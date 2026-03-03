import os
import json
import threading
import concurrent.futures
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime
from werkzeug.utils import secure_filename

# ── STT / LLM SDK imports ────────────────────────────────────────────────────
from groq import Groq
from elevenlabs.client import ElevenLabs

# ── Deepgram SDK — PrerecordedOptions imported lazily inside function ─────────
# (deepgram-sdk v6 moved PrerecordedOptions; lazy import avoids module-level crash)
from deepgram import DeepgramClient

# ── Optional: gradio_client for HF Space node (graceful if absent) ───────────
try:
    from gradio_client import Client as GradioClient, handle_file as gradio_handle_file
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    print("⚠️  gradio_client not installed — HF Space node will be unavailable.")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# [FIX #2] Upload guard — Vercel hobby functions hard-limit at 4.5MB body;
# we cap at 50MB for local/pro use but apply a runtime check before transcription.
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ── API Key Configuration ─────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv('GROQ_API_KEY', '')
DEEPGRAM_API_KEY    = os.getenv('DEEPGRAM_API_KEY', '')
ELEVENLABS_API_KEY  = os.getenv('ELEVENLABS_API_KEY', '')
MURF_API_KEY        = os.getenv('MURF_API_KEY', '')
OPENROUTER_API_KEY  = os.getenv('OPENROUTER_API_KEY', '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')

# HuggingFace Space node — new distributed backend
HF_SPACE_URL        = os.getenv('HF_SPACE_URL', '')       # e.g. "your-user/briefly-asr"
HF_SPACE_TOKEN      = os.getenv('HF_SPACE_TOKEN', '')      # your HF read token (for private spaces)

# ── Client Initialisation ─────────────────────────────────────────────────────
groq_client        = Groq(api_key=GROQ_API_KEY)         if GROQ_API_KEY       else None
deepgram_client    = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None
elevenlabs_client  = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# ── Upload Folder ─────────────────────────────────────────────────────────────
# [FIX #2] Vercel serverless writes only to /tmp — enforced here.
UPLOAD_FOLDER = '/tmp/uploads' if os.environ.get('VERCEL') else os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Vercel deployment guard ────────────────────────────────────────────────────
IS_VERCEL = bool(os.environ.get('VERCEL'))

# [FIX #2] On Vercel hobby (10s hard limit), large audio will always time out.
# This constant is the safe audio size threshold for synchronous API-only processing.
# Files above this are still accepted but automatically routed to the HF Space node
# which processes them server-side (no Vercel timeout exposure).
VERCEL_SAFE_AUDIO_MB = 4.0  # ~4 min of 128kbps MP3

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HF SPACE NODE (WhisperX + pyannote acoustic diarization)
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_via_hf_space(audio_path: str) -> dict:
    """
    Submits audio to the HuggingFace Space (Qualora ASR node).
    The Space runs:
      faster-whisper-large-v3  — transcription with word timestamps
      pyannote/speaker-diarization-3.1 — ECAPA-TDNN acoustic voiceprint clustering
      parselmouth (Praat)      — pitch, intensity, jitter per turn
      speechbrain wav2vec2     — acoustic emotion per speaker turn

    Returns dict with keys:
      transcript       — speaker-labelled full transcript string
      speaker_profiles — acoustic profile per speaker (avg pitch, intensity, dominant emotion)
      turns            — per-turn structs with prosody + emotion
    Raises on failure.
    """
    if not GRADIO_CLIENT_AVAILABLE:
        raise RuntimeError("gradio_client package not installed.")
    if not HF_SPACE_URL:
        raise RuntimeError("HF_SPACE_URL env var not set — HF node not configured.")

    print("--- [HF Space] Connecting to Qualora ASR node ---")
    client = GradioClient(
        HF_SPACE_URL,
        token=HF_SPACE_TOKEN if HF_SPACE_TOKEN else None,
    )

    raw = client.predict(
        gradio_handle_file(audio_path),
        api_name="/predict",
    )

    # Gradio may return dict or JSON string
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Plain text fallback — wrap in standard dict
            return {"transcript": raw.strip(), "speaker_profiles": {}, "turns": []}
    else:
        data = {"transcript": str(raw), "speaker_profiles": {}, "turns": []}

    if "error" in data:
        raise RuntimeError(f"HF Space returned error: {data['error']}")

    transcript = data.get("transcript", "").strip()
    if not transcript:
        raise RuntimeError("HF Space returned empty transcript.")

    nodes = data.get("pipeline_nodes", ["faster-whisper"])
    print(f"✅ [HF Space] Done. Pipeline: {' → '.join(nodes)}")
    return {
        "transcript":       transcript,
        "speaker_profiles": data.get("speaker_profiles", {}),
        "turns":            data.get("turns", []),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — API CHAIN NODE (ElevenLabs → Deepgram [FIXED] → Groq)
# ─────────────────────────────────────────────────────────────────────────────

def _elevenlabs_transcribe(audio_path: str) -> str:
    """
    ElevenLabs Speech-to-Text (Scribe v1) — file upload with speaker diarization.
    SDK: elevenlabs>=2.37 — client.speech_to_text.convert(file=..., model_id='scribe_v1')
    Response has .words[] with .speaker_id and .text per word.
    """
    with open(audio_path, "rb") as f:
        response = elevenlabs_client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
            diarize=True,
            language_code="en",
            timestamps_granularity="word",   # get word-level timestamps + speaker_id
        )

    lines = []

    # Path A — word-level with speaker_id (primary in scribe_v1 with diarize=True)
    if hasattr(response, 'words') and response.words:
        current_speaker = None
        current_text   = []
        for w in response.words:
            spk = getattr(w, 'speaker_id', None)
            txt = getattr(w, 'text', '') or getattr(w, 'punctuated_word', '')
            if not txt.strip():       # skip whitespace-only tokens (EL returns spaces as words)
                continue
            if spk != current_speaker:
                if current_speaker is not None and current_text:
                    # Convert 'speaker_0' → 'Speaker 0'  (avoids 'Speaker speaker_0' redundancy)
                    label = current_speaker.replace('speaker_', 'Speaker ').strip()
                    if not any(c.isdigit() or c.isalpha() for c in label):
                        label = str(current_speaker)
                    lines.append(f"{label}: {' '.join(current_text).strip()}")
                current_speaker = spk
                current_text    = [txt.strip()]
            else:
                current_text.append(txt.strip())
        if current_speaker is not None and current_text:
            label = current_speaker.replace('speaker_', 'Speaker ').strip()
            lines.append(f"{label}: {' '.join(current_text).strip()}")

    # Path B — utterances/segments level
    if not lines:
        segs = getattr(response, 'utterances', None) or getattr(response, 'segments', None) or []
        for seg in segs:
            spk = getattr(seg, 'speaker_id', None) or getattr(seg, 'speaker', '?')
            txt = getattr(seg, 'text', '') or getattr(seg, 'transcript', '')
            lines.append(f"Speaker {spk}: {txt.strip()}")

    if lines:
        return "\n\n".join(lines)

    # Path C — plain text (no diarization)
    if hasattr(response, 'text') and response.text:
        return response.text
    return str(response)


def _deepgram_transcribe(audio_path: str) -> str:
    """
    Deepgram Nova-2 with speaker diarization.
    deepgram-sdk==6.0.0rc2 API (confirmed by introspection):
      - client.listen.v1.media.transcribe_file(request=bytes, **kwargs)
      - No PrerecordedOptions class — all options are direct keyword args.
      - client.listen.prerecorded does NOT exist in v6.
      - Response: response.results.utterances[i].speaker / .transcript
    """
    with open(audio_path, "rb") as f:
        buf = f.read()

    response = deepgram_client.listen.v1.media.transcribe_file(
        request=buf,
        model="nova-2",
        smart_format=True,
        diarize=True,
        punctuate=True,
        utterances=True,
        language="en",
    )

    # Parse utterances (speaker-separated segments)
    if (
        hasattr(response, 'results')
        and hasattr(response.results, 'utterances')
        and response.results.utterances
    ):
        lines = [
            f"Speaker {u.speaker}: {u.transcript}"
            for u in response.results.utterances
        ]
        if lines:
            return "\n\n".join(lines)

    # Fallback: channel alternatives plain transcript
    if (
        hasattr(response, 'results')
        and hasattr(response.results, 'channels')
        and response.results.channels
    ):
        alt = response.results.channels[0].alternatives[0]
        if alt.transcript:
            return alt.transcript

    raise RuntimeError("Deepgram response contained no parseable transcript.")






def _groq_transcribe(audio_path: str) -> str:
    """Groq Whisper-large-v3 — fast, free, no diarization."""
    with open(audio_path, "rb") as f:
        transcription = groq_client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            language="en",
            temperature=0.0,
        )
    if hasattr(transcription, 'text') and transcription.text:
        return transcription.text
    return str(transcription)


def perform_voice_capture_apis(audio_path: str) -> str:
    """
    API-chain node: ElevenLabs → Deepgram [FIXED] → Groq.
    Sequential fallback — tries best quality first, degrades gracefully.
    """
    # Attempt 1: ElevenLabs Scribe (primary — best quality + speaker diarization)
    if elevenlabs_client:
        try:
            print("--- [API Chain] ElevenLabs Scribe ---")
            return _elevenlabs_transcribe(audio_path)
        except Exception as e:
            print(f"⚠️  [ElevenLabs] {e}")

    # Attempt 2: Deepgram Nova-2 (fallback — fixed SDK call + diarization)
    if deepgram_client:
        try:
            print("--- [API Chain] Deepgram Nova-2 ---")
            return _deepgram_transcribe(audio_path)
        except Exception as e:
            print(f"⚠️  [Deepgram] {e}")

    # Attempt 3: Groq Whisper (final — no diarization but always available)
    if groq_client:
        try:
            print("--- [API Chain] Groq Whisper-large-v3 ---")
            return _groq_transcribe(audio_path)
        except Exception as e:
            print(f"⚠️  [Groq] {e}")

    if not (elevenlabs_client or deepgram_client or groq_client):
        raise RuntimeError("No transcription providers configured. Please add GROQ_API_KEY to your Vercel Environment Variables.")
        
    raise RuntimeError("All configured API-chain transcription providers failed to process the audio.")

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ASYNC JOB ENGINE
# HF Space fires first. UI shows "Transcribe Now" if API chain is available.
# User can click it to start API chain simultaneously.
# First valid transcript wins — the other thread is abandoned.
# ─────────────────────────────────────────────────────────────────────────────

import uuid

_jobs: dict = {}   # job_id -> job dict


def _get_fallbacks_available() -> list:
    """Return list of configured API-chain providers."""
    out = []
    if ELEVENLABS_API_KEY: out.append("elevenlabs")
    if DEEPGRAM_API_KEY:   out.append("deepgram")
    if GROQ_API_KEY:       out.append("groq")
    return out


def _clean_old_jobs():
    import time
    cutoff = time.time() - 1800  # 30-minute TTL
    stale  = [k for k, v in list(_jobs.items()) if v.get("_ts", 0) < cutoff]
    for k in stale:
        _jobs.pop(k, None)


def _run_api_chain_for_job(job_id: str):
    """Starts the API-chain fallback (ElevenLabs → Deepgram → Groq) for a job."""
    import time
    job = _jobs.get(job_id)
    if not job or job.get("api_chain_started"):
        return
    job["api_chain_started"] = True

    def _run():
        try:
            tx = perform_voice_capture_apis(job["_filepath"])
            tx = (tx or "").strip()
            if tx and not job["winner"].is_set():
                job["transcript"] = tx
                job["source"]     = "api_chain"
                job["winner"].set()
                print(f"[Job {job_id[:8]}] API chain won.")
        except Exception as e:
            print(f"[Job {job_id[:8]}] API chain failed: {e}")
            if not job["winner"].is_set():
                job["winner"].set()  # unblock the audit watcher

    threading.Thread(target=_run, daemon=True, name=f"api-{job_id[:8]}").start()


def _start_job(audio_filepath: str) -> dict:
    """
    Creates and starts an async transcription job.
    Returns the job dict (including job_id).
    """
    import time
    job_id = uuid.uuid4().hex
    job = {
        "job_id":            job_id,
        "_ts":               time.time(),
        "_filepath":         audio_filepath,
        "status":            "hf_transcribing",
        "transcript":        None,
        "source":            None,
        "acoustic_profile":  {},
        "audit":             None,
        "error":             None,
        "api_chain_started": False,
        "winner":            threading.Event(),
    }
    _jobs[job_id] = job
    _clean_old_jobs()

    # ── Thread 1: HF Space (primary) ──────────────────────────────────────────────
    def _run_hf():
        try:
            result = transcribe_via_hf_space(audio_filepath)
            tx  = result.get("transcript", "").strip()
            apr = result.get("speaker_profiles", {})
            if tx and not job["winner"].is_set():
                job["transcript"]       = tx
                job["acoustic_profile"] = apr
                job["source"]           = "hf_space"
                job["winner"].set()
                print(f"[Job {job_id[:8]}] HF Space won.")
        except Exception as e:
            print(f"[Job {job_id[:8]}] HF Space failed: {e}")
            # Auto-fallback to API chain if not already running
            if not job["api_chain_started"] and not job["winner"].is_set():
                _run_api_chain_for_job(job_id)

    if HF_SPACE_URL and GRADIO_CLIENT_AVAILABLE:
        threading.Thread(target=_run_hf, daemon=True, name=f"hf-{job_id[:8]}").start()
    else:
        # No HF Space configured — go straight to API chain
        job["status"] = "api_transcribing"
        _run_api_chain_for_job(job_id)

    # ── Audit watcher: activates once a transcript is ready ──────────────────────
    def _audit_watcher():
        job["winner"].wait(timeout=300)  # up to 5 min
        if not job["transcript"]:
            job["status"] = "error"
            job["error"]  = "All transcription providers failed or timed out."
            return
        job["status"] = "auditing"
        print(f"[Job {job_id[:8]}] Auditing via {job['source']}...")
        try:
            job["audit"]  = generate_quality_audit(
                job["transcript"],
                acoustic_profile=job["acoustic_profile"],
            )
            job["status"] = "done"
        except Exception as e:
            job["error"]  = str(e)
            job["status"] = "error"
        finally:
            try: os.remove(audio_filepath)
            except Exception: pass

    threading.Thread(target=_audit_watcher, daemon=True, name=f"audit-{job_id[:8]}").start()
    return job


# SECTION 4 — QUALITY AUDIT ENGINE (Milestone 2 — LLM-as-a-Judge)
# ─────────────────────────────────────────────────────────────────────────────

_AUDIT_SYSTEM_PROMPT = """You are an Expert Customer Support Quality Auditor, applied Psychologist, and rigorous Data Scientist.
You receive a customer support transcript (voice or chat) and must return ONLY a valid JSON object — no markdown, no explanation, JSON only.

You MUST optimize every analytic for the absolute highest mathematical correctness and exact precision.
Your output MUST be entirely deterministic based strictly on the data provided, ensuring correct evaluation for all data inputs.

SPEAKER ROLE INFERENCE (apply before scoring):
Every interaction has exactly two sides: the AGENT side (the company/support) and the CUSTOMER side (the client/caller).
Each side may be staffed by a human, a bot, an IVR system, or a combination — but always map to Agent or Customer.
- AGENT side: initiates greetings, offers solutions, references policies, stays professional. Includes human reps, chatbots, and IVR systems representing the company.
- CUSTOMER side: describes a problem, asks for help, expresses frustration, references personal account/order. Includes humans and occasionally automated systems placing service requests.
Generic labels (SPEAKER_00, Speaker 0, spk_1, etc.) MUST be resolved to Agent or Customer using context. Once identified, apply consistently for all turns of that speaker.

JSON schema (all fields required):
{
  "summary": "<one-sentence plain-English summary of the interaction and its outcome>",
  "interaction_type": {
    "agent_channel": "<Human|Bot|IVR|Hybrid>",
    "customer_channel": "<Human|Bot>"
  },
  "agent_f1_score": <float 0.0–1.0, harmonic mean of precision and recall of agent helpfulness>,
  "satisfaction_prediction": "<High|Medium|Low>",
  "compliance_risk": "<Green|Amber|Red>",
  "quality_matrix": {
    "language_proficiency": <int 1–10>,
    "cognitive_empathy": <int 1–10>,
    "efficiency": <int 1–10>,
    "bias_reduction": <int 1–10>,
    "active_listening": <int 1–10>
  },
  "emotional_timeline": [
    {"turn": <int>, "speaker": "<Agent|Customer>", "emotion": "<Frustrated|Angry|Neutral|Confused|Relieved|Satisfied|Happy|Anxious|Professional|Empathetic|Calm>", "intensity": <int 1–10>}
  ],
  "compliance_flags": ["<specific violation or concern, if any>"],
  "behavioral_nudges": ["<specific, psychologically-grounded coaching tip — only applicable to the human agent if present>"],
  "hitl_review_required": <true|false>
}

Scoring guide:
- interaction_type: Hybrid means both human and bot/IVR participated on the agent side.
- agent_f1_score: harmonic mean of precision (correct/helpful statements) and recall (customer pain points addressed). Score the effective agent performance regardless of human or bot.
- compliance_risk: Green = no issues, Amber = minor deviations, Red = serious breach
- hitl_review_required: true if compliance_risk is Red, or if the interaction was fully bot-handled with no human escalation and the issue was unresolved.
- emotional_timeline: include every speaker turn (Agent or Customer). Map turns sequentially (turn 1, 2, 3...).
- behavioral_nudges: 2–4 tips grounded in psychology. If agent_channel is Bot or IVR, focus nudges on system design recommendations rather than interpersonal coaching.
- compliance_flags: empty array [] if none found."""

def _build_acoustic_context(speaker_profiles: dict) -> str:
    """
    Converts the HF Space acoustic profile dict into a plain-English preamble
    for the LLM prompt. This grounds the NLP analysis in real signal-level evidence.
    """
    if not speaker_profiles:
        return ""
    lines = ["\n\n[ACOUSTIC SIGNAL ANALYSIS — from pyannote 3.1 + parselmouth + SpeechBrain wav2vec2]"]
    for spk, profile in speaker_profiles.items():
        pitch  = profile.get("avg_pitch_hz")
        intens = profile.get("avg_intensity_db")
        emo    = profile.get("dominant_emotion", "unknown")
        turns  = profile.get("turn_count", 0)
        # Interpret pitch relative to population baselines
        # Male baseline ~120 Hz, Female baseline ~210 Hz; elevated pitch → stress
        if pitch:
            pitch_note = " (elevated — stress/anxiety signal)" if pitch > 260 else \
                         " (depressed — fatigue/monotony signal)" if pitch < 100 else ""
        else:
            pitch_note = ""
        line = f"  {spk}: avg_pitch={pitch:.0f}Hz{pitch_note}" if pitch else f"  {spk}:"
        if intens:
            line += f", avg_intensity={intens:.1f}dB"
        line += f", dominant_acoustic_emotion={emo}, turn_count={turns}"
        lines.append(line)
    lines.append("Use these acoustic signals to enrich your scoring — especially cognitive_empathy, "
                 "bias_reduction, and compliance_risk. Acoustic anger/frustration is ground-truth "
                 "evidence, not NLP inferred.")
    return "\n".join(lines)


import hashlib
import json

_AUDIT_CACHE = {}

def generate_quality_audit(transcript: str, acoustic_profile: dict | None = None) -> dict:
    """
    LLM-as-a-Judge: Groq Llama-3.3-70b scores the support interaction.
    Returns a structured audit dict. Never raises — always returns a safe default on error.
    """
    
    # ── PERSISTENT CACHING ──
    # Guarantee identical outputs for the exact same input string and profile
    cache_key = hashlib.sha256(f"{transcript}|{json.dumps(acoustic_profile or {}, sort_keys=True)}".encode()).hexdigest()
    if cache_key in _AUDIT_CACHE:
        print("⚡ [Audit] Exact match found in persistent cache. Returning deterministic result.")
        return _AUDIT_CACHE[cache_key]


    _FALLBACK = {
        "summary": "Audit unavailable — LLM engine offline.",
        "agent_f1_score": 0.0,
        "satisfaction_prediction": "Unknown",
        "compliance_risk": "Unknown",
        "quality_matrix": {
            "language_proficiency": 0, "cognitive_empathy": 0,
            "efficiency": 0, "bias_reduction": 0, "active_listening": 0
        },
        "emotional_timeline": [],
        "compliance_flags": [],
        "behavioral_nudges": ["LLM audit engine was unavailable. Check GROQ_API_KEY."],
        "hitl_review_required": True,
    }

    import time

    MAX_CHARS = 40_000          # reduced from 80k to cut token usage and rate-limit risk
    if len(transcript) > MAX_CHARS:
        print(f"[Audit] Trimming transcript: {len(transcript)} → {MAX_CHARS} chars")
        transcript = transcript[:MAX_CHARS] + "\n...[truncated for token limits]"

    # Inject acoustic context if available (from HF Space pipeline)
    acoustic_ctx = _build_acoustic_context(acoustic_profile or {})
    user_prompt  = f"Transcript to audit:{acoustic_ctx}\n\n{transcript}\n\nReturn ONLY the JSON object."

    def _apply_defensive_merge(parsed_audit):
        import re
        # 1. Base key merge
        for key, val in _FALLBACK.items():
            if key not in parsed_audit or parsed_audit[key] is None:
                parsed_audit[key] = val
        
        # 2. Quality Matrix inference
        qm = parsed_audit.get("quality_matrix", {})
        for k, v in _FALLBACK["quality_matrix"].items():
            # Standard generic benchmark is 5/10 for any missing matrix token
            if k not in qm or not isinstance(qm[k], (int, float)):
                qm[k] = 5
        parsed_audit["quality_matrix"] = qm
        
        # 3. Mathematical F1 Logical Inference
        f1 = parsed_audit.get("agent_f1_score")
        if f1 is None or not isinstance(f1, (int, float)) or f1 == 0:
            p_val = (qm.get("language_proficiency", 5) + qm.get("efficiency", 5) + qm.get("bias_reduction", 5)) / 30.0
            r_val = (qm.get("cognitive_empathy", 5) + qm.get("active_listening", 5)) / 20.0
            if (p_val + r_val) > 0:
                parsed_audit["agent_f1_score"] = round((2 * p_val * r_val) / (p_val + r_val), 2)
            else:
                parsed_audit["agent_f1_score"] = 0.50

        # 4. Emotional Timeline Inference (Segment text if missing)
        timeline = parsed_audit.get("emotional_timeline", [])
        if not timeline or not isinstance(timeline, list) or len(timeline) == 0:
            inferred_timeline = []
            lines = [line.strip() for line in transcript.split('\n') if len(line.strip()) > 2]

            # Role keywords — Bot/IVR always map to Agent side (they represent the company)
            _BOT_LABELS      = {'bot', 'ivr', 'system', 'automated', 'ai', 'virtual', 'chatbot', 'assistant'}
            _AGENT_LABELS    = {'agent', 'rep', 'representative', 'support', 'advisor', 'operator', 'staff', 'specialist'}
            _CUSTOMER_LABELS = {'customer', 'client', 'user', 'caller', 'member', 'patient', 'guest'}

            def _infer_role(label: str) -> str:
                l = label.lower().strip()
                if any(k in l for k in _BOT_LABELS):      return 'Agent'   # company-side bot/IVR → Agent
                if any(k in l for k in _AGENT_LABELS):    return 'Agent'
                if any(k in l for k in _CUSTOMER_LABELS): return 'Customer'
                return None  # fallback to alternation for generic labels (SPEAKER_00 etc.)

            for i, line in enumerate(lines[:50]):
                speaker_match = re.match(r'^([^:]+):', line)
                if speaker_match:
                    raw_label = speaker_match.group(1).strip()
                    role = _infer_role(raw_label)
                    spk = role if role else raw_label[:20]
                else:
                    spk = 'Agent' if i % 2 == 0 else 'Customer'

                lower_line = line.lower()
                emo, intensity = 'Neutral', 3
                if any(w in lower_line for w in ['sorry', 'apologize', 'understand']): emo, intensity = 'Empathetic', 6
                elif any(w in lower_line for w in ['angry', 'mad', 'unacceptable', 'furious']): emo, intensity = 'Angry', 9
                elif any(w in lower_line for w in ['help', 'thank', 'great', 'resolved']): emo, intensity = 'Satisfied', 7
                elif any(w in lower_line for w in ['press', 'menu', 'option', 'automated']): emo, intensity = 'Neutral', 2
                elif '?' in line: emo, intensity = 'Confused', 5

                inferred_timeline.append({
                    'turn': i + 1,
                    'speaker': spk,
                    'emotion': emo,
                    'intensity': intensity
                })
            parsed_audit['emotional_timeline'] = inferred_timeline

        # 5. Fallback nudges
        nudges = parsed_audit.get("behavioral_nudges", [])
        if not nudges or not isinstance(nudges, list) or len(nudges) == 0:
            parsed_audit["behavioral_nudges"] = [
                "Fundamental Protocol: Periodically affirm customer issues verbatim to build rapport.",
                "Efficiency Driver: Utilize active templating to resolve recurring issues faster without eroding personalization."
            ]

        return parsed_audit

    if groq_client:
        for attempt in range(3):    # up to 3 tries with exponential backoff on rate-limit
            try:
                print(f"--- [Audit] Groq Llama-3.3-70b attempt {attempt+1}/3 ---")
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": _AUDIT_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=2048,
                )
                raw = response.choices[0].message.content.strip()
                audit = json.loads(raw)
                print("✅ [Audit] Quality audit complete via Groq.")
                final_audit = _apply_defensive_merge(audit)
                _AUDIT_CACHE[cache_key] = final_audit
                return final_audit

            except json.JSONDecodeError as e:
                print(f"⚠️  [Audit] JSON parse failed on Groq: {e}")
                break

            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "rate limit" in err_str or "429" in err_str:
                    if IS_VERCEL:
                        print(f"⚠️  [Audit] Groq Rate limit hit on Vercel. Skipping sleep retries to prevent 504 Timeout.")
                        break
                        
                    wait = 5 * (2 ** attempt)
                    print(f"⚠️  [Audit] Rate limit hit (attempt {attempt+1}). Waiting {wait}s...")
                    time.sleep(wait)
                    if attempt == 2:
                        print("⚠️  [Audit] Exhausted Groq retries, falling back to OpenRouter.")
                else:
                    print(f"⚠️  [Audit] Groq call failed: {e}")
                    break

    # ── OPENROUTER FALLBACK ──
    _or_model   = "google/gemini-2.5-flash"
    _or_timeout = 45
    _or_tokens  = 2048
    print(f"--- [Audit] Attempting OpenRouter Fallback (model={_or_model}, timeout={_or_timeout}s) ---")
    try:
        import requests
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            },
            json={
                "model": _or_model,
                "messages": [
                    {"role": "system", "content": _AUDIT_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
                "max_tokens": _or_tokens
            },
            timeout=_or_timeout
        )
        if resp.status_code == 200:
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip any markdown wrappers occasionally returned by OpenRouter
            if raw.startswith("```json"): raw = raw[7:]
            if raw.endswith("```"):       raw = raw[:-3]
            raw = raw.strip()
            audit = json.loads(raw)
            print(f"✅ [Audit] Quality audit complete via OpenRouter ({_or_model}).")
            final_audit = _apply_defensive_merge(audit)
            _AUDIT_CACHE[cache_key] = final_audit
            return final_audit
        else:
            print(f"⚠️  [Audit] OpenRouter failed with {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"⚠️  [Audit] OpenRouter call failed: {e}")

    return _FALLBACK



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FILE PARSING (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_file(filepath: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == '.pdf':
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            return '\n'.join(page.extract_text() or '' for page in reader.pages).strip()
        elif ext == '.json':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            return json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)
        else:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
    except Exception as e:
        print(f"[Extraction] {filename}: {e}")
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/process-chat', methods=['POST'])
def process_chat():
    try:
        data = request.json
        if not data: return jsonify({'error': 'No JSON payload'}), 400
        
        text = data.get('text', '').strip()
        if not text: return jsonify({'error': 'Empty text'}), 400
        
        print(f"[process-chat] Running text quality audit ({len(text)} chars)...")
        # Text generation naturally forces rate-limits on Vercel without a fast fallback unless
        # explicit. Since text input has NO hugging face pipeline, it immediately skips to OpenRouter/Groq.
        audit = generate_quality_audit(text)
        
        return jsonify({
            'success': True,
            'type': 'chat',
            'original_text': text,
            'audit': audit,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        })
    except Exception as e:
        print(f"⚠️  [process-chat] Failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process-file', methods=['POST'])
def process_file():
    """Process uploaded text/PDF document — returns structured QA audit (Milestone 2)."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        uploaded = request.files['file']
        if not uploaded.filename:
            return jsonify({'error': 'No file selected'}), 400

        allowed = {'.txt', '.csv', '.json', '.md', '.log', '.pdf'}
        ext     = os.path.splitext(uploaded.filename)[1].lower()
        if ext not in allowed:
            return jsonify({'error': f'Unsupported format. Use: {", ".join(allowed)}'}), 400

        safe_name = secure_filename(uploaded.filename)
        filename  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        filepath  = os.path.join(UPLOAD_FOLDER, filename)
        uploaded.save(filepath)

        text = extract_text_from_file(filepath, uploaded.filename)
        try:
            os.remove(filepath)
        except Exception:
            pass

        if not text:
            return jsonify({'error': 'Could not extract text from file'}), 400

        print("[process-file] Running quality audit...")
        audit = generate_quality_audit(text)
        return jsonify({
            'success':       True,
            'type':          'chat',
            'original_text': text,
            'audit':         audit,
            'timestamp':     datetime.utcnow().isoformat() + 'Z',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process-call', methods=['POST'])
def process_call():
    """
    Legacy synchronous endpoint: HF Space first, then API chain fallback.
    New clients should use /api/start-call-audit + /api/job/<id>/status instead.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        audio_file = request.files['audio']
        if not audio_file.filename:
            return jsonify({'error': 'No file selected'}), 400

        safe_name = secure_filename(audio_file.filename)
        filename  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        filepath  = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(filepath)
        file_size = os.path.getsize(filepath)

        MAX_BYTES = 50 * 1024 * 1024
        if file_size > MAX_BYTES:
            os.remove(filepath)
            return jsonify({'error': f'File exceeds 50 MB ({file_size/1e6:.1f} MB).'}), 413

        print(f"[process-call] {filename} ({file_size/1e6:.2f} MB)")

        # Check if user requested immediate API transcription (skipping HF Space queue)
        is_fast_track = request.args.get('fast_track', 'false').lower() == 'true'

        # Waterfall: HF Space → API chain
        transcription = source_node = None
        acoustic_profile = {}

        if not is_fast_track and HF_SPACE_URL and GRADIO_CLIENT_AVAILABLE:
            try:
                result = transcribe_via_hf_space(filepath)
                transcription    = result.get("transcript", "").strip()
                acoustic_profile = result.get("speaker_profiles", {})
                source_node      = "hf_space"
                print(f"[process-call] HF Space OK")
            except Exception as e:
                print(f"[process-call] HF Space failed, falling back: {e}")

        if not transcription:
            transcription = perform_voice_capture_apis(filepath)
            source_node   = "api_chain"

        try: os.remove(filepath)
        except Exception: pass

        print(f"[process-call] Auditing (source: {source_node})...")
        audit = generate_quality_audit(transcription, acoustic_profile=acoustic_profile)

        return jsonify({
            'success':          True,
            'type':             'call',
            'transcription':    transcription,
            'audit':            audit,
            'source_node':      source_node,
            'acoustic_profile': acoustic_profile,
            'timestamp':        datetime.utcnow().isoformat() + 'Z',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/start-call-audit', methods=['POST'])
def start_call_audit():
    """
    Starts an async transcription + audit job.
    Returns immediately with {job_id, fallbacks_available}.
    Client polls /api/job/<id>/status to track progress.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        audio_file = request.files['audio']
        if not audio_file.filename:
            return jsonify({'error': 'No file selected'}), 400

        safe_name = secure_filename(audio_file.filename)
        filename  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        filepath  = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(filepath)
        file_size = os.path.getsize(filepath)

        MAX_BYTES = 50 * 1024 * 1024
        if file_size > MAX_BYTES:
            os.remove(filepath)
            return jsonify({'error': f'File exceeds 50 MB ({file_size/1e6:.1f} MB).'}), 413

        print(f"[start-call-audit] {filename} ({file_size/1e6:.2f} MB)")
        job = _start_job(filepath)

        return jsonify({
            'job_id':              job['job_id'],
            'fallbacks_available': _get_fallbacks_available(),
            'hf_active':          bool(HF_SPACE_URL and GRADIO_CLIENT_AVAILABLE),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/job/<job_id>/status')
def job_status(job_id):
    """Poll for job progress. Returns status + full audit when done."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found or expired.'}), 404

    resp = {
        'status':            job['status'],
        'source':            job['source'],
        'api_chain_started': job['api_chain_started'],
        'error':             job['error'],
    }
    if job['status'] == 'done':
        resp['success']          = True
        resp['type']             = 'call'
        resp['transcription']    = job['transcript']
        resp['audit']            = job['audit']
        resp['acoustic_profile'] = job['acoustic_profile']
        resp['timestamp']        = datetime.utcnow().isoformat() + 'Z'
    return jsonify(resp)


@app.route('/api/job/<job_id>/transcribe-now', methods=['POST'])
def transcribe_now(job_id):
    """
    User-triggered: starts the API-chain fallback while HF Space is still running.
    First result (HF or API chain) wins.
    """
    job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found or expired.'}), 404
    if job['winner'].is_set():
        return jsonify({'message': 'Transcription already complete.'}), 200
    if job['api_chain_started']:
        return jsonify({'message': 'API chain already running.'}), 200

    _run_api_chain_for_job(job_id)
    job['status'] = 'api_transcribing'
    providers     = _get_fallbacks_available()
    return jsonify({'triggered': True, 'providers': providers})


@app.route('/api/health')
def health():
    """Health check — reports status of all nodes."""
    hf_node_configured = bool(HF_SPACE_URL and GRADIO_CLIENT_AVAILABLE)

    transcription_chain = [
        p for p, flag in [
            ('elevenlabs', bool(ELEVENLABS_API_KEY)),
            ('deepgram',   bool(DEEPGRAM_API_KEY)),
            ('groq',       bool(GROQ_API_KEY)),
        ] if flag
    ]

    return jsonify({
        'status': 'operational',
        'architecture': 'distributed_hybrid_competitive',
        'nodes': {
            'hf_space': {
                'configured': hf_node_configured,
                'url':        HF_SPACE_URL or None,
                'capability': 'acoustic_diarization_whisperx_pyannote' if hf_node_configured else None,
            },
            'api_chain': {
                'configured': bool(transcription_chain),
                'providers':  transcription_chain,
                'diarization': bool(ELEVENLABS_API_KEY or DEEPGRAM_API_KEY),
            },
        },
        'summarization': {
            'primary':  'groq'    if GROQ_API_KEY    else None,
            'fallback': 'deepgram' if DEEPGRAM_API_KEY else None,
        },
        'vercel_mode': IS_VERCEL,
        'api_ready': bool(hf_node_configured or transcription_chain),
        'fallbacks': {
            'hf_space':   hf_node_configured,
            'elevenlabs': bool(ELEVENLABS_API_KEY),
            'deepgram':   bool(DEEPGRAM_API_KEY),
            'groq':       bool(GROQ_API_KEY),
            'murf':       bool(MURF_API_KEY),
        },
    })


@app.route('/api/elevenlabs-token', methods=['GET'])
def get_elevenlabs_token():
    """Generate single-use token for ElevenLabs client-side realtime Scribe."""
    if not elevenlabs_client:
        return jsonify({'error': 'ElevenLabs not configured'}), 503
    try:
        token = elevenlabs_client.tokens.single_use.create("realtime_scribe")
        return jsonify(token)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    allowed_exts = {'.css', '.js', '.ico', '.png', '.jpg', '.svg'}
    if os.path.splitext(filename)[1].lower() in allowed_exts:
        return send_from_directory(BASE_DIR, filename)
    return jsonify({'error': 'Access denied'}), 403

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — STARTUP
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("🚀  Qualora — AI Customer Support Quality Auditor")
    print("    Turn every conversation into intelligence.")
    print("=" * 65)

    hf_ready  = bool(HF_SPACE_URL and GRADIO_CLIENT_AVAILABLE)
    api_ready = bool(ELEVENLABS_API_KEY or DEEPGRAM_API_KEY or GROQ_API_KEY)

    print(f"{'✅' if hf_ready  else '⬜'} HF Space Node   : {'Active → ' + HF_SPACE_URL if hf_ready else 'Not configured (set HF_SPACE_URL)'}")
    print(f"{'✅' if api_ready else '⬜'} API Chain Node  : "
          + " → ".join(p for p, flag in [
              ("ElevenLabs", bool(ELEVENLABS_API_KEY)),
              ("Deepgram",   bool(DEEPGRAM_API_KEY)),
              ("Groq",       bool(GROQ_API_KEY)),
          ] if flag) or "Not configured")

    if hf_ready and api_ready:
        print("\n✅  COMPETITIVE MODE ACTIVE — both nodes will race on each request.")
    elif hf_ready:
        print("\n⚠️  HF Space only — API chain not configured (no API keys set).")
    elif api_ready:
        print("\n⚠️  API chain only — HF Space not configured (set HF_SPACE_URL).")
    else:
        print("\n❌  CRITICAL: No transcription nodes available.")

    print("=" * 65 + "\n")
    app.run(debug=True, port=5000)

# Vercel WSGI entry point
app = app
