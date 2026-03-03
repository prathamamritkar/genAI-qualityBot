import os
import re as _re
import json
import copy
import time
import threading
import concurrent.futures

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
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
# Always override OS-level env vars with values from .env so updated keys
# take effect on every process restart without manual env clearing.
load_dotenv(override=True)

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
OPENROUTER_API_KEY  = os.getenv('OPENROUTER_API_KEY', '')  # NO hardcoded fallback for security

# HuggingFace Space node — new distributed backend
HF_SPACE_URL        = os.getenv('HF_SPACE_URL', '')       # e.g. "your-user/briefly-asr"
HF_SPACE_TOKEN      = os.getenv('HF_SPACE_TOKEN', '')      # your HF read token (for private spaces)

# ── Client Initialisation ─────────────────────────────────────────────────────
groq_client        = Groq(api_key=GROQ_API_KEY)         if GROQ_API_KEY       else None
deepgram_client    = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None
elevenlabs_client  = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# ── Upload Folder ─────────────────────────────────────────────────────────────
# Vercel's build system auto-injects VERCEL_ENV. Any other value (incl. manual
# VERCEL=True in .env) must NOT trigger /tmp routing on a local machine.
_ON_VERCEL = bool(os.environ.get('VERCEL_ENV', '').strip())
UPLOAD_FOLDER = '/tmp/uploads' if _ON_VERCEL else os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    """Groq Whisper tiered — T1: large-v3 (best) → T2: large-v3-turbo (fast)."""
    _whisper_models = [
        ("whisper-large-v3",       "T1"),      # Primary: highest quality
        ("whisper-large-v3-turbo", "T2"),      # Fallback: faster
    ]
    last_err = None
    for whisper_model, tier in _whisper_models:
        try:
            with open(audio_path, "rb") as f:
                transcription = groq_client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), f.read()),
                    model=whisper_model,
                    response_format="verbose_json",
                    language="en",
                    temperature=0.0,
                )
            if hasattr(transcription, 'text') and transcription.text:
                print(f"✅ [Groq Whisper] [{tier}] {whisper_model} transcribed successfully.")
                return transcription.text
        except Exception as e:
            print(f"⚠️  [{tier}] {whisper_model} failed: {e}. Trying next model.")
            last_err = e
    raise RuntimeError(f"Both Groq Whisper models failed. Last error: {last_err}")


def perform_voice_capture_apis(audio_path: str) -> tuple:
    """
    API-chain node: ElevenLabs → Deepgram → Groq.
    Sequential fallback — tries best quality first, degrades gracefully.
    Returns (transcript: str, provider_label: str).
    """
    # Attempt 1: ElevenLabs Scribe (primary — best quality + speaker diarization)
    if elevenlabs_client:
        try:
            print("--- [API Chain] ElevenLabs Scribe ---")
            return _elevenlabs_transcribe(audio_path), "ElevenLabs Scribe"
        except Exception as e:
            print(f"⚠️  [ElevenLabs] {e}")

    # Attempt 2: Deepgram Nova-2 (fallback — fixed SDK call + diarization)
    if deepgram_client:
        try:
            print("--- [API Chain] Deepgram Nova-2 ---")
            return _deepgram_transcribe(audio_path), "Deepgram Nova-2"
        except Exception as e:
            print(f"⚠️  [Deepgram] {e}")

    # Attempt 3: Groq Whisper (final — no diarization but always available)
    if groq_client:
        try:
            print("--- [API Chain] Groq Whisper-large-v3 ---")
            return _groq_transcribe(audio_path), "Groq Whisper-large-v3"
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
            tx, provider = perform_voice_capture_apis(job["_filepath"])
            tx = (tx or "").strip()
            if tx and not job["winner"].is_set():
                job["transcript"]            = tx
                job["source"]                = "api_chain"
                job["transcription_provider"] = provider
                job["winner"].set()
                print(f"[Job {job_id[:8]}] API chain won via {provider}.")
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
        "job_id":                job_id,
        "_ts":                   time.time(),
        "_filepath":             audio_filepath,
        "status":                "hf_transcribing",
        "transcript":            None,
        "source":                None,
        "transcription_provider": None,
        "acoustic_profile":      {},
        "audit":                 None,
        "error":                 None,
        "api_chain_started":     False,
        "winner":                threading.Event(),
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
                job["transcript"]            = tx
                job["acoustic_profile"]      = apr
                job["source"]                = "hf_space"
                job["transcription_provider"] = "Faster-Whisper + pyannote"
                job["winner"].set()
                print(f"[Job {job_id[:8]}] HF Space won.")
        except Exception as e:
            print(f"[Job {job_id[:8]}] HF Space failed: {e}")
            # Auto-fallback to API chain if not already running
            if not job["api_chain_started"] and not job["winner"].is_set():
                _run_api_chain_for_job(job_id)

    if HF_SPACE_URL and GRADIO_CLIENT_AVAILABLE:
        threading.Thread(target=_run_hf, daemon=True, name=f"hf-{job_id[:8]}").start()

        # ── Timeout watcher: HF Space must win within 90 s, else start API chain ──
        def _hf_timeout_watcher():
            import time
            time.sleep(90)
            if not job["winner"].is_set() and not job["api_chain_started"]:
                print(f"[Job {job_id[:8]}] HF Space timeout (90s) — starting API chain fallback.")
                job["status"] = "api_transcribing"
                _run_api_chain_for_job(job_id)

        threading.Thread(target=_hf_timeout_watcher, daemon=True, name=f"hftmr-{job_id[:8]}").start()
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

_AUDIT_CACHE = {}


def _repair_json(raw: str) -> dict:
    """Best-effort JSON repair for LLM responses that contain minor syntax errors."""
    # 1. Strip markdown fences
    for fence in ("```json", "```JSON", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):]
            break
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    # 2. Extract outermost JSON object
    brace_start = raw.find('{')
    brace_end   = raw.rfind('}')
    if brace_start != -1 and brace_end != -1:
        raw = raw[brace_start:brace_end + 1]

    # 3. Try clean parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 4. Fix trailing commas before ] or }  (common Gemini issue)
    fixed = _re.sub(r',\s*([}\]])', r'\1', raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 5. Remove unescaped literal newlines + tabs inside strings
    fixed2 = _re.sub(r'(?<!\\)[\n\r]', ' ', fixed)
    fixed2 = fixed2.replace('\t', ' ')
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    # 6. Strip non-printable control characters (except space)
    fixed3 = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', fixed2)
    try:
        return json.loads(fixed3)
    except json.JSONDecodeError:
        pass

    # 7. Truncate to last valid } and retry
    for i in range(len(fixed3) - 1, -1, -1):
        if fixed3[i] == '}':
            try:
                return json.loads(fixed3[:i + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"JSON repair exhausted all strategies. First 200 chars: {raw[:200]}")

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
        # Return a deep copy to preserve metadata for response handlers
        return copy.deepcopy(_AUDIT_CACHE[cache_key])


    _FALLBACK = {
        "summary": "Automated audit unavailable — all configured judge models were exhausted or unreachable.",
        "agent_f1_score": 0.0,
        "satisfaction_prediction": "Unscored",
        "compliance_risk": "Unscored",
        "quality_matrix": {
            "language_proficiency": 0, "cognitive_empathy": 0,
            "efficiency": 0, "bias_reduction": 0, "active_listening": 0
        },
        "emotional_timeline": [],
        "compliance_flags": [],
        "behavioral_nudges": [],
        "hitl_review_required": True,
        # _audit_metadata is injected dynamically just before returning
    }

    # Live audit attempt registry — populated as each model is tried
    _attempted: list[dict] = []

    def _build_exhaustion_metadata() -> dict:
        """Construct audit metadata from actual attempt history."""
        if not _attempted:
            reason = (
                "groq_not_configured" if not groq_client else
                "openrouter_not_configured" if not OPENROUTER_API_KEY else
                "no_models_attempted"
            )
            return {"model_id": "none", "model_label": f"No judge models available [{reason}]",
                    "tier": "!", "attempted": []}
        summary = "; ".join(f"{m['label']} [{m['reason']}]" for m in _attempted)
        return {
            "model_id":         "exhausted",
            "model_label":      f"All {len(_attempted)} judge(s) exhausted",
            "tier":             "!",
            "attempted":        _attempted,
            "attempted_summary": summary,
        }

    MAX_CHARS = 24_000          # covers ~20 min call; well within Groq 128K context
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
        if f1 is None or not isinstance(f1, (int, float)):
            # Only infer F1 if the model didn't provide one
            p_val = (qm.get("language_proficiency", 5) + qm.get("efficiency", 5) + qm.get("bias_reduction", 5)) / 30.0
            r_val = (qm.get("cognitive_empathy", 5) + qm.get("active_listening", 5)) / 20.0
            if (p_val + r_val) > 0:
                parsed_audit["agent_f1_score"] = round((2 * p_val * r_val) / (p_val + r_val), 2)
            else:
                parsed_audit["agent_f1_score"] = 0.50
        # If f1 == 0, respect it as a valid score from the model (do not override)

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
                
                # Emotion detection with more keywords for better variety
                if any(w in lower_line for w in ['sorry', 'apologize', 'understand', 'empathize', 'appreciate', 'grateful']):
                    emo, intensity = 'Empathetic', 7
                elif any(w in lower_line for w in ['angry', 'mad', 'unacceptable', 'furious', 'frustrated', 'annoyed', 'irritated']):
                    emo, intensity = 'Angry', 9
                elif any(w in lower_line for w in ['help', 'thank', 'great', 'resolved', 'excellent', 'perfect', 'happy', 'satisfied']):
                    emo, intensity = 'Satisfied', 8
                elif any(w in lower_line for w in ['press', 'menu', 'option', 'automated', 'please hold', 'transfer']):
                    emo, intensity = 'Neutral', 2
                elif any(w in lower_line for w in ['confused', 'unclear', 'don\'t understand', 'what', 'huh', 'pardon', 'repeat']):
                    emo, intensity = 'Confused', 5
                elif any(w in lower_line for w in ['wait', 'hold', 'patient', 'waiting', 'soon', 'hurry']):
                    emo, intensity = 'Impatient', 6
                elif any(w in lower_line for w in ['concerned', 'worried', 'anxious', 'afraid', 'scared']):
                    emo, intensity = 'Concerned', 5
                # Add alternation for lines without clear emotional markers
                elif 'please' in lower_line:
                    emo, intensity = 'Polite', 4
                elif i > 0 and inferred_timeline:
                    # Add variety by sometimes using different emotions
                    prev_emotion = inferred_timeline[-1]['emotion']
                    pool = ['Neutral', 'Polite', 'Confused', 'Impatient']
                    if prev_emotion in pool:
                        pool.remove(prev_emotion)
                    if pool:
                        emo = pool[i % len(pool)]
                        intensity = 3 + (i % 4)

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

    # ── Enterprise Audit Cascade (Tier-based, Production-Grade) ──
    # Tier 1: Proven QA models — high TPM, high consistency, support json_object mode
    # Tier 2: Fast instant models — high RPD, lower latency
    # Tier 3: Specialist/research models — may not support json_object, handled gracefully
    # Key: (model_id, label, {tier, supports_json_mode})
    _GROQ_AUDIT_MODELS = [
        # T1 — Primary production model: 12K TPM, 1K RPD, proven QA scoring
        ("llama-3.3-70b-versatile",                       "Llama 3.3 70B",           {"tier": 1, "supports_json_mode": True}),
        # T2 — Fast backup: 6K TPM, 14.4K RPD (highest RPD! great rate-limit escape)
        ("llama-3.1-8b-instant",                          "Llama 3.1 8B Instant",    {"tier": 2, "supports_json_mode": True}),
        # T2 — Llama 4 Scout: 30K TPM, 1K RPD, fast
        ("meta-llama/llama-4-scout-17b-16e-instruct",     "Llama 4 Scout 17B",       {"tier": 2, "supports_json_mode": False}),
        # T2 — Llama 4 Maverick: 6K TPM, 1K RPD
        ("meta-llama/llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick 17B",    {"tier": 2, "supports_json_mode": False}),
        # T3 — Kimi K2: 10K TPM, 60 RPM, good for complex reasoning
        ("moonshotai/kimi-k2-instruct",                   "Kimi K2",                 {"tier": 3, "supports_json_mode": False}),
    ]

    if groq_client:
        for model_id, model_label, model_meta in _GROQ_AUDIT_MODELS:
            try:
                tier = model_meta.get("tier", 99)
                supports_json = model_meta.get("supports_json_mode", False)
                tier_str = f"[T{tier}]"
                print(f"--- [Audit] {tier_str} Groq {model_label} ---")

                call_kwargs = dict(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": _AUDIT_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_completion_tokens=2048,
                )
                if supports_json:
                    call_kwargs["response_format"] = {"type": "json_object"}

                response = groq_client.chat.completions.create(**call_kwargs)
                raw = response.choices[0].message.content.strip()
                try:
                    audit = _repair_json(raw)
                except (ValueError, json.JSONDecodeError) as parse_err:
                    _attempted.append({"label": model_label, "tier": f"T{tier}", "reason": "json_parse_error"})
                    print(f"⚠️  [Audit] JSON parse failed on {model_label}: {parse_err}. Trying next model.")
                    continue
                print(f"✅ [Audit] Quality audit complete via {tier_str} {model_label}.")
                final_audit = _apply_defensive_merge(audit)
                final_audit['_audit_metadata'] = {
                    'model_id':    model_id,
                    'model_label': model_label,
                    'tier':        tier,
                }
                _AUDIT_CACHE[cache_key] = copy.deepcopy(final_audit)
                return final_audit

            except Exception as e:
                err_str = str(e).lower()
                tier = model_meta.get("tier", 99)
                if "rate_limit" in err_str or "rate limit" in err_str or "429" in err_str:
                    reason = "rate_limit"
                    print(f"⚠️  [Audit] [T{tier}] Rate limit on {model_label} — trying next model.")
                elif "response_format" in err_str or "json_object" in err_str or "not support" in err_str:
                    reason = "json_mode_unsupported"
                    print(f"⚠️  [Audit] [T{tier}] {model_label} does not support json_object — trying next model.")
                elif "timeout" in err_str or "timed out" in err_str:
                    reason = "timeout"
                    print(f"⚠️  [Audit] [T{tier}] {model_label} timed out — trying next model.")
                elif "401" in err_str or "unauthorized" in err_str or "invalid api" in err_str:
                    reason = "auth_error"
                    print(f"⚠️  [Audit] [T{tier}] {model_label} auth error — trying next model.")
                else:
                    reason = type(e).__name__
                    print(f"⚠️  [Audit] [T{tier}] {model_label} failed ({reason}): {e} — trying next model.")
                _attempted.append({"label": model_label, "tier": f"T{tier}", "reason": reason})
                continue

        print(f"⚠️  [Audit] All {len(_attempted)} Groq model(s) exhausted — escalating to OpenRouter.")

    # ── OPENROUTER ESCALATION ──
    if not OPENROUTER_API_KEY:
        print("⚠️  [Audit] OpenRouter not configured (OPENROUTER_API_KEY missing).")
        _FALLBACK["_audit_metadata"] = _build_exhaustion_metadata()
        return _FALLBACK

    _or_model   = "google/gemini-2.5-flash"
    _or_timeout = 25   # fail fast rather than hang the UI
    _or_tokens  = 2048
    print(f"--- [Audit] Attempting OpenRouter Fallback (model={_or_model}, timeout={_or_timeout}s) ---")
    try:
        import requests as _requests  # lazy — only needed for OpenRouter fallback
        _or_payload = {
            "model": _or_model,
            "messages": [
                {"role": "system", "content": _AUDIT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": _or_tokens
        }
        resp = _requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=_or_payload,
            timeout=_or_timeout
        )
        # Some models via OpenRouter reject json_object mode — retry without it
        if resp.status_code == 400 and "response_format" in resp.text:
            print(f"[Audit] OpenRouter: model rejected json_object mode, retrying without.")
            _or_payload.pop("response_format", None)
            resp = _requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=_or_payload,
                timeout=_or_timeout
            )
        if resp.status_code == 200:
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            or_label = _or_model.split('/')[-1]
            try:
                audit = _repair_json(raw)
            except (ValueError, json.JSONDecodeError) as parse_err:
                _attempted.append({"label": or_label, "tier": "OR", "reason": "json_parse_error"})
                print(f"⚠️  [Audit] OpenRouter JSON repair failed: {parse_err}")
                _FALLBACK["_audit_metadata"] = _build_exhaustion_metadata()
                return _FALLBACK
            print(f"✅ [Audit] Quality audit complete via OpenRouter ({_or_model}).")
            final_audit = _apply_defensive_merge(audit)
            final_audit['_audit_metadata'] = {
                'model_id':    _or_model,
                'model_label': or_label,
                'tier':        'OR',
            }
            _AUDIT_CACHE[cache_key] = copy.deepcopy(final_audit)
            return final_audit
        else:
            or_label = _or_model.split('/')[-1]
            _attempted.append({"label": or_label, "tier": "OR", "reason": f"http_{resp.status_code}"})
            print(f"⚠️  [Audit] OpenRouter failed with {resp.status_code}: {resp.text[:300]}")
    except _requests.exceptions.Timeout:
        or_label = _or_model.split('/')[-1]
        _attempted.append({"label": or_label, "tier": "OR", "reason": "timeout"})
        print(f"⚠️  [Audit] OpenRouter timed out after {_or_timeout}s.")
    except Exception as e:
        or_label = _or_model.split('/')[-1]
        _attempted.append({"label": or_label, "tier": "OR", "reason": type(e).__name__})
        print(f"⚠️  [Audit] OpenRouter call failed: {e}")

    _FALLBACK["_audit_metadata"] = _build_exhaustion_metadata()
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
        
        # Extract & log which model scored this audit
        audit_meta  = audit.pop('_audit_metadata', {})
        model_label = audit_meta.get('model_label') or None
        tier        = audit_meta.get('tier')        or None
        attempted   = audit_meta.get('attempted_summary')
        print(f"[process-chat] Audit scored by [T{tier}] {model_label}")

        response_payload = {
            'success':      True,
            'type':         'chat',
            'audit_scored_by': model_label,
            'audit_tier':   tier,
            'original_text': text,
            'audit':        audit,
            'timestamp':    datetime.utcnow().isoformat() + 'Z',
        }
        if attempted:
            response_payload['audit_attempted_summary'] = attempted
        return jsonify(response_payload)
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
        
        # Extract & log which model scored this audit
        audit_meta  = audit.pop('_audit_metadata', {})
        model_label = audit_meta.get('model_label') or None
        tier        = audit_meta.get('tier')        or None
        attempted   = audit_meta.get('attempted_summary')
        print(f"[process-file] Audit scored by [T{tier}] {model_label}")

        response_payload = {
            'success':       True,
            'type':          'chat',
            'audit_scored_by': model_label,
            'audit_tier':    tier,
            'original_text': text,
            'audit':         audit,
            'timestamp':     datetime.utcnow().isoformat() + 'Z',
        }
        if attempted:
            response_payload['audit_attempted_summary'] = attempted
        return jsonify(response_payload)
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

        # Waterfall: HF Space → API chain (same on every environment)
        transcription = source_node = transcription_provider = None
        acoustic_profile = {}

        if not is_fast_track and HF_SPACE_URL and GRADIO_CLIENT_AVAILABLE:
            try:
                result = transcribe_via_hf_space(filepath)
                transcription    = result.get("transcript", "").strip()
                acoustic_profile = result.get("speaker_profiles", {})
                source_node      = "hf_space"
                transcription_provider = "Faster-Whisper + pyannote"
                print(f"[process-call] HF Space OK")
            except Exception as e:
                print(f"[process-call] HF Space failed, falling back: {e}")

        if not transcription:
            transcription, transcription_provider = perform_voice_capture_apis(filepath)
            source_node = "api_chain"

        try: os.remove(filepath)
        except Exception: pass

        print(f"[process-call] Auditing (source: {source_node})...")
        audit = generate_quality_audit(transcription, acoustic_profile=acoustic_profile)

        # Extract metadata — same contract as /api/job/<id>/status
        audit_meta  = audit.pop('_audit_metadata', {})
        model_label = audit_meta.get('model_label') or None
        tier        = audit_meta.get('tier')        or None
        attempted   = audit_meta.get('attempted_summary')

        response_payload = {
            'success':               True,
            'type':                  'call',
            'transcription':         transcription,
            'audit':                 audit,
            'source':                source_node,        # unified — matches job status response
            'source_node':           source_node,        # kept for any legacy consumers
            'transcription_provider': transcription_provider,
            'audit_scored_by':       model_label,
            'audit_tier':            tier,
            'acoustic_profile':      acoustic_profile,
            'timestamp':             datetime.utcnow().isoformat() + 'Z',
        }
        if attempted:
            response_payload['audit_attempted_summary'] = attempted
        return jsonify(response_payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/start-call-audit', methods=['POST'])
def start_call_audit():
    """
    Starts a transcription + audit job and streams SSE progress events back
    over a single long-lived HTTP connection (text/event-stream).

    This avoids cross-instance in-memory state loss on Vercel — the same
    function invocation that spawned the background threads is also the one
    delivering progress and the final result, staying alive for up to 290 s
    (inside the 300 s maxDuration limit).

    Event shapes:
      { type: 'job_started', job_id, fallbacks_available, hf_active }
      { type: 'status',      status, api_chain_started }
      { type: 'done',        result_type, transcription, audit, ... }
      { type: 'error',       error }
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
        job      = _start_job(filepath)
        job_id   = job['job_id']
        fallbacks = _get_fallbacks_available()
        hf_active = bool(HF_SPACE_URL and GRADIO_CLIENT_AVAILABLE)

        def _sse(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        def _event_stream():
            import time
            # Announce job — client stores job_id and may show "Transcribe Now"
            yield _sse({
                'type':               'job_started',
                'job_id':             job_id,
                'fallbacks_available': fallbacks,
                'hf_active':          hf_active,
            })

            # Stay alive until done, error, or 290 s deadline
            deadline = time.time() + 290
            while time.time() < deadline:
                time.sleep(2)
                j = _jobs.get(job_id)
                if not j:
                    yield _sse({'type': 'error', 'error': 'Job expired or not found.'})
                    return

                st = j.get('status')
                if st == 'done':
                    audit_copy  = copy.deepcopy(j['audit'] or {})
                    audit_meta  = audit_copy.pop('_audit_metadata', {})
                    _attemp_sum = audit_meta.get('attempted_summary')
                    payload = {
                        'type':                   'done',
                        'result_type':            'call',
                        'success':                True,
                        'transcription':          j['transcript'],
                        'audit':                  audit_copy,
                        'acoustic_profile':       j['acoustic_profile'],
                        'timestamp':              datetime.utcnow().isoformat() + 'Z',
                        'source':                 j['source'],
                        'audit_scored_by':        audit_meta.get('model_label') or None,
                        'audit_tier':             audit_meta.get('tier') or None,
                        'transcription_provider': j.get('transcription_provider') or j.get('source') or None,
                    }
                    if _attemp_sum:
                        payload['audit_attempted_summary'] = _attemp_sum
                    yield _sse(payload)
                    return

                elif st == 'error':
                    yield _sse({'type': 'error', 'error': j.get('error') or 'Transcription failed.'})
                    return

                else:
                    yield _sse({
                        'type':               'status',
                        'status':             st,
                        'api_chain_started':  j.get('api_chain_started', False),
                    })

            yield _sse({'type': 'error', 'error': 'Processing timed out after 290 s.'})

        return Response(
            stream_with_context(_event_stream()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control':    'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection':       'keep-alive',
            }
        )
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
        # deepcopy to avoid mutating cached/shared job audit dict
        audit_copy = copy.deepcopy(job['audit'] or {})
        audit_meta = audit_copy.pop('_audit_metadata', {})
        resp['success']                = True
        resp['type']                   = 'call'
        resp['transcription']          = job['transcript']
        resp['audit']                  = audit_copy
        resp['acoustic_profile']       = job['acoustic_profile']
        resp['timestamp']              = datetime.utcnow().isoformat() + 'Z'
        resp['audit_scored_by']        = audit_meta.get('model_label') or None
        resp['audit_tier']             = audit_meta.get('tier') or None
        resp['transcription_provider'] = job.get('transcription_provider') or job.get('source') or None
        _attempted_sum = audit_meta.get('attempted_summary')
        if _attempted_sum:
            resp['audit_attempted_summary'] = _attempted_sum
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


@app.route('/api/admin/clear-cache', methods=['POST'])
def admin_clear_cache():
    """
    Test/admin utility: clears the in-process audit result cache (_AUDIT_CACHE).
    Only callable from localhost to prevent abuse on production.
    Returns the number of entries evicted.
    """
    remote = request.remote_addr or ''
    if remote not in ('127.0.0.1', '::1', 'localhost'):
        return jsonify({'error': 'Forbidden — localhost only'}), 403
    evicted = len(_AUDIT_CACHE)
    _AUDIT_CACHE.clear()
    print(f'[admin] Audit cache cleared — {evicted} entries evicted.')
    return jsonify({'cleared': True, 'evicted': evicted})


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
        'vercel_mode': _ON_VERCEL,
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
