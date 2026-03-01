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

def transcribe_via_hf_space(audio_path: str) -> str:
    """
    Submits audio to the HuggingFace Space (briefly-asr-node).
    The Space runs:  faster-whisper-large-v3  +  pyannote/speaker-diarization-3.1
    Returns speaker-labelled transcript string on success, raises on failure.
    """
    if not GRADIO_CLIENT_AVAILABLE:
        raise RuntimeError("gradio_client package not installed.")
    if not HF_SPACE_URL:
        raise RuntimeError("HF_SPACE_URL env var not set — HF node not configured.")

    print("--- [HF Space] Connecting to ASR node ---")
    client = GradioClient(
        HF_SPACE_URL,
        hf_token=HF_SPACE_TOKEN if HF_SPACE_TOKEN else None,
    )

    raw = client.predict(
        gradio_handle_file(audio_path),
        api_name="/predict",
    )

    # Gradio returns the function's return value — may be dict or JSON string
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw  # plain text transcript returned directly
    else:
        return str(raw)

    if "error" in data:
        raise RuntimeError(f"HF Space returned error: {data['error']}")

    transcript = data.get("transcript", "").strip()
    if not transcript:
        raise RuntimeError("HF Space returned empty transcript.")

    print("✅ [HF Space] Transcription complete.")
    return transcript

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

    raise RuntimeError("All API-chain transcription providers failed or are unconfigured.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — COMPETITIVE EXECUTION ENGINE
# "The Winner Stays" — both nodes fire simultaneously; first valid result wins.
# ─────────────────────────────────────────────────────────────────────────────

def competitive_transcribe(audio_path: str, file_size_bytes: int) -> tuple[str, str]:
    """
    Fires HF Space node and API-chain node simultaneously in separate threads.
    Returns (transcript_string, source_label) for whichever responds first
    with a valid non-empty result within the dynamic timeout.

    Dynamic timeout scales with file size:
        - API chain target window:  ~15–30 s  (fast, network-bound)
        - HF Space target window:   ~45–120 s (slow warm, but more capable)
        - Overall wait:             max of both (up to 120 s)

    On Vercel (IS_VERCEL=True): function timeout is 60 s (set in vercel.json).
    For files above VERCEL_SAFE_AUDIO_MB on Vercel, only the HF Space is tried
    (the API-chain upload itself would approach the Vercel body-size limit).
    """
    file_size_mb   = file_size_bytes / (1024 * 1024)
    overall_timeout = min(max(file_size_mb * 5, 45), 90)  # 45–90 s window
    
    # On Vercel with large file: skip API-chain (body too big for serverless)
    vercel_large_file = IS_VERCEL and file_size_mb > VERCEL_SAFE_AUDIO_MB

    winner = {"transcript": None, "source": None}
    event  = threading.Event()
    lock   = threading.Lock()

    def _attempt(fn, label):
        try:
            result = fn(audio_path)
            if result and result.strip():
                with lock:
                    if not event.is_set():
                        winner["transcript"] = result
                        winner["source"]     = label
                        event.set()
                        print(f"🏆 [Competitive] Winner: {label}")
        except Exception as e:
            print(f"[{label}] node failed: {e}")

    threads = []

    # Always launch HF Space thread if configured
    if HF_SPACE_URL and GRADIO_CLIENT_AVAILABLE:
        t = threading.Thread(
            target=_attempt,
            args=(transcribe_via_hf_space, "hf_space"),
            daemon=True,
        )
        threads.append(t)

    # Launch API-chain thread when safe to do so
    if not vercel_large_file:
        t = threading.Thread(
            target=_attempt,
            args=(perform_voice_capture_apis, "api_chain"),
            daemon=True,
        )
        threads.append(t)
    else:
        print(f"⚠️  [Vercel] File {file_size_mb:.1f}MB > {VERCEL_SAFE_AUDIO_MB}MB threshold — "
              "routing exclusively to HF Space node to avoid serverless timeout.")

    if not threads:
        raise RuntimeError(
            "No transcription nodes available — "
            "configure at least one of: HF_SPACE_URL, ELEVENLABS_API_KEY, DEEPGRAM_API_KEY, GROQ_API_KEY."
        )

    for t in threads:
        t.start()

    event.wait(timeout=overall_timeout)

    if winner["transcript"]:
        return winner["transcript"], winner["source"]

    raise RuntimeError(
        f"All transcription nodes failed or timed out after {overall_timeout:.0f}s. "
        "Check API keys and HF Space status."
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — INSIGHT DISTILLATION (unchanged logic, preserved fully)
# ─────────────────────────────────────────────────────────────────────────────

def generate_quality_audit(text: str, content_type: str = "interaction") -> dict:
    """
    Generate an enterprise-grade Quality Assurance Audit using Groq (Llama-3.3-70b-versatile).
    Strict JSON output enforced. Evaluates Empathy, Compliance, CSAT, Resolution, and Agent Quality.
    """
    if not groq_client:
        raise RuntimeError("Groq API client not configured. Required for QA Auditing.")

    print(f"--- [QA Audit] Groq Llama ({content_type}) ---")
    MODEL_ID = "llama-3.3-70b-versatile"

    system_prompt = (
        "You are an expert Principal AI QA Auditor evaluating a customer support interaction. "
        "Analyze the transcript carefully. Your task is to extract emotional states and evaluate the agent strictly "
        "across core pillars: Empathy, Compliance, and Issue Resolution.\n\n"
        "You MUST output exactly in JSON format, adhering to this schema:\n"
        "{\n"
        '  "thought_process": "Brief step-by-step reasoning of the interaction",\n'
        '  "emotion_timeline": [{"speaker": "...", "turn": 1, "emotion": "Angry/Frustrated/Neutral/Happy"}],\n'
        '  "empathy_score": <int 1-10>,\n'
        '  "compliance_status": "Pass" | "Fail" | "Partial",\n'
        '  "resolution_status": "Resolved" | "Unresolved" | "Escalated",\n'
        '  "csat_score": <int 1-10>,\n'
        '  "efficiency_score": <int 1-10>,\n'
        '  "violations": ["List of protocol violations, empty if none"],\n'
        '  "suggestions": ["Actionable feedback for the agent"],\n'
        '  "summary": "One-line interaction summary"\n'
        "}\n\n"
        "Be highly critical but fair. Default to 'Fail' for compliance if agent ignores security/policy. "
        "Ensure the JSON is perfectly valid."
    )

    user_prompt = f"Evaluate this {content_type}:\n\n{text}"

    try:
        resp = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=MODEL_ID,
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=1024,
        )
        content = resp.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"⚠️  [Groq QA Audit Failed] {e}")
        # Graceful fallback JSON if the LLM fails to parse or errors out
        return {
            "thought_process": "Error evaluating transcript.",
            "emotion_timeline": [],
            "empathy_score": 0,
            "compliance_status": "Fail",
            "resolution_status": "Unresolved",
            "csat_score": 0,
            "efficiency_score": 0,
            "violations": ["System error during evaluation"],
            "suggestions": ["Retry the audit API call"],
            "summary": "Audit failed due to system error."
        }

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
    """Process chat text and return summary."""
    try:
        data      = request.json or {}
        chat_text = data.get('text', '').strip()
        if not chat_text:
            return jsonify({'error': 'No content provided'}), 400

        audit = generate_quality_audit(chat_text, "interaction")
        return jsonify({
            'success':       True,
            'type':          'chat',
            'original_text': chat_text,
            'audit':         audit,
            'timestamp':     datetime.utcnow().isoformat() + 'Z',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process-file', methods=['POST'])
def process_file():
    """Process uploaded text/PDF document and return summary."""
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

        audit = generate_quality_audit(text, "document")
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
    Process audio call via Competitive Execution:
    HF Space (acoustic diarization) races API chain (ElevenLabs → Deepgram → Groq).
    The first valid transcript wins.

    [FIX #2] File size is checked before processing.
    On Vercel with files > VERCEL_SAFE_AUDIO_MB, the request is routed exclusively
    to the HF Space node to avoid the serverless 60-second timeout.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if not audio_file.filename:
            return jsonify({'error': 'No file selected'}), 400

        # Save to temp location
        safe_name  = secure_filename(audio_file.filename)
        filename   = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        filepath   = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(filepath)
        file_size  = os.path.getsize(filepath)

        # [FIX #2] — Runtime file-size guard
        MAX_BYTES = 50 * 1024 * 1024  # 50 MB absolute ceiling
        if file_size > MAX_BYTES:
            os.remove(filepath)
            return jsonify({
                'error': f'File exceeds 50 MB ceiling ({file_size / 1e6:.1f} MB). '
                         'Please compress or split the audio.'
            }), 413

        print(f"[process-call] {filename} ({file_size / 1e6:.2f} MB) — entering competitive engine")

        # ── Competitive Execution ──────────────────────────────────────────
        try:
            transcription, source_node = competitive_transcribe(filepath, file_size)
        finally:
            try:
                os.remove(filepath)
            except Exception:
                pass

        # ── Distillation ───────────────────────────────────────────────────
        print("[process-call] Generating QA Audit...")
        audit = generate_quality_audit(transcription, "voice capture")

        return jsonify({
            'success':       True,
            'type':          'call',
            'transcription': transcription,
            'audit':         audit,
            'source_node':   source_node,   # "hf_space" | "api_chain"
            'timestamp':     datetime.utcnow().isoformat() + 'Z',
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    print("🚀  Briefly Signal Hub — Distributed Hybrid Mode")
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
