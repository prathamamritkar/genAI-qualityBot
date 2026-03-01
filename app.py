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

# ── [FIX #1] Deepgram SDK v6 — correct import with PrerecordedOptions ────────
from deepgram import DeepgramClient, PrerecordedOptions

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
    """ElevenLabs Scribe v2 — premium quality + speaker diarization."""
    with open(audio_path, "rb") as f:
        response = elevenlabs_client.scribe.transcribe(
            audio=f,
            model="scribe_v2",
            language="en",
            diarize=True,
        )

    if hasattr(response, 'segments') and response.segments:
        lines = []
        for seg in response.segments:
            speaker = f"Speaker {seg.speaker}" if hasattr(seg, 'speaker') else "Speaker"
            text    = seg.text if hasattr(seg, 'text') else str(seg)
            lines.append(f"{speaker}: {text}")
        if lines:
            return "\n\n".join(lines)

    if hasattr(response, 'text') and response.text:
        return response.text
    return str(response)


def _deepgram_transcribe(audio_path: str) -> str:
    """
    [FIX #1] Deepgram Nova-2 with acoustic speaker diarization.
    Corrected to use the PrerecordedOptions dataclass and the
    .listen.prerecorded.v("1").transcribe_file() method (SDK v3–v6 compatible).
    The original code used the invalid path .listen.v1.media.transcribe_file()
    which silently failed on every call.
    """
    with open(audio_path, "rb") as f:
        buffer_data = f.read()

    source = {"buffer": buffer_data}

    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        diarize=True,
        punctuate=True,
        utterances=True,
        language="en",
    )

    response = deepgram_client.listen.prerecorded.v("1").transcribe_file(
        source, options
    )

    # Utterances path — speaker-separated segments (richest output)
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

    # Channel alternatives path — plain transcript fallback
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

def generate_insight(text: str, content_type: str = "interaction") -> str:
    """Generate concise insight distillation — Groq primary, Deepgram fallback."""

    # Attempt 1: Groq (Llama-3.3-70b-versatile)
    if groq_client:
        try:
            print(f"--- [Distillation] Groq Llama ({content_type}) ---")
            MODEL_ID   = "llama-3.3-70b-versatile"
            CHUNK_SIZE = 100_000  # ~25k tokens — safe margin

            def _groq_complete(input_text: str) -> str:
                prompt = (
                    f"You are a customer service analyst. Provide a concise one-line summary "
                    f"of this {content_type}. Focus on the main topic, customer concern, or outcome.\n\n"
                    f"{content_type.capitalize()}: {input_text}\n\nOne-line summary:"
                )
                resp = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates concise one-line summaries of customer interactions."},
                        {"role": "user",   "content": prompt},
                    ],
                    model=MODEL_ID,
                    temperature=0.7,
                    max_tokens=150,
                )
                return resp.choices[0].message.content.strip()

            if len(text) <= CHUNK_SIZE:
                return _groq_complete(text)

            # Chunked processing for long transcripts
            print(f"[Distillation] Text {len(text)} chars — chunking...")
            chunks   = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
            summaries = []
            for i, chunk in enumerate(chunks):
                print(f"[Distillation] Chunk {i+1}/{len(chunks)}")
                try:
                    summaries.append(_groq_complete(chunk))
                except Exception as e:
                    print(f"[Distillation] Chunk {i+1} failed: {e}")
                    summaries.append("[chunk failed]")

            final_prompt = (
                f"You are a lead analyst. Synthesize these partial summaries of a long {content_type} "
                f"into one cohesive line.\n\nPartial Summaries:\n"
                + "\n".join(summaries)
                + "\n\nFinal One-line summary:"
            )
            final = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You consolidate multiple summaries into one concise insight."},
                    {"role": "user",   "content": final_prompt},
                ],
                model=MODEL_ID,
                temperature=0.7,
                max_tokens=200,
            )
            return final.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠️  [Groq Distillation] {e}")

    # Attempt 2: Deepgram Text Intelligence (fallback)
    if deepgram_client:
        try:
            print("--- [Distillation] Deepgram Text Intelligence ---")
            if len(text.split()) < 10:
                return text.strip()

            response = deepgram_client.read.v("1").analyze(
                {"buffer": text},
                {"summarize": True, "language": "en"},
            )
            if hasattr(response, 'results'):
                r = response.results
                if hasattr(r, 'summary') and r.summary:
                    return r.summary.short
                if hasattr(r, 'channels') and r.channels:
                    alt = r.channels[0].alternatives[0]
                    if hasattr(alt, 'summaries') and alt.summaries:
                        return alt.summaries[0].summary
        except Exception as e:
            print(f"⚠️  [Deepgram Distillation] {e}")

    raise RuntimeError("All distillation engines failed or are unconfigured.")

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

        summary = generate_insight(chat_text, "interaction")
        return jsonify({
            'success':       True,
            'type':          'chat',
            'original_text': chat_text,
            'summary':       summary,
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

        summary = generate_insight(text, "document")
        return jsonify({
            'success':       True,
            'type':          'chat',
            'original_text': text,
            'summary':       summary,
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
        print("[process-call] Distilling insight...")
        summary = generate_insight(transcription, "voice capture")

        return jsonify({
            'success':       True,
            'type':          'call',
            'transcription': transcription,
            'summary':       summary,
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
