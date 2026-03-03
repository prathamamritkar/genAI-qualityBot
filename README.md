<div align="center">
  <img src="./favicon.svg" width="100" alt="Qualora Logo" />
  <h1>Qualora</h1>
  <p><strong>Turn every customer conversation into structured intelligence.</strong></p>
  <p><em>An enterprise-grade AI auditing platform that transcribes, analyses, and scores support interactions — across voice, text, and file — with zero single point of failure.</em></p>

  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](#)
  [![Flask](https://img.shields.io/badge/Backend-Flask-black?logo=flask&logoColor=white)](#)
  [![HuggingFace](https://img.shields.io/badge/ASR_Node-HuggingFace-yellow?logo=huggingface&logoColor=black)](#)
  [![Vercel](https://img.shields.io/badge/Deploy-Vercel-black?logo=vercel&logoColor=white)](#)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](#)
</div>

---

## What Qualora Does

You upload a call recording (or paste a chat transcript, or drop a file) and within minutes you get a fully structured audit: an agent F1 score, emotional timeline, compliance risk flag, empathy and listening scores, and actionable behavioural coaching nudges — all traceable back to the exact AI model that produced them.

No manual QA. No sampling bias. Every interaction scored consistently.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Audit Output](#audit-output)
3. [Input Modes](#input-modes)
4. [Transcription Architecture](#transcription-architecture)
5. [Audit Engine Cascade](#audit-engine-cascade)
6. [Getting Started](#getting-started)
7. [Hugging Face Space Setup](#hugging-face-space-setup)
8. [Deploying to Vercel](#deploying-to-vercel)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## How It Works

```
Upload audio / paste text / drop a file
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                  TRANSCRIPTION RACE                       │
│                                                           │
│  Thread A ──► HF Space (Faster-Whisper + pyannote)        │
│  Thread B ──► ElevenLabs Scribe                           │
│               └─► Deepgram Nova-2 (sequential fallback    │
│                   └─► Groq Whisper-large-v3  within B)    │
│                                                           │
│  First valid transcript wins. The other thread is dropped.│
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                    AUDIT CASCADE                          │
│                                                           │
│  T1 ─ Groq Llama 3.3 70B (primary, JSON mode)             │
│  T2 ─ Groq Llama 3.1 8B / Llama 4 Scout / Llama 4 Maverick│
│  T3 ─ Groq Kimi K2                                        │
│  T4 ─ OpenRouter Gemini 2.5 Flash  (final safety net)     │
│                                                           │
│  Every attempt is tracked — no hardcoded fallback strings.│
└───────────────────────────────────────────────────────────┘
        │
        ▼
  Structured JSON audit delivered via SSE stream
  (single open connection — no polling, no state loss)
```

The entire voice pipeline runs over a **single Server-Sent Events (SSE) connection**. The browser opens one long-lived request and receives live progress updates (`hf_transcribing → api_transcribing → auditing`) followed by the final result — no repeated polling, no cross-instance data loss.

---

## Audit Output

Every audit produces a deterministic JSON matrix:

```json
{
  "summary": "Customer reported a billing discrepancy; agent resolved without escalation.",
  "agent_f1_score": 0.91,
  "satisfaction_prediction": "High",
  "compliance_risk": "Green",
  "quality_matrix": {
    "language_proficiency": 9,
    "cognitive_empathy": 8,
    "efficiency": 9,
    "bias_reduction": 10,
    "active_listening": 9
  },
  "emotional_timeline": [
    { "turn": 1, "speaker": "Customer", "emotion": "Frustrated", "intensity": 8 },
    { "turn": 2, "speaker": "Agent",    "emotion": "Empathetic",  "intensity": 6 }
  ],
  "hitl_review_required": false,
  "behavioral_nudges": [
    "Mirroring: Repeat the specific billing date back to the customer earlier to validate their frustration sooner."
  ]
}
```

When the HF Space node wins the transcription race, real acoustic data (average pitch, intensity, detected speaker voiceprints from pyannote) is injected into the audit prompt — grounding the LLM's emotion inferences in biometric reality rather than text alone.

Results are **cached by SHA-256 hash** of the transcript. Identical inputs return identically deterministic results; the cache is cleared endpoint-by-endpoint during test runs.

---

## Input Modes

| Mode | Endpoint | Notes |
|---|---|---|
| **Voice upload** | `POST /api/start-call-audit` | MP3, WAV, M4A, WebM — up to 50 MB. SSE stream. |
| **Live recording** | `POST /api/start-call-audit` | Record directly in the browser via MediaRecorder. |
| **Text / chat transcript** | `POST /api/process-chat` | Paste raw `Agent: / Customer:` formatted text. |
| **File upload** | `POST /api/process-file` | `.txt` or `.pdf` (text-layer PDFs). |

---

## Transcription Architecture

Qualora runs two threads the moment an audio file lands:

**Thread A — HF Space (primary)**  
Streams the file to a private Hugging Face Space running `faster-whisper` (int8 quantisation) and `pyannote/speaker-diarization-3.1`. Returns a speaker-labelled transcript plus an acoustic profile (pitch, intensity, speaker voiceprints).

**Thread B — API Chain (concurrent fallback)**  
Runs sequentially *within* its thread: ElevenLabs Scribe → Deepgram Nova-2 → Groq Whisper-large-v3. Each provider is only tried if the previous one throws an exception.

**Automatic triggers for Thread B:**
- User clicks **"Transcribe Now"** — fires Thread B immediately while Thread A is still running
- Thread A throws any exception — Thread B starts automatically
- Thread A hasn't responded after **90 seconds** — timeout watcher starts Thread B

The winner (`job["winner"].is_set()`) locks the transcript. The other thread is abandoned. Provider name is recorded exactly (`"Faster-Whisper + pyannote"`, `"ElevenLabs Scribe"`, `"Deepgram Nova-2"`, `"Groq Whisper-large-v3"`) and surfaced in the audit console log and response payload.

---

## Audit Engine Cascade

The LLM auditor attempts models in order, tracking every attempt with a failure reason code:

| Tier | Model | Notes |
|---|---|---|
| T1 | Groq `llama-3.3-70b-versatile` | JSON mode. Primary choice. |
| T2 | Groq `llama-3.1-8b-instant` → `llama-4-scout` → `llama-4-maverick` | Sequential within tier. |
| T3 | Groq `moonshotai/kimi-k2-instruct` | |
| T4 | OpenRouter `google/gemini-2.5-flash` | Final safety net — different provider entirely. |

Failure reasons (`rate_limit`, `timeout`, `auth_error`, `json_parse_error`, `http_<code>`) are collected into `audit_attempted_summary` and returned in the response for full transparency. No hardcoded fallback strings at any tier.

---

## Getting Started

### Prerequisites
- Python 3.10+
- A Groq API key (free tier works — required for the audit engine)
- Optionally: ElevenLabs, Deepgram, OpenRouter, and HF Space keys for full cascade coverage

### Installation

```bash
git clone https://github.com/yourusername/qualora.git
cd qualora
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory. Values here **always override** OS environment variables (`load_dotenv(override=True)`):

```env
# ── Audit Engine (required) ───────────────────────────────
GROQ_API_KEY=gsk_your_groq_key_here

# ── Audit Fallback (recommended) ─────────────────────────
OPENROUTER_API_KEY=sk-or-your_openrouter_key_here

# ── Transcription API chain (optional — improves coverage) ─
ELEVENLABS_API_KEY=sk_your_elevenlabs_key_here
DEEPGRAM_API_KEY=your_deepgram_key_here

# ── HF Space ASR node (optional — enables diarization) ────
HF_SPACE_URL=your-hf-username/your-space-name
HF_SPACE_TOKEN=hf_your_read_token_here
```

> Only `GROQ_API_KEY` is strictly required. The platform degrades gracefully — missing providers are skipped, and Groq Whisper handles transcription if no other API keys are set.

### Run Locally

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Hugging Face Space Setup

The HF Space node provides free speaker diarization and acoustic profiling that the API chain cannot replicate.

1. Create a **Private Space** on Hugging Face using the **Docker** SDK.
2. Push the contents of the `hf_space/` directory (`app.py`, `requirements.txt`, `Dockerfile`) to your Space.
3. Add these **Repository Secrets** in your Space settings:
   - `HF_TOKEN` — your HuggingFace read token (needed to download pyannote weights)
   - `WHISPER_MODEL` — e.g. `medium` (recommended for free-tier cold boot speed) or `large-v3`
4. Accept the user agreement on the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) model page — the pipeline cannot download weights without this.
5. Set `HF_SPACE_URL` in your local `.env` to match your Space identifier (e.g. `your-username/qualora-asr`).

If the Space is unavailable or not configured, Qualora falls back to the API chain automatically — no manual intervention needed.

---

## Deploying to Vercel

Qualora deploys to Vercel using the `@vercel/python` legacy builder.

```json
{
  "version": 2,
  "builds": [{ "src": "app.py", "use": "@vercel/python" }],
  "routes": [{ "src": "/(.*)", "dest": "app.py" }]
}
```

**Important configuration steps:**

1. Add all environment variables from your `.env` file to **Vercel → Project Settings → Environment Variables**. The `VERCEL_ENV` variable is injected automatically by the platform (used internally to route file uploads to `/tmp`).

2. Set **Max Duration** in **Vercel → Project Settings → Functions** to the highest value your plan allows (60 s on Hobby, up to 800 s on Pro). The SSE stream stays open for the full transcription + audit cycle.

3. The SSE stream emits a `: ping` comment every 15 seconds to prevent Vercel's proxy layer from closing an idle connection mid-transcription.

> **Why SSE and not polling?**  
> On serverless platforms, each HTTP request can land on a different container instance. A polling approach would create a new instance per poll — none of which share the in-memory job state from the original upload instance. SSE keeps one connection open on the same instance from upload to final result delivery.

---

## API Reference

### `POST /api/start-call-audit`
Upload audio and receive a live SSE stream of progress events.

**Request:** `multipart/form-data` with field `audio` (file ≤ 50 MB).

**SSE Event stream:**
```
data: {"type": "job_started", "job_id": "abc123", "fallbacks_available": ["elevenlabs","groq"], "hf_active": true}

data: {"type": "status", "status": "hf_transcribing", "api_chain_started": false}

data: {"type": "status", "status": "auditing", "api_chain_started": false}

data: {"type": "done", "result_type": "call", "transcription": "...", "audit": {...},
       "audit_scored_by": "Llama 3.3 70B", "audit_tier": "T1",
       "transcription_provider": "Faster-Whisper + pyannote", "source": "hf_space", "timestamp": "..."}
```

On failure: `data: {"type": "error", "error": "...description..."}`

---

### `POST /api/job/<job_id>/transcribe-now`
Immediately starts the API chain fallback while the HF Space thread is still running. First valid result wins.

**Response:** `{"triggered": true, "providers": ["elevenlabs", "deepgram", "groq"]}`

---

### `POST /api/process-chat`
Audit a raw text transcript directly.

**Request:** `application/json` — `{"text": "Agent: Hello...\nCustomer: Hi..."}`

**Response:** Full audit payload including `audit_scored_by`, `audit_tier`, `audit_attempted_summary`.

---

### `POST /api/process-file`


Upload a `.txt` or `.pdf` file for auditing.

**Request:** `multipart/form-data` with field `file`.

---

### `GET /api/health`
Returns the configuration status of all nodes and providers.

```json
{
  "api_ready": true,
  "vercel_mode": false,
  "fallbacks": {
    "groq": true,
    "elevenlabs": false,
    "deepgram": false,
    "hf_space": true
  },
  "transcription_chain": ["groq"],
  "audit_chain": ["groq", "openrouter"]
}
```

---

### `POST /api/admin/clear-cache` *(localhost only)*
Evicts all entries from the in-process audit result cache. Used by the test suite to force fresh LLM calls between sections.

**Response:** `{"cleared": true, "evicted": 4}`

---

## Troubleshooting

**All speaker turns labelled "Speaker 0" (no diarization)**  
The HF Space node was unavailable and the API chain handled transcription. Only HF Space (pyannote) produces speaker separation. Configure `HF_SPACE_URL` and ensure the Space is running.

**Vercel SSE stream closes before a result arrives**  
Upgrade to Vercel Pro for longer function durations (the Hobby plan caps at 60 s). Alternatively, ensure ElevenLabs or Deepgram are configured so transcription completes faster and the audit starts sooner.

**`gradio_client` not installed warning on startup**  
The HF Space node is disabled but everything else works normally. Install with `pip install gradio_client>=1.3.0` if you want to enable it.

**HuggingFace pyannote download fails (`401 Unauthorized`)**  
You need to accept the user agreement on [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and set `HF_TOKEN` in your HF Space secrets.

**`.env` changes not being picked up**  
Qualora uses `load_dotenv(override=True)` — restart the server and your `.env` will take precedence over any stale OS-level environment variables.

---

## Running the Test Suite

The exhaustive test suite covers every endpoint, every fallback tier, and every provider — using real conversation samples from `datasets/human_chat.txt`. Server-side audit cache is cleared before each section so every test exercises a fresh LLM call.

```bash
# Server must be running first
python app.py &

python test_exhaustive.py
```

---

## License & Acknowledgements

MIT License. Built with gratitude to the open-source ML community:

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-optimised Whisper inference
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) — Speaker diarization
- [SpeechBrain](https://speechbrain.github.io/) — Acoustic emotion classification
- [Groq](https://groq.com) — Inference API for Llama and Whisper models
- [ElevenLabs](https://elevenlabs.io) — Scribe transcription with diarization
- [Deepgram](https://deepgram.com) — Nova-2 transcription

---

<div align="center"><em>Qualora — unlocking the human element of your support data.</em></div>
