"""
Briefly — Provider Diagnostic Script  (v2 — APIs confirmed via live introspection)
=====================================================================================
ElevenLabs : client.speech_to_text.convert(file=f, model_id='scribe_v1', diarize=True)
Deepgram   : client.listen.v1.media.transcribe_file(request=bytes, diarize=True, ...)
Groq       : client.audio.transcriptions.create(file=(...), model='whisper-large-v3')

Run:  C:\Python314\python.exe test_providers.py
Output written to: test_results.txt
"""

import os, sys, json, traceback, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATASETS_DIR = Path(__file__).parent / "datasets"
TEST_FILES   = [DATASETS_DIR / "call log 1.m4a", DATASETS_DIR / "call log 2.m4a"]
OUT_FILE     = Path(__file__).parent / "test_results.txt"

GROQ_KEY  = os.getenv("GROQ_API_KEY", "")
DG_KEY    = os.getenv("DEEPGRAM_API_KEY", "")
EL_KEY    = os.getenv("ELEVENLABS_API_KEY", "")

results = {}   # collected at end

# ── Output helpers ────────────────────────────────────────────────────────────
lines_buf = []

def w(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    lines_buf.append(line)

def hdr(title):
    w(); w("=" * 65); w(f"  {title}"); w("=" * 65)

def ok(label, detail=""):
    w(f"  PASS  {label}")
    if detail:
        for l in str(detail)[:600].splitlines(): w(f"        {l}")

def fail(label, detail=""):
    w(f"  FAIL  {label}")
    if detail:
        for l in str(detail)[:600].splitlines(): w(f"        {l}")

def skip(label, reason=""):
    w(f"  SKIP  {label}  [{reason}]")


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER 1 — ElevenLabs Scribe v1
# ─────────────────────────────────────────────────────────────────────────────
def test_elevenlabs(audio: Path):
    hdr(f"ElevenLabs speech_to_text.convert  [{audio.name}]")
    if not EL_KEY: skip("ElevenLabs", "ELEVENLABS_API_KEY not set"); return

    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=EL_KEY)

    w(f"\n  SDK attrs with 'speech': {[x for x in dir(client) if 'speech' in x.lower()]}")

    try:
        w("\n  Calling speech_to_text.convert(file=, model_id=scribe_v1, diarize=True)...")
        with open(audio, "rb") as f:
            resp = client.speech_to_text.convert(
                file=f,
                model_id="scribe_v1",
                diarize=True,
                language_code="en",
                timestamps_granularity="word",
            )

        # Inspect response
        attrs = [a for a in dir(resp) if not a.startswith("_")]
        w(f"  Response attrs: {attrs[:25]}")

        has_words = hasattr(resp, "words") and bool(resp.words)
        has_text  = hasattr(resp, "text")  and bool(getattr(resp, "text", None))
        w(f"  has_words={has_words}  has_text={has_text}")

        if has_words:
            words = resp.words
            # Check speaker_id
            sample = words[:5]
            for i, wrd in enumerate(sample):
                spk = getattr(wrd, "speaker_id", "N/A")
                txt = getattr(wrd, "text", getattr(wrd, "punctuated_word", "?"))
                w(f"    word[{i}] speaker_id={spk!r}  text={txt!r}")

            # Group by speaker
            turns = []
            cur_spk, cur_buf = None, []
            for wrd in words:
                spk = getattr(wrd, "speaker_id", None)
                txt = getattr(wrd, "text", "") or getattr(wrd, "punctuated_word", "")
                if spk != cur_spk:
                    if cur_spk is not None: turns.append((cur_spk, " ".join(cur_buf)))
                    cur_spk, cur_buf = spk, [txt]
                else:
                    cur_buf.append(txt)
            if cur_spk is not None: turns.append((cur_spk, " ".join(cur_buf)))

            unique_speakers = set(t[0] for t in turns)
            if len(unique_speakers) > 1:
                ok(f"ElevenLabs diarization — {len(unique_speakers)} distinct speakers, {len(turns)} turns")
                for spk, txt in turns[:3]:
                    w(f"    Speaker {spk}: {txt[:100]}")
                results["elevenlabs"] = "PASS_WITH_DIARIZATION"
            else:
                fail(f"ElevenLabs — only 1 speaker detected (diarize may not have worked)")
                w(f"    speakers found: {unique_speakers}")
                results["elevenlabs"] = "PASS_NO_DIARIZATION"
        elif has_text:
            fail("ElevenLabs — plain text only (no word-level diarization)")
            w(f"    text[:200]: {resp.text[:200]}")
            results["elevenlabs"] = "PASS_NO_DIARIZATION"
        else:
            fail("ElevenLabs — unrecognised response structure")
            w(f"    raw: {str(resp)[:300]}")
            results["elevenlabs"] = "FAIL"

    except Exception:
        fail("ElevenLabs — exception during call")
        w(traceback.format_exc())
        results["elevenlabs"] = "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER 2 — Deepgram Nova-2  (SDK v6 API)
# ─────────────────────────────────────────────────────────────────────────────
def test_deepgram(audio: Path):
    hdr(f"Deepgram Nova-2  (SDK v6)  [{audio.name}]")
    if not DG_KEY: skip("Deepgram", "DEEPGRAM_API_KEY not set"); return

    import deepgram as dg
    w(f"  deepgram version: {getattr(dg, '__version__', 'unknown')}")

    client = dg.DeepgramClient(api_key=DG_KEY)

    # Confirm path exists
    try:
        media = client.listen.v1.media
        w(f"  client.listen.v1.media  OK — methods: {[x for x in dir(media) if not x.startswith('_')]}")
    except Exception as e:
        fail("client.listen.v1.media not found", str(e)); results["deepgram"] = "FAIL"; return

    try:
        w("\n  Calling listen.v1.media.transcribe_file(diarize=True, utterances=True)...")
        with open(audio, "rb") as f:
            buf = f.read()

        resp = client.listen.v1.media.transcribe_file(
            request=buf,
            model="nova-2",
            smart_format=True,
            diarize=True,
            punctuate=True,
            utterances=True,
            language="en",
        )

        w(f"  Response type: {type(resp)}")
        has_results    = hasattr(resp, "results") and resp.results is not None
        has_utterances = has_results and hasattr(resp.results, "utterances") and bool(resp.results.utterances)
        has_channels   = has_results and hasattr(resp.results, "channels")   and bool(resp.results.channels)
        w(f"  has_results={has_results}  has_utterances={has_utterances}  has_channels={has_channels}")

        if has_utterances:
            utts = resp.results.utterances
            speakers = set(u.speaker for u in utts)
            w(f"  {len(utts)} utterances  |  {len(speakers)} speakers: {speakers}")
            for u in utts[:3]:
                w(f"    Speaker {u.speaker}: {u.transcript[:100]}")
            if len(speakers) > 1:
                ok(f"Deepgram diarization — {len(speakers)} speakers, {len(utts)} utterances")
                results["deepgram"] = "PASS_WITH_DIARIZATION"
            else:
                fail("Deepgram — only 1 speaker in utterances")
                results["deepgram"] = "PASS_NO_DIARIZATION"
        elif has_channels:
            alt = resp.results.channels[0].alternatives[0]
            fail("Deepgram — plain transcript only (no utterances/diarization)")
            w(f"    transcript[:200]: {alt.transcript[:200]}")
            results["deepgram"] = "PASS_NO_DIARIZATION"
        else:
            fail("Deepgram — no results, no channels")
            w(f"    raw: {str(resp)[:400]}")
            results["deepgram"] = "FAIL"

    except Exception:
        fail("Deepgram — exception during call")
        w(traceback.format_exc())
        results["deepgram"] = "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER 3 — Groq Whisper-large-v3
# ─────────────────────────────────────────────────────────────────────────────
def test_groq(audio: Path):
    hdr(f"Groq Whisper-large-v3  [{audio.name}]")
    if not GROQ_KEY: skip("Groq", "GROQ_API_KEY not set"); return

    from groq import Groq
    client = Groq(api_key=GROQ_KEY)

    try:
        w("\n  Calling audio.transcriptions.create(model=whisper-large-v3, verbose_json)...")
        with open(audio, "rb") as f:
            data = f.read()
        t = client.audio.transcriptions.create(
            file=(audio.name, data),
            model="whisper-large-v3",
            response_format="verbose_json",
            language="en",
            temperature=0.0,
        )
        text = getattr(t, "text", str(t))
        segs = getattr(t, "segments", [])
        w(f"  segments count: {len(segs)}")
        ok("Groq transcription (no diarization expected)")
        w(f"    text[:200]: {text[:200]}")
        results["groq"] = "PASS"
    except Exception:
        fail("Groq — exception")
        w(traceback.format_exc())
        results["groq"] = "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    w("=" * 65)
    w("  BRIEFLY — PROVIDER DIAGNOSTIC  (API-confirmed v2)")
    w(f"  Python: {sys.version.split()[0]}")
    w(f"  Keys:   EL={'OK' if EL_KEY else 'MISSING'}  "
      f"DG={'OK' if DG_KEY else 'MISSING'}  "
      f"GQ={'OK' if GROQ_KEY else 'MISSING'}")
    available = [f for f in TEST_FILES if f.exists()]
    w(f"  Files:  {[f.name for f in available]}")
    w("=" * 65)

    if not available:
        w("No audio files found in datasets/. Aborting."); sys.exit(1)

    audio = available[0]      # test against first available file

    test_elevenlabs(audio)
    test_deepgram(audio)
    test_groq(audio)

    # ── Summary ────────────────────────────────────────────────────────────
    w(); w("=" * 65); w("  RESULTS SUMMARY"); w("=" * 65)
    for provider, status in results.items():
        icon = "PASS" if "PASS" in status else "FAIL"
        w(f"  {icon:<6} {provider:<15} {status}")
    w("=" * 65)

    # Write to file for clean reading
    OUT_FILE.write_text("\n".join(lines_buf), encoding="utf-8")
    w(f"\n  Full output saved to: {OUT_FILE}")
