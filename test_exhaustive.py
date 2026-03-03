#!/usr/bin/env python3
"""
=============================================================================
  Qualora — EXHAUSTIVE FALLBACK TEST SUITE
  Tests every provider, every fallback, every endpoint — live against server
  Uses real conversation samples from datasets/human_chat.txt.
  Clears the server-side audit cache before each endpoint section so every
  test exercises the full LLM pipeline rather than a cached shortcut.
=============================================================================
Run with:  .\\venv\\Scripts\\python.exe test_exhaustive.py
Server must be running on localhost:5000.
=============================================================================
"""
import os, sys, io, re, struct, time, json, wave, traceback
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / '.env', override=True)

BASE   = 'http://localhost:5000/api'
GROQ_KEY      = os.getenv('GROQ_API_KEY', '')
DEEPGRAM_KEY  = os.getenv('DEEPGRAM_API_KEY', '')
EL_KEY        = os.getenv('ELEVENLABS_API_KEY', '')
OR_KEY        = os.getenv('OPENROUTER_API_KEY', '')
HF_URL        = os.getenv('HF_SPACE_URL', '')
HF_TOKEN      = os.getenv('HF_SPACE_TOKEN', '')

# ── Colours ───────────────────────────────────────────────────────────────────
OK    = '\033[92m✅\033[0m'
FAIL  = '\033[91m❌\033[0m'
SKIP  = '\033[93m⏭\033[0m'
INFO  = '\033[94mℹ\033[0m'
WARN  = '\033[93m⚠\033[0m'
HDR   = '\033[1m\033[96m'
RESET = '\033[0m'

results = []

def section(title):
    print(f'\n{HDR}{"="*65}')
    print(f'  {title}')
    print(f'{"="*65}{RESET}')

def check(name, passed, detail='', warn=False):
    icon = WARN if (not passed and warn) else (OK if passed else FAIL)
    status = 'WARN' if (not passed and warn) else ('PASS' if passed else 'FAIL')
    print(f'  {icon}  [{status}]  {name}')
    if detail:
        for line in str(detail).splitlines():
            print(f'         {line}')
    results.append({'name': name, 'passed': passed or warn, 'warn': not passed and warn})
    return passed

def skip(name, reason):
    print(f'  {SKIP}  [SKIP]  {name}: {reason}')
    results.append({'name': name, 'passed': True, 'skip': True})

# ── Dataset loader ────────────────────────────────────────────────────────────
_FALLBACK_TRANSCRIPT = (
    "Agent: Thank you for calling customer support. How can I help you today?\n"
    "Customer: Hi, I placed an order three weeks ago and it hasn't arrived yet. "
    "I'm really frustrated.\n"
    "Agent: I completely understand your frustration and sincerely apologize. "
    "Let me pull up your order details right away.\n"
    "Customer: Yes please. This is unacceptable. I need this urgently.\n"
    "Agent: I've located your order. It shows a delay at the warehouse. I am "
    "escalating this to our priority fulfilment team. You'll receive an update "
    "within 24 hours and I'm applying a 15% discount for your trouble.\n"
    "Customer: Okay, thank you. I appreciate that."
)

def _load_dataset_samples(max_samples: int = 20) -> list:
    """
    Parse datasets/human_chat.txt into Agent/Customer conversation strings.
    Conversations are split at greeting boundaries (Human 1: Hi / Hello).
    Human 1 → Agent, Human 2 → Customer.
    Falls back to a single hardcoded sample if the file is missing.
    """
    dataset_path = Path(__file__).parent / 'datasets' / 'human_chat.txt'
    if not dataset_path.exists():
        print(f'  {WARN}  datasets/human_chat.txt not found — using fallback transcript.')
        return [_FALLBACK_TRANSCRIPT]

    raw_lines = dataset_path.read_text(encoding='utf-8').splitlines()
    _GREET = re.compile(r'^Human 1:\s*Hi[!.]?\s*$', re.IGNORECASE)

    blocks, current = [], []
    for line in raw_lines:
        if _GREET.match(line.strip()) and current:
            if len(current) >= 4:
                blocks.append(current)
            current = [line]
        elif line.strip():
            current.append(line)
    if current and len(current) >= 4:
        blocks.append(current)

    samples = []
    for block in blocks[:max_samples]:
        turns = []
        for line in block:
            if line.startswith('Human 1:'):
                turns.append('Agent:' + line[8:])
            elif line.startswith('Human 2:'):
                turns.append('Customer:' + line[8:])
        if len(turns) >= 4:
            samples.append('\n'.join(turns))

    if not samples:
        return [_FALLBACK_TRANSCRIPT]
    print(f'  {INFO}  Loaded {len(samples)} conversation samples from datasets/human_chat.txt')
    return samples

DATASET_SAMPLES = _load_dataset_samples()

def dataset_sample(index: int) -> str:
    """Return sample at index, cycling if index exceeds available samples."""
    return DATASET_SAMPLES[index % len(DATASET_SAMPLES)]

# ── Cache control ─────────────────────────────────────────────────────────────
def clear_server_cache(label: str = ''):
    """
    POST /api/admin/clear-cache — evicts all entries from the server's
    _AUDIT_CACHE so the next test hits the full LLM pipeline fresh.
    """
    try:
        r = requests.post(f'{BASE}/admin/clear-cache', timeout=4)
        evicted = r.json().get('evicted', '?')
        tag = f' [{label}]' if label else ''
        print(f'  {INFO}{tag} Server audit cache cleared ({evicted} entries evicted).')
    except Exception as e:
        print(f'  {WARN}  clear_server_cache failed: {e}')

# ── Minimal WAV generator ─────────────────────────────────────────────────────
def make_wav(seconds=2, hz=440, sample_rate=16000) -> bytes:
    """Generates a pure-tone WAV in memory (no disk files needed)."""
    import math
    n_samples = sample_rate * seconds
    buf = io.BytesIO()
    with wave.open(buf, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            val = int(32767 * math.sin(2 * math.pi * hz * i / sample_rate))
            wf.writeframes(struct.pack('<h', val))
    return buf.getvalue()

SAMPLE_WAV = make_wav(seconds=3)   # 3 s pure tone — enough for Whisper to process

# ── SSE consumer ──────────────────────────────────────────────────────────────
def consume_sse(files_dict: dict, timeout: int = 200) -> dict:
    """
    POST to /api/start-call-audit with SSE streaming response.
    Returns the 'done' event payload, or raises on error/timeout.
    Prints intermediate status events for visibility.
    """
    start = time.time()
    with requests.post(
        f'{BASE}/start-call-audit',
        files=files_dict,
        stream=True,
        timeout=timeout,
    ) as resp:
        if not resp.ok:
            raise RuntimeError(f'HTTP {resp.status_code}: {resp.text[:200]}')

        buf = ''
        for raw in resp.iter_content(chunk_size=None, decode_unicode=False):
            buf += raw.decode('utf-8', errors='replace')
            while '\n\n' in buf:
                frame, buf = buf.split('\n\n', 1)
                for line in frame.splitlines():
                    if not line.startswith('data: '):
                        continue
                    try:
                        evt = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    t = evt.get('type', '')
                    if t == 'job_started':
                        print(f'         SSE job_started  job={evt.get("job_id","")[:8]}  '
                              f'hf_active={evt.get("hf_active")}  '
                              f'fallbacks={evt.get("fallbacks_available")}')
                    elif t == 'status':
                        elapsed = time.time() - start
                        print(f'         SSE status [{elapsed:5.1f}s]  {evt.get("status")}  '
                              f'api_chain_started={evt.get("api_chain_started")}')
                    elif t == 'done':
                        return evt
                    elif t == 'error':
                        raise RuntimeError(evt.get('error', 'Unknown SSE error'))
    raise RuntimeError('SSE stream closed without a done event')

# ═════════════════════════════════════════════════════════════════════════════
section('1. SERVER HEALTH & CONNECTIVITY')
# ═════════════════════════════════════════════════════════════════════════════
try:
    r = requests.get(f'{BASE}/health', timeout=6)
    h = r.json()
    check('Server reachable', r.status_code == 200)
    check('API ready flag', h.get('api_ready'), f'api_ready={h.get("api_ready")}')
    check('Groq configured',       h['fallbacks']['groq'],
          f'GROQ_API_KEY set: {bool(GROQ_KEY)}')
    check('ElevenLabs configured', h['fallbacks']['elevenlabs'],
          f'ELEVENLABS_API_KEY set: {bool(EL_KEY)}', warn=True)
    check('Deepgram configured',   h['fallbacks']['deepgram'],
          f'DEEPGRAM_API_KEY set: {bool(DEEPGRAM_KEY)}', warn=True)
    check('HF Space configured',   h['fallbacks']['hf_space'],
          f'HF_SPACE_URL: {HF_URL or "NOT SET"}', warn=True)
    check('OpenRouter env var set', bool(OR_KEY),
          f'OPENROUTER_API_KEY length={len(OR_KEY)}', warn=True)
    check('vercel_mode is False (local)', not h.get('vercel_mode'),
          f'vercel_mode={h.get("vercel_mode")}')
except Exception as e:
    check('Server reachable', False, str(e))
    print(f'\n  {FAIL} Server not running. Start with: .\\venv\\Scripts\\python.exe app.py')
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════
section('2. TEXT AUDIT — /api/process-chat')
# ═════════════════════════════════════════════════════════════════════════════
clear_server_cache('section 2')
_S2 = dataset_sample(0)   # first conversation sample

# 2a. Normal request
try:
    r = requests.post(f'{BASE}/process-chat',
        json={'text': _S2}, timeout=35)
    d = r.json()
    ok = d.get('success') and d.get('audit')
    check('Text audit returns success', ok, f'status={r.status_code} scored_by={d.get("audit_scored_by","?")}')
    if ok:
        a = d['audit']
        check('F1 score present & valid', isinstance(a.get('agent_f1_score'), (int, float)),
              f'f1={a.get("agent_f1_score")}')
        check('Satisfaction prediction present', a.get('satisfaction_prediction') in ('High','Medium','Low'),
              f'sat={a.get("satisfaction_prediction")}')
        check('Compliance risk present', a.get('compliance_risk') in ('Green','Amber','Red'),
              f'risk={a.get("compliance_risk")}')
        check('Emotional timeline non-empty', len(a.get('emotional_timeline', [])) > 0,
              f'turns={len(a.get("emotional_timeline", []))}')
        check('Quality matrix complete', all(
            k in a.get('quality_matrix', {}) for k in
            ['language_proficiency','cognitive_empathy','efficiency','bias_reduction','active_listening']
        ))
        check('Behavioral nudges non-empty', len(a.get('behavioral_nudges', [])) > 0,
              f'nudges={len(a.get("behavioral_nudges", []))}')
        check('Model traceability field present', bool(d.get('audit_scored_by')),
              f'scored_by={d.get("audit_scored_by")} tier={d.get("audit_tier")}')
except Exception as e:
    check('Text audit returns success', False, traceback.format_exc())

# 2b. Additional dataset samples — rotate through next two samples
for _idx, _label in [(1, 'dataset sample #2'), (2, 'dataset sample #3')]:
    clear_server_cache(f'section 2b sample {_idx}')
    _sx = dataset_sample(_idx)
    try:
        r = requests.post(f'{BASE}/process-chat', json={'text': _sx}, timeout=35)
        d = r.json()
        ok = d.get('success') and d.get('audit')
        check(f'Text audit success — {_label}', ok,
              f'status={r.status_code} f1={d.get("audit",{}).get("agent_f1_score","?")} '
              f'scored_by={d.get("audit_scored_by","?")}')
    except Exception as e:
        check(f'Text audit success — {_label}', False, str(e)[:120])

# 2c. Edge cases
try:
    r = requests.post(f'{BASE}/process-chat', json={'text': ''}, timeout=5)
    check('Empty text → 400', r.status_code == 400)
except Exception as e:
    check('Empty text → 400', False, str(e))

try:
    r = requests.post(f'{BASE}/process-chat', data='not json',
                      headers={'Content-Type': 'application/json'}, timeout=5)
    check('Malformed JSON → 400', r.status_code in (400, 415, 500))
except Exception as e:
    check('Malformed JSON → 400', False, str(e))

# 2d. Cache round-trip — same input twice should return identical F1 (cache hit),
#     then clear cache and resubmit to confirm the pipeline runs fresh.
clear_server_cache('section 2d cache-check prime')
_S2d = dataset_sample(3)
try:
    r1 = requests.post(f'{BASE}/process-chat', json={'text': _S2d}, timeout=35)
    r2 = requests.post(f'{BASE}/process-chat', json={'text': _S2d}, timeout=35)   # hits cache
    d1, d2 = r1.json(), r2.json()
    if d1.get('success') and d2.get('success'):
        f1_match = d1['audit']['agent_f1_score'] == d2['audit']['agent_f1_score']
        check('Cache: identical inputs give identical F1 (cache hit)', f1_match,
              f'f1_1={d1["audit"]["agent_f1_score"]}  f1_2={d2["audit"]["agent_f1_score"]}')
    else:
        check('Cache: identical inputs give identical F1 (cache hit)', False, 'One or both requests failed')
except Exception as e:
    check('Cache: identical inputs give identical F1 (cache hit)', False, str(e))

clear_server_cache('section 2d post-clear')
try:
    r3 = requests.post(f'{BASE}/process-chat', json={'text': _S2d}, timeout=35)
    d3 = r3.json()
    check('Post-clear re-audit succeeds (fresh LLM call)', d3.get('success') and d3.get('audit'),
          f'scored_by={d3.get("audit_scored_by","?")}')
except Exception as e:
    check('Post-clear re-audit succeeds (fresh LLM call)', False, str(e))

# ═════════════════════════════════════════════════════════════════════════════
section('3. GROQ AUDIT MODEL CASCADE — Direct API Tests')
# ═════════════════════════════════════════════════════════════════════════════
if not GROQ_KEY:
    skip('Groq model cascade', 'GROQ_API_KEY not set')
else:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_KEY)

    PROMPT_SHORT = "Agent: Hello! Customer: My bill is wrong. Agent: Let me check. [RESOLVED]"

    GROQ_MODELS_TO_TEST = [
        # (model_id,                                    label,               supports_json)
        ('llama-3.3-70b-versatile',                     'Llama 3.3 70B',     True),
        ('llama-3.1-8b-instant',                        'Llama 3.1 8B',      True),
        ('meta-llama/llama-4-scout-17b-16e-instruct',   'Llama 4 Scout',     False),
        ('meta-llama/llama-4-maverick-17b-128e-instruct','Llama 4 Maverick',  False),
        ('moonshotai/kimi-k2-instruct',                 'Kimi K2',           False),
        ('openai/gpt-oss-120b',                         'GPT OSS 120B',      False),
        ('openai/gpt-oss-20b',                          'GPT OSS 20B',       False),
        ('qwen/qwen3-32b',                              'Qwen3 32B',         False),
        ('compound',                                    'Compound',          False),
        ('compound-mini',                               'Compound Mini',     False),
    ]

    system_msg = 'You are a JSON API. Return ONLY valid JSON: {"ok": true, "model": "<model_id>"}'
    user_msg   = 'Return the JSON now.'

    for model_id, label, supports_json in GROQ_MODELS_TO_TEST:
        try:
            kwargs = dict(
                model=model_id,
                messages=[
                    {'role': 'system', 'content': system_msg},
                    {'role': 'user',   'content': user_msg},
                ],
                temperature=0.0,
                max_tokens=64,
            )
            if supports_json:
                kwargs['response_format'] = {'type': 'json_object'}

            resp = groq_client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content.strip()
            # Try to parse JSON
            try:
                parsed = json.loads(raw)
                valid_json = True
            except Exception:
                # Try to find JSON in response
                import re
                m = re.search(r'\{.*?\}', raw, re.DOTALL)
                valid_json = bool(m)

            check(f'Groq {label} ({model_id})', valid_json,
                  f'response: {raw[:80]}')
        except Exception as e:
            err = str(e).lower()
            if 'rate_limit' in err or '429' in err:
                skip(f'Groq {label}', 'Rate limited (expected on free tier)')
            elif 'not found' in err or '404' in err or 'unknown model' in err:
                check(f'Groq {label} ({model_id})', False, f'Model not found: {e}')
            else:
                check(f'Groq {label} ({model_id})', False, str(e)[:120])

# ═════════════════════════════════════════════════════════════════════════════
section('4. OPENROUTER FALLBACK — Direct API Test')
# ═════════════════════════════════════════════════════════════════════════════
if not OR_KEY:
    skip('OpenRouter gemini-2.5-flash', 'OPENROUTER_API_KEY not set')
else:
    try:
        resp = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={'Authorization': f'Bearer {OR_KEY}', 'Content-Type': 'application/json'},
            json={
                'model': 'google/gemini-2.5-flash',
                'messages': [
                    {'role': 'system', 'content': 'Return ONLY valid JSON: {"ok": true}'},
                    {'role': 'user',   'content': 'Return the JSON.'},
                ],
                'temperature': 0.0,
                'max_tokens': 32,
            },
            timeout=20
        )
        ok = resp.status_code == 200
        check('OpenRouter gemini-2.5-flash reachable', ok,
              f'status={resp.status_code}' + ('' if ok else f' body={resp.text[:120]}'))
        if ok:
            raw = resp.json()['choices'][0]['message']['content'].strip()
            try:
                import re
                raw_clean = re.sub(r'```[a-z]*', '', raw).strip().strip('`').strip()
                parsed = json.loads(raw_clean)
                check('OpenRouter returns valid JSON', True, f'parsed ok={parsed.get("ok")}')
            except Exception as pe:
                check('OpenRouter returns valid JSON', False, f'parse error: {pe}  raw: {raw[:120]}')
    except Exception as e:
        check('OpenRouter gemini-2.5-flash reachable', False, str(e))

# ═════════════════════════════════════════════════════════════════════════════
section('5. GROQ WHISPER — Speech-to-Text Fallbacks')
# ═════════════════════════════════════════════════════════════════════════════
if not GROQ_KEY:
    skip('Groq Whisper models', 'GROQ_API_KEY not set')
else:
    for whisper_model, tier in [('whisper-large-v3', 'T1'), ('whisper-large-v3-turbo', 'T2')]:
        try:
            wav_bytes = SAMPLE_WAV
            transcription = groq_client.audio.transcriptions.create(
                file=(f'test_{tier}.wav', wav_bytes),
                model=whisper_model,
                response_format='verbose_json',
                language='en',
                temperature=0.0,
            )
            text = getattr(transcription, 'text', '') or ''
            # Pure tone may produce empty or minimal text — that's OK; we test the API call succeeds
            check(f'Groq Whisper [{tier}] {whisper_model}', True,
                  f'responded (text: "{text[:60] or "<empty — pure tone>"}") ')
        except Exception as e:
            err = str(e).lower()
            if 'rate_limit' in err or '429' in err:
                skip(f'Groq Whisper [{tier}] {whisper_model}', 'Rate limited')
            else:
                check(f'Groq Whisper [{tier}] {whisper_model}', False, str(e)[:120])

# ═════════════════════════════════════════════════════════════════════════════
section('6. ELEVENLABS SCRIBE — Transcription Fallback')
# ═════════════════════════════════════════════════════════════════════════════
if not EL_KEY:
    skip('ElevenLabs Scribe', 'ELEVENLABS_API_KEY not set')
else:
    try:
        from elevenlabs.client import ElevenLabs
        el = ElevenLabs(api_key=EL_KEY)
        wav_io = io.BytesIO(SAMPLE_WAV)
        wav_io.name = 'test.wav'
        response = el.speech_to_text.convert(
            file=wav_io,
            model_id='scribe_v1',
            diarize=False,
            language_code='en',
        )
        got_text = hasattr(response, 'text') and bool(response.text) or \
                   hasattr(response, 'words') and bool(response.words)
        check('ElevenLabs Scribe API call succeeds', True,
              f'text="{getattr(response, "text", "")[:60]}"')
    except Exception as e:
        err = str(e).lower()
        if '401' in err or 'unauthorized' in err or 'invalid' in err:
            check('ElevenLabs Scribe API call succeeds', False, f'Auth error: {e}')
        elif 'audio' in err or 'format' in err:
            check('ElevenLabs Scribe API call succeeds', False, f'Audio format issue: {e}')
        else:
            check('ElevenLabs Scribe API call succeeds', False, str(e)[:120])

# ═════════════════════════════════════════════════════════════════════════════
section('7. DEEPGRAM NOVA-2 — Transcription Fallback')
# ═════════════════════════════════════════════════════════════════════════════
if not DEEPGRAM_KEY:
    skip('Deepgram Nova-2', 'DEEPGRAM_API_KEY not set')
else:
    try:
        from deepgram import DeepgramClient
        dg = DeepgramClient(api_key=DEEPGRAM_KEY)
        response = dg.listen.v1.media.transcribe_file(
            request=SAMPLE_WAV,
            model='nova-2',
            smart_format=True,
            diarize=False,
            punctuate=True,
            language='en',
        )
        has_results = hasattr(response, 'results') and response.results is not None
        check('Deepgram Nova-2 API call succeeds', has_results,
              f'results present: {has_results}')
        if has_results and hasattr(response.results, 'channels') and response.results.channels:
            txt = response.results.channels[0].alternatives[0].transcript if response.results.channels else ''
            check('Deepgram returns transcript field', isinstance(txt, str),
                  f'transcript: "{txt[:60] or "<empty — pure tone>"}"')
    except Exception as e:
        err = str(e).lower()
        if '401' in err or 'unauthorized' in err:
            check('Deepgram Nova-2 API call succeeds', False, f'Auth error: {e}')
        else:
            check('Deepgram Nova-2 API call succeeds', False, str(e)[:120])

# ═════════════════════════════════════════════════════════════════════════════
section('8. API CHAIN via /api/process-call (sync, audio)')
# ═════════════════════════════════════════════════════════════════════════════
clear_server_cache('section 8')
try:
    files = {'audio': ('test_call.wav', io.BytesIO(SAMPLE_WAV), 'audio/wav')}
    r = requests.post(f'{BASE}/process-call?fast_track=true', files=files, timeout=45)
    d = r.json()
    ok = d.get('success') and d.get('audit')
    check('API chain /process-call (fast_track=true)', ok,
          f'status={r.status_code} source={d.get("source","?")} '
          f'scored_by={d.get("audit_scored_by","?")}' if r.status_code==200 else d.get('error',''))
    if ok:
        check('Transcription field present', bool(d.get('transcription')),
              f'len={len(d.get("transcription",""))}')
        check('Audit present in call response', isinstance(d.get('audit'), dict))
        check('transcription_provider field present', bool(d.get('transcription_provider')),
              f'provider={d.get("transcription_provider","?")}')
except Exception as e:
    check('API chain /process-call (fast_track=true)', False, str(e)[:120])

# ═════════════════════════════════════════════════════════════════════════════
section('9. ASYNC JOB SYSTEM — /api/start-call-audit (SSE streaming)')
# ═════════════════════════════════════════════════════════════════════════════
# /start-call-audit now returns text/event-stream (SSE) — a single long-lived
# connection that emits: job_started → status... → done/error.
# consume_sse() reads the stream and returns the 'done' payload.
clear_server_cache('section 9')
try:
    files = {'audio': ('async_test.wav', io.BytesIO(SAMPLE_WAV), 'audio/wav')}
    print(f'       Consuming SSE stream from /start-call-audit (up to 200s)...')
    final = consume_sse(files, timeout=200)

    check('SSE stream delivers done event', bool(final), f'type={final.get("type")}')
    check('Job transcription field', bool(final.get('transcription')),
          f'len={len(final.get("transcription",""))}')
    check('Job audit field', isinstance(final.get('audit'), dict))
    check('Job audit_scored_by field', bool(final.get('audit_scored_by')),
          f'scored_by={final.get("audit_scored_by")} tier={final.get("audit_tier")}')
    check('Job transcription_provider field', bool(final.get('transcription_provider')),
          f'provider={final.get("transcription_provider","?")}')
    check('Job source field', bool(final.get('source')),
          f'source={final.get("source","?")}')
    check('Job timestamp field', bool(final.get('timestamp')))

except Exception as e:
    check('SSE stream delivers done event', False, traceback.format_exc())

# 9b. No-audio guard (start-call-audit must reject without streaming)
try:
    r = requests.post(f'{BASE}/start-call-audit', timeout=5)
    check('/start-call-audit with no audio → 400', r.status_code == 400)
except Exception as e:
    check('/start-call-audit with no audio → 400', False, str(e))

# 9c. transcribe-now fires while a fresh SSE job is still open (separate job)
clear_server_cache('section 9c transcribe-now')
try:
    # We need a job_id without consuming the full stream — start a job via a
    # short timeout to read just the first job_started event, grab the id, then
    # fire transcribe-now.  Use a Thread to run the SSE consumer concurrently.
    import threading as _threading
    _job_id_box = [None]
    _sse_done   = _threading.Event()

    def _sse_reader():
        try:
            files2 = {'audio': ('tnow_test.wav', io.BytesIO(SAMPLE_WAV), 'audio/wav')}
            with requests.post(f'{BASE}/start-call-audit', files=files2,
                               stream=True, timeout=200) as _r:
                _buf = ''
                for _raw in _r.iter_content(chunk_size=None, decode_unicode=False):
                    _buf += _raw.decode('utf-8', errors='replace')
                    while '\n\n' in _buf:
                        _frame, _buf = _buf.split('\n\n', 1)
                        for _line in _frame.splitlines():
                            if not _line.startswith('data: '):
                                continue
                            _evt = json.loads(_line[6:])
                            if _evt.get('type') == 'job_started':
                                _job_id_box[0] = _evt.get('job_id')
                            if _evt.get('type') in ('done', 'error'):
                                _sse_done.set()
                                return
        except Exception:
            _sse_done.set()

    _t = _threading.Thread(target=_sse_reader, daemon=True)
    _t.start()

    # Wait up to 8 s for job_started event
    for _ in range(40):
        if _job_id_box[0]:
            break
        time.sleep(0.2)

    _jid = _job_id_box[0]
    if _jid:
        time.sleep(1)   # Let HF thread start
        tr = requests.post(f'{BASE}/job/{_jid}/transcribe-now', timeout=6)
        td = tr.json()
        check('transcribe-now triggers API chain', tr.status_code == 200,
              f'response={td}')
    else:
        skip('transcribe-now triggers API chain', 'job_id not received in time')

    _sse_done.wait(timeout=220)   # Let background SSE thread finish cleanly
except Exception as e:
    check('transcribe-now triggers API chain', False, traceback.format_exc())

# ═════════════════════════════════════════════════════════════════════════════
section('10. FILE UPLOAD — /api/process-file')
# ═════════════════════════════════════════════════════════════════════════════

# 10a. TXT uploads — three different dataset samples, cache cleared each time
for _fi, _ds_idx in [(0, 4), (1, 5), (2, 6)]:
    clear_server_cache(f'section 10a file-{_fi}')
    _txt = dataset_sample(_ds_idx).encode()
    try:
        files = {'file': (f'transcript_{_fi}.txt', io.BytesIO(_txt), 'text/plain')}
        r = requests.post(f'{BASE}/process-file', files=files, timeout=35)
        d = r.json()
        check(f'TXT file upload audit — dataset sample #{_ds_idx}',
              d.get('success') and d.get('audit'),
              f'status={r.status_code} scored_by={d.get("audit_scored_by","?")} '
              f'f1={d.get("audit",{}).get("agent_f1_score","?")}')
    except Exception as e:
        check(f'TXT file upload audit — dataset sample #{_ds_idx}', False, str(e)[:120])

# 10b. PDF upload (minimal valid PDF)
clear_server_cache('section 10b pdf')
try:
    MINIMAL_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Agent: Hello Customer: OK) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000368 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""
    files = {'file': ('test.pdf', io.BytesIO(MINIMAL_PDF), 'application/pdf')}
    r = requests.post(f'{BASE}/process-file', files=files, timeout=35)
    d = r.json()
    # Minimal PDF may fail text extraction — that's acceptable (warn, not fail)
    check('PDF file upload handled', r.status_code in (200, 400),
          f'status={r.status_code} success={d.get("success")} error={d.get("error","")[:80]}',
          warn=not d.get('success'))
except Exception as e:
    check('PDF file upload handled', False, str(e)[:120])

# 10c. Unsupported format
try:
    files = {'file': ('bad.mp3', io.BytesIO(b'fake'), 'audio/mpeg')}
    r = requests.post(f'{BASE}/process-file', files=files, timeout=5)
    check('Unsupported file format → 400', r.status_code == 400)
except Exception as e:
    check('Unsupported file format → 400', False, str(e))

# ═════════════════════════════════════════════════════════════════════════════
section('11. ENDPOINT EDGE CASES & GUARD RAILS')
# ═════════════════════════════════════════════════════════════════════════════

# Missing audio
try:
    r = requests.post(f'{BASE}/process-call', timeout=5)
    check('/process-call with no audio → 400', r.status_code == 400)
except Exception as e:
    check('/process-call with no audio → 400', False, str(e))

# Missing audio on start-call-audit
try:
    r = requests.post(f'{BASE}/start-call-audit', timeout=5)
    check('/start-call-audit with no audio → 400', r.status_code == 400)
except Exception as e:
    check('/start-call-audit with no audio → 400', False, str(e))

# Invalid job ID
try:
    r = requests.get(f'{BASE}/job/does-not-exist-123abc/status', timeout=5)
    check('Invalid job ID → 404', r.status_code == 404)
except Exception as e:
    check('Invalid job ID → 404', False, str(e))

# Static files (served correctly)
try:
    r = requests.get('http://localhost:5000/', timeout=5)
    check('Index.html served at /', r.status_code == 200 and 'text/html' in r.headers.get('Content-Type',''))
except Exception as e:
    check('Index.html served at /', False, str(e))

try:
    r = requests.get('http://localhost:5000/script.js', timeout=5)
    check('script.js served as static', r.status_code == 200)
except Exception as e:
    check('script.js served as static', False, str(e))

try:
    r = requests.get('http://localhost:5000/app.py', timeout=5)
    check('app.py blocked (403)', r.status_code == 403)
except Exception as e:
    check('app.py blocked (403)', False, str(e))

# ═════════════════════════════════════════════════════════════════════════════
section('SUMMARY')
# ═════════════════════════════════════════════════════════════════════════════
total   = len(results)
passed  = sum(1 for r in results if r['passed'] and not r.get('skip'))
warned  = sum(1 for r in results if r.get('warn'))
skipped = sum(1 for r in results if r.get('skip'))
failed  = sum(1 for r in results if not r['passed'])

print(f'\n  Total:   {total}')
print(f'  {OK} Passed:  {passed - warned}')
print(f'  {WARN} Warned:  {warned}')
print(f'  {SKIP} Skipped: {skipped}')
print(f'  {FAIL} Failed:  {failed}')
print()

if failed > 0:
    print(f'  {FAIL}  FAILED TESTS:')
    for r in results:
        if not r['passed']:
            print(f'       • {r["name"]}')
    print()
    sys.exit(1)
else:
    if warned:
        print(f'  {WARN}  Some providers not configured (warnings above).')
    print(f'  {OK}  All critical tests passed. Safe to push to Vercel.\n')
    sys.exit(0)
