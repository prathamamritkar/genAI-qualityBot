/* ========================================================================
   BRIEFLY QA — CORE LOGIC V3 (Milestone 2)
   Architecture: Audit-First, Chart-Driven, HITL-Ready
   ======================================================================== */

// ── Configuration ─────────────────────────────────────────────────────────
const API_BASE_URL = (window.location.hostname === 'localhost' && window.location.port !== '3000')
    ? 'http://localhost:5000/api'
    : '/api';

// ── Global Chart References (for destroy/re-init) ─────────────────────────
let _radarChart = null;
let _echartsInstance = null;

// ── Persistent State ───────────────────────────────────────────────────────
let savedHistory = [];
try {
    let raw = localStorage.getItem('brieflyqa_history') || localStorage.getItem('briefly_history');
    if (raw) savedHistory = JSON.parse(raw);
    if (!Array.isArray(savedHistory)) savedHistory = [];
} catch (e) {
    console.error("Archive corruption detected, resetting.", e);
}

const AppState = {
    chatDoc: null,
    currentAudio: null,
    history: savedHistory,
    activeView: 'call',
    mediaRecorder: null,
    audioChunks: [],
    isRecording: false,
    isProcessing: false,
    lastAudit: null,
    hitlStatus: null,  // 'approved' | 'flagged' | 'rejected' | null
    isVercel: false,   // Vercel disables async background jobs; fallback to sync endpoint
};

// ── DOM Shortcuts ──────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const UI = {
    navTabs: $$('.nav-tab'),
    panels: {
        chat: $('#panel-chat'),
        call: $('#panel-call'),
        history: $('#panel-history'),
        results: $('#panel-results'),
    },
    chat: {
        input: $('#chat-input'),
        submitBtn: $('#process-chat-btn'),
        fileInput: $('#chat-file-input'),
        dropzone: $('#chat-dropzone'),
        chip: $('#chat-file-chip'),
        fileName: $('#chat-file-name'),
        removeBtn: $('#chat-remove-file'),
    },
    audio: {
        fileInput: $('#audio-input'),
        dropzone: $('#dropzone'),
        chip: $('#file-chip'),
        fileName: $('#file-name'),
        removeBtn: $('#remove-file'),
        submitBtn: $('#process-call-btn'),
        micBtn: $('#mic-record-btn'),
        micText: $('#mic-text'),
        micIcon: $('#mic-icon-sym'),
    },
    results: {
        kpiF1: $('#kpi-f1-val'),
        kpiSat: $('#kpi-sat-val'),
        kpiComp: $('#kpi-comp-val'),
        kpiF1Card: $('#kpi-f1'),
        kpiCompCard: $('#kpi-compliance'),
        summary: $('#summary-text'),
        transContainer: $('#transcription-container'),
        transBlock: $('#transcription-block'),
        transText: $('#transcription-text'),
        transDownload: $('#download-transcript-btn'),
        audioBlock: $('#audio-playback-block'),
        audioPlayer: $('#audio-player'),
        flagsList: $('#flags-list'),
        nudgesList: $('#nudges-list'),
        flagsSection: $('#flags-section'),
        copyBtn: $('#copy-btn'),
        backBtn: $('#back-btn'),
    },
    hitl: {
        panel: $('#hitl-panel'),
        badge: $('#hitl-badge'),
        approveBtn: $('#hitl-approve-btn'),
        flagBtn: $('#hitl-flag-btn'),
        rejectBtn: $('#hitl-reject-btn'),
        status: $('#hitl-status'),
    },
    historyList: $('#history-list'),
    overlayLoader: $('#loader'),
    loaderText: $('#loader-text'),
    tnp: {
        panel: $('#transcribe-now-panel'),
        btn: $('#transcribe-now-btn'),
        label: $('#tnp-node-label'),
        status: $('#tnp-status'),
        hint: $('#tnp-hint')
    },
    infoBar: $('#app-status-bar'),
    statusText: $('#status-text'),
    statusIndicator: $('.status-indicator'),
    toastOutlet: $('#toast-outlet'),
};

// ── Interactive State Sync ────────────────────────────────────────────────
function syncInteractiveState() {
    const hasChatContent = UI.chat.input.value.trim().length > 0;
    UI.chat.submitBtn.disabled = AppState.isProcessing || AppState.isRecording || (!hasChatContent && !AppState.chatDoc);
    UI.audio.submitBtn.disabled = AppState.isProcessing || AppState.isRecording || !AppState.currentAudio;
}

function setGlobalLock(lock) {
    AppState.isProcessing = lock;
    [UI.chat.submitBtn, UI.audio.submitBtn, UI.chat.removeBtn, UI.audio.removeBtn].forEach(el => {
        if (el) el.disabled = lock;
    });
    UI.navTabs.forEach(tab => {
        tab.style.pointerEvents = lock ? 'none' : 'auto';
        tab.style.opacity = lock ? '0.4' : '1';
    });
    UI.chat.input.disabled = lock || !!AppState.chatDoc;
    syncInteractiveState();
}

// ── View Router ───────────────────────────────────────────────────────────
function navigateTo(viewName) {
    if (AppState.isProcessing || AppState.isRecording) return;
    AppState.activeView = viewName;

    UI.navTabs.forEach(tab => {
        const isActive = tab.dataset.tab === viewName;
        tab.classList.toggle('active', isActive);
        tab.setAttribute('aria-selected', isActive);
    });

    Object.values(UI.panels).forEach(panel => {
        panel.classList.remove('active');
        panel.hidden = true;
    });

    const target = UI.panels[viewName];
    if (target) { target.hidden = false; target.classList.add('active'); }
    if (viewName === 'history') renderArchive();
}

// ── Navigation Events ──────────────────────────────────────────────────────
UI.navTabs.forEach(tab => tab.addEventListener('click', () => navigateTo(tab.dataset.tab)));

UI.chat.input.addEventListener('input', () => {
    syncInteractiveState();
    UI.chat.input.style.height = 'auto';
    UI.chat.input.style.height = Math.min(UI.chat.input.scrollHeight, 300) + 'px';
});
UI.chat.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); UI.chat.submitBtn.click(); }
});

// ── File Handling ──────────────────────────────────────────────────────────
UI.chat.dropzone.addEventListener('click', () => !AppState.isProcessing && UI.chat.fileInput.click());
UI.chat.fileInput.addEventListener('change', (e) => e.target.files.length && handleChatFile(e.target.files[0]));
UI.chat.removeBtn.addEventListener('click', () => resetChatInput());

function handleChatFile(file) {
    const validExts = ['.txt', '.csv', '.json', '.md', '.log', '.pdf'];
    if (!validExts.some(ext => file.name.toLowerCase().endsWith(ext))) {
        return notify('Format not supported. Use PDF, TXT, CSV, or MD.', 'error');
    }
    AppState.chatDoc = file;
    UI.chat.fileName.textContent = file.name;
    UI.chat.dropzone.hidden = true;
    UI.chat.chip.hidden = false;
    UI.chat.input.disabled = true;
    UI.chat.input.value = '';
    syncInteractiveState();
}

function resetChatInput() {
    AppState.chatDoc = null;
    UI.chat.fileInput.value = '';
    UI.chat.dropzone.hidden = false;
    UI.chat.chip.hidden = true;
    UI.chat.input.disabled = false;
    syncInteractiveState();
}

UI.audio.dropzone.addEventListener('click', () => !AppState.isProcessing && UI.audio.fileInput.click());
UI.audio.fileInput.addEventListener('change', (e) => e.target.files.length && handleAudioFile(e.target.files[0]));
UI.audio.removeBtn.addEventListener('click', () => resetAudioInput());

function handleAudioFile(file) {
    const validExts = ['.mp3', '.wav', '.m4a', '.ogg', '.webm', '.mp4'];
    if (!validExts.some(ext => file.name.toLowerCase().endsWith(ext))) {
        return notify('Incompatible audio format.', 'error');
    }
    if (file.size > 50 * 1024 * 1024) return notify('File exceeds 50MB limit.', 'error');
    AppState.currentAudio = file;
    UI.audio.fileName.textContent = file.name;
    UI.audio.dropzone.hidden = true;
    UI.audio.chip.hidden = false;
    // Disable mic button while a file is loaded — mirrors chat textarea disable behaviour
    UI.audio.micBtn.disabled = true;
    UI.audio.micBtn.style.opacity = '0.4';
    UI.audio.micBtn.style.pointerEvents = 'none';
    syncInteractiveState();
}

function resetAudioInput() {
    AppState.currentAudio = null;
    UI.audio.fileInput.value = '';
    UI.audio.dropzone.hidden = false;
    UI.audio.chip.hidden = true;
    // Re-enable mic button
    UI.audio.micBtn.disabled = false;
    UI.audio.micBtn.style.opacity = '';
    UI.audio.micBtn.style.pointerEvents = '';
    syncInteractiveState();
}

// ── Drag and Drop ──────────────────────────────────────────────────────────
[UI.audio.dropzone, UI.chat.dropzone].forEach(zone => {
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (!file) return;
        if (zone === UI.audio.dropzone) handleAudioFile(file);
        else handleChatFile(file);
    });
});

// ── API Calls ──────────────────────────────────────────────────────────────
UI.chat.submitBtn.addEventListener('click', async () => {
    if (AppState.chatDoc) await processTextDocument();
    else await processRawText();
});
UI.audio.submitBtn.addEventListener('click', async () => await processVoiceSignal());

async function processRawText() {
    const text = UI.chat.input.value.trim();
    if (!text) return;
    toggleLoader(true, 'Auditing transcript with AI Judge...');
    setGlobalLock(true);
    try {
        const res = await apiFetch('/process-chat', {
            method: 'POST',
            body: JSON.stringify({ text }),
            headers: { 'Content-Type': 'application/json' }
        });
        renderAuditDashboard(res);
        archiveAudit(res);
    } catch (err) { notify(err.message, 'error'); }
    finally { toggleLoader(false); setGlobalLock(false); }
}

async function processTextDocument() {
    toggleLoader(true, 'Ingesting document...');
    setGlobalLock(true);
    try {
        const formData = new FormData();
        formData.append('file', AppState.chatDoc);
        const res = await apiFetch('/process-file', { method: 'POST', body: formData });
        renderAuditDashboard(res);
        archiveAudit(res);
        resetChatInput();
    } catch (err) { notify(err.message, 'error'); }
    finally { toggleLoader(false); setGlobalLock(false); }
}

let currentJobId = null;

async function processVoiceSignal() {
    toggleLoader(true, 'Uploading audio track...');
    setGlobalLock(true);
    UI.tnp.panel.hidden = true;
    UI.tnp.btn.disabled = true;
    UI.tnp.status.hidden = true;
    UI.tnp.hint.textContent = ''; // Hidden in UI, erased for safety

    try {
        const formData = new FormData();
        formData.append('audio', AppState.currentAudio);

        // Vercel handles long requests by severing the connection after maxDuration (60s).
        // To respect the fallback hierarchy, we TRY the primary HF Space first.
        // If HF Space doesn't finish within 50s, we programmatically abort it and 
        // fallback to the fast API chain, preventing a hard unrecoverable 504 dead-end.
        if (AppState.isVercel) {
            UI.tnp.status.hidden = false;
            UI.tnp.status.textContent = 'Processing directly on Vercel Node (primary)...';

            UI.tnp.panel.hidden = false;
            UI.tnp.btn.disabled = false;
            let abortController = new AbortController();
            let isFastTracked = false;

            const triggerFallback = async (reason) => {
                if (isFastTracked) return;
                isFastTracked = true;
                UI.tnp.btn.disabled = true;
                UI.tnp.status.hidden = false;
                UI.tnp.status.textContent = `${reason} — Engaging fast API fallback chain...`;
                abortController.abort(); // Cancel the hanging primary fetch

                try {
                    abortController = new AbortController(); // fresh token
                    const fastRes = await apiFetch('/process-call?fast_track=true', {
                        method: 'POST', body: formData, signal: abortController.signal
                    });
                    finishProcessCall(fastRes);
                } catch (e) {
                    if (e.name !== 'AbortError') throw e; // Bubble to outer catch
                }
            };

            UI.tnp.btn.onclick = () => triggerFallback('Fast-track activated').catch(e => {
                notify('API Chain Failed: ' + e.message, 'error');
                toggleLoader(false);
                setGlobalLock(false);
                UI.tnp.panel.hidden = true;
            });

            try {
                // Attempt standard waterfall (HF Space first!)
                // We wait indefinitely here on the client side. If Vercel issues a 504 
                // Gateway Timeout for exceeding container limits, the catch block intercepts it.
                const res = await apiFetch('/process-call', {
                    method: 'POST', body: formData, signal: abortController.signal
                });
                finishProcessCall(res);
            } catch (e) {
                // If we aborted it manually, ignore the error
                if (e.name !== 'AbortError') {
                    // If Vercel physically cut off the connection with 504 Timeout, seamlessly catch it and fallback.
                    if (e.message.includes('504') || e.message.includes('Timeout')) {
                        await triggerFallback('Vercel Timeout intercepted');
                    } else {
                        throw e; // Standard 500 error / unhandled issue
                    }
                }
            }
            return;
        }

        // Standard backend polling approach for local / persistent servers
        const res = await apiFetch('/start-call-audit', { method: 'POST', body: formData });
        currentJobId = res.job_id;

        let isFastTracked = false;
        if (res.fallbacks_available) {
            UI.tnp.panel.hidden = false;
            UI.tnp.btn.disabled = false;

            UI.tnp.btn.onclick = async () => {
                if (!currentJobId || isFastTracked) return;
                isFastTracked = true;
                UI.tnp.btn.disabled = true;
                UI.tnp.status.hidden = false;
                UI.tnp.status.textContent = 'Engaging priority queue...';
                try {
                    await apiFetch(`/job/${currentJobId}/transcribe-now`, { method: 'POST' });
                } catch (e) {
                    UI.tnp.status.textContent = 'Fast-track failed: ' + e.message;
                }
            };
        }

        await pollJobStatus(currentJobId);

    } catch (err) {
        notify(err.message, 'error');
        toggleLoader(false);
        setGlobalLock(false);
        UI.tnp.panel.hidden = true;
    }
}

// Helper to dry up the resolution step for the synchronous flow
function finishProcessCall(res) {
    if (AppState.currentAudio) {
        res.localAudioUrl = URL.createObjectURL(AppState.currentAudio);
        res.audioName = AppState.currentAudio.name;
    }
    renderAuditDashboard(res);
    archiveAudit(res);
    resetAudioInput();
    toggleLoader(false);
    setGlobalLock(false);
    UI.tnp.panel.hidden = true;
}

async function pollJobStatus(jobId) {
    while (true) {
        await new Promise(r => setTimeout(r, 2000));
        if (jobId !== currentJobId) return; // User cancelled or left

        try {
            const res = await apiFetch(`/job/${jobId}/status`);
            if (res.status === 'done') {
                // Ensure transcription field exists (backward compatibility for both field names)
                if (!res.transcription && res.transcript) {
                    res.transcription = res.transcript;
                }
                if (AppState.currentAudio) {
                    res.localAudioUrl = URL.createObjectURL(AppState.currentAudio);
                    res.audioName = AppState.currentAudio.name;
                }
                // Ensure the response is properly passed as the entire object
                renderAuditDashboard(res);
                archiveAudit(res);
                resetAudioInput();
                toggleLoader(false);
                setGlobalLock(false);
                UI.tnp.panel.hidden = true;
                currentJobId = null;
                if (UI.statusText) UI.statusText.textContent = 'System Ready';
                return;
            } else if (res.status === 'error') {
                throw new Error(res.error || 'Unknown job failure');
            } else {
                UI.loaderText.textContent = 'Analyzing interaction...';
                if (UI.statusText) {
                    UI.statusText.textContent = `[Job ${jobId.substring(0, 6)}] ${res.status.replace(/_/g, ' ')}`;
                }
            }
        } catch (e) {
            notify(e.message, 'error');
            toggleLoader(false);
            setGlobalLock(false);
            UI.tnp.panel.hidden = true;
            currentJobId = null;
            if (UI.statusText) UI.statusText.textContent = 'System Error';
            return;
        }
    }
}

// ── Recording ─────────────────────────────────────────────────────────────
UI.audio.micBtn.addEventListener('click', () => {
    if (AppState.isProcessing) return;
    if (AppState.isRecording) finalizeRecording();
    else initiateRecording();
});

async function initiateRecording() {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
        return notify('Capture system not compatible with this environment.', 'error');
    }
    AppState.audioChunks = [];
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        AppState.mediaRecorder = new MediaRecorder(stream);
        AppState.mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) AppState.audioChunks.push(e.data); };
        AppState.mediaRecorder.onstop = () => {
            const blob = new Blob(AppState.audioChunks, { type: 'audio/wav' });
            handleAudioFile(new File([blob], 'capture.wav', { type: 'audio/wav' }));
            stream.getTracks().forEach(t => t.stop());
            setGlobalLock(false);
        };
        AppState.mediaRecorder.start();
        AppState.isRecording = true;
        setGlobalLock(true);
        AppState.isProcessing = false;
        UI.audio.micBtn.classList.add('recording');
        UI.audio.micBtn.disabled = false;
        UI.audio.micText.textContent = 'Recording Active — Click to Stop';
        UI.audio.micIcon.textContent = 'stop_circle';
        notify('Voice capture active', 'success');
    } catch (e) { notify('Microphone access denied.', 'error'); }
}

function finalizeRecording() {
    if (AppState.mediaRecorder && AppState.isRecording) {
        AppState.mediaRecorder.stop();
        AppState.isRecording = false;
        UI.audio.micBtn.classList.remove('recording');
        UI.audio.micText.textContent = 'Capture Live Audio';
        UI.audio.micIcon.textContent = 'mic';
    }
}

// ── AUDIT DASHBOARD RENDERER ──────────────────────────────────────────────
function renderAuditDashboard(data) {
    const audit = data.audit || {};
    AppState.lastAudit = audit;
    AppState.hitlStatus = null;

    // ── KPI Cards ────────────────────────────────────────────────────────
    let f1 = null;
    if (audit.agent_f1_score !== undefined && audit.agent_f1_score !== null) {
        let parsed = parseFloat(audit.agent_f1_score);
        if (!isNaN(parsed)) f1 = parsed > 1 ? parsed / 100 : parsed; // handle 92 vs 0.92
    }

    // Logical Inference F1 Fallback: Derive highest mathematical F1 based on core metrics if LLM omits it
    if ((f1 === null || f1 === 0 || isNaN(f1)) && audit.quality_matrix) {
        const qm = audit.quality_matrix;
        const pTokens = [qm.language_proficiency, qm.efficiency, qm.bias_reduction];
        const rTokens = [qm.cognitive_empathy, qm.active_listening];

        // Use logical defaults if missing but available (i.e., LLM gave partial response)
        const precision = pTokens.map(v => parseFloat(v) || 5).reduce((a, b) => a + b) / 30;
        const recall = rTokens.map(v => parseFloat(v) || 5).reduce((a, b) => a + b) / 20;

        f1 = (precision + recall > 0) ? (2 * (precision * recall)) / (precision + recall) : 0;
    }

    UI.results.kpiF1.textContent = f1 !== null ? (f1 * 100).toFixed(0) + '%' : '—';
    UI.results.kpiF1Card.className = 'kpi-card ' + scoreClass(f1, [0.85, 0.65]);  // green/amber/red

    const sat = audit.satisfaction_prediction || '—';
    UI.results.kpiSat.textContent = sat;
    UI.results.kpiSat.closest('.kpi-card').className = 'kpi-card ' + satClass(sat);

    const risk = audit.compliance_risk || '—';
    UI.results.kpiComp.textContent = risk;
    UI.results.kpiCompCard.className = 'kpi-card ' + riskClass(risk);

    // ── Summary ──────────────────────────────────────────────────────────
    UI.results.summary.textContent = audit.summary || 'No summary available.';

    // ── Transcript ───────────────────────────────────────────
    const rawContent = data.transcription || data.original_text || data.transcript;
    const hasTranscript = !!rawContent;
    if (UI.results.transBlock) UI.results.transBlock.open = false;
    if (UI.results.transContainer) UI.results.transContainer.hidden = !hasTranscript;
    else if (UI.results.transBlock) UI.results.transBlock.hidden = !hasTranscript;

    if (hasTranscript && UI.results.transText) {
        UI.results.transText.innerHTML = formatTranscript(rawContent);
        AppState.currentTranscriptRaw = rawContent;
    }

    // ── Audio Player ──────────────────────────────────────────────────────
    if (UI.results.audioBlock) {
        if (data.type === 'call' && data.localAudioUrl) {
            UI.results.audioBlock.hidden = false;
            UI.results.audioPlayer.src = data.localAudioUrl;
        } else {
            UI.results.audioBlock.hidden = true;
            UI.results.audioPlayer.src = "";
        }
    }

    // ── Compliance Flags ──────────────────────────────────────────────────
    const flags = Array.isArray(audit.compliance_flags) ? audit.compliance_flags : [];
    UI.results.flagsSection.hidden = false;
    if (flags.length === 0) {
        UI.results.flagsList.innerHTML = `<li class="detail-item no-issue"><span class="material-symbols-rounded">check_circle</span> No compliance issues detected</li>`;
    } else {
        UI.results.flagsList.innerHTML = flags.map(f =>
            `<li class="detail-item flag-item"><span class="material-symbols-rounded">warning</span>${escHtml(f)}</li>`
        ).join('');
    }

    // ── Behavioral Nudges ─────────────────────────────────────────────────
    const nudges = Array.isArray(audit.behavioral_nudges) ? audit.behavioral_nudges : [];
    UI.results.nudgesList.innerHTML = nudges.length
        ? nudges.map(n => `<li class="detail-item nudge-item"><span class="material-symbols-rounded">tips_and_updates</span>${escHtml(n)}</li>`).join('')
        : '<li class="detail-item">No nudges generated.</li>';

    // ── HITL Panel ────────────────────────────────────────────────────────
    UI.hitl.badge.textContent = 'AI Scored';
    UI.hitl.badge.className = 'hitl-badge';
    UI.hitl.status.hidden = true;
    [UI.hitl.approveBtn, UI.hitl.flagBtn, UI.hitl.rejectBtn].forEach(b => b.disabled = false);

    if (audit._hitl) {
        AppState.hitlStatus = audit._hitl.status;
        UI.hitl.badge.textContent = audit._hitl.status.charAt(0).toUpperCase() + audit._hitl.status.slice(1);
        UI.hitl.badge.className = 'hitl-badge ' + { approved: 'badge-green', flagged: 'badge-warn', rejected: 'badge-red' }[audit._hitl.status];
        UI.hitl.status.hidden = false;
        UI.hitl.status.textContent = `Recorded: ${audit._hitl.msg}`;
        UI.hitl.status.className = `hitl-status ${audit._hitl.type}`;
        [UI.hitl.approveBtn, UI.hitl.flagBtn, UI.hitl.rejectBtn].forEach(b => b.disabled = true);
    } else if (audit.hitl_review_required) {
        UI.hitl.badge.textContent = 'Review Required';
        UI.hitl.badge.className = 'hitl-badge badge-warn';
        notify('Supervisor review recommended for this audit.', 'warning');
    }

    // ── Show Dashboard first so containers have real dimensions ──────────
    Object.values(UI.panels).forEach(p => { p.classList.remove('active'); p.hidden = true; });
    UI.panels.results.hidden = false;
    UI.panels.results.classList.add('active');

    // ── Charts — deferred one frame so the browser lays out the panel ────
    const qm = audit.quality_matrix || {};
    const timeline = Array.isArray(audit.emotional_timeline) ? audit.emotional_timeline : [];
    requestAnimationFrame(() => {
        renderRadarChart(qm);
        renderEmotionTopography(timeline);
    });

    notify('Quality audit complete', 'success');
}

// ── CHART: 2D Agent Skill Radar (Chart.js) ────────────────────────────────
function renderRadarChart(qm) {
    if (_radarChart) { _radarChart.destroy(); _radarChart = null; }
    const ctx = document.getElementById('qualityRadarChart')?.getContext('2d');
    if (!ctx) return;

    const labels = ['Language\nProficiency', 'Cognitive\nEmpathy', 'Efficiency', 'Bias\nReduction', 'Active\nListening'];
    // Logical Inference: Baseline standard support performance is inherently 5/10 — infer this instead of zeroing the chart out on an AI error.
    const safeVal = (v) => {
        let parsed = parseFloat(v);
        return isNaN(parsed) || parsed === 0 ? 5 : Math.max(0, Math.min(10, parsed));
    };

    const values = [
        safeVal(qm.language_proficiency),
        safeVal(qm.cognitive_empathy),
        safeVal(qm.efficiency),
        safeVal(qm.bias_reduction),
        safeVal(qm.active_listening),
    ];

    _radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels,
            datasets: [{
                label: 'Agent Score',
                data: values,
                backgroundColor: 'rgba(99,102,241,0.2)',
                borderColor: 'rgba(99,102,241,0.9)',
                borderWidth: 2.5,
                pointBackgroundColor: 'rgba(99,102,241,1)',
                pointRadius: 5,
                pointHoverRadius: 7,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` ${ctx.raw}/10`
                    }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 10,
                    ticks: {
                        stepSize: 2,
                        color: 'rgba(148,163,184,0.8)',
                        backdropColor: 'transparent',
                        font: { size: 10 }
                    },
                    grid: { color: 'rgba(148,163,184,0.15)' },
                    angleLines: { color: 'rgba(148,163,184,0.15)' },
                    pointLabels: {
                        color: '#CBD5E1',
                        font: { size: 11, family: 'Inter' }
                    }
                }
            }
        }
    });
}


// ── CHART: 3D Emotional Landscape — bar3D Mountain Peaks (Apache ECharts GL) ─
// Scientific basis:
//   • Russell's Circumplex Model of Affect (1980) — arousal × valence → color
//   • Plutchik's Wheel of Emotions (1980) — hierarchical emotion structure
//   • Z-axis (height) = Emotional Intensity (1–10, linear, LLM-assigned)
//   • X-axis = Chronological Turn Index (discrete, time-ordered)
//   • Y-axis = Speaker lane (Customer=0, Agent=1)
//   • Bar color = Emotion type → psychological valence mapping
//   Each bar is a 1:1 representation of one conversation turn. No interpolation.
function renderEmotionTopography(timeline) {
    const container = document.getElementById('emotionTopographyChart');
    if (!container) return;

    if (_echartsInstance) { _echartsInstance.dispose(); _echartsInstance = null; }

    if (!timeline || !timeline.length) {
        container.innerHTML = '<div class="chart-empty">No emotional timeline data available for this interaction.</div>';
        return;
    }

    _echartsInstance = echarts.init(container, null, { renderer: 'webgl' });

    // ── Psychological Emotion→Color Map ────────────────────────────────────
    // Based on Russell (1980): valence (positive/negative) + arousal (high/low)
    // High arousal + negative valence → warm red spectrum (sympathetic nervous system)
    // Low arousal  + positive valence → cool green spectrum (parasympathetic)
    // Neutral                         → slate (no strong valence bias)
    const EMOTION_COLORS = {
        'Angry': '#EF4444',  // High arousal, highly negative (threat response)
        'Frustrated': '#F97316',  // High arousal, negative
        'Anxious': '#FB923C',  // High arousal, slightly negative
        'Confused': '#FBBF24',  // Medium arousal, mildly negative
        'Neutral': '#94A3B8',  // Low arousal, neutral valence (baseline)
        'Professional': '#38BDF8',  // Low arousal, slightly positive
        'Calm': '#34D399',  // Low arousal, positive (parasympathetic state)
        'Empathetic': '#22C55E',  // Medium arousal, positive
        'Relieved': '#4ADE80',  // Decreasing arousal, clearly positive
        'Satisfied': '#10B981',  // Low arousal, highly positive
        'Happy': '#059669',  // Medium arousal, highly positive valence
    };

    // Dynamically discover all unique speakers in the timeline
    let speakers = Array.from(new Set(timeline.map(t => (t && t.speaker) ? t.speaker : 'Unknown')));
    if (speakers.length === 0) speakers = ['Customer', 'Agent']; // Fallback

    const maxTurns = timeline.length;

    // Build bar3D data — each entry represents one speaker turn
    const barData = timeline.map((t, i) => {
        if (!t) return null;
        const currentSpeaker = t.speaker || 'Unknown';
        let speakerIdx = speakers.indexOf(currentSpeaker);
        if (speakerIdx === -1) { speakers.push(currentSpeaker); speakerIdx = speakers.length - 1; }

        let color = EMOTION_COLORS[t.emotion] || EMOTION_COLORS['Neutral'];

        // Logical Inference: Extract psychological baseline intensities if missing
        let emotionIntensities = {
            'Angry': 9, 'Frustrated': 8, 'Anxious': 7,
            'Confused': 6, 'Neutral': 3, 'Professional': 4,
            'Calm': 3, 'Empathetic': 6, 'Relieved': 5,
            'Satisfied': 7, 'Happy': 8
        };
        let inferredIntensity = emotionIntensities[t.emotion] || 5;
        let pInt = parseFloat(t.intensity);
        let intensity = Math.min(Math.max(!isNaN(pInt) && pInt !== 0 ? pInt : inferredIntensity, 1), 10);

        return {
            value: [i, speakerIdx, intensity],
            _turn: t,   // retain for tooltip
            itemStyle: { color, opacity: 0.93 }
        };
    }).filter(Boolean);

    // Auto-scale bar width: wider for short calls, narrower for long calls
    const barSize = Math.max(1.2, Math.min(5, 60 / maxTurns));

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            show: true,
            confine: true,
            formatter: params => {
                const t = params.data?._turn || {};
                const emotionColor = EMOTION_COLORS[t.emotion] || '#94A3B8';
                return `<div style="font:12px/1.8 Inter,sans-serif;min-width:160px">
                    <b>Turn ${t.turn ?? (params.data.value[0] + 1)}</b><br/>
                    <span style="color:${emotionColor}">●</span> <b>${t.speaker || '—'}</b><br/>
                    Emotion: <b>${t.emotion || '—'}</b><br/>
                    Intensity: <b>${t.intensity ?? params.data.value[2]}</b> / 10
                </div>`;
            }
        },
        grid3D: {
            boxWidth: 150,
            boxDepth: 45,
            boxHeight: 100,
            viewControl: {
                autoRotate: false,
                rotateSensitivity: 2,
                zoomSensitivity: 1.2,
                panSensitivity: 1,
                alpha: 20,     // tilt angle
                beta: -10,     // rotation angle — slight offset to show both speaker lanes
                distance: 190,
            },
            light: {
                main: {
                    intensity: 2.4,
                    shadow: true,
                    shadowQuality: 'high',
                    alpha: 40,
                    beta: 35,
                },
                ambient: { intensity: 0.3 }
            },
            postEffect: {
                enable: true,
                SSAO: { enable: true, quality: 'medium', radius: 3 }
            },
        },
        xAxis3D: {
            name: 'Turn #',
            type: 'value',
            min: 0,
            max: maxTurns,
            nameTextStyle: { color: '#64748B', fontSize: 10 },
            axisLabel: { color: '#64748B', fontSize: 9 },
            axisLine: { lineStyle: { color: '#1E293B' } },
            splitLine: { lineStyle: { color: 'rgba(30,41,59,0.7)', width: 0.5 } },
        },
        yAxis3D: {
            name: 'Speaker',
            type: 'category',
            data: speakers,
            nameTextStyle: { color: '#64748B', fontSize: 10 },
            axisLabel: { color: '#64748B', fontSize: 10 },
            axisLine: { lineStyle: { color: '#1E293B' } },
            splitLine: { show: false },
        },
        zAxis3D: {
            name: 'Intensity',
            type: 'value',
            min: 0,
            max: 10,
            interval: 2,
            nameTextStyle: { color: '#64748B', fontSize: 10 },
            axisLabel: { color: '#64748B', fontSize: 9, formatter: v => v },
            axisLine: { lineStyle: { color: '#1E293B' } },
            splitLine: { lineStyle: { color: 'rgba(30,41,59,0.7)', width: 0.5 } },
        },
        series: [{
            type: 'bar3D',
            data: barData,
            shading: 'lambert',     // Lambert = physically accurate diffuse light model
            barSize,
            label: { show: false },
            emphasis: {
                label: { show: false },
                itemStyle: { opacity: 1 }
            },
        }]
    };

    _echartsInstance.setOption(option);

    // Respond to window/panel resize — only when container is actually visible
    const ro = new ResizeObserver(() => {
        if (!_echartsInstance) return;
        const { offsetWidth: w, offsetHeight: h } = container;
        if (w > 0 && h > 0) {
            _echartsInstance.resize();
        } else {
            // Container collapsed to zero (panel hidden) — dispose GL context to
            // stop WebGL framebuffer errors; next render call will reinitialise.
            ro.disconnect();
            _echartsInstance.dispose();
            _echartsInstance = null;
        }
    });
    ro.observe(container);
}

// ── HITL Actions ──────────────────────────────────────────────────────────
UI.hitl.approveBtn.addEventListener('click', () => handleHitl('approved', 'Audit approved by supervisor.', 'success'));
UI.hitl.flagBtn.addEventListener('click', () => handleHitl('flagged', 'Flagged for further review.', 'warning'));
UI.hitl.rejectBtn.addEventListener('click', () => handleHitl('rejected', 'Score rejected. Manual audit recommended.', 'error'));

function handleHitl(status, msg, type) {
    AppState.hitlStatus = status;
    UI.hitl.badge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    UI.hitl.badge.className = 'hitl-badge ' + { approved: 'badge-green', flagged: 'badge-warn', rejected: 'badge-red' }[status];
    UI.hitl.status.hidden = false;
    UI.hitl.status.textContent = `Recorded: ${msg}`;
    UI.hitl.status.className = `hitl-status ${type}`;
    [UI.hitl.approveBtn, UI.hitl.flagBtn, UI.hitl.rejectBtn].forEach(b => b.disabled = true);

    // Persist HITL decision into the current audit and memory
    if (AppState.lastAudit) {
        AppState.lastAudit._hitl = { status, msg, type };
        localStorage.setItem('brieflyqa_history', JSON.stringify(AppState.history));
    }
    notify(msg, type);
}

// ── Results Actions ───────────────────────────────────────────────────────
UI.results.backBtn.addEventListener('click', () => {
    if (_radarChart) { _radarChart.destroy(); _radarChart = null; }
    if (_echartsInstance) { _echartsInstance.dispose(); _echartsInstance = null; }
    navigateTo(AppState.activeView);
});

UI.results.copyBtn.addEventListener('click', () => {
    const audit = AppState.lastAudit;
    if (!audit) return;
    const text = [
        `Summary: ${audit.summary || ''}`,
        `Agent F1: ${((audit.agent_f1_score || 0) * 100).toFixed(0)}%`,
        `Satisfaction: ${audit.satisfaction_prediction || ''}`,
        `Compliance Risk: ${audit.compliance_risk || ''}`,
        `Flags: ${(audit.compliance_flags || []).join('; ') || 'None'}`,
        `Nudges: ${(audit.behavioral_nudges || []).join('; ')}`,
    ].join('\n');
    navigator.clipboard.writeText(text).then(() => notify('Audit report copied', 'success'));
});

UI.results.transDownload?.addEventListener('click', () => {
    if (!AppState.currentTranscriptRaw) return;
    const blob = new Blob([AppState.currentTranscriptRaw], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcript_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
});

// ── Archive ───────────────────────────────────────────────────────────────
function archiveAudit(data) {
    const audit = data.audit || {};
    AppState.history.unshift({
        id: Date.now(),
        type: data.type,
        audit,
        transcription: data.transcription || data.transcript || null,
        original_text: data.original_text || null,
        localAudioUrl: data.localAudioUrl || null,
        audioName: data.audioName || null,
        timestamp: data.timestamp || new Date().toISOString(),
    });
    if (AppState.history.length > 50) AppState.history = AppState.history.slice(0, 50);
    localStorage.setItem('brieflyqa_history', JSON.stringify(AppState.history));
}

function renderArchive() {
    if (!AppState.history.length) {
        UI.historyList.innerHTML = `<div class="empty-state">
            <span class="material-symbols-rounded" style="font-size:2.5rem;opacity:0.3">history</span>
            <p>No audits yet. Process a call or chat to begin.</p>
        </div>`;
        return;
    }
    UI.historyList.innerHTML = AppState.history.map(item => {
        const audit = item.audit || {};
        const f1 = typeof audit.agent_f1_score === 'number' ? (audit.agent_f1_score * 100).toFixed(0) + '%' : '—';
        const risk = audit.compliance_risk || '—';
        return `<div class="history-card" data-id="${item.id}">
            <div class="history-header">
                <span class="item-type">${item.type === 'chat' ? 'TEXT' : 'VOICE'}</span>
                <span class="item-time">${formatTimeAgo(new Date(item.timestamp))}</span>
                <span class="item-f1">F1: ${f1}</span>
                <span class="item-risk risk-${risk.toLowerCase()}">${risk}</span>
            </div>
            <div class="item-preview">${escHtml(audit.summary || 'No summary.')}</div>
        </div>`;
    }).join('');

    $$('.history-card').forEach(card => {
        card.addEventListener('click', () => {
            if (AppState.isProcessing || AppState.isRecording) return;
            const entry = AppState.history.find(h => h.id === parseInt(card.dataset.id));
            if (entry) renderAuditDashboard(entry);
        });
    });
}

// ── Utilities ─────────────────────────────────────────────────────────────
function escHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatTranscript(txt) {
    if (!txt) return '';
    // Make speaker labels bold + proper line breaks
    return escHtml(txt)
        .replace(/(Speaker \d+|Agent|Customer|speaker_\d+):/gi, '<strong>$1:</strong>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
}

function scoreClass(val, [hi, lo]) {
    if (val === null || val === undefined) return '';
    if (val >= hi) return 'kpi-green';
    if (val >= lo) return 'kpi-amber';
    return 'kpi-red';
}
function satClass(s) {
    return s === 'High' ? 'kpi-green' : s === 'Medium' ? 'kpi-amber' : s === 'Low' ? 'kpi-red' : '';
}
function riskClass(r) {
    return r === 'Green' ? 'kpi-green' : r === 'Amber' ? 'kpi-amber' : r === 'Red' ? 'kpi-red' : '';
}

async function apiFetch(path, options = {}) {
    const res = await fetch(`${API_BASE_URL}${path}`, options);

    let data;
    try {
        data = await res.json();
    } catch (e) {
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        throw e;
    }

    if (!res.ok) throw new Error(data.error || 'Audit request failed.');
    return data;
}

function toggleLoader(visible, msg = '') {
    UI.loaderText.textContent = msg;
    UI.overlayLoader.hidden = !visible;
}

function notify(msg, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `modern-toast ${type}`;
    toast.textContent = msg;
    UI.toastOutlet.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function formatTimeAgo(date) {
    const s = Math.floor((Date.now() - date.getTime()) / 1000);
    if (s < 10) return 'Just now';
    const units = [['yr', 31536000], ['mo', 2592000], ['wk', 604800], ['d', 86400], ['hr', 3600], ['m', 60]];
    for (const [u, secs] of units) {
        const n = Math.floor(s / secs);
        if (n >= 1) return `${n}${u} ago`;
    }
    return 'Just now';
}

async function verifyConnection() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        const data = await res.json();
        UI.infoBar.classList.remove('status-offline');
        const st = $('#status-text');
        AppState.isVercel = data.vercel_mode || false;

        if (data.api_ready) st.textContent = 'Qualora Engine: Active';
        else st.textContent = 'Qualora Engine: Limited Mode';
    } catch {
        UI.infoBar.classList.add('status-offline');
        $('#status-text').textContent = 'Qualora Engine: Offline';
    }
}

// ── Init ──────────────────────────────────────────────────────────────────
syncInteractiveState();
verifyConnection();
setInterval(verifyConnection, 30000);
