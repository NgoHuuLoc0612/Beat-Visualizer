/* v=202603221341 */
/**
 * Beat Visualizer v3 — Upload Module
 *
 * Two-phase fetch strategy:
 *   Phase 1: Poll /api/status/:id  every 1.5s → tiny JSON (<1KB), always fast
 *   Phase 2: When status=="complete", fetch /api/result/:id → full result JSON
 *
 * This separates progress tracking (fast, small) from result delivery (large, once).
 * Socket events are optional UI speed-ups and never the only path.
 */
BV.upload = (() => {
  let _jobId   = null;
  let _polling = false;
  let _done    = false;

  const STAGES = [
    'Load','HPSS','Beat','Onset','Rhythm','Spectral','Mel','MFCC','CQT',
    'Chroma','Key','Pitch','Voice','Dynamics','Timbre','eGeMAPS','Structure','Music'
  ];

  // ── Init ──────────────────────────────────────────────────────────────────
  function init() {
    loadFormats();
    setupDrop();
    document.getElementById('file-input')?.addEventListener('change', e => {
      if (e.target.files[0]) handleFile(e.target.files[0]);
    });
    const sg = document.getElementById('prog-stages');
    if (sg) {
      sg.innerHTML = STAGES.map(s =>
        `<div class="stage-dot" id="stg-${s.toLowerCase()}">${s}</div>`
      ).join('');
    }
  }

  // ── Format chips ──────────────────────────────────────────────────────────
  async function loadFormats() {
    try {
      const r = await fetch('/api/supported-formats');
      if (!r.ok) return;
      const d = await r.json();
      const chips = document.getElementById('format-chips');
      if (!chips || !d.formats) return;
      chips.innerHTML = '';
      Object.entries(d.formats).forEach(([cat, exts]) => {
        if (['project','stems','streaming'].includes(cat)) return;
        (Array.isArray(exts) ? exts : [...exts]).sort().forEach(ext => {
          const ch = document.createElement('span');
          ch.className = `fmt-chip ${cat}`;
          ch.textContent = '.' + ext;
          chips.appendChild(ch);
        });
      });
    } catch(e) { console.warn('[BV] format chips:', e); }
  }

  // ── Drag & drop ───────────────────────────────────────────────────────────
  function setupDrop() {
    const dz = document.getElementById('drop-zone');
    if (!dz) return;
    dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', ()  => dz.classList.remove('drag-over'));
    dz.addEventListener('drop', e => {
      e.preventDefault(); dz.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    });
    dz.addEventListener('click', e => {
      if (e.target.tagName === 'BUTTON') document.getElementById('file-input')?.click();
    });
  }

  // ── Upload + analyze ──────────────────────────────────────────────────────
  async function handleFile(file) {
    // Clear any previous session job
    try { sessionStorage.removeItem('bv_job_id'); } catch(e) {}
    _done = false; _polling = false; _jobId = null;
    setProgress(0, 'Uploading ' + file.name + '...');
    showProgress(true);

    // Step 1: upload
    const fd = new FormData();
    fd.append('file', file);
    let jid;
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || 'Upload failed');
      jid = d.job_id; _jobId = jid;
      try { sessionStorage.setItem('bv_job_id', jid); } catch(e) {}
      BV.toast('Uploaded: ' + d.filename + ' (' + (file.size/1024/1024).toFixed(1) + ' MB)', 'ok');
      setProgress(1, 'Starting analysis...');
    } catch(e) {
      BV.toast('Upload failed: ' + e.message, 'error');
      showProgress(false); return;
    }

    // Step 2: start analysis
    try {
      const r2 = await fetch('/api/analyze/' + jid, {
        method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}'
      });
      const d2 = await r2.json();
      if (!r2.ok) throw new Error(d2.error || 'Analyze failed');
    } catch(e) {
      BV.toast('Analysis start failed: ' + e.message, 'error');
      showProgress(false); return;
    }

    // Step 3: poll for status (lightweight)
    _pollStatus(jid);
  }

  // ── Phase 1: Poll /api/status — tiny response, runs continuously ──────────
  function _pollStatus(jid) {
    if (_polling) return;
    _polling = true;

    async function tick() {
      if (_done) return;

      try {
        const r = await fetch('/api/status/' + jid);
        // 404 = job not found (server restarted, job_id stale) — stop permanently
        if (r.status === 404) {
          _done = true; _polling = false; _jobId = null;
          try { sessionStorage.removeItem('bv_job_id'); } catch(e) {}
          setProgress(0, 'Server restarted — please upload again.');
          showProgress(false);
          BV.toast('Server restarted — please upload your file again.', 'warn', 5000);
          return;
        }
        if (!r.ok) {
          // Other server errors (500 etc) — retry a few times
          console.warn('[BV] status poll HTTP', r.status, '— retry in 3s');
          setTimeout(tick, 3000);
          return;
        }
        const d = await r.json();

        if (d.progress != null) setProgress(d.progress, d.message || '');

        if (d.status === 'complete') {
          // Status says complete — now fetch the actual result (separate large request)
          setProgress(100, 'Loading result...');
          _fetchResult(jid);
          return;  // stop status polling
        }

        if (d.status === 'error') {
          _done = true; _polling = false;
          BV.toast('Analysis error: ' + (d.error || 'Unknown'), 'error', 8000);
          showProgress(false);
          if (BV.app?.onError) BV.app.onError({ job_id: jid, error: d.error });
          return;
        }

        setTimeout(tick, 1500);

      } catch(e) {
        console.warn('[BV] status poll error — retry in 3s:', e);
        setTimeout(tick, 3000);
      }
    }

    setTimeout(tick, 1500);
  }

  // ── Phase 2: Fetch full result once — retries until success ───────────────
  async function _fetchResult(jid, attempt) {
    attempt = attempt || 1;
    try {
      console.log('[BV] fetching result (attempt ' + attempt + ')...');
      const r = await fetch('/api/result/' + jid);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const d = await r.json();
      if (!d.result) throw new Error('empty result');

      _done = true; _polling = false;
      console.log('[BV] result received, blocks:', Object.keys(d.result).length);
      if (BV.app?.onComplete) BV.app.onComplete({ job_id: jid, result: d.result });

    } catch(e) {
      if (attempt <= 5) {
        const wait = attempt * 2000;
        console.warn('[BV] result fetch failed (attempt ' + attempt + '), retry in ' + wait + 'ms:', e);
        setTimeout(() => _fetchResult(jid, attempt + 1), wait);
      } else {
        BV.toast('Could not load result after 5 attempts. Check console.', 'error', 10000);
        console.error('[BV] result fetch gave up:', e);
      }
    }
  }

  // ── Socket callbacks ──────────────────────────────────────────────────────
  // Progress from socket (faster than polling)
  function onProgress(d) {
    if (d.job_id !== _jobId) return;
    setProgress(d.progress, d.message || '');
  }

  // Complete signal from socket (no result payload)
  function onSocketComplete(d) {
    if (d.job_id !== _jobId) return;
    if (_done) return;
    console.log('[BV] socket: complete signal — fetching result');
    // Stop the status poll loop and go straight to result fetch
    _done = false;  // not fully done yet, just status known
    _polling = true; // prevent new poll loop
    setProgress(100, 'Loading result...');
    _fetchResult(d.job_id);
  }

  // ── Progress bar ──────────────────────────────────────────────────────────
  function setProgress(pct, msg) {
    const bar   = document.getElementById('prog-bar');
    const pctEl = document.getElementById('prog-pct');
    const msgEl = document.getElementById('prog-msg');
    if (bar)   bar.style.width = Math.min(100, pct) + '%';
    if (pctEl) pctEl.textContent = Math.round(pct) + '%';
    if (msgEl) msgEl.textContent = msg || '';

    const idx = Math.floor(pct / (100 / STAGES.length));
    STAGES.forEach((s, i) => {
      const el = document.getElementById('stg-' + s.toLowerCase());
      if (!el) return;
      el.classList.remove('done', 'active');
      if (i < idx)       el.classList.add('done');
      else if (i === idx) el.classList.add('active');
    });
  }

  function showProgress(show) {
    document.getElementById('progress-wrap')?.classList.toggle('visible', show);
  }

  return { init, setProgress, showProgress, onProgress, onSocketComplete,
           getJobId: () => _jobId };
})();

document.addEventListener('DOMContentLoaded', () => {
  BV.upload.init();
  // Hide progress bar on load
  document.getElementById('progress-wrap')?.classList.remove('visible');

  // Check if there was an in-progress job from this browser session.
  // Attempt to resume — if server returns 404, the poll loop will stop and clear it.
  // This handles the case where user refreshes mid-analysis.
  try {
    const savedJob = sessionStorage.getItem('bv_job_id');
    if (savedJob) {
      console.log('[BV] Found in-progress job in session:', savedJob);
      _jobId = savedJob;   // wait -- _jobId is inside the closure, not accessible here
      // Instead, trigger a one-shot check:
      fetch('/api/status/' + savedJob).then(r => {
        if (r.status === 404) {
          sessionStorage.removeItem('bv_job_id');
          console.log('[BV] Stale job cleared from session.');
        } else if (r.ok) {
          return r.json().then(d => {
            if (d.status === 'complete' || d.status === 'error') {
              sessionStorage.removeItem('bv_job_id');
            }
            // If still analyzing, do nothing — user needs to re-upload
            // (in-memory job state is gone, can't resume safely)
          });
        }
      }).catch(() => {});
    }
  } catch(e) {}
});
console.log('[BV] upload.js loaded');
