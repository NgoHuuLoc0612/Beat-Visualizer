/**
 * Beat Visualizer v3 — Realtime WebAudio Analyzer
 * Modes: spectrum, waveform, bands, circular, waterfall
 */
BV.realtime = (() => {
  let audioCtx, analyser, source, gainNode, micStream;
  let rafId, running = false;
  let mode = 'spectrum';
  let waterfallHistory = [];
  const FFT_SIZE = 2048, SMOOTHING = 0.8;
  const BAND_DEFS = [
    {name:'SUB',lo:20,hi:60}, {name:'BASS',lo:60,hi:250},
    {name:'LO-M',lo:250,hi:500}, {name:'MID',lo:500,hi:2000},
    {name:'U-M',lo:2000,hi:4000}, {name:'PRES',lo:4000,hi:6000},
    {name:'BRIL',lo:6000,hi:10000}, {name:'AIR',lo:10000,hi:20000},
  ];
  let chunkCounter = 0;

  function init() {
    document.getElementById('rt-start')?.addEventListener('click', start);
    document.getElementById('rt-stop')?.addEventListener('click', stop);
    document.querySelectorAll('.rt-mode').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.rt-mode').forEach(b => b.classList.remove('active'));
        btn.classList.add('active'); mode = btn.dataset.rtmode;
        if (mode === 'waterfall') waterfallHistory = [];
      });
    });
    _resizeCanvas();
    window.addEventListener('resize', _resizeCanvas);
  }

  function _resizeCanvas() {
    const c = document.getElementById('canvas-rt');
    if (!c) return;
    c.width = c.clientWidth; c.height = c.clientHeight;
  }

  async function start() {
    if (running) return;
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: false });
    } catch(e) {
      BV.toast('Microphone access denied: ' + e.message, 'error'); return;
    }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = FFT_SIZE;
    analyser.smoothingTimeConstant = SMOOTHING;
    gainNode = audioCtx.createGain(); gainNode.gain.value = 1.0;
    source = audioCtx.createMediaStreamSource(micStream);
    source.connect(gainNode); gainNode.connect(analyser);
    running = true;
    document.getElementById('rt-start').style.display = 'none';
    document.getElementById('rt-stop').style.display = '';
    document.getElementById('rt-status').textContent = '● LIVE';
    document.getElementById('rt-status').style.color = 'var(--ok)';
    _frame();
    BV.toast('Microphone connected', 'ok', 1500);
  }

  function stop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    micStream?.getTracks?.().forEach(t => t.stop());
    audioCtx?.close?.();
    document.getElementById('rt-start').style.display = '';
    document.getElementById('rt-stop').style.display = 'none';
    document.getElementById('rt-status').textContent = 'Idle';
    document.getElementById('rt-status').style.color = 'var(--text2)';
    _clearCanvas();
  }

  function _frame() {
    if (!running) return;
    rafId = requestAnimationFrame(_frame);
    const sr = audioCtx.sampleRate;
    const freqData = new Uint8Array(analyser.frequencyBinCount);
    const timeData = new Float32Array(FFT_SIZE);
    analyser.getByteFrequencyData(freqData);
    analyser.getFloatTimeDomainData(timeData);

    const freqs = Array.from({ length: freqData.length }, (_, i) => i * sr / FFT_SIZE);
    const features = _computeFeatures(freqData, timeData, freqs, sr);
    _updateUI(features, freqData);
    _draw(mode, freqData, timeData, freqs, sr);

    // Send 1 in 8 frames to server
    chunkCounter++;
    if (chunkCounter % 8 === 0 && BV.socket?.connected?.()) {
      BV.socket.emit('realtime_chunk', { samples: Array.from(timeData.slice(0, 512)), sample_rate: sr, chunk_id: chunkCounter });
    }
  }

  function _computeFeatures(freq, time, freqs, sr) {
    const rms = Math.sqrt(time.reduce((s, v) => s + v*v, 0) / time.length);
    const rms_db = 20 * Math.log10(rms + 1e-10);
    const peak = Math.max(...time.map(Math.abs));

    // Centroid
    let num = 0, den = 0;
    freq.forEach((v, i) => { num += freqs[i] * v; den += v; });
    const centroid = den > 0 ? num / den : 0;

    // ZCR
    let zc = 0;
    for (let i = 1; i < time.length; i++) if ((time[i] >= 0) !== (time[i-1] >= 0)) zc++;
    const zcr = zc / time.length;

    // Flatness
    const logSum = freq.reduce((s, v) => s + Math.log(v + 1e-10), 0) / freq.length;
    const mean = freq.reduce((s, v) => s + v, 0) / freq.length;
    const flatness = mean > 0 ? Math.exp(logSum) / mean : 0;

    // Band energies
    const bands = {};
    BAND_DEFS.forEach(({ name, lo, hi }) => {
      const idxLo = Math.floor(lo * FFT_SIZE / sr);
      const idxHi = Math.ceil(hi * FFT_SIZE / sr);
      const slice = freq.slice(idxLo, idxHi);
      const e = slice.reduce((s, v) => s + v * v, 0) / (slice.length || 1);
      bands[name] = e;
    });

    return { rms, rms_db, centroid, zcr, flatness, peak, bands };
  }

  function _updateUI(f, freq) {
    document.getElementById('rt-rms').textContent = f.rms_db.toFixed(1);
    document.getElementById('rt-centroid').textContent = Math.round(f.centroid);
    document.getElementById('rt-zcr').textContent = f.zcr.toFixed(3);
    document.getElementById('rt-flat').textContent = f.flatness.toFixed(3);
    document.getElementById('rt-peak').textContent = f.peak.toFixed(3);

    // Band meters
    const maxBandE = Math.max(...Object.values(f.bands)) || 1;
    document.querySelectorAll('.rt-meter-fill').forEach(el => {
      const band = el.dataset.band?.replace(/_/g,'').toUpperCase().slice(0,4);
      const name = BAND_DEFS.find(b => b.name.startsWith(band));
      const key = name?.name;
      const v = key ? (f.bands[key] || 0) / maxBandE : 0;
      el.style.height = (v * 100).toFixed(1) + '%';
    });
  }

  function _draw(mode, freq, time, freqs, sr) {
    const canvas = document.getElementById('canvas-rt');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    if (mode === 'spectrum') {
      ctx.clearRect(0, 0, W, H);
      const bw = W / freq.length;
      const g = ctx.createLinearGradient(0, H, 0, 0);
      g.addColorStop(0, '#00f5d4'); g.addColorStop(0.5, '#7b61ff'); g.addColorStop(1, '#f700ff');
      ctx.fillStyle = g;
      freq.forEach((v, i) => {
        const h = (v / 255) * H;
        ctx.fillRect(i * bw, H - h, Math.max(bw - 0.5, 1), h);
      });
      // Freq labels
      [100,500,1000,2000,5000,10000].forEach(hz => {
        const xi = Math.floor(hz * freq.length / sr * 2);
        if (xi < freq.length) {
          ctx.fillStyle = '#4a5568'; ctx.font = "8px 'Space Mono'";
          ctx.fillText(BV.fmt.hz(hz), xi * bw, H - 3);
        }
      });
    } else if (mode === 'waveform') {
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = '#00f5d4'; ctx.lineWidth = 1.5;
      ctx.beginPath();
      time.forEach((v, i) => {
        const x = (i / time.length) * W, y = (1 - v) * H / 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
      // Mirror
      ctx.strokeStyle = 'rgba(247,0,255,.4)'; ctx.lineWidth = 1;
      ctx.beginPath();
      time.forEach((v, i) => {
        const x = (i / time.length) * W, y = (1 + v) * H / 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    } else if (mode === 'bands') {
      ctx.clearRect(0, 0, W, H);
      const maxE = Math.max(...BAND_DEFS.map(({ name, lo, hi }) => {
        const lo2 = Math.floor(lo*freq.length/sr*2), hi2 = Math.ceil(hi*freq.length/sr*2);
        return freq.slice(lo2, hi2).reduce((s,v)=>s+v,0)/(hi2-lo2||1);
      })) || 1;
      const bw = W / BAND_DEFS.length - 4;
      BAND_DEFS.forEach(({ name, lo, hi }, i) => {
        const lo2 = Math.floor(lo*freq.length/sr*2), hi2 = Math.ceil(hi*freq.length/sr*2);
        const e = freq.slice(lo2, hi2).reduce((s,v)=>s+v,0)/(hi2-lo2||1);
        const h = (e / maxE) * (H - 30); const x = i * (W / BAND_DEFS.length) + 2;
        const g = ctx.createLinearGradient(0, H-30, 0, H-30-h);
        g.addColorStop(0, `hsl(${180-i*22},85%,50%)`); g.addColorStop(1, `hsl(${180-i*22},85%,80%)`);
        ctx.fillStyle = g;
        ctx.fillRect(x, H - 30 - h, bw, h);
        ctx.fillStyle = '#4a5568'; ctx.font = "7px 'Space Mono'";
        ctx.textAlign = 'center';
        ctx.fillText(name, x + bw/2, H - 5);
      });
    } else if (mode === 'circular') {
      ctx.clearRect(0, 0, W, H);
      const cx = W/2, cy = H/2, baseR = Math.min(W,H)*0.25, maxR = Math.min(W,H)*0.45;
      // Outer freq ring
      ctx.beginPath();
      freq.forEach((v, i) => {
        const angle = (i / freq.length) * Math.PI * 2 - Math.PI/2;
        const r = baseR + (v / 255) * (maxR - baseR);
        const x = cx + Math.cos(angle) * r, y = cy + Math.sin(angle) * r;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.strokeStyle = '#00f5d4'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = 'rgba(0,245,212,.05)'; ctx.fill();
      // Inner waveform
      ctx.beginPath();
      const innerR = baseR * 0.5;
      time.forEach((v, i) => {
        const angle = (i / time.length) * Math.PI * 2 - Math.PI/2;
        const r = innerR * (1 + v * 0.5);
        const x = cx + Math.cos(angle) * r, y = cy + Math.sin(angle) * r;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.strokeStyle = '#f700ff'; ctx.lineWidth = 1; ctx.stroke();
    } else if (mode === 'waterfall') {
      // Rolling waterfall
      const rowH = 2;
      const maxRows = Math.floor(H / rowH);
      waterfallHistory.unshift(Array.from(freq));
      if (waterfallHistory.length > maxRows) waterfallHistory.pop();
      ctx.clearRect(0, 0, W, H);
      waterfallHistory.forEach((row, ri) => {
        const bw = W / row.length;
        row.forEach((v, i) => {
          const t = v / 255;
          const [r, g, b] = BV.color.inferno(t);
          ctx.fillStyle = `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
          ctx.fillRect(i * bw, ri * rowH, Math.max(1, bw), rowH);
        });
      });
    }
  }

  function _clearCanvas() {
    const c = document.getElementById('canvas-rt');
    if (!c) return;
    c.getContext('2d').clearRect(0, 0, c.width, c.height);
  }

  function onServerFeatures(d) {
    // Update UI with server-computed features
    if (d.rms_db !== undefined) document.getElementById('rt-rms').textContent = d.rms_db.toFixed(1);
    if (d.centroid !== undefined) document.getElementById('rt-centroid').textContent = Math.round(d.centroid);
  }

  return { init, start, stop, onServerFeatures };
})();

document.addEventListener('DOMContentLoaded', () => BV.realtime.init());
console.log('[BV] realtime.js loaded');
