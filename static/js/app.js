/**
 * Beat Visualizer v3 — Main Orchestrator (v4 analyzer block paths)
 */
BV.app = (() => {
  let _currentTab = 'upload';
  let _result = null, _jobId = null;
  let _player = { playing:false, time:0, duration:0, raf:null };
  let _3dInited = false;
  let _metronome = null;
  let _particleCanvas, _pctx, _particles = [];

  function init() {
    _initTabs(); _initTabScroll(); _initParticles(); _initPlayer();
    _initKeyboard(); _initSubTabs();
    BV.toast('Beat Visualizer v3 ready — drop an audio file to begin', 'ok', 3000);
  }

  // ── Tabs ──────────────────────────────────────────────────────────────────
  function _initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => switchTab(btn.dataset.panel));
    });
  }

  // ── Tab bar: drag-to-scroll + arrow buttons ───────────────────────────────
  function _initTabScroll() {
    const bar   = document.getElementById('tab-bar');
    const btnL  = document.getElementById('tab-scroll-left');
    const btnR  = document.getElementById('tab-scroll-right');
    if (!bar) return;

    // ── Arrow buttons ──────────────────────────────────────────────────────
    const SCROLL_STEP = 160;
    let scrollInterval = null;

    function startScroll(dir) {
      bar.scrollBy({ left: dir * SCROLL_STEP, behavior: 'smooth' });
      scrollInterval = setInterval(() => {
        bar.scrollBy({ left: dir * SCROLL_STEP * 0.6, behavior: 'smooth' });
      }, 180);
    }
    function stopScroll() {
      clearInterval(scrollInterval);
      scrollInterval = null;
    }

    if (btnL) {
      btnL.addEventListener('mousedown',  () => startScroll(-1));
      btnL.addEventListener('touchstart', (e) => { e.preventDefault(); startScroll(-1); }, { passive: false });
      btnL.addEventListener('click', () => bar.scrollBy({ left: -SCROLL_STEP, behavior: 'smooth' }));
    }
    if (btnR) {
      btnR.addEventListener('mousedown',  () => startScroll(1));
      btnR.addEventListener('touchstart', (e) => { e.preventDefault(); startScroll(1); }, { passive: false });
      btnR.addEventListener('click', () => bar.scrollBy({ left: SCROLL_STEP, behavior: 'smooth' }));
    }
    document.addEventListener('mouseup',   stopScroll);
    document.addEventListener('touchend',  stopScroll);

    // Update arrow visibility based on scroll position
    function updateArrows() {
      if (!btnL || !btnR) return;
      const atStart = bar.scrollLeft <= 2;
      const atEnd   = bar.scrollLeft >= bar.scrollWidth - bar.clientWidth - 2;
      btnL.classList.toggle('hidden', atStart);
      btnR.classList.toggle('hidden', atEnd);
    }
    bar.addEventListener('scroll', updateArrows, { passive: true });
    window.addEventListener('resize', updateArrows);
    updateArrows();  // run once on init

    // ── Drag to scroll (mouse) ─────────────────────────────────────────────
    let drag = { active: false, startX: 0, scrollLeft: 0, moved: false };

    bar.addEventListener('mousedown', e => {
      // Only trigger on bar background, not on buttons
      if (e.target.classList.contains('tab-btn')) return;
      drag.active     = true;
      drag.startX     = e.pageX - bar.offsetLeft;
      drag.scrollLeft = bar.scrollLeft;
      drag.moved      = false;
      bar.classList.add('dragging');
    });

    bar.addEventListener('mouseleave', () => {
      drag.active = false;
      bar.classList.remove('dragging');
    });

    document.addEventListener('mouseup', () => {
      drag.active = false;
      bar.classList.remove('dragging');
    });

    bar.addEventListener('mousemove', e => {
      if (!drag.active) return;
      e.preventDefault();
      const x    = e.pageX - bar.offsetLeft;
      const walk = x - drag.startX;
      if (Math.abs(walk) > 3) drag.moved = true;
      bar.scrollLeft = drag.scrollLeft - walk;
    });

    // Prevent click-on-tab from firing if we were dragging
    bar.addEventListener('click', e => {
      if (drag.moved) {
        e.stopImmediatePropagation();
        drag.moved = false;
      }
    }, true);  // capture phase

    // ── Touch drag ────────────────────────────────────────────────────────
    let touch = { startX: 0, scrollLeft: 0 };

    bar.addEventListener('touchstart', e => {
      touch.startX     = e.touches[0].pageX;
      touch.scrollLeft = bar.scrollLeft;
    }, { passive: true });

    bar.addEventListener('touchmove', e => {
      const dx = touch.startX - e.touches[0].pageX;
      bar.scrollLeft = touch.scrollLeft + dx;
    }, { passive: true });
  }

  function switchTab(panel) {
    _currentTab = panel;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.panel === panel));
    document.querySelectorAll('.panel').forEach(p => {
      p.classList.toggle('active', p.id === `panel-${panel}` || p.id === panel);
    });
    if (panel === 'panel-3d') {
      if (_result && !_3dInited) { BV.viz3d.init(_result); _3dInited = true; }
      setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
    }
    // Redraw canvas panels on tab switch (sizing may have changed)
    if (_result && panel === 'waveform')   setTimeout(() => BV.viz2d.renderWaveform(_result),  30);
    if (_result && panel === 'structure')  setTimeout(() => BV.viz2d.renderStructure(_result), 30);
  }

  // ── Sub-tabs ──────────────────────────────────────────────────────────────
  function _initSubTabs() {
    const _activeSub = (sel, cb) => document.querySelectorAll(sel).forEach(btn => btn.addEventListener('click', () => {
      document.querySelectorAll(sel).forEach(b => b.classList.remove('active'));
      btn.classList.add('active'); if (_result) cb(btn);
    }));
    _activeSub('[data-wf]',       btn => BV.viz2d.renderWaveform(_result, btn.dataset.wf));
    _activeSub('[data-spec]',     btn => BV.viz2d.renderSpectrum(_result, btn.dataset.spec));
    _activeSub('[data-chroma]',   btn => BV.viz2d.renderHarmony(_result, btn.dataset.chroma));
    _activeSub('[data-mfcc-view]',btn => BV.viz2d.renderMFCC(_result, btn.dataset.mfccView));

    ['pitch-show-praat','pitch-show-voiced'].forEach(id =>
      document.getElementById(id)?.addEventListener('change', () => _result && BV.viz2d.renderPitch(_result)));
    ['wf-show-beats','wf-show-onsets','wf-show-segs'].forEach(id =>
      document.getElementById(id)?.addEventListener('change', () => _result && BV.viz2d.renderWaveform(_result)));

    document.getElementById('bpm-ring')?.addEventListener('click', _toggleMetronome);
    document.querySelector('.export-dropdown')?.addEventListener('mouseleave', () =>
      document.querySelector('.export-dropdown')?.classList.remove('open'));
  }

  // ── Socket callbacks ──────────────────────────────────────────────────────
  function onProgress(d) { BV.upload.onProgress(d); }

  function onComplete(d) {
    _jobId = d.job_id; _result = d.result;
    BV.upload.setProgress(100, `Analysis complete ✓  (${_result.analysis_time}s)`);
    setTimeout(() => BV.upload.showProgress(false), 2000);
    BV.toast(`✓ Analysis complete in ${_result.analysis_time}s  |  ${Object.keys(_result).length} blocks`, 'ok', 4000);

    // Log capability report
    console.group('[BV] Analysis complete');
    console.log('BPM:', _result.beats?.bpm_consensus?.toFixed(1));
    console.log('Key:', _result.harmony?.key_full);
    console.log('LUFS:', _result.dynamics?.loudness?.integrated_lufs?.toFixed(1));
    console.log('Blocks:', Object.keys(_result).join(', '));
    console.log('Praat:', _result.voice?.available, '| eGeMAPS:', _result.egemaps?.available);
    console.groupEnd();

    _renderAll();
    _initPlayerFromResult();
    document.getElementById('player-bar').classList.remove('hidden');
    switchTab('waveform');
  }

  function onError(d) {
    BV.toast('Analysis error: ' + (d.error||'Unknown'), 'error', 8000);
    BV.upload.showProgress(false);
    console.error('[BV] error', d);
  }

  function _renderAll() {
    if (!_result) return;
    BV.viz2d.renderAll(_result);
  }

  // ── Player ────────────────────────────────────────────────────────────────
  function _initPlayer() {
    document.getElementById('btn-play')?.addEventListener('click', _togglePlay);
    document.getElementById('timeline-wrap')?.addEventListener('click', e => {
      if (!_result) return;
      const rect = e.currentTarget.getBoundingClientRect();
      _player.time = ((e.clientX - rect.left) / rect.width) * _player.duration;
      _updateTimeline();
    });
  }

  function _initPlayerFromResult() {
    _player.duration = _result?.metadata?.duration||0; _player.time = 0;
    _updateTimeline();
    const bt=document.getElementById('beat-ticks');
    if (bt) bt.innerHTML=(_result.beats?.beat_times||[]).map(t=>
      `<div class="beat-tick" style="left:${(t/_player.duration*100).toFixed(2)}%"></div>`).join('');
    const st=document.getElementById('seg-ticks');
    if (st) st.innerHTML=(_result.structure?.segment_times||[]).map(t=>
      `<div class="seg-tick" style="left:${(t/_player.duration*100).toFixed(2)}%"></div>`).join('');
  }

  function _togglePlay() {
    _player.playing = !_player.playing;
    document.getElementById('btn-play').textContent = _player.playing ? '⏸' : '▶';
    if (_player.playing) _playerTick();
    else if (_player.raf) cancelAnimationFrame(_player.raf);
  }

  function _playerTick() {
    if (!_player.playing) return;
    _player.time += 1/60;
    if (_player.time >= _player.duration) _player.time = 0;
    _updateTimeline(); _player.raf = requestAnimationFrame(_playerTick);
  }

  function _updateTimeline() {
    const pct = _player.duration>0 ? (_player.time/_player.duration*100) : 0;
    const fill=document.getElementById('timeline-fill'), cursor=document.getElementById('timeline-cursor');
    if (fill) fill.style.width=pct+'%';
    if (cursor) cursor.style.left=pct+'%';
    const te=document.getElementById('player-time');
    if (te) te.textContent=`${BV.fmt.time(_player.time)} / ${BV.fmt.time(_player.duration)}`;
  }

  // ── Metronome ─────────────────────────────────────────────────────────────
  function _toggleMetronome() {
    if (!_result?.beats?.bpm_consensus) return;
    const bpm = _result.beats.bpm_consensus;
    if (_metronome) {
      try { _metronome.stop(); _metronome.dispose(); } catch(e) {}
      _metronome = null;
      BV.toast('Metronome stopped', 'info', 1000); return;
    }
    try {
      Tone.start();
      Tone.getTransport().bpm.value = bpm;
      _metronome = new Tone.Sequence((time) => {
        new Tone.MetalSynth({
          frequency:400, envelope:{attack:.001,decay:.08,release:.01},
          harmonicity:5.1, modulationIndex:32, resonance:4000, octaves:1.5
        }).toDestination().triggerAttackRelease('32n', time);
        const ring=document.getElementById('bpm-ring');
        if (ring) { ring.classList.remove('pulse'); void ring.offsetWidth; ring.classList.add('pulse'); }
      }, '4n', Infinity).start(0);
      Tone.getTransport().start();
      BV.toast(`♩ Metronome: ${bpm.toFixed(1)} BPM  (click again to stop)`, 'ok', 2500);
    } catch(e) { BV.toast('Tone.js: '+e.message, 'error'); }
  }

  // ── Export ────────────────────────────────────────────────────────────────
  async function exportResult(fmt) {
    if (!_jobId) { BV.toast('No analysis to export', 'warn'); return; }
    document.querySelector('.export-dropdown')?.classList.remove('open');
    const a=document.createElement('a'); a.href=`/api/export/${_jobId}/${fmt}`; a.download=''; a.click();
    BV.toast(`Exporting ${fmt.toUpperCase()}…`, 'ok', 1500);
  }

  // ── Particles ─────────────────────────────────────────────────────────────
  function _initParticles() {
    _particleCanvas = document.getElementById('particle-bg'); if (!_particleCanvas) return;
    _pctx = _particleCanvas.getContext('2d');
    const resize=()=>{ _particleCanvas.width=_particleCanvas.offsetWidth; _particleCanvas.height=_particleCanvas.offsetHeight; };
    resize(); window.addEventListener('resize', resize);
    for (let i=0;i<60;i++) _particles.push(_mkParticle());
    _animParticles();
  }

  function _mkParticle() {
    return { x:Math.random()*(_particleCanvas.width||800), y:Math.random()*(_particleCanvas.height||600),
             vx:(Math.random()-.5)*.4, vy:(Math.random()-.5)*.4,
             r:Math.random()*1.5+.5, alpha:Math.random()*.4+.1,
             color:Math.random()>.5?'0,245,212':'247,0,255' };
  }

  function _animParticles() {
    requestAnimationFrame(_animParticles);
    if (_currentTab !== 'upload') return;
    const W=_particleCanvas.width, H=_particleCanvas.height;
    _pctx.clearRect(0,0,W,H);
    _particles.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0)p.x=W; if(p.x>W)p.x=0; if(p.y<0)p.y=H; if(p.y>H)p.y=0;
      _pctx.beginPath(); _pctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      _pctx.fillStyle=`rgba(${p.color},${p.alpha})`; _pctx.fill();
    });
    for (let i=0;i<_particles.length;i++) {
      for (let j=i+1;j<_particles.length;j++) {
        const dx=_particles[i].x-_particles[j].x, dy=_particles[i].y-_particles[j].y;
        const d=Math.sqrt(dx*dx+dy*dy);
        if (d<100) {
          _pctx.beginPath(); _pctx.moveTo(_particles[i].x,_particles[i].y); _pctx.lineTo(_particles[j].x,_particles[j].y);
          _pctx.strokeStyle=`rgba(0,245,212,${.06*(1-d/100)})`; _pctx.lineWidth=.5; _pctx.stroke();
        }
      }
    }
  }

  // ── Keyboard ──────────────────────────────────────────────────────────────
  function _initKeyboard() {
    const map={'1':'upload','2':'waveform','3':'spectrum','4':'panel-3d','5':'beats',
               '6':'harmony','7':'mfcc','8':'pitch','9':'dynamics','0':'structure','-':'realtime'};
    document.addEventListener('keydown', e => {
      if (e.target.tagName==='INPUT') return;
      if (map[e.key]) switchTab(map[e.key]);
      if (e.key==='m'||e.key==='M') _toggleMetronome();
      if (e.key===' ') { e.preventDefault(); _togglePlay(); }
      if (e.key==='Escape') document.querySelector('.export-dropdown')?.classList.remove('open');
    });
  }

  document.addEventListener('DOMContentLoaded', init);
  return { onProgress, onComplete, onError, switchTab, exportResult };
})();
console.log('[BV] app.js loaded');
