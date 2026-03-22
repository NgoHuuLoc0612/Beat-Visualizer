/**
 * Beat Visualizer v3 — 2D Visualizer (corrected for v4 analyzer key paths)
 *
 * Key path mapping from v4 analyzer:
 *   beats.fourier_tempogram (not fourier_tempogram_mag)
 *   spectral.entropy (not spec_entropy)
 *   chords.chords (separate block, not harmony.chords_over_time)
 *   chords.tonal_tension (not harmony.tonal_tension)
 *   structure.segment_times_laplacian (not segment_times_lp)
 *   structure.self_similarity_path (path SSM)
 *   spectral.band_energy.ultra (9th band, not 'ultrasonic')
 *   gammatone.channels — new block
 *   nmf.basis/activations — new block
 *   stft.main — main STFT block
 */
BV.viz2d = (() => {
  let _R = null;
  let _wfMode = 'standard', _specMode = 'spectrogram', _chromaMode = 'cqt', _mfccView = 'mfcc40';

  // ── helpers ────────────────────────────────────────────────────────────────
  const _get = (obj, path, def=null) => {
    try { return path.split('.').reduce((o,k)=>o[k], obj) ?? def; }
    catch { return def; }
  };
  const _ds = (arr, n=400) => arr?.length > n ? BV.downsample(arr, n) : (arr||[]);
  const _line = (id, times, datasets, extraOpts={}) => {
    BV.createChart(id, 'line', {
      labels: times.map(BV.fmt.time),
      datasets: datasets.map(d => ({borderWidth:1.2,pointRadius:0,...d}))
    }, extraOpts);
  };

  // ── WAVEFORM ──────────────────────────────────────────────────────────────
  function renderWaveform(r, mode) {
    _wfMode = mode || _wfMode;
    const canvas = document.getElementById('canvas-waveform');
    if (!canvas || !r.waveform) return;
    const W = canvas.offsetWidth||800, H = canvas.offsetHeight||200;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    const samples = r.waveform.samples, dur = r.waveform.duration;
    const ds = BV.downsample(samples, W), mid = H/2;
    const showBeats  = document.getElementById('wf-show-beats')?.checked;
    const showOnsets = document.getElementById('wf-show-onsets')?.checked;
    const showSegs   = document.getElementById('wf-show-segs')?.checked;

    ctx.fillStyle = '#0d1017'; ctx.fillRect(0,0,W,H);

    if (_wfMode === 'standard') {
      ctx.beginPath(); ctx.strokeStyle='#00f5d4'; ctx.lineWidth=1.2;
      ds.forEach((v,i)=>{ const y=mid-v*mid*.9; i?ctx.lineTo(i,y):ctx.moveTo(i,y); }); ctx.stroke();
    } else if (_wfMode === 'filled') {
      const g=ctx.createLinearGradient(0,0,0,H);
      g.addColorStop(0,'rgba(0,245,212,.6)'); g.addColorStop(.5,'rgba(0,245,212,.15)'); g.addColorStop(1,'rgba(247,0,255,.1)');
      ctx.beginPath(); ctx.fillStyle=g; ctx.moveTo(0,mid);
      ds.forEach((v,i)=>ctx.lineTo(i,mid-v*mid*.9));
      ctx.lineTo(W,mid); ctx.closePath(); ctx.fill();
    } else if (_wfMode === 'bars') {
      const bw=Math.max(1,W/ds.length-.5);
      ds.forEach((v,i)=>{ const h=Math.abs(v)*mid*.9; const g=ctx.createLinearGradient(0,mid-h,0,mid+h);
        g.addColorStop(0,'#00f5d4'); g.addColorStop(1,'#f700ff'); ctx.fillStyle=g;
        ctx.fillRect(i*(bw+.5),mid-h,bw,h*2); });
    } else if (_wfMode === 'mirror') {
      const g=ctx.createLinearGradient(0,0,0,H);
      g.addColorStop(0,'rgba(0,245,212,.5)'); g.addColorStop(.5,'transparent'); g.addColorStop(1,'rgba(247,0,255,.5)');
      ctx.fillStyle=g; ctx.beginPath(); ctx.moveTo(0,mid);
      ds.forEach((v,i)=>ctx.lineTo(i,mid-v*mid*.9)); ctx.lineTo(W,mid); ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.moveTo(0,mid);
      ds.forEach((v,i)=>ctx.lineTo(i,mid+Math.abs(v)*mid*.9)); ctx.lineTo(W,mid); ctx.closePath(); ctx.fill();
    } else if (_wfMode === 'stereo') {
      const rms=r.energy?.rms||[], rmsT=r.energy?.times||[], maxR=Math.max(...rms)||1;
      const g=ctx.createLinearGradient(0,0,0,H);
      g.addColorStop(0,'rgba(0,245,212,.5)'); g.addColorStop(1,'rgba(0,245,212,.05)');
      ctx.fillStyle=g; ctx.beginPath(); ctx.moveTo(0,H);
      rms.forEach((v,i)=>ctx.lineTo((rmsT[i]/dur)*W, H-(v/maxR)*H*.85));
      ctx.lineTo(W,H); ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.strokeStyle='#00f5d4'; ctx.lineWidth=1.5;
      rms.forEach((v,i)=>{ const x=(rmsT[i]/dur)*W, y=H-(v/maxR)*H*.85; i?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke();
    }

    // overlays
    if (showBeats) {
      ctx.strokeStyle='rgba(0,245,212,.55)'; ctx.lineWidth=1;
      (r.beats?.beat_times||[]).forEach(t=>{ const x=(t/dur)*W; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); });
    }
    if (showOnsets) {
      ctx.strokeStyle='rgba(247,0,255,.4)'; ctx.lineWidth=.8; ctx.setLineDash([3,3]);
      (r.onsets?.onsets_energy||[]).forEach(t=>{ const x=(t/dur)*W; ctx.beginPath(); ctx.moveTo(x,H*.3); ctx.lineTo(x,H); ctx.stroke(); });
      ctx.setLineDash([]);
    }
    if (showSegs) {
      ctx.strokeStyle='rgba(255,107,53,.8)'; ctx.lineWidth=2;
      (r.structure?.segment_times||[]).forEach(t=>{ const x=(t/dur)*W; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); });
    }
  }

  // ── SPECTRUM ──────────────────────────────────────────────────────────────
  function renderSpectrum(r, mode) {
    _specMode = mode || _specMode;
    const titleEl = document.getElementById('spec-card-title');
    const canvas = document.getElementById('canvas-spectrogram');

    if (_specMode === 'spectrogram') {
      if (titleEl) titleEl.textContent = 'STFT SPECTROGRAM (4096-pt STFT, Plasma colormap)';
      if (canvas) BV.drawHeatmap(canvas, r.spectral?.magnitude_db, BV.color.plasma, true);
    } else if (_specMode === 'mel') {
      if (titleEl) titleEl.textContent = 'MEL SPECTROGRAM (128 Mel bands, Inferno)';
      if (canvas) BV.drawHeatmap(canvas, r.mel?.mel_db, BV.color.inferno, true);
    } else if (_specMode === 'cqt') {
      if (titleEl) titleEl.textContent = 'CONSTANT-Q TRANSFORM (84 bins, 36bpo, Viridis)';
      if (canvas) BV.drawHeatmap(canvas, r.cqt?.cqt_db, BV.color.viridis, true);
    } else if (_specMode === 'gammatone') {
      if (titleEl) titleEl.textContent = 'GAMMATONE FILTERBANK (64 channels, Cyan)';
      if (canvas && r.gammatone?.channels) BV.drawHeatmap(canvas, r.gammatone.channels, BV.color.cyan, true);
    } else if (_specMode === 'power') {
      if (titleEl) titleEl.textContent = 'POWER SPECTRUM (time-averaged, dBFS)';
      if (canvas) {
        const freqs = r.spectral?.freqs||[], ps = r.spectral?.power_spectrum_db||[];
        BV.createChart('canvas-spectrogram','line',{
          labels: freqs.map(f=>BV.fmt.hz(f)),
          datasets:[{data:ps,borderColor:'#00f5d4',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(0,245,212,.07)'}]
        },{scales:{x:{display:true,ticks:{color:'#4a5568',maxTicksLimit:10,font:{size:8,family:"'Space Mono'"}}},y:{display:true}}});
      }
    } else if (_specMode === 'bands') {
      if (titleEl) titleEl.textContent = '9-BAND PSYCHOACOUSTIC ENERGY';
      _renderBandChart(r);
    } else if (_specMode === 'bark') {
      if (titleEl) titleEl.textContent = 'BARK SCALE (24 critical bands)';
      _renderBarkChart(r);
    }

    _renderSpectralFeatures(r);
    _renderContrast(r);
    _renderTristimulus(r);
    if (_specMode !== 'bands') _renderBandChart(r);
  }

  function _renderSpectralFeatures(r) {
    const spec = r.spectral; if (!spec) return;
    const step = Math.max(1, Math.floor((spec.times||[]).length/400));
    const t = (spec.times||[]).filter((_,i)=>i%step===0);
    BV.createChart('chart-spec-features','line',{
      labels: t.map(BV.fmt.time),
      datasets:[
        {label:'Centroid',    data:(spec.centroid||[]).filter((_,i)=>i%step===0),   borderColor:'#00f5d4',borderWidth:1.2,pointRadius:0},
        {label:'Rolloff-85',  data:(spec.rolloff_85||[]).filter((_,i)=>i%step===0), borderColor:'#f700ff',borderWidth:1,  pointRadius:0},
        {label:'Rolloff-95',  data:(spec.rolloff_95||[]).filter((_,i)=>i%step===0), borderColor:'#7b61ff',borderWidth:1,  pointRadius:0},
        {label:'Bandwidth',   data:(spec.bandwidth||[]).filter((_,i)=>i%step===0),  borderColor:'#ff6b35',borderWidth:1,  pointRadius:0},
        {label:'Entropy',     data:(spec.entropy||[]).filter((_,i)=>i%step===0),    borderColor:'#ffcc00',borderWidth:1,  pointRadius:0},
      ]
    },{plugins:{legend:{display:true,labels:{color:'#8b9ab0',boxWidth:10,font:{size:9}}}}});
  }

  function _renderContrast(r) {
    const spec = r.spectral; if (!spec?.contrast) return;
    const step = Math.max(1, Math.floor((spec.times||[]).length/300));
    const t = (spec.times||[]).filter((_,i)=>i%step===0);
    BV.createChart('chart-contrast','line',{
      labels: t.map(BV.fmt.time),
      datasets: spec.contrast.map((band,i)=>({
        label:`B${i+1}`, data:band.filter((_,j)=>j%step===0),
        borderColor:`hsl(${i*50},80%,55%)`, borderWidth:1, pointRadius:0
      }))
    },{plugins:{legend:{display:true,labels:{color:'#8b9ab0',boxWidth:8,font:{size:8}}}}});
  }

  function _renderTristimulus(r) {
    const spec = r.spectral; if (!spec?.tristimulus_t1) return;
    const step = Math.max(1, Math.floor((spec.times||[]).length/300));
    const t = (spec.times||[]).filter((_,i)=>i%step===0);
    BV.createChart('chart-tristimulus','line',{
      labels: t.map(BV.fmt.time),
      datasets:[
        {label:'T1 (f0)',  data:(spec.tristimulus_t1||[]).filter((_,i)=>i%step===0), borderColor:'#00f5d4',borderWidth:1.2,pointRadius:0,fill:true,backgroundColor:'rgba(0,245,212,.06)'},
        {label:'T2 (2-4f)',data:(spec.tristimulus_t2||[]).filter((_,i)=>i%step===0), borderColor:'#f700ff',borderWidth:1,  pointRadius:0},
        {label:'T3 (5f+)', data:(spec.tristimulus_t3||[]).filter((_,i)=>i%step===0), borderColor:'#ff6b35',borderWidth:1,  pointRadius:0},
      ]
    },{plugins:{legend:{display:true,labels:{color:'#8b9ab0',boxWidth:8,font:{size:8}}}}});
  }

  function _renderBandChart(r) {
    const be = r.spectral?.band_energy; if (!be) return;
    const labels = Object.keys(be).map(k=>k.replace(/_/g,' ').toUpperCase());
    const data   = Object.values(be);
    const colors = ['#00f5d4','#39d353','#0ea5e9','#7b61ff','#f700ff','#ff6b35','#ffcc00','#ff3b5c','#ffffff'];
    BV.createChart('chart-band-energy','bar',{
      labels, datasets:[{data,backgroundColor:colors.slice(0,data.length),borderWidth:0}]
    },{scales:{x:{display:true,ticks:{color:'#4a5568',font:{size:8,family:"'Space Mono'"}}},y:{display:true}}});
  }

  function _renderBarkChart(r) {
    const be = r.spectral?.bark_energy; if (!be) return;
    BV.createChart('chart-band-energy','bar',{
      labels: be.map(b=>BV.fmt.hz(b.lo)),
      datasets:[{data:be.map(b=>b.energy),backgroundColor:be.map((_,i)=>`hsl(${180-i*7},80%,55%)`),borderWidth:0}]
    },{scales:{x:{display:true,ticks:{color:'#4a5568',font:{size:7,family:"'Space Mono'"}}},y:{display:true}}});
  }

  // ── BEATS ─────────────────────────────────────────────────────────────────
  function renderBeats(r) {
    const beats = r.beats; if (!beats) return;

    // BPM ring
    const bpm = beats.bpm_consensus || 120;
    const el = document.getElementById('bpm-display');
    if (el) el.textContent = bpm.toFixed(1);
    document.getElementById('bpm-cat').textContent = r.music_info?.tempo_category||'';
    const ring = document.getElementById('bpm-ring');
    if (ring) {
      const pct = Math.min(100, ((bpm-40)/260)*100);
      ring.style.setProperty('--pct', pct+'%');
      ring.style.setProperty('--beat', (60/bpm)+'s');
    }
    document.getElementById('stat-bpm-dp').textContent  = beats.bpm?.toFixed(1)||'—';
    document.getElementById('stat-bpm-p').textContent   = beats.bpm_perc?.toFixed(1)||'—';
    document.getElementById('stat-bpm-conf').textContent = beats.bpm_confidence?.toFixed(3)||'—';

    // Onset strength (full + percussive)
    const oe = beats.onset_strength||[], oeP = beats.onset_strength_perc||[];
    const oeT = beats.onset_times||[];
    const step = Math.max(1, Math.floor(oe.length/400));
    _line('chart-onset', oeT.filter((_,i)=>i%step===0), [
      {data:oe.filter((_,i)=>i%step===0), borderColor:'#00f5d4', fill:true, backgroundColor:'rgba(0,245,212,.1)'},
      {data:oeP.filter((_,i)=>i%step===0),borderColor:'#f700ff', fill:false}
    ]);

    // Tempogram heatmap
    const tgC = document.getElementById('canvas-tempogram');
    if (tgC && beats.tempogram?.length) BV.drawHeatmap(tgC, beats.tempogram, BV.color.hot, true);

    // Tempo candidates bar
    const cands = (beats.tempo_candidates||[]).slice(0,8);
    if (cands.length) {
      BV.createChart('chart-tempo-candidates','bar',{
        labels: cands.map(c=>`${c.bpm.toFixed(1)}`),
        datasets:[{data:cands.map(c=>c.strength), backgroundColor:'#00f5d4', borderWidth:0}]
      },{scales:{x:{display:true,ticks:{color:'#4a5568',font:{size:9,family:"'Space Mono'"}}},y:{display:true}}});
    }

    // PLP envelope
    const plp = beats.plp_env||[];
    if (plp.length) {
      const plpStep = Math.max(1,Math.floor(plp.length/400));
      _line('chart-plp', oeT.filter((_,i)=>i%plpStep===0), [
        {data:plp.filter((_,i)=>i%plpStep===0), borderColor:'#7b61ff', fill:true, backgroundColor:'rgba(123,97,255,.1)'}
      ]);
    }

    // Fourier tempogram magnitude (key: beats.fourier_tempogram)
    // This is a complex matrix — render as heatmap of abs
    const ft = beats.fourier_tempogram;
    if (ft?.length) {
      const ftAbs = ft.map(row => (Array.isArray(row[0]) ? row.map(c=>Math.sqrt(c[0]**2+(c[1]||0)**2)) : row));
      const ftC = document.getElementById('canvas-tempogram');
      if (ftC) BV.drawHeatmap(ftC, ftAbs.slice(0,40), BV.color.hot, true);
    }
  }

  // ── HARMONY ───────────────────────────────────────────────────────────────
  function renderHarmony(r, chromaMode) {
    _chromaMode = chromaMode || _chromaMode;
    const h = r.harmony; const ch = r.chroma; const chords = r.chords;
    if (!h) return;

    // Key badge
    document.getElementById('key-name-display').textContent = h.key||'—';
    document.getElementById('key-mode-display').textContent = h.mode||'—';
    document.getElementById('key-conf-display').textContent = 'conf: '+(h.key_confidence?.toFixed(3)||'—');
    const lbl = document.getElementById('key-profiles-lbl');
    if (lbl) lbl.textContent =
      `KS: ${h.ks_estimate?.key} ${h.ks_estimate?.mode}  |  Temperley: ${h.temperley_estimate?.key} ${h.temperley_estimate?.mode}`+
      (h.relative_key ? `  |  Relative: ${h.relative_key}` : '');

    // KS profile
    const ks = h.ks_estimate;
    if (ks?.major_scores) {
      BV.createChart('chart-ks-profile','bar',{
        labels: BV.pitchClasses,
        datasets:[
          {label:'Major',data:ks.major_scores,backgroundColor:'rgba(0,245,212,.5)',borderColor:'#00f5d4',borderWidth:1},
          {label:'Minor',data:ks.minor_scores,backgroundColor:'rgba(247,0,255,.3)',borderColor:'#f700ff',borderWidth:1},
        ]
      },{plugins:{legend:{display:true,labels:{color:'#8b9ab0',boxWidth:8,font:{size:9}}}},
         scales:{x:{display:true,ticks:{color:'#8b9ab0',font:{size:9}}},y:{display:true}}});
    }

    // Pitch wheel
    const pw = document.getElementById('canvas-pitch-wheel');
    if (pw && ch?.chroma_mean) _renderPitchWheel(pw, ch.chroma_mean);

    // Chromagram heatmap
    const chromaC = document.getElementById('canvas-chromagram');
    if (chromaC && ch) {
      const key = {cqt:'chroma_cqt',cens:'chroma_cens',stft:'chroma_stft'}[_chromaMode];
      BV.drawHeatmap(chromaC, ch[key], BV.color.viridis, true);
    }

    // Chord timeline — from r.chords.chords
    const chordList = chords?.chords||[];
    _renderChordTimeline(chordList, r.metadata?.duration||60);

    // Tonal tension — from r.chords.tonal_tension
    const tension = chords?.tonal_tension||[];
    if (tension.length) {
      const tStep = Math.max(1,Math.floor(tension.length/400));
      BV.createChart('chart-tension','line',{
        labels: _ds(ch?.times||[], tension.length).map(BV.fmt.time),
        datasets:[{data:tension.filter((_,i)=>i%tStep===0),borderColor:'#ff3b5c',borderWidth:1.2,pointRadius:0,fill:true,backgroundColor:'rgba(255,59,92,.08)'}]
      },{scales:{y:{min:0,max:1}}});
    }

    // HCDF
    const hcdf = ch?.hcdf||[];
    if (hcdf.length) {
      const hStep = Math.max(1,Math.floor(hcdf.length/400));
      _line('chart-hcdf', (ch.times||[]).filter((_,i)=>i%hStep===0), [
        {data:hcdf.filter((_,i)=>i%hStep===0),borderColor:'#ffcc00',fill:true,backgroundColor:'rgba(255,204,0,.08)'}
      ]);
    }

    // Tonnetz — 6 dims stacked
    const tn = r.tonnetz;
    if (tn?.tonnetz?.length===6) {
      const step = Math.max(1,Math.floor((tn.tonnetz[0]||[]).length/400));
      const cols = ['#00f5d4','#f700ff','#ff6b35','#7b61ff','#ffcc00','#39d353'];
      const dnames = tn.dims || ['5th-cos','5th-sin','M3-cos','M3-sin','m3-cos','m3-sin'];
      BV.createChart('chart-tonnetz','line',{
        labels: (tn.times||[]).filter((_,i)=>i%step===0).map(BV.fmt.time),
        datasets: tn.tonnetz.map((dim,i)=>({
          label:dnames[i], data:dim.filter((_,j)=>j%step===0),
          borderColor:cols[i], borderWidth:1, pointRadius:0
        }))
      },{plugins:{legend:{display:true,labels:{color:'#8b9ab0',boxWidth:8,font:{size:9}}}}});
    }
  }

  function _renderPitchWheel(canvas, chroma_mean) {
    const ctx=canvas.getContext('2d'), W=canvas.width, H=canvas.height, cx=W/2, cy=H/2;
    const r=Math.min(W,H)/2-10, max=Math.max(...chroma_mean)||1, sa=(Math.PI*2)/12;
    ctx.clearRect(0,0,W,H);
    BV.pitchClasses.forEach((pc,i)=>{
      const startA=i*sa-Math.PI/2, v=chroma_mean[i]/max;
      const ir=r*.3, or=r*.3+v*r*.65;
      ctx.beginPath();
      ctx.moveTo(cx+Math.cos(startA)*ir, cy+Math.sin(startA)*ir);
      ctx.arc(cx,cy,or,startA,startA+sa-.03);
      ctx.arc(cx,cy,ir,startA+sa-.03,startA,true);
      ctx.closePath();
      const hue=i*30;
      ctx.fillStyle=`hsla(${hue},85%,55%,${.4+v*.55})`; ctx.strokeStyle=`hsla(${hue},85%,70%,.5)`; ctx.lineWidth=.5;
      ctx.fill(); ctx.stroke();
      ctx.fillStyle=v>.3?'#e8edf5':'#4a5568'; ctx.font="bold 9px 'Space Mono'"; ctx.textAlign='center'; ctx.textBaseline='middle';
      const la=startA+sa/2;
      ctx.fillText(pc, cx+Math.cos(la)*r*.2, cy+Math.sin(la)*r*.2);
    });
  }

  function _renderChordTimeline(chords, duration) {
    const track = document.getElementById('chord-track');
    if (!track||!chords?.length) return;
    track.innerHTML='';
    chords.forEach((c,i)=>{
      const nextT=(chords[i+1]?.time)||duration;
      const x=(c.time/duration)*100, w=((nextT-c.time)/duration)*100;
      const col = BV.chordColor(c.chord||c.label||'N');
      const el=document.createElement('div'); el.className='chord-block';
      el.style.cssText=`left:${x.toFixed(2)}%;width:${Math.max(w,.2).toFixed(2)}%;background:${col}22;border-color:${col}88;color:${col}`;
      el.textContent=(c.chord||c.label||'').replace(':','\u00A0');
      el.title=`${c.chord||c.label} @ ${BV.fmt.time(c.time)} (${(c.score||0).toFixed(2)})`;
      track.appendChild(el);
    });
  }

  // ── MFCC ─────────────────────────────────────────────────────────────────
  function renderMFCC(r, view) {
    _mfccView = view || _mfccView;
    const m = r.mfcc; if (!m) return;
    const canvas = document.getElementById('canvas-mfcc');
    const titleEl = document.getElementById('mfcc-hm-title');
    const matMap = {
      mfcc40: [m.mfcc,         'MFCC-40 MATRIX (Δ/ΔΔ also available)'],
      mfcc13: [m.mfcc13,       'MFCC-13 MATRIX'],
      delta:  [m.mfcc_delta,   'MFCC Δ — VELOCITY'],
      delta2: [m.mfcc_delta2,  'MFCC ΔΔ — ACCELERATION'],
      bfcc:   [m.bfcc,         'BFCC — BARK FREQUENCY CEPSTRAL COEFFICIENTS'],
      cmvn:   [m.mfcc_cmvn,    'MFCC CMVN (cepstral mean-variance normalized)'],
    };
    const [matrix, title] = matMap[_mfccView]||matMap.mfcc40;
    if (titleEl) titleEl.textContent = title;
    if (canvas && matrix?.length) {
      BV.drawHeatmap(canvas, matrix, _mfccView==='delta2'?BV.color.hot:BV.color.magma, false);
    }

    const means = m.mfcc_mean||[], stds = m.mfcc_std||[];
    if (means.length) {
      BV.createChart('chart-mfcc-mean','bar',{
        labels: means.map((_,i)=>`C${i}`),
        datasets:[{data:means,backgroundColor:means.map(v=>v>0?'rgba(0,245,212,.6)':'rgba(247,0,255,.6)'),borderWidth:0}]
      },{scales:{x:{display:true,ticks:{color:'#4a5568',font:{size:8,family:"'Space Mono'"}}},y:{display:true}}});
    }
    if (stds.length) {
      BV.createChart('chart-mfcc-std','bar',{
        labels: stds.map((_,i)=>`C${i}`),
        datasets:[{data:stds,backgroundColor:'rgba(123,97,255,.5)',borderWidth:0}]
      },{scales:{x:{display:true,ticks:{color:'#4a5568',font:{size:8,family:"'Space Mono'"}}},y:{display:true}}});
    }

    // LPC
    const lpc = m.lpc_mean||[];
    if (lpc.length) {
      BV.createChart('chart-lpc','line',{
        labels: lpc.map((_,i)=>`a${i}`),
        datasets:[{label:'Mean',data:lpc,borderColor:'#ff6b35',borderWidth:1.5,pointRadius:3,pointBackgroundColor:'#ff6b35'}]
      },{plugins:{legend:{display:false}},
         scales:{x:{display:true,ticks:{color:'#4a5568',font:{size:9,family:"'Space Mono'"}}},y:{display:true}}});
    }

    // NMF if available
    const nmf = r.nmf;
    if (nmf?.available && nmf.basis?.length) {
      // Render NMF basis as heatmap if canvas exists
      const nmfC = document.getElementById('canvas-mfcc');
      if (nmfC && _mfccView === 'mfcc40') BV.drawHeatmap(nmfC, m.mfcc||[], BV.color.magma, false);
    }
  }

  // ── PITCH ─────────────────────────────────────────────────────────────────
  function renderPitch(r) {
    const pitch = r.pitch; if (!pitch) return;
    const step = Math.max(1, Math.floor((pitch.f0_pyin?.length||1)/500));
    const times = (pitch.f0_times||[]).filter((_,i)=>i%step===0);
    const pyin  = (pitch.f0_pyin||[]).filter((_,i)=>i%step===0);
    const praat = pitch.f0_praat?.f0 ? BV.downsample(pitch.f0_praat.f0, pyin.length) : [];
    const showPraat  = document.getElementById('pitch-show-praat')?.checked;
    const showVoiced = document.getElementById('pitch-show-voiced')?.checked;

    const pyinPlot  = showVoiced ? pyin.map(v=>v>0?v:null)  : pyin;
    const praatPlot = showVoiced ? praat.map(v=>v>0?v:null) : praat;
    const datasets  = [{label:'pYIN',data:pyinPlot,borderColor:'#00f5d4',borderWidth:1.5,pointRadius:0,spanGaps:false}];
    if (showPraat&&praat.length) datasets.push({label:'Praat',data:praatPlot,borderColor:'#f700ff',borderWidth:1,pointRadius:0,spanGaps:false});

    BV.createChart('chart-f0','line',{labels:times.map(BV.fmt.time),datasets},{
      plugins:{legend:{display:true,labels:{color:'#8b9ab0',boxWidth:8,font:{size:9}}}},
      scales:{y:{display:true,ticks:{color:'#4a5568',font:{size:9},callback:v=>Math.round(v)+'Hz'}}}
    });

    const vp=(pitch.voiced_prob||[]).filter((_,i)=>i%step===0);
    _line('chart-voiced-prob', times, [{data:vp,borderColor:'#ffcc00',fill:true,backgroundColor:'rgba(255,204,0,.08)'}],
          {scales:{y:{min:0,max:1}}});

    const ps = pitch.pitch_stats||{};
    document.getElementById('stat-f0-mean').textContent  = ps.mean_hz?.toFixed(1)||'—';
    document.getElementById('stat-f0-range').textContent = ps.range_st?.toFixed(1)||'—';
    document.getElementById('stat-voiced').textContent   = ps.voiced_ratio?(ps.voiced_ratio*100).toFixed(1)+'%':'—';
    const vib = pitch.vibrato||{};
    document.getElementById('stat-vibrato').textContent  = vib.detected?`${vib.rate_hz}Hz`:'None';
    document.getElementById('stat-contour').textContent  = pitch.melodic_contour?.direction||'—';

    _renderVoiceTable(r.voice);
  }

  function _renderVoiceTable(voice) {
    const tbody = document.getElementById('vq-tbody'); if (!tbody) return;
    if (!voice?.available) { tbody.innerHTML='<tr><td colspan="4" style="color:var(--text3);padding:8px 10px">Praat not available or non-vocal audio</td></tr>'; return; }
    const qual=(v,lo,hi)=>v>hi?'bad':v>lo?'warn':'good';
    const rows=[
      ...Object.entries(voice.jitter||{}).map(([k,v])=>[`Jitter (${k})`, typeof v==='number'?v.toExponential(3):v, 'Jitter', qual(v,0.005,0.01)]),
      ...Object.entries(voice.shimmer||{}).map(([k,v])=>[`Shimmer (${k})`, BV.fmt.n4(v), 'Shimmer', qual(v,0.03,0.05)]),
      ['HNR (mean)', (voice.hnr_mean_db?.toFixed(2)||'—')+' dB', 'HNR', voice.hnr_mean_db<10?'bad':voice.hnr_mean_db<20?'warn':'good'],
      ['NHR', voice.nhr?.toExponential(3)||'—', 'NHR', ''],
      ['CPP', (voice.cpp_db?.toFixed(2)||'—')+' dB', 'CPP', ''],
      ['VAD ratio', (voice.vad_ratio*100)?.toFixed(1)+'%', 'VAD', ''],
      ...Object.entries(voice.formants||{}).map(([k,v])=>[`${k} mean`, v.mean_hz?.toFixed(0)+' Hz', 'Formant', '']),
    ];
    tbody.innerHTML=rows.map(([label,val,type,status])=>
      `<tr><td>${label}</td><td class="vq-val${status?' vq-'+status:''}">${val}</td><td style="color:var(--text3)">${type}</td><td class="vq-${status}">${status.toUpperCase()}</td></tr>`
    ).join('');
  }

  // ── DYNAMICS ──────────────────────────────────────────────────────────────
  function renderDynamics(r) {
    const dyn=r.dynamics; if (!dyn) return;
    const loud=dyn.loudness||{};

    const _lufs=(tid,fid,val,lo,hi,unit)=>{
      const el=document.getElementById(tid); if(el) el.textContent=(val?.toFixed?.(1)||'—')+unit;
      const fill=document.getElementById(fid);
      if(fill&&val!=null) fill.style.width=(Math.max(0,Math.min(100,((val-lo)/(hi-lo))*100)))+'%';
    };
    _lufs('lufs-int',  'lufs-fill-int',  loud.integrated_lufs, -60, 0, ' LUFS');
    _lufs('lufs-peak', 'lufs-fill-peak', loud.true_peak_dbtp,  -20, 0, ' dBTP');
    _lufs('lufs-lra',  'lufs-fill-lra',  loud.loudness_range,    0,20, ' LU');

    document.getElementById('stat-dr').textContent    = (dyn.dynamic_range?.toFixed(1)||'—')+' dB';
    document.getElementById('stat-crest').textContent = (dyn.crest_factor?.toFixed(1)||'—')+' dB';

    const mi=r.music_info||{};
    document.getElementById('stat-dance').textContent     = mi.danceability!=null?(mi.danceability*100).toFixed(0)+'%':'—';
    document.getElementById('stat-energy-mi').textContent = mi.energy!=null?(mi.energy*100).toFixed(0)+'%':'—';
    document.getElementById('stat-valence').textContent   = mi.valence!=null?(mi.valence*100).toFixed(0)+'%':'—';
    document.getElementById('stat-acoustic').textContent  = mi.acousticness!=null?(mi.acousticness*100).toFixed(0)+'%':'—';
    const tb=r.timbre||{};
    document.getElementById('stat-bright').textContent    = ((tb.brightness||0)*100).toFixed(1)+'%';
    document.getElementById('stat-warm').textContent      = ((tb.warmth||0)*100).toFixed(1)+'%';

    // RMS envelope
    const step=Math.max(1,Math.floor((dyn.rms||[]).length/500));
    _line('chart-rms', (dyn.times||[]).filter((_,i)=>i%step===0), [
      {data:(dyn.rms_db||[]).filter((_,i)=>i%step===0), borderColor:'#00f5d4',fill:true,backgroundColor:'rgba(0,245,212,.08)'}
    ], {scales:{y:{display:true}}});

    // Short-term loudness
    if (loud.short_term?.length) {
      const dur=r.metadata?.duration||60;
      _line('chart-shortterm-loud', BV.linspace(0,dur,loud.short_term.length), [
        {data:loud.short_term, borderColor:'#7b61ff',fill:true,backgroundColor:'rgba(123,97,255,.08)'}
      ], {scales:{y:{display:true,ticks:{color:'#4a5568',font:{size:9},callback:v=>v+' LUFS'}}}});
    }

    // Genre fingerprint
    const gf=mi.genre_fingerprint||{};
    const gb=document.getElementById('genre-bars');
    if (gb) gb.innerHTML=Object.entries(gf).sort((a,b)=>b[1]-a[1]).map(([k,v])=>{
      const col=BV.genreColors[k]||'#4a5568';
      return `<div class="genre-bar-row">
        <div class="genre-name">${k.replace('_',' ').toUpperCase()}</div>
        <div class="genre-track"><div class="genre-fill" style="width:${(v*100).toFixed(1)}%;background:${col}88;border-left:3px solid ${col}"></div></div>
        <div class="genre-pct">${(v*100).toFixed(1)}%</div></div>`;
    }).join('');

    // Timbre radar
    if (tb.brightness!=null) {
      BV.createChart('chart-timbre-radar','radar',{
        labels:['Brightness','Warmth','Roughness','Odd/Even','Inharmonicity⁻¹','Noisiness⁻¹'],
        datasets:[{
          data:[tb.brightness*100, tb.warmth*100, Math.min(1,tb.roughness*10)*100,
                Math.min(100,Math.abs(tb.odd_even_ratio-1)*40+50),
                (1-Math.min(1,tb.inharmonicity*10))*100,
                (1-Math.min(1,(tb.noisiness||0)))*100],
          borderColor:'#00f5d4', backgroundColor:'rgba(0,245,212,.1)',
          borderWidth:1.5, pointBackgroundColor:'#00f5d4', pointRadius:3
        }]
      },{scales:{r:{grid:{color:'rgba(30,40,64,.8)'},pointLabels:{color:'#8b9ab0',font:{size:9}},ticks:{display:false},min:0,max:100}}});
    }
  }

  // ── STRUCTURE ─────────────────────────────────────────────────────────────
  function renderStructure(r) {
    const st=r.structure; if (!st) return;

    // SSM heatmap
    const ssmC=document.getElementById('canvas-ssm');
    if (ssmC&&st.self_similarity?.length) BV.drawHeatmap(ssmC, st.self_similarity, BV.color.inferno, false);

    // Novelty curve
    const nstep=Math.max(1,Math.floor((st.novelty||[]).length/400));
    if (st.novelty?.length) {
      _line('chart-novelty', (st.novelty_times||[]).filter((_,i)=>i%nstep===0), [
        {data:(st.novelty||[]).filter((_,i)=>i%nstep===0), borderColor:'#ffcc00',fill:true,backgroundColor:'rgba(255,204,0,.08)'}
      ]);
    }

    // Repetition curve (new in v4)
    const rc=st.repetition_curve||[];
    // (shown on novelty chart as secondary series if present)

    // Segment timeline
    const segtl=document.getElementById('seg-timeline');
    if (segtl&&st.segment_times?.length) {
      const dur=r.metadata?.duration||60;
      const bounds=[0,...st.segment_times,dur];
      const cols=['#00f5d4','#f700ff','#ff6b35','#7b61ff','#ffcc00','#39d353','#ff3b5c','#0ea5e9'];
      segtl.innerHTML=bounds.slice(0,-1).map((t,i)=>{
        const w=((bounds[i+1]-t)/dur*100).toFixed(2), col=cols[i%cols.length];
        return `<div class="seg-block" style="flex:${w};background:${col}22;border-right:2px solid ${col}88;color:${col}" title="${BV.fmt.time(t)} → ${BV.fmt.time(bounds[i+1])}">S${i+1}</div>`;
      }).join('');
    }

    // Stats — note: v4 uses segment_times_laplacian not segment_times_lp
    document.getElementById('stat-nsegs').textContent     = st.n_segments||'—';
    document.getElementById('stat-nsegs-lp').textContent  = st.segment_times_laplacian?.length||'—';
    document.getElementById('stat-nsegs-nov').textContent = st.segment_times_novelty?.length||'—';
    document.getElementById('seg-info').textContent =
      `${st.n_segments} segments · Laplacian: ${st.segment_times_laplacian?.length||0} · Novelty: ${st.segment_times_novelty?.length||0}`;

    // Energy + segment overlays
    const en=r.energy;
    if (en?.rms) {
      const step=Math.max(1,Math.floor(en.rms.length/500));
      _line('chart-energy-segs', (en.times||[]).filter((_,i)=>i%step===0), [
        {data:en.rms.filter((_,i)=>i%step===0), borderColor:'#00f5d4',fill:true,backgroundColor:'rgba(0,245,212,.08)'}
      ]);
    }
  }

  // ── Header stats ──────────────────────────────────────────────────────────
  function updateHeaderStats(r) {
    document.getElementById('hdr-bpm').textContent  = r.beats?.bpm_consensus?.toFixed(1)||'—';
    document.getElementById('hdr-key').textContent  = r.harmony?.key_full||'—';
    document.getElementById('hdr-dur').textContent  = BV.fmt.time(r.metadata?.duration);
    document.getElementById('hdr-lufs').textContent = r.dynamics?.loudness?.integrated_lufs?.toFixed(1)||'—';

    const rh=r.rhythm||{};
    const el_ts=document.getElementById('stat-timesig');
    if (el_ts) el_ts.textContent=rh.time_signature||'—';
    const el_bc=document.getElementById('stat-beatcount');
    if (el_bc) el_bc.textContent=r.beats?.beat_count||'—';
    const el_reg=document.getElementById('stat-regularity');
    if (el_reg) el_reg.textContent=(r.beats?.beat_regularity*100)?.toFixed(1)+'%'||'—';
    const el_sw=document.getElementById('stat-swing');
    if (el_sw) el_sw.textContent=rh.swing_factor?.toFixed(3)||'—';
    const el_syn=document.getElementById('stat-sync');
    if (el_syn) el_syn.textContent=rh.syncopation_index?.toFixed(3)||'—';
  }

  // ── Public API ────────────────────────────────────────────────────────────
  function renderAll(r) {
    _R = r;
    try { updateHeaderStats(r); } catch(e) { console.warn('headerStats',e); }
    try { renderWaveform(r);   } catch(e) { console.warn('waveform',e); }
    try { renderSpectrum(r);   } catch(e) { console.warn('spectrum',e); }
    try { renderBeats(r);      } catch(e) { console.warn('beats',e); }
    try { renderHarmony(r);    } catch(e) { console.warn('harmony',e); }
    try { renderMFCC(r);       } catch(e) { console.warn('mfcc',e); }
    try { renderPitch(r);      } catch(e) { console.warn('pitch',e); }
    try { renderDynamics(r);   } catch(e) { console.warn('dynamics',e); }
    try { renderStructure(r);  } catch(e) { console.warn('structure',e); }
  }

  return { renderAll, renderWaveform, renderSpectrum, renderBeats, renderHarmony,
           renderMFCC, renderPitch, renderDynamics, renderStructure, updateHeaderStats,
           getResult:()=>_R };
})();
console.log('[BV] visualizer2d.js loaded');
