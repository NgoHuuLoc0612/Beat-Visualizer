/**
 * Beat Visualizer v3 — Shared Utilities
 */
window.BV = window.BV || {};

BV.fmt = {
  time(s) { if (s==null||isNaN(s)) return '—'; const m=Math.floor(s/60),sec=Math.floor(s%60); return `${m}:${String(sec).padStart(2,'0')}`; },
  hz(f)   { return f>=1000?(f/1000).toFixed(1)+'kHz':Math.round(f)+'Hz'; },
  db(d)   { return (d>=0?'+':'')+d.toFixed(1)+'dB'; },
  pct(v)  { return (v*100).toFixed(1)+'%'; },
  bpm(v)  { return (typeof v==='number'?v.toFixed(1):'—')+' BPM'; },
  n2(v)   { return typeof v==='number'?v.toFixed(2):'—'; },
  n4(v)   { return typeof v==='number'?v.toFixed(4):'—'; },
  sci(v)  { return typeof v==='number'?v.toExponential(2):'—'; },
};

BV.color = {
  _lut(stops) {
    return (t) => {
      t = Math.max(0,Math.min(1,t));
      for (let i=0;i<stops.length-1;i++) {
        const [t0,r0,g0,b0]=stops[i],[t1,r1,g1,b1]=stops[i+1];
        if (t>=t0&&t<=t1) { const f=(t-t0)/(t1-t0); return [r0+f*(r1-r0),g0+f*(g1-g0),b0+f*(b1-b0)]; }
      }
      return [0,0,0];
    };
  },
  get plasma() { return this._lut([[0,13,8,135],[.25,126,3,168],[.5,204,71,120],[.75,248,149,64],[1,240,249,33]]); },
  get inferno() { return this._lut([[0,0,0,4],[.25,87,16,110],[.5,188,55,84],[.75,249,142,9],[1,252,255,164]]); },
  get viridis() { return this._lut([[0,68,1,84],[.25,58,82,139],[.5,32,144,141],[.75,94,201,98],[1,253,231,37]]); },
  get magma()   { return this._lut([[0,0,0,4],[.25,81,18,124],[.5,183,55,121],[.75,251,136,97],[1,252,253,191]]); },
  get cyan()    { return this._lut([[0,8,10,15],[.3,0,50,80],[.6,0,150,130],[.85,0,245,212],[1,255,255,255]]); },
  get hot()     { return this._lut([[0,8,10,15],[.33,120,0,60],[.66,247,0,255],[1,255,200,255]]); },
};

BV.normalize = (arr,lo=null,hi=null) => {
  const mn=lo!==null?lo:Math.min(...arr.flat(2)), mx=hi!==null?hi:Math.max(...arr.flat(2));
  const r=mx-mn||1;
  if (Array.isArray(arr[0])) return arr.map(row=>row.map(v=>(v-mn)/r));
  return arr.map(v=>(v-mn)/r);
};
BV.downsample = (arr,target) => {
  if (arr.length<=target) return arr;
  const step=arr.length/target;
  return Array.from({length:target},(_,i)=>arr[Math.floor(i*step)]);
};
BV.linspace = (a,b,n) => Array.from({length:n},(_,i)=>a+(i/(n-1))*(b-a));

BV.toast = (msg,type='info',ms=3500) => {
  const c=document.getElementById('toast-container'); if (!c) return;
  const t=document.createElement('div'); t.className='toast'+(type!=='info'?` ${type}`:''); t.textContent=msg; c.appendChild(t);
  setTimeout(()=>{t.style.opacity='0';t.style.transform='translateX(20px)';t.style.transition='.3s';},ms-300);
  setTimeout(()=>t.remove(),ms);
};

BV.drawHeatmap = (canvas,matrix,colormap,flipY=true) => {
  if (!matrix||!matrix.length||!matrix[0].length) return;
  const ctx=canvas.getContext('2d'), rows=matrix.length, cols=matrix[0].length;
  canvas.width=cols; canvas.height=rows;
  const img=ctx.createImageData(cols,rows); const cmap=colormap;
  let mn=Infinity,mx=-Infinity;
  for (const row of matrix) for (const v of row) { if(v<mn)mn=v; if(v>mx)mx=v; }
  const range=mx-mn||1;
  for (let r=0;r<rows;r++) {
    const dr=flipY?(rows-1-r):r;
    for (let c=0;c<cols;c++) {
      const t=(matrix[dr][c]-mn)/range; const [red,g,b]=cmap(t); const idx=(r*cols+c)*4;
      img.data[idx]=red; img.data[idx+1]=g; img.data[idx+2]=b; img.data[idx+3]=255;
    }
  }
  ctx.putImageData(img,0,0);
};

const _ch={};
const _DEF={responsive:true,maintainAspectRatio:false,animation:{duration:0},
  plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>` ${BV.fmt.n2(ctx.parsed.y)}`}}},
  scales:{x:{display:false},y:{grid:{color:'rgba(30,40,64,.6)'},ticks:{color:'#4a5568',font:{family:"'Space Mono'",size:9}}}}};

BV.createChart=(id,type,data,opts={})=>{
  const el=document.getElementById(id); if(!el) return null;
  if(_ch[id]){_ch[id].destroy();delete _ch[id];}
  const cfg={type,data,options:BV._merge(JSON.parse(JSON.stringify(_DEF)),opts)};
  return (_ch[id]=new Chart(el.getContext('2d'),cfg));
};
BV.updateChart=(id,data)=>{ const c=_ch[id]; if(!c) return; c.data=data; c.update('none'); };
BV.destroyChart=(id)=>{ if(_ch[id]){_ch[id].destroy();delete _ch[id];} };
BV._merge=(t,s)=>{ for(const k in s){ if(s[k]&&typeof s[k]==='object'&&!Array.isArray(s[k])){if(!t[k])t[k]={};BV._merge(t[k],s[k]);}else{t[k]=s[k];}} return t; };

BV.pitchClasses=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
BV.chordColors={maj:'#00f5d4',min:'#7b61ff',dim:'#ff3b5c',aug:'#ffcc00',maj7:'#00c9a7',min7:'#6048cc',dom7:'#ff9500',sus2:'#39d353',sus4:'#0ea5e9',N:'#2a3650'};
BV.chordColor=c=>{const t=c.split(':')[1]||'N';return BV.chordColors[t]||'#2a3650';};
BV.genreColors={electronic:'#00f5d4',acoustic:'#39d353',hip_hop:'#f700ff',classical:'#7b61ff',rock:'#ff3b5c',jazz:'#ffcc00',edm:'#ff6b35'};
console.log('[BV] utils loaded');
