/**
 * Beat Visualizer v3 — 3D Visualizer (Three.js r128)
 * 5 modes: spectrogram3d, beatSphere, waveRing, particles, tonnetz3d
 */
BV.viz3d = (() => {
  let renderer, scene, camera, animId;
  let currentMode = 'spectrogram3d', autoRotate = true;
  let mouse = { down: false, x: 0, y: 0, rotX: 0.3, rotY: 0.5 };
  let zoomDist = 5;
  let _objects = [];
  let _animFns = [];
  let _R = null;
  let _initialized = false;

  function init(result) {
    _R = result;
    const canvas = document.getElementById('canvas-3d');
    if (!canvas) return;
    if (!_initialized) {
      _initThree(canvas);
      _initControls(canvas);
      _initialized = true;
      _loop();
    }
    setMode(currentMode);
  }

  function _initThree(canvas) {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x080a0f, 1);
    _resize();
    window.addEventListener('resize', _resize);

    scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x080a0f, 8, 25);

    camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
    camera.position.set(0, 2, zoomDist);

    // Lights
    scene.add(new THREE.AmbientLight(0x0d1017, 0.8));
    const dir = new THREE.DirectionalLight(0x00f5d4, 0.6);
    dir.position.set(2, 4, 3); scene.add(dir);
    const dir2 = new THREE.DirectionalLight(0xf700ff, 0.3);
    dir2.position.set(-3, -2, -2); scene.add(dir2);

    // Grid
    const grid = new THREE.GridHelper(10, 20, 0x1e2840, 0x12171f);
    grid.position.y = -1.5; scene.add(grid);
  }

  function _resize() {
    const canvas = document.getElementById('canvas-3d');
    if (!canvas || !renderer || !camera) return;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    renderer.setSize(W, H, false);
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
  }

  function _initControls(canvas) {
    canvas.addEventListener('mousedown', e => { mouse.down = true; mouse.x = e.clientX; mouse.y = e.clientY; });
    canvas.addEventListener('mouseup', () => { mouse.down = false; });
    canvas.addEventListener('mousemove', e => {
      if (!mouse.down) return;
      mouse.rotY += (e.clientX - mouse.x) * 0.008;
      mouse.rotX += (e.clientY - mouse.y) * 0.008;
      mouse.rotX = Math.max(-Math.PI/2, Math.min(Math.PI/2, mouse.rotX));
      mouse.x = e.clientX; mouse.y = e.clientY;
    });
    canvas.addEventListener('wheel', e => { zoomDist = Math.max(1, Math.min(20, zoomDist + e.deltaY * 0.01)); });
    canvas.addEventListener('touchstart', e => { mouse.down=true; mouse.x=e.touches[0].clientX; mouse.y=e.touches[0].clientY; });
    canvas.addEventListener('touchend', () => { mouse.down=false; });
    canvas.addEventListener('touchmove', e => {
      if (!mouse.down) return;
      mouse.rotY += (e.touches[0].clientX - mouse.x) * 0.008;
      mouse.rotX += (e.touches[0].clientY - mouse.y) * 0.008;
      mouse.x=e.touches[0].clientX; mouse.y=e.touches[0].clientY;
    });

    document.querySelectorAll('[data-mode3d]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('[data-mode3d]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        setMode(btn.dataset.mode3d);
      });
    });
    document.getElementById('btn-autorotate')?.addEventListener('click', function() {
      autoRotate = !autoRotate;
      this.textContent = 'AUTO-ROTATE ' + (autoRotate ? '✓' : '✗');
    });
  }

  function _clearScene() {
    _objects.forEach(o => { scene.remove(o); if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    _objects = []; _animFns = [];
  }

  function setMode(mode) {
    currentMode = mode;
    _clearScene();
    if (!_R) return;
    const builders = {
      spectrogram3d, beatSphere, waveRing, particles, tonnetz3d
    };
    (builders[mode] || spectrogram3d)();
  }

  // ── Spectrogram 3D ────────────────────────────────────────────────────────
  function spectrogram3d() {
    const spec = _R.spectral?.magnitude_db;
    if (!spec?.length) return;
    const rows = Math.min(spec.length, 64), cols = Math.min(spec[0]?.length || 0, 120);
    if (!rows || !cols) return;
    const geo = new THREE.PlaneGeometry(8, 5, cols - 1, rows - 1);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position;
    const col = new Float32Array(pos.count * 3);
    let mn = Infinity, mx = -Infinity;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const v = spec[Math.floor(r * spec.length / rows)]?.[Math.floor(c * spec[0].length / cols)] || -80;
      if (v < mn) mn = v; if (v > mx) mx = v;
    }
    const range = mx - mn || 1;
    for (let i = 0; i < pos.count; i++) {
      const r = Math.floor(i / cols), c = i % cols;
      const v = spec[Math.min(r, spec.length-1)]?.[Math.min(c, (spec[0]?.length||1)-1)] || mn;
      const t = (v - mn) / range;
      pos.setY(i, t * 2.5 - 1.5);
      const [red, g, b] = BV.color.plasma(t);
      col[i*3] = red/255; col[i*3+1] = g/255; col[i*3+2] = b/255;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    pos.needsUpdate = true; geo.computeVertexNormals();
    const mat = new THREE.MeshPhongMaterial({ vertexColors: true, wireframe: false, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh); _objects.push(mesh);
    // Wireframe overlay
    const wfMat = new THREE.MeshBasicMaterial({ color: 0x1e2840, wireframe: true, transparent: true, opacity: 0.2 });
    const wfMesh = new THREE.Mesh(geo.clone(), wfMat);
    scene.add(wfMesh); _objects.push(wfMesh);
    camera.position.set(0, 4, 7);
  }

  // ── Beat Sphere ───────────────────────────────────────────────────────────
  function beatSphere() {
    const ch = _R.chroma?.chroma_mean || new Array(12).fill(0.5);
    const bpm = _R.beats?.bpm_consensus || 120;
    const beatPeriod = 60 / bpm * 1000;

    // Central sphere
    const cGeo = new THREE.SphereGeometry(0.6, 32, 32);
    const cMat = new THREE.MeshPhongMaterial({ color: 0x00f5d4, emissive: 0x003322, shininess: 80, transparent: true, opacity: 0.9 });
    const cSphere = new THREE.Mesh(cGeo, cMat);
    scene.add(cSphere); _objects.push(cSphere);

    // Orbit spheres (chroma-weighted, 12 pitch classes)
    const orbitSpheres = [], lines = [];
    const pitchColors = BV.pitchClasses.map((_, i) => new THREE.Color().setHSL(i / 12, 0.9, 0.6));
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2;
      const r = 1.8, y = (ch[i] - 0.5) * 1.2;
      const geo = new THREE.SphereGeometry(0.08 + ch[i] * 0.12, 16, 16);
      const mat = new THREE.MeshPhongMaterial({ color: pitchColors[i], emissive: pitchColors[i], emissiveIntensity: 0.4 });
      const sphere = new THREE.Mesh(geo, mat);
      sphere.position.set(Math.cos(angle) * r, y, Math.sin(angle) * r);
      sphere.userData = { baseAngle: angle, orbitR: r, baseY: y };
      scene.add(sphere); _objects.push(sphere); orbitSpheres.push(sphere);
      // Line to center
      const pts = [new THREE.Vector3(), sphere.position.clone()];
      const lGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const lMat = new THREE.LineBasicMaterial({ color: pitchColors[i], transparent: true, opacity: 0.3 });
      const line = new THREE.Line(lGeo, lMat);
      scene.add(line); _objects.push(line); lines.push({ line, sphere });
    }

    // Beat pulse rings
    const beatToruses = Array.from({ length: 5 }, (_, i) => {
      const geo = new THREE.TorusGeometry(0.7 + i * 0.15, 0.01, 8, 64);
      const mat = new THREE.MeshBasicMaterial({ color: 0x00f5d4, transparent: true, opacity: 0 });
      const torus = new THREE.Mesh(geo, mat);
      scene.add(torus); _objects.push(torus);
      return { torus, mat };
    });

    let lastBeat = 0;
    _animFns.push((t) => {
      const now = performance.now();
      // Orbit animation
      orbitSpheres.forEach((s, i) => {
        const a = s.userData.baseAngle + t * 0.3;
        s.position.set(Math.cos(a) * s.userData.orbitR, s.userData.baseY + Math.sin(t*1.5+i)*0.1, Math.sin(a) * s.userData.orbitR);
      });
      // Update lines
      lines.forEach(({ line, sphere }) => {
        const pts = line.geometry.attributes.position;
        pts.setXYZ(1, sphere.position.x, sphere.position.y, sphere.position.z);
        pts.needsUpdate = true;
      });
      // Beat pulse
      if (now - lastBeat > beatPeriod) {
        lastBeat = now;
        beatToruses.forEach(({ torus, mat }, i) => {
          torus.userData.pulsing = true; torus.userData.pulseT = 0;
          torus.userData.delay = i * 0.08;
        });
        cSphere.scale.setScalar(1.15);
      }
      cSphere.scale.lerp(new THREE.Vector3(1,1,1), 0.1);
      beatToruses.forEach(({ torus, mat }) => {
        if (torus.userData.pulsing) {
          const pt = (torus.userData.pulseT || 0) + 0.03;
          torus.userData.pulseT = pt;
          const scale = 1 + pt * 2;
          torus.scale.setScalar(scale);
          mat.opacity = Math.max(0, 0.7 - pt);
          if (pt > 1) { torus.userData.pulsing = false; torus.userData.pulseT = 0; mat.opacity = 0; torus.scale.setScalar(1); }
        }
      });
    });
    camera.position.set(0, 2.5, 5);
  }

  // ── Wave Ring ─────────────────────────────────────────────────────────────
  function waveRing() {
    const samples = _R.waveform?.samples || [];
    const n = Math.min(samples.length, 512);
    const ds = BV.downsample(samples, n);
    const r0 = 2.0, layers = 4;

    for (let layer = 0; layer < layers; layer++) {
      const pts = [], cols = [];
      for (let i = 0; i <= n; i++) {
        const angle = (i / n) * Math.PI * 2;
        const v = ds[i % n] || 0;
        const r = r0 + v * (0.6 - layer * 0.1) + layer * 0.3;
        pts.push(new THREE.Vector3(Math.cos(angle) * r, v * 0.5 + layer * 0.2 - 0.3, Math.sin(angle) * r));
        const t = (v + 1) / 2;
        const [red, g, b] = BV.color.plasma(t);
        cols.push(red/255, g/255, b/255);
      }
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
      const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 1 - layer * 0.2 });
      const line = new THREE.Line(geo, mat);
      scene.add(line); _objects.push(line);
    }

    // Onset ring (inner)
    const onsets = _R.onsets?.onsets_energy || [];
    const dur = _R.metadata?.duration || 60;
    if (onsets.length) {
      const pts = onsets.map(t => {
        const angle = (t / dur) * Math.PI * 2;
        return new THREE.Vector3(Math.cos(angle) * 1.0, 0, Math.sin(angle) * 1.0);
      });
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.PointsMaterial({ color: 0xf700ff, size: 0.06 });
      const points = new THREE.Points(geo, mat);
      scene.add(points); _objects.push(points);
    }

    let frame = 0;
    _animFns.push((t) => {
      _objects.forEach((o, i) => { if (o.isLine) o.rotation.y = t * (0.1 + i * 0.03); });
    });
    camera.position.set(0, 3, 6);
  }

  // ── Particles ─────────────────────────────────────────────────────────────
  function particles() {
    const spec = _R.spectral?.magnitude_db;
    if (!spec?.length) return;
    const rows = Math.min(spec.length, 60), cols = Math.min(spec[0]?.length || 0, 100);
    const count = rows * cols;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    let mn = Infinity, mx = -Infinity;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const v = spec[Math.floor(r * spec.length / rows)]?.[Math.floor(c * spec[0].length / cols)] || -80;
      if (v < mn) mn = v; if (v > mx) mx = v;
    }
    const range = mx - mn || 1;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const v = spec[Math.floor(r * spec.length / rows)]?.[Math.floor(c * spec[0].length / cols)] || mn;
        const t = (v - mn) / range;
        const x = (c / cols - 0.5) * 8, z = (r / rows - 0.5) * 6, y = t * 3 - 1;
        positions[idx*3] = x + (Math.random()-.5)*.1;
        positions[idx*3+1] = y;
        positions[idx*3+2] = z + (Math.random()-.5)*.1;
        const [red, g, b] = BV.color.inferno(t);
        colors[idx*3] = red/255; colors[idx*3+1] = g/255; colors[idx*3+2] = b/255;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({ size: 0.05, vertexColors: true, transparent: true, opacity: 0.85 });
    const pts = new THREE.Points(geo, mat);
    scene.add(pts); _objects.push(pts);
    _animFns.push((t) => { pts.rotation.y = t * 0.08; });
    camera.position.set(0, 3, 8);
  }

  // ── Tonnetz 3D ────────────────────────────────────────────────────────────
  function tonnetz3d() {
    const ch = _R.chroma?.chroma_mean || new Array(12).fill(0.5);
    const torusR = 1.5, tubeR = 0.6;
    // Torus surface
    const geo = new THREE.TorusGeometry(torusR, tubeR, 32, 64);
    const mat = new THREE.MeshPhongMaterial({ color: 0x1e2840, transparent: true, opacity: 0.25, side: THREE.DoubleSide });
    const torus = new THREE.Mesh(geo, mat);
    scene.add(torus); _objects.push(torus);

    // 12 pitch nodes on torus surface
    const nodes = [], nodePositions = [];
    for (let i = 0; i < 12; i++) {
      // Map pitch class to torus coordinates
      // 5th axis → tube angle, 3rd axis → major ring angle
      const fifthsPos = (i * 7) % 12; // circle of fifths
      const phi = (fifthsPos / 12) * Math.PI * 2;     // ring angle
      const theta = ((i * 4) % 12 / 12) * Math.PI * 2; // tube angle (3rds)
      const x = (torusR + tubeR * Math.cos(theta)) * Math.cos(phi);
      const y = tubeR * Math.sin(theta);
      const z = (torusR + tubeR * Math.cos(theta)) * Math.sin(phi);
      nodePositions.push(new THREE.Vector3(x, y, z));

      const strength = ch[i] || 0.1;
      const r = 0.04 + strength * 0.1;
      const nGeo = new THREE.SphereGeometry(r, 16, 16);
      const hue = i / 12;
      const nMat = new THREE.MeshPhongMaterial({ color: new THREE.Color().setHSL(hue, 0.9, 0.55), emissive: new THREE.Color().setHSL(hue, 0.9, 0.2) });
      const node = new THREE.Mesh(nGeo, nMat);
      node.position.copy(nodePositions[i]);
      scene.add(node); _objects.push(node); nodes.push(node);

      // Sprite label
      const canvas = document.createElement('canvas'); canvas.width = 64; canvas.height = 32;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#00f5d4'; ctx.font = 'bold 18px monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(BV.pitchClasses[i], 32, 16);
      const tex = new THREE.CanvasTexture(canvas);
      const sprGeo = new THREE.PlaneGeometry(0.3, 0.15);
      const sprMat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false });
      const sprite = new THREE.Mesh(sprGeo, sprMat);
      sprite.position.copy(nodePositions[i].clone().multiplyScalar(1.15));
      scene.add(sprite); _objects.push(sprite);
    }

    // 5th-interval edges (connect notes a perfect 5th apart)
    for (let i = 0; i < 12; i++) {
      const j = (i + 7) % 12;
      const pts = [nodePositions[i], nodePositions[j]];
      const lGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const lMat = new THREE.LineBasicMaterial({ color: 0x00f5d4, transparent: true, opacity: 0.35 });
      const line = new THREE.Line(lGeo, lMat);
      scene.add(line); _objects.push(line);
    }
    // Major 3rd edges (4 semitones)
    for (let i = 0; i < 12; i++) {
      const j = (i + 4) % 12;
      const pts = [nodePositions[i], nodePositions[j]];
      const lGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const lMat = new THREE.LineBasicMaterial({ color: 0xf700ff, transparent: true, opacity: 0.2 });
      const line = new THREE.Line(lGeo, lMat);
      scene.add(line); _objects.push(line);
    }

    _animFns.push((t) => { torus.rotation.y = t * 0.12; nodes.forEach((n, i) => n.rotation.y = t + i * 0.5); });
    camera.position.set(0, 2, 5.5);
  }

  // ── Animation loop ────────────────────────────────────────────────────────
  function _loop() {
    animId = requestAnimationFrame(_loop);
    if (!renderer || !scene || !camera) return;
    const t = performance.now() * 0.001;

    // Camera orbit
    if (autoRotate) mouse.rotY += 0.003;
    camera.position.x = Math.sin(mouse.rotY) * Math.cos(mouse.rotX) * zoomDist;
    camera.position.y = Math.sin(mouse.rotX) * zoomDist;
    camera.position.z = Math.cos(mouse.rotY) * Math.cos(mouse.rotX) * zoomDist;
    camera.lookAt(0, 0, 0);

    _animFns.forEach(fn => fn(t));
    renderer.render(scene, camera);
  }

  return { init, setMode, get initialized() { return _initialized; } };
})();
console.log('[BV] visualizer3d.js loaded');
