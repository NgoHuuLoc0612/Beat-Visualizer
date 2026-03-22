/**
 * Beat Visualizer v3 — Socket.IO Client
 *
 * NOTE: analysis_complete from server NO LONGER carries the result payload.
 * Result is fetched via HTTP GET /api/status/:id by upload.js poll loop.
 */
BV.socket = (() => {
  const sock = io({
    transports: ['websocket', 'polling'],
    reconnectionAttempts: 20,
    reconnectionDelay: 1000,
    timeout: 30000,
  });

  sock.on('connect',       () => console.log('[BV] socket connected', sock.id));
  sock.on('disconnect',    () => console.warn('[BV] socket disconnected'));
  sock.on('connect_error', e  => console.warn('[BV] socket connect error:', e.message));

  // Progress updates → upload progress bar (fast path alongside polling)
  sock.on('analysis_progress', d => {
    if (BV.upload && BV.upload.onProgress) BV.upload.onProgress(d);
  });

  // Complete signal — NO result payload, just job_id
  // upload.js poll loop delivers the result via HTTP
  sock.on('analysis_complete', d => {
    if (BV.upload && BV.upload.onSocketComplete) BV.upload.onSocketComplete(d);
  });

  // Error signal
  sock.on('analysis_error', d => {
    if (BV.app && BV.app.onError) BV.app.onError(d);
  });

  // Realtime mic features from server
  sock.on('realtime_features', d => {
    if (BV.realtime && BV.realtime.onServerFeatures) BV.realtime.onServerFeatures(d);
  });

  return {
    emit:      (ev, data) => sock.emit(ev, data),
    connected: ()         => sock.connected,
    raw:       sock,
  };
})();

console.log('[BV] socket.js loaded');
