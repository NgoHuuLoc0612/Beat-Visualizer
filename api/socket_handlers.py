"""WebSocket handlers v4 — analysis progress, realtime chunk, waveform slice, region analysis"""
import logging, numpy as np
log = logging.getLogger(__name__)


def register_socket_handlers(socketio):

    @socketio.on("connect")
    def on_connect(): log.info("Client connected")

    @socketio.on("disconnect")
    def on_disconnect(): log.info("Client disconnected")

    @socketio.on("ping_bv")
    def on_ping(data):
        from flask_socketio import emit
        emit("pong_bv", {"ts": data.get("ts"), "server_time": __import__("time").time()})

    # ── Realtime audio chunk analysis ─────────────────────────────────────────
    @socketio.on("realtime_chunk")
    def on_realtime_chunk(data):
        from flask_socketio import emit
        from core.analyzer import AudioAnalyzer
        try:
            samples = np.array(data["samples"], dtype=np.float32)
            sr      = int(data.get("sample_rate", 44100))
            cid     = data.get("chunk_id", 0)
            if len(samples) < 256: return
            feats   = AudioAnalyzer().analyze_realtime_chunk(samples, sr)
            feats["chunk_id"] = cid
            emit("realtime_features", feats)
        except Exception as e:
            log.error("realtime_chunk: %s", e)
            from flask_socketio import emit as _emit
            _emit("realtime_error", {"error": str(e)})

    # ── Waveform slice streaming (large files) ────────────────────────────────
    @socketio.on("request_waveform_slice")
    def on_waveform_slice(data):
        from flask_socketio import emit
        from api.routes import _jobs, _jobs_lock
        jid   = data.get("job_id"); start = int(data.get("start",0)); length = int(data.get("length",2000))
        with _jobs_lock: job = _jobs.get(jid)
        if not job or not job.get("result"):
            emit("error", {"message": "Job not found or incomplete"}); return
        wf = job["result"].get("waveform", {}); samples = wf.get("samples",[]); times = wf.get("times",[])
        end = min(start+length, len(samples))
        emit("waveform_slice", {"job_id":jid,"start":start,"end":end,
                                 "samples":samples[start:end],"times":times[start:end],"total":len(samples)})

    # ── Region analysis (user-selected time window) ───────────────────────────
    @socketio.on("analyze_region")
    def on_analyze_region(data):
        """Analyze a time region within an already-uploaded file."""
        from flask_socketio import emit
        from api.routes import _jobs, _jobs_lock
        from core.analyzer import AudioAnalyzer
        jid   = data.get("job_id")
        t_start= float(data.get("start", 0))
        t_end  = float(data.get("end",   10))
        with _jobs_lock: job = _jobs.get(jid)
        if not job:
            emit("region_error", {"error":"Job not found"}); return
        try:
            az = AudioAnalyzer()
            y, sr = az.load(job["filepath"], offset=t_start, dur=t_end-t_start)
            # Quick mini-analysis on region
            import librosa
            beats  = az._beats(y, sr, librosa.effects.hpss(y)[1])
            onsets = az._onsets(y, sr)
            spectral=az._spectral(y, sr)
            emit("region_result", {"job_id":jid,"start":t_start,"end":t_end,
                                    "beats":beats,"onsets":onsets,"spectral":spectral})
        except Exception as e:
            log.error("analyze_region: %s", e)
            emit("region_error", {"error": str(e)})

    # ── Spectrogram streaming (send in vertical strips) ───────────────────────
    @socketio.on("stream_spectrogram")
    def on_stream_spectrogram(data):
        from flask_socketio import emit
        from api.routes import _jobs, _jobs_lock
        jid  = data.get("job_id"); block= data.get("block","mel")
        with _jobs_lock: job = _jobs.get(jid)
        if not job or not job.get("result"):
            emit("error", {"message":"Job not ready"}); return
        r = job["result"]
        mat = r.get("mel",{}).get("mel_db") if block=="mel" else \
              r.get("stft",{}).get("main",{}).get("magnitude_db")
        if not mat: emit("error",{"message":f"Block {block} not found"}); return
        STRIP=50
        for i in range(0, len(mat[0]), STRIP):
            strip = [row[i:i+STRIP] for row in mat]
            emit("spectrogram_strip", {"job_id":jid,"block":block,"col_start":i,"strip":strip})
