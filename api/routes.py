"""API Routes v4 — upload, analyze, export, compare, batch"""
import os, sys, json, csv, io, math, time, threading, uuid, logging
from pathlib import Path
from flask import Blueprint, request, jsonify, render_template, send_file, current_app, Response
from werkzeug.utils import secure_filename

log = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)
_jobs = {}
_jobs_lock = threading.Lock()


# ── Windows CP1252 safe logging ───────────────────────────────────────────────
# Background threads on Windows don't inherit the patched stdout, so we patch
# logging at the handler level instead.
def _patch_logging_encoding():
    """Replace any StreamHandlers that use a non-UTF-8 stream with UTF-8 wrappers."""
    root = logging.getLogger()
    for handler in root.handlers:
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'buffer'):
            try:
                enc = getattr(handler.stream, 'encoding', 'utf-8') or 'utf-8'
                if enc.lower().replace('-', '') not in ('utf8', 'utf8bom'):
                    import io as _io
                    handler.stream = _io.TextIOWrapper(
                        handler.stream.buffer, encoding='utf-8', errors='replace')
            except Exception:
                pass

_patch_logging_encoding()


# ── JSON helpers ──────────────────────────────────────────────────────────────
import numpy as _np

def _clean_float(v):
    """Convert any float-like to JSON-safe value — maps Inf/NaN to null (None)."""
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return None


def _clean_value(obj):
    """Recursively sanitize a Python object for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_value(x) for x in obj]
    if isinstance(obj, _np.ndarray):
        return _clean_value(obj.tolist())
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return _clean_float(float(obj))
    if isinstance(obj, float):
        return _clean_float(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    return obj


def _json_safe(obj):
    """Fallback encoder for json.dumps(default=...) — handles remaining numpy types."""
    if isinstance(obj, _np.integer):  return int(obj)
    if isinstance(obj, _np.floating): return _clean_float(float(obj))
    if isinstance(obj, _np.ndarray):  return obj.tolist()
    if isinstance(obj, set):          return sorted(obj)
    if isinstance(obj, bytes):        return obj.decode('utf-8', errors='replace')
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_jsonify(data):
    """JSON response that handles Inf/NaN/numpy/sets — fully standards-compliant."""
    # Deep-clean all floats before serialization so Infinity never appears
    cleaned = _clean_value(data)
    return Response(
        json.dumps(cleaned),          # no custom default needed — all types cleaned
        mimetype='application/json'
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def register_main_routes(app):
    import time as _time
    # Version changes every server restart → forces browser to reload all JS/CSS
    _static_version = str(int(_time.time()))

    @app.route("/")
    def index():
        from flask import make_response
        resp = make_response(render_template("index.html", static_version=_static_version))
        # Never cache the HTML page itself — always get fresh JS version tags
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp
    @app.errorhandler(413)
    def too_large(e): return jsonify({"error": "File too large (max 2 GB)"}), 413
    @app.errorhandler(500)
    def server_error(e): return jsonify({"error": str(e)}), 500


def _all_exts(cfg):
    return {e for s in cfg["ALLOWED_EXTENSIONS"].values() for e in s}

def _allowed(filename, cfg):
    if "." not in filename: return False
    ext = filename.rsplit(".", 1)[1].lower()
    compound = ".".join(filename.lower().split(".")[-2:])
    return ext in _all_exts(cfg) or compound in _all_exts(cfg)

def _ftype(filename, cfg):
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    for cat, exts in cfg["ALLOWED_EXTENSIONS"].items():
        if ext in exts: return cat
    return "unknown"


# ── Upload ────────────────────────────────────────────────────────────────────
@api_bp.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    cfg = current_app.config
    if not _allowed(f.filename, cfg):
        return jsonify({"error": f"Unsupported format. Allowed: {sorted(_all_exts(cfg))}"}), 400

    fname = secure_filename(f.filename)
    jid   = str(uuid.uuid4())
    path  = os.path.join(cfg["UPLOAD_FOLDER"], f"{jid}_{fname}")
    f.save(path)
    fsize = os.path.getsize(path)

    with _jobs_lock:
        _jobs[jid] = {
            "id": jid, "filename": fname, "filepath": path,
            "file_type": _ftype(fname, cfg), "file_size": fsize,
            "status": "uploaded", "progress": 0,
            "message": "Ready for analysis", "created_at": time.time(),
            "result": None, "error": None,
        }

    log.info("Upload: %s (%d B) jid=%s", fname, fsize, jid)
    return jsonify({"job_id": jid, "filename": fname,
                    "file_type": _ftype(fname, cfg), "file_size": fsize, "status": "uploaded"})


# ── Analyze ───────────────────────────────────────────────────────────────────
@api_bp.route("/analyze/<jid>", methods=["POST"])
def analyze(jid):
    with _jobs_lock: job = _jobs.get(jid)
    if not job: return jsonify({"error": "Job not found"}), 404
    if job["status"] in ("analyzing", "complete"):
        return jsonify({"status": job["status"], "job_id": jid})
    opts = request.get_json(silent=True) or {}
    thread = threading.Thread(
        target=_run,
        args=(jid, job["filepath"], current_app._get_current_object(), opts),
        daemon=True,
    )
    thread.start()
    with _jobs_lock:
        _jobs[jid]["status"] = "analyzing"
        _jobs[jid]["progress"] = 1
    return jsonify({"status": "analyzing", "job_id": jid})


def _run(jid, path, app, opts=None):
    # Patch logging encoding for this thread (Windows CP1252 fix)
    _patch_logging_encoding()

    from core.analyzer import AudioAnalyzer
    from app import socketio

    def cb(pct, msg):
        with _jobs_lock:
            if jid in _jobs:
                _jobs[jid]["progress"] = pct
                _jobs[jid]["message"]  = msg
        try:
            socketio.emit("analysis_progress", {"job_id": jid, "progress": pct, "message": msg})
        except Exception:
            pass

    try:
        analyzer = AudioAnalyzer(cache_dir=app.config["ANALYSIS_CACHE_DIR"])
        result   = analyzer.analyze_full(path, callback=cb)
        with _jobs_lock:
            _jobs[jid].update(status="complete", progress=100,
                              message="Analysis complete", result=result)
        # Emit via socket — result is already plain Python (analyzer guarantees this)
        try:
            socketio.emit("analysis_complete", {"job_id": jid})  # result fetched via HTTP GET /api/status
        except Exception as emit_err:
            log.warning("Socket emit failed (result too large?): %s", emit_err)
    except Exception as e:
        log.exception("Analysis failed jid=%s", jid)
        with _jobs_lock:
            _jobs[jid].update(status="error", error=str(e))
        try:
            socketio.emit("analysis_error", {"job_id": jid, "error": str(e)})
        except Exception:
            pass


# ── Status — lightweight, never includes result payload ──────────────────────
@api_bp.route("/status/<jid>")
def status(jid):
    with _jobs_lock: job = _jobs.get(jid)
    if not job: return jsonify({"error": "Not found"}), 404
    r = {k: job[k] for k in ("id", "filename", "file_size", "file_type",
                               "status", "progress", "message")}
    if job["status"] == "error":
        r["error"] = job["error"]
    # NOTE: result is NOT included here — fetch it via GET /api/result/<jid>
    # This keeps /api/status small (<1 KB) so polling is always fast.
    return jsonify(r)


@api_bp.route("/result/<jid>")
def result(jid):
    with _jobs_lock: job = _jobs.get(jid)
    if not job: return jsonify({"error": "Not found"}), 404
    if job["status"] != "complete":
        return jsonify({"error": "Not complete", "status": job["status"]}), 409
    return _safe_jsonify({"result": job["result"]})


@api_bp.route("/result/<jid>/block/<block>")
def result_block(jid, block):
    """Fetch a single analysis block (e.g. beats, mfcc, chroma)."""
    with _jobs_lock: job = _jobs.get(jid)
    if not job or job["status"] != "complete":
        return jsonify({"error": "Not ready"}), 404
    data = job["result"].get(block)
    if data is None:
        return jsonify({"error": f"Block '{block}' not found"}), 404
    return _safe_jsonify({block: data})


# ── Export ────────────────────────────────────────────────────────────────────
@api_bp.route("/export/<jid>/<fmt>")
def export(jid, fmt):
    with _jobs_lock: job = _jobs.get(jid)
    if not job or job["status"] != "complete":
        return jsonify({"error": "No result"}), 404
    r = job["result"]

    if fmt == "json":
        return Response(
            json.dumps(_clean_value(r), indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment; filename=bv_{jid[:8]}.json"},
        )

    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Type", "Time_s", "Value", "Extra"])
        for t in r.get("beats", {}).get("beat_times", []):
            w.writerow(["beat", round(t, 4), 1, ""])
        for t in r.get("onsets", {}).get("onsets_energy", []):
            w.writerow(["onset_energy", round(t, 4), 1, ""])
        for t in r.get("onsets", {}).get("onsets_mel", []):
            w.writerow(["onset_mel", round(t, 4), 1, ""])
        for t, rms in zip(r.get("energy", {}).get("times", []),
                          r.get("energy", {}).get("rms", [])):
            w.writerow(["rms", round(t, 4), round(float(rms), 6), ""])
        for c in r.get("chords", {}).get("chords", []):
            w.writerow(["chord", c.get("time", 0), c.get("score", 0), c.get("chord", "")])
        for t in r.get("structure", {}).get("segment_times", []):
            w.writerow(["segment_boundary", round(t, 4), 1, ""])
        buf.seek(0)
        return Response(
            buf.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=bv_{jid[:8]}.csv"},
        )

    if fmt == "midi":
        try:
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(initial_tempo=float(r["beats"]["bpm_consensus"]))
            inst = pretty_midi.Instrument(program=0, name="Beat Grid")
            for bt in r["beats"].get("beat_times", []):
                inst.notes.append(pretty_midi.Note(velocity=100, pitch=36,
                                                    start=float(bt), end=float(bt) + 0.1))
            for t in r["beats"].get("tatum_times", []):
                inst.notes.append(pretty_midi.Note(velocity=64, pitch=42,
                                                    start=float(t), end=float(t) + 0.05))
            midi.instruments.append(inst)
            buf = io.BytesIO()
            midi.write(buf)
            buf.seek(0)
            return send_file(buf, mimetype="audio/midi",
                             download_name=f"bv_{jid[:8]}_beats.mid", as_attachment=True)
        except Exception as e:
            return jsonify({"error": f"MIDI export failed: {e}"}), 500

    if fmt == "svg":
        dur = r.get("metadata", {}).get("duration", 60) or 60
        W, H = 1200, 80
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                 '<rect width="100%" height="100%" fill="#080a0f"/>']
        for t in r.get("beats", {}).get("beat_times", []):
            x = t / dur * W
            lines.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{H}" stroke="#00f5d4" stroke-width="1.5" opacity="0.7"/>')
        for t in r.get("onsets", {}).get("onsets_energy", []):
            x = t / dur * W
            lines.append(f'<line x1="{x:.1f}" y1="20" x2="{x:.1f}" y2="{H}" stroke="#f700ff" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.5"/>')
        for t in r.get("structure", {}).get("segment_times", []):
            x = t / dur * W
            lines.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{H}" stroke="#ff6b35" stroke-width="2" opacity="0.9"/>')
        lines.append("</svg>")
        return Response("\n".join(lines), mimetype="image/svg+xml",
                        headers={"Content-Disposition": f"attachment; filename=bv_{jid[:8]}_timeline.svg"})

    return jsonify({"error": f"Unknown format: {fmt}. Use json, csv, midi, svg"}), 400


# ── Compare ───────────────────────────────────────────────────────────────────
@api_bp.route("/compare", methods=["POST"])
def compare():
    data = request.get_json()
    jids = data.get("job_ids", [])
    if len(jids) != 2:
        return jsonify({"error": "Provide exactly 2 job_ids"}), 400
    results = []
    for jid in jids:
        with _jobs_lock: job = _jobs.get(jid)
        if not job or job["status"] != "complete":
            return jsonify({"error": f"Job {jid} not complete"}), 404
        results.append(job["result"])
    a, b = results

    def safe_diff(path):
        va, vb = a, b
        for k in path.split("."):
            va = va.get(k) if isinstance(va, dict) else None
            vb = vb.get(k) if isinstance(vb, dict) else None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            return {"a": va, "b": vb, "diff": round(float(vb) - float(va), 4)}
        return {"a": va, "b": vb}

    return _safe_jsonify({
        "job_ids": jids,
        "comparison": {
            "bpm":           safe_diff("beats.bpm_consensus"),
            "key":           {"a": a.get("harmony", {}).get("key_full"),
                              "b": b.get("harmony", {}).get("key_full")},
            "dynamic_range": safe_diff("dynamics.dynamic_range"),
            "brightness":    safe_diff("timbre.brightness"),
            "warmth":        safe_diff("timbre.warmth"),
            "roughness":     safe_diff("timbre.roughness"),
            "danceability":  safe_diff("music_info.danceability"),
            "energy":        safe_diff("music_info.energy"),
            "valence":       safe_diff("music_info.valence"),
            "duration":      safe_diff("metadata.duration"),
            "loudness":      {"a": a.get("dynamics", {}).get("loudness", {}).get("integrated_lufs"),
                              "b": b.get("dynamics", {}).get("loudness", {}).get("integrated_lufs")},
        },
    })


# ── Batch ─────────────────────────────────────────────────────────────────────
@api_bp.route("/batch/upload", methods=["POST"])
def batch_upload():
    files = request.files.getlist("files[]")
    if not files: return jsonify({"error": "No files"}), 400
    cfg = current_app.config
    jobs = []
    for f in files:
        if not _allowed(f.filename, cfg): continue
        fname = secure_filename(f.filename)
        jid   = str(uuid.uuid4())
        path  = os.path.join(cfg["UPLOAD_FOLDER"], f"{jid}_{fname}")
        f.save(path)
        with _jobs_lock:
            _jobs[jid] = {
                "id": jid, "filename": fname, "filepath": path,
                "file_type": _ftype(fname, cfg), "file_size": os.path.getsize(path),
                "status": "uploaded", "progress": 0, "message": "Queued",
                "created_at": time.time(), "result": None, "error": None,
            }
        jobs.append({"job_id": jid, "filename": fname})
    return jsonify({"jobs": jobs, "count": len(jobs)})


@api_bp.route("/batch/analyze", methods=["POST"])
def batch_analyze():
    data  = request.get_json()
    jids  = data.get("job_ids", [])
    started, errors = [], []
    for jid in jids:
        with _jobs_lock: job = _jobs.get(jid)
        if not job:
            errors.append({"job_id": jid, "error": "Not found"}); continue
        if job["status"] in ("analyzing", "complete"): continue
        thread = threading.Thread(
            target=_run,
            args=(jid, job["filepath"], current_app._get_current_object()),
            daemon=True,
        )
        thread.start()
        with _jobs_lock:
            _jobs[jid]["status"] = "analyzing"
            _jobs[jid]["progress"] = 1
        started.append(jid)
    return jsonify({"started": started, "errors": errors})


@api_bp.route("/jobs")
def list_jobs():
    with _jobs_lock:
        jobs = [{"id": j["id"], "filename": j["filename"], "status": j["status"],
                 "progress": j["progress"], "created_at": j["created_at"]}
                for j in _jobs.values()]
    return jsonify({"jobs": sorted(jobs, key=lambda x: x["created_at"], reverse=True),
                    "count": len(jobs)})


@api_bp.route("/jobs/<jid>", methods=["DELETE"])
def delete_job(jid):
    with _jobs_lock: job = _jobs.pop(jid, None)
    if not job: return jsonify({"error": "Not found"}), 404
    try: os.remove(job["filepath"])
    except Exception: pass
    return jsonify({"deleted": jid})


# ── Supported formats — sets converted to sorted lists ───────────────────────
@api_bp.route("/supported-formats")
def supported_formats():
    cfg = current_app.config
    # Convert sets → sorted lists so JSON serialization works
    fmt_dict = {cat: sorted(exts) for cat, exts in cfg["ALLOWED_EXTENSIONS"].items()}
    return jsonify({
        "formats":        fmt_dict,
        "all_extensions": sorted(_all_exts(cfg)),
        "total":          len(_all_exts(cfg)),
        "max_size_mb":    cfg["MAX_CONTENT_LENGTH"] // 1024 // 1024,
    })


@api_bp.route("/health")
def health():
    try:
        from core.analyzer import HAS_PRAAT, HAS_SMILE, HAS_LOUDNORM, HAS_MIDI, HAS_NR
    except ImportError:
        HAS_PRAAT = HAS_SMILE = HAS_LOUDNORM = HAS_MIDI = HAS_NR = False
    return jsonify({
        "status":  "ok",
        "version": "4.0",
        "capabilities": {
            "praat_voice_analysis":  HAS_PRAAT,
            "egemaps_opensmile":     HAS_SMILE,
            "itu_r_bs1770_loudness": HAS_LOUDNORM,
            "midi_analysis":         HAS_MIDI,
            "noise_reduction":       HAS_NR,
        },
    })
