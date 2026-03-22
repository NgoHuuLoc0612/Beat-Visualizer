"""Beat Visualizer v4.0 — Flask + Socket.IO entry point"""
import os, sys, logging
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

# ── Windows: force UTF-8 on stdout/stderr so emoji don't crash CP1252 ────────
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("beat_visualizer.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "bv-v4-secret-key"),
    MAX_CONTENT_LENGTH=2 * 1024**3,   # 2 GB
    UPLOAD_FOLDER=os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads"),
    ANALYSIS_CACHE_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"),
    DEBUG=os.environ.get("DEBUG", "True") == "True",
    # ── 90+ supported formats ────────────────────────────────────────────────
    ALLOWED_EXTENSIONS={
        "lossless":   {"wav","flac","aiff","aif","au","snd","w64","caf","rf64","bwf",
                       "dsf","dff","ape","wv","tta","tak","wavpack","adc","pcm","raw"},
        "lossy":      {"mp3","ogg","oga","aac","m4a","m4b","m4r","wma","opus","spx",
                       "amr","awb","ra","rm","mp2","mp1","ac3","eac3","dts","mka","webm","gsm","g726","g729"},
        "video":      {"mp4","mov","mkv","avi","flv","wmv","ts","m2ts","mts","3gp","3g2",
                       "f4a","f4v","ogv","divx","asf","vob"},
        "midi":       {"mid","midi","smf","kar","rmi","xmf"},
        "tracker":    {"mod","xm","it","s3m","stm","mtm","med","oct","far","umx","669"},
        "project":    {"json","bvp","csv","xml"},
        "multitrack": {"mxf","mlp","thd","dtshd","sd2"},
        "stems":      {"stem.mp4","stem"},
        "streaming":  {"m3u8","mpd"},
    },
)

CORS(app, origins="*")
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=2 * 1024**3,
    ping_timeout=600,
    ping_interval=30,
)

for d in [app.config["UPLOAD_FOLDER"], app.config["ANALYSIS_CACHE_DIR"]]:
    os.makedirs(d, exist_ok=True)

from api.routes import api_bp, register_main_routes
from api.socket_handlers import register_socket_handlers
app.register_blueprint(api_bp, url_prefix="/api")
register_socket_handlers(socketio)
register_main_routes(app)

if __name__ == "__main__":
    exts = {e for s in app.config["ALLOWED_EXTENSIONS"].values() for e in s}
    port = int(os.environ.get("PORT", 5000))
    # ASCII-safe startup banner (no emoji — safe on all terminals)
    log.info("Beat Visualizer v4.0 | %d formats | http://localhost:%d", len(exts), port)
    print(f"\n  *** Beat Visualizer v4.0 ***  http://localhost:{port}\n", flush=True)
    socketio.run(app, host="0.0.0.0", port=port,
                 debug=app.config["DEBUG"], use_reloader=False, log_output=True)
