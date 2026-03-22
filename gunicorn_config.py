# gunicorn_config.py — Production server configuration
import multiprocessing

bind = "0.0.0.0:5000"
worker_class = "eventlet"
workers = 1  # eventlet requires 1 worker
worker_connections = 1000
timeout = 300  # 5 minutes for large file analysis
keepalive = 5
max_requests = 500
max_requests_jitter = 50
preload_app = True
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True
enable_stdio_inheritance = True
