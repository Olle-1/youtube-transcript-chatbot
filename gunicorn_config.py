# gunicorn_config.py
import multiprocessing

# Server socket settings
bind = "0.0.0.0:8080"
workers = multiprocessing.cpu_count() * 2 + 1

# Timeout settings
timeout = 180  # 3 minutes
keep_alive = 120
graceful_timeout = 120

# Worker settings
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Limit request line and header field sizes
limit_request_line = 0
limit_request_fields = 100
limit_request_field_size = 8190