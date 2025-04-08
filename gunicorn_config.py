# gunicorn_config.py
import multiprocessing

# Server socket settings
bind = "0.0.0.0:8080"
workers = 2  # For DigitalOcean App Platform, start with just 2 workers

# Timeout settings - critical for LLM responses
timeout = 300  # 5 minutes - very generous timeout for LLM processing
graceful_timeout = 300  # How long to wait for workers to finish
keep_alive = 120  # Keep connections alive for 2 minutes

# Worker settings
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50  # Restart workers randomly to prevent memory leaks

# Logging
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr

# Explicitly disable the preload to ensure workers are truly separate
preload_app = False