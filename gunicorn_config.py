# Django ASGI application path
wsgi_app = "wise_backend.asgi:application"

# Logging level
loglevel = "debug"

# Worker class (custom Uvicorn worker)
worker_class = "uvicorn.workers.UvicornWorker"

# # Number of worker processes
# workers = multiprocessing.cpu_count() * 2 + 1

# Request timeout
timeout = 1000
