from __future__ import absolute_import, unicode_literals
import os

from celery import Celery, signals
from django.conf import settings
from celery.schedules import crontab
from celery.signals import setup_logging, worker_shutdown
from celery.app.control import Inspect
import logging
# Removed kombu imports since we're not using custom queues

logging = logging.getLogger("celery")

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wise_backend.settings')

app = Celery('wise_backend')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object(settings, namespace='CELERY')
app.conf.broker_connection_retry_on_startup = True
app.conf.update(
    worker_pool_restarts=True,
    task_track_started=True,
    worker_log_format=settings.LOGGING["formatters"]["verbose"]["format"],
    worker_task_log_format=settings.LOGGING["formatters"]["verbose"]["format"],
)

# Simple configuration without custom queues

@setup_logging.connect
def config_loggers(*args, **kwags):
    from logging.config import dictConfig
    from django.conf import settings

    dictConfig(settings.LOGGING)

@worker_shutdown.connect
def graceful_shutdown(sender, **kwargs):
    logging.warning("Celery worker shutting down. Finishing remaining tasks...")

    # Get the current worker's app
    app = sender.app

    # Create an inspector
    inspector = Inspect(app=app)

    # Get active tasks
    active_tasks = inspector.active()

    if active_tasks:
        for worker_name, tasks in active_tasks.items():
            logging.warning(f"Tasks still running on worker {worker_name}:")
            for task in tasks:
                task_id = task.get("id", "Unknown")
                task_name = task.get("name", "Unknown")
                task_args = task.get("args", [])
                task_kwargs = task.get("kwargs", {})

                logging.warning(f"  - Task ID: {task_id}")
                logging.warning(f"    Name: {task_name}")
                logging.warning(f"    Args: {task_args}")
                logging.warning(f"    Kwargs: {task_kwargs}")
    else:
        logging.warning("No active tasks found during shutdown.")

    # Allow the worker to continue with the shutdown process
    pass

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Periodic task schedule
app.conf.beat_schedule = {
    # 'process-movement-logs': {
    #     'task': 'wise_backend.logs.tasks.process_movement_logs_batch',
    #     'schedule': 10.0,  # Every 10 seconds
    # },
    'export-camera-task': {
        'task': 'wise_backend.logs.tasks.start_batch_camera',
        'schedule': crontab(hour=4, minute=30),  # 4:30 AM UTC = 11:30 PM Peru time (UTC-5)
    },
}

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')