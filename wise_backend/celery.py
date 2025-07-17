import os
from celery import Celery
from celery.schedules import crontab

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wise_backend.settings')

app = Celery('wise_backend')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

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
        'schedule': crontab(hour='*/12', minute=0),  # Every 12 hours
    },
}
app.conf.timezone = 'UTC'

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')