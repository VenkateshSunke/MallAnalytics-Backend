# from celery import shared_task
# import requests

# @shared_task
# def send_log_to_backend(person_id, state, timestamp, location, frame, store_name):
#     print(f"🟡 Raw log input: {person_id}, {state}, {timestamp}, {location}, {frame}, {store_name}")

#     # Hardcoded or mapped values (adjust as needed)
#     visit_id = 1
#     store_id_map = {
#         "MENTA": 2,
#         "ZARA": 3,
#         # add more mappings
#     }
#     store_id = store_id_map.get(store_name.upper())

#     payload = {
#         "visit": visit_id,
#         "situation": state,
#         "camera_id": "CAM01",
#         "location": location,
#         "start_time": timestamp,
#         "store": store_id
#     }

#     print(f"✅ Final payload to POST: {payload}")

#     try:
#         response = requests.post("http://localhost:8000/api/movements/", json=payload)
#         print(f"🔵 Response status: {response.status_code}, body: {response.text}")
#     except Exception as e:
#         print(f"❌ Failed to send log: {e}")
from celery import shared_task
import redis
import json
import logging
from django.db import transaction
from django.conf import settings
from django.utils import timezone
from datetime import datetime
from .models import MovementLog
from core.models import UserMovement, Visit, User, MallStore

logger = logging.getLogger(__name__)

# Create Redis client using settings
redis_client = redis.Redis(**settings.REDIS_CONFIG)


def get_or_create_daily_visit(user):
    """Get or create a visit for the user for today's date"""
    today = timezone.now().date()
    current_time = timezone.now()
    
    logger.info(f"get_or_create_daily_visit called for user {user.user_id}. Today's date: {today}, Current time: {current_time}")
    
    # Check if user already has a visit for today (regardless of active/ended status)
    daily_visit = Visit.objects.filter(user=user, visit_date=today).first()
    
    if daily_visit:
        logger.info(f"Found existing visit {daily_visit.visit_id} for user {user.user_id} on {today} (visit_date: {daily_visit.visit_date})")
        return daily_visit
    
    # Create new visit for today
    visit = Visit.objects.create(
        user=user,
        start_time=current_time,
        visit_date=today
    )
    logger.info(f"Created NEW visit {visit.visit_id} for user {user.user_id} on {today} at {current_time}")
    return visit


@shared_task(bind=True, max_retries=3)
def process_movement_logs_batch(self):
    """Process movement logs from Redis queue in batches"""
    batch_size = 100
    logs_batch = []
    
    # Pop logs from Redis queue
    for _ in range(batch_size):
        log_data = redis_client.rpop('movement_logs')
        if not log_data:
            break
        try:
            logs_batch.append(json.loads(log_data))
        except json.JSONDecodeError:
            logger.error("Invalid JSON in movement log")
            continue
    
    if not logs_batch:
        return "No logs to process"
    
    try:
        # Batch insert to database
        with transaction.atomic():
            movement_logs = []
            for log_data in logs_batch:
                try:
                    # Get user_id from the processed data (should already be user_id, not face_id)
                    user_id = log_data.get('user_id')
                    if not user_id:
                        logger.error(f"No user_id in log data: {log_data}")
                        continue
                    
                    # Get existing user (should already exist)
                    try:
                        user = User.objects.get(user_id=user_id)
                        logger.info(f"Processing movement for user {user_id} ({user.name})")
                    except User.DoesNotExist:
                        logger.error(f"User {user_id} not found in database. Skipping movement. Log data: {log_data}")
                        continue
                    
                    # Get or create daily visit for this user
                    visit = get_or_create_daily_visit(user)
                    logger.info(f"BATCH PROCESSING: Using visit {visit.visit_id} (date: {visit.visit_date}) for user {user_id}")
                    
                    # Get store if provided
                    store = None
                    store_name = log_data.get('store')
                    if store_name:
                        store, _ = MallStore.objects.get_or_create(
                            store_code=store_name,
                            defaults={'store_name': store_name}
                        )
                    
                    # Create UserMovement record with visit_id as foreign key
                    movement_record = UserMovement(
                        visit=visit,  # ✅ Using visit_id as foreign key
                        start_time=log_data['timestamp'],
                        situation=log_data.get('state', log_data.get('activity_type', 'movement')),
                        camera_id=log_data.get('camera_id', 'unknown'),
                        location=log_data.get('location', f"Track {log_data.get('track_id', 0)}"),
                        store=store
                    )
                    movement_logs.append(movement_record)
                    logger.info(f"BATCH PROCESSING: Created movement record for visit {visit.visit_id}, timestamp: {log_data['timestamp']}")
                    
                except Exception as e:
                    logger.error(f"Error processing log data: {e}, Data: {log_data}")
                    continue
            
            # Single batch insert (very efficient)
            UserMovement.objects.bulk_create(movement_logs, batch_size=1000)
        
        logger.info(f"Successfully processed {len(logs_batch)} movement logs")
        return f"Processed {len(logs_batch)} logs"
        
    except Exception as exc:
        logger.error(f"Error processing movement logs: {exc}")
        # Retry with exponential backoff
        raise self.retry(countdown=60 * (2 ** self.request.retries))


@shared_task
def start_visit(user_id):
    """Start a new visit for an existing user"""
    try:
        # Get existing user - do not create new users
        try:
            user = User.objects.get(user_id=user_id)
        except User.DoesNotExist:
            error_msg = f"User {user_id} not found. Users must be registered first."
            logger.error(error_msg)
            return error_msg
        
        # Check if user already has an active visit
        active_visit = Visit.objects.filter(user=user, end_time__isnull=True).first()
        if active_visit:
            return f"User {user_id} already has an active visit (ID: {active_visit.visit_id})"
        
        # Create new visit using user_id as foreign key
        visit = Visit.objects.create(
            user=user,  # ✅ Using user as foreign key
            start_time=timezone.now(),
            visit_date=timezone.now().date()
        )
        
        logger.info(f"Started visit {visit.visit_id} for user {user_id} ({user.name})")
        return f"Started visit {visit.visit_id} for user {user_id} ({user.name})"
        
    except Exception as e:
        logger.error(f"Failed to start visit for user {user_id}: {e}")
        raise


@shared_task
def end_visit(user_id):
    """End the active visit for a user"""
    try:
        user = User.objects.get(user_id=user_id)
        
        # Get active visit
        active_visit = Visit.objects.filter(user=user, end_time__isnull=True).first()
        if not active_visit:
            return f"No active visit found for user {user_id}"
        
        # Update visit with end time and calculate duration
        end_time = timezone.now()
        active_visit.end_time = end_time
        active_visit.duration = end_time - active_visit.start_time
        
        # Count stores visited during this visit
        stores_visited = UserMovement.objects.filter(
            visit=active_visit,
            store__isnull=False
        ).values('store').distinct().count()
        active_visit.stores_visited = stores_visited
        
        active_visit.save()
        
        logger.info(f"Ended visit {active_visit.visit_id} for user {user_id}")
        return f"Ended visit {active_visit.visit_id} for user {user_id}. Duration: {active_visit.duration}, Stores visited: {stores_visited}"
        
    except User.DoesNotExist:
        error_msg = f"User {user_id} not found"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        logger.error(f"Failed to end visit for user {user_id}: {e}")
        raise


@shared_task
def add_movement_log_to_queue(user_id, camera_id, location, state, store=None, timestamp=None):
    """Add a single movement log to Redis queue"""
    from datetime import datetime
    
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    log_data = {
        'user_id': user_id,
        'camera_id': camera_id,
        'location': location,
        'state': state,
        'store': store,
        'timestamp': timestamp
    }
    
    try:
        redis_client.lpush('movement_logs', json.dumps(log_data))
        logger.info(f"Added movement log to queue for user {user_id}")
        return f"Successfully queued log for user {user_id}"
    except Exception as e:
        logger.error(f"Failed to queue movement log: {e}")
        raise


@shared_task
def get_queue_status():
    """Get status of Redis queues"""
    from datetime import datetime
    
    try:
        queue_length = redis_client.llen('movement_logs')
        return {
            'movement_logs_queue_length': queue_length,
            'redis_connected': True,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        return {
            'redis_connected': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@shared_task
def clear_movement_logs_queue():
    """Clear the movement logs queue (use carefully!)"""
    try:
        count = redis_client.llen('movement_logs')
        redis_client.delete('movement_logs')
        logger.info(f"Cleared {count} items from movement logs queue")
        return f"Cleared {count} items from queue"
    except Exception as e:
        logger.error(f"Failed to clear queue: {e}")
        raise


@shared_task(bind=True)
def test_celery_task(self):
    """Simple test task to verify Celery is working"""
    logger.info("Test Celery task executed successfully")
    return "Test task completed successfully"


# Utility functions for Redis operations
def push_to_redis_queue(queue_name, data):
    """Helper function to push data to a Redis queue"""
    try:
        redis_client.lpush(queue_name, json.dumps(data))
        return True
    except Exception as e:
        logger.error(f"Failed to push to Redis queue {queue_name}: {e}")
        return False


def get_redis_queue_length(queue_name):
    """Helper function to get Redis queue length"""
    try:
        return redis_client.llen(queue_name)
    except Exception as e:
        logger.error(f"Failed to get queue length for {queue_name}: {e}")
        return -1