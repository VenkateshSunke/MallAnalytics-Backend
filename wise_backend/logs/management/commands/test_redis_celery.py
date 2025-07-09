from django.core.management.base import BaseCommand
from django.conf import settings
import redis
import json
from datetime import datetime
from wise_backend.logs.tasks import (
    test_celery_task, 
    get_queue_status, 
    add_movement_log_to_queue,
    process_movement_logs_batch,
    start_visit,
    end_visit
)


class Command(BaseCommand):
    help = 'Test Redis and Celery setup'

    def add_arguments(self, parser):
        parser.add_argument(
            '--test-redis',
            action='store_true',
            help='Test Redis connection',
        )
        parser.add_argument(
            '--test-celery',
            action='store_true',
            help='Test Celery tasks',
        )
        parser.add_argument(
            '--test-all',
            action='store_true',
            help='Test everything',
        )
        parser.add_argument(
            '--add-sample-data',
            action='store_true',
            help='Add sample movement data to Redis queue',
        )
        parser.add_argument(
            '--process-queue',
            action='store_true',
            help='Process the Redis queue',
        )
        parser.add_argument(
            '--end-sample-visits',
            action='store_true',
            help='End visits for sample users',
        )

    def handle(self, *args, **options):
        if options['test_all']:
            self.test_redis()
            self.test_celery()
        elif options['test_redis']:
            self.test_redis()
        elif options['test_celery']:
            self.test_celery()
        elif options['add_sample_data']:
            self.add_sample_data()
        elif options['process_queue']:
            self.process_queue()
        elif options['end_sample_visits']:
            self.end_sample_visits()
        else:
            self.stdout.write(
                self.style.WARNING('Please specify an option. Use --help for available options.')
            )

    def test_redis(self):
        """Test Redis connection"""
        self.stdout.write("Testing Redis connection...")
        
        try:
            # Test basic Redis connection
            redis_client = redis.Redis(**settings.REDIS_CONFIG)
            redis_client.ping()
            self.stdout.write(
                self.style.SUCCESS('✓ Redis connection successful')
            )
            
            # Test basic operations
            test_key = 'test_key'
            test_value = 'test_value'
            redis_client.set(test_key, test_value, ex=60)  # Expires in 60 seconds
            retrieved_value = redis_client.get(test_key)
            
            # Handle both bytes and string responses
            if retrieved_value:
                if isinstance(retrieved_value, bytes):
                    retrieved_value = retrieved_value.decode()
                
                if retrieved_value == test_value:
                    self.stdout.write(
                        self.style.SUCCESS('✓ Redis read/write operations successful')
                    )
                    redis_client.delete(test_key)
                else:
                    self.stdout.write(
                        self.style.ERROR(f'✗ Redis read/write operations failed. Expected: {test_value}, Got: {retrieved_value}')
                    )
            else:
                self.stdout.write(
                    self.style.ERROR('✗ Redis read/write operations failed - no value retrieved')
                )
            
            # Test Redis queue operations
            queue_name = 'test_queue'
            test_data = {'message': 'test', 'timestamp': datetime.now().isoformat()}
            redis_client.lpush(queue_name, json.dumps(test_data))
            queue_length = redis_client.llen(queue_name)
            
            if queue_length > 0:
                self.stdout.write(
                    self.style.SUCCESS(f'✓ Redis queue operations successful (queue length: {queue_length})')
                )
                redis_client.delete(queue_name)
            else:
                self.stdout.write(
                    self.style.ERROR('✗ Redis queue operations failed')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'✗ Redis connection failed: {str(e)}')
            )

    def test_celery(self):
        """Test Celery tasks"""
        self.stdout.write("Testing Celery tasks...")
        
        try:
            # Test basic Celery task
            self.stdout.write("Running test_celery_task...")
            task = test_celery_task.delay()
            result = task.get(timeout=10)
            self.stdout.write(
                self.style.SUCCESS(f'✓ test_celery_task completed: {result}')
            )
            
            # Test queue status task
            self.stdout.write("Running get_queue_status...")
            task = get_queue_status.delay()
            result = task.get(timeout=10)
            self.stdout.write(
                self.style.SUCCESS(f'✓ get_queue_status completed: {result}')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'✗ Celery test failed: {str(e)}')
            )
            self.stdout.write(
                self.style.WARNING('Make sure Celery worker is running: celery -A wise_backend worker -l info')
            )

    def add_sample_data(self):
        """Add sample movement data to Redis queue"""
        self.stdout.write("Adding sample movement data to Redis queue...")
        
        # First, start visits for users
        users = ['USER001', 'USER002', 'USER003']
        
        try:
            # Start visits
            for user_id in users:
                task = start_visit.delay(user_id)
                result = task.get(timeout=10)
                self.stdout.write(f"Started visit for {user_id}: {result}")
            
            # Add movement logs
            sample_logs = [
                {
                    'user_id': 'USER001',
                    'camera_id': 'CAM01',
                    'location': 'Entrance',
                    'state': 'walking',
                    'store': 'Store A',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'user_id': 'USER002',
                    'camera_id': 'CAM02',
                    'location': 'Food Court',
                    'state': 'standing',
                    'store': 'Store B',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'user_id': 'USER003',
                    'camera_id': 'CAM03',
                    'location': 'Shopping Area',
                    'state': 'browsing',
                    'store': 'Store C',
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            for log_data in sample_logs:
                task = add_movement_log_to_queue.delay(**log_data)
                self.stdout.write(f"Queued movement for {log_data['user_id']} (task: {task.id})")
            
            self.stdout.write(
                self.style.SUCCESS(f'✓ Started visits and added {len(sample_logs)} sample movement logs to queue')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'✗ Failed to add sample data: {str(e)}')
            )

    def process_queue(self):
        """Process the Redis queue"""
        self.stdout.write("Processing Redis queue...")
        
        try:
            task = process_movement_logs_batch.delay()
            result = task.get(timeout=30)
            self.stdout.write(
                self.style.SUCCESS(f'✓ Queue processing completed: {result}')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'✗ Queue processing failed: {str(e)}')
            )
    
    def end_sample_visits(self):
        """End visits for sample users"""
        self.stdout.write("Ending visits for sample users...")
        
        users = ['USER001', 'USER002', 'USER003']
        
        try:
            for user_id in users:
                task = end_visit.delay(user_id)
                result = task.get(timeout=10)
                self.stdout.write(f"Ended visit for {user_id}: {result}")
            
            self.stdout.write(
                self.style.SUCCESS(f'✓ Ended visits for {len(users)} users')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'✗ Failed to end visits: {str(e)}')
            ) 