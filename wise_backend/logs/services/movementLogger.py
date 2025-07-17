import logging
from datetime import datetime
from django.utils import timezone
from django.db import transaction
from core.models import User, Visit, UserMovement, MallStore

class MovementLogger:
    """Django ORM-based movement logger for video processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_or_create_daily_visit(self, user):
        """Get or create a visit for the user for today's date"""
        today = timezone.now().date()
        current_time = timezone.now()
        
        self.logger.info(f"get_or_create_daily_visit called for user {user.user_id}. Today's date: {today}, Current time: {current_time}")
        
        # Check if user already has a visit for today (regardless of active/ended status)
        daily_visit = Visit.objects.filter(user=user, visit_date=today).first()
        
        if daily_visit:
            self.logger.info(f"Found existing visit {daily_visit.visit_id} for user {user.user_id} on {today} (visit_date: {daily_visit.visit_date})")
            return daily_visit
        
        # Create new visit for today
        visit = Visit.objects.create(
            user=user,
            start_time=current_time,
            visit_date=today
        )
        self.logger.info(f"Created NEW visit {visit.visit_id} for user {user.user_id} on {today} at {current_time}")
        return visit
    
    def log_person_movement(
        self, 
        person_id: str, 
        location: tuple, 
        timestamp: datetime, 
        camera_id: str, 
        confidence: float,
        store_id: str | None = None,
        activity_type: str = 'walking',
        bbox: tuple | None = None,
        face_id: str | None = None,
        end_time: datetime | None = None
    ) -> bool:
        """Log person movement directly to Django database (visits and userMovements tables)"""
        try:
            with transaction.atomic():
                # Get user by face_id (person_id should be face_id from AWS recognition)
                try:
                    user = User.objects.get(face_id=person_id)
                    self.logger.info(f"Processing movement for user {person_id} ({user.name})")
                except User.DoesNotExist:
                    # Try alternative lookup by name if face_id not found
                    try:
                        user = User.objects.get(name=person_id)
                        self.logger.info(f"Processing movement for user {person_id} ({user.name}) by name lookup")
                    except User.DoesNotExist:
                        self.logger.error(f"User with face_id or name '{person_id}' not found in database. Skipping movement.")
                        return False
                
                # Get or create daily visit for this user
                visit = self.get_or_create_daily_visit(user)
                self.logger.info(f"Using visit {visit.visit_id} (date: {visit.visit_date}) for user {person_id}")
                
                # Get store if provided
                store = None
                if store_id:
                    store, _ = MallStore.objects.get_or_create(
                        store_code=store_id,
                        defaults={'store_name': store_id}
                    )
                
                # Create UserMovement record with visit_id as foreign key
                movement_record = UserMovement(
                    visit=visit,  # Using visit_id as foreign key
                    start_time=timezone.make_aware(timestamp) if timezone.is_naive(timestamp) else timestamp,
                    situation=activity_type,
                    camera_id=camera_id,
                    location=f"Track {camera_id}_{person_id}",
                    store=store
                )
                
                # Add end_time if provided
                if end_time:
                    movement_record.end_time = timezone.make_aware(end_time) if timezone.is_naive(end_time) else end_time
                
                movement_record.save()
                self.logger.info(f"Created movement record for visit {visit.visit_id}, timestamp: {timestamp}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to log movement: {e}")
            return False