from django.db import models
from django.utils import timezone

class MovementLog(models.Model):
    user_id = models.CharField(max_length=100)
    track_id = models.IntegerField()
    bbox = models.JSONField()
    state = models.CharField(max_length=20)
    store = models.CharField(max_length=100, null=True, blank=True)
    timestamp = models.DateTimeField()

class DailyVideoExport(models.Model):
    """Track daily video exports to prevent duplicates"""
    EXPORT_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    camera_id = models.CharField(max_length=100)
    export_date = models.DateField()  # The date for which the export was requested
    status = models.CharField(max_length=20, choices=EXPORT_STATUS_CHOICES, default='pending')
    export_id = models.CharField(max_length=200, null=True, blank=True)  # Pelco export ID
    export_url = models.URLField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    error_message = models.TextField(null=True, blank=True)
    
    class Meta:
        unique_together = ('camera_id', 'export_date')  # Prevent duplicate exports for same camera/date
        
    def __str__(self):
        return f"Export {self.camera_id} - {self.export_date} ({self.status})"
