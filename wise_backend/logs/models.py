from django.db import models

class MovementLog(models.Model):
    user_id = models.CharField(max_length=100)
    track_id = models.IntegerField()
    bbox = models.JSONField()
    state = models.CharField(max_length=20)
    store = models.CharField(max_length=100, null=True, blank=True)
    timestamp = models.DateTimeField()
