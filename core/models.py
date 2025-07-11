from django.db import models
import uuid
import random
import string
from django.utils import timezone

# --- USERS ---
class User(models.Model):
    user_id = models.CharField(max_length=255, primary_key=True, editable=False)
    name = models.CharField(max_length=255, null=True, blank=True)
    email = models.EmailField(null=True, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    cell_phone = models.CharField(max_length=20, null=True, blank=True)
    picture_url = models.TextField(null=True, blank=True)
    face_id = models.CharField(max_length=100, unique=True, null=True)
    # profiling_questions = models.TextField(null=True, blank=True)

    monthly_visits = models.IntegerField(null=True, blank=True)
    yearly_visits = models.IntegerField(null=True, blank=True)
    life_visits = models.IntegerField(null=True, blank=True)
    avg_time_per_visit_year = models.DurationField(null=True, blank=True)
    avg_time_per_visit_life = models.DurationField(null=True, blank=True)
    stores_visited_month = models.IntegerField(null=True, blank=True)
    stores_visited_life = models.IntegerField(null=True, blank=True)

    first_visit = models.DateField(null=True, blank=True)
    last_visit = models.DateField(null=True, blank=True)
    recency = models.IntegerField(null=True, blank=True)
    monthly_freq = models.IntegerField(null=True, blank=True)

    pattern_1 = models.CharField(max_length=255, null=True, blank=True)
    pattern_2 = models.CharField(max_length=255, null=True, blank=True)
    pattern_3 = models.CharField(max_length=255, null=True, blank=True)
    pattern_4 = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user_id} - {self.name}"
    def save(self, *args, **kwargs):
        if not self.user_id:
            self.user_id = self.generate_unique_user_id()
        super().save(*args, **kwargs)

    @staticmethod
    def generate_unique_user_id():
        while True:
            new_id = 'U' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            if not User.objects.filter(user_id=new_id).exists():
                return new_id

# --- VISITS ---
class Visit(models.Model):
    visit_id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="visits")
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    duration = models.DurationField(null=True, blank=True)
    visit_date = models.DateField(null=True, blank=True)
    stores_visited = models.IntegerField(null=True, blank=True)

# --- MALL STORES ---
class MallStore(models.Model):
    store_code = models.CharField(max_length=255, primary_key=True)
    store_name = models.CharField(max_length=255)
    pattern_characterstic_1 = models.CharField(max_length=255, null=True, blank=True)
    pattern_characterstic_2 = models.CharField(max_length=255, null=True, blank=True)
    pattern_characterstic_3 = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.store_code

# --- USER MOVEMENTS ---
class UserMovement(models.Model):
    movement_id = models.AutoField(primary_key=True)
    visit = models.ForeignKey(Visit, on_delete=models.CASCADE, related_name="movements")
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    situation = models.CharField(max_length=255)
    camera_id = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    store = models.ForeignKey(MallStore, on_delete=models.SET_NULL, null=True, blank=True, related_name="movements")

# --- INTERESTS ---
class Interest(models.Model):
    interest_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

# --- USER INTERESTS ---
class UserInterest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="interests")
    interest = models.ForeignKey(Interest, on_delete=models.CASCADE, related_name="users")
    source = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "interest")

# --- STORE (polygon based) ---
class Store(models.Model):
    id = models.CharField(max_length=255, primary_key=True)
    name = models.CharField(max_length=255)
    category = models.CharField(max_length=255)
    polygon = models.JSONField()  # storing GeoJSON
    is_mapped = models.BooleanField(default=False)

    def __str__(self):
        return self.name

# --- CAMERAS ---
class Camera(models.Model):
    id = models.CharField(max_length=255, primary_key=True)
    position = models.JSONField()  # GeoJSON { type: Point, coordinates: [x, y] }
    orientation = models.FloatField()
    fov_angle = models.FloatField()
    fov_range = models.FloatField()

# --- CALIBRATION ---
class Calibration(models.Model):
    store = models.OneToOneField(Store, on_delete=models.CASCADE, primary_key=True, related_name="calibration")
    matrix = models.JSONField()

class BusinessHour(models.Model):
    DAY_CHOICES = [
        ('Monday', 'Monday'),
        ('Tuesday', 'Tuesday'),
        ('Wednesday', 'Wednesday'),
        ('Thursday', 'Thursday'),
        ('Friday', 'Friday'),
        ('Saturday', 'Saturday'),
        ('Sunday', 'Sunday'),
    ]
    day = models.CharField(max_length=10, choices=DAY_CHOICES)
    start_time = models.TimeField()
    end_time = models.TimeField()

    def __str__(self):
        return f"{self.day}: {self.start_time} - {self.end_time}"


class EmailCampaign(models.Model):
    campaign_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(default=timezone.now)
    business_hours = models.ManyToManyField(BusinessHour, related_name="campaigns")
    is_active = models.BooleanField(default=True)
    sendgrid_list_id = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return self.name
    

class CampaignStep(models.Model):
    campaign = models.ForeignKey(EmailCampaign, on_delete=models.CASCADE, related_name='steps')
    step_order = models.PositiveIntegerField()  # 1 = Day 1, 2 = Day 2 etc.
    subject = models.CharField(max_length=255)
    body = models.TextField()
    send_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    sendgrid_campaign_id = models.CharField(max_length=100, null=True, blank=True)

    class Meta:
        ordering = ['step_order']

    def __str__(self):
        return f"Step {self.step_order} of {self.campaign.name}"


class CampaignContact(models.Model):
    campaign = models.ForeignKey(EmailCampaign, on_delete=models.CASCADE)
    user = models.ForeignKey('User', on_delete=models.CASCADE)
    added_at = models.DateTimeField(auto_now_add=True)
    sendgrid_contact_id = models.CharField(max_length=100, null=True, blank=True)

    class Meta:
        unique_together = ('campaign', 'user')  # Avoid duplicate entries

    def __str__(self):
        return f"{self.user.name} in {self.campaign.name}"

class CampaignStepImage(models.Model):
    """Model to store image metadata for campaign steps"""
    step = models.ForeignKey('CampaignStep', on_delete=models.CASCADE, related_name='images')
    s3_key = models.CharField(max_length=500)
    s3_url = models.URLField()
    original_filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=100)
    file_size = models.IntegerField()
    upload_order = models.IntegerField(default=0)  # For maintaining image order
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['upload_order', 'created_at']