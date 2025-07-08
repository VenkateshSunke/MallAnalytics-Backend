from rest_framework import serializers
from django.core.validators import RegexValidator, validate_email

# from .models import User, Visit, UserMovement, MallStore, Interest, UserInterest, Store, Camera, Calibration
from .models import *;


# --- SCREEN 1: USERS LIST SERIALIZER ---
class UserListSerializer(serializers.ModelSerializer):
    user_id = serializers.CharField(read_only=True)
    class Meta:
        model = User
        fields = [
            'user_id', 'name', 'email', 'cell_phone', 'picture_url',
            'monthly_visits', 'yearly_visits', 'life_visits',
            'recency', 'pattern_1','created_at'
        ]

    def validate(self, data):
        if not data.get('picture_url'):
            raise serializers.ValidationError("Picture URL is required.")
        return data
# --- VISIT SUMMARY FOR USER DETAIL ---
class VisitSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = Visit
        fields = ['visit_id', 'start_time', 'end_time', 'duration', 'visit_date', 'stores_visited']

# --- SCREEN 2: USER + VISITS SERIALIZER ---
class UserDetailWithVisitsSerializer(serializers.ModelSerializer):
    visits = VisitSummarySerializer(many=True, read_only=True)
    user_selected_interests = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = [
            'user_id', 'name', 'email', 'date_of_birth', 'district', 'cell_phone',
            'picture_url',
            'monthly_visits', 'yearly_visits', 'life_visits',
            'avg_time_per_visit_year', 'avg_time_per_visit_life',
            'stores_visited_month', 'stores_visited_life',
            'first_visit', 'last_visit', 'recency', 'monthly_freq',
            'pattern_1', 'pattern_2', 'pattern_3',
            'visits','user_selected_interests','created_at'
        ]
    def get_user_selected_interests(self, obj):
        user_interests = UserInterest.objects.filter(user=obj, source='registration')
        return InterestSerializer([ui.interest for ui in user_interests], many=True).data

class UserCreateSerializer(serializers.ModelSerializer):
    user_selected_interests = serializers.ListField(
        child=serializers.UUIDField(), write_only=True, required=False
    )
    name = serializers.CharField(
        max_length=100,
        required=True,
        validators=[RegexValidator(r'^[a-zA-Z\s]+$', message="Name must contain only letters and spaces.")]
    )
    email = serializers.EmailField(
        required=True,
        validators=[validate_email]
    )
    date_of_birth = serializers.DateField(
        required=True,
        error_messages={"invalid": "Enter a valid date in YYYY-MM-DD format."}
    )
    district = serializers.CharField(required=True, allow_blank=False)
    cell_phone = serializers.CharField(
        required=True,
        validators=[RegexValidator(r'^\+?\d{10,15}$', message="Enter a valid phone number.")]
    )
    picture_url = serializers.CharField(required=False, allow_blank=True)
    face_id = serializers.CharField(required=False, allow_blank=True)

    class Meta:
        model = User
        fields = [
            'user_id', 'name', 'email', 'date_of_birth',
            'district', 'cell_phone', 'picture_url','face_id','user_selected_interests'
        ]

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value
    def create(self, validated_data):
        interests = validated_data.pop("user_selected_interests", [])
        user = User.objects.create(**validated_data)

        # Create UserInterest records
        for interest_id in interests:
            UserInterest.objects.create(
                user=user,
                interest_id=interest_id,
                source="registration"
            )
        return user



# --- USER SERIALIZER ---
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['user_id', 'name', 'email', 'date_of_birth', 'cell_phone', 'picture_url']



# --- VISIT SERIALIZER ---
class VisitSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField()  # Optional: include user ID

    class Meta:
        model = Visit
        fields = '__all__'


# --- SHALLOW/NESTED VARIANTS ---
class ShallowVisitSerializer(serializers.ModelSerializer):
    class Meta:
        model = Visit
        fields = ['visit_id', 'start_time', 'end_time', 'visit_date']


class ShallowUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['user_id', 'name']


# --- NESTED VISIT WITH USER ---
class VisitDetailSerializer(serializers.ModelSerializer):
    user = ShallowUserSerializer()
    class Meta:
        model = Visit
        fields = '__all__'


# --- NESTED USER WITH VISITS + INTERESTS ---
class UserDetailSerializer(serializers.ModelSerializer):
    visits = ShallowVisitSerializer(many=True, read_only=True)
    interests = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = '__all__'

    def get_interests(self, obj):
        return UserInterestSerializer(obj.interests.all(), many=True).data


# --- USER MOVEMENT ---
class UserMovementSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserMovement
        fields = '__all__'


# --- MALL STORE ---
class MallStoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = MallStore
        fields = '__all__'


# --- INTEREST ---
class InterestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Interest
        fields = '__all__'


# --- USER INTEREST ---
class UserInterestSerializer(serializers.ModelSerializer):
    interest = InterestSerializer(read_only=True)

    class Meta:
        model = UserInterest
        fields = '__all__'


# --- STORE ---
class StoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Store
        fields = '__all__'


# --- CAMERA ---
class CameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = Camera
        fields = '__all__'


# --- CALIBRATION ---
class CalibrationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Calibration
        fields = '__all__'

class BusinessHourSerializer(serializers.ModelSerializer):
    class Meta:
        model = BusinessHour
        fields = ['id', 'day', 'start_time', 'end_time']


class EmailCampaignSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmailCampaign
        fields = [
            'campaign_id', 'name', 'created_at', 'is_active'
        ]
    
class CampaignMinimalSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmailCampaign
        fields = ['campaign_id', 'name']


class CampaignContactSerializer(serializers.ModelSerializer):
    user = UserSerializer()
    class Meta:
        model = CampaignContact
        fields = ['campaign', 'user', 'added_at']

class CampaignStepImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = CampaignStepImage
        fields = ['id', 's3_url', 'original_filename', 'content_type', 'file_size', 'upload_order']

class CampaignStepSerializer(serializers.ModelSerializer):
    images = CampaignStepImageSerializer(many=True, read_only=True)
    # Add image_files field for handling uploads
    image_files = serializers.ListField(
        child=serializers.ImageField(),
        write_only=True,
        required=False,
        allow_empty=True
    )
    
    class Meta:
        model = CampaignStep
        fields = ['id', 'step_order', 'subject', 'body', 'send_at', 'sendgrid_campaign_id', 'images', 'image_files']
        read_only_fields = ['sendgrid_campaign_id']

    def create(self, validated_data):
        validated_data.pop('image_files', None)
        return super().create(validated_data)