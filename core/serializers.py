from rest_framework import serializers
from .models import User, Visit, UserMovement, MallStore, Interest, UserInterest, Store, Camera, Calibration


# --- SCREEN 1: USERS LIST SERIALIZER ---
class UserListSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            'user_id', 'name', 'email', 'cell_phone', 'picture_url',
            'monthly_visits', 'yearly_visits', 'life_visits',
            'recency', 'pattern_1'
        ]

# --- VISIT SUMMARY FOR USER DETAIL ---
class VisitSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = Visit
        fields = ['visit_id', 'start_time', 'end_time', 'duration', 'visit_date', 'stores_visited']

# --- SCREEN 2: USER + VISITS SERIALIZER ---
class UserDetailWithVisitsSerializer(serializers.ModelSerializer):
    visits = VisitSummarySerializer(many=True, read_only=True)

    class Meta:
        model = User
        fields = [
            'user_id', 'name', 'email', 'date_of_birth', 'address', 'cell_phone',
            'picture_url', 'profiling_questions',
            'monthly_visits', 'yearly_visits', 'life_visits',
            'avg_time_per_visit_year', 'avg_time_per_visit_life',
            'stores_visited_month', 'stores_visited_life',
            'first_visit', 'last_visit', 'recency', 'monthly_freq',
            'pattern_1', 'pattern_2', 'pattern_3',
            'visits'
        ]

class UserCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            'user_id', 'name', 'email', 'date_of_birth',
            'address', 'cell_phone', 'picture_url', 'profiling_questions'
        ]


# --- USER SERIALIZER ---
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['user_id', 'name', 'email', 'date_of_birth', 'address', 'cell_phone', 'picture_url', 'profiling_questions']



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
