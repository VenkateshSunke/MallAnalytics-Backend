from rest_framework import serializers
from .models import VideoAgentUser

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoAgentUser
        fields = ("id", "username", "email", "created_at", "updated_at")
        read_only_fields = ("id", "created_at", "updated_at")

class UserUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoAgentUser
        fields = ("username", "email")
        
    def validate_email(self, value):
        """
        Validate that the email is unique (case-insensitive)
        """
        if value:
            value = value.lower()
            user = self.context['request'].user
            if VideoAgentUser.objects.filter(email__iexact=value).exclude(id=user.id).exists():
                raise serializers.ValidationError("A user with this email already exists.")
        return value