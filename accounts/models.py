from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _

class VideoAgentUser(AbstractUser):
    email = models.EmailField(unique=True)
    username = models.CharField(_("username"), max_length=254, unique=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["email"],
                name="unique_case_insensitive_email",
                violation_error_message=_("A user with this email already exists."),
            )
        ]

    def __str__(self):
        return self.email