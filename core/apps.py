from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)  # Use module-level logger

class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

