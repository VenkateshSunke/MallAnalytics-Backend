from django.apps import AppConfig
import logging
from django.db import connections
from django.db.utils import OperationalError

logger = logging.getLogger(__name__)  # Use module-level logger

class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        try:
            connections['default'].cursor()
            logger.info("✅ Database connected successfully.")
        except OperationalError:
            logger.error("❌ Database connection failed.")