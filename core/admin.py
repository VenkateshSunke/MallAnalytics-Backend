from django.contrib import admin
from .models import *

admin.site.register(User)
admin.site.register(Interest)
admin.site.register(UserInterest)
admin.site.register(Visit)
admin.site.register(UserMovement)