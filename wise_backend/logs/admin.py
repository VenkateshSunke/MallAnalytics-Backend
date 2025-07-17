from django.contrib import admin
from .models import DailyVideoExport

@admin.register(DailyVideoExport)
class DailyVideoExportAdmin(admin.ModelAdmin):
    list_display = ('camera_id', 'export_date', 'status', 'created_at')
    list_filter = ('status', 'export_date')
    search_fields = ('camera_id', 'export_date')
    date_hierarchy = 'export_date'
    ordering = ('-export_date',)
