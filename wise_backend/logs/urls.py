from django.urls import path
from . import views

urlpatterns = [
    # Visit management (Business Flow)
    path('start-visit/', views.StartVisitView.as_view(), name='start-visit'),
    path('end-visit/', views.EndVisitView.as_view(), name='end-visit'),
    path('visits/', views.VisitListView.as_view(), name='visit-list'),
    
    # Movement log management
    path('add-movement/', views.AddMovementLogView.as_view(), name='add-movement-log'),
    path('movements/', views.MovementLogListView.as_view(), name='movement-log-list'),
    
    # Queue management
    path('queue-status/', views.QueueStatusView.as_view(), name='queue-status'),
    path('process/', views.ProcessLogsView.as_view(), name='process-logs'),
    path('clear-queue/', views.ClearQueueView.as_view(), name='clear-queue'),
    
    # System monitoring
    path('test-celery/', views.TestCeleryView.as_view(), name='test-celery'),
    path('health/', views.SystemHealthView.as_view(), name='system-health'),
    
    # Debugging endpoints
    path('user-visit-detail/', views.UserVisitDetailView.as_view(), name='user-visit-detail'),
    
    # Video processing test endpoint
    path('test-video-processing/', views.TestVideoProcessingView.as_view(), name='test-video-processing'),
] 