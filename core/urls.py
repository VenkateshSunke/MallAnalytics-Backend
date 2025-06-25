from django.urls import path
from . import views

urlpatterns = [
    path('users/', views.UserListView.as_view(), name='user-list'), # Screen 1
    path('users/<str:user_id>/', views.UserDetailView.as_view(), name='user-detail'),#Screen 2
    path('users-create/', views.CreateUserView.as_view(), name='create-user'),
    
    path('upload-photo-local/', views.PhotoUploadView.as_view(), name='upload-photo-local'),
    path('upload-photo/', views.uploadPhotoView.as_view(), name='upload-photo'),
    path('get-photo-url/',views.GeneratePresignedURL.as_view(),name='get-photo-url'),


    path("mapping-data/", views.MappingDataView.as_view()),
    path("import-mapping/", views.ImportMappingView.as_view(),name='import-mapping'),
    path("visits/", views.CreateVisitView.as_view()),
    path("visits/<int:visit_id>/", views.GetVisitView.as_view()),
    path("movements/", views.CreateMovementView.as_view()),
    path("movements/<int:visit_id>/", views.MovementByVisitView.as_view()),
    path("stores/", views.CreateListStoreView.as_view()),
    path("interests/", views.CreateListInterestView.as_view()),
    path("user-interests/", views.CreateUserInterestView.as_view()),
    path("user-interests/<str:user_id>/", views.GetUserInterestsView.as_view()),

    path('business-hours/', views.BusinessHourListView.as_view(), name='business-hour-list'),

    # ---- campaign's related apis ----- #
    path('campaigns/', views.EmailCampaignListCreateView.as_view(), name='campaign-list-create'),
    path('campaigns/<int:campaign_id>/', views.EmailCampaignDetailView.as_view(), name='campaign-detail'),
    path('campaigns/<int:campaign_id>/toggle/', views.EmailCampaignToggleView.as_view(), name='campaign-toggle'),
    path('campaigns/<int:campaign_id>/add-contacts/', views.AddCampaignContactsView.as_view(), name='campaign-add-contacts'),
    path('campaigns/minimal/', views.CampaignMinimalListView.as_view(), name='campaign-minimal'),
    path('campaigns/<int:campaign_id>/contacts/', views.CampaignContactListView.as_view(), name='campaign-contacts'),
    path('campaigns/<int:campaign_id>/steps/', views.CampaignStepListCreateView.as_view(), name='campaign-step-list-create'),
    path('campaigns/<int:campaign_id>/steps/<int:pk>/', views.CampaignStepDetailView.as_view(), name='campaign-step-detail'),
    path('steps/<int:step_id>/schedule/', views.ScheduleCampaignStepView.as_view(), name='schedule-campaign-step'),
    path('steps/<int:step_id>/sendgrid-stats/', views.CampaignStepSendGridStatsView.as_view(), name='campaign-step-sendgrid-stats'),
    
    # ---- dashboard endpoint ---- #
    path('dashboard-metrics/', views.DashboardMetricsView.as_view(), name='dashboard-metrics'),

    path('sendgrid/senders/', views.SendGridSenderListView.as_view(), name='sendgrid-sender-list'),
    path('sendgrid/suppression-groups/', views.SendGridSuppressionGroupListView.as_view(), name='sendgrid-suppression-group-list'),

]
