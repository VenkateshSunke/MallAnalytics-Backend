from django.urls import path
from . import views

urlpatterns = [
    path('users/', views.UserListView.as_view(), name='user-list'), # Screen 1
    path('users/<str:user_id>/', views.UserDetailView.as_view(), name='user-detail'),#Screen 2
    path('users/create/', views.CreateUserView.as_view(), name='create-user'),


    path("mapping-data/", views.MappingDataView.as_view()),
    path("import-mapping/", views.ImportMappingView.as_view()),
    path("visits/", views.CreateVisitView.as_view()),
    path("visits/<int:visit_id>/", views.GetVisitView.as_view()),
    path("movements/", views.CreateMovementView.as_view()),
    path("movements/<int:visit_id>/", views.MovementByVisitView.as_view()),
    path("stores/", views.CreateListStoreView.as_view()),
    path("interests/", views.CreateListInterestView.as_view()),
    path("user-interests/", views.CreateUserInterestView.as_view()),
    path("user-interests/<str:user_id>/", views.GetUserInterestsView.as_view()),
]
