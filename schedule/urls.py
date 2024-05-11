# schedule/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('calculate/', views.your_view),
    path('calculate2/', views.your_view2),
    path('calculate3/', views.your_view3),
    path('my_task/', views.user_task),
    path('calculate4/', views.your_view4),
    path('get_status/<int:task_id>/', views.get_status),
]