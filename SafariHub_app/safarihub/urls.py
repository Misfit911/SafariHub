from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('attraction/<str:attraction_name>/', views.attraction_detail, name='attraction_detail'),
    path('hotel/<str:hotel_name>/', views.hotel_detail, name='hotel_detail'),
    path('tourop/<str:tourop_name>/', views.tourop_detail, name='tourop_detail'),

]
