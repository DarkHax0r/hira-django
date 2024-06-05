from django.urls import path
from . import views

urlpatterns = [
    path('results/', views.analyze_data, name='analyze_data'),
]
