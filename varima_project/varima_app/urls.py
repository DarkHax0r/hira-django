from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path("", views.login_view, name="login"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("results/", views.analyze_data, name="analyze_data"),
    path("laporan/", views.laporan, name="laporan"),
    path("laporan/add/", views.laporan_add, name="laporan_add"),
    path("laporan/import/", views.laporan_import, name="laporan_import"),
    path("logout/", auth_views.LogoutView.as_view(next_page="login"), name="logout"),
    path("laporan/kosongkan/", views.laporan_kosongkan, name="laporan_kosongkan"),
    path("profile/", views.profile_view, name="profile"),
    path('profile/update_password/', views.update_password, name='profile_update_password'),
]
