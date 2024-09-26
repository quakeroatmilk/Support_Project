from django.urls import path
from mysupportapp import views
from .views import download_user_manual_pdf, user_manual, pdf_download_page

urlpatterns = [
    path('', views.recommendation_system, name='recommendation_system'),
    path('recommend/', views.recommend, name='recommend'),
    path('prediction-analysis/', views.prediction_analysis, name='prediction_analysis'),
    path('prediction-result/', views.prediction_result, name='prediction_result'),
    path('user-feedback/', views.display_feedback, name='user_feedback'),
    path('view-feedback/', views.view_feedback, name='view_feedback'),
    path('profile-analytics/', views.profile_analytics, name='profile_analytics'),
    path('organization_analytics/', views.organization_analytic, name='organization_analytics'),
    path('upload-pdf/', views.organization_analytic, name='upload_pdf'),
    path('user-manual/', views.user_manual, name='user_manual'), 
    path('download-user-manual/', views.download_user_manual_pdf, name='download_user_manual_pdf'),
    path('download-pdf/', views.pdf_download_page, name='pdf_download_page'),
]
