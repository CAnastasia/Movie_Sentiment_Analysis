from django.urls import path
from . import views
from .App_views import addition
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.home),
    path('add/<int:nombre1>/<int:nombre2>/', addition.addition),
    path('date_act/', addition.date_actuelle),
    path('test/', addition.index),
    path('imgs/style.css', addition.index)
]


if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)