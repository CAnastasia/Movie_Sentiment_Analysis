from django.urls import path
from . import views
from .App_views import addition
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.home),
    path('add/<int:nombre1>/<int:nombre2>/', addition.addition),
    path('date_act/', addition.date_actuelle),
    path('MachineLearn/', addition.MachineLean),
    path('state/', addition.state),
    path('feature/', addition.feature),
    path('models/', addition.models),
    path('sub/', addition.submission),
    path('desc/', addition.description)
]


if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)