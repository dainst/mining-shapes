from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process/<int:session_id>/<str:seg_model>/<str:features>',
         views.process, name='process'),
]
