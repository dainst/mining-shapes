from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process/<int:session_id>',
         views.process, name='process'),
    path('sessionresult/<int:session_id>',
         views.sessionresult, name='sessionresult')
]
