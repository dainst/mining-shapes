from django.http.response import HttpResponse
from django.shortcuts import render


def index(request):
    return render(request, "mining_shapes/layout.html")
