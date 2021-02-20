from django.shortcuts import render


def index(request):
    return render(request, "mining_shapes/index.html")
