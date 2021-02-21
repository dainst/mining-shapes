from django.shortcuts import render

from .forms import RunSessionForm
from .models import Session
from .model_utils import put_images_in_vesselmodel


def index(request):
    if request.method == "POST":
        form = RunSessionForm(request.POST, request.FILES)
        if form.is_valid():
            session = Session(user=request.user)
            session.save()
            put_images_in_vesselmodel(session, request.FILES.getlist('images'))

            # handle_uploaded_file(request.FILES['file'])

    return render(request, "mining_shapes/index.html", {
        "form": RunSessionForm(),
    })
