from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.http import HttpResponseRedirect


from .forms import RunSessionForm
from .models import Session
from .model_utils import put_images_in_vesselmodel
from .form_utils import get_name_of_choice_field, get_features_from_feature_field

FEATURE_JOIN_STR = '+'


def index(request):
    if request.method == "POST":
        form = RunSessionForm(request.POST, request.FILES)
        if form.is_valid():
            session = Session(user=request.user)
            session.save()
            put_images_in_vesselmodel(session, request.FILES.getlist('images'))
            model_choice = get_name_of_choice_field(form['model'])
            features = get_features_from_feature_field(
                form.cleaned_data.get("features"))
            return HttpResponseRedirect(reverse("process", kwargs={
                'session_id': session.pk,
                'seg_model': model_choice,
                'features': FEATURE_JOIN_STR.join(features)}))

    return render(request, "mining_shapes/index.html", {
        "form": RunSessionForm(),
    })


@login_required
def process(request, session_id: int, seg_model: str, features: str):
    return render(request, "mining_shapes/process.html", {
        "test": session_id,
        "seg": seg_model,
        "features": features.split(FEATURE_JOIN_STR)
    })
