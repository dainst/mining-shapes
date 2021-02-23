from .tasks import run_process
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.http import HttpResponseRedirect


from .forms import RunSessionForm
from .models import Session
from .model_utils import put_images_in_vesselmodel, put_features_in_session
from .form_utils import get_name_of_choice_field, get_features_from_feature_field

# pylint: disable=no-member


def index(request):
    if request.method == "POST":
        form = RunSessionForm(request.POST, request.FILES)
        if form.is_valid():
            model_choice = get_name_of_choice_field(form['model'])

            session = Session(user=request.user, model=model_choice)
            session.save()
            features = get_features_from_feature_field(
                form.cleaned_data.get("features"))
            put_features_in_session(session, features)
            put_images_in_vesselmodel(session, request.FILES.getlist('images'))
            return HttpResponseRedirect(reverse("process", kwargs={'session_id': session.pk}))

    return render(request, "mining_shapes/index.html", {
        "form": RunSessionForm(),
    })


@login_required
def process(request, session_id: int):
    session = Session.objects.get(pk=session_id)
    features = [i.name for i in session.features.all()]

    result = run_process.delay(10)
    return render(request, "mining_shapes/process.html", {
        "session": session,
        "features": features,
        'task_id': result.task_id})
