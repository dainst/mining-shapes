from django.http.response import JsonResponse
from .tasks import run_process
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.urls import reverse
from django.http import HttpResponseRedirect
import json

from .forms import RunSessionForm
from .models import Session, VesselProfile
from .model_utils import put_images_in_vesselmodel, put_features_in_session, edit_seg_image_from_vesselprofile, profile_pagination
from .form_utils import get_name_of_choice_field, get_features_from_feature_field

# pylint: disable=no-member


def index(request):
    if request.method == "POST":
        form = RunSessionForm(request.POST, request.FILES)
        if form.is_valid():
            model_choice = get_name_of_choice_field(form['model'])

            session = Session(user=request.user,
                              model=model_choice, catalog=form.cleaned_data['catalog'])
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
    pid = run_process.delay(session_id)
    return render(request, "mining_shapes/process.html", {
        "session": session,
        "features": features,
        'task_id': pid.task_id,
        "n_files": len(VesselProfile.objects.filter(session=session)),
    })


@login_required
def sessionresult(request, session_id: int):
    session = Session.objects.get(pk=session_id)
    profiles = VesselProfile.objects.filter(session=session)
    return render(request, "mining_shapes/sessionresult.html", {
        "profiles": profile_pagination(request, profiles),
        "session": session_id
    })


@login_required
@csrf_exempt
def editshape(request, shape_id: int):
    profile = VesselProfile.objects.get(pk=shape_id)
    if request.method == "PUT":
        data = json.loads(request.body)
        edit_seg_image_from_vesselprofile(data['polygon'], profile)
        return JsonResponse({
            'message': f"Updated shape {shape_id}",
            'imageUrl': profile.input_image.url,
            'segmentedUrl': profile.segmented_image.url,
            'sessionId': profile.session.pk,
        })

    return render(request, "mining_shapes/editshape.html", {
        "profile": profile
    })


@login_required
def removeshape(request, shape_id: int):
    profile = VesselProfile.objects.get(pk=shape_id)
    session_id = profile.session.pk
    profile.delete()
    return HttpResponseRedirect(reverse("sessionresult", kwargs={'session_id': session_id}))
