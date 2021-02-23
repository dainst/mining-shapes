from celery import shared_task
from celery_progress.backend import ProgressRecorder
import time


from .models import Session, VesselProfile
from .image_processing import load_seg_model, predict_seg_image, read_image
from .model_utils import add_seg_image_to_vesselmodel

# pylint: disable=no-member

# Place for Celery tasks


@shared_task(bind=True)
def run_process(self, session_id: int):
    progress_recorder = ProgressRecorder(self)
    session = Session.objects.get(pk=session_id)
    model = load_seg_model(session.model)
    progressbar_len = len(VesselProfile.objects.filter(session=session))
    for i, profile in enumerate(VesselProfile.objects.filter(session=session), 1):
        image = read_image(profile.input_image.url)
        out = predict_seg_image(model, image)
        add_seg_image_to_vesselmodel(out, profile)
        progress_recorder.set_progress(
            i, progressbar_len, description=profile.filename)
