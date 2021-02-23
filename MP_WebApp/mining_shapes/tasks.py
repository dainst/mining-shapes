from celery import shared_task
from celery_progress.backend import ProgressRecorder
import time

# Place for Celery tasks


@shared_task(bind=True)
def run_process(self, seconds):
    progress_recorder = ProgressRecorder(self)
    result = 0
    for i in range(seconds):
        time.sleep(1)
        result += i
        progress_recorder.set_progress(i + 1, seconds)
    return result
