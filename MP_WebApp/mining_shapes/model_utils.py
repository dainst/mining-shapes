from django.core.files.uploadedfile import InMemoryUploadedFile
from typing import List

from .models import Session, VesselProfile


def put_images_in_vesselmodel(session: Session, img_files: List[InMemoryUploadedFile]):
    for image in img_files:
        profile = VesselProfile(
            filename=image.name,
            input_image=image,
            catalog="test",
            session=session,
        )
        profile.save()
