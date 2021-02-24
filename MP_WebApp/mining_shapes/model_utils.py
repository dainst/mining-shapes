from django.core.files.uploadedfile import InMemoryUploadedFile
from typing import List
import cv2 as cv
import numpy as np
from django.core.files.base import ContentFile

from .models import Session, VesselProfile, FetureType

# pylint: disable=no-member


def put_images_in_vesselmodel(session: Session, img_files: List[InMemoryUploadedFile]):
    for image in img_files:
        profile = VesselProfile(
            filename=image.name,
            input_image=image,
            session=session,
        )
        profile.save()


def put_features_in_session(session: Session, features: List[str]) -> None:
    for feature in features:
        feature_type, _ = FetureType.objects.get_or_create(name=feature)
        feature_type.save()
        session.features.add(feature_type)


def add_seg_image_to_vesselmodel(image: np.ndarray, vesselprofile: VesselProfile) -> None:
    frame_jpg = cv.imencode('.jpg', image)
    file_i = ContentFile(frame_jpg[1])
    vesselprofile.segmented_image.save(
        vesselprofile.filename, file_i, save=True)
    vesselprofile.save()
