from django.core.files.uploadedfile import InMemoryUploadedFile
from typing import List

from .models import Session, VesselProfile, FetureType

# pylint: disable=no-member


def put_images_in_vesselmodel(session: Session, img_files: List[InMemoryUploadedFile]):
    for image in img_files:
        profile = VesselProfile(
            filename=image.name,
            input_image=image,
            catalog="test",
            session=session,
        )
        profile.save()


def put_features_in_session(session: Session, features: List[str]) -> None:
    for feature in features:
        feature_type, _ = FetureType.objects.get_or_create(name=feature)
        feature_type.save()
        session.features.add(feature_type)
