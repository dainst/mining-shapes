from django.core.files.uploadedfile import InMemoryUploadedFile
from typing import List, NamedTuple
import cv2 as cv
import numpy as np
from django.core.files.base import ContentFile
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

from .models import Session, VesselProfile, FetureType

# pylint: disable=no-member


class PointDict(NamedTuple):
    x: str
    y: str


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
    vesselprofile.segmented_image.save(
        vesselprofile.filename, numpy_img_to_content_file(image), save=True)
    vesselprofile.save()


def edit_seg_image_from_vesselprofile(polygon: List[PointDict], vesselprofile: VesselProfile) -> None:
    if not polygon:
        return
    polygon_list = [[point['x'], point['y']] for point in polygon]

    filepath = vesselprofile.segmented_image.path
    orig_image = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    image = np.zeros_like(orig_image, dtype=np.uint8)
    cv.fillPoly(image, pts=np.array([polygon_list]), color=255)

    vesselprofile.segmented_image.save(
        vesselprofile.filename, numpy_img_to_content_file(image), save=True)
    vesselprofile.save()


def numpy_img_to_content_file(image: np.array) -> ContentFile:
    frame_jpg = cv.imencode('.jpg', image)
    return ContentFile(frame_jpg[1])


def profile_pagination(request, post_list) -> Paginator:
    """ Return pagination of Vesselprofile model """
    page = request.GET.get('page', 1)
    paginator = Paginator(post_list, 20)
    try:
        profiles = paginator.page(page)
    except PageNotAnInteger:
        profiles = paginator.page(1)
    except EmptyPage:
        profiles = paginator.page(paginator.num_pages)
    return profiles
