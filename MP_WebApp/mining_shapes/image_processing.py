from tensorflow import keras
import sys
import os
from django.conf import settings
import cv2 as cv
from typing import Tuple
import numpy as np

# pylint: disable=import-error
sys.path.append(os.path.abspath('/home/Code'))
from mining_pages_utils.tensorflow_utils import load_segmentation_model, postprocess_image, is_img_black  # noqa: E402


def load_seg_model(name: str) -> keras.models.Model:
    modelpath = os.path.join(settings.MODELS_DIR, name)
    return load_segmentation_model(modelpath)


def read_image(url: str, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    img = cv.imread(str(settings.BASE_DIR) + url, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, size)
    return img


def predict_seg_image(model: keras.models.Model, image: np.ndarray):
    seg_img = model.predict(image[np.newaxis, ...])
    seg_img = (np.argmax(seg_img[0], axis=2) * 255).astype(np.uint8)
    return seg_img if is_img_black(seg_img) else postprocess_image(seg_img, (512, 512), (512, 512))
