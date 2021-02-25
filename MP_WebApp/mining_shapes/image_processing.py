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


def read_image(url: str) -> np.ndarray:
    img = cv.imread(str(settings.BASE_DIR) + url, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def predict_seg_image(model: keras.models.Model, image: np.ndarray, resize_img_size: Tuple[int, int] = (512, 512)):
    height_orig, width_orig, *_ = image.shape
    img = cv.resize(image, resize_img_size)
    seg_img = model.predict(img[np.newaxis, ...])
    seg_img = (np.argmax(seg_img[0], axis=2) * 255).astype(np.uint8)
    if not is_img_black(seg_img):
        seg_img = postprocess_image(seg_img, resize_img_size, image.shape[:2])
    return cv.resize(seg_img,  (width_orig, height_orig))
