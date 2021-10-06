import numpy as np
import cv2 as cv
import glob
from pathlib import Path
from tensorflow import keras
from typing import Tuple
# pylint: disable=relative-beyond-top-level
from ..db_utils.featureentry import FeatureEntry


class ResnetFeatureVectors:
    """
    @brief: Data generator to return feature vector of given keras model
    @param images_path: location of source images
    @param model: path to keras Model
    @param image: size to rescale images to
    """

    def __init__(self, images_path: str, model: keras.Model, image_size: Tuple[int, int] = (512, 512)) -> None:
        self._images_path = images_path
        self._model = model
        self._image_size = image_size

    def _resourceId_from_file(self, filename: str) -> str:
        return Path(filename).stem

    def _read_img_rgb(self, path: str) -> np.ndarray:
        return cv.cvtColor(cv.imread(path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)

    def __len__(self):
        return len(glob.glob(f"{self._images_path}/*.png"))

    def __iter__(self) -> FeatureEntry:
        for img_path in glob.glob(f"{self._images_path}/*.png"):
            img = cv.resize(self._read_img_rgb(img_path), self._image_size)
            resource_id = self._resourceId_from_file(img_path)
            feature_vec = self._model.predict(
                img[np.newaxis, ...], verbose=False)
            yield FeatureEntry(resource_id, feature_vec.flatten().tolist())
