"""
Helper function for Deep Learning based shape similarity computation
"""
import numpy as np
import cv2 as cv
from tensorflow import keras
from typing import Tuple, List, NamedTuple
import glob
from pathlib import Path
from tqdm import tqdm


class FeatureEntry(NamedTuple):
    id: str
    feature_vec: List[int]


class ResnetFeatureVectors:
    """
    @brief: Data generator to return feature vector of given keras model
    @param images_path: location of source images
    @param model: path to keras Model
    @param image: size to rescale images to
    """

    def __init__(self, images_path: str, model: keras.Model, image_size) -> None:
        self._images_path = images_path
        self._model = model
        self._image_size = image_size

    def _resourceId_from_file(self, filename: str) -> str:
        return Path(filename).stem

    def _read_img_rgb(self, path: str) -> np.ndarray:
        return cv.cvtColor(cv.imread(path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)

    def __len__(self):
        return len(glob.glob(f"{self._images_path}/*.jpg"))

    def __iter__(self) -> FeatureEntry:
        for img_path in glob.glob(f"{self._images_path}/*.jpg"):
            img = cv.resize(self._read_img_rgb(img_path), self._image_size)
            resource_id = self._resourceId_from_file(img_path)
            feature_vec = self._model.predict(
                img[np.newaxis, ...], verbose=False)
            yield FeatureEntry(resource_id, list(feature_vec.flatten()))


def post_featurevector_to_db(path: str, input_shape: Tuple[int, int, int] = (512, 512, 3)) -> None:
    """
    @brief: Read images in path, compute resnet feature vectors and return as json
    @param path: Location of binary images.
    @input_shape shape to scale images
    """

    model = keras.applications.resnet50.ResNet50(
        input_shape=input_shape, include_top=False, pooling='avg', weights='imagenet')
    f = open("hello.txt", "w")

    vector_generator = ResnetFeatureVectors(path, model, input_shape[:2])
    with tqdm(total=len(vector_generator)) as pbar:
        for i in vector_generator:
            f.write(f"{i.id}, {i.feature_vec[:13]}")
            pbar.update(1)

    f.close()
