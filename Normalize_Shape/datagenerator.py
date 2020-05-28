import numpy as np
import os
import math
import cv2 as cv
import random
from xml.dom import minidom
from collections import namedtuple
from typing import Tuple, List, Dict
from tensorflow import keras

from heatmap import generate_heatmap

Point = namedtuple("Point", ["x", "y"])


class DataGenerator(keras.utils.Sequence):
    """
    @brief Data generator for point detection network. 
    @return batch of images and corresponding heatmaps
    @param image_path path of image data
    @param points_xml_path path of keypoints dumped as CVAT xml
    @param image_size Image size given in height, width
    @param batch_size batch size 
    @param shuffle shuffle trainings data
    @param nr_keypoints number of different keypoints
    """

    def __init__(
        self,
        image_path: str,
        points_xml_path: str,
        image_size: Tuple = (200, 200),
        batch_size: int = 32,
        shuffle: bool = True,
        file_types: tuple = ("jpg", "png"),
        nr_keypoints: int = 3,
    ):

        self._image_size = tuple(image_size)
        self._batch_size = batch_size
        self._points_xml_path = points_xml_path
        self._nr_keypoints = nr_keypoints

        self._images = self.create_image_list(image_path, file_types, shuffle)
        self._keypoints = self.create_keypoint_dict(points_xml_path)
        self._remove_images_without_keypoints()

    @staticmethod
    def create_image_list(image_path: str, file_types: Tuple, shuffle: bool) -> List:
        """
        @brief  Creates a list of all files in given image_path and suffles it.
        """
        x = [
            os.path.join(image_path, file)
            for file in os.listdir(image_path)
            if file.lower().endswith(file_types)
        ]

        # shuffle
        if shuffle:
            random.shuffle(x)

        return x

    @staticmethod
    def create_keypoint_dict(keypoint_file: str) -> Dict:
        """
        Read cvat xml file and creates dictionary with keypoints
        """

        def str_coord_to_namedtuple(coord: str):
            x, y, *rest = coord.replace(";", ",").split(",")
            # if rest:
            #    continue
            return Point(int(float(x)), int(float(y)))

        xmldoc = minidom.parse(keypoint_file)
        imagelist = xmldoc.getElementsByTagName("image")
        out = {}
        for image in imagelist:
            temp = {}
            entry = image.attributes["name"].value
            for point in image.getElementsByTagName("points"):
                temp[point.attributes["label"].value] = str_coord_to_namedtuple(
                    point.attributes["points"].value
                )
            out[entry] = temp

        return out

    def _remove_images_without_keypoints(self):
        """
        @brief removes images from image list which dont have keypoints associated
        """
        self._images = [
            image
            for image in self._images
            if os.path.basename(image) in self._keypoints.keys()
        ]
        assert len(self._images) == len(self._keypoints)

    def __len__(self):
        """
        @brief Returns number of batches
        @remark has to be implemented
        """
        return math.ceil(len(self._images) / self._batch_size)

    def __getitem__(self, idx):
        """
        @brief  reads batch of images and preprocesses segmentation masks images.
                Furthermore, scales images. 
        """

        batch_images = self._images[
            idx * self._batch_size : (idx + 1) * self._batch_size
        ]

        images = [self._read_and_resize_image(file_name) for file_name in batch_images]
        heatmaps = self._create_heatmaps(batch_images)

        return np.array(images), heatmaps


    def _create_heatmaps(self, batch_images: List) -> np.ndarray:
        """
        @brief generate heatmaps
        """
        out = np.zeros((len(batch_images), *(self._image_size), self._nr_keypoints))
        height, width = self._image_size

        for img_i, img_name in enumerate(batch_images):
            keypoint_coords = self._keypoints[os.path.basename(img_name)]
            coords = np.zeros((3, 2), dtype=np.int)
            for i, (x, y) in enumerate(keypoint_coords.values()):
                coords[i] = x, y
            out[img_i] = generate_heatmap(height, width, coords, sigma=2)

        return out
    def _read_and_resize_image(self, file_name: str) -> np.ndarray:
        """
        @brief reads and resizes an image
        """
        height, width = self._image_size
        return cv.resize(
            cv.cvtColor(cv.imread(file_name, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB),
            (width, height),
        )
