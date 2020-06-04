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
    @param image_size Image size given in (height, width)
    @param batch_size batch size 
    @param shuffle shuffle trainings data
    @param file_types supported image filetypes
    @param sigma standard dev of generated Gaussian heatmaps
    @param augment_data allow data augmentation
    """

    def __init__(
        self,
        image_path: str,
        points_xml_path: str,
        image_size: Tuple = (256, 256),
        batch_size: int = 32,
        shuffle: bool = True,
        file_types: tuple = ("jpg", "png"),
        sigma: int = 2,
        augment_data: bool = True,
    ):
        assert (
            image_size[0] % 32 == 0 and image_size[1] % 32 == 0
        ), "non valid image_size size. Choose shape % 32 == 0"

        self._image_size = tuple(image_size)
        self._batch_size = batch_size
        self._points_xml_path = points_xml_path
        self._augment_data = augment_data
        self._sigma = sigma

        self.nr_keypoints = self.get_number_of_keypoints(points_xml_path)
        self._images = self.create_image_list(image_path, file_types, shuffle)
        self._keypoints = self.create_keypoint_dict(points_xml_path)
        self._remove_images_without_keypoints()

        # image generator for data augmentation
        dg_args = dict(featurewise_center=False,
                       samplewise_center=False,
                       rotation_range=4,
                       horizontal_flip=True,
                       vertical_flip=False,
                       fill_mode='reflect',
                       data_format='channels_last')
        self._image_aug = keras.preprocessing.image.ImageDataGenerator(
            **dg_args)
        self._keymap_aug = keras.preprocessing.image.ImageDataGenerator(
            **dg_args)

    @staticmethod
    def get_number_of_keypoints(keypoint_file: str) -> int:
        """
        @brief get number of keypoint classes form CVAT xml file
        @param keypoint_file CVAT xml keypoint file
        """
        xmldoc = minidom.parse(keypoint_file)
        labels = len(xmldoc.getElementsByTagName('label'))
        return labels

    @staticmethod
    def create_image_list(image_path: str, file_types: Tuple, shuffle: bool) -> List:
        """
        @brief  Creates a list of all files in given image_path and suffles it.
        @param file_types excepted file types
        @param shuffle shuffle input data
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

    @classmethod
    def create_keypoint_dict(cls, keypoint_file: str) -> Dict:
        """
        Read cvat xml file and creates dictionary with keypoints
        @param keypoint_file CVAT xml keypoint file
        """

        def str_coord_to_namedtuple(coord: str):
            x, y, *_ = coord.replace(";", ",").split(",")
            return Point(float(x), float(y))

        xmldoc = minidom.parse(keypoint_file)
        imagelist = xmldoc.getElementsByTagName("image")
        out = {}
        for image in imagelist:
            temp = {}
            entry = image.attributes["name"].value
            height = int(image.attributes["height"].value)
            width = int(image.attributes["width"].value)
            for point in image.getElementsByTagName("points"):
                temp[point.attributes["label"].value] = DataGenerator.normalize_coords(
                    height,
                    width,
                    str_coord_to_namedtuple(point.attributes["points"].value))

            out[entry] = temp

        return out

    @staticmethod
    def normalize_coords(height: int, width: int, coords: Point):
        """
        @brief normalize coordinates to range [0,1]
        @height image height
        @width image width
        @coords 2d coordinates of pixel
        """
        return Point(x=coords.x / width, y=coords.y / height)

    def _remove_images_without_keypoints(self):
        """
        @brief removes images from image list which dont have keypoints associated
        """
        self._images = [
            image
            for image in self._images
            if os.path.basename(image) in self._keypoints.keys()
        ]

    def _get_keypoint_image_size(self):
        """
        @brief compute size of the keypoint map output image
        """
        im_height, im_width = self._image_size
        map_height = 64
        map_width = map_height * (im_width//im_height)
        return map_height, map_width

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
            idx * self._batch_size: (idx + 1) * self._batch_size
        ]

        images = [self._read_and_resize_image(
            file_name) for file_name in batch_images]
        heatmaps = self._create_heatmaps(batch_images)

        if self._augment_data:
            return self._augment_batch(np.array(images), heatmaps)
        else:
            return np.array(images), heatmaps

    def _create_heatmaps(self, batch_images: List) -> np.ndarray:
        """
        @brief generate heatmaps
        """
        height, width = self._get_keypoint_image_size()
        out = np.zeros((len(batch_images), height, width, self.nr_keypoints))

        for img_i, img_name in enumerate(batch_images):
            keypoint_coords = self._keypoints.get(
                os.path.basename(img_name), (0, 0))
            coords = np.zeros((self.nr_keypoints, 2), dtype=np.int)
            for i, (x, y) in enumerate(keypoint_coords.values()):
                coords[i] = int(x*width), int(y*height)
            out[img_i] = generate_heatmap(
                height, width, coords, sigma=self._sigma)

        return out

    def _augment_batch(self, images: np.ndarray, keymaps: np.ndarray, seed=None) -> Tuple:
        """
        @brief apply data augmentation to masks and input images
        @param images list of images
        @param mask list of masks.
        @param seed random seed to apply same transformation to images and masks
        """
        # keep the seeds synchronized otherwise the augmentation to the images is different from the masks
        np.random.seed(
            seed if seed is not None else np.random.choice(range(9999)))
        seed = np.random.choice(range(9999))

        g_image = self._image_aug.flow(images,
                                       batch_size=images.shape[0],
                                       seed=seed,
                                       shuffle=True)
        g_keymaps = self._keymap_aug.flow(keymaps,
                                          batch_size=keymaps.shape[0],
                                          seed=seed,
                                          shuffle=True)

        return next(g_image), next(g_keymaps)

    def _read_and_resize_image(self, file_name: str) -> np.ndarray:
        """
        @brief reads and resizes an image
        """
        height, width = self._image_size
        return cv.resize(
            cv.cvtColor(cv.imread(file_name, cv.IMREAD_COLOR),
                        cv.COLOR_BGR2RGB),
            (width, height),
        )
