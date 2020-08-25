import numpy as np
import tensorflow as tf
import os
import math
import cv2 as cv
from random import Random

from typing import Tuple, List
from tensorflow import keras
from data_augmentation_utils import augment_to_dashed_profile, augment_to_outlined_profile


class DataGenerator(tf.keras.utils.Sequence):
    """
    Class to read and to preprocess image data for Segmentation Model
    @param image_path path to image data
    @param mask_path path to segmentations masks. \n Should be one channels image where the pixel value represents the class label
    @param classes number of classes 
    @image_size each image and each mask will be scaled to the given size
    @batch_size batch size
    @param shuffle shuffle data or not
    @param file_types tuple of excepted file types
    @param scale image intenseties
    @param augment_data apply data augmentation to input data
    """

    def __init__(self, image_path: str, mask_path: str, labelmap_path: str, image_size: Tuple = (512, 512), batch_size: int = 10, shuffle: bool = True, file_types: tuple = ('jpg', 'png'), scale: int = 0, augment_data=True):
        self._image_path = image_path
        self._mask_path = mask_path
        self._image_size = tuple(image_size)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._file_types = file_types
        self._scale = scale
        self._augment_data = augment_data
        self._labelmap_path = labelmap_path

        self._images, self._masks = self._create_dataset()
        self._read_labelmap()

        # image generator for data augmentation
        dg_args = dict(featurewise_center=False,
                       samplewise_center=False,
                       rotation_range=10,
                       horizontal_flip=True,
                       vertical_flip=True,
                       fill_mode='nearest',
                       data_format='channels_last')
        self._image_generator = keras.preprocessing.image.ImageDataGenerator(
            **dg_args)
        self._mask_generator = keras.preprocessing.image.ImageDataGenerator(
            **dg_args)

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

        batch_images = self._images[idx *
                                    self._batch_size:(idx + 1) * self._batch_size]
        batch_masks = self._masks[idx *
                                  self._batch_size:(idx + 1) * self._batch_size]

        assert len(batch_images) == len(batch_masks)

        # Read image data
        images = [self._read_and_resize_image(
            file_name) for file_name in batch_images]

        # Read mask data
        mask_images = [self._read_and_resize_image(
            file_name) for file_name in batch_masks]

        if self._augment_data:
            images, mask_images = self._augment_batch(
                np.array(images), np.array(mask_images))

        masks = self._mask_image_to_mask_array(mask_images)

        assert len(images) == len(
            masks), "size of image batch doesnt match size of mask list"

        if self._scale == 0:
            return np.array(images).astype(np.int), np.array(masks)
        else:
            return np.array(images).astype(np.float)/self._scale, np.array(masks)

    def _create_dataset(self) -> List:
        """
        @brief  lists all files in given image_path and mask_path. Shuffles list of filenames and assures
                that masks belong to corresponding images
        """
        x = ["{p}/{f}".format(p=self._image_path, f=file)
             for file in os.listdir(self._image_path) if file.lower().endswith(self._file_types)]

        masks = ["{p}/{f}".format(p=self._mask_path, f=file)
                 for file in os.listdir(self._mask_path) if file.lower().endswith(self._file_types)]

        assert len(x) == len(masks), "Number of images and masks do not match"

        # sort
        x.sort()
        masks.sort()

        # shuffle
        seed = np.random.randint(0, 6000)
        if self._shuffle:
            Random(seed).shuffle(x)
            Random(seed).shuffle(masks)

        assert len(np.unique([os.path.basename(x[i]) == os.path.basename(
            masks[i]) for i in range(len(masks))])) == 1, "Mask basenames not like image basenames"

        return x, masks

    def _read_and_resize_image(self, file_name: str) -> np.ndarray:
        """
        @brief reads and resizes an image
        """
        return cv.resize(cv.cvtColor(cv.imread(file_name, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB), self._image_size)

    def _augment_batch(self, images: List, masks: List, seed=None) -> Tuple:
        """
        @brief apply data augmentation to masks and input images
        @param images list of images
        @param mask list of masks.
        @param seed random seed to apply same transformation to images and masks
        """
        generated_images = np.zeros_like(images, dtype=float)
        for i, (image, mask) in enumerate(zip(images, masks)):
            generated_images[i] = self._aug_img_to_dashed_outline(image, mask)

        # keep the seeds synchronized otherwise the augmentation to the images is different from the masks
        np.random.seed(
            seed if seed is not None else np.random.choice(range(9999)))
        seed = np.random.choice(range(9999))

        g_image = self._image_generator.flow(generated_images,
                                             batch_size=images.shape[0],
                                             seed=seed,
                                             shuffle=True)
        g_mask = self._mask_generator.flow(masks,
                                           batch_size=masks.shape[0],
                                           seed=seed,
                                           shuffle=True)

        return next(g_image), next(g_mask)

    def _aug_img_to_dashed_outline(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        @brief randomly augments image to have only outlined profile or dashed profile filling
        @param image input image
        @param correspondig outline mask
        """
        def gray2rgb(img): return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        rand = np.random.randint(0, 5)
        if rand % 2 == 0:
            return gray2rgb(augment_to_dashed_profile(image, mask))
        elif rand == 3:
            return gray2rgb(augment_to_outlined_profile(image, mask))
        else:
            return image

    def _read_labelmap(self) -> None:
        """
        @brief read labelmap.txt file exported from CVAT
        """
        labelmap = open(self._labelmap_path, 'r')
        lines = labelmap.readlines()
        self._nr_classes = len(lines)-1  # exclude header
        self._classes = {}

        for category in range(self._nr_classes):
            splited_dp = lines[category+1].split(':')
            color = tuple([int(i) for i in splited_dp[1].split(',')])
            self._classes[category] = color

        assert self._nr_classes == len(self._classes)
        labelmap.close()

    def _mask_image_to_mask_array(self, mask_images: np.ndarray) -> np.ndarray:
        """
        @brief convert rgb mask image to numpy array with shape (batch_size, rows, cols, classes)
        """
        masks = []
        for maks_image in mask_images:
            mask = np.zeros(shape=(*(self._image_size), self._nr_classes))
            for category, color in self._classes.items():
                img_temp = np.zeros(shape=self._image_size)
                img_temp[np.where((maks_image == color).all(axis=2))] = 1
                mask[:, :, category] = np.copy(img_temp)
            masks.append(mask)

        return np.array(masks)
