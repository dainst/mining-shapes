import numpy as np
import cv2 as cv


def convert_image_gray_if_bgr(image: np.ndarray) -> np.ndarray:
    """converts image to grayscale if is bgr"""
    return image.astype(np.uint8) if is_grayscale_image(image) else bgr2gray(image)


def is_grayscale_image(image: np.ndarray) -> bool:
    """ check if input image is grayscale image """
    return True if (len(image.shape) == 2) else False


def bgr2gray(img: np.ndarray) -> np.ndarray:
    """ Converts bgr input image to 8 bit grayscale image """
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.uint8)
