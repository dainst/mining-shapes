import numpy as np
import cv2 as cv
from dataset_utils.image_utils import convert_image_gray_if_bgr


def generate_random_dashed_array(height, width):
    """
    @brief Generates a 2d sine wave image
    @param height height of output image
    @param width width of output image
    """

    x = np.linspace(-np.pi, np.pi, width)
    rand_per = np.random.randint(4, 8)
    sine1D = 128.0 + (127.0 * np.sin(x * rand_per))
    sine1D = np.uint8(sine1D)
    sine2D = np.ndarray((height, width), dtype=np.uint8)
    rand_rot = np.random.randint(0, 4)
    for i in range(height):
        sine2D[i] = np.roll(sine1D, -i*rand_rot)

    return sine2D


def augment_to_dashed_profile(_image: np.ndarray, _mask: np.ndarray) -> np.ndarray:
    """
    @brief Fills vessel profile with random dashed lines
    @param _image input image
    @param _mask corresponding mask image
    """

    mask = convert_image_gray_if_bgr(_mask)
    image = convert_image_gray_if_bgr(_image)

    # find rectangle around mask
    cnts, _ = cv.findContours(mask, 1, 2)
    out = np.copy(image)
    out[np.where(mask != 0)] = 0
    for cnt in cnts:
        x, y, w, h = cv.boundingRect(cnt)

        # create dased mask
        roi = mask[y:y+h, x:x+w]
        dashed_mask = np.zeros_like(roi)
        dashed_mask[np.where(roi != 0)] = 1
        dashed_mask *= generate_random_dashed_array(h, w)

        # morph original image with morphed mask
        out[y:y+h, x:x+w] += dashed_mask

    return out
