import numpy as np
import cv2 as cv
from dataset_utils.image_utils import convert_image_gray_if_bgr


def augment_to_outlined_profile(_image: np.ndarray, _mask: np.ndarray):
    """
    @brief Returns vessel profile where profile is only reprensented by outline
    @param _image input image
    @param _mask corresponding mask image
    """
    # convert to grayscale
    mask = convert_image_gray_if_bgr(_mask)
    image = convert_image_gray_if_bgr(_image)

    cnts, _ = cv.findContours(mask, 1, 2)

    out = np.copy(image)
    out[np.where(mask != 0)] = 255

    color = np.random.randint(10, 50)
    cv.drawContours(out, cnts, -1, color=color,
                    thickness=np.random.randint(1, 3))

    return out
