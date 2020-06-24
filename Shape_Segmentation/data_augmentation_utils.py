import numpy as np
import cv2 as cv


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
                    thickness=np.random.randint(1, 2))

    return out


def convert_image_gray_if_bgr(image: np.ndarray) -> np.ndarray:
    """converts image to grayscale if is bgr"""
    return image.astype(np.uint8) if is_grayscale_image(image) else bgr2gray(image)


def is_grayscale_image(image: np.ndarray) -> bool:
    """ check if input image is grayscale image """
    return True if (len(image.shape) == 2) else False


def bgr2gray(img: np.ndarray) -> np.ndarray:
    """ Converts bgr input image to 8 bit grayscale image """
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.uint8)
