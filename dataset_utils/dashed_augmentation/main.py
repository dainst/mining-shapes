import numpy as np
import cv2 as cv
from glob import glob
from dataset_utils.image_utils import convert_image_gray_if_bgr


class DashedAugmentation:
    def __init__(self) -> None:
        self.patterns = [cv.imread(img, cv.IMREAD_GRAYSCALE) for img in glob(
            "/home/Code/dataset_utils/dashed_augmentation/patterns/*")]

    def generate_random_dashed_array(self, height, width):
        """
        @brief Return random pattern
        @param height height of output image
        @param width width of output image
        """
        pattern = self.patterns[np.random.randint(0, len(self.patterns))]
        return pattern[:height, :width]

    def augment_to_dashed_profile(self, _image: np.ndarray, _mask: np.ndarray) -> np.ndarray:
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
            dashed_mask *= self.generate_random_dashed_array(h, w)

            # morph original image with morphed mask
            out[y:y+h, x:x+w] += dashed_mask
            color = np.random.randint(10, 50)
            cv.drawContours(out, cnts, -1, color=color,
                            thickness=np.random.randint(1, 3))

        return out
