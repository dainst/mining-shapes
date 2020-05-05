import numpy as np 
import cv2 as cv 
from typing import Tuple


def merge_prediction_and_input(model, filename: str, image_size :Tuple):
    """
    @brief read input image, predict output with model and merge input image and output image together
    @param model keras model
    @filename filename of input image
    @image_size size of the input image. Should match dimensions of the input layer of the model
    """
    img = cv.resize(cv.cvtColor(cv.imread(filename,cv.IMREAD_COLOR), cv.COLOR_BGR2RGB), image_size)
    gray = cv.resize(cv.imread(filename, cv.IMREAD_GRAYSCALE), image_size)
    img_model_shape = np.expand_dims(img, axis=0)

    y = model.predict(img_model_shape, verbose=1)
    prob_to_class = np.argmax(y[0], axis=2)
    return np.concatenate((gray,prob_to_class*255), axis=1)