import numpy as np
from typing import Tuple

def gaussian_k(x0,y0,sigma, width, height):
    """ 
    @brief Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float) ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))


def generate_heatmap(height:int, width:int ,landmarks:np.ndarray,sigma=2):
    """ 
    @brief Generate a full Gaussian heatmap for every landmark in an array
    @param height The height of Heat Map (the height of target output)
    @param width The width  of Heat Map (the width of target output)
    @param landmarks [(x1,y1),(x2,y2)...] containing landmarks
    @param sigma: sigma of Gaussian
    """
    Nlandmarks = len(landmarks)
    hm = np.zeros((height, width, Nlandmarks), dtype = np.float32)
    for i in range(Nlandmarks):
        hm[:,:,i] = gaussian_k(landmarks[i][0],
                                landmarks[i][1],
                                sigma,height=height, width=width)
    return hm


def render_heatmap(heatmap:np.ndarray):
    """
    @brief Sums all heatmap points together in one image
    """
    return np.sum(heatmap,axis=2)

def render_heatmap_in_image(image:np.ndarray,heatmap:np.ndarray, color:Tuple=(0,255,0)):
    """
    @brief Merge source image with heatmap in one images
    @param image 3 channel image
    @param heatmap heatmap array
    @color color in which the heatmaps will be rendered
    """
    assert image.shape[0] == heatmap.shape[0] and image.shape[1] == heatmap.shape[1],\
            "Shape from image and heatmap dont match"
    points = render_heatmap(heatmap)
    img = np.zeros_like(image)
    img[np.where(points >0.4)] = color

    return (image*0.2 + img *0.8)/255
