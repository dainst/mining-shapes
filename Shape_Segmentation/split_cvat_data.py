import glob
import os 
import random
import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 
from typing import List
from shutil import copyfile
import argparse


"""
Script to split mask dataset exported from cvat. 
"""
TEST_DIR = 'test'
TEST_A_DIR = 'testannot'
VAL_DIR = 'val'
VAL_A_DIR = 'valannot'
TRAIN_DIR = 'train'
TRAIN_A_DIR = 'trainannot'

def create_image_list(base_path, basenames:List):
    """
    @brief creats valid path to the input images
    @param base_path path in which the source images are located
    @param list of image filenames
    """
    z = []
    for img in basenames:
        base = os.path.basename(img)
        z.append(os.path.join(base_path, base))
    return z 

def copy_images_to_path(images:List, dst_path:str):
    """
    @brief copy list of images to destination path
    @param images list of filenames
    @param dst_path destination path
    """
    for image in images:
        u = image
        copyfile(u, os.path.join(dst_path,os.path.basename(image)))

def make_output_directory(dir_name:str):
    """
    @brief makes directories for splitted data
    @param dir_name name of base directory
            dir_name -- -val
                        -valannot
                        -train
                        -traianno
                        -test
                        -testannot      
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        os.makedirs(os.path.join(dir_name,VAL_DIR))
        os.makedirs(os.path.join(dir_name,VAL_A_DIR))
        os.makedirs(os.path.join(dir_name,TRAIN_DIR))
        os.makedirs(os.path.join(dir_name,TRAIN_A_DIR))
        os.makedirs(os.path.join(dir_name,TEST_DIR))
        os.makedirs(os.path.join(dir_name,TEST_A_DIR))


def split_data(args):
    """
    @brief splits the mask dataset into train/test/val dataset
    @param args command line arguments
    """
    annotations_p = os.path.join(args.mask,'SegmentationClass/')
    images_p = args.images
    output_p = args.output

    make_output_directory(output_p)

    masks = glob.glob(annotations_p+'*.png')
    images = create_image_list(images_p, masks)

    assert len(masks) == len(images)
    assert len(images) > 1500, "Not enough data to trian network"
    #sort
    masks.sort()
    images.sort()

    #shuffle
    random.shuffle(images, lambda: .5)
    random.shuffle(masks, lambda: .5)

    #split data
    test = images[:300]
    test_a = masks[:300]
    assert len(test) == len(test_a)

    val = images[300:500]
    val_a = masks[300:500]
    assert len(val) == len(val_a)

    train = images[500:]
    train_a = masks[500:]
    assert len(train) == len(train_a)

    #assert correct splitting
    assert len(np.unique([os.path.basename(train[i]) == os.path.basename(train_a[i]) for i in range(len(val_a))])) == 1 
    assert len(np.unique([os.path.basename(test[i]) == os.path.basename(test_a[i]) for i in range(len(val_a))])) == 1 
    assert len(np.unique([os.path.basename(val[i]) == os.path.basename(val_a[i]) for i in range(len(val_a))])) == 1 

    #copy images to output directory
    copy_images_to_path(val, os.path.join(output_p,VAL_DIR))
    copy_images_to_path(test, os.path.join(output_p,TEST_DIR))
    copy_images_to_path(train, os.path.join(output_p,TRAIN_DIR))


    copy_images_to_path(val_a, os.path.join(output_p,VAL_A_DIR))
    copy_images_to_path(train_a, os.path.join(output_p,TRAIN_A_DIR))
    copy_images_to_path(test_a, os.path.join(output_p,TEST_A_DIR))

    #copy labelmap
    copyfile(os.path.join(args.mask,'labelmap.txt'),os.path.join(output_p,'labelmap.txt'))

    print(f'Copied dataset to {output_p}')

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(
        description="Split cvat mask data set in test/val/train datasets")
    parser.add_argument("--images", type=str,
                        help="directory in which source images are located",required=True)
    parser.add_argument("--mask", type=str,
                        help="directory in which exported mask dataset is located", required=True)
    parser.add_argument("--output", type=str,
                        help="directory in which splitted dataset will be saved")
    
    
    args = parser.parse_args()
    split_data(args)
