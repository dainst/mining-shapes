import glob
import os
import random
import numpy as np
from shutil import copyfile
import argparse
import sys

sys.path.append(os.path.abspath('../'))
from dataset_utils.data_split.main import copy_images_to_path, make_output_directory, create_image_list  # noqa: E402

"""
Script to split mask dataset exported from cvat. 
"""
TEST_DIR = 'test'
TEST_A_DIR = 'testannot'
VAL_DIR = 'val'
VAL_A_DIR = 'valannot'
TRAIN_DIR = 'train'
TRAIN_A_DIR = 'trainannot'


def split_data(args):
    """
    @brief splits the mask dataset into train/test/val dataset
    @param args command line arguments
    """
    annotations_p = os.path.join(args.mask, 'SegmentationClass/')
    images_p = args.images
    output_p = args.output

    make_output_directory(
        output_p, [VAL_DIR, VAL_A_DIR, TRAIN_DIR, TRAIN_A_DIR, TEST_DIR, TEST_A_DIR])

    masks = glob.glob(annotations_p+'*.png')
    images = create_image_list(images_p, masks)

    assert len(masks) == len(images)
    assert len(images) > 1500, "Not enough data to trian network"
    # sort
    masks.sort()
    images.sort()

    # shuffle
    random.shuffle(images, lambda: .5)
    random.shuffle(masks, lambda: .5)

    # split data
    test = images[:300]
    test_a = masks[:300]
    assert len(test) == len(test_a)

    val = images[300:500]
    val_a = masks[300:500]
    assert len(val) == len(val_a)

    train = images[500:]
    train_a = masks[500:]
    assert len(train) == len(train_a)

    # assert correct splitting
    assert len(np.unique([os.path.basename(train[i]) == os.path.basename(
        train_a[i]) for i in range(len(val_a))])) == 1
    assert len(np.unique([os.path.basename(test[i]) == os.path.basename(
        test_a[i]) for i in range(len(val_a))])) == 1
    assert len(np.unique([os.path.basename(val[i]) == os.path.basename(
        val_a[i]) for i in range(len(val_a))])) == 1

    # copy images to output directory
    copy_images_to_path(val, os.path.join(output_p, VAL_DIR))
    copy_images_to_path(test, os.path.join(output_p, TEST_DIR))
    copy_images_to_path(train, os.path.join(output_p, TRAIN_DIR))

    copy_images_to_path(val_a, os.path.join(output_p, VAL_A_DIR))
    copy_images_to_path(train_a, os.path.join(output_p, TRAIN_A_DIR))
    copy_images_to_path(test_a, os.path.join(output_p, TEST_A_DIR))

    # copy labelmap
    copyfile(os.path.join(args.mask, 'labelmap.txt'),
             os.path.join(output_p, 'labelmap.txt'))

    print(f'Copied dataset to {output_p}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Split cvat mask data set in test/val/train datasets")
    parser.add_argument("--images", type=str,
                        help="directory in which source images are located", required=True)
    parser.add_argument("--mask", type=str,
                        help="directory in which exported mask dataset is located", required=True)
    parser.add_argument("--output", type=str,
                        help="directory in which splitted dataset will be saved")

    _args = parser.parse_args()
    split_data(_args)
