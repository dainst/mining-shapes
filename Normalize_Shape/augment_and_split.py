"""
Script to augment and split exported point dataset.
Splits image dataset into train, test, val
Applies data augmentation for train dataset. For this it requires valid binary segmentation masks!
"""
import os
import cv2 as cv
import sys
import argparse
import random
import glob
from xml.dom import minidom
from tqdm import tqdm


sys.path.append(os.path.abspath('../'))
from dataset_utils.dashed_augmentation.main import augment_to_dashed_profile  # noqa: E402
from dataset_utils.outlined_augmentation.main import augment_to_outlined_profile  # noqa: E402
from dataset_utils.data_split.main import copy_images_to_path, make_output_directory, create_image_list  # noqa: E402

TEST_DIR = 'test'
VAL_DIR = 'val'
TRAIN_DIR = 'train'


def split_data(_args) -> str:
    """
    @brief splits the mask dataset into train/test/val dataset
    @param args command line arguments
    """
    make_output_directory(_args.output, [VAL_DIR, TRAIN_DIR, TEST_DIR])

    images = create_image_list(
        _args.image_path, glob.glob(_args.image_path+'*.png'))

    assert len(images) > 1500, "Not enough data to trian network"
    # sort
    images.sort()

    # shuffle
    random.shuffle(images, lambda: .5)

    # split data
    test = images[:300]
    val = images[300:500]
    train = images[500:]

    # copy images to output directory
    copy_images_to_path(val, os.path.join(_args.output, VAL_DIR))
    copy_images_to_path(test, os.path.join(_args.output, TEST_DIR))
    copy_images_to_path(train, os.path.join(_args.output, TRAIN_DIR))

    return os.path.join(_args.output, TRAIN_DIR)


def is_image_valid(image_name: str, mask_p: str) -> bool:
    if 'ZenonID' in image_name and image_name in os.listdir(mask_p):
        return True
    else:
        return False


def augment_image(image_, mask_, name):
    """ augment image and mask """
    out = augment_to_outlined_profile(image_, mask_)
    dash = augment_to_dashed_profile(image_, mask_)
    return 'dashed_'+name, dash, 'outlined_'+name, out


def add_to_xml(basename: str, new_name: str, xml_file) -> None:
    images = xml_file.getElementsByTagName('image')
    for img in images:
        if img.attributes['name'].value == basename:  # found image
            xml_file.childNodes[0].appendChild(
                xml_file.createTextNode('\n'))  # add line break
            new_image = create_new_image_node(new_name, img, xml_file)
            for point in img.getElementsByTagName('points'):
                new_image.appendChild(create_new_point_node(point, xml_file))
                new_image.appendChild(
                    xml_file.createTextNode('\n  '))  # add line break

            xml_file.childNodes[0].appendChild(new_image)
            return None


def create_new_image_node(new_name, image_node, xml_file):
    image_ = xml_file.createElement("image")
    image_.setAttribute("id", image_node.attributes['id'].value)
    image_.setAttribute("name", new_name)
    image_.setAttribute("width", image_node.attributes['width'].value)
    image_.setAttribute("height", image_node.attributes['height'].value)
    image_.appendChild(xml_file.createTextNode('\n'))  # add line break
    return image_


def create_new_point_node(point_node, xml_file):
    point = xml_file.createElement("points")
    point.setAttribute("label", point_node.attributes['label'].value)
    point.setAttribute("occluded", point_node.attributes['occluded'].value)
    point.setAttribute("points", point_node.attributes['points'].value)
    point.appendChild(xml_file.createTextNode('\n  '))  # add line break
    return point


def augment_data(points_path: str, aug_points_path: str, images_path: str, masks_path: str):
    """
    @brief: Function to augment and split data for point detection
    @param _args command line arguments
    """

    xmldoc = minidom.parse(points_path)
    print(f'Process {len(os.listdir(images_path))} images')
    with tqdm(total=len(os.listdir(images_path))) as pbar:
        for image_basename in os.listdir(images_path):
            if is_image_valid(image_basename, masks_path):
                image = cv.imread(os.path.join(
                    images_path, image_basename), cv.IMREAD_COLOR)
                mask = cv.imread(os.path.join(
                    masks_path, image_basename), cv.IMREAD_COLOR)
                dashed_name, dashed, out_name, outlined = augment_image(
                    image, mask, image_basename)
                add_to_xml(image_basename, dashed_name, xmldoc)
                add_to_xml(image_basename, out_name, xmldoc)
                cv.imwrite(os.path.join(images_path, dashed_name), dashed)
                cv.imwrite(os.path.join(images_path, out_name), outlined)
            pbar.update(1)

    xmldoc.writexml(open(aug_points_path, 'w'))


def split_augment_data(_args):
    train_path = split_data(_args)
    augment_data(_args.points_path,
                 args.point_aug_path,
                 train_path,
                 args.mask_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split point dataset and augment images if corresponding mask exists")
    parser.add_argument("--points_path", type=str,
                        help="Path of point file. File format should be in CVAT xml format", required=True)
    parser.add_argument("--point_aug_path", type=str,
                        help="Filename for new generated CVAT xml file", required=True)
    parser.add_argument("--image_path", type=str,
                        help="Directory in which source images are located", required=True)
    parser.add_argument("--mask_path", type=str,
                        help="Directory in which corresponding masks are located", required=True)
    parser.add_argument("--output", type=str,
                        help="directory in which splitted dataset will be saved", required=True)

    args = parser.parse_args()
    split_augment_data(args)
