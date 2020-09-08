from typing import List
from shutil import copyfile
import os


def create_image_list(base_path, basenames: List):
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


def copy_images_to_path(images: List, dst_path: str):
    """
    @brief copy list of images to destination path
    @param images list of filenames
    @param dst_path destination path
    """
    for image in images:
        u = image
        copyfile(u, os.path.join(dst_path, os.path.basename(image)))


def make_output_directory(dir_name: str, subdirs: List[str]):
    """
    @brief makes directories for splitted data
    @param dir_name name of base directory
            dir_name -- -subdir1
                        -subdir2
                        -...
                        -subdirx

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        _ = [os.makedirs(os.path.join(dir_name, i)) for i in subdirs]
