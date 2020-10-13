import sys
import tensorflow as tf
#from tensorflow import keras
from tqdm import tqdm
import io
import os
#import math
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List
from object_detection.utils import dataset_util, ops
import segmentation_models as sm

# pylint: disable=import-error
sys.path.append(os.path.abspath('/home/Code/Normalize_Shape'))
from point_detector import PointDetector  # noqa: E402


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in group.object.iterrows():
        box = row['detection_boxes']
        ymin, xmin, ymax, xmax = box
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(row['detection_classesname'].encode('utf8'))
        classes.append(int(row['detection_classes']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_figid(group, path):

    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in group.object.iterrows():

        filename = group.filename.encode('utf8')
        box = row['figid_detection_boxes']
        ymin, xmin, ymax, xmax = box
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(
            str(row['figid_detection_classesname']).encode('utf8'))
        classes.append(int(row['figid_detection_classes']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def run_inference_for_page_series(series: pd.Series, tensor_dict: dict, session: tf.compat.v1.Session) -> pd.Series:
    """
    @brief runs object detection network on page image and appends detection result to series
    @param series pandas Series with image data located in column 'page_imgnp'
    @param tensor_dict
    @param session tensorflow session
    """
    image = series['page_imgnp']
    output_dict = run_inference(tensor_dict, image, session)

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    series['page_detections'] = output_dict
    return series


def run_inference_for_figure_series(series: pd.Series, tensor_dict, session: tf.compat.v1.Session) -> pd.Series:
    """
    @brief runs object detection network on figure image and appends detection result to series
    @param series pandas Series with image data located in column 'page_imgnp'
    @param tensor_dict
    @param session tensorflow session
    """
    image = series['figure_imgnp']
    output_dict = run_inference(tensor_dict, image, session)

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['figid_num_detections'] = int(output_dict['num_detections'][0])
    del output_dict['num_detections']
    output_dict['figid_detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    del output_dict['detection_classes']
    output_dict['figid_detection_boxes'] = output_dict['detection_boxes'][0]
    del output_dict['detection_boxes']
    output_dict['figid_detection_scores'] = output_dict['detection_scores'][0]
    del output_dict['detection_scores']
    if 'detection_masks' in output_dict:
        output_dict['figid_detection_masks'] = output_dict['detection_masks'][0]
        del output_dict['detection_masks']
    series['figid_detections'] = output_dict
    return series


def run_inference(tensor_dict: dict, image: np.ndarray, session: tf.compat.v1.Session):
    """
    @brief run object detection network on input image
    @param tensor_dict
    @param session tensorflow session
    """
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(
            tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                   real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                   real_num_detection, -1, -1])
        detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = session.run(tensor_dict,
                              feed_dict={image_tensor: np.expand_dims(image, 0)})

    return output_dict


def run_vesselprofile_segmentation(vesselpath: str, segmentpath: str, modelpath: str, img_size: Tuple[int, int] = (512, 512)) -> None:
    """
    @brief performs segmentation of vesselprofile images
    @param vesselpath directory of vesselprofile images
    @param path to store segmented images
    @param modelpath location of saved model weights. Weights should be stored in .h5 format
    """
    vessel_image_list = os.listdir(vesselpath)

    # load pretrained model
    seg_model = sm.Unet('resnet34', encoder_weights='imagenet', input_shape=(
        None, None, 3), classes=2, encoder_freeze=True)
    seg_model.load_weights(modelpath)

    # predict segmentations and store to segmentpath
    prog_bar = tqdm(total=len(vessel_image_list)-1)
    #keras.utils.Progbar(len(vessel_image_list)-1, width=30, verbose=1, interval=0.5, unit_name='step')

    for img_name in vessel_image_list:
        image = cv2.imread(os.path.join(
            vesselpath, img_name), cv2.IMREAD_COLOR)
        height_orig, width_orig, *_ = image.shape
        image = cv2.resize(image, img_size)
        seg_img = seg_model.predict(image[np.newaxis, ...])
        seg_img = (np.argmax(seg_img[0], axis=2) * 255).astype(np.uint8)

        if is_img_black(seg_img):
            cv2.imwrite(os.path.join(segmentpath, f"trash_{img_name}"), cv2.resize(
                seg_img, (width_orig, height_orig)))
        else:
            seg_img = postprocess_image(seg_img, img_size, image.shape[:2],)
            cv2.imwrite(os.path.join(segmentpath, img_name), seg_img)

        prog_bar.update(1)
    prog_bar.close()


def is_img_black(img: np.ndarray):
    """ Check if input grayscale image is black """
    return True if not np.any(img) else False


def postprocess_image(img: np.ndarray, seg_shape: Tuple[int, int], orig_shape: Tuple[int, int]) -> np.ndarray:
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=len, reverse=True)

    resize_cnt = []
    # only select 2 largest segments
    for contour in contours[:2]:
        norm_cnt = scale_contour(contour, 1/np.array(seg_shape))
        resize_cnt_temp = scale_contour(norm_cnt, orig_shape)
        resize_cnt.append(resize_cnt_temp.astype(np.int))

    out_img = np.zeros(orig_shape, dtype=np.uint8)
    cv2.fillPoly(out_img, pts=resize_cnt, color=255)
    return out_img


def scale_contour(cnt, shape):
    out_cnt = np.copy(cnt).astype(np.float)
    out_cnt[:, 0, 0] *= shape[1]
    out_cnt[:, 0, 1] *= shape[0]
    return out_cnt


def run_point_detection(vesselpath: str, pointpath: str, modelpath: str):
    vessel_image_list = os.listdir(vesselpath)

    # setup model
    print("Load model")
    img_size = (256, 256)
    model = PointDetector(input_shape=(*img_size, 3))
    model.load_weights(modelpath)
    df = pd.DataFrame(columns=['img_name', 'Top_Rot', 'Base_Rot',
                               'Down_Rot', 'Up_Rot', 'Base_Side', 'Top_Side'])

    # process images
    print("Process images")
    prog_bar = tqdm(total=len(vessel_image_list)-1)
    for img_name in vessel_image_list:
        image = cv2.imread(os.path.join(
            vesselpath, img_name), cv2.IMREAD_COLOR)
        height_orig, width_orig, *_ = image.shape
        image = cv2.resize(image, img_size)
        y = model.predict_img(image)
        norm_coords = heatmaps_to_array(y[0])
        scaled_coords = scale_and_format_coords(
            norm_coords, height_orig, width_orig)
        df = df.append({'img_name': img_name,
                        'Top_Rot': scaled_coords[0], 'Base_Rot': scaled_coords[1],
                        'Down_Rot': scaled_coords[2], 'Up_Rot': scaled_coords[3],
                        'Base_Side': scaled_coords[4], 'Top_Side': scaled_coords[5]}, ignore_index=True)

        prog_bar.update(1)

    prog_bar.close()
    df.to_csv(os.path.join(pointpath, "points.csv"), index=False)


def scale_and_format_coords(norm_coords: List[Tuple[int, int]], height: int, width: int):
    """
    Scale normalized coordinates and set 0 values to no_value string
    """
    scaled_coord = []
    for h, w in norm_coords:
        if h == 0 and w == 0:
            scaled_coord.append("no_value")
        else:
            scaled_coord.append((int(h*height), int(w*width)))
    return scaled_coord


def heatmaps_to_array(heatmap: np.ndarray, nr_classes: int = 6, heatmap_shape: Tuple[int, int] = (64, 64)) -> List[Tuple[int, int]]:
    """
    Convert heatmap to normalized list of coordinates
    """
    for i in range(nr_classes):
        if np.max(heatmap[:, :, i]) < 0.2:
            heatmap[:, :, i] = np.zeros(heatmap_shape)

    max_coords = [np.unravel_index(
        heatmap[:, :, i].argmax(), heatmap[:, :, i].shape) for i in range(nr_classes)]

    norm_coords = [(lambda i, v: (i/heatmap_shape[0], v/heatmap_shape[1]))(i, v)
                   for i, v in max_coords]
    return norm_coords
