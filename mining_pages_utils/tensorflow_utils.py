import tensorflow as tf
import io
import os
import numpy as np
import pandas as pd
from PIL import Image

from object_detection.utils import dataset_util, ops


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