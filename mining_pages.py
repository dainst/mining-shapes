
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from distutils.version import StrictVersion
import pytesseract
import shutil
import tensorflow as tf
import pandas as pd
import os

from mining_pages_utils.image_ocr_utils import load_page, cut_image, ocr_pre_processing, cut_image_savetemp, cut_image_figid
from mining_pages_utils.dataframe_utils import get_page_labelmap_as_df, get_figid_labelmap_as_df, extract_page_detections, extract_detections_figureidv2
from mining_pages_utils.dataframe_utils import filter_best_page_detections, filter_best_vesselprofile_detections, merge_info, split, provide_pagelist
from mining_pages_utils.json_utils import create_find_JSONL, create_type_JSONL, create_drawing_JSONL, create_catalog_JSONL, create_trench_JSONL
from mining_pages_utils.tensorflow_utils import create_tf_example, create_tf_figid, run_inference_for_page_series, run_inference_for_figure_series, run_vesselprofile_segmentation


INPUTDIRECTORY = '/home/images/apply'
GRAPH = '/frozen_inference_graph.pb'
LABELS = '/label_map.pbtxt'
PAGE_MODEL = '/home/models/inference_graph_mining_pages_v8'
FIGID_MODEL = '/home/models/inference_graph_figureid_v1'
SEG_MODEL = 'Shape_Segmentation/train_colab_20200610.h5'
OUTPATH = '/home/images/OUTPUT/'
VESSELLPATH = OUTPATH + 'vesselprofiles/'
SEGMENTPATH = OUTPATH + 'segmented_profiles/'
CSVOUT = OUTPATH + 'mining_pages_allinfo.csv'

classlist = ['pageid', 'pageinfo']
figureclasslist = ['vesselprofilefigure']
figureidclasslist = ['figureid']
pageid_config = r'--psm 6 -c load_system_dawg=0 load_freq_dawg=0'
pagelist = provide_pagelist(INPUTDIRECTORY)



if StrictVersion(tf.version.VERSION) < StrictVersion('1.9.0'):
    raise ImportError(
        'Please upgrade your TensorFlow installation to v1.9.* or later!')

for path in [VESSELLPATH, SEGMENTPATH]:
    if not os.path.exists(path):
        os.makedirs(path)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PAGE_MODEL + GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


with detection_graph.as_default():
    with tf.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {
            output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)

        all_detections_step1 = pd.DataFrame()

        for index, row in pagelist.iterrows():

            img = load_page(row)
            result = run_inference_for_page_series(img, tensor_dict, sess)
            result.drop("page_imgnp", inplace=True)
            all_detections_step1 = all_detections_step1.append(result)


all_detections_step2 = extract_page_detections(
    all_detections_step1, category_index=get_page_labelmap_as_df(PAGE_MODEL + LABELS))

bestpages = filter_best_page_detections(all_detections_step2, classlist, lowest_score=0.7)
pageid_raw = pd.DataFrame()

#perform ocr page number
for index, row in bestpages.iterrows():
    img = cut_image(row)
    img2 = ocr_pre_processing(img)
    result = pytesseract.image_to_string(img2, config=pageid_config)
    row['newinfo'] = result
    pageid_raw = pageid_raw.append(row)

all_detections_step3 = merge_info(all_detections_step2, pageid_raw)
figures = filter_best_vesselprofile_detections(all_detections_step3, figureclasslist,lowest_score= 0.7)

#detect figure id
detection_figureid_graph = tf.Graph()
with detection_figureid_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(FIGID_MODEL + GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


with detection_figureid_graph.as_default():
    with tf.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {
            output.name for op in ops for output in op.outputs}

        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)

        figures_step1 = pd.DataFrame()

        for index, row in figures.iterrows():
            img = cut_image_savetemp(row, VESSELLPATH)
            result = run_inference_for_figure_series(
                img, tensor_dict, sess)
            result.drop("figure_imgnp", inplace=True)
            figures_step1 = figures_step1.append(result)


figid_category_index = get_figid_labelmap_as_df(FIGID_MODEL + LABELS)
figid_detections = figures_step1.apply(
    extract_detections_figureidv2, axis=1)
figures_step2 = figid_detections.merge(
    figid_category_index, on=['figid_detection_classes'], how='left')

#perform ocr figid
figures_step3 = pd.DataFrame()
for index, row in figures_step2.iterrows():
    img = cut_image_figid(row)
    img2 = ocr_pre_processing(img)
    row['figid_raw'] = pytesseract.image_to_string(img2, config=pageid_config)
    figures_step3 = figures_step3.append(row)


with open(OUTPATH + 'catalogs.jsonl', 'w') as f:
    pubs = figures_step3[['pub_key', 'pub_value']].drop_duplicates()
    pubs.apply(create_catalog_JSONL, file=f, axis=1)
with open(OUTPATH + 'trenches.jsonl', 'w') as f:
    pubs = figures_step3[['pub_key', 'pub_value']].drop_duplicates()
    pubs.apply(create_trench_JSONL, file=f, axis=1)
with open(OUTPATH + 'types.jsonl', 'w') as f:
    figures_step3.apply(create_type_JSONL, file=f, axis=1)
with open(OUTPATH + 'finds.jsonl', 'w') as f:
    figures_step3.apply(create_find_JSONL, file=f, axis=1)
with open(OUTPATH + 'drawings.jsonl', 'w') as f:
    figures_step3.apply(create_drawing_JSONL, file=f, axis=1)


TFRECORDOUT = OUTPATH + 'mining_pages.tfrecord'
writer = tf.io.TFRecordWriter(TFRECORDOUT)

shutil.copyfile(PAGE_MODEL + LABELS, OUTPATH + 'pages_label_map.pbtxt')

mining_pages_detections = figures_step3.append(bestpages)
grouped = split(mining_pages_detections, 'page_path')

for group in grouped:
    tf_example = create_tf_example(group,  TFRECORDOUT)
    writer.write(tf_example.SerializeToString())

writer.close()

TFRECORDOUT = OUTPATH + 'mining_figures.tfrecord'
writer = tf.io.TFRecordWriter(TFRECORDOUT)

shutil.copyfile(FIGID_MODEL + LABELS, OUTPATH + 'figures_label_map.pbtxt')
figids = figures_step3[figures_step3.figid_detection_boxes.notnull()]

figsgrouped = split(figids, 'figure_path')

for group in figsgrouped:
    figtf_example = create_tf_figid(group,  TFRECORDOUT)
    writer.write(figtf_example.SerializeToString())

writer.close()
figures_step3.to_csv(CSVOUT)

#Profile segmentation
print('Perform image segmentation')
run_vesselprofile_segmentation(VESSELLPATH, SEGMENTPATH, SEG_MODEL)

