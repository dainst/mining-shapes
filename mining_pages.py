# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from distutils.version import StrictVersion
import pytesseract
import shutil
import tensorflow as tf
import re
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import inflect
import uuid
from shapely.geometry import Polygon
from mining_pages_utils.image_ocr_utils import double_to_singlepage, load_page, cut_image, ocr_pre_processing, ocr_post_processing_pageid, cut_image_savetemp, cut_image_figid, ocr_post_processing_figid
from mining_pages_utils.dataframe_utils import get_page_labelmap_as_df, get_figid_labelmap_as_df, extract_page_detections, extract_page_detections_new,unfold_pagedetections, page_detections_toframe, extract_detections_figureidv2,humanreadID
from mining_pages_utils.dataframe_utils import extract_pdfid, filter_best_page_detections, select_pdfpages, choose_pageid, filter_best_vesselprofile_detections, merge_info,  provide_pagelist, provide_pdf_path, get_pubs_and_configs, pdf_to_image, handleduplicate_humanreadID
from mining_pages_utils.json_utils import create_find_JSONL, create_constructivisttype_JSONL, create_normativtype_JSONL, create_drawing_JSONL, create_catalog_JSONL, create_trench_JSONL
from mining_pages_utils.tensorflow_utils import create_tf_example_new, create_tf_figid, run_inference_for_page_series, run_inference_for_figure_series, build_detectfn, Df2TFrecord, split


if StrictVersion(tf.version.VERSION) < StrictVersion('2.3.0'):
    raise ImportError(
        'Please upgrade your TensorFlow installation to v1.9.* or later!')

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

INPUTDIRECTORY = '/home/images/apply' 
GRAPH = '/frozen_inference_graph.pb'
LABELS = '/label_map.pbtxt'
PAGE_MODEL = '/home/models/faster_rcnn_resnet101_v1_1024x1024_coco17_OCKquick_miningpagesv9_posttrain'
SAVEDMODEL = '/saved_model'
FIGID_MODEL = '/home/models/faster_rcnn_resnet101_v1_1024x1024_coco17_miningfiguresv3'
SEG_MODEL = '/home/models/shape_segmentation/train_colab_20200610.h5'
OUTPATH = '/home/images/OUTPUT/'
VESSELLPATH = OUTPATH + 'vesselprofiles/'
SEGMENTPATH = OUTPATH + 'segmented_profiles/'
CSVOUT = OUTPATH + 'mining_pages_allinfo.csv'
CLEANCSVOUT = OUTPATH + 'mining_pages_clean.csv'








# %%
publist = get_pubs_and_configs(INPUTDIRECTORY)
pdflist = provide_pdf_path(publist)
pdflistv2 = pdflist.apply(pdf_to_image, axis=1)
pagelist = provide_pagelist(pdflistv2)
pagelist = pagelist.apply(extract_pdfid, axis=1)
pagelist = double_to_singlepage(pagelist)
pagelist = provide_pagelist(pdflist)
pagelist = pagelist.apply(extract_pdfid, axis=1)
pagelist = select_pdfpages(pagelist)




# %%

for path in [VESSELLPATH, SEGMENTPATH]:
    if not os.path.exists(path):
        os.makedirs(path)







all_detections_step1 = pd.DataFrame()
miningpagesdetectfn = build_detectfn(PAGE_MODEL + SAVEDMODEL)
for index, row in pagelist.iterrows():
    print('Page ' + os.path.basename(row['page_path']))

    row, page_imgnp = load_page(row)
    input_tensor = tf.convert_to_tensor(page_imgnp)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = miningpagesdetectfn(input_tensor)
    del page_imgnp
    del input_tensor
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    row['page_detections']= detections
    all_detections_step1 = all_detections_step1.append(row)









# %%
#all_detections_step2, keylist =unfold_pagedetections(all_detections_step1)
#all_detections_step22 = extract_page_detections(all_detections_step2, keylist, category_index=get_page_labelmap_as_df(PAGE_MODEL + LABELS))
all_detections_step2 = page_detections_toframe(all_detections_step1).drop(columns='page_detections')

page_category_index = get_page_labelmap_as_df(PAGE_MODEL + LABELS)
all_detections_step2 = all_detections_step2.merge(page_category_index, on=['detection_classes'], how='left')


# %%
pageids = filter_best_page_detections(all_detections_step2 , lowest_score=0.8)
bestpages = choose_pageid(pageids)
pageid_raw = pd.DataFrame()

#perform ocr page number
for index, row in bestpages.iterrows():
    img = cut_image(row)
    img2 = ocr_pre_processing(img)
    result = pytesseract.image_to_string(img2, config=row['pageid_config'])
    row['newinfo'] = result
    
    pageid_raw = pageid_raw.append(row)
all_detections_step3 = merge_info(all_detections_step2, pageid_raw)
all_detections_step3 = all_detections_step3.apply(ocr_post_processing_pageid, axis=1)

figures = filter_best_vesselprofile_detections(all_detections_step3, lowest_score= 0.7)


# %%
for index, row in figures.iterrows():
    print(row['detection_boxes'])


# %%
#detect figure id


figures_step1 = pd.DataFrame()
miningfiguresdetectfn = build_detectfn(FIGID_MODEL + SAVEDMODEL)
for index, row in figures.iterrows():
    print('Figure from ' + os.path.basename(row['page_path']))
    row, figure_imgnp = cut_image_savetemp(row, VESSELLPATH)
    input_tensor = tf.convert_to_tensor(figure_imgnp)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = miningfiguresdetectfn(input_tensor)
    del figure_imgnp
    del input_tensor
    num_detections = int(detections.pop('num_detections'))
    
    detections = {'figid_' + key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    row['figid_detections'] = detections
    detections['figid_num_detections'] = num_detections
    row = extract_detections_figureidv2(row)         
    figures_step1 = figures_step1.append(row)

figid_category_index = get_figid_labelmap_as_df(PAGE_MODEL + LABELS)
figures_step2 = figures_step1.merge(
figid_category_index, on=['figid_detection_classes'], how='left')





# %%
#perform ocr figid
figures_step3 = pd.DataFrame()
for index, row in figures_step2.iterrows():
    print('OCR ' + os.path.basename(row['figure_path']))
    img = cut_image_figid(row)
    img2 = ocr_pre_processing(img)
    row['figid_raw'] = pytesseract.image_to_string(img2, config=row['pageid_config'])
    del img
    del img2
    figures_step3 = figures_step3.append(row)
figures_step3 = figures_step3.apply(ocr_post_processing_figid, axis=1)
figures_step3 = figures_step3.apply(humanreadID, axis=1)
figures_step3 = handleduplicate_humanreadID(figures_step3)


# %%

with open(OUTPATH + 'catalogs.jsonl', 'w') as f:
    pubs = figures_step3[['pub_key', 'pub_value']].drop_duplicates()
    pubs.apply(create_catalog_JSONL, file=f, axis=1)
with open(OUTPATH + 'trenches.jsonl', 'w') as f:
    pubs = figures_step3[['pub_key', 'pub_value']].drop_duplicates()
    pubs.apply(create_trench_JSONL, file=f, axis=1)
with open(OUTPATH + 'types.jsonl', 'w') as f:
    figures_step3.apply(create_constructivisttype_JSONL, file=f, axis=1)
with open(OUTPATH + 'types_standalone.jsonl', 'w') as f:
    figures_step3.apply(create_normativtype_JSONL, file=f, axis=1)
with open(OUTPATH + 'finds.jsonl', 'w') as f:
    figures_step3.apply(create_find_JSONL, file=f, axis=1)
with open(OUTPATH + 'drawings.jsonl', 'w') as f:
    figures_step3.apply(create_drawing_JSONL, file=f, axis=1)







figures_step3.to_csv(CSVOUT)
figures_clean = figures_step3[['pub_key','pub_value','figure_tmpid','HRID','detection_scores', 'detection_classesname','page_imgname','pageid_raw','figid_raw','pageid_clean','figid_clean','pageinfo_raw','figure_path','page_path']]
figures_clean.to_csv(CLEANCSVOUT)


#Profile segmentation
#print('Perform image segmentation')
#run_vesselprofile_segmentation(VESSELLPATH, SEGMENTPATH, SEG_MODEL)


# %%
shutil.copyfile(PAGE_MODEL + LABELS, OUTPATH +'pages_label_map.pbtxt')

mining_pages_detections = figures_step3.append(bestpages)
mining_pages_detections2 = mining_pages_detections.reindex(axis=0)
pages_grouped = mining_pages_detections2.groupby('pub_name')

for name, group in pages_grouped:
    imgdir = os.path.join(OUTPATH, name + '_pages/')
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
    Df2TFrecord(group, 'page_path', OUTPATH + name +'_pages.tfrecord')
    for index, row in group.iterrows():
        imgoutpath = imgdir + os.path.basename(row['page_path'])
        print(imgoutpath)
        if not os.path.exists(imgoutpath):
            shutil.copyfile(row['page_path'], imgoutpath)

shutil.copyfile(FIGID_MODEL + LABELS, OUTPATH + 'figures_label_map.pbtxt')
figures_step3 = figures_step3[figures_step3.detection_boxes.notnull()]

figures_grouped = figures_step3.groupby('pub_name')

for name, group in figures_grouped:
    imgdir = os.path.join(OUTPATH, name + '_figures/')
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
    Df2TFrecord(group, 'figure_path', OUTPATH + name +'_figures.tfrecord')
    for index, row in group.iterrows():
        imgoutpath = imgdir + os.path.basename(str(row['figure_path']))
        print(imgoutpath)
        if not os.path.exists(imgoutpath):
            shutil.copyfile(row['figure_path'], imgoutpath)


# %%



