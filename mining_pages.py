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
import shapely
import geopandas as gpd
import io
from object_detection.utils import dataset_util, ops
from PIL import Image
import re

import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import inflect
import uuid
from shapely.geometry import Polygon
from mining_pages_utils.image_ocr_utils import double_to_singlepage, load_page, cut_image, ocr_pre_processing_page, ocr_pre_processing_figure, ocr_post_processing_pageid, cut_image_savetemp, cut_image_figure, ocr_post_processing_figid
from mining_pages_utils.dataframe_utils import get_page_labelmap_as_df, get_figid_labelmap_as_df, extract_page_detections, extract_page_detections_new,unfold_pagedetections, page_detections_toframe, extract_detections_figureidv2,humanreadID
from mining_pages_utils.dataframe_utils import extract_pdfid, filter_best_page_detections, select_pdfpages, choose_pageid, filter_best_vesselprofile_detections, merge_info,  provide_pagelist, provide_pdf_path, get_pubs_and_configs, pdf_to_imagev2, handleduplicate_humanreadID
from mining_pages_utils.json_utils import create_find_JSONL, create_constructivisttype_JSONL, create_normativtype_JSONL, create_drawing_JSONL, create_catalog_JSONL, create_trench_JSONL
from mining_pages_utils.tensorflow_utils import create_tf_example_new, create_tf_figid, run_inference_for_page_series, run_inference_for_figure_series, build_detectfn, Df2TFrecord, split
from mining_pages_utils.request_utils import getZenonInfo

if StrictVersion(tf.version.VERSION) < StrictVersion('2.3.0'):
    raise ImportError(
        'Please upgrade your TensorFlow installation to v2.3.* or later!')

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

INPUTDIRECTORY = '/home/images/apply' 
GRAPH = '/frozen_inference_graph.pb'
LABELS = '/label_map.pbtxt'
PAGE_MODEL = '/home/models/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8_miningpagesv12'
SAVEDMODEL = '/saved_model'
FIGID_MODEL = '/home/models/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8_miningfiguresv5_OCKquick'
SEG_MODEL = '/home/models/shape_segmentation/train_colab_20200610.h5'
OUTPATH = '/home/images/OUTPUT/'
VESSELLPATH = OUTPATH + 'vesselprofiles/'
SEGMENTPATH = OUTPATH + 'segmented_profiles/'
CSVOUT = OUTPATH + 'mining_pages_allinfo.csv'
CLEANCSVOUT = OUTPATH + 'mining_pages_clean.csv'








# %%
publist = get_pubs_and_configs(INPUTDIRECTORY)
publist = publist.apply(getZenonInfo, axis=1)
pdflist = provide_pdf_path(publist)
pdflistv2 = pdflist.apply(pdf_to_imagev2, axis=1)
print('Finished pdf2images')
pagelist = provide_pagelist(pdflistv2)
print('Created primary pagelist')
pagelist = pagelist.apply(extract_pdfid, axis=1)
double_to_singlepage(pagelist)
pagelist = provide_pagelist(pdflistv2)
pagelist = pagelist.apply(extract_pdfid, axis=1)
#pagelist = select_pdfpages(pagelist)




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
    img2 = ocr_pre_processing_page(img)
    result = pytesseract.image_to_string(img2, config=row['pageid_config'])
    row['newinfo'] = result
    
    pageid_raw = pageid_raw.append(row)
all_detections_step3 = merge_info(all_detections_step2, pageid_raw)
all_detections_step3 = all_detections_step3.apply(ocr_post_processing_pageid, axis=1)

figures = filter_best_vesselprofile_detections(all_detections_step3, lowest_score= 0.8)


# %%
for index, row in figures.iterrows():
    print(row['detection_scores'])


# %%
#detect figure id


figures_step1 = pd.DataFrame()
figure_category_index = get_figid_labelmap_as_df(FIGID_MODEL + LABELS)
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
    print(num_detections)
    print(detections)
    #print(type(detections['detection_multiclass_scores']))
    detectionsDL = { key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    #print(detectionsDL)
    detectionsDL['detection_classesname'] = [ figure_category_index.loc[figure_category_index['figid_detection_classes'] == i, 'figid_detection_classesname'].item() for i in detectionsDL['detection_classes'] ]
    detectionsLD = [{key : value[i] for key, value in detectionsDL.items()} 
         for i in range(num_detections)] 
    #print(detectionsLD)
    row['figure_detections'] = detectionsLD
    #print(row['figure_detections'][0]['detection_scores'])
    row['figure_num_detections'] = num_detections       
    figures_step1 = figures_step1.append(row)

#figid_category_index = get_figid_labelmap_as_df(PAGE_MODEL + LABELS)
#figures_step2 = 
#figure_detections_names = [detection['detection_classes'] :  for detection in figures_step1['figure_detections']]

#figures_step1['figure_detections'].merge(
#figid_category_index, on=['figid_detection_classes'], how='left')





# %%

def filterFigureDetections(series):
    if series['frame_contains']:
        if series['detection_classesname'] in series['allowedFrameDetections']:
            detections = [i for i in series['figure_detections'] if i['detection_classesname'] in series['frame_contains']]
            #print(detections)
            detections_df = pd.DataFrame(detections)
            detections_df = detections_df[detections_df['detection_scores'] == detections_df.groupby('detection_classesname')['detection_scores'].transform('max')]
            series['bestfigure_detections'] = detections_df.to_dict('records')
    if series['figure_contains'] :
        if series['detection_classesname'] in series['allowedFigureDetections'] :
            detections = [i for i in series['figure_detections'] if i['detection_classesname'] in series['figure_contains']]
            detections_df = pd.DataFrame(detections)
            print(detections_df)
            detections_df = detections_df[detections_df['detection_scores'] == detections_df.groupby('detection_classesname')['detection_scores'].transform('max')]
            #detectionsmax = detections_df.groupby(['detection_classesname'], sort=False)['detection_scores'].max()
            #df['count_max'] = df.groupby(['Mt'])['count'].transform(max)
            #detections_dfmax = detections_df[detections_df['detection_scores'] == detections_df.groupby('detection_classesname')['detection_scores'].transform('max')]
            print(detections_df)
            series['bestfigure_detections'] = detections_df.to_dict('records')
    return series


figures_step2 = figures_step1.apply(filterFigureDetections, axis=1)
#for index, row in figures_step2.iterrows():
    #print(row['figure_path'])
    #if type(row['bestfigure_detections']) == list:
        #for i in row['bestfigure_detections']:
            #print(i['detection_classesname'], i['detection_scores'])



# %%
#perform ocr figid

def ocr_post_processing_figure(row, detection):
    if row[str(detection['detection_classesname']) + '_raw']:
        if str(detection['detection_classesname']) + '_exclude_strings' in list(row.keys()):
            for exclude_string in row[str(detection['detection_classesname']) + '_exclude_strings']:
                row[str(detection['detection_classesname']) + '_raw'].replace(exclude_string,"")
        else:
            print(str(detection['detection_classesname']) + '_exclude_strings' + ' not written in config - No strings will be excluded.')
        
        if str(detection['detection_classesname']) + '_regex' in list(row.keys()):
            regex = re.compile(row[str(detection['detection_classesname']) + '_regex'])
            result = re.search(regex, row[str(detection['detection_classesname']) + '_raw'])
            if result:
                row[str(detection['detection_classesname']) + '_clean'] = result.group(1)
            else:
                row['figid_clean'] = 'none'
        else:
            print(str(detection['detection_classesname']) + '_regex' + ' not written in config - No regex filter will be applied')

    return row

figures_step3 = pd.DataFrame()
for index, row in figures_step2.iterrows():
    print('OCR ' + os.path.basename(row['figure_path']))
    detections = []
    if type(row['bestfigure_detections']) == list:
        for detection in row['bestfigure_detections']:
            detection = cut_image_figure(row, detection)
            detection = ocr_pre_processing_figure(detection)
            print(detection['detection_classesname'])
            if str(detection['detection_classesname']) + '_config' in list(row.keys()):
                row[str(detection['detection_classesname']) + '_raw'] = pytesseract.image_to_string(detection['imgnp'], config=row[str(detection['detection_classesname']) + '_config'])
                print(row[str(detection['detection_classesname']) + '_raw'])
            else:
                row[str(detection['detection_classesname']) + '_raw'] = pytesseract.image_to_string(detection['imgnp'], config=row['pageid_config'])
                print(row[str(detection['detection_classesname']) + '_raw'])
        row['figure_detections'] = detections.append(detection)
        row = ocr_post_processing_figure(row, detection)
        del detection['imgnp']
        

    figures_step3 = figures_step3.append(row)




# %%
def generateShapebox(series):
    box = series['detection_boxes']
    ymin, xmin, ymax, xmax = box
    bbox_xmin = int((xmin)*series['page_width'])
    bbox_ymin = int((ymin)*series['page_height'])
    bbox_xmax = int((xmax)*series['page_width'])
    bbox_ymax = int((ymax)*series['page_height'])
    shapebox = shapely.geometry.box(bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, ccw=True)
    series['detection_shapebox'] = shapebox
    return series
def seperateFramesFigures(group):
    infoframes = group[group['detection_classesname'] == 'infoframe']
    figures = group[group['detection_classesname'] != 'infoframe']
    #print(infoframes)
    figuresWithInfo = figures.apply(infosFromFrames, infoframes=infoframes, axis=1)
    return figuresWithInfo


def infosFromFrames(series, infoframes):
    for index, row in infoframes.iterrows():
        if series['detection_shapebox'].within(row['detection_shapebox']):
            for info in row['frame_contains']:
                if info + '_raw' in row.keys():
                #print(row[info + '_raw'])
                    series[info + '_raw'] = row[info + '_raw']
                    series[info + '_clean'] = row[info + '_clean']
    return series

figures_step3 = figures_step3.apply(generateShapebox, axis=1)
figures_geodf = gpd.GeoDataFrame( figures_step3,geometry= 'detection_shapebox')
figures_geodf2 = figures_geodf.groupby('page_path', as_index=False).apply(seperateFramesFigures).reset_index()
#print(figures_geodf2['infoframeid_raw'])
 
    




 


# %%




def humanreadID(Series):
    humanreadID = ''
    if Series['patternHRID']:
        for element in Series['patternHRID']:
            if element in list(Series.keys()):
                humanreadID += str(Series[element]) + '_'
            else:
                print('HRID Element: ' + element + ' does not exist.')
                humanreadID += element + '_'
    else:

        humanreadID = str(Series['figure_tmpid'])

    Series['HRID'] = humanreadID.rstrip('_')
    return Series

figures_step4 = figures_geodf2.apply(humanreadID, axis=1)
figures_step4 = handleduplicate_humanreadID(figures_step4)


# %%

#with open(OUTPATH + 'catalogs.jsonl', 'w') as f:
    #pubs = figures_step3[['pub_key', 'pub_value']].drop_duplicates()
    #pubs.apply(create_catalog_JSONL, file=f, axis=1)
#with open(OUTPATH + 'trenches.jsonl', 'w') as f:
    #pubs = figures_step3[['pub_key', 'pub_value']].drop_duplicates()
    #pubs.apply(create_trench_JSONL, file=f, axis=1)
#with open(OUTPATH + 'types.jsonl', 'w') as f:
    #figures_step3.apply(create_constructivisttype_JSONL, file=f, axis=1)
#with open(OUTPATH + 'types_standalone.jsonl', 'w') as f:
    #figures_step3.apply(create_normativtype_JSONL, file=f, axis=1)
#with open(OUTPATH + 'finds.jsonl', 'w') as f:
    #figures_step3.apply(create_find_JSONL, file=f, axis=1)
#with open(OUTPATH + 'drawings.jsonl', 'w') as f:
    #figures_step3.apply(create_drawing_JSONL, file=f, axis=1)










#Profile segmentation
#print('Perform image segmentation')
#run_vesselprofile_segmentation(VESSELLPATH, SEGMENTPATH, SEG_MODEL)


# %%
#figures_step4.to_csv(CSVOUT)
#figures_clean = figures_step4[['pub_key','pub_value','figure_tmpid','HRID','detection_scores', 'detection_classesname','page_imgname','pageid_raw','figureid_raw','pageid_clean','figureid_clean', 'infoframeid_clean', 'figureinfo_clean', 'pageinfo_raw','figure_path','page_path']]
#figures_clean.to_csv(CLEANCSVOUT)


# %%
def create_tf_example_new(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    fileid = os.path.basename(group.filename)
    fileid = fileid.encode('utf8')
    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        box = row['detection_boxes']
        ymin, xmin, ymax, xmax = box
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(
            row['detection_classesname'].encode('utf8'))
        classes.append(int(row['detection_classes']))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(fileid),
        # 'image/source_id': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(fileid),
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

def create_tf_example_figure(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    fileid = os.path.basename(group.filename)
    fileid = fileid.encode('utf8')
    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        if type(row['bestfigure_detections']) == list:
            for detection in row['bestfigure_detections']:
                box = detection['detection_boxes']
                ymin, xmin, ymax, xmax = box
                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)
                classes_text.append(
                    detection['detection_classesname'].encode('utf8'))
                classes.append(int(detection['detection_classes']))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(fileid),
        # 'image/source_id': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(fileid),
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

def Df2TFrecord(df, imagecolumn, outpath):
    if imagecolumn is 'page_path':
        prefix = ''

    writer = tf.io.TFRecordWriter(outpath)
    grouped = split(df, imagecolumn)
    #print(grouped)

    for group in grouped:
        if imagecolumn is 'figure_path':
        
            tf_example = create_tf_example_figure(group,  outpath)
            writer.write(tf_example.SerializeToString())
        if imagecolumn is 'page_path':
            tf_example = create_tf_example_new(group,  outpath)
            writer.write(tf_example.SerializeToString())

    writer.close()


# %%
shutil.copyfile(PAGE_MODEL + LABELS, OUTPATH +'pages_label_map.pbtxt')

mining_pages_detections = figures_step3.append(bestpages)
mining_pages_detections2 = mining_pages_detections.reindex(axis=0)
pages_grouped = mining_pages_detections2.groupby('catalog_id')

for name, group in pages_grouped:
    imgdir = os.path.join(OUTPATH, name + '_pages/')
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
    Df2TFrecord(group, 'page_path', OUTPATH + name +'_pages.tfrecord')
    for index, row in group.iterrows():
        imgoutpath = imgdir + os.path.basename(row['page_path'])
        #print(imgoutpath)
        if not os.path.exists(imgoutpath):
            shutil.copyfile(row['page_path'], imgoutpath)

shutil.copyfile(FIGID_MODEL + LABELS, OUTPATH + 'figures_label_map.pbtxt')
figures_step3 = figures_step3[figures_step3.detection_boxes.notnull()]

figures_grouped = figures_step3.groupby('catalog_id')

for name, group in figures_grouped:
    imgdir = os.path.join(OUTPATH, name + '_figures/')
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
    Df2TFrecord(group, 'figure_path', OUTPATH + name +'_figures.tfrecord')
    for index, row in group.iterrows():
        imgoutpath = imgdir + os.path.basename(str(row['figure_path']))
        #print(imgoutpath)
        if not os.path.exists(imgoutpath):
            shutil.copyfile(row['figure_path'], imgoutpath)


# %%



