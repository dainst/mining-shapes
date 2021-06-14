# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from distutils.version import StrictVersion
import pytesseract
import zipfile
from datumaro.components.dataset import Dataset
import math
import shutil
import requests
from decimal import Decimal
import tensorflow as tf
from pathlib import Path
import numpy as np
import shapely
from pdf2image import convert_from_path
import uuid
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import io
from object_detection.utils import dataset_util, ops
from PIL import Image
import re
from datetime import datetime
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import inflect
import cv2
from PIL import Image
import imagehash
import uuid
from shapely.geometry import Polygon
from mining_pages_utils.image_ocr_utils import double_to_singlepage, load_page, cut_image, ocr_pre_processing_page, ocr_pre_processing_figure, ocr_post_processing_pageid, cut_image_savetemp, cut_image_figure, ocr_post_processing_figid
from mining_pages_utils.dataframe_utils import get_page_labelmap_as_df, get_figid_labelmap_as_df, extract_page_detections, extract_page_detections_new,unfold_pagedetections, page_detections_toframe, extract_detections_figureidv2,humanreadID
from mining_pages_utils.dataframe_utils import extract_pdfid, filter_best_page_detections, select_pdfpages, choose_pageid, filter_best_vesselprofile_detections, merge_info,  provide_pagelist, provide_pdf_path, get_pubs_and_configs, pdf_to_imagev2, handleduplicate
from mining_pages_utils.json_utils import create_find_JSONL, create_constructivisttype_JSONL, create_normativtype_JSONL, create_drawing_JSONL, create_catalog_JSONL, create_trench_JSONL
from mining_pages_utils.tensorflow_utils import create_tf_example_new, create_tf_figid, run_inference_for_page_series, run_inference_for_figure_series, build_detectfn, Df2TFrecord, split
from mining_pages_utils.request_utils import getZenonInfo, getZenonBibtex, getAllDocs, DocsStructure, selectOfResourceTypes

if StrictVersion(tf.version.VERSION) < StrictVersion('2.3.0'):
    raise ImportError(
        'Please upgrade your TensorFlow installation to v2.3.* or later!')

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

INPUTDIRECTORY = '/home/images/apply' 
ANNOTDIR =  '/home/images/HUMAN_CORR'
GRAPH = '/frozen_inference_graph.pb'
LABELS = '/label_map.pbtxt'
PAGE_MODEL = '/home/models/faster_rcnn_resnet101_v1_1024x1024_coco17_miningpagesv10'
SAVEDMODEL = '/saved_model'
FIGID_MODEL = '/home/models/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8_miningfiguresv5_Hayes1972quickv2'
SEG_MODEL = '/home/models/shape_segmentation/train_colab_20200610.h5'
OUTPATH = '/home/images/OUTPUT/'
VESSELLPATH = OUTPATH + 'vesselprofiles/'
DATASETEXPORT = OUTPATH + 'datasetexport/'
TFRimgPATH = OUTPATH + 'tfr_images/'
SEGMENTPATH = OUTPATH + 'segmented_profiles/'
CSVOUT = OUTPATH + 'mining_pages_allinfo.csv'
CLEANCSVOUT = OUTPATH + 'mining_pages_clean.csv'








# %%
publist = get_pubs_and_configs(INPUTDIRECTORY)
publist = publist.apply(getZenonInfo, axis=1)
publist = publist.apply(getZenonBibtex, axis=1)
pdflist = provide_pdf_path(publist)
pdflistv2 = pdflist.apply(pdf_to_imagev2, axis=1)
print('Finished pdf2images')
pagelist = provide_pagelist(pdflistv2)
print('Created primary pagelist')
pagelist = pagelist.apply(extract_pdfid, writefield='page_pdfid', readfield='page_imgname', axis=1)
double_to_singlepage(pagelist)
pagelist = provide_pagelist(pdflistv2)
pagelist = pagelist.apply(extract_pdfid, writefield='page_pdfid', readfield='page_imgname', axis=1)
#pagelist = select_pdfpages(pagelist)




# %%
def unzip_tfrecords(series, readfield):
    if series[readfield].endswith('.zip'):
        unzippedfolder = Path(os.path.join(ANNOTDIR,series[readfield].replace('.zip','')))
        if not unzippedfolder.exists():
            os.mkdir(Path(os.path.join(ANNOTDIR,series[readfield].replace('.zip',''))))
        with zipfile.ZipFile(os.path.join(ANNOTDIR,series[readfield]), 'r') as zip_ref:
            zipinfos = zip_ref.infolist()
            tfrecordfiles = {}
            for zipinfo in zipinfos:
                # This will do the renaming
                #print(zipinfo.filename)
                #zipinfo.filename = str(i) + str(zipinfo.filename)
                print(zipinfo)
                if str(zipinfo.filename).endswith('.tfrecord'):
                    series[readfield + 'Path'] = os.path.join(unzippedfolder,zipinfo.filename)
                    source = zip_ref.open(zipinfo.filename)
                    if not Path(series[readfield + 'Path']).exists():
                        target = open(series[readfield + 'Path'], "wb")
                        with source, target:
                            shutil.copyfileobj(source, target)
                if str(zipinfo.filename).endswith('.pbtxt'):
                    series[readfield + 'PathPbtxt'] = os.path.join(unzippedfolder,zipinfo.filename)
                    if not Path(series[readfield + 'PathPbtxt']).exists():
                        source = zip_ref.open(zipinfo.filename)
                        target = open(series[readfield + 'PathPbtxt'], "wb")
                        with source, target:
                            shutil.copyfileobj(source, target)


    return series

def importTFrecords(series, readfield):
    dataset= Dataset.import_from(series[readfield], 'tf_detection_api')
    return dataset



def DatumaroDatasetToDF(DatumaroDataset):
    dictlist = []
    for item in dataset:
        dict = vars(item)
        dictlist.append(dict)
    return pd.DataFrame(dictlist)
def decode_image(image_bytes, dtype=np.float32):
    import cv2
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    image = image.astype(dtype)
    assert len(image.shape) in {2, 3}
    if len(image.shape) == 3:
        assert image.shape[2] in {3, 4}
    return image
def DatumaroImageToArray(series, readfield):
    byteimage = series[readfield].get_bytes()
    page = decode_image(byteimage)
    return page

def exportImagesOfDatumaroDataset(dataset):
    for index, row in dataset.iterrows():
        image = DatumaroImageToArray(row, readfield='image')

        if not Path(DATASETEXPORT).exists():
            os.mkdir(DATASETEXPORT)
        cv2.imwrite(DATASETEXPORT + os.path.basename(row['id']) + '.png', image )    




if 'nameOfPageAnnotations' in pagelist.columns:
    pagelist = pagelist.apply(unzip_tfrecords, readfield='nameOfPageAnnotations', axis=1)
pagelistannots = pd.DataFrame()
for name,group in pagelist.groupby('nameOfPageAnnotations' + 'Path'):
    if Path(group.iloc[0]['nameOfPageAnnotationsPath']).exists():
        dataset = importTFrecords(group.iloc[0], readfield = 'nameOfPageAnnotationsPath')
        
        #dataset.export(DATASETEXPORT, 'tf_detection_api', save_images=True)
        DatumaroDF = DatumaroDatasetToDF(dataset)
        exportImagesOfDatumaroDataset(DatumaroDF)
        DatumaroDF['pdfid_regex'] = group.iloc[0]['pdfid_regex']
        DatumaroDF = DatumaroDF.apply(extract_pdfid, writefield='datumaro_page_pdfid', readfield='id', axis=1)
        DatumaroDF = DatumaroDF.drop(columns=['pdfid_regex']).set_index('datumaro_page_pdfid')
        group = group.set_index('page_pdfid')
        groupannotated = group.join(DatumaroDF, how='left', lsuffix='', rsuffix='', sort=False)
        groupannotated = groupannotated.reset_index()
    pagelistannots = pagelistannots.append(groupannotated)

print(pagelistannots['id'])
    #for item in dataset:
        #print(vars(item))
        #item_dict = extract_pdfid(vars(item), writefield='page_pdfid', readfield='id')
        #print(item_dict)


# %%

for path in [VESSELLPATH, SEGMENTPATH]:
    if not os.path.exists(path):
        os.makedirs(path)







all_detections_step1 = pd.DataFrame()
miningpagesdetectfn = build_detectfn(PAGE_MODEL + SAVEDMODEL)
for index, row in pagelistannots.iterrows():
    if not row['annotations'] or 'annotations' not in row.keys():
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
        print(row['page_detections'])
        print(row['annotations'])
        #all_detections_step1 = all_detections_step1.append(row)
    if 'annotations' in row.keys() and row['annotations']:
        img = DatumaroImageToArray(row, readfield='image')
        print('Datumaro page ',img.shape)
        height, width, channel = img.shape
        row['page_width'] = width
        row['page_height'] = height
        row['page_channel'] = channel
    all_detections_step1 = all_detections_step1.append(row)
print(len(all_detections_step1))
#print(all_detections_step1['page_width'])








# %%
def bboxSwitch_yminfirst(box):
    xmin, ymin, xmax, ymax = box
    newbox = [ymin , xmin , ymax , xmax]
    return newbox

def absTorelImagecoords(box, width, height):
    ymin , xmin , ymax , xmax = box
    print(xmin, height)
    rel_xmin = float(xmin)/ float(height)
    print(rel_xmin)
    rel_ymin = float(ymin)/ float(height)
    rel_xmax = float(xmax)/ float(width)
    rel_ymax = float(ymax)/ float(height)
    newcoords = [rel_ymin , rel_xmin , rel_ymax , rel_xmax]
    return newcoords
def relToabsImagecoords(box, width, height):
    ymin , xmin , ymax , xmax = box
    abs_xmin = int(float(xmin)* float(height))

    abs_ymin = int(float(ymin)* float(height))
    abs_xmax = int(float(xmax)* float(width))
    abs_ymax = int(float(ymax)* float(height))
    newcoords = [abs_ymin , abs_xmin , abs_ymax , abs_xmax]
    return newcoords

def page_detections_toframe(df: pd.DataFrame) -> pd.DataFrame:
    all_detections = pd.DataFrame()
    for index, row in df.iterrows():
        if not row['annotations'] or 'annotations' not in row.keys():
            N = int(row['page_detections']['num_detections'])
            for i in range(0, N):
                detection = row
                #detection.drop('detection_boxes', inplace=True)
                detection['detection_boxes'] = row['page_detections']['detection_boxes'][i]
                #detection.drop('detection_classes', inplace=True)
                detection['detection_classes'] = row['page_detections']['detection_classes'][i]
                #detection.drop('detection_scores', inplace=True)
                detection['detection_scores'] = row['page_detections']['detection_scores'][i]
                #if 'page_detections' in detection.keys():
                    #detection.pop('page_detections')
                all_detections = all_detections.append(detection)
        if 'annotations' in row.keys() and row['annotations']:
            for i in row['annotations']:
                #print(row['page_pdfid'], vars(i))
                annotation = vars(i)
                detection = row
                bbox = bboxSwitch_yminfirst(annotation['points'])
                print(row['page_pdfid'])
                print(annotation['points'])
                print(bbox)
                print('These pageshapes are applied to figure bbox: ', detection['page_height'], detection['page_width'])
                detection['detection_boxes'] = absTorelImagecoords(bbox, width=detection['page_width'], height=detection['page_height'])
                print(detection['detection_boxes'], detection['page_width'])
                checkboxabs = relToabsImagecoords(detection['detection_boxes'], width=detection['page_width'], height=detection['page_height'])
                print('This is the return abs box: ',checkboxabs)
                detection['detection_classes'] = annotation['label'] +1
                detection['detection_scores'] = 1.0
                #print(detection.keys())
                all_detections = all_detections.append(detection)
                

    return pd.DataFrame(all_detections)
all_detections_step2 = page_detections_toframe(all_detections_step1)


if not 'nameOfPageAnnotations' in all_detections_step2.columns:
    page_category_index = get_page_labelmap_as_df(PAGE_MODEL + LABELS)
if 'nameOfPageAnnotations' in all_detections_step2.columns:
    page_category_index = get_page_labelmap_as_df(all_detections_step2.iloc[0]['nameOfPageAnnotationsPathPbtxt'])

print(list(set(list(all_detections_step2['detection_classes']))))
all_detections_step2 = all_detections_step2.merge(page_category_index, on=['detection_classes'], how='left')


# %%

def cut_image(row):
    """
    @brief Select an ROI from image
    @param dataframe dataframe with image and ROI metadata
    """
    page_imgnp = cv2.imread(row['page_path'])
    #print(page_imgnp.shape)
    box = row['detection_boxes']
    print(row['detection_boxes'])
    print(page_imgnp.shape)
    ymin, xmin, ymax, xmax = relToabsImagecoords(box, width = row['page_width'], height = row['page_height'])

    bbox_np = page_imgnp[ymin:ymax, xmin:xmax]
    return bbox_np


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
all_detections_step3a = merge_info(all_detections_step2, pageid_raw)


# %%
def guessPdfpageDistance(series):
    if isinstance(series['pageid_clean'], str):
        series['pageDistance'] = int(int(series['page_pdfid']) - int(series['pageid_clean']))
    return series

def pdfpage2realpage(series):
    if 'pdfpage2realpage' in series.keys() and series.get('pdfpage2realpage') is not None :
        series['pageid_clean'] = int(int(series['page_pdfid']) +  int(series['pdfpage2realpage']))
    else:
        guessPdfpageDistance(series)

    return series


all_detections_step3 = all_detections_step3a.apply(ocr_post_processing_pageid, axis=1)
all_detections_step3 = all_detections_step3.apply(pdfpage2realpage, axis=1)

#all_detections_step3 = smoothPageCorrection(all_detections_step3)
print()
figures = filter_best_vesselprofile_detections(all_detections_step3, lowest_score= 0.8)


# %%
print(figures)


# %%
def getHashDuplicates(dfnew, dfold, hashtype):
    duplicates_newdf = dfnew[dfnew[hashtype].isin(dfold[hashtype])].set_index(hashtype)
    duplicates_olddf = dfold[dfold[hashtype].isin(dfnew[hashtype])].set_index(hashtype)
    print(len(dfnew), len(duplicates_newdf))
    print(len(dfold), len(duplicates_olddf))
    concatenated_dataframes = pd.concat([duplicates_newdf, duplicates_olddf], axis=1)
    return concatenated_dataframes

def getHashalmostDuplicates(dfnew, dfold, readfield, percent):
    for index, row in dfnew.iterrows():
        row['figure_detections'] = [annotrow['figure_detections'] for index, annotrow in dfnew.iterrows() if annotrow[readfield] - row[readfield] < 2][0]
        print(row['figure_detections'])
    
   
    #duplicates_newdf = dfnew[dfnew[readfield] - ].set_index(hashtype)

def makeHash(df, pathcol):
    for index,row in df.iterrows():
        hash_av = imagehash.dhash(Image.open(row[pathcol]), 32)
        df.loc[index, 'dhash'] = hash_av
    return df



def cut_image_savetemp(row, img, outpath_base) -> pd.DataFrame:
    """
    @brief Select an ROI from image and writes detected vesselprofile to disk
    @param dataframe dataframe with image and ROI metadata
    """
    box = row['detection_boxes']

    ymin, xmin, ymax, xmax = relToabsImagecoords(box, width = row['page_width'], height = row['page_height'])

    figure_imgnp = img[ymin:ymax, xmin:xmax]
    #print('This is the shape of cut out figure ',figure_imgnp.shape)
    
    figure_height, figure_width, figure_channel = figure_imgnp.shape
    row['figure_height'] = figure_height
    row['figure_width'] = figure_width
    row['figure_channel'] = figure_channel
    row['figure_tmpid'] = uuid.uuid4()
    row['figure_path'] = outpath_base + str(row['page_pdfid']) + '_' + str(row['figure_tmpid']) + '.png'
    cv2.imwrite(str(row['figure_path']), figure_imgnp)

    return row, figure_imgnp

### Load annotated figure tfrecord ###
print(figures.columns)
if 'nameOfFigureAnnotations' in figures.columns:
    figures = figures.apply(unzip_tfrecords, readfield='nameOfFigureAnnotations', axis=1)
    figurelistannots = pd.DataFrame()
    for name,group in figures.groupby('nameOfFigureAnnotations' + 'Path'):
        if Path(group.iloc[0]['nameOfFigureAnnotations' + 'Path']).exists():
            print('Yes figure tfrecords exists')
            dataset = importTFrecords(group.iloc[0], readfield = 'nameOfFigureAnnotations' + 'Path')
            DatumaroDF = DatumaroDatasetToDF(dataset) 
        figurelistannots = figurelistannots.append(DatumaroDF)

### Load DatumaroImage and generate Hash ###

    for index, row in figurelistannots.iterrows():
        figure_np = DatumaroImageToArray(row, readfield='image')
        #print('This is theshape of Datumaro figure ', figure_np.shape )
        figurelistannots.loc[index, 'dhash']  = imagehash.dhash(Image.fromarray(np.uint8(figure_np)))
        if not Path(TFRimgPATH).exists():
            os.mkdir(TFRimgPATH)
        cv2.imwrite(TFRimgPATH + os.path.basename(row['id']) + '.png', figure_np )
        #imgplot = plt.imshow(figure_np)
        #print(row['detection_classesname'])
        #plt.show()



### detect figure id and generate Hash of incoming figures ###
figures_step1 = pd.DataFrame()
figure_category_index = get_figid_labelmap_as_df(FIGID_MODEL + LABELS)
print(figure_category_index)
miningfiguresdetectfn = build_detectfn(FIGID_MODEL + SAVEDMODEL)

for index, row in figures.iterrows():
    print('Figure from ' + os.path.basename(row['page_path']))
    #print(row.keys())
    #if 'nameOfFigureAnnotations' in figures_step1.columns:
        #page_imgnp = DatumaroImageToArray(row, readfield = 'image')
    #if 'nameOfFigureAnnotations' not in row.keys():
    page_imgnp = cv2.imread(row['page_path'])
    row, figure_imgnp = cut_image_savetemp(row, img = page_imgnp, outpath_base=VESSELLPATH)
    row['dhash'] = imagehash.dhash(Image.fromarray(np.uint8(figure_imgnp)))
    #print(row['dhash'])
    
    if 'nameOfFigureAnnotations' not in row.keys():
        input_tensor = tf.convert_to_tensor(figure_imgnp)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = miningfiguresdetectfn(input_tensor)
        print(detections)
        del figure_imgnp
        del input_tensor
        num_detections = int(detections.pop('num_detections'))
        #print(num_detections)
        #print(detections)
        #print(type(detections['detection_multiclass_scores']))
        detectionsDL = { key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        #print(detectionsDL)
        print('Here are the detection classes in the dataset ',detectionsDL['detection_classes'] )
        print('And this is the index ',figure_category_index)
        detectionsDL['detection_classesname'] = [ figure_category_index.loc[figure_category_index['figid_detection_classes'] == i, 'figid_detection_classesname'].item() for i in detectionsDL['detection_classes'] ]
        detectionsLD = [{key : value[i] for key, value in detectionsDL.items()} for i in range(num_detections)]      
        #print(detectionsLD)
        row['figure_detections'] = detectionsLD
        #print(row['figure_detections'][0]['detection_scores'])
        row['figure_num_detections'] = num_detections
    figures_step1 = figures_step1.append(row)
if 'nameOfFigureAnnotations' in figures_step1.columns:
    print(figures_step1['dhash'])
    duplicates = getHashDuplicates(figures_step1 , figurelistannots, hashtype='dhash')
    

    

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

def ocr_replacestrings(replace_strings, text):
    for replace_string in replace_strings :
        text = text.replace(replace_string['search'],replace_string['replace'])
    return text


def ocr_post_processing_figure(row, detection, matchgroup):
    if detection['detection_classesname'] + '_raw' in row.keys():
        row[str(detection['detection_classesname']) + '_raw'] = ocr_replacestrings(replace_strings = row[str(detection['detection_classesname']) + '_replace_strings'], text = row[str(detection['detection_classesname']) + '_raw'] )



        if str(detection['detection_classesname']) + '_regex' in list(row.keys()):
            regex = re.compile(row[str(detection['detection_classesname']) + '_regex'])
            result = re.search(regex, row[str(detection['detection_classesname']) + '_raw'])
            if result is not None:
                row[str(detection['detection_classesname']) + '_clean'] = result.group(matchgroup)
                print('This was found in ' + detection['detection_classesname'] + '_clean: ' + row[str(detection['detection_classesname']) + '_clean'])


            elif result is None and row.get(str(detection['detection_classesname']) + '_searchextensive'):
                print('Yes this will use textall! ')
                row['textall_raw'] = ocr_replacestrings(replace_strings = row[str(detection['detection_classesname']) + '_replace_strings'], text = row['textall_raw'] )
                regex = re.compile(row[str(detection['detection_classesname']) + '_regex'])
                allresult = re.search(regex, row['textall_raw'])
                if allresult is not None:
                    row[str(detection['detection_classesname']) + '_clean'] = allresult.group(matchgroup)
                    print('This was found in textall ' + detection['detection_classesname'] + '_clean: ' + row[str(detection['detection_classesname']) + '_clean'])

            elif str(detection['detection_classesname']) == 'figureid' and row.get('infoframeid_searchextensive'):
                row[str(detection['detection_classesname']) + '_raw'] = ocr_replacestrings(replace_strings = row['infoframeid_replace_strings'], text = row[str(detection['detection_classesname']) + '_raw'] )
                regex = re.compile(row['infoframeid_regex'])
                antiresult = re.search(regex, row[str(detection['detection_classesname']) + '_raw'])
                if antiresult is not None:
                    print(str(detection['detection_classesname']) + '_raw' + ' matches to infoframeid_regex and will be written as such')
                    row['infoframeid_clean'] = antiresult.group(int(row['infoframeid_regexgroup']))
                    row[str(detection['detection_classesname']) + '_clean']  = None
            
            else:
                print('Fällt durchs Netz')





        else:
            print(str(detection['detection_classesname']) + '_regex' + ' not written in config - No regex filter will be applied')
     
        
    return row

def ocr_post_processing_textall(row):
    if 'textall_replace_strings' in row.keys():
        row['textall_raw'] = ocr_replacestrings(replace_strings = row['textall_replace_strings'], text = row['textall_raw'] )
    
    textall_regex = re.compile(row['textall_regex'])
    result = re.search(textall_regex, row['textall_raw'])
    if result is not None:
        row['textall_clean'] = result.group(1)
    else:
        row['textall_clean'] = 'none'


    return row


figures_step3 = pd.DataFrame()

for index, row in figures_step2.iterrows():
    if row['detection_classesname'] == 'infoframe':

        figure_imgnp = cv2.imread(row['figure_path'])
        figure_imgnp = ocr_pre_processing_page(figure_imgnp)
        row['textall_raw'] = pytesseract.image_to_string(figure_imgnp, config=row['textall_config'])
        print(row['textall_raw'])
        del figure_imgnp
        row = ocr_post_processing_textall(row)
    detections = []
    if type(row['bestfigure_detections']) == list:
        for detection in row['bestfigure_detections']:
            detection = cut_image_figure(row, detection)
            detection = ocr_pre_processing_figure(detection)

            if str(detection['detection_classesname']) + '_config' in list(row.keys()):
                row[str(detection['detection_classesname']) + '_raw'] = pytesseract.image_to_string(detection['imgnp'], config=row[str(detection['detection_classesname']) + '_config'])
                del detection['imgnp']
 
            else:
                row[str(detection['detection_classesname']) + '_raw'] = pytesseract.image_to_string(detection['imgnp'], config=row['pageid_config'])
   
            row['figure_detections'] = detections.append(detection)
            row = ocr_post_processing_figure(row, detection, matchgroup=int(row[detection['detection_classesname'] + '_regexgroup']))
    figures_step3 = figures_step3.append(row)






# %%
def generateShapebox(series):
    absbox = relToabsImagecoords(series['detection_boxes'], width= series['page_width'], height= series['page_height'])
    xmin, ymin, xmax, ymax = absbox
    shapebox = shapely.geometry.box(xmin, ymin, xmax, ymax, ccw=True)
    series['detection_shapebox'] = shapebox
    return series

def infosFromFrames(series, infoframes):
    for index, row in infoframes.iterrows():
        #if series['detection_shapebox'].within(row['detection_shapebox']):
        if (series['detection_shapebox'].intersection(row['detection_shapebox']).area/series['detection_shapebox'].area)*100 > 60:
            for info in row['frame_contains']:
                if info + '_clean' in row.keys():
                    print('Iswithin ' + str(row[info + '_clean']))
                    series[info + '_raw'] = row[info + '_raw']
                    series[info + '_clean'] = row[info + '_clean']
    return series

def seperateFramesFigures(group):
    infoframes = group[group['detection_classesname'] == 'infoframe']
    figures = group[group['detection_classesname'] != 'infoframe']
    #print(infoframes)
    figuresWithInfo = figures.apply(infosFromFrames, infoframes=infoframes, axis=1)
    return figuresWithInfo




figures_step3 = figures_step3.apply(generateShapebox, axis=1)
figures_geodf = gpd.GeoDataFrame( figures_step3,geometry= 'detection_shapebox')
figures_geodf2 = figures_geodf.groupby('page_path', as_index=False).apply(seperateFramesFigures).reset_index()
 
    




 


# %%
print(len(figures_geodf2['infoframeid_clean']), len(figures_geodf2['infoframeid_clean'].dropna()))
print(len(figures_geodf2['figureid_clean']), len(figures_geodf2['figureid_clean'].dropna()))
print(figures_geodf2[['infoframeid_clean','figureid_clean']])


# %%
def handleduplicate(df, field):
    #grouped = df.groupby(field)
    #df[field].fillna('no' + str(field))
    print(len(df[field]))
    groupsize = df.groupby( field ).size()
    cumcount = df.groupby( field ).cumcount()
    df = df.set_index(field)
    df['groupsize']= groupsize
    
    df = df.reset_index()
    df['cumcount']= cumcount
    #print(len(groupsize))
    #df = df.groupby(field).cumcount().to_frame('count').reset_index()
    #size = grouped.size()
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    #print(groupsize)
    print(df['cumcount'])
    #df = pd.concat((df, groupsize), axis=1)
    #df.rename( columns={0 :'groupsize'}, inplace=True )
    #print(groupsize)
    #
    #df = pd.concat([df, size], axis=1)
    #df.rename( columns={0 :'size'}, inplace=True )
    #print(df)

    #df.rename( columns={'Unnamed: 0':'count'}, inplace=True )
    #print(df.head())
    #df['count'] = '_' + str(df['count'])
    #print(df.columns)
    dfnew = pd.DataFrame()
    df['cumcount'].fillna(0, inplace=True)
    df['groupsize'].fillna(0, inplace=True)
    print(df[['cumcount','groupsize']])
    for index, row in df.iterrows():
        if int(row['cumcount']) == 0 and int(row['groupsize'])== 1:
            row[field + '_undup'] = row[field]
        else: 
            row[field + '_undup'] = str(row[field]) + '_' + str(row['cumcount'])
        dfnew = dfnew.append(row)




    #df[field + '_undup'] = df[field].astype(str).add('_') + df['count'].astype(str)
    #df = df.drop('count',axis=1)
    #print(df[field + '_undup'])
    #df = df.assign(field = df.groupby(field).cumcount().rename_axis([field,'count']))
    #df['HRID_undup'] += g.cumcount().add(g.HRID_undup.transform('count') == 1, '')
    #print( df[field + '_undup'])

    return dfnew

def humanreadID_drawing(Series):
    humanreadID = ''
    if 'patternHRID_drawing' in Series.keys():
        for element in Series['patternHRID_drawing']:
            if element['field'] in Series.keys():
                if isinstance(Series[element['field']], float)  and not math.isnan(Series[element['field']]):
                    Series[element['field']] = str(int(Series[element['field']]))
                humanreadID += str(Series[element['field']]) + str(element['seperator'])
            else:
                print('HRID Element: ' + element + ' does not exist.')
                humanreadID += element + '_'
    else:

        humanreadID = str(Series['figure_tmpid'])

    Series['HRID_drawing'] = humanreadID
    return Series



def humanreadID(Series):
 
    humanreadID = ''
    if 'patternHRID' in Series.keys():
        for element in Series['patternHRID']:
            if element['field'] in Series.keys():
                if isinstance(Series[element['field']], float) and not math.isnan(Series[element['field']]):
                    Series[element['field']] = str(int(Series[element['field']]))
                humanreadID += str(Series[element['field']]) + str(element['seperator'])
 
            else:
                print('HRID Element: ' + element['field']+ ' does not exist.')
                humanreadID += 'nan' + '_'
    else:

        humanreadID = str(Series['figure_tmpid'])
    Series['HRID'] = humanreadID
    #print(Series['HRID'])
    return Series



#print(figures_geodf2)
figures_step4 = figures_geodf2.apply(humanreadID, axis=1)
#print(figures_step4['HRID'])
figures_step4 = makeHash(figures_step4,'figure_path')
print(len(figures_step4))
figures_step4.drop_duplicates(subset='dhash', keep='first', inplace=True)
print(len(figures_step4))
figures_step4 = handleduplicate(figures_step4, field='HRID')
figures_step4 = figures_step4.apply(humanreadID_drawing, axis=1)
figures_step4 = handleduplicate(figures_step4, field='HRID_drawing')
print(type(figures_step4['pageid_clean'][0]))


# %%


#print(json.dumps(allDocs, indent=4, sort_keys=True))
#categoriesStructureDF = DocsStructure(allDocs)
#print(categoriesStructureDF)

#for index,row in categoriesStructureDF.iterrows():
    #print(index, row)
#groups = categoriesStruc
#def makeHash(df):
def imagestoreToDF(group):
    filelist = os.listdir(os.path.join(group['imagestore'][0], group['exportProject'][0])) 
    listofdicts = [{'id_existing': file, 'imagestore_path_existing': os.path.join(group['imagestore'][0], group['exportProject'][0], file ) } for file in filelist]
    return pd.DataFrame(listofdicts)



def getDocsByDF(df, id):
    doclist = []
    for index,row in df.iterrows():
        doc = {}
        doc['id'] = row[id]
        #doc['_deleted']
        doclist.append(doc)
    DOChull={}
    DOChull['docs']= doclist
    response = requests.post(pouchDB_url_bulkget , auth=auth, json= DOChull)
    result = json.loads(response.text)
    return result

def updateDuplicatesInDB(olddocs , newdocs):

    #doclistdelete = [doc['_deleted'] = True for doc in result['docs']]
    DOChull={}
    DOChull['docs']= doclistdelete
    answer = requests.post(pouchDB_url_bulk , auth=auth, json= DOChull)
    return answer
    
def getOldIds(series, olddocs_df):
    series['figure_tmpid'] = series['id_existing']
    series['rev'] = olddocs_df[olddocs_df['_id'] == series['id_existing']]['_rev']
    return series

def findExistingIdentifiers(newDOC, oldDOC):
    return sameIdentifierDOC

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]
def bulkSaveChanges(DOC, pouchDB_url_bulk, auth ):
    chunks = list(divide_chunks(DOC['docs'], 200))
    for chunk in chunks:
        #print(json.dumps(chunk, indent=4, sort_keys=True))
        chunkhull = {'docs':[]}
        chunkhull['docs'] = chunk
        answer = requests.post(pouchDB_url_bulk , auth=auth, json=chunkhull)
        print(answer)
    return print('Documents uploaded')
def pageCorrStat(df):
    return pageDistance
def pageCorrection(series):
    return series


    


def enterCreated(doc):
    now = datetime.now()
    daytoSec = now.strftime('%Y-%m-%dT%H:%M:%S')
    sec = "{:.3f}".format(Decimal(now.strftime('.%f')))
    doc['created'] = {}
    doc['created']['user'] = 'Script mhaibt'
    doc['created']['date'] = daytoSec + str(sec)[1:] + 'Z'
    return doc
def pathToStore(series):
    series['figure_imagestorepath'] = os.path.join(series['imagestore'], series['exportProject'], str(series['figure_tmpid']) )
    return series
def imageToStore(series):
    shutil.copyfile(str(series['figure_path']), series['figure_imagestorepath'])
    return series

def formatResults(results):
    docs = [resultitem['docs'][0]['ok'] for resultitem in results]
    return docs

def createCatalogDocs (series, docshull):
    doc = {}   
    doc['_id'] = series['catalog_uuid']
    #if 'rev' in series.keys():
        #doc['_rev'] = series['rev']
    doc['modified'] = []
    doc = enterCreated(doc)
    resource = doc['resource'] = {}
    resource['type'] = 'TypeCatalog'
    resource['id'] = series['catalog_uuid']
    resource['identifier'] = series['catalog_id']
    resource['shortDescription'] =  series['pub_quote']
    #resource['originalFilename'] = os.path.basename(series['figure_path'])
    #resource['width'] = int(series['figure_width'])
    #resource['height'] = int(series['figure_height'])
    #resource['id'] = str(series['figure_tmpid'])
    #resource['description'] = series['textall_raw'] + ' ' + series['pageinfo_raw']
    resource['literature'] = []
    if series['pub_key'] == 'ZenonID':
        resource['literature'].append({'zenonId' : str(series['pub_value']), 'quotation' : str(series['pub_quote']) })
    
    resource['relations'] = {}
    resource['relations']['isDepictedIn'] = []
    resource['relations']['isDepictedIn'].append(series['catalogcoverdrawing_uuid'])
    doc['resource'] = resource
    return docshull['docs'].append(doc)

def createDrawingDocs (series, docshull):
    doc = {}   
    doc['_id'] = str(series['figure_tmpid'])
    if 'rev' in series.keys():
        doc['_rev'] = series['rev']
    doc['modified'] = []
    doc = enterCreated(doc)
    resource = doc['resource'] = {}
    resource['type'] = 'Drawing'
    resource['identifier'] = series['HRID_drawing']
    resource['shortDescription'] = 'infoframeid: ' + str(series.get('infoframeid_clean')) + ' ' + series['detection_classesname'] + ' ' +series['HRID']
    resource['originalFilename'] = os.path.basename(series['figure_path'])
    resource['width'] = int(series['figure_width'])
    resource['height'] = int(series['figure_height'])
    resource['id'] = str(series['figure_tmpid'])
    if 'textall_clean' in series.keys():
        resource['description'] = 'textall: ' + str(series['textall_raw'] )
    if 'pageinfo_clean' in series.keys():
        resource['description'] += 'pageinfo: ' + str(series['pageinfo_clean'])
    if 'figureinfo_clean' in series.keys():
        resource['description'] += 'figureinfo: ' + str(series['figureinfo_clean'])

    resource['literature'] = []
    if series['pub_key'] == 'ZenonID':
        litdict = {'zenonId' : str(series['pub_value']), 'quotation' : str(series['pub_quote'])}
    if series.get('pageid_clean') is not None:
        litdict['page'] = str(series['pageid_clean'])
    if series.get("figureid_clean") is not None:
        litdict['figure'] = str(series['figureid_clean'])
    resource['literature'].append(litdict)
    doc['resource'] = resource
    return docshull['docs'].append(doc)

def createCatalogDrawingDocs (series, docshull):
    doc = {}   
    doc['_id'] = series['catalogcoverdrawing_uuid']
    #if 'rev' in series.keys():
        #doc['_rev'] = series['rev']
    doc['modified'] = []
    doc = enterCreated(doc)
    resource = doc['resource'] = {}
    resource['type'] = 'Drawing'
    resource['identifier'] = os.path.basename(series['catalogcoverpath'])
    resource['shortDescription'] =  'catalog cover'
    resource['originalFilename'] = os.path.basename(series['catalogcoverpath'])
    resource['width'] = int(series['catalogcover_width'])
    resource['height'] = int(series['catalogcover_height'])
    resource['id'] = str(series['catalogcoverdrawing_uuid'])
    resource['literature'] = []
    if series['pub_key'] == 'ZenonID':
        resource['literature'].append({'zenonId' : str(series['pub_value']), 'quotation' : str(series['pub_quote']) })
    resource['relations'] = {}
    resource['relations']['depicts']=[str(series['catalog_uuid'])]
    doc['resource'] = resource
    return docshull['docs'].append(doc)
def makeUuid (series, fieldname):
    series[fieldname] = str(uuid.uuid4())
    return series

def createTypeDocs (series, docshull, useuuid, useidentifier, liesWithin= None, isDepictedIn= None):
    doc = {}   
    doc['_id'] = str(series[useuuid])
    if 'rev' in series.keys():
        doc['_rev'] = series['rev']
    doc['modified'] = []
    doc = enterCreated(doc)
    resource = doc['resource'] = {}
    resource['type'] = 'Type'
    resource['id'] = str(series[useuuid])
    resource['identifier'] = str(series[useidentifier])
    #resource['id'] = str(series['catalogcoverdrawing_uuid'])
    resource['shortDescription'] = 'infoframeid: ' + str(series.get('infoframeid_clean')) + ' ' + series['detection_classesname'] + ' ' +series['HRID']
    #resource['originalFilename'] = os.path.basename(series['figure_path'])
    #resource['width'] = int(series['figure_width'])
    #resource['height'] = int(series['figure_height'])
    #resource['id'] = str(series['figure_tmpid'])
    if 'textall_clean' in series.keys():
        resource['description'] = 'textall: ' + str(series['textall_raw'] )
    if 'pageinfo_clean' in series.keys():
        resource['description'] += 'pageinfo: ' + str(series['pageinfo_clean'])
    if 'figureinfo_clean' in series.keys():
        resource['description'] += 'figureinfo: ' + str(series['figureinfo_clean'])
    resource['literature'] = []
    if series['pub_key'] == 'ZenonID':
        litdict = {'zenonId' : str(series['pub_value']), 'quotation' : str(series['pub_quote'])}
    if series.get("pageid_clean") is not None:
        litdict['page'] = str(series['pageid_clean'])
    if series.get("figureid_clean") is not None:
        litdict['figure'] = str(series['figureid_clean'])
    resource['literature'].append(litdict)
    resource['relations'] = {}
    if liesWithin:
        resource['relations']['liesWithin'] = [] 
        resource['relations']['liesWithin'].append(str(series[liesWithin]))
    if isDepictedIn:
        resource['relations']['isDepictedIn'] = []
        resource['relations']['isDepictedIn'].append(str(series[isDepictedIn]))
    doc['resource'] = resource
    
    return docshull['docs'].append(doc)
def attachImagesToDoc(series, isDepictedIn, DOC):
    if DOC['resource']['relations'].get('isDepictedIn') is None:
        DOC['resource']['relations']['isDepictedIn'] = []
    return DOC['resource']['relations']['isDepictedIn'].append(str(series[isDepictedIn]))


def getCatalogCover(df):
    if 'catalog_cover_pdfpage' in df.columns:
        firstrow = df.iloc[0]
        convert_from_path(firstrow['pubpdf_path'], fmt='png', thread_count=1, output_file= 'CatalogCover_'+firstrow['catalog_id'], first_page=int(firstrow['catalog_cover_pdfpage']),dpi=200, single_file=True, paths_only=False, use_pdftocairo=True, output_folder=firstrow['pubfolder_path'])
        
        firstrow['catalogcoverpath']=os.path.join(firstrow['pubfolder_path'],'CatalogCover_'+firstrow['catalog_id']+'.png')
        img = cv2.imread(firstrow['catalogcoverpath'])
        height, width, channels = img.shape
        firstrow['catalogcover_width'] = width
        firstrow['catalogcover_height'] = height

    return firstrow



def createRelationMapping(docs, relation):
    mapdictlist = []
    for doc in docs:
        if relation in doc['resource']['relations'].keys():
            for relateddocid in doc['resource']['relations'][relation]:
                mapdict = {}
                mapdict[relation + 'id']= relateddocid
                mapdict['docid']= doc['_id']
                mapdictlist.append(mapdict)
    
    return mapdictlist

def writeReverseRelation(df, mapdicts, relation, rev_relation):
    newdf = pd.DataFrame()
    for index,row in df.iterrows():
        docids = [item['docid'] for item in mapdicts if item[relation + 'id']==row['_id']]
        if docids:
            if not 'relations' in row.keys():
                row['relations']= {}
            row['relations'][rev_relation]= docids
            newdf= newdf.append(row)
    return newdf

def DOCtoDF(DOC):
    DFdocs = pd.DataFrame(DOC)
    print(DFdocs.columns)
    DFdocs = DFdocs.drop('resource', axis=1)
    DFresources = pd.DataFrame([i['resource'] for i in DOC])
    for col in DFdocs.columns:
        DFresources[str(col)]=DFdocs[str(col)]
    docfields = DFdocs.columns

    return DFresources, docfields
   

def DFtoDOC(DFresources, docfields):
    DF = DFresources
    columns = [i for i in DFresources.columns if not i in docfields]
    #print(columns)
    DOC = []
    for index,row in DF.iterrows():
        #print(type(row[columns]))
        #dd = defaultdict(list)
        #print('Before DROP:')  
        cleanrow=row[columns].dropna() 
        #äprint(cleanrow)
        row['resource']= cleanrow.to_dict()
        #row['resource'] = {k: row['resource'][k] for k in row['resource'] if not isnan(row['resource'][k])}
        #print(row)
        #print('After DROP:')
        row = row.drop(columns)
        #print(row)

        DOC.append(row.to_dict())
    DOChull={}
    DOChull['docs']=DOC
    return DOChull


projects_grouped = figures_step4.groupby('exportProject')

for name, group in projects_grouped:
    #print(group['catalog_id'])
    group_drawingsclean = group
    print(type(group))
    db_name = name
    auth = ('' , group['authProjectPass'][0])
    print(auth)
    db_url = group['db_url'][0]
    print(db_url)
    pouchDB_url_find = f'{db_url}/{db_name}/_find'
    pouchDB_url_put = f'{db_url}/{db_name}/'
    pouchDB_url_bulk = f'{db_url}/{db_name}/_bulk_docs'
    pouchDB_url_bulkget = f'{db_url}/{db_name}/_bulk_get'
    idaifieldconfigpath = Path('/home/idaifieldConfigs')
    #group.apply(createDrawingDocs, axis=1)
    imagestoredf = imagestoreToDF(group)
    
    if not imagestoredf.empty:
        imagestoredf = makeHash(imagestoredf, 'imagestore_path_existing')
        
        #group = makeHash(group,'figure_path')
        duplicates_df = getHashDuplicates(group, imagestoredf, 'dhash')
        print(len(duplicates_df))

        if 'handleDuplicateImages' in  duplicates_df.columns and not duplicates_df.empty:
            config_grouped = duplicates_df.groupby('handleDuplicateImages')
            for config_name, config_group in config_grouped:        
                if config_name == 'skip':
                    continue

                elif config_name == 'update':
                    toUpdateDocs = getDocsByDF(config_group, 'id_existing')
                    #print(json.dumps(toUpdateDocs, indent=4, sort_keys=True))
                    toUpdateDocs_formated = formatResults(toUpdateDocs['results'])
                    toUpdateDocs_df = pd.DataFrame(toUpdateDocs_formated)
                    config_group.apply(getOldIds, olddocs_df=toUpdateDocs_df, axis=1)
                    updateWithDocs = {"docs" : []}
                    config_group.apply(createDrawingDocs, docshull=updateWithDocs,  axis=1)
                    config_group = config_group.apply(pathToStore, axis=1)
                    config_group.apply(imageToStore, axis=1)
                    bulkSaveChanges(updateWithDocs, pouchDB_url_bulk, auth )
            group_drawingsclean = pd.concat([group, duplicates_df]).drop_duplicates(subset='figure_path' ,keep=False)
    cleannewDocs = {"docs" : []}
    group_drawingsclean.apply(createDrawingDocs, docshull=cleannewDocs,  axis=1)
    
    group_drawingsclean = group_drawingsclean.apply(pathToStore, axis=1)
    
    group_drawingsclean.apply(imageToStore, axis=1)
    
    pub_grouped = group.groupby('catalog_id')
    
    for pub_name, pub_group in pub_grouped :
        #print(pub_name)
        #print(pub_group)
        catalogrow = getCatalogCover(pub_group)
        catalogrow['catalogcoverdrawing_uuid'] = str(uuid.uuid4())
        catalogrow['catalog_uuid'] = str(uuid.uuid4())
        CatalogDrawingDocs = {"docs" : []}
        createCatalogDrawingDocs (catalogrow, CatalogDrawingDocs)
        bulkSaveChanges(CatalogDrawingDocs, pouchDB_url_bulk, auth )
        shutil.copyfile(str(catalogrow['catalogcoverpath']), os.path.join(catalogrow['imagestore'], catalogrow['exportProject'], os.path.basename(catalogrow['catalogcoverdrawing_uuid']).replace('.png','') ))
        
        pub_group['catalog_uuid'] = catalogrow['catalog_uuid'] 
        CatalogDocs = {"docs" : []}
        createCatalogDocs(catalogrow, docshull=CatalogDocs)
        
        TypeDocs = {"docs" : []}
        #print(type(pub_group['infoframe_becomesCategory']))
        if 'Type' in pub_group.iloc[0]['infoframe_becomesCategory']:

            for infoframeid_name, infoframeid_group in pub_group.groupby('infoframeid_clean'):
                #infoframeid_group = infoframeid_group.apply(makeUuid, fieldname='type_id', axis=1)
                infoframeid_group = makeUuid( infoframeid_group, fieldname='type_id')  
                #print(infoframeid_group) 
                infoframeid_group.head(1).apply(createTypeDocs, docshull=TypeDocs, useuuid='type_id', useidentifier='HRID' , liesWithin='catalog_uuid', axis=1)
                if 'Type' in infoframeid_group.iloc[0]['figure_becomesCategory']:
                    infoframeid_group = infoframeid_group.apply(makeUuid, fieldname='subtype_id', axis=1) 
                    infoframeid_group.apply(createTypeDocs, docshull=TypeDocs, useuuid='subtype_id', useidentifier='HRID_undup', liesWithin='type_id', isDepictedIn='figure_tmpid', axis=1)
                elif 'Drawing' in infoframeid_group.iloc[0]['figure_becomesCategory']:
                    infoframeid_group.apply(attachImagesToDoc, isDepictedIn='figure_tmpid', DOC= TypeDocs['docs'][-1], axis=1)
                    
        else:
            if 'Type' in pub_group.iloc[0]['figure_becomesCategory']:
                pub_group = pub_group.apply(makeUuid, fieldname='type_id', axis=1)  
                pub_group.apply(createTypeDocs, docshull=TypeDocs, useuuid='type_id', useidentifier='HRID_undup' , liesWithin='catalog_uuid', isDepictedIn='figure_tmpid', axis=1)
            elif 'Drawing' in pub_group.iloc[0]['figure_becomesCategory']:
                continue



        
        DrawingsDF, docfields = DOCtoDF(cleannewDocs['docs'])
        print('length of TypeDoc: ', len(TypeDocs['docs']))
        isDepictedIn_map = createRelationMapping(TypeDocs['docs'], relation='isDepictedIn')
        print('length of mapping: ', len(isDepictedIn_map))
        print(isDepictedIn_map)
        DrawingsDF = writeReverseRelation(DrawingsDF, mapdicts=isDepictedIn_map, relation='isDepictedIn', rev_relation='depicts')
        #DFresources = removeUndepicts(DFresources)
        print('length with depicts: ', len(DrawingsDF))
        #DrawingsDF = DrawingsDF.apply(addModifiedEntry, axis=1)
        DrawingsDOC = DFtoDOC(DrawingsDF, docfields)
        
        
        bulkSaveChanges(DrawingsDOC, pouchDB_url_bulk, auth )
        bulkSaveChanges(CatalogDocs, pouchDB_url_bulk, auth )
        bulkSaveChanges(TypeDocs, pouchDB_url_bulk, auth )



    #print(json.dumps(cleannewDocs, indent=4, sort_keys=True))
    

    #print(duplicateImages[['HRID','hash_av']])



    #allDocs = getAllDocs(group['authProject'][0], pouchDB_url_find)textal
    #allDocs = allDocs['docs']
    #allDocs = [doc for doc in  allDocs if 'type' in doc['resource'].keys()]
    #existingDrawings = selectOfResourceTypes(['Drawing'], allDocs)
    #findExistingIdentifiers(DrawingDocs, existingDrawings)


    #DocHull['docs'] = DrawingDocs
    #print(json.dumps(DrawingDocs, indent=4, sort_keys=True))
    #figures_step4.apply(imageToStore)
    #figures_step5 = figures_step4.apply(pathToStore, axis=1)
    


    #print(figures_step5['hash_av'])
#figures_step6.apply(imageToStore, axis=1)


#duplicateimages = findDuplicateImages()
#print(duplicateimages)

#bulkSaveChanges(DocHull, pouchDB_url_bulk, auth )
#print(AllDocs)



    
    

#cleanrow=row[columns].dropna() 
#äprint(cleanrow)
#row['resource']= cleanrow.to_dict()

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
figures_step4.to_csv(CSVOUT)
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



