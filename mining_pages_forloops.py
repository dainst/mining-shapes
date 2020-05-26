
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image



import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import pytesseract
import shutil
import json
import math




import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util, label_map_util, ops
from collections import namedtuple, OrderedDict

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
  


# This is needed to display the images.




#from utils import visualization_utils as vis_util
inputdirectory = '/home/images/apply'
GRAPH = '/frozen_inference_graph.pb'
LABELS = '/label_map.pbtxt'
PAGE_MODEL = '/home/models/inference_graph_mining_pages_v8'
FIGID_MODEL = '/home/models/inference_graph_figureid_v1'
OUTPATH = '/home/images/OUTPUT/'

def get_labelmap_as_df(PATH_TO_LABELS): 
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    category_index = pd.DataFrame(category_index).T
    category_index = category_index.rename(columns={'id':'detection_classes', 'name':'detection_classesname'})
    return category_index

def get_figid_labelmap_as_df(PATH_TO_LABELS): 
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    category_index = pd.DataFrame(category_index).T
    category_index = category_index.rename(columns={'id':'figid_detection_classes', 'name':'figid_detection_classesname'})
    return category_index

def provide_pagelist(inputdirectory, pagelist):

    for pub_id in os.listdir(inputdirectory): 
        pub_key, pub_value = pub_id.split('_')
        pub_path = os.path.join(inputdirectory, pub_id)
        pub = {}
        pub['pub_key'] = pub_key
        pub['pub_value'] = pub_value                   
        for page_imgname in os.listdir(pub_path) :
            
            if page_imgname.endswith((".png",".jpg")) and 'Thumbs' not in page_imgname :                
                page = pub
                page_path = os.path.join(pub_path, page_imgname)
                page['page_imgname'] = page_imgname
                page['page_path'] = page_path               
                pagelist.append(page.copy())
    return pd.DataFrame(pagelist)
        
        
    
                
            
    




def run_inference_for_single_image(image, graph):
    
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def run_inference_for_series(series, graph):
    image = series['page_imgnp']
    
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

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

def run_inference_for_figureseries(series, graph):
    image = series['figure_imgnp']
    
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

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


    
def extract_detections_pageold(page, output_dict):
    page_detections = []
    N = len(output_dict['detection_boxes'])    
    for i in range(N):
        detection = page
        box = output_dict['detection_boxes'][i]
        ymin, xmin, ymax, xmax = box
        detection['bbox_xmin'] = int((xmin)*page_width)
        detection['bbox_ymin'] = int((ymin)*page_height)
        detection['bbox_xmax'] = int((xmax)*page_width)
        detection['bbox_ymax'] = int((ymax)*page_height)
        detection['detection_score'] = output_dict['detection_scores'][i]
        detection_class = output_dict['detection_classes'][i]
        if detection_class == 1:
            detection['detection_class'] = 'vesselprofilefigure'
        elif detection_class == 2:
            detection['detection_class'] = 'pageid'            
        elif detection_class == 3:
            detection['detection_class'] = 'pageinfo'
        page_detections.append(detection.copy())
    return page_detections


def extract_detections_page(df, category_index):
    page_detectionsaslist = pd.DataFrame(df['page_detections'].tolist()).reindex(df.index)
    df = pd.concat([df,page_detectionsaslist], axis=1)
    all_detections = pd.DataFrame()
    N = df['num_detections'].max()
    for i in range(0,N):
        detection = df.applymap(lambda x: x[i] if type(x)== np.ndarray else x)
        all_detections = all_detections.append(detection)
    
    all_detections = all_detections.merge(category_index, on=['detection_classes'], how='left')
    return all_detections





def extract_detections_figureid(df, figid_category_index):
    figid_detections = pd.DataFrame(df['figid_detections'].tolist()).reindex(df.index)    
    detections = figid_detections.applymap(lambda x: x[0] if type(x)== np.ndarray else x).reindex(figid_detections.index)    
    detections = detections.merge(figid_category_index, on=['figid_detection_classes'], how='left')
    df = pd.concat([df,detections])
    return figid_detections


    #df = pd.concat([df, figure_detectionsaslist], axis=1)
    
    
    #all_detections = pd.DataFrame()
    #N = 1
    #for i in range(0,N):

def extract_detections_figureidv2(df, figid_category_index):
    figid_detectionsdict = df['figid_detections']
    df['figid_detection_scores'] = figid_detectionsdict['figid_detection_scores'][0]
    df['figid_detection_boxes'] = figid_detectionsdict['figid_detection_boxes'][0]
    df['figid_detection_classes'] = figid_detectionsdict['figid_detection_classes'][0]
    df['figid_num_detections'] = figid_detectionsdict['figid_num_detections']
    #figid_detections = pd.DataFrame(df['figid_detections'].tolist()).reindex(df.index)    
    #detections = figid_detections.applymap(lambda x: x[0] if type(x)== np.ndarray else x).reindex(figid_detections.index)    
    #df = df.merge(figid_category_index, on=['figid_detection_classes'], how='left')
    #df = pd.concat([df,detections])
    return df


    #df = pd.concat([df, figure_detectionsaslist], axis=1)
    
    
    #all_detections = pd.DataFrame()
    #N = 1
    #for i in range(0,N):





def filter_bestdetections_max1(all_detections, classlist, lowest_score):
    pageids = (all_detections[(all_detections['detection_classesname'].isin(classlist)) &
      (all_detections['detection_scores'] >= lowest_score)])
    bestdetections = (pageids[pageids['detection_scores'] == pageids
                     .groupby(['pub_key','pub_value', 'page_imgname', 'detection_classesname'])['detection_scores'].transform('max')])
    return bestdetections

def filter_bestdetections(all_detections, classlist, lowest_score):
    bestdetections = (all_detections[(all_detections['detection_classesname'].isin(classlist)) &
      (all_detections['detection_scores'] >= lowest_score)])
    return bestdetections

def filter_bestdetections_figid(all_detections, classlist, lowest_score):
    figids = (all_detections[(all_detections['figid_detection_classesname'].isin(classlist)) &
      (all_detections['figid_detection_scores'] >= lowest_score)])
    bestdetections = (figids[figids['figid_detection_scores'] == figids
                     .groupby(['pub_key','pub_value', 'page_imgname', 'figure_tmpid'])['figid_detection_scores'].transform('max')])
    return bestdetections





def cut_image(dataframe):
    page_imgnp = cv2.imread(dataframe['page_path'])
    box = dataframe['detection_boxes']
    ymin, xmin, ymax, xmax = box
    bbox_xmin = int((xmin)*dataframe['page_width'])
    bbox_ymin = int((ymin)*dataframe['page_height'])
    bbox_xmax = int((xmax)*dataframe['page_width'])
    bbox_ymax = int((ymax)*dataframe['page_height'])
    bbox_np = page_imgnp[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]
    
    return bbox_np

def cut_image_savetemp(dataframe):
    page_imgnp = cv2.imread(dataframe['page_path'])
    box = dataframe['detection_boxes']
    ymin, xmin, ymax, xmax = box
    bbox_xmin = int((xmin)*dataframe['page_width'])
    bbox_ymin = int((ymin)*dataframe['page_height'])
    bbox_xmax = int((xmax)*dataframe['page_width'])
    bbox_ymax = int((ymax)*dataframe['page_height'])
    bbox_np = page_imgnp[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]
    figure_height, figure_width, figure_channel = bbox_np.shape
    dataframe['figure_height'] = figure_height
    dataframe['figure_width'] = figure_width
    dataframe['figure_channel'] = figure_channel
    dataframe['figure_imgnp'] = bbox_np
    dataframe['figure_tmpid'] = dataframe.name
    dataframe['figure_path'] = OUTPATH + str(dataframe['pub_key']) + '_' + str(dataframe['pub_value']) + '_' + 'tempid' + str(dataframe['figure_tmpid']) + '.png'
    cv2.imwrite( str(dataframe['figure_path']), bbox_np )
    
    return dataframe
 
def cut_image_figid(dataframe):
    figure_imgnp = cv2.imread(dataframe['figure_path'])
    box = dataframe['figid_detection_boxes']
    ymin, xmin, ymax, xmax = box
    bbox_xmin = int((xmin)*dataframe['figure_width'])
    bbox_ymin = int((ymin)*dataframe['figure_height'])
    bbox_xmax = int((xmax)*dataframe['figure_width'])
    bbox_ymax = int((ymax)*dataframe['figure_height'])
    bbox_np = figure_imgnp[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]
    
    return bbox_np                               
 
def ocrPreProcessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image,5)
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    row, col = image.shape[:2]
    bottom = image[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]
    
    bordersize = 20
    image = cv2.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )
    return image



def ocrPostProcessing_Pageid(row):
    pageid_raw = row['pageid_raw']
    row['pageid_int'] = [int(s) for s in pageid_raw.split() if s.isdigit()]
    return row


def  merge_info(all_detections, bestpages_result ):
    for detection_classesname in bestpages_result.detection_classesname.unique():
        #print (detection_class)
        selected_info = bestpages_result[bestpages_result['detection_classesname'] == detection_classesname]
        newinfo_name = detection_classesname + '_raw'
        selected_info = selected_info.rename(columns={'newinfo' : newinfo_name })
        all_detections = all_detections.merge(selected_info[[newinfo_name,'pub_key','pub_value','page_imgname']], on=['pub_key','pub_value','page_imgname'], how='left')
                    
    return all_detections
       
def load_figure(series):
    figure_imgnp = cv2.imread( str(series['figure_path']))
    figure_height, figure_width, figure_channel = figure_imgnp.shape
    series['figure_width'] = figure_width
    series['figure_height'] = figure_height
    series['figure_channel'] = figure_channel
    series['figure_imgnp'] = figure_imgnp
        
    return series

def load_page(series):
    page_imgnp = cv2.imread( str(series['page_path']))

    page_height, page_width, page_channel = page_imgnp.shape
    series['page_width'] = page_width
    series['page_height'] = page_height
    series['page_channel'] = page_channel
    series['page_imgnp'] = page_imgnp
        
    return  series

def class_text_to_int(row_label):
    if row_label == 'vesselprofilefigure':
        return 1
    if row_label == 'pageid':
        return 2
    if row_label == 'pageinfo':
        return 3
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def createFIND_JSONL(df, file):   
    FIND_template = '{"category":"","identifier":"","relations":{"isChildOf":"","isDepictedIn":[],"isInstanceOf":[]}}'
    FIND = json.loads(FIND_template)
    print(FIND)
    #print(df['figure_tmpid'])
    FIND["identifier"] = 'Find_' + str(df['figure_tmpid'])
    FIND["category"] = 'Pottery'
    
    #FIND["shortDescription"] = str(df['figid_raw'])
    relations = FIND["relations"]
    relations["isChildOf"] = 'Findspot_refferedtoin_' + str(df['pub_key']) + '_' + str(df['pub_value'])
    InstanceOfList = relations["isInstanceOf"]
    typename = 'Type_' + str(df['pub_key']) + '_' + str(df['pub_value']) + '_' + 'tempid' + str(df['figure_tmpid'])
    InstanceOfList.append(typename)
    #print(type(relations['isDepictedIn']))
    depictedInList = relations["isDepictedIn"]
    imagename = str(df['pub_key']) + '_' + str(df['pub_value']) + '_' + 'tempid' +  str(df['figure_tmpid'] )+ '.png'
    depictedInList.append(imagename)
    print(type(FIND))
    json.dump(FIND, file)
    file.write("\n")
  
def createTYPE_JSONL(df, file):   
    TYPE_template = '{"category":"","identifier":"","relations":{"isChildOf":""}}'
    TYPE = json.loads(TYPE_template)
    #print(df['figure_tmpid'])
    TYPE["identifier"] = 'Type_' + str(df['pub_key']) + '_' + str(df['pub_value']) + '_' + 'tempid' + str(df['figure_tmpid'])
    TYPE["category"] = 'Type'
    relations = TYPE["relations"]
    relations["isChildOf"] = 'Catalog_' + str(df['pub_key']) + '_' + str(df['pub_value'])
    #print(type(relations['isDepictedIn']))
    json.dump(TYPE, file)
    file.write("\n")
    
def createDRAWING_JSONL(df, file):   
    DRAWING_template = '{"category":"","identifier":"", "description":"none","literature":[{"quotation":"none","zenonId":""}]}'
    DRAWING = json.loads(DRAWING_template)
    #print(df['figure_tmpid'])
    DRAWING["identifier"] = str(df['pub_key']) + '_' + str(df['pub_value']) + '_' + 'tempid' +  str(df['figure_tmpid'] )+ '.png'
    DRAWING["category"] = 'Drawing'
    DRAWING["description"] = 'PAGEID_RAW: ' + str(df['pageid_raw']) + 'PAGEINFO_RAW: ' + str(df['pageinfo_raw'])
    literature = DRAWING["literature"]
    literature0 = literature[0]
    literature0['zenonId'] = str(df['pub_key']) + '_' + str(df['pub_value'])
    
    literature0['quotation'] = str(df['figid_raw'])
    if not literature0['quotation']:
        literature0['quotation'] = 'no page detected'

    #print(DRAWING(relations['isDepictedIn']))
    json.dump(DRAWING, file)
    file.write("\n")

def createCATALOG_JSONL(df, file):
    CATALOG_template = '{"category":"","identifier":"","shortDescription":"In what aspects differ types in this catalog and what do they have in common?", "relations":{"isDepictedIn":[]}}'
    CATALOG = json.loads(CATALOG_template)
    #print(df['figure_tmpid'])
    CATALOG["identifier"] = 'Catalog_' + str(df['pub_key']) + '_' + str(df['pub_value'])
    relations = CATALOG["relations"]
    depictedInList = relations["isDepictedIn"]
    depictedInList.append('Catalogcover_' + str(df['pub_key']) + '_' + str(df['pub_value']) + '.png')
    CATALOG["category"] = 'TypeCatalog'
    json.dump(CATALOG, file)
    file.write("\n")

def createTRENCH_JSONL(df, file):
    TRENCH_template = '{"category":"","identifier":"","shortDescription":"Where have the Objects been found?"}'
    TRENCH = json.loads(TRENCH_template)
    #print(df['figure_tmpid'])
    TRENCH["identifier"] = 'Findspot_refferedtoin_' + str(df['pub_key']) + '_' + str(df['pub_value'])
    TRENCH["category"] = 'Trench'
    json.dump(TRENCH, file)
    file.write("\n")




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

    for index, row in group.object.iterrows():
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



    for index, row in group.object.iterrows():
        
        filename = group.filename.encode('utf8')
        box = row['figid_detection_boxes']
        ymin, xmin, ymax, xmax = box
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(str(row['figid_detection_classesname']).encode('utf8'))
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
classlist= ['pageid', 'pageinfo']
figureclasslist= ['vesselprofilefigure']
figureidclasslist= ['figureid']
pageid_config = r'--psm 6 -c load_system_dawg=0 load_freq_dawg=0'
pagelist = []
pagelist = provide_pagelist(inputdirectory, pagelist)





detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PAGE_MODEL + GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
  

try:
    with detection_graph.as_default():
        with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
                e = 0

                all_detections_step1 = pd.DataFrame()

                for index, row in pagelist.iterrows():
           
                  img = load_page(row)
                  result = run_inference_for_series(img, detection_graph)
                  result.drop("page_imgnp", inplace=True)
                  all_detections_step1 = all_detections_step1.append(result)
                                
                
                
 
               
except Exception as e:
    print(e)

category_index = get_labelmap_as_df(PAGE_MODEL + LABELS)
all_detections_step2 = extract_detections_page(all_detections_step1, category_index=category_index)

bestpages = filter_bestdetections_max1(all_detections_step2, classlist , 0.7 )

pageid_raw = pd.DataFrame()
for index, row in bestpages.iterrows():

    img = cut_image(row)
    img2 = ocrPreProcessing(img)
    result = pytesseract.image_to_string(img2, config=pageid_config)
    row['newinfo'] = result

    pageid_raw = pageid_raw.append(row)




#bestpages_result = pd.concat([bestpages, pageid_raw], axis=1)
all_detections_step3 = merge_info(all_detections_step2, pageid_raw)

figures = filter_bestdetections(all_detections_step3, figureclasslist , 0.7 )


detection_figureid_graph = tf.Graph()
with detection_figureid_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(FIGID_MODEL + GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

try:
    with detection_figureid_graph.as_default():
        with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                
                tensor_dict = {}
                for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
                e = 0
                
                
                #print (figures_imgnp)
                figures_step1 = pd.DataFrame()

                for index, row in figures.iterrows():
                             


                  img = cut_image_savetemp(row)
                  result = run_inference_for_figureseries(img, detection_figureid_graph)
                  result.drop("figure_imgnp", inplace=True)
                  figures_step1 = figures_step1.append(result)
                            


             
                                       
                                                  
                
                

except Exception as e:
    print(e)

figid_category_index = get_figid_labelmap_as_df(FIGID_MODEL + LABELS)
#figures_step2 = figures_step1.merge(figid_category_index, on=['figid_detection_classes'], how='left')
#figid_detections= extract_detections_figureid(figures_step1, figid_category_index= figid_category_index)
figid_detections= figures_step1.apply(extract_detections_figureidv2, figid_category_index= figid_category_index, axis=1)
figures_step2 = figid_detections.merge(figid_category_index, on=['figid_detection_classes'], how='left')

#bestfigid = filter_bestdetections_figid(figures_step2, figureidclasslist , 0.6 )
figures_step3 = pd.DataFrame()
for index, row in figures_step2.iterrows():
    img = cut_image_figid(row)
    img2 = ocrPreProcessing(img)
    row['figid_raw'] = pytesseract.image_to_string(img2, config=pageid_config)
    figures_step3= figures_step3.append(row)



with open(OUTPATH + 'catalogs.jsonl', 'w') as f:
    pubs = figures_step3[['pub_key','pub_value']].drop_duplicates()
    pubs.apply(createCATALOG_JSONL, file = f, axis = 1)
with open(OUTPATH + 'trenches.jsonl', 'w') as f:
    pubs = figures_step3[['pub_key','pub_value']].drop_duplicates()
    pubs.apply(createTRENCH_JSONL, file = f, axis = 1)
with open(OUTPATH + 'types.jsonl', 'w') as f:
    figures_step3.apply(createTYPE_JSONL, file = f, axis = 1)
with open(OUTPATH + 'finds.jsonl', 'w') as f:
    figures_step3.apply(createFIND_JSONL, file = f, axis = 1)
with open(OUTPATH + 'drawings.jsonl', 'w') as f:
    figures_step3.apply(createDRAWING_JSONL, file = f, axis = 1)
    #for i in figures_step8.index:
        #FIND = createFIND_JSONL(figures_step8[i])
        #f.write("%s\n" % FIND)
    

    
#df['json'] = df.apply(lambda x: x.to_json(), axis=1)    

#with jsonlines.open(, 'w') as outfile:
            #outfile.write(figures_jsonl)

TFRECORDOUT = OUTPATH + 'mining_pages.tfrecord'
writer = tf.python_io.TFRecordWriter(TFRECORDOUT )

shutil.copyfile(PAGE_MODEL + LABELS, OUTPATH + 'pages_label_map.pbtxt' )

mining_pages_detections = figures_step3.append(bestpages)
grouped = split(mining_pages_detections, 'page_path')

for group in grouped:
    tf_example = create_tf_example(group,  TFRECORDOUT)
    writer.write(tf_example.SerializeToString())
  
writer.close()

TFRECORDOUT = OUTPATH + 'mining_figures.tfrecord'
writer = tf.python_io.TFRecordWriter(TFRECORDOUT )

shutil.copyfile(FIGID_MODEL + LABELS, OUTPATH + 'figures_label_map.pbtxt' )
figids = figures_step3[figures_step3.figid_detection_boxes.notnull()]

figsgrouped = split(figids, 'figure_path')

for group in figsgrouped:
    figtf_example = create_tf_figid(group,  TFRECORDOUT)
    writer.write(figtf_example.SerializeToString())
  
writer.close() 
CSVOUT = OUTPATH + 'mining_pages_allinfo.csv'
figures_step3.to_csv(CSVOUT)




