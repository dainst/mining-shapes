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
import jsonlines
import re



import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
  


# This is needed to display the images.


from utils import label_map_util

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

def provide_pagelist(inputdirectory, pagelist = []):

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


def extract_detections_page(output_dict):
    all_detections = pd.DataFrame()
    N = output_dict['num_detections'].max()
    for i in range(0,N):
        detection = output_dict.applymap(lambda x: x[i] if type(x)== np.ndarray else x)
        all_detections = all_detections.append(detection)
    return all_detections





def extract_detections_figureid(output_dict):
    all_detections = pd.DataFrame()
    N = 3
    for i in range(0,N):
        detection = output_dict[['figid_detection_boxes','figid_detection_scores','figid_detection_classes','figid_num_detections','pub_key','pub_value', 'page_imgname','figure_tmpid']].applymap(lambda x: x[i] if type(x)== np.ndarray else x)
        all_detections = all_detections.append(detection)

    return all_detections





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
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
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




def create_tf_example(group):

    path = group['page_path']

    with tf.gfile.GFile(path, 'rb') as fid:
        encoded_jpg = fid.read()
    #encoded_jpg_io = io.BytesIO(encoded_jpg)
    #image = Image.open(encoded_jpg_io)
    width = int(group['page_width'])
    height = int(group['page_height'])

    filename = group.page_path.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    box = group['figid_detection_boxes']
    ymin, xmin, ymax, xmax = box

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

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
pagelist = provide_pagelist(inputdirectory)

pagelist = pagelist.apply(load_page, axis=1)



# %%
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


                
                output_dict_raw = pagelist['page_imgnp'].apply(run_inference_for_single_image, graph='detection_graph')
                
                
                
 
               
except Exception as e:
    print(e)

pagelist = pagelist.drop(columns='page_imgnp') 

output_dict = pd.DataFrame(output_dict_raw.tolist()).reindex(output_dict_raw.index)
pagelist = pd.concat([pagelist, output_dict], axis=1)
all_detections_step1 = extract_detections_page(pagelist)
# %%
category_index = get_labelmap_as_df(PAGE_MODEL + LABELS)
all_detections_step2 = all_detections_step1.merge(category_index, on=['detection_classes'], how='left')



bestpages = filter_bestdetections_max1(all_detections_step2, classlist , 0.7 )
pageid_imgnp = bestpages.apply(cut_image, axis=1)

pageid_imgnp_mod = pageid_imgnp.apply(ocrPreProcessing)

pageid_raw = pageid_imgnp_mod.apply(pytesseract.image_to_string, config=pageid_config)

pageid_raw = pd.Series(pageid_raw, name='newinfo')


bestpages_result = pd.concat([bestpages, pageid_raw], axis=1)
all_detections_step3 = merge_info(all_detections_step2, bestpages_result)
# %%
#all_detections_step3 = all_detections_step3x.apply(ocrPostProcessing_Pageid, axis=1)
# %%
figures = filter_bestdetections(all_detections_step3, figureclasslist , 0.7 )

figures_step1 = figures.apply(cut_image_savetemp, axis=1)


figures_step2 = figures_step1.apply(load_figure, axis=1)

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
                


                figures_output_dict_raw = figures_step2['figure_imgnp'].apply(run_inference_for_single_image, graph='detection_figureid_graph')
                
                
                
                

except Exception as e:
    print(e)


figures_step3 = figures_step2.drop(columns='figure_imgnp')
# %%
figures_output_dict = pd.DataFrame(figures_output_dict_raw.tolist()).add_prefix('figid_').reindex(figures_output_dict_raw.index)

figures_step4 = pd.concat([figures_step3, figures_output_dict], axis=1)



#

figures_detections = extract_detections_figureid(figures_step4)
#figures_step5 = pd.concat([figures_detections, figures_step4], axis=1)
figures_step4x = figures_step4.drop(columns=['figid_detection_boxes','figid_detection_scores','figid_detection_classes','figid_num_detections'])
figures_step5 = figures_detections.merge(figures_step4x, on=['pub_key','pub_value', 'page_imgname','figure_tmpid'], how='left')
figid_category_index = get_figid_labelmap_as_df(FIGID_MODEL + LABELS)
#figures_step5x =figures_step.drop(columns='figure_imgnp')
figures_step6 = figures_step5.merge(figid_category_index, on=['figid_detection_classes'], how='left')



bestfigid = filter_bestdetections_figid(figures_step6, figureidclasslist , 0.6 )
# %%
figid_imgnp = bestfigid.apply(cut_image_figid, axis=1)

figid_imgnp_mod = figid_imgnp.apply(ocrPreProcessing)
# %%
figid_raw = figid_imgnp_mod.apply(pytesseract.image_to_string, config=pageid_config)

figid_raw = pd.Series(figid_raw, name='figid_raw')
# %%
figures_step7 = pd.concat([bestfigid, figid_raw], axis=1)
cols_to_use = figures_step7.columns.difference(figures_step4x.columns)

figures_step8 = pd.merge(figures_step4x, figures_step7[cols_to_use], left_index=True, right_index=True, how='left')
#figures_step8.to_json(r'catalog_out.jsonl')
#figures_jsonl = figures_step8.to_json(orient='records', lines=True)

with open(OUTPATH + 'catalogs.jsonl', 'w') as f:
    pubs = figures_step8[['pub_key','pub_value']].drop_duplicates()
    pubs.apply(createCATALOG_JSONL, file = f, axis = 1)
with open(OUTPATH + 'trenches.jsonl', 'w') as f:
    pubs = figures_step8[['pub_key','pub_value']].drop_duplicates()
    pubs.apply(createTRENCH_JSONL, file = f, axis = 1)
with open(OUTPATH + 'types.jsonl', 'w') as f:
    figures_step8.apply(createTYPE_JSONL, file = f, axis = 1)
with open(OUTPATH + 'finds.jsonl', 'w') as f:
    figures_step8.apply(createFIND_JSONL, file = f, axis = 1)
with open(OUTPATH + 'drawings.jsonl', 'w') as f:
    figures_step8.apply(createDRAWING_JSONL, file = f, axis = 1)
    #for i in figures_step8.index:
        #FIND = createFIND_JSONL(figures_step8[i])
        #f.write("%s\n" % FIND)
    

    
#df['json'] = df.apply(lambda x: x.to_json(), axis=1)    

#with jsonlines.open(, 'w') as outfile:
            #outfile.write(figures_jsonl)

# %%
TFRECORDOUT = OUTPATH + 'mining_pages.tfrecord'
writer = tf.python_io.TFRecordWriter(TFRECORDOUT )

shutil.copyfile(PAGE_MODEL + LABELS, OUTPATH + 'mining_pages_label_map.pbtxt' )
shutil.copyfile(FIGID_MODEL + LABELS, OUTPATH + 'figid_label_map.pbtxt' )

#grouped = split(figures_step7, 'page_path')
TFRECORDOUT = OUTPATH + 'mining_pages.tfrecord'
tf_example = figures_step8.apply(create_tf_example, axis=1)
writer.write(tf_example.SerializeToString())
   
writer.close()               


