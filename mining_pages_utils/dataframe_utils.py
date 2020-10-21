import pandas as pd
import numpy as np
import os
import yaml
import re
from collections import namedtuple

from object_detection.utils import label_map_util
from pdf2image import convert_from_path

def pdf_to_image(dataframe):
    if(pd.notnull(dataframe['pubpdf_path'])):
        pnglist = []
        for file in os.listdir(dataframe['pubfolder_path']):
            if file.endswith('.png'):
                pnglist.append(file)
        reg = re.compile(os.path.basename(dataframe['pubpdf_path'])+'.*png')
        pngexist =  filter(reg.match, pnglist)
        if len(list(pngexist)) is 0:
                convert_from_path(dataframe['pubpdf_path'], dpi=300, fmt='png', thread_count=4, output_file=os.path.basename(dataframe['pubpdf_path']), output_folder=dataframe['pubfolder_path'])
    return dataframe
def get_pubs_and_configs(inputdirectory):
        publist = []
        for pub_id in os.listdir(inputdirectory):
            pub_key, pub_value = pub_id.split('_')
            pubfolder_path = os.path.join(inputdirectory, pub_id)
            with open(pubfolder_path + '/config.yml') as ymlfile:
                pub = yaml.load(ymlfile, Loader=yaml.FullLoader)
            pub['pub_key'] = pub_key
            pub['pub_value'] = pub_value
            pub['pubfolder_path'] = str(pubfolder_path)
            publist.append(pub.copy())
        return pd.DataFrame(publist)    

def get_page_labelmap_as_df(path_to_labels: str) ->pd.DataFrame:
    """
    @brief read page labelmap and return as pandas DataFrame
    """
    category_index = label_map_util.create_category_index_from_labelmap(
        path_to_labels, use_display_name=True)
    category_index = pd.DataFrame(category_index).T
    category_index = category_index.rename(
        columns={'id': 'detection_classes', 'name': 'detection_classesname'})
    return category_index


def get_figid_labelmap_as_df(path_to_labels: str) ->pd.DataFrame:
    """
    @brief read figure id labelmap and return as pandas DataFrame
    """
    category_index = label_map_util.create_category_index_from_labelmap(
        path_to_labels, use_display_name=True)
    category_index = pd.DataFrame(category_index).T
    category_index = category_index.rename(
        columns={'id': 'figid_detection_classes', 'name': 'figid_detection_classesname'})
    return category_index

def provide_pdf_path(dataframe):
    pdflist=[]
    for index, row in dataframe.iterrows():
        for file in os.listdir(row['pubfolder_path']):
            if file.endswith('.pdf'):
                pdfs = row 
                pdfs['pubpdf_path'] = str(os.path.join(row['pubfolder_path'], file))
                pdflist.append(pdfs.copy())
                break
        else:
            nopdfs = row
            pdflist.append(nopdfs.copy())
    return pd.DataFrame(pdflist)

def provide_pagelist(dataframe: pd.DataFrame) ->pd.DataFrame:
    """
    @brief reads metadata of page images from inputdirectory and stores it into 
            a pandas DataFrame
    @param inputdirectory directory with page images stored in .png or .jpg format
    @return DataFrame with cols: pub_key, pub_value, pub_imagename, page_path 
    """
    pagelist = []
    for index, row in dataframe.iterrows():
        
        if(pd.notnull(row['pubpdf_path'])):
            filelist = os.listdir(row['pubfolder_path'])
            reg = re.compile(os.path.basename(row['pubpdf_path'])+'[0-9]*-[0-9]*\.png')
            pnglist =  list(filter(reg.match, filelist))
            reg_pdfpageid = re.compile(os.path.basename(row['pubpdf_path'])+'[0-9]*-([0-9]*)\.png')
            for page_imgname in pnglist:
                page = row
                page_path = os.path.join(row['pubfolder_path'], page_imgname)
                page['page_imgname'] = page_imgname
                page['page_path'] = page_path
                page['page_pdfid'] =  int(re.search(reg_pdfpageid, page_imgname).group(1))
                pagelist.append(page.copy())

        else:
            for page_imgname in os.listdir(row['pubfolder_path']):    
                if page_imgname.endswith((".png", ".jpg")) and 'Thumbs' not in page_imgname:
                    page = row
                    page_path = os.path.join(row['pubfolder_path'], page_imgname)
                    page['page_imgname'] = page_imgname
                    page['page_path'] = page_path
                    pagelist.append(page.copy())
    return pd.DataFrame(pagelist)


def unfold_pagedetections(df: pd.DataFrame) ->pd.DataFrame:
    """
    @brief extracts page detection metadata from dataframe df
    """
    rows = []
    keylist = []
    for index, row in df.iterrows():
        page_detections = row['page_detections']
        
        for key in page_detections:
            row[key] = page_detections[key]
            if not key is 'num_detections':
                keylist.append(key)
        rows.append(row.copy())
    return pd.DataFrame(rows), list(set(keylist))

def page_detections_toframe(df: pd.DataFrame) ->pd.DataFrame:
    all_detections = []
    for index, row in df.iterrows():
        N = int(row['page_detections']['num_detections'])
        for i in range(0, N):
            detection = row
            #detection.drop('detection_boxes', inplace=True)
            detection['detection_boxes'] = row['page_detections']['detection_boxes'][i]
            #detection.drop('detection_classes', inplace=True)
            detection['detection_classes'] = row['page_detections']['detection_classes'][i]
            #detection.drop('detection_scores', inplace=True)
            detection['detection_scores'] = row['page_detections']['detection_scores'][i]
            all_detections.append(detection.copy())

    return pd.DataFrame(all_detections)
        
def extract_page_detections(df: pd.DataFrame, keylist: list, category_index: pd.DataFrame) ->pd.DataFrame:
    all_detections = []
    for index, row in df.iterrows():
        N = row['num_detections']
        for i in range(0, N):
            detection = row
            for key in keylist:
                detectionarray = detection[key]
                #detectionlist = detectionarray.tolist()
                print(detectionarray[i])
                print(type(detectionarray[i]))
                #row.drop(key, inplace=True)
                #value = str(detectionarray[i])
                #detection[key]= detectionlist[i]
                #print(str(detection['page_pdfid']) +' '+ key + ' '+ str(i) + ' ' + str(detection[key]))
            #all_detections.append(detection.copy())
            
    return pd.DataFrame(all_detections)



def extract_page_detections_new(df: pd.DataFrame, category_index: pd.DataFrame) ->pd.DataFrame:
    """
    @brief extracts page detection metadata from dataframe df
    """
    for index, row in df.iterrows():
        page_detections = row['page_detections']
        boxes = page_detections['detection_boxes']
        #print(boxes)


def extract_detections_figureidv2(df: pd.DataFrame):
    figid_detectionsdict = df['figid_detections']
    df['figid_detection_scores'] = figid_detectionsdict['figid_detection_scores'][0]
    df['figid_detection_boxes'] = figid_detectionsdict['figid_detection_boxes'][0]
    df['figid_detection_classes'] = figid_detectionsdict['figid_detection_classes'][0]
    df['figid_num_detections'] = figid_detectionsdict['figid_num_detections']
    return df


def filter_best_page_detections(all_detections: pd.DataFrame, classlist: list, lowest_score: float):
    """
    @brief thresholds and selects only detections above certain detection score
    @classlist list of all classes 
    @param lowest_score score threshold
    """
    pageids = (all_detections[(all_detections['detection_classesname'].isin(classlist)) &
                              (all_detections['detection_scores'] >= lowest_score)])
    bestdetections = (pageids[pageids['detection_scores'] == pageids
                              .groupby(['pub_key', 'pub_value', 'page_imgname', 'detection_classesname'])['detection_scores'].transform('max')])
    return bestdetections


def filter_best_vesselprofile_detections(all_detections: pd.DataFrame, classlist: list, lowest_score: float):
    """
    @brief thresholds and selects only vesselprofile detections above certain detection score
    @classlist list of all classes 
    @param lowest_score score threshold
    """
    bestdetections = (all_detections[(all_detections['detection_classesname'].isin(classlist)) &
                                     (all_detections['detection_scores'] >= lowest_score)])
    return bestdetections


# def filter_bestdetections_figid(all_detections: pd.DataFrame, classlist: list, lowest_score: float):
#     figids = (all_detections[(all_detections['figid_detection_classesname'].isin(classlist)) &
#                              (all_detections['figid_detection_scores'] >= lowest_score)])
#     bestdetections = (figids[figids['figid_detection_scores'] == figids
#                              .groupby(['pub_key', 'pub_value', 'page_imgname', 'figure_tmpid'])['figid_detection_scores'].transform('max')])
#     return bestdetections


def merge_info(all_detections: pd.DataFrame, bestpages_result: pd.DataFrame):
    if not bestpages_result.empty:
        print(bestpages_result.detection_classesname.unique())
        for detection_classesname in bestpages_result.detection_classesname.unique():
            selected_info = bestpages_result[bestpages_result['detection_classesname']
                                            == detection_classesname]
            newinfo_name = detection_classesname + '_raw'
            selected_info = selected_info.rename(columns={'newinfo': newinfo_name})
            #all_detections = pd.DataFrame(columns=['pageid_raw', 'pageid_info'])
            all_detections = all_detections.merge(selected_info[[newinfo_name, 'pub_key', 'pub_value', 'page_imgname']], on=['pub_key', 'pub_value', 'page_imgname'], how='left')
        if 'pageid_raw' not in all_detections.columns:
            all_detections['pageid_raw'] = ''
        if 'pageinfo_raw' not in all_detections.columns:
            all_detections['pageinfo_raw'] = ''
    else:
        all_detections['pageid_raw'] = ''
        all_detections['pageinfo_raw'] = ''

    return all_detections


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
