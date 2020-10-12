import pandas as pd
import numpy as np
import os
from collections import namedtuple

from object_detection.utils import label_map_util


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

def provide_pdf_path(inputdirectory):
    listOfFiles = []
    for (dirpath, dirnames, filenames) in os.walk(inputdirectory):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames if file.endswith(".pdf")] 
    return listOfFiles

def provide_pagelist(inputdirectory: str) ->pd.DataFrame:
    """
    @brief reads metadata of page images from inputdirectory and stores it into 
            a pandas DataFrame
    @param inputdirectory directory with page images stored in .png or .jpg format
    @return DataFrame with cols: pub_key, pub_value, pub_imagename, page_path 
    """

    pagelist = []
    for pub_id in os.listdir(inputdirectory):
        pub_key, pub_value = pub_id.split('_')
        pub_path = os.path.join(inputdirectory, pub_id)
        pub = {}
        pub['pub_key'] = pub_key
        pub['pub_value'] = pub_value
        for page_imgname in os.listdir(pub_path):

            if page_imgname.endswith((".png", ".jpg")) and 'Thumbs' not in page_imgname:
                page = pub
                page_path = os.path.join(pub_path, page_imgname)
                page['page_imgname'] = page_imgname
                page['page_path'] = page_path
                pagelist.append(page.copy())
    return pd.DataFrame(pagelist)


def extract_page_detections(df: pd.DataFrame, category_index: pd.DataFrame) ->pd.DataFrame:
    """
    @brief extracts page detection metadata from dataframe df
    """
    page_detectionsaslist = pd.DataFrame(
        df['page_detections'].tolist()).reindex(df.index)
    df = pd.concat([df, page_detectionsaslist], axis=1)
    all_detections = pd.DataFrame()
    N = df['num_detections'].max()
    for i in range(0, N):
        detection = df.applymap(lambda x: x[i] if type(x) == np.ndarray else x)
        all_detections = all_detections.append(detection)

    all_detections = all_detections.merge(
        category_index, on=['detection_classes'], how='left')
    return all_detections


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
    for detection_classesname in bestpages_result.detection_classesname.unique():
        selected_info = bestpages_result[bestpages_result['detection_classesname']
                                         == detection_classesname]
        newinfo_name = detection_classesname + '_raw'
        selected_info = selected_info.rename(columns={'newinfo': newinfo_name})
        all_detections = all_detections.merge(selected_info[[newinfo_name, 'pub_key', 'pub_value', 'page_imgname']], on=[
                                              'pub_key', 'pub_value', 'page_imgname'], how='left')

    return all_detections


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
