import cv2 as cv2
import numpy as np
import pandas as pd
import uuid
import os
import nltk
import re



def double_to_singlepage(dataframe):
    for index,row in dataframe.iterrows():
        if row['expectDoublepages']:
            doublepage_imgnp = cv2.imread(row['page_path'])
            os.remove(row['page_path'])
            height, width, channel = doublepage_imgnp.shape
            splitline = int(width*row['splitpercent'])

            pageleft_imgnp = doublepage_imgnp[0:height, 0:splitline]
            pageright_imgnp = doublepage_imgnp[0:height, splitline:width]
            pdfpageid_left = 2*row['page_pdfid'] - 1
            pdfpageid_right = 2*row['page_pdfid']
            #result = re.sub(r"(\d.*?)\s(\d.*?)", r"\1 \2", string1)
            #print(result)
            #reg = re.compile('.*[0-9]*-[0-9]*\.png')
            pagepath_left = re.sub(r'-[0-9]*\.png', '-' + str(2*row['page_pdfid'] - 1) + '.png', row['page_path'])
            pagepath_right = re.sub(r'-[0-9]*\.png', '-' + str(2*row['page_pdfid'])  + '.png', row['page_path'])
            print(pagepath_left)
            print(pagepath_right)
            cv2.imwrite(pagepath_left, pageleft_imgnp)
            cv2.imwrite(pagepath_right, pageright_imgnp)




    #dataframe['figid_clean'] = dataframe['figid_raw']


def cut_image(dataframe: pd.DataFrame) -> np.ndarray:
    """
    @brief Select an ROI from image
    @param dataframe dataframe with image and ROI metadata
    """
    page_imgnp = cv2.imread(dataframe['page_path'])
    box = dataframe['detection_boxes']
    ymin, xmin, ymax, xmax = box
    bbox_xmin = int((xmin)*dataframe['page_width'])
    bbox_ymin = int((ymin)*dataframe['page_height'])
    bbox_xmax = int((xmax)*dataframe['page_width'])
    bbox_ymax = int((ymax)*dataframe['page_height'])
    bbox_np = page_imgnp[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]

    return bbox_np


def cut_image_savetemp(dataframe: pd.DataFrame, outpath_base: str) -> pd.DataFrame:
    """
    @brief Select an ROI from image and writes detected vesselprofile to disk
    @param dataframe dataframe with image and ROI metadata
    """
    page_imgnp = cv2.imread(dataframe['page_path'])
    box = dataframe['detection_boxes']
    ymin, xmin, ymax, xmax = box
    bbox_xmin = int((xmin)*dataframe['page_width'])
    bbox_ymin = int((ymin)*dataframe['page_height'])
    bbox_xmax = int((xmax)*dataframe['page_width'])
    bbox_ymax = int((ymax)*dataframe['page_height'])
    figure_imgnp = page_imgnp[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]
    del page_imgnp
    figure_height, figure_width, figure_channel = figure_imgnp.shape
    dataframe['figure_height'] = figure_height
    dataframe['figure_width'] = figure_width
    dataframe['figure_channel'] = figure_channel
    dataframe['figure_tmpid'] = uuid.uuid4()
    dataframe['figure_path'] = outpath_base + str(dataframe['figure_tmpid']) + '.png'
    cv2.imwrite(str(dataframe['figure_path']), figure_imgnp)

    return dataframe, figure_imgnp


def cut_image_figid(dataframe: pd.DataFrame) -> np.ndarray:
    figure_imgnp = cv2.imread(dataframe['figure_path'])
    box = dataframe['figid_detection_boxes']
    ymin, xmin, ymax, xmax = box
    bbox_xmin = int((xmin)*dataframe['figure_width'])
    bbox_ymin = int((ymin)*dataframe['figure_height'])
    bbox_xmax = int((xmax)*dataframe['figure_width'])
    bbox_ymax = int((ymax)*dataframe['figure_height'])
    bbox_np = figure_imgnp[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]

    return bbox_np


def load_figure(series: pd.Series) -> pd.Series:
    """
    @brief Loads figure image from given series
    @return Series with page data  
    """
    return load_image_from_pdseries(series, 'figure_path', 'figure')


def load_page(row):
    page_imgnp = cv2.imread(str(row['page_path']))
    print(str(row['page_path']))
    assert len(page_imgnp.shape) == 3
    page_height, page_width, page_channel = page_imgnp.shape
    row['page_width'] = page_width
    row['page_height'] = page_height
    row['page_channel'] = page_channel
    #row['page_imgnp'] = page_imgnp

    return row, page_imgnp


def load_image_from_pdseries(series, column_name: str, col_base_name: str):
    """
    @brief Loads image from path given in pd series
    @param series pandas Series
    @param column_name column in which image path is written in series
    @param col_base_name base name for column name for returned pandas series
    @return pdSeries which in includes image data and image dimensions
    """
    page_imgnp = cv2.imread(str(series[column_name]))
    print(str(series[column_name]))
    assert len(page_imgnp.shape) == 3
    page_height, page_width, page_channel = page_imgnp.shape
    series[f'{col_base_name}_width'] = page_width
    series[f'{col_base_name}_height'] = page_height
    series[f'{col_base_name}_channel'] = page_channel
    series[f'{col_base_name}_imgnp'] = page_imgnp

    return series



def ocr_pre_processing(image: np.ndarray) -> np.ndarray:
    """
    @brief applies adaptive thresholding to preprocess image for   
            optical character recognition 
    @param image image to be preprocessed
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image, 5)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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


def ocr_post_processing_pageid(row):
    pageid_raw = str(row['pageid_raw']).replace("\n","")
    pageid_raw = str(pageid_raw).replace(' ','')
    
    pageid_regex = re.compile(row['pageid_regex'])
    result = re.search(pageid_regex, pageid_raw)
    if result:
        row['pageid_clean'] = result.group(1)
    else:
        row['pageid_clean'] = 'none'

    return row

def ocr_post_processing_figid(row):
    figid_raw = str(row['figid_raw']).replace("\n","")
    figid_raw = str(figid_raw).replace(' ','')
    
    figid_regex = re.compile(row['figid_regex'])
    result = re.search(figid_regex, figid_raw)
    if result:
        row['figid_clean'] = result.group(1)
    else:
        row['figid_clean'] = 'none'

    return row


