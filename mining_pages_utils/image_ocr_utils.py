
import cv2
import numpy as np
import pandas as pd


def cut_image(dataframe: pd.DataFrame) -> np.ndarray:
    page_imgnp = cv2.imread(dataframe['page_path'])
    box = dataframe['detection_boxes']
    ymin, xmin, ymax, xmax = box
    bbox_xmin = int((xmin)*dataframe['page_width'])
    bbox_ymin = int((ymin)*dataframe['page_height'])
    bbox_xmax = int((xmax)*dataframe['page_width'])
    bbox_ymax = int((ymax)*dataframe['page_height'])
    bbox_np = page_imgnp[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]

    return bbox_np


def cut_image_savetemp(dataframe: pd.DataFrame, outpath: str) -> pd.DataFrame:
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
    dataframe['figure_path'] = outpath + str(dataframe['pub_key']) + '_' + str(
        dataframe['pub_value']) + '_' + 'tempid' + str(dataframe['figure_tmpid']) + '.png'
    cv2.imwrite(str(dataframe['figure_path']), bbox_np)

    return dataframe


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
    figure_imgnp = cv2.imread(str(series['figure_path']))
    figure_height, figure_width, figure_channel = figure_imgnp.shape
    series['figure_width'] = figure_width
    series['figure_height'] = figure_height
    series['figure_channel'] = figure_channel
    series['figure_imgnp'] = figure_imgnp

    return series


def load_page(series: pd.Series) -> pd.Series:
    page_imgnp = cv2.imread(str(series['page_path']))

    page_height, page_width, page_channel = page_imgnp.shape
    series['page_width'] = page_width
    series['page_height'] = page_height
    series['page_channel'] = page_channel
    series['page_imgnp'] = page_imgnp

    return series


def ocr_pre_processing(image: np.ndarray) -> np.ndarray:
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
    pageid_raw = row['pageid_raw']
    row['pageid_int'] = [int(s) for s in pageid_raw.split() if s.isdigit()]
    return row
