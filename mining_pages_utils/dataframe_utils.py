import pandas as pd
import numpy as np
import os
import yaml
import json
import re
import inflect
from collections import namedtuple

from object_detection.utils import label_map_util
from pdf2image import convert_from_path


def humanreadID(Series):
    humanreadID = ''
    if Series['patternHRID']:
        print(Series.keys())
        for element in Series['patternHRID']:
            if Series[element]:
                humanreadID += Series[element] + '_'
            if Series[element] is 'none':
                humanreadID += Series[element] + '_'
                humanreadID += str(Series['figure_tmpid'])

        Series['HRID'] = humanreadID.rstrip('_')
    return Series


def handleduplicate(df, field):
    #p = inflect.engine()
    g = df.groupby(field)
    df['HRID_undup'] = df['HRID']
    df['HRID_undup'] += g.cumcount().add(1).radd('_DUP').mask(g.HRID_undup.transform('count') == 1, '')

    return df


def pdf_to_image(dataframe):
    if(pd.notnull(dataframe['pubpdf_path'])):
        pnglist = []
        for file in os.listdir(dataframe['pubfolder_path']):
            if file.endswith('.png'):
                pnglist.append(file)
        reg = re.compile(os.path.basename(dataframe['pubpdf_path'])+'.*png')
        pngexist = filter(reg.match, pnglist)
        if len(list(pngexist)) is 0:
            convert_from_path(dataframe['pubpdf_path'], fmt='png', thread_count=4, output_file=os.path.basename(
                dataframe['pubpdf_path']), first_page=None, last_page=None, output_folder=dataframe['pubfolder_path'])
    return dataframe


def pdf_to_imagev2(series):
    if 'expectImagesnotPdf' not in series.keys() and pd.notnull(series['pubpdf_path']) or not series['expectImagesnotPdf'] and pd.notnull(series['pubpdf_path']):
        pnglist = [file for file in os.listdir(series['pubfolder_path']) if file.endswith('.png')]
        print(pnglist)
        for pagetuple in series['select_pdfpages']:
            alreadypng = []
            for i in range(pagetuple[0],pagetuple[1]):
                i = str(i).zfill(len(str(pagetuple[1])))
                print(i)
                reg = re.compile(os.path.basename(series['pubpdf_path'])+'.*' + str(i) + '.png')
                pngexist = filter(reg.match, pnglist)
                alreadypng = alreadypng + list(pngexist)
            if not len(alreadypng) == range(pagetuple[0],pagetuple[1]):
                convert_from_path(series['pubpdf_path'], fmt='png', thread_count=6, output_file=os.path.basename(
                    series['pubpdf_path']), first_page=pagetuple[0],dpi=300,last_page=pagetuple[1], paths_only=False, use_pdftocairo=True, output_folder=series['pubfolder_path'])
            else:
                print('Pdf-pages' + str(pagetuple[0]) + ' to ' + str(pagetuple[1]) + ' already completed.' )
    return series


def get_pubs_and_configs(inputdirectory):
    publist = []
    for pub_id in os.listdir(inputdirectory):
        pub_key, pub_value = pub_id.split('_')
        pubfolder_path = os.path.join(inputdirectory, pub_id)
        with open(pubfolder_path + '/config.json') as configfile:
            pub = json.load(configfile)
        pub['pub_key'] = pub_key
        pub['pub_value'] = pub_value
        pub['pubfolder_path'] = str(pubfolder_path)
        publist.append(pub.copy())
    return pd.DataFrame(publist)


def get_page_labelmap_as_df(path_to_labels: str) -> pd.DataFrame:
    """
    @brief read page labelmap and return as pandas DataFrame
    """
    category_index = label_map_util.create_category_index_from_labelmap(
        path_to_labels, use_display_name=True)
    category_index = pd.DataFrame(category_index).T
    category_index = category_index.rename(
        columns={'id': 'detection_classes', 'name': 'detection_classesname'})
    return category_index


def get_figid_labelmap_as_df(path_to_labels: str) -> pd.DataFrame:
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
    pdflist = []
    for index, row in dataframe.iterrows():
        if not 'expectImagesnotPdf' in row.keys() or not row['expectImagesnotPdf']:
            for file in os.listdir(row['pubfolder_path']):
                if file.endswith('.pdf'):
                    pdfs = row
                    pdfs['pubpdf_path'] = str(
                        os.path.join(row['pubfolder_path'], file))
                    pdflist.append(pdfs.copy())
        if 'expectImagesnotPdf' in row.keys() and row['expectImagesnotPdf']:
            print('Expect no pdf in folder but images.')
            pdflist.append(row.copy())


    return pd.DataFrame(pdflist)


def provide_pagelist(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    @brief reads metadata of page images from inputdirectory and stores it into 
            a pandas DataFrame
    @param inputdirectory directory with page images stored in .png or .jpg format
    @return DataFrame with cols: pub_key, pub_value, pub_imagename, page_path 
    """
    pagelist = []
    for index, row in dataframe.iterrows():

        if 'expectImagesnotPdf' not in row.keys() and pd.notnull(row['pubpdf_path']) or not row['expectImagesnotPdf'] and pd.notnull(row['pubpdf_path']):
            filelist = os.listdir(row['pubfolder_path'])
            reg = re.compile(os.path.basename(
                row['pubpdf_path'])+'[0-9]*-[0-9]*\.png')
            pnglist = list(filter(reg.match, filelist))
            for page_imgname in pnglist:
                page = row
                page_path = os.path.join(row['pubfolder_path'], page_imgname)
                page['page_imgname'] = page_imgname
                page['page_path'] = page_path

                pagelist.append(page.copy())

        if 'expectImagesnotPdf' in row.keys() and row['expectImagesnotPdf']:
        
            for page_imgname in os.listdir(row['pubfolder_path']):
                if page_imgname.endswith((".png", ".jpg")) and 'Thumbs' not in page_imgname:
                    page = row
                    page_path = os.path.join(
                        row['pubfolder_path'], page_imgname)
                    page['page_imgname'] = page_imgname
                    page['page_path'] = page_path
                    pagelist.append(page.copy())
    return pd.DataFrame(pagelist)


def extract_pdfid(series, readfield, writefield ):
    reg_pdfpageid = re.compile(series['pdfid_regex'])
    series[writefield] = int(
        re.search(reg_pdfpageid, series[readfield]).group(1))
    return series


def select_pdfpages(dataframe: pd.DataFrame) -> pd.DataFrame:

    pagelist = pd.DataFrame()
    for index, row in dataframe.iterrows():
        print(type(row['page_pdfid']))
        for p in row['select_pdfpages']:
            print(p)
            if row['page_pdfid'] in range(p[0], p[1]):
                pagelist = pagelist.append(row)
    return pagelist


def unfold_pagedetections(df: pd.DataFrame) -> pd.DataFrame:
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


def page_detections_toframe(df: pd.DataFrame) -> pd.DataFrame:
    all_detections = []
    for index, row in df.iterrows():
        if not row['annotations']:
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
        if row['annotations']:
            for i in row['annotations']:
                detection = row
                print(i, type(i))

    return pd.DataFrame(all_detections)


def extract_page_detections(df: pd.DataFrame, keylist: list, category_index: pd.DataFrame) -> pd.DataFrame:
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
            # all_detections.append(detection.copy())

    return pd.DataFrame(all_detections)


def extract_page_detections_new(df: pd.DataFrame, category_index: pd.DataFrame) -> pd.DataFrame:
    """
    @brief extracts page detection metadata from dataframe df
    """
    for index, row in df.iterrows():
        page_detections = row['page_detections']
        boxes = page_detections['detection_boxes']

        # print(boxes)


def extract_detections_figureidv2(df: pd.DataFrame):
    figid_detectionsdict = df['figid_detections']
    df['figid_detection_scores'] = figid_detectionsdict['figid_detection_scores'][0]
    df['figid_detection_boxes'] = figid_detectionsdict['figid_detection_boxes'][0]
    df['figid_detection_classes'] = figid_detectionsdict['figid_detection_classes'][0]
    df['figid_num_detections'] = figid_detectionsdict['figid_num_detections']
    return df.drop(columns='figid_detections')


def filter_best_page_detections(all_detections, lowest_score):
    """
    @brief thresholds and selects only detections above certain detection score
    @classlist list of all classes 
    @param lowest_score score threshold
    """
    pageids = pd.DataFrame()
    for index, row in all_detections.iterrows():
        if row['detection_classesname'] in row['allowedPageDetections'] and row['detection_scores'] >= lowest_score:
            pageids = pageids.append(row)

    return pageids


def choose_pageid(pageids):
    bestdetections = (pageids[pageids['detection_scores'] == pageids
                              .groupby(['pub_key', 'pub_value', 'page_imgname', 'detection_classesname'])['detection_scores'].transform('max')])
    return bestdetections


def filter_best_vesselprofile_detections(all_detections: pd.DataFrame, lowest_score: float):
    """
    @brief thresholds and selects only vesselprofile detections above certain detection score
    @classlist list of all classes 
    @param lowest_score score threshold
    """
    bestdetections = pd.DataFrame()
    for index, row in all_detections.iterrows():
        print(row['detection_scores'], row['detection_classesname'] )
    
        if row['detection_classesname'] in row['allowedFigureDetections'] + row['allowedFrameDetections']  and row['detection_scores'] >= lowest_score:
            bestdetections = bestdetections.append(row)
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
            selected_info = selected_info.rename(
                columns={'newinfo': newinfo_name})
            #all_detections = pd.DataFrame(columns=['pageid_raw', 'pageid_info'])
            all_detections = all_detections.merge(selected_info[[newinfo_name, 'pub_key', 'pub_value', 'page_imgname']], on=[
                                                  'pub_key', 'pub_value', 'page_imgname'], how='left')
        if 'pageid_raw' not in all_detections.columns:
            all_detections['pageid_raw'] = ''
        if 'pageinfo_raw' not in all_detections.columns:
            all_detections['pageinfo_raw'] = ''
    else:
        all_detections['pageid_raw'] = ''
        all_detections['pageinfo_raw'] = ''

    return all_detections

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