import json
import pandas as pd
from typing import TextIO


def create_find_JSONL(df: pd.DataFrame, file: TextIO):
    FIND_template = '{"category":"","identifier":"","relations":{"isChildOf":"","isDepictedIn":[],"isInstanceOf":[]}}'
    FIND = json.loads(FIND_template)
    FIND["identifier"] = 'Find_' + str(df['figure_tmpid'])
    FIND["category"] = 'Pottery'

    relations = FIND["relations"]
    relations["isChildOf"] = 'Findspot_refferedtoin_' + \
        str(df['pub_key']) + '_' + str(df['pub_value'])
    InstanceOfList = relations["isInstanceOf"]
    typename = 'Type_' + str(df['figure_tmpid'])
    InstanceOfList.append(typename)
    depictedInList = relations["isDepictedIn"]
    imagename = str(df['figure_tmpid']) + '.png'
    depictedInList.append(imagename)
    json.dump(FIND, file)
    file.write("\n")



def create_constructivisttype_JSONL(df: pd.DataFrame, file: TextIO):
    TYPE_template = '{"category":"","identifier":"","relations":{"isChildOf":""}}'
    TYPE = json.loads(TYPE_template)
    TYPE["identifier"] = str(df['HRID'])
    TYPE["category"] = 'Type'
    relations = TYPE["relations"]
    relations["isChildOf"] = 'Catalog_' + \
        str(df['pub_key']) + '_' + str(df['pub_value'])
    json.dump(TYPE, file)
    file.write("\n")

def create_normativtype_JSONL(df: pd.DataFrame, file: TextIO):
    TYPE_template = '{"category":"","identifier":"", "shortDescription":"Extracted by Mining Shapes","literature":[{"quotation":"none","zenonId":""}],"relations":{"isChildOf":"","isDepictedIn":[]}}'
    TYPE = json.loads(TYPE_template)
    TYPE["identifier"] = str(df['HRID'])
    TYPE["category"] = 'Type'
    relations = TYPE["relations"]
    relations["isChildOf"] = 'Catalog_' + \
        str(df['pub_key']) + '_' + str(df['pub_value'])
    depictedInList = relations["isDepictedIn"]
    imagename = str(df['figure_tmpid']) + '.png'
    depictedInList.append(imagename)
    TYPE["shortDescription"] = 'PAGEID_RAW: ' + \
        str(df['pageid_raw']) + '   ' + 'PAGEINFO_RAW: ' + str(df['pageinfo_raw'])\
        + '   ' + 'FIGID_RAW: ' + str(df['figid_raw'])
    literature = TYPE["literature"]
    literature0 = literature[0]
    literature0['zenonId'] = str(df['pub_value'])

    literature0['quotation'] = 'p. ' + str(df['pageid_clean']) + ', fig. ' + str(df['figid_clean'])
    json.dump(TYPE, file)
    file.write("\n")


def create_drawing_JSONL(df: pd.DataFrame, file: TextIO):
    DRAWING_template = '{"category":"","identifier":"", "description":"Extracted by Mining Shapes","literature":[{"quotation":"none","zenonId":""}]}'
    DRAWING = json.loads(DRAWING_template)
    DRAWING["identifier"] = str(df['figure_tmpid']) + '.png'
    DRAWING["category"] = 'Drawing'
    DRAWING["description"] = 'PAGEID_RAW: ' + \
        str(df['pageid_raw']) + '   ' + 'PAGEINFO_RAW: ' + str(df['pageinfo_raw'])\
        + '   ' + 'FIGID_RAW: ' + str(df['figid_raw'])
    literature = DRAWING["literature"]
    literature0 = literature[0]
    literature0['zenonId'] = str(df['pub_value'])
    literature0['quotation'] = 'p. ' + str(df['pageid_clean']) + ', fig. ' + str(df['figid_clean'])

    json.dump(DRAWING, file)
    file.write("\n")


def create_catalog_JSONL(df: pd.DataFrame, file: TextIO):
    CATALOG_template = '{"category":"","identifier":"","shortDescription":"Extracted by Mining Shapes", "relations":{"isDepictedIn":[]}}'
    CATALOG = json.loads(CATALOG_template)
    CATALOG["identifier"] = 'Catalog_' + \
        str(df['pub_key']) + '_' + str(df['pub_value'])
    relations = CATALOG["relations"]
    depictedInList = relations["isDepictedIn"]
    depictedInList.append(
        'Catalogcover_' + str(df['pub_key']) + '_' + str(df['pub_value']) + '.png')
    CATALOG["category"] = 'TypeCatalog'
    json.dump(CATALOG, file)
    file.write("\n")


def create_trench_JSONL(df: pd.DataFrame, file: TextIO):
    TRENCH_template = '{"category":"","identifier":"","shortDescription":"Where have the Objects been found?"}'
    TRENCH = json.loads(TRENCH_template)
    TRENCH["identifier"] = 'Findspot_refferedtoin_' + \
        str(df['pub_key']) + '_' + str(df['pub_value'])
    TRENCH["category"] = 'Trench'
    json.dump(TRENCH, file)
    file.write("\n")
