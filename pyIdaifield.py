from pydoc import doc
import requests
import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import operator
import numpy as np
import math
import matplotlib.pyplot as plt
import re
from math import isnan
import itertools
from json import JSONDecoder
import seaborn as sns
import colorcet as cc
from decimal import Decimal  
#from decimal import Decimal  

#pouchDB_url_find = f'{db_url}/{db_name}/_find'
#ouchDB_url_put = f'{db_url}/{db_name}/'

#pouchDB_url_alldbs = f'{db_url}/_all_dbs'



def DOCtoDF(DOC):
    DFdocs = pd.DataFrame(DOC)
    print(DFdocs.columns)
    DFdocs = DFdocs.drop('resource', axis=1)
    DFresources = pd.DataFrame([i['resource'] for i in DOC])
    for col in DFdocs.columns:
        DFresources[str(col)]=DFdocs[str(col)]
    docfields = DFdocs.columns

    return DFresources, docfields


def allDocsToDf(DOC):
    outsideDF = pd.DataFrame([dict['doc'] for dict in DOC['rows']])
    outsideDF.drop(columns=['resource','_attachments'] , axis=1, inplace=True)
    insideDOC = [dict['doc'].get('resource') for dict in DOC['rows']]
    insideDOC = [i for i in insideDOC if i]
    print(len(insideDOC))
    DFdocs = pd.DataFrame(insideDOC)
    docfields = list(outsideDF.columns)
    print(docfields)
    for col in docfields:
        DFdocs[str(col)]=outsideDF[str(col)]
    return DFdocs, docfields
   

def DFtoDOC(DFresources, docfields):
    DF = DFresources
    columns = [i for i in DFresources.columns if not i in docfields]
    #print(columns)
    DOC = []
    for index,row in DF.iterrows():
        
        cleanrow = row.drop(docfields)
        cleanrow=cleanrow.dropna() 
        row['resource']= cleanrow.to_dict()
        row = row.drop(columns)
        dict = row.to_dict()
        clean_dict = {k: dict[k] for k in dict.keys() if not isinstance(dict[k], (float)) or not isnan(dict[k])}

        DOC.append(clean_dict)
    
    DOChull={}
    DOChull['docs']=DOC
    return DOChull

def getAllDocs(db_url, auth, db_name):
    pouchDB_url_all = f'{db_url}/{db_name}/_all_docs'
    pouchDB_url_base = f'{db_url}/{db_name}'
    response = requests.get(pouchDB_url_base, auth=auth)
    result = json.loads(response.text)
    print('The database contains so much docs: ', result['doc_count'])
    if result['doc_count'] > 10000:
        collect = {"total_rows":0,"rows":[], "offset": 0}
        limit = math.ceil(result['doc_count'] / 10000)

        for i in range(limit):
            
            response = requests.get(pouchDB_url_all, auth=auth, params={'limit':10000, 'include_docs':True, 'skip': i * 10000})
            i = i + 1
            result = json.loads(response.text)
            print('This is round ' + str(i) + 'offset :', str(result['offset']) )
            collect['total_rows'] = collect['total_rows'] + result['total_rows']
            collect['offset'] = result['offset']
            collect['rows'] = collect['rows'] + result['rows']
    else:
        response = requests.get(pouchDB_url_all, auth=auth,params = {'include_docs':True})
        collect = json.loads(response.text)
    return collect

def flatten(t):
    return [item for sublist in t for item in sublist]

def getListOfDBs(db_url, auth):
    pouchDB_url_alldbs = f'{db_url}/_all_dbs'
    response = requests.get(pouchDB_url_alldbs, auth=auth)
    result = json.loads(response.text)
    return result

def addModifiedEntry(series):
    now = datetime.now()
    entry = {}
    entry['user'] = 'Script mhaibt'
    daytoSec = now.strftime('%Y-%m-%dT%H:%M:%S')
    sec = "{:.3f}".format(Decimal(now.strftime('.%f')))
    entry['date'] = daytoSec + str(sec)[1:] + 'Z'
    if not 'modified' in series.keys():
        series['modified']=[]
    series['modified'].append(entry)
    return series

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]
def bulkSaveChanges(db_url, auth, db_name, DOC):
    pouchDB_url_bulk = f'{db_url}/{db_name}/_bulk_docs'
    chunks = list(divide_chunks(DOC['docs'], 200))
    for chunk in chunks:
        #print(json.dumps(chunk, indent=4, sort_keys=True))
        chunkhull = {'docs':[]}
        chunkhull['docs'] = chunk
        answer = requests.post(pouchDB_url_bulk , auth=auth, json=chunkhull)
        print(answer)
    return print('Documents uploaded')