import requests
import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
from itertools import groupby
import pybtex
from pybtex.database import parse_string
from  pybtex import format_from_string
from collections import Mapping, Set, Sequence
string_types = (str, unicode) if str is bytes else (str, bytes)
iteritems = lambda mapping: getattr(mapping, 'iteritems', mapping.items)()




def getZenonBibtex(series):
    exporturl = 'https://zenon.dainst.org/Record/' + series['pub_value'] + '/Export?style='
    exportstyle ='BibTeX'
    result = requests.get(exporturl + exportstyle)
    print(result.text)
    parse_string
    pybtex.errors.set_strict_mode(False)
    data = parse_string(result.text, 'bibtex')
    del data.entries[str(series['pub_value'])].fields['crossref']
    #list(data.entry.fields.keys())
    bib_data = data.to_string('bibtex')
    quote = pybtex.format_from_string(bib_data, 'unsrt', min_crossrefs=0, citations=['*'], output_backend='plaintext')
    #bib_data = format_from_string(result.text, 'unsrt')
    print(quote)
    series['pub_quote'] = quote
    #bibquote = make_bibliography(aux_filename, style=None, output_encoding=None, bib_format=None, **kwargs)
    #print (quote)
    return series

def getZenonInfo(series):
    if series['pub_key'] == 'ZenonID':
        zenonlink_url = 'https://zenon.dainst.org/api/v1/record?id=' + \
            series['pub_value'] + '&field[]=DAILinks'
        zenonbase_url = 'https://zenon.dainst.org/api/v1/record?id=' + \
            series['pub_value']
        s = requests.Session()
        zenonlinks = s.get(zenonlink_url)
        zenonlinks = json.loads(zenonlinks.text)
        zenonlinks = zenonlinks['records'][0]
        zenonbase = s.get(zenonbase_url)
        zenonbase = json.loads(zenonbase.text)
        zenonbase = zenonbase['records'][0]
        zenonbase['gazetteerlinks'] = zenonlinks['DAILinks']['gazetteer']
        zenonbase['thesaurilinks'] = zenonlinks['DAILinks']['thesauri']
        series['pub_info'] = zenonbase
        print('For pub_key: ' + series['pub_key'] + ' found pub_info!')
    else:
        series['pub_info'] = {}
        print('For pub_key: ' + series['pub_key'] + ' no pub_info')
    return series

def idOfIdentifier(identifier, auth, pouchDB_url_find):
    queryByIdentifier={'selector':{}}
    queryByIdentifier['selector']['resource.identifier'] = {'$eq':str(identifier)}
    response = requests.post(pouchDB_url_find, auth=auth, json=queryByIdentifier)
    result = json.loads(response.text)
    #print (result)
    return result['docs'][0]['resource']['id']

def identifierOfId(id, auth, pouchDB_url_find):
    queryByIdentifier={'selector':{}}
    queryByIdentifier['selector']['resource.id'] = {'$eq':str(id)}
    response = requests.post(pouchDB_url_find, auth=auth, json=queryByIdentifier)
    result = json.loads(response.text)
    #print (result)
    return result['docs'][0]['resource']['identifier']


def getDocsRecordedInIdentifier(identifier, auth, pouchDB_url_find):
    querydict={'selector':{}}
    querydict['selector']['resource.relations.isRecordedIn'] = {'$elemMatch': str(idOfIdentifier(str(identifier), auth=auth, pouchDB_url_find=pouchDB_url_find))}
    response = requests.post(pouchDB_url_find, auth=auth, json=querydict)
    result = json.loads(response.text)
    return result

def getAllDocs(auth, pouchDB_url_find):
    querydict={'selector':{}}
    querydict['selector']['resource.id'] = {'$gt': 'Null'}
    response = requests.post(pouchDB_url_find, auth=auth, json=querydict)
    result = json.loads(response.text)
    return result

def DOCtoDF(DOC):
    DFdocs = pd.DataFrame(DOC)
    print(DFdocs.columns)
    DFdocs = DFdocs.drop('resource', axis=1)
    DFresources = pd.DataFrame([i['resource'] for i in DOC])
    for col in DFdocs.columns:
        DFresources[str(col)]=DFdocs[str(col)]
    docfields = DFdocs.columns
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
    return DFresources, docfields
def objwalk(obj, path=(), memo=None):
    if memo is None:
        memo = set()
    iterator = None
    if isinstance(obj, Mapping):
        iterator = iteritems
    #elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, string_types) and not isinstance(obj, np.ndarray):
        #iterator = enumerate
    if iterator:
        if id(obj) not in memo:
            memo.add(id(obj))
            for path_component, value in iterator(obj):
                for result in objwalk(value, path + (path_component,), memo):
                    yield result
            memo.remove(id(obj))
    else:
        yield path, obj

def find(element, JSON):        
    paths = element.replace('(','').replace(')','').split(",")
    data = JSON
    for i in range(0,len(paths)):
        data = data[paths[i]]
    return data

def getIdaifieldConfigs(idaifieldconfigpath):
    # read file
    with open(os.path.join(idaifieldconfigpath,Path('Library/Categories.json')), 'r') as myfile:
        data=myfile.read()
    # parse file
    categories = json.loads(data)

    return categories

def selectOfResourceTypes(listOfIncludedTypes, allDocs):
    selectedResources = [obj for obj in allDocs if obj['resource']['type'] in listOfIncludedTypes ]
    return selectedResources
#categories = getIdaifieldConfigs(idaifieldconfigpath)

def DocsStructure(allDocs):
    allDocs.sort(key=lambda x:x['resource']['type'])
    categoryStructureList = []
    for k,v in groupby(allDocs,key=lambda x:x['resource']['type']):
        print(k,v)
        for obj in v:
            categoryStructure={}
            categoryStructure['category']= k
            for path,obj in objwalk(obj, path=(), memo=None):
                categoryStructure['fieldpath'] = path
                categoryStructure['datatype'] = str(type(obj))
                #print(categoryStructure)
                categoryStructureList.append(categoryStructure.copy())
    categoriesStructureDF = pd.DataFrame(categoryStructureList)
    categoriesStructureDF = categoriesStructureDF.drop_duplicates()
    return categoriesStructureDF

#result = getDocsRecordedInIdentifier('WarkaEnvironsSurvey', auth=auth, pouchDB_url_find=pouchDB_url_find)
#allDocs = getAllDocs(auth=auth, pouchDB_url_find=pouchDB_url_find )
#listoftypes = set([i['resource']['type'] for i in allDocs['docs'] if 'type' in i['resource'] ])
#DFresources, docfields = DOCtoDF(allDocs['docs'])

#
#print(listoftypes)
#print(type(allDocs['docs'][84]['resource']['featureVectors']['resnet']))
#allDocs = allDocs['docs']
#allDocs = [doc for doc in  allDocs if 'type' in doc['resource'].keys()]
#categoriesStructureDF = DocsStructure(allDocs)
#print(categoriesStructureDF)
#for index,row in categoriesStructureDF.iterrows():
    #print(index, row)
#groups = categoriesStructureDF.groupby(['category','fieldpath','datatype'])
#for group in groups:





#print(categories['Image:default'])
#print(categories['Drawing:default'])
#print(categories['Type:default'])
#print(categories['TypeCatalog:default'])



def addModifiedEntry(doc):
    now = datetime.now()
    entry = {}
    entry['user'] = 'Script mhaibt'
    daytoSec = now.strftime('%Y-%m-%dT%H:%M:%S')
    sec = "{:.3f}".format(Decimal(now.strftime('.%f')))
    entry['date'] = daytoSec + str(sec)[1:] + 'Z'
    #print(doc)
    doc['modified'].append(entry)
    return doc

#for doc in filteredResources:
    #print (doc)
    #doc['resource']['relations']['isRecordedIn'].clear() 
    #doc['resource']['relations']['isRecordedIn'].append(str(idOfIdentifier('WarkaEnvironsSurvey', auth=auth, pouchDB_url_find=pouchDB_url_find)))
    #doc = addModifiedEntry(doc)