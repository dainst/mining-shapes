import requests
import os
import pandas as pd
import json
from datetime import datetime
import operator
import numpy as np
import math
import matplotlib.pyplot as plt
from math import isnan
import itertools
from decimal import Decimal  
#from decimal import Decimal  
auth = ('', 'blub')
db_url = 'http://host.docker.internal:3000'
db_name = 'idaishapes'
pouchDB_url_find = f'{db_url}/{db_name}/_find'
pouchDB_url_put = f'{db_url}/{db_name}/'
pouchDB_url_bulk = f'{db_url}/{db_name}/_bulk_docs'
pouchDB_url_all = f'{db_url}/{db_name}/_all_docs'
imagestore = '/home/imagestore/'
FIG_SIZE = [20,30]


def getAllDocs():
    querydict={'selector':{}}
    querydict['selector']['_id'] = {'$gt': None}
    response = requests.post(pouchDB_url_find, auth=auth, json=querydict)
    result = json.loads(response.text)
    return result

def getAllDocsv2():
    response = requests.post(pouchDB_url_all, auth=auth)
    result = json.loads(response.text)
    return result

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
def getDocsNotRecordedInIdentifier(identifier, auth, pouchDB_url_find):
    querydict={'selector':{'$not':{'resource.relations.isRecordedIn':{}}}}
    selector1 = {'$in': [str(idOfIdentifier(str(identifier), auth=auth, pouchDB_url_find=pouchDB_url_find)), str(idOfIdentifier('StadtSurvey', auth=auth, pouchDB_url_find=pouchDB_url_find)) ]}
    querydict['selector']['$not']['resource.relations.isRecordedIn'] = selector1
    response = requests.post(pouchDB_url_find, auth=auth, json=querydict)
    result = json.loads(response.text)
    print (querydict)
    return result


def saveChanges(doc, pouchDB_url_put, auth ):
    requests.put(pouchDB_url_put + doc['_id'], auth=auth, json=doc)

def bulkSaveChanges(DOC, pouchDB_url_bulk, auth ):
    answer = requests.post(pouchDB_url_bulk , auth=auth, json=DOC)
    print(answer)

def statOfRessouceTypes(result):
    dataset = sorted(result['docs'], key= lambda x: x['resource']['type'])
    grouped = itertools.groupby(dataset,key = lambda x: x['resource']['type'])
    for k, v in grouped:
        print (k, len(list(v)))
    print('That was statOfRessouceTypes')
    return grouped

def statOfRecordedIn(result):
    dataset = sorted(result, key= lambda x: x['resource']['relations']['isRecordedIn'][0])
    grouped = itertools.groupby(dataset,key = lambda x: x['resource']['relations']['isRecordedIn'][0])
    for k, v in grouped:
        print (str(identifierOfId(str(k), auth=auth, pouchDB_url_find=pouchDB_url_find)), len(list(v)))
    print('That was statOfRecordedIn')
    return grouped
def statOfLiesWithin(result):
    dataset = sorted(result, key= lambda x: x['resource']['relations']['liesWithin'][0])
    grouped = itertools.groupby(dataset,key = lambda x: x['resource']['relations']['liesWithin'][0])
    for k, v in grouped:
        print (str(identifierOfId(str(k), auth=auth, pouchDB_url_find=pouchDB_url_find)), len(list(v)))
    print('That was statOfLiesWithin')
    return grouped

def filterOfResourceTypes(listOfNotIncludedTypes, result):
    filteredResources = [obj for obj in result['docs'] if not obj['resource']['type'] in listOfNotIncludedTypes ]
    return filteredResources
def selectOfResourceTypes(listOfIncludedTypes, result):
    selectedResources = [obj for obj in result if 'type' in obj['resource'].keys() ]
    selectedResources = [obj for obj in selectedResources if obj['resource']['type'] in listOfIncludedTypes ]
    return selectedResources



def findEliminateNone(row):
    #test = json.loads(row['dimensionDiameter'][0])
    if type(row['dimensionDiameter']) == list:
        #if 'dimensionDiameter' in row['dimensionDiameter'][0]:
        #print(json.loads(row['dimensionDiameter'][0]))
        #if 'inputRangeEndValue' in row['dimensionDiameter'][0]:
        if 'inputValue' in row['dimensionDiameter'][0] and row['dimensionDiameter'][0]['inputValue'] == None:
            #if row['dimensionDiameter'][0]['isRange']== False:
            print (row['identifier'])
                #print (row['modified'])
            print(row['dimensionDiameter'])
            del row['dimensionDiameter'][0]
                #print(row['dimensionDiameter'])
    #elif math.isnan(row['dimensionDiameter']):
        #print('Nonono')

    return row


def cleanNans(series, listOfFields):
    cleanseries = pd.Series()
    nanseries = pd.Series('object')
    if not series[listOfFields].hasnans:
        if not series[listOfFields].isnull().values.any():
        #print(series[listOfFields])
            cleanseries = series

    return cleanseries 
    #thelist = ['temper', 'temperAmount', 'temperParticles']
    #notnull = series.notnull()
    #good = series.loc[notnull]
    #df=pd.DataFrame(series)
    #print(df.columns)
    #if pd.Series(thelist).isin(df.columns).all():
        #print (good['temper'])


    #else:
        #print('NOLIST: ', series['temper'] )
    #res = subset.groupby(DFresources['temper'].map(tuple))['Count'].sum()
    #print(series[listOfFields])
    #for column in interestproperties:
        #listvalues = [str(i) for i in DFresources[column]]
        #print(column, set(listvalues) )    return series



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
    
    #print(columns)
    DOC = []
    for index,row in DF.iterrows():
        #print(type(row[columns]))
        #dd = defaultdict(list)
        #print('Before DROP:')  
        cleanrow=row.dropna() 
        columns = [i for i in cleanrow.keys() if not i in docfields]
        #äprint(cleanrow)
        cleanrow['resource']= cleanrow[columns].to_dict()
        #row['resource'] = {k: row['resource'][k] for k in row['resource'] if not isnan(row['resource'][k])}
        #print(row)
        #print('After DROP:')
        cleanrow = cleanrow.drop(columns)
        #print(row)

        DOC.append(cleanrow.to_dict())
    DOChull={}
    DOChull['docs']=DOC
    return DOChull


def addModifiedEntry(doc):
    now = datetime.now()
    entry = {}
    entry['user'] = 'Script mhaibt'
    daytoSec = now.strftime('%Y-%m-%dT%H:%M:%S')
    sec = "{:.3f}".format(Decimal(now.strftime('.%f')))
    entry['date'] = daytoSec + str(sec)[1:] + 'Z'
    #print(entry)
    #doc['modified']=[]
    doc['modified'].append(entry)
    #print(doc['modified'])
    return doc

def searchReplaceIdentifier(series, searchstr, replacestr):
    if searchstr in series['identifier']:
        print(series['identifier'])
        series['identifier'] = series['identifier'].replace(searchstr, replacestr)
        print('changed to: ',series['identifier'])
    return series

def updateFigID(series):
    if 'AUWE4' in series['identifier'] and 'page' in series['literature'][0].keys() :
        series['literature'][0]['page']= 'Tafel ' + str(series['literature'][0]['page'])
    return series

    


#result = getDocsRecordedInIdentifier(identifier='AUWE1', auth, pouchDB_url_find)    
result = getAllDocs()
print('AllDocs: ', len(result['docs']))
mistakes = [doc['_id'] for doc in result['docs'] if '_id' in doc['resource'].keys() and '_rev' in doc['resource'].keys() ]

#print(json.dumps(mistakes, indent=4, sort_keys=True))
print(len(mistakes))
#listOfIncludedTypes = ['Type']
#allTypes = selectOfResourceTypes(['Type'], result['docs'])
#print('Drawings: ', len(Drawings) )
#listOfIncludedTypes = ['Type', 'TypeCatalog']
#TypesandCatalogs = selectOfResourceTypes(['Type', 'TypeCatalog'], result['docs'])
#print('allTypes: ', len(allTypes) )
#print('ORIGINAL RESOURCE')
#print(allTypes[0])
#DFresources, docfields = DOCtoDF(result['docs'])

#DFresourcesAUWE4 = DFresources[DFresources['identifier'].str.contains('AUWE4')]
#DFresources_ident = DFresourcesAUWE1.apply(searchReplaceIdentifier, searchstr='AUWE1_p', replacestr='AUWE1_Tafel', axis=1)
#DFresources_ident = DFresources_ident.apply(searchReplaceIdentifier, searchstr='AUWE1drawing_p', replacestr='AUWE1drawing_Tafel', axis=1)
#print('changing so many resources: ',len(DFresourcesAUWE4))
#DFresources_figid = DFresourcesAUWE4.apply(updateFigID, axis=1)
#print('changing so many resources: ',len(DFresources_ident))
#DFresources_final = DFresources_figid.apply(addModifiedEntry, axis=1)
#print(docfields)
#DOC = DFtoDOC(DFresources_final, docfields)
#print(json.dumps(DOC['docs'][0], indent=4, sort_keys=True))
#bulkSaveChanges(DOC, pouchDB_url_bulk, auth)
        



#print(json.dumps(DOC['docs'], indent=4, sort_keys=True))
#bulkSaveChanges(DOC, pouchDB_url_bulk, auth)
#print('DF RESOURCE')
