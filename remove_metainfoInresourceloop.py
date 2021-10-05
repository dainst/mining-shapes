import requests
import os
import pandas as pd
import json
from datetime import datetime
import operator
import numpy as np
import math
import matplotlib.pyplot as plt
import shutil
from math import isnan
import itertools
from decimal import Decimal  
#from decimal import Decimal  

FIG_SIZE = [20,30]

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]
def bulkSaveChanges(DOC, pouchDB_url_bulk, auth ):
    chunks = list(divide_chunks(DOC['docs'], 200))
    for chunk in chunks:
        #print(json.dumps(chunk, indent=4, sort_keys=True))
        chunkhull = {'docs':[]}
        chunkhull['docs'] = chunk
        answer = requests.post(pouchDB_url_bulk , auth=auth, json=chunkhull)
        print(answer)
    return print('Documents uploaded')

def getListOfDBs():
    response = requests.get(pouchDB_url_alldbs, auth=auth)
    result = json.loads(response.text)
    return result

def getAllDocsv2(db_name):
    pouchDB_url_all = f'{db_url}/{db_name}/_all_docs'
    response = requests.post(pouchDB_url_all, auth=auth)
    result = json.loads(response.text)
    return result


def getAllDocs(db_name):
    pouchDB_url_find = f'{db_url}/{db_name}/_find'
    querydict={'selector':{}}
    querydict['selector']['_id'] = {'$gt': None}
    response = requests.post(pouchDB_url_find, auth=auth, json=querydict)
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

def saveChanges(doc, pouchDB_url_put, auth ):
    requests.put(pouchDB_url_put + doc['_id'], auth=auth, json=doc)



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
def replaceValue (series, key, value, newvalue):
    #print(key, series[key])
    if type(series[key])==list:
        if value in series[key]:
            
            series[key].remove(value)
            series[key].append(newvalue)
            series[key] = list(set(series[key]))
            print(key, series[key])
    return series

def combineColumns (series, columnlist, newColumn):
    for column in columnlist:
        if type(series[column])==list:
            #print(column, ' is a List')
            series[newColumn]=series[column]
        if type(series[column])==str:
        #print(column, ' is a List')
            series[newColumn]=series[column]
        else:
            if math.isnan(series[column]):
                #print(column, ' is NaN')
                continue
                #series.drop(column)
    columns = [i for i in columnlist if not i == newColumn]
    #print(columns)
    series = series.drop(columns)
    return series

def strColumnToList (series, column):
    if type(series[column]) == str:
        valueaslist = []
        valueaslist.append(series[column])
        series[column] = valueaslist
    return series


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

def correctDiameter(row):
    #test = json.loads(row['dimensionDiameter'][0])
    if type(row['dimensionDiameter']) == list:
        #if 'dimensionDiameter' in row['dimensionDiameter'][0]:
        #print(row['dimensionDiameter'][0])
        #if not math.isnan(row['dimensionDiameter']):
            #row['dimensionDiameter'] = np.nan
            #print(row['identifier'], row['dimensionDiameter'] )
            #row = row.drop('dimensionDiameter')
            #if row['dimensionDiameter']:
                #print(row['dimensionDiameter'] )
        if 'isRange' in row['dimensionDiameter'][0] and row['dimensionDiameter'][0]['isRange']== False and 'rangeMin' in row['dimensionDiameter'][0]:
            print(row['dimensionDiameter'][0]['inputValue'], row['dimensionDiameter'][0]['rangeMin'], row['dimensionDiameter'][0]['rangeMax'])
            del row['dimensionDiameter'][0]['rangeMin']
            del row['dimensionDiameter'][0]['rangeMax']
            if 'inputRangeEndValue' in row['dimensionDiameter'][0]:
                print(row['dimensionDiameter'][0])
                del row['dimensionDiameter'][0]['inputRangeEndValue']
                print(row['dimensionDiameter'][0])
                #row['dimensionDiameter'][0].pop('inputRangeEndValue')
            #del row['dimensionDiameter'][0]['inputRangeEndValue']
            #row['dimensionDiameter'][0]['isRange']
        #if 'inputValue' in row['dimensionDiameter'][0] and row['dimensionDiameter'][0]['inputValue'] == None:
            #if row['dimensionDiameter'][0]['isRange']== False:
            #print (row['identifier'])
                #print (row['modified'])
            #print(row['dimensionDiameter'])
            #del row['dimensionDiameter'][0]
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
def typoStat(df):
    notnull = df.notnull()
    good = df[notnull]
    good = good.astype({'temper': 'string'})
    res = good.groupby(['temper', 'temperAmount', 'temperParticles']).size()
    print(res)
    #return res


def fillEmptyProcessor(series):
    if 'processor' in series and not type(series['processor'])==list:
        if math.isnan(series['processor']):
            series['processor']=series['created']['user']
            print(series[['type','processor']])
    
    return series


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
        cleanrow=row.dropna() 
        #äprint(cleanrow)
        row['resource']= cleanrow.to_dict()
        #row['resource'] = {k: row['resource'][k] for k in row['resource'] if not isnan(row['resource'][k])}
        #print(row)
        #print('After DROP:')
        row = row.drop(columns)
        #print(row)
        dict = row.to_dict()
        clean_dict = {k: dict[k] for k in dict.keys() if not isinstance(dict[k], (float)) or not isnan(dict[k])}

        DOC.append(clean_dict)
    
    DOChull={}
    DOChull['docs']=DOC
    return DOChull
def simpleTypoPlot (df, listOfFields):
    df = good.replace(np.nan,0)
    df = df.astype({'temper': 'string'})
    df = df.groupby(listOfFields).size()
    df.plot.bar()
    plt.show()
 
def plot_bargraph_with_groupings(df, groupby, colourby, title, xlabel, ylabel):
    """
    Plots a dataframe showing the frequency of datapoints grouped by one column and coloured by another.
    df : dataframe
    groupby: the column to groupby
    colourby: the column to color by
    title: the graph title
    xlabel: the x label,
    ylabel: the y label
    """

    import matplotlib.patches as mpatches
    df = df.astype({'temper': 'string'})

    # Makes a mapping from the unique colourby column items to a random color.
    ind_col_map = {x:y for x, y in zip(df[colourby].unique(),
                               [plt.cm.Paired(np.arange(len(df[colourby].unique())))][0])}
    #print(ind_col_map)


    # Find when the indicies of the soon to be bar graphs colors.
    unique_comb = df[[groupby, colourby]].drop_duplicates()
    name_ind_map = {x:y for x, y in zip(unique_comb[groupby], unique_comb[colourby])}
    c = df[groupby].value_counts().index.map(lambda x: ind_col_map[name_ind_map[x]])

    # Makes the bargraph.
    ax = df[groupby].value_counts().plot(kind='bar',
                                         figsize=FIG_SIZE,
                                         title=title)
    # Makes a legend using the ind_col_map
    legend_list = []
    for key in ind_col_map.keys():
        legend_list.append(mpatches.Patch(color=ind_col_map[key], label=key))

    # display the graph.
    plt.legend(handles=legend_list)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def addModifiedEntry(doc):
    now = datetime.now()
    entry = {}
    entry['user'] = 'Script mhaibt'
    daytoSec = now.strftime('%Y-%m-%dT%H:%M:%S')
    sec = "{:.3f}".format(Decimal(now.strftime('.%f')))
    entry['date'] = daytoSec + str(sec)[1:] + 'Z'
    #print(entry)
    doc['modified'].append(entry)
    #print(doc['modified'])
    return doc

def deleteImagestorefile(series):
    figure_imagestorepath = os.path.join('imagestore', 'exportProject', str(series['_id']) )
    if os.path.exists(figure_imagestorepath):
      os.remove(figure_imagestorepath)
    return series

def handleduplicate(df, field):
    print(len(df[field]))
    groupsize = df.groupby( field ).size()
    cumcount = df.groupby( field ).cumcount()
    df = df.set_index(field)
    df['groupsize']= groupsize
    df = df.reset_index()
    df['cumcount']= cumcount
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    #print(df['cumcount'])
    dfnew = pd.DataFrame()
    df['cumcount'].fillna(0, inplace=True)
    df['groupsize'].fillna(0, inplace=True)
    print(df[['cumcount','groupsize']])
    for index, row in df.iterrows():
        if int(row['cumcount']) == 0 and int(row['groupsize'])== 1:
            row[field + '_undup'] = row[field]
        else: 
            row[field + '_undup'] = str(row[field]) + '_' + str(row['cumcount'])
        dfnew = dfnew.append(row)
    return dfnew

def pathToStore(series):
    series['figure_imagestorepath'] = os.path.join(series['imagestore'], series['exportProject'], str(series['figure_tmpid']) )
    return series
def imageToStore(series):
    shutil.copyfile(str(series['figure_path']), series['figure_imagestorepath'])
    return series
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

auth = ('', 'blub')
db_url = 'http://host.docker.internal:3000'
#db_name = 'shapes_import'
#pouchDB_url_find = f'{db_url}/{db_name}/_find'
#pouchDB_url_put = f'{db_url}/{db_name}/'
#pouchDB_url_bulk = f'{db_url}/{db_name}/_bulk_docs'
#pouchDB_url_all = f'{db_url}/{db_name}/_all_docs'
pouchDB_url_alldbs = f'{db_url}/_all_dbs'
imagestore = '/home/imagestore/'
#exportProject = 'shapes_import'
alldblist = getListOfDBs()
shapesdblist = [db for db in alldblist if db.endswith('_ed')]
print(alldblist)
excludelist = ['ock_ed']
selectlist = ['hayes1972_ed']
targetdb = ['idaishapes']
#shapesdblist = [db for db in shapesdblist if not db in excludelist ]
## for testing only one db ##
#shapesdblist = [db for db in shapesdblist if db in selectlist ]
print(shapesdblist)


for db in targetdb:
    print(db)
    result = getAllDocs(db)
    print('AllDocs: ', len(result['docs']))
    mistakes = [doc for doc in result['docs'] if '_id' in doc['resource'].keys() or '_rev' in doc['resource'].keys()  or 'attachments' in doc['resource'].keys() or '_attachments' in doc['resource'].keys()]
    print(len(mistakes))
    corrected = []
    for doc in mistakes:
        remove_key = doc['resource'].pop('_id', None)
        remove_key = doc['resource'].pop('_rev', None)
        remove_key = doc['resource'].pop('modified', None) 
        remove_key = doc['resource'].pop('created', None)
        remove_key = doc['resource'].pop('attachments', None)  
        remove_key = doc['resource'].pop('_attachments', None)
        addModifiedEntry(doc)    	
        corrected.append(doc)
    #print(json.dumps(mistakes, indent=4, sort_keys=True))
    docshull = {}
    docshull['docs']=corrected
    pouchDB_url_bulk = f'{db_url}/{db}/_bulk_docs'
    #bulkSaveChanges(docshull, pouchDB_url_bulk, auth)

    #shutil.copyfile(os.path.join(imagestore, db), os.path.join(imagestore, 'null'))
    #docs = [doc for doc in docs if 'type' in doc['resource'].keys()]
    #Drawings = [doc for doc in docs if doc['resource']['type']=='Drawing' ]
    #DFresources, docfields = DOCtoDF(dbdocs['docs'])

   #DFnew = handleduplicate(DFresources, 'identifier')
    
    
    #DFundup = DFnew[DFnew['identifier_undup']!=DFnew['identifier']]
    #print('Drawings',len(docs[docs['type']=='Drawing']))
    #print('Types',len(docs[docs['type']=='Type']))
    #print(DFundup[DFundup['type']=='Type'][['identifier','identifier_undup']])
    #DFundup['identifier'] = DFundup['identifier_undup']
    #DFundup = DFundup.drop(columns=['cumcount', 'groupsize', 'identifier_undup'])
    #DFundup = DFundup.apply(addModifiedEntry, axis=1)
    #DOCundup = DFtoDOC(DFundup, docfields)
    #docshull = {}
    #docshull['docs']=docs_noattach

    #print(json.dumps(docshull['docs'], indent=4, sort_keys=True))
    #pouchDB_url_bulk = f'{db_url}/idaishapes/_bulk_docs'
    #bulkSaveChanges(docshull, pouchDB_url_bulk, auth)
