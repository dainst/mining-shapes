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
db_name = 'shapes_import'
pouchDB_url_find = f'{db_url}/{db_name}/_find'
pouchDB_url_put = f'{db_url}/{db_name}/'
pouchDB_url_bulk = f'{db_url}/{db_name}/_bulk_docs'
imagestore = '/home/imagestore/'
exportProject = 'shapes_import'
FIG_SIZE = [20,30]


def getAllDocs():
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
    print(entry)
    doc['modified'].append(entry)
    #print(doc['modified'])
    return doc

def deleteImagestorefile(series):
    figure_imagestorepath = os.path.join('imagestore', 'exportProject', str(series['_id']) )
    if os.path.exists(figure_imagestorepath):
      os.remove(figure_imagestorepath)
    return series

    
result = getAllDocs()
#result = getDocsRecordedInIdentifier('WarkaEnvironsSurvey', auth=auth, pouchDB_url_find=pouchDB_url_find)
print(len(result['docs']))
#print(result)
#grouped = statOfRessouceTypes(result)
#print (grouped)
#listOfNIncludedTypes = []
#filteredResources = filterOfResourceTypes(listOfNotIncludedTypes, result)
#print(len(filteredResources))
#p = statOfRecordedIn(filteredResources)
#listOfNotIncludedTypes = ['Drawing','Place','Project','Photo','Image','TypeCatalog','Survey']
#filteredResources = filterOfResourceTypes(listOfNotIncludedTypes, result)
listOfIncludedTypes = ['Drawing']
selectedResources = selectOfResourceTypes(listOfIncludedTypes, result['docs'])
#print('ORIGINAL RESOURCE')
#print(selectedResources[0])
DFresources, docfields = DOCtoDF(selectedResources)
print(docfields)
print(len(DFresources))
DFresources_clean = DFresources[DFresources.identifier.str.startswith('OCK_type')]
DFresources_clean['_deleted'] = True
DFresources_clean = DFresources_clean.apply(addModifiedEntry, axis=1)
#print(DFresources_clean.columns)
#print(DFresources_clean['_id'])
print(type(DFresources_clean))
DFresources_clean = DFresources_clean.apply(deleteImagestorefile, axis=1)
print(type(docfields))
print(docfields)
docfields = docfields.append(pd.Index(['_deleted']))
print(len(DFresources_clean))
DFresources_clean = DFresources_clean[['_id','_rev','_deleted']]
DOC = DFtoDOC(DFresources_clean, docfields)
print(json.dumps(DOC['docs'], indent=4, sort_keys=True))
bulkSaveChanges(DOC, pouchDB_url_bulk, auth)
#print('DF RESOURCE')
#DFresources = DFresources.apply(replaceValue,key='temper', value='quartz', newvalue = 'quartz sand',axis=1)
#listOfFields = ['temper', 'temperAmount', 'temperParticles']
#statDF  = DFresources.apply(cleanNans,listOfFields=listOfFields, axis=1)
#notnull = statDF.notnull()
#good = statDF[notnull]
#print(good[['temper', 'temperAmount', 'temperParticles']])
#simpleTypoPlot (good, listOfFields)


#plot_bargraph_with_groupings(df=good, groupby='temper', colourby='identifier', title='Material', xlabel='Types', ylabel='frequency')
#DFresources = DFresources.apply(correctDiameter, axis=1)



#DFresources = DFresources.apply(combineColumns, columnlist=['Processor','processor'], newColumn='Processor', axis=1)
#DFresources = DFresources.apply(strColumnToList, column='processor', axis=1)
#DFresources = DFresources.apply(replaceValue,key='processor', value='MvE', newvalue = 'Margarete van Ess',axis=1)
#DFresources= DFresources.rename(columns={"Processor": "processor"})
#print(DFresources.columns)
#listOfFields= ['temper', 'temperAmount', 'temperParticles']
#groups = DFresources.groupby(listOfFields)
#if type(DFresources['temper']) == list:

#DFresources.groupby(by='temper').size()
#for group in groups:
#print (statsDF)
#print(groups)

#DFresources=DFresources.apply(addModifiedEntry, axis=1)

#interestproperties = ['type', 'amount', 'condition', 'manufacturingMethod', 'temper', 'temperAmount', 'temperParticles','coatOutsideType', 'vesselpart', 'processor', 'clayColorOutside','period', 'clayColorInside', 'Firing','periodEnd', 'conditionPercent','FormComparisonCompendia', 'FormComparisonCompendiadetail', 'Drawing','decorationTechnique', 'vesselForm','DecorationComparisonCompendiadetail', 'DecorationComparisonCompendia','dimensionLength', 'dimensionWidth', 'dimensionThickness', 'ApplicationComparisonCompendia','ApplicationComparisonCompendiadetail']
#interestproperties = ['processor']

#for column in interestproperties:
    #listvalues = [str(i) for i in DFresources[column]]
    #print(column, set(listvalues) )
#print (json.dumps(DOC['docs'], indent=1))
#print(DOC['docs'][['identifier']=='WESid_13_43'])
#bulkSaveChanges(DOC, pouchDB_url_bulk, auth)
#x = statOfLiesWithin(selectedResources)
#p = statOfRessouceTypes(filteredResources)


#
#for doc in filteredResources:
    #print (doc)
    #doc['resource']['relations']['isRecordedIn'].clear() 
    #doc['resource']['relations']['isRecordedIn'].append(str(idOfIdentifier('WarkaEnvironsSurvey', auth=auth, pouchDB_url_find=pouchDB_url_find)))
    #doc = addModifiedEntry(doc)
    #print (doc)
    #saveChanges(doc, pouchDB_url_put= pouchDB_url_put, auth= auth)

#result_df = pd.DataFrame(result)
#print(result)
