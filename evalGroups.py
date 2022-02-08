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
auth = ('admin', 'Celt1!wedged')
authlocal = ('','blub')
db_url = 'http://field.dainst.org/sync'
db_name = 'shapes_import'
pouchDB_url_find = f'{db_url}/{db_name}/_find'
pouchDB_url_put = f'{db_url}/{db_name}/'
pouchDB_url_bulk = f'{db_url}/{db_name}/_bulk_docs'
imagestore = '/home/imagestore/'
exportProject = 'shapes_import'
FIG_SIZE = [20,30]


def allDocsToDf(DOC):
    insideDOC = [dict['doc'].get('resource') for dict in DOC['rows']]
    insideDOC = [i for i in insideDOC if i]
    print(len(insideDOC))
    DFdocs = pd.DataFrame(insideDOC)
    #print(DFdocs.columns)
    docfields = list(set(flatten([list(dict['doc'].keys()) for dict in DOC['rows']])))
    #print(docfields)
    #DFdocs = DFdocs.drop('resource', axis=1)
 
    return DFdocs, docfields
   

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

def getAllDocsv2(db_name):
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
            print()
    else:
        response = requests.get(pouchDB_url_all, auth=auth,params = {'include_docs':True})
        collect = json.loads(response.text)

    return collect

def getAllDocs(db_name):
    pouchDB_url_find = f'{db_url}/{db_name}/_find'
    querydict={'selector':{}}
    querydict['selector']['_id'] = {'$gt': None}
    #print(requests.post(pouchDB_url_find, auth=auth, json=querydict))
    response = requests.post(pouchDB_url_find, auth=auth, json=querydict)
    result = json.loads(response.text)
    return result


def flatten(t):
    return [item for sublist in t for item in sublist]

def getListOfDBs():
    pouchDB_url_alldbs = f'{db_url}/_all_dbs'
    response = requests.get(pouchDB_url_alldbs, auth=auth)
    result = json.loads(response.text)
    return result

def extract_json_objects(text, decoder=JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data

    Does not attempt to look for JSON arrays, text, or other JSON types outside
    of a parent JSON object.

    """
    pos = 0
    while True:
        match = text.find('{', pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            yield result
            pos = match + index
        except ValueError:
            pos = match + 1


alldblist = getListOfDBs()
#alldblist = [db for db in alldblist if not db.endswith('_ed') or db.endswith('_edv2') ]
print(alldblist)
testdbs = ['bourgou','uruk','gadara_bm',  'goebekli_tepe', 'kalapodi-mig2019', 'sel-akropolis', 'seli-bauteile', 'meninx-project', 'selinunt-1','fgap', 'ayamonte', 'pergamongrabung-1', 'wuq18','monte-turcisi-project','olympia_sht', 'bogazkoy-hattusa', 'heliopolis-project','milet']
sel = ['pergamongrabung-1']
with open('C:/Users/mhaibt/Documents/GitHub/idai-field/core/config/Library/Forms.json') as f:
    forms = json.load(f)
    f.close()
with open('C:/Users/mhaibt/Documents/GitHub/idai-field/core/config/Library/Categories.json') as f:
    categoriesrel = json.load(f)
    f.close()
with open('C:/Users/mhaibt/Documents/GitHub/mining-shapes/builtin.json') as f:
    builtins = json.load(f)
    f.close()
#regex = re.compile('([^\s,:}{[\]]+)')
#result = re.sub(regex, r'"\1"', text)
#result = re.sub(r'"true"', 'True', result)
#result = re.sub(r'"false"', 'false', result)
#with open('C:/Users/mhaibt/Documents/GitHub/mining-shapes/builtin.json','w') as f:
    #f.write(result)
#print(builtins)
#builtin = json.loads(result)
#regex = re.compile("true"', 'true', result)
#print(builtins.keys())
#print(categoriesrel.keys())
#builtandcore = z = {**builtins, **categoriesrel}
#print(builtandcore.keys())
allfieldssoll = []
allgroupnames = []
for key in forms.keys():
    category = forms[key]['categoryName']
    print(category)
    if category in categoriesrel.keys():
        parent = categoriesrel[category].get('parent')
    if category in builtins.keys():
        parent = builtins[category].get('parent')

    #coreparent = categoriesrel[category].get('parent')
    #builtinparent = builtins[category].get('parent')
        
    #builtparent = builtins[category].get('parent')

    fieldsfromcategory = flatten([group['fields'] for group in forms[key]['groups']])
    if parent:
        print('And this is my parent:', parent)
        formsparentfields = []
        corefields = []
        if parent in forms.keys():
            print('parent defined in forms.json')
            formsparentfields = flatten([group['fields'] for group in forms[parent]['groups']])
        if parent in categoriesrel.keys():
            print('Parent defined in categories.json')
            coreparentfields = list(categoriesrel[parent]['fields'].keys())
        if parent in builtins.keys():
            print('parent defined in builtin.json')
            builtinparentfields = flatten([group['fields'] for group in builtins[parent]['minimalForm']['groups']])

        fieldsfromcategory = fieldsfromcategory + formsparentfields + coreparentfields + builtinparentfields
    fieldsfromcategory = [category +'.' + item for item in fieldsfromcategory]
    print(fieldsfromcategory)
    groupnamefromcategory = [group['name'] for group in forms[key]['groups']]
    allfieldssoll = allfieldssoll + fieldsfromcategory
    allgroupnames.append(groupnamefromcategory)
#print(list(set(flatten(allgroupnames))))
print(list(set(allfieldssoll)))
corefields = list(set(allfieldssoll))
def fieldValueFrequency(df, fieldValue):
    df.plot(fieldValue, kind = 'hist')


mergeS = pd.Series()
mergeDf = pd.DataFrame()
#mergeSList = []

for db in testdbs:
    dbCount = pd.Series()
    print ('Processing ' + db)
    pathTemp = 'C:/Users/mhaibt/Documents/evalIdaiField/'+ db + '.json'

    
    if not Path(pathTemp).exists():
        allDOCs = getAllDocsv2(db)
        with open('C:/Users/mhaibt/Documents/evalIdaiField/'+ db + '.json', 'w') as f:
            f.write(json.dumps(allDOCs))
    with open('C:/Users/mhaibt/Documents/evalIdaiField/'+ db + '.json', 'r') as f:
        allDOCs = json.load(f)
    if allDOCs['total_rows'] > 3:
        df, docfields = allDocsToDf(allDOCs)
        #print(df.columns)
        #print(type(df))
        #plt.figure()
        
        #df['type'].value_counts().plot(ax=ax, kind='bar')
        #print('Single df:',type(df.count()))
        for name, group in df.groupby('type'):
            for column in group.columns:
                #print (column)
                group.rename({ column : name + '.' + column }, axis='columns', inplace = True)
            groupcount = group.count()
            groupcount.rename(db + '.' + name, inplace = True)
            groupcount = groupcount.where(lambda x : x!=0).dropna()
            #groupcount = groupcount.where(lambda x : ~groupcount.keys().endswith('.identifier')).dropna()
            #mergedf = mergedf.add(groupcount, fill_value=0)
            #print(groupcount)
            dbCount = dbCount.add(groupcount, fill_value=0)
    
    #dbCount = dbCount.where(lambda x : x!=0).dropna()
    #dbCount = dbCount[[not i.endswith(('.identifier','.relations','.id','.type', '.originalFilename','Photo.width','Photo.height','Photo.shortDescription','Drawing.width','Drawing.height','.geometry', '.featureVectors', '.shortDescription' )) for i in dbCount.index]]
    dbCount.rename(db , inplace = True)
    #print(dbCount)
    #fig, ax = plt.subplots()
    #dbCount.plot(ax=ax, kind='bar')
    
    plt.show()

    mergeDf=mergeDf.append(dbCount)
    #mergedf = mergedf.add(dbCount, fill_value=0)        
            #print(groupcount)
            #print(type(group), group.columns )
            #mergedf = mergedf.append(group.count() )
            
        
        #mergedf = pd.concat([mergedf, df.count()], ignore_index=True,axis=1)
    #pd.set_option("display.max_rows", None, "display.max_columns", None) 
    #mergedf = mergedf.where(lambda x : x!=0).dropna()
    #mergedf = mergedf.where(~'.identifier').dropna()
    #print(mergedf)
        #df.groupby('type').plot(kind = 'bar')
#mergedf = mergedf.groupby(by=mergedf.columns, axis=1).sum()


fig, ax = plt.subplots()
mergeDfT = mergeDf.T
#print(mergeDfT)
mergeDfT['sum'] = mergeDfT.sum(axis=1)
#mergeDfT.sort_values('sum', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last', ignore_index=False, key=None)
mergeDfT.sort_index(axis= 0, ascending = True, inplace=True)
mergeDfT.drop(['sum'], axis=1, inplace=True)
mergeDfT.fillna(0, inplace=True)
#mergeDfT= mergeDfT[mergeDfT.sum(axis=1) > 19]
coreselect = [rowname for rowname in mergeDfT.index.tolist() if rowname in corefields]
print('Length of coreselect:', len(coreselect))
if 'pottery.identifier' in corefields:
    print('Ja ist da')
print('Length of corefields', len(corefields))
mergeDfT = mergeDfT.loc[coreselect]
notcoreselect = [field for field in corefields if not field in coreselect]
for field in notcoreselect:
    new_row = pd.Series(data={}, name= field)
    mergeDfT = mergeDfT.append(new_row, ignore_index=False)
#append row to the dataframe
mergeDfT.fillna(0, inplace=True)
mergeDfT.sort_index(axis= 0, ascending = True, inplace=True)
print('This is the length of the final DF: ', len(mergeDfT))
mergeDfT.to_csv('C:/Users/mhaibt/Documents/evalIdaiField/AllDBs_usageOfCorefields.csv')  

#print()
#mergeDfT = mergeDfT[(mergeDfT != 0)]
#print(mergeDfT.index.tolist())
#mergeDfT = mergeDfT.dropna()
#print(mergeDfT)
#mergeDfT.plot(xlabel="new x", ylabel="new y")
#plt.figure(figsize=(8, 6), dpi=80)
#ax.tick_params(axis='y', which='major', pad=10)
#fig.set_dpi(300)
fig.set_size_inches(20, 120)
ax.xaxis.tick_top()
sns.color_palette('pastel', len(testdbs))
#colors = {'milet':'red', 'heliopolis-project':'green', 'bogazkoy-hattusa':'blue', 'olympia_sht':'yellow', 'monte-turcisi-project':'orange', 'wuq18':'purple','pergamongrabung-1': 'grey', 'ayamonte':''}
mergeDfT.plot.barh(xlabel="new x", ylabel="new y",stacked=True, color=sns.color_palette(cc.glasbey, len(testdbs)), fontsize=8, ax=ax)
#sns.plotbarh(xlabel="new x", ylabel="new y", data = mergeDfT, stacked=True, fontsize=6, ax=ax)
fig.savefig('C:/Users/mhaibt/Documents/evalIdaiField/test2png.png', dpi=150)
#ax.set_xticklabels(fontsize = 10)
#plt.show()
        
    
        #perprojectstring = json.dumps(perproject)
        #with open('C:/Users/mhaibt/Documents/allProjectsallFields.json', 'a') as f:
            #f.write(perprojectstring + '\n')



    



    

