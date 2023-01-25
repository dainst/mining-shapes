import pyIdaifield as pydai
import json

def changeFigure (series):
    if series['type'] == 'Drawing' and series['literature'] and len(series['identifier'].split('_'))==3:
        series['literature'][0]['figure'] = series['identifier'].split('_')[2]
    return series
def changeQuotation (series):
    if series['literature'] :
        series['literature'][0]['quotation'] = 'Hayes, John W. (1972). Late Roman Pottery. London: British School at Rome. '
    return series

#auth = ('admin', 'Celt1!wedged')
auth = ('','blub')
db_url = 'http://field.dainst.org/sync'
db_url= 'http://localhost:3000'
imagestore = '/home/imagestore/'

dblist=['hayes1972_edv2']

alldocs = pydai.getAllDocs(db_url,auth,'hayes1972_edv2')
allDF, docfields = pydai.allDocsToDf(alldocs)
print(docfields)
changeDF = allDF[(allDF['type'] == 'Type') & (allDF['literature'].apply(lambda x: isinstance(x, list)))]
#changeDF = changeDF.apply(changeFigure, axis=1)
changeDF = changeDF.apply(changeQuotation, axis=1)
changeDF = changeDF.apply(pydai.addModifiedEntry, axis=1)
print(changeDF.columns)
DOCS = pydai.DFtoDOC(changeDF, docfields)#
pydai.bulkSaveChanges(db_url, auth, 'hayes1972_edv2', DOCS)
#print(json.dumps(DOCS, indent=4, sort_keys=True))
#print(changeDF['literature'].apply(lambda x: x[0].get('figure')))


