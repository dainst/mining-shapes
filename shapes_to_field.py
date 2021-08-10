import os
from mining_pages_utils.tensorflow_utils import run_vesselprofile_segmentation
from Shape_Similarity.ResNetFeatureVectors.deep_learning_similarity_utils import resnet_featurevector_to_db
from Shape_Similarity.FourierDescriptors.fourier_similarity_utils import fourier_featurevector_to_db
from mining_pages_utils.request_utils import getListOfDBs
import datetime

import requests
import json

seg_model = "/home/models/saved-no_outlined-model-21-0.96.hdf5"
auth = ('', 'blub')
db_url = 'http://host.docker.internal:3000'

descriptor = 'resnet'




def getListOfDBs():
    response = requests.get(pouchDB_url_alldbs, auth=auth)
    result = json.loads(response.text)
    return result

def process_files(db_name):



    if not os.path.exists(seg_path):
        os.mkdir(seg_path)

    print("Perform segementation")
    run_vesselprofile_segmentation(
        data_path, seg_path, seg_model, mark_black_img=False, resize_img=True)

def computeFeaturevectors(db_name):
    print("Compute feature vectors")
    if descriptor == 'resnet':
        resnet_featurevector_to_db(seg_path, db_url, db_name, auth)
    elif descriptor == 'fourier':
        fourier_featurevector_to_db(seg_path, db_url, db_name, auth)
    else:
        raise ValueError('not valid descriptor type')

pouchDB_url_alldbs = f'{db_url}/_all_dbs'
alldbs = getListOfDBs()
list = [item for item in alldbs if item.endswith('_ed')]
list = [item for item in list if not item.endswith('ock_ed')]
print(list)
imagestore = '/home/imagestore/'
for db_name in list:
    data_path = "/home/imagestore/" + db_name +'/'
    seg_path = "/home/images/SEGMENT_RESULTS/" + db_name +'/'
    #process_files(db_name)
    print(seg_path)
    computeFeaturevectors(db_name)
