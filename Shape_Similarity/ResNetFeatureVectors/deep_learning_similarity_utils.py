"""
Helper function for Deep Learning based shape similarity computation
"""
from tensorflow import keras
from typing import Tuple
from tqdm import tqdm
# pylint: disable=relative-beyond-top-level
from .resnetfeaturevectors import ResnetFeatureVectors
from ..db_utils.db_requests import put_data_in_pouchdb


def resnet_featurevector_to_db(
        path: str,
        db_url: str,
        db_name: str,
        auth: Tuple[str, str]) -> None:
    """
    @brief: Read images in path, compute resnet feature vectors and PUT data into RUNNING PouchDB instance
            of idai-field
    @param path: Location of binary images.
    @param db_url: url of running pouchDB instance
    @db_name: name of database
    @param auth: authentication for PouchDB (username, password)
    """
    pouchDB_url = f'{db_url}/{db_name}'

    model = keras.applications.resnet50.ResNet50(
        include_top=False, pooling='avg', weights='imagenet')

    vector_generator = ResnetFeatureVectors(path, model)
    with tqdm(total=len(vector_generator)) as pbar:
        for feature in vector_generator:
            put_data_in_pouchdb(pouchDB_url,
                                auth=auth, feature=feature, feature_type='resnet')
            pbar.update(1)
