import requests
from typing import Tuple
from mining_pages_utils.request_utils import addModifiedEntry
from .featureentry import FeatureEntry
import datetime


def put_data_in_pouchdb(url: str, auth: Tuple[str, str], feature: FeatureEntry, feature_type: str) -> None:
    doc_url = f"{url}/{feature.id}"
    res = requests.get(doc_url, auth=auth)
    if res.status_code != 404:
        payload = res.json()
        rev = payload['_rev']
        if not 'featureVectors' in payload['resource'].keys():
            payload['resource']['featureVectors'] = {
                feature_type: feature.feature_vec}
        else:
            payload['resource']['featureVectors'][feature_type] = feature.feature_vec
        payload = addModifiedEntry(payload)
        stat = requests.put(f"{doc_url}?rev={rev}", auth=auth, json=payload)
