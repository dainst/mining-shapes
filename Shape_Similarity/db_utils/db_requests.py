import requests
from typing import Tuple

from .featureentry import FeatureEntry


def put_data_in_pouchdb(url: str, auth: Tuple[str, str], feature: FeatureEntry, feature_type: str) -> None:
    doc_url = f"{url}/{feature.id}"
    res = requests.get(doc_url, auth=auth)
    if res.status_code != 404:
        payload = res.json()
        rev = payload['_rev']
        if not payload['resource']['featureVectors']:
            payload['resource']['featureVectors'] = {
                feature_type: feature.feature_vec}
        else:
            payload['resource']['featureVectors'][feature_type] = feature.feature_vec
        stat = requests.put(f"{doc_url}?rev={rev}", auth=auth, json=payload)
