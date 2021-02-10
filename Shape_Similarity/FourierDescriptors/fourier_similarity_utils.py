from typing import Tuple
from tqdm import tqdm

from .FourierDescriptorPhase import FourierDescriptorPhase
# pylint: disable=relative-beyond-top-level
from ..db_utils.db_requests import put_data_in_pouchdb


def fourier_featurevector_to_db(
        path: str,
        db_url: str,
        db_name: str,
        auth: Tuple[str, str],
        fd_harmonics: int = 20) -> None:
    """
    @brief: Read images in path, compute resnet feature vectors and PUT data into RUNNING PouchDB instance
            of idai-field
    @param path: Location of binary images.
    @param db_url: url of running pouchDB instance
    @db_name: name of database
    @param auth: authentication for PouchDB (username, password)
    @param fd_harmonics number of fourier descriptor (FD) harmonics
    """
    pouchDB_url = f'{db_url}/{db_name}'

    descriptor_generator = FourierDescriptorPhase(path, fd_harmonics)
    with tqdm(total=len(descriptor_generator)) as pbar:
        for feature in descriptor_generator:
            put_data_in_pouchdb(pouchDB_url,
                                auth=auth, feature=feature, feature_type='phaseFourier')
            pbar.update(1)
