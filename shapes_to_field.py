import os
from mining_pages_utils.tensorflow_utils import run_vesselprofile_segmentation
from Shape_Similarity.deep_learning_similarity_utils import featurevector_to_db


def process_files():
    data_path = "/home/Data/field-raw-images"
    seg_path = "/home/Data/field-segmented-images"
    seg_model = "/home/Data/Models/training_resnet34_512_512_epochs_100_2020-10-08.h5"
    auth = ('', 'LMA47ezK')
    db_url = 'http://host.docker.internal:3000'
    db_name = 'idaishapes'

    if not os.path.exists(seg_path):
        os.makedirs(seg_path)

    print("Perform segementation")
    run_vesselprofile_segmentation(
        data_path, seg_path, seg_model, mark_black_img=False, resize_img=False)

    print("Compute feature vectors")
    featurevector_to_db(seg_path, db_url, db_name, auth)


process_files()
