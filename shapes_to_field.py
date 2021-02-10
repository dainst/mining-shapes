import os
from mining_pages_utils.tensorflow_utils import run_vesselprofile_segmentation
from Shape_Similarity.ResNetFeatureVectors.deep_learning_similarity_utils import resnet_featurevector_to_db
from Shape_Similarity.FourierDescriptors.fourier_similarity_utils import fourier_featurevector_to_db


def process_files():
    data_path = "/home/Data/field-raw-images"
    seg_path = "/home/Data/field-segmented-images"
    seg_model = "/home/Data/Models/saved-no_outlined-model-17-0.96.hdf5"
    auth = ('', 'LMA47ezK')
    db_url = 'http://host.docker.internal:3000'
    db_name = 'idaishapes'
    descriptor = 'fourier'

    if not os.path.exists(seg_path):
        os.makedirs(seg_path)

    print("Perform segementation")
    run_vesselprofile_segmentation(
        data_path, seg_path, seg_model, mark_black_img=False, resize_img=True)

    print("Compute feature vectors")
    if descriptor == 'resnet':
        resnet_featurevector_to_db(seg_path, db_url, db_name, auth)
    elif descriptor == 'fourier':
        fourier_featurevector_to_db(seg_path, db_url, db_name, auth)
    else:
        raise ValueError('not valid descriptor type')


process_files()
