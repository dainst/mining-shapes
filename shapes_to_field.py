import os
from mining_pages_utils.tensorflow_utils import run_vesselprofile_segmentation
from Shape_Similarity.deep_learning_similarity_utils import post_featurevector_to_db


def process_files():
    data_path = "/home/Data/field-raw-images"
    seg_path = "/home/Data/field-segmented-images"
    seg_model = "/home/Data/Models/training_resnet34_512_512_epochs_100_2020-10-08.h5"

    if not os.path.exists(seg_path):
        os.makedirs(seg_path)

    print("Perform segementation")
    run_vesselprofile_segmentation(
        data_path, seg_path, seg_model, mark_black_img=False, resize_img=False)

    print("Compute feature vectors")
    post_featurevector_to_db("/home/Data/field-segmented-images")


process_files()
