
import os
from mining_pages_utils.tensorflow_utils import run_vesselprofile_segmentation, run_point_detection


SEG_MODEL = '/home/Code/Shape_Segmentation/training_resnet34_512_512_epochs_100_2020-10-09.h5'
#POINT_MODEL = '/home/images/OUTPUT/Point_Detector_trained_80_2020-10-12.h5'
POINT_MODEL = '/home/Code/Normalize_Shape/point_network_trained_80_256x256_20200723_augdata.h5'

OUTPATH = '/home/images/OUTPUT/'
VESSELLPATH = OUTPATH + 'vesselprofiles/'
SEGMENTPATH = OUTPATH + 'segmented_profiles/'
POINTPATH = OUTPATH + 'points'

for path in [POINTPATH, SEGMENTPATH]:
    if not os.path.exists(path):
        os.makedirs(path)

# Profile segmentation
print('Perform image segmentation')
run_vesselprofile_segmentation(VESSELLPATH, SEGMENTPATH, SEG_MODEL)

# Point detection
print('Perform point detection')
run_point_detection(VESSELLPATH, POINTPATH, POINT_MODEL)
