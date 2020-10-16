
import os
import argparse
from mining_pages_utils.tensorflow_utils import run_vesselprofile_segmentation, run_point_detection


def process_vesselprofiles(_args):

    OUTPATH = _args.outpath
    SEGMENTPATH = os.path.join(OUTPATH, 'segmented_profiles/')
    POINTPATH = os.path.join(OUTPATH, 'points/')
    VESSELLPATH = _args.vesselpath

    SEG_MODEL = _args.segment_model
    POINT_MODEL = _args.point_model

    # Create output dirs
    for path in [POINTPATH, SEGMENTPATH]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Profile segmentation
    print('Perform image segmentation')
    run_vesselprofile_segmentation(VESSELLPATH, SEGMENTPATH, SEG_MODEL)

    # Point detection
    print('Perform point detection')
    run_point_detection(VESSELLPATH, POINTPATH, POINT_MODEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run profile segmentation and perform point detection."
    )
    parser.add_argument("--vesselpath", type=str,
                        help="Path in which vesselprofiles are located", required=True)
    parser.add_argument("--outpath", type=str,
                        help="Output path to save processed vesselprofiles", required=True)
    parser.add_argument("--segment_model", type=str,
                        help="Path of segmetation model (should be a Keras .h5 file)", required=True)
    parser.add_argument("--point_model", type=str,
                        help="Path of point detection model (should be a Keras .h5 file)", required=True)

    args = parser.parse_args()
    process_vesselprofiles(args)
