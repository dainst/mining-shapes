import os
import tensorflow as tf
from mining_pages_utils.tensorflow_utils import run_vesselprofile_segmentation, readNoextensionImage, build_detectfn
from Shape_Similarity.ResNetFeatureVectors.deep_learning_similarity_utils import resnet_featurevector_to_db
from Shape_Similarity.FourierDescriptors.fourier_similarity_utils import fourier_featurevector_to_db
from mining_pages_utils.request_utils import getListOfDBs
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from PIL import Image
from IPython.display import display
import datetime
import numpy
import requests
import json

seg_model = "E:/mining_shapes/MODELS/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8_profilesegment/saved_model"
auth = ('', 'blub')
db_url = 'http://host.docker.internal:3000'
PATH_TO_LABELS = "E:/mining_shapes/MODELS/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8_profilesegment/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

descriptor = 'resnet'




def getListOfDBs():
    response = requests.get(pouchDB_url_alldbs, auth=auth)
    result = json.loads(response.text)
    return result

def process_files(db_name):
    if not os.path.exists(seg_path):
        os.mkdir(seg_path)
    vessel_image_list = os.listdir(data_path)
    print(vessel_image_list)
    miningfiguresdetectfn = build_detectfn(seg_model)

    for vessel in vessel_image_list:
        image = readNoextensionImage(data_path + vessel)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        output_dict = miningfiguresdetectfn(input_tensor)
        num_detections = int(output_dict.pop('num_detections'))
        print(num_detections)
        output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(numpy.int64)
        
        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])      
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        vis_util.visualize_boxes_and_labels_on_image_array(image, output_dict['detection_boxes'], output_dict['detection_classes'],output_dict['detection_scores'],category_index,instance_masks=output_dict.get('detection_masks_reframed', None),use_normalized_coordinates=True,line_thickness=8)
        display(Image.fromarray(image))



def computeFeaturevectors(db_name):
    print("Compute feature vectors")
    if descriptor == 'resnet':
        resnet_featurevector_to_db(seg_path, db_url, db_name, auth)
    elif descriptor == 'fourier':
        fourier_featurevector_to_db(seg_path, db_url, db_name, auth)
    else:
        raise ValueError('not valid descriptor type')

pouchDB_url_alldbs = f'{db_url}/_all_dbs'
alldbs = getListOfDBs()#
selectlist = ['lattara6_edv2', 'sidikhrebish_ed', 'urukcatalogs_ed', 'sabratha_ed', 'tallzira_ed', 'hayes1972_edv2', 'bonifay2004_ed', 'simitthus_ed']
list = [item for item in alldbs if item.endswith('_ed') or item.endswith('_edv2') ]
list = [item for item in list if not item.endswith('ock_ed')]
print(list)
imagestore = 'E:/mining_shapes/imagestore/'
for db_name in selectlist:
    data_path = imagestore + db_name +'/'
    seg_path = "E:/mining_shapes/segmented_profiles/" + db_name +'/'
    process_files(db_name)
    print(seg_path)
    #computeFeaturevectors(db_name)
