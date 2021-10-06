
import tensorflow as tf
import os
from pathlib import Path
import datetime
import pandas as pd
import numpy as np
#import Keras
#from datumaro.components.project import Project
from datumaro.components.dataset import Dataset
import datumaro.plugins.transforms as transforms
from datumaro.components.project import Environment, Project
from datumaro.components.operations import merge_categories, MergingStrategy
from datumaro.components.operations import IntersectMerge
from datumaro.components.extractor import (Importer, Extractor, Transform, DatasetItem, Bbox, AnnotationType, Label,
    LabelCategories, PointsCategories, MaskCategories)
import zipfile
import shutil
#import random
#from sklearn.model_selection import train_test_split
#print(tf.version.VERSION)

DIR = "E:/Traindata/Trainingdata_fromCVAT/mining_pages"
TRAIN_PART = 0.7

def provide_recorddf(path) -> pd.DataFrame:
    list_of_files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            row = {}
            if filename in ['test.tfrecord', 'val.tfrecord', 'train.tfrecord']:
                row['filename'] = filename
                row['filepath'] = os.path.join(dirpath, filename)
                list_of_files.append(row)
                print(row)
    return pd.DataFrame(list_of_files)

def unzip_tfrecords(path):
    i = 1
    listoftfrecordfiles = []
    for file in os.listdir(path):
        if file.endswith('.zip'):

                listoftfrecordfiles.append(tfrecordfiles)
                #zip_ref.extractall(DIR)
                i = i + 1
    return listoftfrecordfiles




def dataset_shapes(dataset):
    try:
        return [x.get_shape().as_list() for x in dataset._tensors]
    except TypeError:
        return dataset._tensors.get_shape().as_list()

def loadtfrecord(path):

    dataset = tf.data.TFRecordDataset(path, compression_type=None, buffer_size=None, num_parallel_reads=None)
    return dataset

def converttolist(path):
    records = []
    for record in tf.data.Iterator(path):
        records.append(record)
    return records

def traintestsplit(dataset):
    split = 3
    dataset_train = dataset.window(split, split + 1).flat_map(lambda ds: ds)
    dataset_test = dataset.skip(split).window(1, split + 1).flat_map(lambda ds: ds)
    return dataset_train , dataset_test

def writetrainandtest(train,test, i):
    test_writer = os.path.join(DIR, 'x0' +str(i) + "_test.tfrecord")
    train_writer = os.path.join(DIR, 'x0' + str(i) + "_train.tfrecord")

    writer = tf.data.experimental.TFRecordWriter(test_writer)
    writer.write(test)
    writer = tf.data.experimental.TFRecordWriter(train_writer)
    writer.write(train)
def extract_fn(data_record):
    features = {
        # Extract features using the keys set during creation
        "image/class/label":    tf.FixedLenFeature([], tf.int64),
        "image/encoded":        tf.VarLenFeature(tf.string),
    }
    sample = tf.parse_single_example(data_record, features)
    label = sample['image/class/label']
    dense = tf.sparse_tensor_to_dense(sample['image/encoded'])

    # Comment it if you got an error and inspect just dense:
    image = tf.image.decode_image(dense, dtype=tf.float32) 

    return dense, image, label
def correctionsMiningPages(listoftrainsets,listoftestsets,listofvalsets):
    merger = IntersectMerge()
    merged_trainset = merger(listoftrainsets)
    del listoftrainsets
    merged_trainset = merged_trainset.transform('remap_labels', {'stampbox': 'infoframe' }, default='keep')
    merged_trainset = merged_trainset.transform('remap_labels', {'pageid': 'figureid' }, default='keep')
    trainset_path = os.path.join(DIR, 'trainset.tfrecord')
    print('trainset')
    print(merged_trainset.categories())
    print(len(merged_trainset))
    merged_trainset.export(trainset_path, 'tf_detection_api', save_images=True)
    merged_testset = merger(listoftestsets)
    del listoftestsets

    merged_testset = merged_testset.transform('remap_labels', {'stampbox': 'infoframe' }, default='keep')
    
    merged_testset = merged_testset.transform('remap_labels', {'pageid': 'figureid' }, default='keep')
    testset_path = os.path.join(DIR, 'testset.tfrecord')
    print('testset')
    print(merged_testset.categories())
    print(len(merged_testset))
    merged_testset.export(testset_path, 'tf_detection_api', save_images=True)
    merged_valset = merger(listofvalsets)
    del listofvalsets

    merged_valset = merged_valset.transform('remap_labels', {'stampbox': 'infoframe' }, default='keep')
    merged_valset = merged_valset.transform('remap_labels', {'pageid': 'figureid' }, default='keep')
    valset_path = os.path.join(DIR, 'valset.tfrecord')
    print('valset')
    print(merged_valset.categories())
    print(len(merged_valset))
    merged_valset.export(valset_path, 'tf_detection_api', save_images=True)

def splitEachRecord(listoftfrecordfiles):
    listoftrainsets = []
    listoftestsets = []
    listofvalsets = []
    for record in listoftfrecordfiles:
        print(record['name'])
        
        
        dataset= Dataset.import_from(record['tfrpath'], 'tf_detection_api')
        #cleanset = dataset.select(lambda item: len(item.annotations) <= 2)
        #for item in cleanset:
            #print(item.annotations)


        print(record['tfrpath'])
        if 'task_mining_figures_zenonid_000066595_300pages-2021_02_16_12_47_10-tfrecord 1.0' in record['name']:
            print(len(dataset))
            dataset=dataset.select(lambda item: len(item.annotations) != 0)
            print(len(dataset))
        if 'task_mining_pages_zenonid_000147534_selectedpages-2021_02_23_16_00_02-tfrecord 1.0' in record['name']:
            dataset = dataset.transform('remap_labels', {'stampfigure':'stampfigure','vesselprofilefigure': 'vesselprofilefigure', 'pageid':'pageid', 'pageinfo':'pageinfo', 'vesselimage':'vesselimage' }, default='delete')
        if 'task_mining_pages_zenonid_000009465-2020_11_10_09_30_39-tfrecord 1.0' in record['name']:
            dataset = dataset.transform('remap_labels', {'stampfigure':'stampfigure','vesselprofilefigure': 'vesselprofilefigure', 'pageid':'pageid', 'pageinfo':'pageinfo', 'vesselimage':'vesselimage' }, default='delete')
        if 'task_zenonid_000267012_and_zenonid_001508696-2020_11_20_09_34_56-tfrecord 1.0' in record['name']:
            dataset = dataset.transform('remap_labels', {'stampfigure':'stampfigure','vesselprofilefigure': 'vesselprofilefigure', 'pageid':'pageid', 'pageinfo':'pageinfo', 'vesselimage':'vesselimage' }, default='delete')
        if 'task_zenonid_001344933_and_zenonid_001346932-2020_11_12_14_13_46-tfrecord 1.0' in record['name']:
            dataset = dataset.transform('remap_labels', {'stampfigure':'stampfigure','vesselprofilefigure': 'vesselprofilefigure', 'pageid':'pageid', 'pageinfo':'pageinfo', 'vesselimage':'vesselimage' }, default='delete')

        splitted = transforms.RandomSplit(dataset, splits=[('train', 0.60), ('test', 0.15), ('val', 0.25)])
        train = splitted.get_subset('train')
        trainset = Dataset.from_extractors(train)
        test = splitted.get_subset('test')
        testset = Dataset.from_extractors(test)
        val = splitted.get_subset('val')
        valset = Dataset.from_extractors(val)
        #train, test = splitter
        listoftrainsets.append(trainset)
        listoftestsets.append(testset)
        listofvalsets.append(valset)

    return listoftrainsets,listoftestsets,listofvalsets

listofrecords = provide_recorddf(DIR) 
print(listofrecords.iloc[0]['filepath'])
onimageList = []
annotationList = []
for index,record in listofrecords.iterrows():
    dataset= Dataset.import_from(record['filepath'], 'tf_detection_api')
    for data in dataset:
        dict = vars(data)
        onimageList.append(dict)
        for anno in dict['annotations']:
            annoplus = vars(anno)
            annoplus['page']=dict['id']
            annotationList.append(annoplus)

annotationDF = pd.DataFrame(annotationList)
#print(len(annotationDF.groupby('label')))
#print(annotationDF.iloc[10])
for name,group in annotationDF.groupby('label'):
    print(name, len(group))
    #print(record['filepath'], len(dataset))
#dataset = Environment().make_importer('E:/Traindata/Trainingdata_fromCVAT/mining_figures/Bonifay2004quick/testset.tfrecord').make_dataset()
# load a Datumaro project
#project = Project.load('E:/Traindata/Trainingdata_fromCVAT/mining_pages/datumaroproject')



#dataset = Dataset.from_extractors(dataset1, dataset2)
#ms = MergingStrategy()

#print(merged.categories())
#print(dataset2.categories())
#mergecats = ms.merge([dataset1.categories(), dataset2.categories()])
#categories = dataset.transform.categories()
#print(categories.get(AnnotationType.label))
#newdataset = dataset.transform('remap_labels', {'pageid': 'dog' }, default='delete')
#print(mergecats)
#for key, value in categories.items() :
    #print(type(categories[key]))
#print(dataset[0])
#for item in dataset:
    #print(item)



  #print(item.annotations)
#dataset.export(cocofile, 'coco')
# create a dataset
#dataset = project.make_dataset()

# keep only annotated images
#dataset.select(lambda item: len(item.annotations) != 0)
#print(dataset.categories())

# change dataset labels
#dataset.transform('remap_labels', {'0': '666'}, default = 'delete')

# iterate over dataset elements
#i = 0
#for item in dataset:
    #if i < 20:
        #for a in item.annotations:
            #print(a.label)
    #i = i +1 

#{<AnnotationType.label: 1>: LabelCategories(attributes=set(), items=[LabelCategories.Category(name='pageid', parent='', attributes=set()), LabelCategories.Category(name='pageinfo', parent='', attributes=set()), LabelCategories.Category(name='vesselprofilefigure', parent='', attributes=set()), LabelCategories.Category(name='vesselimage', parent='', attributes=set()), LabelCategories.Category(name='infoframe', parent='', attributes=set()), LabelCategories.Category(name='stampfigure', parent='', attributes=set())], _indices={'pageid': 0, 'pageinfo': 1, 'vesselprofilefigure': 2, 'vesselimage': 3, 'infoframe': 4, 'stampfigure': 5})}
    #for i in item.annotations:
        #print(i)
  #print(item.id, item.annotations)

# export the resulting dataset in COCO format
#dataset.export('dst/dir', 'coco')
#dataset = tf.data.TFRecordDataset(filename, compression_type=None, buffer_size=None, num_parallel_reads=None)
#for example in tf.compat.v1.python_io.tf_record_iterator(filename):
    #print(tf.train.Example.FromString(example))
#for element in dataset:
    #print(elemen)
#print(list(dataset.as_numpy_iterator()))

#dataset = dataset.map(extract_fn)
#iterator = dataset.make_one_shot_iterator()
#next_element = iterator.get_next()


#tf.enable_eager_execution()
#for images, labels in dataset.take(1):  # only take first element of dataset
    #numpy_images = images.numpy()
    #numpy_labels = labels.numpy()
    #print(numpy_labels)
    #image = image.reshape(IMAGE_SHAPE)


#for tfrecords in listoftfrecordfiles:
    #dataset=loadtfrecord(tfrecords['tfrpath'])
    #for element in dataset.as_numpy_iterator():

    #train, test = traintestsplit(record)
    #writetrainandtest(train,test, tfrecords['id'])
