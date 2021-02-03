
import tensorflow as tf
import os
#import Keras
import zipfile
import shutil
#import random
#from sklearn.model_selection import train_test_split

DIR = "/home/images/apply/testtfrec/"

TRAIN_PART = 0.7


def unzip_tfrecords(path):
    i = 1
    listoftfrecordfiles = []
    for file in os.listdir(path):
        if file.endswith('.zip'):

            with zipfile.ZipFile(DIR + file, 'r') as zip_ref:
                zipinfos = zip_ref.infolist()
                tfrecordfiles = {}
                for zipinfo in zipinfos:
                    # This will do the renaming
                    #print(zipinfo.filename)
                    #zipinfo.filename = str(i) + str(zipinfo.filename)
                    print(zipinfo)
                    if str(zipinfo.filename).endswith('.tfrecord'):
                        tfrecordfiles['tfrpath'] = DIR + str(i) + zipinfo.filename
                        source = zip_ref.open(zipinfo.filename)
                        target = open(tfrecordfiles['tfrpath'], "wb")
                        with source, target:
                            shutil.copyfileobj(source, target)
                    if str(zipinfo.filename).endswith('.pbtxt'):
                        tfrecordfiles['id'] = i
                        tfrecordfiles['pbtxtpath'] = DIR + str(i) + zipinfo.filename
                        source = zip_ref.open(zipinfo.filename)
                        target = open(tfrecordfiles['pbtxtpath'], "wb")
                        with source, target:
                            shutil.copyfileobj(source, target)
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


listoftfrecordfiles = unzip_tfrecords(DIR)

for tfrecords in listoftfrecordfiles:
    record=loadtfrecord(tfrecords['tfrpath'])
    train, test = traintestsplit(record)
    writetrainandtest(train,test, tfrecords['id'])
