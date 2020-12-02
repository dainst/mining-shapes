
import tensorflow as tf
import os
import zipfile
import shutil
import random

DIR = "/home/trainmodels/INCOMING_TFrecords/"

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
                        tfrecordfiles['pbtxtpath'] = DIR + str(i) + zipinfo.filename
                        source = zip_ref.open(zipinfo.filename)
                        target = open(tfrecordfiles['pbtxtpath'], "wb")
                        with source, target:
                            shutil.copyfileobj(source, target)
                listoftfrecordfiles.append(tfrecordfiles)   
                #zip_ref.extractall(DIR)
                i = i + 1
    return listoftfrecordfiles


def converttolist(path):
    records = []
    for record in tf.python_io.tf_record_iterator(path):
        records.append(record)
    return records

def traintestsplit(tfrecord):
    n_total = len(tfrecord)
    split_idx = int(n_total * TRAIN_PART)

    random.shuffle(tfrecord)

    train = tfrecord[:split_idx]
    test = tfrecord[split_idx:]

    print("Length of records:", len(tfrecord))
    print("Length train/test: %d/%d" % (len(train), len(test)))
    return train,test 

def writetrainandtest(train,test, i):
    test_writer = tf.io.TFRecordWriter(os.path.join(DIR, str(i) + "_test.tfrecord"))
    train_writer = tf.io.TFRecordWriter(os.path.join(DIR, str(i) + "_train.tfrecord"))
    for record in train:
        train_writer.write(record)

    for record in test:
        test_writer.write(record)
    test_writer.flush()
    train_writer.flush()

listoftfrecordfiles = unzip_tfrecords(DIR)
trains = []
tests = []
for tfrecords in listoftfrecordfiles:
    record=converttolist(tfrecords['tfrpath'])
    train,test=traintestsplit(record)
    trains.extend(train)
    tests.extend(test)
writetrainandtest(trains,tests, 'settest')

               

