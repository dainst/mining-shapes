
import tensorflow as tf
import os
import zipfile
import shutil
import random

from tensorflow.python.data.ops.dataset_ops import DatasetSpec

DIR = "E:/Traindata/Trainingdata_fromCVAT/profile_segmentation/"

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

def importAsDataset(path):
    for batch in tf.data.TFRecordDataset(path).map(decode_fn):
        print(batch)
    



def converttolist(path):
    records = []
    for record in tf.data.Iterator(path):
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

def decode_fn(record_bytes):
    features = {
    # Extract features using the keys set during creation
    "image/class/label":    tf.io.FixedLenFeature([], tf.int64),
    "image/encoded":        tf.io.VarLenFeature(tf.string),
    }
    return tf.io.parse_single_example(record_bytes, features)


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

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

dataset = importAsDataset("E:/Traindata/Trainingdata_fromCVAT/profile_segmentation/1default.tfrecord")
#print(dataset)
#decoded = decode_fn(dataset)
#dataset_length = [i for i,_ in enumerate(decoded)][-1] + 1
#print(dataset_length)   
#train,test=traintestsplit(record)
    #trains.extend(train)
    #tests.extend(test)
#writetrainandtest(trains,tests, 'settest')

               

