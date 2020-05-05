
import tensorflow as tf
import os
import random

DIR = "/home/tf_records/"

TRAIN_PART = 0.7

file = os.path.join(DIR, "default.tfrecord")

records = []
for record in tf.python_io.tf_record_iterator(file):
    # Print an example record and exit the loop
    # print(tf.train.Example.FromString(record))
    # break

    # add the record (a binary string) to the list of records
    records.append(record)

n_total = len(records)
split_idx = int(n_total * TRAIN_PART)

random.shuffle(records)

train = records[:split_idx]
test = records[split_idx:]

print("Length of records:", len(records))
print("Length train/test: %d/%d" % (len(train), len(test)))

# Acutally write the train and test files
test_writer = tf.io.TFRecordWriter(os.path.join(DIR, "test.tfrecord"))
train_writer = tf.io.TFRecordWriter(os.path.join(DIR, "train.tfrecord"))

for record in train:
    train_writer.write(record)

for record in test:
    test_writer.write(record)

test_writer.flush()
train_writer.flush()
