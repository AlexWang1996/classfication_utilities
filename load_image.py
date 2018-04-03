
#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Alex Wang'

import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from split_dataset import *

IMAGE_LENGTH = 299
IMAGE_WIDTH = 299
IMAGE_DEPTH = 3


def make_tfrecords(data_path, TRAIN_RECORDS_NAME, TEST_RECORDS_NAME):
    train_images,test_images = prepare_train_and_test(data_path)
    train_writer = tf.python_io.TFRecordWriter(data_path + TRAIN_RECORDS_NAME)
    test_writer = tf.python_io.TFRecordWriter(data_path + TEST_RECORDS_NAME)

    for index,whole_image in enumerate(train_images):
        for image in whole_image:
                train_image = Image.open(image)
                train_image = train_image.resize((IMAGE_LENGTH,IMAGE_WIDTH))
                train_image_raw = train_image.tobytes()
                train_example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_image_raw]))
                }))
                train_writer.write(train_example.SerializeToString())
    train_writer.close()

    for index,whole_image in enumerate(test_images):
        for image in whole_image:
                test_image = Image.open(image)
                test_image = test_image.resize((IMAGE_LENGTH,IMAGE_WIDTH))
                test_image_raw = test_image.tobytes()
                test_example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[test_image_raw]))
                }))
                test_writer.write(test_example.SerializeToString())
    test_writer.close()




def read_and_decode(filename, n_classes, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [IMAGE_LENGTH, IMAGE_WIDTH, IMAGE_DEPTH])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                      batch_size=batch_size,
                                                      capacity=2000,
                                                      min_after_dequeue=1000)
    label_batch = tf.one_hot(label_batch, n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int64)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    return image_batch, label_batch



def get_batch(image,label,batch_size):
    image_batch, label_batch = tf.train.shuffle_batch([image,label],batch_size=batch_size,num_threads=16,capacity=20000,min_after_dequeue=1000)
    return image_batch,tf.reshape(label_batch,[batch_size])

def get_test_batch(image,label,batch_size):
    image_batch, label_batch=tf.train.shuffle_batch([image,label],batch_size=batch_size)
    return image_batch, tf.reshape(label_batch,[batch_size])

def test(data_path, TRAIN_RECORDS_NAME, TEST_RECORDS_NAME, BATCH_SIZE):
   # make_tfrecords(data_path)
    train_image,train_label = read_and_decode(data_path + TRAIN_RECORDS_NAME)
    test_image,test_label = read_and_decode(data_path + TEST_RECORDS_NAME)
    train_image_batch, train_label_batch = get_batch(train_image, train_label,BATCH_SIZE)
    test_image_batch, test_label_batch = get_batch(test_image, test_label, BATCH_SIZE)
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        batch_image_train, batch_label_train = session.run([train_image_batch, train_label_batch])
        batch_image_test, batch_label_test = session.run([test_image_batch, test_label_batch])
        print (len(batch_image_train))
        print (len(batch_image_test))
        print (len(batch_label_train))
        print (len(batch_label_test))
        coord.request_stop()
        coord.join(threads)

def show(data_path, tfrecords_name):
    filename_queue = tf.train.string_input_producer([tfrecords_name])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [IMAGE_LENGTH, IMAGE_WIDTH, IMAGE_DEPTH])
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1000):
            example, l = sess.run([image, label])
            img = Image.fromarray(example, 'RGB')
            img.save(data_path + str(i) + '_''Label_' + str(l) + '.jpg')
            print(example, l)
        coord.request_stop()
        coord.join(threads)




'''
if __name__ == '__main__':
    test()
'''