#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Alex Wang'



from load_image import *
import tensorflow as tf
# import vgg16
from xception import *
import os


data_path = '/home/wsf/sources/tensorflow-image-detection/training_dataset/'
train_logs_dir = './train_log'
val_logs_dir = './val_log'
model_path = './model.pb'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

IMAGE_WIDTH = 299
IMAGE_LENGTH = 299
IMAGE_DEPTH = 3
BATCH_SIZE = 32
num_class = 27
TRAIN_RECORDS_NAME = "train.tfrecords"
TEST_RECORDS_NAME = "test.tfrecords"
TRAIN_RECORDS_PATH = data_path + TRAIN_RECORDS_NAME
TEST_RECORDS_PATH = data_path + TEST_RECORDS_NAME


train_batch, train_label_batch = read_and_decode(TRAIN_RECORDS_PATH, num_class, BATCH_SIZE)
test_batch, test_label_batch = read_and_decode(TEST_RECORDS_PATH, num_class, BATCH_SIZE)


X = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3], name="input")
Y = tf.placeholder(dtype=tf.int64, shape=[None, num_class])

#replace model
logits,end_points = xception(X,num_classes=num_class)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
tf.summary.scalar("loss", loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuarcy", accuracy)

summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(train_logs_dir, sess.graph)
    val_writer = tf.summary.FileWriter(val_logs_dir,sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in np.arange(30000):
        if coord.should_stop():
            break

        train_images, train_labels = sess.run([train_batch, train_label_batch])
        val_images, val_labels = sess.run([test_batch, test_label_batch])

        train_summary, _, train_loss, train_acc = sess.run([summary_op, train_op, loss, accuracy], feed_dict={X: train_images, Y: train_labels})
        val_summary, val_loss, val_acc = sess.run([summary_op,loss, accuracy], feed_dict={X: val_images, Y: val_labels})
        train_writer.add_summary(train_summary, step)
        val_writer.add_summary(val_summary,step)

        if step % 10 == 0:
            print('Step %d, loss %f, acc %.2f%% --- * val_loss %f, val_acc %.2f%%' % (
            step, train_loss, train_acc * 100.0, val_loss, val_acc * 100.0))

    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,['input','Xception/block15_logits'])
    with tf.gfile.GFile(model_path,"wb") as f:
        f.write(output_graph_def.SerializeToString())

    checkpoint_path = os.path.join(train_logs_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=step)

    coord.request_stop()
    coord.join(threads)

