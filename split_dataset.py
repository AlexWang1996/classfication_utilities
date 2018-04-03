#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Alex Wang'


from sklearn.model_selection import train_test_split
import os
import sys
import shutil

# data_path = '/home/wsf/sources/tensorflow-image-detection/training_dataset/'


'''
the path where the keras datasets will be found, when using keras ImageDataGenerator.flow_from_directory
'''
keras_train_path = '/home/wsf/sources/tensorflow-image-detection/training_dataset/train/'
keras_test_path = '/home/wsf/sources/tensorflow-image-detection/training_dataset/valid/'



def prepare_train_and_test(filepath):
    '''
    if not os.path.exists(keras_train_path):
        os.mkdir(keras_train_path)
    if not os.path.exists(keras_test_path):
        os.mkdir(keras_test_path)
    '''
    class_dist = os.listdir(filepath)
    image_whole_train_data = []
    image_whole_test_data = []
    print(len(class_dist))
    for index,item in enumerate(class_dist):
        if item != 'train' and item != 'valid':
            image_path = filepath + item + '/'
            train_dir = image_path +  'train'
            test_dir =  image_path +  'valid'
            image_data = []
            image_label = []
            for image in os.listdir(image_path):
                if image.endswith('.jpg'):
                    image_name = image_path + image
                    image_data.append(image_name)
                    image_label.append(index)
            image_data_train,image_data_test, image_label_train,image_label_test= \
                train_test_split(image_data,image_label,shuffle=True, random_state=1)
            image_whole_train_data.append(image_data_train)
            image_whole_test_data.append(image_data_test)

            '''
            for images in image_data_test:
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)
                shutil.copyfile(images,test_dir + "/" + images.split('/')[-1])
            if os.path.exists(keras_test_path + item):
                shutil.rmtree(keras_test_path + item)
            shutil.move(test_dir,keras_test_path)
            os.rename(keras_test_path + 'valid',keras_test_path + item)

            for images in image_data_train:
                if not os.path.exists(train_dir):
                    os.mkdir(train_dir)
                shutil.copyfile(images,train_dir + "/" + images.split('/')[-1])
            if os.path.exists(keras_train_path + item):
                shutil.rmtree(keras_train_path + item)
            shutil.move(train_dir, keras_train_path)
            os.rename(keras_train_path + 'train', keras_train_path + item)
            '''

            print(len(image_label_train))
            print(len(image_data_test))

    print(len(image_whole_train_data))
    print(len(image_whole_test_data))

    return image_whole_train_data, image_whole_test_data

'''
if __name__ == '__main__':
    prepare_train_and_test(data_path)
'''



