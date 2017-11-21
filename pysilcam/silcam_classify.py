  # -*- coding: utf-8 -*-
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import pandas as pd
import os

'''
SilCam TensorFlow analysis for classification of particle types
'''


def get_class_labels(model_path='/mnt/ARRAY/classifier/model/particle-classifier.tfl'):
    ''' read the header file that defines the catagories

    class_labels = get_class_labels(model_path='/mnt/ARRAY/classifier/model/particle-classifier.tfl')
     '''
    headerfile = path, filename = os.path.split(model_path)
    header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
    OUTPUTS = len(header.columns)
    class_labels = header.columns
    return class_labels


def load_model(model_path='/mnt/ARRAY/classifier/model/particle-classifier.tfl'):
    ''' load the trained tfl model
    
    model, class_labels = load_model(model_path='/mnt/ARRAY/classifier/model/particle-classifier.tfl')

    model is the trained tf model ready for use in prediction
    '''
    headerfile = path, filename = os.path.split(model_path)
    header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
    OUTPUTS = len(header.columns)
    class_labels = header.columns


    # Same network definition as before
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)

    network = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.75)
    network = fully_connected(network, OUTPUTS, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    model = tflearn.DNN(network, tensorboard_verbose=0,
            checkpoint_path=model_path)
    model.load(model_path)

    return model, class_labels

def predict(img, model):
    ''' use tfl model to classify particles

    prediction = predict(img, model)

    img is a particle ROI, corrected and treated with the silcam
    explode_contrast function

    model is the loaded tfl model

    prediction is the probability of the roi belonging to each class
    '''
    # Load the image file
    #img = scipy.ndimage.imread(args.image, mode="RGB")

    # Scale it to 32x32
    img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

    # Predict
    prediction = model.predict([img])

    #print('other,  copepod,  diatom chain')
    #print(prediction)
    return prediction

