# -*- coding: utf-8 -*-
import os
import scipy
import numpy as np
import pandas as pd
from skimage.transform import resize

import torch
from torch import nn

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from torch_tools.network import ParticleClassifier
import torch_tools.ml_utils as util
import torch_tools.ml_config as config

'''
SilCam classification of particle types. Built now using pytorch,
previously used tensorflow.
'''


def check_model(model_path):
    '''
    Raises errors if classification model is not found

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                  '/mnt/ARRAY/classifier/model/particle-classifier.pt'
                                  usually obtained from settings.NNClassify.model_path

    '''
    path, filename = os.path.split(model_path)
    if os.path.exists(path) is False:
        raise Exception(path + ' not found')

    header_file = os.path.join(path, 'header.tfl.txt')
    if os.path.isfile(header_file) is False:
        raise Exception(header_file + ' not found')


def get_class_labels(model_path):
    '''
    Read the header file that defines the catagories of particles in the model

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                  '/mnt/ARRAY/classifier/model/particle-classifier.tfl'
                                  usually obtained from settings.NNClassify.model_path

    Returns:
        class_labels (str)      : labelled catagories which can be predicted
     '''
    path, filename = os.path.split(model_path)
    header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
    class_labels = header.columns
    return class_labels


def load_model_tf(model_path):
    '''
    Load the trained tensorflow model

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                  '/mnt/ARRAY/classifier/model/particle-classifier.tfl'

    Returns:
        model (tf model object) : loaded tfl model from load_model()
    '''
    path, filename = os.path.split(model_path)
    header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
    OUTPUTS = len(header.columns)
    class_labels = header.columns

    tf.reset_default_graph()

    # Same network definition as in tfl_tools scripts
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


def predict_tf(img, model):
    '''
    Use tensorflow model to classify particles

    Args:
        img (uint8)             : a particle ROI, corrected and treated with the silcam
                                  explode_contrast function
        model (tf model object) : loaded tfl model from load_model()

    Returns:
        prediction (array)      : the probability of the roi belonging to each class
    '''

    # Scale it to 32x32
    img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

    # Predict
    prediction = model.predict([img])

    return prediction


def load_model(model_path):
    '''
    Load the trained torch model

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                    '/mnt/ARRAY/classifier/model/particle-classifier.pt'

    Returns:
        model (pt model object) : loaded pt model from load_model()
        class_labels            : list of the classes from which the model predicts
    '''

    class_labels = get_class_labels(model_path)

    model = ParticleClassifier()
    model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # We assume model will be being used in for prediction:
    model.eval()

    return model, class_labels


def predict(img, model):
    '''
    Use torch model to classify particles

    Args:
        img (uint8)                : a particle ROI, corrected and treated with the silcam
                                     explode_contrast function
        model (torch model object) : loaded torch model from load_model()

    Returns:
        prediction (array)         : the probability of the roi belonging to each class
    '''

    # print("img going in: ", type(img), img.shape, img.dtype)
    img = resize(img, (config.image_size, config.image_size, 3), mode='reflect', preserve_range=True)
    # print("img post resize in: ", type(img), img.shape, img.dtype)
    img = util.pred_transform(img.astype(np.uint8))
    # print("img after transform", type(img))

    prediction = model(img.unsqueeze(0))
    # print('prediction: ', prediction)
    prediction = nn.Softmax(dim=1)(prediction)
    # print('prediction after applying softmax: ', prediction)
    prediction = prediction.cpu().detach().numpy()  # = torch.var(prediction).data.numpy()
    # print('prediction numpy array: ', prediction)

    return prediction
