  # -*- coding: utf-8 -*-
import torch
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf
import scipy
import numpy as np
import pandas as pd
import os

from torch_tools.net import *
from torch_tools.dataloader import *
from torchvision import transforms
from torch import nn
from skimage import io


'''
SilCam TensorFlow analysis for classification of particle types
'''


def check_model(model_path):
    '''
    Raises errors if classification model is not found

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                  '/mnt/ARRAY/classifier/model/particle-classifier.tfl'
                                  usually obtained from settings.NNClassify.model_path

    '''
    path, filename = os.path.split(model_path)
    if os.path.exists(path) is False:
        raise Exception(path + ' not found')

    header_file = os.path.join(path, 'header.tfl.txt')
    if os.path.isfile(header_file) is False:
        raise Exception(header_file + ' not found')


def get_class_labels(model_path='/mnt/ARRAY/classifier/model/particle-classifier.tfl'):
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


def load_model_tf(model_path='/mnt/ARRAY/classifier/model/particle-classifier.tfl'):
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


def load_model(model_path='/mnt/ARRAY/classifier/model/particle-classifier.pt'):
    '''
    Load the trained torch model

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                '/mnt/ARRAY/classifier/model/particle-classifier.pt'

    Returns:
        model (tf model object) : loaded tfl model from load_model()
    '''
    path, filename = os.path.split(model_path)
    header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
    # OUTPUTS = len(header.columns)
    class_labels = header.columns

    model = COAPNet(num_classes=len(class_labels))
    name = 'COAPModNet'
    # remap everything onto CPU: loading weights trained on GPU to CPU
    # model.load_state_dict(torch.load(model_path,
    #                                  map_location=lambda storage, loc: storage))  # 'cpu'
    # model.load_state_dict(torch.load(model_path))   # ,map_location='cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    return model, class_labels


def predict(img, model):
    '''
    Use torch model to classify particles

    Args:
        img (uint8)             : a particle ROI, corrected and treated with the silcam
                                explode_contrast function
        model (torch model object) : loaded torch model from load_model()

    Returns:
        prediction (array)      : the probability of the roi belonging to each class
    '''

    # Scale it to 32x32
    # img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

    # Predict
    # prediction = model.predict([img])
    # print('before reading the image ')
    # image = io.imread(img)
    # print('io.imread(img) ', img.shape)
    image = transform.resize(img, (64, 64))
    # print('transform.resize(image, (64, 64)) ', image.shape)
    image = image.transpose((2, 0, 1))
    # print('image.transpose((2, 0, 1)) ', image.shape)
    image = torch.from_numpy(image).float()
    # print('torch.from_numpy(image).float() ', image.shape)
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image = norm(image)
    # print('image.shape ', image.shape)
    image = image[np.newaxis, :]
    # print('image.shape', image.shape)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    out_predict = model(image.float())
    # print('out_predict: ', out_predict)
    out_predict = nn.Softmax(dim=1)(out_predict)
    # print('out_predict after applying softmax: ', out_predict)
    out_predict = out_predict.cpu().detach().numpy()  # = torch.var(out_predict).data.numpy()
    # print('out_predict numpy array: ', out_predict)

    return out_predict
