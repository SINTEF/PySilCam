# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import pandas as pd
import scipy
import tensorflow.keras as keras

'''
SilCam TensorFlow analysis for classification of particle types
'''


def check_model(model_path):
    '''
    Raises errors if classification model is not found, or if it is not a valid file.

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                  '/mnt/ARRAY/classifier/model/particle_classifier.h5'
                                  usually obtained from settings.NNClassify.model_path

    '''
    path, filename = os.path.split(model_path)
    if not h5py.is_hdf5(model_path):
        if os.path.exists(path) is False:
            raise Exception(
                path + ' not found. Please see '
                + 'github.com/SINTEF/PySilCam/wiki/Installation,-setup-and-contributions '
                + 'for help.')
        else:
            raise Exception(
                model_path + ' is not valid hdf5 file. The predition model now '
                + 'uses a tensorflow.keras .h5 file, not a .tfl file.')

    header_file = os.path.join(path, 'header.tfl.txt')
    if os.path.isfile(header_file) is False:
        raise Exception(header_file + ' not found')


def get_class_labels(model_path):
    '''
    Read the header file that defines the catagories of particles in the model

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                  '/testdata/model_name/particle_classifier.h5'
                                  usually obtained from settings.NNClassify.model_path

    Returns:
        class_labels (str)      : labelled catagories which can be predicted
     '''
    path, filename = os.path.split(model_path)
    header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
    class_labels = header.columns
    return class_labels


def load_model(model_path):
    '''
    Load the trained tensorflow keras model

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                  '/testdata/model_name/particle_classifier.h5'

    Returns:
        model (tf model object) : loaded tf.keras model from load_model()
    '''
    path, filename = os.path.split(model_path)
    header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
    class_labels = header.columns

    model = keras.models.load_model(model_path)

    return model, class_labels


def predict(img, model):
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
    img = (img - 195.17760394934288) / 56.10742134506719  # Image preprocessing that matches the TFL model

    # Predict
    prediction = model.predict(np.expand_dims(img, 0))

    return prediction
