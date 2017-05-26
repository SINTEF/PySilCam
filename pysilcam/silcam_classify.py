  # -*- coding: utf-8 -*-
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np


def load_model(model_path='/mnt/ARRAY/classifier/model/particle-classifier.tfl'):
    OUTPUTS = 3

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
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, OUTPUTS, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    model = tflearn.DNN(network, tensorboard_verbose=0,
            checkpoint_path=model_path)
    model.load(model_path)

    return model

def predict(img, model):
    # Load the image file
    #img = scipy.ndimage.imread(args.image, mode="RGB")

    # Scale it to 32x32
    img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

    # Predict
    prediction = model.predict([img])

    #print('other,  copepod,  diatom chain')
    #print(prediction)
    return prediction


