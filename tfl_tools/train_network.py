from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import numpy as np
import logging

logging.basicConfig(filename='train_network.log',level=logging.DEBUG)

logging.info('Load the data set')
X, Y = pickle.load(open("DATA.pkl", "rb"))
X2, Y2 = pickle.load(open("DATA_12.pkl", "rb"))
X = np.vstack((X,X2))
Y = np.vstack((Y,Y2))

X_test, Y_test = pickle.load(open("DATA_backup.pkl", "rb"))
logging.info('pickle file loaded')

outputs = np.shape(Y)[1]
logging.info(outputs,' outputs found')

X = np.float64(X)
Y = np.float64(Y)
X_test = np.float64(X_test)
Y_test = np.float64(Y_test)

logging.info('Shuffle the data')
X, Y = shuffle(X, Y)

logging.info('Make sure the data is normalized')
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
logging.info('make extra data')
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

logging.info('Define our network architecture:')

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

logging.info('Step 1: Convolution')
network = conv_2d(network, 32, 3, activation='relu')

logging.info('Step 2: Max pooling')
network = max_pool_2d(network, 2)

logging.info('Step 3: Convolution again')
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, outputs, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='particle-classifier.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=200, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='particle-classifier')

# Save model when training is complete to a file
model.save("particle-classifier.tfl")
logging.info("Network trained and saved as particle-classifier.tfl!")
