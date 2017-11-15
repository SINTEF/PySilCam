# Import tflearn and some helpers
import matplotlib.pyplot as plt
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import numpy as np
import skimage.io
import skimage.transform
import os
import pandas as pd
# -----------------------------

DATABASE_PATH = '/mnt/ARRAY/silcam_classification_database'
IMXY = 32


# -----------------------------
def find_classes(d=DATABASE_PATH):
    classes = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    print(classes)
    return classes


def add_im_to_stack(stack,im):
    blank = np.zeros([1, IMXY, IMXY, 3],dtype='uint8')
    imrs = skimage.transform.resize(im, (IMXY, IMXY, 3), mode='reflect',
            preserve_range=True)
    imrs = np.uint8(imrs)
            
    stack = np.vstack((stack, blank))
    stack[-1,:] = imrs
    return stack


def add_class_to_stack(stack,classes,classification_index):
    tag = np.zeros((1,len(classes)),dtype='uint8')
    tag[0][classification_index] = 1
    stack = np.vstack((stack, tag[0]))
    return stack

# -----------------------------
print('Formatting database....')
classes = find_classes()

X = np.zeros([0, IMXY, IMXY, 3],dtype='uint8')
Y = np.zeros((0,len(classes)),dtype='uint8')

for c_ind, c in enumerate(classes):
    print('  ',c)
    filepath = os.path.join(DATABASE_PATH,c)
    #files = os.listdir(filepath)
    files = [o for o in os.listdir(filepath) if o.endswith('.tiff')]
    for f in files:
        im = skimage.io.imread(os.path.join(filepath,f))
        X = add_im_to_stack(X, im)
        Y = add_class_to_stack(Y, classes, c_ind)

print('  Done.')

print('Splitting validation and training data')

print('Toal shape:', np.shape(Y), np.shape(X))

X, Y = shuffle(X, Y)

X_test = np.zeros([0, IMXY, IMXY, 3],dtype='uint8')
Y_test = np.zeros((0,len(classes)),dtype='uint8')

for c in range(len(classes)):
    ind = np.argwhere(Y[:,c]==1)
    print(len(ind),'images in class',c)
    #step = np.max([int(np.round(len(ind)/10)),1])
    step = 10
    print('  to be shortened by', step)
    ind = np.array(ind[np.arange(0,len(ind),step)]).flatten()
    #ind = np.array(ind[0::step]).flatten()

    Y_test = np.vstack((Y_test,Y[ind,:]))
    X_test = np.vstack((X_test,X[ind,:,:,:]))
    print('  test shape:', np.shape(Y_test), np.shape(X_test))

    Y = np.delete(Y,ind,0)
    X = np.delete(X,ind,0)
    print('  data shape:', np.shape(Y), np.shape(X))

print('OK.')

# -----------------------------
df = pd.DataFrame(columns = classes)
df.to_csv('header.tfl.txt', index=False)
# -----------------------------

outputs = np.shape(Y)[1]

X = np.float64(X)
Y = np.float64(Y)
X_test = np.float64(X_test)
Y_test = np.float64(Y_test)

print('Shuffle the data')
X, Y = shuffle(X, Y)

print('Make sure the data is normalized')
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
print('make extra data')
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

print('Define our network architecture:')

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

print('Step 1: Convolution')
network = conv_2d(network, 32, 3, activation='relu')

print('Step 2: Max pooling')
network = max_pool_2d(network, 2)

print('Step 3: Convolution again')
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.75)

# Step 8: Fully-connected neural network with outputs to make the final prediction
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
print("Network trained and saved as particle-classifier.tfl!")
