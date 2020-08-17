import os
import sys
import time
import numpy as np

import skimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# silcam_dir = "/Users/odin/Sintef/SilCam/PySilCam"
silcam_dir = "/home/william/SilCam/PySilCam"
sys.path.append(silcam_dir)

import torch_tools.ml_config as config
import torch_tools.ml_utils as util
from torch_tools.network import COAP

# data_dir = "/Users/odin/Sintef/SilCam"
data_dir = "/home/william/SilCam/pysilcam-testdata/unittest-data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")


print('====== Loading training data:')

classes = util.find_classes(train_dir)
# If we use the random crop, then we shuold load the images in full size
# then crop as a part of the transformers.
# X = np.zeros([0, config.image_size_on_load, config.image_size_on_load, 3], dtype='uint8')
X = np.zeros([0, config.image_size, config.image_size, 3], dtype='uint8')
Y = np.zeros((0, len(classes)), dtype='uint8')

for c_ind, c in enumerate(classes):
    print('  ', c)
    filepath = os.path.join(train_dir, c)
    files = [f for f in os.listdir(filepath) if f.endswith('.tiff')]
    for f in files:
        im = skimage.io.imread(os.path.join(filepath, f))
#         X = util.add_im_to_stack(X, im, imsize=config.image_size_on_load)
        X = util.add_im_to_stack(X, im, imsize=config.image_size)
        Y = util.add_class_to_stack(Y, classes, c_ind)


print('Splitting train and validation data.')
print('Toal shape:', np.shape(Y), np.shape(X))
X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                  test_size=config.validation_frac,
                                                  random_state=config.random_state,
                                                  stratify=Y)

train_transform = transforms.Compose([
    transforms.Lambda(util.gaussian_blur),
    transforms.ToPILImage(),
    # transforms.RandomCrop(config.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-config.aug_max_angle, config.aug_max_angle)),
    transforms.ToTensor(),
    transforms.Normalize(config.rgb_means, config.rgb_stds),
])
val_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize(config.rgb_means, config.rgb_stds),
])

trainset = util.SilcamDataset(X_train, Y_train, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=config.batch_size,
                         shuffle=True, num_workers=0)
testset = util.SilcamDataset(X_val, Y_val, transform=val_transform)
testloader = DataLoader(testset, batch_size=config.batch_size,
                        shuffle=True, num_workers=0)


def train_net(net, epochs, criterion, optimizer, model_dir, min_epochs_save=10):

    best_acc = 0

    start_time = time.time()
    for epoch in range(epochs):
        for data in trainloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
        print('[%d] loss: %.3f' %(epoch + 1, loss), end=' ')
        print("Time: %d" % (time.time() - start_time), end=' ')
        acc = util.calc_accuracy(net, testloader)
        print("Acc: {:.2f}".format(acc), end=' ')

        # If the val-accuracy is the highest yet, save the model
        # if epoch > min_epochs_save and acc > best_acc:
        if acc > best_acc:
            torch.save(net.state_dict(), model_dir)
            best_acc = acc
            print('Saved', end=' ')
        print()
        start_time = time.time()

    print('Finished training, hope it worked!')


run_name = 'coap_32_testing'
criterion = CrossEntropyLoss()

net = COAP()
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, eps=config.epsilon)
model_dir = os.path.join(data_dir, 'model', run_name + '.pt')
print('======  Training network: ' + run_name)
print('======  Params:')
print('  image_size = {}'.format(config.image_size))
print('  learning_rate = {}'.format(config.learning_rate))
print('  epsilon = {}'.format(config.epsilon))
print('  epochs = {}'.format(config.epochs))
print('======  Optimizer:')
print(optimizer)
print('======  Network:')
print(net)
start_time_training = time.time()
train_net(net, config.epochs, criterion, optimizer, model_dir)
print('======  Done in {:.1f} mins \n'.format((time.time() - start_time_training) / 60))
