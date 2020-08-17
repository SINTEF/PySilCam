import os
import skimage
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import torch_tools.ml_config as config


# ---- Data loading

def find_classes(data_dir):
    classes = [c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))]
    classes = [c for c in classes if not c.startswith('.')]
    classes.sort()
    return classes


def add_im_to_stack(stack, im, imsize=config.image_size_on_load):
    im = skimage.transform.resize(im,
                                  (imsize, imsize, 3),
                                  mode='reflect',
                                  preserve_range=True)
    im = np.uint8(im)
    stack = np.vstack((stack, np.expand_dims(im, axis=0)))
    return stack


def add_class_to_stack(stack, classes, classification_index):
    tag = np.zeros((1, len(classes)), dtype='uint8')
    tag[0][classification_index] = 1
    stack = np.vstack((stack, tag[0]))
    return stack


def get_class_name(y, classes):
    return classes[np.argmax(y)]


def unison_shuffle_arrays(a, b):
    assert a.shape[0] == b.shape[0]
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]


def show_image(x, y=None, classes=None):
    plt.imshow(x)
    if (y is not None) and (classes is not None):
        plt.title(get_class_name(y, classes))
    else:
        plt.title("Unknown class (y not given)")
    plt.show()


class SilcamDataset(Dataset):

    def __init__(self, X, Y, transform=None):
        # These inputs should be nparrays (loaded outside of this)
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, index):
        image = self.X[index]
        label = self.Y[index].argmax()

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ---- Data processing / Augmentation

def gaussian_blur(image):
    rand = np.random.rand() * config.aug_blur_max
    # skimage converts unit8 to float64
    image = skimage.filters.gaussian(image, sigma=rand, multichannel=True)
    image = image * 256
    return image.astype('uint8')


def to_numpy(torch_image):
    '''To convert a torch.Tensor (CxHxW) back to numpy (HxWxC)'''
    return (torch_image.numpy().transpose((1, 2, 0)) + 1) / 2


# ---- Training

def calc_accuracy(net, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """
    Function to adjust learning rate when training. Currrently this
    is not used. If you wanted to use it you would do something like:

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, config.learning_rate)
        for data in trainloader:
            ...
    """
    lr = learning_rate
    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
