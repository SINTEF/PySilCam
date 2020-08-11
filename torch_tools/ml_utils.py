import os
import skimage
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


# ---- Data processing / Augmentation

def gaussian_blur(image):
    rand = np.random.rand()*config.aug_blur_max
    # skimage converts unit8 to float64
    image = skimage.filters.gaussian(image, sigma=rand, multichannel=True)
    image = image * 256
    return image.astype('uint8')


def to_numpy(torch_image):
    '''To convert a torch.Tensor (CxHxW) back to numpy (HxWxC)'''
    return (torch_image.numpy().transpose((1, 2, 0)) + 1) / 2
