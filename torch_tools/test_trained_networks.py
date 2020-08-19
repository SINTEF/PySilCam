import os
import sys
import time
import glob
import numpy as np
# import matplotlib.pyplot as plt

import skimage
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# silcam_dir = "/Users/odin/Sintef/SilCam/PySilCam"
silcam_dir = "/home/william/SilCam/PySilCam"
sys.path.append(silcam_dir)

import torch_tools.ml_config as config
import torch_tools.ml_utils as util
import pysilcam.silcam_classify as sccl
# import pysilcam.postprocess as scpp

# data_dir = "/Users/odin/Sintef/SilCam"
data_dir = "/home/william/SilCam/pysilcam-testdata/unittest-data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
model_dir = os.path.join(data_dir, 'model', 'net_params_4.pt')


class CoapNet(nn.Module):
    '''
    We are assuming that the image_size is divisible by 4.
    '''

    def __init__(self, num_classes, image_size=32):
        super(CoapNet, self).__init__()
        # Assumes 2 max pool layers:
        self.num_conv_features = int((image_size / 4)**2 * 64)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(self.num_conv_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), (2, 2))
        x = self.dropout1(x)
        x = x.view(-1, self.num_conv_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


print("---- Getting the data:")
classes = util.find_classes(train_dir)
print(classes)

# In train_network.py there are test and val sets, corresponding
# to a 90/10 split of the TRAINING data, here we load all the test
# data, and use the val_transform.
print('---- Loading data:')
X = np.zeros([0, config.image_size, config.image_size, 3], dtype='uint8')
Y = np.zeros((0, len(classes)), dtype='uint8')

for c_ind, c in enumerate(classes):
    print('  ', c)
    filepath = os.path.join(test_dir, c)
    files = [f for f in os.listdir(filepath) if f.endswith('.tiff')]
    for f in files:
        im = skimage.io.imread(os.path.join(filepath, f))
        X = util.add_im_to_stack(X, im, imsize=config.image_size)
        Y = util.add_class_to_stack(Y, classes, c_ind)

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(config.rgb_means, config.rgb_stds),
])

testset_size = Y.shape[0]
testset = util.SilcamDataset(X, Y, transform=val_transform)
testloader = DataLoader(testset, batch_size=config.batch_size,
                        shuffle=True, num_workers=0)
print('Finished loading, total number of test images: ', testset_size)


print("---- Loading network:")


class COAP(nn.Module):
    def __init__(self):
        super(COAP, self).__init__()
        self.num_conv_features = int(128 * 8 * 8)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(self.num_conv_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), (2, 2))
        x = self.dropout1(x)
        x = x.view(-1, self.num_conv_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model_dir = os.path.join(data_dir, 'model', 'coap_32.pt')
net = COAP()


print("---- Testing network (pytorch version of tf model):")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load(model_dir))

correct = 0
total = 0
start_time = time.time()
with torch.no_grad():
    for images, labels in testloader:

        net.eval()
        net.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
run_time = time.time() - start_time
print('Predict time: ', str(run_time / testset_size))
print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

# This code was built around aya's code from branch pytorch-cpu. Very untidy, but this
# won't work on other branches.
# print("---- Testing Ayas version of COAP:")
# model, class_labels = sccl.load_model(model_path='/home/william/SilCam/pysilcam-testdata/COAPNetTorch/COAPModNet_GPU_model.pt')

# X = []
# Y = []
# for cat in class_labels:
#     files = glob.glob(os.path.join(test_dir, cat, '*.tiff'))
#     for f in files:
#         im = imread(f)
#         # im = scpp.explode_contrast(imread(f))
#         X.append(im)
#         Y.append(cat)

# correct = 0
# total = len(Y)
# start_time = time.time()
# for x, y in zip(X, Y):
#     prediction = sccl.predict(x, model)  # run prediction from silcam_classify
#     ind = np.argmax(prediction)  # find the highest score
#     if class_labels[ind] == y:
#         correct += 1
# run_time = time.time() - start_time
# print('Num images:', total)
# print('Predict time: ', str(run_time / total))
# print('Accuracy of the network on the test images: %d %%' % (
#     100 * correct / total))


print("---- Testing original TF model:")
model, class_labels = sccl.load_model_tf(model_path='/home/william/SilCam/pysilcam-testdata/tflmodel/particle-classifier.tfl')

correct = 0
total = len(Y)
start_time = time.time()
for x, y in zip(X, Y):
    prediction = sccl.predict_tf(x, model)  # run prediction from silcam_classify
    ind = np.argmax(prediction)  # find the highest score
    if class_labels[ind] == y:
        correct += 1
run_time = time.time() - start_time
print('Predict time: ', str(run_time / total))
print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
