import os
import sys
import time
import psutil
import numpy as np
# import matplotlib.pyplot as plt

import skimage
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import ml_config as config

# silcam_dir = "/Users/odin/Sintef/SilCam/PySilCam"
silcam_dir = "/home/william/SilCam/PySilCam"
sys.path.append(silcam_dir)

# data_dir = "/Users/odin/Sintef/SilCam"
data_dir = "/home/william/SilCam/pysilcam-testdata/unittest-data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
model_dir = os.path.join(data_dir, 'model', 'net_params_4.pt')


def find_classes(d=train_dir):
    classes = [c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))]
    classes = [c for c in classes if not c.startswith('.')]
    classes.sort()
    return classes


def add_im_to_stack(stack, im):
    im = skimage.transform.resize(im,
                                  (config.image_size, config.image_size, 3),
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


# def show_image(x, y=None):
#     plt.imshow(x)
#     if y is not None:
#         plt.title(get_class_name(y, classes))
#     else:
#         plt.title("Unknown class (y not given)")
#     plt.show()


def gaussian_blur(image):
    rand = np.random.rand() * config.aug_blur_max
    # skimage converts unit8 to float64
    image = skimage.filters.gaussian(image, sigma=rand, multichannel=True)
    image = image * 256
    return image.astype('uint8')


def to_numpy(torch_image):
    '''To convert a torch.Tensor (CxHxW) back to numpy (HxWxC)'''
    return (torch_image.numpy().transpose((1, 2, 0)) + 1) / 2

# First CoapNet like thing, give roughly 75% acc.
# class CoapNet(nn.Module):
#     '''
#     We are assuming that the image_size is divisible by 4.
#     '''

#     def __init__(self, num_classes, image_size=32):
#         super(CoapNet, self).__init__()
#         # Assumes 2 max pool layers:
#         self.num_conv_features = int((image_size / 4)**2 * 64)
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
#         self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
#         self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
#         self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
#         self.fc1 = nn.Linear(self.num_conv_features, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Dropout(0.75)
#         self.fc5 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, (2, 2))
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = F.max_pool2d(x, (2, 2))
#         x = x.view(-1, self.num_conv_features)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         x = self.fc5(x)
#         return x


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


def adjust_learning_rate(optimizer, epoch, start_lr):
    """Gradually decay learning rate"""
    lr = start_lr
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


print("---- Getting the Process ID:")
pid = os.getpid()
ps = psutil.Process(pid)
print("Pricess ID:", pid)

print("---- Getting the data:")
classes = find_classes()
print("Classes:", classes)

# with open('header.tfl.txt', 'w') as f:
#     f.write(",".join(classes))

X = np.zeros([0, config.image_size, config.image_size, 3], dtype='uint8')
Y = np.zeros((0, len(classes)), dtype='uint8')

print("Loading data for class:")
for c_ind, c in enumerate(classes):
    print('  ', c)
    filepath = os.path.join(train_dir, c)
    files = [f for f in os.listdir(filepath) if f.endswith('.tiff')]
    for f in files:
        im = skimage.io.imread(os.path.join(filepath, f))
        X = add_im_to_stack(X, im)
        Y = add_class_to_stack(Y, classes, c_ind)

print('---- Splitting train and validation data.')
print('Toal shape:', np.shape(Y), np.shape(X))
X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                  test_size=0.1,
                                                  random_state=config.random_state,
                                                  stratify=Y)

print("---- Defining the data cleaning and augmentation:")
train_transform = transforms.Compose([
    transforms.Lambda(gaussian_blur),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-config.aug_max_angle, config.aug_max_angle)),
    transforms.ToTensor(),
    transforms.Normalize(config.rgb_means, config.rgb_stds),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(config.rgb_means, config.rgb_stds),
])

# This is terrible form, but I believe in the original training
# all images are used in training, so testing this:
# train_set = SilcamDataset(X, Y, transform=train_transform)
train_set = SilcamDataset(X_train, Y_train, transform=train_transform)
trainloader = DataLoader(train_set,
                         batch_size=config.batch_size,
                         shuffle=True,
                         num_workers=0)

# Again, terrible, but following the logic above, we might as well test
# on all images.
# val_set = SilcamDataset(X, Y, transform=val_transform)
val_set = SilcamDataset(X_val, Y_val, transform=val_transform)
valloader = DataLoader(val_set,
                       batch_size=config.batch_size,
                       shuffle=True,
                       num_workers=0)

print("---- Training:")
# Can set a manual seed for the init of the network, this seems not
# to have an effect though
# torch.manual_seed(np.random.randint(0, 10000))
net = nn.Sequential(CoapNet(num_classes=len(classes)), nn.Softmax(1))
# print(net)

criterion = nn.CrossEntropyLoss()
# Have tried various optimisers, seems Adam results in the time it takes
# to run an epoch slowing. SGD doens't optimise well.
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in valloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Acc after init: %d %%' % (100 * correct / total))
# print('Confusion matrix:')
# print(confusion_matrix(labels, predicted))
for epoch in range(200):  # loop over the dataset multiple times

    running_loss = 0.0
    start_time = time.time()
    # Below line has modified LR, but it's commented:
    # adjust_learning_rate(optimizer, epoch, config.start_lr)
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print_freq = 20
    print('[Epoch %d] loss: %.3f' % (epoch + 1, loss), end=' lr: ')
    for param_group in optimizer.param_groups:
        print(param_group['lr'], end=' ')
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Acc: %d%%' % (100 * correct / total), end=" ")
    print("Time: %d" % (time.time() - start_time))
    # print(ps.memory_info())
print('Finished Training')

print(model_dir)
torch.save(net.state_dict(), model_dir)

print("---- Testing network:")
net = nn.Sequential(CoapNet(num_classes=len(classes)), nn.Softmax(1))
net.load_state_dict(torch.load(model_dir))

net.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in valloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
