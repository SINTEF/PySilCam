import os
import sys
import time
import numpy as np

import skimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

# silcam_dir = "/Users/odin/Sintef/SilCam/PySilCam"
silcam_dir = "/home/william/SilCam/PySilCam"
sys.path.append(silcam_dir)

import torch_tools.ml_config as config
import torch_tools.ml_utils as util

# data_dir = "/Users/odin/Sintef/SilCam"
data_dir = "/home/william/SilCam/pysilcam-testdata/unittest-data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")


classes = util.find_classes(train_dir)
print(classes)

print('Formatting database (loading training data)....  again as I will try without RandomCrop.')
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
                                                  test_size=0.1,
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                          shuffle=True, num_workers=0)
testset = util.SilcamDataset(X_val, Y_val, transform=val_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                         shuffle=True, num_workers=0)


def calc_accuracy(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total


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
        acc = calc_accuracy(net)
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


run_name = 'coap_32'
learning_rate = 0.001
epsilon = 1e-08
epochs = 200
criterion = nn.CrossEntropyLoss()

net = COAP()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=epsilon)
model_dir = os.path.join(data_dir, 'model', run_name + '.pt')
print('======  Training network: ' + run_name)
print('======  Params:')
print('  image_size = {}'.format(config.image_size))
print('  learning_rate = {}'.format(learning_rate))
print('  epsilon = {}'.format(epsilon))
print('  epochs = {}'.format(epochs))
print('======  Optimizer:')
print(optimizer)
print('======  Network:')
print(net)
start_time_training = time.time()
train_net(net, epochs, criterion, optimizer, model_dir)
print('======  Done in {:.1f} mins \n'.format((time.time() - start_time_training) / 60))
