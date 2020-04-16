import torch
import torch.nn as nn

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=False, **kwargs):
        super(Unit, self).__init__()
        self.max_pool = max_pool

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels) # added to improve the training accuracy
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        if (self.max_pool):
            output = self.maxpool(output)

        return output

'''
COAP net structure as described by Bjarne
network = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep))
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = fully_connected(network, 256, activation='relu')
network = fully_connected(network, 256, activation='relu')
network = fully_connected(network, OUTPUTS, activation='softmax'
'''
class COAPNet(nn.Module):
    def __init__(self, num_classes=10):
        super(COAPNet, self).__init__()

        self.unit1 = Unit(in_channels=3, out_channels=64, max_pool=True)
        self.unit2 = Unit(in_channels=64, out_channels=128, max_pool=True)
        self.unit3 = Unit(in_channels=128, out_channels=256, max_pool=True)
        self.unit4 = Unit(in_channels=256, out_channels=512, max_pool=True)
        # Add all the units into the Sequential layer in exact order
        self.features = nn.Sequential(self.unit1,
                                      self.unit2,
                                      self.unit3,
                                      self.unit4) # input image size is 64x64x3: 32-16-8-4

        self.classifier = nn.Sequential(
            nn.Linear(in_features=4*4*512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, input):
        output = self.features(input)
        output = output.view(-1, 4*4*512)
        output = self.classifier(output)
        #output = nn.Softmax(dim=1)(output)  # removed to improve the training accuracy
        return output
