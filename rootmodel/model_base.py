import torch
import torch.nn as nn
import torch.nn.functional as F
from rootmodel.model_util import *


# ResNet34
class ResNet34_1(nn.Module):
    def __init__(self):
        super(ResNet34_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BaLtchNorm2d(64)
        self.relu = nn.ReU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # [B,3,256,256] -> [B,64,64,64]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ResNet34_2(nn.Module):
    def __init__(self):
        super(ResNet34_2, self).__init__()
        self.layer1 = ResNet34_Block(64, 64)
        self.layer2 = ResNet34_Block(64, 64)
        self.layer3 = ResNet34_Block(64, 64)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # [B,64,64,64] -> [B,64,64,64]
        x_1 = self.relu(self.layer1(x) + x)
        x_2 = self.relu(self.layer2(x_1) + x_1)
        x_3 = self.relu(self.layer3(x_2) + x_2)
        return x_3

class ResNet34_3(nn.Module):
    def __init__(self):
        super(ResNet34_3, self).__init__()
        self.layer1 = ResNet34_Block(64, 128, s=2)
        self.downconv = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, 0),
            nn.BatchNorm2d(128)
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = ResNet34_Block(128, 128)
        self.layer3 = ResNet34_Block(128, 128)
        self.layer4 = ResNet34_Block(128, 128)

    def forward(self, x):
        # [B,64,64,64] -> [B,128,32,32]
        x_1 = self.relu(self.layer1(x) + self.downconv(x))
        x_2 = self.relu(self.layer2(x_1) + x_1)
        x_3 = self.relu(self.layer3(x_2) + x_2)
        x_4 = self.relu(self.layer4(x_3) + x_3)
        return x_4

class ResNet34_4(nn.Module):
    def __init__(self):
        super(ResNet34_4, self).__init__()
        self.layer1 = ResNet34_Block(128, 256, s=2)
        self.downconv = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, 0),
            nn.BatchNorm2d(256)
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = ResNet34_Block(256, 256)
        self.layer3 = ResNet34_Block(256, 256)
        self.layer4 = ResNet34_Block(256, 256)
        self.layer5 = ResNet34_Block(256, 256)
        self.layer6 = ResNet34_Block(256, 256)

    def forward(self, x):
        # [B,128,32,32] -> [B,256,16,16]
        x_1 = self.relu(self.layer1(x) + self.downconv(x))
        x_2 = self.relu(self.layer2(x_1) + x_1)
        x_3 = self.relu(self.layer3(x_2) + x_2)
        x_4 = self.relu(self.layer4(x_3) + x_3)
        x_5 = self.relu(self.layer5(x_4) + x_4)
        x_6 = self.relu(self.layer6(x_5) + x_5)
        return x_6

class ResNet34_5(nn.Module):
    def __init__(self):
        super(ResNet34_5, self).__init__()
        self.layer1 = ResNet34_Block(256, 512, s=2)
        self.downconv = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, 0),
            nn.BatchNorm2d(512)
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = ResNet34_Block(512, 512)
        self.layer3 = ResNet34_Block(512, 512)

    def forward(self, x):
        # [B,256,16,16] -> [B,512,8,8]
        x_1 = self.relu(self.layer1(x) + self.downconv(x))
        x_2 = self.relu(self.layer2(x_1) + x_1)
        x_3 = self.relu(self.layer3(x_2) + x_2)
        return x_3
