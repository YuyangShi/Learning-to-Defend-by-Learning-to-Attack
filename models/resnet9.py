import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0,pool=0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.pool = pool
        self.pooling = nn.MaxPool2d(pool)

    def forward(self, x):
        out = self.conv1(x)
        if self.pool != 0:
            out = self.pooling(out)
        out = self.bn1(out)
        out = self.relu1(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None


    def forward(self, x):
        if not self.equalInOut:
            raise('Network Error')
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = out.add(x)
        return out
        # if not self.equalInOut:
        #     x = self.relu1(self.bn1(x))
        # else:
        #     out = self.relu1(self.bn1(x))
        # out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        # if self.droprate > 0:
        #     out = F.dropout(out, p=self.droprate, training=self.training)
        # out = self.conv2(out)
        # return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class ResNet9(nn.Module):
    def __init__(self,num_classes=10, dropRate=0.0):
        super(ResNet9, self).__init__()
        nChannels = [64,128,256,512]


        self.conv1 = BasicBlock(3,nChannels[0],stride=1,pool=0)
        self.conv2 = BasicBlock(nChannels[0], nChannels[1], stride=1, pool=2)
        self.conv3 = BasicBlock(nChannels[1], nChannels[2], stride=1, pool=2)
        self.conv4 = BasicBlock(nChannels[2], nChannels[3], stride=1, pool=2)

        self.block1 = ResBlock(nChannels[1], nChannels[1], stride=1)
        self.block2 = ResBlock(nChannels[3], nChannels[3], stride=1)
        self.pool = nn.MaxPool2d(kernel_size=4)
        self.fc = nn.Linear(nChannels[3], num_classes)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.block1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.block2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



