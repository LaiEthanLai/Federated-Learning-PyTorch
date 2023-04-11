#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
from functools import partial
from einops.layers.torch import Reduce

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

conv3x3 = partial(nn.Conv2d, kernel_size = 3, padding=1, bias=False)
conv1x1 = partial(nn.Conv2d, kernel_size = 1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out

class MulitBranchCNN(nn.Module):
    def __init__(self, inplanes=16, layers = [4, 4, 4], num_classes = 10) -> None:
        super().__init__()

        self.inplanes = inplanes
        self.conv1 = conv3x3(3, self.inplanes)
        self.bn1 = nn.BatchNorm2d(16)
        self.lrelu = nn.LeakyReLU()
        self.expansion = 1
        self.layers = layers

        downsample = None
        stride = 1
        planes = 16
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        name = 'group0_layer'
        for i in range(layers[0]):
            setattr(self, f'{name}{i}',BasicBlock(self.inplanes, planes, stride, downsample))
        
        self.exit0 = nn.Sequential(
            conv3x3(self.inplanes, planes),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(),
            Reduce('b c h w -> b c', reduction='mean'),
            nn.Linear(planes, num_classes),
            nn.LeakyReLU()
        )

        
        downsample = None
        stride = 2
        planes = 32
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        name = 'group1_layer'
        setattr(self, f'{name}{0}',BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.expansion
        for i in range(1, layers[1]):
            setattr(self, f'{name}{i}',BasicBlock(self.inplanes, planes))
            
        
        self.exit1 = nn.Sequential(
            Reduce('b c h w -> b c', reduction='mean'),
            nn.Linear(planes, planes // 2),
            nn.LeakyReLU(),
            nn.Linear(planes // 2, num_classes),
            nn.LeakyReLU()
        )


        downsample = None
        stride = 2
        planes = 64
        self.inplanes = planes 
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        name = 'group2_layer'
        setattr(self, f'{name}{0}',BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.expansion
        for i in range(1, layers[2]):
            setattr(self, f'{name}{i}',BasicBlock(self.inplanes, planes))
            
        
        self.exit2 = nn.Sequential(
            Reduce('b c h w -> b c', reduction='mean'),
            nn.Linear(planes, planes // 2),
            nn.LeakyReLU(),
            nn.Linear(planes // 2, planes // 2),
            nn.LeakyReLU(),
            nn.Linear(planes // 2, num_classes),
            nn.LeakyReLU()
        )


    def forward(self, x, idx: int):

        x = self.lrelu(self.bn1(self.conv1(x)))

        for g in range(3):
            for l in range(self.layers[g]):
                # print(x.shape)
                # print(f'group{g}_layer{l}')
                # print(getattr(self, f'group{g}_layer{l}'))
                x = getattr(self, f'group{g}_layer{l}')(x)
            
            if idx == g:
                return getattr(self, f'exit{idx}')(x)
       
        return 