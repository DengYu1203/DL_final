#!/usr/bin/env python
# coding: utf-8

import sys

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import math
import torch.utils.model_zoo as model_zoo

class AM(nn.Module):
    
    def __init__(self, input_channel):
        
        super(AM, self).__init__() 
        
        self.conv_dilation_1_x_32 = torch.nn.Conv2d(input_channel, 32, kernel_size=3, padding=1, dilation=1)
        self.conv_dilation_1_32_32 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1)
        self.conv_dilation_2_32_32 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.conv_dilation_4_32_32 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4)
        self.conv_dilation_8_32_32 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=8, dilation=8)
        self.conv_1_32_32 = torch.nn.Conv2d(32, 32, kernel_size=1)
        self.conv_1_32_128 = torch.nn.Conv2d(32, 128, kernel_size=1)
        self.prelu = nn.PReLU()
        self.bn_128 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = self.conv_dilation_1_x_32(x)
        x = self.conv_dilation_1_32_32(x)
        x = self.conv_dilation_2_32_32(x)
        x = self.conv_dilation_2_32_32(x)
        x = self.conv_dilation_4_32_32(x)
        x = self.conv_dilation_4_32_32(x)
        x = self.conv_dilation_8_32_32(x)
        x = self.conv_1_32_32(x)
        x = self.conv_1_32_128(x)
        x = self.prelu(x)
        x = self.bn_128(x)
        
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # stride/2 maybe applied on conv1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # Conv + BatchNorm + RelU
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

        # Downsample: feature Map size/2 || Channel increase
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 conv
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet34(nn.Module):

    def __init__(self, block, layers = [3, 4, 6, 3], num_classes=1000):
        self.inplanes = 64
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # kaiming weight normal after default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # construct layer/stage conv2_x,conv3_x,conv4_x,conv5_x
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # when to need downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # inplanes expand for next block
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)

        return x

class UpScale(nn.Module):
    
    def __init__(self, input_num):
        
        super(UpScale, self).__init__() 
        
        self.conv = nn.Conv2d(input_num, 128, kernel_size=3,padding=1, bias=False)
        self.upsample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3,padding=1, bias=False),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=3,padding=1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(1)
        )
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.upsample_1(x)
        
        return x

class FAB_AMNet(nn.Module):
    
    def __init__(self):
        
        super(FAB_AMNet, self).__init__() 
        
        self.resnet34 = ResNet34(BasicBlock)
        self.AM_128 = AM(128)
        self.AM_128_1 = AM(128)
        self.AM_128_2 = AM(128)
        self.AM_3 = AM(3)
        self.UpScale_256_1 = UpScale(256)
        self.UpScale_256_2 = UpScale(256)
        self.UpScale_256_3 = UpScale(256)
        self.UpDim = torch.nn.Conv2d(3, 128, kernel_size=1)
        self.ReduceDim = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3,padding=1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()
        
        self.resnet34.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth'))
        
    def forward(self, x, y):
        
        x = self.resnet34(x)
        x = self.AM_128(x)
        y = self.resnet34(y)
        y = self.AM_128(y)
        z = torch.cat((x,y),dim = 1)
        fore = self.UpScale_256_1(z)
        fore_o = self.sigmoid(fore)
        back = self.UpScale_256_2(z)
        back_o = self.sigmoid(back)
        middle = self.UpScale_256_3(z)
        
        D1 = torch.cat((fore,middle,back),dim = 1)
        
        D2 = self.AM_3(D1)
        D1 = self.UpDim(D1)
        
        D2 = self.AM_128_1(D1+D2)
        D2 = self.AM_128_2(D1+D2)
        disparity = self.ReduceDim(D2)
        
        return fore_o, back_o, disparity