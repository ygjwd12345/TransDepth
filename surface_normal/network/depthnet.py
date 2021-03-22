##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Guanglei Yang
## Department of Information Engineering and Computer Science, University of Trento
## Email: guanglei.yang@studenti.unitn.it or yangguanglei.phd@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree

## 2020.2.28 change 2 * width to width
## change 378 to 384
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
import sys, os
from encoding.nn import AttentionGraphCondKernel
from torch.nn import BatchNorm2d

model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, norm_layer=BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
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

        out = out + residual
        out = self.relu_inplace(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, norm_layer=None, **kwargs):
        self.attentiongraph = True
        self.inplanes = 128
        self.feat_h = 240
        self.feat_w = 320
        self.img_height = 480
        self.img_width =640
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = norm_layer(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = norm_layer(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.dec_conv3 = self.make_conv(512, 256, 1, 1, 0)
        self.dec_deconv4 = self.make_deconv(1024, 384, 4, 2, 1)
        self.dec_deconv5 = self.make_deconv(2048, 512, 8, 4, 2)
        
        self.dropout_ = nn.Dropout(p=0.4)
            
        #self.final_deconv = self.make_deconv(512+384+256, int((512+384+256)/2), 4, 2, 1) 

        self.pred_scale5 = self.make_prediction(256, self.num_classes, 4, 2, 1)
        self.pred_scale4 = self.make_prediction(384, self.num_classes, 4, 2, 1)
        self.pred_scale3 = self.make_prediction(256, self.num_classes, 4, 2, 1)

        self.AttentionGraphCondKernel = AttentionGraphCondKernel([256, 384, 512], width=256, norm_layer=norm_layer) if self.attentiongraph else None

        # self.head = NonLocal(2048, num_classes)
        # self.head = NonLocalCat(2048, num_classes)
        #self.head = Glore(2048, 512, num_classes)
        # self.head = LessAtt(2048, num_classes)
        # self.head = LessAttCat(2048, num_classes)
        # self.head = LessAttHigh(2048, num_classes)
        #self.head = LessAttHighGLU(2048, num_classes)

        # self.head = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=0, bias=False),
        #                           InPlaceABNSync(512),
        #                           nn.Dropout2d(0.1),
        #                           nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        #self.dsn = nn.Sequential(
        #    nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
        #    eLU(inplace=True),),
        #    nn.Dropout2d(0.1),
        #    nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        #)

    def make_deconv(self, input_channels, output_channels, kernel_size_, stride_, padding_):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size_, stride=stride_, padding=padding_),
            self.norm_layer(output_channels),
            nn.ReLU(inplace=True),
            nn.Upsample([self.feat_h, self.feat_w], mode='bilinear')
            )

    def make_conv(self, input_channels, output_channels, kernel_size_, stride_, padding_):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size_, stride=stride_, padding=padding_),
            self.norm_layer(output_channels),
            nn.ReLU(inplace=True),
            nn.Upsample([self.feat_h, self.feat_w], mode='bilinear')
            )
    
    def make_prediction(self, input_channels, output_channels, kernel_size_, stride_, padding_):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size_, stride=stride_, padding=padding_),
            nn.Upsample([self.img_height, self.img_width], mode='bilinear')
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid), norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x,epoch=100,rank=1):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)

        x = self.layer2(x)
        dec_conv3_f = self.dec_conv3(x)
        pred_scale3_ = self.pred_scale3(dec_conv3_f)

        x = self.layer3(x)
        dec_deconv4_f = self.dec_deconv4(x)
        pred_scale4_ = self.pred_scale4(dec_deconv4_f)
        
        x = self.layer4(x)
        dec_deconv5_f = self.dec_deconv5(x)
      
        if self.attentiongraph:
            final_ = self.AttentionGraphCondKernel(dec_conv3_f, dec_deconv4_f, dec_deconv5_f,rank)
            pred_final = self.pred_scale5(final_)
            return tuple([pred_final, pred_scale4_, pred_scale3_])
        return tuple([pred_scale4_, pred_scale3_]) 

def DepthNet(num_classes=1, norm_layer=BatchNorm2d):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, norm_layer)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model
