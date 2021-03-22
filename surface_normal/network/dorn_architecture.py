import os
import torch
import torch.nn as nn
import torchvision.models
import collections
import math
from AttentionGraphCondKernel import AttentionGraphCondKernel
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import numpy as np
def weights_init(modules, type='xavier'):
    m = modules
    if isinstance(m, nn.Conv2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            m.weight.data.fill_(1.0)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Module):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    m.weight.data.fill_(1.0)

                if m.bias is not None:
                    m.bias.data.zero_()


class FullImageEncoder(nn.Module):
    def __init__(self, dataset='kitti'):
        super(FullImageEncoder, self).__init__()
        self.global_pooling = nn.AvgPool2d(8, stride=8, padding=(1, 0))  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(2048 * 4 * 5, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 卷积
        self.upsample = nn.UpsamplingBilinear2d(size=(30, 40))  # KITTI 49X65 NYU 33X45
        self.dataset = dataset
        weights_init(self.modules(), 'xavier')

    def forward(self, x):
        x1 = self.global_pooling(x)

        # print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * 4 * 5)
        x4 = self.relu(self.global_fc(x3))
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)
        out = self.upsample(x5)
        return out


class SceneUnderstandingModuleBN(nn.Module):
    def __init__(self, output_channel=136, dataset='kitti', mode='L2'):
        super(SceneUnderstandingModuleBN, self).__init__()
        self.encoder = FullImageEncoder(dataset=dataset)
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=6, dilation=6),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=12, dilation=12),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=18, dilation=18),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512 * 5, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, output_channel, 1),
            # nn.UpsamplingBilinear2d(scale_factor=8)
            nn.UpsamplingBilinear2d(size=(240, 320))
        )
        weights_init(self.modules(), type='kaiming')
        self.mode = mode

    def forward(self, x):
        x1 = self.encoder(x)

        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print('cat x6 size:', x6.size())
        out = self.concat_process(x6)

        return out


class SceneUnderstandingModule(nn.Module):
    def __init__(self, output_channel=136, dataset='kitti', mode='L2'):
        super(SceneUnderstandingModule, self).__init__()
        self.encoder = FullImageEncoder(dataset=dataset)
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=18, dilation=18),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512 * 5, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, output_channel, 1),
            # nn.UpsamplingBilinear2d(scale_factor=8)
            nn.UpsamplingBilinear2d(size=(240, 320))
        )
        weights_init(self.modules(), type='xavier')
        self.mode = mode

    def forward(self, x):
        x1 = self.encoder(x)

        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print('cat x6 size:', x6.size())
        out = self.concat_process(x6)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, freeze=True):
        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(101)](pretrained=pretrained)

        self.channel = in_channels

        self.conv1 = nn.Sequential(collections.OrderedDict([
            ('conv1_1', nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn_2', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1_3', nn.BatchNorm2d(128)),
            ('relu1_3', nn.ReLU(inplace=True))
        ]))
        self.bn1 = nn.BatchNorm2d(128)
        # self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.layer2 = pretrained_model._modules['layer2']

        self.layer3 = pretrained_model._modules['layer3']
        self.layer3[0].conv2.stride = (1, 1)
        self.layer3[0].downsample[0].stride = (1, 1)

        self.layer4 = pretrained_model._modules['layer4']
        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)

        # clear memory
        del pretrained_model

        if pretrained:
            weights_init(self.conv1, type='kaiming')
            weights_init(self.layer1[0].conv1, type='kaiming')
            weights_init(self.layer1[0].downsample[0], type='kaiming')
            # weights_init(self.layer3[0].conv2, type='kaiming')
            # weights_init(self.layer3[0].downsample[0], type='kaiming')
            # weights_init(self.layer4[0].conv2, 'kaiming')
            # weights_init(self.layer4[0].downsample[0], 'kaiming')
        else:
            weights_init(self.modules(), type='kaiming')

        if freeze:
            self.freeze()

    def forward(self, x):
        # print(pretrained_model._modules)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # print('conv1:', x.size())

        x = self.maxpool(x)

        # print('pool:', x.size())

        x1 = self.layer1(x)
        # print('layer1 size:', x1.size())
        x2 = self.layer2(x1)
        # print('layer2 size:', x2.size())
        x3 = self.layer3(x2)
        # print('layer3 size:', x3.size())
        x4 = self.layer4(x3)
        # print('layer4 size:', x4.size())
        return x4

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class DORN(nn.Module):
    def __init__(self, output_size=(240, 320), pretrained=True, freeze=True, output_channel=3, training_mode='train_L2_loss'):
        super(DORN, self).__init__()

        self.output_size = output_size
        # self.feature_extractor = ResNet(pretrained=pretrained, freeze=freeze)
        self.aspp_module = SceneUnderstandingModule(output_channel=output_channel, mode=training_mode)
        ### vit
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        config_vit.patches.grid = (int( 320/ 16), int(240 / 16))
        self.encoder = ViT_seg(config_vit, img_size=[ 320,240], num_classes=config_vit.n_classes).cuda()
        self.encoder.load_from(weights=np.load(config_vit.pretrained_path))
        ### scale 3
        self.AttentionGraphCondKernel = AttentionGraphCondKernel([512, 1024, 2048], width=512)
    def forward(self, x):
        # x1 = self.feature_extractor(x)
        skip_feat = self.encoder(x)
        skip_feat[5] = self.AttentionGraphCondKernel(skip_feat[2],skip_feat[3],skip_feat[4],skip_feat[5],3)
        x2 = self.aspp_module(skip_feat[5])
        return x2