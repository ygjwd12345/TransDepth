import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models
import collections
import math
from AttentionGraphCondKernel import AttentionGraphCondKernel
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import numpy as np
from bts import bts
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


class ResNetPyramids(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, freeze=True, backbone='resnext101'):
        super(ResNetPyramids, self).__init__()
        if backbone == 'resnext101':
            pretrained_model = torchvision.models.__dict__['resnext101_32x8d'](pretrained=pretrained)
        else:
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

        if backbone == 'resnext101':
            self.layer1[0].conv1 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        else:
            self.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)


        self.layer2 = pretrained_model._modules['layer2']

        self.layer3 = pretrained_model._modules['layer3']
        # self.layer3[0].conv2.stride = (1, 1)
        # self.layer3[0].downsample[0].stride = (1, 1)

        self.layer4 = pretrained_model._modules['layer4']
        # self.layer4[0].conv2.stride = (1, 1)
        # self.layer4[0].downsample[0].stride = (1, 1)

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
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)

        # print('conv1:', x.size())

        x = self.maxpool(x)
        # print(x.shape)

        # print('pool:', x.size())

        x1 = self.layer1(x)
        # print('layer1 size:', x1.size())
        x2 = self.layer2(x1)
        # print('layer2 size:', x2.size())
        x3 = self.layer3(x2)
        # print('layer3 size:', x3.size())
        x4 = self.layer4(x3)
        # print('layer4 size:', x4.size())
        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}
        # return x4

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class SimpleUpsample(nn.Module):
    def __init__(self, in_ch, scale_upsample=2, ch_downsample=1, out_spatial=None):
        super(SimpleUpsample, self).__init__()
        if out_spatial is not None:
            self.simple_upsample = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // ch_downsample, 3, 1, 1),
                nn.BatchNorm2d(in_ch // ch_downsample),
                nn.ReLU(inplace=True),
                nn.Upsample(size=out_spatial, mode='bilinear', align_corners=False),
            )
        else:
            self.simple_upsample = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // ch_downsample, 3, 1, 1),
                nn.BatchNorm2d(in_ch // ch_downsample),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=scale_upsample, mode='bilinear', align_corners=False),
            )

    def forward(self, x):
        x = self.simple_upsample(x)
        return x


class ChannelReduction(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ChannelReduction, self).__init__()
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.channel_reduction(x)
        return x


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch, d1, d2, d3, d4):
        super(ASPP, self).__init__()
        self.aspp_d1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 3, padding=d1, dilation=d1),
            nn.BatchNorm2d(in_ch // 4),
            nn.ReLU(inplace=True)
        )
        self.aspp_d2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 3, padding=d2, dilation=d2),
            nn.BatchNorm2d(in_ch // 4),
            nn.ReLU(inplace=True)
        )
        self.aspp_d3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 3, padding=d3, dilation=d3),
            nn.BatchNorm2d(in_ch // 4),
            nn.ReLU(inplace=True)
        )
        if d4 == 'full':
            self.aspp_d4 = ASPPPooling(in_channels=in_ch, out_channels=in_ch // 4)
        else:
            self.aspp_d4 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 4, 3, padding=d4, dilation=d4),
                nn.BatchNorm2d(in_ch // 4),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        d1 = self.aspp_d1(x)
        d2 = self.aspp_d2(x)
        d3 = self.aspp_d3(x)
        d4 = self.aspp_d4(x)
        return torch.cat((d1, d2, d3, d4), dim=1)


class PlainFPN(nn.Module):
    def __init__(self, output_size=(240, 320), in_channels=3, training_mode='train_L2_loss', backbone='resnet101'):
        super(PlainFPN, self).__init__()
        self.output_size = output_size
        self.mode = training_mode
        self.resnet_pyramids = ResNetPyramids(in_channels=in_channels, pretrained=True, backbone=backbone)

        self.feature1_upsampling = nn.Sequential(
            ChannelReduction(in_ch=256, out_ch=128)
        )

        self.feature2_upsampling = nn.Sequential(
            ChannelReduction(in_ch=512, out_ch=256),
            SimpleUpsample(in_ch=256, scale_upsample=2, ch_downsample=2)
        )

        self.feature3_upsampling = nn.Sequential(
            ChannelReduction(in_ch=1024, out_ch=512),
            SimpleUpsample(in_ch=512, scale_upsample=2, ch_downsample=2),
            SimpleUpsample(in_ch=256, scale_upsample=2, ch_downsample=2)
        )

        self.feature4_upsampling = nn.Sequential(
            ChannelReduction(in_ch=2048, out_ch=1024),
            SimpleUpsample(in_ch=1024, out_spatial=(15, 20), ch_downsample=2),
            SimpleUpsample(in_ch=512, scale_upsample=2, ch_downsample=2),
            SimpleUpsample(in_ch=256, scale_upsample=2, ch_downsample=2),
        )

        self.feature_concat = nn.Sequential(
            # nn.Dropout2d(0.5),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.5),
            nn.Conv2d(64, 3, 1),
            nn.UpsamplingBilinear2d(size=(240, 320)),
        )


    def forward(self, x):
        features = self.resnet_pyramids(x)
        z1 = self.feature1_upsampling(features['x1'])
        z2 = self.feature2_upsampling(features['x2'])
        z3 = self.feature3_upsampling(features['x3'])
        z4 = self.feature4_upsampling(features['x4'])

        y = self.feature_concat(z1 + z2 + z3 + z4)

        return y


class ASPP_FPN(nn.Module):
    def __init__(self, output_size=(240, 320), in_channels=3, training_mode='train_L2_loss', backbone='resnext101'):
        super(ASPP_FPN, self).__init__()
        self.output_size = output_size
        self.mode = training_mode
        # self.resnet_pyramids = ResNetPyramids(in_channels=in_channels, pretrained=True, backbone=backbone)

        self.feature1_upsampling = nn.Sequential(
            ASPP(in_ch=256, d1=1, d2=8, d3=16, d4=24),
        )

        self.feature2_upsampling = nn.Sequential(
            ASPP(in_ch=512, d1=1, d2=4, d3=8, d4=12),
            SimpleUpsample(in_ch=512, scale_upsample=2, ch_downsample=2)
        )

        self.feature3_upsampling = nn.Sequential(
            ASPP(in_ch=1024, d1=1, d2=2, d3=4, d4=6),
            SimpleUpsample(in_ch=1024, scale_upsample=4, ch_downsample=4)
        )

        self.feature4_upsampling = nn.Sequential(
            ASPP(in_ch=2048, d1=1, d2=2, d3=3, d4='full'),
            SimpleUpsample(in_ch=2048, out_spatial=(60,80), ch_downsample=8)
        )

        self.feature_concat = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
            nn.UpsamplingBilinear2d(size=(240, 320)),
        )
        ### vit
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(320/ 16), int(256 / 16))
        self.encoder = ViT_seg(config_vit, img_size=[ 320,256], num_classes=config_vit.n_classes).cuda()
        self.encoder.load_from(weights=np.load(config_vit.pretrained_path))
        ### scale 3
        self.AttentionGraphCondKernel = AttentionGraphCondKernel([512, 1024, 2048], width=512)
        self.decoder = bts([64, 256, 512, 1024, 2048])

    def forward(self, x):
        # features = self.resnet_pyramids(x)
        skip_feat = self.encoder(x)
        skip_feat[5] = self.AttentionGraphCondKernel(skip_feat[2],skip_feat[3],skip_feat[4],skip_feat[5],3)
        # z1 = self.feature1_upsampling(skip_feat[2])
        # z2 = self.feature2_upsampling(skip_feat[3])
        # z3 = self.feature3_upsampling(skip_feat[4])
        # z4 = self.feature4_upsampling(skip_feat[5])
        # y = self.feature_concat(z1 + z2 + z3 + z4)
        # z1 = self.feature1_upsampling(features['x1'])
        # z2 = self.feature2_upsampling(features['x2'])
        # z3 = self.feature3_upsampling(features['x3'])
        # z4 = self.feature4_upsampling(features['x4'])
        # y = self.feature_concat(z1 + z2 + z3 + z4)
        # for i in range(len(skip_feat)):
        #     print(skip_feat[i].shape)
        y=self.decoder(skip_feat,518.8579 )
        return y
