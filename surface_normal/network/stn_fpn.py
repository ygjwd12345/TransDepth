import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models
import collections
import math
from warping_2dof_alignment import Warping2DOFAlignment


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
    def __init__(self, in_channels=3, pretrained=True, freeze=True):
        super(ResNetPyramids, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(101)](pretrained=pretrained)

        # for m in pretrained_model.modules():
        #     print(m)
        # exit()
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

    def forward(self, x_input):
        # print(pretrained_model._modules)

        x = self.conv1(x_input)
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
        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}
        # return x4

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ModifiedFPN(nn.Module):
    def __init__(self, output_size=(240, 320), in_channels=3, training_mode='train_L2_loss', use_mask=False):
        super(ModifiedFPN, self).__init__()
        self.output_size = output_size
        self.mode = training_mode
        self.resnet_pyramids = ResNetPyramids(in_channels=in_channels, pretrained=True)
        self.use_mask = use_mask
        self.feature1_upsamping = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature2_upsamping = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature3_upsamping = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature4_upsamping = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(15, 20)),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        if 'mix_loss' in training_mode:
            self.feature_concat = nn.Sequential(
                # nn.Conv2d(256, 128, 3, 1, 1),
                # nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.5),
                nn.Conv2d(64, 3, 1),
            )
        else:
            self.feature_concat = nn.Sequential(
                # nn.Dropout2d(0.5),
                # nn.Conv2d(256, 128, 1),
                # nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.5),
                nn.Conv2d(64, 3, 1, 1, 1),
                nn.UpsamplingBilinear2d(size=(240, 320)),
            )

        # weights_init(self.feature_concat, type='xavier')

    def forward(self, x):
        features = self.resnet_pyramids(x)
        if self.use_mask:
            feature_mask = x[:, 0:1] + x[:, 1:2] + x[:, 2:3] > 1e-2
            feature_mask = feature_mask.float().detach()
            feature1_mask = nn.functional.interpolate(feature_mask, size=(60, 80), mode='nearest')
            feature2_mask = nn.functional.interpolate(feature_mask, size=(30, 40), mode='nearest')
            feature3_mask = nn.functional.interpolate(feature_mask, size=(15, 20), mode='nearest')
            feature4_mask = nn.functional.interpolate(feature_mask, size=(8, 10), mode='nearest')

            z1 = self.feature1_upsamping(features['x1'] * feature1_mask)
            z2 = self.feature2_upsamping(features['x2'] * feature2_mask)
            z3 = self.feature3_upsamping(features['x3'] * feature3_mask)
            z4 = self.feature4_upsamping(features['x4'] * feature4_mask)
            y = self.feature_concat((z1 + z2 + z3 + z4) * feature1_mask)
            return y
        else:
            z1 = self.feature1_upsamping(features['x1'])
            z2 = self.feature2_upsamping(features['x2'])
            z3 = self.feature3_upsamping(features['x3'])
            z4 = self.feature4_upsamping(features['x4'])
            y = self.feature_concat(z1 + z2 + z3 + z4)
            return y


class FPNWarpInputMultiDirections(nn.Module):
    def __init__(self, output_size=(240, 320), in_channels=3,
                    training_mode='train_L2_loss',
                    fc_img=np.array([0.5 * 577.87061, 0.5 * 580.25851]),
                    cc_img=np.array([0.5 * 319.87654, 0.5 * 239.87603]),
                    use_mask=False):
        super(FPNWarpInputMultiDirections, self).__init__()
        self.output_size = output_size
        self.mode = training_mode
        self.use_mask = use_mask

        fc = fc_img
        cc = cc_img
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=fc[0], fy=fc[1], cx=cc[0], cy=cc[1])

        self.resnet_pyramids = ResNetPyramids(in_channels=in_channels, pretrained=True)
        self.feature1_upsamping = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature2_upsamping = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature3_upsamping = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.feature4_upsamping = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(15, 20)),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.feature_concat = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
            nn.UpsamplingBilinear2d(size=(240, 320)),
        )

    def forward(self, x):
        features = self.resnet_pyramids(x)
        if self.use_mask:
            feature_mask = x1[:, 0:1] + x1[:, 1:2] + x1[:, 2:3] > 1e-2
            feature_mask = feature_mask.float().detach()
            feature1_mask = nn.functional.interpolate(feature_mask, size=(60, 80), mode='nearest')
            feature2_mask = nn.functional.interpolate(feature_mask, size=(30, 40), mode='nearest')
            feature3_mask = nn.functional.interpolate(feature_mask, size=(15, 20), mode='nearest')
            feature4_mask = nn.functional.interpolate(feature_mask, size=(8, 10), mode='nearest')

            z1 = self.feature1_upsamping(features['x1'] * feature1_mask)
            z2 = self.feature2_upsamping(features['x2'] * feature2_mask)
            z3 = self.feature3_upsamping(features['x3'] * feature3_mask)
            z4 = self.feature4_upsamping(features['x4'] * feature4_mask)
            y = self.feature_concat((z1 + z2 + z3 + z4) * feature1_mask)
        else:
            z1 = self.feature1_upsamping(features['x1'])
            z2 = self.feature2_upsamping(features['x2'])
            z3 = self.feature3_upsamping(features['x3'])
            z4 = self.feature4_upsamping(features['x4'])
            y = self.feature_concat(z1 + z2 + z3 + z4)
        return y


class WarpingParametersPrediction(nn.Module):
    def __init__(self, in_channels=3):
        super(WarpingParametersPrediction, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(18)](pretrained=True)

        self.channel = in_channels
        # for m in pretrained_model.modules():
        #     print(m)
        # exit()

        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        self.avg_pool = pretrained_model._modules['avgpool']
        self.warping_params_output = nn.Sequential(nn.Linear(512, 128),
                                                   nn.ReLU(True),
                                                   nn.Dropout(),
                                                   nn.Linear(128, 6))

        # clear memory
        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.avg_pool(x4)
        z = self.warping_params_output(torch.flatten(y, 1))
        return z


class SpatialWarpingFPN(nn.Module):
    def __init__(self, fc_img=np.array([0.5 * 577.87061, 0.5 * 580.25851])):
        super(SpatialWarpingFPN, self).__init__()
        self.generalized_view_cnn = ModifiedFPN(in_channels=3, training_mode='train_robust_acos_loss', use_mask=False)
        self.warp_params_cnn = WarpingParametersPrediction()
        self.canonical_view_cnn = FPNWarpInputMultiDirections(in_channels=3, training_mode='train_robust_acos_loss',
                                                              use_mask=False, fc_img = fc_img)
        fc = fc_img
        cc = np.array([160, 120])
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=fc[0], fy=fc[1], cx=cc[0], cy=cc[1])

        # self.generalized_view_cnn.load_state_dict(torch.load('./checkpoints/FPN_generalized_view.ckpt'))
        # self.warp_params_cnn.load_state_dict(torch.load('./checkpoints/FPN_warping_params.ckpt'))
        # self.canonical_view_cnn.load_state_dict(torch.load('./checkpoints/FPN_canonical_view.ckpt'))

    def forward(self, x):
        # Step 1: Generalized view
        n_pred_g = self.generalized_view_cnn(x)
        # NOTE: Do I want this to be here?
        n_pred_g = torch.nn.functional.normalize(n_pred_g, dim=1)

        # Step 2: Construct warping parameters
        v = self.warp_params_cnn(n_pred_g)
        I_g = torch.nn.functional.normalize(v[:, 0:3], dim=1, eps=1e-6)
        I_a = torch.nn.functional.normalize(v[:, 3:6], dim=1, eps=1e-6)

        # Step 3: Construct image sampler forward and inverse
        R_inv, img_sampler, inv_img_sampler = self.warp_2dof_alignment.image_sampler_forward_inverse(I_g, I_a)

        # Step 4: Warp input to be canonical
        w_x = torch.nn.functional.grid_sample(x, img_sampler, padding_mode='zeros', mode='bilinear')

        # Step 5: Canonical view
        w_y = self.canonical_view_cnn(w_x)

        # Step 6: Inverse warp the output to be pixel wise with input
        y = torch.nn.functional.grid_sample(w_y, inv_img_sampler, padding_mode='zeros', mode='bilinear')
        y = y.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        n_pred_c = (R_inv.bmm(y)).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        # Step 7: Join the information between generalized and canonical

        return {'n1': n_pred_g, 'I_g': v[:, 0:3], 'I_a': v[:, 3:6], 'n2': n_pred_c,
                'W_I': w_x, 'W_O': w_y}


if __name__ == '__main__':
    warp_param_net = WarpingParametersPrediction()
    warp_param_net.cuda()
