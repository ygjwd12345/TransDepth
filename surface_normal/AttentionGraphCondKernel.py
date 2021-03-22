##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Guanglei Yang
## Department of Information Engineering and Computer Science, University of Trento
## Email: guanglei.yang@studenti.unitn.it or yangguanglei.phd@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Probabilistic graph attention model with conditional Kernels"""
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, BCELoss, CrossEntropyLoss, Sigmoid
from torch.nn.functional import unfold
# from encoding.nn import BatchNorm2d
import time

from torch.autograd import Variable

torch_ver = torch.__version__[:3]

__all__ = ['AttentionGraphCondKernel','AttentionGraphCondKernel_2','AttentionGraphCondKernel_3','AttentionGraphCondKernel_4']

class AttentionGatedMSG(nn.Module):
    def __init__(self, width=512, norm_layer=None, up_kwargs=None):
        super(AttentionGatedMSG, self).__init__()
        self.up_kwargs = up_kwargs
        self.ks = 3

        # kernel prediction based on the combined two different scales of features
        self.kernel_prediction_1 = nn.Conv2d(2 * width, 9, kernel_size=3, dilation=1, bias=True,
                                             padding=1)  # 4 groups of kernels and each kernel with 9 kernel values
        self.kernel_prediction_2 = nn.Conv2d(2 * width, 9, kernel_size=3, dilation=4, bias=True,
                                             padding=4)  # 4 groups of kernels and each kernel with 9 kernel values
        self.kernel_prediction_3 = nn.Conv2d(2 * width, 9, kernel_size=3, dilation=8, bias=True,
                                             padding=8)  # 4 groups of kernels and each kernel with 9 kernel values

        # kernel prediction for attention
        self.kernel_se_1 = nn.Conv2d(width, 9, kernel_size=3, dilation=1, bias=True,
                                     padding=1)  # one channel attention map
        self.kernel_sr_1 = nn.Conv2d(width, 9, kernel_size=3, dilation=1, bias=True,
                                     padding=1)  # one channel attention map

        self.kernel_se_2 = nn.Conv2d(width, 9, kernel_size=3, dilation=4, bias=True, padding=4)
        self.kernel_sr_2 = nn.Conv2d(width, 9, kernel_size=3, dilation=4, bias=True, padding=4)

        self.kernel_se_3 = nn.Conv2d(width, 9, kernel_size=3, dilation=8, bias=True, padding=8)
        self.kernel_sr_3 = nn.Conv2d(width, 9, kernel_size=3, dilation=8, bias=True, padding=8)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.combination_msgs = nn.Sequential(nn.Conv2d(3 * width, width, kernel_size=1),
                                              nn.ReLU(inplace=True))

    def struc_att(self, att,epoch=100, rank=3):
        bs, W, h, w = att.size()
        output=torch.zeros(bs, W, h, w).cuda()
        if rank == 0:
            output = att
        else:
            for i in range(rank):
                ch_weights = torch.randn(bs, W).cuda()
                ch_ag_weights = self.softmax(ch_weights).unsqueeze(-1).unsqueeze(-1)
                sp_weights = (ch_ag_weights * att).sum(1, True)
                sp_ag_weights = self.sigmoid(sp_weights)
                ch_weights = (sp_ag_weights * att).sum(-1).sum(-1)
                # ch_ag_weights = self.softmax(ch_weights).unsqueeze(-1).unsqueeze(-1)
                output=sp_ag_weights * ch_ag_weights * att+output
        return output


    def forward(self, sr,se,epoch=100,rank=3):
        # input[0] is last scale feature map
        inputs_se = se  # the feature map sending message
        inputs_sr = sr  # the feature map receiving message
        input_concat = torch.cat((inputs_se, inputs_sr), 1)
        # weight prediction for different dilation rates
        dy_weights_1 = self.kernel_prediction_1(input_concat)
        dy_weights_1_ = dy_weights_1.view(dy_weights_1.size(0), 1, self.ks ** 2, dy_weights_1.size(2),
                                          dy_weights_1.size(3))
        dy_weights_2 = self.kernel_prediction_2(input_concat)
        dy_weights_2_ = dy_weights_2.view(dy_weights_2.size(0), 1, self.ks ** 2, dy_weights_2.size(2),
                                          dy_weights_2.size(3))
        dy_weights_3 = self.kernel_prediction_3(input_concat)
        dy_weights_3_ = dy_weights_3.view(dy_weights_3.size(0), 1, self.ks ** 2, dy_weights_3.size(2),
                                          dy_weights_3.size(3))

        dy_kernel_se_1 = self.kernel_se_1(inputs_se).unsqueeze(1)
        dy_kernel_sr_1 = self.kernel_sr_1(inputs_sr).unsqueeze(1)
        dy_kernel_se_2 = self.kernel_se_2(inputs_se).unsqueeze(1)
        dy_kernel_sr_2 = self.kernel_sr_2(inputs_sr).unsqueeze(1)
        dy_kernel_se_3 = self.kernel_se_3(inputs_se).unsqueeze(1)
        dy_kernel_sr_3 = self.kernel_sr_3(inputs_sr).unsqueeze(1)
        # new add 2020 2 12
        # unfold inputs
        f_se = inputs_se.shape  ##feature maps have the same shape
        f_sr = inputs_sr.shape
        inputs_se_1 = unfold(inputs_se, kernel_size=3, dilation=1, padding=1).view(f_se[0], f_se[1], self.ks ** 2,
                                                                                   f_se[2], f_se[3])
        inputs_sr_1 = unfold(inputs_sr, kernel_size=3, dilation=1, padding=1).view(f_sr[0], f_sr[1], self.ks ** 2,
                                                                                   f_sr[2], f_sr[3])
        inputs_se_2 = unfold(inputs_se, kernel_size=3, dilation=4, padding=4).view(f_se[0], f_se[1], self.ks ** 2,
                                                                                   f_se[2], f_se[3])
        inputs_sr_2 = unfold(inputs_sr, kernel_size=3, dilation=4, padding=4).view(f_sr[0], f_sr[1], self.ks ** 2,
                                                                                   f_sr[2], f_sr[3])
        inputs_se_3 = unfold(inputs_se, kernel_size=3, dilation=8, padding=8).view(f_se[0], f_se[1], self.ks ** 2,
                                                                                   f_se[2], f_se[3])
        inputs_sr_3 = unfold(inputs_sr, kernel_size=3, dilation=8, padding=8).view(f_sr[0], f_sr[1], self.ks ** 2,
                                                                                   f_sr[2], f_sr[3])

        # attention prediction

        attention_map_1 = inputs_sr * ((dy_weights_1_ * inputs_se_1).sum(2)) + (dy_kernel_se_1 * inputs_se_1).sum(2) + (
                    dy_kernel_sr_1 * inputs_sr_1).sum(2)

        attention_map_2 = inputs_sr * ((dy_weights_2_ * inputs_se_2).sum(2)) + (dy_kernel_se_2 * inputs_se_2).sum(2) + (
                    dy_kernel_sr_2 * inputs_sr_2).sum(2)

        attention_map_3 = inputs_sr * ((dy_weights_3_ * inputs_se_3).sum(2)) + (dy_kernel_se_3 * inputs_se_3).sum(2) + (
                    dy_kernel_sr_3 * inputs_sr_3).sum(2)
        # sturcure attention
        attention_map_1 = self.struc_att(attention_map_1,epoch,rank=rank)
        attention_map_2 = self.struc_att(attention_map_2,epoch,rank=rank)
        attention_map_3 = self.struc_att(attention_map_3,epoch,rank=rank)

        # attention gated message calcultation with different dilation rate
        message_1 = attention_map_1 * ((dy_weights_1_ * inputs_se_1).sum(2))
        message_2 = attention_map_2 * ((dy_weights_2_ * inputs_se_2).sum(2))
        message_3 = attention_map_3 * ((dy_weights_3_ * inputs_se_3).sum(2))

        # final message
        message_f = self.combination_msgs(torch.cat([message_1, message_2, message_3], 1))
        return message_f , attention_map_1


class AttentionGraphCondKernel(nn.Module):
    def __init__(self, ms_featmaps, width=512, norm_layer=nn.BatchNorm2d, up_kwargs=None):
        super(AttentionGraphCondKernel, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        # scale 3
        # self.attention_MSG_43 = AttentionGatedMSG(width=512)
        # self.attention_MSG_53 = AttentionGatedMSG(width=512)
        # self.combination_msgs_3 = nn.Sequential(nn.Conv2d(3*width, width, kernel_size=1),
        #                                      norm_layer(width),
        #                                      nn.ReLU(inplace=True))
        # scale 4
        # self.attention_MSG_34 = AttentionGatedMSG(width=512)
        # self.attention_MSG_54 = AttentionGatedMSG(width=512)
        # self.combination_msgs_4 = nn.Sequential(nn.Conv2d(3*width, width, kernel_size=1),
        #                                      norm_layer(width),
        #                                      nn.ReLU(inplace=True))
        # scale 5
        self.attention_MSG_45 = AttentionGatedMSG(width=width)
        self.attention_MSG_35 = AttentionGatedMSG(width=width)
        self.attention_MSG_55 = AttentionGatedMSG(width=width)
        self.combination_msgs_51 = nn.Sequential(nn.Conv2d(4 * width, 4 * width, kernel_size=3,stride=2,padding=1),
                                                norm_layer(4 * width),
                                                nn.ReLU(inplace=True))
        self.combination_msgs_52 = nn.Sequential(nn.Conv2d(4 * width, 4 * width, kernel_size=3,stride=2,padding=1),
                                                norm_layer(4 * width),
                                                nn.ReLU(inplace=True))

    def forward(self, c1, c2, c3, c4,epoch=100,rank=3):
        feats = [self.conv5(c4), self.conv4(c3), self.conv3(c2)]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w))
        feats[-3] = F.upsample(feats[-3], (h, w))
        # import scipy.io as sio
        #
        # sio.savemat('feat5.mat', {'feat': feats[-3].cpu().data.numpy()})
        # sio.savemat('feat4.mat', {'feat': feats[-2].cpu().detach().numpy()})
        # sio.savemat('feat3.mat', {'feat': feats[-1].cpu().data.numpy()})
        # import pdb
        # pdb.set_trace()

        # message passing from scale 4 & 5 to 3:
        # att_MSG_43 = self.attention_MSG_43(feats[2], feats[1])
        # att_MSG_53 = self.attention_MSG_53(feats[2], feats[0])
        # output_3 = self.combination_msgs_3(torch.cat([att_MSG_43, att_MSG_53, feats[2]], 1))

        # message passing from scale 3 & 5 to 4
        # att_MSG_34 = self.attention_MSG_43(feats[1], output_3)
        # att_MSG_54 = self.attention_MSG_53(feats[1], feats[0])
        # output_4 = self.combination_msgs_4(torch.cat([att_MSG_34, att_MSG_54, feats[1]], 1))

        # message passing from scale 3 & 4 to 5 mutlti scale
        att_MSG_35, attention_map_35 = self.attention_MSG_35(feats[0], feats[2],epoch,rank=rank)
        att_MSG_45, attention_map_45 = self.attention_MSG_45(feats[0], feats[1],epoch,rank=rank)
        att_MSG_55, attention_map_55 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # # message passing from scale 3 & 4 to 5 single scale
        # att_MSG_35, attention_map_35 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # att_MSG_45, attention_map_45 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # att_MSG_55, attention_map_55 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)

        output_final_ = self.combination_msgs_51(torch.cat([att_MSG_35, att_MSG_45, att_MSG_55, feats[0]], 1))
        output_final = self.combination_msgs_52(output_final_ )
        # output_final = torch.cat([output_3, output_4, output_5], 1)

        return output_final
class AttentionGraphCondKernel_2(nn.Module):
    def __init__(self, ms_featmaps, width=1024, norm_layer=nn.BatchNorm2d, up_kwargs=None):
        super(AttentionGraphCondKernel_2, self).__init__()
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        # layer 5
        self.attention_MSG_45 = AttentionGatedMSG(width=width)
        self.attention_MSG_55 = AttentionGatedMSG(width=width)
        self.combination_msgs_51 = nn.Sequential(nn.Conv2d(3 * width, 2 * width, kernel_size=3,stride=2,padding=1),
                                                norm_layer(2 * width),
                                                nn.ReLU(inplace=True))

    def forward(self, c1, c2, c3, c4,epoch=100,rank=3):
        feats = [self.conv5(c4), self.conv4(c3)]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w))

        # message passing from scale 3 & 4 to 5 mutlti scale
        att_MSG_45, attention_map_45 = self.attention_MSG_45(feats[0], feats[1],epoch,rank=rank)
        att_MSG_55, attention_map_55 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # # message passing from scale 3 & 4 to 5 single scale
        # att_MSG_35, attention_map_35 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # att_MSG_45, attention_map_45 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # att_MSG_55, attention_map_55 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)

        output_final = self.combination_msgs_51(torch.cat([att_MSG_45, att_MSG_55, feats[0]], 1))

        return output_final

class AttentionGraphCondKernel_3(nn.Module):
    def __init__(self, ms_featmaps, width=512, norm_layer=nn.BatchNorm2d, up_kwargs=None):
        super(AttentionGraphCondKernel_3, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        # layer 5
        self.attention_MSG_45 = AttentionGatedMSG(width=width)
        self.attention_MSG_35 = AttentionGatedMSG(width=width)
        self.attention_MSG_55 = AttentionGatedMSG(width=width)
        self.combination_msgs_51 = nn.Sequential(nn.Conv2d(4 * width, 4 * width, kernel_size=3,stride=2,padding=1),
                                                norm_layer(4 * width),
                                                nn.ReLU(inplace=True))
        self.combination_msgs_52 = nn.Sequential(nn.Conv2d(4 * width, 4 * width, kernel_size=3,stride=2,padding=1),
                                                norm_layer(4 * width),
                                                nn.ReLU(inplace=True))

    def forward(self, c1, c2, c3, c4,epoch=100,rank=3):
        feats = [self.conv5(c4), self.conv4(c3), self.conv3(c2)]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w))
        feats[-3] = F.upsample(feats[-3], (h, w))

        # message passing from scale 3 & 4 to 5 mutlti scale
        att_MSG_35, attention_map_35 = self.attention_MSG_35(feats[0], feats[2],epoch,rank=rank)
        att_MSG_45, attention_map_45 = self.attention_MSG_45(feats[0], feats[1],epoch,rank=rank)
        att_MSG_55, attention_map_55 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # # message passing from scale 3 & 4 to 5 single scale
        # att_MSG_35, attention_map_35 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # att_MSG_45, attention_map_45 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # att_MSG_55, attention_map_55 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)

        output_final_ = self.combination_msgs_51(torch.cat([att_MSG_35, att_MSG_45, att_MSG_55, feats[0]], 1))
        output_final = self.combination_msgs_52(output_final_ )

        return output_final

class AttentionGraphCondKernel_4(nn.Module):
    def __init__(self, ms_featmaps, width=256, norm_layer=nn.BatchNorm2d, up_kwargs=None):
        super(AttentionGraphCondKernel_4, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ms_featmaps[-4], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        # layer 5
        self.attention_MSG_25 = AttentionGatedMSG(width=width)
        self.attention_MSG_45 = AttentionGatedMSG(width=width)
        self.attention_MSG_35 = AttentionGatedMSG(width=width)
        self.attention_MSG_55 = AttentionGatedMSG(width=width)
        self.combination_msgs_51 = nn.Sequential(nn.Conv2d(5 * width, 6 * width, kernel_size=3,stride=2,padding=1),
                                                norm_layer(6 * width),
                                                nn.ReLU(inplace=True))
        self.combination_msgs_52 = nn.Sequential(nn.Conv2d(6 * width, 8 * width, kernel_size=3,stride=2,padding=1),
                                                norm_layer(8 * width),
                                                nn.ReLU(inplace=True))
        self.combination_msgs_53 = nn.Sequential(nn.Conv2d(8 * width, 8 * width, kernel_size=3,stride=2,padding=1),
                                                norm_layer(8 * width),
                                                nn.ReLU(inplace=True))

    def forward(self, c1, c2, c3, c4,epoch=100,rank=3):
        feats = [self.conv5(c4), self.conv4(c3), self.conv3(c2),self.conv4(c1)]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w))
        feats[-3] = F.upsample(feats[-3], (h, w))
        feats[-4] = F.upsample(feats[-4], (h, w))

        # message passing from scale 2 & 3 & 4 to 5 mutlti scale
        att_MSG_25, attention_map_25 = self.attention_MSG_35(feats[0], feats[3],epoch,rank=rank)
        att_MSG_35, attention_map_35 = self.attention_MSG_35(feats[0], feats[2],epoch,rank=rank)
        att_MSG_45, attention_map_45 = self.attention_MSG_45(feats[0], feats[1],epoch,rank=rank)
        att_MSG_55, attention_map_55 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # # message passing from scale 3 & 4 to 5 single scale
        # att_MSG_35, attention_map_35 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # att_MSG_45, attention_map_45 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)
        # att_MSG_55, attention_map_55 = self.attention_MSG_55(feats[0], feats[0],epoch,rank=rank)

        output_final_ = self.combination_msgs_51(torch.cat([att_MSG_25,att_MSG_35, att_MSG_45, att_MSG_55, feats[0]], 1))
        output_final_ = self.combination_msgs_52(output_final_ )
        output_final = self.combination_msgs_53(output_final_ )

        return output_final