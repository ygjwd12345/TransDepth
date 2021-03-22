import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
'''
image  is  2011_09_26/2011_09_26_drive_0005_sync/image_03/data/0000000138.jpg
label  is data_depth_annotated/val/2011_09_26_drive_0005_sync/proj_depth/groundtruth/image_03/0000000138.png
'''
root="./dataset/kitti_dataset/"
num_l=0
num_i=0
with open('../train_test_inputs/eigen_train_files_with_gt.txt','r') as f:
    for line in f:
        p=list(line.strip('\n').split(' '))
        label_path=p[1]
        image_path=p[0]
        # print(label_path)
        if not os.path.exists(os.path.join('./dataset/kitti_dataset/train/',label_path)):
            num_l=num_l+1
        if not os.path.exists(os.path.join(root,image_path)):
            num_i=num_i+1


print('total number image no exist in train is %d'%(num_i))
print('total number label no exist train is  %d'%(num_l))
# i=1
with open('../train_test_inputs/eigen_test_files_with_gt.txt','r') as f:
    for line in f:
        # print(i)
        p=list(line.strip('\n').split(' '))
        label_path=p[1]
        image_path=p[0]
        # rgb_img = cv2.imread(os.path.join(root,image_path))
        # rgb_img = np.float32(rgb_img)
        # # processing
        # img = cv2.resize(rgb_img, (621, 188))
        # i=i+1
        # print(label_path)
        if not os.path.exists(os.path.join('./dataset/kitti_dataset/val/',label_path)):
            num_l=num_l+1
        if not os.path.exists(os.path.join(root,image_path)):
            num_i=num_i+1


print('total number image no exist test is %d'%(num_i))
print('total number label no exist test is %d'%(num_l))
# np.savetxt(r'eigen_test_files_n.txt',np.column_stack((path_i,path_l)),fmt='%s')