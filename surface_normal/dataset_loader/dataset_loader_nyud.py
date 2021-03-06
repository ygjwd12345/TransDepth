import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import skimage.io as sio
import pickle
import numpy as np
import cv2
import os
from PIL import Image
class NYUD_Dataset(Dataset):
    def __init__(self, usage='test'):
        self.to_tensor = transforms.ToTensor()
        self.root = './datasets/nyu-normal'
        self.usage=usage
        if usage == 'test':
            self.idx = [int(line.rstrip('\n')) for line in open(os.path.join(self.root, 'testsplit.txt'))]
        else:
            self.idx = [int(line.rstrip('\n')) for line in open(os.path.join(self.root, 'trainsplit.txt'))]
        self.data_len = len(self.idx)
        self.img_trans = transforms.Compose([
                                         transforms.RandomHorizontalFlip(),
                                         # transforms.RandomCrop(size=(240, 320), padding=None),
                                         transforms.ToTensor(),
                                         # transforms.Normalize([0.485, 0.456, 0.406],
                                         #                     [0.229, 0.224, 0.225]),
                                        ])
        self.sn_trans = transforms.Compose([
                                         # transforms.RandomCrop(size=(240, 320), padding=None),
                                         transforms.ToTensor(),
                                         # transforms.Normalize([0.485, 0.456, 0.406],
                                         #                     [0.229, 0.224, 0.225]),
                                        ])

    def __getitem__(self, index):
        # Get image name from the pandas df
        image_id = self.idx[index]
        rgb_info = os.path.join(self.root, 'test', '%05d-rgb.jpg' % image_id)
        rgb_img = Image.open(rgb_info)

        if self.usage =='test':
            rgb_img = rgb_img.resize((320, 256))
            rgb_tensor = self.img_trans(rgb_img)
            input_tensor = rgb_tensor
            norm_info = os.path.join(self.root, 'nyu_normals_gt/test/', '%05d.png' % image_id)
            norm_img = Image.open(norm_info)
            norm_img = norm_img .resize((320, 256))
            norm_img = np.asarray(norm_img) / 255.0 * 2.0 - 1.0

            normal_input_tensor = self.sn_trans(norm_img)
            mask_info = os.path.join(self.root, 'masks/', '%04d.png' % (image_id - 1))
            mask = sio.imread(mask_info)
            mask= cv2.resize(mask, (320, 256), interpolation=cv2.INTER_NEAREST)
            mask_img = torch.Tensor(mask.astype(np.uint8))
        else:
            rgb_img = rgb_img.resize((320, 256))
            rgb_tensor = self.img_trans(rgb_img)
            input_tensor = rgb_tensor
            norm_info = os.path.join(self.root, 'nyu_normals_gt/train/', '%05d.png' % image_id)
            norm_img = Image.open(norm_info)
            norm_img = norm_img .resize((320, 256))
            norm_img = np.asarray(norm_img )/ 255.0 * 2.0 - 1.0

            normal_input_tensor = self.sn_trans(norm_img)
            mask_info = os.path.join(self.root, 'masks/', '%04d.png' % (image_id-1))
            mask = sio.imread(mask_info)
            mask= cv2.resize(mask, (320, 256), interpolation=cv2.INTER_NEAREST)
            mask_img = torch.Tensor(mask.astype(np.uint8))
        # print(input_tensor.shape)
        # print(normal_input_tensor.shape)
        return {'image': input_tensor, 'Z': normal_input_tensor, 'mask': mask_img}

    def __len__(self):
        return self.data_len