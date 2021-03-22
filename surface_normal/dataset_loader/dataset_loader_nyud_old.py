import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import skimage.io as sio
import pickle
import numpy as np
import cv2
import os


class NYUD_Dataset(Dataset):
    def __init__(self, usage='test'):
        self.to_tensor = transforms.ToTensor()
        self.root = './datasets/nyu-normal'
        self.idx = [int(line.rstrip('\n')) for line in open(os.path.join(self.root, 'testsplit.txt'))]
        self.data_len = len(self.idx)

    def __getitem__(self, index):
        # Get image name from the pandas df
        image_id = self.idx[index]

        rgb_info = os.path.join(self.root, 'test', '%05d-rgb.jpg' % image_id)
        rgb_img = sio.imread(rgb_info)
        rgb_img = cv2.resize(rgb_img, (320, 256), interpolation=cv2.INTER_CUBIC)
        rgb_tensor = self.to_tensor(rgb_img)
        input_tensor = np.zeros((3, rgb_img.shape[0], rgb_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = rgb_tensor

        norm_info = os.path.join(self.root, 'nyu_normals_gt/test/', '%05d.png' % image_id)
        norm_img = sio.imread(norm_info)

        origin_img = sio.imread(os.path.join(self.root, 'test', '%05d-norm.png' % image_id))
        origin_img = (origin_img / 255.0 - 0.5) * 2
        mask = np.linalg.norm(origin_img, axis=2) > 0.5
        temp = 255 - norm_img
        norm_img[:, :, 0] = temp[:, :, 2]
        norm_img[:, :, 1] = temp[:, :, 1]
        norm_img[:, :, 2] = temp[:, :, 0]
        norm_img[:, :, 0] = norm_img[:, :, 0] * mask + (1 - mask) * 128
        norm_img[:, :, 1] = norm_img[:, :, 1] * mask + (1 - mask) * 128
        norm_img[:, :, 2] = norm_img[:, :, 2] * mask + (1 - mask) * 128
        norm_img = cv2.resize(norm_img, (320, 256), interpolation=cv2.INTER_NEAREST)
        norm_img = norm_img / 255.0 * 2.0 - 1.0

        mask = np.linalg.norm(norm_img, axis=2) > 0.5
        mask_img = mask & (norm_img[:, :, 2] > 0)

        mask_img = torch.Tensor(mask_img.astype(np.uint8))
        normal_input_tensor = self.to_tensor(norm_img.astype('float32'))

        return {'image': input_tensor, 'Z': normal_input_tensor, 'mask': mask_img}

    def __len__(self):
        return self.data_len