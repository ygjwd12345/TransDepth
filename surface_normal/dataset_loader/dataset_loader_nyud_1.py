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
        self.trans = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                        ])

    def __getitem__(self, index):
        # Get image name from the pandas df
        image_id = self.idx[index]
        rgb_info = os.path.join(self.root, 'test', '%05d-rgb.jpg' % image_id)
        rgb_img = Image.open(rgb_info)
        rgb_img =rgb_img.resize((320, 240))
        rgb_tensor =self.trans(rgb_img)
        input_tensor = rgb_tensor
        if self.usage =='test':
            norm_info = os.path.join(self.root, 'nyu_normals_gt/test/', '%05d.png' % image_id)
        else:
            norm_info = os.path.join(self.root, 'nyu_normals_gt/train/', '%05d.png' % image_id)
        norm_img = sio.imread(norm_info)
        mask_info = os.path.join(self.root, 'masks/', '%04d.png' % (image_id-1))
        mask = sio.imread(mask_info)
        norm_img = cv2.resize(norm_img, (320, 240), interpolation=cv2.INTER_NEAREST)
        norm_img = norm_img / 255.0 * 2.0 - 1.0
        mask= cv2.resize(mask, (320, 240), interpolation=cv2.INTER_NEAREST)
        mask_img = torch.Tensor(mask.astype(np.uint8))
        normal_input_tensor = self.to_tensor(norm_img.astype('float32'))

        return {'image': input_tensor, 'Z': normal_input_tensor, 'mask': mask_img}

    def __len__(self):
        return self.data_len