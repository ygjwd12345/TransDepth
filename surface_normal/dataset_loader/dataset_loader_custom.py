import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import skimage.io as sio
import pickle
import numpy as np
import cv2
import os
import fnmatch

class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        self.to_tensor = transforms.ToTensor()

        self.root = dataset_path
        self.idx = fnmatch.filter(os.listdir(self.root), '*.png')
        self.data_len = len(self.idx)

    def __getitem__(self, index):
        image_name = self.idx[index]
        rgb_info = os.path.join(self.root, image_name)
        rgb_img = sio.imread(rgb_info)
        rgb_img = cv2.resize(rgb_img, (320, 240), interpolation=cv2.INTER_CUBIC)
        rgb_tensor = self.to_tensor(rgb_img)
        input_tensor = np.zeros((3, rgb_img.shape[0], rgb_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = rgb_tensor
        return {'image': input_tensor}

    def __len__(self):
        return self.data_len