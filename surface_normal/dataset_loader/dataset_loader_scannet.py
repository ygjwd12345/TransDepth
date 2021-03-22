import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import skimage.io as sio
import pickle
import numpy as np
import cv2

class ScannetDataset(Dataset):
    def __init__(self, root='./datasets/scannet-frames/',
                       usage='test',
                       train_test_split ='./data/scannet_standard_train_test_val_split.pkl',
                       frameGap=200):

        self.root = root
        self.to_tensor = transforms.ToTensor()

        self.train_test_split = train_test_split
        self.data_info = pickle.load(open(train_test_split, 'rb'))[usage]

        if usage == 'train':
            self.idx = [i for i in range(0, len(self.data_info[0]), 1)]
        elif usage == 'test':
            self.idx = [i for i in range(0, len(self.data_info[0]), frameGap)]
        elif usage == 'val':
            self.idx = [i for i in range(0, len(self.data_info[0]), frameGap)]

        self.data_len = len(self.idx)
        self.root = root

    def __getitem__(self, index):
        if self.train_test_split == './data/framenet_train_test_split.pkl':  # get proper path from framenet pkl
            color_info = self.data_info[0][self.idx[index]]
            orient_info = self.data_info[1][self.idx[index]][:-10] + 'normal.png'
            mask_info = self.data_info[2][self.idx[index]]

            color_info = self.root + '/' + color_info[27:]
            orient_info = self.root + '/' + orient_info[27:]
            mask_info = self.root + '/' + mask_info[27:]

        else:
            color_info = self.data_info[0][self.idx[index]]
            orient_info = self.data_info[1][self.idx[index]][:-10] + 'normal.png'
            mask_info = self.data_info[2][self.idx[index]]

            color_info = self.root + '/' + color_info
            orient_info = self.root + '/' + orient_info
            mask_info = self.root + '/' + mask_info

        # Image resize and load
        color_img = cv2.resize(sio.imread(color_info), (320, 256), interpolation=cv2.INTER_CUBIC)
        orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 256), interpolation=cv2.INTER_NEAREST)
        orient_img = cv2.resize(sio.imread(orient_info), (320, 256), interpolation=cv2.INTER_NEAREST)

        # To tensor
        color_tensor = self.to_tensor(color_img)
        orient_mask_tensor = torch.Tensor(orient_mask_tensor/255.0)
        Z = -self.to_tensor(orient_img) + 0.5

        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor

        return {'image': input_tensor, 'mask': orient_mask_tensor, 'Z': Z}

    def __len__(self):
        return self.data_len