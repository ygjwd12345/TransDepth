import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import skimage.io as sio
import pickle
import numpy as np
import cv2
import os

class KinectAzureDataset(Dataset):
    def __init__(self, root='./datasets/KinectAzure',
                       usage='test_full',
                       train_test_split = './data/kinect_azure_test_datasets.pkl'):
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()
        self.train_test_plit = train_test_split
        self.data_info = pickle.load(open(train_test_split, 'rb'))[usage]

        # for color_path in self.data_info[2]:
        #     folder_path = color_path[:-16]
        #     print(color_path)
        #     if not os.path.exists(os.path.join('./datasets/KinectAzure', folder_path)):
        #         os.makedirs(os.path.join('./datasets/KinectAzure', folder_path))
        #     cmd = 'cp %s %s' % (os.path.join(self.root, color_path), os.path.join('./datasets/KinectAzure', color_path))
        #     os.system(cmd)

        self.idx = [i for i in range(0, len(self.data_info[0]), 1)]
        self.data_len = len(self.idx)
        self.root = root

    def __getitem__(self, index):
        color_info = os.path.join(self.root, self.data_info[0][self.idx[index]])
        orient_info = os.path.join(self.root, self.data_info[1][self.idx[index]])
        mask_info = os.path.join(self.root, self.data_info[2][self.idx[index]])

        # Image resize and load
        color_img = cv2.resize(sio.imread(color_info), (320, 240), interpolation=cv2.INTER_CUBIC)
        orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
        orient_img = cv2.resize(sio.imread(orient_info), (320, 240), interpolation=cv2.INTER_NEAREST)

        # To tensor
        color_tensor = self.to_tensor(color_img)
        orient_mask_tensor = torch.Tensor(orient_mask_tensor/255.0)
        Z = -self.to_tensor(orient_img) + 0.5

        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor

        return {'image': input_tensor, 'mask': orient_mask_tensor, 'Z': Z}

    def __len__(self):
        return self.data_len