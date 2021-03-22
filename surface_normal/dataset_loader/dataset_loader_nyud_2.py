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
import random
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
        if self.usage =='test':
            norm_info = os.path.join(self.root, 'nyu_normals_gt/test/', '%05d.png' % image_id)
        else:
            norm_info = os.path.join(self.root, 'nyu_normals_gt/train/', '%05d.png' % image_id)
        mask_info = os.path.join(self.root, 'masks/', '%04d.png' % (image_id-1))
        rgb_img = Image.open(rgb_info)
        norm_img = Image.open(norm_info)
        mask = Image.open(mask_info)

        if self.usage =='train':
            random_angle = (random.random() - 0.5) * 2 * 2.5
            rgb_img = self.rotate_image(rgb_img, random_angle)
            norm_img = self.rotate_image(norm_img, random_angle)
            mask = self.rotate_image(mask, random_angle)
            rgb_img = np.asarray(rgb_img, dtype=np.float32)
            norm_img = np.asarray(norm_img, dtype=np.float32)
            rgb_img, norm_img = self.random_crop(rgb_img, norm_img, 416, 544)
            rgb_img, norm_img = self.train_preprocess(rgb_img, norm_img)

        # rgb_img =rgb_img.resize((320, 240))
            rgb_img=cv2.resize(rgb_img, (320, 240), interpolation=cv2.INTER_NEAREST)

            rgb_tensor =self.trans(rgb_img)

            norm_img=cv2.resize(norm_img, (320, 240), interpolation=cv2.INTER_NEAREST)
        else:
            rgb_img = np.asarray(rgb_img, dtype=np.float32)
            norm_img = np.asarray(norm_img, dtype=np.float32)
            rgb_img=cv2.resize(rgb_img, (320, 240), interpolation=cv2.INTER_NEAREST)
            rgb_tensor =self.trans(rgb_img)

            norm_img=cv2.resize(norm_img, (320, 240), interpolation=cv2.INTER_NEAREST)
        input_tensor = rgb_tensor

        # norm_img = norm_img.resize((320, 240))
        norm_img = norm_img / 255.0 * 2.0 - 1.0
        mask= mask.resize((320, 240))
        mask_img = torch.Tensor(np.uint8(mask))
        normal_input_tensor = self.to_tensor(norm_img.astype('float32'))

        return {'image': input_tensor, 'Z': normal_input_tensor, 'mask': mask_img}

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        img_=np.array(img)
        depth_=np.array(depth)
        assert img_.shape[0] >= height
        assert img_.shape[1] >= width
        assert img_.shape[0] == depth_.shape[0]
        assert img_.shape[1] == depth_.shape[1]
        x = random.randint(0, img_.shape[1] - width)
        y = random.randint(0, img_.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation

        brightness = random.uniform(0.75, 1.25)

        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    def __len__(self):
        return self.data_len