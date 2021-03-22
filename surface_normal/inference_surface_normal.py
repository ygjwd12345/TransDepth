import torch
import numpy as np
import skimage.io as sio
import argparse
from torch.utils.data import DataLoader
from network import dorn_architecture, fpn_architecture, stn_fpn
from dataset_loader.dataset_loader_custom import CustomDataset
from dataset_loader.dataset_loader_scannet import ScannetDataset
from dataset_loader.dataset_loader_nyud import NYUD_Dataset
from dataset_loader.dataset_loader_kinectazure import KinectAzureDataset
import os
import time
import cv2


def parsing_configurations():
    parser = argparse.ArgumentParser(description='Inference for surface normal estimation')
    parser.add_argument('--log_folder', type=str, default='')
    parser.add_argument('--operation', type=str, default='inference')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--test_dataset', type=str, default='custom folder')
    parser.add_argument('--net_architecture', type=str, default='d_fpn_resnext101')
    args = parser.parse_args()

    config = {'LOG_FOLDER': args.log_folder,
              'CKPT_PATH': args.checkpoint_path,
              'OPERATION': args.operation,
              'BATCH_SIZE': args.batch_size,
              'TEST_DATASET': args.test_dataset,
              'ARCHITECTURE': args.net_architecture}
    return config


def log(str, fp=None):
    if fp is not None:
        fp.write('%s\n' % (str))
        fp.flush()
    print(str)


def saving_rgb_tensor_to_file(rgb_tensor, path):
    output_rgb_img = np.uint8((rgb_tensor.permute(1, 2, 0).detach().cpu()) * 255)
    sio.imsave(path, output_rgb_img)
def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min)/(max-min)
def saving_normal_tensor_to_file(normal_tensor, path):
    normal_tensor = torch.nn.functional.normalize(normal_tensor, dim=0)
    # normal_tensor = minmaxscaler(normal_tensor)

    output_normal_img = np.uint8((normal_tensor.permute(1, 2, 0).detach().cpu() + 1) * 127.5)

    # output_normal_img = np.uint8((normal_tensor.permute(1, 2, 0).detach().cpu() ) * 255)

    # temp=255 -output_normal_img
    sio.imsave(path, output_normal_img)

def saving_gt_tensor_to_file(normal_tensor, path):
    normal_tensor = torch.nn.functional.normalize(normal_tensor, dim=0)
    output_normal_img = np.uint8((normal_tensor.permute(1, 2, 0).detach().cpu() + 1) * 127.5)

    # output_normal_img = np.uint8((normal_tensor.permute(1, 2, 0).detach().cpu() ) * 255)

    # temp=255 -output_normal_img
    sio.imsave(path, output_normal_img)
def Normalize(dir_x):
    dir_x_l = torch.sqrt(torch.sum(dir_x ** 2,dim=1) + 1e-6).view(dir_x.shape[0], 1, dir_x.shape[2], dir_x.shape[3])
    dir_x_l = torch.cat([dir_x_l, dir_x_l, dir_x_l], dim=1)
    return dir_x / dir_x_l


def create_dataset_loader(config):
    if config['TEST_DATASET'] == 'kinect_azure_full':
        test_dataset = KinectAzureDataset(usage='test_full')
    elif config['TEST_DATASET'] == 'kinect_azure_biased_viewing_directions':
        test_dataset = KinectAzureDataset(usage='test_biased_viewing_directions')
    elif config['TEST_DATASET'] == 'kinect_azure_unseen_viewing_directions':
        test_dataset = KinectAzureDataset(usage='test_unseen_viewing_directions')
    elif config['TEST_DATASET'] == 'nyud':
        test_dataset = NYUD_Dataset(usage='test')
    elif 'scannet' in config['TEST_DATASET']:
        test_dataset = ScannetDataset(usage='test', train_test_split='./data/scannet_standard_train_test_val_split.pkl',
                                      frameGap=20)

    test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=16)

    return test_dataloader


def create_network(config):
    if 'dorn' in config['ARCHITECTURE']:
        cnn = dorn_architecture.DORN(output_channel=3, training_mode=config['OPERATION'])

    elif 'p_fpn' in config['ARCHITECTURE']:
        if 'resnet101' in config['ARCHITECTURE']:
            cnn = fpn_architecture.PlainFPN(in_channels=3, training_mode=config['OPERATION'], backbone='resnet101')
        else:
            raise Exception('Network architecture not implemented!')

    elif 'd_fpn' in config['ARCHITECTURE']:
        if 'resnext101' in config['ARCHITECTURE']:
            cnn = fpn_architecture.ASPP_FPN(in_channels=3, training_mode=config['OPERATION'], backbone='resnext101')
        elif 'resnet101' in config['ARCHITECTURE']:
            cnn = fpn_architecture.ASPP_FPN(in_channels=3, training_mode=config['OPERATION'], backbone='resnet101')
        else:
            raise Exception('Network architecture not implemented!')

    elif config['ARCHITECTURE'] == 'stn_fpn':
            cnn = stn_fpn.SpatialWarpingFPN(fc_img=np.array([202., 202.]))

    else:
        raise Exception('Network architecture not implemented!')

    cnn = cnn.cuda()
    return cnn


def forward_cnn(sample_batched, cnn):
    output_prediction = cnn(sample_batched['image'])
    return output_prediction


if __name__ == '__main__':

    _saving_indices = 0

    # Step 1. Configuration file
    config = parsing_configurations()
    if config['LOG_FOLDER'] != '':
        if not os.path.exists(config['LOG_FOLDER']):
            os.makedirs(config['LOG_FOLDER'])

    # Step 2. Create dataset loader
    test_dataloader = create_dataset_loader(config)

    # Step 3. Create cnn
    cnn = create_network(config)
    if config['CKPT_PATH'] is not '':
        print('Loading checkpoint from %s' % config['CKPT_PATH'])
        cnn.load_state_dict(torch.load(config['CKPT_PATH']))

    counter = 0
    runtime_measurements = []
    # a=torch.zeros(654,3,240,320)
    with torch.no_grad():
        print('<INFERENCE MODE>')
        cnn.eval()
        for iter, sample_batched in enumerate(test_dataloader):
            print(iter, '/', len(test_dataloader))
            sample_batched = {data_key: sample_batched[data_key].cuda() for data_key in sample_batched}
            torch.cuda.synchronize()
            start_time = time.time()
            output_prediction = forward_cnn(sample_batched, cnn)
            torch.cuda.synchronize()
            runtime_measurements.append((time.time() - start_time) / config['BATCH_SIZE'])

            if config['ARCHITECTURE'] == 'stn_fpn':
                for i in range(output_prediction['n2'].shape[0]):
                    saving_rgb_tensor_to_file(sample_batched['image'][i],
                                              os.path.join(config['LOG_FOLDER'], 'input_%d.png' % _saving_indices))
                    saving_normal_tensor_to_file(output_prediction['n2'][i],
                                                 os.path.join(config['LOG_FOLDER'],
                                                              'normal_pred_%d.png' % _saving_indices))
                    _saving_indices += 1
            else:
                for i in range(output_prediction.shape[0]):
                    saving_rgb_tensor_to_file(sample_batched['image'][i],
                                              os.path.join(config['LOG_FOLDER'], 'input_%d.png' % _saving_indices))
                    saving_normal_tensor_to_file(output_prediction[i],
                                                 os.path.join(config['LOG_FOLDER'],
                                                              'normal_pred_%d.png' % _saving_indices))

                    saving_gt_tensor_to_file(sample_batched['Z'][i],
                                                 os.path.join(config['LOG_FOLDER'],
                                                              'normal_gt_%d.png' % _saving_indices))
                    _saving_indices += 1

    # np.save('./pre_sn.npy',a.cpu().numpy())
    print('Median of inference time per image: %.4f (s)' % np.median(np.asarray(runtime_measurements)))
