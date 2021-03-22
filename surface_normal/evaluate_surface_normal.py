import torch
import numpy as np
import skimage.io as sio
import argparse
from torch.utils.data import DataLoader
from network import dorn_architecture, fpn_architecture, stn_fpn
from dataset_loader.dataset_loader_scannet import ScannetDataset
from dataset_loader.dataset_loader_nyud import NYUD_Dataset
from dataset_loader.dataset_loader_kinectazure import KinectAzureDataset
import os
import scipy.io as scio
import time


dataset_dict = {'scannet_standard': './data/scannet_standard_train_test_val_split.pkl',
                'scannet_framenet': './data/framenet_train_test_split.pkl'}

def parsing_configurations():
    parser = argparse.ArgumentParser(description='Test surface normal estimation')
    parser.add_argument('--log_folder', type=str, default='')
    parser.add_argument('--operation', type=str, default='evaluate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--test_dataset', type=str, default='scannet_standard')
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


def saving_rgb_tensor_to_file(rgb_tensor, path,i):
    output_rgb_img = np.uint8((rgb_tensor.permute(1, 2, 0).detach().cpu()) * 255)
    path=os.path.join(path,str(i)+'_img'+'.jpg')
    sio.imsave(path, output_rgb_img)


def saving_normal_tensor_to_file(normal_tensor, path,i):
    normal_tensor=normalize3D(normal_tensor)
    # output_normal_img = np.uint8((normal_tensor.permute(1, 2, 0).detach().cpu() + 1) * 127.5)
    # output_normal_img = np.uint8((normal_tensor.permute(1, 2, 0).detach().cpu() + 1) * 127.5)
    # output_normal_img=255-output_normal_img
    path=os.path.join(path,str(i)+'_sn'+'.jpg')
    sio.imsave(path, normal_tensor.permute(1, 2, 0).detach().cpu())

def normalize3D(x, eps=1e-12):
    N = torch.zeros(x.shape)
    nx = x[0]
    ny = x[1]
    nz = x[2]
    n = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2).clamp_min(eps)
    nx = nx / n
    ny = ny / n
    nz = nz / n
    N[0] = nx
    N[1] = ny
    N[2] = nz
    return N
def Normalize(dir_x):
    dir_x_l = torch.sqrt(torch.sum(dir_x ** 2,dim=1) + 1e-6).view(dir_x.shape[0], 1, dir_x.shape[2], dir_x.shape[3])
    dir_x_l = torch.cat([dir_x_l, dir_x_l, dir_x_l], dim=1)
    return dir_x / dir_x_l


def compute_surface_normal_angle_error(sample_batched, output_pred, mode='evaluate'):
    if 'Z' in sample_batched:
        surface_normal_pred = output_pred
        if mode == 'evaluate':
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'])
            prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
            return torch.acos(prediction_error) * 180.0 / np.pi


total_normal_errors = None
def accumulate_prediction_error(sample_batched, angle_error_prediction):
    global total_normal_errors
    mask = sample_batched['mask'] > 0
    if total_normal_errors is None:
        total_normal_errors = angle_error_prediction[mask].data.cpu().numpy()
    else:
        total_normal_errors = np.concatenate((total_normal_errors, angle_error_prediction[mask].data.cpu().numpy()))


def log_normal_stats(normal_error_in_angle, fp=None):
    log('Mean %f, Median %f, Rmse %f, 5deg %f, 7.5deg %f, 11.25deg %f, 22.5deg %f, 30deg %f' %
    (np.average(normal_error_in_angle), np.median(normal_error_in_angle),
     np.sqrt(np.sum(normal_error_in_angle * normal_error_in_angle) / normal_error_in_angle.shape),
     np.sum(normal_error_in_angle < 5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 7.5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 11.25) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 22.5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 30) / normal_error_in_angle.shape[0]), fp)


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
        test_dataset = ScannetDataset(usage='test', train_test_split=dataset_dict[config['TEST_DATASET']],
                                      frameGap=1)
    print(len(test_dataset))
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
        if 'kinect_azure' in config['TEST_DATASET']:
            cnn = stn_fpn.SpatialWarpingFPN(fc_img=np.array([202., 202.]))
        else:
            cnn = stn_fpn.SpatialWarpingFPN()

    else:
        raise Exception('Network architecture not implemented!')

    cnn = cnn.cuda()
    return cnn


def forward_cnn(sample_batched, cnn):
    output_prediction = cnn(sample_batched['image'])
    return output_prediction


if __name__ == '__main__':
    # Step 1. Configuration file
    config = parsing_configurations()
    # Create logger file
    evaluate_stat_file = None
    if config['LOG_FOLDER'] != '':
        if not os.path.exists(config['LOG_FOLDER']):
            os.makedirs(config['LOG_FOLDER'])
        evaluate_stat_file = open(config['LOG_FOLDER'] + '/evaluate_surface_normal_stat.txt', 'w')
    log(config, evaluate_stat_file)

    # Step 2. Create dataset loader
    test_dataloader = create_dataset_loader(config)


    # Step 3. Create cnn
    cnn = create_network(config)
    if config['CKPT_PATH'] is not '':
        print('Loading checkpoint from %s' % config['CKPT_PATH'])
        cnn.load_state_dict(torch.load(config['CKPT_PATH']))

    evaluation_mode = 'evaluate'

    total_normal_errors = None
    ### save output
    with torch.no_grad():
        print('<EVALUATION MODE>')
        cnn.eval()
        total_normal_errors = None
        for iter, sample_batched in enumerate(test_dataloader):
            print(iter, '/', len(test_dataloader))
            sample_batched = {data_key: sample_batched[data_key].cuda() for data_key in sample_batched}
            output_prediction = forward_cnn(sample_batched, cnn)
            if config['ARCHITECTURE'] == 'stn_fpn':
                angle_error_prediction = compute_surface_normal_angle_error(sample_batched,
                                                                            output_prediction['n2'],
                                                                            mode=evaluation_mode)
            else:
                angle_error_prediction = compute_surface_normal_angle_error(sample_batched,
                                                                            output_prediction,
                                                                            mode=evaluation_mode)
            accumulate_prediction_error(sample_batched, angle_error_prediction)
        print('========= SURFACE NORMAL EVALUATION ERROR STATS =========')
        log_normal_stats(total_normal_errors, evaluate_stat_file)

