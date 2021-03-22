import torch
import numpy as np
import skimage.io as sio
import argparse
from torch.utils.data import DataLoader
from network import dorn_architecture, fpn_architecture
from dataset_loader.dataset_loader_scannet import ScannetDataset
from dataset_loader.dataset_loader_nyud import NYUD_Dataset
from dataset_loader.dataset_loader_kinectazure import KinectAzureDataset
import os
import time
from torch import distributed
from apex.parallel import DistributedDataParallel
from apex import amp
from torch.utils.data.distributed import DistributedSampler


dataset_dict = {'scannet_standard': './data/scannet_standard_train_test_val_split.pkl',
                'scannet_framenet': './data/framenet_train_test_split.pkl'}

def parsing_configurations():
    parser = argparse.ArgumentParser(description='Train surface normal estimation')
    parser.add_argument('--log_folder', type=str, default='')
    parser.add_argument('--operation', type=str, default='evaluate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_dataset', type=str, default='scannet_standard')
    parser.add_argument('--test_dataset', action='append', required=False)
    parser.add_argument('--val_dataset', action='append', required=False)
    parser.add_argument('--net_architecture', type=str, default='dorn')
    parser.add_argument('--exp', type=str, default='noname', help = 'experiment name')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--opt_level', type=str, choices=['O0', 'O1', 'O2', 'O3'], default='O0')

    args = parser.parse_args()

    # default dataset
    if args.test_dataset is None:
        args.test_dataset = ['scannet_standard']

    if args.val_dataset is None:
        args.val_dataset = ['scannet_standard']

    config = {'LOG_FOLDER': args.log_folder,
              'CKPT_PATH': args.checkpoint_path,
              'OPERATION': args.operation,
              'BATCH_SIZE': args.batch_size,
              'LEARNING_RATE': args.learning_rate,
              'TRAIN_DATASET': args.train_dataset,
              'TEST_DATASET': args.test_dataset,
              'VAL_DATASET': args.val_dataset,
              'ARCHITECTURE': args.net_architecture,
              'epoch':args.epoch,
              'local_rank':args.local_rank,
              'opt_level':args.opt_level}

    return config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log(str, fp=None):
    if fp is not None:
        fp.write('%s\n' % str)
        fp.flush()
    print(str)


def Normalize(dir_x):
    dir_x_l = torch.sqrt(torch.sum(dir_x ** 2,dim=1) + 1e-6).view(dir_x.shape[0], 1, dir_x.shape[2], dir_x.shape[3])
    dir_x_l = torch.cat([dir_x_l, dir_x_l, dir_x_l], dim=1)
    return dir_x / dir_x_l


def compute_surface_normal_angle_error(sample_batched, output_pred, mode='evaluate', angle_type='delta'):
    if 'Z' in sample_batched:
        surface_normal_pred = output_pred
        if mode == 'evaluate':
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'])
            prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
            return torch.acos(prediction_error) * 180.0 / np.pi
        elif mode == 'train_L2_loss':
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'], dim=1, eps=1e-6)
            mask = sample_batched['mask'] > 0
            mask = mask.detach()
            return -torch.sum(prediction_error[mask]), 1.0-torch.mean(prediction_error[mask])
        elif mode == 'train_acos_loss':
            mask = sample_batched['mask'] > 0
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'], dim=1, eps=1e-6)
            acos_mask = mask.float() \
                    * (prediction_error.detach() < 0.999).float() * (prediction_error.detach() > -0.999).float()
            acos_mask = acos_mask > 0.0
            optimize_loss = torch.sum(torch.acos(prediction_error[acos_mask]))
            logging_loss = torch.mean(torch.acos(prediction_error[acos_mask]))
            return optimize_loss, logging_loss
        elif mode == 'train_robust_acos_loss':
            mask = sample_batched['mask'] > 0
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'], dim=1, eps=1e-6)
            acos_mask = mask.float() \
                   * (prediction_error.detach() < 0.9999).float() * (prediction_error.detach() > 0.0).float()
            cos_mask = mask.float() * (prediction_error.detach() <= 0.0).float()
            acos_mask = acos_mask > 0.0
            cos_mask = cos_mask > 0.0
            optimize_loss = torch.sum(torch.acos(prediction_error[acos_mask])) - torch.sum(prediction_error[cos_mask])
            logging_loss = optimize_loss.detach() / (torch.sum(cos_mask) + torch.sum(acos_mask))
            return optimize_loss, logging_loss


total_normal_errors = None
best_median_error = None

def accumulate_prediction_error(sample_batched, angle_error_prediction):
    global total_normal_errors
    mask = sample_batched['mask'] > 0
    if total_normal_errors is None:
        total_normal_errors = angle_error_prediction[mask].data.cpu().numpy()
    else:
        total_normal_errors = np.concatenate((total_normal_errors, angle_error_prediction[mask].data.cpu().numpy()))


def log_normal_stats(epoch, iter, normal_error_in_angle, fp=None):
    log('Epoch %d, Iter %d, Mean %f, Median %f, Rmse %f, 5deg %f, 7.5deg %f, 11.25deg %f, 22.5deg %f, 30deg %f' %
    (epoch, iter, np.average(normal_error_in_angle), np.median(normal_error_in_angle),
     np.sqrt(np.sum(normal_error_in_angle * normal_error_in_angle) / normal_error_in_angle.shape),
     np.sum(normal_error_in_angle < 5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 7.5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 11.25) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 22.5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 30) / normal_error_in_angle.shape[0]), fp)


def create_dataset_loader(config,world_size,rank):
    if config['TRAIN_DATASET'] == 'scannet_standard':
        train_dataset = ScannetDataset(usage='train', train_test_split=dataset_dict[config['TRAIN_DATASET']])
        train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                                      sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank),
                                      num_workers=16, drop_last=True)
    elif config['TRAIN_DATASET'] == 'nyud':
        train_dataset = NYUD_Dataset(usage='train')
        train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                                      num_workers=4)


    test_dataloader_list = {}
    for test_split in config['TEST_DATASET']:
        if 'scannet' in test_split:
            test_dataset = ScannetDataset(usage='test', train_test_split=dataset_dict[test_split],
                                          frameGap=200)
            test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                         sampler=DistributedSampler(test_dataset, num_replicas=world_size, rank=rank),
                                         num_workers=4)
            test_dataloader_list[test_split] = test_dataloader
        elif 'nyud' in test_split:
            test_dataset = NYUD_Dataset(usage='test')
            test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                         sampler=DistributedSampler(test_dataset, num_replicas=world_size, rank=rank),
                                          num_workers=4)
            test_dataloader_list['nyud'] = test_dataloader

    val_dataloader_list = {}
    for val_split in config['VAL_DATASET']:
        if 'scannet_standard' in val_split:
            val_dataset = ScannetDataset(usage='val', train_test_split=dataset_dict[val_split],
                                         frameGap=200)
            val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'],
                                        sampler=DistributedSampler(val_dataset, num_replicas=world_size, rank=rank),
                                        num_workers=4, drop_last=True)
            val_dataloader_list[val_split] = val_dataloader
        elif 'nyud' in val_split:
            val_dataset = NYUD_Dataset(usage='test')
            val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'],
                                        sampler=DistributedSampler(val_dataset, num_replicas=world_size, rank=rank),
                                         num_workers=4, drop_last=True)
            val_dataloader_list[val_split] = val_dataloader
        else:
            raise Exception('Validation split is not implemented for this dataset!')

    return train_dataloader, test_dataloader_list, val_dataloader_list


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
    else:
        raise Exception('Network architecture not implemented!')

    cnn = cnn.cuda()
    return cnn


def forward_cnn(sample_batched, cnn, config):
    output_prediction = cnn(sample_batched['image'])
    return output_prediction


if __name__ == '__main__':
    # Step 1. Configuration file
    config = parsing_configurations()

    # Create logger file
    training_loss_file = None
    evaluate_stat_file_list = {}
    validation_stat_file_list = {}

    if config['LOG_FOLDER'] != '':

        if not os.path.exists(config['LOG_FOLDER']):
            os.makedirs(config['LOG_FOLDER'])
        training_loss_file = open(config['LOG_FOLDER'] + '/training_loss.txt', 'w')

        for test_split in config['TEST_DATASET']:
            evaluate_stat_file_list[test_split] = open(config['LOG_FOLDER'] + '/evaluate_stat_%s.txt' % test_split, 'w')

        for val_split in config['VAL_DATASET']:
            validation_stat_file_list[val_split] = open(config['LOG_FOLDER'] + '/validation_stat_%s.txt' % val_split, 'w')

    log(config, training_loss_file)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    distributed.init_process_group(backend='nccl')
    device_id, device = config['local_rank'], torch.device(config['local_rank'])
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)
    print(f"Device: {device}")
    print(f"Total batch size is {config['BATCH_SIZE'] * world_size}")

    # Step 2. Create dataset loader
    train_dataloader, test_dataloader_list, val_dataloader_list = create_dataset_loader(config,world_size,rank)

    # Step 3. Create cnn
    cnn = create_network(config)
    if config['CKPT_PATH'] is not '':
        print('Loading checkpoint from %s' % config['CKPT_PATH'])
        cnn.load_state_dict(torch.load(config['CKPT_PATH']))


    # Step 3.1: Count the number of trainable parameters
    log(cnn, training_loss_file)
    log('Number of trainable parameters: %d' % count_parameters(cnn), training_loss_file)

    # Step 4. Create optimizer
    optimizer = None
    if 'train' in config['OPERATION']:
        optimizer = torch.optim.Adam(cnn.parameters(), lr=config['LEARNING_RATE'], betas=(0.9, 0.999))
    cnn, optimizer = amp.initialize(cnn.to(device), optimizer, opt_level=config['opt_level'])
    # Put the model on GPU
    cnn = DistributedDataParallel(cnn.to(device), delay_allreduce=True)
    # Step 5. Learning loop
    if 'train' in config['OPERATION']:
        for epoch in range(0, config['epoch']):
            for iter, sample_batched in enumerate(train_dataloader):
                cnn.train()
                sample_batched = {data_key: sample_batched[data_key].to(device, dtype=torch.float32) for data_key in sample_batched}
                # zero the parameter gradients
                optimizer.zero_grad()

                # Step 5a: Forward pass
                output_prediction = forward_cnn(sample_batched, cnn, config)

                # Step 5b: Compute loss
                # print(sample_batched['Z'].dtype)
                # print(sample_batched['mask'].dtype)
                losses, logging_losses = compute_surface_normal_angle_error(sample_batched,
                                                                            output_prediction,
                                                                            mode=config['OPERATION'],
                                                                            angle_type='delta')
                # Step 5c: Backward pass and update
                # losses.backward()
                with amp.scale_loss(losses, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                # Step 5d. Print loss value
                if iter % 10 == 0:
                    log('Epoch %d, Iter %d, Loss %.4f' % (epoch, iter, logging_losses), training_loss_file)
                # Step 5e. Print robust evaluation stats
                if iter % 40 == 0 and iter > 0:
                    evaluation_mode = 'evaluate_mix_loss' if 'mix_loss' in config['OPERATION'] else 'evaluate'
                    with torch.no_grad():
                        cnn.eval()
                        print('<=============================TEST MODE===============================>')
                        for name, test_dataloader in test_dataloader_list.items():
                            print('<%s dataset>' % name.upper())
                            total_normal_errors = None
                            for _, eval_batch in enumerate(test_dataloader):
                                eval_batch = {data_key: eval_batch[data_key].to(device, dtype=torch.float32) for data_key in eval_batch}
                                output_prediction = forward_cnn(eval_batch, cnn, config)
                                angle_error_prediction = compute_surface_normal_angle_error(eval_batch,
                                                                                            output_prediction,
                                                                                            mode=evaluation_mode,
                                                                                            angle_type='delta')



                                accumulate_prediction_error(eval_batch, angle_error_prediction)

                            log_normal_stats(epoch, iter, total_normal_errors, evaluate_stat_file_list[name])

                        print('<==========================VALIDATION MODE============================>')
                        for name, val_dataloader in val_dataloader_list.items():
                            print('<%s dataset>' % name.upper())
                            total_normal_errors = None
                            for _, eval_batch in enumerate(val_dataloader):
                                eval_batch = {data_key: eval_batch[data_key].to(device, dtype=torch.float32) for data_key in eval_batch}
                                output_prediction = forward_cnn(eval_batch, cnn, config)
                                angle_error_prediction = compute_surface_normal_angle_error(eval_batch,
                                                                                            output_prediction,
                                                                                            mode=evaluation_mode,
                                                                                            angle_type='delta')


                                accumulate_prediction_error(eval_batch, angle_error_prediction)
                            log_normal_stats(epoch, iter, total_normal_errors, validation_stat_file_list[name])

                            current_median_error = np.median(total_normal_errors)
                            if best_median_error is None:
                                best_median_error = current_median_error
                                print('Best median error in validation: %f, saving checkpoint epoch %d, iter %d' % (best_median_error, epoch, iter))
                            else:
                                if current_median_error < best_median_error:
                                    best_median_error = current_median_error
                                    print('Best median error in validation: %f, saving checkpoint epoch %d, iter %d' % (best_median_error, epoch, iter))
                                    path = config['LOG_FOLDER'] + '/model-final.ckpt'
                                    torch.save(cnn.state_dict(), path)
                            torch.distributed.barrier()


