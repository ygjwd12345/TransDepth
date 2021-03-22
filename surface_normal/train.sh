#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 train_surface_normal.py  --log_folder './log/training_release/' --operation 'train_L2_loss' --learning_rate 0.0001 --batch_size 4 --net_architecture 'd_fpn_resnext101' --train_dataset 'scannet_standard' --test_dataset 'scannet_standard'  --test_dataset 'nyud' --val_dataset 'scannet_standard'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_surface_normal.py  --log_folder './log/training_release/' --operation 'train_L2_loss' --learning_rate 0.0001 --batch_size 4 --net_architecture 'd_fpn_resnext101' --train_dataset 'scannet_standard' --test_dataset 'scannet_standard'  --test_dataset 'nyud' --val_dataset 'scannet_standard'

CUDA_VISIBLE_DEVICES=0 python train_surface_normal.py --log_folder './log/transformer_sn/' --operation 'train_robust_acos_loss' --learning_rate 0.0001 --batch_size 16 --net_architecture 'd_fpn_resnext101' --train_dataset 'nyud' --test_dataset 'nyud' --val_dataset 'nyud'  --epoch 50
CUDA_VISIBLE_DEVICES=0 python train_surface_normal.py --log_folder './log/transformer_sn/' --operation 'train_L2_loss' --learning_rate 0.0001 --batch_size 16 --net_architecture 'd_fpn_resnext101' --train_dataset 'nyud' --test_dataset 'nyud' --val_dataset 'nyud' --epoch 50
CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch train_surface_normal.py --log_folder './log/training_release/' --operation 'train_acos_loss' --learning_rate 0.0001 --batch_size 48 --net_architecture 'd_fpn_resnext101' --train_dataset 'scannet_standard' --test_dataset 'scannet_standard'  --test_dataset 'nyud' --val_dataset 'scannet_standard'
CUDA_VISIBLE_DEVICES=1 python train_surface_normal.py --log_folder './log/training_release/' --operation 'train_L2_loss' --learning_rate 0.0001 --batch_size 4 --net_architecture 'd_fpn_resnext101' --train_dataset 'scannet_standard' --test_dataset 'scannet_standard'  --test_dataset 'nyud' --val_dataset 'scannet_standard'




