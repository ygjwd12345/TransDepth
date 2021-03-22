#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python evaluate_surface_normal.py --checkpoint_path './log/training_release/model-final.ckpt' --log_folder './log/testing_release/' --batch_size 16 --net_architecture 'd_fpn_resnext101' --test_dataset 'scannet_standard'

CUDA_VISIBLE_DEVICES=0 python evaluate_surface_normal.py --checkpoint_path './log/transformer/model-final.ckpt' --log_folder './log/testing_release/' --batch_size 16 --net_architecture 'd_fpn_resnext101' --test_dataset 'nyud'

