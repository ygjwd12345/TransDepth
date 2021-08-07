Official PyTorch code for [Transformer-Based Attention Networks for Continuous Pixel-Wise Prediction](https://arxiv.org/abs/2103.12091). <br>
Guanglei Yang, [Hao Tang](http://disi.unitn.it/~hao.tang/), Mingli Ding, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en), [Elisa Ricci](https://scholar.google.com/citations?hl=en&user=xf1T870AAAAJ&view_op=list_works&sortby=pubdate). <br>
ICCV 2021 <br>
Apply Transformer into depth predciton and surface normal estimation.

## Prepare pretrain model
we choose R50-ViT-B_16 as our encoder.
```bash root transformerdepth
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz 
mkdir ./model/vit_checkpoint/imagenet21k 
mv R50+ViT-B_16.npz ./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

## Prepare Dateset
### prepare nyu
```bash
mkdir -p pytorch/dataset/nyu_depth_v2
python utils/download_from_gdrive.py 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP pytorch/dataset/nyu_depth_v2/sync.zip
cd pytorch/dataset/nyu_depth_v2
unzip sync.zip
```
### prepare kitti
```bash
cd dataset
mkdir kitti_dataset
cd kitti_dataset
### image move kitti_archives_to_download.txt into kitti_dataset
wget -i kitti_archives_to_download.txt

### label
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
unzip data_depth_annotated.zip
cd train
mv * ../
cd ..  
rm -r train
cd val
mv * ../
cd ..
rm -r val
rm data_depth_annotated.zip
```
## Environment 
```bash
pip install -r requirement.txt
```

## Run
Train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python bts_main.py arguments_train_nyu.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 python bts_main.py arguments_train_eigen.txt
```
Test: Pick up nice result
```bash
CUDA_VISIBLE_DEVICES=1 python bts_test.py arguments_test_nyu.txt
python ../utils/eval_with_pngs.py --pred_path vis_att_bts_nyu_v2_pytorch_att/raw/ --gt_path ../../dataset/nyu_depth_v2/official_splits/test/ --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop
CUDA_VISIBLE_DEVICES=1 python bts_test.py arguments_test_eigen.txt
python ../utils/eval_with_pngs.py --pred_path vis_att_bts_eigen_v2_pytorch_att/raw/ --gt_path ./dataset/kitti_dataset/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --do_kb_crop --garg_crop
```
Debug
```bash
CUDA_VISIBLE_DEVICES=1 python bts_main.py arguments_train_nyu_debug.txt
```

## Download Pretrained Model

```bash
sh scripts/download_TransDepth_model.sh kitti_depth

sh scripts/download_TransDepth_model.sh nyu_depth

sh scripts/download_TransDepth_model.sh nyu_surfacenormal
```

Note: Please try to execute the command line a second time, if it doesn’t work the first time.


# Reference
[BTS](https://github.com/cogaplex-bts/bts)

[ViT](https://github.com/jeonsworld/ViT-pytorch)

[TransUNet](https://github.com/Beckschen/TransUNet)


[Do‘s code](https://github.com/MARSLab-UMN/TiltedImageSurfaceNormal)
# Visualization result share
We provide all vis result of all tasks. [link](https://www.dropbox.com/sh/iv4zb4fl3vn294i/AACGjH0jIPtyZ8qwr_erLKr9a?dl=0)
