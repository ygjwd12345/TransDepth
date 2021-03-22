### prepare nyu
cd ./pytorch
mkdir dataset
mkdir dataset/nyu_depth_v2
python ../utils/download_from_gdrive.py 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP ./dataset/nyu_depth_v2/sync.zip
cd dataset
cd nyu_depth_v2
unzip sync.zip
### prepare model save fold !!!!
mkdir models
### run nyu in depth_nyu/pytorch
CUDA_VISIBLE_DEVICES=3,4 python bts_main.py arguments_train_nyu.txt
python bts_main.py arguments_train_eigen_L.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 python bts_main.py arguments_train_nyu.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 python bts_main.py arguments_train_eigen.txt

## test
CUDA_VISIBLE_DEVICES=1 python bts_test.py arguments_test_nyu.txt
python ../utils/eval_with_pngs.py --pred_path vis_att_bts_nyu_v2_pytorch_att/raw/ --gt_path ../../dataset/nyu_depth_v2/official_splits/test/ --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop
CUDA_VISIBLE_DEVICES=1 python bts_test.py arguments_test_eigen.txt
python ../utils/eval_with_pngs.py --pred_path vis_att_bts_eigen_v2_pytorch_att/raw/ --gt_path ./dataset/kitti_dataset/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --do_kb_crop --garg_crop
### prepare kitti
cd dataset
mkdir kitti_dataset
cd kitti_dataset
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
unzip data_depth_annotated.zip
cd train
mv * ../
rm train
cd val
mv * ../
rm val
rm data_depth_annotated.zip
#unzip '*.zip'
python bts_main.py arguments_train_eigen.txt

docker run -it --rm --gpus all -v /data2/gyang/PGA-net:/home pga
#### debug
CUDA_VISIBLE_DEVICES=1 python bts_main.py arguments_train_nyu_debug.txt
