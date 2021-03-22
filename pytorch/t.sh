CUDA_VISIBLE_DEVICES=1 python bts_main.py arguments_train_nyu_debug.txt
CUDA_VISIBLE_DEVICES=1 python bts_main.py arguments_train_eigen_debug.txt
CUDA_VISIBLE_DEVICES=1 python bts_test.py arguments_test_eigen.txt
CUDA_VISIBLE_DEVICES=1 python bts_test.py arguments_test_nyu.txt
