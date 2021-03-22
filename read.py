import os
with open('./train_test_inputs/eigen_train_files_with_gt.txt', 'r') as f:
    filenames = f.readlines()


for line3 in filenames:
    gt_path =line3.split()[1]
    img_path =line3.split()[0]
    # print(img_path.split('/')[0])
    depth_path = os.path.join('./dataset/kitti_dataset/',  gt_path)
    print(depth_path)
    break



