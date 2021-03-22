import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
sn=cv.imread('./datasets/nyu-normal/nyu_normals_gt/test/00001.png')
# print(sn[1,1,0])
plt.imshow(255-sn)# 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()