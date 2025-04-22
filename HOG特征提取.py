import os
import time

import numpy as np
import cv2
import skimage


winSize = (512,128)     # 检测窗口尺寸（需与图像尺寸一致）
blockSize = (16, 16)    # 块大小（必须是cellSize的整数倍）
blockStride = (8, 8)    # 块滑动步长（建议是cellSize的1/2）
cellSize = (8, 8)       # 细胞单元大小
nbins = 9
hog = cv2.HOGDescriptor(

    _winSize=winSize,
    _blockSize=blockSize,
    _blockStride=blockStride,
    _cellSize=cellSize,
    _nbins=nbins
)


if __name__ == '__main__':
    imgPath = r'./data/images'
    imgFiles = os.listdir(imgPath)

    imgFiles.sort(key=lambda x: int(x[3:-4]))


    HOG = []
    for imgFile in imgFiles:
        img = cv2.imread(os.path.join(imgPath, imgFile), 0)
        HOG.append(hog.compute(img))
        print(imgFile, 'done')

    HOG = np.array(HOG)
    print(HOG.dtype)
    print(HOG.shape)
    np.save(r'./data/HOG', HOG)