import os

import cv2
import numpy as np
import skimage

def get_glcm(img, distances = np.array([1,2,4,8],dtype=np.int32),
             angles = np.array([0, np.pi/4, np.pi/2, np.pi*3/4], dtype=np.float32),
             props=['contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM'],
             levels = 256):
    img = img//(256//levels)
    glcmsNormed = skimage.feature.graycomatrix(img, distances, angles, levels, True, True)  # 归一化后返回np.float64类型

    F = np.array([], dtype=np.float64)
    n_features = len(distances) * len(angles) * len(props)
    F = np.empty(n_features, dtype=np.float64)

    # 提取矩阵特征，对比度，相异性， 同质性， 能量， 相关性， 能量平方
    cnt = 0
    for prop in props:
        result = skimage.feature.graycoprops(glcmsNormed, prop).ravel()
        F[cnt:cnt+len(distances) * len(angles)] = result
        cnt+=len(distances) * len(angles)
    return F

if __name__ == '__main__':

    # img = cv2.imread(r'./street.jpg', 0)
    # F = get_glcm(img)
    # print(F.shape, F.dtype)
    # print(F)


    imgPath = r'./data/images'
    imgFiles = os.listdir(imgPath)

    imgFiles.sort(key=lambda x: int(x[3:-4]))


    GLCM = []
    for imgFile in imgFiles:
        img = cv2.imread(os.path.join(imgPath, imgFile), 0)  # 灰度图形式读取图片
        GLCM.append(get_glcm(img, levels=64))
        print(imgFile, 'done')

    GLCM = np.array(GLCM)
    print(GLCM.dtype)
    print(GLCM.shape)
    np.save(r'./data/GLCM64Thresh', GLCM)

