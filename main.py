import cv2
import numpy as np
import joblib
import skimage
from sklearn.preprocessing import StandardScaler


# 要求输入的图像为（512，1408）
class DefectDetector(object):
    def __init__(self, step=64):
        self.step = 64  # 拆分图片时候的步长
        self.partSize = (512, 128)
        self.model = joblib.load(r'./mySVMRBF_GCLM64.pkl')  # 支撑向量机模型，注意共生矩阵特征的levels为64
        self.standerScaler = StandardScaler()

    # 共生矩阵特征提取
    def getGlcm(self,img, distances=np.array([1, 2, 4, 8], dtype=np.int32),
                 angles=np.array([0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], dtype=np.float32),
                 props=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
                 levels=256):
        img = img // (256 // levels)
        glcmsNormed = skimage.feature.graycomatrix(img, distances, angles, levels, True, True)  # 归一化后返回np.float64类型

        F = np.array([], dtype=np.float64)
        n_features = len(distances) * len(angles) * len(props)
        F = np.empty(n_features, dtype=np.float64)

        # 提取矩阵特征，对比度，相异性， 同质性， 能量， 相关性， 能量平方
        cnt = 0
        for prop in props:
            result = skimage.feature.graycoprops(glcmsNormed, prop).ravel()
            F[cnt:cnt + len(distances) * len(angles)] = result
            cnt += len(distances) * len(angles)
        return F

    # 输入图片，输出存在缺陷的矩形框，如果没有输出[]
    def detect(self, img):
        img = cv2.resize(img, (512, 1408))
        F = np.zeros((img.shape[0]//self.step, 96), dtype=np.float64)
        print('F.shape: ', F.shape)
        # 与处理图像，将图像拆分为partSize一截一截的
        for i in range(img.shape[0]//self.step):
            F[i,:] = self.getGlcm(img[i*self.step:i*self.step+self.partSize[1]], levels=64)

        #进行标准化
        # F = self.standerScaler.fit_transform(F)
        detectResult = self.model.predict(F)
        print(detectResult)
        detectResult = np.where(detectResult==1)[0]*self.step
        return detectResult


if __name__ == '__main__':
    img = cv2.imread(r'./KolektorSDD/kos05/part5.jpg', 0)  # 有裂纹
    # img = cv2.imread(r'./KolektorSDD/kos05/part1.jpg', 0)  # 无裂纹
    print('img.shape: ', img.shape)
    myDefectDetector = DefectDetector()
    result = myDefectDetector.detect(img)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    for i in result:
        cv2.imshow('defect{}'.format(i), img[i:i+128])
    cv2.waitKey(0)
