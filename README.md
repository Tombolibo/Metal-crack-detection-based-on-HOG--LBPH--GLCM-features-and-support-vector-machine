# Metal crack detection based on HOG, LBPH, GLCM features and support vector machine
## 基于HOG，LBPH，GLCM特征和支撑向量机的金属裂纹检测

使用KolektorSDD金属裂纹数据集，分别对图像进行HOG特征、LBPH特征、GLCM特征提取。<br/>
对提取特征及进行支撑向量机训练，其中GLCM特征所训练模型效果最好测试集混淆矩阵如下：<br/>
[[1445   51]<br/>
 [   9  154]]<br/>
（原始数据无裂缝数量远大于有裂缝数量）<br/>
