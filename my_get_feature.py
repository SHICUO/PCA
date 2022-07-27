# coding:utf-8
# @Time:2022/7/11 10:17
# @Author:LHT
# @File:my_get_feature
# @GitHub:https://github.com/SHICUO
# @Contact:lin1042528352@163.com
# @Software:PyCharm

import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import matplotlib.pyplot as plt
from torchvision.models import vgg

# BASE_DIR = os.path.dirname(__file__)


if __name__ == '__main__':
    # 设置特征脸数量
    k = 256
    # 获取特征向量
    img_mean = np.loadtxt('img_mean.txt', dtype=np.float32)
    eig_vector = np.fromfile('eig_vector.bin', dtype=np.float32)
    # eig_vector = np.loadtxt('eig_vector.txt', dtype=np.float32)
    # print(eig_vector)
    # 取前面k个
    eig_faces = eig_vector.reshape(10304, -1, order='C')[:, :k]     # order='C'横着读写
    # print(eig_faces)

    # 恢复特征脸图片并保存，先加均值，再乘255
    for i in tqdm.tqdm(range(k)):
        # print(eig_faces[:, i])
        # print(eig_faces[:, i]+img_mean)
        eig_face = (eig_faces[:, i]+img_mean).reshape(112, -1)*255
        # print(eig_face)
        eig_face[eig_face > 255] = 255
        eig_face[eig_face < 0] = 0
        eig_face = eig_face.astype(np.uint8)
        # plt.hist(eig_face)
        # plt.show()
        # cv2.imshow('face', eig_face)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join("eigenfaces", "%s.bmp" % i), eig_face)

    # 获取输入400张图片的权重，并重构
    for face_class in tqdm.tqdm(range(1, 41)):
        for face_n in range(1, 11):
            face_o = cv2.imread('ORL/s%d/%d.bmp' % (face_class, face_n), cv2.IMREAD_GRAYSCALE)
            face = face_o.reshape(-1).astype(np.float32)/255.0 - img_mean  # 标准化
            eig_face_weight = face.dot(eig_faces)  # 权重
            np.savetxt(os.path.join("eig_face_weight", "%s_%s.txt" % (face_class, face_n)), eig_face_weight, fmt='%.5f')

            # 重构
            re_face = eig_faces.dot(eig_face_weight.T)
            re_face = (re_face+img_mean)*255
            # re_face_norm = re_face+abs(min(re_face))
            # re_face_norm = 1 * (np.power((1.0 + re_face_norm), 2.5))
            # plt.figure()
            # plt.hist(re_face_norm)
            re_face_norm = re_face.reshape(112, -1)
            re_face_norm = cv2.normalize(re_face_norm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cv2.imwrite(os.path.join("refactoring", "%s_%s.bmp" % (face_class, face_n)), re_face_norm)
            # re_face_norm = cv2.equalizeHist(re_face_norm)
            # plt.figure()
            # plt.hist(re_face_norm)
            # plt.show()
            # re_face = re_face*255
            # re_face[re_face > 255] = 255
            # re_face[re_face < 0] = 0
            # re_face = re_face.astype(np.uint8).reshape(112, -1)
            # cv2.imshow('re_face', re_face_norm)
            # cv2.waitKey(0)
            # cv2.imshow('face', face_o)
            # cv2.waitKey(0)

