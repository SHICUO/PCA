# coding:utf-8
# @Time:2022/7/9 12:54
# @Author:LHT
# @File:my_pca
# @GitHub:https://github.com/SHICUO
# @Contact:lin1042528352@163.com
# @Software:PyCharm

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    sta_time = time.time()

    img_list = pd.DataFrame()
    # 读取图片数据，形成400*n维
    for dir_name in os.listdir('ORL'):
        if dir_name.startswith("s"):
            for img_name in os.listdir(os.path.join("ORL", dir_name)):
                if img_name.endswith(".bmp"):
                    img_path = os.path.join("ORL", dir_name, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # cv2.imshow('test', img)
                    # cv2.waitKey(0)
                    img = pd.DataFrame(img.reshape(-1).astype(np.float32)/255)
                    # print(img)
                    img_list = pd.concat([img_list, img], axis=1)

    # print(img_list.shape)
    # print(img_list.iloc[:, 4].to_numpy().reshape(112, 92))
    # 计算平均值
    img_mean = (img_list.sum(axis=1)/img_list.shape[1]).to_numpy()   # 10304*1
    # 标准化
    img_list = img_list.to_numpy().T - img_mean.T    # 400*10304
    # 计算协方差矩阵的特征值和对应的特征向量
    conv = img_list.dot(img_list.T)
    eig_value, eig_vector = np.linalg.eig(conv)
    # print(eig_vector)
    # print(sum(abs(eig_value[0]*eig_vector[:, 0] - conv.dot(eig_vector[:, 0])) <= 0.001))
    # 求原协方差矩阵的特征向量
    eig_vector = img_list.T.dot(eig_vector)   # 10304*400
    # print(eig_vector)
    # np.savetxt('eig_vector.txt', eig_vector, fmt="%.5f")
    # 保存数据
    if not os.path.exists('eig_value.txt'):
        np.savetxt('img_mean.txt', img_mean, fmt="%.5f")
        np.savetxt('eig_value.txt', eig_value, fmt="%.5f")
        eig_vector.tofile('eig_vector.bin')     # 向量数据多保存为二进制文件读取速度快
        print("保存文件成功！")
    end_time = time.time()

    # img_list = []
    # for i in range(1, 41):
    #     for j in range(1, 11):
    #         file = "ORL/s%d/%d.bmp" % (i, j)
    #         img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    #         img = img.astype(np.float32) / 255.0
    #         img_list.append(img)
    #
    # imgs = np.zeros([400, 10304], dtype=np.float32)
    # for i in range(0, 400):
    #     imgs[i, :] = img_list[i].reshape(-1)
    #
    # imgs_mean = imgs.sum(axis=0) / 400.0
    # imgs = imgs - imgs_mean
    # # conv = imgs.transpose(1,0).dot(imgs)
    # conv = imgs.dot(imgs.transpose(1, 0))
    # eig_value, eig_vector = np.linalg.eig(conv)
    # eig_value = eig_value.astype(np.float32)
    # eig_vector = eig_vector.astype(np.float32)
    # eig_vector = imgs.transpose(1, 0).dot(eig_vector)
    # np.savetxt('eig_value.txt', eig_value, fmt="%.5f")
    # eig_vector.tofile('eig_vector.bin')
    # end_time = time.time()

    # 显示平均脸和特征脸
    cv2.imshow('mean_face', (img_mean*255).reshape(112, 92).astype(np.uint8))
    cv2.waitKey(0)
    # for i in range(40):
    #     cv2.imshow('eig_face{:}'.format(i), (eig_vector[:, i]*255).reshape(112, 92).astype(np.uint8))
    #     cv2.waitKey(0)

    print("Cost of time:{:.5f}s".format(end_time-sta_time))
