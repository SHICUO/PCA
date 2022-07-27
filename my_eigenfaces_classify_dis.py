# coding:utf-8
# @Time:2022/7/12 22:30
# @Author:LHT
# @File:my_eigenfaces_classify_dis
# @GitHub:https://github.com/SHICUO
# @Contact:lin1042528352@163.com
# @Software:PyCharm

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_data():
    # 处理特征权重数据，把每一类人脸前9个作为训练集，最后一个作为测试集
    # 读取数据文件名字
    file_name = []
    for file in os.listdir("eig_face_weight"):
        if file.endswith(".txt"):
            file_name.append(file)

    # 创建两个list存储训练集和测试集数据
    # 训练9——测试1
    train_data = np.zeros([360, 256])
    train_label = np.zeros(360, dtype=np.uint8)
    test_data = np.zeros([40, 256])
    test_label = np.zeros(40, dtype=np.uint8)
    for f in file_name:
        cla_ord = f.split(".")[0]
        k, n = cla_ord.split("_")
        k = int(k)
        n = int(n)
        if n < 10:
            train_data[(k - 1) * 9 + (n - 1)] = np.loadtxt(os.path.join("eig_face_weight", f))
            train_label[(k - 1) * 9 + (n - 1)] = k-1
        else:
            test_data[k-1] = np.loadtxt(os.path.join("eig_face_weight", f))
            test_label[k-1] = k-1

    # data = np.zeros([400, 256])
    # label = np.zeros(400, dtype=np.uint8)
    # for f in file_name:
    #     cla_ord = f.split(".")[0]
    #     k, n = cla_ord.split("_")
    #     k = int(k)
    #     n = int(n)
    #     if n < 10:
    #         data[(k - 1) * 10 + (n - 1)] = np.loadtxt(os.path.join("eig_face_weight", f))
    #         label[(k - 1) * 10 + (n - 1)] = k
    #     else:
    #         data[k * 10 - 1] = np.loadtxt(os.path.join("eig_face_weight", f))
    #         label[k * 10 - 1] = k
    # train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1, random_state=520)

    # print(train_data)
    # print(train_label)
    # print(test_data)
    # print(np.sort(test_label))
    return train_data, train_label, test_data, test_label


if __name__ == '__main__':

    cla = 40    # 40类人脸
    train_data, _, test_data, _ = get_data()  # 不需要标签
    means = []
    for i in range(cla):
        temp = train_data[i * 9: i * 9 + 9]
        means.append(temp.sum(axis=0) / 9)

    # 根据与mean最短距离判断测试数据是哪类
    acc = 0
    for i in range(cla):
        text_x = test_data[i]
        min_dis = 10000
        text_y = -1
        for j in range(cla):
            diff = text_x - means[j]
            dis = np.sqrt(diff.dot(diff))
            if dis < min_dis:
                min_dis = dis
                text_y = j
        if text_y == i:
            acc += 1
        print("Class %d predict for %d" % (i, text_y))
    print("Acc:{:.2%}".format(acc / cla))

