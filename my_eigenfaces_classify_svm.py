# coding:utf-8
# @Time:2022/7/13 13:50
# @Author:LHT
# @File:my_eigenfaces_classify_svm
# @GitHub:https://github.com/SHICUO
# @Contact:lin1042528352@163.com
# @Software:PyCharm

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from my_eigenfaces_classify_dis import get_data
from sklearn.svm import SVC
import time


if __name__ == '__main__':
    sta_time = time.time()
    cla = 40  # 40类人脸
    train_data, train_label, test_data, test_label = get_data()
    # end_time = time.time()

    # 使用向量机预测
    model = SVC()
    model.fit(train_data, train_label)

    pre_labels = model.predict(test_data)
    acc = 0
    for i, pre_label in enumerate(pre_labels):
        if pre_label == i:
            acc += 1
        print("Class %d predict for %d" % (i+1, pre_label+1))
    print("Acc:{:.2%}".format(acc / cla))

    end_time = time.time()
    print("Cost of time:{:.5f}s".format(end_time - sta_time))
