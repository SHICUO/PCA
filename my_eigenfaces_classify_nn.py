# coding:utf-8
# @Time:2022/7/13 14:08
# @Author:LHT
# @File:my_eigenfaces_classify_nn
# @GitHub:https://github.com/SHICUO
# @Contact:lin1042528352@163.com
# @Software:PyCharm

import os
import matplotlib.pyplot as plt
from my_eigenfaces_classify_dis import get_data
import time
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    def __init__(self, input_features: int, num_classes: int = 40):
        super(Network, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    sta_time = time.time()
    cla = 40  # 40类人脸
    vis_model = False

    # 读取数据并打乱
    train_data, train_label, test_data, test_label = get_data()
    train_data = torch.from_numpy(train_data).float().to(device)
    train_label = torch.from_numpy(train_label).long().to(device)

    # 加载模型
    nn_model = Network(input_features=256)
    if vis_model:
        from torchsummary import summary
        summary(nn_model, (1, 1, 256), device='cpu')  # input_size=chw
    nn_model.to(device)

    # 最优：b360,lr0.001,用Adam,全连接中间参数4096,最高test_acc97.5%,time16s
    max_epoch = 400
    batch_size = 360    # 8 24 40 72 120 360
    learning_rate = 0.001   # 0.0001
    lr_decay_step = 1
    vis_batch = 1
    # 设置交叉熵损失
    loss_func = nn.CrossEntropyLoss()
    # 设置优化器
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate, momentum=0.9)
    # 设置学习率下降策略  当损失值近乎不变时（一个阈值内1-thr），学习率才下降
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    # train
    Iteration = int(360 / batch_size)    # 每batch迭代次数

    train_curve = []
    nn_model.train()
    for epoch in range(max_epoch):
        shuff_index = torch.randperm(360)
        train_x = train_data[shuff_index]
        train_y = train_label[shuff_index]

        loss_mean = 0.
        correct = 0.
        total = 0.
        val_loss = torch.tensor(0)  # 无验证集，此损失值设置为5个batch的loss平均值
        for batch_idx in range(Iteration):
            batch_x = train_x[batch_idx*batch_size: (batch_idx+1)*batch_size, :]
            batch_y = train_y[batch_idx*batch_size: (batch_idx+1)*batch_size]
            pre_y = nn_model(batch_x)   # (8,40)

            optimizer.zero_grad()
            loss = loss_func(pre_y, batch_y)
            loss.backward()
            optimizer.step()    # 更新权重

            # 统计loss值，每5个迭代显示一次
            total += batch_y.size(0)
            _, predicted = torch.max(pre_y.data, dim=1)  # 按行 (8, 1)
            correct += (predicted == batch_y).cpu().sum().numpy()

            train_curve.append(loss.item())   # Iteration*400个损失值
            loss_mean += loss.item()
            if (batch_idx+1) % vis_batch == 0:
                loss_mean = loss_mean/vis_batch
                val_loss = loss_mean
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>2}/{:0>2}] Loss:{:.8f} Acc:{:.2%}".format(
                    epoch+1, max_epoch, batch_idx+1, Iteration, loss_mean, correct / total))
                loss_mean = 0.

        scheduler.step(val_loss)        # 更新学习率

    curve_x = range(len(train_curve))
    curve_y = train_curve
    plt.plot(curve_x, curve_y, label='Train')
    plt.title("batch_size:%d" % batch_size)
    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()

    test_x = torch.from_numpy(test_data).float().to(device)
    test_y = torch.from_numpy(test_label).long().to(device)
    test_y_pre = nn_model(test_x)
    test_y_pre = test_y_pre.softmax(dim=1)
    _, test_y_pre = torch.max(test_y_pre, 1)
    print(test_y_pre)
    acc = (test_y_pre == test_y).cpu().sum().numpy()
    print("Test set Acc:{:.2%}".format(acc / cla))

    end_time = time.time()
    print("Cost of time:{:.5f}s".format(end_time - sta_time))
