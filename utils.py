import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from datasets import TS_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        # 可以选择"w"
        self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal


def load_UCR_data(datadir, dataset):
    train_data = np.loadtxt(datadir + dataset + '/' + dataset + '_TRAIN.txt')
    test_data = np.loadtxt(datadir + dataset + '/' + dataset + '_TEST.txt')

    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
    print('load UCR data', X_train.shape, X_test.shape)
    return (X_train, y_train, X_test, y_test)


def partition_tsdata(dataset, datadir, partition, n_nets, alpha):
    # partition_strategy = "homo"
    # partition_strategy = "hetero-dir"
    print('---------------load UCR daset-------------------')
    X_train, y_train, X_test, y_test = load_UCR_data(datadir, dataset)
    n_train = X_train.shape[0]
    if partition == "homo":
        idxs = np.random.permutation(n_train)  # 随机排序
        batch_idxs = np.array_split(idxs, n_nets)  # 把idxs分为 nnets份
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}  # 用户id->数据的字典
    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        K = len(np.unique(y_train))
        N = y_train.shape[0]
        net_dataidx_map = {}
        while (min_size < 1) or (dataset == 'mnist' and min_size < 100):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]  # 取出10个类分别对应的下标集合
                np.random.shuffle(idx_k)  # 打乱下标，重复alphanets次
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))  # 地雷克雷分布
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()  # 归一化
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # 数据采用地雷克雷分布分配给用户
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    # fanhui
    return net_dataidx_map


def get_ts_loader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = TS_truncated

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True)
    test_ds = dl_obj(datadir, train=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    # print('get ts loader length: ', len(train_ds), len(test_ds))

    return train_dl, test_dl


def seed_experiment(seed=0):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seeded everything %d", seed)

