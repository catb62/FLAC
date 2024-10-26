import numpy as np
import torch

class TS_truncated():

    def __init__(self, root, dataidxs=None, train=True):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        if self.train:
            ts_dataobj = self.__load_ucr__(self.root + '_TRAIN.txt', normalize=False)
            ts_dataobj = torch.tensor(ts_dataobj, dtype=torch.float)
            data = ts_dataobj[:, 1:]
            target = ts_dataobj[:, 0]
        else:
            ts_dataobj = self.__load_ucr__(self.root + '_TEST.txt', normalize=False)
            ts_dataobj = torch.tensor(ts_dataobj, dtype=torch.float)
            data = ts_dataobj[:, 1:]
            target = ts_dataobj[:, 0]

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        ts, target = self.data[index], self.target[index]

        return ts, target

    def __len__(self):
        return len(self.data)

    def __load_ucr__(self, path, normalize=False):
        data = np.loadtxt(path)
        # nan
        mask = np.isnan(data)
        data[mask] = 0
        if normalize:
            mean = data[:, 1:].mean(axis=1, keepdims=True)
            std = data[:, 1:].std(axis=1, keepdims=True)
            data[:, 1:] = (data[:, 1:] - mean) / (std + 1e-8)
        return data
