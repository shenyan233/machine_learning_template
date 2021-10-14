import os
import numpy
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, k_fold, kth_fold, dataset_path, config=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config
        self.k_fold = k_fold
        self.kth_fold = kth_fold
        self.dataset_path = dataset_path

    def setup(self, stage=None) -> None:
        # 得到全部数据的list
        # dataset_list = get_dataset_list(dataset_path)
        x, y = self.get_fit_dataset_list()
        if stage == 'fit' or stage is None:
            x_train, y_train, x_val, y_val = self.get_dataset_lists(x, y)
            self.train_dataset = CustomDataset(x_train, y_train, self.config)
            self.val_dataset = CustomDataset(x_val, y_val, self.config)
        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset(x, y, self.config)

    def get_fit_dataset_list(self):
        if not os.path.exists(self.dataset_path + '/dataset_list.txt'):
            x = torch.randn(self.config['dataset_len'], self.config['dim_in'])
            noise = torch.randn(self.config['dataset_len'])
            y = torch.cos(1.5 * x[:, 0]) * (x[:, 1] ** 2.0) + noise
            with open(self.dataset_path + '/dataset_list.txt', 'w', encoding='utf-8') as f:
                for line in range(self.config['dataset_len']):
                    f.write(' '.join([str(temp) for temp in x[line].tolist()]) + ' ' + str(y[line].item()) + '\n')
            print('已生成新的数据list')
        else:
            dataset_list = open(self.dataset_path + '/dataset_list.txt').readlines()
            dataset_list = [[float(temp) for temp in item.strip('\n').split(' ')] for item in dataset_list]
            x = torch.from_numpy(numpy.array(dataset_list)[:, 0:self.config['dim_in']]).float()
            y = torch.from_numpy(numpy.array(dataset_list)[:, self.config['dim_in']]).float()
        return x, y

    def get_dataset_lists(self, x: Tensor, y):
        # 得到一个fold的数据量和不够组成一个fold的剩余数据的数据量
        num_1fold, remainder = divmod(self.config['dataset_len'], self.k_fold)
        # 分割全部数据, 得到训练集, 验证集, 测试集
        x_val = x[num_1fold * self.kth_fold:(num_1fold * (self.kth_fold + 1) + remainder)]
        y_val = y[num_1fold * self.kth_fold:(num_1fold * (self.kth_fold + 1) + remainder)]
        temp = torch.ones(x.shape[0])
        temp[num_1fold * self.kth_fold:(num_1fold * (self.kth_fold + 1) + remainder)] = 0
        x_train = x[temp == 1]
        y_train = y[temp == 1]
        return x_train, y_train, x_val, y_val

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)


class CustomDataset(Dataset):
    def __init__(self, x, y, config):
        super().__init__()
        self.x = x
        self.y = y
        self.config = config

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]

    def __len__(self):
        return self.x.shape[0]
