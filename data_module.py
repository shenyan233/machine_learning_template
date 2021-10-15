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
        dataset_list = self.get_dataset_list()
        if stage == 'fit' or stage is None:
            dataset_train, dataset_val = self.get_dataset_lists(dataset_list)
            self.train_dataset = CustomDataset(dataset_train, self.config)
            self.val_dataset = CustomDataset(dataset_val, self.config)
        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset(dataset_list, self.config)

    def get_dataset_list(self):
        if not os.path.exists(self.dataset_path + '/dataset_list.txt'):
            # 针对数据拟合获得dataset
            dataset = torch.randn(self.config['dataset_len'], self.config['dim_in'] + 1)
            noise = torch.randn(self.config['dataset_len'])
            dataset[:, self.config['dim_in']] = torch.cos(1.5 * dataset[:, 0]) * (dataset[:, 1] ** 2.0) + torch.cos(
                torch.sin(dataset[:, 2] ** 3)) + torch.arctan(dataset[:, 4]) + noise
            assert (dataset[torch.isnan(dataset)].shape[0] == 0)

            with open(self.dataset_path + '/dataset_list.txt', 'w', encoding='utf-8') as f:
                for line in range(self.config['dataset_len']):
                    f.write(' '.join([str(temp) for temp in dataset[line].tolist()]) + '\n')
            print('已生成新的数据list')
        else:
            dataset_list = open(self.dataset_path + '/dataset_list.txt').readlines()
            # 针对数据拟合获得dataset
            dataset_list = [[float(temp) for temp in item.strip('\n').split(' ')] for item in dataset_list]
            dataset = torch.Tensor(dataset_list).float()

        return dataset

    def get_dataset_lists(self, dataset_list: Tensor):
        # 得到一个fold的数据量和不够组成一个fold的剩余数据的数据量
        num_1fold, remainder = divmod(self.config['dataset_len'], self.k_fold)
        # 分割全部数据, 得到训练集, 验证集, 测试集
        dataset_val = dataset_list[num_1fold * self.kth_fold:(num_1fold * (self.kth_fold + 1) + remainder), :]
        temp = torch.ones(dataset_list.shape[0])
        temp[num_1fold * self.kth_fold:(num_1fold * (self.kth_fold + 1) + remainder)] = 0
        dataset_train = dataset_list[temp == 1]
        return dataset_train, dataset_val

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
    def __init__(self, dataset, config):
        super().__init__()
        self.x = dataset[:, 0:config['dim_in']]
        self.y = dataset[:, config['dim_in']]

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]

    def __len__(self):
        return self.x.shape[0]
