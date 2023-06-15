# get_fit_dataset_lists or get_dataset_list must be overridden
# get_test_dataset_lists must be overridden
datasets_path = '/home/shen/dataset'
import importlib
import os
import random
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, num_workers, config):
        super().__init__()
        self.num_workers = num_workers
        self.config = config
        self.dataset_path = datasets_path + '/' + config['dataset_name']
        self.pin_memory = True

        if config['is_check']:
            imported = importlib.import_module('dataset.check')
        else:
            imported = importlib.import_module('dataset.%(dataset_name)s' % config)
        self.custom_dataset = imported.CustomDataset

        assert 'get_dataset_list' in dir(imported) or 'get_fit_dataset_lists' in dir(imported)
        if 'get_dataset_list' in dir(imported):
            self.get_dataset_list = imported.get_dataset_list
        if 'get_fit_dataset_lists' in dir(imported):
            self.get_fit_dataset_lists = imported.get_fit_dataset_lists
        assert 'get_test_dataset_lists' in dir(imported)
        self.get_test_dataset_lists = imported.get_test_dataset_lists

    def setup(self, stage=None) -> None:
        if stage == 'fit' or stage is None:
            dataset_train, dataset_val = self.get_fit_dataset_lists(self.dataset_path)
            self.train_dataset = self.custom_dataset(dataset_train, 'train', self.config, self.dataset_path)
            self.val_dataset = self.custom_dataset(dataset_val, 'val', self.config, self.dataset_path)
        if stage == 'test' or stage is None:
            dataset_test = self.get_test_dataset_lists(self.dataset_path)
            self.test_dataset = self.custom_dataset(dataset_test, 'test', self.config, self.dataset_path)

    def get_fit_dataset_lists(self, dataset_path):
        k_fold_dataset_list = self.get_k_fold_dataset_list(dataset_path)
        dataset_train, dataset_val = self.divide_fit_dataset_lists(k_fold_dataset_list)
        return dataset_train, dataset_val

    def get_k_fold_dataset_list(self, dataset_path):
        # Get the list of data used for K folding and segmentation, and generate TXT file for saving
        # 得到用于K折分割的数据的list, 并生成txt文件进行保存
        if not os.path.exists(dataset_path + '/k_fold_dataset_list.txt'):
            # Get the list of data used for k-fold splitting
            # 获得用于k折分割的数据的list
            dataset = self.get_dataset_list(dataset_path)
            random.shuffle(dataset)
            written = dataset

            with open(dataset_path + '/k_fold_dataset_list.txt', 'w', encoding='utf-8') as f:
                for line in written:
                    f.write(line.replace('\\', '/') + '\n')
            print('A new list of k-fold data has been generated|已生成新的k折数据list')
        else:
            dataset = open(dataset_path + '/k_fold_dataset_list.txt').readlines()
            dataset = [item.strip('\n') for item in dataset]
        return dataset

    def divide_fit_dataset_lists(self, dataset_list: list):
        # The amount of data that makes up a fold and the amount of data that is not enough to make up a fold
        # 得到一个fold的数据量和不够组成一个fold的剩余数据的数据量
        num_1fold, remainder = divmod(len(dataset_list), self.config['k_fold'])
        # The training set, validation set and test set are obtained by dividing all the data
        # 分割全部数据, 得到训练集, 验证集, 测试集
        dataset_val = dataset_list[
                      num_1fold * self.config['kth_fold']:(num_1fold * (self.config['kth_fold'] + 1) + remainder)]
        del (dataset_list[num_1fold * self.config['kth_fold']:(num_1fold * (self.config['kth_fold'] + 1) + remainder)])
        dataset_train = dataset_list
        return dataset_train, dataset_val

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'], shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
