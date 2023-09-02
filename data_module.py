# get_fit_dataset_lists or get_dataset_list must be overridden
# get_test_dataset_lists must be overridden
import importlib
import os
import random
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from global_param import datasets_path


class DataModule(pl.LightningDataModule):
    def __init__(self, num_workers, config):
        super().__init__()
        self.num_workers = num_workers
        self.config = config
        self.dataset_path = datasets_path + '/' + config['dataset_name']
        if config['accelerator'] == 'cpu':
            self.pin_memory = False
        else:
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
            dataset_train, dataset_val = self.get_fit_dataset_lists(self.dataset_path, self.config)
            self.train_dataset = self.custom_dataset(dataset_train, 'train', self.config, self.dataset_path)
            self.val_dataset = self.custom_dataset(dataset_val, 'val', self.config, self.dataset_path)
        if stage == 'test' or stage is None:
            dataset_test = self.get_test_dataset_lists(self.dataset_path)
            self.test_dataset = self.custom_dataset(dataset_test, 'test', self.config, self.dataset_path)

    def get_fit_dataset_lists(self, dataset_path, config):
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
        """
        Since the loss method used by PL to calculate and verify epoch calculated the mean value of the loss of
        each batch, the mean value would also be calculated when multiple samples were calculated within each batch,
        which led to the mean value calculation twice (this problem would also occur in official CELOSS). If the sizes
        of two batches are different at this time, the weight of each loss is not equal, which leads to the error of
        obtaining Loss. This situation often occurs, because the amount of data cannot be divided exactly by size, so
        basically the size of the last batch is different from the size of the previous batch.
        In order to ensure the accuracy of verification set calculation of Loss, batch size of verification set is
        redefined in this method. In addition, the backpropagation of the training set is not affected, but
        the loss record of the training set is. However, since batCH_size has a great influence on the training results,
        the impact of loss record is ignored and the training set Batch size is not redefined.
        由于pl计算验证epoch的loss的方法为每个batch的loss求均值, 而每个batch内计算多个样本时同样会求均值,
        这导致了两次求均值(官方的celoss也会出现该问题). 如果这时存在两个batch的size不同,则会导致每个loss
        的权重不相等, 导致求loss的错误. 这种情况常常出现, 因为数据量不能整除size, 所以基本上最后一个batch
        的size与前面的batch的size不同.
        为了保证验证集计算loss的准确性, 该方法中对于验证集的batch size进行了重新定义.
        此外, 训练集的反向传播不受影响, 但训练集的loss记录会受影响. 然而, 由于batch_size对训练结果具有较大
        的影响, 因此, 忽略loss记录的影响, 不对训练集batch size进行重新定义.
        """
        val_batch_size = 1
        for num in range(self.config['batch_size']):
            if (len(self.val_dataset) % (self.config['batch_size'] - num)) == 0:
                val_batch_size = self.config['batch_size'] - num
                break
        return DataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        test_batch_size = 1
        for num in range(self.config['batch_size']):
            if (len(self.test_dataset) % (self.config['batch_size'] - num)) == 0:
                test_batch_size = self.config['batch_size'] - num
                break
        return DataLoader(self.test_dataset, batch_size=test_batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)