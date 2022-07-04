import random
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import glob
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class DataModule(pl.LightningDataModule):
    def __init__(self, num_workers, config):
        super().__init__()
        self.num_workers = num_workers
        self.config = config
        self.dataset_path = './dataset/' + config['dataset_name']

    def setup(self, stage=None) -> None:
        if stage == 'fit' or stage is None:
            dataset_train, dataset_val = self.get_fit_dataset_lists(self.dataset_path)
            self.train_dataset = CustomDataset(self.dataset_path, dataset_train, 'train', self.config, )
            self.val_dataset = CustomDataset(self.dataset_path, dataset_val, 'val', self.config, )
        if stage == 'test' or stage is None:
            dataset_test = self.get_test_dataset_lists(self.dataset_path)
            self.test_dataset = CustomDataset(self.dataset_path, dataset_test, 'test', self.config, )

    def get_fit_dataset_lists(self, dataset_path):
        dataset_train = glob.glob(dataset_path + '/train/image/*.png')
        dataset_val = glob.glob(dataset_path + '/test/image/*.png')
        return dataset_train, dataset_val

    def get_test_dataset_lists(self, dataset_path):
        return glob.glob(dataset_path + '/test/image/*.png')

    def get_dataset_list(self, dataset_path):
        return None

    def get_k_fold_dataset_list(self):
        # Get the list of data used for K folding and segmentation, and generate TXT file for saving
        # 得到用于K折分割的数据的list, 并生成txt文件进行保存
        if not os.path.exists(self.dataset_path + '/k_fold_dataset_list.txt'):
            # Get the list of data used for k-fold splitting
            # 获得用于k折分割的数据的list
            dataset = self.get_dataset_list(self.dataset_path)
            if dataset is None:
                return self.dataset_path
            random.shuffle(dataset)
            written = dataset

            with open(self.dataset_path + '/k_fold_dataset_list.txt', 'w', encoding='utf-8') as f:
                for line in written:
                    f.write(line.replace('\\', '/') + '\n')
            print('A new list of k-fold data has been generated|已生成新的k折数据list')
        else:
            dataset = open(self.dataset_path + '/k_fold_dataset_list.txt').readlines()
            dataset = [item.strip('\n') for item in dataset]
        return dataset

    def kfolds_divide_dataset_lists(self, dataset_list: list):
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
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'], shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)


class CustomDataset(Dataset):
    def __init__(self, dataset_path, dataset, stage, config, ):
        super().__init__()
        self.dataset = dataset
        # The mean and variance here are derived from ImageNet
        # 此处的均值和方差来源于ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if stage == 'train':
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(config['dim_in'], 4),
                transforms.ToTensor(),
                normalize, ])
        elif stage == 'val':
            stage = 'test'
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                normalize, ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                normalize, ])
        self.labels = open(dataset_path + '/' + stage + '/label.txt').readlines()

    def __getitem__(self, idx):
        # Note: In order to meet the requirements of the initial weight algorithm, the mean value of
        # the input parameters is required to be 0. Transforms.Normalize() can be used
        # 注意: 为了满足初始化权重算法的要求, 需要输入参数的均值为0. 可以使用transforms.Normalize()
        image_path = self.dataset[idx]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        image = self.trans(image)
        label = torch.Tensor([int(self.labels[int(image_name.strip('.png'))].strip('\n'))])
        return image_name, image, label.long().squeeze(dim=0)

    def __len__(self):
        return int(len(self.dataset))
