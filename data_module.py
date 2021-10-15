import glob
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms


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
        k_fold_dataset_list = self.get_k_fold_dataset_list()
        if stage == 'fit' or stage is None:
            dataset_train, dataset_val = self.get_fit_dataset_lists(k_fold_dataset_list)
            self.train_dataset = CustomDataset(self.dataset_path, dataset_train, self.config, 'train')
            self.val_dataset = CustomDataset(self.dataset_path, dataset_val, self.config, 'train')
        if stage == 'test' or stage is None:
            dataset_test = self.get_test_dataset_lists(k_fold_dataset_list)
            self.test_dataset = CustomDataset(self.dataset_path, dataset_test, self.config, 'test')

    def get_k_fold_dataset_list(self):
        # 得到用于K折分割的数据的list, 并生成文件夹进行保存
        if not os.path.exists(self.dataset_path + '/k_fold_dataset_list.txt'):
            # 获得用于k折分割的数据的list
            dataset = glob.glob(self.dataset_path + '/train/image/*.png')
            random.shuffle(dataset)
            written = dataset

            with open(self.dataset_path + '/k_fold_dataset_list.txt', 'w', encoding='utf-8') as f:
                for line in written:
                    f.write(line.replace('\\', '/') + '\n')
            print('已生成新的k折数据list')
        else:
            dataset = open(self.dataset_path + '/k_fold_dataset_list.txt').readlines()
            dataset = [item.strip('\n') for item in dataset]
        return dataset

    def get_fit_dataset_lists(self, dataset_list: list):
        # 得到一个fold的数据量和不够组成一个fold的剩余数据的数据量
        num_1fold, remainder = divmod(len(dataset_list), self.k_fold)
        # 分割全部数据, 得到训练集, 验证集, 测试集
        dataset_val = dataset_list[num_1fold * self.kth_fold:(num_1fold * (self.kth_fold + 1) + remainder)]
        del (dataset_list[num_1fold * self.kth_fold:(num_1fold * (self.kth_fold + 1) + remainder)])
        dataset_train = dataset_list
        return dataset_train, dataset_val

    def get_test_dataset_lists(self, dataset_list):
        dataset = glob.glob(self.dataset_path + '/test/image/*.png')
        return dataset

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
    def __init__(self, dataset_path, dataset, config, type):
        super().__init__()
        self.dataset = dataset
        self.trans = transforms.ToTensor()
        self.labels = open(dataset_path + '/' + type + '/label.txt').readlines()

    def __getitem__(self, idx):
        image_path = self.dataset[idx]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        image = self.trans(image)
        label = torch.Tensor([int(self.labels[int(image_name.strip('.png'))].strip('\n'))])
        return image_name, image, label.long()

    def __len__(self):
        return len(self.dataset)
