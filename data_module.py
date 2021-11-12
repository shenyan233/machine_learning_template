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
        # k_fold_dataset_list = self.get_k_fold_dataset_list()
        if stage == 'fit' or stage is None:
            # dataset_train, dataset_val = self.get_fit_dataset_lists(k_fold_dataset_list)
            dataset_train = glob.glob(self.dataset_path + '/train/image/*.png')
            dataset_val = glob.glob(self.dataset_path + '/test/image/*.png')
            self.train_dataset = CustomDataset(self.dataset_path, dataset_train,  'train', self.config,)
            self.val_dataset = CustomDataset(self.dataset_path, dataset_val,  'val', self.config,)
        if stage == 'test' or stage is None:
            dataset_test = glob.glob(self.dataset_path + '/test/image/*.png')
            self.test_dataset = CustomDataset(self.dataset_path, dataset_test, 'test', self.config,)

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
        """
        由于pl计算验证epoch的loss的方法为每个batch的loss求均值, 而每个batch内计算多个样本时同样会求均值,
        这导致了两次求均值(官方的celoss也会出现该问题). 如果这时存在两个batch的size不同,则会导致每个loss
        的权重不相等, 导致求loss的错误. 这种情况常常出现, 因为数据量不能整除size, 所以基本上最后一个batch
        的size与前面的batch的size不同.
        为了保证验证集计算loss的准确性, 该方法中对于验证集的batch size进行了重新定义.
        此外, 训练集的反向传播不受影响, 但训练集的loss记录会受影响. 然而, 由于batch_size对训练结果具有较大
        的影响, 因此, 忽略loss记录的影响, 不对训练集batch size进行重新定义.
        """
        val_batch_size = 1
        for num in range(self.batch_size):
            if len(self.val_dataset) % (self.batch_size - num) == 0:
                val_batch_size = self.batch_size - num
                break
        return DataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)


class CustomDataset(Dataset):
    def __init__(self, dataset_path, dataset, stage, config, ):
        super().__init__()
        self.dataset = dataset
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
        # 注意: 为了满足初始化权重算法的要求, 需要输入参数的均值为0. 可以使用transforms.Normalize()
        image_path = self.dataset[idx]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        image = self.trans(image)
        label = torch.Tensor([int(self.labels[int(image_name.strip('.png'))].strip('\n'))])
        return image_name, image, label.long()

    def __len__(self):
        return len(self.dataset)
