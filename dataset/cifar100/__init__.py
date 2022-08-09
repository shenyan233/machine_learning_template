import glob
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def get_fit_dataset_lists(dataset_path):
    dataset_train = glob.glob(dataset_path + '/train/image/*.png')
    dataset_val = glob.glob(dataset_path + '/test/image/*.png')
    return dataset_train, dataset_val


def get_test_dataset_lists(dataset_path):
    return glob.glob(dataset_path + '/test/image/*.png')


class CustomDataset(Dataset):
    def __init__(self, dataset, stage, config, dataset_path):
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
        dataset_path = './dataset/' + config['dataset_name']
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
