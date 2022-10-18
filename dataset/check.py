import torch
from torch.utils.data import Dataset


def get_fit_dataset_lists(dataset_path):
    dataset_train = [0]
    dataset_val = [0]
    return dataset_train, dataset_val


def get_test_dataset_lists(dataset_path):
    return [0]


class CustomDataset(Dataset):
    def __init__(self, dataset, stage, config, dataset_path):
        super().__init__()
        self.config = config

    def __getitem__(self, idx):
        input = torch.rand((3, self.config['dim_in'], self.config['dim_in']))
        label = (torch.rand(1)>0.5).int()
        return 'check', input, label.squeeze().long()

    def __len__(self):
        return 2000
