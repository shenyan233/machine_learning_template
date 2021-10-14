import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, config=None):
        super().__init__()
        # TODO 使用k折交叉验证
        # divide_dataset(config['dataset_path'], [0.8, 0, 0.2])
        if config['flag']:
            self.x = torch.randn(config['dataset_len'], 2)
            noise = torch.randn(config['dataset_len'], )
            self.y = 1.0 * self.x[:, 0] + 2.0 * self.x[:, 1] + noise
        else:
            x_1 = torch.randn(config['dataset_len'])
            x_2 = torch.randn(config['dataset_len'])
            x_useful = torch.cos(1.5 * x_1) * (x_2 ** 2)
            x_1_rest_small = torch.randn(config['dataset_len'], 15) + 0.01 * x_1.unsqueeze(1)
            x_1_rest_large = torch.randn(config['dataset_len'], 15) + 0.1 * x_1.unsqueeze(1)
            x_2_rest_small = torch.randn(config['dataset_len'], 15) + 0.01 * x_2.unsqueeze(1)
            x_2_rest_large = torch.randn(config['dataset_len'], 15) + 0.1 * x_2.unsqueeze(1)
            self.x = torch.cat([x_1[:, None], x_2[:, None], x_1_rest_small, x_1_rest_large, x_2_rest_small, x_2_rest_large],
                          dim=1)
            self.y = (10 * x_useful) + 5 * torch.randn(config['dataset_len'])

        self.y_train, self.y_test = self.y[:50000], self.y[50000:]
        self.x_train, self.x_test = self.x[:50000, :], self.x[50000:, :]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomDataset(self.x_train, self.y_train, self.config)
            self.val_dataset = CustomDataset(self.x_test, self.y_test, self.config)
        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset(self.x, self.y, self.config)

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
