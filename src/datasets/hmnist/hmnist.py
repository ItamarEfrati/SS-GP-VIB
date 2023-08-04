import os

import pytorch_lightning as pl
import numpy as np
import requests

from typing import Optional
from tqdm import tqdm
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset

from hydra.utils import to_absolute_path


class HealingMNIST(pl.LightningDataModule):

    def __init__(self,
                 download_dir: str,
                 download_url: str,
                 file_name: str,
                 batch_size: int,
                 num_workers: int,
                 val_split: int,
                 test_split: int,
                 is_test_only: bool,
                 is_data_missing: bool):
        """

        :param download_dir: the directory to save the data
        :param download_url: from where to download the data
        :param file_name: the name of the downloaded file
        :param batch_size:
        :param num_workers:
        :param val_split: the split between validation and train
        :param test_split: the split between train and test data
        :param is_test_only: take only the test data for train an test (performed on gp-vae)
        :param is_data_missing: take full data or data with missing values
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if not os.path.exists(to_absolute_path(self.hparams.download_dir)):
            os.makedirs(to_absolute_path(self.hparams.download_dir))
        file_path = os.path.join(to_absolute_path(self.hparams.download_dir), self.hparams.file_name)
        if not os.path.exists(file_path):
            headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
            response = requests.get(self.hparams.download_url, stream=True, headers=headers)
            content_length = int(response.headers['Content-Length'])
            pbar = tqdm(total=content_length)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=20_000_000):
                    if chunk:
                        f.write(chunk)
                    pbar.update(len(chunk))

    def setup(self, stage: Optional[str] = None):
        file_path = os.path.join(to_absolute_path(self.hparams.download_dir), self.hparams.file_name)
        data = np.load(file_path)

        # full data
        # missing and mask, 1 is an evidence for missing data, True is observe

        data_name_format = "x_{}_miss" if self.hparams.is_data_missing else 'x_{}_full'

        if self.hparams.is_test_only:
            X = tensor(data[data_name_format.format('test')])
            train_tensors = [X[:self.hparams.test_split]]
            val_tensors = [X[:self.hparams.test_split]]
            test_tensors = [X[self.hparams.test_split:]]

            labels = tensor(data['y_test'])
            train_tensors.append(labels[:self.hparams.test_split])
            val_tensors.append(labels[:self.hparams.test_split])
            test_tensors.append(labels[self.hparams.test_split:])
        else:
            train_tensors = [tensor(data[data_name_format.format('train')])[:self.hparams.val_split]]
            val_tensors = [tensor(data[data_name_format.format('train')])[self.hparams.val_split:]]
            test_tensors = [tensor(data[data_name_format.format('test')])[self.hparams.test_split:]]

            train_tensors.append(tensor(data['y_train'])[:self.hparams.val_split])
            val_tensors.append(tensor(data['y_train'])[self.hparams.val_split:])
            test_tensors.append(tensor(data['y_test'])[self.hparams.test_split:])

        self.train_dataset = TensorDataset(*train_tensors)
        self.val_dataset = TensorDataset(*val_tensors)
        self.test_dataset = TensorDataset(*test_tensors)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=False, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=False)
