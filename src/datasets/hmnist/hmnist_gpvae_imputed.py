import torch
import pytorch_lightning as pl
from hydra.utils import to_absolute_path

from torch.utils.data import DataLoader, TensorDataset
from typing import Optional


class HealingMNISTImputedGPVAE(pl.LightningDataModule):

    def __init__(self,
                 x_path: str,
                 y_path: str,
                 batch_size: int,
                 num_workers: int,
                 is_test_only: bool,
                 test_split: int):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: Optional[str] = None):
        X = torch.load(to_absolute_path(self.hparams.x_path))
        y = torch.load(to_absolute_path(self.hparams.y_path))

        X = torch.concat(X)
        y = torch.concat(y)

        if self.hparams.is_test_only:
            X = X[60_000:]
            y = y[60_000:]
            X_train, y_train = X[:self.hparams.test_split], y[:self.hparams.test_split]
            X_test, y_test = X[self.hparams.test_split:], y[self.hparams.test_split:]
        else:
            X_train, y_train = X[:60_000], y[:60_000]
            X_test, y_test = X[60_000 + self.hparams.test_split:], y[60_000 + self.hparams.test_split:]

        self.train_dataset = TensorDataset(X_train, y_train)
        self.test_dataset = TensorDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True)
