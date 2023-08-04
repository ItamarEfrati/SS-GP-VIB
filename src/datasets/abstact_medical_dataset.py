import torch
import numpy as np
from abc import ABC, abstractmethod

from sklearn.preprocessing import StandardScaler


class MedicalDataset(ABC):

    @abstractmethod
    def read_files(self):
        pass

    @abstractmethod
    def get_users_split(self):
        pass

    @staticmethod
    def _scale_data(data_df, train_users, val_users, test_users):
        ss = StandardScaler()
        train_df = data_df.loc[train_users].copy()
        X_train = ss.fit_transform(train_df.values)
        X_train = np.nan_to_num(X_train, nan=0)
        train_df[:] = X_train

        val_df = data_df.loc[val_users].copy()
        X_val = ss.transform(val_df.values)
        X_val = np.nan_to_num(X_val, nan=0)
        val_df[:] = X_val

        test_df = data_df.loc[test_users].copy()
        X_test = ss.transform(test_df.values)
        X_test = np.nan_to_num(X_test, nan=0)
        test_df[:] = X_test

        return train_df, val_df, test_df

    @staticmethod
    def _pandas_to_tensors(df, users):
        data = df.groupby('record_id') \
            .apply(lambda g: torch.tensor(g.values, dtype=torch.float)) \
            .loc[users].values

        return torch.stack(list(data))
