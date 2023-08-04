import os
import pathlib
import zipfile

import pytorch_lightning as pl
import pandas as pd
import numpy as np
import requests
import torch
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from hydra.utils import to_absolute_path

from typing import Optional, List


class Physionet2019(pl.LightningDataModule):
    time_feature = 'ICULOS'
    vital_features = [
        'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    lab_features = [
        'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
        'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
        'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
        'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
        'WBC', 'Fibrinogen', 'Platelets'
    ]
    label = 'SepsisLabel'
    ts_features = [time_feature] + vital_features + lab_features

    def __init__(self,
                 download_dir: str,
                 batch_size: int,
                 num_workers: int,
                 data_urls: List[str],
                 file_name: str,
                 label: str,
                 is_test: bool):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # region Private

    @staticmethod
    def _get_time_series_data(data):
        if any(data[['Time', 'Parameter']].duplicated()):
            data = data.groupby(['Time', 'Parameter'], as_index=False).mean().reset_index()
        time_series = data.pivot(index='Time', columns='Parameter', values='Value')
        time_series = time_series.reindex(range(48))
        return time_series

    @staticmethod
    def _get_txt_files_contents(local_file):
        contents = {}
        with zipfile.ZipFile(local_file, "r") as f:
            for name in list(filter(lambda x: x.endswith('psv'), list(f.namelist()))):
                contents[name] = pd.read_csv(f.open(name), sep='|', header=0)
        return contents

    def _handle_single_participant_data(self, record_id, record_df):
        time_series_data = record_df[self.ts_features + [self.label]].copy()
        time_series_data['record_id'] = int(record_id.split('/')[1].split('.')[0][1:])
        time_series_data.set_index(['record_id', self.time_feature], inplace=True)

        # ts_data_mask = record_df['Parameter'].isin(self.ts_features)
        # time_series_data = record_df[ts_data_mask]
        # time_series_data = self._get_time_series_data(time_series_data)
        # time_series_data['record_id'] = int(record_id)
        # time_series_data = time_series_data.set_index('record_id', append=True).swaplevel(0, 1)
        # time_series_data = time_series_data.reset_index('Time')
        return time_series_data

    def _download_files(self, urls):
        local_files = []
        for url in urls:
            file_name = url.split('/')[-1].split('?')[0]
            local_file_path = os.path.join(to_absolute_path(self.hparams.download_dir), file_name)
            local_files.append(local_file_path)
            if os.path.exists(local_file_path):
                continue
            headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
            response = requests.get(url, stream=True, headers=headers)
            content_length = int(response.headers['Content-Length'])
            pbar = tqdm(total=content_length)
            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=max(20_000_000, content_length)):
                    if chunk:
                        f.write(chunk)
                    pbar.update(len(chunk))
        return local_files

    def _get_sets_data(self, local_files):
        sets_list = []
        set_df_list = []
        for local_file in local_files:
            records_list = []
            current_set = pathlib.Path(local_file).stem.split('.')[0]
            sets_list.append(current_set)
            set_contents = self._get_txt_files_contents(local_file)
            for record_id, record_df in set_contents.items():
                records_list.append(self._handle_single_participant_data(record_id, record_df))

            set_df_list.append(pd.concat(records_list))
        return set_df_list, sets_list

    def _get_users_split(self):
        splits = {}
        data_dir = to_absolute_path(self.hparams.download_dir)
        for split in ['train', 'test', 'val']:
            with open(os.path.join(data_dir, f'{split}_records.txt'), 'r') as f:
                splits[split] = list(map(lambda x: int(x.strip()), f.readlines()))
        if self.hparams.is_test:
            splits['train'] += splits['val']
        else:
            splits['test'] = splits['val']

        return splits['train'], splits['test']

    # endregion

    def prepare_data(self):
        if not os.path.exists(to_absolute_path(self.hparams.download_dir)):
            os.makedirs(to_absolute_path(self.hparams.download_dir))
        file_path = os.path.join(to_absolute_path(self.hparams.download_dir), self.hparams.file_name)

        is_data_already_exists = os.path.exists(file_path)
        if is_data_already_exists:
            return

        data_local_files = self._download_files(self.hparams.data_urls)

        set_df_list, sets_list = self._get_sets_data(data_local_files)

        data_df = pd.concat(set_df_list, keys=sets_list, names=['set', 'record_id', self.time_feature])
        data_df.to_csv(file_path)

    def setup(self, stage: Optional[str] = None):
        file_path = os.path.join(to_absolute_path(self.hparams.download_dir), self.hparams.file_name)
        data_df = pd.read_csv(file_path,
                              usecols=self.ts_features + ['record_id', self.hparams.label],
                              index_col='record_id')

        ts_data = data_df[self.ts_features]
        y = data_df[self.hparams.label].groupby('record_id').first()

        train_users, test_users = self._get_users_split()

        ss = StandardScaler()
        X_train = ss.fit_transform(ts_data.loc[train_users]).reshape(-1, 48, 37)
        X_train = np.nan_to_num(X_train, nan=0)
        X_test = ss.transform(ts_data.loc[test_users]).reshape(-1, 48, 37)
        X_test = np.nan_to_num(X_test, nan=0)
        y_train, y_test = y.loc[train_users], y.loc[test_users]
        train_tensors = [torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train.values, dtype=torch.int)]
        test_tensors = [torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test.values, dtype=torch.int)]

        self.train_dataset = TensorDataset(*train_tensors)
        self.test_dataset = TensorDataset(*test_tensors)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True)


if __name__ == '__main__':
    download_dir = r"data\\physionet_2019"
    data_urls = ['https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',
                 'https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip']

    file_name = 'time_series_48.csv'
    batch_size = 64
    num_workers = 4

    p = Physionet2019(download_dir=download_dir, data_urls=data_urls,
                      file_name=file_name, batch_size=batch_size, num_workers=num_workers, label='In-hospital_death',
                      is_test=True)

    p.prepare_data()
