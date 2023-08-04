import math
import os
import pathlib
from datetime import datetime, timedelta
from glob import glob

import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from hydra.utils import to_absolute_path

from datasets.abstact_medical_dataset import MedicalDataset
from datasets.samplers import LoaderWrapper


class Mimic3(pl.LightningDataModule, MedicalDataset):
    # Blacklisted instances due to unusually many observations compared to the
    # overall distribution.
    blacklist = [
        # Criterion for exclusion: more than 1000 distinct timepoints
        # In training data
        '73129_episode2_timeseries.csv', '48123_episode2_timeseries.csv',
        '76151_episode2_timeseries.csv', '41493_episode1_timeseries.csv',
        '65565_episode1_timeseries.csv', '55205_episode1_timeseries.csv',
        '41861_episode1_timeseries.csv', '58242_episode4_timeseries.csv',
        '54073_episode1_timeseries.csv', '46156_episode1_timeseries.csv',
        '55639_episode1_timeseries.csv', '89840_episode1_timeseries.csv',
        '43459_episode1_timeseries.csv', '10694_episode2_timeseries.csv',
        '51078_episode2_timeseries.csv', '90776_episode1_timeseries.csv',
        '89223_episode1_timeseries.csv', '12831_episode2_timeseries.csv',
        '80536_episode1_timeseries.csv',
        # In validation data
        '78515_episode1_timeseries.csv', '62239_episode2_timeseries.csv',
        '58723_episode1_timeseries.csv', '40187_episode1_timeseries.csv',
        '79337_episode1_timeseries.csv', '60552_episode1_timeseries.csv',
        # In testing data
        '51177_episode1_timeseries.csv', '70698_episode1_timeseries.csv',
        '48935_episode1_timeseries.csv', '54353_episode2_timeseries.csv',
        '19223_episode2_timeseries.csv', '58854_episode1_timeseries.csv',
        '80345_episode1_timeseries.csv', '48380_episode1_timeseries.csv'
    ]
    demographics = ['Height']
    vitals = [
        'Weight', 'Heart Rate', 'Mean blood pressure',
        'Diastolic blood pressure', 'Systolic blood pressure',
        'Oxygen saturation', 'Respiratory rate'
    ]
    lab_measurements = [
        'Capillary refill rate', 'Glucose', 'pH', 'Temperature']

    interventions = [
        'Fraction inspired oxygen', 'Glascow coma scale eye opening',
        'Glascow coma scale motor response', 'Glascow coma scale total',
        'Glascow coma scale verbal response'
    ]
    coma_scale_eye_opening_replacements = {
        "1 No Response": 1,
        "None": 1,
        "2 To pain": 2,
        "To Pain": 2,
        "3 To speech": 3,
        "To Speech": 3,
        "4 Spontaneously": 4,
        "Spontaneously": 4,
    }
    coma_scale_motor_replacements = {
        "1 No Response": 1,
        "No response": 1,
        "2 Abnorm extensn": 2,
        "Abnormal extension": 2,
        "3 Abnorm flexion": 3,
        "Abnormal Flexion": 3,
        "4 Flex-withdraws": 4,
        "Flex-withdraws": 4,
        "5 Localizes Pain": 5,
        "Localizes Pain": 5,
        "6 Obeys Commands": 6,
        "Obeys Commands": 6
    }
    coma_scale_verbal_replacements = {
        "No Response-ETT": 0,
        "1.0 ET/Trach": 0,
        "1 No Response": 1,
        "No Response": 1,
        "2 Incomp sounds": 2,
        "Incomprehensible sounds": 2,
        "3 Inapprop words": 3,
        "Inappropriate Words": 3,
        "4 Confused": 4,
        "Confused": 4,
        "5 Oriented": 5,
        "Oriented": 5,
    }

    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int,
                 is_test: bool,
                 class_weight_file_name: str,
                 aggregation_minutes: int,
                 demo_file_name: str,
                 ts_file_name_format: str,
                 folder_format: str,
                 use_sample_weight: bool
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.sampler = None

    # region Private

    def _get_indices_time_range(self):
        last_interval = datetime(1900, 1, 1) + timedelta(minutes=48) - timedelta(
            seconds=self.hparams.aggregation_minutes)
        return pd.date_range(start=datetime(1900, 1, 1), end=last_interval, freq=f"{self.hparams.aggregation_minutes}s")

    def _get_time_series_data(self, time_series):
        time_series['Hours'] = time_series['Hours'].apply(
            lambda x: x if x.minute < 48 else datetime(1900, 1, 1, 0, 47, 59))
        if self.hparams.aggregation_minutes is not None:
            ts_group = time_series.groupby([pd.Grouper(freq=f'{self.hparams.aggregation_minutes}s', key='Hours')])
            time_series_mean = ts_group.mean()
            time_series_std = ts_group.std()
            time_series = time_series_mean.merge(time_series_std, left_index=True, right_index=True,
                                                 suffixes=('_mean', '_std'))
        return time_series

    def _get_timeseries_data(self, data, record_id):
        time_series_data = self._get_time_series_data(data)
        time_range_indices = self._get_indices_time_range()
        time_series_data = time_series_data.reindex(time_range_indices)
        time_series_data.index.name = 'Hours'
        time_series_data['record_id'] = record_id
        time_series_data = time_series_data.set_index('record_id', append=True).swaplevel(0, 1)
        return time_series_data

    def _read_data_for_instance(self, file_path):
        record_id = pathlib.Path(file_path).name
        data = pd.read_csv(file_path)

        ts_data = data[['Hours'] + self.vitals + self.lab_measurements + self.interventions]
        ts_data['Hours'] = ts_data['Hours'].apply(lambda x: "%02d:%02d" % (int(x), math.ceil((x % 1) * 60)))
        ts_data['Hours'] = pd.to_datetime(ts_data['Hours'], format="%M:%S")
        ts_data[self.interventions] = self._preprocess_coma_scales(ts_data[self.interventions])
        ts_data = self._get_timeseries_data(ts_data, record_id)

        demo = data[self.demographics].replace({-1: float('NaN')}).mean().fillna(-1).to_frame().T
        demo['record_id'] = record_id
        demo.set_index('record_id', inplace=True)
        return demo, ts_data

    def _preprocess_coma_scales(self, data):
        to_replace = {
            "Glascow coma scale eye opening":
                self.coma_scale_eye_opening_replacements,
            "Glascow coma scale motor response":
                self.coma_scale_motor_replacements,
            "Glascow coma scale verbal response":
                self.coma_scale_verbal_replacements
        }
        coma_scale_columns = list(to_replace.keys())
        coma_scales = data[coma_scale_columns]
        coma_scales = coma_scales.astype(str)
        coma_scales = coma_scales.replace(
            to_replace=to_replace
        )
        coma_scales = coma_scales.astype(float)
        data = data.copy()
        data[coma_scale_columns] = coma_scales
        return data

    # endregion

    def get_users_split(self):
        splits = {}
        for split in ['train', 'test', 'val']:
            splits[split] = pd.read_csv(os.path.join(self.hparams.data_dir, f'{split}_listfile.csv'), index_col='stay')
            splits[split] = splits[split][~splits[split].index.isin(self.blacklist)]
        return splits['train'], splits['val'], splits['test']

    def read_files(self):
        # relevant files paths
        folder_name = self.hparams.folder_format.format(self.hparams.aggregation_minutes)
        folder_path = os.path.join(to_absolute_path(self.hparams.data_dir), folder_name)

        ts_file_name = self.hparams.ts_file_name_format.format(self.hparams.aggregation_minutes)
        ts_file_path = os.path.join(folder_path, ts_file_name)

        demo_file_path = os.path.join(folder_path, self.hparams.demo_file_name)

        # reading files
        demo_df = pd.read_csv(demo_file_path, index_col='record_id')

        ts_data_df = pd.read_csv(ts_file_path, index_col=['record_id', 'Hours'], parse_dates=['Hours'])

        return ts_data_df, demo_df

    def prepare_data(self):
        folder_name = self.hparams.folder_format.format(self.hparams.aggregation_minutes)
        folder_path = os.path.join(to_absolute_path(self.hparams.data_dir), folder_name)
        is_data_already_exists = os.path.exists(folder_path)
        if is_data_already_exists:
            return
        else:
            os.makedirs(folder_path)
        records_list = []
        demo_list = []
        for file in tqdm(glob(self.hparams.data_dir + "\**\*_timeseries.csv", recursive=True)):
            demo, records = self._read_data_for_instance(file)
            records_list.append(records)
            demo_list.append(demo)
        data_df = pd.concat(records_list)
        demo_df = pd.concat(demo_list)
        assert all(data_df.groupby('record_id').apply(lambda g: g.shape[0] == 48))

        ts_file_name = self.hparams.ts_file_name_format.format(self.hparams.aggregation_minutes)
        ts_file_path = os.path.join(folder_path, ts_file_name)
        data_df.to_csv(ts_file_path)

        demo_file_path = os.path.join(folder_path, self.hparams.demo_file_name)
        demo_df.to_csv(demo_file_path)

    def setup(self, stage: Optional[str] = None):
        ts_data_df, demo_df = self.read_files()

        demo_df.loc[demo_df['Height'] == -1] = np.nan

        train_users, val_users, test_users = self.get_users_split()

        y_train, y_val, y_test = train_users['y_true'], val_users['y_true'], test_users['y_true']
        train_users, val_users, test_users = train_users.index, val_users.index, test_users.index

        ts_train_df, ts_val_df, ts_test_df = self._scale_data(ts_data_df, train_users, val_users, test_users)
        demo_train_df, demo_val_df, demo_test_df = self._scale_data(demo_df, train_users, val_users, test_users)
        indications = (~ts_data_df.filter(regex=".*_mean").isna()).astype(int)
        train_indications, val_indications, test_indications = indications.loc[train_users], \
                                                               indications.loc[val_users], \
                                                               indications.loc[test_users]

        train_tensors = [self._pandas_to_tensors(ts_train_df, train_users),
                         torch.tensor(demo_train_df.values, dtype=torch.float),
                         self._pandas_to_tensors(train_indications, train_users),
                         torch.tensor(y_train.values, dtype=torch.long).reshape(-1)]

        val_tensors = [self._pandas_to_tensors(ts_val_df, val_users),
                       torch.tensor(demo_val_df.values, dtype=torch.float),
                       self._pandas_to_tensors(val_indications, val_users),
                       torch.tensor(y_val.values, dtype=torch.long).reshape(-1)]

        test_tensors = [self._pandas_to_tensors(ts_test_df, test_users),
                        torch.tensor(demo_test_df.values, dtype=torch.float),
                        self._pandas_to_tensors(test_indications, test_users),
                        torch.tensor(y_test.values, dtype=torch.long).reshape(-1)]

        self.train_dataset = TensorDataset(*train_tensors)
        self.val_dataset = TensorDataset(*val_tensors)
        self.test_dataset = TensorDataset(*test_tensors)

        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

        class_weight = class_sample_count.sum() / class_sample_count

        class_weight_file = os.path.join(to_absolute_path(self.hparams.data_dir),
                                         self.hparams.class_weight_file_name)
        if not os.path.exists(class_weight_file):
            torch.save(torch.tensor(class_weight), class_weight_file)

        if self.hparams.use_sample_weight:

            class_balance = class_sample_count / class_sample_count.sum()
            weight = 1. / class_sample_count
            samples_weight = torch.tensor([weight[t] for t in train_tensors[-1]])
            self.sampler = WeightedRandomSampler(weights=samples_weight.type('torch.DoubleTensor'),
                                                 num_samples=len(samples_weight),
                                                 replacement=True)
            n_majority = int(np.max(class_balance) * class_sample_count.sum())
            n_minority = int(np.min(class_balance) * class_sample_count.sum())

            self.n_steps = min(
                math.ceil(2 * n_majority / self.hparams.batch_size),
                math.ceil(3 * 2 * n_minority / self.hparams.batch_size)
            )
        else:
            self.sampler = None

    def train_dataloader(self):
        shuffle = not self.hparams.use_sample_weight
        dataloader = DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.num_workers, shuffle=shuffle,
                                persistent_workers=False, sampler=self.sampler, pin_memory=True)
        return LoaderWrapper(dataloader, self.n_steps) if self.hparams.use_sample_weight else dataloader

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=False, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          persistent_workers=True, pin_memory=True, shuffle=False)
