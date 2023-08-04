import math
import os
import pathlib
import tarfile
from datetime import timedelta, datetime

import pytorch_lightning as pl
import pandas as pd
import numpy as np
import requests
import torch

from tqdm import tqdm
from typing import Optional, List
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from hydra.utils import to_absolute_path

from datasets.abstact_medical_dataset import MedicalDataset
from datasets.samplers import LoaderWrapper


class Physionet2012(pl.LightningDataModule, MedicalDataset):
    blacklist = [
        140501, 150649, 140936, 143656, 141264, 145611, 142998, 147514, 142731,
        150309, 155655, 156254
    ]

    expanded_static_features = [
        'Age', 'Gender=0', 'Gender=1', 'Height', 'ICUType=1', 'ICUType=2',
        'ICUType=3', 'ICUType=4'
    ]

    outcome_features = ['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death']

    static_features = [
        'Age', 'Gender', 'Height', 'ICUType'
    ]
    categorical_demographics = {
        'Gender': [0, 1],
        'ICUType': [1, 2, 3, 4]
    }

    ts_features = [
        'Weight', 'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
        'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
        'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
        'Urine', 'WBC', 'pH'
    ]

    def __init__(self,
                 download_dir: str,
                 batch_size: int,
                 num_workers: int,
                 data_urls: List[str],
                 outcome_urls: List[str],
                 folder_format: str,
                 ts_file_name_format: str,
                 sr_file_name_format: str,
                 length_file_name_format: str,
                 demo_file_name: str,
                 target_file_name: str,
                 class_weight_file_name: str,
                 label: str,
                 aggregation_minutes: int,
                 use_sample_weight=False
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.sampler = None
        self.n_steps = None
        self.class_weight = None

    # region Private

    def _get_indices_time_range(self):
        last_interval = datetime(1900, 1, 1) + timedelta(minutes=48) - timedelta(
            seconds=self.hparams.aggregation_minutes)
        return pd.date_range(start=datetime(1900, 1, 1), end=last_interval, freq=f"{self.hparams.aggregation_minutes}s")

    @staticmethod
    def _get_txt_files_contents(local_file):
        contents = {}
        tar = tarfile.open(local_file, "r:gz")
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                participant_id = member.name.split('/')[1].split('.')[0]
                contents[participant_id] = pd.read_csv(f, sep=',', header=0)
        return contents

    @staticmethod
    def _get_outcome_data(outcome_local_files):
        outcomes = []
        for outcome_file in outcome_local_files:
            outcome_df = pd.read_csv(outcome_file, header=0, sep=',')
            outcome_df = outcome_df.rename(columns={'RecordID': 'record_id'})
            outcome_df = outcome_df.set_index('record_id')
            outcomes.append(outcome_df)
        return outcomes

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

    def get_users_split(self):
        splits = {}
        data_dir = to_absolute_path(self.hparams.download_dir)
        for split in ['train', 'test', 'val']:
            with open(os.path.join(data_dir, f'{split}_records.txt'), 'r') as f:
                splits[split] = list(map(lambda x: int(x.strip()), f.readlines()))
                splits[split] = [user for user in splits[split] if user not in self.blacklist]
        assert len(set(splits['train']).intersection(set(splits['test']))) == 0
        assert len(set(splits['train']).intersection(set(splits['val']))) == 0
        assert len(set(splits['test']).intersection(set(splits['val']))) == 0
        return splits['train'], splits['val'], splits['test']

    def _get_time_series_data(self, data):
        data['Time'] = data['Time'].apply(lambda x: x if x.minute < 48 else datetime(1900, 1, 1, 0, 47, 59))
        if any(data[['Time', 'Parameter']].duplicated()):
            data = data.groupby(['Time', 'Parameter'], as_index=False).mean().reset_index()

        time_series = data.pivot(index='Time', columns='Parameter', values='Value')
        if self.hparams.aggregation_minutes is not None:
            ts_group = time_series.groupby([pd.Grouper(freq=f'{self.hparams.aggregation_minutes}s', level='Time')])
            time_series_mean = ts_group.mean()
            time_series_std = ts_group.std()
            time_series = time_series_mean.merge(time_series_std, left_index=True, right_index=True,
                                                 suffixes=('_mean', '_std'))
        return time_series

    def _get_timeseries_data(self, record_df, record_id):
        ts_data_mask = record_df['Parameter'].isin(self.ts_features)
        time_series_data = record_df[ts_data_mask]
        time_series_data = self._get_time_series_data(time_series_data)
        time_range_indices = self._get_indices_time_range()
        time_series_data = time_series_data.reindex(time_range_indices)
        time_series_data['record_id'] = int(record_id)
        time_series_data = time_series_data.set_index('record_id', append=True).swaplevel(0, 1)

        # time_series_rate = self._get_samples_times(time_series_data)
        # time_series_sample_times = pd.DataFrame(record_df.Time.unique())
        # time_series_length = record_df.Time.nunique()
        # time_series_rate = time_series_rate.reindex(time_range_indices)
        # time_series_sample_times['record_id'] = int(record_id)
        # time_series_sample_times = time_series_sample_times.set_index('record_id', append=True).swaplevel(0, 1)

        # return time_series_data, time_series_sample_times, time_series_length
        return time_series_data

    def _get_static_data(self, record_df, record_id):
        # Extract statics
        statics_indicator = record_df['Parameter'].isin(
            ['RecordID'] + self.static_features)
        statics = record_df[statics_indicator]

        # Handle duplicates in statics
        duplicated_statics = statics[['Time', 'Parameter']].duplicated()
        if duplicated_statics.sum() > 0:
            # Average over duplicate measurements
            statics = statics.groupby(['Time', 'Parameter'], as_index=False) \
                .mean().reset_index()
        statics = statics.pivot(
            index='Time', columns='Parameter', values='Value')
        statics = statics.reindex().reset_index()
        statics = statics.iloc[0]

        # Be sure we are loading the correct record
        assert str(int(statics['RecordID'])) == record_id
        # Drop RecordID
        statics = statics[self.static_features]

        # Do one hot encoding for categorical features
        for demo, values in self.categorical_demographics.items():
            cur_demo = statics[demo]
            # Transform categorical values into zero based index
            to_replace = {val: values.index(val) for val in values}
            # Ensure we don't have unexpected values
            if cur_demo in to_replace.keys():
                indicators = to_replace[cur_demo]
                one_hot_encoded = np.eye(len(values))[indicators]
            else:
                # We have a few cases where the categorical variables are not
                # available. Then we should just return zeros for all
                # categories.
                one_hot_encoded = np.zeros(len(to_replace.values()))
            statics.drop(columns=demo, inplace=True)
            columns = [f'{demo}={val}' for val in values]
            statics = pd.concat([statics, pd.Series(one_hot_encoded, index=columns)])

        # Ensure same order
        statics = statics[self.expanded_static_features]
        statics['record_id'] = record_id
        statics = statics.to_frame().T.set_index('record_id')

        return statics

    def _save_ts_data_files(self, folder_path, set_records_list, sets_names_list):
        file_name = self.hparams.ts_file_name_format.format(self.hparams.aggregation_minutes)
        ts_file_path = os.path.join(folder_path, file_name)
        data_df = pd.concat(set_records_list, keys=sets_names_list, names=['set', 'record_id', 'Time'])
        data_df.to_csv(ts_file_path)

    def _save_demo_data_files(self, folder_path, set_demo_list, sets_names_list):
        demo_file_path = os.path.join(folder_path, self.hparams.demo_file_name)
        demo_df = pd.concat(set_demo_list, keys=sets_names_list, names=['set', 'record_id'])
        demo_df.to_csv(demo_file_path)

    def _save_sample_rate_data_files(self, folder_path, set_sample_rate_list, sets_names_list):
        file_name = self.hparams.sr_file_name_format.format(self.hparams.aggregation_minutes)
        sr_file_path = os.path.join(folder_path, file_name)
        sample_rate_df = pd.concat(set_sample_rate_list, keys=sets_names_list, names=['set', 'record_id', 'Time'])
        sample_rate_df.to_csv(sr_file_path)

    def _save_length_files(self, folder_path, set_length_list, sets_names_list):
        file_name = self.hparams.length_file_name_format.format(self.hparams.aggregation_minutes)
        length_file_path = os.path.join(folder_path, file_name)
        length_df = pd.concat(set_length_list, keys=sets_names_list, names=['set', 'record_id'])
        length_df.to_csv(length_file_path)

    def _save_outcome_files(self, folder_path, sets_names_list):
        outcome_local_files = self._download_files(self.hparams.outcome_urls)
        outcome_df_list = self._get_outcome_data(outcome_local_files)
        outcome_df = pd.concat(outcome_df_list, keys=sets_names_list, names=['set', 'record_id'])
        target_file_path = os.path.join(folder_path, self.hparams.target_file_name)
        outcome_df.to_csv(target_file_path)

    def _handle_single_participant_data(self, record_id, record_df):
        record_df['Time'] = pd.to_datetime(record_df['Time'], format="%M:%S")
        static_data = self._get_static_data(record_df, record_id)
        # time_series_data, time_series_rate, time_series_length = self._get_timeseries_data(record_df, record_id)
        time_series_data = self._get_timeseries_data(record_df, record_id)
        # return time_series_data, time_series_rate, static_data, time_series_length

        return time_series_data, static_data

    def _handle_single_set_file(self, local_file):
        records_list = []
        records_sample_rate_list = []
        demographics_list = []
        # time_series_length_list = []
        set_contents = self._get_txt_files_contents(local_file)
        for record_id, record_df in tqdm(set_contents.items()):
            # time_series_data, ts_sample_freq_list, static_data, time_series_length = \
            #     self._handle_single_participant_data(record_id, record_df)
            time_series_data, static_data = self._handle_single_participant_data(record_id, record_df)
            records_list.append(time_series_data)
            # records_sample_rate_list.append(ts_sample_freq_list)
            demographics_list.append(static_data)
            # time_series_length_list.append((record_id, time_series_length))
        # return demographics_list, records_list, records_sample_rate_list, time_series_length_list

        return demographics_list, records_list

    def _parse_files(self):
        print("Parsing set files")
        local_data_files = self._download_files(self.hparams.data_urls)
        sets_names_list = []
        set_records_list = []
        # set_sample_rate_list = []
        set_demo_list = []
        # set_length_list = []

        for local_file in local_data_files:
            current_set = pathlib.Path(local_file).stem.split('.')[0]
            sets_names_list.append(current_set)
            print(f"Parsing {current_set}")
            # demographics_list, records_list, records_sample_rate_list, time_series_length_list = \
            #     self._handle_single_set_file(local_file)
            demographics_list, records_list = self._handle_single_set_file(local_file)

            set_records_list.append(pd.concat(records_list))
            # set_sample_rate_list.append(pd.concat(records_sample_rate_list))
            set_demo_list.append(pd.concat(demographics_list))
            # set_length_list.append(pd.DataFrame(time_series_length_list, columns=['record_id', 'length']))
        # return set_demo_list, set_records_list, set_sample_rate_list, set_length_list, sets_names_list
        return set_demo_list, set_records_list, sets_names_list

    def _handle_data(self, folder_path):
        # set_demo_list, set_records_list, set_sample_rate_list, set_length_list, sets_names_list = self._parse_files()
        set_demo_list, set_records_list, sets_names_list = self._parse_files()

        self._save_outcome_files(folder_path, sets_names_list)
        # self._save_length_files(folder_path, set_length_list, sets_names_list)
        self._save_ts_data_files(folder_path, set_records_list, sets_names_list)
        self._save_demo_data_files(folder_path, set_demo_list, sets_names_list)
        # self._save_sample_rate_data_files(folder_path, set_sample_rate_list, sets_names_list)

    def read_files(self):
        # relevant files paths
        folder_name = self.hparams.folder_format.format(self.hparams.aggregation_minutes)
        folder_path = os.path.join(to_absolute_path(self.hparams.download_dir), folder_name)

        target_file_path = os.path.join(folder_path, self.hparams.target_file_name)

        file_name = self.hparams.sr_file_name_format.format(self.hparams.aggregation_minutes)
        sr_file_path = os.path.join(folder_path, file_name)

        ts_file_name = self.hparams.ts_file_name_format.format(self.hparams.aggregation_minutes)
        ts_file_path = os.path.join(folder_path, ts_file_name)

        file_name = self.hparams.length_file_name_format.format(self.hparams.aggregation_minutes)
        length_file_path = os.path.join(folder_path, file_name)

        demo_file_path = os.path.join(folder_path, self.hparams.demo_file_name)

        # reading files
        target_df = pd.read_csv(target_file_path, index_col='record_id', usecols=['record_id', self.hparams.label])

        demo_df = pd.read_csv(demo_file_path,
                              usecols=self.expanded_static_features + ['record_id'],
                              index_col='record_id')

        ts_data_df = pd.read_csv(ts_file_path,
                                 # usecols=self.ts_features + ['record_id', 'Time'],
                                 index_col=['record_id', 'Time'],
                                 parse_dates=['Time'])
        # keep order for consistency
        ts_data_df = ts_data_df.drop(columns=['set'])
        # ts_data_df = ts_data_df[self.ts_features]

        # sr_data_df = pd.read_csv(sr_file_path,
        #                          usecols=['record_id', 'Time', 'sample_time'],
        #                          index_col=['record_id', 'Time'],
        #                          parse_dates=['Time'])

        # length_df = pd.read_csv(length_file_path,
        #                         usecols=['record_id.1', 'length'],
        #                         index_col=['record_id.1'])

        return ts_data_df, demo_df, None, target_df, None

    # @staticmethod
    # def _scale_data(data_df, train_users, val_users, test_users):
    #     ss = StandardScaler()
    #     train_df = data_df.loc[train_users].copy()
    #     X_train = ss.fit_transform(train_df.values)
    #     X_train = np.nan_to_num(X_train, nan=0)
    #     train_df[:] = X_train
    #
    #     val_df = data_df.loc[val_users].copy()
    #     X_val = ss.transform(val_df.values)
    #     X_val = np.nan_to_num(X_val, nan=0)
    #     val_df[:] = X_val
    #
    #     test_df = data_df.loc[test_users].copy()
    #     X_test = ss.transform(test_df.values)
    #     X_test = np.nan_to_num(X_test, nan=0)
    #     test_df[:] = X_test
    #
    #     return train_df, val_df, test_df
    #
    # @staticmethod
    # def _pandas_to_tensors(df, users):
    #     data = df.groupby('record_id') \
    #         .apply(lambda g: torch.tensor(g.values, dtype=torch.float)) \
    #         .loc[users].values
    #
    #     return torch.stack(list(data))

    # endregion

    def prepare_data(self):
        os.makedirs(to_absolute_path(self.hparams.download_dir), exist_ok=True)
        folder_name = self.hparams.folder_format.format(self.hparams.aggregation_minutes)
        folder_path = os.path.join(to_absolute_path(self.hparams.download_dir), folder_name)
        is_data_already_exists = os.path.exists(folder_path)
        if is_data_already_exists:
            return
        os.makedirs(folder_path, exist_ok=True)
        self._handle_data(folder_path)

    def setup(self, stage: Optional[str] = None):
        ts_data_df, demo_df, sr_data_df, target_df, length_df = self.read_files()
        train_users, val_users, test_users = self.get_users_split()
        y_train, y_val, y_test = target_df.loc[train_users], target_df.loc[val_users], target_df.loc[test_users]

        ts_train_df, ts_val_df, ts_test_df = self._scale_data(ts_data_df, train_users, val_users, test_users)
        # sr_train_df, sr_val_df, sr_test_df = self._scale_data(sr_data_df, train_users, val_users, test_users)
        # demo_train_df, demo_val_df, demo_test_df = self._scale_data(demo_df, train_users, val_users, test_users)
        demo_train_df, demo_val_df, demo_test_df = demo_df.loc[train_users], demo_df.loc[val_users], demo_df.loc[
            test_users]
        indications = (~ts_data_df.filter(regex=".*_mean").isna()).astype(int)
        train_indications, val_indications, test_indications = indications.loc[train_users], indications.loc[val_users], \
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

        class_weight_file = os.path.join(to_absolute_path(self.hparams.download_dir),
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
