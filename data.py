"""
Data pre-processing for the dataset.
"""

import os
import math
import random
from collections import Counter
from itertools import islice
from time import time

import torch
from torch import nn
import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
import networkx as nx
from sklearn.utils import shuffle

from utils import create_if_noexists, remove_if_exists, intersection


pd.options.mode.chained_assignment = None
CLASS_COL = 'driver'
SET_NAMES = [(0, 'train'), (1, 'val'), (2, 'test')]
MIN_TRIP_LEN = 6
TRIP_COLS = ['tod', 'road', 'road_prop', 'lng', 'lat', 'weekday']


class Data:
    def __init__(self, name, base_path, dataset_path):
        self.name = name

        self.base_path = base_path
        self.dataset_path = dataset_path

        self.df_path = f'{self.dataset_path}/{self.name}.h5'
        self.meta_dir = f'{self.base_path}/meta/{self.name}'
        self.stat_path = f'{self.meta_dir}/stat.h5'

        self.get_meta_path = lambda meta_type, select_set: os.path.join(
            self.meta_dir, f'{meta_type}_{select_set}.npz')

    """ Load functions for loading dataframes and meta. """

    def read_hdf(self):
        # Load the raw data from HDF files.
        # One set of raw dataset is composed of one HDF file with four keys.
        # The trips contains the sequences of trajectories, with three columns: trip, time, road
        self.trips = pd.read_hdf(self.df_path, key='trips')
        # The trip_info contains meta information about trips. For now, this is mainly used for class labels.
        self.trip_info = pd.read_hdf(self.df_path, key='trip_info')
        # The road_info contains meta information about roads.
        self.road_info = pd.read_hdf(self.df_path, key='road_info')
        # self.trips = pd.merge(self.trips, self.road_info[['road', 'lng', 'lat']], on='road', how='left')

        # Add some columns to the trip
        self.trips['minutes'] = self.trips['time'].apply(lambda x: x.timestamp() / 60)
        self.trips['tod'] = self.trips['time'].apply(lambda x: x.timestamp() % (24 * 60 * 60) / (24 * 60 * 60))
        self.trips['weekday'] = self.trips['time'].dt.weekday
        self.stat = self.trips.describe()

        num_road = int(self.road_info['road'].max() + 1)
        num_class = int(self.trip_info[CLASS_COL].max() + 1)
        self.data_info = pd.Series([num_road, num_class], index=['num_road', 'num_class'])
        print('Loaded DataFrame from', self.df_path)

        num_trips = self.trip_info.shape[0]
        self.train_val_test_trips = (self.trip_info['trip'].iloc[:int(num_trips * 0.8)],
                                     self.trip_info['trip'].iloc[int(num_trips * 0.8):int(num_trips * 0.9)],
                                     self.trip_info['trip'].iloc[int(num_trips * 0.9):])

        create_if_noexists(self.meta_dir)
        self.stat.to_hdf(self.stat_path, key='stat')
        self.data_info.to_hdf(self.stat_path, key='info')
        print(self.stat)
        print(self.data_info)
        print('Dumped dataset info into', self.stat_path)

        self.valid_trips = [self.get_valid_trip_id(i) for i in range(3)]

    def load_stat(self):
        # Load statistical information for features.
        self.stat = pd.read_hdf(self.stat_path, key='stat')
        self.data_info = pd.read_hdf(self.stat_path, key='info')

    def load_meta(self, meta_type, select_set):
        meta_path = self.get_meta_path(meta_type, select_set)
        loaded = np.load(meta_path)
        print('Loaded meta from', meta_path)
        return list(loaded.values())

    def get_valid_trip_id(self, select_set):
        select_trip_id = self.train_val_test_trips[select_set]
        trips = self.trips[self.trips['trip'].isin(select_trip_id)]
        valid_trip_id = []
        for _, group in tqdm(trips.groupby('trip'), desc='Filtering trips', total=select_trip_id.shape[0]):
            if (not group.isna().any().any()) and group.shape[0] >= MIN_TRIP_LEN:
                valid_trip_id.append(group.iloc[0]['trip'])
        return valid_trip_id

    def dump_meta(self, meta_type, select_set):
        select_trip_id = self.valid_trips[select_set]
        trips = self.trips[self.trips['trip'].isin(select_trip_id)]
        trip_info = self.trip_info[self.trip_info['trip'].isin(select_trip_id)]
        max_trip_len = max(Counter(trips['trip']).values())

        if meta_type == 'trip':
            arrs, valid_lens = [], []
            for _, group in tqdm(trips.groupby('trip'), desc='Gathering trips', total=len(select_trip_id)):
                arr = group[TRIP_COLS].to_numpy()
                valid_len = arr.shape[0]
                # Pad all trips to the maximum length by repeating the last item.
                arr = np.concatenate([arr, np.repeat(arr[-1:], max_trip_len - valid_len, axis=0)], 0)
                arrs.append(arr)
                valid_lens.append(valid_len)
            arrs, valid_lens = np.stack(arrs, 0), np.array(valid_lens)
            for col_i in [0, 2, 3, 4]:
                col_name = TRIP_COLS[col_i]
                arrs[:, :, col_i] = (arrs[:, :, col_i] - self.stat.loc['mean', col_name]) / \
                    self.stat.loc['std', col_name]
            meta = [arrs, valid_lens]

        elif meta_type == 'class':
            classes = trip_info[CLASS_COL].to_numpy()
            meta = [classes]

        elif meta_type == 'tte':
            travel_times = []
            for _, row in tqdm(trip_info.iterrows(), desc='Gathering TTEs', total=trip_info.shape[0]):
                travel_times.append((row['end'] - row['start']).total_seconds() / 60)
            travel_times = np.array(travel_times)
            meta = [travel_times]

        else:
            raise NotImplementedError('No meta type', meta_type)

        create_if_noexists(self.meta_dir)
        meta_path = self.get_meta_path(meta_type, select_set)
        np.savez(meta_path, *meta)
        print('Saved meta to', meta_path)


class Normalizer(nn.Module):
    def __init__(self, stat, feat_cols, feat_names=None):
        super().__init__()

        self.stat = stat
        self.feat_cols = feat_cols
        self.feat_names = feat_names if feat_names is not None \
            else [TRIP_COLS[feat_col] for feat_col in feat_cols]

    def forward(self, batch):
        """ Normalize the input batch. """
        x = torch.clone(batch)
        for col, name in zip(self.feat_cols, self.feat_names):
            x[:, :, col] = (x[:, :, col] - self.stat.loc['mean', name]) / self.stat.loc['std', name]
        return x


class Denormalizer(nn.Module):
    def __init__(self, stat, feat_cols, feat_names=None):
        super().__init__()

        self.stat = stat
        self.feat_names = feat_names
        self.feat_cols = feat_cols
        self.feat_names = feat_names if feat_names is not None \
            else [TRIP_COLS[feat_col] for feat_col in feat_cols]

    def forward(self, select_cols, batch):
        """ Denormalize the input batch. """
        x = torch.clone(batch)
        for col, name in zip(self.feat_cols, self.feat_names):
            if col in select_cols:
                x[:, :, col] = x[:, :, col] * self.stat.loc['std', name] + self.stat.loc['mean', name]
        return x


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--base', help='base path', type=str, default='sample')
    parser.add_argument('--dataset', help='dataset path', type=str, default='sample')
    parser.add_argument('-n', '--name', help='the name of the dataset', type=str, required=True)
    parser.add_argument('-t', '--types', help='the type of meta data to dump', type=str, required=True)
    parser.add_argument('-i', '--indices', help='the set index to dump meta', type=str)

    args = parser.parse_args()

    data = Data(args.name, args.base, args.dataset)
    data.read_hdf()
    for type in args.types.split(','):
        for i in args.indices.split(','):
            data.dump_meta(type, int(i))
            # Test if we can load meta from the file
            meta = data.load_meta(type, i)
