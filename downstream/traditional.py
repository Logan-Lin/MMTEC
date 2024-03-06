import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import networkx as nx

from downstream.trainer import *
from utils import cal_regression_metric


class PathTTE:
    def __init__(self, data, weighted=True):
        self.data = data
        self.weighted = weighted

        self.base_key = f'PathTTE_{data.name}_w{weighted}'

    def train(self, portion):
        self.data.build_network()
        nx.set_edge_attributes(self.data.network, 0, 'weight')
        nx.set_edge_attributes(self.data.network, 0, 'count')

        trip_meta = self.data.get_trip_meta(0)
        road_te, road_count = (np.zeros(self.data.data_info['num_road']) for _ in range(2))

        for row in tqdm(list(zip(*trip_meta))[:int(trip_meta[0].shape[0] * portion)],
                        desc='Calculating the mean travel time of road links'):
            trip, valid_len = row
            trip = trip[:valid_len]
            link_te = trip[1:, 0] - trip[:-1, 0]
            link_index = trip[:-1, 1].astype(int)

            if self.weighted:
                for o, d, te in zip(trip[:-1, 1], trip[1:, 1], link_te):
                    o, d = int(o), int(d)
                    self.data.network[o][d]['weight'] += te
                    self.data.network[o][d]['count'] += 1

            road_te[link_index] += link_te
            road_count[link_index] += 1

        roads_wo_val = road_count < 1
        road_count[roads_wo_val] = 1
        road_te = road_te / road_count
        mean_te = np.mean(road_te[~roads_wo_val])
        road_te[roads_wo_val] = mean_te

        if self.weighted:
            for o, d in self.data.network.edges():
                edge = self.data.network[o][d]
                if edge['count'] > 0 and edge['weight'] > 0:
                    self.data.network[o][d]['weight'] = edge['count'] / edge['weight']
                else:
                    self.data.network[o][d]['weight'] = 1 / mean_te
        else:
            nx.set_edge_attributes(self.data.network, 1, 'weight')

        self.road_te = road_te

    def eval(self, set_index):
        set_name = ["train", "val", "test"][set_index]
        trip_meta = self.data.get_trip_meta(set_index)
        labels = self.data.get_tte_meta(set_index)

        pres = []
        for row in tqdm(zip(*trip_meta),
                        desc=f'Evaluting on {set_name} set',
                        total=labels.shape[0]):
            trip, valid_len = row
            o, d = int(trip[0, 1]), int(trip[-1, 1])
            path = nx.dijkstra_path(self.data.network, source=o, target=d, weight='weight')
            tt_pre = self.road_te[path].sum() / 60

            pres.append(tt_pre)
        pres = np.array(pres)

        cal_regression_metric(self.data.base_path, labels, pres, save_name=f'{self.base_key}_{set_name}')


class LrTTE:
    def __init__(self, data):
        self.data = data

        self.lr = LinearRegression()
        self.base_key = f'LrTTE_{data.name}'

    def train(self, portion):
        trips, _ = self.data.get_trip_meta(0)
        tt_label = self.data.get_tte_meta(0)
        select_portion = int(trips.shape[0] * portion)
        trips, tt_label = trips[:select_portion], tt_label[:select_portion]
        odt = np.concatenate([trips[:, 0, [0, 2, 3]], trips[:, -1, [2, 3]]], -1)
        self.lr.fit(odt, tt_label)

    def eval(self, set_index):
        set_name = ["train", "val", "test"][set_index]
        trips, _ = self.data.get_trip_meta(set_index)
        labels = self.data.get_tte_meta(set_index)
        odt = np.concatenate([trips[:, 0, [0, 2, 3]], trips[:, -1, [2, 3]]], -1)
        pres = self.lr.predict(odt)

        cal_regression_metric(self.data.base_path, labels, pres, save_name=f'{self.base_key}_{set_name}')
