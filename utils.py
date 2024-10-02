"""
Utility functions for the project.
"""

import math
import os
import shutil
from os.path import exists

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, accuracy_score, roc_auc_score
from tqdm import trange

from model import sample


def create_if_noexists(*paths):
    for path in paths:
        if not exists(path):
            os.makedirs(path)


def remove_if_exists(path):
    if exists(path):
        os.remove(path)


def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        if end_index - start_index > 1:
            yield data[start_index:end_index]


def clean_dirs(*dirs):
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)


def clean_small_caches():
    cache_dirs = ['sample/model_save', 'sample/model_cache',
                  'sample/accuracy', 'sample/prediction', 'sample/generation']
    clean_dirs(*cache_dirs)


def regulate_batch(batch, device):
    """
    Regulate all sequences into the same length.

    :returns: the regulated batch as a torch Tensor, and a sequence recording the valid length of the input sequences.
    """
    valid_len = [len(s) for s in batch]  # valid lengths of sequences.
    max_len = max(*valid_len)
    # We fill all sequences in a batch to the maximum length, by extending the last value of each sequence.
    batch = np.stack([np.concatenate([s, np.repeat(s[-1:], max_len - s.shape[0], axis=0)], 0)
                     for s in batch], 0)  # (B, L, E)

    batch = torch.from_numpy(batch).float().to(device)
    valid_len = torch.tensor(valid_len).long().to(device)  # (B)
    return batch, valid_len


def mean_absolute_percentage_error(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape


def cal_regression_metric(label, pres):
    rmse = math.sqrt(mean_squared_error(label, pres))
    mae = mean_absolute_error(label, pres)
    mape = mean_absolute_percentage_error(label, pres)

    s = pd.Series([rmse, mae, mape], index=['rmse', 'mae', 'mape'])
    return s


def top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


def cal_classification_metric(labels, pres):
    """
    :param labels: classification label, with shape (N).
    :param pres: predicted classification distribution, with shape (N, num_class).
    """
    pres_index = pres.argmax(-1)  # (N)
    macro_f1 = f1_score(labels, pres_index, average='macro')
    macro_recall = recall_score(labels, pres_index, average='macro')
    acc = accuracy_score(labels, pres_index)
    n_list = [5, 10]
    top_n_acc = [top_n_accuracy(labels, pres, n) for n in n_list]

    s = pd.Series([macro_f1, macro_recall, acc] + top_n_acc,
                  index=['macro_f1', 'macro_rec'] +
                  [f'acc@{n}' for n in [1] + n_list])
    return s


def intersection(lst1, lst2):
    lst3 = list(set(lst1) & set(lst2))
    return lst3


class BatchPreparer:
    def __init__(self):
        pass

    @staticmethod
    def fetch_prepare_func(meta_type):
        if meta_type == 'trip':
            return BatchPreparer.prepare_trip_batch
        elif meta_type == 'graph':
            return BatchPreparer.prepare_graph_signal_batch
        else:
            raise NotImplementedError('No prepare function for meta type "' + meta_type + '".')

    @staticmethod
    def prepare_trip_batch(batch_meta, device):
        batch_trip, valid_len = batch_meta
        batch_trip = np.stack(batch_trip, 0)
        batch_trip = torch.from_numpy(batch_trip).float().to(device)
        valid_len = torch.tensor(valid_len).long().to(device)

        return batch_trip, valid_len

    @staticmethod
    def prepare_graph_signal_batch(batch_meta, device):
        batch_signal, = batch_meta
        batch_signal = torch.stack([torch.from_numpy(meta.todense()) for meta in batch_signal], 0)  # (B, N, E)
        batch_signal = batch_signal.float().to(device)
        return batch_signal,


def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")
    total, used = devices_info[cuda_device].split(',')
    return total, used


def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    for _ in trange(256, desc='Occupying CUDA device'):
        x = torch.FloatTensor(1024, block_mem).to('cuda:0')
        del x


def cal_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
