"""
Downstream trainer.
"""

import math
from abc import abstractmethod
from time import time

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import accuracy_score

from utils import create_if_noexists, cal_classification_metric, cal_regression_metric, next_batch, mean_absolute_percentage_error, BatchPreparer
from data import SET_NAMES
from pretrain.trainer import Trainer as PreTrainer


class Trainer:
    def __init__(self, task_name, base_name, metric_type, data, models,
                 predictor, batch_size, num_epoch, lr, device,
                 meta_types=[], label_meta=None,
                 es_epoch=-1, finetune=False, save_prediction=False):
        self.task_name = task_name
        self.metric_type = metric_type

        self.data = data
        # All models feed into the downstream trainer will be used for calculating the embedding vectors.
        # The embedding vectors will be concatenated along the feature dimension.
        self.models = [model.to(device) for model in models]
        # The predictor is fed with the embedding vectors, and output the prediction required by the downstream task.
        self.predictor = predictor.to(device)

        self.batch_size = batch_size
        self.es_epoch = es_epoch
        self.num_epoch = num_epoch
        self.lr = lr
        self.device = device
        self.finetune = finetune
        self.save_prediction = save_prediction
        self.meta_types = ['trip'] + meta_types  # The first type of meta must be trip.
        self.label_metas = [label_meta] if label_meta is not None else []

        model_name = '_'.join([f'{model.name}-ds-{model.sampler.name}' for model in models])
        self.base_key = f'{task_name}/{base_name}/{model_name}_ft{int(finetune)}'
        self.model_cache_dir = f'{data.base_path}/model_cache/{self.base_key}'
        self.model_save_dir = f'{data.base_path}/model_save/{self.base_key}'
        self.metric_save_name = f'{self.base_key}'

        self.optimizer = torch.optim.Adam(PreTrainer.gather_all_param(*self.models, self.predictor), lr=lr)

    def prepare_batch_iter(self, select_set):
        metas = []
        self.meta_lengths = []
        for meta_type in self.meta_types + self.label_metas:
            meta = self.data.load_meta(meta_type, select_set)
            metas += meta
            self.meta_lengths.append(len(meta))
        self.batch_iter = list(zip(*metas))
        self.num_iter = math.ceil((len(self.batch_iter) - 1) / self.batch_size)

    def prepare_batch_meta(self, batch_meta):
        pointer = 0
        zipped = list(zip(*batch_meta))
        prepared_metas = []
        for i, meta_type in enumerate(self.meta_types):
            meta_prepare_func = BatchPreparer.fetch_prepare_func(meta_type)
            prepared_metas += meta_prepare_func(zipped[pointer:pointer + self.meta_lengths[i]], self.device)
            pointer += self.meta_lengths[i]
        label = self.cal_label(zipped, *prepared_metas[:2])
        return prepared_metas, label

    def train(self):
        num_noimprove_epoches = 0
        best_metric = 0.0
        self.epoch_loss_log, self.epoch_metric_log = [], []
        for epoch in range(self.num_epoch):
            train_loss = self.train_epoch()
            self.epoch_loss_log.append(train_loss)
            print('Epoch %d, avg loss: %.5f' % (epoch+1, train_loss))

            if epoch == 0:
                self.save_models('best')
            if self.es_epoch > -1:
                val_metric = self.eval(1, full_metric=False)
                if val_metric > best_metric:
                    best_metric = val_metric
                    num_noimprove_epoches = 0
                    self.save_models('best')
                else:
                    num_noimprove_epoches += 1
            if self.es_epoch > -1 and num_noimprove_epoches >= self.es_epoch:
                self.load_models('best')
                print(f'Early stopping, rolling back to {epoch - self.es_epoch + 1}-th epoch')
                break

        self.save_models()
        return self.models, self.predictor

    def train_epoch(self):
        self.prepare_batch_iter(0)
        self.train_state()

        loss_log = []
        for batch_meta in tqdm(next_batch(shuffle(self.batch_iter), self.batch_size),
                               desc=f'{self.task_name} training', total=self.num_iter):
            meta, label = self.prepare_batch_meta(batch_meta)
            encodes = self.forward_encoders(*meta)
            pre = self.predictor(encodes).squeeze(-1)

            loss = self.loss_func(pre, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_log.append(loss.item())
        return float(np.mean(loss_log))

    def eval(self, set_index, full_metric=True):
        set_name = SET_NAMES[set_index][1]
        self.prepare_batch_iter(set_index)
        self.eval_state()

        pres, labels = [], []
        for batch_meta in tqdm(next_batch(self.batch_iter, self.batch_size),
                               desc=f'Evaluating on {set_name} set', total=self.num_iter):
            meta, label = self.prepare_batch_meta(batch_meta)
            encodes = self.forward_encoders(*meta)
            pre = self.predictor(encodes).squeeze(-1)

            pres.append(pre.detach().cpu().numpy())
            labels.append(label.cpu().numpy())
        pres, labels = np.concatenate(pres, 0), np.concatenate(labels, 0)

        if full_metric:
            aux_log = {'epoch_loss': ','.join(map(str, self.epoch_loss_log)),
                       'epoch_metric': ','.join(map(str, self.epoch_metric_log))} \
                if hasattr(self, 'epoch_loss_log') else None
            self.metric_and_save(labels, pres, set_name, aux_log)
        else:
            if self.metric_type == 'regression':
                mape = mean_absolute_percentage_error(labels, pres)
                self.epoch_metric_log.append(mape)
                return 1 / (mape + 1e-6)
            elif self.metric_type == 'classification':
                acc = accuracy_score(labels, pres.argmax(-1))
                self.epoch_metric_log.append(acc)
                return acc

    @abstractmethod
    def cal_label(self, zipped_meta, trip, valid_len):
        pass

    def loss_func(self, pre, label):
        pass

    def forward_encoders(self, *x):
        """ Feed the input to all encoders and concatenate the embedding vectors.  """
        encodes = [encoder(*x) for encoder in self.models]
        if not self.finetune:
            encodes = [encode.detach() for encode in encodes]
        encodes = torch.cat(encodes, -1)
        return encodes  # (B, num_encoders * E)

    def train_state(self):
        """ Turn all models and the predictor into training mode.  """
        for encoder in self.models:
            encoder.train()
        self.predictor.train()

    def eval_state(self):
        """ Turn all models and the predictor into evaluation mode.  """
        for encoder in self.models:
            encoder.eval()
        self.predictor.eval()

    def save_models(self, epoch=None):
        """ Save the encoder model and the predictor model. """
        for model in (*self.models, self.predictor):
            if epoch is not None:
                create_if_noexists(self.model_cache_dir)
                save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
            else:
                create_if_noexists(self.model_save_dir)
                save_path = f'{self.model_save_dir}/{model.name}.model'
                print('Saved model to', save_path)
            torch.save(model.state_dict(), save_path)

    def load_model(self, model, epoch=None):
        """ Load one of the encoder. """
        if epoch is not None:
            save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
        else:
            save_path = f'{self.model_save_dir}/{model.name}.model'
            print('Load model from', save_path)
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        return model

    def load_models(self, epoch=None):
        """ Load all encoders. """
        for i, encoder in enumerate(self.models):
            self.models[i] = self.load_model(encoder, epoch)
        self.predictor = self.load_model(self.predictor, epoch)

    def metric_and_save(self, labels, pres, save_name, aux_meta=None):
        """ Calculate the evaluation metric, then save the metric and the prediction result. """
        if self.metric_type == 'classification':
            metric = cal_classification_metric(labels, pres)
        elif self.metric_type == 'regression':
            metric = cal_regression_metric(labels, pres)
        else:
            raise NotImplementedError(f'No type "{type}".')
        print(metric)
        if aux_meta is not None:
            metric = pd.concat([metric.astype(str), pd.Series(aux_meta)])

        metric_save_dir = f'{self.data.base_path}/accuracy/{self.metric_save_name}'
        create_if_noexists(metric_save_dir)
        metric.to_hdf(f'{metric_save_dir}/{save_name}.h5',
                      key=f't{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}', format='table')

        if self.save_prediction:
            prediction_save_dir = f'{self.data.base_path}/prediction/{self.metric_save_name}'
            create_if_noexists(prediction_save_dir)
            np.savez(f'{prediction_save_dir}/{save_name}.npz', labels=labels, pres=pres)


class Classification(Trainer):
    """ A helper class for trajectory classification. """

    def __init__(self, **kwargs):
        super().__init__(task_name='classification', metric_type='classification',
                         label_meta='class', **kwargs)
        self.loss_func = F.cross_entropy

    def cal_label(self, zipped_meta, trip, valid_len):
        return torch.tensor(zipped_meta[-1]).long().to(self.device)


class Destination(Trainer):
    """ A helper class for destination prediction. """

    def __init__(self, pre_length, **kwargs):
        super().__init__(task_name='destination', metric_type='classification', **kwargs)
        self.pre_length = pre_length
        self.loss_func = F.cross_entropy

    def forward_encoders(self, *x):
        trip, valid_len = x[:2]
        return super().forward_encoders(trip, valid_len-self.pre_length, *x[2:])

    def cal_label(self, zipped_meta, trip, valid_len):
        return trip[:, -1, 1].long().detach()


# TODO: write a new version of similar trajectory search.
# class Search(Trainer):
#     """ A helper class for the most similar trajectory search task. """

#     def __init__(self, distance_method, detour_portion, num_target, num_negative, **kwargs):
#         assert distance_method in ['l2', 'cosine']
#         super().__init__(
#             task_name=f'search',
#             metric_type='classification',
#             **kwargs)
#         self.distance_method = distance_method
#         self.num_target = num_target
#         self.num_negative = num_negative
#         self.detour_portion = detour_portion
#         self.cosine_sim = nn.CosineSimilarity(dim=-1)

#     def train(self):
#         print('Similar Trajectory Search do not require training.')
#         return self.models, self.predictor

#     def eval(self, set_index):
#         set_name = SET_NAMES[set_index][1]
#         trip_meta = self.data.get_trip_meta(set_index)
#         detour_meta = self.data.get_detour_meta(set_index, self.num_target, self.detour_portion)
#         self.eval_state()

#         targets, pos_embeds, pos_indices, neg_indices = [], [], [], []

#         batch_iter = list(zip(*detour_meta))
#         i = 0
#         for batch_meta in tqdm(next_batch(batch_iter, self.batch_size),
#                                desc=f'Calculating postive samples on {set_name} set',
#                                total=math.ceil((len(batch_iter) - 1) / self.batch_size)):
#             zipped = list(zip(*batch_meta))
#             batch_detour, detour_len = prepare_trip_batch(zipped[1:3], self.device)
#             neg_i = torch.from_numpy(np.stack(zipped[-1], 0)).long()[:,
#                                                                      :self.num_negative].to(self.device)  # (B, num_neg)

#             queries = torch.cat([encoder(batch_detour, detour_len).detach()
#                                  for encoder in self.models], -1)  # (B, E)
#             # pos_indices.append(torch.arange(i, i+queries.size(0)).long().to(self.device))
#             pos_indices.append(torch.tensor(zipped[0]).long().to(self.device))
#             i += queries.size(0)
#             pos_embeds.append(queries)
#             neg_indices.append(neg_i)

#         s_time = time()
#         batch_iter = list(zip(*trip_meta))
#         for batch_meta in tqdm(next_batch(batch_iter, self.batch_size),
#                                desc=f'Calculating all targets on {set_name} set',
#                                total=len(batch_iter) // self.batch_size):
#             zipped = list(zip(*batch_meta))
#             batch_trip, valid_len = prepare_trip_batch(zipped[:2], self.device)
#             target = torch.cat([encoder(batch_trip, valid_len).detach() for encoder in self.models], -1)  # (B, E)
#             targets.append(target)
#         targets = torch.cat(targets, 0)  # (num_total, E)
#         e_time = time()
#         print('Used', e_time - s_time, 'sec to calculate embeddings for all trajectories.')

#         sims = []
#         for pos_embed, pos_i, neg_i in tqdm(zip(pos_embeds, pos_indices, neg_indices),
#                                             desc=f'Scoring similar trajectories on {set_name} set',
#                                             total=len(pos_embeds)):
#             target_embed = targets[pos_i]  # (B, E)
#             neg_embed = targets[neg_i]  # (B, num_neg, E)
#             keys = torch.cat([target_embed.unsqueeze(1), neg_embed], 1)  # (B, num_neg+1, E)
#             queries = repeat(pos_embed, 'B E -> B N E', N=keys.size(1))

#             if self.distance_method == 'l2':
#                 sim = queries - keys  # (B, N+1, E)
#                 sim = 1 / (torch.sqrt(torch.pow(sim, 2).sum(-1)) + 1e-6)  # (B, N)
#                 sims.append(sim.detach().cpu().numpy())
#             elif self.distance_method == 'cosine':
#                 sim = self.cosine_sim(queries, keys)  # (B, N)
#                 sims.append(sim.detach().cpu().numpy())
#         sims = np.concatenate(sims)  # (num_pos, num_neg+1)
#         labels = np.zeros(sims.shape[0]).astype(int)  # (num_pos)
#         self.metric_and_save(labels, sims, set_name)


class TTE(Trainer):
    """ A helper class for travel time estimation evaluation. """

    def __init__(self, **kwargs):
        super().__init__(task_name=f'tte', metric_type='regression',
                         label_meta='tte', **kwargs)
        self.loss_func = F.mse_loss

    def cal_label(self, zipped_meta, trip, valid_len):
        return torch.tensor(zipped_meta[-1]).float().to(self.device)
