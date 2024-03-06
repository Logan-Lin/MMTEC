from random import randint
from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from einops import rearrange

from pretrain.func import *


class AutoRegressive(nn.Module):
    def __init__(self, paired=True):
        super().__init__()
        self.paired = paired

        self.name = 'AutoReg'
        if not paired:
            self.name += 'Unp'

    def forward(self, models, trip, valid_len):
        if self.paired:
            num_models = len(models)
            encoders, decoders = models[:num_models//2], models[num_models//2:]
            losses = 0.0
            for encoder, decoder in zip(encoders, decoders):
                encode = encoder(trip, valid_len)
                label, recovery, loss = decoder(trip, valid_len, encode)
                losses += loss
            return losses
        else:
            encodes = [encoder(trip, valid_len) for encoder in models[:-1]]
            encode = torch.cat(encodes, 1)
            label, recovery, loss = models[-1](trip, valid_len, encode)
            return loss


class MLM(nn.Module):
    """ Masked Language Model. """

    def __init__(self, num_roads, embed_dim, hidden_size, road_col=None, aux_cols=None):
        super().__init__()
        self.name = 'MLM-' + (str(road_col) if road_col is not None else '') + \
            (''.join(map(str, aux_cols)) if aux_cols is not None else '') + \
            f'-e{embed_dim}-h{hidden_size}'

        self.road_col = road_col
        self.aux_cols = aux_cols

        if road_col is not None:
            self.road_predictor = nn.Sequential(nn.Linear(embed_dim, hidden_size, bias=False),
                                                nn.BatchNorm1d(hidden_size),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(hidden_size, num_roads))
        else:
            self.road_predictor = None

        if len(aux_cols):
            self.aux_predictor = nn.Sequential(nn.Linear(embed_dim, hidden_size, bias=False),
                                               nn.BatchNorm1d(hidden_size),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(hidden_size, len(aux_cols)))
        else:
            self.aux_predictor = None

    def forward(self, models, trip, valid_len):
        min_valid_len = valid_len.min().long()
        mask_i = randint(0, min_valid_len - 1)

        encode = self.forward_masked_encoder(trip, valid_len, mask_i, models)

        loss = 0.0
        if self.road_predictor is not None:
            road_label = trip[:, mask_i, self.road_col].long()  # (B)
            road_pre = self.road_predictor(encode)
            road_loss = F.cross_entropy(road_pre, road_label)
            loss += road_loss
        if self.aux_predictor is not None:
            aux_label = trip[:, mask_i, self.aux_cols].float()  # (B, num_aux)
            aux_pre = self.aux_predictor(encode)
            aux_loss = F.mse_loss(aux_pre, aux_label)
            loss += aux_loss

        return loss

    def forward_masked_encoder(self, trip, valid_len, mask_i, models):
        masked_trip = torch.cat([trip[:, :mask_i], trip[:, (mask_i+1):]], 1)
        encodes = [model(masked_trip, valid_len-1) for model in models]
        encode = torch.cat(encodes, -1)  # (B, num_models * E)
        return encode
