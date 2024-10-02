"""
Implementation of the Transformer Encoder and Decoder.
"""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from model.base import Encoder, Decoder, TimeEmbed
from model.rnn import RnnDecoder


class PositionalEncoding(nn.Module):
    """
    A type of trigonometric encoding for indicating items' positions in sequences.
    """

    def __init__(self, embed_size, max_len):
        super().__init__()

        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TransformerEncoder(Encoder):
    def __init__(self, d_model, num_embed, output_size,
                 sampler, road_col=None, aux_cols=None,
                 nhead=8, num_layers=2, hidden_size=128,
                 pre_embed=None, pre_embed_update=True):
        super().__init__(sampler, 'Transformer-' +
                         (str(road_col) if road_col is not None else '') +
                         (''.join(map(str, aux_cols)) if aux_cols is not None else '') +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}-h{nhead}' +
                         (f'-preU{pre_embed_update}' if pre_embed is not None else ''))

        self.d_model = d_model
        self.road_col = road_col
        self.aux_cols = aux_cols
        self.output_size = output_size

        self.pos_encode = PositionalEncoding(d_model, max_len=500)
        if road_col is not None:
            self.grid_embed = nn.Embedding(num_embed, d_model)
            if pre_embed is not None:
                self.grid_embed.weight = nn.Parameter(torch.from_numpy(pre_embed),
                                                      requires_grad=pre_embed_update)
        else:
            self.grid_embed = None
        if aux_cols is not None and len(aux_cols) > 0:
            self.aux_linear = nn.Linear(len(aux_cols), d_model)
        else:
            self.aux_linear = None
        self.out_linear = nn.Sequential(nn.Linear(d_model, output_size, bias=False),
                                        nn.BatchNorm1d(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size, bias=False),
                                        nn.BatchNorm1d(output_size))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, hidden_size, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, trip, valid_len):
        B, L, E_in = trip.shape

        src_mask = repeat(torch.arange(end=L, device=trip.device),
                          'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
        x, src_mask = self.sampler(trip, src_mask)

        h = torch.zeros(B, x.size(1), self.d_model).to(x.device)
        if self.road_col is not None:
            h += self.grid_embed(x[:, :, self.road_col].long())  # (B, L, E)
        if self.aux_linear is not None:
            h += self.aux_linear(x[:, :, self.aux_cols])
        h += self.pos_encode(h)

        memory = self.encoder(h, src_key_padding_mask=src_mask)  # (B, L, E)
        memory = torch.nan_to_num(memory)
        memory = self.out_linear(memory.mean(1))  # (B, E_out)
        return memory


class TransformerDecoder(Decoder):
    def __init__(self, dis_feats, num_embeds, con_feats, encode_size,
                 d_model, hidden_size, num_layers, num_heads):
        super().__init__(f'Trans-' + ''.join(map(str, dis_feats + con_feats)) +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}-h{num_heads}')
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.d_model = d_model

        self.memory_linear = nn.Linear(encode_size, d_model)
        self.pos_encode = PositionalEncoding(d_model, max_len=500)
        self.start_token = nn.Parameter(torch.randn(d_model), requires_grad=True)

        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads,
                                           dim_feedforward=hidden_size, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)

        if dis_feats is not None and len(dis_feats) > 0:
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
            self.dis_predictors = nn.ModuleList([nn.Linear(d_model, num_embed) for num_embed in num_embeds])
        else:
            self.dis_embeds = None
            self.dis_predictors = None

        if con_feats is not None and len(con_feats) > 0:
            self.con_linear = nn.Linear(len(con_feats), d_model)
            self.con_predictor = nn.Linear(d_model, len(con_feats))
        else:
            self.con_linear = None
            self.con_predictor = None

    def forward(self, trip, valid_len, encode):
        B, L = trip.size(0), trip.size(1)

        x = trip
        memory = self.memory_linear(encode).unsqueeze(1)  # (B, 1, E)

        h = torch.zeros(B, x.size(1), self.d_model).to(x.device)   # (B, L, E)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(x[:, :, dis_feat].long())  # (B, L, E)
        if self.con_linear is not None:
            h += self.con_linear(x[:, :, self.con_feats])

        tgt = torch.cat([repeat(self.start_token, 'E -> B 1 E', B=B),
                         h[:, :-1]], 1)  # (B, L, E), the target sequence
        tgt += self.pos_encode(tgt)
        tgt_mask = self.gen_casual_mask(L).to(tgt.device)
        out = self.transformer(tgt, memory, tgt_mask=tgt_mask)
        out = RnnDecoder.get_pack_data(out, valid_len)

        predict, label = [], []
        loss = 0.0

        if self.dis_predictors is not None:
            for dis_feat, dis_predictor in zip(self.dis_feats, self.dis_predictors):
                dis_pre, dis_label, dis_loss = RnnDecoder.get_pre_label(x, valid_len, dis_feat,
                                                                        out, dis_predictor, True)
                predict.append(dis_pre)
                label.append(dis_label)
                loss += dis_loss

        if self.con_predictor is not None:
            aux_pre, aux_label, aux_loss = RnnDecoder.get_pre_label(x, valid_len, self.con_feats,
                                                                    out, self.con_predictor, False)
            predict.append(aux_pre)
            label.append(aux_label)
            loss += aux_loss

        return np.concatenate(label, -1), np.concatenate(predict, -1), loss

    @staticmethod
    def gen_casual_mask(seq_len, include_self=True):
        """
        Generate a casual mask which prevents i-th output element from
        depending on any input elements from "the future".
        Note that for PyTorch Transformer model, sequence mask should be
        filled with -inf for the masked positions, and 0.0 else.

        :param seq_len: length of sequence.
        :return: a casual mask, shape (seq_len, seq_len)
        """
        if include_self:
            mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
        else:
            mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
        return mask.bool()
