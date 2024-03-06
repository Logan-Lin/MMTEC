"""
All samplers should be able to accept multiple arguments, each argument is an tensor.
They should also return multiple values, with the number of values the same as the number of input arguments.
"""

import torch
from torch import nn
import networkx as nx


class Sampler(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name


class KHopSampler(Sampler):
    def __init__(self, jump, select):
        super().__init__(f'khop{jump}-{select}')

        self.jump = jump
        self.select = select

    def forward(self, *tensors):
        tensors = (tensor[:, self.select-1::self.jump] for tensor in tensors)
        return tensors


class PassSampler(Sampler):
    """ 
    As the name suggest, this sampler won't change the tensor given to it, only to output as it is. 
    """
    def __init__(self):
        super().__init__('pass')

    def forward(self, *tensors):
        return tensors


class IndexSampler(Sampler):
    def __init__(self, index):
        super().__init__(f'index{index}')
        self.index = index

    def forward(self, *tensors):
        tensors = (tensor[:, self.index] for tensor in tensors)
        return tensors


class PoolSampler(Sampler):
    def __init__(self, pool_method='mean'):
        super().__init__(f'pool{pool_method}')

        if pool_method == 'mean':
            self.pool_func = torch.mean
        elif pool_method == 'max':
            self.pool_func = torch.max
        elif pool_method == 'min':
            self.pool_func = torch.min
        else:
            raise NotImplementedError(pool_method)
    
    def forward(self, *tensors):
        tensors = (self.pool_func(tensor, dim=1) for tensor in tensors)
        return tensors