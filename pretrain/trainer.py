import math
from time import time

import numpy as np
import torch
from sklearn.utils import shuffle
from tqdm import tqdm

from utils import create_if_noexists, next_batch, cal_model_size, BatchPreparer
from data import SET_NAMES


class Trainer:
    """
    Base class of the pre-training helper class.
    Implements most of the functions shared by all types of pre-trainers.
    """

    def __init__(self, data, models, trainer_name,
                 loss_func, batch_size, num_epoch, lr, device, cache_epoches=False,
                 meta_types=[], suffix=None):
        self.data = data
        # The list of models may have different usage in different types of trainers.
        self.models = [model.to(device) for model in models]
        self.trainer_name = trainer_name

        self.loss_func = loss_func.to(device)
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.lr = lr
        self.device = device
        self.cache_epoches = cache_epoches

        self.meta_types = ['trip'] + meta_types
        model_name = '_'.join([model.name for model in models])
        self.BASE_KEY = f'{trainer_name}/{loss_func.name}/{self.data.name}_{model_name}_b{batch_size}-lr{lr}'
        if suffix is not None:
            self.BASE_KEY += suffix
        self.model_cache_dir = f'{data.base_path}/model_cache/{self.BASE_KEY}'
        self.model_save_dir = f'{data.base_path}/model_save/{self.BASE_KEY}'

        self.optimizer = torch.optim.Adam(self.gather_all_param(*self.models, self.loss_func), lr=lr)

        for model in models + [loss_func]:
            print(model.name, 'size', cal_model_size(model), 'MB')

    def prepare_batch_iter(self, select_set):
        metas = []
        self.meta_lengths = []
        for meta_type in self.meta_types:
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
        return prepared_metas

    def train(self, start=-1):
        """
        :param start: if given a value of 0 or higher, will try to load the trained model 
            cached after the start-th epoch training, and resume training from there.
        """
        self.prepare_batch_iter(0)
        self.train_epoches(start)

    def train_epoches(self, start=-1):
        self.train_state()

        if start > -1:
            self.load_models(start)
            print('Resumed training from epoch', start)

        for epoch_i in range(start+1, self.num_epoch):
            s_time = time()
            train_loss = self.train_epoch(epoch_i)
            e_time = time()
            print('Epoch %d, avg loss: %.5f' % (epoch_i+1, train_loss))

            if epoch_i == start + 1:
                print('Train time:', (e_time - s_time) / 60, 'min/epoch.')

            if self.cache_epoches and epoch_i < self.num_epoch - 1:
                self.save_models(epoch_i)
        self.save_models()

    def train_epoch(self, epoch_i):
        """
        :param metas: series of input meta data. Notice that different types of meta data should be ordered.
            Here, the order is grid, coor and road.
        """
        loss_log = []
        for batch_meta in tqdm(next_batch(shuffle(self.batch_iter), self.batch_size),
                               desc=f'{self.trainer_name} training {epoch_i+1}-th epoch',
                               total=self.num_iter):
            self.optimizer.zero_grad()
            loss = self.loss_func(self.models, *self.prepare_batch_meta(batch_meta))
            loss.backward()
            self.optimizer.step()

            loss_log.append(loss.item())
        return float(np.mean(loss_log))

    @staticmethod
    def gather_all_param(*models):
        parameters = []
        for encoder in models:
            parameters += list(encoder.parameters())
        return parameters

    def save_models(self, epoch=None):
        """ Save all models. """
        for model in (*self.models, self.loss_func):
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
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        print('Load model from', save_path)
        return model

    def load_models(self, epoch=None):
        """ Load all encoders. """
        for i, model in enumerate(self.models):
            self.models[i] = self.load_model(model, epoch)
        self.loss_func = self.load_model(self.loss_func, epoch)

    def get_models(self):
        for model in self.models:
            model.eval()
        return self.models

    def train_state(self):
        for model in self.models:
            model.train()
        self.loss_func.train()

    def eval_state(self):
        for model in self.models:
            model.eval()
        self.loss_func.eval()


class ContrastiveTrainer(Trainer):
    """
    Trainer for contrastive pre-training.
    """

    def __init__(self, **kwargs):
        """
        :encoders: a list of encoders. Each encoder should be able to accept a series of batch_meta input, and have a 'name' field.
        """
        super().__init__(trainer_name='contrastive', **kwargs)


class GenerativeTrainer(Trainer):
    """
    Trainer for generative pre-training.
    Contains a generate function for evaluating the recovered input.
    """

    def __init__(self, **kwargs):
        super().__init__(trainer_name='generative', **kwargs)
        self.generation_save_dir = f'{self.data.base_path}/generation/{self.BASE_KEY}'

    def generate(self, set_index, save_gen=False, **gen_params):
        self.prepare_batch_iter(set_index)
        self.eval_state()

        gen_dicts = []
        for batch_meta in tqdm(next_batch(self.batch_iter, self.batch_size),
                               desc='Generating', total=self.num_iter):
            gen_dict, gen_save_name = self.loss_func.generate(self.models, *self.prepare_batch_meta(batch_meta),
                                                              **gen_params)
            gen_dicts.append(gen_dict)
        numpy_dict = {key: np.concatenate([gen_dict[key] for gen_dict in gen_dicts], 0) for key in gen_dicts[0].keys()}

        create_if_noexists(self.generation_save_dir)
        generation_save_path = f'{self.generation_save_dir}/{gen_save_name}-{SET_NAMES[set_index][1]}.npz'
        np.savez(generation_save_path, **numpy_dict)
        print('Saved generation to', generation_save_path)


class MomentumTrainer(Trainer):
    """
    Trainer for momentum-style parameter updating.
    Requires the loss function contains extra "teacher" models symmetric to the base models.
    The parameters of the teacher models will be updated in a momentum-style.
    """

    def __init__(self, momentum, teacher_momentum, weight_decay, eps, warmup_epoch=10, **kwargs):
        super().__init__(trainer_name='momentum',
                         suffix=f'_m{momentum}-tm{teacher_momentum}-wd{weight_decay}-eps{eps}-we{warmup_epoch}',
                         **kwargs)

        self.momentum = momentum
        self.teacher_momentum = teacher_momentum
        self.warmup_epoch = warmup_epoch
        self.lamda = 1 / (kwargs['batch_size'] * eps / self.models[0].output_size)

        self.optimizer = torch.optim.SGD(self.gather_all_param(*self.models, self.loss_func), lr=self.lr,
                                         momentum=momentum, weight_decay=weight_decay)

    def train(self, start=-1):
        self.prepare_batch_iter(0)
        # The schedules are used for controlling the learning rate, momentum, and lamda.
        self.momentum_schedule = self.cosine_scheduler(self.teacher_momentum, 1,
                                                       self.num_epoch, self.num_iter)
        self.lr_schedule = self.cosine_scheduler(self.lr, 0, self.num_epoch,
                                                 self.num_iter, warmup_epochs=self.warmup_epoch)
        self.lamda_schedule = self.lamda_scheduler(8/self.lamda, 1/self.lamda, self.num_epoch, self.num_iter,
                                                   warmup_epochs=self.warmup_epoch)
        self.train_epoches(start)

    def train_epoch(self, epoch_i):
        loss_log = []
        for batch_i, batch_meta in tqdm(enumerate(next_batch(shuffle(self.batch_iter), self.batch_size)),
                                        desc=f'{self.trainer_name} training {epoch_i+1}-th epoch',
                                        total=self.num_iter):
            it = self.num_iter * epoch_i + batch_i
            cur_lr = self.lr_schedule[it]
            lamda_inv = self.lamda_schedule[it]
            momentum = self.momentum_schedule[it]

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr

            self.optimizer.zero_grad()
            loss = self.loss_func(self.models, *self.prepare_batch_meta(batch_meta),
                                  lamda_inv=lamda_inv)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                for encoder, teacher in zip(self.models, self.loss_func.teachers):
                    for param_q, param_k in zip(encoder.parameters(), teacher.parameters()):
                        param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

            loss_log.append(loss.item())
        return float(np.mean(loss_log))

    @staticmethod
    def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule

    @staticmethod
    def lamda_scheduler(start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule


class NoneTrainer():
    def __init__(self, models, data, device):
        self.models = [model.to(device) for model in models]
        self.BASE_KEY = f'end2end/none/{data.name}'
        self.device = device
        self.model_save_dir = f'{data.base_path}/model_save/{self.BASE_KEY}'

    def save_models(self):
        """ Save all models. """
        create_if_noexists(self.model_save_dir)
        for model in self.models:
            save_path = f'{self.model_save_dir}/{model.name}.model'
            torch.save(model.state_dict(), save_path)

    def load_model(self, model):
        """ Load one of the encoder. """
        save_path = f'{self.model_save_dir}/{model.name}.model'
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        print('Load model from', save_path)
        return model

    def load_models(self):
        """ Load all encoders. """
        for i, model in enumerate(self.models):
            self.models[i] = self.load_model(model)

    def get_models(self):
        for model in self.models:
            model.eval()
        return self.models
