"""
Main entry for training and evaluation.
"""

import json
import os
from argparse import ArgumentParser
import copy

from torch.cuda import is_available as cuda_available

from data import Data, Normalizer
from pretrain import trainer as PreTrainer, contrastive_losses
from model import sample as EncSampler, ode, transformer, induced_att
from downstream import predictor as DownPredictor, trainer as DownTrainer
import utils


if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--base', help='base path', type=str, default='sample')
    parser.add_argument('--dataset', help='dataset path', type=str, default='sample')
    parser.add_argument('-c', '--config', help='name of the config file to use', type=str, required=True)
    parser.add_argument('--device', help='name of the device to use', type=str, default='cuda:0' if cuda_available() else 'cpu')
    args = parser.parse_args()
    device = args.device

    # Load config file
    with open(f'config/{args.config}.json', 'r') as fp:
        config = json.load(fp)

    # Each config file can contain multiple entries. Each entry is a different set of configuration.
    for num_entry, entry in enumerate(config):
        print(f'\n{"=" * 30}\n===={num_entry+1}/{len(config)} experiment entry====')

        # Load dataset.
        data_entry = entry['data']
        data = Data(data_entry['name'], args.base, args.dataset)
        data.load_stat()
        num_roads = data.data_info['num_road']

        # Each entry can be repeated for several times.
        num_repeat = entry.get('repeat', 1)
        for repeat_i in range(num_repeat):
            print(f'\n----{num_entry+1}/{len(config)} experiment entry, {repeat_i+1}/{num_repeat} repeat----\n')

            models = []
            for model_entry in entry['models']:
                # Prepare samplers for models.
                if 'sampler' in model_entry:
                    sampler_entry = model_entry['sampler']
                    sampler_name = sampler_entry['name']
                    sampler_config = sampler_entry.get('config', {})

                    if sampler_name == 'khop':
                        sampler = EncSampler.KHopSampler(**sampler_config)
                    elif sampler_name == 'pass':
                        sampler = EncSampler.PassSampler()
                    elif sampler_name == 'index':
                        sampler = EncSampler.IndexSampler(**sampler_config)
                    elif sampler_name == 'pool':
                        sampler = EncSampler.PoolSampler(**sampler_config)
                    else:
                        raise NotImplementedError(f'No sampler called "{sampler_name}".')
                else:
                    sampler = EncSampler.PassSampler()

                # Create models.
                model_name = model_entry['name']
                model_config = model_entry.get('config', {})
                if model_name == 'ia':
                    models.append(induced_att.InducedAttEncoder(
                        sampler=sampler,
                        **model_config))
                elif model_name == 'transformer_encoder':
                    models.append(transformer.TransformerEncoder(
                        sampler=sampler,
                        **model_config))
                elif model_name == 'cde':
                    models.append(ode.CDEEncoder(sampler=sampler, **model_config))
                else:
                    raise NotImplementedError(f'No encoder called "{model_name}".')

            if 'pretrain' in entry:
                # Create pre-training loss function.
                pretrain_entry = entry['pretrain']
                loss_entry = pretrain_entry['loss']
                loss_name = loss_entry['name']

                loss_param = loss_entry.get('config', {})
                if loss_name == 'infonce':
                    loss_func = contrastive_losses.InfoNCE(**loss_param)
                elif loss_name == 'mec':
                    loss_func = contrastive_losses.MEC(**loss_param,
                                                       teachers=(copy.deepcopy(model) for model in models))
                else:
                    raise NotImplementedError(f'No loss function called "{loss_name}".')

                # Create pre-trainer.
                pretrainer_entry = pretrain_entry['trainer']
                pretrainer_name = pretrainer_entry['name']
                pretrainer_comm_param = {"data": data, "models": models, "loss_func": loss_func, "device": device}
                pretrainer_config = pretrainer_entry.get('config', {})
                if pretrainer_name == 'contrastive':
                    pre_trainer = PreTrainer.ContrastiveTrainer(
                        **pretrainer_comm_param,
                        **pretrainer_config)
                elif pretrainer_name == 'momentum':
                    pre_trainer = PreTrainer.MomentumTrainer(
                        **pretrainer_comm_param,
                        **pretrainer_config)
                else:
                    raise NotImplementedError(f'No loss function called "{pretrainer_name}".')

                # Pre-training on the training set, or load from trained cache.
                if pretrain_entry.get('load', False):
                    pre_trainer.load_models()
                else:
                    pre_trainer.train(pretrain_entry.get('resume', -1))

                models = pre_trainer.get_models()

                for i_model, model in enumerate(models):
                    model_entry = entry['models'][i_model]
                    # In case the model has a different sampler for downstream task.
                    if 'down_sampler' in model_entry:
                        down_sampler_entry = model_entry['down_sampler']
                        model.sampler = utils.load_sampler(
                            down_sampler_entry['name'], down_sampler_entry.get('config', {}))
            else:
                pre_trainer = PreTrainer.NoneTrainer(models=models, data=data, device=device)
                pre_trainer.save_models()
                print('Skip pretraining.')

            # Downstream evaluation
            if 'downstream' in entry:
                num_down = len(entry['downstream'])
                for down_i, down_entry in enumerate(entry['downstream']):
                    print(f'\n....{num_entry+1}/{len(config)} experiment entry, {repeat_i+1}/{num_repeat} repeat, '
                          f'{down_i+1}/{num_down} downstream task ....\n')

                    if down_i > 0:
                        pre_trainer.load_models()
                        models = pre_trainer.get_models()

                    down_models = [models[i] for i in down_entry['select_models']]
                    down_embed_size = sum([model.output_size for model in down_models])
                    down_task = down_entry['task']
                    down_config = down_entry.get('config', {})

                    down_comm_params = {
                        "data": data, "models": down_models, "device": device,
                        "base_name": pre_trainer.BASE_KEY}
                    predictor_entry = down_entry.get('predictor', {})
                    predictor_config = predictor_entry.get('config', {})
                    if down_task == 'classification':
                        data.load_class_meta()
                        predictor = DownPredictor.FCPredictor(
                            input_size=down_embed_size, output_size=data.data_info['num_class'],
                            **predictor_config)
                        down_trainer = DownTrainer.Classification(
                            predictor=predictor,
                            **down_comm_params, **down_config)
                    elif down_task == 'destination':
                        predictor = DownPredictor.FCPredictor(
                            input_size=down_embed_size, output_size=num_roads,
                            **predictor_config)
                        down_trainer = DownTrainer.Destination(
                            predictor=predictor,
                            **down_comm_params, **down_config)
                    elif down_task == 'search':
                        data.load_detour_meta(down_entry['config']['num_target'],
                                              down_entry['config']['detour_portion'])
                        predictor = DownPredictor.NonePredictor()
                        down_trainer = DownTrainer.Search(
                            predictor=predictor,
                            **down_comm_params, **down_config)
                    elif down_task == 'tte':
                        data.load_tte_meta()
                        predictor = DownPredictor.FCPredictor(
                            input_size=down_embed_size, output_size=1,
                            **predictor_config)
                        down_trainer = DownTrainer.TTE(
                            predictor=predictor,
                            **down_comm_params, **down_config)
                    else:
                        raise NotImplementedError(f'No downstream task called "{down_task}".')

                    if down_entry.get('load', False):
                        down_trainer.load_models()
                    else:
                        down_trainer.train()
                    down_trainer.eval(down_entry['eval_set'])
            else:
                print('Finishing program without performing downstream tasks.')
