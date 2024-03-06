# Pre-training General Trajectory Embeddings with Maximum Multi-view Entropy Coding

Paper published to TKDE: [Pre-print](https://arxiv.org/abs/2207.14539), [Early-access](https://ieeexplore.ieee.org/abstract/document/10375102/)

![Framework](framework.webp)

The published code not only includes the implementation of MMTEC, but is also a general framework for pre-training and evaluating trajectory representations. It should be easy to implement new pre-training trajectory representation method based on this framework.

## Pre-trainer and pretext loss

The pre-trainers and pretext losses are included in the `/pretrain` directory.

### Pre-trainer

`/pretrain/trainer.py` includes commonly used pre-trainers. 

- The `Trainer` class is an abstract class, with implementation of the common functions: fetching mini-batches, feed mini-batches into loss functions, save and load the pre-trained models.
- The `ContrastiveTrainer` class is the trainer for contrastive-style pre-training. As for now, it doesn't include any additional function to the `Trainer` class.
- The `GenerativeTrainer` class is the trainer for generative-style pre-training. It includes a `generation` function that can be used to evaluate the generation accuracy of a trained encoder-decoder pair.
- The `MomentumTrainer` can be regarded as a special version of contrastive-style pre-trainer. It implements a momentum training scheme, with student-teacher pairs.
- The `NoneTrainer` is reserved for end-to-end training scenarios.

### Pretext loss

As their names suggest, `contrastive_losses.py` and `generative_losses.py` store contrastive- and generative-style loss functions. The loss functions have to obey two basic standards:

- They need to be a subclass of `torch.nn.Module`. This is because some loss functions may include extra discriminators or predictors.
- The `forward` function is the implementation of the loss's calculation.

We already include some widely used and SOTA pretext losses. For contrastive losses, we include the Maximum Entropy Coding loss and the InfoNCE loss. For generative losses, we include the Auto-regressive loss and the MLM loss.

## Representation model

The encoder models and decoder models (if applicable) are all stored in the `/model` directory. Noted that samplers for the encoders are stored in `/model/sample.py`.

## Downstream tasks

We include four downstream tasks for evaluating the performance of pre-training representation methods. In `downstream/trainer.py`, `Classification` class implements the classification task, `Destination` implements the destination prediction task, `Search` implements the similar trajectory search task, `TTE` implements the travel time estimation task.

You can also add your own tasks, just implement a new downstream trainer based on the abstract class `Trainer`. To add your own predictor for the downstream tasks, just add a new model to `downstream/predictor.py`.

## Dataset

The `Data` class in `data.py` is a helper class for dataset pre-processing. Since fetching trajectory sequences and calculating labels for downstream tasks is time-consuming, we advise pre-calculate and store them before experiments. You can do this by directly run the `data.py` through Python.

Note that the storage directory of metadata, model parameters and results are controlled by the `Data.base_path` parameter. Change it according to your specific running environment.

## The configuration file system

You can use configuration files to control all the parameters in experiments. The config files are all stored in `/config` directory, and are all JSON files.

During experiments, use the following command line arguments to specify a config file:

```bash
python main.py -c <config_file_path> --cuda <cuda_device_index>
```

## Localized, small-scale test and debug

If "small" is contained in the dataset name, the dataset source files and the results will be all contained in the `/sample` directory.

A sample version of the Chengdu and Xian datasets are provided, enabling faster local debug.