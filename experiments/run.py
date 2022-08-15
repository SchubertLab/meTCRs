import argparse
import os.path
import sys

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from meTCRs.models.embeddings.transformer import TransformerEncoder

sys.path.append(os.path.join(sys.path[0], '..'))

from meTCRs.dataloader.data_module import DataModule
from meTCRs.models.distances.euclidean import Euclidean
from meTCRs.models.embeddings.cnn import Cnn
from meTCRs.models.embeddings.mlp import Mlp
from meTCRs.models.embeddings.lstm import Lstm
from meTCRs.models.losses.barlow_twin_loss import BarlowTwinLoss
from meTCRs.models.losses.contrastive_loss import ContrastiveLoss


parser = argparse.ArgumentParser(description='Run experiments without hyperparameter optimization')
parser.add_argument('configuration_file', type=str)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run(save_path: str,
        data_sets: list[dict],
        data_params: dict,
        dist_type: str,
        loss_type: str,
        loss_params: dict,
        model_type: str,
        model_params: dict,
        optimizer_params: dict,
        test_params: dict,
        trainer_params: dict,
        early_stopping_params: dict,
        seed: int,
        debug: bool):
    set_seed(seed)

    data = setup_data(data_params, data_sets, debug, seed)
    distance = get_distance(dist_type)
    loss = get_loss(distance, loss_params, loss_type)
    model = get_model(loss, model_type, data.dimension, model_params, optimizer_params, test_params)
    trainer = get_trainer(save_path, trainer_params, early_stopping_params)

    trainer.fit(model, datamodule=data)
    result = trainer.test(datamodule=data)

    return result[0]['test_result']


def get_trainer(save_path: str, trainer_params: dict, early_stopping_params: dict):
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", **early_stopping_params)
    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, every_n_epochs=1)
    trainer = Trainer(**trainer_params,
                      default_root_dir=save_path,
                      callbacks=[early_stopping, model_checkpoint])

    return trainer


def setup_data(data_params: dict, data_sets: list[dict], debug: bool, seed: int):
    for data_set in data_sets:
        data_set['file'] = os.path.join(os.path.dirname(__file__), '..', 'data', data_set['file'])
    data = DataModule(data_sets, **data_params)
    data.setup(debug=debug, seed=seed)
    return data


def get_model(loss, model_type: str, input_dimension: torch.Size, model_params: dict, optimizer_params: dict, test_params: dict):
    if model_type == 'mlp':
        model = Mlp(loss=loss,
                    input_dimension=input_dimension,
                    optimizer_params=optimizer_params,
                    test_params=test_params,
                    **model_params)
    elif model_type == 'cnn':
        model = Cnn(input_size=input_dimension[0],
                    number_labels=input_dimension[1],
                    optimizer_params=optimizer_params,
                    test_params=test_params,
                    loss=loss,
                    **model_params)
    elif model_type == 'lstm':
        model = Lstm(loss=loss,
                     number_labels=input_dimension[1],
                     optimizer_params=optimizer_params,
                     test_params=test_params,
                     **model_params)
    elif model_type == 'transformer':
        model = TransformerEncoder(loss=loss,
                                   input_size=input_dimension[0],
                                   number_labels=input_dimension[1],
                                   optimizer_params=optimizer_params,
                                   test_params=test_params,
                                   **model_params)
    else:
        raise NotImplementedError("model of type {} is not implemented".format(model_type))
    return model


def get_loss(distance, loss_params: dict, loss_type: str):
    if loss_type == 'contrastive':
        loss = ContrastiveLoss(distance, **loss_params)
    elif loss_type == 'barlow-twin':
        loss = BarlowTwinLoss(**loss_params)
    else:
        raise NotImplementedError("loss of type {} is not implemented".format(loss_type))
    return loss


def get_distance(dist_type: str):
    if dist_type == 'l2':
        distance = Euclidean()
    else:
        raise NotImplementedError("distance of type {} is not implemented".format(dist_type))
    return distance


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.configuration_file, 'r') as f:
        config = yaml.safe_load(f)

    score = run(**config)

    print('Experiment finished with score: {}'.format(score))
