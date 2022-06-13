import os.path

import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from meTCRs.evaluation.pairwise_distance import pairwise_distance_evaluation
from meTCRs.dataloader.data_module import DataModule
from meTCRs.models.distances.euclidean import Euclidean
from meTCRs.models.embeddings.barlow_twins import BarlowTwins
from meTCRs.models.embeddings.mlp import Mlp
from meTCRs.models.losses.contrastive_loss import ContrastiveLoss


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
        trainer_params: dict,
        early_stopping_params: dict,
        seed: int,
        debug: bool):
    set_seed(seed)

    data = setup_data(data_params, data_sets, debug, seed)
    distance = get_distance(dist_type)
    loss = get_loss(distance, loss_params, loss_type)
    model = get_model(loss, model_type, data.dimension, model_params, optimizer_params)
    trainer = get_trainer(save_path, trainer_params, early_stopping_params)

    trainer.fit(model, datamodule=data)

    evaluation_results = pairwise_distance_evaluation(model, dist_type, data.val_data)

    return evaluation_results['score']


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


def get_model(loss, model_type: str, number_inputs: int, model_params: dict, optimizer_params: dict):
    if model_type == 'mlp':
        model = Mlp(loss=loss, number_inputs=number_inputs, optimizer_params=optimizer_params, **model_params)
    elif model_type == 'barlow-twins':
        encoder = Mlp(number_inputs=number_inputs,
                      number_outputs=model_params['latent_space_dim'],
                      number_hidden=model_params['encoder_hidden_dims'])
        projection_head = Mlp(number_inputs=model_params['latent_space_dim'],
                              number_outputs=model_params['number_outputs'],
                              number_hidden=model_params['projection_head_hidden'])
        model = BarlowTwins(encoder, projection_head, batch_size= optimizer_params=optimizer_params)
    else:
        raise NotImplementedError("model of type {} is not implemented".format(model_type))
    return model


def get_loss(distance, loss_params: dict, loss_type: str):
    if loss_type == 'contrastive':
        loss = ContrastiveLoss(distance, **loss_params)
    else:
        raise NotImplementedError("loss of type {} is not implemented".format(loss_type))
    return loss


def get_distance(dist_type: str):
    if dist_type == 'l2':
        distance = Euclidean()
    else:
        raise NotImplementedError("distance of type {} is not implemented".format(dist_type))
    return distance
