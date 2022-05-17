import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.insert(0, os.pardir)

from meTCRs.evaluation.pairwise_distance import pairwise_distance_evaluation
from meTCRs.dataloader.VDJdb_data_module import VDJdbDataModule
from meTCRs.models.distances.euclidean import Euclidean
from meTCRs.models.embeddings.mlp import Mlp
from meTCRs.models.losses.contrastive_loss import ContrastiveLoss


def run(save_path: str,
        data_path: str,
        data_params: dict,
        dist_type: str,
        loss_type: str,
        loss_params: dict,
        model_type: str,
        model_params: dict,
        optimizer_params: dict,
        trainer_params: dict,
        early_stopping: bool,
        debug: bool):
    data = setup_data(data_params, data_path, debug)
    distance = get_distance(dist_type)
    loss = get_loss(distance, loss_params, loss_type)
    model = get_model(loss, model_type, data.dimension, model_params, optimizer_params)
    trainer = get_trainer(save_path, trainer_params, early_stopping)

    trainer.fit(model, datamodule=data)

    evaluation_results = pairwise_distance_evaluation(model, distance, data.val_data)

    return evaluation_results['score']


def get_trainer(save_path: str, trainer_params: dict, early_stopping: bool):
    if early_stopping:
        trainer = Trainer(**trainer_params,
                          default_root_dir=save_path,
                          callbacks=[EarlyStopping(monitor='val_loss', mode='min')])
    else:
        trainer = Trainer(**trainer_params, default_root_dir=save_path)

    return trainer


def setup_data(data_params: dict, data_path: str, debug: bool):
    data = VDJdbDataModule(data_path, **data_params)
    data.setup(debug=debug)
    return data


def get_model(loss, model_type: str, number_inputs: int, model_params: dict, optimizer_params: dict):
    if model_type == 'mlp':
        model = Mlp(loss=loss, number_inputs=number_inputs, optimizer_params=optimizer_params, **model_params)
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
