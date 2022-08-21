import itertools

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam

from meTCRs.dataloader.utils.pair_maker import pair_maker
from meTCRs.evaluation.mean_average_precision import MeanAveragePrecision
from meTCRs.models.embeddings.embedding import Embedding
from meTCRs.models.losses.barlow_twin_loss import BarlowTwinLoss


class FullyConnected(LightningModule):
    def __init__(self, input_dimension: int, hidden_size: int, output_dimension: int):
        super(FullyConnected, self).__init__()

        self._model = nn.Sequential(
            nn.Linear(input_dimension, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dimension)
        )

    def forward(self, x):
        return self._model(x.type(torch.float32))


class BarlowTwins(LightningModule):
    def __init__(self,
                 encoder: Embedding,
                 projector_hidden_size: int,
                 projector_output_size: int,
                 evaluation_layer_hidden_size: int,
                 evaluation_layer_output_size: int,
                 barlow_lmd: float,
                 barlow_regulator: float,
                 metric_loss,
                 optimizer_params,
                 test_params):
        super(BarlowTwins, self).__init__()

        self.save_hyperparameters(ignore=['encoder'])

        self._encoder = encoder
        self._projection_head = FullyConnected(encoder.output_size, projector_hidden_size, projector_output_size)
        self._evaluation_layer = FullyConnected(encoder.output_size,
                                                evaluation_layer_hidden_size,
                                                evaluation_layer_output_size)

        self._barlow_loss = BarlowTwinLoss(barlow_lmd, barlow_regulator)
        self._metric_loss = metric_loss

        self._optimizer_params = optimizer_params

        self._test = MeanAveragePrecision(**test_params)

    def forward(self, x):
        return self._evaluation_layer(self._encoder(x))

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            barlow_loss = self._perform_barlow_step(batch)
            self.log('train_barlow_loss', barlow_loss)

            return barlow_loss

        if optimizer_idx == 1:
            metric_loss = self._perform_evaluation_layer_step(batch)
            self.log('train_metric_loss', metric_loss)

            return metric_loss

        if optimizer_idx is None or optimizer_idx > 1:
            raise IndexError('No training step for optimizer_idx {} available'.format(optimizer_idx))

    def validation_step(self, batch, batch_idx):
        loss = self._perform_evaluation_layer_step(batch)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_index):
        input_sequence, labels = batch
        return {'embeddings': self(input_sequence).detach(), 'labels': labels}

    def test_epoch_end(self, outputs):
        embedded_sequences = torch.cat([out['embeddings'] for out in outputs])
        labels = [label for out in outputs for label in out['labels']]

        test_result = self._test(embedded_sequences, labels)
        self.log('test_result', test_result)
        return test_result

    def _perform_barlow_step(self, batch):
        input_sequences, labels = batch
        embeddings = self._projection_head(self._encoder(input_sequences))
        z1, z2, _, _ = pair_maker(labels, embeddings)

        return self._barlow_loss(z1, z2, _, _)

    def _perform_evaluation_layer_step(self, batch):
        input_sequences, labels = batch

        with torch.no_grad():
            embeddings = self._encoder(input_sequences).detach()

        embeddings = self._evaluation_layer(embeddings)

        z_a1, z_p, z_a2, z_n = pair_maker(labels, embeddings)

        return self._metric_loss(z_a1, z_p, z_a2, z_n)

    def configure_optimizers(self):
        barlow_params = [self._encoder.parameters(), self._projection_head.parameters()]
        barlow_optimizer = Adam(itertools.chain(*barlow_params), **self._optimizer_params)
        metric_optimizer = Adam(self._evaluation_layer.parameters(), **self._optimizer_params)

        return barlow_optimizer, metric_optimizer
