import math
from typing import List

import torch
from pytorch_lightning import LightningModule
from torch import nn, float32
from torch.optim import Adam

from meTCRs.dataloader.utils.pair_maker import pair_maker


class Mlp(LightningModule):
    def __init__(self,
                 input_dimension: torch.Size,
                 number_outputs: int,
                 number_hidden: List[int],
                 loss=None,
                 optimizer_params=None):
        super().__init__()
        self.save_hyperparameters()

        if optimizer_params is None:
            self._optimizer_params = {}
        else:
            self._optimizer_params = optimizer_params

        number_inputs = math.prod(input_dimension)

        self._model = self._setup_model(number_inputs, number_outputs, number_hidden)

        self._loss = loss

    def configure_optimizers(self):
        return Adam(self.parameters(), **self._optimizer_params)

    def forward(self, x):
        x = x.flatten(1)
        return self._model(x.type(float32))

    def training_step(self, batch, batch_index):
        loss = self._perform_step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_index):
        loss = self._perform_step(batch)
        self.log('val_loss', loss)

    def _perform_step(self, batch):
        if self._loss is None:
            raise ValueError("`_perform_step` requires a loss function but loss is None")

        input_sequence, labels = batch
        embeddings = self(input_sequence.type(float32))
        anchor1, positive, anchor2, negative = pair_maker(labels, embeddings)

        return self._loss(anchor1, positive, anchor2, negative)

    @staticmethod
    def _setup_model(number_inputs: int, number_outputs: int, number_hidden: List[int]):
        layers = [nn.Linear(number_inputs, number_hidden[0]), nn.ReLU(), nn.BatchNorm1d(number_hidden[0])]
        for i in range(len(number_hidden)-1):
            layers += [nn.Linear(number_hidden[i], number_hidden[i+1]), nn.ReLU(), nn.BatchNorm1d(number_hidden[i+1])]
        layers += [nn.Linear(number_hidden[-1], number_outputs)]

        return nn.Sequential(*layers)


