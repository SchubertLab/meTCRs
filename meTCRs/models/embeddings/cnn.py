import math

import torch
from torch import nn
from torch.optim import Adam

from pytorch_lightning import LightningModule

from meTCRs.dataloader.utils.pair_maker import pair_maker


class Cnn(LightningModule):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 number_labels: int,
                 embedding_size: int,
                 number_features: list[int],
                 kernel_sizes: list[int],
                 strides: list[int],
                 loss,
                 optimizer_params: dict):
        super(Cnn, self).__init__()

        self.save_hyperparameters()

        if optimizer_params is None:
            self._optimizer_params = {}
        else:
            self._optimizer_params = optimizer_params

        self._output_channels = input_size

        self._embedding = nn.Parameter(torch.randn((number_labels, embedding_size)))
        self._cnn_blocks = self._build_cnn_blocks(embedding_size, kernel_sizes, number_features, strides)
        self._output_layer = nn.Linear(number_features[-1] * self._output_channels, output_size)

        self._loss = loss

    def _build_cnn_blocks(self, embedding_size, kernel_sizes, number_features, strides):
        cnn_blocks = []
        input_channels = [embedding_size] + number_features[:-1]

        for _in, _out, _kernel, _stride in zip(input_channels, number_features, kernel_sizes, strides):
            _padding = (_kernel - 1) // 2

            self._output_channels = math.floor((self._output_channels + 2 * _padding - _kernel) / _stride + 1)

            block = [nn.Conv1d(_in, _out, _kernel, _stride, padding=_padding),
                     nn.BatchNorm1d(_out),
                     nn.ReLU(inplace=True)]

            cnn_blocks.append(nn.Sequential(*block))

        return nn.Sequential(*cnn_blocks)

    def forward(self, x):
        x = torch.matmul(x.type(torch.float32), self._embedding).permute(0, 2, 1)
        x = self._cnn_blocks(x)
        x = x.flatten(1)
        return self._output_layer(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), **self._optimizer_params)

    def _perform_step(self, batch):
        if self._loss is None:
            raise ValueError("`_perform_step` requires a loss function but loss is None")

        input_sequence, labels = batch
        embeddings = self(input_sequence.type(torch.float32))
        anchor1, positive, anchor2, negative = pair_maker(labels, embeddings)

        return self._loss(anchor1, positive, anchor2, negative)

    def training_step(self, batch, batch_index):
        loss = self._perform_step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_index):
        loss = self._perform_step(batch)
        self.log('val_loss', loss)



