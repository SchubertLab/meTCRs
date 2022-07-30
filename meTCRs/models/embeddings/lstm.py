import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam

from meTCRs.dataloader.utils.pair_maker import pair_maker


class Lstm(LightningModule):
    def __init__(self,
                 number_labels: int,
                 embedding_size: int,
                 hidden_size: int,
                 number_layers: int,
                 output_size: int,
                 loss,
                 optimizer_params: dict):
        super(Lstm, self).__init__()

        self.save_hyperparameters()

        if optimizer_params is None:
            self._optimizer_params = {}
        else:
            self._optimizer_params = optimizer_params

        if output_size is None:
            output_size = 0

        self._embedding = nn.Parameter(torch.randn((number_labels, embedding_size)))
        self._lstm = nn.LSTM(input_size=embedding_size,
                             hidden_size=hidden_size,
                             num_layers=number_layers,
                             proj_size=output_size,
                             batch_first=True)

        self._loss = loss

    def forward(self, x):
        x = F.normalize(x.type(torch.float32), dim=-1)
        x = torch.matmul(x, self._embedding)
        _, (_, c_n) = self._lstm(x)
        return c_n[-1]

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
