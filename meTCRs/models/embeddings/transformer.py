import math

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam

from meTCRs.dataloader.utils.pair_maker import pair_maker


class PositionalEncoding(nn.Module):
    """
    Taken from Pytorch Lightning Tutorial 5: Transformers and Multi-Head Attention
    https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
    """

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(1).transpose(0, 1)

        self.register_parameter("positional_encoding", nn.Parameter(positional_encoding, requires_grad=False))

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1)]


class TransformerEncoder(LightningModule):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 number_labels: int,
                 embedding_size: int,
                 number_heads: int,
                 forward_expansion: int,
                 number_layers: int,
                 loss,
                 optimizer_params: dict):
        super(TransformerEncoder, self).__init__()

        self.save_hyperparameters()

        if optimizer_params is None:
            self._optimizer_params = {}
        else:
            self._optimizer_params = optimizer_params

        self._embedding = nn.Parameter(torch.randn((number_labels, embedding_size)))

        self._positional_encoding = PositionalEncoding(embedding_size, input_size)

        encoding_layer = nn.TransformerEncoderLayer(d_model=embedding_size,
                                                    nhead=number_heads,
                                                    dim_feedforward=embedding_size * forward_expansion,
                                                    batch_first=True)

        self._transformer_encoder = nn.TransformerEncoder(encoding_layer, number_layers)

        self._reduction = nn.Linear(input_size * embedding_size, output_size)

        self._loss = loss

    def forward(self, x):
        x = F.normalize(x.type(torch.float32), dim=-1)
        x = torch.matmul(x.type(torch.float32), self._embedding)
        x = self._positional_encoding(x)
        x = self._transformer_encoder(x)
        x = x.flatten(1)
        return self._reduction(x)

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
