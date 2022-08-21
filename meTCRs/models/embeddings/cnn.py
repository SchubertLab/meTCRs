import math

import torch
import torch.nn.functional as F
from torch import nn

from meTCRs.models.embeddings.embedding import Embedding


class Cnn(Embedding):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 number_labels: int,
                 embedding_size: int,
                 number_features: list[int],
                 kernel_sizes: list[int],
                 strides: list[int],
                 *args,
                 **kwargs):
        super(Cnn, self).__init__(*args, **kwargs)

        self._output_channels = input_size

        self._embedding = nn.Parameter(torch.randn((number_labels, embedding_size)))
        self._cnn_blocks = self._build_cnn_blocks(embedding_size, kernel_sizes, number_features, strides)
        self._output_layer = nn.Linear(number_features[-1] * self._output_channels, output_size)

        self._output_size = output_size

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
        x = F.normalize(x.type(torch.float32), dim=-1)
        x = torch.matmul(x, self._embedding).permute(0, 2, 1)
        x = self._cnn_blocks(x)
        x = x.flatten(1)
        return self._output_layer(x)

    @property
    def output_size(self):
        return self._output_size
