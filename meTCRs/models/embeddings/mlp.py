import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from meTCRs.models.embeddings.embedding import Embedding


class Mlp(Embedding):
    def __init__(self,
                 input_dimension: torch.Size,
                 number_outputs: int,
                 number_hidden: List[int],
                 *args,
                 **kwargs):
        super(Mlp, self).__init__(*args, **kwargs)

        number_inputs = math.prod(input_dimension)

        self._model = self._setup_model(number_inputs, number_outputs, number_hidden)

        self._output_size = number_outputs

    def forward(self, x):
        x = F.normalize(x.type(torch.float32), dim=-1)
        x = x.flatten(1)
        return self._model(x)

    @staticmethod
    def _setup_model(number_inputs: int, number_outputs: int, number_hidden: List[int]):
        layers = [nn.Linear(number_inputs, number_hidden[0]), nn.ReLU(), nn.BatchNorm1d(number_hidden[0])]
        for i in range(len(number_hidden)-1):
            layers += [nn.Linear(number_hidden[i], number_hidden[i+1]), nn.ReLU(), nn.BatchNorm1d(number_hidden[i+1])]
        layers += [nn.Linear(number_hidden[-1], number_outputs)]

        return nn.Sequential(*layers)

    @property
    def output_size(self):
        return self._output_size


