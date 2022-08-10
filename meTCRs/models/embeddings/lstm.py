import torch
from torch import nn
import torch.nn.functional as F

from meTCRs.models.embeddings.embedding import Embedding


class Lstm(Embedding):
    def __init__(self,
                 number_labels: int,
                 embedding_size: int,
                 hidden_size: int,
                 number_layers: int,
                 output_size: int,
                 *args,
                 **kwargs):
        super(Lstm, self).__init__(*args, **kwargs)

        if output_size is None:
            output_size = 0

        self._embedding = nn.Parameter(torch.randn((number_labels, embedding_size)))
        self._lstm = nn.LSTM(input_size=embedding_size,
                             hidden_size=hidden_size,
                             num_layers=number_layers,
                             proj_size=output_size,
                             batch_first=True)

    def forward(self, x):
        x = F.normalize(x.type(torch.float32), dim=-1)
        x = torch.matmul(x, self._embedding)
        _, (_, c_n) = self._lstm(x)
        return c_n[-1]