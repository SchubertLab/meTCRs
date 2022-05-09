from pytorch_lightning import LightningModule
from torch import nn, float32
from torch.optim import Adam

from meTCRs.dataloader.utils.pair_maker import pair_maker


class Mlp(LightningModule):
    def __init__(self, loss, number_inputs, number_outputs, number_hidden, optimizer_params=None):
        super().__init__()

        if optimizer_params is None:
            self._optimizer_params = {}
        else:
            self._optimizer_params = optimizer_params

        self.model = nn.Sequential(
            nn.Linear(number_inputs, number_hidden),
            nn.BatchNorm1d(number_hidden),
            nn.ReLU(),
            nn.Linear(number_hidden, number_hidden),
            nn.BatchNorm1d(number_hidden),
            nn.ReLU(),
            nn.Linear(number_hidden, number_outputs)
        )

        self.loss = loss

    def configure_optimizers(self):
        return Adam(self.parameters(), **self._optimizer_params)

    def forward(self, x):
        return self.model(x.type(float32))

    def training_step(self, batch, batch_index):
        return self._perform_step(batch)

    def validation_step(self, batch, batch_index):
        loss = self._perform_step(batch)
        self.log('val_loss', loss)

    def _perform_step(self, batch):
        input_sequence, labels = batch
        embeddings = self.model(input_sequence.type(float32))
        anchor1, positive, anchor2, negative = pair_maker(labels, embeddings)
        return self.loss(anchor1, positive, anchor2, negative)


