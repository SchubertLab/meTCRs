from pytorch_lightning import LightningModule
from torch import nn, float32
from torch.optim import Adam

from meTCRs.dataloader.utils.pair_maker import pair_maker
from meTCRs.models.utils.losses import ContrastiveLoss


class Mlp(LightningModule):
    def __init__(self, number_inputs, number_outputs, number_hidden):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(number_inputs, number_hidden),
            nn.ReLU(),
            nn.Linear(number_hidden, number_hidden),
            nn.ReLU(),
            nn.Linear(number_hidden, number_outputs)
        )

        self.loss = ContrastiveLoss()

    def configure_optimizers(self):
        return Adam(self.parameters())

    def forward(self, x):
        return self.model(x.type(float32))

    def training_step(self, batch, batch_index):
        input_sequence, labels = batch

        embeddings = self.model(input_sequence.type(float32))

        anchor1, positive, anchor2, negative = pair_maker(labels, embeddings)

        return self.loss(anchor1, positive, anchor2, negative)


