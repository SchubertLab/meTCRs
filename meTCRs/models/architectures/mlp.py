from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam

from meTCRs.models.utils.losses import contrastive_loss


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

    def configure_optimizers(self):
        return Adam(self.parameters())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_index):
        input_sequence, labels = batch

        batch_size, _ = input_sequence.shape

        assert(batch_size % 2 == 0)

        embeddings_1 = self.model(input_sequence[:batch_size//2].float())
        embeddings_2 = self.model(input_sequence[batch_size//2:].float())

        labels_1 = labels[:batch_size//2]
        labels_2 = labels[batch_size//2:]

        return contrastive_loss((embeddings_1, embeddings_2), (labels_1, labels_2))


