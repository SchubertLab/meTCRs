from pytorch_lightning import LightningModule
from torch.optim import Adam

from meTCRs.dataloader.utils.pair_maker import pair_maker
from meTCRs.models.losses.barlow_twin_loss import BarlowTwinLoss


class BarlowTwins(LightningModule):
    def __init__(self, encoder, projection_head, batch_size, loss_lmd, optimizer_params):
        super(BarlowTwins, self).__init__()

        self._encoder = encoder
        self._projection_head = projection_head
        self._loss = BarlowTwinLoss(batch_size, loss_lmd)
        self._optimizer_params = optimizer_params

    def forward(self, x):
        return self._encoder(x)

    def training_step(self, batch, batch_idx):
        loss = self._perform_step(batch)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._perform_step(batch)
        self.log('val_loss', loss)

        return loss

    def _perform_step(self, batch):
        input_sequences, labels = batch
        x1, x2, _, _ = pair_maker(labels, input_sequences)

        z1 = self._projection_head(self._encoder(x1))
        z2 = self._projection_head(self._encoder(x2))

        return self._loss(z1, z2)

    def configure_optimizers(self):
        # TODO Consider warmup
        return Adam(self.parameters(), **self._optimizer_params)
