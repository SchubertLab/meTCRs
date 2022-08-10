import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam

from meTCRs.dataloader.utils.pair_maker import pair_maker
from meTCRs.evaluation.mean_average_precision import MeanAveragePrecision


class Embedding(LightningModule):
    def __init__(self, loss, optimizer_params: dict, test_params: dict):
        super(Embedding, self).__init__()

        self.save_hyperparameters()

        if optimizer_params is None:
            self._optimizer_params = {}
        else:
            self._optimizer_params = optimizer_params

        if test_params is None:
            test_params = {}

        self._loss = loss
        self._test = MeanAveragePrecision(**test_params)

    def forward(self, *args, **kwargs) -> any:
        raise NotImplementedError

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

    def test_step(self, batch, batch_index):
        input_sequence, labels = batch
        return {'embeddings': self(input_sequence).detach(), 'labels': labels}

    def test_epoch_end(self, outputs):
        embedded_sequences = torch.cat([out['embeddings'] for out in outputs])
        labels = [label for out in outputs for label in out['labels']]

        test_result = self._test(embedded_sequences, labels)
        self.log('test_result', test_result)
        return test_result
