from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.model_selection import train_test_split

from meTCRs.dataloader.dataset import TCREpitopeDataset

DATA_SEPARATOR = '\t'


class VDJdbDataModule(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, classes_per_batch: int):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.train_set = None
        self.val_set = None
        self.dimension = None

    def setup(self, stage: Optional[str] = None) -> None:
        raw_data = pd.read_csv(self.data_path, sep=DATA_SEPARATOR)

        self.dimension = self._get_dimension(raw_data)

        padded_cdr = self._pad(raw_data)
        cdr_tokens = self._tokenize(padded_cdr)
        cdr_token_ids = self._encode(cdr_tokens)

        epitopes = list(raw_data['Epitope'])

        tcr_train, tcr_val, epitope_train, epitope_val = train_test_split(cdr_token_ids,
                                                                          epitopes,
                                                                          train_size=0.8)

        self.train_set = TCREpitopeDataset(tcr_data=tcr_train,
                                           epitope_data=epitope_train,
                                           batch_size=self.batch_size,
                                           classes_per_batch=self.classes_per_batch,
                                           total_batches=len(tcr_train) // self.batch_size)

        self.val_set = TCREpitopeDataset(tcr_data=tcr_val,
                                         epitope_data=epitope_val,
                                         batch_size=self.batch_size,
                                         classes_per_batch=self.classes_per_batch,
                                         total_batches=len(tcr_val) // self.batch_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set,
                                           batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set,
                                           batch_size=self.batch_size)

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    @staticmethod
    def _encode(tokens):
        unique_tokens = sorted(set([t for element in tokens for t in element]))
        token_to_id = {t: id_ for id_, t in enumerate(unique_tokens)}
        token_ids = [[token_to_id[t] for t in token] for token in tokens]

        return token_ids

    @staticmethod
    def _tokenize(sequences):
        return sequences.apply(lambda x: list(x))

    def _pad(self, raw_data):
        return raw_data['CDR3'].apply(lambda x: x.ljust(self.dimension, '-'))

    @staticmethod
    def _get_dimension(raw_data):
        return max(raw_data['CDR3'].apply(lambda x: len(x)))
