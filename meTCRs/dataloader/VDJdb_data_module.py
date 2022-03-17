from typing import Optional

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule

from meTCRs.dataloader.dataset import TCREpitopeDataset

DATA_SEPARATOR = '\t'


class VDJdbDataModule(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.dimension = None

    def setup(self, stage: Optional[str] = None) -> None:
        raw_data = pd.read_csv(self.data_path, sep=DATA_SEPARATOR)

        self.dimension = self._get_dimension(raw_data)

        padded_cdr = self._pad(raw_data)
        cdr_tokens = self._tokenize(padded_cdr)
        cdr_token_ids = self._encode(cdr_tokens)

        dataset = TCREpitopeDataset(tcr_data=cdr_token_ids, epitope_data=raw_data['Epitope'])
        self.train_set, self.val_set, self.test_set = self._split_dataset(dataset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, drop_last=True)

    # TODO Remove `drop_last` if proper sampling is used

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, drop_last=True)

    def predict_dataloader(self):
        pass

    @staticmethod
    def _split_dataset(dataset):
        train_set_len = int(len(dataset) * 0.6)
        val_set_len = int(len(dataset) * 0.2)
        test_set_len = len(dataset) - train_set_len - val_set_len

        return torch.utils.data.random_split(
            dataset,
            [train_set_len, val_set_len, test_set_len]
        )

    @staticmethod
    def _encode(tokens):
        unique_tokens = sorted(set([t for element in tokens for t in element]))
        token_to_id = {t: id_ for id_, t in enumerate(unique_tokens)}
        token_ids = [[token_to_id[t] for t in token] for token in tokens]

        return token_ids

    @staticmethod
    def _tokenize(padded_sequences):
        tokens = padded_sequences.apply(lambda x: list(x))
        return tokens

    def _pad(self, raw_data):
        padded_sequences = raw_data['CDR3'].apply(lambda x: x.ljust(self.dimension, '-'))
        return padded_sequences

    @staticmethod
    def _get_dimension(raw_data):
        dimension = max(raw_data['CDR3'].apply(lambda x: len(x)))
        return dimension
