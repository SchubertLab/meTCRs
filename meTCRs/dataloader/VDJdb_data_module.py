from typing import Optional

import torch
from torch.nn.functional import one_hot
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from meTCRs.dataloader.dataset import TCREpitopeDataset

DATA_SEPARATOR = '\t'
CDR_SEQUENCE_KEY = 'CDR3'


class VDJdbDataModule(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, encoding: str):
        super().__init__()
        self._data_path = data_path
        self._batch_size = batch_size
        self._classes_per_batch = batch_size // 2
        self._encoding = encoding
        self._train_set = None
        self._val_set = None
        self._dimension = None

    def setup(self, stage: Optional[str] = None, debug=False, seed=1) -> None:
        skip_rows = (lambda i: i > 0 and np.random.choice([True, False], p=[0.99, 0.01])) if debug else None

        raw_data = pd.read_csv(self._data_path, sep=DATA_SEPARATOR, skiprows=skip_rows)

        processed_cdr_sequences = self._process_cdr_sequences(raw_data)

        self._dimension = processed_cdr_sequences.shape[1]

        epitopes = list(raw_data['Epitope'])

        tcr_train, tcr_val, epitope_train, epitope_val = train_test_split(processed_cdr_sequences,
                                                                          epitopes,
                                                                          random_state=seed,
                                                                          train_size=0.8)

        self._train_set = TCREpitopeDataset(tcr_data=tcr_train,
                                            epitope_data=epitope_train,
                                            batch_size=self._batch_size,
                                            classes_per_batch=self._classes_per_batch,
                                            total_batches=len(tcr_train) // self._batch_size)

        self._val_set = TCREpitopeDataset(tcr_data=tcr_val,
                                          epitope_data=epitope_val,
                                          batch_size=self._batch_size,
                                          classes_per_batch=self._classes_per_batch,
                                          total_batches=len(tcr_val) // self._batch_size)

    def _process_cdr_sequences(self, raw_data):
        size = self._get_size(raw_data[CDR_SEQUENCE_KEY])
        trimmed = self._trim(raw_data[CDR_SEQUENCE_KEY])
        padded = self._pad(trimmed, size)
        tokenized = self._tokenize(padded)
        encoded = self._encode(tokenized)
        return encoded

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self._train_set,
                                           batch_size=self._batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self._val_set,
                                           batch_size=self._batch_size)

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def _encode(self, tokens):
        unique_tokens = sorted(set([t for element in tokens for t in element]))
        token_to_id = {t: id_ for id_, t in enumerate(unique_tokens)}
        token_ids = torch.tensor([[token_to_id[t] for t in token] for token in tokens])

        if self._encoding == 'ordinal':
            return token_ids
        elif self._encoding == 'one_hot':
            return one_hot(token_ids).flatten(1)
        else:
            raise NotImplementedError('Encoding of type {} is not defined.'.format(self._encoding))

    @staticmethod
    def _tokenize(sequences):
        return sequences.apply(lambda x: list(x))

    @staticmethod
    def _pad(input_data, size):
        return input_data.apply(lambda x: x.ljust(size, '-'))

    @staticmethod
    def _trim(input_data):
        return input_data.apply(lambda x: x[1:-1])

    @staticmethod
    def _get_size(input_data):
        return max(input_data.apply(lambda x: len(x)))

    @property
    def dimension(self):
        return self._dimension

    @property
    def val_data(self):
        return self._val_set.tcr_data, self._val_set.epitope_data

