from typing import Optional

import torch
from torch.nn.functional import one_hot
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule

from meTCRs.dataloader.dataset import TCREpitopeDataset

DATA_SEPARATOR = '\t'
CDR_SEQUENCE_KEY = 'CDR3'
EPITOPE_KEY = 'Epitope'


class VDJdbDataModule(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, encoding: str, class_sampling_method: str):
        super().__init__()
        self._data_path = data_path
        self._batch_size = batch_size
        self._classes_per_batch = batch_size // 2
        self._encoding = encoding
        self._train_set = None
        self._val_set = None
        self._dimension = None
        self._class_sampling_method = class_sampling_method

    def setup(self, stage: Optional[str] = None, debug=False, seed=1) -> None:
        skip_rows = (lambda i: i > 0 and np.random.choice([True, False], p=[0.99, 0.01])) if debug else None

        raw_data = pd.read_csv(self._data_path, sep=DATA_SEPARATOR, skiprows=skip_rows)

        raw_train_set, raw_val_set = self._train_val_split(raw_data, train_fraction=0.8, random_state=seed)

        train_set = self._remove_small_class_data(raw_train_set, min_class_size=2)
        val_set = self._remove_small_class_data(raw_val_set, min_class_size=2)

        sequence_size = self._get_sequence_size(raw_data)

        processed_cdr_train_sequences = self._process_cdr_sequences(train_set, sequence_size)
        processed_cdr_val_sequences = self._process_cdr_sequences(val_set, sequence_size)

        self._dimension = processed_cdr_train_sequences.shape[1]

        self._train_set = TCREpitopeDataset(tcr_data=processed_cdr_train_sequences,
                                            epitope_data=train_set[EPITOPE_KEY],
                                            batch_size=self._batch_size,
                                            classes_per_batch=self._classes_per_batch,
                                            total_batches=len(processed_cdr_train_sequences) // self._batch_size,
                                            class_sampling_method=self._class_sampling_method)

        self._val_set = TCREpitopeDataset(tcr_data=processed_cdr_val_sequences,
                                          epitope_data=val_set[EPITOPE_KEY],
                                          batch_size=self._batch_size,
                                          classes_per_batch=self._classes_per_batch,
                                          total_batches=len(processed_cdr_val_sequences) // self._batch_size,
                                          class_sampling_method=self._class_sampling_method)

    def _process_cdr_sequences(self, data, sequence_size: int):
        trimmed = self._trim(data[CDR_SEQUENCE_KEY])
        padded = self._pad(trimmed, sequence_size)
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

    @property
    def dimension(self):
        return self._dimension

    @property
    def val_data(self):
        return self._val_set.tcr_data, self._val_set.epitope_list

    @staticmethod
    def _get_sequence_size(raw_data):
        return max(raw_data[CDR_SEQUENCE_KEY].apply(lambda x: len(x)))

    @staticmethod
    def _train_val_split(data: pd.DataFrame, train_fraction: float, random_state: int):
        train_set = data.sample(frac=train_fraction, random_state=random_state)
        complement_mask = ~data.index.isin(train_set)
        val_set = data[complement_mask]

        return train_set, val_set

    @staticmethod
    def _remove_small_class_data(raw_data, min_class_size):
        class_sizes = raw_data.groupby(EPITOPE_KEY)[EPITOPE_KEY].transform(len)

        return raw_data[class_sizes >= min_class_size]
