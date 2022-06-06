from typing import Optional

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.nn.functional import one_hot

from meTCRs.dataloader.IEDB_processor import prepare_iedb
from meTCRs.dataloader.VDJdb_processor import prepare_vdjdb
from meTCRs.dataloader.dataset import TCREpitopeDataset
from meTCRs.dataloader.utils.amino_acids import AMINO_ACID_ENUMERATION

DATA_SEPARATOR = '\t'
CDR_SEQUENCE_KEY = 'CDR3b'
EPITOPE_KEY = 'Epitope'


class DataModule(LightningDataModule):
    def __init__(self, data_sets: list[dict], batch_size: int, encoding: str, class_sampling_method: str):
        super().__init__()
        self._data_sets = data_sets
        self._batch_size = batch_size
        self._classes_per_batch = batch_size // 2
        self._encoding = encoding
        self._train_set = None
        self._val_set = None
        self._dimension = None
        self._class_sampling_method = class_sampling_method

    def setup(self, stage: Optional[str] = None, debug=False, seed=1) -> None:
        data = self._concatenate_datasets()

        if debug:
            data = data.sample(frac=0.01, random_state=seed)

        raw_train_set, raw_val_set = self._train_val_split(data, train_fraction=0.8, random_state=seed)

        train_set = self._remove_small_class_data(raw_train_set, min_class_size=2)
        val_set = self._remove_small_class_data(raw_val_set, min_class_size=2)

        sequence_size = self._get_sequence_size(data)

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
        padded = self._pad(data[CDR_SEQUENCE_KEY], sequence_size)
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
        token_ids = torch.tensor([[AMINO_ACID_ENUMERATION[t] for t in token] for token in tokens])

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
        complement_mask = ~data.index.isin(train_set.index)
        val_set = data[complement_mask]

        return train_set, val_set

    @staticmethod
    def _remove_small_class_data(raw_data, min_class_size):
        class_sizes = raw_data.groupby(EPITOPE_KEY)[EPITOPE_KEY].transform(len)

        return raw_data[class_sizes >= min_class_size]

    def _concatenate_datasets(self):
        data_frames = []

        for data_set in self._data_sets:
            if data_set['source'] == 'IEDB':
                data_frames.append(prepare_iedb(data_set['file']))
            elif data_set['source'] == 'VDJdb':
                data_frames.append(prepare_vdjdb(data_set['file']))
            else:
                raise NotImplementedError('Cannot process dataset of source {}'.format(data_set['source']))

        data = pd.concat(data_frames)
        data.drop_duplicates(inplace=True)

        return data

