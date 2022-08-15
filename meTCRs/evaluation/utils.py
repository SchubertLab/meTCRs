import torch
from tqdm import tqdm

from meTCRs.dataloader.data_module import DataModule


def default_compare(i, j, labels):
    return labels[i] == labels[j]


def get_embedding(model, use_batched_data: bool, data_module: DataModule):
    if use_batched_data:
        embedded_sequences = torch.tensor([])
        labels = []
        for sequence_batch, label_batch in tqdm(iter(data_module.test_dataloader())):
            embedded_sequences = torch.cat([embedded_sequences, model(sequence_batch).detach()])
            labels += label_batch
        return embedded_sequences, labels
    else:
        sequences, labels = data_module.val_data
        return model(sequences), labels
