import torch
from tqdm import tqdm

from meTCRs.dataloader.data_module import DataModule
from meTCRs.models.embeddings.embedding import Embedding


def default_compare(i, j, labels):
    return labels[i] == labels[j]


def get_embedding(model: Embedding, use_batched_data: bool, data_module: DataModule):
    model.eval()
    if use_batched_data:
        embedded_sequences = torch.tensor([])
        labels = []
        for sequence_batch, label_batch in tqdm(iter(data_module.test_dataloader())):
            with torch.no_grad():
                reconstruction = model(sequence_batch).detach()
            embedded_sequences = torch.cat([embedded_sequences, reconstruction])
            labels += label_batch
        return embedded_sequences, labels
    else:
        sequences, labels = data_module.val_data
        with torch.no_grad():
            reconstruction = model(sequences)
        return reconstruction, labels
