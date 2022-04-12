import itertools

import torch
from torch.nn import MSELoss


def pair_maker(labels, embeddings):
    pair_indices = list(itertools.combinations(range(len(labels)), 2))

    anchor1 = []
    positive = []
    anchor2 = []
    negative = []

    for i, j in pair_indices:
        if labels[i] == labels[j]:
            anchor1.append(embeddings[i])
            positive.append(embeddings[j])
        else:
            anchor2.append(embeddings[i])
            negative.append(embeddings[j])

    return torch.stack(anchor1), torch.stack(positive), torch.stack(anchor2), torch.stack(negative)
