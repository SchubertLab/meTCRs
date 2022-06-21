import itertools

import torch


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

    return _stack(anchor1), _stack(positive), _stack(anchor2), _stack(negative)


def _stack(tensors: list[torch.Tensor]):
    if len(tensors) == 0:
        return torch.tensor([])
    else:
        return torch.stack(tensors)
