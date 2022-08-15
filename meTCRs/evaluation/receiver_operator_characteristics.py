from itertools import product

import torch
from sklearn.metrics import roc_auc_score, roc_curve

from meTCRs.utils import function_timer


@function_timer
def roc_auc(embedding, dist_type, data, calculate_curve=False):
    sequences, epitopes = data
    embedded_sequences = embedding(sequences)

    if dist_type == 'l2':
        pairs = [0 if epitopes[i] == epitopes[j] else 1 for i, j in product(range(len(epitopes)), repeat=2)]
        distances = torch.cdist(embedded_sequences, embedded_sequences).detach().numpy().flatten(order='F')
    else:
        raise NotImplementedError('Pairwise distance evaluation not implemented for dist_type {}'.format(dist_type))

    results = {'score': roc_auc_score(y_true=pairs, y_score=distances)}

    if calculate_curve:
        results['tpr'], results['fpr'], results['thresholds'] = roc_curve(y_true=pairs, y_score=distances)

    return results
