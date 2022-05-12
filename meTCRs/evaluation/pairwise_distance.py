from itertools import product

from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import torch

from meTCRs.utils import function_timer


@function_timer
def pairwise_distance_evaluation(embedding, distance, data, calculate_curve=False):
    sequences, epitopes = data
    embedded_sequences = embedding(torch.Tensor(sequences))

    # TODO Find better solution for pair building. This is due to the necessary transformations of the distance matrix.
    pairs = [0 if epitopes[i] == epitopes[j] else 1 for i, j in product(range(len(epitopes)), repeat=2)]

    # TODO Find solution for generic distances
    distances = torch.cdist(embedded_sequences, embedded_sequences).detach().numpy().flatten(order='F')

    results = {'score': roc_auc_score(y_true=pairs, y_score=distances)}

    if calculate_curve:
        results['tpr'], results['fpr'], results['thresholds'] = roc_curve(y_true=pairs, y_score=distances)

    return results
