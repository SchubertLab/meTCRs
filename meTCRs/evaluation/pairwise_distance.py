from itertools import combinations

from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor


def pairwise_distance_evaluation(embedding, distance, data):
    sequences, epitopes = data
    embedded_sequences = embedding(sequences)
    pairs = []
    distances = []

    for i, j in combinations(range(len(data)), 2):
        pairs.append(0 if epitopes[i] == epitopes[j] else 1)
        distances.append(distance(embedded_sequences[i], embedded_sequences[j]))

    print(pairs, distances)

    score = roc_auc_score(y_true=pairs, y_score=distances)
    tpr, fpr, thresholds = roc_curve(y_true=pairs, y_score=distances)

    return score, tpr, fpr, thresholds
