from bottleneck import argpartition


def recall_at_k(distance_matrix, labels, k_values):
    """
    Calculate the Recall at k values given a pairwise distance matrix
    :param distance_matrix: numpy array (n, n) with pairwise distances
    :param labels: list (n) with corresponding antigen specificity
    :param k_values: list of k values for R@k
    :return: dict {k_value: score} containing the evaluation score
    """
    recalls = {}
    for k in k_values:
        correct, count = 0., 0.
        for i in range(labels.shape[0]):
            distance_matrix[i, i] = 1e10
            k_neighbors = argpartition(distance_matrix[i], k)[:k]
            if any(labels[i] == labels[neighbor] for neighbor in k_neighbors):
                correct += 1
            count += 1
        recalls[k] = correct/count
    return recalls
