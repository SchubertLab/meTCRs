import torch

from meTCRs.evaluation.base_metric import BaseMetric


class Precision(BaseMetric):
    def __init__(self, k: int, *args, **kwargs):
        super(Precision, self).__init__(*args, **kwargs)

        self._k = k

    def __call__(self, embedded_sequences: torch.Tensor, labels: list):
        pairwise_distances = self._get_pairwise_distances(embedded_sequences)
        knn_tensor = pairwise_distances.topk(k=self._k + 1, dim=1, largest=False, sorted=True).indices
        match_matrix = self._get_match_matrix(labels, index_matrix=knn_tensor, limit=self._k)

        return torch.mean(match_matrix)

