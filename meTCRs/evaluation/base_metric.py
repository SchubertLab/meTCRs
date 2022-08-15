import torch

from meTCRs.evaluation.utils import default_compare


class BaseMetric:
    def __init__(self, dist_type: str, compare=default_compare):
        self._dist_type = dist_type
        self._compare = compare

    def __call__(self, embedded_sequences: torch.Tensor, labels: list):
        raise NotImplementedError

    def _get_pairwise_distances(self, embedded_sequences: torch.Tensor):
        if self._dist_type == 'l2':
            pairwise_distances = torch.cdist(embedded_sequences, embedded_sequences)
        elif self._dist_type == 'random':
            length = len(embedded_sequences)
            pairwise_distances = torch.rand((length, length))
        else:
            raise NotImplementedError('Evaluation method not implemented for dist_type {}'.format(self._dist_type))

        return pairwise_distances

    def _get_match_matrix(self, labels, index_matrix, limit):
        match_matrix = torch.tensor([])

        for key, index_vector in enumerate(index_matrix):
            match_vector = self._create_match_vector(key, index_vector, labels, limit)
            match_matrix = torch.cat([match_matrix, match_vector], dim=0)

        return match_matrix

    def _create_match_vector(self, key, index_vector, labels, limit):
        match = []
        for idx in reversed(index_vector):
            idx = int(idx)
            if idx != key:
                match.append(self._compare(idx, key, labels))
        match_vector = torch.tensor([match[:limit]])
        return match_vector
