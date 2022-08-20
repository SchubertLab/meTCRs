import torch


class MeanAveragePrecision:
    def __init__(self, dist_type: str, R: int):
        self._dist_type = dist_type
        self._r = R

    def __call__(self, embedded_sequences: torch.Tensor, labels: list):
        if self._dist_type == 'l2':
            pairwise_distances = torch.cdist(embedded_sequences, embedded_sequences)
        elif self._dist_type == 'random':
            length = len(embedded_sequences)
            pairwise_distances = torch.rand((length, length))
        else:
            raise NotImplementedError('MAP evaluation not implemented for dist_type {}'.format(self._dist_type))

        knn_tensor = pairwise_distances.topk(k=self._r + 1, dim=1, largest=False, sorted=True).indices

        return torch.mean(torch.sum(self._map_at_r(knn_tensor, labels), dim=1))

    def _map_at_r(self, knn_tensor, labels):
        match_matrix = torch.tensor([])

        for key, knn in enumerate(knn_tensor):
            match_vector = self._create_match_vector(key, knn, labels)
            match_matrix = torch.cat([match_matrix, match_vector], dim=0)

        weight_matrix = torch.triu(torch.stack([1 / r * torch.ones(self._r) for r in reversed(range(1, self._r + 1))]))

        return torch.matmul(match_matrix, weight_matrix) * match_matrix / self._r

    def _create_match_vector(self, key, knn, labels):
        match = []
        for idx in reversed(knn):
            idx = int(idx)
            if idx != key:
                match.append(self._compare(idx, key, labels))
        match_vector = torch.tensor([match[:self._r]])
        return match_vector

    @staticmethod
    def _compare(i, j, labels):
        return labels[i] == labels[j]
