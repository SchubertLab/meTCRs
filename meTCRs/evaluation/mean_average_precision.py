import torch

from meTCRs.dataloader.data_module import DataModule


class MeanAveragePrecision:
    def __init__(self, dist_type: str, R: int, data: DataModule):
        self._dist_type = dist_type
        self._r = R
        self._data = data
        self._labels = None

    def __call__(self, model, use_batched_data: True):
        embedded_sequences, labels = self._get_embedding(model, use_batched_data)
        self._labels = labels

        if self._dist_type == 'l2':
            pairwise_distances = torch.cdist(embedded_sequences, embedded_sequences)
        elif self._dist_type == 'random':
            length = len(embedded_sequences)
            pairwise_distances = torch.rand((length, length))
        else:
            raise NotImplementedError('MAP evaluation not implemented for dist_type {}'.format(self._dist_type))

        knn_tensor = pairwise_distances.topk(k=self._r + 1, dim=1, largest=False, sorted=True).indices

        return float(torch.mean(torch.sum(self._map_at_r(knn_tensor), dim=1)))

    def _get_embedding(self, model, use_batched_data):
        if use_batched_data:
            embedded_sequences = torch.tensor([])
            labels = []
            for sequence_batch, label_batch in iter(self._data.test_dataloader()):
                embedded_sequences = torch.cat([embedded_sequences, model(sequence_batch)])
                labels += label_batch
            return embedded_sequences, labels
        else:
            sequences, labels = self._data.val_data
            return model(sequences), labels

    def _map_at_r(self, knn_tensor):
        match_matrix = torch.tensor([])

        for key, knn in enumerate(knn_tensor):
            match_vector = self._create_match_vector(key, knn)
            match_matrix = torch.cat([match_matrix, match_vector], dim=0)

        weight_matrix = torch.triu(torch.stack([1 / r * torch.ones(self._r) for r in reversed(range(1, self._r + 1))]))

        return torch.matmul(match_matrix, weight_matrix) * match_matrix / self._r

    def _create_match_vector(self, key, knn):
        match = []
        for idx in reversed(knn):
            idx = int(idx)
            if idx != key:
                match.append(self._labels[idx] == self._labels[key])
        match_vector = torch.tensor([match[:self._r]])
        return match_vector

    def _compare(self, i, j):
        return self._labels[i] == self._labels[j]
