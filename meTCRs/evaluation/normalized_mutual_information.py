import numpy as np
import sklearn.cluster
import torch
import math


class Clustering:
    def __init__(self):
        pass

    def __call__(self, embedded_sequences: torch.Tensor, n_clusters: int) -> np.ndarray:
        pass


class KMeans(Clustering):
    def __call__(self, embedded_sequences: torch.Tensor, n_clusters: int):
        k_means = sklearn.cluster.KMeans(n_clusters)
        return k_means.fit_predict(embedded_sequences.detach().numpy())


class NormalizedMutualInformation:
    def __init__(self, clustering: Clustering = KMeans()):
        self._clustering = clustering

    def __call__(self, embedded_sequences: torch.Tensor, labels: list):
        clusters = self._clustering(embedded_sequences, n_clusters=len(set(labels)))
        mutual_information = self._mutual_information(clusters, labels)
        cluster_entropy = self._entropy(clusters)
        label_entropy = self._entropy(labels)

        return 2 * mutual_information / (cluster_entropy + label_entropy)

    @staticmethod
    def _mutual_information(clusters, classes):
        mutual_information = 0
        for cluster in set(clusters):
            cluster_indices = np.argwhere(np.array(clusters) == cluster).flatten()
            for cls in set(classes):
                cls_indices = np.argwhere(np.array(classes) == cls).flatten()
                intersection_size = len(set(cluster_indices).intersection(set(cls_indices)))
                frac = intersection_size / len(classes)
                if frac > 0:
                    exp = len(classes)*intersection_size / (len(cluster_indices)*len(cls_indices))
                    mutual_information += frac * math.log(exp)
        return mutual_information

    @staticmethod
    def _entropy(labels):
        entropy = 0
        for label in set(labels):
            label_indices = np.argwhere(np.array(labels) == label).flatten()
            frac = len(label_indices) / len(labels)
            if frac > 0:
                entropy -= frac * math.log(frac)
        return entropy
