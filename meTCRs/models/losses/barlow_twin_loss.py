import torch
from torch.nn import Module


class BarlowTwinLoss(Module):
    """
    Taken from https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/barlow-twins.html
    :param batch_size: int, The batch size of the input tensors
    :param lmd: float, The regularization parameter lambda
    """
    def __init__(self, batch_size: int, lmd: float):
        super(BarlowTwinLoss).__init__()

        self._batch_size = batch_size
        self._lmd = lmd

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_correlation = torch.matmul(z1_norm.T, z2_norm) / self._batch_size

        diagonal_term = torch.sum((1 - torch.diagonal(cross_correlation))**2)
        off_diagonal_term = self._get_off_diagonal_term(cross_correlation)

        return diagonal_term + self._lmd * off_diagonal_term

    @staticmethod
    def _get_off_diagonal_term(cross_correlation: torch.Tensor):
        clone = cross_correlation.clone()
        return torch.sum(clone**2)
