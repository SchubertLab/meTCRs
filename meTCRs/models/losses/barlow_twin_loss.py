import torch
from torch.nn import Module


class BarlowTwinLoss(Module):
    """
    Taken from https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/barlow-twins.html
    :param lmd: float, The regularization parameter lambda
    """
    def __init__(self, lmd: float, regulator: float):
        super(BarlowTwinLoss, self).__init__()

        self._lmd = lmd
        self._regulator = torch.tensor(regulator)

    def forward(self, anchor1: torch.Tensor, positive: torch.Tensor, anchor2: torch.Tensor, negative: torch.Tensor):
        z1_norm = (anchor1 - torch.mean(anchor1, dim=0)) / (torch.maximum(torch.std(anchor1, dim=0), self._regulator))
        z2_norm = (positive - torch.mean(positive, dim=0)) / (torch.maximum(torch.std(positive, dim=0), self._regulator))

        cross_correlation = torch.matmul(z1_norm.T, z2_norm) / len(anchor1)

        diagonal_term = torch.sum((1 - torch.diagonal(cross_correlation))**2)
        off_diagonal_term = self._get_off_diagonal_term(cross_correlation)

        return diagonal_term + self._lmd * off_diagonal_term

    @staticmethod
    def _get_off_diagonal_term(cross_correlation: torch.Tensor):
        clone = cross_correlation.clone()
        return torch.sum(clone**2)
