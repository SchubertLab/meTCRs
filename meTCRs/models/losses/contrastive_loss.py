import torch
from torch.nn import ReLU, Module


class ContrastiveLoss(Module):
    def __init__(self, distance, alpha=0.1, reduction='mean'):
        super(ContrastiveLoss, self).__init__()

        self.distance = distance
        self.alpha = alpha
        self.reduction = reduction
        self.relu = ReLU()

    def forward(self, anchor1: torch.Tensor, positive: torch.Tensor, anchor2: torch.Tensor, negative: torch.Tensor):
        positive_loss = self.distance(anchor1, positive)
        negative_loss = self.relu(self.alpha - self.distance(anchor2, negative))

        if len(anchor1) == 0:
            return self._reduce(negative_loss)
        if len(anchor2) == 0:
            return self._reduce(positive_loss)

        concatenated_loss = torch.cat([negative_loss, positive_loss])
        return self._reduce(concatenated_loss)

    def _reduce(self, losses):
        if self.reduction == 'mean':
            return torch.mean(losses)
        elif self.reduction == 'sum':
            return torch.sum(losses)
        else:
            return losses
