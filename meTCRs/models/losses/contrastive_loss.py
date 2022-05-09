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

        concatenated = torch.cat([negative_loss, positive_loss])

        if self.reduction == 'mean':
            return torch.mean(concatenated)
        elif self.reduction == 'sum':
            return torch.sum(concatenated)
        else:
            return concatenated
