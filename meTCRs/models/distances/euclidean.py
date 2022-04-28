from torch.nn import Module
from torch.linalg import norm


class Euclidean(Module):
    def __init__(self):
        super(Euclidean, self).__init__()
        self.distance = lambda x, y: norm(x - y, 2, dim=-1)

    def forward(self, x, y):
        return self.distance(x, y)
