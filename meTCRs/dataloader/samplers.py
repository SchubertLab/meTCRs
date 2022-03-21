import numpy as np
from torch.utils.data.sampler import Sampler

from meTCRs.models.utils.list_inversion_dict import list_inversion_dict


class HierarchicalSampler(Sampler):
    """
    Inspired by MPerClassSampler (https://github.com/KevinMusgrave/pytorch-metric-learning/blob/58247798ca9bf62ff49874e5cd07c41424e64fe9/src/pytorch_metric_learning/utils/common_functions.py)
    For each batch a certain number of classes is sampled to which the labels of the batch will belong. In the second
    step, for each class a fixed number of samples is drawn.
    """

    def __init__(self, labels, batch_size: int, classes_per_batch: int, total_batches: int, *args, **kwargs):
        """
        :param labels: iterable, contains the labels of each data point
        :param batch_size: int, the length of each batch
        :param classes_per_batch: int, number of classes each batch should contain, must be smaller or equal than the
                                  total number of classes available in the labels iterable
        :param total_batches: int, total number of batches to be created
        :param args:
        :param kwargs:
        """
        super(HierarchicalSampler, self).__init__(*args, **kwargs)

        assert (batch_size % classes_per_batch != 0, "batch_size must be multiple of classes_per_batch")

        self.class_dict = list_inversion_dict(labels)
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.total_batches = total_batches
        self.samples_per_class = batch_size // classes_per_batch

    def __iter__(self):
        indices = []

        for _ in range(self.total_batches):
            classes = np.random.choice(self.class_dict.keys(), size=self.classes_per_batch, replace=False)
            for cls in classes:
                indices.append(np.random.choice(self.class_dict[cls], size=self.samples_per_class))

        return iter(indices)

    def __len__(self):
        return self.total_batches * self.batch_size
