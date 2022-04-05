import numpy as np
from torch.utils.data import IterableDataset

from meTCRs.dataloader.utils.list_inversion_dict import list_inversion_dict


class TCREpitopeDataset(IterableDataset):
    """
    Inspired by MPerClassSampler (https://github.com/KevinMusgrave/pytorch-metric-learning/blob/58247798ca9bf62ff49874e5cd07c41424e64fe9/src/pytorch_metric_learning/utils/common_functions.py)
    For each batch a certain number of classes is sampled to which the labels of the batch will belong. In the second
    step, for each class a fixed number of samples is drawn.
    """

    def __init__(self, tcr_data, epitope_data, batch_size: int, classes_per_batch: int, total_batches: int):
        """
        :param tcr_data: iterable, contains the tcr sequences
        :param epitope_data: iterable, contains the epitopes related to each
        :param batch_size: int, the length of each batch
        :param classes_per_batch: int, number of classes each batch should contain, must be smaller or equal than the
                                  total number of classes available in the labels iterable
        :param total_batches: int, total number of batches to be created
        """
        self.tcr_data = tcr_data
        self.epitope_data = epitope_data
        self.class_dict = list_inversion_dict(epitope_data)
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.total_batches = total_batches
        self.samples_per_class = batch_size // classes_per_batch

    def __len__(self):
        return self.total_batches * self.batch_size

    def __iter__(self):
        samples = []

        for _ in range(self.total_batches):
            classes = np.random.choice(list(self.class_dict.keys()), size=self.classes_per_batch, replace=False)
            for cls in classes:
                indices = np.random.choice(self.class_dict[cls], size=self.samples_per_class)
                for idx in indices:
                    samples.append((np.array(self.tcr_data[idx]), self.epitope_data[idx]))

        return iter(samples)
