from meTCRs.evaluation.precision import Precision
from meTCRs.evaluation.recall import Recall
from meTCRs.evaluation.utils import default_compare


class F1Score:
    def __init__(self, k: int, dist_type: str = 'l2', compare=default_compare):
        self._precision = Precision(k, compare=compare, dist_type=dist_type)
        self._recall = Recall(k, compare=compare, dist_type=dist_type)

    def __call__(self, embedding_sequences, labels):
        pr = self._precision(embedding_sequences, labels)
        re = self._recall(embedding_sequences, labels)

        return 2 * pr * re / (pr + re)
