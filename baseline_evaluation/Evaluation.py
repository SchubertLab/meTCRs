import numpy as np

from tcrmatch_c import tcrmatch

import DataLoader as Data
from Utils import Configurations as Config
import baseline_evaluation.Metrices as Metrices


def evaluate_baseline(source='test'):
    methods = {
        'uniform_noise': uniform_noise_wrapper,
        'tcr_match': tcr_match_wrapper,
        # 'normal_noise': normal_noise_wrapper,
        # todo: add tcr_dist + own method
    }
    for name, func in methods.items():
        print(f'---{name}---')
        evaluate_method_on_recall(func, source)


def evaluate_method_on_recall(distance_method, source):
    print('->load data')
    sequences, labels = load_data(source)
    print(f'{len(labels)} sequences')
    print('->calculate distances')
    distance_matrix = distance_method(sequences)
    k_values = [1, 10, 100, 1000]
    print('->evaluate metrics')
    recall_values = Metrices.recall_at_k(distance_matrix, labels, k_values)
    for k, result in recall_values.items():
        print(f'Recall@{k}: {result}')


# <- Methods to calculate the distance matrix of a list of sequences by different methods ->
def tcr_match_wrapper(sequences):
    distance_matrix = np.zeros(shape=(len(sequences), len(sequences)))
    for i, seq_1 in enumerate(sequences):
        if i % 100 == 0:
            print(i)
        for j, seq_2 in enumerate(sequences):
            # print(tcrmatch(seq_1, seq_2))
            distance_matrix[i, j] = tcrmatch(seq_1, seq_2)[-1]
    return distance_matrix


def tcr_dist_wrapper(sequences):
    raise NotImplementedError


def my_awesome_shit_wrapper(sequences):
    raise NotImplementedError


def uniform_noise_wrapper(sequences):
    distance_matrix = np.random.uniform(size=(len(sequences), len(sequences)))
    return distance_matrix


def normal_noise_wrapper(sequences):
    distance_matrix = np.random.normal(size=(len(sequences), len(sequences)))
    return distance_matrix


# <-Functions for loading a dataset in the form sequences, labels->
def load_data(source):
    if source == 'test':
        path_data = Config.PATH_DATA_TEST
    elif source == 'val':
        path_data = Config.PATH_DATA_VAL
    else:
        path_data = Config.PATH_DATA_TRAIN
    data = Data.BatchSampler(16, 2, path_dataset=path_data, length_max=-99, do_embed=False)
    data = data.dataset
    sequences, labels = reorder_dataset(data)
    return sequences, labels


def reorder_dataset(data):
    """ input data: dictionary with key (representing the label) and values (list of corresponding sequences
        output: list of sequences, list of corresponding labels """
    sequences = []
    labels = []
    for label, sequence_list in data.items():
        labels += [label] * len(sequence_list)
        sequences += sequence_list
    labels = np.array(labels)
    return sequences, labels


if __name__ == '__main__':
    evaluate_baseline('val')
    evaluate_baseline('test')
