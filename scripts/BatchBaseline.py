import numpy as np
import baseline_evaluation.Metrices as Metrics
import DataLoader as DataLoader
import baseline_evaluation.Evaluation as Eval
from Utils import AminoAcids, Configurations as Config


def baseline_batch_random_uniform():
    recalls = {1: 0, 4: 0, 8: 0, 31: 0}
    for _ in range(10000):
        random_array = np.random.uniform(size=(16, 16))
        random_array = (random_array + random_array.T)/2
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7], dtype=np.float)

        recall = Metrics.recall_at_k(random_array, labels, [1, 4, 8])
        # print(recall)
        for key, value in recall.items():
            recalls[key] += value

    for key, value in recalls.items():
        print(f'Recall@{key}: {value/10000}')


def baseline_batch_tcr_match():
    dataset = DataLoader.BatchSampler(16, 2, '../data/full_val.csv', encoding=None).get_dataset()

    recalls = {1: 0, 4: 0, 8: 0, 31: 0}
    idx = 0
    for batch, labels in dataset:
        sequences = numeric_sequence_to_string(batch.numpy())
        distance_matrix = Eval.tcr_match_wrapper(sequences)
        labels = labels.numpy()

        recall = Metrics.recall_at_k(distance_matrix, labels, [1, 4, 8])
        for key, value in recall.items():
            recalls[key] += value
        idx += 1
        print(idx)
        if idx == 1000:
            break

    for key, value in recalls.items():
        print(f'Recall@{key}: {value/1000}')


def numeric_sequence_to_string(sequences):
    sequences_str_list = []
    for seq in sequences:
        seq_str = ''
        for letter_code in seq:
            letter_code = int(letter_code)
            seq_str += AminoAcids.LETTER_CODES[letter_code]
        seq_str = seq_str.replace('_', '')
        sequences_str_list.append(seq_str)
    return sequences_str_list


if __name__ == '__main__':
    baseline_batch_random_uniform()
    # baseline_batch_tcr_match()
