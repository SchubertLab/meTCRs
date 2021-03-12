import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm

from tcrmatch_c import tcrmatch
from evaluation.CdrDistance import get_cdr_dist_value

from MultiBatchModels import DataLoader as Data
from Utils import Configurations as Config
import Utils.AminoAcids as Amino
import evaluation.Metrices as Metrics


def evaluate_performance(source='val', dataset='IEDB'):
    """
    Calculate the performance values of baseline and our method
    :param source: str indicating the dataset 'val' or 'test'
    :param dataset: str key indicating which dataset to use
    :return: dict {method_name: performance_summary} containing the scores for all methods
    """
    print(source)
    methods = {
        # 'uniform_noise': uniform_noise_wrapper,
        # 'tcr_match': tcr_match_wrapper,
        # 'normal_noise': normal_noise_wrapper,
        # 'cdr_dist': cdr_dist_wrapper,
        # 'MultiBatch': multibatch_model_wrapper,
        # 'AE': autoencoder_wrapper,
        # 'PairedModel': paired_classifier_wrapper,
    }
    performance = {}
    for name, func in methods.items():
        print(f'---{name}---')
        scores = evaluate_method_on_recall(func, source, dataset, tag=name)
        performance[name] = scores
    return performance


def evaluate_method_on_recall(distance_method, source, dataset, tag, model=None):
    """
    Calculate Recall at [1, 10, 100] for a given method
    :param distance_method: function list of sequences as input and returning a pairwise distance matrix
    :param source: source of the dataset ('val', 'train')
    :param dataset: str key indicating which dataset to use
    :param tag: unique identifier for saving the model performance
    :param model: trained tensorflow model
    :return: dict {k_value: score} R@k score for different ks
    """
    print('->load data')
    sequences, labels = load_data(source, dataset)
    print(f'{len(labels)} sequences')
    print('->calculate distances')
    if model is None:
        distance_matrix = distance_method(sequences)
    else:
        distance_matrix = distance_method(sequences, model)
    k_values = [1, 10, 100]
    print('->evaluate metrics')
    recall_values = Metrics.recall_at_k(distance_matrix, labels, k_values)
    for k, result in recall_values.items():
        print(f'Recall@{k}: {result}')
    log_performance(tag, recall_values, dataset)
    return recall_values


def log_performance(tag, scores, dataset):
    """
    Logs all performances to common file
    :param tag: identifier for the model
    :param scores: dict {k_value: score} containing the R@k values
    :param dataset: dataset tag for creating an individual file for each set
    :return: write output to logfile all_scores.csv
    """
    path_log = Config.PATH_LOGS + f'/all_scores_{dataset}.csv'
    if not os.path.exists(path_log):
        with open(path_log, 'w') as log_file:
            line = ['tag'] + ['R@'+str(k) for k in scores]
            line = ','.join(line) + '\n'
            log_file.write(line)
    with open(path_log, 'a+') as log_file:
        line = [tag] + [str(score) for score in scores.values()]
        line = ','.join(line) + '\n'
        log_file.write(line)


# <- Methods to calculate the distance matrix of a list of sequences by different methods ->
def tcr_match_wrapper(sequences):
    """
    Function calculating distance matrix based on TCRmatch algorithm
    :param sequences: list of sequences
    :return: numpy array (num_sequences, num_sequences) containing pairwise distances
    """
    distance_matrix = np.zeros(shape=(len(sequences), len(sequences)), dtype=np.float16)
    for i, seq_1 in enumerate(tqdm(sequences)):
        for j, seq_2 in enumerate(sequences):
            distance_matrix[i, j] = 1 - tcrmatch(seq_1, seq_2)[-1]
    return distance_matrix


def cdr_dist_wrapper(sequences):
    """
    Function calculating distance matrix based on CDRdist algorithm
    :param sequences: list of sequences
    :return: numpy array (num_sequences, num_sequences) containing pairwise distances
    """
    n = len(sequences)
    distance_matrix = np.zeros(shape=(n, n), dtype=np.float16)
    for i, seq_1 in enumerate(tqdm(sequences)):
        for j, seq_2 in enumerate(sequences):
            distance_matrix[i, j] = get_cdr_dist_value(seq_1, seq_2)
    return distance_matrix


def multibatch_model_wrapper(sequences):
    """
    Function calculating distance matrix based our trained model for latent space embedding.
    :param sequences: list of sequences
    :return: numpy array (num_sequences, num_sequences) containing pairwise distances
    """
    trained_model = tf.keras.models.load_model(Config.PATH_MODEL_MULTI_BATCH, compile=False)
    n = len(sequences)
    distance_matrix = np.zeros(shape=(n, n))
    embeddings = []
    for i, seq_1 in enumerate(tqdm(sequences)):
        seq_1 = embed_sequence(seq_1, do_one_hot=True)
        embed_1 = trained_model.predict(seq_1)[0]
        embeddings.append(embed_1)

    for i, embed_1 in enumerate(tqdm(embeddings)):
        for j, embed_2 in enumerate(embeddings):
            distance = euclidean_distance(embed_1, embed_2)
            distance_matrix[i][j] = distance
    return distance_matrix


def autoencoder_wrapper(sequences):
    """
    Function calculating distance matrix based on our autoencoder model
    :param sequences: list of sequences
    :return: numpy array (num_sequences, num_sequences) containing pairwise distances
    """
    trained_model = tf.keras.models.load_model(Config.PATH_MODEL_AE, compile=False)
    n = len(sequences)
    distance_matrix = np.zeros(shape=(n, n))
    embeddings = []
    for i, seq_1 in enumerate(sequences):
        seq_1 = embed_sequence(seq_1)
        embed_1 = trained_model.predict(seq_1)[0]
        embeddings.append(embed_1)

    for i, embed_1 in enumerate(embeddings):
        for j, embed_2 in enumerate(embeddings):
            distance = euclidean_distance(embed_1, embed_2)
            distance_matrix[i][j] = distance
    return distance_matrix


def paired_classifier_wrapper(sequences, model=None):
    """
    Function calculating distance matrix based on our paired data processing model
    :param sequences: list of sequences
    :param model: trained tensorflow model
    :return: numpy array (num_sequences, num_sequences) containing pairwise distances
    """
    if model is None:
        model = tf.keras.models.load_model(Config.PATH_MODEL_PAIRED, compile=False, custom_objects={'tf': tf})

    n = len(sequences)
    distance_matrix = np.zeros(shape=(n, n))

    sequences_numeric = []
    for seq in sequences:
        sequences_numeric.append(embed_sequence(seq))
    sequences_numeric = np.stack(sequences_numeric)

    for i in tqdm(range(sequences_numeric.shape[0])):
        seq_1 = sequences_numeric[i]
        seq_1 = np.repeat(seq_1, len(sequences), axis=0)
        seq_1 = np.expand_dims(seq_1, 1)

        paired_seqs = np.concatenate([seq_1, sequences_numeric], axis=1)
        prediction = model.predict(paired_seqs, batch_size=500)
        prediction = prediction.flatten()
        distance_matrix[i, :] = prediction
    return distance_matrix


def uniform_noise_wrapper(sequences):
    """
    Function calculating distance matrix of purely uniform noise
    :param sequences: list of sequences
    :return: numpy array (num_sequences, num_sequences) containing pairwise distances
    """
    distance_matrix = np.random.uniform(size=(len(sequences), len(sequences)))
    return distance_matrix


def normal_noise_wrapper(sequences):
    """
    Function calculating distance matrix of purely normal noise
    :param sequences: list of sequences
    :return: numpy array (num_sequences, num_sequences) containing pairwise distances
    """
    distance_matrix = np.random.normal(size=(len(sequences), len(sequences)))
    return distance_matrix


def euclidean_distance(x, y):
    """
    Calculates the euclidean distance between 2 vectors
    :param x: first numpy vector
    :param y: second numpy vector
    :return: euclidean distance value
    """
    dist = x-y
    dist = np.square(dist)
    dist = np.sum(dist)
    dist = np.sqrt(dist)
    return dist


def embed_sequence(sequence, do_one_hot=False):
    """
    Embeds an amino acid as numeric values
    :param sequence: str representing the amino acid sequence
    :param do_one_hot: embed each letter as one hot vector, if False: return the numeric value of the amino acid
    :return: numpy array of the embedded amino acid
    """
    length_padding = 25 - len(sequence)
    sequence_embedded = sequence + '_' * length_padding
    sequence_embedded = list(sequence_embedded)
    sequence_embedded = [Amino.letter_code_to_int(letter) for letter in sequence_embedded]

    if do_one_hot:
        one_hot = np.zeros(shape=(25, 21), dtype=np.float32)
        for idx, code in enumerate(sequence_embedded):
            one_hot[idx, code] = 1.
        one_hot = np.expand_dims(one_hot, axis=[0])
        return one_hot
    sequence_embedded = np.array(sequence_embedded)
    sequence_embedded = np.expand_dims(sequence_embedded, axis=[0])
    return sequence_embedded


# <-Functions for loading a dataset in the form sequences, labels->
def load_data(source, dataset):
    """
    Load the data used for evaluation
    :param source: str indicating the dataset ('val' or 'test')
    :param dataset: str key indicating which dataset to use
    :return: list of amino acid sequences, list of corresponding epitopes
    """
    if source == 'test':
        path_data = Config.PATH_DATA_TEST[dataset]
    elif source == 'val':
        path_data = Config.PATH_DATA_VAL[dataset]
    else:
        path_data = Config.PATH_DATA_TRAIN[dataset]
    data = Data.BatchSampler(16, 2, path_dataset=path_data, do_embed=False)
    data = data.dataset
    sequences, labels = reorder_dataset(data)
    return sequences, labels


def reorder_dataset(data):
    """
    Transform the dataset from an dict to lists
    :param data: evaluation data in the form {antigen: [tcrs]}
    :return: list of tcrs, list of corresponding antigens
    """
    sequences = []
    labels = []
    for label, sequence_list in data.items():
        labels += [label] * len(sequence_list)
        sequences += sequence_list
    labels = np.array(labels)
    return sequences, labels


if __name__ == '__main__':
    evaluate_performance('val', dataset='TcrMatch_old')

# just some performances notes
'''
vdjdb_val
random
Recall@1: 0.44154302670623147
Recall@10: 0.772700296735905
Recall@100: 0.9402571711177052
Recall@1000: 0.9970326409495549

tcr match
Recall@1: 0.590108803165183
Recall@10: 0.8478733926805143
Recall@100: 0.9556874381800198
Recall@1000: 0.9962413452027695

my 20 out, 1 lstm
Recall@1: 0.5612265084075173
Recall@10: 0.8367952522255193
Recall@100: 0.9592482690405539
Recall@1000: 0.997626112759644

conv 30 out
Recall@1: 0.5768545994065282
Recall@10: 0.8498516320474777
Recall@100: 0.9576656775469832
Recall@1000: 0.9974282888229475

auto encoder
Recall@1: 0.5708035003977725
Recall@10: 0.8406921241050119
Recall@100: 0.9576372315035799
Recall@1000: 0.997016706443914
'''


'''
tcr_match_val
---uniform_noise---
->load data
75
3327 sequences
->calculate distances
->evaluate metrics
Recall@1: 0.34595731890592124
Recall@10: 0.7382025849113315
Recall@100: 0.8845807033363391
---tcr_match---
->load data
75
3327 sequences
->calculate distances
->evaluate metrics
Recall@1: 0.4099789600240457
Recall@10: 0.826570483919447
Recall@100: 0.9380823564773069
--- my stuff ---
->evaluate metrics
Recall@1: 0.44063721070033063
Recall@10: 0.8322813345356177
Recall@100: 0.9284640817553351

'''