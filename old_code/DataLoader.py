import tensorflow as tf
import pandas as pd

import os
from Utils import AminoAcids as Amino

# import random


MAX_LENGTH = 23
DEPTH_ONE_HOT = len(Amino.LETTER_CODES) + 1
PATH_DATA = os.path.join('data_db')


def load_data_from_file(source_tag, is_positive=True, fraction=1.):
    if is_positive:
        path_input = os.path.join(PATH_DATA, 'positive_')
    else:
        path_input = os.path.join(PATH_DATA, 'negative_')
    path_input += source_tag + '.csv'
    raw_data = pd.read_csv(path_input)[['trimmed_seq1', 'trimmed_seq2']]
    if fraction != 1:
        raw_data = raw_data.sample(frac=fraction)
    return raw_data


def apply_padding(pd_data):
    padding_func = padding_end
    pd_data = pd_data.applymap(padding_func)
    return pd_data


def padding_end(input_string):
    return input_string + '_' * (MAX_LENGTH-len(input_string))


def create_labels(amount, is_positive):
    value = 1.
    if is_positive:
        value = 0.
    labels = [value] * amount
    # for testing only
    # labels = [random.randint(0, 1) for i in range(amount)]
    labels = tf.data.Dataset.from_tensor_slices(labels)
    return labels


def convert_pd_to_tf(pd_data):
    tf_data = tf.data.Dataset.from_tensor_slices(pd_data.values)
    return tf_data


def convert_to_one_hot(chains):
    chains = tf.strings.bytes_split(chains)
    for letter in Amino.LETTER_CODES:
        chains = tf.strings.regex_replace(chains, letter, str(Amino.letter_code_to_int(letter)))
    chains = tf.strings.regex_replace(chains, '_', '20')  # '-1')
    chains = tf.strings.to_number(chains, out_type=tf.dtypes.int32)
    # chains = tf.one_hot(chains, DEPTH_ONE_HOT)
    # chains = (chains-0.5) * 2
    chains = chains.to_tensor()
    return chains


def set_shape(data):
    data.set_shape((2, MAX_LENGTH))  # , DEPTH_ONE_HOT))
    return data


def get_data(batch_size, source_tag='train', amount_neg_data=600000, use_static_neg=True, fraction=1):
    positive_data = prepare_data(source_tag=source_tag, fraction=fraction)
    if use_static_neg:
        negative_data = prepare_data(source_tag=source_tag, is_positive=False, fraction=fraction)
    else:
        negative_data = prepare_negative_data(source_tag=source_tag, amount_neg_data=amount_neg_data)
    datasets = [positive_data, negative_data]
    choice_dataset = tf.data.Dataset.range(2).repeat(len(positive_data))
    data = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
    data = data.shuffle(10000, reshuffle_each_iteration=True)
    data = data.batch(batch_size)
    data = data.prefetch(2)
    return data


def prepare_data(source_tag, is_positive=True, fraction=1):
    data = load_data_from_file(source_tag, is_positive, fraction=fraction)
    data = apply_padding(data)
    labels = create_labels(len(data), is_positive=is_positive)
    data = convert_pd_to_tf(data)
    data = data.map(convert_to_one_hot)
    data = data.map(set_shape)
    data = tf.data.Dataset.zip((data, labels))
    return data


def prepare_negative_data(source_tag, amount_neg_data=600000):
    data = load_negative_data_from_file(source_tag)
    data = apply_padding(data)
    data = convert_pd_to_tf(data)
    data = create_negative_pairs(data, amount_neg_data)
    data = data.map(convert_to_one_hot)
    data = data.map(set_shape)
    labels = create_labels(len(data), is_positive=False)
    data = tf.data.Dataset.zip((data, labels))
    return data


def load_negative_data_from_file(source_tag):
    path_input = os.path.join(PATH_DATA, 'full_')
    path_input += source_tag + '.csv'
    raw_data = pd.read_csv(path_input)[['trimmed_seq']]
    return raw_data


# not sure about this suppresses a warning
@tf.autograph.experimental.do_not_convert
def create_negative_pairs(data, amount_pairs):
    current_amount = 0
    full_data = None
    while current_amount < amount_pairs:
        data2 = data
        data2 = data2.shuffle(50000, reshuffle_each_iteration=True)
        paired_data = tf.data.Dataset.zip((data, data2))
        paired_data = paired_data.map(lambda *t: tf.stack(t, axis=0))
        paired_data = paired_data.map(lambda t: tf.reshape(t, t.shape[:-1]))
        if full_data is None:
            full_data = paired_data
        else:
            full_data = full_data.concatenate(paired_data)
        current_amount += len(data)
    return full_data


if __name__ == '__main__':
    train_data = get_data(source_tag='train', batch_size=1, amount_neg_data=1000)
    for batch, target in train_data:
        print(batch.shape)
        print(batch[0][0])
        print(batch[0][1])
        print(target)
        break
