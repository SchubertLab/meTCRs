import tensorflow as tf
import numpy as np
import pandas as pd
import random

from Utils import AminoAcids as Amino


class BatchSampler:
    def __init__(self, batch_size, samples_per_class, path_dataset='', length_max=25, min_entries_per_group=None,
                 do_embed=True, do_weight=False, encoding='one_hot'):
        self.path_data = path_dataset

        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.min_entries_per_group = min_entries_per_group

        self.do_embed = do_embed
        self.encoding = encoding
        self.do_weight = do_weight

        self.length_max = length_max

        self.blosum = Amino.read_blosum()
        self.dataset = self.load_dataset(path_dataset)

    def load_dataset(self, path_dataset):
        """
        Process dataset from file to groups of tcrs.
        :param path_dataset: the path to an csv file.
        :return: dictionary {epitope: list of tcrs}.
        """

        df_samples = pd.read_csv(path_dataset)
        df_samples = df_samples.drop_duplicates(['CDR3_beta'])
        samples_by_label = {}
        class_to_label = {}
        next_label = 0
        for _, row in df_samples.iterrows():
            class_name = row['Antigen']
            sequence = row['CDR3_beta']
            if len(sequence) > self.length_max:
                continue

            if class_name not in class_to_label:
                class_to_label[class_name] = next_label
                next_label += 1

            label = class_to_label[class_name]
            if label not in samples_by_label:
                samples_by_label[label] = []

            sequence = self.embed_sequence(sequence)
            samples_by_label[label].append(sequence)
        samples_by_label = self.filter_groups(samples_by_label)
        return samples_by_label

    def filter_groups(self, samples_by_label):
        """
        Filter epitope groups if not enough tcrs are present.
        :param samples_by_label: dictionary {epitope: tcr list}.
        :return: dictionary containing only the filtered groups.
        """
        if self.min_entries_per_group:
            samples_by_label = {antigen: tcrs for antigen, tcrs in samples_by_label.items()
                                if len(tcrs) >= self.min_entries_per_group}  # self.samples_per_class}
        else:
            samples_by_label = {antigen: tcrs for antigen, tcrs in samples_by_label.items()
                                if len(tcrs) >= self.samples_per_class}
        return samples_by_label

    def get_weight_list(self):
        """
        Linear weights by amount of tcrs in an epitope group.
        :return: list where each entry corresponds to the weights for one epitope or None if samples are not weighted.
        """
        if self.do_weight is None:
            return None

        weights = []
        for group_label, members in self.dataset.items():
            weights.append(len(members))
        weights = [float(w)/sum(weights) for w in weights]
        return weights

    def generate_sample(self):
        """
        Generator for sequences with class label
        :return: yield sequence, class label as numpy arrays.
        """
        classes_per_batch = int(self.batch_size/self.samples_per_class)
        group_list = list(self.dataset.keys())
        weight_list = self.get_weight_list()
        while True:
            classes = np.random.choice(group_list, size=classes_per_batch, replace=False, p=weight_list)
            for cls in classes:
                samples = np.random.choice(range(len(self.dataset[cls])), size=self.samples_per_class, replace=False)
                for idx in samples:
                    yield self.dataset[cls][idx], np.array(cls)

    def generate_paired_sample(self):
        """
        Generator for positive and negative pairs of tcrs.
        :return: yields (sequence 1, sequence 2), label as numpy arrays.
        """
        group_list = list(self.dataset.keys())
        weight_list = self.get_weight_list()

        half_batch = int(self.batch_size/2)
        while True:
            classes_pos = np.random.choice(group_list, size=half_batch, replace=False, p=weight_list)
            classes_neg = np.random.choice(group_list, size=half_batch, replace=False, p=weight_list)
            for cls_pos, cls_neg in zip(classes_pos, classes_neg):
                sample_pos = random.sample(self.dataset[cls_pos], k=2)
                yield sample_pos, np.array(1., dtype=np.float32)
                sample_neg_1 = random.sample(self.dataset[cls_pos], k=1)[0]
                sample_neg_2 = random.sample(self.dataset[cls_neg], k=1)[0]
                yield [sample_neg_1, sample_neg_2], np.array(0., dtype=np.float32)

    def embed_sequence(self, sequence):
        """
        Create numpy representation of an amino acid string.
        :param sequence: string of amino acid single letter encodings.
        :return: string of sequence or numpy array, either one hot encoded or array of amino acids indices.
        """
        if not self.do_embed:
            return sequence
        length_padding = self.length_max - len(sequence)
        sequence_embedded = sequence + '_' * length_padding
        sequence_embedded = list(sequence_embedded)
        sequence_embedded = [Amino.letter_code_to_int(letter) for letter in sequence_embedded]

        if self.encoding == 'one_hot':
            return self.one_hot_encoding(sequence_embedded)
        elif self.encoding == 'blosum':
            return self.blosum_encoding(sequence_embedded)
        else:
            return np.array(sequence_embedded)

    def one_hot_encoding(self, sequence):
        """
        One hot encode an amino acid sequence.
        :param sequence: list of integers indicating the amino acid
        :return: numpy array of shape (max_length, 21) of one hot encoded amino acids.
        """
        sequence_embedded = np.zeros(shape=(self.length_max, 21), dtype=np.float32)
        for idx, code in enumerate(sequence):
            sequence_embedded[idx, code] = 1.
        return sequence_embedded

    def blosum_encoding(self, sequence):
        """
        Encoding based on BLOSUM45 amino acid coding.
        :param sequence: list of integers indicating the amino acid
        :return: numpy array of shape (max_length, 21) of blosum encoding
        """
        sequence_embedded = np.zeros(shape=(self.length_max, 21), dtype=np.float32)
        for idx, code in enumerate(sequence):
            amino_letter = Amino.LETTER_CODES[code]
            sequence_embedded[idx, :] = self.blosum[amino_letter]
        return sequence_embedded

    def get_dataset(self, do_paired=False):
        """
        Creates TF dataset that can be used with e.g. keras API.
        :param do_paired: boolean, whether a sample consists of pos / neg Pairs or single Sequences with class label.
        :return: tf.data.Dataset that can be used as a handle to the data.
        """
        if do_paired:
            dataset = tf.data.Dataset.from_generator(self.generate_paired_sample, (tf.float32, tf.float32))
        else:
            dataset = tf.data.Dataset.from_generator(self.generate_sample, (tf.float32, tf.int16))
        dataset = dataset.batch(self.batch_size)
        return dataset


if __name__ == '__main__':
    """ Print the first Batch for testing purposes. """
    sampler = BatchSampler(32, 2, 'data/dl_vdj_train.csv', do_embed=True, do_weight=True)
    dl = sampler.get_dataset(do_paired=False)
    # print(dl.dataset)
    for batch, target in dl:
        print(batch)
        print(target)
        break
