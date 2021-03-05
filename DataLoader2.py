import tensorflow as tf
import numpy as np
import pandas as pd

import random

import Utils.AminoAcids as Amino


class DataLoader:
    def __init__(self, batch_size, path_dataset='', length_max=25):
        self.batch_size = batch_size
        self.path_dataset = path_dataset
        self.length_max = length_max

        self.sequence_label_pairs = self.load_dataset()
        self.sequences_by_label = self.sort_by_label()

    def load_dataset(self):
        raw_data = pd.read_csv(self.path_dataset)
        sequence_label_pairs = []
        labels = {}
        cur_label = 0
        for _, row in raw_data.iterrows():
            antigen = row['Antigen']
            sequence = row['CDR3_beta']
            sequence = self.embed_sequence(sequence)
            if len(sequence) > self.length_max:
                continue
            if antigen not in labels:
                labels[antigen] = cur_label
                cur_label += 1
            label = labels[antigen]
            sequence_label_pairs.append([sequence, label])
        return sequence_label_pairs

    def sort_by_label(self):
        sequences_by_label = {}
        for sequence, label in self.sequence_label_pairs:
            if label not in sequences_by_label:
                sequences_by_label[label] = []
            sequences_by_label[label].append(sequence)
        return sequences_by_label

    def embed_sequence(self, sequence):
        length_padding = self.length_max - len(sequence)
        sequence = sequence + '_' * length_padding
        sequence = [Amino.letter_code_to_int(letter) for letter in sequence]
        return np.array(sequence)

    def generate_dataset(self):
        random.shuffle(self.sequence_label_pairs)
        for seq_anchor, label_anchor in self.sequence_label_pairs:
            # positive pair
            seq_pos = random.choice(self.sequences_by_label[label_anchor])
            yield [seq_anchor, seq_pos], np.array(0., dtype=np.float32)
            # negative pair
            seq_neg, label_neg = random.choice(self.sequence_label_pairs)
            while label_neg == label_anchor:
                seq_neg, label_neg = random.choice(self.sequence_label_pairs)
            yield [seq_anchor, seq_neg],  np.array(1., dtype=np.float32)

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.generate_dataset, (tf.float32, tf.float32))
        dataset = dataset.batch(self.batch_size)
        return dataset


if __name__ == '__main__':
    dl = DataLoader(4, path_dataset='data/full_train.csv')
    for batch, target in dl.get_dataset():
        print('-----')
        for i in range(4):
            print(Amino.tensor_to_amino_acid(batch[i, 0], is_one_hot=False))
            print(Amino.tensor_to_amino_acid(batch[i, 1], is_one_hot=False))
            print(target[i].numpy())
        break
