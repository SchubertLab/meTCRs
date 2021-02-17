import tensorflow as tf
import pandas as pd
import numpy as np

import Utils.AminoAcids as Amino


class LoaderAE:
    def __init__(self, path_data, batch_size=32):
        self.path_data = path_data
        self.length_max = 25
        self.batch_size = batch_size

        self.tcrs = self.load_dataset()
        self.training_data = self.create_dataset()

    def load_dataset(self):
        df = pd.read_csv(self.path_data)['CDR3_beta']
        df = df.drop_duplicates()
        df = df.reset_index()
        df['numeric'] = df.apply(lambda row: self.embed_sequence(row['CDR3_beta']), axis=1)
        df['one_hot'] = df.apply(lambda row: self.embed_one_hot(row['numeric']), axis=1)
        mask = df['CDR3_beta'].str.len() <= self.length_max
        df = df[mask]
        df = df.sample(frac=1)
        return df

    def embed_sequence(self, sequence):
        length_padding = self.length_max - len(sequence)
        sequence_embedded = sequence + '_' * length_padding
        sequence_embedded = list(sequence_embedded)
        sequence_embedded = [Amino.letter_code_to_int(letter) for letter in sequence_embedded]
        sequence_embedded = np.array(sequence_embedded, dtype=np.float32)
        return sequence_embedded

    def embed_one_hot(self, numeric_sequence):
        shape = (self.length_max, len(Amino.LETTER_CODES))
        one_hot = np.zeros(shape=shape, dtype=np.float32)
        for idx in range(self.length_max):
            one_hot[idx, int(numeric_sequence[idx])] = 1
        return one_hot

    def create_dataset(self):
        tcrs_numeric = np.stack(self.tcrs['numeric'].values, axis=0)
        tcrs_one_hot = np.stack(self.tcrs['one_hot'].values, axis=0)
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor(tcrs_numeric),
                tf.convert_to_tensor(tcrs_one_hot)
            )
        )
        dataset = tf.data.Dataset.shuffle(dataset, buffer_size=500)
        dataset = tf.data.Dataset.batch(dataset, self.batch_size)
        return dataset


if __name__ == '__main__':
    dl = LoaderAE('data/dl_vdj_train.csv').training_data
    for batch in dl:
        print(batch)
        break