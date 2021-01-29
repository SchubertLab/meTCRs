import tensorflow as tf
import numpy as np
import pandas as pd

from Utils import AminoAcids as Amino


class BatchSampler:
    def __init__(self, classes_per_batch, samples_per_class, path_dataset='', length_max=25, do_embed=True):
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batch_size = self.classes_per_batch * self.samples_per_class

        self.length_max = length_max

        self.dataset = self.load_dataset(path_dataset, do_embed)

    def load_dataset(self, path_dataset, do_embed):
        df_samples = pd.read_csv(path_dataset)
        samples_by_label = {}
        class_to_label = {}
        next_label = 0
        for _, row in df_samples.iterrows():
            class_name = row['Antigen']
            if class_name not in class_to_label:
                class_to_label[class_name] = next_label
                next_label += 1
            label = class_to_label[class_name]
            if label not in samples_by_label:
                samples_by_label[label] = []
            sequence = row['CDR3_beta']
            if do_embed:
                sequence = self.embed_sequence(sequence)
            samples_by_label[label].append(sequence)
        samples_by_label = {antigen: tcrs for antigen, tcrs in samples_by_label.items()
                            if len(tcrs) >= self.samples_per_class}
        assert self.samples_per_class <= min([len(value) for value in samples_by_label.values()])
        return samples_by_label

    def generate_batch(self):
        while True:
            classes = np.random.choice(list(self.dataset.keys()), size=self.classes_per_batch, replace=False)
            for cls in classes:
                samples = np.random.choice(range(len(self.dataset[cls])), size=self.samples_per_class, replace=False)
                for idx in samples:
                    yield self.dataset[cls][idx], np.array(cls)

    def embed_sequence(self, sequence):
        length_padding = self.length_max - len(sequence)
        sequence_embedded = sequence + '_' * length_padding
        sequence_embedded = list(sequence_embedded)
        sequence_embedded = [Amino.letter_code_to_int(letter) for letter in sequence_embedded]
        sequence_embedded = np.array(sequence_embedded)
        return sequence_embedded

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.generate_batch, (tf.float32, tf.int16))
        dataset = dataset.batch(self.batch_size)
        return dataset


if __name__ == '__main__':
    ''' Print the first Batch for testing purposes. '''
    sampler = BatchSampler(16, 2, 'data/full_train.csv')
    dl = sampler.get_dataset()
    # print(dl.dataset)
    for batch in dl:
        print(batch[0])
        break

