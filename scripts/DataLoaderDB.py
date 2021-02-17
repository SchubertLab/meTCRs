import string
import numpy as np
import random
import Utils.AminoAcids as Amino
import tensorflow as tf


class DebugDL:
    def __init__(self, classes_per_batch, samples_per_class, path_dataset='', length_max=25):
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batch_size = classes_per_batch * samples_per_class

        self.length_max = length_max

    def generate_batch(self):
        while True:
            for idx in range(self.classes_per_batch):
                samples = self.create_class_samples()
                for sample in samples:
                    yield self.embed_sequence(sample), np.array(idx)

    def embed_sequence(self, sequence):
        length_padding = self.length_max - len(sequence)
        sequence_embedded = sequence + '_' * length_padding
        sequence_embedded = list(sequence_embedded)
        sequence_embedded = [Amino.letter_code_to_int(letter) for letter in sequence_embedded]
        sequence_embedded = np.array(sequence_embedded)
        return sequence_embedded

    def create_class_samples(self):
        length = random.randint(8, self.length_max-3)
        base_sample = ''
        for _ in range(length):
            base_sample += random.choice(Amino.LETTER_CODES[1:])
        samples = [base_sample]
        while len(samples) != self.samples_per_class:
            change_idx = random.randint(0, len(base_sample))
            new_letter = random.choice(Amino.LETTER_CODES[1:])
            new_sample = base_sample[:change_idx] + new_letter + base_sample[change_idx+1:]
            samples.append(new_sample)
        return samples

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.generate_batch, (tf.float32, tf.int16))
        dataset = dataset.batch(self.batch_size)
        return dataset


if __name__ == '__main__':
    dl = DebugDL(16, 2, '', 25).get_dataset()
    for batch, label in dl:
        # print(batch)
        print(label)