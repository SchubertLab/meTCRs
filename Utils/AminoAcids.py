import Utils.Configurations as Dirs
import pandas as pd
import numpy as np


LETTER_CODES = [
    '_', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]


def letter_code_to_int(letter_code):
    return LETTER_CODES.index(letter_code)


def int_to_letter_code(int_code):
    return LETTER_CODES[int_code]


def one_hot_to_letter_code(one_hot_vector):
    if sum(one_hot_vector) == 0:
        return '_'
    position = one_hot_vector.index(1)
    return LETTER_CODES[position]


def tensor_to_amino_acid(tensor, is_one_hot=True):
    amino_acid = ''
    for row in tensor:
        if is_one_hot:
            amino_acid += one_hot_to_letter_code(row.numpy().tolist())
        else:
            amino_acid += LETTER_CODES[int(row.numpy())]
    return amino_acid


def read_blosum():
    encodings = pd.read_csv(Dirs.PATH_BLOSUM_45, sep=' ', index_col=0)
    invalid_letters = ['*', 'B', 'J', 'Z', 'X']
    encodings = encodings.drop(invalid_letters, axis=0)
    encodings = encodings.drop(invalid_letters, axis=1)
    encodings.loc['_'] = [0] * 20
    encodings['_'] = 0
    # encodings = (encodings - encodings.mean()) / encodings.std()

    encoding_dict = {}
    for letter in encodings.columns:
        encoding_dict[letter] = encodings[letter].values.astype(np.float32)
    return encoding_dict


if __name__ == '__main__':
    print(read_blosum())