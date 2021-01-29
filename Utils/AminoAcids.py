LETTER_CODES = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '_'
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


def tensor_to_amino_acid(tensor):
    amino_acid = ''
    for row in tensor:
        amino_acid += one_hot_to_letter_code(row.numpy().tolist())
    return amino_acid
