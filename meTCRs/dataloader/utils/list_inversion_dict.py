from typing import Dict


def list_inversion_dict(iterable) -> Dict:
    """
    Converts an iterable into a dictionary where each element serves as a key and the value is a list of indices where
    this element occurs in the iterable
    :param iterable: iterable, the iterable (i.e. list, array etc.) to be inverted
    :return: dictionary, whose keys are the elements of the iterable and the values are lists of indices
    """
    output_dict = {}
    for idx, element in enumerate(iterable):
        if element in output_dict.keys():
            output_dict[element].append(idx)
        else:
            output_dict[element] = [idx]

    return output_dict
