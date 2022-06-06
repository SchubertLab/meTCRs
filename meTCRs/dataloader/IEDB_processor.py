import pandas as pd

from meTCRs.dataloader.utils.amino_acids import AMINO_ACIDS

CDR_SEQUENCE_KEY = 'Chain 2 CDR3 Calculated'
EPITOPE_KEY = 'Description'


def remove_nans(data):
    return data.dropna(subset=[CDR_SEQUENCE_KEY, EPITOPE_KEY])


def remove_invalid_epitopes(data):
    mask = data[EPITOPE_KEY].apply(lambda epitope: any(c not in AMINO_ACIDS for c in epitope))
    return data[~mask]


def prepare_iedb(path):
    data = pd.read_csv(path)
    data = remove_nans(data)
    data = remove_invalid_epitopes(data)

    return pd.DataFrame({'CDR3b': data[CDR_SEQUENCE_KEY], 'Epitope': data[EPITOPE_KEY]})
