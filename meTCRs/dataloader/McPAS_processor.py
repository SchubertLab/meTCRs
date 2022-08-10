import pandas as pd

from meTCRs.dataloader.utils.amino_acids import AMINO_ACIDS

CDR_SEQUENCE_KEY = 'CDR3.beta.aa'
EPITOPE_KEY = 'Epitope.peptide'


def remove_nans(data):
    return data.dropna(subset=[CDR_SEQUENCE_KEY, EPITOPE_KEY])


def remove_invalid_proteins(data, key):
    mask = data[key].apply(lambda protein: any(c not in AMINO_ACIDS for c in protein))
    return data[~mask]


def trim_cdr(data):
    def trim(x: str):
        trimmed = x
        if trimmed.startswith('C'):
            trimmed = x[1:]
        if trimmed.endswith('F'):
            trimmed = x[:-1]
        return trimmed

    data[CDR_SEQUENCE_KEY] = data[CDR_SEQUENCE_KEY].apply(trim)

    return data


def prepare_mcpas(path, extended: bool = False):
    data = pd.read_csv(path, encoding='latin1')
    data = remove_nans(data)
    data = remove_invalid_proteins(data, CDR_SEQUENCE_KEY)
    data = remove_invalid_proteins(data, EPITOPE_KEY)
    data = trim_cdr(data)

    if extended:
        return data
    else:
        return pd.DataFrame({'CDR3b': data[CDR_SEQUENCE_KEY], 'Epitope': data[EPITOPE_KEY]})
