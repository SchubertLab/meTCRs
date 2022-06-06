import pandas as pd

CDR_SEQUENCE_KEY = 'CDR3'
EPITOPE_KEY = 'Epitope'


def trim_sequences(data):
    data[CDR_SEQUENCE_KEY] = data[CDR_SEQUENCE_KEY].apply(lambda x: x[1:-1])


def prepare_vdjdb(path: str):
    data = pd.read_csv(path, sep='\t')
    trim_sequences(data)

    return pd.DataFrame({'CDR3b': data[CDR_SEQUENCE_KEY], 'Epitope': data[EPITOPE_KEY]})
