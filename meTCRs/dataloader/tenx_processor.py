import pandas as pd

CDR_SEQUENCE_KEY = 'TRB_cdr3'
EPITOPE_KEY = 'Epitope'


def trim_cdr(data):
    def trim(x: str):
        trimmed = x
        if trimmed.startswith('C'):
            trimmed = trimmed[1:]
        if trimmed.endswith('F'):
            trimmed = trimmed[:-1]
        return trimmed

    data[CDR_SEQUENCE_KEY] = data[CDR_SEQUENCE_KEY].apply(trim)

    return data


def prepare_10x(path, extended: bool = False):
    data = pd.read_csv(path)
    data = trim_cdr(data)

    if extended:
        return data
    else:
        return pd.DataFrame({'CDR3b': data[CDR_SEQUENCE_KEY], 'Epitope': data[EPITOPE_KEY]})
