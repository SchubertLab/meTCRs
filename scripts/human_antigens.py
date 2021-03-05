import pandas as pd
import os


def format_data():
    files = get_all_files()
    full_data = read_data(files)
    data_split = split_data(full_data, ['train', 'val', 'test'], [0.6, 0.2, 0.2])
    save_data(data_split)


def save_data(data):
    for name, df in data.items():
        path_file = 'human_' + name + '.csv'
        df.to_csv(path_file, index=False)


def split_data(data, names, fractions):
    subsets = {}
    start = 0
    for name, frac in zip(names, fractions):
        end = start + int(frac*len(data))
        subsets[name] = data[start:end]
        start = end
    return subsets


def read_data(files):
    subsets = []
    for path_file in files:
        df = pd.read_csv(path_file, delimiter='\t')
        antigen = path_file.split('\\')[1]
        antigen = antigen.split('-')[1]
        df['Antigen'] = antigen
        df = df.rename({'aminoAcid': 'CDR3_beta'}, axis=1)
        df = df.rename({'beta': 'CDR3_beta'}, axis=1)
        df = df[['Antigen', 'CDR3_beta']]
        subsets.append(df)
    data = pd.concat(subsets, ignore_index=True)
    data = data.sample(frac=1)
    return data


def get_all_files():
    path_base = 'Human_Antigens'
    dirs = [x[0] for x in os.walk(path_base)]

    file_names = []
    for dir in dirs:
        file_names += [dir+'\\'+name for name in os.listdir(dir) if name.endswith('.tsv')]

    print(file_names)
    return file_names




if __name__ == '__main__':
    format_data()