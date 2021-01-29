import pandas as pd
import os
import shutil
import random

from Utils import AminoAcids as Amino

PATH_INPUT = os.path.join('../data', 'IEDB_data.tsv')
PATH_OUTPUT = os.path.join('../data', 'clustered_data')
PATH_OUTPUT_INVALID = os.path.join('../data', 'single_data')
PATH_OUTPUT_PAIRS_POS = os.path.join('../data', 'positive_')
PATH_OUTPUT_PAIRS_NEG = os.path.join('../data', 'negative_')
PATH_OUTPUT_SPLIT = os.path.join('../data', 'full_')


MAX_LENGTH = 30
MIN_LENGTH = 1


# Functions for separating the data based on epitope specificity
def copy_to_files_by_epitope():
    """ Stores TCRs in separate file based on their epitope specificity. """
    purge_directories()
    data = read_data()
    tcrs_by_epitope = {}
    for idx, sample in data.iterrows():
        if idx % 100 == 0:
            print(str(idx) + '/' + str(len(data)))
        add_sample(tcrs_by_epitope, sample)
    write_collection_to_file(tcrs_by_epitope)


def purge_directories():
    purge_directory(PATH_OUTPUT)
    purge_directory(PATH_OUTPUT_INVALID)


def purge_directory(path_dir):
    for file_name in os.listdir(path_dir):
        file_name_full = os.path.join(PATH_OUTPUT, file_name)
        os.remove(file_name_full)


def read_data():
    full_data = pd.read_csv(PATH_INPUT, sep='\t', index_col=False)
    return full_data


def add_sample(epitope_dict, sample):
    for epitope in sample['epitopes'].split(','):
        if epitope not in epitope_dict.keys():
            epitope_dict[epitope] = pd.DataFrame(columns=sample.keys())
        refined_sample = sample.copy()
        refined_sample['epitopes'] = epitope
        if is_valid_amino_acid(epitope) and is_valid_amino_acid(sample['trimmed_seq']):
            epitope_dict[epitope] = epitope_dict[epitope].append(refined_sample)


def is_valid_amino_acid(amino_accid):
    for letter in amino_accid:
        if letter not in Amino.LETTER_CODES:
            return False
    if len(amino_accid) > MAX_LENGTH:
        return False
    if len(amino_accid) < MIN_LENGTH:
        return False
    return True


def write_collection_to_file(tcrs_by_epitope):
    for epitope, samples in tcrs_by_epitope.items():
        samples.drop_duplicates(subset=['trimmed_seq'], inplace=True)
        if len(samples) > 1:
            path_out_sample = os.path.join(PATH_OUTPUT, epitope + '.csv')
        else:
            path_out_sample = os.path.join(PATH_OUTPUT_INVALID, epitope + '.csv')
        samples.to_csv(path_out_sample, index=False)


# Functions to create a datasets of positively paired TCRs in training, validation, evaluation splits
def create_paired_positive_data_split(fractions=None, tags=None):
    if fractions is None:
        fractions = [0.7, 0.15, 0.15]
    if tags is None:
        tags = ['train', 'val', 'test']
    create_folders(tags)

    file_list_full = [file for file in os.listdir(PATH_OUTPUT) if '.csv' in file]
    n = len(file_list_full)
    splits = []
    random.seed(2903)
    for fraction in fractions:
        partial_split = random.sample(file_list_full, int(n*fraction))
        splits.append(partial_split)
        file_list_full = [file_name for file_name in file_list_full if file_name not in partial_split]
    splits[-1] += file_list_full  # append last files, if they are leftover by rounding errors
    for partial_split, tag in zip(splits, tags):
        create_paired_data_positive(partial_split, tag)


def create_folders(tags):
    for tag in tags:
        path_folder = os.path.join(PATH_OUTPUT, tag)
        if os.path.exists(path_folder):
            os.remove(path_folder)
        os.mkdir(path_folder)


def create_paired_data_positive(file_list, tag):
    path_output = PATH_OUTPUT_PAIRS_POS + tag
    rows = []
    for idx, file_name in enumerate(file_list):
        print(idx)
        copy_file_by_tag(file_name, tag)
        current_file = pd.read_csv(os.path.join(PATH_OUTPUT, file_name))
        current_file.reset_index()
        rows += match_pairs(current_file)
    paired_data = pd.DataFrame(rows, columns=['trimmed_seq1', 'original_seq1', 'receptor_group1',
                                              'trimmed_seq2', 'original_seq2', 'receptor_group2', 'epitope'])
    paired_data = paired_data.sample(frac=1)
    paired_data.to_csv(path_output + '.csv', index=False)


def copy_file_by_tag(file_name, tag):
    path_old = os.path.join(PATH_OUTPUT, file_name)
    path_new = os.path.join(PATH_OUTPUT, tag, file_name)
    shutil.copy(path_old, path_new)


def match_pairs(df, max_combinations=15000):
    rows = []
    count = 0
    for j in range(len(df)):
        for i in range(len(df) - j - 1):
            tcr1 = df.iloc[i].tolist()
            tcr2 = df.iloc[i + j + 1].tolist()
            new_pair = tcr1[:-1] + tcr2
            rows.append(new_pair)
            count += 1
            if count == max_combinations:
                return rows
    return rows


# Functions to create a datasets of positively paired TCRs in training, validation, evaluation splits
def create_paired_negative_data_split(tags=None):
    if tags is None:
        tags = ['train', 'val', 'test']
    amounts_samples = get_amount_positive_samples(tags)
    for tag, amount in zip(tags, amounts_samples):
        tcr_list = load_full_data(tag)
        negative_pairs = create_random_pairs(tcr_list, amount)
        write_negatives(negative_pairs, tag)


def get_amount_positive_samples(tags):
    amounts = []
    for tag in tags:
        path_positive_data = PATH_OUTPUT_PAIRS_POS + tag + '.csv'
        data_pos = pd.read_csv(path_positive_data)
        amount = len(data_pos)
        amounts.append(amount)
    return amounts


def load_full_data(tag):
    path_input = PATH_OUTPUT_SPLIT + tag + '.csv'
    data_full = pd.read_csv(path_input)
    return data_full


def create_random_pairs(tcr_list, amount):
    count = 0
    n = len(tcr_list)
    print(n)
    rows = []
    while count < amount:
        if count % 1000 == 0:
            print(str(count) + '/' + str(amount))
        tcr1 = tcr_list.iloc[random.randint(0, n-1)].to_list()
        tcr2 = tcr_list.iloc[random.randint(0, n-1)].to_list()
        if tcr1[-1] != tcr2[-1]:
            pair = tcr1[:-1] + tcr2
            rows.append(pair)
            count += 1
    paired_data = pd.DataFrame(rows, columns=['trimmed_seq1', 'original_seq1', 'receptor_group1',
                                              'trimmed_seq2', 'original_seq2', 'receptor_group2', 'epitope'])
    return paired_data


def write_negatives(data, tag):
    path_output = PATH_OUTPUT_PAIRS_NEG + tag
    data = data.sample(frac=1)
    data.to_csv(path_output + '.csv', index=False)


# Functions to create files of tcrs based on their split
def create_tcr_files_by_split(tags=None):
    if tags is None:
        tags = ['train', 'val', 'test']

    for tag in tags:
        tcrs_by_tag = get_tcrs_in_split(tag)
        write_tcr_split_to_file(tcrs_by_tag, tag)


def get_tcrs_in_split(tag):
    tcr_df = pd.DataFrame()
    path_split = os.path.join(PATH_OUTPUT, tag)
    file_list = [file for file in os.listdir(path_split) if file.endswith('.csv')]
    for file_name in file_list:
        path_file = os.path.join(path_split, file_name)
        single_df = pd.read_csv(path_file)
        tcr_df = pd.concat([tcr_df, single_df])
    return tcr_df


def write_tcr_split_to_file(tcrs, tag):
    path_out = PATH_OUTPUT_SPLIT + tag + '.csv'
    tcrs.to_csv(path_out, index=False)


if __name__ == '__main__':
    copy_to_files_by_epitope()
    create_paired_positive_data_split()
    create_tcr_files_by_split()
    create_paired_negative_data_split()
