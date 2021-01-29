import pandas as pd
import os
from collections import defaultdict


PATH_INPUT = os.path.join('../data', 'IEDB_data.tsv')
PATH_OUTPUT = os.path.join('../data', 'clustered_data')


def read_data():
    full_data = pd.read_csv(PATH_INPUT, sep='\t', index_col=False)
    return full_data


def amount_tcrs_in_file(path_file):
    with open(path_file) as f:
        nonempty_lines = [line.strip("\n") for line in f if line != "\n"]
    return len(nonempty_lines) - 1


def calculate_possible_pairings(amounts_per_epitope):
    amount = sum([(n - 1) * n / 2 for n in amounts_per_epitope.values()])
    return int(amount)


def length_information():
    length_tcrs = defaultdict(lambda: 0)
    raw_data = read_data()
    for _, sample in raw_data.iterrows():
        tcr_length = len(sample['trimmed_seq'])
        length_tcrs[tcr_length] += 1

    # plt.bar(length_tcrs.keys(), length_tcrs.values())
    # plt.show()
    return length_tcrs


def run_analysis():
    amount_tcrs_per_epitope = {}
    for file_name in os.listdir(PATH_OUTPUT):
        file_name_full = os.path.join(PATH_OUTPUT, file_name)
        epitope = file_name.split('.')[0]
        count = amount_tcrs_in_file(file_name_full)
        if count <= 100000:
            amount_tcrs_per_epitope[epitope] = count
    print('Min: ' + str(min(amount_tcrs_per_epitope.values())))
    print('Max: ' + str(max(amount_tcrs_per_epitope.values())))
    print('Avg: ' + str(int(sum(amount_tcrs_per_epitope.values()) / len(amount_tcrs_per_epitope.values()))))
    print('Possible pairings: ' + str(calculate_possible_pairings(amount_tcrs_per_epitope)))

    tcrs_length = length_information()
    print('Max length:' + str(max(tcrs_length.keys())))
    print('Min length:' + str(min(tcrs_length.keys())))


if __name__ == '__main__':
    run_analysis()
