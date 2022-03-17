import torch
import numpy as np


class TCREpitopeDataset(torch.utils.data.Dataset):
    def __init__(self, tcr_data, epitope_data):
        self.tcr_data = tcr_data
        self.epitope_data = epitope_data

    def __len__(self):
        return len(self.tcr_data)

    def __getitem__(self, index):
        return np.array(self.tcr_data[index]), self.epitope_data[index]
