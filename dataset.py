import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class EurDataset(Dataset):
    def __init__(self, split='train'):
        
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        file_path = os.path.join(data_dir, f'{split}_data.pkl')

        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def collate_data(batch):
    batch_size = len(batch)
    max_len = max(map(len, batch))   # max length in batch
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=len, reverse=True)

    for i, sent in enumerate(sort_by_len):
        sents[i, :len(sent)] = sent  # padding

    return torch.from_numpy(sents)
