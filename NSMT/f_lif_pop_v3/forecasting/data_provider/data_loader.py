"""Dataset_ETT_hour adapted from model_v1; identical 12/4/4-month boundaries."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, flag='train'):
        self.seq_len, self.pred_len = args.seq_len, args.pred_len
        self.flag = flag
        self.scaler = StandardScaler()
        self.path = os.path.join(args.root_path, args.data_path)
        frame = pd.read_csv(self.path)
        self.columns = [c for c in frame.columns if c != 'date']
        values = frame[self.columns].to_numpy(dtype=np.float64)
        if len(values) < 14400 or values.shape[1] != 7 or not np.isfinite(values).all():
            raise ValueError('Expected finite ETT-hour data with seven variables')
        self.scaler.fit(values[:8640])
        values = self.scaler.transform(values).astype(np.float32)
        self.start, self.end = {'train': (0, 8640), 'val': (8640 - self.seq_len, 11520),
                               'test': (11520 - self.seq_len, 14400)}[flag]
        self.data_x = torch.from_numpy(values[self.start:self.end].copy())
        self.data_y = self.data_x
        if self.__len__() < 1:
            raise ValueError('No complete windows')

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len
        # Keep the repository's four-value batch interface. Time marks unused.
        return self.data_x[s_begin:s_end], self.data_y[s_end:r_end], torch.empty(0), torch.empty(0)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
