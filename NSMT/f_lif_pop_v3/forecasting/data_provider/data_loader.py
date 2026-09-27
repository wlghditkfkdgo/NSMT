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
        # 2O D-BL: 'future' has its targets in [14400, 17420) with the 336 preceding rows allowed
        # as context -- 2,925 windows at H96. It is opened once, through the global registry.
        if flag == 'future' and len(values) < 17420:
            raise ValueError('the future period needs rows up to 17420')
        self.start, self.end = {'train': (0, 8640), 'val': (8640 - self.seq_len, 11520),
                               'test': (11520 - self.seq_len, 14400),
                               'future': (14400 - self.seq_len, 17420)}[flag]
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


class Dataset_Custom(Dataset):
    """Dataset_Custom adapted from model_v1 for weather (prereg 2R): the repository's 70/10/20 row
    split, every variable (features 'M') with the target column 'OT' moved last, StandardScaler fit
    on the train rows only. Each split keeps the seq_len rows before it as context, as in model_v1.
    Time marks are unused, as in Dataset_ETT_hour."""
    def __init__(self, args, flag='train'):
        self.seq_len, self.pred_len = args.seq_len, args.pred_len
        self.flag = flag
        self.scaler = StandardScaler()
        self.path = os.path.join(args.root_path, args.data_path)
        frame = pd.read_csv(self.path)
        cols = [c for c in frame.columns if c not in ('date', 'OT')]
        self.columns = cols + ['OT']                                  # model_v1: target last
        values = frame[self.columns].to_numpy(dtype=np.float64)
        if values.shape != (52696, 21) or not np.isfinite(values).all():
            raise ValueError('Expected finite weather data: 52,696 rows x 21 variables')
        num_train = int(len(values) * 0.7)
        num_test = int(len(values) * 0.2)
        num_vali = len(values) - num_train - num_test
        self.borders = {'train': (0, num_train),
                        'val': (num_train - self.seq_len, num_train + num_vali),
                        'test': (len(values) - num_test - self.seq_len, len(values))}
        self.scaler.fit(values[:num_train])
        values = self.scaler.transform(values).astype(np.float32)
        self.start, self.end = self.borders[flag]
        self.data_x = torch.from_numpy(values[self.start:self.end].copy())
        self.data_y = self.data_x
        if self.__len__() < 1:
            raise ValueError('No complete windows')

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len
        return self.data_x[s_begin:s_end], self.data_y[s_end:r_end], torch.empty(0), torch.empty(0)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
