import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import random
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch

TASK = Path(__file__).resolve().parent
NSMT = TASK.parents[1]
sys.path.insert(0, str(NSMT))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)


def set_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_arguments():
    parser = argparse.ArgumentParser(description='f-LIF population forecasting: first experiment')
    parser.add_argument('--model', default='myModel', choices=['myModel'])
    parser.add_argument('--data', default='ETTh1', choices=['ETTh1', 'ETTh2'])
    parser.add_argument('--root_path', default=str(NSMT / 'forecasting/dataset/ETT-small'))
    parser.add_argument('--suite', default='ett-first-20260914')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('-nd', '--num_device', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('-emb', '--embedding_dim', dest='embed_dim', type=int, default=32)
    parser.add_argument('--num_population', type=int, default=4)
    parser.add_argument('--head_dim', type=int, default=32)
    parser.add_argument('--head_mode', choices=['flatten', 'last'], default='flatten')
    parser.add_argument('--heterogeneous', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--retrieval', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tau_min', type=float, default=2.)
    parser.add_argument('--tau_max', type=float, default=16.)
    parser.add_argument('--threshold', type=float, default=1.)
    parser.add_argument('--memory_strength', type=float, default=0.05)
    parser.add_argument('--temperature', type=float, default=0.25)
    parser.add_argument('--input_scale', type=float, default=2.)
    parser.add_argument('--max_train_batches', type=int, default=0, help='Smoke only; zero uses all windows')
    parser.add_argument('--max_eval_batches', type=int, default=0, help='Smoke only; zero uses all windows')
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--config', default=None, help='Existing run directory for standalone test')
    return parser.parse_args()


class Config():
    def __init__(self):
        self.date = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%y%m%d')

    def set_args(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.dataset = self.data
        self.data_path = self.data + '.csv'
        self.variant = ('heterogeneous' if self.heterogeneous else 'homogeneous') + ('_retrieval' if self.retrieval else '_no_memory')
        self.run_id = f'{self.data}_p{self.pred_len}_{self.head_mode}_{self.variant}_seed{self.seed}'
        tag = self.date
        for name in ['model', 'seq_len', 'pred_len', 'patch_size', 'embed_dim', 'num_population', 'head_dim', 'lr']:
            tag += f'+{name}+{getattr(self, name)}'
        self.save_result_path = str(TASK / 'log' / self.suite / self.data / self.date / tag / f'seed{self.seed}_{self.head_mode}_{self.variant}')
        self.save_log_path = os.path.join(self.save_result_path, 'log')
        self.save_model_state_path = os.path.join(self.save_result_path, 'model_state')
        self.result_path = str(TASK / 'results' / self.suite / (self.run_id + '.json'))
        if os.path.exists(self.save_result_path) or os.path.exists(self.result_path):
            raise FileExistsError('Run exists; use a new --suite to preserve artifacts')
        self.device = torch.device('cpu' if self.cpu else f'cuda:{self.num_device}')
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        os.makedirs(self.save_log_path)
        os.makedirs(self.save_model_state_path)

    def load_args(self, config_path, config):
        for key, value in torch.load(os.path.join(config_path, 'model_state/config.pt'), map_location='cpu').items():
            setattr(self, key, value)
        self.device = torch.device('cpu' if config.cpu else f'cuda:{config.num_device}')

    def save_arg(self):
        torch.save(vars(self), os.path.join(self.save_model_state_path, 'config.pt'))

    def print_info(self):
        with open(os.path.join(self.save_result_path, 'logargs.txt'), 'w') as handle:
            for key, value in vars(self).items():
                line = f'{key:-<30s}{str(value):->70s}'
                print(line)
                print(line, file=handle)
