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
    parser = argparse.ArgumentParser(description='Population f-LIF v3-A')
    parser.add_argument('--model', default='myModel', choices=['myModel', 'GRU'])
    parser.add_argument('--task', default='recall', choices=['recall', 'ett'])
    parser.add_argument('--data', default='ETTh1', choices=['ETTh1', 'ETTh2', 'recall'])
    parser.add_argument('--root_path', default=str(NSMT / 'forecasting/dataset/ETT-small'))
    parser.add_argument('--suite', default='v3a-phaseD-20260921')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('-nd', '--num_device', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--scheduler_patience', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)

    # 회상 과제 (사전등록 §4). 난이도 축은 n_keys이며 run_range는 고정한다.
    recall_arg = parser.add_argument_group('synthetic recall task')
    recall_arg.add_argument('--n_keys', type=int, default=3)
    recall_arg.add_argument('--run_range', nargs=2, type=int, default=[2, 5])
    recall_arg.add_argument('--min_gap', type=int, default=1, help='minimum intervening runs before re-query')
    recall_arg.add_argument('--cue_mode', choices=['onehot', 'code'], default='onehot')
    recall_arg.add_argument('--cue_noise', type=float, default=0.)
    recall_arg.add_argument('--distractor', type=float, default=0.)
    recall_arg.add_argument('--n_train', type=int, default=8000)
    recall_arg.add_argument('--n_val', type=int, default=1000)
    recall_arg.add_argument('--n_test', type=int, default=1000)
    recall_arg.add_argument('--data_seed', type=int, default=20260921)

    # 텐서 모양. 회상 과제도 ETT와 같은 T=42가 되도록 맞춘다.
    shape_arg = parser.add_argument_group('tensor shape')
    shape_arg.add_argument('--seq_len', type=int, default=336)
    shape_arg.add_argument('--pred_len', type=int, default=96)
    shape_arg.add_argument('--patch_size', type=int, default=8)
    shape_arg.add_argument('-emb', '--embedding_dim', dest='embed_dim', type=int, default=32)
    shape_arg.add_argument('--head_dim', type=int, default=32)
    shape_arg.add_argument('--head_mode', choices=['flatten', 'last'], default='flatten')
    shape_arg.add_argument('--readout', choices=['spike', 'analog'], default='spike',
                           help='analog reads the soma membrane WITH gradient (diagnostic, D8)')

    # 뉴런. tau와 alpha는 사전등록 고정값이며 학습하지 않는다 (D7, F1).
    neuron_arg = parser.add_argument_group('population f-LIF neuron')
    neuron_arg.add_argument('--num_population', type=int, default=4)
    neuron_arg.add_argument('--alpha', type=float, default=.7)
    neuron_arg.add_argument('--tau', nargs='+', type=float, default=[4., 8., 16., 32.])
    neuron_arg.add_argument('--heterogeneous', action=argparse.BooleanOptionalAction, default=True)
    neuron_arg.add_argument('--tau_s', type=float, default=2.)
    neuron_arg.add_argument('--threshold', type=float, default=1.)
    neuron_arg.add_argument('--surrogate_scale', type=float, default=5.)
    neuron_arg.add_argument('--input_scale', type=float, default=8., help='D11 calibration; 8.0 recall, 6.0 ETTh1')
    neuron_arg.add_argument('--input_norm', choices=['none', 'frozen'], default='frozen',
                            help='frozen: per-unit standardisation estimated once on train, no batch stats')

    # 선택자. mode가 곧 실험 조건이다.
    select_arg = parser.add_argument_group('selector')
    select_arg.add_argument('--mode', default='sparse',
                            choices=['full', 'dense', 'sparse', 'recent', 'mass_matched', 'oracle'])
    select_arg.add_argument('--theta', type=float, default=5.5,
                            help='score temperature; Phase B calibrates it (recall r2: 5.5)')
    select_arg.add_argument('--key_norm', choices=['none', 'frozen'], default='none',
                            help='frozen standardises xi before Q/K; not the main stability fix')
    select_arg.add_argument('--eta_init', type=float, default=-4., help='sigmoid pre-activation; -4 -> 0.018')
    select_arg.add_argument('--eta_fixed', type=float, default=None, help='pin eta instead of learning it')
    select_arg.add_argument('--cap', action=argparse.BooleanOptionalAction, default=True,
                            help='R3 coefficient cap; --no-cap is the ablation only')

    parser.add_argument('--g11_every', type=int, default=10, help='watch max|u| every N train batches')
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
        self.dataset = 'recall' if self.task == 'recall' else self.data
        self.data_path = self.data + '.csv'
        self.num_patches = self.seq_len // self.patch_size
        self.variant = ('heterogeneous' if self.heterogeneous else 'homogeneous') + '_' + self.mode
        if not self.cap:
            self.variant += '_nocap'
        if self.eta_fixed is not None:
            self.variant += f'_eta{self.eta_fixed:g}'
        if self.task == 'recall':
            self.variant += f'_k{self.n_keys}'
        # 모델·α·readout이 run_id에 없으면 서로 다른 실험이 같은 이름으로 충돌한다 (audit A10).
        self.run_id = (f'{self.model}_{self.task}_{self.dataset}_p{self.pred_len}_{self.head_mode}'
                       f'_a{self.alpha}_{self.readout}_{self.variant}_seed{self.seed}')
        tag = self.date
        for name in ['model', 'task', 'seq_len', 'pred_len', 'patch_size', 'embed_dim',
                     'num_population', 'alpha', 'head_dim', 'lr']:
            tag += f'+{name}+{getattr(self, name)}'
        self.save_result_path = str(TASK / 'log' / self.suite / self.dataset / self.date / tag
                                    / f'seed{self.seed}_{self.head_mode}_{self.variant}')
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
        for key, value in torch.load(os.path.join(config_path, 'model_state/config.pt'),
                                     map_location='cpu').items():
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


def neuron_kwargs(config):
    """The subset of Config that PopulationNeuron accepts, so callers do not drift."""
    return dict(num_population=config.num_population, alpha=config.alpha, tau=tuple(config.tau),
                heterogeneous=config.heterogeneous, max_length=config.num_patches,
                theta=config.theta, eta_init=config.eta_init,
                eta_fixed=config.eta_fixed, cap=config.cap,      # audit A06: 이름만 바뀌면 안 된다
                key_norm=config.key_norm,
                tau_s=config.tau_s, threshold=config.threshold,
                surrogate_scale=config.surrogate_scale)
