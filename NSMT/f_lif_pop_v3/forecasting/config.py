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

NEURON_CLASSES = ['lif', 'flif', 'pop_flif', 'sel_flif', 'sel_pop_flif']   # neurons.NEURONS
SINGLE = ('lif', 'flif', 'sel_flif')                              # 한 뉴런: tau 하나
SELECTIVE = ('sel_flif', 'sel_pop_flif')
POPULATION = ('pop_flif', 'sel_pop_flif')                          # --soma 의 기본값이 lif 인 클래스


def soma_of(config):
    """--soma with the class default filled in (neurons.DEFAULT_SOMA)."""
    soma = getattr(config, 'soma', None)
    return soma if soma is not None else ('lif' if config.neuron in POPULATION else 'none')


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
    parser.add_argument('--model', default='myModel', choices=['myModel', 'GRU', 'Linear'])
    parser.add_argument('--task', default='recall', choices=['recall', 'ett'])
    parser.add_argument('--data', default='ETTh1', choices=['ETTh1', 'ETTh2', 'weather', 'recall'],
                        help="weather (prereg 2R) needs --root_path <NSMT>/forecasting/dataset/weather")
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
    recall_arg.add_argument('--n_confirm', type=int, default=1000,
                            help='held-out split used once, after selection (D-AI)')
    recall_arg.add_argument('--data_seed', type=int, default=20260921)

    # 텐서 모양. 회상 과제도 ETT와 같은 T=42가 되도록 맞춘다.
    shape_arg = parser.add_argument_group('tensor shape')
    shape_arg.add_argument('--seq_len', type=int, default=336)
    shape_arg.add_argument('--pred_len', type=int, default=96)
    shape_arg.add_argument('--patch_size', type=int, default=8)
    shape_arg.add_argument('-emb', '--embedding_dim', dest='embed_dim', type=int, default=32)
    shape_arg.add_argument('--head_dim', type=int, default=32)
    shape_arg.add_argument('--head_mode', choices=['flatten', 'last', 'linear'], default='flatten',
                           help="'linear' = model_v1's single nn.Linear readout on the flattened spikes (prereg 2Q)")
    shape_arg.add_argument('--revin', action=argparse.BooleanOptionalAction, default=False,
                           help='prereg 2O: per-window, per-channel standardisation of the input, inverted on the output')
    shape_arg.add_argument('--readout', choices=['spike', 'analog', 'drive'], default='spike',
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
    # 뉴런 클래스 (neurons.py). legacy는 layers.PopulationNeuron 그대로이며 지금까지의 실행을 재현한다.
    neuron_arg.add_argument('--neuron', default='legacy', choices=['legacy'] + NEURON_CLASSES,
                            help='legacy = layers.PopulationNeuron (every run so far); the others are the '
                                 'neurons.py classes, checked against spikeDE in analysis/spikede_parity.py')
    neuron_arg.add_argument('--soma', choices=['lif', 'none'], default=None,
                            help="'lif' = silent f-LIF member(s) read by a spikeDE LIF soma (v3 arrangement); "
                                 "'none' = the f-LIF member(s) fire themselves (K trains). Default: lif for the "
                                 "population classes, none for the single-neuron classes")
    neuron_arg.add_argument('--fde_method', choices=['pred', 'gl'], default='pred',
                            help='fractional solver of the neurons.py classes (spikeDE pred | gl)')
    neuron_arg.add_argument('--fde_step', type=float, default=1.,
                            help="integration step h of the neurons.py classes (the paper's scripts use 0.1)")
    neuron_arg.add_argument('--fde_memory', type=int, default=0,
                            help='spikeDE short memory: only the last N steps enter; 0 = full history')

    # 선택자. mode가 곧 실험 조건이다.
    select_arg = parser.add_argument_group('selector')
    select_arg.add_argument('--mode', default='sparse',
                            choices=['full', 'dense', 'sparse', 'recent', 'mass_matched', 'oracle', 'hard'])
    # mode=hard (prereg 2J): kernel kept, past slots gated 0/1 by a statistic. No eta, no cap.
    select_arg.add_argument('--hard_axis', choices=['unit', 'shared', 'input'], default='shared',
                            help='descriptor the similarity is computed on (2J fixes shared)')
    select_arg.add_argument('--hard_stat', choices=['pearson', 'cosine', 'recent', 'random', 'oracle'], default='pearson',
                            help='recent/random: same-budget content-free controls (2K); oracle: answer slots (2M)')
    select_arg.add_argument('--hard_q', type=float, default=1.,
                            help='fraction of past slots kept per query; 1.0 is the neutral limit (== full)')
    select_arg.add_argument('--select_part', choices=['all', 'input'], default='all',
                            help="selective neurons.py classes: 'all' drops whole past increments (v3); "
                                 "'input' drops only their input part and keeps leak and reset (2Q (A) candidate)")
    select_arg.add_argument('--theta', type=float, default=5.5,
                            help='score temperature; Phase B calibrates it (recall r2: 5.5)')
    select_arg.add_argument('--key_norm', choices=['none', 'frozen'], default='none',
                            help='frozen standardises xi before Q/K; not the main stability fix')
    select_arg.add_argument('--eta_init', type=float, default=-4., help='sigmoid pre-activation; -4 -> 0.018')
    select_arg.add_argument('--eta_fixed', type=float, default=None, help='pin eta instead of learning it')
    select_arg.add_argument('--qk_eps', type=float, default=1e-6,
                            help='soft-norm floor; the Jacobian of the normalisation is bounded by 1/eps')
    select_arg.add_argument('--qk_norm', action=argparse.BooleanOptionalAction, default=False,
                            help='L2-normalise the projected q and k before the distance (priority 2)')
    select_arg.add_argument('--cap', action=argparse.BooleanOptionalAction, default=True,
                            help='R3 coefficient cap; --no-cap is the ablation only')

    parser.add_argument('--g11_every', type=int, default=10, help='watch max|u| every N train batches')
    parser.add_argument('--max_train_batches', type=int, default=0, help='Smoke only; zero uses all windows')
    parser.add_argument('--max_eval_batches', type=int, default=0, help='Smoke only; zero uses all windows')
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--require_calibration', action=argparse.BooleanOptionalAction, default=True,
                        help='fail instead of training without a frozen G11 bound')
    parser.add_argument('--calibration_dir', default='calibration',
                        help='folder under results/ that calibrate.py writes and train.py reads')
    parser.add_argument('--config', default=None, help='Existing run directory for standalone test')

    return parser.parse_args()


def parse_defaults():
    """Parser defaults only, for filling attributes absent from an older saved config."""
    import sys
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        return parse_arguments()
    finally:
        sys.argv = argv


class Config():
    def __init__(self):
        self.date = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%y%m%d')

    def set_args(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.dataset = 'recall' if self.task == 'recall' else self.data
        self.data_path = self.data + '.csv'
        self.num_patches = self.seq_len // self.patch_size
        if self.neuron == 'legacy':
            self.variant = ('heterogeneous' if self.heterogeneous else 'homogeneous') + '_' + self.mode
            if self.qk_norm:
                self.variant += '_qknorm'
            if not self.cap:
                self.variant += '_nocap'
            if self.eta_fixed is not None:
                self.variant += f'_eta{self.eta_fixed:g}'
            if self.mode == 'hard':
                self.variant += f'-{self.hard_axis}-{self.hard_stat}-q{self.hard_q:g}'
        else:
            check_neuron_args(self)
            self.variant = neuron_variant(self)
        if self.revin:
            self.variant += '_revin'
        if self.task == 'recall':
            self.variant += f'_k{self.n_keys}'
        # 모델·α·readout이 run_id에 없으면 서로 다른 실험이 같은 이름으로 충돌한다 (audit A10).
        self.run_id = (f'{self.model}_{self.task}_{self.dataset}_p{self.pred_len}_{self.head_mode}'
                       f'_a{self.alpha}_{self.readout}_{self.variant}_seed{self.seed}')
        tag = self.date
        for name in ['model', 'task', 'readout', 'seq_len', 'pred_len', 'patch_size', 'embed_dim',
                     'num_population', 'alpha', 'head_dim', 'lr']:
            tag += f'+{name}+{getattr(self, name)}'
        self.save_result_path = str(TASK / 'log' / self.suite / self.dataset / self.date / tag
                                    / f'seed{self.seed}_{self.head_mode}_{self.readout}_{self.variant}')
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
        # 나중에 추가된 인자는 옛 config에 없다. parser 기본값으로 채운다.
        for key, value in vars(parse_defaults()).items():
            if not hasattr(self, key):
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


def check_neuron_args(config):
    """Refuse the settings a neurons.py class would silently ignore. No-op for the legacy neuron.

    The classes fix their selection when they are built (selective ones read --hard_q,
    --hard_stat and --hard_axis), so --mode only states which kind of class it is: 'full'
    without selection, 'hard' with it -- the legacy names of the same two computations.
    """
    if getattr(config, 'neuron', 'legacy') == 'legacy':
        return
    config.soma = soma_of(config)                                    # 클래스 기본값을 채운다
    single, selective = config.neuron in SINGLE, config.neuron in SELECTIVE
    problems = []
    if config.model != 'myModel':
        problems.append('--model myModel')
    if config.mode != ('hard' if selective else 'full'):
        problems.append(f"--mode {'hard' if selective else 'full'} (got {config.mode!r})")
    if single and (len(config.tau) != 1 or config.num_population != 1 or not config.heterogeneous):
        problems.append('one --tau and --num_population 1 (a single neuron)')
    if not single and len(config.tau) != config.num_population:
        problems.append('one --tau per --num_population member')
    if config.neuron == 'lif' and (config.alpha != 1. or config.fde_method != 'pred' or config.fde_memory):
        problems.append('--alpha 1 and no fractional solver options for the Euler LIF')
    if selective and (config.fde_method != 'pred' or config.hard_stat == 'oracle'):
        problems.append("--fde_method pred and a statistic other than 'oracle' for selection")
    if not selective and (config.hard_q != 1. or config.select_part != 'all'):
        problems.append('--hard_q 1 and --select_part all without selection')
    if config.neuron == 'lif' and config.soma == 'lif':
        problems.append('--soma none for the Euler LIF (there is no LIF-with-soma class)')
    if config.readout != 'spike' and config.soma != 'lif':
        problems.append("--readout spike: analog/drive read the soma, which needs --soma lif")
    if config.qk_norm or not config.cap or config.eta_fixed is not None or config.key_norm != 'none':
        problems.append('no soft-selector options (--qk_norm, --no-cap, --eta_fixed, --key_norm): legacy only')
    if config.fde_step <= 0. or config.fde_memory < 0:
        problems.append('--fde_step > 0 and --fde_memory >= 0')
    if problems:
        raise ValueError(f'--neuron {config.neuron} needs: ' + '; '.join(problems))


def neuron_variant(config):
    """Name of a neurons.py configuration: class, time constants, soma, solver, selection."""
    name = config.neuron
    tag = f"{name}-tau{'-'.join(f'{t:g}' for t in config.tau)}"
    if name in POPULATION:
        tag += f'-soma{soma_of(config)}' + ('' if config.heterogeneous else '-homog')
    elif soma_of(config) == 'lif':
        tag += '-somalif'                                            # f-LIF with soma (한 뉴런)
    if config.fde_method != 'pred':
        tag += f'-{config.fde_method}'
    if config.fde_step != 1.:
        tag += f'-h{config.fde_step:g}'
    if config.fde_memory:
        tag += f'-mem{config.fde_memory}'
    if name in SELECTIVE:
        tag += f'-{config.hard_axis}-{config.hard_stat}-q{config.hard_q:g}'
        if config.select_part != 'all':
            tag += f'-part{config.select_part}'

    return tag


def calibration_tag(config):
    """Filename tag of the D11 calibration a run uses. Empty for the legacy neuron, so the
    existing files keep matching. A neurons.py configuration gets its own file, selection
    included: a selective one is probed as its q = 1 class but must pass its own at-init check,
    so it can neither borrow its base's file nor block it (the latest file of a tag is used)."""
    if getattr(config, 'neuron', 'legacy') == 'legacy':
        return ''

    return '_n-' + neuron_variant(config)


def neuron_kwargs(config):
    """The subset of Config that the neuron accepts, so callers do not drift."""
    if getattr(config, 'neuron', 'legacy') != 'legacy':
        # neurons.build_neuron이 받는 이름으로. num_population은 보정 대조용 (K = len(tau)).
        return dict(neuron=config.neuron, num_population=config.num_population, alpha=config.alpha,
                    tau=tuple(config.tau), heterogeneous=config.heterogeneous,
                    max_length=config.num_patches, tau_s=config.tau_s, threshold=config.threshold,
                    surrogate_scale=config.surrogate_scale, soma=soma_of(config),
                    method=config.fde_method, step=config.fde_step, memory=config.fde_memory or None,
                    q=config.hard_q, stat=config.hard_stat, axis=config.hard_axis,
                    part=config.select_part, seed=config.seed)
    return dict(num_population=config.num_population, alpha=config.alpha, tau=tuple(config.tau),
                heterogeneous=config.heterogeneous, max_length=config.num_patches,
                theta=config.theta, eta_init=config.eta_init,
                eta_fixed=config.eta_fixed, cap=config.cap,      # audit A06: 이름만 바뀌면 안 된다
                key_norm=config.key_norm, qk_norm=config.qk_norm, qk_eps=config.qk_eps,
                tau_s=config.tau_s, threshold=config.threshold,
                surrogate_scale=config.surrogate_scale,
                hard_axis=config.hard_axis, hard_stat=config.hard_stat, hard_q=config.hard_q,
                hard_seed=config.seed)                          # 2K: random control's generator
