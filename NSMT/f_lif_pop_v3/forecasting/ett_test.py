import torch
import numpy as np
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from scipy import stats

import os
import sys
import glob
import json
import math
import hashlib
import argparse
from pathlib import Path

from config import set_random_seed, Config, parse_defaults, TASK
from model import LOAD_MODEL
from data_provider.data_factory import data_provider
import layers

sys.path.insert(0, str(TASK.parent / 'analysis'))
from split_registry import open_once                            # noqa: E402

# prereg 2N: the one look at the ETT test split for the hard design (D-BF, D-BG).
# Every run of a dataset is checked against the registered manifest BEFORE the split is
# claimed; each model then passes the whole split once, error and states from one forward.

SEEDS = (7, 13, 21, 42, 123, 256, 512, 1024)
CONDS = ('q1', 'pearson', 'gru')
TORCH = '1.12.0+cu113'                                          # D-BE: 이 버전의 CUDA topk가 동점 규칙이다
MDE = .005                                                      # O8
# audit 42 A20-MANIFEST: every setting a verdict depends on -- training budget, tensor shape,
# readout and head, neuron constants, input path -- not only the condition fields.
REGISTERED = {'task': 'ett', 'seq_len': 336, 'patch_size': 8, 'embed_dim': 32, 'head_dim': 32,
              'head_mode': 'flatten', 'readout': 'spike', 'epoch': 50, 'batch_size': 128, 'lr': 1e-3,
              'weight_decay': .01, 'patience': 10, 'scheduler_patience': 5, 'num_workers': 0,
              'max_train_batches': 0, 'max_eval_batches': 0, 'num_population': 4, 'alpha': .7,
              'tau': [4., 8., 16., 32.], 'heterogeneous': True, 'tau_s': 2., 'threshold': 1.,
              'surrogate_scale': 5., 'input_norm': 'frozen'}
HARD = {'q1': {'model': 'myModel', 'mode': 'hard', 'hard_axis': 'shared', 'hard_stat': 'pearson', 'hard_q': 1.},
        'pearson': {'model': 'myModel', 'mode': 'hard', 'hard_axis': 'shared', 'hard_stat': 'pearson', 'hard_q': .5},
        'gru': {'model': 'GRU'}}
# §5 참고 기준선 (v2, 같은 프로토콜의 다른 파이프라인). 검정하지 않는다.
REFERENCE = {'ETTh1': {'ridge': .3702, 'ridge_lastvalue': .3696, 'window_mean': .706},
             'ETTh2': {'ridge': .3010, 'ridge_lastvalue': .2719, 'window_mean': .385}}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_run(suite, data, pred_len, cond, seed):
    """run directory and result json of one finished training run."""
    variant = 'heterogeneous_full' if cond == 'gru' else \
        f"heterogeneous_hard-shared-pearson-q{HARD[cond]['hard_q']:g}"
    runs = glob.glob(str(TASK / 'log' / suite / data / '*' / '*' / f'seed{seed}_*_{variant}'))
    prefix = 'GRU' if cond == 'gru' else 'myModel'
    results = glob.glob(str(TASK / 'results' / suite / f'{prefix}_ett_{data}_p{pred_len}_*_{variant}_seed{seed}.json'))
    if len(runs) != 1 or len(results) != 1:
        raise SystemExit(f"[ett_test] {data} {cond} seed {seed}: {len(runs)} run dirs, {len(results)} results")

    return runs[0], results[0]


def load_model(run, device):
    base = parse_defaults()
    base.cpu = device.type == 'cpu'
    base.num_device = 0
    args = Config()
    args.load_args(run, base)
    args.device = device                                            # max_eval_batches는 덮어쓰지 않고 대조한다
    model = LOAD_MODEL[args.model](args, train=False)

    return model, args


def check_manifest(args, result, data, pred_len, cond, seed, bound):
    """Every registered field; returns the list of mismatches (empty = OK)."""
    want = {**REGISTERED, **HARD[cond], 'data': data, 'pred_len': pred_len, 'seed': seed}
    bad = [f"{k}: {getattr(args, k, None)!r} != {v!r}" for k, v in want.items() if getattr(args, k, None) != v]
    if cond != 'gru':
        if getattr(args, 'g11_bound', None) != bound:
            bad.append(f"g11_bound {getattr(args, 'g11_bound', None)!r} != {bound!r}")
        if getattr(args, 'input_scale', None) != 6.:
            bad.append(f"input_scale {getattr(args, 'input_scale', None)!r} != 6.0")
    if getattr(args, 'data_path', None) != f'{data}.csv':
        bad.append(f"data_path {getattr(args, 'data_path', None)!r} != '{data}.csv'")
    train = json.load(open(result)).get('train', {})
    if not 1 <= (train.get('epochs_run') or 0) <= REGISTERED['epoch']:
        bad.append(f"epochs_run {train.get('epochs_run')!r} outside 1..{REGISTERED['epoch']}")

    return bad


def test(args:Config, model=None, cond='pearson', flag='test'):
    """Whole test split, one forward per batch: MSE/MAE, safety, kept mass, ties, firing rate.

    flag='val' is the dry run before the test split is claimed; it writes nothing.
    """

    set_random_seed(args.seed)

    _, loader = data_provider(args, flag=flag)

    mse = MeanSquaredError().to(args.device)
    mae = MeanAbsoluteError().to(args.device)

    if model is None:
        model = LOAD_MODEL[args.model](args, train=False)

    spiking = args.model == 'myModel'
    peak, finite, nonfinite, windows = 0., True, 0, 0
    kept, kept_n, ties, tie_n, mismatch, rate = 0., 0, 0, 0, 0, []
    with torch.no_grad():
        model.eval()

        for i, batch in enumerate(loader):
            x, y, _, _ = batch
            x = x.float().to(args.device)
            y = y.float().to(args.device)

            output, aux = model(x, mode=args.mode, return_aux=True)   # 오차와 상태를 같은 forward에서
            mse.update(output.contiguous(), y.contiguous())
            mae.update(output.contiguous(), y.contiguous())
            windows += x.shape[0]

            if not spiking:
                if not torch.isfinite(output).all():
                    finite, nonfinite = False, nonfinite + 1
                continue

            state = aux['state']                                       # [T, B*C, D, K]
            if not (torch.isfinite(state).all() and torch.isfinite(output).all()):
                finite, nonfinite = False, nonfinite + 1               # 최대값에 섞지 않는다
            else:
                peak = max(peak, state.abs().max().item())
            rate.append(aux['spikes'].mean().item())

            b = model.embedding.neuron.b
            share = torch.zeros(state.shape[1], dtype=torch.float64, device=args.device)
            for n in range(1, len(aux['coeff'])):
                c = aux['coeff'][n].double()                           # [B*C, D, n] = m * b
                share += (c.sum(-1) / b[1:n + 1].sum().double()).mean(-1)
            kept += (share / (len(aux['coeff']) - 1)).sum().item()     # 시퀀스(=창 x 채널) 가중
            kept_n += state.shape[1]

            if cond == 'pearson':                                      # 학습된 모델의 top-k 경계 동점
                selector = model.embedding.neuron.selector
                current = model.embedding.current(layers.to_patches(x, args.patch_size))
                zero = torch.zeros_like(state[0])
                for n in range(1, current.shape[0]):
                    xi = torch.cat([state[n - 1], current[n].unsqueeze(-1)], dim=-1)
                    hist = torch.stack([torch.cat([state[j - 1] if j else zero, current[j].unsqueeze(-1)], -1)
                                        for j in range(n)], dim=-2)
                    m, sim = selector.hard_mask(xi, hist)
                    mismatch += int(not torch.equal(m, (aux['coeff'][n] > 0).to(m.dtype)))   # 재구성 = 모델의 마스크
                    k = max(1, int(round(selector.hard_q * n)))
                    if k < n:
                        ordered = sim[:, 0, :].sort(dim=-1, descending=True).values
                        ties += int((ordered[:, k - 1] == ordered[:, k]).sum())
                    tie_n += xi.shape[0]

    bound = getattr(args, 'g11_bound', None)
    test_result = {
        'mse' : mse.compute().item(),
        'mae' : mae.compute().item(),
        'windows' : windows,
        'finite' : finite,
        'nonfinite_batches' : nonfinite,
        'max_abs_state' : peak if spiking else None,
        'bound' : bound if spiking else None,
        'safe' : finite and (not spiking or (bound is not None and peak < bound)),
        'kept_mass_frac' : kept / kept_n if kept_n else None,
        'firing_rate' : float(np.mean(rate)) if rate else None,
        'ties' : [ties, tie_n] if cond == 'pearson' else None,
        'mask_mismatch' : mismatch if cond == 'pearson' else None,
    }

    print("Test was successfully done")

    head = ','.join(["mse", "mae", "windows", "safe", "max_abs_state", "bound", "kept_mass_frac",
                     "firing_rate", "ties"])
    results_csv = ','.join([
        f"{test_result['mse']:.6f}",
        f"{test_result['mae']:.6f}",
        f"{test_result['windows']}",
        f"{test_result['safe']}",
        f"{test_result['max_abs_state']:.4f}" if spiking else "None",
        f"{bound:.4f}" if spiking else "None",
        f"{test_result['kept_mass_frac']:.4f}" if spiking else "None",
        f"{test_result['firing_rate']:.4f}" if spiking else "None",
        f"{ties}/{tie_n}" if cond == 'pearson' else "None", ]
    )

    for k, v in test_result.items():
        if isinstance(v, float):
            print(f" > {k:16s}:{v:>9.6f}")

    if flag != 'test':
        return test_result

    with open(args.save_log_path + '/final+result.csv', 'a', encoding='utf-8') as log_csv:
        print(head, end="\n", file=log_csv)
        print(results_csv, end="\n", file=log_csv)

    print(f"Final result saved to `{args.save_log_path}`")

    return test_result


def paired(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n)

    return {'deltas': d.tolist(), 'mean': float(mean), 'sd': float(sd),
            'ci95': [float(mean - half), float(mean + half)], 'relative': float(mean / np.mean(b)),
            'a_lower': int((d < 0).sum()), 'n': n}


def verdict(t):
    """D-BG wording for a paired difference a - b."""
    lo, hi = t['ci95']
    if hi < 0:
        word = 'a가 test MSE를 낮춘다'
    elif lo > 0:
        word = 'a가 test MSE를 높인다'
    else:
        word = '차이를 주장하지 않는다'
    if abs(t['mean']) < MDE:
        word += f' (|평균| {abs(t["mean"]):.4f} < MDE {MDE}: 검출 한계 아래)'

    return word


def parse_arguments():

    parser = argparse.ArgumentParser(description='prereg 2N: one look at the ETT test split for the hard design')

    parser.add_argument('--suite', dest='suite', default='etthard-20260925', help='training suite of the 48 runs')
    parser.add_argument('--data', dest='data', nargs='?', choices=['ETTh1', 'ETTh2'], required=True)
    parser.add_argument('--pred_len', dest='pred_len', nargs='?', type=int, default=96)
    parser.add_argument('-nd', '--num_device', dest='num_device', nargs='?', type=int, default=0,
                        help='CUDA device index (default: %(default)s); training ran on CUDA, so does the test')
    parser.add_argument('--out', dest='out', default=None, help='record directory under results/ (default: <suite>-test-<data>)')

    return parser.parse_args()


if __name__ == '__main__':

    config = parse_arguments()
    if torch.__version__ != TORCH:
        raise SystemExit(f"[ett_test] torch {torch.__version__} != {TORCH}: the declared tie rule is that torch.topk")
    device = torch.device(f'cuda:{config.num_device}')
    calibration = sorted(glob.glob(str(TASK / 'results' / 'calibration' / f'{config.data}_a0.7_norm-frozen_seed7_260925-*.json')))
    bound = json.load(open(calibration[-1]))['picked']['declared_bound']

    # ---- 1. 개방 전 전체 대조: 하나라도 어긋나면 test를 열지 않는다 ------------------------------
    runs, errors = {}, {}
    for cond in CONDS:
        for seed in SEEDS:
            run, result = find_run(config.suite, config.data, config.pred_len, cond, seed)
            model, args = load_model(run, device)
            bad = check_manifest(args, result, config.data, config.pred_len, cond, seed, bound)
            runs[(cond, seed)] = (model, args, sha256(Path(run) / 'model_state' / 'best+model.pt'), result)
            if bad:
                errors[f'{cond}/{seed}'] = bad
    if errors:
        for k, v in errors.items():
            print(f"[ett_test] manifest {k}: {v}")
        raise SystemExit(f"[ett_test] {len(errors)} run(s) fail the manifest; {config.data} test was NOT opened")
    print(f"[ett_test] manifest OK for all {len(runs)} runs of {config.data} H{config.pred_len}; "
          f"torch {torch.__version__}; frozen G11 bound {bound:.1f}")

    # ---- 2-3. 전역 등록부 잠금 -> 기록 생성 -------------------------------------------------------
    out = TASK / 'results' / (config.out or f'{config.suite}-test-{config.data}')
    out.mkdir(parents=True, exist_ok=False)
    record_path = out / 'ett_test_record.json'
    open_once(f'{config.data}-test-H{config.pred_len}', 'prereg 2N: hard design on ETT', record_path)
    head = {'prereg': '2N', 'data': config.data, 'pred_len': config.pred_len, 'suite': config.suite,
            'torch': torch.__version__, 'device': torch.cuda.get_device_name(device), 'bound': bound,
            'checkpoint_sha256': {f'{c}/{s}': v[2] for (c, s), v in runs.items()},
            'source_sha256': {f: sha256(TASK / f) for f in ('layers.py', 'ours.py', 'model.py', 'train.py', 'test.py',
                                                           'config.py', 'ett_test.py', 'data_provider/data_loader.py')}}
    with open(record_path, 'x') as handle:
        json.dump({**head, 'status': f'opening {config.data} test'}, handle, indent=2)

    # ---- 4. 모델마다 test 전체를 한 번 ------------------------------------------------------------
    rows, mse = [], {c: [] for c in CONDS}
    for seed in SEEDS:
        for cond in CONDS:
            model, args, _, result = runs[(cond, seed)]
            print(f"{f' {config.data} {cond} seed {seed} ':=^100s}")
            r = test(args, model, cond)
            r.update(cond=cond, seed=seed, epochs_run=json.load(open(result))['train']['epochs_run'])
            rows.append(r)
            mse[cond].append(r['mse'])

    unsafe = sorted({(r['cond'], r['seed']) for r in rows if not r['safe']})

    def blocked(*conds):
        return [u for u in unsafe if u[0] in conds]

    tests = {'pearson-q1': paired(mse['pearson'], mse['q1']),
             'pearson-gru': paired(mse['pearson'], mse['gru']),
             'q1-gru': paired(mse['q1'], mse['gru'])}
    print(f"{' RESULT ':=^100s}")
    print(f"[ett_test] {config.data} H{config.pred_len} mean test MSE: "
          + ', '.join(f'{c} {np.mean(v):.6f}' for c, v in mse.items())
          + ' | reference (v2, not tested): ' + ', '.join(f'{k} {v}' for k, v in REFERENCE[config.data].items()))
    for name, t in tests.items():
        a, b = name.split('-')
        bl = blocked(a, b)
        word = f'판정 보류: unsafe {bl}' if bl else verdict(t)
        t['verdict'], t['blocked'] = word, bl
        print(f"[ett_test] {'1차' if name == 'pearson-q1' else '2차'} {name:<12} mean {t['mean']:+.6f} "
              f"95% CI [{t['ci95'][0]:+.6f}, {t['ci95'][1]:+.6f}] rel {100 * t['relative']:+.2f}% "
              f"a<b {t['a_lower']}/{t['n']} -> {word}")
    if unsafe:
        print(f"[ett_test] UNSAFE models (their verdicts are withheld): {unsafe}")
    t_all = [r['ties'] for r in rows if r['ties'] is not None]
    print(f"[ett_test] pearson top-k boundary ties {sum(t[0] for t in t_all)}/{sum(t[1] for t in t_all)}")

    json.dump({**head, 'status': 'done', 'rows': rows, 'mse': mse, 'tests': tests, 'unsafe': unsafe,
               'reference_v2': REFERENCE[config.data]}, open(record_path, 'w'), indent=2)
    print(f"Record saved to `{record_path}`")
