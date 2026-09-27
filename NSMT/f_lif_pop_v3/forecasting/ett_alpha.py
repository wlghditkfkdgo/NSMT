import torch
import numpy as np
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from scipy import stats

import glob
import hashlib
import json
import math
import argparse
from pathlib import Path

from config import set_random_seed, Config
from data_provider.data_factory import data_provider
import layers
from ett_test import (sha256, load_model, stopping_evidence, REGISTERED, SEEDS, TORCH, MDE,
                      DATA_ROOT, DATA_SHA256, DATASETS, TASK)
from ett_future import CALIBRATION_R

# prereg 2P: the spiking fairness control. Same spiking model, R on, with ordinary LIF branches
# (alpha=1, every b_d = 1) against the fractional kernel (alpha=0.7). ETT test and future are
# both opened, so this is validation only and exploratory (D-BT); no split is registered.
# Audit 52 A25 applied from the start: per-channel errors from the same forward, every
# comparison carries the unsafe/failed cells it uses, no relative change against zero, strict JSON.

S2O, S2P = 'etthard-revin-20260927', 'etthard-alpha1-20260927'
SPLIT = 'val'
PEARSON = {'hard_axis': 'shared', 'hard_stat': 'pearson'}
CONDITIONS = {
    'q1_R':         {'suite': S2O, 'model': 'myModel', 'mode': 'hard', 'revin': True, 'alpha': .7, 'hard_q': 1., **PEARSON},
    'pearson_R':    {'suite': S2O, 'model': 'myModel', 'mode': 'hard', 'revin': True, 'alpha': .7, 'hard_q': .5, **PEARSON},
    'q1_R_a1':      {'suite': S2P, 'model': 'myModel', 'mode': 'hard', 'revin': True, 'alpha': 1., 'hard_q': 1., **PEARSON},
    'pearson_R_a1': {'suite': S2P, 'model': 'myModel', 'mode': 'hard', 'revin': True, 'alpha': 1., 'hard_q': .5, **PEARSON},
    'gru_R':        {'suite': S2O, 'model': 'GRU', 'mode': 'full', 'revin': True, 'alpha': .7},
    'linear_R':     {'suite': S2O, 'model': 'Linear', 'mode': 'full', 'revin': True, 'alpha': .7},
}
# R 켠 보정: alpha=0.7은 2O의 고정 artifact, alpha=1은 2P D-BS (commit 3cde96941)
CALIBRATION = {**{(d, .7): v for d, v in CALIBRATION_R.items()},
               ('ETTh1', 1.): ('ETTh1_a1.0_norm-frozen_revin_seed7_260927-214003.json',
                               '98d3caa9ffc3da811f9c58d46053825c4e945f12bd46663f002c47895fcc39a3'),
               ('ETTh2', 1.): ('ETTh2_a1.0_norm-frozen_revin_seed7_260927-214002.json',
                               '5875ea5fe51fa7774584ebabb65ef03cfcf488867143cbdce2c7e8b0146b88e0')}
REPRO_TOL = 1e-6                                                # D-BU: 다시 잰 val MSE와 best_val_loss
CONTRASTS = {                                                   # D-BV: name -> (a, b) or ((a, b), (c, d)) for J
    'F_q = q1_R(0.7) - q1_R(1)':             ('q1_R', 'q1_R_a1'),
    'F_p = pearson_R(0.7) - pearson_R(1)':   ('pearson_R', 'pearson_R_a1'),
    'S_on(1) = pearson_R(1) - q1_R(1)':      ('pearson_R_a1', 'q1_R_a1'),
    'J = S_on(0.7) - S_on(1)':               (('pearson_R', 'q1_R'), ('pearson_R_a1', 'q1_R_a1')),
    'q1_R(1) - gru_R':                       ('q1_R_a1', 'gru_R'),
    'q1_R(1) - linear_R':                    ('q1_R_a1', 'linear_R'),
    'q1_R(0.7) - gru_R':                     ('q1_R', 'gru_R'),
    'q1_R(0.7) - linear_R':                  ('q1_R', 'linear_R'),
}
LABELLED = ('F_q', 'F_p')                                       # D-BV 표현 규칙을 적용하는 alpha 대비


def variant_of(spec):
    if spec['model'] == 'myModel':
        v = f"heterogeneous_hard-shared-pearson-q{spec['hard_q']:g}"
    else:
        v = 'heterogeneous_full'

    return v + ('_revin' if spec['revin'] else '')


def find_cell(data, pred_len, cond, seed):
    """(status, run dir, result json). Exactly one of: 'done' (checkpoint + result, no G11 record)
    or 'g11' (G11 record, no checkpoint, no result) -- 2P D-BX. Anything else stops here."""
    spec = CONDITIONS[cond]
    variant = variant_of(spec)
    runs = glob.glob(str(TASK / 'log' / spec['suite'] / data / '*'
                         / f"*+model+{spec['model']}+*+alpha+{spec['alpha']}+*" / f'seed{seed}_*_{variant}'))
    results = glob.glob(str(TASK / 'results' / spec['suite']
                            / f"{spec['model']}_ett_{data}_p{pred_len}_*_a{spec['alpha']}_*_{variant}_seed{seed}.json"))
    if len(runs) != 1 or len(results) > 1:
        raise SystemExit(f"[ett_alpha] {data} {cond} seed {seed}: {len(runs)} run dirs, {len(results)} results")
    run = Path(runs[0])
    has_ckpt = (run / 'model_state' / 'best+model.pt').exists()
    has_g11 = (run / 'G11_violation.json').exists()
    if has_ckpt and results and not has_g11:
        return 'done', str(run), results[0]
    if has_g11 and not has_ckpt and not results:
        return 'g11', str(run), None
    raise SystemExit(f"[ett_alpha] {data} {cond} seed {seed}: checkpoint {has_ckpt}, result {bool(results)}, "
                     f"G11 record {has_g11} -- neither a finished run nor a recorded G11 failure")


def calibration_of(data, alpha):
    return CALIBRATION[(data, alpha)]


def calibrated(data, alpha):
    return json.load(open(TASK / 'results' / 'calibration' / calibration_of(data, alpha)[0]))['picked']


def check_manifest(args, result, data, pred_len, cond, seed):
    """Every registered field for a finished run; returns the mismatches (empty = OK)."""
    spec = CONDITIONS[cond]
    want = {**REGISTERED, 'data': data, 'pred_len': pred_len, 'seed': seed,
            **{k: v for k, v in spec.items() if k != 'suite'}}
    bad = [f"{k}: {getattr(args, k, None)!r} != {v!r}" for k, v in want.items() if getattr(args, k, None) != v]
    picked = calibrated(data, spec['alpha'])
    if spec['model'] == 'myModel':
        if getattr(args, 'g11_bound', None) != picked['declared_bound']:
            bad.append(f"g11_bound {getattr(args, 'g11_bound', None)!r} != {picked['declared_bound']!r}")
        if getattr(args, 'input_scale', None) != picked['input_scale']:
            bad.append(f"input_scale {getattr(args, 'input_scale', None)!r} != {picked['input_scale']!r}")
    if getattr(args, 'data_path', None) != f'{data}.csv':
        bad.append(f"data_path {getattr(args, 'data_path', None)!r} != '{data}.csv'")
    if Path(getattr(args, 'root_path', '')).resolve() != DATA_ROOT:
        bad.append(f"root_path {getattr(args, 'root_path', None)!r} != {str(DATA_ROOT)!r}")
    payload = json.load(open(result))
    train, provenance = payload.get('train', {}), payload.get('provenance', {})
    if not 1 <= (train.get('epochs_run') or 0) <= REGISTERED['epoch']:
        bad.append(f"epochs_run {train.get('epochs_run')!r} outside 1..{REGISTERED['epoch']}")
    if payload.get('test') is not None or payload.get('test_skipped') is not True:
        bad.append('test was not skipped in training')
    if provenance.get('checkpoint_sha256') != sha256(Path(args.save_model_state_path) / 'best+model.pt'):
        bad.append('checkpoint sha256 differs from the one the training run recorded')
    if not str(provenance.get('device', '')).startswith('cuda'):
        bad.append(f"trained on {provenance.get('device')!r}, not CUDA")
    if provenance.get('torch') != TORCH:
        bad.append(f"trained with torch {provenance.get('torch')!r} != {TORCH}")
    if (provenance.get('calibration') or {}).get('file') != calibration_of(data, spec['alpha'])[0]:
        bad.append(f"calibration {(provenance.get('calibration') or {}).get('file')!r} != "
                   f"{calibration_of(data, spec['alpha'])[0]!r}")
    bad += stopping_evidence(args, train, cond, data, pred_len, seed)

    return bad


def check_failure(run, data, pred_len, cond, seed):
    """A registered G11 failure (2P D-BX): the record, the config and the stdout must agree."""
    spec = CONDITIONS[cond]
    bad = []
    saved = torch.load(Path(run) / 'model_state' / 'config.pt', map_location='cpu')
    want = {**REGISTERED, 'data': data, 'pred_len': pred_len, 'seed': seed, 'suite': spec['suite'],
            **{k: v for k, v in spec.items() if k != 'suite'}}
    bad += [f"config {k}: {saved.get(k)!r} != {v!r}" for k, v in want.items() if saved.get(k) != v]
    picked = calibrated(data, spec['alpha'])
    record = json.load(open(Path(run) / 'G11_violation.json'))
    if record.get('run_id') != saved.get('run_id'):
        bad.append(f"G11 record run_id {record.get('run_id')!r} != config {saved.get('run_id')!r}")
    if record.get('bound') != picked['declared_bound'] or saved.get('g11_bound') != picked['declared_bound']:
        bad.append(f"G11 bound {record.get('bound')!r} / config {saved.get('g11_bound')!r} != {picked['declared_bound']!r}")
    if record.get('calibration_file') != calibration_of(data, spec['alpha'])[0]:
        bad.append(f"G11 record calibration {record.get('calibration_file')!r}")
    if not record.get('max_abs_state', 0.) >= record.get('bound', float('inf')):
        bad.append('G11 record does not show the bound reached')
    stdout = glob.glob(str(TASK / 'log' / spec['suite'] / f'ett_{data}_p{pred_len}_{cond}_seed{seed}_*.stdout'))
    if len(stdout) != 1 or 'FloatingPointError: G11 violated' not in Path(stdout[0]).read_text():
        bad.append(f"{len(stdout)} stdout file(s) with the G11 stop")

    return bad, record


def test(args:Config, model, cond, flag=SPLIT):
    """Whole split, one forward per batch: MSE/MAE, per-channel MSE, safety, kept mass, ties, firing rate."""

    set_random_seed(args.seed)

    _, loader = data_provider(args, flag=flag)

    mse = MeanSquaredError().to(args.device)
    mae = MeanAbsoluteError().to(args.device)

    spiking = args.model == 'myModel'
    count_ties = cond.startswith('pearson')
    peak, finite, nonfinite, windows = 0., True, 0, 0
    kept, kept_n, ties, tie_n, mismatch, spikes, spike_n = 0., 0, 0, 0, 0, 0., 0
    sse, per_channel_n = None, 0
    with torch.no_grad():
        model.eval()

        for i, batch in enumerate(loader):
            x, y, _, _ = batch
            x = x.float().to(args.device)
            y = y.float().to(args.device)

            output, aux = model(x, mode=args.mode, return_aux=True)   # 오차와 상태를 같은 forward에서
            mse.update(output.contiguous(), y.contiguous())
            mae.update(output.contiguous(), y.contiguous())
            err = (output.double() - y.double()).square().sum(dim=(0, 1))   # [C] 채널별 (감사 52 A25)
            sse = err if sse is None else sse + err
            per_channel_n += output.shape[0] * output.shape[1]
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
            spikes += aux['spikes'].double().mean(dim=(0, 2)).sum().item()
            spike_n += aux['spikes'].shape[1]

            b = model.embedding.neuron.b
            share = torch.zeros(state.shape[1], dtype=torch.float64, device=args.device)
            for n in range(1, len(aux['coeff'])):
                c = aux['coeff'][n].double()                           # [B*C, D, n] = m * b
                share += (c.sum(-1) / b[1:n + 1].sum().double()).mean(-1)
            kept += (share / (len(aux['coeff']) - 1)).sum().item()
            kept_n += state.shape[1]

            if count_ties:                                             # 학습된 모델의 top-k 경계 동점
                from ours import window_norm
                selector = model.embedding.neuron.selector
                xin = window_norm(x)[0]                                # 모델이 본 입력 = R 변환 후
                current = model.embedding.current(layers.to_patches(xin, args.patch_size))
                zero = torch.zeros_like(state[0])
                for n in range(1, current.shape[0]):
                    xi = torch.cat([state[n - 1], current[n].unsqueeze(-1)], dim=-1)
                    hist = torch.stack([torch.cat([state[j - 1] if j else zero, current[j].unsqueeze(-1)], -1)
                                        for j in range(n)], dim=-2)
                    m, sim = selector.hard_mask(xi, hist)
                    mismatch += int(not torch.equal(m, (aux['coeff'][n] > 0).to(m.dtype)))
                    k = max(1, int(round(selector.hard_q * n)))
                    if k < n:
                        ordered = sim[:, 0, :].sort(dim=-1, descending=True).values
                        ties += int((ordered[:, k - 1] == ordered[:, k]).sum())
                    tie_n += xi.shape[0]

    bound = getattr(args, 'g11_bound', None)
    test_result = {
        'mse' : mse.compute().item(),
        'mae' : mae.compute().item(),
        'mse_per_channel' : (sse / per_channel_n).tolist(),
        'windows' : windows,
        'finite' : finite,
        'nonfinite_batches' : nonfinite,
        'max_abs_state' : peak if spiking else None,
        'bound' : bound if spiking else None,
        'safe' : finite and (not spiking or (bound is not None and peak < bound)),
        'kept_mass_frac' : kept / kept_n if kept_n else None,
        'firing_rate' : spikes / spike_n if spike_n else None,
        'ties' : [ties, tie_n] if count_ties else None,
        'mask_mismatch' : mismatch if count_ties else None,
    }

    print("Test was successfully done")

    head = ','.join(["split", "mse", "mae", "windows", "safe", "max_abs_state", "bound", "kept_mass_frac",
                     "firing_rate", "ties", "mse_per_channel"])
    results_csv = ','.join([
        f"{flag}-2P",
        f"{test_result['mse']:.6f}",
        f"{test_result['mae']:.6f}",
        f"{test_result['windows']}",
        f"{test_result['safe']}",
        f"{test_result['max_abs_state']:.4f}" if spiking else "None",
        f"{bound:.4f}" if spiking else "None",
        f"{test_result['kept_mass_frac']:.4f}" if spiking else "None",
        f"{test_result['firing_rate']:.4f}" if spiking else "None",
        f"{ties}/{tie_n}" if count_ties else "None",
        ' '.join(f"{v:.6f}" for v in test_result['mse_per_channel']), ]
    )

    for k, v in test_result.items():
        if isinstance(v, float):
            print(f" > {k:16s}:{v:>9.6f}")

    if CONDITIONS[cond]['suite'] == S2P:                               # 2O 재사용 run의 로그는 건드리지 않는다
        with open(args.save_log_path + '/final+result.csv', 'a', encoding='utf-8') as log_csv:
            print(head, end="\n", file=log_csv)
            print(results_csv, end="\n", file=log_csv)
        print(f"Final result saved to `{args.save_log_path}`")

    return test_result


def paired(a, b, level=.95, relative=True):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n)

    return {'deltas': d.tolist(), 'mean': float(mean), 'sd': float(sd), 'level': level,
            'ci': [float(mean - half), float(mean + half)],
            'relative': float(mean / np.mean(b)) if relative else None,     # 0과의 비교에는 정의하지 않는다
            'a_lower': int((d < 0).sum()), 'n': n}


def label(t):
    """2P D-BV wording, corrected by D-BY (audit 54): each alpha runs at its OWN calibrated scale, so
    the contrast is alpha together with its calibration, not the kernel alone. Exploratory only."""
    if t['ci'][1] < 0 and t['mean'] <= -MDE:
        return 'validation에서 α=0.7(자체 보정) 쪽 오차가 낮은 탐색적 신호'
    if t['ci'][0] > 0 and t['mean'] >= MDE:
        return 'validation에서 α=1(자체 보정) 쪽 오차가 낮은 탐색적 신호'
    return 'validation에서 차이를 보이지 못함 (무효과·동등성의 증거가 아님)'


def analyse(rows):
    """Blocked cells, the D-BV contrasts and the per-channel F_q, from the evaluation rows.

    A comparison is withheld when any cell it uses failed in training (G11), was unsafe in the
    evaluation, or did not reproduce its stored best validation MSE (audit 52 A25-UNSAFE-SECONDARY).
    """
    blocked_cells = {(r['cond'], r['seed']): ('G11 failure in training' if r['status'] != 'evaluated'
                                              else 'unsafe in evaluation' if not r['safe']
                                              else 'validation MSE not reproduced')
                     for r in rows if r['status'] != 'evaluated' or not r['safe'] or not r['repro_ok']}
    mse = {(r['cond'], r['seed']): r['mse'] for r in rows if r['status'] == 'evaluated'}
    contrasts = {}
    for name, spec in CONTRASTS.items():
        uses = [c for pair in spec for c in pair] if isinstance(spec[0], tuple) else list(spec)
        block = sorted([list(k) + [v] for k, v in blocked_cells.items() if k[0] in uses])
        if block:
            contrasts[name] = {'uses': uses, 'blocked': block, 'status': 'withheld'}
            continue
        if isinstance(spec[0], tuple):
            (a, b), (c, d) = spec
            diff = [(mse[(a, s)] - mse[(b, s)]) - (mse[(c, s)] - mse[(d, s)]) for s in SEEDS]
            t = paired(diff, [0.] * len(diff), relative=False)
        else:
            a, b = spec
            t = paired([mse[(a, s)] for s in SEEDS], [mse[(b, s)] for s in SEEDS])
        t.update(uses=uses, blocked=[], status='reported (exploratory, 95% CI)')
        if name.split(' ')[0] in LABELLED:
            t['label'] = label(t)
        contrasts[name] = t

    per_channel = None                                          # 채널별 F_q (기술만)
    if contrasts['F_q = q1_R(0.7) - q1_R(1)']['status'] != 'withheld':
        pc = {c: np.array([next(r['mse_per_channel'] for r in rows if r['cond'] == c and r['seed'] == s)
                           for s in SEEDS]) for c in ('q1_R', 'q1_R_a1')}
        per_channel = {'F_q_mean': (pc['q1_R'] - pc['q1_R_a1']).mean(0).tolist(),
                       'F_q_a_lower': ((pc['q1_R'] - pc['q1_R_a1']) < 0).sum(0).tolist(),
                       'q1_R_mean': pc['q1_R'].mean(0).tolist(), 'q1_R_a1_mean': pc['q1_R_a1'].mean(0).tolist()}

    return blocked_cells, contrasts, per_channel


def gate(data, pred_len, device):
    """Manifest for all 6 conditions x 8 seeds of one dataset, before any validation pass.
    Returns the errors, the finished runs and the recorded G11 failures."""
    errors, runs, failures = [], {}, {}
    for alpha in (.7, 1.):
        name, digest = calibration_of(data, alpha)
        if sha256(TASK / 'results' / 'calibration' / name) != digest:
            errors.append(f"{data}: calibration {name} sha256 is not the fixed one")
    if sha256(DATA_ROOT / f'{data}.csv') != DATA_SHA256[data]:
        errors.append(f"{data}: data CSV sha256 is not the fixed one")
    for cond in CONDITIONS:
        for seed in SEEDS:
            status, run, result = find_cell(data, pred_len, cond, seed)
            if status == 'g11':
                bad, record = check_failure(run, data, pred_len, cond, seed)
                failures[(cond, seed)] = {'run': run, 'record': record,
                                          'config_sha256': sha256(Path(run) / 'model_state' / 'config.pt')}
            else:
                model, args = load_model(run, device)
                bad = check_manifest(args, result, data, pred_len, cond, seed)
                runs[(cond, seed)] = (model, args, sha256(Path(run) / 'model_state' / 'best+model.pt'),
                                      sha256(Path(run) / 'model_state' / 'config.pt'), result)
            errors += [f"{data} {cond}/{seed}: {b}" for b in bad]

    return errors, runs, failures


def parse_arguments():

    parser = argparse.ArgumentParser(description='prereg 2P: alpha=1 spiking control on ETT validation (exploratory)')

    parser.add_argument('--data', dest='data', nargs='?', choices=list(DATASETS), required=True)
    parser.add_argument('--pred_len', dest='pred_len', nargs='?', type=int, default=96)
    parser.add_argument('-nd', '--num_device', dest='num_device', nargs='?', type=int, default=0,
                        help='CUDA device index (default: %(default)s); training ran on CUDA, so does the evaluation')
    parser.add_argument('--out', dest='out', default=None, help='record directory under results/ (default: <S2P>-val-<data>)')

    return parser.parse_args()


if __name__ == '__main__':

    config = parse_arguments()
    if torch.__version__ != TORCH:
        raise SystemExit(f"[ett_alpha] torch {torch.__version__} != {TORCH}: the declared tie rule is that torch.topk")
    device = torch.device(f'cuda:{config.num_device}')

    # ---- 1. 관문: 48칸이 모두 '완료' 또는 '기록된 G11 실패'이고 지문이 맞아야 한다 (2P D-BU, D-BX) ------
    errors, runs, failures = gate(config.data, config.pred_len, device)
    if errors:
        for e in errors:
            print(f"[ett_alpha] manifest {e}")
        raise SystemExit(f"[ett_alpha] {len(errors)} manifest error(s); nothing evaluated")
    print(f"[ett_alpha] manifest OK for {config.data} H{config.pred_len}: {len(runs)} finished runs, "
          f"{len(failures)} recorded G11 failures; torch {torch.__version__}")

    out = TASK / 'results' / (config.out or f'{S2P}-{SPLIT}-{config.data}')
    out.mkdir(parents=True, exist_ok=False)
    record_path = out / 'ett_alpha_record.json'
    head = {'prereg': '2P', 'data': config.data, 'pred_len': config.pred_len, 'split': SPLIT,
            'exploratory': True, 'suites': [S2O, S2P],
            'torch': torch.__version__, 'device': torch.cuda.get_device_name(device),
            'checkpoint_sha256': {f'{c}/{s}': v[2] for (c, s), v in runs.items()},
            'config_sha256': {**{f'{c}/{s}': v[3] for (c, s), v in runs.items()},
                              **{f'{c}/{s}': v['config_sha256'] for (c, s), v in failures.items()}},
            'data_sha256': DATA_SHA256[config.data],
            'calibration': {str(a): calibration_of(config.data, a) for a in (.7, 1.)},
            'source_sha256': {f: sha256(TASK / f) for f in ('layers.py', 'ours.py', 'model.py', 'train.py', 'test.py',
                                                           'config.py', 'ett_test.py', 'ett_future.py', 'ett_alpha.py',
                                                           'data_provider/data_loader.py')}}

    # ---- 2. 모델마다 validation 전체를 한 번 ---------------------------------------------------------
    rows, mse = [], {c: {} for c in CONDITIONS}
    for seed in SEEDS:
        for cond in CONDITIONS:
            if (cond, seed) in failures:
                rows.append({'cond': cond, 'seed': seed, 'status': 'g11_failure_in_training',
                             'g11': failures[(cond, seed)]['record'], 'safe': False})
                continue
            model, args, _, _, result = runs[(cond, seed)]
            print(f"{f' {config.data} {cond} seed {seed} ':=^100s}")
            r = test(args, model, cond)
            stored = json.load(open(result))['train']
            if args.model == 'myModel':                                  # 감사 54 (c)1: 동작점 고정 표
                emb = model.embedding
                frozen = torch.cat([emb.norm_mean.flatten(), emb.norm_std.flatten()]).double().cpu().numpy().tobytes()
                r.update(alpha=args.alpha, input_scale=args.input_scale, calibration=args.calibration_file,
                         frozen_norm_sha256_16=hashlib.sha256(frozen).hexdigest()[:16])
            r.update(cond=cond, seed=seed, status='evaluated', epochs_run=stored['epochs_run'],
                     best_val_loss=stored['best_val_loss'],
                     repro_ok=abs(r['mse'] - stored['best_val_loss']) <= REPRO_TOL)
            rows.append(r)
            mse[cond][seed] = r['mse']

    blocked_cells, contrasts, per_channel = analyse(rows)

    print(f"{' RESULT (exploratory, validation) ':=^100s}")
    print(f"[ett_alpha] {config.data} H{config.pred_len} {SPLIT} mean MSE: "
          + ', '.join(f"{c} {np.mean(list(v.values())):.6f} (n={len(v)})" for c, v in mse.items() if v))
    for name, t in contrasts.items():
        if t['status'] == 'withheld':
            print(f"[ett_alpha] {name:<38} WITHHELD: {len(t['blocked'])} blocked cell(s), e.g. {t['blocked'][:2]}")
            continue
        rel = '' if t['relative'] is None else f"rel {100 * t['relative']:+.2f}%  "
        print(f"[ett_alpha] {name:<38} mean {t['mean']:+.6f}  95% CI [{t['ci'][0]:+.6f}, {t['ci'][1]:+.6f}]  "
              f"{rel}a<b {t['a_lower']}/{t['n']}" + (f"  -> {t['label']}" if 'label' in t else ''))
    if per_channel:
        print(f"[ett_alpha] per-channel F_q mean: " + ' '.join(f"{v:+.4f}" for v in per_channel['F_q_mean'])
              + f"  (a<b {per_channel['F_q_a_lower']})")
    print(f"[ett_alpha] blocked cells: {sorted(blocked_cells.items())}")
    t_all = [r['ties'] for r in rows if r.get('ties') is not None]
    print(f"[ett_alpha] pearson top-k boundary ties {sum(t[0] for t in t_all)}/{sum(t[1] for t in t_all)}; "
          f"mask mismatches {sum(r.get('mask_mismatch') or 0 for r in rows)}; "
          f"reproduction worst |diff| {max(abs(r['mse'] - r['best_val_loss']) for r in rows if r['status'] == 'evaluated'):.2e}")

    with open(record_path, 'x') as handle:
        json.dump({**head, 'status': 'done', 'rows': rows,
                   'mse': {c: {str(s): v for s, v in m.items()} for c, m in mse.items()},
                   'blocked_cells': [list(k) + [v] for k, v in sorted(blocked_cells.items())],
                   'contrasts': contrasts, 'per_channel_F_q': per_channel}, handle, indent=2, allow_nan=False)
    print(f"Record saved to `{record_path}`")
