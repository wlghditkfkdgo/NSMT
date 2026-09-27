import torch
import numpy as np
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

import re
import glob
import json
import argparse
from pathlib import Path

from config import set_random_seed, Config
from data_provider.data_factory import data_provider
import layers
from ett_test import (sha256, load_model, stopping_evidence, REGISTERED, SEEDS, TORCH, MDE,
                      DATA_ROOT, DATA_SHA256, DATASETS, TASK)
from ett_future import CALIBRATION_R
from ett_alpha import paired

# prereg 2Q (B): model_v1's readout -- one non-spiking nn.Linear on the flattened spikes -- against
# the v3 two-stage head, same spiking model (alpha=0.7, R on, 2O calibration). Validation only,
# exploratory (D-CB). Audit 56 A29 applied: every cell of EVERY dataset is checked before any
# forward, and a cell that stopped in training is matched on run_uuid, operating point, no-test
# and the exact stop line.

S2O, S2Q = 'etthard-revin-20260927', 'etthard-readout-20260928'
SPLIT = 'val'
PEARSON = {'hard_axis': 'shared', 'hard_stat': 'pearson'}
BASE = {'model': 'myModel', 'mode': 'hard', 'revin': True, 'alpha': .7, **PEARSON}
CONDITIONS = {
    'q1_R':          {'suite': S2O, 'head_mode': 'flatten', 'hard_q': 1., **BASE},
    'pearson_R':     {'suite': S2O, 'head_mode': 'flatten', 'hard_q': .5, **BASE},
    'q1_R_lin':      {'suite': S2Q, 'head_mode': 'linear', 'hard_q': 1., **BASE},
    'pearson_R_lin': {'suite': S2Q, 'head_mode': 'linear', 'hard_q': .5, **BASE},
}
REPRO_TOL = 1e-6
CONTRASTS = {                                                   # D-CB
    'H_q = q1_R(linear) - q1_R(flatten)':           ('q1_R_lin', 'q1_R'),
    'H_p = pearson_R(linear) - pearson_R(flatten)': ('pearson_R_lin', 'pearson_R'),
    'S(flatten) = pearson_R - q1_R':                ('pearson_R', 'q1_R'),
    'S(linear) = pearson_R - q1_R':                 ('pearson_R_lin', 'q1_R_lin'),
}
HEAD_RULE = ('H_q', 'H_p')                                      # D-CB의 D head 결정 규칙에 쓰는 대비


def variant_of(spec):
    return f"heterogeneous_hard-shared-pearson-q{spec['hard_q']:g}" + ('_revin' if spec['revin'] else '')


def find_cell(data, pred_len, cond, seed, spec=None):
    """(status, run dir, result json): 'done' (checkpoint + result, no G11 record) or 'g11'
    (G11 record, no checkpoint, no result). Anything else stops here."""
    spec = spec or CONDITIONS[cond]
    variant = variant_of(spec)
    runs = glob.glob(str(TASK / 'log' / spec['suite'] / data / '*'
                         / f"*+model+{spec['model']}+*+alpha+{spec['alpha']}+*"
                         / f"seed{seed}_{spec['head_mode']}_spike_{variant}"))
    results = glob.glob(str(TASK / 'results' / spec['suite']
                            / f"{spec['model']}_ett_{data}_p{pred_len}_{spec['head_mode']}_a{spec['alpha']}_spike_{variant}_seed{seed}.json"))
    if len(runs) != 1 or len(results) > 1:
        raise SystemExit(f"[ett_readout] {data} {cond} seed {seed}: {len(runs)} run dirs, {len(results)} results")
    run = Path(runs[0])
    has_ckpt = (run / 'model_state' / 'best+model.pt').exists()
    has_g11 = (run / 'G11_violation.json').exists()
    if has_ckpt and results and not has_g11:
        return 'done', str(run), results[0]
    if has_g11 and not has_ckpt and not results:
        return 'g11', str(run), None
    raise SystemExit(f"[ett_readout] {data} {cond} seed {seed}: checkpoint {has_ckpt}, result {bool(results)}, "
                     f"G11 record {has_g11} -- neither a finished run nor a recorded G11 failure")


def calibrated(data):
    """(file, fixed sha256, picked operating point) of the alpha=0.7 R-on calibration (2O)."""
    name, digest = CALIBRATION_R[data]
    return name, digest, json.load(open(TASK / 'results' / 'calibration' / name))['picked']


def check_manifest(args, result, data, pred_len, cond, seed):
    """Every registered field for a finished run; returns the mismatches (empty = OK)."""
    spec = CONDITIONS[cond]
    want = {**REGISTERED, 'data': data, 'pred_len': pred_len, 'seed': seed,
            **{k: v for k, v in spec.items() if k != 'suite'}}
    bad = [f"{k}: {getattr(args, k, None)!r} != {v!r}" for k, v in want.items() if getattr(args, k, None) != v]
    name, _, picked = calibrated(data)
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
    if (provenance.get('calibration') or {}).get('file') != name:
        bad.append(f"calibration {(provenance.get('calibration') or {}).get('file')!r} != {name!r}")
    bad += stopping_evidence(args, train, cond, data, pred_len, seed)

    return bad


def check_failure(run, data, pred_len, cond, seed, spec=None, calibration=None):
    """A cell that stopped on G11 (audit 56 A29-FAILURE-PROVENANCE). The config, the G11 record and
    the stdout must describe the same run at the fixed operating point:
    run_id and run_uuid, input_scale / calibration file / bound, no-test, and the stop line's
    epoch, batch, peak and bound."""
    spec = spec or CONDITIONS[cond]
    name, picked = calibration if calibration else (calibrated(data)[0], calibrated(data)[2])
    bad = []
    saved = torch.load(Path(run) / 'model_state' / 'config.pt', map_location='cpu')
    want = {**REGISTERED, 'data': data, 'pred_len': pred_len, 'seed': seed, 'suite': spec['suite'],
            **{k: v for k, v in spec.items() if k != 'suite'}}
    bad += [f"config {k}: {saved.get(k)!r} != {v!r}" for k, v in want.items() if saved.get(k) != v]
    if saved.get('input_scale') != picked['input_scale']:
        bad.append(f"config input_scale {saved.get('input_scale')!r} != {picked['input_scale']!r}")
    if saved.get('calibration_file') != name:
        bad.append(f"config calibration_file {saved.get('calibration_file')!r} != {name!r}")
    if saved.get('g11_bound') != picked['declared_bound']:
        bad.append(f"config g11_bound {saved.get('g11_bound')!r} != {picked['declared_bound']!r}")
    if saved.get('test') is not False:
        bad.append(f"config test {saved.get('test')!r} (the run was not --no-test)")
    record = json.load(open(Path(run) / 'G11_violation.json'))
    for key in ('run_id', 'run_uuid'):
        if record.get(key) != saved.get(key):
            bad.append(f"G11 record {key} {record.get(key)!r} != config {saved.get(key)!r}")
    if record.get('bound') != picked['declared_bound']:
        bad.append(f"G11 record bound {record.get('bound')!r} != {picked['declared_bound']!r}")
    if record.get('calibration_file') != name:
        bad.append(f"G11 record calibration {record.get('calibration_file')!r} != {name!r}")
    if not record.get('max_abs_state', 0.) >= record.get('bound', float('inf')):
        bad.append('G11 record does not show the bound reached')
    stdout = glob.glob(str(TASK / 'log' / spec['suite'] / f'ett_{data}_p{pred_len}_{cond}_seed{seed}_*.stdout'))
    if len(stdout) != 1:
        bad.append(f"{len(stdout)} stdout files")
        return bad, record
    line = re.search(r'FloatingPointError: G11 violated: max\|u\| = ([0-9.]+) reached the bound ([0-9.]+) '
                     r'frozen at calibration \(epoch (\d+), batch (\d+)\)', Path(stdout[0]).read_text())
    if not line:
        bad.append('stdout has no G11 stop line')
    else:
        peak, bound, epoch, batch = float(line[1]), float(line[2]), int(line[3]), int(line[4])
        if (epoch, batch) != (record.get('epoch'), record.get('batch')) or abs(peak - record.get('max_abs_state', -1.)) > 5e-4 \
                or abs(bound - record.get('bound', -1.)) > 5e-4:
            bad.append(f"stdout stop line (epoch {epoch}, batch {batch}, peak {peak}, bound {bound}) != G11 record")

    return bad, record


def test(args:Config, model, cond, flag=SPLIT):
    """Whole split, one forward per batch: MSE/MAE, per-channel MSE, safety, kept mass, ties, firing rate."""

    set_random_seed(args.seed)

    _, loader = data_provider(args, flag=flag)

    mse = MeanSquaredError().to(args.device)
    mae = MeanAbsoluteError().to(args.device)

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
            err = (output.double() - y.double()).square().sum(dim=(0, 1))   # [C] 채널별
            sse = err if sse is None else sse + err
            per_channel_n += output.shape[0] * output.shape[1]
            windows += x.shape[0]

            state = aux['state']                                       # [T, B*C, D, K]
            if not (torch.isfinite(state).all() and torch.isfinite(output).all()):
                finite, nonfinite = False, nonfinite + 1
            else:
                peak = max(peak, state.abs().max().item())
            spikes += aux['spikes'].double().mean(dim=(0, 2)).sum().item()
            spike_n += aux['spikes'].shape[1]

            b = model.embedding.neuron.b
            share = torch.zeros(state.shape[1], dtype=torch.float64, device=args.device)
            for n in range(1, len(aux['coeff'])):
                c = aux['coeff'][n].double()
                share += (c.sum(-1) / b[1:n + 1].sum().double()).mean(-1)
            kept += (share / (len(aux['coeff']) - 1)).sum().item()
            kept_n += state.shape[1]

            if count_ties:
                from ours import window_norm
                selector = model.embedding.neuron.selector
                xin = window_norm(x)[0]
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
        'max_abs_state' : peak,
        'bound' : bound,
        'safe' : finite and bound is not None and peak < bound,
        'kept_mass_frac' : kept / kept_n if kept_n else None,
        'firing_rate' : spikes / spike_n if spike_n else None,
        'ties' : [ties, tie_n] if count_ties else None,
        'mask_mismatch' : mismatch if count_ties else None,
    }

    print("Test was successfully done")

    head = ','.join(["split", "mse", "mae", "windows", "safe", "max_abs_state", "bound", "kept_mass_frac",
                     "firing_rate", "ties", "mse_per_channel"])
    results_csv = ','.join([
        f"{flag}-2Q",
        f"{test_result['mse']:.6f}",
        f"{test_result['mae']:.6f}",
        f"{test_result['windows']}",
        f"{test_result['safe']}",
        f"{test_result['max_abs_state']:.4f}",
        f"{bound:.4f}",
        f"{test_result['kept_mass_frac']:.4f}",
        f"{test_result['firing_rate']:.4f}",
        f"{ties}/{tie_n}" if count_ties else "None",
        ' '.join(f"{v:.6f}" for v in test_result['mse_per_channel']), ]
    )

    for k, v in test_result.items():
        if isinstance(v, float):
            print(f" > {k:16s}:{v:>9.6f}")

    if CONDITIONS[cond]['suite'] == S2Q:                               # 2O 재사용 run의 로그는 건드리지 않는다
        with open(args.save_log_path + '/final+result.csv', 'a', encoding='utf-8') as log_csv:
            print(head, end="\n", file=log_csv)
            print(results_csv, end="\n", file=log_csv)
        print(f"Final result saved to `{args.save_log_path}`")

    return test_result


def analyse(rows):
    """Blocked cells and the D-CB contrasts for one dataset; a comparison that uses a cell that
    failed in training, was unsafe or did not reproduce its best validation MSE is withheld."""
    blocked = {(r['cond'], r['seed']): ('G11 failure in training' if r['status'] != 'evaluated'
                                        else 'unsafe in evaluation' if not r['safe']
                                        else 'validation MSE not reproduced')
               for r in rows if r['status'] != 'evaluated' or not r['safe'] or not r['repro_ok']}
    mse = {(r['cond'], r['seed']): r['mse'] for r in rows if r['status'] == 'evaluated'}
    out = {}
    for name, (a, b) in CONTRASTS.items():
        block = sorted([list(k) + [v] for k, v in blocked.items() if k[0] in (a, b)])
        if block:
            out[name] = {'uses': [a, b], 'blocked': block, 'status': 'withheld'}
            continue
        t = paired([mse[(a, s)] for s in SEEDS], [mse[(b, s)] for s in SEEDS])
        t.update(uses=[a, b], blocked=[], status='reported (exploratory, 95% CI)')
        out[name] = t

    return blocked, out


def head_decision(per_data):
    """D-CB: keep the user's 'linear' unless one of H_q, H_p on either dataset is clearly worse
    (95% lower bound > 0 and mean >= +MDE); a withheld H counts against deciding."""
    reasons = []
    for data, contrasts in per_data.items():
        for name, t in contrasts.items():
            if name.split(' ')[0] not in HEAD_RULE:
                continue
            if t['status'] == 'withheld':
                reasons.append(f'{data} {name}: withheld')
            elif t['ci'][0] > 0 and t['mean'] >= MDE:
                reasons.append(f"{data} {name}: mean {t['mean']:+.6f}, 95% CI [{t['ci'][0]:+.6f}, {t['ci'][1]:+.6f}]")
    return ('flatten' if reasons else 'linear'), reasons


def gate(pred_len, device):
    """Every cell of every dataset, before any forward (A29-GATE-COVERAGE)."""
    errors, runs, failures = [], {}, {}
    for data in DATASETS:
        name, digest, _ = calibrated(data)
        if sha256(TASK / 'results' / 'calibration' / name) != digest:
            errors.append(f"{data}: calibration {name} sha256 is not the fixed one")
        if sha256(DATA_ROOT / f'{data}.csv') != DATA_SHA256[data]:
            errors.append(f"{data}: data CSV sha256 is not the fixed one")
        for cond in CONDITIONS:
            for seed in SEEDS:
                status, run, result = find_cell(data, pred_len, cond, seed)
                if status == 'g11':
                    bad, record = check_failure(run, data, pred_len, cond, seed)
                    failures[(data, cond, seed)] = {'run': run, 'record': record,
                                                    'config_sha256': sha256(Path(run) / 'model_state' / 'config.pt')}
                else:
                    model, args = load_model(run, device)
                    bad = check_manifest(args, result, data, pred_len, cond, seed)
                    runs[(data, cond, seed)] = (model, args, sha256(Path(run) / 'model_state' / 'best+model.pt'),
                                                sha256(Path(run) / 'model_state' / 'config.pt'), result)
                errors += [f"{data} {cond}/{seed}: {b}" for b in bad]

    return errors, runs, failures


def parse_arguments():

    parser = argparse.ArgumentParser(description='prereg 2Q (B): single Linear readout vs two-stage head on ETT validation')

    parser.add_argument('--pred_len', dest='pred_len', nargs='?', type=int, default=96)
    parser.add_argument('-nd', '--num_device', dest='num_device', nargs='?', type=int, default=0)
    parser.add_argument('--out', dest='out', default=None, help='record directory under results/ (default: <S2Q>-val)')

    return parser.parse_args()


if __name__ == '__main__':

    config = parse_arguments()
    if torch.__version__ != TORCH:
        raise SystemExit(f"[ett_readout] torch {torch.__version__} != {TORCH}: the declared tie rule is that torch.topk")
    device = torch.device(f'cuda:{config.num_device}')

    # ---- 1. 관문: 두 데이터셋의 모든 칸을 어떤 forward보다 먼저 (A29) ------------------------------
    errors, runs, failures = gate(config.pred_len, device)
    if errors:
        for e in errors:
            print(f"[ett_readout] manifest {e}")
        raise SystemExit(f"[ett_readout] {len(errors)} manifest error(s); nothing evaluated")
    print(f"[ett_readout] manifest OK for every dataset: {len(runs)} finished runs, {len(failures)} recorded G11 failures; "
          f"torch {torch.__version__}")

    out = TASK / 'results' / (config.out or f'{S2Q}-{SPLIT}')
    out.mkdir(parents=True, exist_ok=False)
    record_path = out / 'ett_readout_record.json'
    head = {'prereg': '2Q (B)', 'pred_len': config.pred_len, 'split': SPLIT, 'exploratory': True,
            'suites': [S2O, S2Q], 'torch': torch.__version__, 'device': torch.cuda.get_device_name(device),
            'checkpoint_sha256': {'/'.join(map(str, k)): v[2] for k, v in runs.items()},
            'config_sha256': {**{'/'.join(map(str, k)): v[3] for k, v in runs.items()},
                              **{'/'.join(map(str, k)): v['config_sha256'] for k, v in failures.items()}},
            'data_sha256': {d: DATA_SHA256[d] for d in DATASETS},
            'calibration': {d: list(CALIBRATION_R[d]) for d in DATASETS},
            'source_sha256': {f: sha256(TASK / f) for f in ('layers.py', 'ours.py', 'model.py', 'train.py', 'test.py',
                                                           'config.py', 'ett_test.py', 'ett_future.py', 'ett_alpha.py',
                                                           'ett_readout.py', 'data_provider/data_loader.py')}}

    # ---- 2. 모델마다 validation 전체를 한 번 --------------------------------------------------------
    per_data, rows_all, blocked_all = {}, [], {}
    for data in DATASETS:
        rows = []
        for seed in SEEDS:
            for cond in CONDITIONS:
                if (data, cond, seed) in failures:
                    rows.append({'data': data, 'cond': cond, 'seed': seed, 'status': 'g11_failure_in_training',
                                 'g11': failures[(data, cond, seed)]['record'], 'safe': False})
                    continue
                model, args, _, _, result = runs[(data, cond, seed)]
                print(f"{f' {data} {cond} seed {seed} ':=^100s}")
                r = test(args, model, cond)
                stored = json.load(open(result))['train']
                n_params = sum(p.numel() for p in model.parameters())
                r.update(data=data, cond=cond, seed=seed, status='evaluated', head_mode=args.head_mode,
                         parameters=n_params, epochs_run=stored['epochs_run'], best_val_loss=stored['best_val_loss'],
                         repro_ok=abs(r['mse'] - stored['best_val_loss']) <= REPRO_TOL)
                rows.append(r)
        blocked, contrasts = analyse(rows)
        per_data[data], blocked_all[data] = contrasts, blocked
        rows_all += rows
    decision, reasons = head_decision(per_data)

    print(f"{' RESULT (exploratory, validation) ':=^100s}")
    for data in DATASETS:
        means = {c: np.mean([r['mse'] for r in rows_all if r['data'] == data and r['cond'] == c and r['status'] == 'evaluated'])
                 for c in CONDITIONS}
        print(f"[ett_readout] {data} val mean MSE: " + ', '.join(f"{c} {v:.6f}" for c, v in means.items()))
        for name, t in per_data[data].items():
            if t['status'] == 'withheld':
                print(f"[ett_readout] {data} {name:<46} WITHHELD: {t['blocked'][:2]}")
                continue
            print(f"[ett_readout] {data} {name:<46} mean {t['mean']:+.6f}  95% CI [{t['ci'][0]:+.6f}, {t['ci'][1]:+.6f}]  "
                  f"rel {100 * t['relative']:+.2f}%  a<b {t['a_lower']}/{t['n']}")
        print(f"[ett_readout] {data} blocked cells: {sorted(blocked_all[data].items())}")
    ev = [r for r in rows_all if r['status'] == 'evaluated']
    print(f"[ett_readout] parameters by head: " + ', '.join(sorted({f"{r['head_mode']} {r['parameters']}" for r in ev})))
    print(f"[ett_readout] reproduction worst |diff| {max(abs(r['mse'] - r['best_val_loss']) for r in ev):.2e}; "
          f"all safe {all(r['safe'] for r in ev)}; mask mismatches {sum(r.get('mask_mismatch') or 0 for r in ev)}")
    print(f"[ett_readout] D-CB head for D: {decision}" + (f" ({'; '.join(reasons)})" if reasons else
                                                        " (no H clearly worse: 95% lower bound > 0 with mean >= +0.005)"))

    with open(record_path, 'x') as handle:
        json.dump({**head, 'status': 'done', 'rows': rows_all,
                   'blocked_cells': {d: [list(k) + [v] for k, v in sorted(b.items())] for d, b in blocked_all.items()},
                   'contrasts': per_data, 'head_for_D': {'head_mode': decision, 'reasons': reasons}},
                  handle, indent=2, allow_nan=False)
    print(f"Record saved to `{record_path}`")
