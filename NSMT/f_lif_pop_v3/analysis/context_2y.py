"""Prereg 2Y: what the output Linear reads x window length x the neuron's state, on the 2W pick, VALIDATION only
(exploratory).

    python context_2y.py [-nd <gpu>]

Cells per base (q1_R: R on; q1: R off), dataset and seed {7, 13, 21} -- '<mem|nomem>-<L>-<all|last1>':
    mem-336-all      the 2W selection runs (state across patches, 336 steps, the Linear reads all 42 steps), not retrained
    mem-96-all       96-step window (T = 12)
    mem-336-last1    the Linear reads the last step only (128 spikes)
    mem-96-last1
    nomem-336-all    v = 0 at every patch: no state crosses patches
    nomem-336-last1  only the last patch can reach the forecast
Every cell passes its manifest (its own calibration manifest) and re-measures its stored best validation MSE
(REPRO_TOL); a blocked cell withholds every comparison it enters. Reported per base and dataset: mean val MSE, ratio
to the 2T run of the same seed, firing rate, epochs, training time, and the paired differences (95% t-interval with
2 degrees of freedom: wide, descriptive)
    C1  nomem-336-all   - mem-336-all     the state's share when the Linear reads every step
    C2  nomem-336-last1 - mem-336-last1   the state's share when it reads only the last step
    C3  mem-96-last1    - mem-336-last1   does input older than 96 steps reach the last state
    C4  mem-96-all      - mem-336-all     does the longer window help when every step is read
    C5  mem-336-last1   - mem-336-all     what the Linear loses by reading only the last step
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))

from config import TASK, set_random_seed                       # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
import rerun_common as rc                                      # noqa: E402

SEEDS = (7, 13, 21)
BASES = ('q1_R', 'q1')
DATASETS = ('ETTh1', 'ETTh2')
SUITE = 'etthard-2y-20261023'
CELLS = {   # cell -> (training suite, stdout label, calibration suite)
    'mem-336-all': ('etthard-2w-20261022', '{base}-direct-0.0005-all', 'rerun2u-cal-20261021'),
    'mem-96-all': (SUITE, '{base}-mem-96-all', 'rerun2y-cal-L96-20261023'),
    'mem-336-last1': (SUITE, '{base}-mem-336-last1', 'rerun2u-cal-20261021'),
    'mem-96-last1': (SUITE, '{base}-mem-96-last1', 'rerun2y-cal-L96-20261023'),
    'nomem-336-all': (SUITE, '{base}-nomem-336-all', 'rerun2y-cal-nomem-20261023'),
    'nomem-336-last1': (SUITE, '{base}-nomem-336-last1', 'rerun2y-cal-nomem-20261023'),
}
CONTRASTS = {'C1': ('nomem-336-all', 'mem-336-all'), 'C2': ('nomem-336-last1', 'mem-336-last1'),
             'C3': ('mem-96-last1', 'mem-336-last1'), 'C4': ('mem-96-all', 'mem-336-all'),
             'C5': ('mem-336-last1', 'mem-336-all')}


def measure(base, data, cell, device, manifests):
    suite, label, cal_suite = CELLS[cell]
    spec = rc.cell_2y(f'{base}-{cell}')
    if cal_suite not in manifests:
        manifests[cal_suite] = rc.calibration_manifest(cal_suite)
    cal = manifests[cal_suite][rc.calibration_label('ett', base, data)]
    rows, errors = [], []
    for seed in SEEDS:
        if cal['refused']:
            bad, _ = rc.refused_cell(suite, 'ett', data, spec, seed, cal)
            errors += bad
            rows.append({'seed': seed, 'status': 'calibration_refused'})
            continue
        status, run, result = rc.find_cell(suite, 'ett', data, spec, seed)
        if status == 'g11':
            rows.append({'seed': seed, 'status': 'g11_failure_in_training'})
            continue
        model, args = rc.load_model(run, device)
        errors += [f'{base}-{cell} {data} seed {seed}: {e}' for e in
                   rc.check_manifest(args, result, 'ett', spec, seed, cal, data=data, cond=label.format(base=base))]
        set_random_seed(args.seed)
        _, loader = data_provider(args, 'val')
        r = rc.one_pass(model, loader, args, ties=False)
        train = json.loads(Path(result).read_text())['train']
        ref = rc.find_cell('etthard-spikede-20261021', 'ett', data, rc.ETT_CONDITIONS[base], seed)[2]
        r.update(seed=seed, status='evaluated', best_val_loss=train['best_val_loss'], epochs_run=train['epochs_run'],
                 train_seconds=train.get('seconds'), num_patches=args.num_patches, seq_len=args.seq_len,
                 head_in=model.head.in_features, repro_ok=abs(r['mse'] - train['best_val_loss']) <= rc.REPRO_TOL,
                 ref_2T=json.loads(Path(ref).read_text())['train']['best_val_loss'])
        rows.append(r)
    return rows, errors


def blocked(rows):
    return [r['seed'] for r in rows if r['status'] != 'evaluated' or not r['safe'] or not r['repro_ok']]


def main():
    ap = argparse.ArgumentParser(description='prereg 2Y: readout x window x state on validation')
    ap.add_argument('-nd', '--num_device', type=int, default=0)
    ap.add_argument('--out', default='etthard-2y-20261023-val')
    cli = ap.parse_args()
    if torch.__version__ != rc.TORCH:
        raise SystemExit(f'torch {torch.__version__} != {rc.TORCH}')
    device = torch.device(f'cuda:{cli.num_device}')
    out_dir = TASK / 'results' / cli.out
    out_dir.mkdir(parents=True, exist_ok=False)
    manifests, record, errors = {}, {'prereg': '2Y', 'split': 'val', 'exploratory': True, 'seeds': list(SEEDS),
                                     'per': {}}, []
    for base in BASES:
        for data in DATASETS:
            cells = {}
            for cell in CELLS:
                print(f"{f' {base} {data} {cell} ':=^80s}")
                cells[cell], bad = measure(base, data, cell, device, manifests)
                errors += bad
            block = {}
            for cell, rows in cells.items():
                ev = [r for r in rows if r['status'] == 'evaluated']
                block[cell] = {'rows': rows, 'blocked_seeds': blocked(rows),
                               'mean_mse': float(np.mean([r['mse'] for r in ev])) if ev else None,
                               'mean_ratio_2T': float(np.mean([r['mse'] / r['ref_2T'] for r in ev])) if ev else None,
                               'firing_rate': float(np.mean([r['firing_rate'] for r in ev])) if ev else None,
                               'epochs': float(np.mean([r['epochs_run'] for r in ev])) if ev else None,
                               'train_seconds': float(np.mean([r['train_seconds'] or 0 for r in ev])) if ev else None}
            contrasts = {}
            for name, (a_cell, b_cell) in CONTRASTS.items():
                if blocked(cells[a_cell]) or blocked(cells[b_cell]):
                    contrasts[name] = {'a': a_cell, 'b': b_cell, 'status': 'withheld'}
                    continue
                a = [next(r['mse'] for r in cells[a_cell] if r['seed'] == s) for s in SEEDS]
                b = [next(r['mse'] for r in cells[b_cell] if r['seed'] == s) for s in SEEDS]
                t = rc.paired(a, b)
                t.update(a=a_cell, b=b_cell, status='reported (exploratory, validation, 3 seeds)', category=rc.category('dir95', t))
                contrasts[name] = t
            block['contrasts'] = contrasts
            record['per'][f'{base}/{data}'] = block
    if errors:
        for e in errors:
            print(f'[2y] manifest {e}')
        json.dump({**record, 'status': 'manifest errors', 'errors': errors}, open(out_dir / 'context_2y_record.json', 'x'),
                  indent=2, allow_nan=False)
        raise SystemExit(f'[2y] {len(errors)} manifest error(s); the comparisons are NOT to be used')
    print(f"{' RESULT (validation, exploratory, 3 seeds) ':=^80s}")
    for key, block in record['per'].items():
        for cell in CELLS:
            b = block[cell]
            print(f"[2y] {key:<11} {cell:<16} val {b['mean_mse']:.4f}  vs 2T x{b['mean_ratio_2T']:.4f}  rate "
                  f"{b['firing_rate']:.3f}  epochs {b['epochs']:.1f}  train {b['train_seconds']:.0f} s")
        for name, t in block['contrasts'].items():
            if t.get('status') == 'withheld':
                print(f"[2y] {key:<11} {name} {t['a']} - {t['b']}: withheld")
                continue
            print(f"[2y] {key:<11} {name} {t['a']} - {t['b']}: {t['mean']:+.5f} [{t['ci'][0]:+.5f}, {t['ci'][1]:+.5f}] "
                  f"rel {100 * t['relative']:+.2f}% a<b {t['a_lower']}/3 -> {t['category']}")
    record['source_sha256'] = {f: rc.sha256(TASK / f) for f in rc.SOURCES}
    record['analysis_sha256'] = rc.sha256(Path(__file__))
    json.dump({**record, 'status': 'done'}, open(out_dir / 'context_2y_record.json', 'x'), indent=2, allow_nan=False)
    print(f"Record saved to `{out_dir / 'context_2y_record.json'}`")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
