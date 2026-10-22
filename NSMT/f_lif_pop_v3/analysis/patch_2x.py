"""Prereg 2X: overlapping patches on the 2W pick, VALIDATION only (exploratory).

    python patch_2x.py [-nd <gpu>]

Cells per base (q1_R: R on; q1: R off), dataset and seed {7, 13, 21}, all the 2W pick (population axis kept,
per-step BN with gamma from D11, no hidden block, one output Linear, lr 0.0005, weight decay on all):
    p8s8    length 8, stride 8 (T = 42, no overlap)  the 2W selection runs, not retrained
    p8s4    length 8, stride 4 (T = 83)              trained for 2X
    p16s8   length 16, stride 8 (T = 41)             trained for 2X
Every cell passes its manifest (its own calibration manifest) and re-measures its stored best validation MSE
(REPRO_TOL); a blocked cell withholds every comparison it enters. Reported per base and dataset: mean val MSE,
the paired difference to p8s8 over the 3 seeds (95% t-interval with 2 degrees of freedom: wide, descriptive),
the ratio to the 2T run of the same seed, firing rate, epochs and training time.
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
CELLS = {   # patching -> (suite, stdout label, calibration suite)
    'p8s8': ('etthard-2w-20261022', '{base}-direct-0.0005-all', 'rerun2u-cal-20261021'),
    'p8s4': ('etthard-2x-20261022', '{base}-p8s4', 'rerun2x-cal-p8s4-20261022'),
    'p16s8': ('etthard-2x-20261022', '{base}-p16s8', 'rerun2x-cal-p16s8-20261022'),
}


def measure(base, data, patch, device, manifests):
    suite, label, cal_suite = CELLS[patch]
    spec = rc.cell_2x(f'{base}-{patch}')
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
        errors += [f'{base}-{patch} {data} seed {seed}: {e}' for e in
                   rc.check_manifest(args, result, 'ett', spec, seed, cal, data=data, cond=label.format(base=base))]
        set_random_seed(args.seed)
        _, loader = data_provider(args, 'val')
        r = rc.one_pass(model, loader, args, ties=False)
        train = json.loads(Path(result).read_text())['train']
        ref = rc.find_cell('etthard-spikede-20261021', 'ett', data, rc.ETT_CONDITIONS[base], seed)[2]
        r.update(seed=seed, status='evaluated', best_val_loss=train['best_val_loss'], epochs_run=train['epochs_run'],
                 train_seconds=train.get('seconds'), num_patches=args.num_patches,
                 repro_ok=abs(r['mse'] - train['best_val_loss']) <= rc.REPRO_TOL,
                 ref_2T=json.loads(Path(ref).read_text())['train']['best_val_loss'])
        rows.append(r)
    return rows, errors


def blocked(rows):
    return [r['seed'] for r in rows if r['status'] != 'evaluated' or not r['safe'] or not r['repro_ok']]


def main():
    ap = argparse.ArgumentParser(description='prereg 2X: overlapping patches on validation')
    ap.add_argument('-nd', '--num_device', type=int, default=0)
    ap.add_argument('--out', default='etthard-2x-20261022-val')
    cli = ap.parse_args()
    if torch.__version__ != rc.TORCH:
        raise SystemExit(f'torch {torch.__version__} != {rc.TORCH}')
    device = torch.device(f'cuda:{cli.num_device}')
    out_dir = TASK / 'results' / cli.out
    out_dir.mkdir(parents=True, exist_ok=False)
    manifests, record, errors = {}, {'prereg': '2X', 'split': 'val', 'exploratory': True, 'seeds': list(SEEDS),
                                     'per': {}}, []
    for base in BASES:
        for data in DATASETS:
            cells = {}
            for patch in CELLS:
                print(f"{f' {base} {data} {patch} ':=^80s}")
                cells[patch], bad = measure(base, data, patch, device, manifests)
                errors += bad
            block = {}
            for patch, rows in cells.items():
                ev = [r for r in rows if r['status'] == 'evaluated']
                block[patch] = {'rows': rows, 'blocked_seeds': blocked(rows),
                                'mean_mse': float(np.mean([r['mse'] for r in ev])) if ev else None,
                                'mean_ratio_2T': float(np.mean([r['mse'] / r['ref_2T'] for r in ev])) if ev else None,
                                'firing_rate': float(np.mean([r['firing_rate'] for r in ev])) if ev else None,
                                'epochs': float(np.mean([r['epochs_run'] for r in ev])) if ev else None,
                                'train_seconds': float(np.mean([r['train_seconds'] or 0 for r in ev])) if ev else None}
            for patch in ('p8s4', 'p16s8'):
                if blocked(cells[patch]) or blocked(cells['p8s8']):
                    block[patch]['vs_p8s8'] = {'status': 'withheld'}
                    continue
                a = [next(r['mse'] for r in cells[patch] if r['seed'] == s) for s in SEEDS]
                b = [next(r['mse'] for r in cells['p8s8'] if r['seed'] == s) for s in SEEDS]
                t = rc.paired(a, b)
                t.update(status='reported (exploratory, validation, 3 seeds)', category=rc.category('dir95', t))
                block[patch]['vs_p8s8'] = t
            record['per'][f'{base}/{data}'] = block
    if errors:
        for e in errors:
            print(f'[2x] manifest {e}')
        json.dump({**record, 'status': 'manifest errors', 'errors': errors}, open(out_dir / 'patch_2x_record.json', 'x'),
                  indent=2, allow_nan=False)
        raise SystemExit(f'[2x] {len(errors)} manifest error(s); the comparisons are NOT to be used')
    print(f"{' RESULT (validation, exploratory, 3 seeds) ':=^80s}")
    for key, block in record['per'].items():
        for patch in CELLS:
            b = block[patch]
            line = (f"[2x] {key:<11} {patch:<6} val {b['mean_mse']:.4f}  vs 2T x{b['mean_ratio_2T']:.4f}  rate "
                    f"{b['firing_rate']:.3f}  epochs {b['epochs']:.1f}  train {b['train_seconds']:.0f} s")
            t = b.get('vs_p8s8')
            if t and t.get('status') != 'withheld':
                line += (f"  - p8s8 {t['mean']:+.5f} [{t['ci'][0]:+.5f}, {t['ci'][1]:+.5f}] rel {100 * t['relative']:+.2f}% "
                         f"a<b {t['a_lower']}/3 -> {t['category']}")
            print(line)
    record['source_sha256'] = {f: rc.sha256(TASK / f) for f in rc.SOURCES}
    record['analysis_sha256'] = rc.sha256(Path(__file__))
    json.dump({**record, 'status': 'done'}, open(out_dir / 'patch_2x_record.json', 'x'), indent=2, allow_nan=False)
    print(f"Record saved to `{out_dir / 'patch_2x_record.json'}`")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
