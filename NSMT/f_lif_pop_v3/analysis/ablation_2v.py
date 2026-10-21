"""Prereg 2V: where does 2U lose against 2T? The three changes taken apart on VALIDATION (exploratory).

    python ablation_2v.py [-nd <gpu>] [--out <results dir name>]

2 x 2 x 2 cells per base (q1_R: R on; q1: R off), dataset and seed, all without selection (pop_flif):
    soma   lif (silent members read by a soma, 2T)        | none (the members fire, 2U)
    norm   frozen (train-split standardisation x scale)   | bn (per-step BN, gamma from D11, 2U)
    head   flatten (Linear(D*,32) -> Linear(1344,96))     | hidden (Linear + BN + SpikingJelly LIF -> Linear, 2U)
lif-frozen-flatten is the 2T run and none-bn-hidden the 2U run (not retrained); the other six are trained for 2V.

Every cell passes its manifest (its own calibration manifest) and re-measures its stored best validation MSE
(REPRO_TOL) before it is used; a G11 stop, a refused calibration, an unsafe model or a failed reproduction
blocks every effect it enters (no seed is dropped). Reported, all as paired 95% t-intervals over the 8 seeds:
    single changes from 2T     2T -> soma none | -> bn | -> hidden, each alone
    main effects               level B - level A of one factor, averaged over the 4 cells of the other two
    two-way interactions       (difference of differences), averaged over the third factor
    2U - 2T                    all three together
plus per cell: mean validation MSE, train loss at the best epoch, best epoch, member / soma firing rate, hidden
firing rate, max|u|. No held-out period is read; nothing is registered.
"""
import sys
import csv
import glob
import json
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))

from config import TASK, set_random_seed                       # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
import rerun_common as rc                                      # noqa: E402

DATASETS = ('ETTh1', 'ETTh2')
BASES = ('q1_R', 'q1')
LEVELS = {'soma': ('lif', 'none'), 'norm': ('frozen', 'bn'), 'head': ('flatten', 'hidden')}
FACTORS = tuple(LEVELS)
T2 = ('lif', 'frozen', 'flatten')
U2 = ('none', 'bn', 'hidden')


def cell_name(base, combo):
    return f"{base}-{'-'.join(combo)}"


def curve(run):
    rows = list(csv.DictReader(open(Path(run) / 'log' / 'best_log_0.csv')))
    val = [float(r['val_loss']) for r in rows]
    b = int(np.argmin(val))
    return {'train_at_best': float(rows[b]['train_loss']), 'best_epoch': b,
            'train_min': min(float(r['train_loss']) for r in rows)}


def measure(base, data, combo, device, manifests):
    suite, label, cal_suite = rc.SOURCE_2V[combo]
    name = cell_name(base, combo)
    spec = rc.cell_2v(name)
    cond = label.format(base=base, cell=name)
    if cal_suite not in manifests:
        manifests[cal_suite] = rc.calibration_manifest(cal_suite)
    cal = manifests[cal_suite][rc.calibration_label('ett', base, data)]
    rows, errors = [], []
    for seed in rc.SEEDS:
        if cal['refused']:
            bad, info = rc.refused_cell(suite, 'ett', data, spec, seed, cal)
            errors += bad
            rows.append({'seed': seed, 'status': 'calibration_refused'})
            continue
        status, run, result = rc.find_cell(suite, 'ett', data, spec, seed)
        if status == 'g11':
            bad, record = rc.check_failure(run, 'ett', spec, seed, cal)
            errors += bad
            rows.append({'seed': seed, 'status': 'g11_failure_in_training', 'g11': record})
            continue
        model, args = rc.load_model(run, device)
        errors += [f'{name} {data} seed {seed}: {e}' for e in
                   rc.check_manifest(args, result, 'ett', spec, seed, cal, data=data, cond=cond)]
        set_random_seed(args.seed)
        _, loader = data_provider(args, 'val')
        r = rc.one_pass(model, loader, args, ties=False)
        best = json.loads(Path(result).read_text())['train']['best_val_loss']
        r.update(seed=seed, status='evaluated', suite=suite, best_val_loss=best,
                 repro_ok=abs(r['mse'] - best) <= rc.REPRO_TOL, **curve(run))
        rows.append(r)
    return rows, errors


def blocked_seeds(rows):
    return [r['seed'] for r in rows if r['status'] != 'evaluated' or not r['safe'] or not r['repro_ok']]


def effect(cells, a_combos, b_combos):
    """Paired over seeds: mean over b_combos minus mean over a_combos; withheld when any cell is blocked."""
    used = a_combos + b_combos
    block = sorted({(c, s) for c in used for s in blocked_seeds(cells[c])})
    if block:
        return {'status': 'withheld', 'blocked': [list(map(str, b)) for b in block]}
    val = {c: {r['seed']: r['mse'] for r in cells[c]} for c in used}
    a = [np.mean([val[c][s] for c in a_combos]) for s in rc.SEEDS]
    b = [np.mean([val[c][s] for c in b_combos]) for s in rc.SEEDS]
    t = rc.paired(b, a)
    t['status'] = 'reported (exploratory, validation)'
    t['category'] = rc.category('dir95', t)
    return t


def analyse(cells):
    combos = list(itertools.product(*LEVELS.values()))
    out = {'single_from_2T': {}, 'main': {}, 'interaction': {}}
    for i, f in enumerate(FACTORS):
        one = list(T2)
        one[i] = LEVELS[f][1]
        out['single_from_2T'][f] = effect(cells, [T2], [tuple(one)])
        lo = [c for c in combos if c[i] == LEVELS[f][0]]
        hi = [c for c in combos if c[i] == LEVELS[f][1]]
        out['main'][f] = effect(cells, lo, hi)
    for (i, f), (j, g) in itertools.combinations(enumerate(FACTORS), 2):
        # (f hi, g hi) - (f hi, g lo) - (f lo, g hi) + (f lo, g lo), each averaged over the third factor
        pick = lambda x, y: [c for c in combos if c[i] == LEVELS[f][x] and c[j] == LEVELS[g][y]]
        block = sorted({(c, s) for c in combos for s in blocked_seeds(cells[c])})
        if block:
            out['interaction'][f'{f} x {g}'] = {'status': 'withheld'}
            continue
        val = {c: {r['seed']: r['mse'] for r in cells[c]} for c in combos}
        m = lambda cs, s: np.mean([val[c][s] for c in cs])
        diff = [m(pick(1, 1), s) - m(pick(1, 0), s) - m(pick(0, 1), s) + m(pick(0, 0), s) for s in rc.SEEDS]
        t = rc.paired(diff, [0.] * len(diff), relative=False)
        t.update(status='reported (exploratory, validation)', category=rc.category('dir95', t))
        out['interaction'][f'{f} x {g}'] = t
    out['2U - 2T'] = effect(cells, [T2], [U2])
    return out


def main():
    ap = argparse.ArgumentParser(description='prereg 2V: the 2T -> 2U changes taken apart on validation')
    ap.add_argument('-nd', '--num_device', type=int, default=0)
    ap.add_argument('--out', default='etthard-2v-20261022-val')
    cli = ap.parse_args()
    if torch.__version__ != rc.TORCH:
        raise SystemExit(f'torch {torch.__version__} != {rc.TORCH}')
    device = torch.device(f'cuda:{cli.num_device}')
    out_dir = TASK / 'results' / cli.out
    out_dir.mkdir(parents=True, exist_ok=False)
    manifests, record, errors = {}, {'prereg': '2V', 'split': 'val', 'exploratory': True, 'per': {}}, []
    for base in BASES:
        for data in DATASETS:
            cells = {}
            for combo in itertools.product(*LEVELS.values()):
                print(f"{f' {base} {data} {cell_name(base, combo)} ':=^90s}")
                cells[combo], bad = measure(base, data, combo, device, manifests)
                errors += bad
            record['per'][f'{base}/{data}'] = {
                'cells': {'-'.join(c): rows for c, rows in cells.items()},
                'means': {'-'.join(c): {k: (float(np.mean([r[k] for r in rows if r['status'] == 'evaluated']))
                                            if any(r['status'] == 'evaluated' for r in rows) else None)
                                        for k in ('mse', 'train_at_best', 'best_epoch', 'train_min', 'firing_rate',
                                                  'max_abs_state')}
                          | {'hidden_firing_rate': (float(np.mean([r['hidden_firing_rate'] for r in rows
                                                                   if r['status'] == 'evaluated']))
                                                    if c[2] == 'hidden' else None),
                             'blocked_seeds': blocked_seeds(rows)}
                          for c, rows in cells.items()},
                'effects': analyse(cells)}
    if errors:
        for e in errors:
            print(f'[2v] manifest {e}')
        json.dump({**record, 'status': 'manifest errors', 'errors': errors}, open(out_dir / 'ablation_2v_record.json', 'x'),
                  indent=2, allow_nan=False)
        raise SystemExit(f'[2v] {len(errors)} manifest error(s); the effects above are NOT to be used')
    print(f"{' RESULT (validation, exploratory) ':=^90s}")
    for key, block in record['per'].items():
        print(f'[2v] {key} mean val MSE: ' + ', '.join(f"{c} {m['mse']:.4f}" for c, m in block['means'].items()
                                                       if m['mse'] is not None))
        for group in ('single_from_2T', 'main', 'interaction'):
            for name, t in block['effects'][group].items():
                if t.get('status') == 'withheld':
                    print(f'[2v] {key} {group:<15} {name:<14} WITHHELD')
                    continue
                rel = '' if t.get('relative') is None else f" rel {100 * t['relative']:+.2f}%"
                print(f"[2v] {key} {group:<15} {name:<14} {t['mean']:+.5f} [{t['ci'][0]:+.5f}, {t['ci'][1]:+.5f}]{rel} "
                      f"a<b {t['a_lower']}/8 -> {t['category']}")
        t = block['effects']['2U - 2T']
        print(f"[2v] {key} 2U - 2T          {t['mean']:+.5f} [{t['ci'][0]:+.5f}, {t['ci'][1]:+.5f}] rel {100 * t['relative']:+.2f}%")
    record['source_sha256'] = {f: rc.sha256(TASK / f) for f in rc.SOURCES}
    record['analysis_sha256'] = rc.sha256(Path(__file__))
    json.dump({**record, 'status': 'done'}, open(out_dir / 'ablation_2v_record.json', 'x'), indent=2, allow_nan=False)
    print(f"Record saved to `{out_dir / 'ablation_2v_record.json'}`")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
