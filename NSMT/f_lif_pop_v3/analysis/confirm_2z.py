"""Prereg 2Z: confirm 2Y C1 (the neuron state reset at every patch beat the state kept) with a learning rate chosen
for each family, 8 seeds and the test split (exploratory: test was opened before; recorded as a reopening under 2Z).

    python confirm_2z.py --stage select [-nd <gpu>]    pick one lr per family on validation (seeds 7/13/21)
    python confirm_2z.py --stage queue                 write the queue of the 5 further seeds at the picked lrs
    python confirm_2z.py --stage test [-nd <gpu>]      reproduce validation for every run, then reopen test and compare

Families on the 2W structure (direct, weight decay on all, 336 steps, the output Linear reads every step):
    mem    the state crosses patches   (lr 0.0005 = the 2W pick; 0.001 / 0.002 = the 2W grid cells, not retrained)
    nomem  v = 0 at every patch        (lr 0.0005 = the 2Y cell; 0.001 / 0.002 trained for 2Z)
Selection rule (fixed before any 2Z number was seen; the 2W rule restricted to these cells):
    score   mean over (base q1_R / q1, dataset ETTh1 / ETTh2, seed 7 / 13 / 21) of val MSE / the 2T run's best val MSE
    eligible every cell finished (no G11 stop), passed its manifest, safe, reproduced its best val MSE (1e-6)
    pick    the lowest score; within 0.005 of it the O8 lr 0.001, then the lower score. One lr per family.
Comparisons (per base x dataset, 8 seeds {7, 13, 21, 42, 123, 256, 512, 1024}, paired 95% t; the 2N verdict: lowers /
raises when the interval excludes 0, otherwise no difference shown):
    Z1  nomem - mem on test (primary)      Z1v the same on validation
    Z2  nomem - 2T on test                 Z3  mem - 2T on test         (2T: etthard-spikede-20261021, the soma structure)
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))
sys.path.insert(0, str(HERE))

from config import TASK, set_random_seed                       # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
import rerun_common as rc                                      # noqa: E402

SUITE = 'etthard-2z-20261023'
SELECT_SEEDS = (7, 13, 21)
FINAL_SEEDS = tuple(rc.SEEDS)
BASES = ('q1_R', 'q1')
DATASETS = ('ETTh1', 'ETTh2')
TIE = .005
DEFAULT_LR = '0.001'
CAL = {'mem': 'rerun2u-cal-20261021', 'nomem': 'rerun2y-cal-nomem-20261023'}
OUT_SELECT = TASK / 'results' / SUITE / 'confirm_2z_select.json'


def where(family, lr, seed):
    """(training suite, stdout label) of a cell."""
    if seed in SELECT_SEEDS:
        if family == 'mem':
            return 'etthard-2w-20261022', '{base}-direct-' + lr + '-all'
        if lr == '0.0005':
            return 'etthard-2y-20261023', '{base}-nomem-336-all'
    return SUITE, '{base}-' + family + '-' + lr


def evaluate(family, lr, base, data, seed, device, manifests, split='val'):
    suite, label = where(family, lr, seed)
    spec = rc.cell_2z(f'{base}-{family}-{lr}')
    if CAL[family] not in manifests:
        manifests[CAL[family]] = rc.calibration_manifest(CAL[family])
    cal = manifests[CAL[family]][rc.calibration_label('ett', base, data)]
    status, run, result = rc.find_cell(suite, 'ett', data, spec, seed)
    row = {'family': family, 'lr': lr, 'base': base, 'data': data, 'seed': seed, 'suite': suite, 'status': status}
    if status != 'done':
        return row, []
    model, args = rc.load_model(run, device)
    errors = [f'{family}-{lr} {base} {data} {seed}: {e}' for e in
              rc.check_manifest(args, result, 'ett', spec, seed, cal, data=data, cond=label.format(base=base))]
    set_random_seed(args.seed)
    _, loader = data_provider(args, split)
    r = rc.one_pass(model, loader, args, ties=False)
    best = json.loads(Path(result).read_text())['train']['best_val_loss']
    row.update(mse=r['mse'], safe=r['safe'], firing_rate=r['firing_rate'], best_val_loss=best,
               repro_ok=(abs(r['mse'] - best) <= rc.REPRO_TOL) if split == 'val' else None,
               epochs_run=json.loads(Path(result).read_text())['train']['epochs_run'])
    return row, errors


def ref_2t(base, data, seed):
    _, _, result = rc.find_cell('etthard-spikede-20261021', 'ett', data, rc.ETT_CONDITIONS[base], seed)
    return json.loads(Path(result).read_text())['train']['best_val_loss']


def select(device):
    if OUT_SELECT.exists():
        raise SystemExit(f'[2z] {OUT_SELECT} exists; the selection is recorded')
    manifests, rows, errors = {}, [], []
    for family in ('mem', 'nomem'):
        for lr in rc.LRS_2Z:
            for base in BASES:
                for data in DATASETS:
                    for seed in SELECT_SEEDS:
                        row, bad = evaluate(family, lr, base, data, seed, device, manifests)
                        row['ref_2T'] = ref_2t(base, data, seed)
                        rows.append(row)
                        errors += bad
                        print(f"[2z] {family:<5} lr {lr:<6} {base:<4} {data} seed {seed:>2}: "
                              + (f"val {row['mse']:.4f} (2T {row['ref_2T']:.4f})" if row['status'] == 'done' else row['status']))
    if errors:
        for e in errors:
            print(f'[2z] manifest {e}')
        raise SystemExit(f'[2z] {len(errors)} manifest error(s); nothing selected')
    table, picks = {}, {}
    for family in ('mem', 'nomem'):
        for lr in rc.LRS_2Z:
            rs = [r for r in rows if r['family'] == family and r['lr'] == lr]
            ok = all(r['status'] == 'done' and r['safe'] and r['repro_ok'] for r in rs)
            table[f'{family}-{lr}'] = {'eligible': ok,
                                       'score_vs_2T': float(np.mean([r['mse'] / r['ref_2T'] for r in rs])) if ok else None,
                                       'by': {f'{b}/{d}': float(np.mean([r['mse'] for r in rs if r['base'] == b and r['data'] == d]))
                                              for b in BASES for d in DATASETS} if ok else None}
        cand = {lr: table[f'{family}-{lr}']['score_vs_2T'] for lr in rc.LRS_2Z if table[f'{family}-{lr}']['eligible']}
        if not cand:
            raise SystemExit(f'[2z] no eligible lr for {family}')
        best = min(cand.values())
        tied = [lr for lr, v in cand.items() if v - best <= TIE]
        picks[family] = DEFAULT_LR if DEFAULT_LR in tied else min(tied, key=lambda lr: cand[lr])
        print(f"[2z] {family}: " + '  '.join(f"lr {lr} {v:.4f}" for lr, v in cand.items())
              + f"  -> within {TIE} of {best:.4f}: {tied}; picked {picks[family]}")
    OUT_SELECT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SELECT, 'x') as handle:
        json.dump({'prereg': '2Z', 'split': 'val', 'seeds': list(SELECT_SEEDS), 'rule': __doc__.split('Selection rule')[1].split('Comparisons')[0],
                   'table': table, 'picked': picks, 'rows': rows,
                   'source_sha256': {f: rc.sha256(TASK / f) for f in rc.SOURCES}}, handle, indent=2, allow_nan=False)
    print(f'[2z] recorded to {OUT_SELECT}')


def queue():
    picks = json.loads(OUT_SELECT.read_text())['picked']
    q = TASK / 'scripts' / 'queues' / f'rerun-ett-2z-final-{picks["mem"]}-{picks["nomem"]}.txt'
    if q.exists():
        raise SystemExit(f'{q} exists')
    lines = [f'SUITE={SUITE} bash f_lif_pop_v3/forecasting/scripts/rerun2z_ett.sh {data} 96 {base}-{family}-{picks[family]} {seed}'
             for family in ('mem', 'nomem') for base in BASES for seed in FINAL_SEEDS if seed not in SELECT_SEEDS
             for data in DATASETS]
    q.write_text('\n'.join(lines) + '\n')
    with open(TASK / 'results' / SUITE / 'run_commands_final.txt', 'x') as handle:
        handle.write(f'# 2Z final runs at the picked lrs mem {picks["mem"]}, nomem {picks["nomem"]}\n' + '\n'.join(lines) + '\n')
    print(f'[2z] {len(lines)} runs in {q}')


def test(device):
    from split_registry import reopen
    picks = json.loads(OUT_SELECT.read_text())['picked']
    out = TASK / 'results' / f'{SUITE}-test'
    out.mkdir(parents=True, exist_ok=False)
    manifests, errors, rows = {}, [], {}
    # 1. every run reproduces its validation MSE before test is read
    for family in ('mem', 'nomem'):
        for base in BASES:
            for data in DATASETS:
                for seed in FINAL_SEEDS:
                    row, bad = evaluate(family, picks[family], base, data, seed, device, manifests)
                    errors += bad
                    if row['status'] != 'done' or not row['safe'] or not row['repro_ok']:
                        errors.append(f"{family} {base} {data} {seed}: {row['status']}, safe {row.get('safe')}, repro {row.get('repro_ok')}")
                    rows[(family, base, data, seed)] = {'val': row}
    cal2t = rc.calibration_manifest('rerun-cal-20261021')
    for base in BASES:
        for data in DATASETS:
            for seed in FINAL_SEEDS:
                status, run, result = rc.find_cell('etthard-spikede-20261021', 'ett', data, rc.ETT_CONDITIONS[base], seed)
                model, args = rc.load_model(run, device)
                errors += [f'2T {base} {data} {seed}: {e}' for e in rc.check_manifest(
                    args, result, 'ett', rc.ETT_CONDITIONS[base], seed, cal2t[rc.calibration_label('ett', base, data)],
                    data=data, cond=base)]
                set_random_seed(args.seed)
                _, loader = data_provider(args, 'val')
                r = rc.one_pass(model, loader, args, ties=False)
                best = json.loads(Path(result).read_text())['train']['best_val_loss']
                if status != 'done' or not r['safe'] or abs(r['mse'] - best) > rc.REPRO_TOL:
                    errors.append(f'2T {base} {data} {seed}: val {r["mse"]} vs {best}')
                rows[('2T', base, data, seed)] = {'val': {'mse': r['mse'], 'best_val_loss': best}}
    if errors:
        for e in errors:
            print(f'[2z] {e}')
        json.dump({'status': 'blocked before test', 'errors': errors}, open(out / 'confirm_2z_record.json', 'x'), indent=2)
        raise SystemExit(f'[2z] {len(errors)} problem(s); test NOT opened')
    print(f'[2z] validation reproduced by all {len(rows)} runs; opening test')
    record_path = out / 'confirm_2z_record.json'
    for data in DATASETS:
        reopen(f'{data}-test-H96', 'prereg 2Z: confirm 2Y C1 (nomem vs mem) with 8 seeds', record_path, appendix='2Z')
    # 2. test
    for (family, base, data, seed), entry in rows.items():
        if family == '2T':
            _, run, _ = rc.find_cell('etthard-spikede-20261021', 'ett', data, rc.ETT_CONDITIONS[base], seed)
        else:
            suite, _ = where(family, picks[family], seed)
            _, run, _ = rc.find_cell(suite, 'ett', data, rc.cell_2z(f'{base}-{family}-{picks[family]}'), seed)
        model, args = rc.load_model(run, device)
        set_random_seed(args.seed)
        _, loader = data_provider(args, 'test')
        r = rc.one_pass(model, loader, args, ties=False)
        entry['test'] = {'mse': r['mse'], 'mae': r['mae'], 'safe': r['safe'], 'windows': r['windows'], 'firing_rate': r['firing_rate']}
        if not r['safe']:
            errors.append(f'{family} {base} {data} {seed}: test pass not safe')
    contrasts = {}
    for base in BASES:
        for data in DATASETS:
            def vec(family, split):
                return [rows[(family, base, data, s)][split]['mse'] for s in FINAL_SEEDS]
            for name, a, b, split in (('Z1', 'nomem', 'mem', 'test'), ('Z1v', 'nomem', 'mem', 'val'),
                                      ('Z2', 'nomem', '2T', 'test'), ('Z3', 'mem', '2T', 'test')):
                t = rc.paired(vec(a, split), vec(b, split))
                t.update(a=a, b=b, split=split, category=rc.category('dir95', t))
                contrasts[f'{base}/{data}/{name}'] = t
                print(f"[2z] {base:<4} {data} {name:<3} {a}-{b} ({split}): {t['mean']:+.5f} [{t['ci'][0]:+.5f}, {t['ci'][1]:+.5f}] "
                      f"rel {100 * t['relative']:+.2f}% a<b {t['a_lower']}/8 -> {t['category']}")
    means = {f'{f}/{b}/{d}/{sp}': float(np.mean([rows[(f, b, d, s)][sp]['mse'] for s in FINAL_SEEDS]))
             for f in ('mem', 'nomem', '2T') for b in BASES for d in DATASETS for sp in ('val', 'test')}
    json.dump({'prereg': '2Z', 'exploratory': True, 'picked': picks, 'seeds': list(FINAL_SEEDS),
               'target_rows': rc.TARGET_ROWS['test'], 'means': means, 'contrasts': contrasts,
               'rows': {'/'.join(map(str, k)): v for k, v in rows.items()}, 'errors_after_test': errors,
               'analysis_sha256': rc.sha256(Path(__file__)),
               'source_sha256': {f: rc.sha256(TASK / f) for f in rc.SOURCES}}, open(record_path, 'x'), indent=2, allow_nan=False)
    print(f'[2z] recorded to {record_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['select', 'queue', 'test'])
    ap.add_argument('-nd', '--num_device', type=int, default=0)
    a = ap.parse_args()
    if a.stage == 'queue':
        return queue()
    if torch.__version__ != rc.TORCH:
        raise SystemExit(f'torch {torch.__version__} != {rc.TORCH}')
    device = torch.device(f'cuda:{a.num_device}')
    select(device) if a.stage == 'select' else test(device)


if __name__ == '__main__':
    main()
