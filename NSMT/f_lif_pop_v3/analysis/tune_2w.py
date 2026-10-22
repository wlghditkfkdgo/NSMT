"""Prereg 2W: tune the 2U base on VALIDATION, then queue the selected setting for the held-out looks.

    python tune_2w.py --stage select [-nd <gpu>]       pick one setting from the grid (validation, seeds 7/13/21)
    python tune_2w.py --stage queue                    write the final queue for the picked setting

Grid (rerun_common.GRID_2W, 12 settings) on the 2U base (population axis kept, per-step BN with gamma from D11):
arch direct | hmem x lr 0.0005 | 0.001 | 0.002 x wd all | xnb, each trained for q1 and q1_R on ETTh1 and ETTh2
with seeds 7, 13, 21 (144 runs, suite etthard-2w-20261022).

Selection rule (fixed before any 2W number was seen):
    score   mean over (base, dataset, seed) of  val MSE / the 2T run's best val MSE at the same (base, dataset, seed)
    eligible every cell finished (no G11 stop), passed its manifest, safe, and reproduced its best val MSE (1e-6)
    pick    the lowest score; within 0.005 of it, the setting with the fewest changes from (direct, 0.001, all),
            then the lower score
One setting is picked for both R on and R off, so 2O's R-on / R-off contrasts compare the same model.
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

SUITE = 'etthard-2w-20261022'
CAL = 'rerun2u-cal-20261021'
SELECT_SEEDS = (7, 13, 21)
BASES = ('q1_R', 'q1')
DATASETS = ('ETTh1', 'ETTh2')
TIE = .005
DEFAULT = ('direct', '0.001', 'all')


def stored(suite, base, data, seed, spec):
    _, _, result = rc.find_cell(suite, 'ett', data, spec, seed)
    return json.loads(Path(result).read_text())['train']['best_val_loss']


def select(device):
    out = TASK / 'results' / SUITE / 'tune_2w_select.json'
    if out.exists():
        raise SystemExit(f'[2w] {out} exists; the selection is recorded')
    cal = rc.calibration_manifest(CAL)
    rows, errors = [], []
    for cfg in rc.GRID_2W:
        for base in BASES:
            for data in DATASETS:
                c = cal[rc.calibration_label('ett', base, data)]
                for seed in SELECT_SEEDS:
                    name = f'{base}-{cfg}'
                    spec = rc.cell_2w(name)
                    status, run, result = rc.find_cell(SUITE, 'ett', data, spec, seed)
                    row = {'cfg': cfg, 'base': base, 'data': data, 'seed': seed, 'status': status,
                           'ref_2T': stored('etthard-spikede-20261021', base, data, seed, rc.ETT_CONDITIONS[base]),
                           'ref_2U': stored('etthard-2u-20261021', base, data, seed, rc.ETT_CONDITIONS_2U[base])}
                    if status == 'g11':
                        rows.append(row)
                        continue
                    model, args = rc.load_model(run, device)
                    errors += [f'{name} {data} {seed}: {e}' for e in
                               rc.check_manifest(args, result, 'ett', spec, seed, c, data=data, cond=name)]
                    set_random_seed(args.seed)
                    _, loader = data_provider(args, 'val')
                    r = rc.one_pass(model, loader, args, ties=False)
                    best = json.loads(Path(result).read_text())['train']['best_val_loss']
                    row.update(mse=r['mse'], safe=r['safe'], repro_ok=abs(r['mse'] - best) <= rc.REPRO_TOL,
                               firing_rate=r['firing_rate'], hidden_firing_rate=r['hidden_firing_rate'],
                               epochs_run=json.loads(Path(result).read_text())['train']['epochs_run'])
                    rows.append(row)
                    print(f"[2w] {cfg:<18} {base:<5} {data} seed {seed:>3}: val {r['mse']:.4f} (2T {row['ref_2T']:.4f}, "
                          f"2U {row['ref_2U']:.4f})")
    if errors:
        for e in errors:
            print(f'[2w] manifest {e}')
        raise SystemExit(f'[2w] {len(errors)} manifest error(s); nothing selected')
    table = {}
    for cfg in rc.GRID_2W:
        rs = [r for r in rows if r['cfg'] == cfg]
        ok = all(r['status'] == 'done' and r['safe'] and r['repro_ok'] for r in rs)
        table[cfg] = {'eligible': ok,
                      'score_vs_2T': float(np.mean([r['mse'] / r['ref_2T'] for r in rs])) if ok else None,
                      'score_vs_2U': float(np.mean([r['mse'] / r['ref_2U'] for r in rs])) if ok else None,
                      'by': {f'{b}/{d}': float(np.mean([r['mse'] for r in rs if r['base'] == b and r['data'] == d]))
                             for b in BASES for d in DATASETS} if ok else None,
                      'changes': sum(x != y for x, y in zip(cfg.split('-'), DEFAULT))}
    eligible = {k: v for k, v in table.items() if v['eligible']}
    if not eligible:
        raise SystemExit('[2w] no eligible setting')
    best = min(v['score_vs_2T'] for v in eligible.values())
    tied = [k for k, v in eligible.items() if v['score_vs_2T'] - best <= TIE]
    pick = min(tied, key=lambda k: (eligible[k]['changes'], eligible[k]['score_vs_2T']))
    print(f"{' 2W selection (validation, seeds 7/13/21) ':=^90s}")
    for k, v in sorted(table.items(), key=lambda kv: (kv[1]['score_vs_2T'] is None, kv[1]['score_vs_2T'] or 0)):
        if v['eligible']:
            print(f"[2w] {k:<18} val/2T {v['score_vs_2T']:.4f}  val/2U {v['score_vs_2U']:.4f}  "
                  + '  '.join(f"{b} {m:.4f}" for b, m in v['by'].items()) + ('  <- picked' if k == pick else ''))
        else:
            print(f'[2w] {k:<18} NOT eligible')
    print(f'[2w] best {best:.4f}; within {TIE}: {tied} -> picked {pick}')
    with open(out, 'x') as handle:
        json.dump({'prereg': '2W', 'split': 'val', 'seeds': list(SELECT_SEEDS), 'rule': __doc__.split('Selection rule')[1],
                   'table': table, 'tied': tied, 'picked': pick, 'rows': rows,
                   'source_sha256': {f: rc.sha256(TASK / f) for f in rc.SOURCES}}, handle, indent=2, allow_nan=False)
    print(f'[2w] recorded to {out}')


def queue():
    sel = json.loads((TASK / 'results' / SUITE / 'tune_2w_select.json').read_text())
    cfg = sel['picked']
    q = TASK / 'scripts' / 'queues' / f'rerun-ett-2w-final-{cfg}.txt'
    if q.exists():
        raise SystemExit(f'{q} exists')
    lines = []
    for base in ('q1_R', 'q1', 'pearson_R', 'pearson'):
        for seed in rc.SEEDS:
            if base in BASES and seed in SELECT_SEEDS:                 # already trained for the selection
                continue
            for data in DATASETS:
                lines.append(f'SUITE={SUITE} bash f_lif_pop_v3/forecasting/scripts/rerun2w_ett.sh {data} 96 {base}-{cfg} {seed}')
    q.write_text('\n'.join(lines) + '\n')
    with open(TASK / 'results' / SUITE / 'run_commands_final.txt', 'x') as handle:
        handle.write(f'# 2W final runs for the picked setting {cfg}\n' + '\n'.join(lines) + '\n')
    print(f'[2w] {len(lines)} runs in {q}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['select', 'queue'])
    ap.add_argument('-nd', '--num_device', type=int, default=0)
    a = ap.parse_args()
    if a.stage == 'select':
        if torch.__version__ != rc.TORCH:
            raise SystemExit(f'torch {torch.__version__} != {rc.TORCH}')
        select(torch.device(f'cuda:{a.num_device}'))
    else:
        queue()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
