"""Per dataset, which single configuration (pred_len excluded) reproduces ALL four paper cells?

usage: python nsmt_paper_consistent_configs.py <repo-root>/forecasting
Companion of nsmt_paper_match_runs.py (same PAPER table and argument parser).
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsmt_paper_match_runs import PAPER, read_args          # noqa: E402

ROOT = Path(sys.argv[1])
CFG = ['seed', 'epoch', 'batch_size', 'lr', 'weight_decay', 'scheduler', 'patience', 'alpha', 'embed_dim',
       'num_heads', 'num_layers', 'time_num_layers', 'patch_size', 'mlp_ratios', 'max_ratio', 'gating',
       'keep_ratio', 'bias', 'tau', 'seq_len', 'c_in', 'data', 'label_len', 'num_workers', 'warm_up_epoch']

runs = []
for csvp in ROOT.rglob('final+result.csv'):
    run = csvp.parent.parent
    ap = run / 'logargs.txt'
    if not ap.exists():
        continue
    a = read_args(ap)
    if a.get('model') != 'myModel':
        continue
    rows = [r for r in csv.DictReader(csvp.open()) if re.match(r'^[0-9.]+$', r.get('mse', '') or '')]
    if not rows:
        continue
    runs.append((run, a, float(rows[0]['mse']), float(rows[0]['mae'])))

for ds, table in PAPER.items():
    by_cfg = {}
    for run, a, m, e in runs:
        if a.get('dataset') != ds:
            continue
        pl = int(a['pred_len'])
        if pl not in table or (round(m, 3), round(e, 3)) != table[pl]:
            continue
        key = tuple((k, a.get(k)) for k in CFG)
        by_cfg.setdefault(key, {}).setdefault(pl, []).append(str(run.relative_to(ROOT)))
    full = {k: v for k, v in by_cfg.items() if len(v) == 4}
    print(f'\n### {ds}: {len(full)} configuration(s) reproduce all 4 horizons (of {len(by_cfg)} partial)')
    for k, v in full.items():
        print('   ' + ' '.join(f'{a}={b}' for a, b in k))
        for pl in (96, 192, 336, 720):
            print(f'      p{pl}: ' + ' | '.join(v[pl]))
