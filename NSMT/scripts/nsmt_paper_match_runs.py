"""usage: python nsmt_paper_match_runs.py <repo-root>/forecasting

Match the NSMT paper's Table 13 cells (MSE/MAE, 3 decimals) to saved runs under root forecasting/.

A run = a directory holding log/final+result.csv (first data row = the test written right after
training; later rows are re-tests) and logargs.txt (the full argument dump).
"""
import csv, re, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else None
PAPER = {  # Table 13, NSMT column: dataset -> {pred_len: (mse, mae)}
    'ETTh1': {96: (.378, .386), 192: (.426, .417), 336: (.473, .441), 720: (.461, .454)},
    'ETTh2': {96: (.288, .334), 192: (.358, .380), 336: (.403, .412), 720: (.409, .429)},
    'ETTm1': {96: (.317, .345), 192: (.364, .371), 336: (.390, .391), 720: (.446, .425)},
    'ETTm2': {96: (.175, .251), 192: (.241, .297), 336: (.302, .335), 720: (.399, .393)},
    'traffic': {96: (.481, .290), 192: (.486, .294), 336: (.504, .299), 720: (.539, .320)},
    'electricity': {96: (.178, .260), 192: (.181, .263), 336: (.198, .280), 720: (.237, .313)},
    'exchange': {96: (.081, .198), 192: (.184, .302), 336: (.335, .418), 720: (.823, .681)},
    'weather': {96: (.174, .209), 192: (.217, .248), 336: (.274, .289), 720: (.349, .339)},
}
KEYS = ['model', 'seed', 'epoch', 'batch_size', 'lr', 'weight_decay', 'scheduler', 'patience', 'alpha',
        'embed_dim', 'num_heads', 'num_layers', 'time_num_layers', 'patch_size', 'mlp_ratios', 'max_ratio',
        'gating', 'keep_ratio', 'bias', 'tau', 'seq_len', 'pred_len', 'c_in', 'data', 'features']

def read_args(p):
    out = {}
    for line in p.read_text(errors='replace').splitlines():
        m = re.match(r'^([A-Za-z_]+)-+(.*)$', line)
        if m:
            out[m.group(1)] = m.group(2).lstrip('-')
    return out


if __name__ == '__main__':
    runs = []
    for csvp in ROOT.rglob('final+result.csv'):
        run = csvp.parent.parent
        args_p = run / 'logargs.txt'
        if not args_p.exists():
            continue
        a = read_args(args_p)
        if a.get('model') != 'myModel':
            continue
        rows = [r for r in csv.DictReader(csvp.open()) if r.get('mse') not in (None, 'mse')]
        rows = [r for r in rows if re.match(r'^[0-9.]+$', r.get('mse', ''))]
        if not rows:
            continue
        r = rows[0]
        runs.append((run, a, float(r['mse']), float(r['mae']), len(rows)))

    print(f'{len(runs)} myModel runs with results')
    hits = {}
    for ds, table in PAPER.items():
        for pl, (mse, mae) in table.items():
            cands = [(run, a, m, e, n) for run, a, m, e, n in runs
                     if a.get('dataset') == ds and a.get('pred_len') == str(pl)
                     and round(m, 3) == mse and round(e, 3) == mae]
            hits[(ds, pl)] = cands
            print(f'\n== {ds} p{pl}: paper {mse:.3f}/{mae:.3f} -> {len(cands)} matching run(s)')
            for run, a, m, e, n in cands:
                print(f'   {run.relative_to(ROOT)}   mse {m:.6f} mae {e:.6f} rows {n}')
                print('   ' + ' '.join(f'{k}={a.get(k)}' for k in KEYS))
    missing = [k for k, v in hits.items() if not v]
    print('\nunmatched cells:', missing)
