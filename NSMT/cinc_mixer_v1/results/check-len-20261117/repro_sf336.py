"""Report-only reproduction check of the ETT input-length test (2026-11-18, PROJECT_LOG, exp/f-lif-pop-v3; pre-registered 11-17 20:30).
The 6 Spikformer-style ETTh2 rows of this suite (lr 5e-4, 1 layer, input 336) should equal the other session's input-336 Spikformer-style
rows (exp/nsmt-flif-neo, nsmt_flif_v1/forecasting/results/flif-v1-ett-len336/raw.txt, read only): same code path, same settings, same seed.
Compares every RESULT field the two rows share except exp / sha / time_s. ETTm2 is not compared (different lr). Fail closed (exit 1).
usage (from cinc_mixer_v1/): python -B results/check-len-20261117/repro_sf336.py
"""
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OURS = os.path.join(C, 'results', 'flif-v1-ett-spkmux-len336', 'raw.txt')
OTHER = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1/forecasting/results/flif-v1-ett-len336/raw.txt'
IGNORE = {'exp', 'sha', 'time_s', 'tag', 'lr', 'layers', 'dir'}


def rows(p, keep):
    out = []
    for ln in open(p):
        if ln.startswith('SKIP'):
            continue
        if ln.startswith('RESULT'):
            r = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
            if keep(r):
                out.append(r)
    return out


ours = rows(OURS, lambda r: r.get('tag') == 'spk_ssa_lr0.0005_L1_s336' and r['ds'] == 'ETTh2')
other = rows(OTHER, lambda r: r.get('cond') == 'spk_ssa' and r['ds'] == 'ETTh2')
bad = 0
if len(ours) != 6 or len(other) != 6:
    print(f'FAIL  {len(ours)} / {len(other)} rows, expected 6 / 6'); sys.exit(1)
for r in sorted(ours, key=lambda r: (int(r['pl']), r['seed'])):
    q = [o for o in other if o['pl'] == r['pl'] and o['seed'] == r['seed']]
    if len(q) != 1:
        print(f'FAIL  {len(q)} other-session rows for ETTh2 H{r["pl"]} seed {r["seed"]}'); bad += 1; continue
    keys = sorted((set(r) & set(q[0])) - IGNORE)
    diff = [k for k in keys if r[k] != q[0][k]]
    print(f'ETTh2 H{r["pl"]} seed {r["seed"]}: mse {r["mse"]} / {q[0]["mse"]}, {len(keys)} shared fields, {len(diff)} differ {diff}')
    bad += bool(diff)
if bad:
    print(f'{bad} FAILED'); sys.exit(1)
print('all 6 Spikformer-style ETTh2 rows equal the other session\'s input-336 rows in every shared field')
