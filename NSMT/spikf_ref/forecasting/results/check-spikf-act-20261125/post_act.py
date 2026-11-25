"""Post hoc analysis of the SpikF head activation runs (2026-11-26, exp/f-lif-pop-v3; written after training, not pre-registered).
Reads results/spikf-act-ett/raw.txt and our two models' cells (cinc_mixer_v1/results/flif-v1-ett-spkmux-retune/judge.json). Reports:
  P1 per dataset (4-horizon mean): ours - SpikF-LIF and ours - SpikF-GELU (MSE / MAE; + = ours higher = worse), cells where ours is lower
  P2 energy: AC / MAC operations per activation, the MAC share of the energy, firing rate; ours / SpikF-LIF per cell (mean, min, max)
  P3 LIF - GELU by seed (16-cell mean MSE / MAE) and by horizon (mean r over the 4 datasets)
  P4 training length: runs that reached the epoch cap, runs whose tested (best-validation) epoch is the last epoch run
Checks (fail closed): 96 rows, no SKIP, every value finite, energy_uJ = 0.9 pJ x AC + 4.6 pJ x MAC within 1e-6 relative.
usage (from spikf_ref/forecasting/): python -B results/check-spikf-act-20261125/post_act.py
"""
import json
import math
import os
import sys

F = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = os.path.dirname(os.path.dirname(F))
raw = open(os.path.join(F, 'results', 'spikf-act-ett', 'raw.txt')).read().splitlines()
if any(l.startswith('SKIP') for l in raw):
    sys.exit('STOP: SKIP line')
R = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in raw if l.startswith('RESULT')]
if len(R) != 96 or len({(r['ds'], r['pl'], r['act'], r['seed']) for r in R}) != 96:
    sys.exit(f'STOP: {len(R)} rows')
NUM = ('mse', 'mae', 'energy_uJ', 'AC_op', 'MAC_op', 'firing_rate', 'epochs_run', 'epochs_max', 'best_epoch')
for r in R:
    for k in NUM:
        r[k] = float(r[k])
        if not math.isfinite(r[k]):
            sys.exit(f'STOP: {k} not finite')
    e = (0.9 * r['AC_op'] + 4.6 * r['MAC_op']) * 1e-6
    if abs(e - r['energy_uJ']) > 1e-6 * r['energy_uJ']:
        sys.exit(f'STOP: energy formula {e} vs {r["energy_uJ"]}')
ours = json.load(open(os.path.join(N, 'cinc_mixer_v1', 'results', 'flif-v1-ett-spkmux-retune', 'judge.json')))['models']
DS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')
PL = ('96', '192', '336', '720')
SEEDS = ('42', '7', '2026')
cells = [(d, p) for d in DS for p in PL]
mean = lambda xs: sum(xs) / len(xs)


def cm(d, p, act, k):
    return mean([r[k] for r in R if r['ds'] == d and r['pl'] == p and r['act'] == act])


print('P1 ours - SpikF (seed means; + = ours higher = worse), per dataset (mean of 4 horizons)')
for mk in ('first', 'ext'):
    for act in ('lif', 'gelu'):
        print(f'  {mk} - SpikF-{act.upper()}')
        for d in DS:
            dm = mean([ours[mk]['cells'][f'{d} {p}']['mse'] - cm(d, p, act, 'mse') for p in PL])
            da = mean([ours[mk]['cells'][f'{d} {p}']['mae'] - cm(d, p, act, 'mae') for p in PL])
            print(f'    {d}: MSE {dm:+.4f}  MAE {da:+.4f}')
        lo_m = sum(ours[mk]['cells'][f'{d} {p}']['mse'] < cm(d, p, act, 'mse') for d, p in cells)
        lo_a = sum(ours[mk]['cells'][f'{d} {p}']['mae'] < cm(d, p, act, 'mae') for d, p in cells)
        print(f'    cells where ours is lower: MSE {lo_m} / 16, MAE {lo_a} / 16')
print('P2 energy (mean over 48 runs per activation)')
for act in ('gelu', 'lif'):
    rs = [r for r in R if r['act'] == act]
    ac, mac, en = mean([r['AC_op'] for r in rs]), mean([r['MAC_op'] for r in rs]), mean([r['energy_uJ'] for r in rs])
    print(f'  {act}: AC {ac / 1e6:.3f} M, MAC {mac / 1e6:.3f} M, energy {en:.2f} uJ, MAC share of energy {4.6e-6 * mac / en * 100:.2f} %, '
          f'firing rate {mean([r["firing_rate"] for r in rs]):.4f}')
for mk in ('first', 'ext'):
    q = [ours[mk]['cells'][f'{d} {p}']['syn_energy_uJ'] / cm(d, p, 'lif', 'energy_uJ') for d, p in cells]
    print(f'  {mk} / SpikF-LIF energy per cell: mean {mean(q):.4f}, min {min(q):.4f}, max {max(q):.4f} '
          f'(= SpikF-LIF uses {1 / mean(q):.0f}x on average)')
print('P3 LIF - GELU')
for s in SEEDS:
    dm = mean([[r for r in R if (r['ds'], r['pl'], r['act'], r['seed']) == (d, p, 'lif', s)][0]['mse']
               - [r for r in R if (r['ds'], r['pl'], r['act'], r['seed']) == (d, p, 'gelu', s)][0]['mse'] for d, p in cells])
    da = mean([[r for r in R if (r['ds'], r['pl'], r['act'], r['seed']) == (d, p, 'lif', s)][0]['mae']
               - [r for r in R if (r['ds'], r['pl'], r['act'], r['seed']) == (d, p, 'gelu', s)][0]['mae'] for d, p in cells])
    print(f'  seed {s}: 16-cell mean MSE {dm:+.4f}, MAE {da:+.4f}')
for p in PL:
    rr = mean([100 * (cm(d, p, 'lif', 'mse') - cm(d, p, 'gelu', 'mse')) / cm(d, p, 'gelu', 'mse') for d in DS])
    ra = mean([100 * (cm(d, p, 'lif', 'mae') - cm(d, p, 'gelu', 'mae')) / cm(d, p, 'gelu', 'mae') for d in DS])
    print(f'  H{p}: mean r MSE {rr:+.2f} %, MAE {ra:+.2f} %')
print('P4 training length')
for act in ('gelu', 'lif'):
    rs = [r for r in R if r['act'] == act]
    cap = sum(r['epochs_run'] == r['epochs_max'] for r in rs)
    last = sum(r['best_epoch'] == r['epochs_run'] for r in rs)
    capl = sum(r['epochs_run'] == r['epochs_max'] and r['best_epoch'] == r['epochs_run'] for r in rs)
    print(f'  {act}: reached the epoch cap {cap} / 48; tested epoch = last epoch run {last} / 48; both {capl} / 48')
    for d in DS:
        rd = [r for r in rs if r['ds'] == d]
        print(f'    {d}: cap {int(rd[0]["epochs_max"])}, reached {sum(r["epochs_run"] == r["epochs_max"] for r in rd)} / 12, '
              f'best = last {sum(r["best_epoch"] == r["epochs_run"] for r in rd)} / 12')
