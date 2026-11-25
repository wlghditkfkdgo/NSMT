"""Post hoc report of the ETTh1 / ETTm1 retuning (2026-11-25, exp/f-lif-pop-v3; written after the driver ended, not pre-registered,
not used by the judgment). Reads only raw.txt of the retune suite and of the 11-24 full table, and judge.json; imports nothing from
this repo.
  1 run facts: training commit of every row, epochs run (50-epoch cap reached?), best epoch, GPU time
  2 the 11-24 copied settings (ETTh2 / ETTm2 finals applied to ETTh1 / ETTm1) against the retuned finals on the selection cells
    (H96 / H720): seed-mean best validation L1 and test MSE
  3 test MAE per cell and the 8-cell means (ETTh1 / ETTm1) of both, MSE and MAE
usage (from cinc_mixer_v1/): python -B results/check-retune-20261125/post_rt.py > results/check-retune-20261125/post_rt.txt
"""
import collections
import json
import os
import statistics as st

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(s):
    out = []
    for l in open(os.path.join(R, s, 'raw.txt')):
        if l.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return out


rt, f16 = rows('flif-v1-ett-spkmux-retune'), rows('flif-v1-ett-spkmux-full16')
J = json.load(open(os.path.join(R, 'flif-v1-ett-spkmux-retune', 'judge.json')))
print('## 1 run facts')
print(f'rows {len(rt)}; training commit {dict(collections.Counter(r["sha"] for r in rt))}')
ep = [int(r['best_epoch_val_epochs'].split(':')[2]) for r in rt]
be = [int(r['best_epoch_val_epochs'].split(':')[0]) for r in rt]
print(f'epochs run: max {max(ep)}, rows that reached 50: {sum(e >= 50 for e in ep)}, median {st.median(ep)}')
print(f'best epoch (0-based): median {st.median(be)}, rows with best epoch 0: {sum(b == 0 for b in be)} '
      f'(ETTh1 {sum(1 for r, b in zip(rt, be) if b == 0 and r["ds"] == "ETTh1")}, ETTm1 {sum(1 for r, b in zip(rt, be) if b == 0 and r["ds"] == "ETTm1")})')
for ds in ('ETTh1', 'ETTm1'):
    t = [int(r['time_s']) for r in rt if r['ds'] == ds]
    print(f'{ds}: run time median {st.median(t)} s, max {max(t)} s, sum {sum(t) / 3600:.1f} GPU h')
print(f'all rows: {sum(int(r["time_s"]) for r in rt) / 3600:.1f} GPU h')
OLD = {'a': {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt16_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt16_mlp2_wd0.1'},
       'x': {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt24_wd0.005', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt24_e256_blk2_wd0.02'}}
NAME = {'a': 'first chain', 'x': 'extension'}


def m(rs, ds, pl, tag, k):
    v = [float(r[k]) for r in rs if r['ds'] == ds and r['pl'] == pl and r['tag'] == tag]
    assert len(v) == 3, (ds, pl, tag, len(v))
    return sum(v) / 3


print('\n## 2 selection cells: 11-24 copied setting (old) vs retuned final (new), seed means')
print('| model | cell | validation L1 old | new | change | test MSE old | new | change |')
print('|---|---|--:|--:|--:|--:|--:|--:|')
for ch in 'ax':
    for ds in ('ETTh1', 'ETTm1'):
        o, n = OLD[ch][ds], J['final_tags'][ch][ds]
        for pl in ('96', '720'):
            vo, vn, to, tn = m(f16, ds, pl, o, 'val_loss_best'), m(rt, ds, pl, n, 'val_loss_best'), m(f16, ds, pl, o, 'mse'), m(rt, ds, pl, n, 'mse')
            print(f'| {NAME[ch]} | {ds} H{pl} | {vo:.6f} | {vn:.6f} | {(vn / vo - 1) * 100:+.2f} % | {to:.4f} | {tn:.4f} | {(tn / to - 1) * 100:+.2f} % |')
print('\n## 3 test MAE per cell and 8-cell means (ETTh1 / ETTm1)')
print('| model | cell | MAE old | new | change |')
print('|---|---|--:|--:|--:|')
for ch in 'ax':
    s = collections.defaultdict(float)
    for ds in ('ETTh1', 'ETTm1'):
        o, n = OLD[ch][ds], J['final_tags'][ch][ds]
        for pl in ('96', '192', '336', '720'):
            ao, an, mo, mn = m(f16, ds, pl, o, 'mae'), m(rt, ds, pl, n, 'mae'), m(f16, ds, pl, o, 'mse'), m(rt, ds, pl, n, 'mse')
            s['ao'] += ao; s['an'] += an; s['mo'] += mo; s['mn'] += mn
            print(f'| {NAME[ch]} | {ds} H{pl} | {ao:.4f} | {an:.4f} | {an - ao:+.4f} |')
    print(f'| {NAME[ch]} | 8-cell mean MSE | {s["mo"] / 8:.4f} | {s["mn"] / 8:.4f} | {(s["mn"] - s["mo"]) / 8:+.4f} ({(s["mn"] / s["mo"] - 1) * 100:+.2f} %) |')
    print(f'| {NAME[ch]} | 8-cell mean MAE | {s["ao"] / 8:.4f} | {s["an"] / 8:.4f} | {(s["an"] - s["ao"]) / 8:+.4f} ({(s["an"] / s["ao"] - 1) * 100:+.2f} %) |')
