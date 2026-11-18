"""Summary of the mean-pooling energy recount (2026-11-19, report only): LIF SpkMixer with / without mean pooling and SF-de, energy under the
earlier rule (E_old) and the pooled-integer rule (E_pool; energy_recount_pool.py). Ratios as summarize_l2.compare / cmp_mit: ETT = mean over the
4 cells of (seed-mean A / seed-mean B), MIT-BIH = seed-mean A / seed-mean B. Accuracy comparisons are the logged ones (unchanged by the recount).
usage (from cinc_mixer_v1/): python -B results/check-pool-energy-20261119/summarize_pool.py"""
import csv, json, os
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(H)
P = list(csv.DictReader(open(os.path.join(H, 'energy_pool.csv'))))
CELLS = [('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720')]
m = lambda v: sum(v) / len(v)
sel = json.load(open(os.path.join(R, 'flif-v1-ett-sfde', 'selected_lr.json')))['selected']
LY = {'ETTh2': '1', 'ETTm2': '2'}
sf = {}
for r in csv.DictReader(open(os.path.join(R, 'flif-v1-ett-sfde', 'energy_recount.csv'))):
    if r['tag'] == f"spk_ssa_lr{sel[r['ds']]}_L{LY[r['ds']]}_nalif":
        sf.setdefault((r['ds'], r['pl']), []).append(float(r['total_fft_uJ_corr']))
def ett(kind, col):
    v = {}
    for r in P:
        if r['task'] == 'ETT' and r['tag'].endswith('_mpmean') == (kind == 'mean'):
            v.setdefault((r['ds'], r['pl']), []).append(float(r[col]))
    return v
L = ['# Mean-pooling energy recount (pooled-integer rule), LIF SpkMixer', '',
     '| task | arm | energy µJ, earlier rule | energy µJ, pooled-integer rule | ratio vs LIF base (earlier -> pooled) | ratio vs SF-de (earlier -> pooled) |', '|---|---|--:|--:|---|---|']
base_o, mean_o, mean_p = ett('base', 'E_old'), ett('mean', 'E_old'), ett('mean', 'E_pool')
assert all(len(base_o[c]) == len(mean_o[c]) == len(sf[c]) == 3 for c in CELLS), 'missing ETT rows'
avg = lambda d: m([m(d[c]) for c in CELLS])
rat = lambda a, b: m([m(a[c]) / m(b[c]) for c in CELLS])
L.append(f"| ETT | LIF base | {avg(base_o):.2f} | {avg(ett('base', 'E_pool')):.2f} | 1 | {rat(base_o, sf):.4f} |")
L.append(f"| ETT | LIF + mean pooling | {avg(mean_o):.2f} | {avg(mean_p):.2f} | {rat(mean_o, base_o):.4f} -> {rat(mean_p, base_o):.4f} | {rat(mean_o, sf):.4f} -> {rat(mean_p, sf):.4f} |")
L.append(f"| ETT | SF-de | {avg(sf):.2f} | {avg(sf):.2f} | | 1 |")
mt = lambda kind, col: [float(r[col]) for r in P if r['task'] == 'MITBIH' and r['tag'].endswith('_mpmean') == (kind == 'mean')]
sfm = [float(r['total_fft_uJ_corr']) for r in csv.DictReader(open(os.path.join(R, 'flif-v1-mitbih-sfde', 'energy_recount.csv')))
       if r['tag'] == 'spk_ssa_lr0.01_L2_nalif' and r['val_fold'] == '-1']
bo, mo, mp = mt('base', 'E_old'), mt('mean', 'E_old'), mt('mean', 'E_pool')
assert len(bo) == len(mo) == len(sfm) == 3, 'missing MIT-BIH rows'
L.append(f"| MIT-BIH | LIF base | {m(bo):.2f} | {m(mt('base', 'E_pool')):.2f} | 1 | {m(bo) / m(sfm):.4f} |")
L.append(f"| MIT-BIH | LIF + mean pooling | {m(mo):.2f} | {m(mp):.2f} | {m(mo) / m(bo):.4f} -> {m(mp) / m(bo):.4f} | {m(mo) / m(sfm):.4f} -> {m(mp) / m(sfm):.4f} |")
L.append(f"| MIT-BIH | SF-de | {m(sfm):.2f} | {m(sfm):.2f} | | 1 |")
L += ['', 'ETT per cell (seed mean, µJ): ' + '; '.join(f"{d} H{p}: base {m(base_o[(d, p)]):.2f}, mean {m(mean_o[(d, p)]):.2f} -> {m(mean_p[(d, p)]):.2f}, SF-de {m(sf[(d, p)]):.2f}" for d, p in CELLS)]
open(os.path.join(H, 'summary.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
