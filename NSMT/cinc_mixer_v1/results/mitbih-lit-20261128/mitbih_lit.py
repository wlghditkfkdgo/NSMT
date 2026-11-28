"""MIT-BIH: our models next to the SNN ECG models of the NSMT paper Table 5 (2026-11-28, exp/f-lif-pop-v3; user "SNN 분류 model 문헌 표를
NSMT 원문에서 가져와서 비교해줘"). No training.
Literature rows: NSMT/papers/NSMT.pdf Table 5 ("The reported results are taken from the corresponding papers"), copied from the text layer:
  Y. Xing et al. [44] (Electronics 2022), A. Rana and K. Kim [30] (Sensors 2024), M. Scrugli et al. [33] (DATE 2024),
  K. Buettner and A. George [5] (ISVLSI 2021), B. Reddy et al. [32] (VDAT 2024), NSMT (paper).
  Their split / class set / F1 averaging / energy measurement are those of each paper and were NOT checked here.
Our rows (logged RESULT lines, seeds 42 / 7 / 2026, 70 / 30 random beat split of the Kachuee et al. 2018 file, 5 AAMI classes, test file
5class_mitbih_test_3_187.csv, n_test 32,833; macro precision / recall / F1 over the 5 classes; energy = syops syn_energy_uJ):
  SpkMixer IAND base (flif-v1-mitbih-res spkmx_lif_muxlin_iand_lr0.01_L2), SpkMixer ADD (flif-v1-mitbih-spkmux-tune), SF-de PE on
  (flif-v1-mitbih-sfde; internal, not a paper baseline per the 11-24 decision), NSMT rerun (the other session's nsmt_flif_v1
  classification/results/flif-v1-cls cond base, read only; same data files and seeds).
Paired comparison (same split and seeds): SpkMixer IAND vs NSMT rerun, MIT-BIH rule (mean d >= +0.5 %p and >= 2 of 3 > 0 = better).
usage (from cinc_mixer_v1/): python -B results/mitbih-lit-20261128/mitbih_lit.py
"""
import json
import math
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(C, 'results')
NSMT_RAW = ('/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/'
            'nsmt_flif_v1/classification/results/flif-v1-cls/raw.txt')
SEEDS = ('42', '7', '2026')
K = ('f1', 'overall_acc', 'precision', 'sensitivity', 'syn_energy_uJ', 'n_test')
mean = lambda x: sum(x) / len(x)
LIT = [('Y. Xing et al. [44]', 98.26, 89.09, '346.33 µJ'), ('A. Rana and K. Kim [30]', 93.8, 86.4, '278 (mW)'),
       ('M. Scrugli et al. [33]', 98.4, None, '50.98 µJ'), ('K. Buettner and A. George [5]', 97.8, 87.9, '34.5 µJ'),
       ('B. Reddy et al. [32]', 85.94, None, '—'), ('NSMT (paper)', 98.45, 91.91, '19.05 µJ')]


def stop(m):
    print(f'STOP: {m}')
    sys.exit(2)


def get(path, pred):
    v = {}
    for l in open(path):
        if not l.startswith('RESULT'):
            continue
        r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
        if pred(r):
            if r['seed'] in v:
                stop(f'duplicate {path} seed {r["seed"]}')
            v[r['seed']] = {k: float(r[k]) for k in K}
    if sorted(v) != sorted(SEEDS) or not all(math.isfinite(x) for d in v.values() for x in d.values()):
        stop(f'{path}: seeds {sorted(v)} or non-finite')
    if any(d['n_test'] != 32833 for d in v.values()):
        stop(f'{path}: n_test != 32833')
    return v


OURS = {'SpkMixer IAND (base, lr 0.01)': get(os.path.join(RS, 'flif-v1-mitbih-res', 'raw.txt'), lambda r: r.get('tag') == 'spkmx_lif_muxlin_iand_lr0.01_L2'),
        'SpkMixer ADD (lr 0.02)': get(os.path.join(RS, 'flif-v1-mitbih-spkmux-tune', 'raw.txt'), lambda r: r.get('tag') == 'spkmx_lif_muxlin_add_lr0.02_L2'),
        'NSMT rerun (same split / seeds)': get(NSMT_RAW, lambda r: r.get('cond') == 'base'),
        'SF-de PE on (internal)': get(os.path.join(RS, 'flif-v1-mitbih-sfde', 'raw.txt'), lambda r: r.get('tag') == 'spk_ssa_lr0.01_L2_nalif')}
A, B = OURS['SpkMixer IAND (base, lr 0.01)'], OURS['NSMT rerun (same split / seeds)']
pair = {}
for k in ('f1', 'overall_acc', 'precision', 'sensitivity'):
    d = [100 * (A[s][k] - B[s][k]) for s in SEEDS]
    m = mean(d)
    pair[k] = dict(d_pp=d, mean_pp=m, verdict='A better' if m >= 0.5 and sum(t > 0 for t in d) >= 2 else ('A worse' if m <= -0.5 and sum(t < 0 for t in d) >= 2 else 'no clear difference'))
pair['energy_ratio'] = mean([A[s]['syn_energy_uJ'] for s in SEEDS]) / mean([B[s]['syn_energy_uJ'] for s in SEEDS])
sm = {n: {k: mean([v[s][k] for s in SEEDS]) for k in K} | {'f1_sd': (sum((v[s]['f1'] - mean([v[t]['f1'] for t in SEEDS])) ** 2 for s in SEEDS) / 3) ** .5}
      for n, v in OURS.items()}
ia = sm['SpkMixer IAND (base, lr 0.01)']
vs = [dict(row=n, acc=a, f1=f, energy=e, d_acc=100 * ia['overall_acc'] - a, d_f1=(100 * ia['f1'] - f) if f is not None else None) for n, a, f, e in LIT]
json.dump(dict(ours=sm, literature=LIT, pair_iand_vs_nsmt_rerun=pair, iand_minus_literature=vs), open(os.path.join(HERE, 'summary_mitbih_lit.json'), 'w'), indent=1)
L = ['# MIT-BIH: SNN ECG models of the NSMT paper Table 5 and our models (no training)', '', 'Sources and caveats: mitbih_lit.py docstring.', '',
     '| model | Acc % | macro P % | macro R % | macro F1 % | energy | source |', '|---|--:|--:|--:|--:|--:|---|']
for n, a, f, e in LIT:
    L.append(f'| {n} | {a:.2f} | — | — | {"—" if f is None else f"{f:.2f}"} | {e} | NSMT Table 5 (from the cited paper) |')
for n, m in sm.items():
    L.append(f"| {n} | {100 * m['overall_acc']:.2f} | {100 * m['precision']:.2f} | {100 * m['sensitivity']:.2f} | {100 * m['f1']:.2f} ± {100 * m['f1_sd']:.2f} | "
             f"{m['syn_energy_uJ']:.2f} µJ (syops) | logged, seeds 42 / 7 / 2026 |")
L += ['', '| SpkMixer IAND minus | Acc %p | F1 %p |', '|---|--:|--:|']
for x in vs:
    df1 = '—' if x['d_f1'] is None else f"{x['d_f1']:+.2f}"
    L.append(f"| {x['row']} | {x['d_acc']:+.2f} | {df1} |")
L += ['', 'Paired, same split and seeds: SpkMixer IAND vs NSMT rerun -- ' + '; '.join(
    f"{k} {' / '.join(f'{t:+.2f}' for t in pair[k]['d_pp'])} (mean {pair[k]['mean_pp']:+.2f} %p, {pair[k]['verdict']})" for k in ('f1', 'overall_acc', 'precision', 'sensitivity'))
      + f"; energy {pair['energy_ratio']:.3f}x"]
open(os.path.join(HERE, 'summary_mitbih_lit.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
