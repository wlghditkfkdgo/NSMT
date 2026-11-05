"""Results of SpkMixer muxlin next to the NSMT paper and the Spikformer-style model (2026-11-05, exp/f-lif-pop-v3; user: "Muxlin+Neo-mixer을
실험한 결과 중 가장 효과적이었던 버전으로 결과표를 정리해줘. 그리고 NSMT 원문의 모델 결과랑 Spikformer와 비교해줘."). Report only, no
training. All values are recomputed from RESULT lines; the other session's files in the main tree (nsmt_flif_v1, exp/nsmt-flif-neo) are read
only; the paper values are typed from NSMT/papers/NSMT.pdf (Tables 5, 6, 9, 14; checked against the PDF text layer).

Versions:
    muxlin    "most effective" = the validation-selected settings of each task's tuning (no test value chooses them). The default-setting
              runs are shown next to them; on ETT the setting with the lowest test MSE per cell is also shown (post hoc, optimistic).
    NSMT      paper   = the published numbers (ETT: one run, seed 42, patience 2-3, last weights: the rerun with seed 42 gives them)
              rerun-a = our rerun with the paper's training rule (flif-v1-main cond base, 3 seeds)
              rerun-b = our rerun with the rule of the muxlin / Spikformer-style ETT runs (flif-v1-s2p10 cond base: patience 10, best
                        validation weights, 3 seeds); the paired ETT comparisons use this one
    Spikformer-style  tuned and default (this branch's selections).
Verdict rules are the tasks' existing ones: ETT R <= -1 % and >= 8 / 12 pairs (summarize_ett_tune.compare), MIT-BIH / CinC mean >= +0.5 %p and
>= 2 / 3 seeds, AD R >= +2 % and >= 6 / 9 pairs (AUC-PR). NSMT was not rerun on AD (nsmt_base/anomaly_detection is unverified), so AD has
paper values only and no paired verdict. Energy: total = synapses + neuron state updates, head recounted as AC where its input is an
average of 0/1 spikes (total_fft_uJ_corr); syn = total - neuron_model_fft_uJ, the quantity of the paper's Tables 5, 6. NSMT's head reads 0/1
per step (syops AC), so its logged values need no recount. Fail closed: a missing seed -> SystemExit.
usage: python compare_nsmt.py   ->  results/compare-nsmt-20261105/summary.md
"""
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402
import summarize_ett_tune as et                                           # noqa: E402
import summarize_mitbih_tune as sm                                        # noqa: E402
import summarize_ad_tune as ad                                            # noqa: E402
import summarize_cinc_tune as ct                                          # noqa: E402

MAIN = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1'
MC, MF = os.path.join(MAIN, 'classification', 'results'), os.path.join(MAIN, 'forecasting', 'results')
OUT = os.path.join(HERE, 'results', 'compare-nsmt-20261105')
SEEDS = (42, 7, 2026)
PAPER_ETT = {('ETTh2', 96): (0.288, 0.334), ('ETTh2', 720): (0.409, 0.429), ('ETTm2', 96): (0.175, 0.251), ('ETTm2', 720): (0.399, 0.393)}   # Table 14
PAPER_REF = {'SpikF': {('ETTh2', 96): (0.290, 0.336), ('ETTh2', 720): (0.416, 0.436), ('ETTm2', 96): (0.175, 0.251), ('ETTm2', 720): (0.405, 0.397)},
             'iSpikformer': {('ETTh2', 96): (0.308, 0.349), ('ETTh2', 720): (0.438, 0.439), ('ETTm2', 96): (0.178, 0.256), ('ETTm2', 720): (0.406, 0.398)}}
PAPER_SYN_H720 = {'ETTh2': 44.90, 'ETTm2': 46.72}                         # Table 6, synapses only
PAPER_MIT = dict(acc=98.45, f1=91.91, syn=19.05)                          # Table 5
PAPER_AD = {'MSL': (88.24, 67.54, 76.51), 'SMAP': (91.61, 64.57, 75.75), 'PSM': (98.54, 92.56, 95.46)}   # Table 9, P / R / F1


def need(v, what):
    if v is None or len(v) != 3 or any(x is None for x in v):
        raise SystemExit(f'missing runs: {what}')
    return v


def nsmt_ett(suite, best):
    R = {}
    for r in parse(os.path.join(MF, suite, 'raw.txt')):
        if r['cond'] != 'base':
            continue
        ds, pl = r['dataset'].split('_p')
        s = '_best' if best else ''
        k = (ds, int(pl), int(r['seed']))
        if k in R:
            raise SystemExit(f'{suite}: duplicate {k}')
        R[k] = dict(mse=float(r['mse' + s]), mae=float(r['mae' + s]), E=float(r['total_fft_uJ' + s]), syn=float(r['syn_energy' + s]), seed=int(r['seed']))
    return lambda ds, pl: need([R.get((ds, pl, x)) for x in SEEDS], f'NSMT {suite} {ds} {pl}')


def m(rr, k, scale=1.):
    return float(np.mean([scale * float(x[k]) for x in rr]))


def sd(rr, k, scale=1.):
    return float(np.std([scale * float(x[k]) for x in rr]))


def ett(L):
    D = et.load()
    mux_t = et.rows(D, et.MUX, 'iand', {'ETTh2': 0.0005, 'ETTm2': 0.0001}, 1)
    mux_d = et.rows(D, et.MUX, 'or', et.LR0, et.L0)
    spk_t = et.rows(D, et.REF, '-', {'ETTh2': 0.0005, 'ETTm2': 0.00025}, et.L0)
    spk_d = et.rows(D, et.REF, '-', et.LR0, et.L0)
    a, b = nsmt_ett('flif-v1-main', False), nsmt_ett('flif-v1-s2p10', True)
    for f in (mux_t, mux_d, spk_t, spk_d):
        for r in (x for c in et.CELLS for x in need(f(*c), f'ett {c}')):
            r['syn'] = float(r['E']) - float(r['neuron_model_fft_uJ'])
    L += ['## ETT test MSE / MAE (scaled data; seeds 42, 7, 2026, mean; paper = one run)', '',
          '| model | ' + ' | '.join(f'{d} H{p}' for d, p in et.CELLS) + ' | mean MSE |', '|---|' + '---|' * 5]
    def row(name, f):
        v = [f(*c) for c in et.CELLS]
        L.append(f'| {name} | ' + ' | '.join(f"{m(x, 'mse'):.4f} / {m(x, 'mae'):.4f}" for x in v) + f" | {np.mean([m(x, 'mse') for x in v]):.4f} |")
    row('muxlin (validation-selected)', mux_t)
    row('muxlin (default settings)', mux_d)
    # post hoc: the muxlin setting with the lowest seed-mean test MSE per cell (chosen on test, optimistic)
    best = {}
    for k, r in D.items():
        if k[0] != et.MUX:
            continue
        best.setdefault((k[2], k[3], k[1], k[4], k[5]), []).append(r)
    def hoc(ds, pl):
        c = {k: v for k, v in best.items() if k[:2] == (ds, pl) and len(v) == 3}
        return min(c.items(), key=lambda kv: m(kv[1], 'mse'))
    v = [hoc(*c) for c in et.CELLS]
    L.append('| muxlin (lowest test MSE per cell, post hoc) | ' + ' | '.join(f"{m(x[1], 'mse'):.4f} / {m(x[1], 'mae'):.4f}" for x in v) + f" | {np.mean([m(x[1], 'mse') for x in v]):.4f} |")
    row('NSMT rerun-b (same rule as muxlin)', b)
    row('NSMT rerun-a (paper rule)', a)
    L.append('| NSMT paper (Table 14) | ' + ' | '.join(f'{PAPER_ETT[c][0]:.3f} / {PAPER_ETT[c][1]:.3f}' for c in et.CELLS) + f' | {np.mean([PAPER_ETT[c][0] for c in et.CELLS]):.4f} |')
    row('Spikformer-style (tuned)', spk_t)
    row('Spikformer-style (default)', spk_d)
    for n, t in PAPER_REF.items():
        L.append(f'| {n} (paper Table 14) | ' + ' | '.join(f'{t[c][0]:.3f} / {t[c][1]:.3f}' for c in et.CELLS) + f' | {np.mean([t[c][0] for c in et.CELLS]):.4f} |')
    L += ['', 'Post-hoc settings (residual, lr, mixer layers): ' + '; '.join(f'{c[0]} H{c[1]} {x[0][2]}, {x[0][3]:g}, {x[0][4]}' for c, x in zip(et.CELLS, v)), '',
          '## ETT energy (µJ per window; total / synapses only)', '', '| model | ' + ' | '.join(f'{d} H{p}' for d, p in et.CELLS) + ' |', '|---|' + '---|' * 4]
    for name, f in (('muxlin (validation-selected)', mux_t), ('NSMT rerun-b', b), ('NSMT rerun-a', a), ('Spikformer-style (tuned)', spk_t)):
        L.append(f'| {name} | ' + ' | '.join(f"{m(f(*c), 'E'):.2f} / {m(f(*c), 'syn'):.2f}" for c in et.CELLS) + ' |')
    L.append('| NSMT paper (Table 6, synapses) | ' + ' | '.join(f'— / {PAPER_SYN_H720[c[0]]:.2f}' if c[1] == 720 else '—' for c in et.CELLS) + ' |')
    L += ['', '## ETT paired comparisons (R % = mean of per-cell seed-mean r; rule: R <= -1 % and >= 8 / 12 lower)', '',
          '| A vs B | R % | lower pairs | verdict | energy A/B | per cell r % |', '|---|---:|---:|---|---:|---|']
    for name, A, B in (('muxlin (selected) vs NSMT rerun-b', mux_t, b), ('muxlin (selected) vs NSMT rerun-a', mux_t, a), ('muxlin (selected) vs Spikformer-style (tuned)', mux_t, spk_t),
                       ('NSMT rerun-b vs Spikformer-style (tuned)', b, spk_t)):
        c = et.compare(D, A, B, et.CELLS)
        if c is None:
            raise SystemExit(f'ett incomplete: {name}')
        L.append(f"| {name} | {c['R']:+.2f} | {c['lo']}/{c['n']} | {c['verdict']} | {c['energy']:.2f} | " + '; '.join(x.split(' (')[0] for x in c['cells']) + ' |')
    L.append('')


def verdict(d):
    x = float(np.mean(d))
    return 'better' if x >= 0.5 and sum(z > 0 for z in d) >= 2 else 'worse' if x <= -0.5 and sum(z < 0 for z in d) >= 2 else 'no clear difference'


def mitbih(L):
    D = sm.load()
    sel = {s: json.load(open(os.path.join(sm.TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
    mux_t = need(sm.seeds(D, sm.MUX, sm.tuned_cfg(sm.MUX, sel), -1), 'mitbih muxlin tuned')
    mux_d = need(sm.seeds(D, sm.MUX, (sm.R0, sm.LR0, sm.L0), -1), 'mitbih muxlin default')
    spk_t = need(sm.seeds(D, sm.REF, sm.tuned_cfg(sm.REF, sel), -1), 'mitbih spikformer tuned')
    spk_d = need(sm.seeds(D, sm.REF, ('-', sm.LR0, sm.L0), -1), 'mitbih spikformer default')
    base = {int(r['seed']): r for r in parse(os.path.join(MC, 'flif-v1-cls', 'raw.txt')) if r['cond'] == 'base'}
    nsmt = need([base.get(s) for s in SEEDS], 'mitbih NSMT rerun')
    for r in nsmt:
        r['E'] = float(r['total_fft_uJ'])
    L += ['## MIT-BIH test (70 / 30 random beat split; seeds 42, 7, 2026)', '',
          '| model | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy total µJ | synapses µJ |', '|---|---:|---:|---|---:|---:|']
    syn = lambda r: float(r['E']) - float(r['neuron_model_fft_uJ'])
    for name, rr in (('muxlin (validation-selected: ADD, lr 0.02, 2 layers)', mux_t), ('muxlin (default: OR, lr 0.01, 2 layers)', mux_d), ('NSMT rerun', nsmt),
                     ('Spikformer-style (tuned, lr 0.005)', spk_t), ('Spikformer-style (default, lr 0.01)', spk_d)):
        L.append(f"| {name} | {m(rr, 'f1', 100):.2f} ± {sd(rr, 'f1', 100):.2f} | {m(rr, 'overall_acc', 100):.2f} | " + ' / '.join(f"{m(rr, f'f1_{i}', 100):.2f}" for i in range(5))
                 + f" | {m(rr, 'E'):.2f} | {np.mean([syn(r) for r in rr]):.2f} |")
    L.append(f"| NSMT paper (Table 5) | {PAPER_MIT['f1']:.2f} | {PAPER_MIT['acc']:.2f} | — | — | {PAPER_MIT['syn']:.2f} |")
    L += ['', '| A vs B | A − B %p (seed 42, 7, 2026) | mean | verdict | energy A/B |', '|---|---|---:|---|---:|']
    for name, A, B in (('muxlin (selected) vs NSMT rerun', mux_t, nsmt), ('muxlin (selected) vs Spikformer-style (tuned)', mux_t, spk_t),
                       ('muxlin (selected) vs Spikformer-style (default)', mux_t, spk_d), ('NSMT rerun vs Spikformer-style (default)', nsmt, spk_d)):
        d = [100 * (float(a['f1']) - float(b['f1'])) for a, b in zip(A, B)]
        L.append(f"| {name} | {', '.join(f'{x:+.2f}' for x in d)} | {np.mean(d):+.2f} | {verdict(d)} | {m(A, 'E') / m(B, 'E'):.2f} |")
    L.append('')


def anomaly(L):
    D = ad.load()
    sel = {s: json.load(open(os.path.join(ad.TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
    ck = ad.CONDKEY[ad.MUX]
    L += ['## AD test (TSLib procedure; seeds 42, 7, 2026, mean; NSMT was not rerun here)', '',
          '| data | model | AUC-PR % | AUC-ROC % | raw F1 % | point-adjusted F1 % | energy µJ |', '|---|---|---:|---:|---:|---:|---:|']
    triv = {}
    p = os.path.join(ad.TUNE, 'posthoc_trivial_baselines.md')
    for line in open(p):
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) == 7 and c[0] in ad.DSS:
            triv[(c[0], c[1])] = c
    pairs = {'mux': [], }
    for ds in ad.DSS:
        rows = [(f"muxlin (validation-selected: {sel['res'][ck][ds].upper()}, lr {sel['lr'][ck][ds]:g}, {sel['layers'][ck][ds]} layers)",
                 ad.seeds(D, ad.MUX, ds, (sel['res'][ck][ds], sel['lr'][ck][ds], sel['layers'][ck][ds]), 0)),
                ('muxlin (default: OR, lr 5e-4, 2 layers)', ad.seeds(D, ad.MUX, ds, (ad.R0, ad.LR0, ad.L0), 0)),
                (f"Spikformer-style (tuned, lr {sel['lr']['spk_ssa'][ds]:g})", ad.seeds(D, ad.REF, ds, ('-', sel['lr']['spk_ssa'][ds], ad.L0), 0))]
        for name, rr in rows:
            need(rr, f'ad {ds} {name}')
            L.append(f"| {ds} | {name} | {m(rr, 'aucpr', 100):.2f} | {m(rr, 'aucroc', 100):.2f} | {m(rr, 'raw_f1', 100):.2f} | {m(rr, 'pa_f1', 100):.2f} | {m(rr, 'total_fft_uJ_corr'):.2f} |")
        L.append(f'| {ds} | NSMT paper (Table 9) | — | — | — | {PAPER_AD[ds][2]:.2f} | — |')
        for t in ('random', 'win-mean', 'train-dev'):
            c = triv[(ds, t)]
            L.append(f'| {ds} | no training: {t} | {c[2]} | {c[3]} | {c[4]} | {c[5]} | 0 |')
        pairs.setdefault('rows', []).append((ds, rows[0][1], rows[1][1], rows[2][1]))
    d = {'selected vs Spikformer-style (tuned)': [], 'default vs Spikformer-style (tuned)': []}
    for ds, s, dflt, r in pairs['rows']:
        d['selected vs Spikformer-style (tuned)'].append([100 * (float(a['aucpr']) / float(b['aucpr']) - 1) for a, b in zip(s, r)])
        d['default vs Spikformer-style (tuned)'].append([100 * (float(a['aucpr']) / float(b['aucpr']) - 1) for a, b in zip(dflt, r)])
    L += ['', '| muxlin vs (AUC-PR; rule R >= +2 % and >= 6 / 9 higher) | R % | higher pairs | verdict |', '|---|---:|---:|---|']
    for k, v in d.items():
        R = float(np.mean([np.mean(x) for x in v])); fl = [z for x in v for z in x]; hi = sum(z > 0 for z in fl)
        L.append(f"| {k} | {R:+.2f} | {hi}/9 | {'better' if R >= 2 and hi >= 6 else 'worse' if R <= -2 and 9 - hi >= 6 else 'no clear difference'} |")
    L.append('')


def cinc(L):
    D = ct.load()
    mux = need(ct.seeds(D, ct.MUX, ('or', 0.001, ct.L0)), 'cinc muxlin')
    spk_t, spk_d = need(ct.seeds(D, ct.REF, ('-', 0.002, ct.L0)), 'cinc spk tuned'), need(ct.seeds(D, ct.REF, ('-', 0.001, ct.L0)), 'cinc spk default')
    base = {int(r['seed']): r for r in parse(os.path.join(MC, 'flif-v1-rhythm-cinc', 'raw.txt')) if r['cond'] == 'nsmt_lif' and int(r['fold']) == 0}
    nsmt = need([base.get(s) for s in SEEDS], 'cinc NSMT rerun')
    for r in nsmt:
        r['E'] = float(r['total_fft_uJ'])
    L += ['## CinC fold 0 test (support only, not a main result; 30 epochs; the paper has no CinC result)', '',
          '| model | CinC score % | macro-F1 % | energy µJ |', '|---|---:|---:|---:|']
    for name, rr in (('muxlin (OR, lr 1e-3, 2 layers)', mux), ('NSMT rerun (LIF)', nsmt), ('Spikformer-style (tuned, lr 2e-3)', spk_t), ('Spikformer-style (default, lr 1e-3)', spk_d)):
        L.append(f"| {name} | {m(rr, 'score', 100):.2f} ± {sd(rr, 'score', 100):.2f} | {m(rr, 'macro_f1', 100):.2f} | {m(rr, 'E'):.1f} |")
    L += ['', '| A vs B | A − B %p (seed 42, 7, 2026) | mean | verdict | energy A/B |', '|---|---|---:|---|---:|']
    for name, A, B in (('muxlin vs NSMT rerun', mux, nsmt), ('muxlin vs Spikformer-style (tuned)', mux, spk_t), ('NSMT rerun vs Spikformer-style (default)', nsmt, spk_d)):
        d = [100 * (float(a['score']) - float(b['score'])) for a, b in zip(A, B)]
        L.append(f"| {name} | {', '.join(f'{x:+.2f}' for x in d)} | {np.mean(d):+.2f} | {verdict(d)} | {m(A, 'E') / m(B, 'E'):.2f} |")
    L.append('')


def main():
    L = ['# SpkMixer muxlin vs the NSMT paper and the Spikformer-style model (report only, 2026-11-05)', '', 'Sources and definitions: see the module docstring.', '']
    ett(L); mitbih(L); anomaly(L); cinc(L)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
