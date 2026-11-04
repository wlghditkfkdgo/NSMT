"""Candidate comparison across tasks (2026-11-04, exp/f-lif-pop-v3; user: pick the setting / model most likely to be adopted for our
contribution over CinC, MIT-BIH, ETT and AD). Report only, no training. Reads this branch's results and, read-only, the other
session's query-form results in the main tree (nsmt_flif_v1, branch exp/nsmt-flif-neo; nothing there is written).
    A  SpkMixer muxlin LIF (this branch; tuned settings = summarize_*_tune selections)
    B  NSMT query form, population LIF / population f-LIF Neocortex (nsmt4q_pop_lif / nsmt4q_pop_flif, 90 epochs on CinC)
Baselines: Spikformer-style default (lr 1e-3 CinC, 5e-4 ETT) and tuned (this branch's selections: CinC 2e-3, ETTh2 5e-4, ETTm2
2.5e-4). Pairs are same-seed (42 / 7 / 2026). CinC rule: mean >= +0.5 %p and >= 2/3 seeds (per fold), 5-fold pooled 15 pairs with a
95 % t interval; ETT rule (summarize_ett_tune.compare): R <= -1 % and >= 8 / 12 lower pairs; energy = total_fft_int_uJ (CinC,
else total_fft_uJ) / total_fft_uJ_corr (ETT). Fail closed: a missing seed -> SystemExit.
usage: python compare_candidates.py   ->  results/candidates-20261104/summary.md
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402
import summarize_ett_tune as et                                           # noqa: E402
import summarize_cinc_tune as ct                                          # noqa: E402

MAIN = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1'
MC, MF = os.path.join(MAIN, 'classification', 'results'), os.path.join(MAIN, 'forecasting', 'results')
OUT = os.path.join(HERE, 'results', 'candidates-20261104')
SEEDS = (42, 7, 2026)
Q = ('nsmt4q_pop_lif', 'nsmt4q_pop_flif')


def need(rows, what):
    if rows is None or len(rows) != 3 or any(r is None for r in rows):
        raise SystemExit(f'missing runs: {what}')
    return rows


def tint(d):
    d = np.asarray(d, float); n = len(d); m = float(d.mean()); s = float(d.std(ddof=1))
    t = {14: 2.1448, 2: 4.3027}[n - 1]
    return m, m - t * s / math.sqrt(n), m + t * s / math.sqrt(n)


def cinc(L):
    D = ct.load()
    tuned = lambda b, lr, lay=ct.L0, res='or': need(ct.seeds(D, b, ('-', lr, lay) if b == ct.REF else (res, lr, lay)), f'cinc {b} {lr}')
    spk_t, spk_d, mux = tuned(ct.REF, 0.002), tuned(ct.REF, 0.001), tuned(ct.MUX, 0.001)
    q0 = {r['cond']: r for r in []}
    rows0 = [r for r in parse(os.path.join(MC, 'flif-v1-rhythm-query-cinc', 'raw.txt')) if r['cond'] in Q]
    rows14 = [r for r in parse(os.path.join(MC, 'flif-v1-rhythm-query-cv-cinc', 'raw.txt')) if r['cond'] in Q]
    spk14 = [r for r in parse(os.path.join(MC, 'flif-v1-rhythm-folds-cinc', 'raw.txt')) if r['cond'] == 'spk_ssa']
    def pick(rows, cond, fold):
        v = {int(r['seed']): r for r in rows if r['cond'] == cond and int(r.get('fold', 0)) == fold}
        return need([v.get(s) for s in SEEDS], f'cinc {cond} fold {fold}')
    E = lambda r: float(r['total_fft_int_uJ']) if 'total_fft_int_uJ' in r else float(r['total_fft_uJ'])
    def pair(A, B, name):
        d = [100 * (float(a['score']) - float(b['score'])) for a, b in zip(A, B)]
        m = float(np.mean(d)); v = 'better' if m >= 0.5 and sum(x > 0 for x in d) >= 2 else 'worse' if m <= -0.5 and sum(x < 0 for x in d) >= 2 else 'no clear difference'
        e = np.mean([E(a) for a in A]) / np.mean([E(b) for b in B])
        L.append(f'| {name} | {np.mean([100 * float(a["score"]) for a in A]):.2f} | {np.mean([100 * float(b["score"]) for b in B]):.2f} | '
                 f'{m:+.2f} ({", ".join(f"{x:+.2f}" for x in d)}) | {v} | {e:.2f} |')
    L += ['## CinC fold 0 (test CinC score %, best-validation weights)', '',
          '| A vs B | A mean | B mean | A − B %p (seed 42, 7, 2026) | verdict | energy A/B |', '|---|---:|---:|---|---|---:|']
    pair(mux, spk_t, 'muxlin (30 ep) vs Spikformer-style tuned (lr 2e-3, 30 ep)')
    pair(mux, spk_d, 'muxlin (30 ep) vs Spikformer-style default (lr 1e-3, 30 ep)')
    for c in Q:
        pair(pick(rows0, c, 0), spk_t, f'{c} (90 ep) vs Spikformer-style tuned')
        pair(pick(rows0, c, 0), spk_d, f'{c} (90 ep) vs Spikformer-style default')
        pair(mux, pick(rows0, c, 0), f'muxlin vs {c}')
    L += ['', '## CinC folds 0-4: B vs Spikformer-style default (15 pairs; reproduces the other session\'s order summary)', '',
          '| model | 5-fold mean | Spikformer-style mean | pooled %p | 95 % t interval | pairs ahead | energy A/B |', '|---|---:|---:|---:|---|---:|---:|']
    for c in Q:
        A = pick(rows0, c, 0) + sum((pick(rows14, c, f) for f in (1, 2, 3, 4)), [])
        B = spk_d + sum((pick(spk14, 'spk_ssa', f) for f in (1, 2, 3, 4)), [])
        d = [100 * (float(a['score']) - float(b['score'])) for a, b in zip(A, B)]
        m, lo, hi = tint(d)
        L.append(f'| {c} | {np.mean([100 * float(a["score"]) for a in A]):.2f} | {np.mean([100 * float(b["score"]) for b in B]):.2f} | '
                 f'{m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {sum(x > 0 for x in d)}/15 | {np.mean([E(a) for a in A]) / np.mean([E(b) for b in B]):.2f} |')
    L.append('')


def ett(L):
    D = et.load()
    def qrows(cond):
        R = {}
        en = {}
        import csv
        for sub in ('flif-v1-ett-query', 'flif-v1-ett-query-m2'):
            for e in csv.DictReader(open(os.path.join(MF, sub, 'energy_recount.csv'))):
                en[(e['cond'], e['ds'], int(e['pl']), int(e['seed']))] = float(e['total_fft_uJ_corr'])
            for r in parse(os.path.join(MF, sub, 'raw.txt')):
                if r['cond'] == cond:
                    k = (r['ds'], int(r['pl']), int(r['seed']))
                    if k in R:
                        raise SystemExit(f'duplicate {cond} {k}')
                    r['E'] = en.get((cond, r['ds'], int(r['pl']), int(r['seed'])), float('nan'))
                    R[k] = r
        return lambda ds, pl: need([R.get((ds, pl, s)) for s in SEEDS], f'ett {cond} {ds} {pl}')
    spk_d = et.rows(D, et.REF, '-', et.LR0, et.L0)
    spk_t = et.rows(D, et.REF, '-', {'ETTh2': 0.0005, 'ETTm2': 0.00025}, et.L0)
    mux_t = et.rows(D, et.MUX, 'iand', {'ETTh2': 0.0005, 'ETTm2': 0.0001}, 1)
    L += ['## ETT (ETTh2 / ETTm2 x H96 / H720, test MSE; R = mean of per-cell seed-mean r, r = 100 (MSE_A − MSE_B) / MSE_B)', '',
          '| A vs B | R % | lower pairs | verdict | energy A/B | per cell r % (seed 42, 7, 2026) |', '|---|---:|---:|---|---:|---|']
    for name, A in [('muxlin tuned', mux_t)] + [(c, qrows(c)) for c in Q]:
        for bn, B in (('Spikformer-style default', spk_d), ('Spikformer-style tuned', spk_t)):
            v = et.compare(D, A, B, et.CELLS)
            if v is None:
                raise SystemExit(f'ett incomplete: {name} vs {bn}')
            L.append(f'| {name} vs {bn} | {v["R"]:+.2f} | {v["lo"]}/{v["n"]} | {v["verdict"]} | {v["energy"]:.2f} | {"; ".join(v["cells"])} |')
    v = et.compare(D, qrows(Q[0]), mux_t, et.CELLS)
    L.append(f'| {Q[0]} vs muxlin tuned | {v["R"]:+.2f} | {v["lo"]}/{v["n"]} | {v["verdict"]} | {v["energy"]:.2f} | {"; ".join(v["cells"])} |')
    L.append('')


def main():
    L = ['# Candidate comparison (report only, 2026-11-04)', '', 'Sources: see the module docstring. Values are recomputed from RESULT lines.', '']
    cinc(L)
    ett(L)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
