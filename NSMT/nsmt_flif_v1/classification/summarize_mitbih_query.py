"""MODI 2026-11-01: the query form (Neocortex as the attention query, hippocampus keys / values) with population neurons on MIT-BIH
(docs/MODI_PROJECT_LOG.md 2026-11-01, user decision after the CinC 5-fold check). Kachuee 5-class heartbeats, the ECG framework
(30 epochs, CAL-weighted CE, last-epoch weights), seeds 42 / 7 / 2026; metric = test macro-F1 (%).
Stage 1 (default settings, full training file): suite flif-v1-mitbih-query (nsmt4q_pop_lif, nsmt4q_pop_flif); comparisons from
  flif-v1-gen-mitbih (spk_ssa) and flif-v1-mitbih-dual (nsmt4_pop_*, spkneo_pop_*).
  M1 (main, criterion 2) query population n - Spikformer-style; M2 query - replace (nsmt4_pop_n); M3 query - append / B form
  (spkneo_pop_n); report: query population f-LIF - query population LIF.
Stage 2 (--tune; run only if neither query population is 'better' than Spikformer-style in M1): suite flif-v1-mitbih-query-tune,
  peak learning rate {0.0025, 0.005, 0.01 (default), 0.02} x {nsmt4q_pop_lif, nsmt4q_pop_flif, spk_ssa}, all with --mitbih_val
  (fold 0 of a stratified 5-fold split of the training file held out as validation, 80 / 20 training). Selection per model = the
  learning rate with the highest mean last-epoch validation macro-F1 over the 3 seeds. T1 selected - default learning rate (test,
  same split); T2 query population n at its selected learning rate - Spikformer-style at its selected learning rate (criterion 2);
  report: T2 at the default learning rate, and every learning rate side by side (validation and test).
Rules (fail-closed; a missing / duplicated / non-finite value -> incomplete): better = mean A - B >= +0.5 %p and A > B in >= 2 of 3
  seeds, worse = mirror; criterion (2) = better, or mean >= -0.5 %p with energy A/B <= 0.5.
usage: python summarize_mitbih_query.py [--tune]  ->  results/flif-v1-mitbih-query[-tune]-summary.md
"""
import argparse
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 2026)
LRS = (0.0025, 0.005, 0.01, 0.02)
DEFAULT_LR = 0.01


def parse(suite):
    out = []
    p = os.path.join(HERE, 'results', suite, 'raw.txt')
    if not os.path.exists(p):
        return out
    for line in open(p):
        if not line.startswith('RESULT'):
            continue
        d = dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)
        r = {'cond': d['cond'], 'seed': int(d['seed']), 'lr': float(d.get('lr', DEFAULT_LR)), 'val': int(d.get('val', 0))}
        for k in ('f1', 'f1_0', 'f1_1', 'f1_2', 'f1_3', 'f1_4', 'overall_acc', 'total_fft_uJ', 'val_f1'):
            try:
                r[k] = float(d[k])
            except (KeyError, ValueError):
                r[k] = float('nan')
        out.append(r)
    return out


def fin(x):
    return isinstance(x, float) and math.isfinite(x)


class Table:
    def __init__(self):
        self.by = {}

    def add(self, tag, suite, cond, lr=DEFAULT_LR, val=0):
        for r in parse(suite):
            if r['cond'] == cond and math.isclose(r['lr'], lr) and r['val'] == val:
                self.by.setdefault((tag, r['seed']), []).append(r)

    def get(self, tag, seed, key='f1'):
        v = self.by.get((tag, seed), [])
        return v[0] if len(v) == 1 and fin(v[0][key]) else None

    def seeds(self, tag, key='f1'):
        rr = [self.get(tag, s, key) for s in SEEDS]
        return None if any(r is None for r in rr) else rr


def verdict(d):
    m = float(np.mean(d))
    return m, ('better' if m >= 0.5 and sum(z > 0 for z in d) >= 2 else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= 2 else 'no clear difference')


def paired(T, rows, crit=False):
    out = ['| comparison | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B |' + (' criterion (2) |' if crit else ''),
           '|---|---|---|---|---|---|---|---|---|' + ('---|' if crit else '')]
    for q, a, b in rows:
        A, B = T.seeds(a), T.seeds(b)
        if A is None or B is None:
            out.append(f'| {q} | {a} | {b} | | | | | incomplete | |' + (' |' if crit else '')); continue
        d = [100 * (x['f1'] - y['f1']) for x, y in zip(A, B)]; m, v = verdict(d)
        er = float(np.mean([x['total_fft_uJ'] for x in A]) / np.mean([y['total_fft_uJ'] for y in B]))
        c = ('met' if v == 'better' or (m >= -0.5 and er <= 0.5) else 'not met') if crit else ''
        out.append(f"| {q} | {a} | {b} | {100 * np.mean([x['f1'] for x in A]):.2f} | {100 * np.mean([y['f1'] for y in B]):.2f} | "
                   f"{' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {v} | {er:.2f} |" + (f' {c} |' if crit else ''))
    return out


def means(T, tags):
    out = ['| model | macro-F1 (mean ± sd) | accuracy | F1 N / S / V / F / Q | energy µJ |', '|---|---|---|---|---|']
    for t, name in tags:
        rr = T.seeds(t)
        if rr is None:
            out.append(f'| {name} | incomplete | | | |'); continue
        f = lambda k: 100 * np.array([r[k] for r in rr])
        out.append(f"| {name} | {f('f1').mean():.2f} ± {f('f1').std(ddof=1):.2f} | {f('overall_acc').mean():.2f} | "
                   + ' / '.join(f"{f(f'f1_{i}').mean():.2f}" for i in range(5))
                   + f" | {np.mean([r['total_fft_uJ'] for r in rr]):.2f} |")
    return out


def stage1():
    T = Table()
    for n in ('pop_lif', 'pop_flif'):
        T.add(f'q_{n}', 'flif-v1-mitbih-query', f'nsmt4q_{n}')
        T.add(f'r_{n}', 'flif-v1-mitbih-dual', f'nsmt4_{n}')
        T.add(f'b_{n}', 'flif-v1-mitbih-dual', f'spkneo_{n}')
    T.add('spk', 'flif-v1-gen-mitbih', 'spk_ssa')
    L = ['# Query form with population neurons on MIT-BIH, stage 1 — summary', '',
         'See the module docstring for sources and rules. Test macro-F1 (%), last-epoch weights, seeds 42 / 7 / 2026.', '']
    L += means(T, [('spk', 'Spikformer-style (spk_ssa)'), ('q_pop_lif', 'query population LIF (nsmt4q_pop_lif)'),
                   ('q_pop_flif', 'query population f-LIF (nsmt4q_pop_flif)'), ('r_pop_lif', 'replace population LIF (nsmt4_pop_lif)'),
                   ('r_pop_flif', 'replace population f-LIF (nsmt4_pop_flif)'), ('b_pop_lif', 'B form population LIF (spkneo_pop_lif)'),
                   ('b_pop_flif', 'B form population f-LIF (spkneo_pop_flif)')])
    L += ['', '## M1 (main) query population − Spikformer-style', '']
    L += paired(T, [(f'M1 query {n} − Spikformer-style', f'q_{n}', 'spk') for n in ('pop_lif', 'pop_flif')], crit=True)
    L += ['', '## M2 query − replace ([NSMT T=4]), M3 query − append (B form), report', '']
    L += paired(T, [(f'M2 query − replace, {n}', f'q_{n}', f'r_{n}') for n in ('pop_lif', 'pop_flif')]
                + [(f'M3 query − B form, {n}', f'q_{n}', f'b_{n}') for n in ('pop_lif', 'pop_flif')]
                + [('report: query population f-LIF − query population LIF', 'q_pop_flif', 'q_pop_lif')])
    m1 = []
    for n in ('pop_lif', 'pop_flif'):
        A, B = T.seeds(f'q_{n}'), T.seeds('spk')
        m1.append(None if A is None or B is None else verdict([100 * (x['f1'] - y['f1']) for x, y in zip(A, B)])[1])
    trig = 'incomplete' if None in m1 else ('no (a query population is better)' if 'better' in m1 else 'yes (run stage 2)')
    L += ['', f'**Stage 2 trigger (neither query population better than Spikformer-style in M1):** {trig}', '']
    return L, 'flif-v1-mitbih-query-summary.md'


def stage2():
    T = Table()
    models = [('q_pop_lif', 'nsmt4q_pop_lif', 'query population LIF'), ('q_pop_flif', 'nsmt4q_pop_flif', 'query population f-LIF'),
              ('spk', 'spk_ssa', 'Spikformer-style')]
    for tag, cond, _ in models:
        for lr in LRS:
            T.add(f'{tag}@{lr:g}', 'flif-v1-mitbih-query-tune', cond, lr=lr, val=1)
    L = ['# Query form with population neurons on MIT-BIH, stage 2 (learning rate) — summary', '',
         'See the module docstring for sources and rules. All runs train on 80 % of the training file (--mitbih_val); validation = the',
         'held-out 20 % (last epoch, selection only); test = the 30 % test file (last-epoch weights). Default learning rate marked *.', '',
         '## every learning rate: validation macro-F1 / test macro-F1 (mean over seeds, %)', '',
         '| model | ' + ' | '.join(f'{lr:g}' + ('*' if lr == DEFAULT_LR else '') for lr in LRS) + ' | selected |', '|---|' + '---|' * (len(LRS) + 1)]
    sel = {}
    for tag, _, name in models:
        cells, vm = [], {}
        for lr in LRS:
            rr = T.seeds(f'{tag}@{lr:g}', 'val_f1')
            if rr is None or T.seeds(f'{tag}@{lr:g}') is None:
                cells.append('incomplete'); continue
            vm[lr] = float(np.mean([r['val_f1'] for r in rr]))
            cells.append(f"{100 * vm[lr]:.2f} / {100 * np.mean([r['f1'] for r in rr]):.2f}")
        sel[tag] = max(vm, key=lambda k: vm[k]) if len(vm) == len(LRS) else None
        L.append(f'| {name} | ' + ' | '.join(cells) + f" | {'incomplete' if sel[tag] is None else f'{sel[tag]:g}'} |")
    L += ['', '## energy µJ (mean over seeds)', '', '| model | ' + ' | '.join(f'{lr:g}' for lr in LRS) + ' |', '|---|' + '---|' * len(LRS)]
    for tag, _, name in models:
        L.append(f'| {name} | ' + ' | '.join('incomplete' if T.seeds(f'{tag}@{lr:g}') is None else f"{np.mean([r['total_fft_uJ'] for r in T.seeds(f'{tag}@{lr:g}')]):.2f}" for lr in LRS) + ' |')
    if any(v is None for v in sel.values()):
        L += ['', '**T1 / T2:** incomplete (a selection is missing)', '']
        return L, 'flif-v1-mitbih-query-tune-summary.md'
    L += ['', '## T1 selected − default learning rate (test)', '']
    L += paired(T, [(f'T1 {name} @{sel[tag]:g} − @{DEFAULT_LR:g}', f'{tag}@{sel[tag]:g}', f'{tag}@{DEFAULT_LR:g}')
                    for tag, _, name in models if sel[tag] != DEFAULT_LR])
    if all(sel[tag] == DEFAULT_LR for tag, _, _ in models):
        L.append('| (every model selected the default) | | | | | | | | |')
    L += ['', '## T2 query population − Spikformer-style, each at its selected learning rate (main) and at the default (report)', '']
    L += paired(T, [(f"T2 {name} @{sel[tag]:g} − Spikformer-style @{sel['spk']:g}", f'{tag}@{sel[tag]:g}', f"spk@{sel['spk']:g}")
                    for tag, _, name in models[:2]]
                + [(f'report: {name} − Spikformer-style, both @{DEFAULT_LR:g}', f'{tag}@{DEFAULT_LR:g}', f'spk@{DEFAULT_LR:g}')
                   for tag, _, name in models[:2]], crit=True)
    return L, 'flif-v1-mitbih-query-tune-summary.md'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tune', action='store_true')
    L, name = stage2() if ap.parse_args().tune else stage1()
    out = os.path.join(HERE, 'results', name)
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
