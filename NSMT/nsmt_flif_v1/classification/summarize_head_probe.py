"""MODI 2026-11-03: summary of the head readout diagnostic (head_probe.py; docs/MODI_PROJECT_LOG.md 2026-11-03 15:36, rules fixed
there). Reads results/flif-v1-head-probe/*.json (full runs, not --check_only); writes results/flif-v1-head-probe-summary.md.
Models nsmt4q_pop_lif / nsmt4q_pop_flif, seeds 42 / 7 / 2026; MIT-BIH (6 checkpoints), ETTh2 / ETTm2 x H96 / H720 (24).
Rules (fail-closed: a missing / duplicated checkpoint, a non-finite value -> incomplete):
  MIT-BIH, per model (3 seeds): d = test macro-F1(R_k) - test macro-F1(R0), %p; better = mean >= +0.5 and >= 2 of 3 seeds > 0;
    worse = mean <= -0.5 and >= 2 of 3 < 0; else no clear difference.
  ETT, per model (4 cells x 3 seeds = 12 pairs): r = test MSE(R_k) / test MSE(R0) - 1; better = mean <= -1 % and >= 8 of 12 < 0;
    worse = mirror; per cell (3 seeds): mean <= -1 % and >= 2 of 3.
  Conclusion per task and readout: better for both models -> readout bottleneck there; for one -> neuron-dependent (weak); else none.
  Report only: R0 probe - trained head, R6 - R0 probe, R6 - trained head, R7 - R0 probe, MIT-BIH per-class F1, lam at the cap,
  ETTm2 query models (trained head) vs earlier ETTm2 models.
usage: python summarize_head_probe.py
"""
import glob
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.join(HERE, '..', 'forecasting', 'results')
SEEDS = (42, 7, 2026)
MODELS = ('nsmt4q_pop_lif', 'nsmt4q_pop_flif')
NAMES = {'nsmt4q_pop_lif': 'query pop LIF', 'nsmt4q_pop_flif': 'query pop f-LIF'}
CELLS = (('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720'))
CMP_CLS = ('R1', 'R2', 'R3', 'R4', 'R5')
CMP_ETT = ('R2', 'R3', 'R4', 'R5')
DESC = {'R0': 'current head input', 'R1': 'per token', 'R2': 'per T_h step', 'R3': '+ embedding spikes', 'R4': '+ Neocortex output',
        'R5': '+ membrane potentials', 'R6': 'raw input', 'R7': 'all sources per token'}
LINEAR_BEST = {('ETTh2', '96'): 0.291515, ('ETTh2', '720'): 0.420805, ('ETTm2', '96'): 0.182854, ('ETTm2', '720'): 0.407828}
# ^ 10-18 linear baseline (window normalisation + Linear(96 -> H)), test MSE of the best-validation weights, mean of 3 seeds:
#   results/flif-v1-s4/summary.md "test MSE" table (raw.txt 'mse' there is the last-epoch weights)


def fin(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def load():
    by = {}
    for p in sorted(glob.glob(os.path.join(HERE, 'results', 'flif-v1-head-probe', '*.json'))):
        j = json.load(open(p))
        if j.get('check_only'):
            continue
        key = (j['task'], j['cond'], j['data'], j['pred_len'], j['seed'])
        by.setdefault(key, []).append(j)
    return by


def get(by, task, cond, data, pl, seed):
    v = by.get((task, cond, data, pl, seed), [])
    return v[0] if len(v) == 1 else None


def verdict(d, thr, need, lower_better=False):
    m = float(np.mean(d))
    s = -1 if lower_better else 1
    good = sum(s * z > 0 for z in d) >= need and s * m >= thr
    bad = sum(s * z < 0 for z in d) >= need and s * m <= -thr
    return m, 'better' if good else 'worse' if bad else 'no clear difference'


def main():
    by = load()
    L = ['# Head readout diagnostic (frozen query-form bodies, new linear readouts) — summary', '',
         'See the module docstring and head_probe.py for sources, probes and rules. Probes R_k are compared with the R0 probe of the '
         'same checkpoint.', '']
    concl = {}
    # ------------------------------------------------ MIT-BIH
    L += ['## MIT-BIH (test macro-F1 %, 80 / 20 --mitbih_val checkpoints, lr 0.01)', '',
          '| model | readout | dim | seed 42 | seed 7 | seed 2026 | mean | lam (per seed) |', '|---|---|---|---|---|---|---|---|']
    J = {(c, s): get(by, 'mitbih', c, 'MITBIH', None, s) for c in MODELS for s in SEEDS}
    for c in MODELS:
        for r in ('head',) + tuple(f'R{i}' for i in range(8)):
            v = [J[(c, s)]['readouts'].get(r) if J[(c, s)] else None for s in SEEDS]
            if any(x is None or not fin(x.get('test_f1')) for x in v):
                L.append(f'| {NAMES[c]} | {r} | | incomplete | | | | |'); continue
            lam = ' / '.join(f"{x['lam']:g}{'*' if x.get('lam_at_cap') else ''}" for x in v) if r != 'head' else 'trained head'
            L.append(f"| {NAMES[c]} | {r} {DESC.get(r, '')} | {v[0].get('dim', 384)} | " + ' | '.join(f"{100 * x['test_f1']:.2f}" for x in v)
                     + f" | {100 * np.mean([x['test_f1'] for x in v]):.2f} | {lam} |")
    L += ['', '(* = lam at the cap of the grid.)', '', '### verdicts: R_k probe − R0 probe (%p)', '',
          '| readout | model | per seed (42 / 7 / 2026) | mean | > 0 | verdict |', '|---|---|---|---|---|---|']
    for r in CMP_CLS:
        for c in MODELS:
            v = [(J[(c, s)]['readouts'].get(r), J[(c, s)]['readouts'].get('R0')) if J[(c, s)] else (None, None) for s in SEEDS]
            if any(a is None or b is None or not fin(a.get('test_f1')) or not fin(b.get('test_f1')) for a, b in v):
                L.append(f'| {r} | {NAMES[c]} | incomplete | | | incomplete |'); concl.setdefault(('mitbih', r), []).append('incomplete'); continue
            d = [100 * (a['test_f1'] - b['test_f1']) for a, b in v]
            m, vd = verdict(d, 0.5, 2)
            concl.setdefault(('mitbih', r), []).append(vd)
            L.append(f"| {r} {DESC[r]} | {NAMES[c]} | {' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {sum(z > 0 for z in d)}/3 | {vd} |")
    L += ['', '### report only (%p, mean of 3 seeds)', '',
          '| model | R0 probe − trained head | R6 − R0 probe | R6 − trained head | R7 − R0 probe |', '|---|---|---|---|---|']
    for c in MODELS:
        if any(J[(c, s)] is None for s in SEEDS):
            L.append(f'| {NAMES[c]} | incomplete | | | |'); continue
        g = lambda r: np.array([J[(c, s)]['readouts'][r]['test_f1'] for s in SEEDS])
        L.append(f"| {NAMES[c]} | {100 * np.mean(g('R0') - g('head')):+.2f} | {100 * np.mean(g('R6') - g('R0')):+.2f} | "
                 f"{100 * np.mean(g('R6') - g('head')):+.2f} | {100 * np.mean(g('R7') - g('R0')):+.2f} |")
    L += ['', 'Per-class test F1 (%, mean of 3 seeds; classes N / S / V / F / Q):', '',
          '| model | readout | N | S | V | F | Q |', '|---|---|---|---|---|---|---|']
    for c in MODELS:
        if any(J[(c, s)] is None for s in SEEDS):
            continue
        for r in ('head',) + tuple(f'R{i}' for i in range(8)):
            pc = np.mean([J[(c, s)]['readouts'][r]['test_f1_cls'] for s in SEEDS], axis=0)
            L.append(f'| {NAMES[c]} | {r} | ' + ' | '.join(f'{100 * z:.2f}' for z in pc) + ' |')
    # ------------------------------------------------ ETT
    L += ['', '## ETT (test MSE, scaled data, best-validation checkpoints)', '']
    E = {(c, ds, pl, s): get(by, 'ett', c, ds, pl, s) for c in MODELS for ds, pl in CELLS for s in SEEDS}
    for ds, pl in CELLS:
        L += [f'### {ds} H{pl}', '', '| model | readout | dim | seed 42 | seed 7 | seed 2026 | mean | lam (per seed) |', '|---|---|---|---|---|---|---|---|']
        for c in MODELS:
            for r in ('head', 'R0', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'):
                v = [E[(c, ds, pl, s)]['readouts'].get(r) if E[(c, ds, pl, s)] else None for s in SEEDS]
                if any(x is None or not fin(x.get('test_mse')) for x in v):
                    L.append(f'| {NAMES[c]} | {r} | | incomplete | | | | |'); continue
                lam = ' / '.join(f"{x['lam']:g}{'*' if x.get('lam_at_cap') else ''}" for x in v) if r != 'head' else 'trained head'
                L.append(f"| {NAMES[c]} | {r} {DESC.get(r, '')} | {v[0].get('dim', '')} | " + ' | '.join(f"{x['test_mse']:.6f}" for x in v)
                         + f" | {np.mean([x['test_mse'] for x in v]):.6f} | {lam} |")
        L.append('')
    L += ['### verdicts: R_k probe / R0 probe − 1 (%)', '',
          '| readout | model | pooled 12 pairs | < 0 | verdict | per cell (mean, verdict) |', '|---|---|---|---|---|---|']
    for r in CMP_ETT:
        for c in MODELS:
            pairs = [(E[(c, ds, pl, s)], ds, pl) for ds, pl in CELLS for s in SEEDS]
            if any(j is None or not fin(j['readouts'].get(r, {}).get('test_mse')) or not fin(j['readouts']['R0'].get('test_mse'))
                   for j, _, _ in pairs):
                L.append(f'| {r} | {NAMES[c]} | incomplete | | incomplete | |'); concl.setdefault(('ett', r), []).append('incomplete'); continue
            d = [100 * (j['readouts'][r]['test_mse'] / j['readouts']['R0']['test_mse'] - 1) for j, _, _ in pairs]
            m, vd = verdict(d, 1.0, 8, lower_better=True)
            concl.setdefault(('ett', r), []).append(vd)
            pc = []
            for i, (ds, pl) in enumerate(CELLS):
                mc, vc = verdict(d[3 * i:3 * i + 3], 1.0, 2, lower_better=True)
                pc.append(f'{ds} H{pl} {mc:+.2f} {vc}')
            L.append(f"| {r} {DESC[r]} | {NAMES[c]} | {m:+.2f} | {sum(z < 0 for z in d)}/12 | {vd} | {'; '.join(pc)} |")
    L += ['', '### report only (%, mean over 3 seeds per cell)', '',
          '| model | cell | R0 probe / trained head − 1 | R6 / R0 probe − 1 | R6 / trained head − 1 | R7 / R0 probe − 1 |', '|---|---|---|---|---|---|']
    for c in MODELS:
        for ds, pl in CELLS:
            js = [E[(c, ds, pl, s)] for s in SEEDS]
            if any(j is None for j in js):
                L.append(f'| {NAMES[c]} | {ds} H{pl} | incomplete | | | |'); continue
            g = lambda r: np.array([j['readouts'][r]['test_mse'] for j in js])
            f = lambda a, b: f'{100 * np.mean(g(a) / g(b) - 1):+.2f}'
            L.append(f"| {NAMES[c]} | {ds} H{pl} | {f('R0', 'head')} | {f('R6', 'R0')} | {f('R6', 'head')} | {f('R7', 'R0')} |")
    # ------------------------------------------------ conclusions
    L += ['', '## conclusions (rule of the pre-registration)', '', '| task | readout | pop LIF | pop f-LIF | conclusion |', '|---|---|---|---|---|']
    for task, rs in (('mitbih', CMP_CLS), ('ett', CMP_ETT)):
        for r in rs:
            v = concl.get((task, r), ['incomplete', 'incomplete'])
            nb = sum(x == 'better' for x in v)
            cc = 'incomplete' if 'incomplete' in v else 'readout bottleneck' if nb == 2 else 'neuron-dependent (weak)' if nb == 1 else 'no evidence'
            L.append(f"| {'MIT-BIH' if task == 'mitbih' else 'ETT'} | {r} {DESC[r]} | {v[0]} | {v[1]} | {cc} |")
    # ------------------------------------------------ ETTm2 report: query models vs earlier ETTm2 models (trained heads)
    L += ['', '## report: ETT test MSE of the trained models (mean of 3 seeds, best-validation weights)', '',
          '| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | source |', '|---|---|---|---|---|---|']
    srcs = [('Spikformer-style', 'spk_ssa', 'flif-v1-gen-ett'), ('B form pop LIF', 'spkneo_pop_lif', 'flif-v1-ett-dual'),
            ('B form pop f-LIF', 'spkneo_pop_flif', 'flif-v1-ett-dual'), ('replace pop LIF', 'nsmt4_pop_lif', 'flif-v1-ett-dual'),
            ('replace pop f-LIF', 'nsmt4_pop_flif', 'flif-v1-ett-dual'), ('query pop LIF', 'nsmt4q_pop_lif', 'flif-v1-ett-query + -m2'),
            ('query pop f-LIF', 'nsmt4q_pop_flif', 'flif-v1-ett-query + -m2')]
    rows = {}
    for suite in ('flif-v1-gen-ett', 'flif-v1-ett-dual', 'flif-v1-ett-query', 'flif-v1-ett-query-m2'):
        p = os.path.join(FC, suite, 'raw.txt')
        if not os.path.exists(p):
            continue
        for ln in open(p):
            if ln.startswith('RESULT'):
                g = dict(re.findall(r'(\w+)=(\S+)', ln))
                rows.setdefault((g['cond'], g['ds'], g['pl']), []).append((int(g['seed']), float(g['mse'])))
    for name, cond, src in srcs:
        cells = []
        for ds, pl in CELLS:
            v = rows.get((cond, ds, pl), [])
            ok = sorted(s for s, _ in v) == sorted(SEEDS) and all(fin(x) for _, x in v)
            cells.append(f'{np.mean([x for _, x in v]):.4f}' if ok else 'incomplete')
        L.append(f'| {name} | ' + ' | '.join(cells) + f' | {src} |')
    L.append('| linear baseline (10-18) | ' + ' | '.join(f'{LINEAR_BEST[c]:.4f}' for c in CELLS) + ' | flif-v1-s4/summary.md |')
    out = os.path.join(HERE, 'results', 'flif-v1-head-probe-summary.md')
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
