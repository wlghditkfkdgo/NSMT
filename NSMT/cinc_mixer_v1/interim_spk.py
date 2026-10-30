"""Interim post-hoc contrasts for the SpkMixer rerun (PROJECT_LOG 2026-10-30, exp/f-lif-pop-v3; CinC fold 0, test score %).

Not pre-registered (the pre-registered verdicts are summarize_spk.py --stage verdicts). Reads results/<suite>/raw.txt and the
Spikformer-style reference (spk_ssa) copy, and reports, over whatever seeds are present (marked per line):
  - neuron vs LIF at the same fusion (OR residual), paired by seed;
  - popg f-LIF vs popg LIF;
  - fusion contrasts for the non-LIF neurons, including incomplete ones (interim, seeds listed);
  - per-class F1 (N, AF, Other, Noisy), macro-F1 and score means, and LIF sdlin_or - spk_ssa per class;
  - run hours per neuron.
Writes results/<suite>/interim_<YYYYMMDD-HHMM>.txt (exclusive create).
"""
import argparse
import collections
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize_rhythm as S                                              # noqa: E402

SEEDS = (42, 7, 2026)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-rhythm-spk-cinc')
    ap.add_argument('--ref_suite', default='flif-v1-rhythm-cinc')
    a = ap.parse_args()
    R = os.path.join(HERE, 'results', a.suite)
    rows = [r for r in S.parse(os.path.join(R, 'raw.txt')) if int(r.get('fold', 0)) == 0 and int(r.get('crop', 0)) == 0]
    ref = [r for r in S.parse(os.path.join(HERE, 'results', a.ref_suite, 'raw.txt'))
           if r['cond'] == 'spk_ssa' and int(r.get('fold', 0)) == 0]
    if len(ref) != 3:
        raise SystemExit(f'reference spk_ssa fold 0 needs 3 runs, found {len(ref)}')

    def get(c, k='score'):
        return {int(r['seed']): 100 * float(r[k]) for r in rows if r['cond'] == c}

    def refd(k='score'):
        return {int(r['seed']): 100 * float(r[k]) for r in ref}

    def diff(x, y):
        s = [q for q in SEEDS if q in x and q in y]
        v = [x[q] - y[q] for q in s]
        return (float(np.mean(v)) if v else float('nan')), v, s

    def line(label, x, y):
        m, v, s = diff(x, y)
        return f'  {label}: {m:+.2f} per seed {[round(q, 2) for q in v]} seeds {s}'

    L = [f'# SpkMixer interim post-hoc contrasts ({datetime.datetime.now():%Y-%m-%d %H:%M} KST), RESULT rows {len(rows)}',
         'test score %, CinC fold 0, residual OR unless named; not pre-registered', '',
         'neuron vs LIF (same fusion, OR residual)']
    for n in ('flif', 'pop_lif', 'pop_flif', 'popg_lif', 'popg_flif'):
        for c in ('or', 'sdor', 'sdlin'):
            L.append(line(f'{n} {c}', get(f'spk_{n}_{c}_or'), get(f'spk_lif_{c}_or')))
    L.append('popg f-LIF vs popg LIF')
    for c in ('or', 'sdor', 'sdlin'):
        L.append(line(c, get(f'spk_popg_flif_{c}_or'), get(f'spk_popg_lif_{c}_or')))
    L.append('fusion contrasts (non-LIF neurons, interim)')
    for n in ('flif', 'pop_lif', 'pop_flif', 'popg_lif', 'popg_flif'):
        for x, y in (('sdor', 'or'), ('sdlin', 'or'), ('sdlin', 'sdor')):
            L.append(line(f'{n} {x} - {y}', get(f'spk_{n}_{x}_or'), get(f'spk_{n}_{y}_or')))
    keys = ('f1_0', 'f1_1', 'f1_2', 'f1_3', 'macro_f1', 'score')
    L.append('class F1 means: N, AF, Other, Noisy, macro-F1, score (seeds present)')
    L.append('  spk_ssa ' + str([round(float(np.mean(list(refd(k).values()))), 2) for k in keys]))
    for c in sorted({r['cond'] for r in rows}):
        L.append(f'  {c} (n={len(get(c))}) ' + str([round(float(np.mean(list(get(c, k).values()))), 2) for k in keys]))
    L.append('LIF sdlin_or - spk_ssa per class')
    for k in keys:
        L.append(line(k, get('spk_lif_sdlin_or', k), refd(k)))
    t = collections.defaultdict(list)
    for r in rows:
        p = r['cond'].split('_')
        n = '_'.join(p[1:3]) if p[1] in ('pop', 'popg') else p[1]
        t[n].append(float(r['time_s']) / 3600)
    L.append('run hours by neuron')
    for n, v in t.items():
        L.append(f'  {n}: {min(v):.1f}-{max(v):.1f} h, n={len(v)}')
    out = os.path.join(R, f'interim_{datetime.datetime.now():%Y%m%d-%H%M}.txt')
    with open(out, 'x') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('wrote', out)


if __name__ == '__main__':
    main()
