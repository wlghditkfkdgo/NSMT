"""Stage 9 decisions and summary (MODI_PROJECT_LOG 2026-10-19).

--decide : from results/<lr_suite>/raw.txt and results/<crop_suite>/raw.txt write results/<crop_suite>/decision.json
           LR    = the lr in {0.001, 0.003} with the higher best-validation score of nsmt_lif seed 42 (tie -> 0.001)
           gate  = passes when best-val score(mix_lif_bidir, full) - best-val score(mix_lif_bidir, 8 tokens) >= 1 %p;
                   a missing run fails the gate (fail-closed)
--summary: results/<suite>/summary.md with per-condition means, energy and the pre-registered verdicts R1-R4.
Verdict: A better than B if mean(A-B) >= +0.5 %p and A > B in >= 2 of 3 seeds (CinC) / >= 4 of 5 seeds (SHD); worse if the
mirror holds; otherwise no clear difference; a missing seed or non-finite value -> incomplete.
"""
import argparse
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COND_LABEL = {'nsmt_lif': 'NSMT, spikeDE LIF', 'nsmt_flif': 'NSMT, f-LIF', 'nsmt_pop': 'NSMT, population f-LIF (calibrated)',
              'nsmt_lif_ce': 'NSMT, spikeDE LIF, plain CE', 'spk_ssa': 'Hippo-only, Spikformer SSA (T=4)',
              'mix_flif_bidir': 'Neo-mixer, bidirectional f-LIF', 'mix_pop_bidir': 'Neo-mixer, bidirectional population f-LIF',
              'mix_lif_bidir': 'Neo-mixer, bidirectional spikeDE LIF', 'mix_lif_bidir_ce': 'Neo-mixer, bidirectional LIF, plain CE',
              'mix_none': 'no token mixing (floor)'}


def parse(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        if line.startswith('RESULT '):
            d = dict(re.findall(r'(\S+?)=(\S+)', line))
            for k, v in list(d.items()):
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
            rows.append(d)
    return rows


def best_val(d):
    v = str(d.get('best_epoch_val', '')).split(':')
    return float(v[1]) if len(v) == 2 else float('nan')


def decide(a):
    lr_rows = parse(os.path.join(HERE, 'results', a.lr_suite, 'raw.txt'))
    crop_rows = parse(os.path.join(HERE, 'results', a.crop_suite, 'raw.txt'))
    out = {}
    for data in ('CINC2017', 'SHD'):
        cand = {}
        for r in lr_rows:
            if r.get('data') == data and r.get('cond') == 'nsmt_lif' and int(r.get('seed', -1)) == 42:
                cand[r['lr']] = best_val(r)
        lr = 0.001
        if len(cand) == 2 and all(math.isfinite(v) for v in cand.values()):
            lr = 0.003 if cand.get(0.003, -1) > cand.get(0.001, -1) else 0.001
        sc = {}
        for r in crop_rows:
            if r.get('data') == data and r.get('cond') in ('mix_lif_bidir', 'mix_none'):
                sc[(r['cond'], int(r['crop']))] = best_val(r)
        full, short = sc.get(('mix_lif_bidir', 0), float('nan')), sc.get(('mix_lif_bidir', 8), float('nan'))
        gap = full - short
        out[data] = dict(lr=lr, lr_candidates={str(k): v for k, v in cand.items()}, lr_complete=len(cand) == 2,
                         crop_scores={f'{c}@{k}': v for (c, k), v in sorted(sc.items())}, gate_gap_pp=100 * gap if math.isfinite(gap) else None,
                         gate_pass=bool(math.isfinite(gap) and gap >= 0.01))
    path = os.path.join(HERE, 'results', a.crop_suite, 'decision.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(json.dumps(out, indent=1))
    return out


def verdict(rows, a_cond, b_cond, seeds, need):
    da = {int(r['seed']): float(r['score']) for r in rows if r['cond'] == a_cond}
    db = {int(r['seed']): float(r['score']) for r in rows if r['cond'] == b_cond}
    if not all(s in da and s in db and math.isfinite(da[s]) and math.isfinite(db[s]) for s in seeds):
        return 'incomplete', [], float('nan')
    d = [100 * (da[s] - db[s]) for s in seeds]
    m = float(np.mean(d))
    if m >= 0.5 and sum(x > 0 for x in d) >= need:
        return 'better', d, m
    if m <= -0.5 and sum(x < 0 for x in d) >= need:
        return 'worse', d, m
    return 'no clear difference', d, m


def summary(a):
    R = os.path.join(HERE, 'results', a.suite)
    rows = [r for r in parse(os.path.join(R, 'raw.txt')) if int(r.get('crop', 0)) == 0]
    data = rows[0]['data'] if rows else '?'
    seeds = (42, 7, 2026) if data == 'CINC2017' else (42, 7, 2026, 123, 1024)
    need = 2 if data == 'CINC2017' else 4
    metric = 'CinC score (mean F1 of N, AF, Other)' if data == 'CINC2017' else 'test accuracy'
    L = [f'# Stage 9 `{a.suite}` ({data}) — summary', '', f'Best-validation weights on the test set; {metric}; mean ± sd over seeds {seeds} (%).', '',
         '| condition | n | score | macro-F1 (all classes) | accuracy | last-epoch score | synaptic µJ | total µJ (FFT / direct) | neocortex firing | params | time s |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    def ms(v, s=100., nd=2):
        v = np.asarray([x for x in v], dtype=float)
        return 'n/a' if len(v) == 0 or not np.all(np.isfinite(v)) else f'{s * v.mean():.{nd}f} ± {s * v.std(ddof=1) if len(v) > 1 else 0:.{nd}f}'
    energy = {}
    for c, lab in COND_LABEL.items():
        rr = [r for r in rows if r['cond'] == c]
        if not rr:
            continue
        energy[c] = np.mean([float(r['total_fft_uJ']) for r in rr])
        rho = [r.get('rho_neo') for r in rr]
        rho_s = ms(rho, 1., 4) if all(isinstance(x, float) for x in rho) else 'n/a'
        L.append(f"| {lab} | {len(rr)} | {ms([r['score'] for r in rr])} | {ms([r['macro_f1'] for r in rr])} | {ms([r['acc'] for r in rr])} | "
                 f"{ms([r['last_score'] for r in rr])} | {ms([r['syn_energy_uJ'] for r in rr], 1.)} | {ms([r['total_fft_uJ'] for r in rr], 1.)} / "
                 f"{ms([r['total_direct_uJ'] for r in rr], 1.)} | {rho_s} | {int(rr[0]['params'])} | {np.mean([r['time_s'] for r in rr]):.0f} |")
    Q = [('R1 memory requirement', c, 'mix_none') for c in ('nsmt_lif', 'nsmt_flif', 'nsmt_pop', 'spk_ssa', 'mix_flif_bidir', 'mix_pop_bidir', 'mix_lif_bidir')]
    Q += [('R2 neuron in NSMT', 'nsmt_flif', 'nsmt_lif'), ('R2 neuron in NSMT', 'nsmt_pop', 'nsmt_lif'),
          ('R3 neuron in Neo-mixer', 'mix_flif_bidir', 'mix_lif_bidir'), ('R3 neuron in Neo-mixer', 'mix_pop_bidir', 'mix_lif_bidir')]
    Q += [('R4 vs Hippo-only (Spikformer)', c, 'spk_ssa') for c in ('mix_flif_bidir', 'mix_pop_bidir', 'mix_lif_bidir', 'nsmt_lif', 'nsmt_flif', 'nsmt_pop')]
    if data == 'CINC2017':
        Q += [('R5 CAL vs plain CE (report)', 'nsmt_lif', 'nsmt_lif_ce'), ('R5 CAL vs plain CE (report)', 'mix_lif_bidir', 'mix_lif_bidir_ce')]
    L += ['', '## Pre-registered comparisons (%p, paired by seed)', '', '| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |',
          '|---|---|---|---|---|---|---|---|']
    for q, x, y in Q:
        v, d, m = verdict(rows, x, y, seeds, need)
        er = energy.get(x, float('nan')) / energy.get(y, float('nan')) if energy.get(y) else float('nan')
        crit = ''
        if q.startswith('R4') and v != 'incomplete':
            crit = 'met' if (v == 'better') or (m >= -0.5 and math.isfinite(er) and er <= 0.5) else 'not met'
        L.append(f"| {q} | {COND_LABEL.get(x, x)} | {COND_LABEL.get(y, y)} | {' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {v} | {er:.2f} | {crit} |")
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--decide', action='store_true')
    ap.add_argument('--lr_suite', default='flif-v1-rhythm-lr')
    ap.add_argument('--crop_suite', default='flif-v1-rhythm-crop')
    ap.add_argument('--suite', default=None)
    a = ap.parse_args()
    decide(a) if a.decide else summary(a)
