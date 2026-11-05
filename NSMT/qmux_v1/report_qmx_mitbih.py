"""Independent recomputation and report-only tables for the MIT-BIH stage of the query-form bidirectional Neocortex test (2026-11-05, exp/f-lif-pop-v3).
No training and nothing read from a checkpoint. Own parser and own arithmetic (no import from summarize_qmx.py), so a summarizer bug shows up as a mismatch.
Fail closed: a missing run, a duplicate key, a non-finite value, a run that did not complete 30 epochs or tested on another set size, or a value that differs from
results/qmx-mitbih/summary.md stops the script with exit 2 and writes nothing. It also re-derives the adoption table from both summaries.
Report only (no verdict): class F1 changes, firing rates, seed spread, the highest-F1 condition against the references, the ETT lowest-MSE condition on MIT-BIH.
usage: python -B report_qmx_mitbih.py   ->   results/qmx-mitbih/report_extra.md
"""
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1/classification/results'
CM = os.path.join(HERE, '..', 'cinc_mixer_v1', 'results')
SUITE = os.path.join(HERE, 'results', 'qmx-mitbih')
SEEDS = (42, 7, 2026)
NEURONS = ('lif', 'flif', 'pop_lif', 'pop_flif')
RES = ('or', 'iand')
ETT_BEST = 'qmx_pop_flif_or_or'     # lowest mean test MSE of the ETT stage (results/qmx-ett/report_extra.md)
CLASSES = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}


def stop(msg):
    print('STOP:', msg)
    sys.exit(2)


def read_result(path, used=lambda c: True):
    out = []
    for line in open(path):
        if line.startswith('SKIP'):
            m = re.search(r'\bcond=(\S+)', line)
            if m is None or used(m[1]):
                stop(f'SKIP line of a used condition in {path}: {line[:120]}')
        if line.startswith('RESULT'):
            r = dict(tok.split('=', 1) for tok in line.split()[1:] if '=' in tok)
            if used(r['cond']):
                out.append(r)
    return out


def mean(v):
    return sum(v) / len(v)


def num(r, key):
    x = float(r[key])
    if not math.isfinite(x):
        stop(f'non-finite {key} in {r.get("cond")} seed {r.get("seed")}')
    return x


# ---- runs
PLAIN = [f'nsmt4q_{n}' for n in NEURONS]
NEW = [f'qmx_{n}_{f}_{r}' for n in NEURONS for f in ('muxlin', 'or') for r in RES]
own = read_result(os.path.join(SUITE, 'raw.txt'))
if len(own) != 54:
    stop(f'{len(own)} RESULT lines in this suite, expected 54')
for r in own:
    if int(r['epochs_run']) != 30 or int(r['n_test']) != 32833 or abs(float(r['lr']) - 0.01) > 1e-12:
        stop(f'run differs from the setting: {r["cond"]} seed {r["seed"]}')
is_plain = lambda c: c in PLAIN
Q = {}
for r in read_result(os.path.join(MAIN, 'flif-v1-mitbih-query', 'raw.txt'), is_plain) + own:
    k = (r['cond'], int(r['seed']))
    if k in Q:
        stop(f'duplicate run {k}')
    r['E'] = num(r, 'total_fft_uJ')
    Q[k] = r
for c in PLAIN + NEW:
    for s in SEEDS:
        if (c, s) not in Q:
            stop(f'missing run {c} seed {s}')
        num(Q[(c, s)], 'f1')

# ---- references
DFLT = {}
for r in read_result(os.path.join(CM, 'flif-v1-mitbih-spkmux', 'baseline_spk_ssa_raw.txt')):
    DFLT[int(r['seed'])] = dict(r, E=num(r, 'total_fft_uJ'))
tune = os.path.join(CM, 'flif-v1-mitbih-spkmux-tune')
en = {(e['tag'], int(e['seed'])): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(tune, 'energy_recount.csv')))}
TUNED = {}
for r in read_result(os.path.join(tune, 'raw.txt'), lambda c: c == 'spk_ssa'):
    if abs(float(r['lr']) - 0.005) < 1e-12 and int(r['val_fold']) == -1:
        TUNED[int(r['seed'])] = dict(r, E=en[(r['tag'], int(r['seed']))])
NSMT = {}
for r in read_result(os.path.join(MAIN, 'flif-v1-cls', 'raw.txt'), lambda c: c == 'base'):
    NSMT[int(r['seed'])] = dict(r, E=num(r, 'total_fft_uJ'))
for T, name in ((DFLT, 'Spikformer-style default'), (TUNED, 'Spikformer-style tuned'), (NSMT, 'NSMT rerun')):
    for s in SEEDS:
        if s not in T:
            stop(f'missing {name} seed {s}')


def q(c):
    return [Q[(c, s)] for s in SEEDS]


def ref(T):
    return [T[s] for s in SEEDS]


def compare(ra, rb):
    d = [100 * (num(a, 'f1') - num(b, 'f1')) for a, b in zip(ra, rb)]
    m = mean(d)
    v = 'better' if m >= 0.5 and sum(z > 0 for z in d) >= 2 else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= 2 else 'no clear difference'
    return dict(m=m, d=d, verdict=v, energy=mean([a['E'] for a in ra]) / mean([b['E'] for b in rb]))


# ---- agreement with summary.md, every pre-registered row
SUMMARY = {}
for line in open(os.path.join(SUITE, 'summary.md')):
    m = re.match(r'\| (V[1-4]) (\w+) (OR|IAND): [^|]+\| ([+-][\d.]+) \| ([^|]+) \| ([a-z ]+) \| ([\d.]+) \|', line)
    if m:
        SUMMARY[(m[1], m[2], m[3].lower())] = (float(m[4]), [float(z) for z in m[5].split(',')], m[6], float(m[7]))
if len(SUMMARY) != 32:
    stop(f'{len(SUMMARY)} pre-registered rows in summary.md, expected 32')
COMP = {}
for n in NEURONS:
    for r in RES:
        mx, o, p = q(f'qmx_{n}_muxlin_{r}'), q(f'qmx_{n}_or_{r}'), q(f'nsmt4q_{n}')
        COMP[('V1', n, r)], COMP[('V2', n, r)] = compare(mx, p), compare(mx, o)
        COMP[('V3', n, r)], COMP[('V4', n, r)] = compare(o, p), compare(mx, ref(DFLT))
for k, c in COMP.items():
    m_, d_, v_, e_ = SUMMARY[k]
    if abs(m_ - c['m']) > 0.0051 or any(abs(a - b) > 0.0051 for a, b in zip(d_, c['d'])) or v_ != c['verdict'] or abs(e_ - c['energy']) > 0.0051:
        stop(f'summary.md differs at {k}: {SUMMARY[k]} vs {c}')
print('32 pre-registered rows of summary.md reproduced (mean, per-seed values, verdict, energy)')

# ---- adoption, from the two summaries
ETT = {}
for line in open(os.path.join(HERE, 'results', 'qmx-ett', 'summary.md')):
    m = re.match(r'\| V1 (\w+) (OR|IAND): [^|]+\| [+-][\d.]+ \| \d+/12 \| ([a-z ]+) \|', line)
    if m:
        ETT[(m[1], m[2].lower())] = m[3]
if len(ETT) != 8:
    stop(f'{len(ETT)} ETT V1 rows, expected 8')
adopted = [k for k, v in ETT.items() if v == 'better' and COMP[('V1',) + k]['verdict'] != 'worse']
rows = re.findall(r'^\| \w+ \| (?:OR|IAND) \| [^|]+ \| [^|]+ \| (yes|no) \|$', open(os.path.join(SUITE, 'summary.md')).read(), flags=re.M)
if len(rows) != 8 or (adopted and 'yes' not in rows) or (not adopted and 'yes' in rows):
    stop(f'adoption table of summary.md disagrees: {rows} vs {adopted}')
print(f'adoption table reproduced: {len(adopted)} of 8 adopted')

# ---- report
L = ['# MIT-BIH stage: independent recomputation and report-only tables', '',
     'The 32 pre-registered rows and the adoption table of summary.md were reproduced by separate code. All 54 runs: 30 epochs, lr 0.01, 32833 test samples.',
     'Everything below is report only. Plain LIF and f-LIF controls come from this suite, plain population controls from flif-v1-mitbih-query.', '']

L += ['## V1 by class: muxlin minus plain query, F1 %p (seed mean)', '', '| neuron | residual | macro | N | S | V | F | Q |', '|---|---|---:|---:|---:|---:|---:|---:|']
for n in NEURONS:
    for r in RES:
        a, b = q(f'qmx_{n}_muxlin_{r}'), q(f'nsmt4q_{n}')
        cl = [mean([100 * (num(x, f'f1_{i}') - num(y, f'f1_{i}')) for x, y in zip(a, b)]) for i in CLASSES.values()]
        L.append(f"| {n} | {r.upper()} | {COMP[('V1', n, r)]['m']:+.2f} | " + ' | '.join(f'{z:+.2f}' for z in cl) + ' |')

L += ['', '## Firing rates, energy, seed range (mean over 3 seeds)', '', '| condition | rho_neo | rho_hippo | energy µJ | lowest and highest seed macro-F1 % |', '|---|---:|---:|---:|---|']
for n in NEURONS:
    for c in [f'nsmt4q_{n}'] + [f'qmx_{n}_{f}_{r}' for f in ('muxlin', 'or') for r in RES]:
        rr = q(c)
        f = [100 * num(x, 'f1') for x in rr]
        L.append(f"| {c} | {mean([num(x, 'rho_neo') for x in rr]):.4f} | {mean([num(x, 'rho_hippo') for x in rr]):.4f} | {mean([x['E'] for x in rr]):.2f} | {min(f):.2f}, {max(f):.2f} |")


def cmp_rows(name, c):
    n_c = re.match(r'qmx_(\w+?)_(muxlin|or)_', c)[1]
    out = [f'', f'## {name}: {c} {100 * mean([num(x, "f1") for x in q(c)]):.2f}', '', '| comparison | mean %p | seed 42, 7, 2026 | verdict | energy A/B |', '|---|---:|---|---|---:|']
    for lab, rb in ((f'vs nsmt4q_{n_c}', q(f'nsmt4q_{n_c}')), ('vs Spikformer-style (default)', ref(DFLT)), ('vs Spikformer-style (tuned)', ref(TUNED)), ('vs NSMT rerun', ref(NSMT))):
        k = compare(q(c), rb)
        out.append(f"| {lab} | {k['m']:+.2f} | " + ', '.join(f'{z:+.2f}' for z in k['d']) + f" | {k['verdict']} | {k['energy']:.2f} |")
    return out


f1 = {c: mean([num(x, 'f1') for x in q(c)]) for c in NEW + PLAIN}
order = sorted(f1, key=f1.get, reverse=True)
L += cmp_rows('Highest mean macro-F1 (chosen after seeing the test values, so biased)', next(c for c in order if c.startswith('qmx_')))
L += cmp_rows('The ETT lowest-MSE condition on MIT-BIH (also chosen after seeing test values)', ETT_BEST)
L += ['', f'Range of the mean macro-F1 over the 20 query conditions: {100 * min(f1.values()):.2f} to {100 * max(f1.values()):.2f} %.']

# Hippocampus-only NSMT (same training rule, separately trained, no Neocortex path). A reference, not an ablation of the query model: its attention and energy are NSMT's.
HIPPO = {}
for r in read_result(os.path.join(MAIN, 'flif-v1-cls', 'raw.txt'), lambda c: c == 'hippo_only'):
    if int(r['seed']) in HIPPO:
        stop(f'duplicate hippo_only seed {r["seed"]}')
    HIPPO[int(r['seed'])] = dict(r, E=num(r, 'total_fft_uJ'))
for s in SEEDS:
    if s not in HIPPO:
        stop(f'missing hippo_only seed {s}')
check = compare(ref(NSMT), ref(HIPPO))
if abs(check['m'] + 0.25) > 0.0051:      # value logged by commit 07986f989 (NSMT vs hippo-only)
    stop(f'NSMT vs hippo_only {check["m"]:.3f} %p, expected -0.25 from commit 07986f989')
L += ['', f'## Reference: Hippocampus-only NSMT at its own setting (no Neocortex path, same training rule; macro-F1 {100 * mean([num(x, "f1") for x in ref(HIPPO)]):.2f}; NOT an ablation of the query model, which has T = 4)', '',
      '| comparison | mean %p | seed 42, 7, 2026 | verdict | energy A/B |', '|---|---:|---|---|---:|']
for name, ra in [('NSMT rerun', ref(NSMT))] + [(f'nsmt4q_{n}', q(f'nsmt4q_{n}')) for n in NEURONS] + [(f'qmx_{n}_muxlin_{r}', q(f'qmx_{n}_muxlin_{r}')) for n in NEURONS for r in RES]:
    k = compare(ra, ref(HIPPO))
    L.append(f"| {name} vs hippo-only | {k['m']:+.2f} | " + ', '.join(f'{z:+.2f}' for z in k['d']) + f" | {k['verdict']} | {k['energy']:.2f} |")

path = os.path.join(SUITE, 'report_extra.md')
with open(path, 'w') as f:
    f.write('\n'.join(L) + '\n')
print('wrote', path)
print('\n'.join(L))
