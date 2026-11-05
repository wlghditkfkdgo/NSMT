"""Independent recomputation and report-only tables for the ETT stage of the query-form bidirectional Neocortex test (2026-11-05, exp/f-lif-pop-v3).
No training and nothing read from a checkpoint. Own parser and own arithmetic (no import from summarize_qmx.py), so a summarizer bug shows up as a mismatch.
Fail closed: a missing run, a duplicate key, a non-finite value, a recount that does not reproduce the logged MSE, a non-binary head input, or a value that differs
from results/qmx-ett/summary.md stops the script with exit 2 and writes nothing.
Report only (no verdict): best validation epoch, seed spread, firing rates, validation ranking, the lowest-MSE condition against the references.
usage: python -B report_qmx_ett.py   ->   results/qmx-ett/report_extra.md
"""
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1/forecasting/results'
CM = os.path.join(HERE, '..', 'cinc_mixer_v1', 'results')
SUITE = os.path.join(HERE, 'results', 'qmx-ett')
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
SEEDS = (42, 7, 2026)
NEURONS = ('lif', 'flif', 'pop_lif', 'pop_flif')
RES = ('or', 'iand')


def stop(msg):
    print('STOP:', msg)
    sys.exit(2)


def read_result(path, used=lambda c: True):
    """A SKIP line stops the script when its condition is one this report uses (flif-v1-s2p10 has two SKIP lines of neo_flif_both, a condition not read here)."""
    out = []
    for line in open(path):
        if line.startswith('SKIP'):
            m = re.search(r'\bcond=(\S+)', line)
            if m is None or used(m[1]):
                stop(f'SKIP line of a used condition in {path}: {line[:120]}')
        if line.startswith('RESULT'):
            out.append(dict(tok.split('=', 1) for tok in line.split()[1:] if '=' in tok))
    return out


def mean(v):
    return sum(v) / len(v)


def sd(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def finite(x, what):
    x = float(x)
    if not math.isfinite(x):
        stop(f'non-finite {what}')
    return x


def energy_map(path):
    M = {}
    for e in csv.DictReader(open(path)):
        k = (e['cond'], e['ds'], int(e['pl']), int(e['seed']), e.get('tag', ''))
        if k in M:
            stop(f'duplicate energy row {k}')
        M[k] = e
    return M


# ---- runs
Q = {}


def add_query(raw, en, keep):
    M = energy_map(en)
    for r in read_result(raw, keep):
        if not keep(r['cond']):
            continue
        k = (r['cond'], r['ds'], int(r['pl']), int(r['seed']))
        if k in Q:
            stop(f'duplicate run {k}')
        e = M.get((r['cond'], r['ds'], int(r['pl']), int(r['seed']), r.get('tag', '')))
        if e is None:
            stop(f'no energy row for {k}')
        r['E'] = finite(e['total_fft_uJ_corr'], f'energy {k}')
        Q[k] = r


is_plain = lambda c: c in {f'nsmt4q_{n}' for n in NEURONS}
for s in ('flif-v1-ett-query', 'flif-v1-ett-query-m2'):
    add_query(os.path.join(MAIN, s, 'raw.txt'), os.path.join(MAIN, s, 'energy_recount.csv'), is_plain)
add_query(os.path.join(SUITE, 'raw.txt'), os.path.join(SUITE, 'energy_recount.csv'), lambda c: True)

NEW = [f'qmx_{n}_{f}_{r}' for n in NEURONS for f in ('muxlin', 'or') for r in RES]
PLAIN = [f'nsmt4q_{n}' for n in NEURONS]
for c in NEW + PLAIN:
    for ds, pl in CELLS:
        for s in SEEDS:
            if (c, ds, pl, s) not in Q:
                stop(f'missing run {c} {ds} H{pl} seed {s}')
            finite(Q[(c, ds, pl, s)]['mse'], 'mse')

# ---- recount of this suite
rec = list(csv.DictReader(open(os.path.join(SUITE, 'energy_recount.csv'))))
if len(rec) != 204:
    stop(f'energy_recount has {len(rec)} rows, expected 204')
worst = 0.0
for e in rec:
    d = abs(finite(e['mse_recomputed'], 'mse_recomputed') - finite(e['mse_logged'], 'mse_logged'))
    worst = max(worst, d)
    if d > 1e-5:
        stop(f'recount does not reproduce the logged MSE: {e["cond"]} {e["ds"]} {e["pl"]} {e["seed"]} ({d})')
    if e['binary'] != 'True':
        stop(f'head input is not binary: {e["cond"]} {e["ds"]} {e["pl"]} {e["seed"]}')
print(f'recount: 204 rows, largest |recomputed - logged| MSE {worst:.2e}, head input binary in all')

# ---- references
base, tune = os.path.join(CM, 'flif-v1-ett-spkmux'), os.path.join(CM, 'flif-v1-ett-spkmux-tune')
en_b = {(e['ds'], int(e['pl']), int(e['seed'])): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(base, 'baseline_spk_ssa_energy.csv')))}
en_t = {(e['tag'], e['ds'], int(e['pl']), int(e['seed'])): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(tune, 'energy_recount.csv')))}
TUNED = {}
for r in read_result(os.path.join(base, 'baseline_spk_ssa_raw.txt')):
    if r['ds'] == 'ETTh2':
        r['E'] = en_b[(r['ds'], int(r['pl']), int(r['seed']))]
        TUNED[(r['ds'], int(r['pl']), int(r['seed']))] = r
for r in read_result(os.path.join(tune, 'raw.txt')):
    if r['cond'] == 'spk_ssa' and r['ds'] == 'ETTm2' and abs(float(r['lr']) - 2.5e-4) < 1e-12:
        k = (r['ds'], int(r['pl']), int(r['seed']))
        if k in TUNED:
            stop(f'duplicate tuned Spikformer-style row {k}')
        r['E'] = en_t[(r.get('tag', ''), r['ds'], int(r['pl']), int(r['seed']))]
        TUNED[k] = r
NSMT = {}
for r in read_result(os.path.join(MAIN, 'flif-v1-s2p10', 'raw.txt'), lambda c: c == 'base'):
    if r['cond'] == 'base':
        ds, pl = r['dataset'].split('_p')
        NSMT[(ds, int(pl), int(r['seed']))] = dict(mse=r['mse_best'], E=float(r['total_fft_uJ_best']))
for T, name in ((TUNED, 'Spikformer-style'), (NSMT, 'NSMT')):
    for ds, pl in CELLS:
        for s in SEEDS:
            if (ds, pl, s) not in T:
                stop(f'missing {name} {ds} H{pl} seed {s}')


def q(c):
    return lambda ds, pl, s: Q[(c, ds, pl, s)]


def ref(T):
    return lambda ds, pl, s: T[(ds, pl, s)]


def compare(fa, fb):
    cell, pairs, ers = [], [], []
    for ds, pl in CELLS:
        ra, rb = [fa(ds, pl, s) for s in SEEDS], [fb(ds, pl, s) for s in SEEDS]
        r = [100 * (finite(a['mse'], 'mse') - finite(b['mse'], 'mse')) / finite(b['mse'], 'mse') for a, b in zip(ra, rb)]
        cell.append(mean(r))
        pairs += r
        ers.append(mean([finite(a['E'], 'E') for a in ra]) / mean([finite(b['E'], 'E') for b in rb]))
    R, lo, hi = mean(cell), sum(z < 0 for z in pairs), sum(z > 0 for z in pairs)
    v = 'better' if R <= -1 and lo >= 8 else 'worse' if R >= 1 and hi >= 8 else 'no clear difference'
    return dict(R=R, lo=lo, verdict=v, energy=mean(ers), cells=cell)


# ---- agreement with summary.md, every pre-registered row
SUMMARY = {}
for line in open(os.path.join(SUITE, 'summary.md')):
    m = re.match(r'\| (V[1-4]) (\w+) (OR|IAND): [^|]+\| ([+-][\d.]+) \| (\d+)/12 \| ([a-z ]+) \| ([\d.]+) \|', line)
    if m:
        SUMMARY[(m[1], m[2], m[3].lower())] = (float(m[4]), int(m[5]), m[6], float(m[7]))
if len(SUMMARY) != 32:
    stop(f'{len(SUMMARY)} pre-registered rows in summary.md, expected 32')
COMP = {}
for n in NEURONS:
    for r in RES:
        mx, o, p = q(f'qmx_{n}_muxlin_{r}'), q(f'qmx_{n}_or_{r}'), q(f'nsmt4q_{n}')
        COMP[('V1', n, r)], COMP[('V2', n, r)] = compare(mx, p), compare(mx, o)
        COMP[('V3', n, r)], COMP[('V4', n, r)] = compare(o, p), compare(mx, ref(TUNED))
for k, c in COMP.items():
    R, lo, v, er = SUMMARY[k]
    if abs(R - c['R']) > 0.0051 or lo != c['lo'] or v != c['verdict'] or abs(er - c['energy']) > 0.0051:
        stop(f'summary.md differs at {k}: {SUMMARY[k]} vs {c}')
print('32 pre-registered rows of summary.md reproduced (R, lower pairs, verdict, energy)')

# ---- report
L = ['# ETT stage: independent recomputation and report-only tables', '',
     f'Recount: 204 rows, largest |recomputed - logged| MSE {worst:.2e}, head input binary in all. The 32 pre-registered rows of summary.md were reproduced by separate code.',
     'Everything below is report only. Best epoch counts from 0.', '']


def be(r):
    a, _, c = r['best_epoch_val_epochs'].split(':')
    return int(a), int(c)


def group(fu):
    """Condition names of one row of the epoch table."""
    return [f'nsmt4q_{n}' for n in NEURONS] if fu == 'plain' else [f'qmx_{n}_{fu}_{r}' for n in NEURONS for r in RES]


L += ['## Best validation epoch (median over runs; share of runs with best epoch <= 1)', '', '| Neocortex | ' + ' | '.join(f'{d} H{p}' for d, p in CELLS) + ' |', '|---|' + '---:|' * 4]
for fu, name in (('plain', 'plain query'), ('muxlin', 'bidirectional muxlin'), ('or', 'bidirectional OR')):
    row = []
    for ds, pl in CELLS:
        v = sorted(be(Q[(c, ds, pl, s)])[0] for c in group(fu) for s in SEEDS)
        row.append(f'{v[len(v) // 2]} ({100 * sum(x <= 1 for x in v) / len(v):.0f} %, n = {len(v)})')
    L.append(f'| {name} | ' + ' | '.join(row) + ' |')

L += ['', '## Seed spread: standard deviation over the 3 seeds as % of the mean MSE', '', '| condition | ' + ' | '.join(f'{d} H{p}' for d, p in CELLS) + ' |', '|---|' + '---:|' * 4]
for c in PLAIN + [f'qmx_{n}_muxlin_or' for n in NEURONS]:
    row = []
    for ds, pl in CELLS:
        v = [float(Q[(c, ds, pl, s)]['mse']) for s in SEEDS]
        row.append(f'{100 * sd(v) / mean(v):.2f}')
    L.append(f'| {c} | ' + ' | '.join(row) + ' |')

L += ['', '## Firing rates and energy (mean over 12 runs)', '', '| condition | rho_neo | rho_hippo | energy µJ |', '|---|---:|---:|---:|']
for n in NEURONS:
    for c in [f'nsmt4q_{n}'] + [f'qmx_{n}_{f}_{r}' for f in ('muxlin', 'or') for r in RES]:
        rows = [Q[(c, ds, pl, s)] for ds, pl in CELLS for s in SEEDS]
        L.append(f"| {c} | {mean([float(r['rho_neo']) for r in rows]):.4f} | {mean([float(r['rho_hippo']) for r in rows]):.4f} | {mean([r['E'] for r in rows]):.2f} |")


def val(r):
    return float(r['best_epoch_val_epochs'].split(':')[1])


L += ['', '## Validation only: mean over cells of best validation L1 relative to the plain query of the same neuron, %', '', '| neuron | muxlin OR | muxlin IAND | bidirectional OR, residual OR | bidirectional OR, residual IAND |', '|---|---:|---:|---:|---:|']
for n in NEURONS:
    row = []
    for f, r in (('muxlin', 'or'), ('muxlin', 'iand'), ('or', 'or'), ('or', 'iand')):
        v = [100 * (mean([val(Q[(f'qmx_{n}_{f}_{r}', ds, pl, s)]) for s in SEEDS]) / mean([val(Q[(f'nsmt4q_{n}', ds, pl, s)]) for s in SEEDS]) - 1) for ds, pl in CELLS]
        row.append(f'{mean(v):+.2f}')
    L.append(f'| {n} | ' + ' | '.join(row) + ' |')

ms = {c: mean([mean([float(Q[(c, ds, pl, s)]['mse']) for s in SEEDS]) for ds, pl in CELLS]) for c in NEW + PLAIN}
order = sorted(ms, key=ms.get)
best, runner = order[0], order[1]
L += ['', f'## Lowest mean test MSE (chosen after seeing the test values, so biased): {best} {ms[best]:.4f}; next {runner} {ms[runner]:.4f}', '', '| comparison | R % | lower pairs | verdict | energy A/B | per cell r % |', '|---|---:|---:|---|---:|---|']
n_best = re.match(r'qmx_(\w+?)_(muxlin|or)_', best)[1]
for name, fb in ((f'vs nsmt4q_{n_best}', q(f'nsmt4q_{n_best}')), ('vs Spikformer-style (tuned)', ref(TUNED)), ('vs NSMT rerun', ref(NSMT))):
    c = compare(q(best), fb)
    L.append(f"| {name} | {c['R']:+.2f} | {c['lo']}/12 | {c['verdict']} | {c['energy']:.2f} | " + ', '.join(f'{z:+.2f}' for z in c['cells']) + ' |')

L += ['', '## How close the V1 rows came to the rule (better needs R <= -1 % and at least 8 of 12 pairs lower)', '', '| neuron | residual | R % | lower pairs | verdict |', '|---|---|---:|---:|---|']
for n in NEURONS:
    for r in RES:
        c = COMP[('V1', n, r)]
        L.append(f"| {n} | {r.upper()} | {c['R']:+.2f} | {c['lo']}/12 | {c['verdict']} |")

path = os.path.join(SUITE, 'report_extra.md')
with open(path, 'w') as f:
    f.write('\n'.join(L) + '\n')
print('wrote', path)
print('\n'.join(L))
