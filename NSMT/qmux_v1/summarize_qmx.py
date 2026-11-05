"""Query form with the bidirectional Neocortex (2026-11-05, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; neo_mux.py). Per neuron n (LIF, f-LIF,
population LIF, population f-LIF) and residual r (OR, IAND), same seeds 42 / 7 / 2026:
    V1 (main)  qmx_n_muxlin_r vs nsmt4q_n          the new Neocortex vs the plain one-direction query model
    V2         qmx_n_muxlin_r vs qmx_n_or_r        what the membrane-difference gate adds over bidirectional OR
    V3         qmx_n_or_r vs nsmt4q_n              what bidirectionality alone adds
    V4         qmx_n_muxlin_r vs Spikformer-style  ETT: tuned (criterion 2), default (report); MIT-BIH: default lr 0.01 (criterion 2; the query runs use
                                                    lr 0.01), tuned (report)
    report     qmx_n_muxlin_r vs NSMT rerun; residual IAND vs OR; the validation choice of the residual (ETT only, MIT-BIH has no validation here)
Rules (the tasks' existing ones): ETT r = 100 (MSE_A - MSE_B) / MSE_B per seed, R = mean over the 4 cells of the seed-mean r, better = R <= -1 % and >= 8 of 12
pairs lower, worse = mirror; criterion (2) = better or (R <= +1 % and energy <= 0.5x). MIT-BIH: macro-F1 difference per seed, better = mean >= +0.5 %p and >= 2 of 3
seeds, worse = mirror; criterion (2) = better or (mean >= -0.5 and energy <= 0.5x). Energy: ETT total_fft_uJ_corr, MIT-BIH total_fft_uJ.
ETT residual choice (validation only): per neuron, score = mean over the 4 cells of [seed-mean best validation L1 with IAND] / [the same with OR]; below 1 picks IAND.
Adoption, written with --task mitbih (needs both): a neuron and residual whose ETT V1 is better and whose MIT-BIH V1 is not worse.
Controls: nsmt4q_* from the source results (read only; SHA256 printed) and from this suite for the ones the source did not run. Spikformer-style and NSMT rerun rows: read only.
Fail closed: a missing run, a SKIP line or a non-finite value -> incomplete (exit 2).
usage: python summarize_qmx.py --task ett|mitbih [--suite <suite>]   ->  results/<suite>/summary.md
"""
import argparse
import csv
import hashlib
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402

MAIN = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1'
CM = os.path.join(HERE, '..', 'cinc_mixer_v1', 'results')
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
SEEDS = (42, 7, 2026)
NEURONS = ('lif', 'flif', 'pop_lif', 'pop_flif')
RES = ('or', 'iand')
SHA = []


class Incomplete(Exception):
    pass


def sha(p):
    SHA.append(f'{os.path.relpath(p, HERE)} {hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]}')


def ett_rows():
    D = {}
    def add(raw, en, name=None):
        energy = {}
        if os.path.exists(en):
            for e in csv.DictReader(open(en)):
                energy[(e['cond'], e['ds'], int(e['pl']), int(e['seed']), e.get('tag', ''))] = float(e['total_fft_uJ_corr'])
        for r in parse(raw):
            k = (name or r['cond'], r['ds'], int(r['pl']), int(r['seed']))
            if k in D:
                raise SystemExit(f'duplicate {k} in {raw}')
            r['E'] = energy.get((r['cond'], r['ds'], int(r['pl']), int(r['seed']), str(r.get('tag', ''))), float('nan'))
            r['val'] = float(str(r['best_epoch_val_epochs']).split(':')[1])
            D[k] = r
    for s in ('flif-v1-ett-query', 'flif-v1-ett-query-m2'):
        raw = os.path.join(MAIN, 'forecasting', 'results', s, 'raw.txt'); sha(raw)
        add(raw, os.path.join(MAIN, 'forecasting', 'results', s, 'energy_recount.csv'))
    return D, add


def spk_ett():
    """Spikformer-style rows: default = the reused rows; tuned = ETTh2 default lr, ETTm2 lr 2.5e-4 (selected_lr.json of the ETT tuning)."""
    base, tune = os.path.join(CM, 'flif-v1-ett-spkmux'), os.path.join(CM, 'flif-v1-ett-spkmux-tune')
    sha(os.path.join(base, 'baseline_spk_ssa_raw.txt'))
    en = {(e['ds'], int(e['pl']), int(e['seed'])): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(base, 'baseline_spk_ssa_energy.csv')))}
    dflt = {}
    for r in parse(os.path.join(base, 'baseline_spk_ssa_raw.txt')):
        r['E'] = en.get((r['ds'], int(r['pl']), int(r['seed'])), float('nan'))
        dflt[(r['ds'], int(r['pl']), int(r['seed']))] = r
    entu = {(e['tag'], e['ds'], int(e['pl']), int(e['seed'])): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(tune, 'energy_recount.csv')))}
    tuned = {k: v for k, v in dflt.items() if k[0] == 'ETTh2'}
    for r in parse(os.path.join(tune, 'raw.txt')):
        if r['cond'] == 'spk_ssa' and r['ds'] == 'ETTm2' and abs(float(r['lr']) - 0.00025) < 1e-12:
            r['E'] = entu.get((str(r['tag']), r['ds'], int(r['pl']), int(r['seed'])), float('nan'))
            tuned[(r['ds'], int(r['pl']), int(r['seed']))] = r
    return dflt, tuned


def nsmt_ett():
    raw = os.path.join(MAIN, 'forecasting', 'results', 'flif-v1-s2p10', 'raw.txt'); sha(raw)
    R = {}
    for r in parse(raw):
        if r['cond'] == 'base':
            ds, pl = r['dataset'].split('_p')
            R[(ds, int(pl), int(r['seed']))] = dict(mse=float(r['mse_best']), mae=float(r['mae_best']), E=float(r['total_fft_uJ_best']))
    return R


def cmp_ett(fa, fb):
    """fa, fb: (ds, pl, seed) -> row. Returns the verdict dict; Incomplete on a missing run or a non-finite value."""
    cell, pairs, ers = [], [], []
    for ds, pl in CELLS:
        ra, rb = [fa(ds, pl, s) for s in SEEDS], [fb(ds, pl, s) for s in SEEDS]
        if any(x is None for x in ra + rb):
            raise Incomplete(f'{ds} H{pl}')
        v = [float(x['mse']) for x in ra + rb] + [float(x['E']) for x in ra + rb]
        if not all(math.isfinite(z) for z in v):
            raise Incomplete(f'{ds} H{pl}: non-finite')
        r = [100 * (float(a['mse']) - float(b['mse'])) / float(b['mse']) for a, b in zip(ra, rb)]
        cell.append(float(np.mean(r))); pairs += r
        ers.append(np.mean([float(a['E']) for a in ra]) / np.mean([float(b['E']) for b in rb]))
    R, lo, hi = float(np.mean(cell)), sum(z < 0 for z in pairs), sum(z > 0 for z in pairs)
    v = 'better' if R <= -1 and lo >= 8 else 'worse' if R >= 1 and hi >= 8 else 'no clear difference'
    er = float(np.mean(ers))
    return dict(R=R, lo=lo, verdict=v, energy=er, crit='met' if v == 'better' or (R <= 1 and er <= 0.5) else 'not met', cells=cell)


def task_ett(L, suite):
    D, add = ett_rows()
    p = os.path.join(HERE, 'results', suite, 'raw.txt')
    add(p, os.path.join(HERE, 'results', suite, 'energy_recount.csv'))
    if any(line.startswith('SKIP') for line in open(p)):
        raise Incomplete(f'SKIP lines in {p}')
    dflt, tuned = spk_ett(); nsmt = nsmt_ett()
    q = lambda c: (lambda ds, pl, s: D.get((c, ds, pl, s)))
    d_ = lambda T: (lambda ds, pl, s: T.get((ds, pl, s)))
    out = {}
    L += ['## ETT test MSE (seed mean) and energy', '', '| condition | ' + ' | '.join(f'{d} H{p_}' for d, p_ in CELLS) + ' | mean MSE | energy µJ (4-cell mean) | rho_neo |', '|---|' + '---:|' * 7]
    def line(name, f):
        v = [[f(ds, pl, s) for s in SEEDS] for ds, pl in CELLS]
        if any(x is None for r in v for x in r):
            L.append(f'| {name} | incomplete |' + ' |' * 6); return
        mse = [float(np.mean([float(x['mse']) for x in r])) for r in v]
        e = float(np.mean([float(x['E']) for r in v for x in r]))
        rho = [str(x.get('rho_neo', '')) for r in v for x in r]
        rv = [float(z) for z in rho if z not in ('', 'None')]
        L.append(f"| {name} | " + ' | '.join(f'{z:.4f}' for z in mse) + f" | {np.mean(mse):.4f} | {e:.2f} | {np.mean(rv) if rv else float('nan'):.4f} |")
    line('Spikformer-style (tuned)', d_(tuned)); line('NSMT rerun (same training rule)', d_(nsmt))
    for n in NEURONS:
        line(f'query {n} (plain)', q(f'nsmt4q_{n}'))
        for fu in ('muxlin', 'or'):
            for r in RES:
                line(f'query {n}, bidirectional {fu}, residual {r.upper()}', q(f'qmx_{n}_{fu}_{r}'))
    L += ['', '## Pre-registered comparisons', '', '| comparison | R % | lower pairs | verdict | energy A/B | criterion (2) | per cell r % |', '|---|---:|---:|---|---:|---|---|']
    def cmp_line(name, fa, fb, key=None):
        try:
            c = cmp_ett(fa, fb)
        except Incomplete as ex:
            L.append(f'| {name} | incomplete ({ex}) | | | | | |')
            out[key or name] = None
            return
        L.append(f"| {name} | {c['R']:+.2f} | {c['lo']}/12 | {c['verdict']} | {c['energy']:.2f} | {c['crit']} | " + ', '.join(f'{z:+.2f}' for z in c['cells']) + ' |')
        out[key or name] = c
    for n in NEURONS:
        for r in RES:
            cmp_line(f'V1 {n} {r.upper()}: muxlin vs plain query', q(f'qmx_{n}_muxlin_{r}'), q(f'nsmt4q_{n}'), ('V1', n, r))
    for n in NEURONS:
        for r in RES:
            cmp_line(f'V2 {n} {r.upper()}: muxlin vs bidirectional OR', q(f'qmx_{n}_muxlin_{r}'), q(f'qmx_{n}_or_{r}'))
    for n in NEURONS:
        for r in RES:
            cmp_line(f'V3 {n} {r.upper()}: bidirectional OR vs plain query', q(f'qmx_{n}_or_{r}'), q(f'nsmt4q_{n}'))
    for n in NEURONS:
        for r in RES:
            cmp_line(f'V4 {n} {r.upper()}: muxlin vs Spikformer-style (tuned)', q(f'qmx_{n}_muxlin_{r}'), d_(tuned))
    L += ['', '## Report', '', '| comparison | R % | lower pairs | verdict | energy A/B | criterion (2) | per cell r % |', '|---|---:|---:|---|---:|---|---|']
    for n in NEURONS:
        for r in RES:
            cmp_line(f'{n} {r.upper()}: muxlin vs Spikformer-style (default)', q(f'qmx_{n}_muxlin_{r}'), d_(dflt))
            cmp_line(f'{n} {r.upper()}: muxlin vs NSMT rerun', q(f'qmx_{n}_muxlin_{r}'), d_(nsmt))
    for n in NEURONS:
        cmp_line(f'{n}: muxlin IAND vs muxlin OR', q(f'qmx_{n}_muxlin_iand'), q(f'qmx_{n}_muxlin_or'))
        cmp_line(f'{n}: plain query vs Spikformer-style (tuned)', q(f'nsmt4q_{n}'), d_(tuned))
    L += ['', '## Residual choice (validation only: mean over cells of best validation L1, IAND / OR)', '', '| neuron | muxlin score | chosen |', '|---|---:|---|']
    for n in NEURONS:
        try:
            s_ = []
            for ds, pl in CELLS:
                a, b = [q(f'qmx_{n}_muxlin_iand')(ds, pl, s) for s in SEEDS], [q(f'qmx_{n}_muxlin_or')(ds, pl, s) for s in SEEDS]
                if any(x is None for x in a + b):
                    raise Incomplete('')
                s_.append(np.mean([x['val'] for x in a]) / np.mean([x['val'] for x in b]))
            sc = float(np.mean(s_))
            L.append(f"| {n} | {sc:.4f} | {'IAND' if sc < 1 else 'OR'} |")
        except Incomplete:
            L.append(f'| {n} | incomplete | |')
    return out


def mit_rows(suite):
    D = {}
    raw = os.path.join(MAIN, 'classification', 'results', 'flif-v1-mitbih-query', 'raw.txt'); sha(raw)
    p = os.path.join(HERE, 'results', suite, 'raw.txt')
    if any(line.startswith('SKIP') for line in open(p)):
        raise Incomplete(f'SKIP lines in {p}')
    for path in (raw, p):
        for r in parse(path):
            k = (r['cond'], int(r['seed']))
            if k in D:
                raise SystemExit(f'duplicate {k} in {path}')
            r['E'] = float(r['total_fft_uJ'])
            D[k] = r
    return D


def spk_mit():
    base, tune = os.path.join(CM, 'flif-v1-mitbih-spkmux'), os.path.join(CM, 'flif-v1-mitbih-spkmux-tune')
    sha(os.path.join(base, 'baseline_spk_ssa_raw.txt'))
    dflt = {int(r['seed']): dict(r, E=float(r['total_fft_uJ'])) for r in parse(os.path.join(base, 'baseline_spk_ssa_raw.txt'))}
    en = {(e['tag'], int(e['seed'])): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(tune, 'energy_recount.csv')))}
    tuned = {}
    for r in parse(os.path.join(tune, 'raw.txt')):
        if r['cond'] == 'spk_ssa' and abs(float(r['lr']) - 0.005) < 1e-12 and int(r['val_fold']) == -1:
            tuned[int(r['seed'])] = dict(r, E=en.get((str(r['tag']), int(r['seed'])), float('nan')))
    return dflt, tuned


def cmp_mit(ra, rb):
    if any(x is None for x in ra + rb):
        raise Incomplete('missing')
    d = [100 * (float(a['f1']) - float(b['f1'])) for a, b in zip(ra, rb)]
    if not all(math.isfinite(z) for z in d):
        raise Incomplete('non-finite')
    m = float(np.mean(d))
    v = 'better' if m >= 0.5 and sum(z > 0 for z in d) >= 2 else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= 2 else 'no clear difference'
    er = float(np.mean([float(a['E']) for a in ra]) / np.mean([float(b['E']) for b in rb]))
    return dict(m=m, d=d, verdict=v, energy=er, crit='met' if v == 'better' or (m >= -0.5 and er <= 0.5) else 'not met')


def task_mit(L, suite):
    D = mit_rows(suite)
    dflt, tuned = spk_mit()
    base = {int(r['seed']): dict(r, E=float(r['total_fft_uJ'])) for r in parse(os.path.join(MAIN, 'classification', 'results', 'flif-v1-cls', 'raw.txt')) if r['cond'] == 'base'}
    q = lambda c: [D.get((c, s)) for s in SEEDS]
    t = lambda T: [T.get(s) for s in SEEDS]
    L += ['## MIT-BIH test macro-F1 (seed mean) and energy', '', '| condition | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy µJ | rho_neo |', '|---|---:|---:|---|---:|---:|']
    def line(name, rr):
        if any(x is None for x in rr):
            L.append(f'| {name} | incomplete | | | | |'); return
        rho = [float(x['rho_neo']) for x in rr if str(x.get('rho_neo')) not in ('None', '', 'nan')]
        L.append(f"| {name} | {np.mean([100 * float(x['f1']) for x in rr]):.2f} ± {np.std([100 * float(x['f1']) for x in rr]):.2f} | {np.mean([100 * float(x['overall_acc']) for x in rr]):.2f} | "
                 + ' / '.join(f"{np.mean([100 * float(x[f'f1_{i}']) for x in rr]):.2f}" for i in range(5))
                 + f" | {np.mean([float(x['E']) for x in rr]):.2f} | {np.mean(rho) if rho else float('nan'):.4f} |")
    line('Spikformer-style (default, lr 0.01)', t(dflt)); line('Spikformer-style (tuned, lr 0.005)', t(tuned)); line('NSMT rerun', t(base))
    for n in NEURONS:
        line(f'query {n} (plain)', q(f'nsmt4q_{n}'))
        for fu in ('muxlin', 'or'):
            for r in RES:
                line(f'query {n}, bidirectional {fu}, residual {r.upper()}', q(f'qmx_{n}_{fu}_{r}'))
    out = {}
    L += ['', '## Pre-registered comparisons', '', '| comparison | mean %p | seed 42, 7, 2026 | verdict | energy A/B | criterion (2) |', '|---|---:|---|---|---:|---|']
    def cl(name, a, b, key=None):
        try:
            c = cmp_mit(a, b)
        except Incomplete as ex:
            L.append(f'| {name} | incomplete ({ex}) | | | | |'); out[key or name] = None; return
        L.append(f"| {name} | {c['m']:+.2f} | {', '.join(f'{z:+.2f}' for z in c['d'])} | {c['verdict']} | {c['energy']:.2f} | {c['crit']} |"); out[key or name] = c
    for n in NEURONS:
        for r in RES:
            cl(f'V1 {n} {r.upper()}: muxlin vs plain query', q(f'qmx_{n}_muxlin_{r}'), q(f'nsmt4q_{n}'), ('V1', n, r))
    for n in NEURONS:
        for r in RES:
            cl(f'V2 {n} {r.upper()}: muxlin vs bidirectional OR', q(f'qmx_{n}_muxlin_{r}'), q(f'qmx_{n}_or_{r}'))
    for n in NEURONS:
        for r in RES:
            cl(f'V3 {n} {r.upper()}: bidirectional OR vs plain query', q(f'qmx_{n}_or_{r}'), q(f'nsmt4q_{n}'))
    for n in NEURONS:
        for r in RES:
            cl(f'V4 {n} {r.upper()}: muxlin vs Spikformer-style (default)', q(f'qmx_{n}_muxlin_{r}'), t(dflt))
    L += ['', '## Report', '', '| comparison | mean %p | seed 42, 7, 2026 | verdict | energy A/B | criterion (2) |', '|---|---:|---|---|---:|---|']
    for n in NEURONS:
        for r in RES:
            cl(f'{n} {r.upper()}: muxlin vs Spikformer-style (tuned)', q(f'qmx_{n}_muxlin_{r}'), t(tuned))
            cl(f'{n} {r.upper()}: muxlin vs NSMT rerun', q(f'qmx_{n}_muxlin_{r}'), t(base))
    for n in NEURONS:
        cl(f'{n}: muxlin IAND vs muxlin OR', q(f'qmx_{n}_muxlin_iand'), q(f'qmx_{n}_muxlin_or'))
        cl(f'{n}: plain query vs Spikformer-style (default)', q(f'nsmt4q_{n}'), t(dflt))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--task', choices=['ett', 'mitbih'], required=True); ap.add_argument('--suite', default=None); a = ap.parse_args()
    suite = a.suite or f'qmx-{a.task}'
    L = [f'# Query form with the bidirectional Neocortex — {a.task} ({suite})', '', 'Rules and sources: module docstring.', '']
    incomplete = False
    try:
        res = (task_ett if a.task == 'ett' else task_mit)(L, suite)
    except Incomplete as ex:
        L.append(f'incomplete: {ex}'); res = {}; incomplete = True
    incomplete |= any(v is None for v in res.values()) or not res
    if a.task == 'mitbih' and not incomplete:
        L2 = []
        ett = task_ett(L2, 'qmx-ett')
        L += ['', '## Adoption (pre-registered): ETT V1 better and MIT-BIH V1 not worse', '', '| neuron | residual | ETT V1 | MIT-BIH V1 | adopted |', '|---|---|---|---|---|']
        for n in NEURONS:
            for r in RES:
                e, m = ett.get(('V1', n, r)), res.get(('V1', n, r))
                if e is None or m is None:
                    L.append(f'| {n} | {r.upper()} | incomplete | | |'); incomplete = True; continue
                L.append(f"| {n} | {r.upper()} | {e['verdict']} (R {e['R']:+.2f} %) | {m['verdict']} ({m['m']:+.2f} %p) | {'yes' if e['verdict'] == 'better' and m['verdict'] != 'worse' else 'no'} |")
    L += ['', 'Source files read (SHA256, first 16): ' + '; '.join(sorted(set(SHA)))]
    d = os.path.join(HERE, 'results', suite); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    if incomplete:
        print('incomplete'); sys.exit(2)


if __name__ == '__main__':
    main()
