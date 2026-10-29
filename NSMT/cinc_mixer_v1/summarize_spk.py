"""Summary / verdicts for the SpkMixer rerun (2026-10-29, PROJECT_LOG, exp/f-lif-pop-v3; CinC fold 0, seeds 42 / 7 / 2026).

Conditions: spk_<neuron>_<comb>_<cf>, neuron in lif / flif / pop_lif / pop_flif / popg_lif / popg_flif, comb in or / sdor / sdlin
(/ linmx reference), cf in iand / or / add. Verdict rule = summarize_rhythm.verdict (mean >= +0.5 %p and >= 2 of 3 seeds the same
direction = better; mirror = worse; else no clear difference; a missing / non-finite run = incomplete). Criterion (2) vs the
Spikformer-style model (spk_ssa, stage-9 suite copy): better, or mean >= -0.5 %p with total energy <= 0.5x. Criterion energy =
total_fft_int_uJ for EVERY condition (energy.SynOpRecorder: integer rule for add, user decision 10-29, and the neuron input-AC term
counted for every neuron class -- syops' custom LIF hook matches LIFNeuron / FLIFNeuron by exact type and so skips PopulationFLIF,
which makes total_fft_uJ 1.46 uJ (9 layers x 179,712 inputs) too small for pop_lif / pop_flif; smoke 10-29). For lif / flif / popg with
iand / or the two columns are identical (check s5, smoke); the syops-rule total_fft_uJ is reported beside it. The Spikformer-style
reference has no recorder column; its total_fft_uJ (all-LIF model, syops rule = recorder rule) is used.

  --stage select    B1: residual selection on LIF (3 fusions x 3 residuals x 3 seeds = 27 runs, all required): the residual with
                    the highest mean best-VALIDATION score over its 9 runs; ties (< 1e-9) -> iand. Writes select_residual.{md,json}.
  --stage verdicts  B2 / C with the selected residual (or --cf): per neuron sdor - or, sdlin - or, sdlin - sdor; every condition
                    vs spk_ssa with criterion (2); LIF linmx reference vs sdlin and vs spk_ssa; the B1 residual contrasts
                    (or - iand, add - iand per fusion, test score) as a report. Writes summary.md.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize_rhythm as S                                              # noqa: E402

SEEDS, NEED = (42, 7, 2026), 2
NEURONS = ('lif', 'flif', 'pop_lif', 'pop_flif', 'popg_lif', 'popg_flif')
LABEL = {'lif': 'LIF', 'flif': 'f-LIF', 'pop_lif': 'population LIF (soma)', 'pop_flif': 'population f-LIF (soma)',
         'popg_lif': 'popg LIF (tau groups, no soma)', 'popg_flif': 'popg f-LIF (tau groups, no soma)'}
COMBS, CFS = ('or', 'sdor', 'sdlin'), ('iand', 'or', 'add')


def cond(n, comb, cf):
    return f'spk_{n}_{comb}_{cf}'


def energy(r):
    """criterion energy of a row (uJ): the recorder total (integer rule for add; neuron AC term for every neuron class)."""
    return float(r['total_fft_int_uJ'])


def val_score(r):
    s = str(r.get('best_epoch_val', ''))
    if ':' not in s:
        return float('nan')
    return float(s.split(':')[1])


def rows_of(rows, c):
    return sorted([r for r in rows if r['cond'] == c and int(r.get('fold', 0)) == 0], key=lambda r: int(r['seed']))


def complete(rows, c):
    rr = rows_of(rows, c)
    return {int(r['seed']) for r in rr} == set(SEEDS) and len(rr) == 3 and all(math.isfinite(float(r['score'])) for r in rr)


def ms(v, k=100):
    v = [k * float(x) for x in v]
    return f'{np.mean(v):.2f} ± {np.std(v):.2f}' if len(v) > 1 else (f'{v[0]:.2f}' if v else 'n/a')


def load(a):
    R = os.path.join(HERE, 'results', a.suite)
    os.makedirs(R, exist_ok=True)
    rows = [r for r in S.parse(os.path.join(R, 'raw.txt')) if int(r.get('crop', 0)) == 0]
    ref = [r for r in S.parse(os.path.join(HERE, 'results', a.ref_suite, 'raw.txt')) if r['cond'] == 'spk_ssa' and int(r.get('fold', 0)) == 0]
    return R, rows, ref


def select(a):
    R, rows, _ = load(a)
    conds = [cond('lif', comb, cf) for comb in COMBS for cf in CFS]
    missing = [c for c in conds if not complete(rows, c)]
    L = ['# B1 residual selection (LIF, fold 0): mean best-validation score over 3 fusions x 3 seeds', '',
         '| residual | runs | mean validation score | mean test score | mean total µJ (criterion rule) |', '|---|---:|---:|---:|---:|']
    out = dict(rule='highest mean best-validation score over spk_lif_{or,sdor,sdlin}_<cf> x seeds 42/7/2026; ties < 1e-9 -> iand',
               missing=missing, by_cf={})
    best, best_v = None, -1.
    for cf in CFS:
        rr = [r for comb in COMBS for r in rows_of(rows, cond('lif', comb, cf))]
        vs = [val_score(r) for r in rr]
        if len(rr) == 9 and all(math.isfinite(v) for v in vs):
            mv = float(np.mean(vs)) * 100
            out['by_cf'][cf] = dict(n=9, val=mv, test=float(np.mean([r['score'] for r in rr])) * 100, uJ=float(np.mean([energy(r) for r in rr])))
            L.append(f"| {cf.upper()} | 9 | {mv:.2f} | {out['by_cf'][cf]['test']:.2f} | {out['by_cf'][cf]['uJ']:.1f} |")
            if mv > best_v + 1e-9 or (abs(mv - best_v) <= 1e-9 and cf == 'iand'):
                best, best_v = cf, mv
        else:
            L.append(f'| {cf.upper()} | {len(rr)} | incomplete | | |')
    if missing:
        out['selected'] = None
        L += ['', f'**Incomplete:** missing or non-finite: {missing}. No residual selected.']
    else:
        out['selected'] = best
        L += ['', f'**Selected residual: {best.upper()}** (rule: highest mean best-validation score; ties -> IAND).']
    L += ['', '## Per condition (test score, best-validation weights; mean ± sd over seeds)', '',
          '| condition | n | validation | test score | macro-F1 | total µJ syops rule | total µJ integer rule | time s |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for comb in COMBS:
        for cf in CFS:
            rr = rows_of(rows, cond('lif', comb, cf))
            if rr:
                L.append(f"| {cond('lif', comb, cf)} | {len(rr)} | {ms([val_score(r) for r in rr])} | {ms([r['score'] for r in rr])} | {ms([r['macro_f1'] for r in rr])} | "
                         f"{np.mean([float(r['total_fft_uJ']) for r in rr]):.1f} | {np.mean([float(r['total_fft_int_uJ']) for r in rr]):.1f} | {np.mean([r['time_s'] for r in rr]):.0f} |")
    open(os.path.join(R, 'select_residual.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(out, open(os.path.join(R, 'select_residual.json'), 'w'), indent=1)
    print('\n'.join(L))


def verdicts(a):
    R, rows, ref = load(a)
    cf = a.cf
    if cf is None:
        sel = os.path.join(R, 'select_residual.json')
        if not os.path.exists(sel):
            raise SystemExit('run --stage select first (or pass --cf)')
        cf = json.load(open(sel))['selected']
        if cf is None:
            raise SystemExit('no residual selected (B1 incomplete)')
    allrows = rows + ref
    e_ref = float(np.mean([float(r['total_fft_uJ']) for r in ref])) if ref else float('nan')
    L = [f'# SpkMixer rerun `{a.suite}` — verdicts with residual {cf.upper()}', '',
         f'CinC fold 0, test score (mean F1 of N / AF / Other, %), best-validation weights, seeds {SEEDS}. Verdict: mean >= +0.5 %p and >= 2/3 seeds '
         f'= better; mirror = worse; else no clear difference. Spikformer-style reference `spk_ssa`: {ms([r["score"] for r in ref])}, {e_ref:.1f} µJ '
         f'({len(ref)} runs, suite `{a.ref_suite}`).', '',
         '## Scores', '', '| neuron | OR fusion | sdor (F v R v s_d) | sdlin (W[F v R; s_d]) | linmx (reference, real input) |', '|---|---:|---:|---:|---:|']
    for n in NEURONS:
        cells = []
        for comb in ('or', 'sdor', 'sdlin', 'linmx'):
            rr = rows_of(rows, cond(n, comb, cf))
            cells.append(f"{ms([r['score'] for r in rr])} ({len(rr)})" if rr else '—')
        L.append(f'| {LABEL[n]} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Energy (total µJ, FFT neuron model; criterion rule / syops rule)', '', '| neuron | OR | sdor | sdlin | linmx |', '|---|---:|---:|---:|---:|']
    for n in NEURONS:
        cells = []
        for comb in ('or', 'sdor', 'sdlin', 'linmx'):
            rr = rows_of(rows, cond(n, comb, cf))
            cells.append(f"{np.mean([energy(r) for r in rr]):.1f} / {np.mean([float(r['total_fft_uJ']) for r in rr]):.1f}" if rr else '—')
        L.append(f'| {LABEL[n]} | ' + ' | '.join(cells) + ' |')

    def vline(tag, A, B):
        v, d, m = S.verdict(allrows, A, B, SEEDS, NEED)
        return f'| {tag} | {A} − {B} | ' + (f'{m:+.2f} | {", ".join(f"{x:+.2f}" for x in d)} | {v} |' if v != 'incomplete' else 'incomplete | | incomplete |')

    L += ['', '## V1 fusion effect per neuron (same residual)', '', '| question | contrast | mean %p | per seed | verdict |', '|---|---|---:|---|---|']
    for n in NEURONS:
        L.append(vline(f'V1a {LABEL[n]}: s_d by OR', cond(n, 'sdor', cf), cond(n, 'or', cf)))
        L.append(vline(f'V1b {LABEL[n]}: s_d by learned fusion', cond(n, 'sdlin', cf), cond(n, 'or', cf)))
        L.append(vline(f'V1c {LABEL[n]}: learned vs OR form', cond(n, 'sdlin', cf), cond(n, 'sdor', cf)))
    L += ['', '## V2 vs Spikformer-style (criterion (2): better, or mean >= −0.5 %p with energy <= 0.5x)', '',
          '| condition | mean %p | per seed | verdict | energy / ref | criterion (2) |', '|---|---:|---|---|---:|---|']
    for n in NEURONS:
        for comb in ('or', 'sdor', 'sdlin', 'linmx'):
            c = cond(n, comb, cf)
            if not rows_of(rows, c):
                continue
            v, d, m = S.verdict(allrows, c, 'spk_ssa', SEEDS, NEED)
            if v == 'incomplete':
                L.append(f'| {c} | incomplete | | incomplete | | incomplete |'); continue
            ratio = float(np.mean([energy(r) for r in rows_of(rows, c)])) / e_ref
            crit = 'met' if (v == 'better' or (m >= -0.5 and ratio <= 0.5)) else 'not met'
            L.append(f'| {c} | {m:+.2f} | {", ".join(f"{x:+.2f}" for x in d)} | {v} | {ratio:.2f}x | {crit} |')
    L += ['', '## V3 reference (3): LIF linmx (real-valued fusion input)', '', '| question | contrast | mean %p | per seed | verdict |', '|---|---|---:|---|---|',
          vline('V3a linmx vs sdlin', cond('lif', 'linmx', cf), cond('lif', 'sdlin', cf)),
          vline('V3b linmx vs sdor', cond('lif', 'linmx', cf), cond('lif', 'sdor', cf))]
    L += ['', '## B1 residual contrasts (LIF, test score; the selection itself used validation)', '', '| fusion | contrast | mean %p | per seed | verdict |', '|---|---|---:|---|---|']
    for comb in COMBS:
        L.append(vline(f'{comb}: OR − IAND', cond('lif', comb, 'or'), cond('lif', comb, 'iand')))
        L.append(vline(f'{comb}: ADD − IAND', cond('lif', comb, 'add'), cond('lif', comb, 'iand')))
    L += ['', '## All runs', '', '| condition | seed | validation | test score | macro-F1 | AF F1 | Other F1 | Noisy F1 | total µJ syops | total µJ integer | best epoch | time s |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows, key=lambda r: (r['cond'], int(r['seed']))):
        L.append(f"| {r['cond']} | {int(r['seed'])} | {100 * val_score(r):.2f} | {100 * float(r['score']):.2f} | {100 * float(r['macro_f1']):.2f} | {100 * float(r['f1_1']):.2f} | "
                 f"{100 * float(r['f1_2']):.2f} | {100 * float(r['f1_3']):.2f} | {float(r['total_fft_uJ']):.1f} | {float(r['total_fft_int_uJ']):.1f} | "
                 f"{str(r.get('best_epoch_val', '')).split(':')[0]} | {float(r['time_s']):.0f} |")
    skips = sum(1 for line in open(os.path.join(R, 'raw.txt')) if line.startswith('SKIP')) if os.path.exists(os.path.join(R, 'raw.txt')) else 0
    L += ['', f'RESULT lines: {len(rows)}; SKIP lines: {skips}.']
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-rhythm-spk-cinc')
    ap.add_argument('--ref_suite', default='flif-v1-rhythm-cinc', help='stage-9 suite copy holding spk_ssa fold 0 (results/BASELINE_COPIES.txt)')
    ap.add_argument('--stage', choices=['select', 'verdicts'], required=True)
    ap.add_argument('--cf', choices=list(CFS), default=None)
    a = ap.parse_args()
    (select if a.stage == 'select' else verdicts)(a)
