"""AD with SF-de (2026-11-18, exp/f-lif-pop-v3; PROJECT_LOG AD SF-de pre-registration; user decision 11-18 "이대로 AD를 진행").
SF-de = the Spikformer-style model with the spikeDE LIF in every layer (--neuron_all lif, tag _nalif), tuned as the spikingjelly one (SF-sj)
was on 11-02 / 11-03 (summarize_ad_tune.py, stage lr): suite flif-v1-ad-sfde.
  select  per dataset, score(v) = seed-mean best validation MSE at lr v / that at 5e-4 (holdout runs: first 80 % trained, last 20 %
          validated; no test values); the lowest wins, ties (< 1e-12) keep 5e-4; a missing run or a SKIP line -> no selection.
          results/flif-v1-ad-sfde/selected_lr.json is never overwritten with a different selection.
  final   whole training file at the selected lr, test AUC-PR of the best-validation weights. Comparisons (summarize_ad_tune.compare:
          per seed r = 100 (AUC-PR_A - AUC-PR_B) / AUC-PR_B, R = mean over MSL / SMAP / PSM of the seed-mean r; better = R >= +2 % and
          >= 6 of 9 pairs higher, worse = mirror; energy = total_fft_uJ_corr):
            E1 muxlin(tuned, 11-03 rows) vs SF-de(tuned)                      verdict + paper bar
            report: muxlin(default) vs SF-de (paper bar), SF-de vs SF-sj(tuned), muxlin(tuned) vs SF-sj(tuned) (= 11-03 A1)
          Paper bar (user decision 11-18, per task, sign only): the SpkMixer's R > 0 (AUC-PR higher on average) and energy A/B < 1.
Fail closed (exit 2, nothing written): a missing / duplicate run, a SKIP line in the suite, a non-finite AUC-PR or energy_check != ok.
usage: python -B summarize_adsf.py --stage select|final
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                                        # noqa: E402
import summarize_ad_tune as sat                                                           # noqa: E402

SUITE = os.environ.get('ADSF_DRY') or os.path.join(HERE, 'results', 'flif-v1-ad-sfde')   # ADSF_DRY=<folder>: dry runs
DSS, SEEDS = sat.DSS, sat.SEEDS
LRS = ('0.0001', '0.00025', '0.0005', '0.001', '0.002')
LR0 = '0.0005'


def stop(msg):
    print('STOP:', msg)
    sys.exit(2)


def sfde_rows():
    raw = os.path.join(SUITE, 'raw.txt')
    if not os.path.exists(raw):
        stop(f'no {raw}')
    skips = [l for l in open(raw) if l.startswith('SKIP')]
    if skips:
        stop(f'{len(skips)} SKIP line(s) in {raw}')
    D = {}
    for r in parse(raw):
        k = (r['tag'], r['ds'], int(r['holdout']), int(r['seed']))
        if k in D:
            stop(f'duplicate run {k}')
        D[k] = r
    return D


def tag(lr, ho):
    return f'spk_ssa_lr{lr}_L2{"_ho" if ho else ""}_nalif'


def three(D, lr, ds, ho):
    v = [D.get((tag(lr, ho), ds, ho, s)) for s in SEEDS]
    if any(x is None for x in v):
        stop(f'missing SF-de run lr {lr} {ds} holdout {ho}: {[s for s, x in zip(SEEDS, v) if x is None]}')
    return v


def stage_select():
    D = sfde_rows()
    sel, scores = {}, {}
    L = ['# AD SF-de learning-rate selection (validation only)', '',
         'Score = seed-mean best validation MSE / that at 5e-4 (holdout: first 80 % trained, last 20 % validated; lower is better).', '',
         '| dataset | lr | val MSE per seed (42, 7, 2026) | mean | score | best epoch (mean) |', '|---|---|---|---:|---:|---:|']
    for ds in DSS:
        m = {}
        for lr in LRS:
            rr = three(D, lr, ds, 1)
            if not all(math.isfinite(float(x['val_mse_best'])) for x in rr):
                stop(f'non-finite validation MSE lr {lr} {ds}')
            m[lr] = float(np.mean([x['val_mse_best'] for x in rr]))
            L.append(f"| {ds} | {lr} | {', '.join('%.6f' % x['val_mse_best'] for x in rr)} | {m[lr]:.6f} | | {np.mean([x['best_epoch'] for x in rr]):.1f} |")
        sc = {lr: m[lr] / m[LR0] for lr in LRS}
        best = min(sc.values())
        win = LR0 if abs(sc[LR0] - best) < 1e-12 else min(sc, key=lambda v: sc[v])
        sel[ds], scores[ds] = win, sc
        L.append(f"| {ds} | **selected {win}** | scores: {', '.join(f'{k} {v:.5f}' for k, v in sc.items())} | | | |")
    p = os.path.join(SUITE, 'selected_lr.json')
    if os.path.exists(p):
        old = json.load(open(p))['selected']
        if old != sel:
            stop(f'{p} exists with a different selection {old} (new {sel}); not overwritten')
    else:
        json.dump({'stage': 'AD SF-de lr (validation only)', 'selected': sel, 'scores': scores}, open(p, 'x'), indent=1)
    L += ['', f'selected_lr.json: {json.dumps(sel)}']
    return os.path.join(SUITE, 'selection.md'), L


def stage_final():
    D = sfde_rows()
    sel = json.load(open(os.path.join(SUITE, 'selected_lr.json')))['selected']
    T = sat.load()
    tsel = {s: json.load(open(os.path.join(sat.TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
    mk, rk = sat.CONDKEY[sat.MUX], sat.CONDKEY[sat.REF]
    mux_t = lambda ds: sat.seeds(T, sat.MUX, ds, (tsel['res'][mk][ds], tsel['lr'][mk][ds], tsel['layers'][mk][ds]), 0)
    mux_d = lambda ds: sat.seeds(T, sat.MUX, ds, (sat.R0, sat.LR0, sat.L0), 0)
    sfsj = lambda ds: sat.seeds(T, sat.REF, ds, ('-', tsel['lr'][rk][ds], sat.L0), 0)
    sfde = lambda ds: three(D, sel[ds], ds, 0)
    for name, f in (('muxlin(tuned)', mux_t), ('muxlin(default)', mux_d), ('SF-sj(tuned)', sfsj), ('SF-de(tuned)', sfde)):
        for ds in DSS:
            if not sat.ok_rows(f(ds)):
                stop(f'{name} {ds}: missing run, non-finite AUC-PR or energy_check != ok')
    Q = [('E1 muxlin(tuned) vs SF-de(tuned)', mux_t, sfde, True), ('(report) muxlin(default) vs SF-de(tuned)', mux_d, sfde, True),
         ('(report) SF-de(tuned) vs SF-sj(tuned)', sfde, sfsj, False), ('(report) muxlin(tuned) vs SF-sj(tuned) (= 11-03 A1)', mux_t, sfsj, False)]
    L = ['# AD: SpkMixer (muxlin) vs SF-de', '', 'Rules: module docstring of summarize_adsf.py. Seeds 42 / 7 / 2026; test AUC-PR of the best-validation weights.', '',
         f'SF-de selected lr: {json.dumps(sel)}', '',
         '| comparison | R (AUC-PR, %) | pairs higher | verdict | energy A/B | paper bar (R > 0 and energy < 1) |', '|---|---:|---|---|---:|---|']
    det = ['', '| comparison | per-dataset mean r (seeds 42, 7, 2026) |', '|---|---|']
    for q, A, B, bar in Q:
        c = sat.compare(A, B)
        pb = ('met' if c['R'] > 0 and c['energy'] < 1 else 'not met') + f" (R {c['R']:+.3f}, energy {c['energy']:.4f})" if bar else ''
        L.append(f"| {q} | {c['R']:+.2f} | {c['hi']}/{c['n']} | {c['verdict']} | {c['energy']:.2f} | {pb} |")
        det.append(f"| {q.split(' (=')[0]} | {' ; '.join(c['cells'])} |")
    L += det
    for ds in DSS:
        L += ['', f'## {ds} (seed mean ± std; report)', '', '| model | lr | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch | time s (seeds) |',
              '|---|---|---:|---:|---:|---:|---:|---:|---|']
        for name, f in (('SF-de (tuned)', sfde), ('SF-sj (tuned)', sfsj), ('muxlin (tuned)', mux_t), ('muxlin (default)', mux_d)):
            rr = f(ds)
            ms = lambda k: f"{np.mean([100 * x[k] for x in rr]):.2f} ± {np.std([100 * x[k] for x in rr]):.2f}"
            L.append(f"| {name} | {rr[0]['lr']:g} | {ms('aucpr')} | {ms('aucroc')} | {ms('raw_f1')} | {ms('pa_f1')} | "
                     f"{np.mean([x['total_fft_uJ_corr'] for x in rr]):.2f} | {np.mean([x['best_epoch'] for x in rr]):.1f} | {', '.join('%d' % x['time_s'] for x in rr)} |")
    return os.path.join(SUITE, 'summary.md'), L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['select', 'final'], required=True); a = ap.parse_args()
    p, L = {'select': stage_select, 'final': stage_final}[a.stage]()
    with open(p, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', p); print('\n'.join(L))


if __name__ == '__main__':
    main()
