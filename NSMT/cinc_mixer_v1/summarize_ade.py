"""AD energy test (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG AD energy pre-registration; user request 11-19 "AD energy를 낮추는 실험을
하자", decisions "(가) (나) 둘 다" and "energy 계산: 기존 syops 라이브러리의 규칙을 그대로"). Suite flif-v1-ad-energy (scripts/make_queue_ade.sh).
  (A) mean pooling: the 11-03 tuned muxlin settings (MSL OR / 2e-3 / M 2, SMAP ADD / 2e-3 / M 1, PSM ADD / 1e-3 / M 3) with the T_h = 4 repeats
      averaged before the mixer (--spk_mixpool mean, tag _mpmean), whole training file.
  (B) one mixer layer per direction (M = 1) for every dataset, re-selected on validation in the 11-03 order without the layer stage:
      lr {1e-4, 2.5e-4, 5e-4*, 1e-3, 2e-3} with OR, then residual {OR*, IAND, ADD} at the selected lr (* = default).
      Selection (as summarize_ad_tune.py): score = seed-mean best validation MSE / that of the default (holdout: first 80 % trained,
      last 20 % validated, no test pass); the lowest wins, ties (< 1e-12) keep the default; a missing run or a SKIP line -> no
      selection; selected_<stage>.json is never overwritten with a different selection.
      "res" = the selected residual; "spike" = the best of {OR*, IAND} (0/1 spikes only, the residuals syops counts as AC); the
      spike arm gets its own final runs only where "res" is ADD.
Final comparisons (test AUC-PR of the best-validation weights; summarize_ad_tune.compare: per seed r = 100 (A - B) / B, R = mean over
MSL / SMAP / PSM of the seed-mean r; better = R >= +2 % and >= 6 of 9 pairs higher, worse = mirror):
  G1 (B) M1(tuned) vs SF-de(tuned)              verdict + paper bar
  G2 (B) M1(tuned) vs muxlin(tuned, 11-03)      verdict + energy ratio
  G3 (A) mean vs SF-de(tuned)                   verdict + paper bar
  G4 (A) mean vs muxlin(tuned, 11-03)           verdict + energy ratio
  G5 (B) M1 0/1 residual vs SF-de(tuned)        verdict + paper bar (only where it differs from G1)
  report: muxlin(tuned) / muxlin(default) vs SF-de; reproducibility of runs whose setting already ran on 11-03 (same values expected).
Energy (user 11-19): the syops library rule as published (github.com/iCGY96/syops-counter HEAD 4232f77 = installed syops/ops.py):
total_fft_uJ, syops' own count in the run (a Linear / BN input with <= 2 distinct values in [0, 1] -> AC x rate, otherwise MAC; one AC per
neuron input element) + the f-LIF-paper neuron energy. The rule used until 11-19 (total_fft_uJ_corr: ADD body as integer accumulates,
averaged head input as T_h binary steps) is reported next to it. Energy A/B = mean over the 3 datasets of the ratio of seed means.
Paper bar (user 11-18, sign only): R > 0 and energy A/B < 1 (syops rule).
Fail closed (exit 2, nothing written): a SKIP line in the suite, a missing / duplicate run, a non-finite value, energy_check != ok.
usage: python -B summarize_ade.py --stage lr|res|final      (ADE_DRY=<folder>: read the suite from that folder; dry runs)
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
import summarize_adsf as sa                                                               # noqa: E402

SUITE = os.environ.get('ADE_DRY') or os.path.join(HERE, 'results', 'flif-v1-ad-energy')
DSS, SEEDS = sat.DSS, sat.SEEDS
LRS, LR0, R0 = ('0.0001', '0.00025', '0.0005', '0.001', '0.002'), '0.0005', 'or'
RESS = ('or', 'iand', 'add')
MK = sat.CONDKEY[sat.MUX]
EK = {'syops': 'total_fft_uJ', 'current': 'total_fft_uJ_corr'}


def stop(msg):
    print('STOP:', msg)
    sys.exit(2)


def load():
    raw = os.path.join(SUITE, 'raw.txt')
    if not os.path.exists(raw):
        stop(f'no {raw}')
    if any(l.startswith('SKIP') for l in open(raw)):
        stop(f'SKIP line(s) in {raw}')
    D = {}
    for r in parse(raw):
        if not r['cond'].startswith('spkmx_lif_muxlin_'):
            stop(f'unexpected condition {r["cond"]}')
        pool = 'mean' if r['tag'].endswith('_mpmean') else 'none'
        k = (r['cond'].split('_')[3], r['ds'], float(r['lr']), int(r['layers']), int(r['holdout']), int(r['seed']), pool)
        if k in D:
            stop(f'duplicate run {k}')
        D[k] = r
    return D


def three(D, res, ds, lr, layers, ho, pool='none'):
    v = [D.get((res, ds, float(lr), int(layers), ho, s, pool)) for s in SEEDS]
    if any(x is None for x in v):
        stop(f'missing run {res} {ds} lr {lr} M {layers} holdout {ho} pool {pool}: seeds {[s for s, x in zip(SEEDS, v) if x is None]}')
    for x in v:
        if ho and not math.isfinite(float(x['val_mse_best'])):
            stop(f'non-finite validation MSE {x["tag"]} {ds}')
    return v


def write_sel(stage, sel, scores):
    p = os.path.join(SUITE, f'selected_{stage}.json')
    if os.path.exists(p):
        old = json.load(open(p))['selected']
        if old != sel:
            stop(f'{p} exists with a different selection {old} (new {sel}); not overwritten')
        return
    json.dump({'stage': f'AD energy (B) {stage} (validation only)', 'selected': sel, 'scores': scores}, open(p, 'x'), indent=1)


def pick(sc, dflt):
    best = min(sc.values())
    return dflt if abs(sc[dflt] - best) < 1e-12 else min(sc, key=lambda v: sc[v])


def vrow(L, ds, cand, rr, m, sc):
    L.append(f"| {ds} | {cand} | {', '.join('%.6f' % float(x['val_mse_best']) for x in rr)} | {m:.6f} | {sc:.5f} | "
             f"{np.mean([x['best_epoch'] for x in rr]):.1f} |")


def stage_lr(D):
    L = ['# AD energy test (B), M = 1: lr selection (validation only, OR residual)', '',
         'Score = seed-mean best validation MSE / that at lr 5e-4 (holdout: first 80 % trained, last 20 % validated; lower is better).', '',
         '| dataset | lr | val MSE per seed (42, 7, 2026) | mean | score | best epoch (mean) |', '|---|---|---|---:|---:|---:|']
    sel, scores = {}, {}
    for ds in DSS:
        m = {lr: float(np.mean([float(x['val_mse_best']) for x in three(D, R0, ds, lr, 1, 1)])) for lr in LRS}
        sc = {lr: m[lr] / m[LR0] for lr in LRS}
        for lr in LRS:
            vrow(L, ds, lr, three(D, R0, ds, lr, 1, 1), m[lr], sc[lr])
        sel[ds], scores[ds] = pick(sc, LR0), sc
        L.append(f'| {ds} | **selected {sel[ds]}** | | | | |')
    write_sel('lr', sel, scores)
    return os.path.join(SUITE, 'selection_lr.md'), L + ['', f'selected_lr.json: {json.dumps(sel)}']


def stage_res(D):
    lr = json.load(open(os.path.join(SUITE, 'selected_lr.json')))['selected']
    L = ['# AD energy test (B), M = 1: residual selection (validation only) at the selected lr', '',
         'Score = seed-mean best validation MSE / that of OR. "res" = lowest of {OR, IAND, ADD}; "spike" = lowest of {OR, IAND} '
         '(0/1 residuals); ties keep OR.', '',
         '| dataset | residual | val MSE per seed (42, 7, 2026) | mean | score | best epoch (mean) |', '|---|---|---|---:|---:|---:|']
    sel, scores = {'res': {}, 'spike': {}}, {}
    for ds in DSS:
        m = {r: float(np.mean([float(x['val_mse_best']) for x in three(D, r, ds, lr[ds], 1, 1)])) for r in RESS}
        sc = {r: m[r] / m[R0] for r in RESS}
        for r in RESS:
            vrow(L, ds, f'{r} (lr {lr[ds]})', three(D, r, ds, lr[ds], 1, 1), m[r], sc[r])
        sel['res'][ds], sel['spike'][ds] = pick(sc, R0), pick({r: sc[r] for r in ('or', 'iand')}, R0)
        scores[ds] = sc
        L.append(f"| {ds} | **selected res {sel['res'][ds]}, spike {sel['spike'][ds]}** | | | | |")
    write_sel('res', sel, scores)
    return os.path.join(SUITE, 'selection_res.md'), L + ['', f'selected_res.json: {json.dumps(sel)}']


def erows(f, key):
    return lambda ds: [dict(r, total_fft_uJ_corr=float(r[key])) for r in f(ds)]


def emean(f, key):
    return float(np.mean([np.mean([float(r[key]) for r in f(ds)]) for ds in DSS]))


def stage_final(D):
    lr = json.load(open(os.path.join(SUITE, 'selected_lr.json')))['selected']
    sel = json.load(open(os.path.join(SUITE, 'selected_res.json')))['selected']
    T = sat.load()
    tsel = {s: json.load(open(os.path.join(sat.TUNE, f'selected_{s}.json')))['selected'][MK] for s in ('lr', 'layers', 'res')}
    SD = sa.sfde_rows()
    sfsel = json.load(open(os.path.join(sa.SUITE, 'selected_lr.json')))['selected']
    B1 = lambda ds: three(D, sel['res'][ds], ds, lr[ds], 1, 0)
    B2 = lambda ds: three(D, sel['spike'][ds], ds, lr[ds], 1, 0)
    A = lambda ds: three(D, tsel['res'][ds], ds, tsel['lr'][ds], tsel['layers'][ds], 0, 'mean')
    mux_t = lambda ds: sat.seeds(T, sat.MUX, ds, (tsel['res'][ds], tsel['lr'][ds], tsel['layers'][ds]), 0)
    mux_d = lambda ds: sat.seeds(T, sat.MUX, ds, (sat.R0, sat.LR0, sat.L0), 0)
    sfde = lambda ds: sa.three(SD, sfsel[ds], ds, 0)
    arms = {'(A) mean': A, '(B) M1': B1, '(B) M1 0/1 residual': B2, 'muxlin(tuned, 11-03)': mux_t, 'muxlin(default)': mux_d, 'SF-de(tuned)': sfde}
    for name, f in arms.items():
        for ds in DSS:
            rr = f(ds)
            if not sat.ok_rows(rr) or not all(math.isfinite(float(x[k])) and float(x[k]) > 0 for x in rr for k in EK.values()):
                stop(f'{name} {ds}: missing run, non-finite AUC-PR / energy or energy_check != ok')
    g5 = any(sel['spike'][ds] != sel['res'][ds] for ds in DSS)
    Q = [('G1 (B) M1(tuned) vs SF-de(tuned)', B1, sfde, True), ('G2 (B) M1(tuned) vs muxlin(tuned, 11-03)', B1, mux_t, False),
         ('G3 (A) mean vs SF-de(tuned)', A, sfde, True), ('G4 (A) mean vs muxlin(tuned, 11-03)', A, mux_t, False)]
    if g5:
        Q.append(('G5 (B) M1 0/1 residual vs SF-de(tuned)', B2, sfde, True))
    Q += [('(report) muxlin(tuned, 11-03) vs SF-de(tuned)', mux_t, sfde, True), ('(report) muxlin(default) vs SF-de(tuned)', mux_d, sfde, True)]
    cfg = {'(A) mean': {ds: f"{tsel['res'][ds]} / lr {tsel['lr'][ds]:g} / M {tsel['layers'][ds]} / mean" for ds in DSS},
           '(B) M1': {ds: f"{sel['res'][ds]} / lr {lr[ds]} / M 1" for ds in DSS},
           '(B) M1 0/1 residual': {ds: f"{sel['spike'][ds]} / lr {lr[ds]} / M 1" for ds in DSS}}
    L = ['# AD energy test: (A) mean pooling, (B) one mixer layer per direction', '',
         'Rules and sources: module docstring of summarize_ade.py. Seeds 42 / 7 / 2026; test AUC-PR of the best-validation weights.', '',
         'Settings: ' + json.dumps(cfg), '', f'G5 run: {"yes" if g5 else "no (the selected residual is already 0/1 on every dataset; G5 = G1)"}', '',
         '| comparison | R (AUC-PR, %) | pairs higher | verdict | energy A/B syops | paper bar (syops) | energy A/B current rule | paper bar (current rule) |',
         '|---|---:|---|---|---:|---|---:|---|']
    det = ['', '| comparison | per-dataset mean r (seeds 42, 7, 2026) | energy A/B per dataset, syops (current) |', '|---|---|---|']
    for q, X, Y, bar in Q:
        c1, c0 = sat.compare(erows(X, EK['syops']), erows(Y, EK['syops'])), sat.compare(erows(X, EK['current']), erows(Y, EK['current']))
        if c1 is None or c0 is None:
            stop(f'{q}: incomplete')
        pb = lambda c: ('met' if c['R'] > 0 and c['energy'] < 1 else 'not met') if bar else ''
        L.append(f"| {q} | {c1['R']:+.3f} | {c1['hi']}/{c1['n']} | {c1['verdict']} | {c1['energy']:.4f} | {pb(c1)} | {c0['energy']:.4f} | {pb(c0)} |")
        per = '; '.join(f"{ds} {np.mean([float(r[EK['syops']]) for r in X(ds)]) / np.mean([float(r[EK['syops']]) for r in Y(ds)]):.3f} "
                        f"({np.mean([float(r[EK['current']]) for r in X(ds)]) / np.mean([float(r[EK['current']]) for r in Y(ds)]):.3f})" for ds in DSS)
        det.append(f"| {q.split(' (')[0] if q.startswith('G') else q} | {' ; '.join(c1['cells'])} | {per} |")
    L += det
    for ds in DSS:
        L += ['', f'## {ds} (seed mean ± std; report)', '',
              '| arm | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ syops | energy µJ current rule | best epoch | time s (seeds) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for name, f in arms.items():
            if name == '(B) M1 0/1 residual' and not g5:
                continue
            rr = f(ds)
            ms = lambda k: f"{np.mean([100 * float(x[k]) for x in rr]):.2f} ± {np.std([100 * float(x[k]) for x in rr]):.2f}"
            L.append(f"| {name} | {ms('aucpr')} | {ms('aucroc')} | {ms('raw_f1')} | {ms('pa_f1')} | {np.mean([float(x[EK['syops']]) for x in rr]):.2f} | "
                     f"{np.mean([float(x[EK['current']]) for x in rr]):.2f} | {np.mean([x['best_epoch'] for x in rr]):.1f} | "
                     f"{', '.join('%d' % x['time_s'] for x in rr)} |")
    L += ['', '## Reproducibility (report): runs of this suite whose setting also ran on 11-03 (flif-v1-ad-spkmux-tune)', '',
          '| setting | holdout | seed | value here | value 11-03 | same |', '|---|---|---|---|---|---|']
    nrep = 0
    for k, r in sorted(D.items(), key=lambda kv: str(kv[0])):
        res, ds, lr_, lay, ho, s, pool = k
        if pool != 'none':
            continue
        o = T.get((sat.MUX, res, ds, lr_, lay, ho, s))
        if o is None:
            continue
        key = 'val_mse_best' if ho else 'aucpr'
        a, b = float(r[key]), float(o[key])
        nrep += 1
        L.append(f'| {res} {ds} lr {lr_:g} M {lay} | {ho} | {s} | {key} {a:.6f} | {b:.6f} | {"yes" if a == b else "NO"} |')
    if not nrep:
        L.append('| (none) | | | | | |')
    return os.path.join(SUITE, 'summary.md'), L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['lr', 'res', 'final'], required=True); a = ap.parse_args()
    D = load()
    p, L = {'lr': stage_lr, 'res': stage_res, 'final': stage_final}[a.stage](D)
    open(p, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'-> {p}')


if __name__ == '__main__':
    main()
