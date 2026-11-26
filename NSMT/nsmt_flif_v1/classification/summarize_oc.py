"""MODI 2026-11-26: pre-registered summary of the order / accumulation check (docs/MODI_PROJECT_LOG.md 2026-11-26 pre-registration).

Rows: results/<suite>/raw.txt, RESULT lines of scripts/run_oc.sh. Fail closed: every expected run must have exactly one RESULT line with
epochs 30, no batch limit, finite test values, one code version (sha); a SKIP line or anything else stops the summary.
Metric: SC test accuracy; MITBIH, SLEEPIN, SLEEPCTX test macro-F1 (accuracy reported). Differences in %p, paired by seed (SC, MITBIH:
42 / 7 / 2026) or fold (SLEEP*: 0-4).
(B) order, per level (SC, MITBIH, SLEEPIN, SLEEPCTX with L 16) and reference model m in {gru, tf}: d_m = orig - shuf.
    V_B = the larger of the two models' mean d; m* = that model; n = number of pairs, pos = pairs of m* with d > 0.
    strong: V_B >= 3.0 and pos = n; moderate (not strong): V_B >= 1.0 and pos >= n - 1; weak: otherwise.
(A) accumulation (gru, original order): SC d = acc(k 25 = gru_orig) - acc(k 10); SLEEPCTX d = MF1(L 16) - MF1(L 1).
    needed: mean d >= 2.0 and pos >= n (SC, n = 3) / pos >= n - 1 (SLEEPCTX, n = 5); otherwise not shown. k 5 and L 4 are reported.
A level passes the check if (B) is strong or moderate, or (A) is needed. MITBIH is the calibration case (order known to be weak in our SNN
runs, 11-19: -0.67 %p). Report only: PPT-style permuted test (models trained in the original order, tested with shuffled tokens).
usage (classification folder): python summarize_oc.py [--suite flif-v1-oc]
"""
import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = {'SLEEPCTX': (['gru_orig_L16', 'gru_shuf_L16', 'tf_orig_L16', 'tf_shuf_L16', 'gru_orig_L4', 'gru_orig_L1'], 'fold', [0, 1, 2, 3, 4]),
        'SLEEPIN': (['gru_orig', 'gru_shuf', 'tf_orig', 'tf_shuf'], 'fold', [0, 1, 2, 3, 4]),
        'SC': (['gru_orig', 'gru_shuf', 'tf_orig', 'tf_shuf', 'gru_orig_k5', 'gru_orig_k10'], 'seed', [42, 7, 2026]),
        'MITBIH': (['gru_orig', 'gru_shuf', 'tf_orig', 'tf_shuf'], 'seed', [42, 7, 2026])}
MET = {'SC': 'test_acc', 'MITBIH': 'test_mf1', 'SLEEPIN': 'test_mf1', 'SLEEPCTX': 'test_mf1'}


def kv(line):
    return dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)


def load(suite):
    lines = open(os.path.join(HERE, 'results', suite, 'raw.txt')).read().splitlines()
    skips = [l for l in lines if l.startswith('SKIP')]
    if skips:
        raise SystemExit(f'STOP: {len(skips)} SKIP lines, first: {skips[0][:200]}')
    rows = [kv(l) for l in lines if l.startswith('RESULT')]
    table, shas = {}, set()
    for r in rows:
        ds, cond = r['ds'], r['cond']
        if ds not in PLAN or cond not in PLAN[ds][0]:
            raise SystemExit(f'STOP: unexpected row {ds} {cond}')
        key = (ds, cond, int(r['fold']) if PLAN[ds][1] == 'fold' else int(r['seed']))
        if key in table:
            raise SystemExit(f'STOP: duplicate row {key}')
        if r['epochs'] != '30' or r['maxb'] != '0':
            raise SystemExit(f'STOP: {key} epochs {r["epochs"]} maxb {r["maxb"]}')
        if PLAN[ds][1] == 'fold' and r['seed'] != '42':
            raise SystemExit(f'STOP: {key} seed {r["seed"]}')
        vals = {k: float(r[k]) for k in ('test_acc', 'test_mf1', 'ptest_acc', 'ptest_mf1', 'best_val', 'params', 'time_s', 'peak_mem_gb')}
        for k in ('test_acc', 'test_mf1', 'best_val'):
            if not math.isfinite(vals[k]):
                raise SystemExit(f'STOP: {key} {k} = {vals[k]}')
        if '_orig' in cond and not (math.isfinite(vals['ptest_acc']) and math.isfinite(vals['ptest_mf1'])):
            raise SystemExit(f'STOP: {key} permuted test missing')
        table[key] = vals; shas.add(r['sha'])
    want = {(ds, c, u) for ds, (cs, _, us) in PLAN.items() for c in cs for u in us}
    if set(table) != want:
        raise SystemExit(f'STOP: {len(want - set(table))} expected runs missing, e.g. {sorted(want - set(table))[:3]}')
    if len(shas) != 1:
        raise SystemExit(f'STOP: code versions {sorted(shas)}')
    return table, shas.pop()


def pairs(table, ds, a, b, metric):
    us = PLAN[ds][2]
    return np.array([table[(ds, a, u)][metric] - table[(ds, b, u)][metric] for u in us])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suite', default='flif-v1-oc'); a = ap.parse_args()
    table, sha = load(a.suite)
    out, dec = [], {'suite': a.suite, 'sha': sha, 'levels': {}}
    out.append(f'# Order / accumulation check — {a.suite} (sha {sha})\n')
    out.append('Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-26. Values in % (SC accuracy, others macro-F1), mean ± std over '
               'seeds (SC, MIT-BIH) or folds (Sleep-EDF); differences in %p, paired.\n')
    out.append('## Scores\n\n| level | condition | metric | accuracy | permuted test (report only) |\n|---|---|---|---|---|')
    for ds, (cs, _, us) in PLAN.items():
        for c in cs:
            m = np.array([table[(ds, c, u)][MET[ds]] for u in us]); acc = np.array([table[(ds, c, u)]['test_acc'] for u in us])
            pm = np.array([table[(ds, c, u)]['ptest_' + MET[ds].split('_')[1]] for u in us])
            pt = f'{pm.mean():.2f} ± {pm.std():.2f}' if '_orig' in c else '—'
            out.append(f'| {ds} | {c} | {m.mean():.2f} ± {m.std():.2f} | {acc.mean():.2f} ± {acc.std():.2f} | {pt} |')
    out.append('\n## (B) order: orig − shuf\n\n| level | model | d per pair | mean d | positive |\n|---|---|---|---|---|')
    for ds, sfx in (('SC', ''), ('MITBIH', ''), ('SLEEPIN', ''), ('SLEEPCTX', '_L16')):
        n = len(PLAN[ds][2]); res = {}
        for m in ('gru', 'tf'):
            d = pairs(table, ds, f'{m}_orig{sfx}', f'{m}_shuf{sfx}', MET[ds])
            res[m] = d
            out.append(f'| {ds} | {m} | {" / ".join(f"{v:+.2f}" for v in d)} | {d.mean():+.2f} | {int((d > 0).sum())}/{n} |')
        ms = max(res, key=lambda k: res[k].mean()); vb = float(res[ms].mean()); pos = int((res[ms] > 0).sum())
        verdict = 'strong' if vb >= 3.0 and pos == n else 'moderate' if vb >= 1.0 and pos >= n - 1 else 'weak'
        dec['levels'][ds] = {'B': dict(V_B=vb, model=ms, pos=pos, n=n, verdict=verdict,
                                       d={k: v.tolist() for k, v in res.items()})}
    out.append('\n## (A) accumulation (gru, original order)\n\n| level | comparison | d per pair | mean d | positive |\n|---|---|---|---|---|')
    for ds, full, short, extra in (('SC', 'gru_orig', 'gru_orig_k10', 'gru_orig_k5'), ('SLEEPCTX', 'gru_orig_L16', 'gru_orig_L1', 'gru_orig_L4')):
        n = len(PLAN[ds][2])
        d = pairs(table, ds, full, short, MET[ds]); dx = pairs(table, ds, full, extra, MET[ds])
        pos = int((d > 0).sum()); need = n if ds == 'SC' else n - 1
        verdict = 'needed' if d.mean() >= 2.0 and pos >= need else 'not shown'
        out.append(f'| {ds} | {full} − {short} | {" / ".join(f"{v:+.2f}" for v in d)} | {d.mean():+.2f} | {pos}/{n} |')
        out.append(f'| {ds} | {full} − {extra} (report) | {" / ".join(f"{v:+.2f}" for v in dx)} | {dx.mean():+.2f} | {int((dx > 0).sum())}/{n} |')
        dec['levels'][ds]['A'] = dict(V_A=float(d.mean()), pos=pos, n=n, verdict=verdict, d=d.tolist(), d_report=dx.tolist())
    out.append('\n## PPT-style permuted test (report only): test − permuted test, models trained in the original order\n')
    out.append('| level | model | drop |\n|---|---|---|')
    for ds, sfx in (('SC', ''), ('MITBIH', ''), ('SLEEPIN', ''), ('SLEEPCTX', '_L16')):
        k = MET[ds].split('_')[1]
        for m in ('gru', 'tf'):
            v = np.array([table[(ds, f'{m}_orig{sfx}', u)][MET[ds]] - table[(ds, f'{m}_orig{sfx}', u)]['ptest_' + k] for u in PLAN[ds][2]])
            out.append(f'| {ds} | {m} | {v.mean():+.2f} ± {v.std():.2f} |')
    out.append('\n## Verdicts (pre-registered rules)\n\n| level | (B) order | (A) accumulation | passes |\n|---|---|---|---|')
    for ds, L in dec['levels'].items():
        b = L['B']; aa = L.get('A')
        p = b['verdict'] in ('strong', 'moderate') or (aa is not None and aa['verdict'] == 'needed')
        L['passes'] = bool(p)
        bs = f'{b["verdict"]} (V_B {b["V_B"]:+.2f} %p, {b["model"]}, {b["pos"]}/{b["n"]})'
        As = f'{aa["verdict"]} (V_A {aa["V_A"]:+.2f} %p, {aa["pos"]}/{aa["n"]})' if aa else '—'
        out.append(f'| {ds} | {bs} | {As} | {"yes" if p else "no"}{" (calibration)" if ds == "MITBIH" else ""} |')
    out.append('\nRun time / peak GPU memory: ' + ', '.join(
        f'{ds} {np.mean([table[k]["time_s"] for k in table if k[0] == ds]):.0f} s / {max(table[k]["peak_mem_gb"] for k in table if k[0] == ds):.1f} GB'
        for ds in PLAN))
    txt = '\n'.join(out) + '\n'
    print(txt)
    with open(os.path.join(HERE, 'results', a.suite, 'summary.md'), 'w') as f:
        f.write(txt)
    with open(os.path.join(HERE, 'results', a.suite, 'decision.json'), 'w') as f:
        json.dump(dec, f, indent=1)


if __name__ == '__main__':
    main()
