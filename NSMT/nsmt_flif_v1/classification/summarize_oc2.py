"""MODI 2026-11-26 (evening): pre-registered summary of the second diagnostic batch (docs/MODI_PROJECT_LOG.md 2026-11-26 pre-registration).

Rows: results/<suite>/raw.txt (scripts/run_oc.sh). Fail closed as summarize_oc.py: every expected run exactly once, epochs 30, no
batch limit, finite values, one sha, no SKIP line.

SLEEPPSD (Sleep-EDF, token = the spectrum of one 30-s epoch; the front end the user chose for (나), ①). Metric test macro-F1, pairs =
folds 0-4. Same rules as SLEEPCTX in summarize_oc.py:
  (B) d_m = orig_L16 - shuf_L16 for m in {gru, tf}; V_B = the larger mean; strong V_B >= 3.0 and 5/5 positive; moderate V_B >= 1.0 and
      >= 4/5; weak otherwise.  (A) d = gru_orig_L16 - gru_orig_L1; needed: mean >= 2.0 and >= 4/5.  Passes if (B) strong / moderate or
      (A) needed. The SNN comparison (나) starts only if SLEEPPSD passes; otherwise the result goes to the user first.
MBIP (MIT-BIH, DS1 -> DS2, token = one beat: 117 points + its pre-RR; L = past beats + 1). Primary metric: test S F1 (%), the class
whose label depends on timing; pairs = seeds 42 / 7 / 2026.
  (A) accumulation: d = gru_orig_L17 - gru_orig_L1; needed: mean >= 2.0 and 3/3.
  (B) order: d_m = m_orig_L17 - m_shuf_L17; V_B = the larger mean; strong >= 3.0 and 3/3; moderate >= 1.0 and >= 2/3; weak otherwise.
  (C) two-number summary: d = gru_orig_L17 - gru_orig_L1_rr (current beat + pre-RR + mean pre-RR of the 16 beats before);
      "beaten": mean >= 2.0 and 3/3; otherwise "not beaten".
  Reading: order needed beyond the summary = (B) strong / moderate and (C) beaten; accumulation beyond the summary = (A) needed and
  (C) beaten, (B) weak; past helps but the summary is not beaten = (A) needed, (C) not beaten; otherwise past beats not shown to help.
Report only: the same differences on 4-class macro-F1 and V F1, S sensitivity / positive predictivity, S sensitivity in records 232 and
222, best original-order model - best shuffled model (post hoc in the first batch, reported here from the start), PPT-style permuted test,
and the SLEEPPSD vs CNN front end (flif-v1-oc SLEEPCTX) comparison.
usage (classification folder): python summarize_oc2.py [--suite flif-v1-oc2]
"""
import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MB_CONDS = ['gru_orig_L17', 'gru_shuf_L17', 'tf_orig_L17', 'tf_shuf_L17', 'gru_orig_L5', 'gru_orig_L2', 'gru_orig_L1', 'gru_orig_L1_rr']
PLAN = {'MBIP': (MB_CONDS, 'seed', [42, 7, 2026]),
        'SLEEPPSD': (['gru_orig_L16', 'gru_shuf_L16', 'tf_orig_L16', 'tf_shuf_L16', 'gru_orig_L4', 'gru_orig_L1'], 'fold', [0, 1, 2, 3, 4])}
NUM = ('test_acc', 'test_mf1', 'best_val', 'ptest_acc', 'ptest_mf1', 'params', 'time_s', 'peak_mem_gb')
MBX = ('S_se', 'S_pp', 'S_f1', 'V_se', 'V_pp', 'V_f1', 'S_se_232', 'S_se_222')


def kv(line):
    return dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)


def load(suite):
    lines = open(os.path.join(HERE, 'results', suite, 'raw.txt')).read().splitlines()
    skips = [l for l in lines if l.startswith('SKIP')]
    if skips:
        raise SystemExit(f'STOP: {len(skips)} SKIP lines, first: {skips[0][:200]}')
    table, shas = {}, set()
    for r in (kv(l) for l in lines if l.startswith('RESULT')):
        ds, cond = r['ds'], r['cond']
        if ds not in PLAN or cond not in PLAN[ds][0]:
            raise SystemExit(f'STOP: unexpected row {ds} {cond}')
        key = (ds, cond, int(r['fold']) if PLAN[ds][1] == 'fold' else int(r['seed']))
        if key in table:
            raise SystemExit(f'STOP: duplicate row {key}')
        if r['epochs'] != '30' or r['maxb'] != '0' or (PLAN[ds][1] == 'fold' and r['seed'] != '42'):
            raise SystemExit(f'STOP: {key} epochs {r["epochs"]} maxb {r["maxb"]} seed {r["seed"]}')
        keys = NUM + (MBX if ds == 'MBIP' else ())
        v = {k: float(r[k]) for k in keys}
        must = ['test_acc', 'test_mf1', 'best_val'] + (list(MBX) if ds == 'MBIP' else []) + (['ptest_acc', 'ptest_mf1'] if '_orig' in cond else [])
        bad = [k for k in must if not math.isfinite(v[k])]
        if bad:
            raise SystemExit(f'STOP: {key} not finite: {bad}')
        table[key] = v; shas.add(r['sha'])
    want = {(ds, c, u) for ds, (cs, _, us) in PLAN.items() for c in cs for u in us}
    if set(table) != want:
        raise SystemExit(f'STOP: {len(want - set(table))} expected runs missing, e.g. {sorted(want - set(table))[:3]}')
    if len(shas) != 1:
        raise SystemExit(f'STOP: code versions {sorted(shas)}')
    return table, shas.pop()


def pairs(table, ds, a, b, met):
    return np.array([table[(ds, a, u)][met] - table[(ds, b, u)][met] for u in PLAN[ds][2]])


def fmt(d):
    return ' / '.join(f'{v:+.2f}' for v in d)


def order_verdict(vb, pos, n):
    return 'strong' if vb >= 3.0 and pos == n else 'moderate' if vb >= 1.0 and pos >= n - 1 else 'weak'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suite', default='flif-v1-oc2'); a = ap.parse_args()
    table, sha = load(a.suite)
    out, dec = [f'# Second diagnostic batch — {a.suite} (sha {sha})\n',
                'Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-26 (evening). Values in %, mean ± std over seeds (MBIP) or folds '
                '(SLEEPPSD); differences in %p, paired.\n'], {'suite': a.suite, 'sha': sha}
    # ---- SLEEPPSD
    us = PLAN['SLEEPPSD'][2]; n = len(us)
    out.append('## SLEEPPSD (Sleep-EDF, epoch-spectrum tokens)\n\n| condition | macro-F1 | accuracy | permuted test MF1 (report) |\n|---|---|---|---|')
    for c in PLAN['SLEEPPSD'][0]:
        m = np.array([table[('SLEEPPSD', c, u)]['test_mf1'] for u in us]); ac = np.array([table[('SLEEPPSD', c, u)]['test_acc'] for u in us])
        pt = np.array([table[('SLEEPPSD', c, u)]['ptest_mf1'] for u in us])
        out.append(f'| {c} | {m.mean():.2f} ± {m.std():.2f} | {ac.mean():.2f} ± {ac.std():.2f} | {f"{pt.mean():.2f}" if "_orig" in c else "—"} |')
    res = {m: pairs(table, 'SLEEPPSD', f'{m}_orig_L16', f'{m}_shuf_L16', 'test_mf1') for m in ('gru', 'tf')}
    ms = max(res, key=lambda k: res[k].mean()); vb = float(res[ms].mean()); pos = int((res[ms] > 0).sum())
    vB = order_verdict(vb, pos, n)
    dA = pairs(table, 'SLEEPPSD', 'gru_orig_L16', 'gru_orig_L1', 'test_mf1'); dA4 = pairs(table, 'SLEEPPSD', 'gru_orig_L16', 'gru_orig_L4', 'test_mf1')
    vA = 'needed' if dA.mean() >= 2.0 and (dA > 0).sum() >= n - 1 else 'not shown'
    best = max(np.mean([table[('SLEEPPSD', f'{m}_orig_L16', u)]['test_mf1'] for u in us]) for m in ('gru', 'tf')) - \
        max(np.mean([table[('SLEEPPSD', f'{m}_shuf_L16', u)]['test_mf1'] for u in us]) for m in ('gru', 'tf'))
    passes = vB in ('strong', 'moderate') or vA == 'needed'
    out += ['', '| test | d per fold | mean | positive | verdict |', '|---|---|---|---|---|',
            f'| (B) gru orig − shuf | {fmt(res["gru"])} | {res["gru"].mean():+.2f} | {int((res["gru"] > 0).sum())}/{n} | |',
            f'| (B) tf orig − shuf | {fmt(res["tf"])} | {res["tf"].mean():+.2f} | {int((res["tf"] > 0).sum())}/{n} | **{vB}** (V_B {vb:+.2f}, {ms}) |',
            f'| (A) L16 − L1 | {fmt(dA)} | {dA.mean():+.2f} | {int((dA > 0).sum())}/{n} | **{vA}** |',
            f'| L16 − L4 (report) | {fmt(dA4)} | {dA4.mean():+.2f} | {int((dA4 > 0).sum())}/{n} | |',
            f'| best orig − best shuf (report) | | {best:+.2f} | | |',
            '', f'**SLEEPPSD passes: {"yes" if passes else "no"}** → (나) SNN comparison {"may start" if passes else "waits for the user"}.']
    ref = os.path.join(HERE, 'results', 'flif-v1-oc', 'raw.txt')
    if os.path.exists(ref):                                                          # report only: CNN front end, first batch
        rr = [kv(l) for l in open(ref) if l.startswith('RESULT') and ' ds=SLEEPCTX ' in l]
        cmp_ = []
        for c in ('gru_orig_L16', 'tf_orig_L16', 'gru_orig_L1'):
            v = [float(r['test_mf1']) for r in rr if r['cond'] == c]
            if len(v) == 5:
                cmp_.append(f'{c} CNN {np.mean(v):.2f} vs spectrum {np.mean([table[("SLEEPPSD", c, u)]["test_mf1"] for u in us]):.2f}')
        out.append('\nFront end (report only, macro-F1): ' + '; '.join(cmp_))
    dec['SLEEPPSD'] = dict(B=dict(V_B=vb, model=ms, pos=pos, n=n, verdict=vB, d={k: v.tolist() for k, v in res.items()}),
                           A=dict(V_A=float(dA.mean()), pos=int((dA > 0).sum()), n=n, verdict=vA, d=dA.tolist()), best_diff=float(best), passes=bool(passes))
    # ---- MBIP
    us = PLAN['MBIP'][2]; n = len(us)
    out.append('\n## MBIP (MIT-BIH DS1 → DS2, token = one beat)\n\n| condition | S F1 | S Se | S +P | V F1 | macro-F1 (4) | S Se 232 | S Se 222 | permuted test macro-F1 (report) |\n|---|---|---|---|---|---|---|---|---|')
    for c in MB_CONDS:
        g = lambda k: np.array([table[('MBIP', c, u)][k] for u in us])
        ptm = f'{g("ptest_mf1").mean():.2f}' if '_orig' in c else '—'
        out.append(f'| {c} | {g("S_f1").mean():.2f} ± {g("S_f1").std():.2f} | {g("S_se").mean():.2f} | {g("S_pp").mean():.2f} | {g("V_f1").mean():.2f} | '
                   f'{g("test_mf1").mean():.2f} ± {g("test_mf1").std():.2f} | {g("S_se_232").mean():.2f} | {g("S_se_222").mean():.2f} | {ptm} |')
    tests = {}
    dA = pairs(table, 'MBIP', 'gru_orig_L17', 'gru_orig_L1', 'S_f1')
    vA = 'needed' if dA.mean() >= 2.0 and (dA > 0).sum() == n else 'not shown'
    res = {m: pairs(table, 'MBIP', f'{m}_orig_L17', f'{m}_shuf_L17', 'S_f1') for m in ('gru', 'tf')}
    ms = max(res, key=lambda k: res[k].mean()); vb = float(res[ms].mean()); pos = int((res[ms] > 0).sum()); vB = order_verdict(vb, pos, n)
    dC = pairs(table, 'MBIP', 'gru_orig_L17', 'gru_orig_L1_rr', 'S_f1')
    vC = 'beaten' if dC.mean() >= 2.0 and (dC > 0).sum() == n else 'not beaten'
    if vB in ('strong', 'moderate') and vC == 'beaten':
        reading = 'order needed beyond the two-number summary'
    elif vA == 'needed' and vC == 'beaten':
        reading = 'accumulation beyond the summary; order not shown'
    elif vA == 'needed':
        reading = 'past beats help, but the two-number summary is not beaten'
    else:
        reading = 'past beats not shown to help'
    out += ['', '| test (S F1) | d per seed | mean | positive | verdict |', '|---|---|---|---|---|',
            f'| (A) gru L17 − L1 | {fmt(dA)} | {dA.mean():+.2f} | {int((dA > 0).sum())}/{n} | **{vA}** |',
            f'| (B) gru orig − shuf (L17) | {fmt(res["gru"])} | {res["gru"].mean():+.2f} | {int((res["gru"] > 0).sum())}/{n} | |',
            f'| (B) tf orig − shuf (L17) | {fmt(res["tf"])} | {res["tf"].mean():+.2f} | {int((res["tf"] > 0).sum())}/{n} | **{vB}** (V_B {vb:+.2f}, {ms}) |',
            f'| (C) gru L17 − L1 + two numbers | {fmt(dC)} | {dC.mean():+.2f} | {int((dC > 0).sum())}/{n} | **{vC}** |']
    for lab, a_, b_ in (('L17 − L5 (report)', 'gru_orig_L17', 'gru_orig_L5'), ('L5 − L2 (report)', 'gru_orig_L5', 'gru_orig_L2'),
                        ('L2 − L1 (report)', 'gru_orig_L2', 'gru_orig_L1'), ('L1 + two numbers − L1 (report)', 'gru_orig_L1_rr', 'gru_orig_L1')):
        d = pairs(table, 'MBIP', a_, b_, 'S_f1'); out.append(f'| {lab} | {fmt(d)} | {d.mean():+.2f} | {int((d > 0).sum())}/{n} | |')
    best = max(np.mean([table[('MBIP', f'{m}_orig_L17', u)]['S_f1'] for u in us]) for m in ('gru', 'tf')) - \
        max(np.mean([table[('MBIP', f'{m}_shuf_L17', u)]['S_f1'] for u in us]) for m in ('gru', 'tf'))
    out.append(f'| best orig − best shuf (report) | | {best:+.2f} | | |')
    out.append('\n| same tests on macro-F1 (4 classes), report only | (A) | (B) gru | (B) tf | (C) |\n|---|---|---|---|---|')
    mm = [pairs(table, 'MBIP', 'gru_orig_L17', 'gru_orig_L1', 'test_mf1'), pairs(table, 'MBIP', 'gru_orig_L17', 'gru_shuf_L17', 'test_mf1'),
          pairs(table, 'MBIP', 'tf_orig_L17', 'tf_shuf_L17', 'test_mf1'), pairs(table, 'MBIP', 'gru_orig_L17', 'gru_orig_L1_rr', 'test_mf1')]
    out.append('| mean d | ' + ' | '.join(f'{d.mean():+.2f} ({int((d > 0).sum())}/{n})' for d in mm) + ' |')
    out.append(f'\n**MBIP reading (pre-registered): {reading}.**')
    dec['MBIP'] = dict(A=dict(V=float(dA.mean()), pos=int((dA > 0).sum()), verdict=vA, d=dA.tolist()),
                       B=dict(V_B=vb, model=ms, pos=pos, verdict=vB, d={k: v.tolist() for k, v in res.items()}),
                       C=dict(V=float(dC.mean()), pos=int((dC > 0).sum()), verdict=vC, d=dC.tolist()), reading=reading, best_diff=float(best))
    out.append('\nRun time / peak GPU memory: ' + ', '.join(
        f'{ds} {np.mean([table[k]["time_s"] for k in table if k[0] == ds]):.0f} s / {max(table[k]["peak_mem_gb"] for k in table if k[0] == ds):.1f} GB' for ds in PLAN))
    txt = '\n'.join(out) + '\n'
    print(txt)
    open(os.path.join(HERE, 'results', a.suite, 'summary.md'), 'w').write(txt)
    json.dump(dec, open(os.path.join(HERE, 'results', a.suite, 'decision.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
