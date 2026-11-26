"""MODI 2026-11-27: pre-registered summary of the PTB order check (docs/MODI_PROJECT_LOG.md 2026-11-27 pre-registration).

Rows: results/<suite>/raw.txt (scripts/run_oc.sh). Fail closed as summarize_oc.py (every expected run once, epochs 30, no batch limit,
finite values, one sha, no SKIP).
PTB (token = 10 points x 15 leads, PPT's patch, 30 tokens) is judged; PTB30 (token = 30 points, 10 tokens) is reported only.
Metric: test macro-F1 (%) over the two classes; pairs = seeds 42 / 7 / 2026.
  (B) d_m = m_orig - m_shuf for m in {gru, tf} (shuf = tokens shuffled per sample in training, validation and test); V_B = the larger
      mean; strong V_B >= 3.0 and 3/3 positive; moderate V_B >= 1.0 and >= 2/3; weak otherwise.
  Reading: strong = PTB is a candidate task for the order claim (the condition set before the check); moderate = weak candidate;
  weak = not a candidate.
Report only: the same on accuracy, best original-order model - best shuffled model, PPT-style permuted test (models trained in the
original order, tested with shuffled tokens; PPT Table 3 reports PatchTST 78.6 -> 76.3 accuracy and PITS 82.0 -> 71.6 on PTB), PTB30.
usage (classification folder): python summarize_oc3.py [--suite flif-v1-oc3]
"""
import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ('PTB', 'PTB30')
CONDS = ('gru_orig', 'gru_shuf', 'tf_orig', 'tf_shuf')
SEEDS = (42, 7, 2026)


def kv(line):
    return dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)


def load(suite):
    lines = open(os.path.join(HERE, 'results', suite, 'raw.txt')).read().splitlines()
    sk = [l for l in lines if l.startswith('SKIP')]
    if sk:
        raise SystemExit(f'STOP: {len(sk)} SKIP lines: {sk[0][:160]}')
    T, shas = {}, set()
    for r in (kv(l) for l in lines if l.startswith('RESULT')):
        key = (r['ds'], r['cond'], int(r['seed']))
        if r['ds'] not in LEVELS or r['cond'] not in CONDS or key[2] not in SEEDS:
            raise SystemExit(f'STOP: unexpected row {key}')
        if key in T:
            raise SystemExit(f'STOP: duplicate {key}')
        if r['epochs'] != '30' or r['maxb'] != '0':
            raise SystemExit(f'STOP: {key} epochs {r["epochs"]} maxb {r["maxb"]}')
        v = {k: float(r[k]) for k in ('test_acc', 'test_mf1', 'ptest_acc', 'ptest_mf1', 'best_val', 'time_s', 'peak_mem_gb')}
        need = ['test_acc', 'test_mf1', 'best_val'] + (['ptest_acc', 'ptest_mf1'] if r['cond'].endswith('orig') else [])
        bad = [k for k in need if not math.isfinite(v[k])]
        if bad:
            raise SystemExit(f'STOP: {key} not finite {bad}')
        T[key] = v; shas.add(r['sha'])
    want = {(l, c, s) for l in LEVELS for c in CONDS for s in SEEDS}
    if set(T) != want:
        raise SystemExit(f'STOP: missing {sorted(want - set(T))[:3]}')
    if len(shas) != 1:
        raise SystemExit(f'STOP: code versions {sorted(shas)}')
    return T, shas.pop()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suite', default='flif-v1-oc3'); a = ap.parse_args()
    T, sha = load(a.suite)
    g = lambda l, c, k: np.array([T[(l, c, s)][k] for s in SEEDS])
    out, dec = [f'# PTB order check — {a.suite} (sha {sha})\n', 'Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-27. Test values in %, mean ± std over seeds 42 / 7 / 2026.\n'], {}
    for l in LEVELS:
        out += [f'## {l} ({"judged" if l == "PTB" else "report only"})', '', '| condition | macro-F1 | accuracy | permuted test macro-F1 / acc (report) |', '|---|---|---|---|']
        for c in CONDS:
            pt = f'{g(l, c, "ptest_mf1").mean():.2f} / {g(l, c, "ptest_acc").mean():.2f}' if c.endswith('orig') else '—'
            out.append(f'| {c} | {g(l, c, "test_mf1").mean():.2f} ± {g(l, c, "test_mf1").std():.2f} | {g(l, c, "test_acc").mean():.2f} ± {g(l, c, "test_acc").std():.2f} | {pt} |')
        res = {m: g(l, f'{m}_orig', 'test_mf1') - g(l, f'{m}_shuf', 'test_mf1') for m in ('gru', 'tf')}
        ms = max(res, key=lambda k: res[k].mean()); vb = float(res[ms].mean()); pos = int((res[ms] > 0).sum())
        v = 'strong' if vb >= 3.0 and pos == 3 else 'moderate' if vb >= 1.0 and pos >= 2 else 'weak'
        best = max(g(l, f'{m}_orig', 'test_mf1').mean() for m in ('gru', 'tf')) - max(g(l, f'{m}_shuf', 'test_mf1').mean() for m in ('gru', 'tf'))
        out += ['', '| test | d per seed | mean | positive |', '|---|---|---|---|']
        for m in ('gru', 'tf'):
            da = g(l, f'{m}_orig', 'test_acc') - g(l, f'{m}_shuf', 'test_acc')
            dp = g(l, f'{m}_orig', 'test_acc') - g(l, f'{m}_orig', 'ptest_acc')
            out.append(f'| (B) {m} orig − shuf, macro-F1 | {" / ".join(f"{x:+.2f}" for x in res[m])} | {res[m].mean():+.2f} | {int((res[m] > 0).sum())}/3 |')
            out.append(f'| {m} orig − shuf, accuracy (report) | {" / ".join(f"{x:+.2f}" for x in da)} | {da.mean():+.2f} | {int((da > 0).sum())}/3 |')
            out.append(f'| {m} test − permuted test, accuracy (PPT-style, report) | {" / ".join(f"{x:+.2f}" for x in dp)} | {dp.mean():+.2f} | |')
        out.append(f'| best orig − best shuf, macro-F1 (report) | | {best:+.2f} | |')
        reading = {'strong': 'candidate task for the order claim', 'moderate': 'weak candidate', 'weak': 'not a candidate'}[v]
        out += ['', f'**{l}: (B) {v} (V_B {vb:+.2f} %p, {ms}, {pos}/3){" → " + reading if l == "PTB" else ""}.**', '']
        dec[l] = dict(V_B=vb, model=ms, pos=pos, verdict=v, best_diff=float(best), d={k: x.tolist() for k, x in res.items()},
                      reading=reading if l == 'PTB' else 'report only')
    out.append('Run time / peak GPU memory: ' + ', '.join(f'{l} {np.mean([T[k]["time_s"] for k in T if k[0] == l]):.0f} s / {max(T[k]["peak_mem_gb"] for k in T if k[0] == l):.1f} GB' for l in LEVELS))
    txt = '\n'.join(out) + '\n'
    print(txt)
    open(os.path.join(HERE, 'results', a.suite, 'summary.md'), 'w').write(txt)
    json.dump(dec, open(os.path.join(HERE, 'results', a.suite, 'decision.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
