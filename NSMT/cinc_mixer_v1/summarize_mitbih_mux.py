"""Verdicts of the MIT-BIH extension of the fusion MUX (2026-11-01, PROJECT_LOG, exp/f-lif-pop-v3). SpkMixer LIF muxlin / sdlin with the
OR residual, seeds 42 / 7 / 2026, the other session's MIT-BIH protocol (run_mitbih2.sh: last-epoch weights, test macro-F1).
Rule (the other session's summarize_gen.py --mitbih, MODI_PROJECT_LOG 2903-2906): per-seed macro-F1 differences in %p, paired by seed;
better = mean >= +0.5 %p and A > B in >= 2 of 3 seeds, worse mirror, else no clear difference; criterion (2) = better, or
(mean >= -0.5 %p and energy A/B <= 0.5), energy = mean logged total_fft_uJ.
Pre-registered: M1 muxlin vs Spikformer-style (criterion 2), M2 muxlin vs sdlin, M3 sdlin vs Spikformer-style (criterion 2).
Spikformer-style rows = results/<suite>/baseline_spk_ssa_raw.txt (copy of the other session's 3 runs).
usage: python summarize_mitbih_mux.py [--suite flif-v1-mitbih-spkmux]  ->  results/<suite>/summary.md
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402

SEEDS = (42, 7, 2026)
REF, MUX, SDL = 'spk_ssa', 'spkmx_lif_muxlin_or', 'spkmx_lif_sdlin_or'
LAB = {REF: 'Spikformer-style (reused)', SDL: 'sdlin (L)', MUX: 'muxlin (s_d ? L : F ^ R)'}
CLS = ('N', 'S', 'V', 'F', 'Q')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suite', default='flif-v1-mitbih-spkmux'); a = ap.parse_args()
    R = os.path.join(HERE, 'results', a.suite)
    rows = [r for r in parse(os.path.join(R, 'raw.txt')) if r['cond'] in (MUX, SDL)] + \
           [r for r in parse(os.path.join(R, 'baseline_spk_ssa_raw.txt')) if r['cond'] == REF]
    D = {}
    for r in rows:
        k = (r['cond'], int(r['seed']))
        if k in D:
            raise SystemExit(f'duplicate {k}')
        D[k] = r
    ms = lambda v: f'{np.mean(v):.2f} ± {np.std(v):.2f}' if len(v) > 1 else (f'{v[0]:.2f}' if v else '—')
    L = [f'# MIT-BIH extension of the fusion MUX `{a.suite}` (LIF, residual OR; last-epoch weights, test set, seeds {SEEDS})', '',
         '| condition | n | macro-F1 % | accuracy % | ' + ' | '.join(f'F1 {c}' for c in CLS) + ' | total µJ | params | epochs | time s |',
         '|---|---:|---:|---:|' + '---:|' * len(CLS) + '---:|---:|---:|---:|']
    for c in (REF, SDL, MUX):
        rr = [D[(c, s)] for s in SEEDS if (c, s) in D]
        if not rr:
            L.append(f'| {LAB[c]} | 0 |' + ' |' * (len(CLS) + 6)); continue
        L.append(f"| {LAB[c]} | {len(rr)} | {ms([100 * r['f1'] for r in rr])} | {ms([100 * r['overall_acc'] for r in rr])} | "
                 + ' | '.join(f"{np.mean([100 * r[f'f1_{i}'] for r in rr]):.2f}" for i in range(len(CLS)))
                 + f" | {np.mean([r['total_fft_uJ'] for r in rr]):.2f} | {int(rr[0]['params'])} | {int(rr[0]['epochs_run'])} | {np.mean([r['time_s'] for r in rr]):.0f} |")
    L += ['', '## Pre-registered comparisons (macro-F1, %p, paired by seed)', '',
          '| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |', '|---|---:|---|---|---:|---|']
    for q, x, y, crit in (('M1 muxlin vs Spikformer-style', MUX, REF, True), ('M2 muxlin vs sdlin', MUX, SDL, False),
                          ('M3 sdlin vs Spikformer-style', SDL, REF, True)):
        if not all((x, s) in D and (y, s) in D for s in SEEDS):
            L.append(f'| {q} | incomplete | | incomplete | | |'); continue
        d = [100 * (D[(x, s)]['f1'] - D[(y, s)]['f1']) for s in SEEDS]
        m = float(np.mean(d))
        v = 'better' if m >= 0.5 and sum(z > 0 for z in d) >= 2 else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= 2 else 'no clear difference'
        er = float(np.mean([D[(x, s)]['total_fft_uJ'] for s in SEEDS])) / float(np.mean([D[(y, s)]['total_fft_uJ'] for s in SEEDS]))
        c2 = ('met' if v == 'better' or (m >= -0.5 and er <= 0.5) else 'not met') if crit else ''
        L.append(f"| {q} | {m:+.2f} | {', '.join(f'{z:+.2f}' for z in d)} | {v} | {er:.2f} | {c2} |")
    skips = sum(1 for l in open(os.path.join(R, 'raw.txt')) if l.startswith('SKIP')) if os.path.exists(os.path.join(R, 'raw.txt')) else 0
    L += ['', f'RESULT rows (new): {sum(1 for k in D if k[0] != REF)}; SKIP lines: {skips}; baseline rows: {sum(1 for k in D if k[0] == REF)}.']
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
