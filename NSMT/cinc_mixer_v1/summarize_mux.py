"""Verdicts for the fusion MUX of the SpkMixer (2026-10-30, PROJECT_LOG, exp/f-lif-pop-v3; CinC fold 0, seeds 42 / 7 / 2026, LIF,
OR residual). Pre-registered contrasts (verdict rule = summarize_rhythm.verdict: mean >= +0.5 %p and >= 2 of 3 seeds the same
direction = better; mirror = worse; else no clear difference; a missing run = incomplete):
  X1  muxor  - sdor     (i): majority vs OR over the same three bits F, R, s_d (same initial model)
  X2  muxlin - sdlin    (ii): MUX around the learned fusion (same initial model)
  X3  muxor - or, muxlin - muxor
  X4  muxor, muxlin vs the Spikformer-style model with criterion (2) (better, or mean >= -0.5 %p with energy <= 0.5x)
Controls spk_lif_{or,sdor,sdlin}_or come from the B1 suite (flif-v1-rhythm-spk-cinc). Criterion energy = summarize_spk.energy
(recorder total; the element-wise gates inside the fusion count 0, as the max of the OR fusion). Reported only: per-class F1,
macro-F1, energy. Writes results/<suite>/summary.md.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize_rhythm as S                                              # noqa: E402
from summarize_spk import SEEDS, NEED, energy, rows_of, complete, ms, val_score   # noqa: E402

NEW = ('spk_lif_muxor_or', 'spk_lif_muxlin_or')
CTRL = ('spk_lif_or_or', 'spk_lif_sdor_or', 'spk_lif_sdlin_or')
LABEL = {'spk_lif_or_or': 'or: F v R', 'spk_lif_sdor_or': 'sdor: F v R v s_d', 'spk_lif_sdlin_or': 'sdlin: L',
         'spk_lif_muxor_or': '(i) muxor: s_d ? F v R : F ^ R', 'spk_lif_muxlin_or': '(ii) muxlin: s_d ? L : F ^ R'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-rhythm-spk-mux-cinc')
    ap.add_argument('--ctrl_suite', default='flif-v1-rhythm-spk-cinc')
    ap.add_argument('--ref_suite', default='flif-v1-rhythm-cinc')
    a = ap.parse_args()
    R = os.path.join(HERE, 'results', a.suite)
    new = [r for r in S.parse(os.path.join(R, 'raw.txt')) if int(r.get('crop', 0)) == 0 and r['cond'] in NEW]
    ctrl = [r for r in S.parse(os.path.join(HERE, 'results', a.ctrl_suite, 'raw.txt')) if int(r.get('crop', 0)) == 0 and r['cond'] in CTRL]
    ref = [r for r in S.parse(os.path.join(HERE, 'results', a.ref_suite, 'raw.txt')) if r['cond'] == 'spk_ssa' and int(r.get('fold', 0)) == 0]
    for c in CTRL:
        if not complete(ctrl, c):
            raise SystemExit(f'control {c} incomplete in {a.ctrl_suite}')
    if len(ref) != 3:
        raise SystemExit(f'reference spk_ssa needs 3 runs, found {len(ref)}')
    rows = new + ctrl
    allrows = rows + ref
    e_ref = float(np.mean([float(r['total_fft_uJ']) for r in ref]))
    L = [f'# Fusion MUX `{a.suite}` — verdicts (LIF, residual OR, CinC fold 0)', '',
         f'Test score (mean F1 of N / AF / Other, %), best-validation weights, seeds {SEEDS}. Verdict: mean >= +0.5 %p and >= 2/3 seeds = better; '
         f'mirror = worse; else no clear difference. Spikformer-style `spk_ssa`: {ms([r["score"] for r in ref])}, {e_ref:.1f} µJ. Controls from `{a.ctrl_suite}` (B1).', '',
         '## Scores', '', '| fusion | n | validation | test score | macro-F1 | N F1 | AF F1 | Other F1 | Noisy F1 | total µJ (criterion) |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in CTRL + NEW:
        rr = rows_of(rows, c)
        if not rr:
            L.append(f'| {LABEL[c]} | 0 | — | — | — | — | — | — | — | — |'); continue
        L.append(f"| {LABEL[c]} | {len(rr)} | {ms([val_score(r) for r in rr])} | {ms([r['score'] for r in rr])} | {ms([r['macro_f1'] for r in rr])} | "
                 f"{ms([r['f1_0'] for r in rr])} | {ms([r['f1_1'] for r in rr])} | {ms([r['f1_2'] for r in rr])} | {ms([r['f1_3'] for r in rr])} | "
                 f"{np.mean([energy(r) for r in rr]):.1f} |")
    L.append(f"| Spikformer-style | 3 | — | {ms([r['score'] for r in ref])} | {ms([r['macro_f1'] for r in ref])} | {ms([r['f1_0'] for r in ref])} | "
             f"{ms([r['f1_1'] for r in ref])} | {ms([r['f1_2'] for r in ref])} | {ms([r['f1_3'] for r in ref])} | {e_ref:.1f} |")

    def vline(tag, A, B):
        v, d, m = S.verdict(allrows, A, B, SEEDS, NEED)
        return f'| {tag} | {A} − {B} | ' + (f'{m:+.2f} | {", ".join(f"{x:+.2f}" for x in d)} | {v} |' if v != 'incomplete' else 'incomplete | | incomplete |')

    L += ['', '## Pre-registered contrasts', '', '| question | contrast | mean %p | per seed (42, 7, 2026) | verdict |', '|---|---|---:|---|---|',
          vline('X1 (i) majority vs OR of F, R, s_d', 'spk_lif_muxor_or', 'spk_lif_sdor_or'),
          vline('X2 (ii) MUX around the learned fusion', 'spk_lif_muxlin_or', 'spk_lif_sdlin_or'),
          vline('X3a (i) vs spike OR fusion', 'spk_lif_muxor_or', 'spk_lif_or_or'),
          vline('X3b (ii) vs (i)', 'spk_lif_muxlin_or', 'spk_lif_muxor_or')]
    L += ['', '## X4 vs Spikformer-style (criterion (2))', '', '| condition | mean %p | per seed | verdict | energy / ref | criterion (2) |', '|---|---:|---|---|---:|---|']
    for c in NEW:
        v, d, m = S.verdict(allrows, c, 'spk_ssa', SEEDS, NEED)
        if v == 'incomplete':
            L.append(f'| {c} | incomplete | | incomplete | | incomplete |'); continue
        ratio = float(np.mean([energy(r) for r in rows_of(rows, c)])) / e_ref
        crit = 'met' if (v == 'better' or (m >= -0.5 and ratio <= 0.5)) else 'not met'
        L.append(f'| {c} | {m:+.2f} | {", ".join(f"{x:+.2f}" for x in d)} | {v} | {ratio:.2f}x | {crit} |')
    L += ['', '## New runs', '', '| condition | seed | validation | test score | macro-F1 | AF F1 | Other F1 | Noisy F1 | total µJ | best epoch | time s |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(new, key=lambda r: (r['cond'], int(r['seed']))):
        L.append(f"| {r['cond']} | {int(r['seed'])} | {100 * val_score(r):.2f} | {100 * float(r['score']):.2f} | {100 * float(r['macro_f1']):.2f} | "
                 f"{100 * float(r['f1_1']):.2f} | {100 * float(r['f1_2']):.2f} | {100 * float(r['f1_3']):.2f} | {energy(r):.1f} | "
                 f"{str(r.get('best_epoch_val', '')).split(':')[0]} | {float(r['time_s']):.0f} |")
    raw = os.path.join(R, 'raw.txt')
    skips = sum(1 for line in open(raw) if line.startswith('SKIP')) if os.path.exists(raw) else 0
    L += ['', f'RESULT lines (new): {len(new)}; SKIP lines: {skips}.']
    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
