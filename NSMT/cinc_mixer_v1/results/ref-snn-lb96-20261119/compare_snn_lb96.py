"""SNN forecasting models at look-back 96 on ETT, from the NSMT and SpikF papers, vs the current SpkMixer (2026-11-19, exp/f-lif-pop-v3;
PROJECT_LOG; user request 11-19 "NSMT 원문 혹은 SpikF 원문에서 얻을 수 있는 SNN 시계열 모델의 Look-back window=96 설정의 ETT 실험 값들을 현재
SpikMixer와 비교해줘"). No training.
Paper values (test MSE / MAE, input 96), transcribed below:
  NSMT  Table 14 "Full comparisons on SNN-based models" (SpikF, SpikeRNN, iSpikformer, NSMT); text: "we use the same look-back window size
        L = 96 for all methods ... All results are taken from [43] and [41]" (NSMT's own column is its result).
  SpikF Table 12 "More benchmarks on ETT dataset" (SpikF, SpikeRNN, iSpikformer; text: "SNN-based models are equipped with SPE");
        Table 1 caption: "All baseline models are configured with a uniform look-back window of L = 96".
  --verify <nsmt_layout.txt> <spikf_layout.txt> (pdftotext -layout of the two PDFs, kept local) re-reads both tables from the text and
  checks every transcribed number; the three models present in both tables must agree.
Ours: LIF SpkMixer (tuned: muxlin, IAND, 1 mixer layer, lr ETTh2 5e-4 / ETTm2 1e-4) and SF-de, input 96, seeds 42 / 7 / 2026, test values of
the best-validation weights, loaded with summarize_pn's loaders and checks. Our runs cover ETTh2 / ETTm2 x H96 / H720 only (user decision 12).
Per paper model: R = mean over the 4 cells of 100 (ours - paper) / paper (negative = ours lower = better), cells lower, 4-cell means, and the
difference of the 4-cell mean MSEs in % (the measure of research_notes/SpikMixer 선행 연구 조사/spiking_time_series.md; it weights the H720 cells more).
usage (from cinc_mixer_v1/): python -B results/ref-snn-lb96-20261119/compare_snn_lb96.py [--verify NSMT.txt SpikF.txt]
"""
import os
import re
import sys

import numpy as np

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
import summarize_pn as sp                                                          # noqa: E402
from summarize_l2 import CELLS, SEEDS                                              # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
DSO, HO = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'), ('96', '192', '336', '720', 'Avg')
# (ds, H) -> model -> (MSE, MAE); NSMT Table 14
P = {('ETTh2', 96): {'SpikF': (0.290, 0.336), 'SpikeRNN': (0.295, 0.335), 'iSpikformer': (0.308, 0.349), 'NSMT': (0.288, 0.334)},
     ('ETTh2', 720): {'SpikF': (0.416, 0.436), 'SpikeRNN': (0.428, 0.438), 'iSpikformer': (0.438, 0.439), 'NSMT': (0.409, 0.429)},
     ('ETTm2', 96): {'SpikF': (0.175, 0.251), 'SpikeRNN': (0.176, 0.251), 'iSpikformer': (0.178, 0.256), 'NSMT': (0.175, 0.251)},
     ('ETTm2', 720): {'SpikF': (0.405, 0.397), 'SpikeRNN': (0.409, 0.396), 'iSpikformer': (0.406, 0.398), 'NSMT': (0.399, 0.393)}}
MODELS = ('NSMT', 'SpikF', 'SpikeRNN', 'iSpikformer')


def table_rows(path, title, n):
    """rows of n numbers after the line containing title, in order ETTh1 / ETTh2 / ETTm1 / ETTm2 x 96 / 192 / 336 / 720 / Avg."""
    L = open(path, encoding='utf-8').read().splitlines()
    i = next(k for k, l in enumerate(L) if title in l)
    rx = re.compile(r'(?:^|\s)(96|192|336|720|Avg)\s+((?:\d\.\d{3}\s+){%d}\d\.\d{3})' % (n - 1))
    rows = []
    for l in L[i:i + 120]:
        m = rx.search(l)
        if m:
            rows.append((m.group(1), [float(x) for x in m.group(2).split()]))
        if len(rows) == 20:
            break
    assert [h for h, _ in rows] == list(HO) * 4, [h for h, _ in rows]
    return {(DSO[k // 5], rows[k][0]): rows[k][1] for k in range(20)}


def verify(nsmt, spikf):
    bad = 0
    T14 = table_rows(nsmt, 'Table 14: Full comparisons on SNN-based models', 8)      # SpikF, SpikeRNN, iSpikformer, NSMT
    T12 = table_rows(spikf, 'Table 12. More benchmarks on ETT dataset', 10)          # SpikF, SpikeRNN, iSpikformer, FITS, FEDformer
    for (ds, H), mm in P.items():
        a, b = T14[(ds, str(H))], T12[(ds, str(H))]
        for j, m in enumerate(('SpikF', 'SpikeRNN', 'iSpikformer', 'NSMT')):
            if tuple(a[2 * j:2 * j + 2]) != mm[m]:
                print(f'MISMATCH NSMT T14 {ds} H{H} {m}: {a[2 * j:2 * j + 2]} vs {mm[m]}'); bad += 1
        for j, m in enumerate(('SpikF', 'SpikeRNN', 'iSpikformer')):
            if tuple(b[2 * j:2 * j + 2]) != mm[m]:
                print(f'MISMATCH SpikF T12 {ds} H{H} {m}: {b[2 * j:2 * j + 2]} vs {mm[m]}'); bad += 1
    same = all(T14[k][:6] == T12[k][:6] for k in T14)
    print(f'verify: transcription mismatches {bad}; NSMT T14 = SpikF T12 for SpikF / SpikeRNN / iSpikformer on all 20 rows: {same}')
    return bad == 0 and same, T14


def ours(T):
    return {(ds, pl): {k: [float(T[(ds, pl, s)][k]) for s in SEEDS] for k in ('mse', 'mae')} for ds, pl in CELLS}


ok, T14 = (True, None)
if len(sys.argv) > 1 and sys.argv[1] == '--verify':
    ok, T14 = verify(sys.argv[2], sys.argv[3])
    if not ok:
        sys.exit(1)
O = {'LIF SpkMixer (ours)': ours(sp.arms_ett_one('LIF')), 'SF-de (ours)': ours(sp.load_sfde_ett())}
L = ['# SNN models at look-back 96 on ETT (NSMT Table 14 / SpikF Table 12) vs the current SpkMixer', '',
     'Rules and sources: module docstring of compare_snn_lb96.py. Ours: seed mean ± std (42 / 7 / 2026).',
     ('Transcription verified against both papers\' text (--verify).' if T14 else 'Transcription not verified in this run.'), '',
     '## Test MSE / MAE', '', '| model | ' + ' | '.join(f'{ds} H{pl} MSE / MAE' for ds, pl in CELLS) + ' | 4-cell mean MSE / MAE |', '|---|' + '---|' * 5]
for n, v in O.items():
    cells = [f"{np.mean(v[c]['mse']):.4f} ± {np.std(v[c]['mse']):.4f} / {np.mean(v[c]['mae']):.4f}" for c in CELLS]
    L.append(f'| {n} | ' + ' | '.join(cells) + f" | {np.mean([np.mean(v[c]['mse']) for c in CELLS]):.4f} / {np.mean([np.mean(v[c]['mae']) for c in CELLS]):.4f} |")
for m in MODELS:
    L.append(f'| {m} | ' + ' | '.join(f'{P[c][m][0]:.3f} / {P[c][m][1]:.3f}' for c in CELLS) +
             f' | {np.mean([P[c][m][0] for c in CELLS]):.4f} / {np.mean([P[c][m][1] for c in CELLS]):.4f} |')
L += ['', '## Ours vs each paper model (R = mean of the 4 per-cell relative differences; negative = ours better)', '',
      '| ours | vs | R MSE % | cells lower (MSE) | R MAE % | cells lower (MAE) | 4-cell mean MSE difference % | per-cell MSE % (ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720) |',
      '|---|---|--:|--:|--:|--:|--:|---|']
for n, v in O.items():
    for m in MODELS:
        r = {k: [100 * (np.mean(v[c][k]) - P[c][m][i]) / P[c][m][i] for c in CELLS] for i, k in enumerate(('mse', 'mae'))}
        dm = 100 * (np.mean([np.mean(v[c]['mse']) for c in CELLS]) - np.mean([P[c][m][0] for c in CELLS])) / np.mean([P[c][m][0] for c in CELLS])
        L.append(f"| {n} | {m} | {np.mean(r['mse']):+.2f} | {sum(x < 0 for x in r['mse'])}/4 | {np.mean(r['mae']):+.2f} | "
                 f"{sum(x < 0 for x in r['mae'])}/4 | {dm:+.2f} | {', '.join(f'{x:+.2f}' for x in r['mse'])} |")
L += ['', '## Seeds of the LIF SpkMixer that beat each paper value (MSE, per cell; report)', '', '| cell | seeds 42 / 7 / 2026 | ' + ' | '.join(MODELS) + ' |', '|---|---|' + '--:|' * 4]
v = O['LIF SpkMixer (ours)']
for c in CELLS:
    L.append(f"| {c[0]} H{c[1]} | {', '.join('%.4f' % x for x in v[c]['mse'])} | " + ' | '.join(f"{sum(x < P[c][m][0] for x in v[c]['mse'])}/3" for m in MODELS) + ' |')
if T14:
    L += ['', '## Cells of the papers not run here (ETTh1, ETTm1, H192, H336; user decision 12): NSMT Table 14 averages over 4 horizons (MSE)', '',
          '| dataset | SpikF | SpikeRNN | iSpikformer | NSMT |', '|---|--:|--:|--:|--:|']
    for ds in DSO:
        a = T14[(ds, 'Avg')]
        L.append(f'| {ds} | {a[0]:.3f} | {a[2]:.3f} | {a[4]:.3f} | {a[6]:.3f} |')
open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
