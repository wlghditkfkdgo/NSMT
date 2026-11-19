"""ETT input length by the DNN criterion (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG; user decision 11-19 "DNN 보다 평균적으로 좋은 성능을
보여주어야 함. 그러므로 이 기준으로 정할 것"). No training.
Ours: LIF SpkMixer (tuned: muxlin, IAND, 1 mixer layer, lr ETTh2 5e-4 / ETTm2 1e-4) and SF-de, seed means (42 / 7 / 2026) of the test
MSE of the final runs, input 96 (results/flif-v1-ett-spkmux-tune, flif-v1-ett-sfde) and 336 (flif-v1-ett-spkmux-len336, -pn336), loaded
with summarize_pn's loaders and checks.
DNN: published test MSE at the same input length, copied (with their table numbers) in research_notes/SpikMixer 선행 연구 조사/
spiking_time_series.md (L 96: iTransformer Table 10 incl. its re-runs of RLinear / PatchTST / TimesNet / DLinear; SpikF Table 9-10
(iTransformer, MAE loss) and 12 (FITS); TimeMixer Table 13 incl. its PatchTST re-run) and results/ref-dnn-l336-20261118/tables.md
(L 336: PatchTST Table 3, DLinear Table 2).
Per DNN: R = mean over the 4 cells of 100 (ours - DNN) / DNN (negative = ours lower MSE = better), cells lower, 4-cell mean MSE.
Not checked: whether the published protocol equals ours (test batching, normalisation, loss); orientation only.
usage (from cinc_mixer_v1/): python -B results/ett-length-dnn-20261119/compare_dnn.py -> results/ett-length-dnn-20261119/summary.md
"""
import os
import sys

import numpy as np

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
import summarize_pn as sp                                                          # noqa: E402
from summarize_l2 import CELLS, SEEDS                                              # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
# cell order: ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720
DNN = {96: [('iTransformer', (0.297, 0.427, 0.180, 0.412), 'iTransformer Table 10'),
            ('iTransformer, MAE loss', (0.294, 0.423, 0.182, 0.411), 'SpikF Table 9, 10'),
            ('RLinear', (0.288, 0.420, 0.182, 0.407), 'iTransformer Table 10'),
            ('PatchTST (re-run)', (0.302, 0.431, 0.175, 0.402), 'iTransformer Table 10'),
            ('PatchTST (other re-run)', (0.308, 0.436, 0.183, 0.412), 'TimeMixer Table 13'),
            ('TimeMixer', (0.289, 0.412, 0.175, 0.391), 'TimeMixer Table 13'),
            ('TimesNet', (0.340, 0.462, 0.187, 0.408), 'iTransformer Table 10'),
            ('FITS', (0.295, 0.431, 0.183, 0.407), 'SpikF Table 12'),
            ('DLinear (re-run)', (0.333, 0.831, 0.193, 0.554), 'iTransformer Table 10')],
       336: [('PatchTST/42', (0.274, 0.379, 0.165, 0.367), 'PatchTST Table 3'),
             ('DLinear', (0.289, 0.605, 0.167, 0.397), 'DLinear Table 2 / PatchTST Table 3'),
             ('NLinear', (0.277, 0.394, 0.167, 0.368), 'DLinear Table 2'),
             ('Linear', (0.288, 0.698, 0.168, 0.413), 'DLinear Table 2')]}


def cells(T):
    return [float(np.mean([float(T[(ds, pl, s)]['mse']) for s in SEEDS])) for ds, pl in CELLS]


ours = {96: {'LIF SpkMixer': cells(sp.arms_ett_one('LIF')), 'SF-de': cells(sp.load_sfde_ett())}}
raw, en = os.path.join(sp.L336, 'raw.txt'), os.path.join(sp.L336, 'energy_recount.csv')
ours[336] = {'LIF SpkMixer': cells(sp.load_ett(raw, en, lambda ds: f'spkmx_lif_muxlin_iand_lr{sp.LR_MX[ds]}_L1_s336', 'spkmx_lif_muxlin_iand')),
             'SF-de': cells(sp.load_sfde_ett('_s336'))}
L = ['# ETT input length by the DNN criterion', '', 'Rules and sources: module docstring of compare_dnn.py.', '']
for Lb in (96, 336):
    L += [f'## Input {Lb}', '', '| model | ' + ' | '.join(f'{ds} H{pl}' for ds, pl in CELLS) + ' | 4-cell mean |', '|---|' + '--:|' * 5]
    for n, v in ours[Lb].items():
        L.append(f'| {n} (ours) | ' + ' | '.join(f'{x:.4f}' for x in v) + f' | {np.mean(v):.4f} |')
    for n, v, src in DNN[Lb]:
        L.append(f'| {n} ({src}) | ' + ' | '.join(f'{x:.3f}' for x in v) + f' | {np.mean(v):.4f} |')
    L += ['', '| ours | vs DNN | R % (< 0 = ours better) | cells lower | ours better on average |', '|---|---|--:|--:|---|']
    for n, o in ours[Lb].items():
        wins = 0
        for d, v, _ in DNN[Lb]:
            r = [100 * (a - b) / b for a, b in zip(o, v)]
            R = float(np.mean(r)); wins += R < 0
            L.append(f"| {n} | {d} | {R:+.2f} | {sum(x < 0 for x in r)}/4 | {'yes' if R < 0 else 'no'} |")
        L.append(f'| {n} | **summary** | | | better on average than {wins} of {len(DNN[Lb])} DNN rows |')
    L.append('')
open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
