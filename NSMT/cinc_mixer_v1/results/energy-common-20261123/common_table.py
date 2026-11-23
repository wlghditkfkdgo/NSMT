"""Energy of every ETT model under one rule: the syops library as published (2026-11-23, exp/f-lif-pop-v3; PROJECT_LOG (나); user question
11-23 "'우리 규칙'이 뭐야? 현재 synops 라이브러리를 그대로 따르고 있지 않아?"). No training.
Rule (syops 0.0.5 = github iCGY96/syops-counter HEAD 4232f77, ops.py byte-identical): every nn module hook -- Linear / Conv / BN / pool
input with <= 2 distinct values in [0, 1] -> AC x firing rate, otherwise MAC; spikingjelly LIF / IF neurons: 1 AC per input element;
ops outside nn modules (matmul, fft, elementwise) are not counted; E = 0.9 pJ x AC + 4.6 pJ x MAC; per input window of all 7 channels
(syops averages over the test loader's samples). No neuron term of the f-LIF paper (that term is our addition; shown separately).
Sources:
  ours / SF-de  logged syn_energy_uJ of the final runs (train_forecast.final_test: syops with custom_modules_hooks = syops' own
                LIF_syops_counter_hook for neurons.py LIFNeuron / FLIFNeuron, i.e. the syops neuron rule applied to a class syops does not
                list); loaded with summarize_ettune.py's loaders (seed means of 42 / 7 / 2026); + the logged total_fft_uJ (with the neuron term)
  SF-de pt16    results/flif-v1-ett-sfde-pt16 (only when its summary.json exists)
  NSMT          the NSMT reproduction runs of 2026-10-14 (main tree NSMT/nsmt_base/forecasting/log/ett-paper-check-20261014, seed 42,
                read only): column "energy" of final+result.csv = syops as published without custom hooks; H720 = NSMT Table 6
  SpikF         results/energy-common-20261123/spikf_syops.csv (count_spikf.py: H720 reproduces NSMT Table 6 and the logged SpikF runs;
                H96 = the H720 checkpoint with an H96 output layer, MAC exact, AC from the H720 encoder)
usage (from cinc_mixer_v1/): python -B results/energy-common-20261123/common_table.py > results/energy-common-20261123/summary.md
"""
import csv
import glob
import importlib.util
import json
import os
import sys

import numpy as np

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, C)
NSMT_LOG = ('/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/'
            'nsmt_base/forecasting/log/ett-paper-check-20261014')
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))


def stop(m):
    print('STOP:', m); sys.exit(2)


def load_se(chain):
    os.environ.pop('ET_DRY', None); os.environ['ET_CHAIN'] = chain
    spec = importlib.util.spec_from_file_location(f'se_{chain or "first"}', os.path.join(C, 'summarize_ettune.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def ours(se, cfg_from=None):
    D, E = se.load(), se.energy_rows()
    cfg = json.load(open(os.path.join(se.OUT, 'final.json')))['config'] if cfg_from is None else cfg_from
    return se.arm(D, E, cfg), E


s1, sx = load_se(''), load_se('x')
T1, E1 = ours(s1)
TX, _ = ours(sx)
S8 = s1.sfde_rows(E1)


def sm(T, k):
    return [float(np.mean([float(T[(d, p, s)][k]) for s in s1.SEEDS])) for d, p in CELLS]


rows = [('LIF SpkMixer, first-chain final (patch 16)', sm(T1, 'syn_energy_uJ'), sm(T1, 'total_fft_uJ')),
        ('LIF SpkMixer, extension final (patch 24)', sm(TX, 'syn_energy_uJ'), sm(TX, 'total_fft_uJ')),
        ('SF-de patch 8 (criteria SF-de)', sm(S8, 'syn_energy_uJ'), sm(S8, 'total_fft_uJ'))]
p16 = os.path.join(C, 'results', 'flif-v1-ett-sfde-pt16', 'summary.json')
if os.path.exists(p16):
    T16 = {}
    for l in open(os.path.join(C, 'results', 'flif-v1-ett-sfde-pt16', 'raw.txt')):
        if l.startswith('RESULT'):
            r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
            T16[(r['ds'], int(r['pl']), int(r['seed']))] = r
    rows.append(('SF-de patch 16', sm(T16, 'syn_energy_uJ'), sm(T16, 'total_fft_uJ')))
nsmt = []
for ds, pl in CELLS:
    f = glob.glob(os.path.join(NSMT_LOG, ds, '*', f'*dataset+{ds}+seq_len+96+pred_len+{pl}+*', 'log', 'final+result.csv'))
    if len(f) != 1:
        stop(f'NSMT {ds} H{pl}: {len(f)} result files')
    r = next(csv.DictReader(open(f[0])))
    nsmt.append(float(r['energy'].split()[0]))
TABLE6 = {('ETTh2', 720): 44.90, ('ETTm2', 720): 46.72}
for (c, v) in zip(CELLS, nsmt):
    if c in TABLE6 and abs(v - TABLE6[c]) > 1e-9:
        stop(f'NSMT {c} {v} != Table 6 {TABLE6[c]}')
rows.append(('NSMT (10-14 reproduction, seed 42)', nsmt, None))
sp = {(r['ds'], int(r['pl'])): float(r['energy_uJ']) for r in csv.DictReader(open(os.path.join(OUT, 'spikf_syops.csv')))}
rows.append(('SpikF (H96: output layer resized, see count_spikf.py)', [sp[c] for c in CELLS], None))
L = ['# ETT energy under one rule: syops as published (µJ per input window, all 7 channels)', '',
     'Rules and sources: module docstring of common_table.py. Ours / SF-de: seed means; NSMT: seed 42; SpikF: one trained checkpoint.', '',
     '| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4-cell mean |', '|---|--:|--:|--:|--:|--:|']
for nm, v, _ in rows:
    L.append(f'| {nm} | ' + ' | '.join(f'{x:.2f}' for x in v) + f' | {np.mean(v):.2f} |')
L += ['', 'With the f-LIF-paper neuron term added (our runs only; the rule used for the verdicts so far):', '',
      '| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | neuron term share |', '|---|--:|--:|--:|--:|---|']
for nm, v, w in rows:
    if w is not None:
        L.append(f'| {nm} | ' + ' | '.join(f'{x:.2f}' for x in w) + ' | ' + ', '.join(f'{100 * (b - a) / b:.1f} %' for a, b in zip(v, w)) + ' |')
base = rows[0][1]
L += ['', 'Ratio first-chain final / other model, per cell (syops as published):', '', '| vs | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean of cell ratios |',
      '|---|--:|--:|--:|--:|--:|']
for nm, v, _ in rows[1:]:
    r = [a / b for a, b in zip(base, v)]
    L.append(f'| {nm} | ' + ' | '.join(f'{x:.3f}' for x in r) + f' | {np.mean(r):.3f} |')
print('\n'.join(L))
