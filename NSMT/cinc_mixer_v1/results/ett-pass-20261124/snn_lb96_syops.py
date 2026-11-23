"""SNN baselines at look-back 96 from the SpikF paper vs our ETT models, energy = syops only (2026-11-24, exp/f-lif-pop-v3; PROJECT_LOG;
user 11-24: "1. 현재 Look-back window가 96이라면, Spikformer의 값을 SpikF 원 논문에 제시된 값으로 하게 되면 SNN baseline들 중에서 우리
모델이 어떤 우위/열위를 갖게 돼? 2. energy의 neuron항은 'syops만 쓰기'"). No training.
Paper rows (test MSE / MAE, input 96; cells ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720):
  SpikF Table 12 "More benchmarks on ETT dataset" (= NSMT Table 14; transcription verified 11-19, results/ref-snn-lb96-20261119):
  SpikF, SpikeRNN and iSpikformer ("SNN-based models are equipped with SPE"); the SpikF paper has no plain Spikformer row at
  input 96 -- its Spikformer-family baseline is iSpikformer (inverted Spikformer, Lv et al. 2024b). NSMT Table 14: NSMT.
Ours (seed means of 42 / 7 / 2026, loaded with summarize_ettune.py's loaders and checks): first-chain final, extension final, SF-de
patch 16 (results/flif-v1-ett-sfde-pt16, same row checks as summarize_sfpt16.py), SF-de patch 8.
Energy (user decision 11-24: syops only = syn_energy_uJ, no f-LIF neuron term): ours from the runs; NSMT / SpikF from
results/energy-common-20261123 (NSMT 10-14 reproduction seed 42; SpikF recount); SpikeRNN / iSpikformer: not available (not reported,
not counted) -> any energy criterion against them is "cannot judge" (fail closed).
Pass criteria (pass_ett_nsmt.judge, NSMT DNN rows) re-judged with syops-only energy: (2) against SF-de patch 8 (as defined); report:
the accuracy part of (2) against the paper's iSpikformer (its energy part cannot be judged).
usage (from cinc_mixer_v1/): python -B results/ett-pass-20261124/snn_lb96_syops.py > results/ett-pass-20261124/summary.md
"""
import csv
import importlib.util
import json
import math
import os
import sys

import numpy as np

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
sys.path.insert(0, os.path.join(C, 'results', 'ett-pass-20261123'))
import pass_ett_nsmt as pn                                                           # noqa: E402
pe = pn.pe
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
PAPER = {'NSMT': ((0.288, 0.409, 0.175, 0.399), (0.334, 0.429, 0.251, 0.393), 'NSMT Table 14'),
         'SpikF': ((0.290, 0.416, 0.175, 0.405), (0.336, 0.436, 0.251, 0.397), 'SpikF Table 12'),
         'SpikeRNN': ((0.295, 0.428, 0.176, 0.409), (0.335, 0.438, 0.251, 0.396), 'SpikF Table 12'),
         'iSpikformer': ((0.308, 0.438, 0.178, 0.406), (0.349, 0.439, 0.256, 0.398), 'SpikF Table 12')}


def stop(m):
    print('STOP:', m); sys.exit(2)


# the transcription must equal the verified 11-19 table and pass_ett's SpikF / NSMT rows
import importlib.machinery                                                            # noqa: E402
ref = open(os.path.join(C, 'results', 'ref-snn-lb96-20261119', 'compare_snn_lb96.py')).read()
ns = {}
exec(ref[ref.index('P = {'):ref.index('MODELS =')], ns)
for m in PAPER:
    for j, c in enumerate(CELLS):
        if ns['P'][c][m] != (PAPER[m][0][j], PAPER[m][1][j]):
            stop(f'{m} {c}: {ns["P"][c][m]} != {(PAPER[m][0][j], PAPER[m][1][j])}')
if PAPER['SpikF'][:2] != pe.SNN['SpikF'][:2] or PAPER['NSMT'][:2] != pe.NSMT[:2]:
    stop('SpikF / NSMT rows differ from pass_ett')


def load_se(chain):
    os.environ.pop('ET_DRY', None); os.environ['ET_CHAIN'] = chain
    spec = importlib.util.spec_from_file_location(f'se_{chain or "first"}', os.path.join(C, 'summarize_ettune.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


s1, sx = load_se(''), load_se('x')
SEEDS = s1.SEEDS


def final_arm(se):
    D, E = se.load(), se.energy_rows()
    return se.arm(D, E, json.load(open(os.path.join(se.OUT, 'final.json')))['config']), E


T1, E1 = final_arm(s1)
TX, _ = final_arm(sx)
S8 = s1.sfde_rows(E1)
S16 = {}
for r in s1.raw(os.path.join(C, 'results', 'flif-v1-ett-sfde-pt16', 'raw.txt')):
    ds = r['ds']
    want = {'ETTh2': 'spk_ssa_lr0.0005_L1_pt16_nalif', 'ETTm2': 'spk_ssa_lr0.0001_L2_pt16_nalif'}[ds]
    k = (ds, int(r['pl']), int(r['seed']))
    if r['tag'] != want or k in S16:
        stop(f'bad SF-de pt16 row {k} {r["tag"]}')
    S16[k] = s1.finite_row(r, f'SF-de pt16 {k}')
if sorted(S16) != sorted((d, p, s) for d, p in CELLS for s in SEEDS):
    stop('SF-de pt16 rows incomplete')
OURS = [('LIF SpkMixer first-chain final (patch 16)', T1), ('LIF SpkMixer extension final (patch 24)', TX),
        ('SF-de patch 16 (ours, Spikformer-style)', S16), ('SF-de patch 8 (ours, criteria SF-de)', S8)]


def cell(T, k):
    v = [float(np.mean([float(T[(d, p, s)][k]) for s in SEEDS])) for d, p in CELLS]
    if not all(math.isfinite(x) for x in v):
        stop(f'non-finite {k}')
    return v


ec = {r['ds'] + r['pl']: float(r['energy_uJ']) for r in csv.DictReader(open(os.path.join(C, 'results', 'energy-common-20261123', 'spikf_syops.csv')))}
EN = {'SpikF': [ec['ETTh296'], ec['ETTh2720'], ec['ETTm296'], ec['ETTm2720']], 'NSMT': [33.41, 44.90, 38.46, 46.72]}   # common_table.py sources
L = ['# SNN models at look-back 96 (SpikF paper rows) vs ours; energy = syops only (user 11-24)', '',
     'Rules and sources: module docstring of snn_lb96_syops.py. The SpikF paper has no plain Spikformer row at input 96; its '
     'Spikformer-family baseline is iSpikformer (with SPE).', '', '## Test MSE / MAE and energy', '',
     '| model | source | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4-cell MSE / MAE | energy syops µJ (4-cell mean; H720 ETTh2 / ETTm2) |',
     '|---|---|---|---|---|---|---|---|']
rows = []
for nm, T in OURS:
    ms, ma, e = cell(T, 'mse'), cell(T, 'mae'), cell(T, 'syn_energy_uJ')
    rows.append((nm, 'ours, 3 seeds', ms, ma, e))
for m, (ms, ma, src) in PAPER.items():
    rows.append((m, src + ' (paper value)', list(ms), list(ma), EN.get(m)))
rows.sort(key=lambda r: np.mean(r[2]))
for nm, src, ms, ma, e in rows:
    es = f'{np.mean(e):.2f} ({e[1]:.2f} / {e[3]:.2f})' if e else 'not available'
    L.append(f'| {nm} | {src} | ' + ' | '.join(f'{a:.4f} / {b:.4f}' for a, b in zip(ms, ma)) + f' | {np.mean(ms):.4f} / {np.mean(ma):.4f} | {es} |')
L += ['', '## First-chain final vs each paper SNN row (negative = ours lower = better)', '',
      '| vs | 4-cell MSE diff (ref − ours) | 4-cell MAE diff | MSE % per cell | cells lower MSE / MAE | 0.001 rule (MSE and MAE) | energy ratio ours / ref (syops) |',
      '|---|--:|--:|---|---|---|---|']
om, oa, oe = cell(T1, 'mse'), cell(T1, 'mae'), cell(T1, 'syn_energy_uJ')
for m, (ms, ma, _) in PAPER.items():
    dm, da = np.mean(ms) - np.mean(om), np.mean(ma) - np.mean(oa)
    pc = [100 * (a - b) / b for a, b in zip(om, ms)]
    lo = (sum(a < b for a, b in zip(om, ms)), sum(a < b for a, b in zip(oa, ma)))
    ok = dm >= pe.MARGIN - pe.EPS and da >= pe.MARGIN - pe.EPS
    er = f'{np.mean([a / b for a, b in zip(oe, EN[m])]):.3f} (cells {", ".join(f"{a / b:.3f}" for a, b in zip(oe, EN[m]))})' if m in EN else 'not available'
    L.append(f'| {m} | {dm:+.4f} | {da:+.4f} | {", ".join(f"{x:+.2f}" for x in pc)} | {lo[0]}/4 / {lo[1]}/4 | {"passes" if ok else "does not pass"} | {er} |')
L += ['', '## Pass criteria re-judged with syops-only energy (NSMT DNN rows; pass_ett_nsmt.judge)', '']
VS = {c: {k: [float(S8[(c[0], c[1], s)][k]) for s in SEEDS] for k in ('mse', 'mae')} for c in CELLS}
ES = {c: [float(S8[(c[0], c[1], s)]['syn_energy_uJ']) for s in SEEDS] for c in CELLS}
out = {}
for nm, T in OURS[:2]:
    V = {c: {k: [float(T[(c[0], c[1], s)][k]) for s in SEEDS] for k in ('mse', 'mae')} for c in CELLS}
    EA = {c: [float(T[(c[0], c[1], s)]['syn_energy_uJ']) for s in SEEDS] for c in CELLS}
    r, lines, _, er = pn.judge(f'{nm}; (2) against SF-de patch 8, energy syops only', V, VS, EA, ES, pn.DNN_NSMT)
    out[nm] = {'pass': r, 'energy_ab_syops_only': er}
    L += lines
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rejudge_syops_only.json'), 'w'), indent=1)
print('\n'.join(L))
