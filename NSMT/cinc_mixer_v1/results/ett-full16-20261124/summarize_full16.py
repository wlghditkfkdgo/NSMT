"""Full ETT table: both final LIF SpkMixer configurations on ETTh1 / ETTh2 / ETTm1 / ETTm2 x H 96 / 192 / 336 / 720, input 96
(2026-11-24, exp/f-lif-pop-v3; PROJECT_LOG full-ETT-table pre-registration; user decisions 11-24: SNN baselines = the SpikF paper's rows,
energy must be below SpikF, "두 개다 고려한채로 확장하자").
Models (no selection on the new cells: the settings chosen on ETTh2 / ETTm2 validation are applied as they are; ETTh1 takes the ETTh2
settings, ETTm1 the ETTm2 settings):
  first  first-chain final   ETTh* lr 5e-4 patch 16 wd 0.1;          ETTm* lr 1e-4 patch 16 mlp 2 wd 0.1
  ext    extension final     ETTh* lr 5e-4 patch 24 wd 0.005;        ETTm* lr 1e-4 patch 24 d_model 256 blocks 2 wd 0.02
Rows: the 4 cells trained before (ETTh2 / ETTm2 x H96 / H720; summarize_ettune.py's loaders and checks, final.json of each chain) + the
72 new runs of suite flif-v1-ett-spkmux-full16 (12 cells x 2 models x seeds 42 / 7 / 2026) with its energy_recount.csv
(energy_recount_ett.py: recomputed test MSE within 1e-5, syops body + head = logged synaptic energy within 1e-3).
References (results/ett-full16-20261124/refs16.json, make_refs16.py, verified against both papers' text): SNN = SpikF, SpikeRNN,
iSpikformer (SpikF Table 12); DNN = Informer, FEDformer, PatchTST, ETSformer, Autoformer (NSMT Table 13); NSMT (the user's own paper):
report row, not a criterion.
Pre-registered judgment, per model:
  accuracy  16-cell mean (seed mean per cell, then the mean over the 16 cells; the references' 16 cells, not their Avg rows):
            ref - ours >= 0.001 for MSE and for MAE, against each of the 3 SNN rows and each of the 5 DNN rows
  energy    syops only (logged syn_energy_uJ, seed mean): below SpikF's syops energy in every one of the 16 cells
            (SpikF: results/ett-full16-20261124/spikf16, count_spikf16.py)
  PASS = all 8 accuracy rows and the energy rule.
Report only: the 4-cell values of 11-23 (must reproduce final.json), per-dataset tables with the papers' Avg rows, cells where ours is
lower, NSMT margins, energy against NSMT (10-14 reproduction, seed 42, syops as published), seed std.
Fail closed (exit 2, nothing written): SKIP line, duplicate, unexpected row, missing run, non-finite value, missing / non-reproducing
energy row, a missing SpikF count. env FULL_DIR=<folder>: read the new suite from <folder> (dry runs; the summary is written there).
usage (from cinc_mixer_v1/): python -B results/ett-full16-20261124/summarize_full16.py  ->  <suite>/summary.md, <suite>/summary.json
"""
import csv
import glob
import importlib.util
import json
import math
import os
import sys

import numpy as np

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, C)
SUITE = 'flif-v1-ett-spkmux-full16'
DIR = os.environ.get('FULL_DIR') or os.path.join(C, 'results', SUITE)
DSS, PLS, SEEDS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'), (96, 192, 336, 720), (42, 7, 2026)
CELLS = [(d, p) for d in DSS for p in PLS]
OLD = [('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720)]
NEW = [c for c in CELLS if c not in OLD]
MARGIN, EPS = 0.001, 1e-12
NSMT_LOG = ('/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/'
            'nsmt_base/forecasting/log/ett-paper-check-20261014')
# model -> dataset family -> (tag, the RESULT fields that must match)
CFG = {'first': {'h': ('spkmx_lif_muxlin_iand_lr0.0005_L1_pt16_wd0.1', dict(lr='0.0005', layers='1', patch='16', emb='64', mlpr='0.5', blocks='1', wd='0.1')),
                 'm': ('spkmx_lif_muxlin_iand_lr0.0001_L1_pt16_mlp2_wd0.1', dict(lr='0.0001', layers='1', patch='16', emb='64', mlpr='2', blocks='1', wd='0.1'))},
       'ext': {'h': ('spkmx_lif_muxlin_iand_lr0.0005_L1_pt24_wd0.005', dict(lr='0.0005', layers='1', patch='24', emb='64', mlpr='0.5', blocks='1', wd='0.005')),
               'm': ('spkmx_lif_muxlin_iand_lr0.0001_L1_pt24_e256_blk2_wd0.02', dict(lr='0.0001', layers='1', patch='24', emb='256', mlpr='1', blocks='2', wd='0.02'))}}
NAME = {'first': 'first-chain final (patch 16)', 'ext': 'extension final (patch 24)'}


def stop(m):
    print('STOP:', m); sys.exit(2)


def fam(ds):
    return 'h' if ds.startswith('ETTh') else 'm'


def load_se(chain):
    os.environ.pop('ET_DRY', None); os.environ['ET_CHAIN'] = chain
    spec = importlib.util.spec_from_file_location(f'se_{chain or "first"}', os.path.join(C, 'summarize_ettune.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


s1, sx = load_se(''), load_se('x')
T = {}
for mk, se in (('first', s1), ('ext', sx)):
    fj = json.load(open(os.path.join(se.OUT, 'final.json')))
    for ds in ('ETTh2', 'ETTm2'):
        if fj['tags'][ds] != CFG[mk][fam(ds)][0]:
            stop(f'{mk} {ds}: final.json tag {fj["tags"][ds]} != {CFG[mk][fam(ds)][0]}')
    A = se.arm(se.load(), se.energy_rows(), fj['config'])
    T[mk] = {k: v for k, v in A.items()}
    m4 = float(np.mean([np.mean([A[(d, p, s)]['mse'] for s in SEEDS]) for d, p in OLD]))
    a4 = float(np.mean([np.mean([A[(d, p, s)]['mae'] for s in SEEDS]) for d, p in OLD]))
    if abs(m4 - fj['tuned']['mse']) > 1e-9 or abs(a4 - fj['tuned']['mae']) > 1e-9:
        stop(f'{mk}: 4-cell means {m4} / {a4} != final.json {fj["tuned"]["mse"]} / {fj["tuned"]["mae"]}')
p = os.path.join(DIR, 'raw.txt')
if not os.path.exists(p):
    stop(f'{p} missing')
en = {}
for e in csv.DictReader(open(os.path.join(DIR, 'energy_recount.csv'))):
    k = (e['tag'], e['ds'], int(e['pl']), int(e['seed']))
    if k in en:
        stop(f'duplicate energy row {k}')
    en[k] = e
TAG2M = {(CFG[mk][f][0], f): mk for mk in CFG for f in 'hm'}
nnew = 0
for r in s1.raw(p):                                                                  # stops on a SKIP line
    ds = r.get('ds'); f = fam(ds or 'x')
    mk = TAG2M.get((r.get('tag'), f))
    if mk is None or ds not in DSS or r.get('cond') != 'spkmx_lif_muxlin_iand':
        stop(f'unexpected row {r.get("tag")} {ds} {r.get("pl")} {r.get("seed")}')
    for kk, vv in CFG[mk][f][1].items():
        if r.get(kk) != vv:
            stop(f'{mk} {ds} {r.get("pl")} {r.get("seed")}: {kk}={r.get(kk)} != {vv}')
    k = (ds, int(r['pl']), int(r['seed']))
    if (k[0], k[1]) not in NEW or k[2] not in SEEDS:
        stop(f'unexpected cell {mk} {k}')
    if k in T[mk]:
        stop(f'duplicate row {mk} {k}')
    r = s1.finite_row(r, f'{mk} {k}')
    e = en.get((r['tag'], ds, k[1], k[2]))
    if e is None:
        stop(f'no energy row for {mk} {k}')
    if abs(float(e['mse_recomputed']) - r['mse']) > 1e-5:
        stop(f'recomputed MSE {e["mse_recomputed"]} != {r["mse"]} {mk} {k}')
    body = float(e['body_syn_uJ']) + float(e['head_MAC']) * 4.6e-6
    if abs(body - r['syn_energy_uJ']) > 1e-3 * r['syn_energy_uJ']:
        stop(f'syops body + head {body} != {r["syn_energy_uJ"]} {mk} {k}')
    T[mk][k] = r; nnew += 1
for mk in T:
    miss = [(d, p_, s) for d, p_ in CELLS for s in SEEDS if (d, p_, s) not in T[mk]]
    if miss or len(T[mk]) != 48:
        stop(f'{mk}: missing runs {miss} ({len(T[mk])} rows)')
    for v in T[mk].values():
        if not all(math.isfinite(float(v[x])) for x in ('mse', 'mae', 'syn_energy_uJ')):
            stop('non-finite value')
if nnew != 72:
    stop(f'{nnew} new rows, want 72')
REF = json.load(open(os.path.join(HERE, 'refs16.json')))['models']
SP = {}
for ds in DSS:
    f = os.path.join(HERE, 'spikf16', f'spikf_syops_{ds}.csv')
    if not os.path.exists(f):
        stop(f'{f} missing')
    for r in csv.DictReader(open(f)):
        SP[(r['ds'], int(r['pl']))] = float(r['energy_uJ'])
if sorted(SP) != sorted(CELLS) or not all(math.isfinite(v) and v > 0 for v in SP.values()):
    stop('SpikF energy rows incomplete')
NE = {}
for ds, pl in CELLS:
    f = glob.glob(os.path.join(NSMT_LOG, ds, '*', f'*dataset+{ds}+seq_len+96+pred_len+{pl}+*', 'log', 'final+result.csv'))
    NE[(ds, pl)] = float(next(csv.DictReader(open(f[0])))['energy'].split()[0]) if len(f) == 1 else None


def sm(mk, c, k):
    return float(np.mean([float(T[mk][(c[0], c[1], s)][k]) for s in SEEDS]))


def ref(m, c, j):
    return REF[m]['rows'][f'{c[0]} {c[1]}'][j]


L = ['# Full ETT table, input 96: both final LIF SpkMixer configurations vs the SpikF paper\'s SNN rows and the NSMT paper\'s DNN rows', '',
     'Rules and sources: module docstring of summarize_full16.py. Ours: seed mean of 42 / 7 / 2026 (test values of the best-validation '
     'weights). ETTh1 / ETTm1 use the settings chosen on ETTh2 / ETTm2 (no selection on them). Paper rows: single published numbers.', '']
out = {'models': {}}
SHOW = ('SpikF', 'SpikeRNN', 'iSpikformer', 'PatchTST', 'NSMT')
for ds in DSS:
    L += [f'## {ds} (test MSE / MAE)', '', '| H | first-chain (pt 16) | extension (pt 24) | ' + ' | '.join(SHOW) + ' |', '|---|---|---|' + '---|' * len(SHOW)]
    for pl in PLS:
        c = (ds, pl)
        L.append(f'| {pl} | ' + ' | '.join(f'{sm(mk, c, "mse"):.4f} / {sm(mk, c, "mae"):.4f}' for mk in ('first', 'ext')) + ' | '
                 + ' | '.join(f'{ref(m, c, 0):.3f} / {ref(m, c, 1):.3f}' for m in SHOW) + ' |')
    L.append('| Avg | ' + ' | '.join(f'{np.mean([sm(mk, (ds, p_), "mse") for p_ in PLS]):.4f} / {np.mean([sm(mk, (ds, p_), "mae") for p_ in PLS]):.4f}'
                                     for mk in ('first', 'ext')) + ' | '
             + ' | '.join(f'{REF[m]["rows"][f"{ds} Avg"][0]:.3f} / {REF[m]["rows"][f"{ds} Avg"][1]:.3f}' for m in SHOW) + ' |')
    L.append('')
L += ['## Judgment (16-cell means; margin = ref − ours, ≥ +0.0010 needed for MSE and MAE)', '']
for mk in ('first', 'ext'):
    om = float(np.mean([sm(mk, c, 'mse') for c in CELLS])); oa = float(np.mean([sm(mk, c, 'mae') for c in CELLS]))
    o4 = (float(np.mean([sm(mk, c, 'mse') for c in OLD])), float(np.mean([sm(mk, c, 'mae') for c in OLD])))
    L += [f'### {NAME[mk]}: 16-cell MSE / MAE **{om:.4f} / {oa:.4f}** (the 4 cells of 11-23: {o4[0]:.4f} / {o4[1]:.4f})', '',
          '| reference | role | ref 16-cell MSE / MAE | margin MSE / MAE | passes | cells with lower MSE / MAE (of 16) |', '|---|---|---|---|---|---|']
    res = {}
    for m in ('SpikF', 'SpikeRNN', 'iSpikformer', 'Informer', 'FEDformer', 'PatchTST', 'ETSformer', 'Autoformer', 'NSMT'):
        rm = float(np.mean([ref(m, c, 0) for c in CELLS])); ra = float(np.mean([ref(m, c, 1) for c in CELLS]))
        dm, da = rm - om, ra - oa
        ok = dm >= MARGIN - EPS and da >= MARGIN - EPS
        lo = (sum(sm(mk, c, 'mse') < ref(m, c, 0) for c in CELLS), sum(sm(mk, c, 'mae') < ref(m, c, 1) for c in CELLS))
        role = {'SNN': 'SNN (SpikF paper)', 'DNN': 'DNN (NSMT Table 13)', 'own': 'report only (own paper)'}[REF[m]['role']]
        L.append(f'| {m} | {role} | {rm:.4f} / {ra:.4f} | {dm:+.4f} / {da:+.4f} | {"yes" if ok else "no"} | {lo[0]} / {lo[1]} |')
        res[m] = dict(role=REF[m]['role'], ref_mse=rm, ref_mae=ra, margin_mse=dm, margin_mae=da, passes=bool(ok), lower_mse=lo[0], lower_mae=lo[1])
    er = {c: sm(mk, c, 'syn_energy_uJ') / SP[c] for c in CELLS}
    below = sum(v < 1 for v in er.values())
    acc = all(r['passes'] for r in res.values() if r['role'] != 'own')
    eok = below == 16
    L += ['', f'Energy (syops): below SpikF in **{below} / 16** cells (needed 16); ours / SpikF mean of cell ratios {np.mean(list(er.values())):.4f}, '
          f'largest {max(er.values()):.4f} ({max(er, key=er.get)[0]} H{max(er, key=er.get)[1]}).',
          f'Accuracy rows (3 SNN + 5 DNN): {"all pass" if acc else "NOT all pass: " + ", ".join(m for m, r in res.items() if r["role"] != "own" and not r["passes"])}. '
          f'**Verdict: {"PASS" if acc and eok else "NOT PASSED"}**', '']
    out['models'][mk] = dict(name=NAME[mk], mse16=om, mae16=oa, mse4=o4[0], mae4=o4[1], rows=res, energy_cells_below_spikf=below,
                             energy_ratio_spikf_mean=float(np.mean(list(er.values()))), accuracy_all=bool(acc), energy_ok=bool(eok),
                             verdict='PASS' if acc and eok else 'NOT PASSED',
                             cells={f'{c[0]} {c[1]}': dict(mse=sm(mk, c, 'mse'), mae=sm(mk, c, 'mae'), syn_energy_uJ=sm(mk, c, 'syn_energy_uJ'),
                                                         mse_std=float(np.std([float(T[mk][(c[0], c[1], s)]['mse']) for s in SEEDS]))) for c in CELLS})
L += ['## Energy per cell (syops µJ per input window; ours seed mean, SpikF one checkpoint, NSMT seed 42)', '',
      '| cell | first-chain | extension | SpikF | NSMT | first / SpikF | ext / SpikF | first / NSMT | ext / NSMT |', '|---|--:|--:|--:|--:|--:|--:|--:|--:|']
for c in CELLS:
    a, b = sm('first', c, 'syn_energy_uJ'), sm('ext', c, 'syn_energy_uJ')
    ne = NE[c]
    L.append(f'| {c[0]} H{c[1]} | {a:.2f} | {b:.2f} | {SP[c]:.2f} | ' + (f'{ne:.2f}' if ne else 'not available') + f' | {a / SP[c]:.4f} | {b / SP[c]:.4f} | '
             + (f'{a / ne:.3f} | {b / ne:.3f}' if ne else '— | —') + ' |')
a16, b16 = [sm('first', c, 'syn_energy_uJ') for c in CELLS], [sm('ext', c, 'syn_energy_uJ') for c in CELLS]
L.append(f'| 16-cell mean | {np.mean(a16):.2f} | {np.mean(b16):.2f} | {np.mean([SP[c] for c in CELLS]):.2f} | '
         + (f'{np.mean([NE[c] for c in CELLS]):.2f}' if all(NE[c] for c in CELLS) else 'not available') + ' | | | | |')
out['energy'] = {f'{c[0]} {c[1]}': dict(first=sm('first', c, 'syn_energy_uJ'), ext=sm('ext', c, 'syn_energy_uJ'), spikf=SP[c], nsmt=NE[c]) for c in CELLS}
L += ['', '## Seed spread (std of test MSE over the 3 seeds, mean over the 16 cells)', '']
for mk in ('first', 'ext'):
    L.append(f'- {NAME[mk]}: {np.mean([out["models"][mk]["cells"][f"{c[0]} {c[1]}"]["mse_std"] for c in CELLS]):.4f}')
json.dump(out, open(os.path.join(DIR, 'summary.json'), 'w'), indent=1)
open(os.path.join(DIR, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
