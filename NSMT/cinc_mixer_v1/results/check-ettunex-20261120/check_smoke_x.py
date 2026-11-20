"""Smoke check of the ETT tuning extension (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning extension pre-registration).
suite smoke-ettunex-20261120: 1 epoch, seed 42, H720, per dataset the default and each new value alone (PATCH 24 / 32, WD 0.005 / 0.02 /
0.05) = 12 runs.
  S1 12 RESULT, 0 SKIP; every run's saved config (model_state/config.pt) holds the knob values of its RESULT fields
  S2 the default runs have the parameter count of the 11-01 rows (same shape); PATCH changes it, WD does not
  S3 energy_recount_ett.py rows for all 12 (the summarizer's checks: recomputed MSE within 1e-5, syops body + head MAC x 4.6 pJ = logged
     synaptic energy within 1e-3, relative)
  report: 1-epoch syops energy and parameters relative to the default run of the dataset, run time
usage (from cinc_mixer_v1/): python -B results/check-ettunex-20261120/check_smoke_x.py > results/check-ettunex-20261120/check_smoke_x.txt
"""
import csv
import os
import sys

import torch

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(C, 'results')
fails = 0


def check(label, cond, detail=''):
    global fails
    fails += not cond
    print(f'{"PASS" if cond else "FAIL"} {label}{(": " + detail) if detail else ""}')


def rows(suite):
    out, skip = [], 0
    for l in open(os.path.join(RES, suite, 'raw.txt')):
        skip += l.startswith('SKIP')
        if l.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return out, skip


sm, sk = rows('smoke-ettunex-20261120')
check('S1 smoke: 12 RESULT, 0 SKIP', len(sm) == 12 and sk == 0, f'{len(sm)} RESULT, {sk} SKIP')
for r in sm:
    cfg = torch.load(os.path.join(r['dir'], 'model_state', 'config.pt'), map_location='cpu')
    got = (cfg['patch_size'], cfg['embed_dim'], float(cfg['mlp_ratios']), cfg['num_layers'], float(cfg['weight_decay']), cfg['time_num_layers'], float(cfg['lr']))
    want = (int(r['patch']), int(r['emb']), float(r['mlpr']), int(r['blocks']), float(r['wd']), 1, float(r['lr']))
    check(f'S1 {r["ds"]} {r["tag"]}: config.pt = RESULT fields', got == want, f'{got}' if got != want else '')
LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
tune, _ = rows('flif-v1-ett-spkmux-tune')
ref = {r['ds']: r for r in tune if r['cond'] == 'spkmx_lif_muxlin_iand' and r.get('tag') == f'spkmx_lif_muxlin_iand_lr{LR[r["ds"]]}_L1'
       and r['pl'] == '720' and r['seed'] == '42'}
dflt = {r['ds']: r for r in sm if r['tag'] == f'spkmx_lif_muxlin_iand_lr{LR[r["ds"]]}_L1'}
for ds in LR:
    check(f'S2 {ds} default smoke params = 11-01 row params ({ref[ds]["params"]})', dflt[ds]['params'] == ref[ds]['params'], dflt[ds]['params'])
    for r in sm:
        if r['ds'] != ds or r is dflt[ds]:
            continue
        shape = r['patch'] != dflt[ds]['patch']
        check(f'S2 {ds} {r["tag"]}: params {"change" if shape else "unchanged (WD only)"}', (r['params'] != dflt[ds]['params']) == shape, r['params'])
E = {(e['tag'], e['ds'], e['pl'], e['seed']): e for e in csv.DictReader(open(os.path.join(RES, 'smoke-ettunex-20261120', 'energy_recount.csv')))}
for r in sm:
    e = E.get((r['tag'], r['ds'], r['pl'], r['seed']))
    if e is None:
        check(f'S3 {r["ds"]} {r["tag"]}: energy row', False); continue
    syn, body = float(r['syn_energy_uJ']), float(e['body_syn_uJ']) + float(e['head_MAC']) * 4.6e-6
    check(f'S3 {r["ds"]} {r["tag"]}: recount MSE and syops body + head', abs(float(e['mse_recomputed']) - float(r['mse'])) <= 1e-5
          and abs(body - syn) <= 1e-3 * syn, f'mse {e["mse_recomputed"]} / {r["mse"]}, syn {body:.4f} / {syn:.4f}')
print('\nReport (1 epoch, seed 42, H720; energy = logged total_fft_uJ, syops rule; relative to the default run of the dataset):')
print('| dataset | knob | tokens | params | params / default | energy µJ | energy / default | time s |')
print('|---|---|--:|--:|--:|--:|--:|--:|')
for r in sorted(sm, key=lambda r: (r['ds'], int(r['patch']), float(r['wd']))):
    d = dflt[r['ds']]
    kn = r['tag'][len(f'spkmx_lif_muxlin_iand_lr{LR[r["ds"]]}_L1'):] or '(default)'
    P = int(r['patch']); tok = int((96 - P) / (P // 2) + 1) + 1
    print(f'| {r["ds"]} | {kn} | {tok} | {r["params"]} | {int(r["params"]) / int(d["params"]):.2f} | {float(r["total_fft_uJ"]):.2f} | '
          f'{float(r["total_fft_uJ"]) / float(d["total_fft_uJ"]):.2f} | {r["time_s"]} |')
print('ALL SMOKE CHECKS PASSED' if fails == 0 else f'{fails} FAILED')
sys.exit(0 if fails == 0 else 1)
