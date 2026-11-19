"""Smoke / reproduction / memory check of the ETT tuning (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning pre-registration).
  S1 smoke-ettune-20261119 (1 epoch, seed 42, H720): 30 RESULT, no SKIP; every run's saved config (model_state/config.pt) holds the
     knob values of its RESULT fields (patch_size, embed_dim, mlp_ratios, num_layers = blocks, weight_decay; time_num_layers 1, lr)
  S2 the default smoke runs have the parameter count of the 11-01 tuned rows (same shape); a shape knob changes it, WD does not
  R  check-ettune-repro-20261120 (full training, seed 42, default configuration, today's runner): every shared RESULT field equals the
     11-01 tuned row (flif-v1-ett-spkmux-tune) except time_s, sha, dir
  report: 1-epoch syops energy and params relative to the default smoke run, time, peak GPU memory per run (mem_all.txt, 2 s samples,
     8 runs on 4 GPUs at once)
usage (from cinc_mixer_v1/): python -B results/check-ettune-20261120/check_smoke.py > results/check-ettune-20261120/check_smoke.txt
"""
import collections
import os
import sys

import torch

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(C, 'results')
fails = 0


def check(label, cond, detail=''):
    global fails
    fails += not cond
    print(f'{"PASS" if cond else "FAIL"} {label}{(": " + detail) if detail else ""}')


def rows(suite):
    out, skip = [], 0
    for l in open(os.path.join(RES, suite, 'raw.txt')):
        if l.startswith('SKIP'):
            skip += 1
        if l.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return out, skip


sm, sk = rows('smoke-ettune-20261119')
check('S1 smoke: 30 RESULT, 0 SKIP', len(sm) == 30 and sk == 0, f'{len(sm)} RESULT, {sk} SKIP')
for r in sm:
    cfg = torch.load(os.path.join(r['dir'], 'model_state', 'config.pt'), map_location='cpu')
    got = (cfg['patch_size'], cfg['embed_dim'], float(cfg['mlp_ratios']), cfg['num_layers'], float(cfg['weight_decay']), cfg['time_num_layers'], float(cfg['lr']))
    want = (int(r['patch']), int(r['emb']), float(r['mlpr']), int(r['blocks']), float(r['wd']), 1, float(r['lr']))
    check(f'S1 {r["ds"]} {r["tag"]}: config.pt = RESULT fields', got == want, f'{got}' if got != want else '')
tune, _ = rows('flif-v1-ett-spkmux-tune')
LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
ref = {r['ds']: r for r in tune if r['cond'] == 'spkmx_lif_muxlin_iand' and r.get('tag') == f'spkmx_lif_muxlin_iand_lr{LR[r["ds"]]}_L1'
       and r['pl'] == '720' and r['seed'] == '42'}
dflt = {r['ds']: r for r in sm if r['tag'] == f'spkmx_lif_muxlin_iand_lr{LR[r["ds"]]}_L1'}
for ds in LR:
    check(f'S2 {ds} default smoke params = 11-01 tuned row params ({ref[ds]["params"]})', dflt[ds]['params'] == ref[ds]['params'], dflt[ds]['params'])
    for r in sm:
        if r['ds'] != ds or r is dflt[ds]:
            continue
        shape = any(r[f] != dflt[ds][f] for f in ('patch', 'emb', 'mlpr', 'blocks'))
        check(f'S2 {ds} {r["tag"]}: params {"change" if shape else "unchanged (WD only)"}', (r['params'] != dflt[ds]['params']) == shape, r['params'])
rp, rk = rows('check-ettune-repro-20261120')
check('R repro: 2 RESULT, 0 SKIP', len(rp) == 2 and rk == 0, f'{len(rp)} / {rk}')
for r in rp:
    o = [t for t in tune if t['cond'] == r['cond'] and t.get('tag') == r['tag'] and t['ds'] == r['ds'] and t['pl'] == r['pl'] and t['seed'] == r['seed']]
    if len(o) != 1:
        check(f'R {r["ds"]} H{r["pl"]}: one 11-01 row', False, str(len(o))); continue
    shared = [k for k in r if k in o[0] and k not in ('time_s', 'sha', 'dir')]
    diff = [k for k in shared if r[k] != o[0][k]]
    check(f'R {r["ds"]} H{r["pl"]} seed {r["seed"]}: {len(shared)} shared fields equal (mse {r["mse"]} vs {o[0]["mse"]}, sha {r["sha"]} vs {o[0]["sha"]})',
          not diff and len(shared) >= 30, f'DIFFER {diff}' if diff else '')
peak = collections.defaultdict(int)
mp = os.path.join(OUT, 'mem_all.txt')
for l in open(mp) if os.path.exists(mp) else []:
    p = l.split()
    if len(p) == 4:
        peak[p[2]] = max(peak[p[2]], int(p[3]))
print('\nReport (1 epoch, seed 42, H720; energy = logged total_fft_uJ, syops rule; relative to the default run of the dataset):')
print('| dataset | knobs | params | params / default | energy µJ | energy / default | time s | peak GPU MiB |')
print('|---|---|--:|--:|--:|--:|--:|--:|')
for r in sorted(sm, key=lambda r: (r['ds'], r['tag'])):
    d = dflt[r['ds']]
    kn = r['tag'][len(f'spkmx_lif_muxlin_iand_lr{LR[r["ds"]]}_L1'):] or '(default)'
    print(f'| {r["ds"]} | {kn} | {r["params"]} | {int(r["params"]) / int(d["params"]):.2f} | {float(r["total_fft_uJ"]):.2f} | '
          f'{float(r["total_fft_uJ"]) / float(d["total_fft_uJ"]):.2f} | {r["time_s"]} | {peak.get(r["tag"] + "_seed42", 0) or "-"} |')
print(f'\nPeak GPU memory of any sampled run: {max(peak.values()) if peak else "-"} MiB')
print('ALL SMOKE CHECKS PASSED' if fails == 0 else f'{fails} FAILED')
sys.exit(0 if fails == 0 else 1)
