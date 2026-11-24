"""Smoke / reproduction / memory check of the ETTh1 / ETTm1 retuning (2026-11-25, exp/f-lif-pop-v3; PROJECT_LOG pre-registration of
2026-11-25). Run after the smoke pool, the reproduction pool and energy_recount_ett.py --suite smoke-retune-20261125.
  S1 smoke-retune-20261125 (1 epoch, seed 42, H720; per dataset the start configuration and 4 others covering every candidate value
     that is new for ETTh1 / ETTm1: lr 2.5e-4 / 2e-3 / 1e-4, M 1 / 3, patch 4 / 24 / 32, d_model 128 / 256, MLP 0.5 / 4, blocks 2 / 3,
     wd 0 / 0.005): 10 RESULT, 0 SKIP; each row's RESULT fields = the knobs of its queue line, tag = rt_plan.tag()
  S2 every run's saved config (model_state/config.pt) holds those values (patch_size, embed_dim, mlp_ratios, num_layers = blocks,
     weight_decay, time_num_layers = M, lr) and 1 epoch
  S3 a shape knob changes the parameter count against the start configuration of the dataset
  S4 energy_recount_ett.py on the smoke suite: 10 rows, recomputed test MSE within 1e-5, syops body + head = logged synaptic energy
     within 1e-3 (the judge's checks)
  R1 check-retune-repro-20261125, the 11-24 full-table queue line of ETTh1 H96 seed 42 (first-chain settings, runner default 100
     epochs): every shared RESULT field equals the full16 row except time_s, sha, dir
  R2 the same run written the retune way (all seven knobs explicit, EPOCHS=50): the same tag and the same shared fields (explicit
     default knobs do not change training; the 50-epoch cap did not bind there)
  report: peak GPU memory per run (mem_all.txt, 2 s samples), 1-epoch time
usage (from cinc_mixer_v1/): python -B results/check-retune-20261125/check_smoke_rt.py > results/check-retune-20261125/check_smoke_rt.txt
"""
import collections
import csv
import os
import sys

import torch

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
import rt_plan as P                                                              # noqa: E402

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
        skip += l.startswith('SKIP')
        if l.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return out, skip


def kv(s):
    return dict(x.split('=') for x in s.split())


WANT = []
for ds in ('ETTm1', 'ETTh1'):
    st = ' '.join(f'{k}={P.START[ds][k]}' for k in P.KN)
    for s in (st, 'LR=0.00025 LAYERS=3 PATCH=8 EMB=64 MLPR=1 BLOCKS=1 WD=0.06', 'LR=0.002 LAYERS=1 PATCH=32 EMB=64 MLPR=1 BLOCKS=1 WD=0.005',
              'LR=0.0001 LAYERS=3 PATCH=4 EMB=256 MLPR=4 BLOCKS=3 WD=0.06', 'LR=0.0005 LAYERS=1 PATCH=24 EMB=128 MLPR=0.5 BLOCKS=2 WD=0'):
        WANT.append((ds, kv(s)))
sm, sk = rows('smoke-retune-20261125')
check('S1 smoke: 10 RESULT, 0 SKIP', len(sm) == 10 and sk == 0, f'{len(sm)} RESULT, {sk} SKIP')
start = {}
for ds, env in WANT:
    got = [r for r in sm if r['ds'] == ds and all(r[P.FIELD[k]] == env[k] for k in P.KN)]
    if len(got) != 1:
        check(f'S1 {ds} {env}: one row', False, str(len(got))); continue
    r = got[0]
    check(f'S1 {ds} {r["tag"]}: fields = queue knobs, tag = rt_plan.tag', r['tag'] == P.tag(ds, env), P.tag(ds, env))
    cfg = torch.load(os.path.join(r['dir'], 'model_state', 'config.pt'), map_location='cpu')
    g = (cfg['patch_size'], cfg['embed_dim'], float(cfg['mlp_ratios']), cfg['num_layers'], float(cfg['weight_decay']), cfg['time_num_layers'], float(cfg['lr']), cfg['epoch'])
    w = (int(env['PATCH']), int(env['EMB']), float(env['MLPR']), int(env['BLOCKS']), float(env['WD']), int(env['LAYERS']), float(env['LR']), 1)
    check(f'S2 {ds} {r["tag"]}: config.pt = knobs, 1 epoch', g == w, f'{g} vs {w}' if g != w else '')
    if env == P.START[ds]:
        start[ds] = r
for r in sm:
    if r['ds'] in start and r is not start[r['ds']]:
        s0 = start[r['ds']]
        shape = any(r[f] != s0[f] for f in ('patch', 'emb', 'mlpr', 'blocks', 'layers'))
        check(f'S3 {r["ds"]} {r["tag"]}: params {r["params"]} vs start {s0["params"]}', (r['params'] != s0['params']) == shape)
ep = os.path.join(RES, 'smoke-retune-20261125', 'energy_recount.csv')
E = list(csv.DictReader(open(ep))) if os.path.exists(ep) else []
check('S4 energy recount: 10 rows', len(E) == 10, str(len(E)))
for e in E:
    r = [x for x in sm if x['tag'] == e['tag'] and x['ds'] == e['ds']][0]
    body = float(e['body_syn_uJ']) + float(e['head_MAC']) * 4.6e-6
    check(f'S4 {e["ds"]} {e["tag"]}: recomputed MSE {e["mse_recomputed"]} vs {r["mse"]}, syops body + head {body:.4f} vs {float(r["syn_energy_uJ"]):.4f}',
          abs(float(e['mse_recomputed']) - float(r['mse'])) <= 1e-5 and abs(body - float(r['syn_energy_uJ'])) <= 1e-3 * float(r['syn_energy_uJ']))
rp, rk = rows('check-retune-repro-20261125')
f16, _ = rows('flif-v1-ett-spkmux-full16')
ref = [t for t in f16 if t['tag'] == 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt16_wd0.1' and t['ds'] == 'ETTh1' and t['pl'] == '96' and t['seed'] == '42']
check('R repro: 2 RESULT, 0 SKIP; one full16 row', len(rp) == 2 and rk == 0 and len(ref) == 1, f'{len(rp)} / {rk} / {len(ref)}')
for nm, r in zip(('R1 (11-24 line, 100 epochs)', 'R2 (retune form, explicit knobs, 50 epochs)'),
                 sorted(rp, key=lambda r: '+epoch+50+' in r['dir'])):
    o = ref[0]
    shared = [k for k in r if k in o and k not in ('time_s', 'sha', 'dir')]
    diff = [k for k in shared if r[k] != o[k]]
    check(f'{nm}: {len(shared)} shared fields equal the full16 row (mse {r["mse"]} vs {o["mse"]}, epochs run {r["best_epoch_val_epochs"]}, sha {r["sha"]} vs {o["sha"]})',
          not diff and len(shared) >= 30, f'DIFFER {diff}' if diff else '')
peak = collections.defaultdict(int)
mp = os.path.join(OUT, 'mem_all.txt')
for l in open(mp) if os.path.exists(mp) else []:
    p = l.split()
    if len(p) == 4:
        peak[p[2]] = max(peak[p[2]], int(p[3]))
print('\nReport (1 epoch, seed 42, H720; syops energy = logged total_fft_uJ):')
print('| dataset | knobs | params | energy µJ | time s | peak GPU MiB |')
print('|---|---|--:|--:|--:|--:|')
for r in sorted(sm, key=lambda r: (r['ds'], r['tag'])):
    kn = ' '.join(f'{k} {r[P.FIELD[k]]}' for k in P.KN)
    print(f'| {r["ds"]} | {kn} | {r["params"]} | {float(r["total_fft_uJ"]):.2f} | {r["time_s"]} | {peak.get(r["tag"] + "_seed42", "n/a")} |')
print(f'\npeak GPU memory of any sampled process: {max(peak.values()) if peak else "n/a"} MiB')
print(f'{"ALL PASS" if fails == 0 else str(fails) + " FAIL"}')
sys.exit(0 if fails == 0 else 1)
