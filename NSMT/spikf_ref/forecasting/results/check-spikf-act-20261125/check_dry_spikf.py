"""Dry run of summarize_spikf_act.py and recheck_spikf_act.py on fabricated rows (2026-11-26, exp/f-lif-pop-v3; PROJECT_LOG SpikF-LIF
pre-registration). No training. Temporary suites results/dry_spikf_* are removed at the end.
  D1 every cell: GELU = paper value + 0.002 (MSE and MAE), LIF = GELU x 1.02 for all 48 pairs, energy GELU 900 / LIF 700 µJ ->
     R +2 % 'LIF worse' (48 higher), energy ratio 0.7778, 16-cell means and the C margins equal hand values; the independent recheck passes
  D2 LIF = GELU x 1.04 in 31 pairs and x 0.995 in 17 -> R +2.41 % but 31 < 32 pairs higher -> 'no clear difference'
  N1-N5 fail closed (exit 2, no summary written): a SKIP line, a missing run, a duplicate run, a nan, mixed evaluation settings
usage (from spikf_ref/forecasting/): python -B results/check-spikf-act-20261125/check_dry_spikf.py
"""
import json
import os
import re
import shutil
import subprocess
import sys

F = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = os.path.dirname(os.path.dirname(F))
PY = sys.executable
refs = json.load(open(os.path.join(N, 'cinc_mixer_v1', 'results', 'ett-full16-20261124', 'refs16.json')))['models']['SpikF']['rows']
ours = json.load(open(os.path.join(N, 'cinc_mixer_v1', 'results', 'flif-v1-ett-spkmux-retune', 'judge.json')))['models']
cells = [(d, p) for d in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2') for p in ('96', '192', '336', '720')]
SEEDS = ('42', '7', '2026')
fails = 0


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}')


def row(d, p, act, s, mse, mae, e, fast='1_fe256_sy1', sha='dry0000'):
    return (f'RESULT exp=spikf_ref_ett ds={d} pl={p} act={act} seed={s} epochs_max=5 patience=3 fast={fast} mse={mse!r} mae={mae!r} '
            f'val_loss_best=0.5 best_epoch=2 epochs_run=4 energy_uJ={e!r} AC_op=1.0 MAC_op=2.0 total_op=3.0 params=1 firing_rate=0.5 '
            f'time_s=1 sha={sha} dir=x\n')


def build(fac):
    L = []
    k = 0
    for d, p in cells:
        g_mse, g_mae = refs[f'{d} {p}'][0] + 0.002, refs[f'{d} {p}'][1] + 0.002
        for s in SEEDS:
            f = fac(k)
            k += 1
            L.append(row(d, p, 'gelu', s, g_mse, g_mae, 900.0))
            L.append(row(d, p, 'lif', s, g_mse * f, g_mae * f, 700.0))
    return L


def run(name, lines):
    d = os.path.join(F, 'results', name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'raw.txt'), 'w').writelines(lines)
    r = subprocess.run([PY, '-B', os.path.join(F, 'summarize_spikf_act.py'), '--suite', name], capture_output=True, text=True)
    return r.returncode, d


base = build(lambda k: 1.02)
rc, d = run('dry_spikf_D1', base)
S = json.load(open(os.path.join(d, 'summary.json'))) if rc == 0 else None
check('D1 summarizer exit 0', rc == 0)
if S:
    pm = sum(refs[f'{a} {b}'][0] for a, b in cells) / 16
    pa = sum(refs[f'{a} {b}'][1] for a, b in cells) / 16
    check('D1 GELU 16-cell MSE = paper + 0.002', abs(S['mean16']['gelu']['mse'] - (pm + 0.002)) < 1e-12)
    check('D1 LIF 16-cell MAE = 1.02 (paper + 0.002)', abs(S['mean16']['lif']['mae'] - 1.02 * (pa + 0.002)) < 1e-12)
    check('D1 B MSE: R +2 %, 48 higher, LIF worse', abs(S['B']['mse']['R_pct'] - 2) < 1e-9 and S['B']['mse']['pairs_higher'] == 48 and S['B']['mse']['verdict'] == 'LIF worse')
    check('D1 energy ratio 700 / 900', abs(S['energy_ratio_lif_gelu_mean'] - 700 / 900) < 1e-12)
    for mk, mv in ours.items():
        want = 1.02 * (pm + 0.002) - mv['mse16']
        check(f'D1 C {mk} vs SpikF-LIF margin MSE {want:+.6f}', abs(S['C'][mk]['lif']['margin_mse'] - want) < 1e-12)
        below = sum(mv['cells'][f'{a} {b}']['syn_energy_uJ'] < 700 for a, b in cells)
        check(f'D1 C {mk} vs SpikF-LIF energy cells below = {below}', S['C'][mk]['lif']['energy_cells_below'] == below)
    r = subprocess.run([PY, '-B', os.path.join(F, 'results', 'check-spikf-act-20261125', 'recheck_spikf_act.py'), '--suite', 'dry_spikf_D1'],
                       capture_output=True, text=True)
    check('D1 independent recheck passes', r.returncode == 0 and 'ALL CHECKS PASSED' in r.stdout, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
rc, d = run('dry_spikf_D2', build(lambda k: 1.04 if k < 31 else 0.995))   # R = (31 x 4 - 17 x 0.5) / 48 = +2.41 %; v1 used 1.02 / 0.99 (R +0.94 %, a bad case)
S = json.load(open(os.path.join(d, 'summary.json'))) if rc == 0 else None
check('D2 R > +1 % with 31 higher -> no clear difference', S is not None and S['B']['mse']['R_pct'] > 1 and S['B']['mse']['pairs_higher'] == 31
      and S['B']['mse']['verdict'] == 'no clear difference', f'R {S["B"]["mse"]["R_pct"]:.3f}' if S else '')
bad = {'N1 SKIP line': base + ['SKIP(rc=1, no result) ds=ETTh1 pl=96 act=lif seed=42\n'],
       'N2 missing run': base[:-1],
       'N3 duplicate run': base + [base[0]],
       'N4 nan': [re.sub(r' mse=\S+', ' mse=nan', base[0], count=1)] + base[1:],
       'N5 mixed evaluation settings': [base[0].replace('fast=1_fe256_sy1', 'fast=0')] + base[1:]}
for i, (nm, lines) in enumerate(bad.items()):
    rc, d = run(f'dry_spikf_N{i + 1}', lines)
    check(f'{nm}: exit 2, no summary', rc == 2 and not os.path.exists(os.path.join(d, 'summary.json')), f'rc {rc}')
for nm in os.listdir(os.path.join(F, 'results')):
    if nm.startswith('dry_spikf_'):
        shutil.rmtree(os.path.join(F, 'results', nm))
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
