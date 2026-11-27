"""Checks before the T test (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG T-test pre-registration; user "T를 8로 늘려보자"). No training.
  r0 runner unchanged without TH: run_ett_spk.sh (copy before this change, `--old`) and the worktree runner give the same training
     arguments (stub python) for earlier conditions; TH=8 changes only --hippo_T 4 -> 8 and adds the tag suffix _T8; TH=8 with a
     non-LIF-SpkMixer condition is refused
  r1 the model at hippo_T 8 builds 8 repeats ([8, B', N, P] tokens, embedding / MLP neurons over 8 steps) and its parameters equal the
     T 4 model's at the same seed (T has no parameters)
usage (from cinc_mixer_v1/): python -B results/check-t8-20261127/check_t8.py --old <dir with the runner copy before the change>
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, C)
os.chdir(C)
ap = argparse.ArgumentParser()
ap.add_argument('--old', required=True)
a = ap.parse_args()
fails = 0


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}', flush=True)


tmp = tempfile.mkdtemp(prefix='t8_')
stub = os.path.join(tmp, 'stubpy')
open(stub, 'w').write('#!/usr/bin/env bash\necho "ARGS $*"\nexit 3\n')
os.chmod(stub, 0o755)


def runner_args(runner, code_dir, envs, ds):
    res, log = tempfile.mkdtemp(prefix='res_'), tempfile.mkdtemp(prefix='log_')
    p = subprocess.run(['bash', runner, ds, '96', '42', '0'], capture_output=True, text=True, env=dict(os.environ, PY=stub, RES=res, LOG=log, SUITE='t8', **envs))
    txt = ''.join(open(os.path.join(log, f)).read() for f in os.listdir(log) if f.endswith('.stdout'))
    m = re.search(r'^ARGS (.*)$', txt, re.M)
    shutil.rmtree(res); shutil.rmtree(log)
    return (m.group(1).replace(os.path.abspath(code_dir), '<W>').replace(log, '<LOG>') if m else None), p.stdout + p.stderr


old_r = os.path.join(os.path.abspath(a.old), 'scripts', 'run_ett_spk.sh')
new_r = os.path.join(C, 'scripts', 'run_ett_spk.sh')
KN = [('ETTh1', dict(COND='spkmx_lif_muxlin_iand', LR='0.00025', LAYERS='1', EMB='32', MLPR='4', WD='0.1')),
      ('ETTm1', dict(COND='spkmx_lif_muxlin_iand', LR='0.0005', LAYERS='2', PATCH='16', EMB='32', MLPR='0.5')),
      ('ETTm1', dict(COND='spkmx_lif_msxdpca_iand', LR='0.0005', LAYERS='2', PATCH='16', EMB='32', MLPR='0.5')),
      ('ETTh2', dict(COND='spkmx_lif_sdlin_or')), ('ETTh2', dict(COND='spk_ssa', LR='0.0005', PATCH='16'))]
for ds, kn in KN:
    o, _ = runner_args(old_r, a.old, kn, ds)
    n, _ = runner_args(new_r, C, kn, ds)
    check(f'r0 {kn["COND"]} {ds}: same training arguments without TH', o is not None and o == n, '' if o == n else f'\n  {o}\n  {n}')
for ds, kn in KN[:3]:
    n4, _ = runner_args(new_r, C, kn, ds)
    n8, _ = runner_args(new_r, C, dict(kn, TH='8'), ds)
    ok = n4 is not None and n8 is not None and n8.replace('--hippo_T 8', '--hippo_T 4').replace('_T8 ', ' ', 1) == n4 and '--hippo_T 8' in n8 and '_T8' in n8
    check(f'r0 {kn["COND"]} {ds}: TH=8 changes only --hippo_T and adds _T8', ok, '' if ok else f'\n  {n4}\n  {n8}')
n, out = runner_args(new_r, C, dict(KN[4][1], TH='8'), 'ETTh2')
check('r0 TH=8 with spk_ssa is refused', n is None and 'LIF SpkMixer conditions only' in out)
shutil.rmtree(tmp)
import torch                                    # noqa: E402
from forecast_wrap import Forecaster            # noqa: E402
from spk_mixer import SpkMixer                  # noqa: E402
P = {}
for T in (4, 8):
    torch.manual_seed(42)
    inner = SpkMixer(seq_len=96, num_classes=1, depths=1, mixer='lif', T_h=T, neo_tau=2.0, bias=False, tau=2.0, mixer_comb='msxdpca',
                     connect_f='IAND', data_patch_size=16, embed_dim=32, mlp_ratios=.5, mixer_layers=2)
    m = Forecaster(inner, 32, 96, bias=False)
    tok = inner.tokens(torch.randn(6, 1, 96))
    with torch.no_grad():
        h = inner.forward_parts(torch.randn(6, 1, 96))[4]
    P[T] = dict(m.named_parameters())
    check(f'r1 T {T}: tokens {tuple(tok.shape)}, block output {tuple(h.shape)}', tuple(tok.shape) == (T, 6, 12, 16) and tuple(h.shape) == (T, 6, 12, 32))
check('r1 parameters at T 8 equal those at T 4 (same seed)', list(P[4]) == list(P[8]) and all(torch.equal(P[4][k], P[8][k]) for k in P[4]))
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
