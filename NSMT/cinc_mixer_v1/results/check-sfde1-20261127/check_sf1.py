"""Checks before SF-de on ETTh1 / ETTm1 (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "SF-de에 대해서 ETTh1, ETTm1의
순서 섞기 실험을 진행해줘"). No training.
  r0 runner: earlier conditions give the same training arguments as the runner before this change (stub python)
  r1 spk_ssa on ETTh1 / ETTm1: the arguments equal those of the same settings on ETTh2 / ETTm2 except the dataset name / file
usage (from cinc_mixer_v1/): python -B results/check-sfde1-20261127/check_sf1.py --old <copy of cinc_mixer_v1 before the change>
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
ap = argparse.ArgumentParser()
ap.add_argument('--old', required=True)
a = ap.parse_args()
fails = 0


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}', flush=True)


tmp = tempfile.mkdtemp(prefix='sf1_')
stub = os.path.join(tmp, 'stubpy')
open(stub, 'w').write('#!/usr/bin/env bash\necho "ARGS $*"\nexit 3\n')
os.chmod(stub, 0o755)


def args_of(runner, code_dir, envs, ds):
    res, log = tempfile.mkdtemp(prefix='res_'), tempfile.mkdtemp(prefix='log_')
    p = subprocess.run(['bash', runner, ds, '96', '42', '0'], capture_output=True, text=True, env=dict(os.environ, PY=stub, RES=res, LOG=log, SUITE='sf1', **envs))
    txt = ''.join(open(os.path.join(log, f)).read() for f in os.listdir(log) if f.endswith('.stdout'))
    m = re.search(r'^ARGS (.*)$', txt, re.M)
    shutil.rmtree(res); shutil.rmtree(log)
    return (m.group(1).replace(os.path.abspath(code_dir), '<W>').replace(log, '<LOG>') if m else None), p.stdout + p.stderr


old_r, new_r = os.path.join(a.old, 'scripts', 'run_ett_spk.sh'), os.path.join(C, 'scripts', 'run_ett_spk.sh')
for ds, kn in (('ETTh1', dict(COND='spkmx_lif_muxlin_iand', LR='0.00025', LAYERS='1', EMB='32', MLPR='4', WD='0.1')),
               ('ETTm1', dict(COND='spkmx_lif_msxdpca_iand', LR='0.0005', LAYERS='2', PATCH='16', EMB='32', MLPR='0.5', TMEM='off')),
               ('ETTh2', dict(COND='spk_ssa', NALL='lif', LR='0.0005', PATCH='16')), ('ETTm2', dict(COND='spk_ssa', NALL='lif', LR='0.0001'))):
    o, _ = args_of(old_r, a.old, kn, ds)
    n, _ = args_of(new_r, C, kn, ds)
    check(f'r0 {kn["COND"]} {ds}: same arguments as before', o is not None and o == n, '' if o == n else f'\n  {o}\n  {n}')
o, out = args_of(old_r, a.old, dict(COND='spk_ssa', NALL='lif', LR='0.0005'), 'ETTh1')
check('r0 the old runner refused spk_ssa on ETTh1', o is None and 'LIF SpkMixer conditions only' in out)
for ds1, ds2, kn in (('ETTh1', 'ETTh2', dict(COND='spk_ssa', NALL='lif', LR='0.0005')), ('ETTm1', 'ETTm2', dict(COND='spk_ssa', NALL='lif', LR='0.0001', PATCH='16'))):
    n1, _ = args_of(new_r, C, kn, ds1)
    n2, _ = args_of(new_r, C, kn, ds2)
    ok = n1 is not None and n2 is not None and n1.replace(ds1, ds2) == n2
    check(f'r1 spk_ssa {ds1} = {ds2} arguments with the dataset swapped', ok, '' if ok else f'\n  {n1}\n  {n2}')
shutil.rmtree(tmp)
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
