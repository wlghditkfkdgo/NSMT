"""Runner check for the MIT-BIH memoryless control (2026-11-28, exp/f-lif-pop-v3). No training. With a stub python that prints its
arguments: without TMEM the training arguments of run_mitbih_spk.sh equal those of the runner before this change; TMEM=off adds only
--token_memory off and the tag suffix _tmoff; TMEM=off with spk_ssa is refused.
usage (from cinc_mixer_v1/): python -B results/check-mitmemoff-20261128/check_runner.py --old <copy before the change>
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument('--old', required=True); a = ap.parse_args()
fails = 0


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}')


tmp = tempfile.mkdtemp()
stub = os.path.join(tmp, 'stub'); open(stub, 'w').write('#!/usr/bin/env bash\necho "ARGS $*"\nexit 3\n'); os.chmod(stub, 0o755)


def args_of(runner, code, envs):
    res, log = tempfile.mkdtemp(), tempfile.mkdtemp()
    p = subprocess.run(['bash', runner, '42', '0'], capture_output=True, text=True, env=dict(os.environ, PY=stub, RES=res, LOG=log, SUITE='mm', **envs))
    txt = ''.join(open(os.path.join(log, f)).read() for f in os.listdir(log) if f.endswith('.stdout'))
    m = re.search(r'^ARGS (.*)$', txt, re.M)
    shutil.rmtree(res); shutil.rmtree(log)
    return (m.group(1).replace(os.path.abspath(code), '<W>').replace(log, '<LOG>') if m else None), p.stdout + p.stderr


old, new = os.path.join(a.old, 'scripts', 'run_mitbih_spk.sh'), os.path.join(C, 'scripts', 'run_mitbih_spk.sh')
for kn in (dict(COND='spkmx_lif_muxlin_add', LR='0.02'), dict(COND='spk_ssa', NALL='lif', LR='0.01'), dict(COND='spkmx_lif_muxlin_or', LR='0.02', VAL_FOLD='0')):
    o, _ = args_of(old, a.old, kn); n, _ = args_of(new, C, kn)
    check(f'{kn}: same arguments without TMEM', o is not None and o == n, '' if o == n else f'\n  {o}\n  {n}')
n0, _ = args_of(new, C, dict(COND='spkmx_lif_muxlin_add', LR='0.02'))
n1, _ = args_of(new, C, dict(COND='spkmx_lif_muxlin_add', LR='0.02', TMEM='off'))
check('TMEM=off adds only --token_memory off and _tmoff', n0 is not None and n1 is not None and ' --token_memory off' in n1
      and n1.replace(' --token_memory off', '', 1).replace('_tmoff', '', 1) == n0)
n2, out = args_of(new, C, dict(COND='spk_ssa', NALL='lif', LR='0.01', TMEM='off'))
check('TMEM=off with spk_ssa is refused', n2 is None and 'LIF SpkMixer only' in out)
shutil.rmtree(tmp)
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
