"""Queue of stage 1 of the fusion redesign (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG pre-registration).
10 arms (fusion muxlin / sxd / sxdpc / sxp / sxppc x directional weights independent / shared; LIF, ADD residual, one block) x
ETTh1 / ETTm1 x H 96 / 720 x seeds 42 / 7 / 2026 = 120 lines, slow cells first (ETTm1, H720). First-chain settings:
  ETTh1  LR=0.00025 LAYERS=1 EMB=32 MLPR=4 WD=0.1                 (patch 8, blocks 1)
  ETTm1  LR=0.0005  LAYERS=2 PATCH=16 EMB=32 MLPR=0.5             (wd 0.06; blocks 1 instead of the first chain's 3)
Each line: SUITE=<suite> COND=spkmx_lif_<comb>_add WSHARE=<0|1> <knobs> bash cinc_mixer_v1/scripts/run_ett_spk.sh <ds> <pl> <seed>
(gpu_pool2.sh appends the GPU and runs it from NSMT/). The runner tag is rebuilt here exactly as run_ett_spk.sh builds it; every
line must have a distinct (dataset, horizon, tag, seed) = run folder and stdout name. Refuses to overwrite a non-empty queue.
usage (from NSMT/): python -B cinc_mixer_v1/scripts/make_queue_sxa.py [--suite flif-v1-ett-sxa-s1]  ->  prints <queue>:<lines>
"""
import argparse
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--suite', default='flif-v1-ett-sxa-s1')
a = ap.parse_args()
KNOBS = {'ETTh1': dict(LR='0.00025', LAYERS='1', EMB='32', MLPR='4', WD='0.1'),
         'ETTm1': dict(LR='0.0005', LAYERS='2', PATCH='16', EMB='32', MLPR='0.5')}
MLP0 = {'ETTh1': '0.5', 'ETTm1': '1'}
ARMS = [(c, w) for c in ('muxlin', 'sxd', 'sxdpc', 'sxp', 'sxppc') for w in ('0', '1')]


def tag(ds, cond, kn, ws):                                                   # run_ett_spk.sh: TAGC
    k = ''
    if kn.get('PATCH', '8') != '8':
        k += f'_pt{kn["PATCH"]}'
    if kn.get('EMB', '64') != '64':
        k += f'_e{kn["EMB"]}'
    if kn.get('MLPR', MLP0[ds]) != MLP0[ds]:
        k += f'_mlp{kn["MLPR"]}'
    if kn.get('BLOCKS', '1') != '1':
        k += f'_blk{kn["BLOCKS"]}'
    if kn.get('WD', '0.06') != '0.06':
        k += f'_wd{kn["WD"]}'
    return f'{cond}_lr{kn["LR"]}_L{kn["LAYERS"]}{k}' + ('_ws' if ws == '1' else '')


lines, keys = [], set()
for ds in ('ETTm1', 'ETTh1'):
    for pl in ('720', '96'):
        for s in ('42', '7', '2026'):
            for comb, ws in ARMS:
                cond = f'spkmx_lif_{comb}_add'
                kn = KNOBS[ds]
                knobs = ' '.join(f'{k}={v}' for k, v in kn.items())
                lines.append(f'SUITE={a.suite} COND={cond} WSHARE={ws} {knobs} bash cinc_mixer_v1/scripts/run_ett_spk.sh {ds} {pl} {s}')
                keys.add((ds, pl, tag(ds, cond, kn, ws), s))
if not (len(lines) == len(set(lines)) == len(keys) == 120):
    sys.exit(f'STOP: {len(lines)} lines, {len(set(lines))} distinct, {len(keys)} distinct run keys (want 120)')
qd = os.path.join(HERE, 'queues')
os.makedirs(qd, exist_ok=True)
q = os.path.join(qd, f'{a.suite}-{datetime.date.today():%Y%m%d}.txt')
if os.path.exists(q) and os.path.getsize(q) > 0:
    sys.exit(f'STOP: queue {q} exists and is not empty')
open(q, 'w').write('\n'.join(lines) + '\n')
print(f'{q}:{len(lines)}')
