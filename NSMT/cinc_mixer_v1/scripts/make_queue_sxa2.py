"""Queue of stage 2 of the fusion redesign (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG stage-2 pre-registration).
Reads the stage-1 selection (results/flif-v1-ett-sxa-s1/selected_s1.json: Q / K / V form d | p and shared yes | no) and queues the
three stage-2 arms in that form: sx<f>ot (OT plan mask, no MUX), msx<f>pc (Pearson mask + MUX), msx<f>ot (OT mask + MUX); LIF, ADD,
one block, first-chain settings as stage 1; ETTh1 / ETTm1 x H 96 / 720 x seeds 42 / 7 / 2026 = 36 lines, slow cells first.
The stage-1 rows of the selected Pearson arm and of muxlin independent are reused (same seeds, paired initial weights).
usage (from NSMT/): python -B cinc_mixer_v1/scripts/make_queue_sxa2.py [--suite flif-v1-ett-sxa-s2]  ->  prints <queue>:<lines>
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--suite', default='flif-v1-ett-sxa-s2')
ap.add_argument('--s1_suite', default='flif-v1-ett-sxa-s1')
a = ap.parse_args()
sel = json.load(open(os.path.join(C, 'results', a.s1_suite, 'selected_s1.json')))
if sel['comb'] not in ('sxdpc', 'sxppc') or not isinstance(sel['shared'], bool):
    sys.exit(f'STOP: unexpected selection {sel}')
f = sel['comb'][2]
ws = '1' if sel['shared'] else '0'
COMBS = (f'sx{f}ot', f'msx{f}pc', f'msx{f}ot')
KNOBS = {'ETTh1': dict(LR='0.00025', LAYERS='1', EMB='32', MLPR='4', WD='0.1'),
         'ETTm1': dict(LR='0.0005', LAYERS='2', PATCH='16', EMB='32', MLPR='0.5')}
lines = []
for ds in ('ETTm1', 'ETTh1'):
    for pl in ('720', '96'):
        for s in ('42', '7', '2026'):
            for comb in COMBS:
                knobs = ' '.join(f'{k}={v}' for k, v in KNOBS[ds].items())
                lines.append(f'SUITE={a.suite} COND=spkmx_lif_{comb}_add WSHARE={ws} {knobs} bash cinc_mixer_v1/scripts/run_ett_spk.sh {ds} {pl} {s}')
keys = {(l.split()[1], l.split()[-3], l.split()[-2], l.split()[-1]) for l in lines}
if not (len(lines) == len(set(lines)) == len(keys) == 36):
    sys.exit(f'STOP: {len(lines)} lines, {len(keys)} distinct (cond, dataset, horizon, seed) keys (want 36)')
qd = os.path.join(HERE, 'queues')
os.makedirs(qd, exist_ok=True)
q = os.path.join(qd, f'{a.suite}-{datetime.date.today():%Y%m%d}.txt')
if os.path.exists(q) and os.path.getsize(q) > 0:
    sys.exit(f'STOP: queue {q} exists and is not empty')
open(q, 'w').write('\n'.join(lines) + '\n')
print(f'{q}:{len(lines)}')
