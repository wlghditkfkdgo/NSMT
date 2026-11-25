"""Queue of the SpikF-LIF experiment (2026-11-26, exp/f-lif-pop-v3; PROJECT_LOG SpikF-LIF pre-registration; user request 11-25).
4 ETT datasets x H 96 / 192 / 336 / 720 x seeds 42 / 7 / 2026 x head activation {gelu, lif} = 96 lines, slow cells first (ETTm, long H).
Each line: SUITE=spikf-act-ett FAST=1 EVBS=256 SYBS=<s> ACT=<a> bash spikf_ref/forecasting/scripts/run_spikf.sh <ds> <pl> <seed>
(gpu_pool2.sh appends the GPU and runs the line from NSMT/). Refuses to overwrite a non-empty queue file; checks that every line has a
distinct run folder and stdout name.
usage (from NSMT/): python -B spikf_ref/forecasting/scripts/make_queue_spikf.py   ->   prints <queue path>:<lines>
"""
import argparse
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--sybs', default='1', choices=['1'])                  # syops at batch > 1 crashes (neuron state not reset between batches)
ap.add_argument('--suite', default='spikf-act-ett')
a = ap.parse_args()
DS = ('ETTm1', 'ETTm2', 'ETTh1', 'ETTh2')
PL = ('720', '336', '192', '96')
SEEDS = ('42', '7', '2026')
ACTS = ('lif', 'gelu')
lines, folders, outs = [], set(), set()
for ds in DS:
    for pl in PL:
        for s in SEEDS:
            for act in ACTS:
                lines.append(f'SUITE={a.suite} FAST=1 EVBS=256 SYBS={a.sybs} ACT={act} bash spikf_ref/forecasting/scripts/run_spikf.sh {ds} {pl} {s}')
                ft = f'_fe256_sy{a.sybs}'
                folders.add(f'{ds}/p{pl}/{act}{ft}_seed{s}')
                outs.add(f'spikf_{ds}_p{pl}_{act}{ft}_seed{s}')
if not (len(lines) == len(folders) == len(outs) == 96):
    sys.exit(f'STOP: {len(lines)} lines, {len(folders)} run folders, {len(outs)} stdout names (want 96 each)')
qd = os.path.join(HERE, 'queues')
os.makedirs(qd, exist_ok=True)
q = os.path.join(qd, f'{a.suite}-{datetime.date.today():%Y%m%d}.txt')
if os.path.exists(q) and os.path.getsize(q) > 0:
    sys.exit(f'STOP: queue {q} exists and is not empty')
open(q, 'w').write('\n'.join(lines) + '\n')
print(f'{q}:{len(lines)}')
