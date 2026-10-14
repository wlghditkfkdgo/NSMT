"""Build the post-training queues (future-interval evaluation and memory probe) from a suite's raw.txt.
usage: python make_post_queues.py --suite flif-v1-main   -> scripts/queues/{post-future,post-probe}-<suite>.txt
Queue lines are evaluated from NSMT/ by scripts/gpu_pool2.sh, which appends the GPU index (so lines end in -nd)."""
import os
import re
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--suite', default='flif-v1-main'); ap.add_argument('--windows', type=int, default=512)
a = ap.parse_args()
dirs = []
for line in open(os.path.join(HERE, 'results', a.suite, 'raw.txt')):
    if line.startswith('RESULT'):
        kv = dict(re.findall(r'(\w+)=(\S+)', line)); dirs.append((kv['cond'], kv['dir']))
fut = os.path.join(HERE, 'scripts', 'queues', f'post-future-{a.suite}.txt')
prb = os.path.join(HERE, 'scripts', 'queues', f'post-probe-{a.suite}.txt')
cd = 'cd nsmt_flif_v1/forecasting && LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib OMP_NUM_THREADS=4 /home/yschoi/.conda/envs/snn_recall/bin/python'
with open(fut, 'w') as f:
    for c, d in dirs:
        f.write(f'{cd} ett_future.py --run "{d}" --split future --out results/{a.suite}/future -nd\n')
with open(prb, 'w') as f:
    for c, d in dirs:
        if c != 'hippo_only':
            f.write(f'{cd} neo_memory_probe.py --run "{d}" --windows {a.windows} --out results/{a.suite}/memory_probe -nd\n')
print(fut, sum(1 for _ in open(fut)), 'lines;', prb, sum(1 for _ in open(prb)), 'lines')
