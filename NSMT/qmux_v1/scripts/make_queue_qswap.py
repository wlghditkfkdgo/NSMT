"""make_queue_qswap.py (2026-11-07, exp/f-lif-pop-v3; PROJECT_LOG query-swap pre-registration): queue of the plain query checkpoints, gpu_pool2.sh lines without the gpu argument, run from NSMT/.
Fail closed: a SKIP line of a used condition, a missing or duplicate run, a missing checkpoint.
  all    4 neurons x ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026 = 48, + 4 neurons x MIT-BIH x seeds = 12
  smoke  checks only: pop LIF ETTh2 H96 seed 42 with --max_batches 2 per source file (5 sources), so no replaced metric is seen
  repro  clean only, whole test split, one run per source file (5 sources): the logged metric reproduces from this folder's code
usage: python -B make_queue_qswap.py all|smoke|repro
"""
import os
import re
import sys

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1'
RAW = dict(ett_h2=os.path.join(MAIN, 'forecasting/results/flif-v1-ett-query/raw.txt'),
           ett_m2_pop=os.path.join(MAIN, 'forecasting/results/flif-v1-ett-query-m2/raw.txt'),
           ett_m2_lif=os.path.join(W, 'results/qmx-ett/raw.txt'),
           mit_pop=os.path.join(MAIN, 'classification/results/flif-v1-mitbih-query/raw.txt'),
           mit_lif=os.path.join(W, 'results/qmx-mitbih/raw.txt'))
NEURONS = ('pop_lif', 'pop_flif', 'lif', 'flif')
SEEDS = ('42', '7', '2026')
CELLS = (('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720'))


def stop(msg):
    print('STOP:', msg)
    sys.exit(1)


def rows(path, conds):
    out = {}
    for line in open(path):
        m = re.search(r'\bcond=(\S+)', line)
        if not m or m[1] not in conds:
            continue
        if line.startswith('SKIP'):
            stop(f'SKIP of a used condition in {path}: {line[:120]}')
        if line.startswith('RESULT'):
            r = dict(re.findall(r'(\w+)=(\S+)', line))
            key = (r.get('ds'), r.get('pl'), r['cond'], r['seed'])
            if key in out:
                stop(f'duplicate {key} in {path}')
            out[key] = r
    return out


def source(ds, n):
    if ds == 'ETTh2':
        return RAW['ett_h2']
    return RAW['ett_m2_pop'] if n.startswith('pop') else RAW['ett_m2_lif']


def ett_jobs():
    jobs = []
    for ds, pl in CELLS:
        for n in NEURONS:
            for s in SEEDS:
                raw = source(ds, n)
                r = rows(raw, {f'nsmt4q_{n}'}).get((ds, pl, f'nsmt4q_{n}', s))
                if r is None:
                    stop(f'missing {ds} H{pl} {n} seed {s} in {raw}')
                jobs.append(('ett', raw, r['dir'], f'ett_{ds}_H{pl}_{n}_s{s}', 'checkpoint.pth'))
    return jobs


def mit_jobs():
    jobs = []
    for n in NEURONS:
        raw = RAW['mit_pop'] if n.startswith('pop') else RAW['mit_lif']
        got = rows(raw, {f'nsmt4q_{n}'})
        for s in SEEDS:
            r = got.get((None, None, f'nsmt4q_{n}', s))
            if r is None:
                stop(f'missing MIT-BIH {n} seed {s} in {raw}')
            jobs.append(('mitbih', raw, r['dir'], f'mitbih_{n}_s{s}', 'best+model.pt'))
    return jobs


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage not in ('all', 'smoke', 'repro'):
        stop('usage: all|smoke|repro')
    jobs = ett_jobs() + mit_jobs()
    if len(jobs) != 60 or len({j[3] for j in jobs}) != 60:
        stop(f'{len(jobs)} jobs, expected 60 distinct')
    for j in jobs:
        ck = os.path.join(j[2], 'model_state', j[4])
        if not (os.path.isfile(ck) and os.path.isfile(os.path.join(j[2], 'model_state', 'config.pt'))):
            stop(f'no checkpoint {ck}')
    if stage != 'all':                                                 # one run per source file
        seen, keep = set(), []
        for j in jobs:
            if j[1] not in seen and j[3].endswith('_s42'):
                seen.add(j[1]); keep.append(j)
        if len(keep) != 5:
            stop(f'{len(keep)} sources, expected 5')
        jobs = keep
    q = os.path.join(W, 'scripts', 'queues', f'qswap-{stage}-20261107.txt')
    os.makedirs(os.path.dirname(q), exist_ok=True)
    env = dict(all='', smoke='OUT=smoke-qswap-20261107 EXTRA="--max_batches 2" ', repro='OUT=check-qswap-repro-20261107 EXTRA="--conds clean" ')[stage]
    with open(q, 'w') as f:
        for task, raw, run, name, _ in jobs:
            f.write(f"{env}bash qmux_v1/scripts/run_qswap.sh {task} '{raw}' '{run}' {name}\n")
    print(f'{q}:{len(jobs)}')


if __name__ == '__main__':
    main()
