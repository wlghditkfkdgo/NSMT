"""MODI 2026-11-26 ((나) Sleep-EDF SNN comparison, docs/MODI_PROJECT_LOG.md 2026-11-26 pre-registration): stage 1 -> stage 2.

Stage 1 (suite flif-v1-sleep-lr): the 5 models x lr {0.001, 0.003}, fold 0, seed 42. For every model the lr with the higher best
validation macro-F1 (best_epoch_val = epoch:score, validation only) is chosen; a tie keeps 0.001. Fail closed: exactly one RESULT line
per (model, lr), no SKIP, epochs 30, batch 64, finite scores, one sha. Writes results/flif-v1-sleep-lr/selected.json and the stage-2
queue (folds 1-4 at the chosen lr, suite flif-v1-sleep); fold 0 of the main comparison is the stage-1 run with the chosen lr.
usage (classification folder): python select_sleep.py --queue <stage-2 queue file> [--stage1 flif-v1-sleep-lr] [--main flif-v1-sleep]
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ['spkall_lif', 'q_pop_lif', 'q_pop_flif', 'kv_pop_lif', 'kv_pop_flif']
LRS = ['0.001', '0.003']
RUNNER = 'nsmt_flif_v1/classification/scripts/run_sleep.sh'


def kv(line):
    return dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)


def rows(suite):
    p = os.path.join(HERE, 'results', suite, 'raw.txt')
    lines = open(p).read().splitlines()
    sk = [l for l in lines if l.startswith('SKIP')]
    if sk:
        raise SystemExit(f'STOP: {len(sk)} SKIP lines in {suite}: {sk[0][:160]}')
    return [kv(l) for l in lines if l.startswith('RESULT')]


def check(r, key):
    if r['epochs'] != '30' or r['bs'] != '64' or r['seed'] != '42' or r['data'] != 'SLEEPEDF':
        raise SystemExit(f'STOP: {key} epochs {r["epochs"]} bs {r["bs"]} seed {r["seed"]} data {r["data"]}')
    for k in ('score', 'macro_f1', 'acc', 'syn_energy_uJ'):
        if not math.isfinite(float(r[k])):
            raise SystemExit(f'STOP: {key} {k} = {r[k]}')
    v = float(r['best_epoch_val'].split(':')[1])
    if not math.isfinite(v):
        raise SystemExit(f'STOP: {key} validation score {r["best_epoch_val"]}')
    return v


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--queue', required=True)
    ap.add_argument('--stage1', default='flif-v1-sleep-lr'); ap.add_argument('--main', default='flif-v1-sleep')
    a = ap.parse_args()
    t, shas = {}, set()
    for r in rows(a.stage1):
        key = (r['cond'], r['lr'], r['fold'])
        if r['cond'] not in CONDS or r['lr'] not in LRS or r['fold'] != '0':
            raise SystemExit(f'STOP: unexpected stage-1 row {key}')
        if key in t:
            raise SystemExit(f'STOP: duplicate stage-1 row {key}')
        t[key] = check(r, key); shas.add(r['sha'])
    want = {(c, lr, '0') for c in CONDS for lr in LRS}
    if set(t) != want:
        raise SystemExit(f'STOP: stage-1 rows missing: {sorted(want - set(t))}')
    if len(shas) != 1:
        raise SystemExit(f'STOP: code versions {sorted(shas)}')
    sel = {}
    for c in CONDS:
        v1, v3 = t[(c, '0.001', '0')], t[(c, '0.003', '0')]
        sel[c] = dict(lr='0.003' if v3 > v1 else '0.001', val={'0.001': v1, '0.003': v3})
    out = dict(stage1=a.stage1, main=a.main, sha=shas.pop(), rule='higher best validation macro-F1, tie -> 0.001', selected=sel)
    json.dump(out, open(os.path.join(HERE, 'results', a.stage1, 'selected.json'), 'w'), indent=1)
    with open(a.queue, 'w') as f:
        for fold in (1, 2, 3, 4):
            for c in CONDS:
                f.write(f'SUITE={a.main} LR={sel[c]["lr"]} bash {RUNNER} {c} {fold}\n')
    for c in CONDS:
        print(f'{c:12s} val 0.001 {sel[c]["val"]["0.001"]:.4f}  0.003 {sel[c]["val"]["0.003"]:.4f}  -> lr {sel[c]["lr"]}')
    print(f'stage-2 queue: {a.queue} ({4 * len(CONDS)} lines)')


if __name__ == '__main__':
    main()
