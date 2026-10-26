"""Recover the RESULT lines that run_rhythm5.sh failed to write for the CinC fold runs (MODI_PROJECT_LOG 2026-10-25).

run_rhythm5.sh finds a run's directory with `ls -td <log>/*/*+<cond>+lr<lr>+seed<s>+*/seed<s>_* | head -1`, i.e. the most
recently modified directory of that condition and seed -- the fold is not in the pattern. When folds of the same condition and
seed ran at the same time, it picked another fold's directory that had no final+result.csv yet and wrote a SKIP line although
the run had finished. This script reads every SKIP line of results/<suite>/raw.txt, finds the run's own directory (inner name
ending in _f<fold>), checks that its final+result.csv / last/final+result.csv / best_epoch.txt exist and that its saved config
has this condition's fold and seed, and writes the RESULT line in run_rhythm5.sh's format to results/<suite>/raw_recovered.txt
(created with 'x'). raw.txt is not modified. A recovered key that already has a RESULT line, or a SKIP line without exactly one
matching directory, stops the script (fail closed). time_s is not recoverable and is written as na.
usage: python recover_fold_results.py --suite flif-v1-rhythm-folds-cinc
"""
import argparse
import csv
import glob
import os
import re

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suite', required=True)
    a = ap.parse_args()
    R = os.path.join(HERE, 'results', a.suite)
    raw = open(os.path.join(R, 'raw.txt')).read().splitlines()
    have = set()
    for line in raw:
        if line.startswith('RESULT '):
            d = dict(re.findall(r'(\S+?)=(\S+)', line))
            have.add((d['cond'], int(d['fold']), int(d['seed'])))
    out = []
    for line in raw:
        if not line.startswith('SKIP'):
            continue
        d = dict(re.findall(r'(\S+?)=(\S+)', line))
        cond, fold, seed, lr, data = d['cond'], int(d['fold']), int(d['seed']), d['lr'], d['data']
        if (cond, fold, seed) in have:
            raise SystemExit(f'{cond} fold {fold} seed {seed} has a RESULT line already')
        log = os.path.join(HERE, 'log', a.suite, data)
        dirs = [p for p in glob.glob(os.path.join(log, '*', f'*+{cond}+lr{lr}+seed{seed}+*', f'seed{seed}_*')) if p.endswith(f'_f{fold}')]
        if len(dirs) != 1:
            raise SystemExit(f'{cond} fold {fold} seed {seed}: {len(dirs)} directories')
        D = dirs[0]
        cfg = torch.load(os.path.join(D, 'model_state', 'config.pt'), map_location='cpu')
        if int(cfg['fold']) != fold or int(cfg['seed']) != seed or cfg['data'] != data:
            raise SystemExit(f'{D}: config fold {cfg["fold"]} seed {cfg["seed"]} data {cfg["data"]}')
        b = next(csv.DictReader(open(os.path.join(D, 'log', 'final+result.csv'))))
        l = next(csv.DictReader(open(os.path.join(D, 'log', 'last', 'final+result.csv'))))
        vals = ' '.join(f'{k}={b[k]}' for k in b if k != 'weights') + ' ' + ' '.join(f'last_{k}={l[k]}' for k in ('score', 'acc', 'macro_f1'))
        be = open(os.path.join(D, 'log', 'best_epoch.txt')).read().strip().replace(' ', ':')
        stdout = d['out']
        par = re.findall(r'parameters : (\d+)', open(stdout, errors='ignore').read())
        if not par:
            raise SystemExit(f'{stdout}: no parameter count')
        sha = re.search(r'_([0-9a-f]{7,})\.stdout$', stdout).group(1)
        out.append(f'RESULT exp=nsmt_flif_v1_rhythm data={data} cond={cond} fold={fold} seed={seed} lr={lr} crop={d.get("crop", 0)} {vals} '
                   f'best_epoch_val={be} params={par[-1]} time_s=na sha={sha} dir={D} recovered=1')
        have.add((cond, fold, seed))
    with open(os.path.join(R, 'raw_recovered.txt'), 'x') as handle:
        handle.write('\n'.join(out) + '\n')
    print(f'{len(out)} RESULT lines recovered -> {os.path.join(R, "raw_recovered.txt")}')


if __name__ == '__main__':
    main()
