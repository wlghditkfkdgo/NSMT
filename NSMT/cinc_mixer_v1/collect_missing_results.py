"""Append RESULT lines for finished runs that scripts/run_rhythm.sh did not record (MODI 2026-10-21).

Why: run_rhythm.sh was edited while aux-loss jobs were running; bash reads a script by byte offset, so the instances that
reached the post-training lines after the edit executed garbage and wrote neither RESULT nor SKIP (training and the
final+result.csv / last/ / best_epoch.txt artifacts are intact). This rebuilds the line from those artifacts in the same
field order as run_rhythm.sh and marks it `collected=1`; `time_s` comes from the stdout's "done in Ns" (training + both
tests, i.e. slightly less than the script's wall time).
usage: python collect_missing_results.py --suite flif-v1-rhythm-aux-cinc [--dry]
"""
import argparse
import csv
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', required=True); ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    raw = os.path.join(HERE, 'results', a.suite, 'raw.txt')
    have = set()
    for l in open(raw):
        if l.startswith(('RESULT', 'SKIP')):
            d = dict(re.findall(r'(\S+?)=(\S+)', l)); have.add((d['data'], d['cond'], d['seed'], d.get('crop', '0')))
    added = 0
    for out in sorted(glob.glob(os.path.join(HERE, 'log', a.suite, '*.stdout'))):
        m = re.match(r'(CINC2017|SHD)_(.+)_f(\d+)_seed(\d+)_lr([0-9.e-]+)_(\d{8})_([0-9a-f]+)\.stdout$', os.path.basename(out))
        if not m:
            continue
        data, tagc, fold, seed, lr, _, sha = m.groups()
        crop = '0'
        mm = re.match(r'(.+)_crop(\d+)$', tagc)
        if mm:
            tagc, crop = mm.groups()
        if (data, tagc, seed, crop) in have:
            continue
        txt = open(out, errors='replace').read()
        done = re.search(r'done in (\d+)s', txt)
        if not done:
            continue                                                   # still running or crashed: leave to the script / user
        dirs = sorted(glob.glob(os.path.join(HERE, 'log', a.suite, data, '*', f'*+{tagc}+lr{lr}+seed{seed}+*', f'seed{seed}_*')), key=os.path.getmtime)
        if not dirs:
            print('no run dir for', out); continue
        D = dirs[-1]
        csvp, lcsv = os.path.join(D, 'log', 'final+result.csv'), os.path.join(D, 'log', 'last', 'final+result.csv')
        if not (os.path.exists(csvp) and os.path.exists(lcsv)):
            print('incomplete artifacts for', D); continue
        b = next(csv.DictReader(open(csvp))); last = next(csv.DictReader(open(lcsv)))
        vals = ' '.join(f'{k}={b[k]}' for k in b if k != 'weights') + ' ' + ' '.join(f'last_{k}={last[k]}' for k in ('score', 'acc', 'macro_f1'))
        be = open(os.path.join(D, 'log', 'best_epoch.txt')).read().split()
        par = re.search(r'number of parameters : (\d+)', txt)
        line = (f'RESULT exp=nsmt_flif_v1_rhythm data={data} cond={tagc} fold={fold} seed={seed} lr={lr} crop={crop} {vals} '
                f'best_epoch_val={be[0]}:{be[1]} params={par.group(1) if par else "nan"} time_s={done.group(1)} sha={sha} dir={D} collected=1')
        print(line[:160] + ' ...')
        if not a.dry:
            open(raw, 'a').write(line + '\n')
        added += 1
    print(f'{added} lines {"would be " if a.dry else ""}appended to {raw}')


if __name__ == '__main__':
    main()
