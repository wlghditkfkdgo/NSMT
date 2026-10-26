"""Report only (MODI_PROJECT_LOG 2026-10-26): does a model's final training loss go with its CinC test score?

For every CinC fold-0 condition of the stage-9 line of this branch (suites flif-v1-rhythm-{cinc, fix, bix, lin, aux}-cinc, the
30-epoch runs; crop 0) take each run's last-epoch train CE (log/best_log_0.csv, epoch 29), its best validation score and epoch,
and its test score (best-validation weights, RESULT line). Average over seeds per condition and report Spearman's rank
correlation between the mean final train CE and the mean test score (and the best validation score), plus the share of runs
whose best validation epoch is in the last 6 of 30 epochs. No training, no new evaluation.
Fail-closed: a condition needs exactly one run per seed (42, 7, 2026), each with a 30-row log; anything else stops the script.
usage: python fit_vs_score.py [--out results/fit-vs-score-20261026]
"""
import argparse
import csv
import os
import re

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ('flif-v1-rhythm-cinc', 'flif-v1-rhythm-fix-cinc', 'flif-v1-rhythm-bix-cinc', 'flif-v1-rhythm-lin-cinc', 'flif-v1-rhythm-aux-cinc')
SEEDS = (42, 7, 2026)
EPOCHS = 30


def runs():
    out = {}
    for s in SUITES:
        for line in open(os.path.join(HERE, 'results', s, 'raw.txt')):
            if not line.startswith('RESULT '):
                continue
            d = dict(re.findall(r'(\S+?)=(\S+)', line))
            if d['data'] != 'CINC2017' or int(d.get('crop', 0)) != 0 or int(d.get('fold', 0)) != 0:
                continue
            key = (d['cond'], int(d['seed']))
            if key in out and out[key]['dir'] != d['dir']:
                raise SystemExit(f'two different runs for {key}: {out[key]["dir"]} / {d["dir"]}')
            log = list(csv.DictReader(open(os.path.join(d['dir'], 'log', 'best_log_0.csv'))))
            if len(log) != EPOCHS:
                raise SystemExit(f'{key}: {len(log)} epochs in {d["dir"]}')
            val = np.array([float(r['val_score']) for r in log])
            out[key] = dict(cond=d['cond'], seed=key[1], suite=s, dir=d['dir'], score=100 * float(d['score']),
                            macro_f1=100 * float(d['macro_f1']), ce_last=float(log[-1]['train_ce']),
                            best_val=100 * float(val.max()), best_epoch=int(val.argmax()), uJ=float(d['total_fft_uJ']))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'fit-vs-score-20261026'))
    a = ap.parse_args()
    R = runs()
    conds = sorted({c for c, _ in R})
    rows = []
    for c in conds:
        rr = [R.get((c, s)) for s in SEEDS]
        if any(r is None for r in rr) or any(not np.isfinite(r[k]) for r in rr for k in ('score', 'ce_last', 'best_val')):
            raise SystemExit(f'{c}: missing seed or non-finite value')
        rows.append(dict(cond=c, suite=rr[0]['suite'], n=3, **{k: float(np.mean([r[k] for r in rr]))
                                                              for k in ('score', 'macro_f1', 'ce_last', 'best_val', 'best_epoch', 'uJ')}))
    rows.sort(key=lambda r: -r['score'])
    ce = np.array([r['ce_last'] for r in rows]); sc = np.array([r['score'] for r in rows]); bv = np.array([r['best_val'] for r in rows])
    rho_s, p_s = spearmanr(ce, sc); rho_v, p_v = spearmanr(ce, bv)
    late = np.mean([r['best_epoch'] >= EPOCHS - 6 for r in R.values()])
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'fit_vs_score.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = [f'conditions {len(rows)} (x 3 seeds = {len(R)} runs), suites {", ".join(SUITES)}, CinC fold 0, {EPOCHS} epochs',
         f'Spearman(final train CE, test score)      = {rho_s:+.3f} (p = {p_s:.2e})',
         f'Spearman(final train CE, best val score)  = {rho_v:+.3f} (p = {p_v:.2e})',
         f'runs whose best validation epoch is in the last 6 epochs (24-29): {late * 100:.1f} %', '',
         f'{"condition":24s} {"score":>7s} {"macroF1":>8s} {"train CE":>9s} {"best val":>9s} {"best ep":>8s} {"uJ":>9s}']
    L += [f'{r["cond"]:24s} {r["score"]:7.2f} {r["macro_f1"]:8.2f} {r["ce_last"]:9.3f} {r["best_val"]:9.2f} {r["best_epoch"]:8.1f} {r["uJ"]:9.1f}' for r in rows]
    open(os.path.join(a.out, 'fit_vs_score.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
