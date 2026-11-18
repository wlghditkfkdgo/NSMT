"""MODI 2026-11-19 (user request 11-19, item 3): per-epoch train / validation loss trends of the current experiments (no training).

ETT (best_log_0.csv: train L1 = mean over the epoch's batches in training mode, val L1 = validation split in eval mode; the lr of
each epoch from the run's stdout; ReduceLROnPlateau x0.1 after 2 stalled epochs, early stopping after 10):
  per run: epochs run E, best-validation epoch b, train change epoch 0 -> end and best -> end, train change over the last 3 epochs,
  share of epochs where train fell, validation change epoch 0 -> best and best -> end, val - train gap at best and at the end, lr
  at the best epoch, number of lr cuts.
  flags: overfit = train fell >= 2 % after the best epoch while validation rose >= 1 %; first-epoch = b == 0; still-falling =
  train fell >= 0.5 % over the last 3 epochs; best-at-end = b == E - 1.
MIT-BIH (best_log_0.csv; 30 epochs, cosine lr, the last epoch is tested; the per-epoch "val" columns are the TEST file -- nothing is
selected on them, they are only read here): train CE epoch 0 -> end and over the last 5 epochs, test CE minimum epoch and the rise
from it to the end, test macro-F1 maximum epoch and the drop from it to the end.
  flag: overfit = test CE rose >= 10 % from its minimum to the end while train CE fell >= 30 % over the same epochs.
Thresholds are descriptive (set before reading the numbers); nothing here is a pre-registered verdict.
usage (classification folder): python loss_trends.py [--out_md F] [--out_csv F]
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.join(HERE, '..', 'forecasting', 'results')
FL = os.path.join(HERE, '..', 'forecasting', 'log')
CL = os.path.join(HERE, 'results')
CELLS = (('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720'))
ETT = [  # (group, suite, cond, model label)
    ('input 96', 'flif-v1-gen-ett', 'spk_ssa', 'Spikformer-style'),
    ('input 96', 'flif-v1-ett-ms', 'nsmt4q_pop_lif_msn', 'base pop LIF'),
    ('input 96', 'flif-v1-ett-ms', 'nsmt4q_pop_flif_msn', 'base pop f-LIF'),
    ('input 96 SpikeDE', 'flif-v1-ett-sde', 'spkall_lif', 'Spikformer-style SpikeDE'),
    ('input 96 SpikeDE', 'flif-v1-ett-sde', 'nsmt4q_pop_lif_msn_sde', 'base pop LIF SpikeDE'),
    ('input 96 SpikeDE', 'flif-v1-ett-sde', 'nsmt4q_pop_flif_msn_sde', 'base pop f-LIF SpikeDE'),
    ('input 96 XNOR', 'flif-v1-ett-xnor', 'spk_ssa_xl', 'Spikformer-style XNOR'),
    ('input 96 XNOR', 'flif-v1-ett-xnor', 'nsmt4q_pop_lif_msn_xl', 'base pop LIF XNOR'),
    ('input 96 XNOR', 'flif-v1-ett-xnor', 'nsmt4q_pop_flif_msn_xl', 'base pop f-LIF XNOR'),
] + [(f'input {L}', f'flif-v1-ett-len{L}', c, lab) for L in (336, 720) for c, lab in
     (('spk_ssa', 'Spikformer-style'), ('nsmt4q_pop_lif_msn', 'base pop LIF'), ('nsmt4q_pop_flif_msn', 'base pop f-LIF'))]
MB = [('flif-v1-gen-mitbih', 'spk_ssa', 'Spikformer-style', {}),
      ('flif-v1-mitbih-ms', 'nsmt4q_pop_lif_msn', 'base pop LIF', dict(lr='0.01', val='0')),
      ('flif-v1-mitbih-ms', 'nsmt4q_pop_flif_msn', 'base pop f-LIF', dict(lr='0.01', val='0')),
      ('flif-v1-mitbih-sde', 'spkall_lif', 'Spikformer-style SpikeDE', {}),
      ('flif-v1-mitbih-sde', 'nsmt4q_pop_lif_msn_sde', 'base pop LIF SpikeDE', dict(lr='0.01', val='0')),
      ('flif-v1-mitbih-sde', 'nsmt4q_pop_flif_msn_sde', 'base pop f-LIF SpikeDE', dict(lr='0.01', val='0'))]


def results(path, **sel):
    out = []
    for ln in open(path):
        if ln.startswith('RESULT'):
            d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
            if all(d.get(k) == str(v) for k, v in sel.items()):
                out.append(d)
    return out


def read_log(d):
    rows = list(csv.DictReader(open(os.path.join(d, 'log', 'best_log_0.csv'))))
    if not rows or [int(r['epoch']) for r in rows] != list(range(len(rows))):
        raise RuntimeError(f'bad epoch column in {d}')
    return rows


def ett_lrs(suite, ds, pl, cond, seed, sha, n):
    fs = [f for f in glob.glob(os.path.join(FL, suite, f'ett_{ds}_p{pl}_{cond}_seed{seed}_*_{sha}.stdout'))]
    lrs = []
    if len(fs) == 1:
        for ln in open(fs[0], errors='replace'):
            m = re.match(r'epoch (\d+) \(lr=([0-9.e+-]+)\) train L1', ln)
            if m:
                lrs.append(float(m.group(2)))
    return lrs if len(lrs) == n else None


def pct(a, b):
    return 100. * (a / b - 1.)


def ett_run(group, suite, cond, label, d):
    rows = read_log(d['dir'])
    tr = np.array([float(r['train_loss']) for r in rows]); va = np.array([float(r['val_loss']) for r in rows])
    if not (np.isfinite(tr).all() and np.isfinite(va).all()):
        raise RuntimeError(f'non-finite loss in {d["dir"]}')
    E = len(rows); b = int(np.argmin(va))
    be = d['best_epoch_val_epochs'].split(':')
    if int(be[0]) != b or int(be[2]) != E:
        raise RuntimeError(f'{d["dir"]}: log best {b} / {E} vs RESULT {be}')
    lrs = ett_lrs(suite, d['ds'], d['pl'], cond, d['seed'], d['sha'], E)
    falls = np.mean(tr[1:] < tr[:-1]) if E > 1 else float('nan')
    last3 = pct(tr[-1], tr[-4]) if E >= 4 else float('nan')
    return dict(task='ETT', group=group, suite=suite, cond=cond, model=label, cell=f"{d['ds']} H{d['pl']}", seed=d['seed'], E=E, best=b,
                tr_all=pct(tr[-1], tr[0]), tr_after=pct(tr[-1], tr[b]), tr_last3=last3, falls=100 * falls,
                va_to_best=pct(va[b], va[0]), va_after=pct(va[-1], va[b]), gap_best=va[b] - tr[b], gap_end=va[-1] - tr[-1],
                # lrs[i] = lr after epoch i's scheduler step (used in epoch i + 1); the start lr is 5e-4 for ETTh2 / ETTm2 here
                lr_end=(lrs[-1] if lrs else None), cuts=(sum(1 for i, v in enumerate(lrs) if v < (5e-4 if i == 0 else lrs[i - 1]) * 0.99) if lrs else None),
                overfit=pct(tr[-1], tr[b]) <= -2 and pct(va[-1], va[b]) >= 1, first=b == 0, falling=E >= 4 and last3 <= -0.5, at_end=b == E - 1)


def mb_run(suite, cond, label, d):
    rows = read_log(d['dir'])
    tr = np.array([float(r['train_ce']) for r in rows]); te = np.array([float(r['val_ce']) for r in rows]); f1 = np.array([float(r['val_f1']) for r in rows])
    if len(rows) != 30 or not (np.isfinite(tr).all() and np.isfinite(te).all()):
        raise RuntimeError(f'{d["dir"]}: {len(rows)} epochs or non-finite')
    if abs(f1[-1] - float(d['f1'])) > 1e-5:
        raise RuntimeError(f'{d["dir"]}: last-epoch F1 {f1[-1]} vs RESULT {d["f1"]}')
    m = int(np.argmin(te)); k = int(np.argmax(f1))
    return dict(task='MIT-BIH', suite=suite, cond=cond, model=label, seed=d['seed'], tr_all=pct(tr[-1], tr[0]), tr_last5=pct(tr[-1], tr[-6]),
                te_min_ep=m, te_rise=pct(te[-1], te[m]), tr_fall_since_min=pct(tr[-1], tr[m]), f1_max_ep=k, f1_max=100 * f1[k],
                f1_end=100 * f1[-1], f1_drop=100 * (f1[-1] - f1[k]), ce_end_train=tr[-1], ce_end_test=te[-1],
                overfit=pct(te[-1], te[m]) >= 10 and pct(tr[-1], tr[m]) <= -30)


def fmt(v, f='{:+.1f}'):
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_md', default=os.path.join(CL, 'flif-v1-loss-trends.md'))
    ap.add_argument('--out_csv', default=os.path.join(CL, 'flif-v1-loss-trends.csv'))
    a = ap.parse_args()
    E, M = [], []
    for group, suite, cond, label in ETT:
        for ds, pl in CELLS:
            rows = results(os.path.join(FC, suite, 'raw.txt'), ds=ds, pl=pl, cond=cond)
            if len(rows) != 3:
                raise RuntimeError(f'{suite} {cond} {ds} {pl}: {len(rows)} rows')
            E += [ett_run(group, suite, cond, label, d) for d in rows]
    for suite, cond, label, sel in MB:
        rows = results(os.path.join(CL, suite, 'raw.txt'), cond=cond, **sel)
        if len(rows) != 3:
            raise RuntimeError(f'{suite} {cond}: {len(rows)} rows')
        M += [mb_run(suite, cond, label, d) for d in rows]
    keys = sorted({k for r in E + M for k in r})
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in E + M]
    L = ['# Loss trends of the current experiments (per-epoch logs; no training) — summary', '',
         'See the module docstring for the definitions. Values are medians over the runs of a row unless a count. Train / validation L1 '
         '(ETT) are on different splits and modes; the gap is descriptive.', '',
         '## ETT by model and input length (12 runs per row: 4 cells x 3 seeds)', '',
         '| input | model | epochs (median, min–max) | best epoch (median; # at epoch 0) | train change, epoch 0 → end % | train change after best % | '
         'train fell in % of epochs | train, last 3 epochs % | val change, epoch 0 → best % | val change, best → end % | gap val − train at best / end | '
         '# overfit | # train still falling | # best at last epoch |', '|' + '---|' * 14]
    for group, suite, cond, label in ETT:
        R = [r for r in E if r['suite'] == suite and r['cond'] == cond]
        med = lambda k: float(np.median([r[k] for r in R]))
        L.append(f"| {group} | {label} | {med('E'):.0f} ({min(r['E'] for r in R)}–{max(r['E'] for r in R)}) | {med('best'):.0f}; {sum(r['first'] for r in R)} | "
                 f"{med('tr_all'):+.1f} | {med('tr_after'):+.1f} | {med('falls'):.0f} | {med('tr_last3'):+.2f} | {med('va_to_best'):+.1f} | {med('va_after'):+.1f} | "
                 f"{med('gap_best'):+.3f} / {med('gap_end'):+.3f} | {sum(r['overfit'] for r in R)} | {sum(r['falling'] for r in R)} | {sum(r['at_end'] for r in R)} |")
    L += ['', '## ETT by cell (input 96 base and Spikformer-style, spikingjelly; 3 seeds per row)', '',
          '| cell | model | best epoch per seed | epochs per seed | train change after best % (per seed) | val change best → end % (per seed) | overfit |',
          '|---|---|---|---|---|---|---|']
    for ds, pl in CELLS:
        for suite, cond, label in (('flif-v1-gen-ett', 'spk_ssa', 'Spikformer-style'), ('flif-v1-ett-ms', 'nsmt4q_pop_lif_msn', 'base pop LIF'),
                                   ('flif-v1-ett-ms', 'nsmt4q_pop_flif_msn', 'base pop f-LIF')):
            R = sorted([r for r in E if r['suite'] == suite and r['cond'] == cond and r['cell'] == f'{ds} H{pl}'], key=lambda r: r['seed'])
            L.append(f"| {ds} H{pl} | {label} | {' / '.join(str(r['best']) for r in R)} | {' / '.join(str(r['E']) for r in R)} | "
                     f"{' / '.join(fmt(r['tr_after']) for r in R)} | {' / '.join(fmt(r['va_after']) for r in R)} | {sum(r['overfit'] for r in R)}/3 |")
    L += ['', '## MIT-BIH (30 epochs, cosine lr, last epoch tested; "test" = the per-epoch test-file columns, read only)', '',
          '| model | train CE epoch 0 → end % | train CE last 5 epochs % | test CE min epoch (per seed) | test CE rise min → end % (per seed) | '
          'test F1 max epoch (per seed) | F1 max → end %p (per seed) | train / test CE at the end | # overfit |', '|---|---|---|---|---|---|---|---|---|']
    for suite, cond, label, _ in MB:
        R = sorted([r for r in M if r['suite'] == suite and r['cond'] == cond], key=lambda r: r['seed'])
        med = lambda k: float(np.median([r[k] for r in R]))
        L.append(f"| {label} | {med('tr_all'):+.1f} | {med('tr_last5'):+.1f} | {' / '.join(str(r['te_min_ep']) for r in R)} | "
                 f"{' / '.join(fmt(r['te_rise']) for r in R)} | {' / '.join(str(r['f1_max_ep']) for r in R)} | {' / '.join(fmt(r['f1_drop'], '{:+.2f}') for r in R)} | "
                 f"{med('ce_end_train'):.4f} / {med('ce_end_test'):.4f} | {sum(r['overfit'] for r in R)}/3 |")
    open(a.out_md, 'w').write('\n'.join(L) + '\n')
    print(a.out_md, len(E), 'ETT runs', len(M), 'MIT-BIH runs')


if __name__ == '__main__':
    main()
