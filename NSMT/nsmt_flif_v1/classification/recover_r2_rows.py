"""MODI 2026-11-24: write the RESULT lines of the range-2 ETT search runs that finished but got a SKIP line (docs/MODI_PROJECT_LOG.md
2026-11-24).

Why: run_ett_dual.sh / run_ett_mixer.sh found a run's folder after training with a pattern that had no lr when LR was unset (the
dataset default 0.0005). In the range-2 search the default-lr runs shared their folder tag with the LR=0.00025 / 0.001 runs of the
same model, batch and warmup, the pattern matched all of them and `ls -td | head -1` took the newest -- for 5 runs a folder of another
lr that was still training (no final+result.csv), so the runner wrote a SKIP line although the run had finished. (Fixed in the
runners on 11-24; check_r2.py f.) No other row was affected: every RESULT line of the search passes select_r2.check_row.

For every SKIP line of results/<suite>/raw.txt not yet answered, this script
  - reads model, dataset, horizon, seed, batch (_b), warmup (_w), lr (_lr) and sha from the SKIP line and its stdout file name,
  - requires the stdout to end the training ("done in Ns", no Traceback) and to name the parameter count,
  - finds the run's folder with the lr always in the pattern and requires exactly one,
  - requires the saved config (model_state/config.pt) to hold this dataset, horizon, seed, lr, batch, warmup, epochs, input 96 and the
    range-1 / bias-off settings (wd_exclude_1d 1, val_per_epoch 4, ssa_pe 0, weight decay 0.06, no head dropout),
  - and builds the RESULT line exactly as the runner does (field order and values from final+result.csv, last/final+result.csv,
    best_epoch.txt), with time_s = the stdout's "done in" seconds (training + tests: a few seconds less than the runner's wall time),
    followed by out=<the SKIP line's stdout> collected=1.
Everything is checked before anything is written; any failure stops the script without writing (fail closed). The lines are appended to
raw.txt (append-only: the SKIP lines stay). select_r2.py accepts a SKIP line only when exactly one collected=1 line names its stdout.
usage (classification folder): python recover_r2_rows.py [--suite flif-v1-ett-r2s] [--dry]
"""
import argparse
import csv
import glob
import os
import re

import torch

import select_r2 as S

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = {'ETTh2': dict(lr='0.0005', bs=64, e=100), 'ETTm2': dict(lr='0.0005', bs=64, e=50)}   # run_ett_dual.sh / run_ett_mixer.sh HP


def kv(line):
    return dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)


def rebuild(skip, suite):
    d = kv(skip); ds, pl, cond, seed, out = d['ds'], int(d['pl']), d['cond'], int(d['seed']), d['out']
    if ds not in DEFAULT:
        raise SystemExit(f'{out}: dataset {ds}')
    m = re.fullmatch(rf'ett_{ds}_p{pl}_{re.escape(cond)}(?:_b(\d+))?(?:_w([0-9.]+))?(?:_lr([0-9.e-]+))?_seed{seed}_(\d{{8}})_([0-9a-f]+)\.stdout',
                     os.path.basename(out))
    if not m:
        raise SystemExit(f'{out}: stdout name')
    bs, wu, lr, _, sha = m.groups()
    lrf, E = lr or DEFAULT[ds]['lr'], DEFAULT[ds]['e']
    txt = open(out, errors='replace').read()
    done, par = re.findall(r'done in (\d+)s', txt), re.findall(r'number of parameters : (\d+)', txt)
    if 'Traceback' in txt or not done or not par:
        raise SystemExit(f'{out}: unfinished or crashed run (done in: {done}, parameters: {par})')
    tag = cond + (f'_b{bs}' if bs else '') + (f'_w{wu}' if wu else '')
    log = os.path.join(S.FC, '..', 'log', suite, ds, f'p{pl}')
    dirs = glob.glob(os.path.join(log, '*', f'*+{tag}+seed{seed}+*+epoch+{E}+lr+{lrf}+*', f'seed{seed}_*'))
    if len(dirs) != 1:
        raise SystemExit(f'{out}: {len(dirs)} run folders')
    D = os.path.abspath(dirs[0])
    cfg = torch.load(os.path.join(D, 'model_state', 'config.pt'), map_location='cpu')
    want = dict(data=ds, pred_len=pl, seed=seed, lr=float(lrf), batch_size=int(bs or DEFAULT[ds]['bs']), lr_warmup=float(wu) if wu else -1.0,
                epoch=E, seq_len=96, wd_exclude_1d=1, val_per_epoch=4, ssa_pe=0, weight_decay=0.06, head_dropout=0.0)
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise SystemExit(f'{D}: saved config differs {bad}')
    csvp, lcsv, bep = (os.path.join(D, 'log', p) for p in ('final+result.csv', 'last/final+result.csv', 'best_epoch.txt'))
    b, l = next(csv.DictReader(open(csvp))), next(csv.DictReader(open(lcsv)))
    vals = ' '.join(f'{k}={b[k]}' for k in b if k != 'weights') + ' ' + ' '.join(f'last_{k}={l[k]}' for k in ('mse', 'mae'))
    be = open(bep).read().strip().replace(' ', ':')
    exp = 'nsmt_flif_v1_gen_ett' if cond.startswith('spk') else 'nsmt_flif_v1_ett_dual'
    line = (f'RESULT exp={exp} ds={ds} pl={pl} cond={cond} seed={seed} {vals} best_epoch_val_epochs={be} params={par[-1]} time_s={done[-1]} '
            f'seq_len=96{f" lr={lr}" if lr else ""} wdx=1 vpe=4{f" bs={bs}" if bs else ""}{f" wu={wu}" if wu else ""} pe=0 sha={sha} dir={D} '
            f'out={out} collected=1')
    c = (lrf, str(bs or DEFAULT[ds]['bs']), wu or 'none')
    m_ = [k for k in S.MODELS if S.cond(k) == cond]
    if len(m_) != 1 or c not in S.combos('ett'):
        raise SystemExit(f'{out}: not a range-2 model / combination ({cond}, {c})')
    S.check_row('ett', kv(line), m_[0], c, f'recovered {out}')         # the same check select_r2 applies to every row
    return line, (ds, pl, cond, seed, bs, wu, lr)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suite', default=S.SEARCH['ett']); ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    raw = os.path.join(S.FC, a.suite, 'raw.txt')
    lines = open(raw).read().splitlines()
    res = [kv(x) for x in lines if x.startswith('RESULT')]
    done = {d['out'] for d in res if d.get('collected') == '1'}
    have = {(d['ds'], int(d['pl']), d['cond'], int(d['seed']), d.get('bs'), d.get('wu'), d.get('lr')) for d in res}
    todo = [x for x in lines if x.startswith('SKIP') and kv(x)['out'] not in done]
    new = []
    for s in todo:
        line, key = rebuild(s, a.suite)
        if key in have:
            raise SystemExit(f'{kv(s)["out"]}: a RESULT line for {key} exists already')
        have.add(key); new.append(line)
    for x in new:
        print(x[:200] + ' ...')
    if not a.dry and new:
        with open(raw, 'a') as f:
            f.write(''.join(x + '\n' for x in new))
    print(f'{len(new)} lines {"would be " if a.dry else ""}appended to {raw} ({len(todo)} SKIP lines unanswered before)')


if __name__ == '__main__':
    main()
