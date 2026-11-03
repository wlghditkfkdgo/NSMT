"""Head input forms, post-hoc observations (2026-11-03, exp/f-lif-pop-v3; report only, not pre-registered). For every form and
model, next to the control rows that summarize_hp.py compares against (same seeds 42 / 7 / 2026): first / last-epoch training
loss and, on CinC, the best validation score (log/best_log_0.csv), test of the last-epoch weights (CinC last_score; MIT-BIH tests
the last-epoch weights already), firing outside the Neo-mixer (rho_hippo) and in it (rho_neo), run time, and the Frobenius norms
of head.weight and of the new term (pool_readout.weight, attn pool_q) in model_state/best+model.pt.
usage: python posthoc_hp.py --task cinc|mitbih   ->  results/flif-v1-headpool/posthoc_<task>.txt
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_hp import hp_raw, OUT, FORMS                               # noqa: E402

SEEDS = (42, 7, 2026)


def num(x):
    return float(x) if x not in (None, 'None', '') else float('nan')


def info(r, task):
    d = r['dir']
    rows = list(csv.DictReader(open(os.path.join(d, 'log', 'best_log_0.csv'))))
    if len(rows) != 30:
        raise SystemExit(f'{d}: {len(rows)} epochs, expected 30')
    tl = [float(x['train_loss']) for x in rows]
    o = {'seed': int(r['seed']), 'first': tl[0], 'last': tl[-1], 'rho_h': num(r['rho_hippo']), 'rho_n': num(r.get('rho_neo')),
         't': int(r['time_s']), 'sha': r.get('sha')}
    if task == 'cinc':
        vs = [float(x['val_score']) for x in rows]
        o.update(test=float(r['score']), bestval=max(vs), bep=int(np.argmax(vs)), last_test=float(r['last_score']))
    else:
        o.update(test=float(r['f1']))
    sd = torch.load(os.path.join(d, 'model_state', 'best+model.pt'), map_location='cpu')
    sd = sd.get('model', sd)
    o['head'] = float(sd['head.weight'].norm())
    o['pool'] = next((float(sd[k].norm()) for k in ('pool_readout.weight', 'pool_q') if k in sd), float('nan'))
    return o


def collect(task):
    if task == 'cinc':
        import summarize_cinc_tune as s
        sel = {k: json.load(open(os.path.join(s.TUNE, f'selected_{k}.json')))['selected'] for k in ('lr', 'layers', 'res')}
        def cfg(b):
            ck = s.CONDKEY[b]
            return ('-', sel['lr'][ck], s.L0) if b == s.REF else (sel['res'][ck], sel['lr'][ck], sel['layers'][ck])
        pick = lambda D, b: s.seeds(D, b, cfg(b))
        key = lambda r, base, res: (base, res, float(r['lr']), int(r['layers']), int(r['seed']))
    else:
        import summarize_mitbih_tune as s
        sel = {k: json.load(open(os.path.join(s.TUNE, f'selected_{k}.json')))['selected'] for k in ('lr', 'layers', 'res')}
        pick = lambda D, b: s.seeds(D, b, s.tuned_cfg(b, sel), -1)
        key = lambda r, base, res: (base, res, float(r['lr']), int(r['layers']), int(r['val_fold']), int(r['seed']))
    out = {'control': {b: pick(s.load(), b) for b in (s.MUX, s.REF)}}
    for form, rows in hp_raw(task).items():
        H = {}
        for r in rows:
            base, res = s.split(r['cond'])
            H[key(r, base, res)] = r
        out[form] = {b: pick(H, b) for b in (s.MUX, s.REF)}
    names = {s.MUX: 'muxlin', s.REF: 'Spikformer-style'}
    res = {}
    for k, by in out.items():
        for b, rr in by.items():
            if rr is None or sorted(int(r['seed']) for r in rr) != sorted(SEEDS):
                raise SystemExit(f'{k} {b}: missing runs')
            res[(k, names[b])] = sorted((info(r, task) for r in rr), key=lambda o: SEEDS.index(o['seed']))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--task', choices=['cinc', 'mitbih'], required=True); a = ap.parse_args()
    res = collect(a.task)
    cols = [('first', 'train loss, epoch 0', '{:.4f}'), ('last', 'train loss, epoch 29', '{:.4f}'), ('test', 'test (reported)', '{:.4f}')]
    if a.task == 'cinc':
        cols += [('bestval', 'best validation score', '{:.4f}'), ('bep', 'best epoch', '{}'), ('last_test', 'test, last-epoch weights', '{:.4f}')]
    cols += [('rho_h', 'rho_hippo', '{:.4f}'), ('rho_n', 'rho_neo', '{:.4f}'), ('t', 'time s', '{}'),
             ('head', '|head.weight|', '{:.2f}'), ('pool', '|new term|', '{:.3f}')]
    L = [f'# head input forms, post-hoc (report only) — {a.task}; values per seed 42 / 7 / 2026', '']
    for model in ('muxlin', 'Spikformer-style'):
        L += [f'## {model}', '', '| form | ' + ' | '.join(c[1] for c in cols) + ' |', '|---|' + '---|' * len(cols)]
        for k in ('control',) + FORMS:
            rr = res[(k, model)]
            L.append(f'| {k} | ' + ' | '.join(' / '.join(c[2].format(o[c[0]]) for o in rr) for c in cols) + ' |')
        L.append('')
    shas = sorted({o['sha'] for (k, _), rr in res.items() if k != 'control' for o in rr})
    L.append(f'form-run sha: {", ".join(map(str, shas))}')
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f'posthoc_{a.task}.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
