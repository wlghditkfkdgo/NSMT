"""MODI 2026-11-21 (4-2 coupling placement, report only; pre-registration d53119ef4 "학습이 끝나면 학습 없이 (나) model에 무작위
token 순서 평가(order_eval_rand.py 방식)를 돌려 보고한다"): test-time random token-order shuffle of the stored 4-2 models (no training).

Models (all at the range-1 training, relative position bias off, no aux):
  (나)  nsmt4q_pop_<lif|flif>_msn_sde_ncna  (suites flif-v1-ett-nc / flif-v1-mitbih-nc)  -- the pre-registered target
  (가)  nsmt4q_pop_<lif|flif>_msn_sde_nc    (same suites)                                -- added for reference (not pre-registered)
  base nsmt4q_pop_<lif|flif>_msn_sde       (suites flif-v1-ett-r1 / flif-v1-mitbih-r1)  -- added for reference (not pre-registered):
       the same training as (가) / (나); the earlier order_eval_rand.py models were trained before range 1
ETT ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026, MIT-BIH seeds 42 / 7 / 2026 -> 72 + 18 = 90 runs. Weights = the tested ones.

Random order, variants and repeats exactly as order_eval_rand.py (its functions are imported, not copied): every row of a batch gets
its own permutation of the N tokens, generator seed 1000 + r, R = 5 repeats.
  normal  true order; must reproduce the logged test metric (ETT MSE and MAE |diff| <= 2e-6, MIT-BIH macro-F1 <= 1e-6)
  body    the same permutation for the hippocampus input and the Neocortex input; the hippocampus output put back before the head
  all     as body, not put back (ETT: the flat head reads the tokens at the wrong positions; MIT-BIH: = body)
  neo     only the Neocortex reads the tokens in the random order (neo_shuffle); its outputs are put back to the true positions
In (가) / (나) the Neocortex output enters the hippocampus embedding current (forward and Forecaster.step_features both call
_tokens_mc / hippo_blocks through the instance, so order_eval_rand.install applies unchanged).
Checks (fail-closed, any failure stops): normal reproduces the log; the stored config matches the condition (neo_current / hippo_attn,
ssa_pe 0, wd_exclude_1d 1, ETT val_per_epoch 4); (나): body = neo (the hippocampus mixes no tokens, so the only order the model sees is
the Neocortex's; ETT |MSE ratio - 1| <= 1e-4 in every repeat, MIT-BIH >= 99.99 % identical predictions); MIT-BIH all = body (>= 99.99 %);
the normal metric is restored after the variants; every value finite; complete and unique rows.
usage (classification folder):
  python order_eval_nc.py --task ett --shard 0/4      (one process per GPU; CUDA_VISIBLE_DEVICES picks the GPU)
  python order_eval_nc.py --task mb  --shard 0/1
  python order_eval_nc.py --summarize                 -> results/flif-v1-nc-order.{json,md}
  python order_eval_nc.py --smoke                     (one (나) and one base run per task, R = 1, checks only; writes nothing)
Rows go to results/flif-v1-nc-order/<task>-shard<i>of<k>.jsonl (one line per run; an existing file is refused).
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'forecasting'))
import order_eval_rand as OR                                            # noqa: E402  (draw / RandPerms / install / build / evaluate)

RES = os.path.join(HERE, 'results')
FC = os.path.join(HERE, '..', 'forecasting', 'results')
OUTD = os.path.join(RES, 'flif-v1-nc-order')
NEU = ('lif', 'flif')
KINDS = ('base', 'nc', 'ncna')
KNAME = {'base': 'base', 'nc': '(가) Neocortex current + attention', 'ncna': '(나) Neocortex current, no attention'}
NNAME = {'lif': 'pop LIF', 'flif': 'pop f-LIF'}
FLAGS = {'base': (0, 1), 'nc': (1, 1), 'ncna': (1, 0)}                  # (neo_current, hippo_attn)
SEEDS = (42, 7, 2026)
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
SUITES = {'ett': {'base': os.path.join(FC, 'flif-v1-ett-r1'), 'new': os.path.join(FC, 'flif-v1-ett-nc')},
          'mb': {'base': os.path.join(RES, 'flif-v1-mitbih-r1'), 'new': os.path.join(RES, 'flif-v1-mitbih-nc')}}
R = OR.R
TOL_R, TOL_AGREE = OR.TOL_R, OR.TOL_AGREE


def cond_of(kind, n):
    return f'nsmt4q_pop_{n}_msn_sde' + ('' if kind == 'base' else '_' + kind)


def runs(task):
    """[(kind, neuron, ds, pl, seed, RESULT dict)] in a fixed order; exactly one row per key; one sha per suite."""
    out = []
    for kind in KINDS:
        suite = SUITES[task]['base' if kind == 'base' else 'new']
        got = {}
        for ln in open(os.path.join(suite, 'raw.txt')):
            if not ln.startswith('RESULT'):
                continue
            d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
            hit = [n for n in NEU if d['cond'] == cond_of(kind, n)]
            if not hit:
                continue
            k = (hit[0], d.get('ds'), int(d['pl']) if 'pl' in d else None, int(d['seed']))
            if k in got:
                raise RuntimeError(f'{suite}: duplicate {kind} {k}')
            got[k] = d
        keys = [(n, ds, pl, s) for n in NEU for (ds, pl) in (CELLS if task == 'ett' else ((None, None),)) for s in SEEDS]
        miss = [k for k in keys if k not in got]
        if miss or len(got) != len(keys):
            raise RuntimeError(f'{suite} {kind}: {len(got)} rows, missing {miss}')
        for k in keys:
            d = got[k]
            if d.get('pe') != '0' or d.get('wdx') != '1' or (task == 'ett' and d.get('vpe') != '4'):
                raise RuntimeError(f'{suite} {kind} {k}: pe={d.get("pe")} wdx={d.get("wdx")} vpe={d.get("vpe")} (want 0 / 1 / 4)')
            out.append((kind,) + k + (d,))
    return out


def one_run(task, kind, n, ds, pl, seed, d):
    t0 = time.time()
    m, body, a = OR.build(task, d['dir'])
    key = f'{task} {kind} {n} {ds} {pl} {seed}'
    want = FLAGS[kind]
    got = (int(getattr(a, 'neo_current', 0)), int(getattr(a, 'hippo_attn', 1)))
    if got != want or int(body.neo_current) != want[0] or int(body.hippo_attn) != want[1]:
        raise RuntimeError(f'{key}: neo_current / hippo_attn config {got}, model {(body.neo_current, body.hippo_attn)}, want {want}')
    if int(a.ssa_pe) != 0 or int(getattr(a, 'wd_exclude_1d', 0)) != 1 or (task == 'ett' and int(getattr(a, 'val_per_epoch', 1)) != 4):
        raise RuntimeError(f'{key}: ssa_pe {a.ssa_pe}, wd_exclude_1d {getattr(a, "wd_exclude_1d", None)}, '
                           f'val_per_epoch {getattr(a, "val_per_epoch", None)}')
    n_tok = int(body.num_patches)
    v0, mae0, p0 = OR.evaluate(task, m, a)
    if task == 'ett':
        if not (abs(v0 - float(d['mse'])) <= 2e-6 and abs(mae0 - float(d['mae'])) <= 2e-6):
            raise RuntimeError(f'{key}: normal mse {v0:.6f} / mae {mae0:.6f} != logged {d["mse"]} / {d["mae"]}')
    elif not abs(v0 - float(d['f1'])) <= 1e-6:
        raise RuntimeError(f'{key}: normal f1 {v0:.6f} != logged {d["f1"]}')
    row = dict(task=task, kind=kind, neuron=n, ds=ds, pl=pl, seed=seed, n_tokens=n_tok, normal=v0, normal_mae=mae0, sha=d.get('sha'), rep={})
    for r in range(R):
        rep = {}
        for v in ('body', 'all', 'neo'):
            gen = torch.Generator().manual_seed(1000 + r)
            if v == 'neo':
                if getattr(body, 'neo_shuffle', 'missing') is not None:
                    raise RuntimeError(f'{key}: the model must be built without neo_shuffle')
                body.neo_shuffle = OR.RandPerms(n_tok, gen)
                val, mae, p = OR.evaluate(task, m, a)
                body.neo_shuffle = None
            else:
                undo = OR.install(body, v, gen)
                val, mae, p = OR.evaluate(task, m, a)
                undo()
            rep[v] = dict(val=val, mae=mae)
            if p is not None:
                rep[v]['changed'] = float((p != p0).float().mean())
                rep[v]['_p'] = p
        if task == 'ett':
            if kind == 'ncna' and not abs(rep['body']['val'] / rep['neo']['val'] - 1) <= TOL_R:
                raise RuntimeError(f'{key} r{r}: (나) body {rep["body"]["val"]:.6f} != neo {rep["neo"]["val"]:.6f}')
        else:
            if kind == 'ncna':
                ag = float((rep['body']['_p'] == rep['neo']['_p']).float().mean())
                if not ag >= TOL_AGREE:
                    raise RuntimeError(f'{key} r{r}: (나) body / neo agree on {ag:.4%} only')
            ag = float((rep['body']['_p'] == rep['all']['_p']).float().mean())
            if not ag >= TOL_AGREE:
                raise RuntimeError(f'{key} r{r}: MIT-BIH all / body agree on {ag:.4%} only')
            for v in rep:
                rep[v].pop('_p')
        for v in rep:
            for k_, x in rep[v].items():
                if x is not None and not math.isfinite(x):
                    raise RuntimeError(f'{key} r{r} {v}: non-finite {k_}')
        row['rep'][r] = rep
    vb, _, pb = OR.evaluate(task, m, a)
    if vb != v0 or (pb is not None and not torch.equal(pb, p0)):
        raise RuntimeError(f'{key}: normal metric not restored ({vb} vs {v0})')
    row['time_s'] = round(time.time() - t0, 1)
    del m, body
    torch.cuda.empty_cache()
    return row


def shard_main(task, shard):
    i, k = (int(x) for x in shard.split('/'))
    if not 0 <= i < k:
        raise ValueError('--shard i/k with 0 <= i < k')
    todo = runs(task)[i::k]
    os.makedirs(OUTD, exist_ok=True)
    path = os.path.join(OUTD, f'{task}-shard{i}of{k}.jsonl')
    if os.path.exists(path) or os.path.exists(path + '.part'):
        raise RuntimeError(f'{path} (or .part) exists (refusing to overwrite)')
    with open(path + '.part', 'w') as f:
        for kind, n, ds, pl, seed, d in todo:
            row = one_run(task, kind, n, ds, pl, seed, d)
            f.write(json.dumps(row) + '\n'); f.flush()
            print(f"done {task} {kind} {n} {ds} {pl} {seed} normal {row['normal']:.6f} body r0 {row['rep'][0]['body']['val']:.6f} "
                  f"({row['time_s']} s)", flush=True)
    os.rename(path + '.part', path)
    print(f'shard {i}/{k} of {task}: {len(todo)} runs -> {path}; all checks passed', flush=True)


def summarize():
    rows = []
    for fn in sorted(os.listdir(OUTD)):
        if fn.endswith('.part'):
            raise RuntimeError(f'unfinished shard {fn}')
        if fn.endswith('.jsonl'):
            rows += [json.loads(ln) for ln in open(os.path.join(OUTD, fn))]
    for task in ('ett', 'mb'):
        want = {(kd, n, ds, pl, s) for kd, n, ds, pl, s, _ in runs(task)}
        got = [(r['kind'], r['neuron'], r['ds'], r['pl'], r['seed']) for r in rows if r['task'] == task]
        if len(got) != len(set(got)) or set(got) != want:
            raise RuntimeError(f'{task}: {len(got)} rows ({len(set(got))} unique), {len(want)} expected')
        for r in rows:
            if r['task'] == task and sorted(int(x) for x in r['rep']) != list(range(R)):
                raise RuntimeError(f'{task} {r["kind"]} {r["neuron"]} {r["seed"]}: repeats {sorted(r["rep"])}')
    shas = sorted({r['sha'] for r in rows})

    def rel(r, v):        # ETT: mean over repeats of (shuffled MSE / normal MSE - 1), %
        return OR.mean(100 * (r['rep'][str(k)][v]['val'] / r['normal'] - 1) for k in range(R))

    def drop(r, v):       # MIT-BIH: mean over repeats of (shuffled - normal) macro-F1, %p
        return OR.mean(100 * (r['rep'][str(k)][v]['val'] - r['normal']) for k in range(R))

    def changed(r, v):
        return OR.mean(100 * r['rep'][str(k)][v]['changed'] for k in range(R))

    L = ['# 4-2 models: test-time token-order shuffle with a random order per row (stored models, no training; report only)', '',
         'See the docstring of order_eval_nc.py. (나) is the pre-registered target (d53119ef4); the base (range 1) and (가) rows are added for '
         f'reference. R = {R} repeats (generator seed 1000 + r). Run shas: {", ".join(shas)}.', '',
         '- body: tokens shuffled for both paths, the hippocampus output put back before the head (does the body use the order?)',
         '- neo: only the Neocortex reads the tokens in the random order, its outputs put back',
         '- all: as body, not put back (ETT: the flat head also reads the tokens at the wrong positions)', '',
         '## ETT (r = shuffled test MSE / normal test MSE - 1, %; mean of the R repeats per run, then over the 12 runs = 4 cells x 3 seeds)', '',
         '| neuron | model | normal MSE (mean of 12) | body r % (min .. max of 12) | runs body r > 0 | neo r % (min .. max) | all r % | largest repeat sd of body r |',
         '|---|---|---|---|---|---|---|---|']
    ett = [r for r in rows if r['task'] == 'ett']
    mb = [r for r in rows if r['task'] == 'mb']
    out = {'ett': {}, 'mb': {}, 'shas': shas, 'R': R}
    for n in NEU:
        for kind in KINDS:
            g = [r for r in ett if r['neuron'] == n and r['kind'] == kind]
            if len(g) != 12:
                raise RuntimeError(f'ett {n} {kind}: {len(g)} runs')
            b = [rel(r, 'body') for r in g]
            nn_ = [rel(r, 'neo') for r in g]
            al = [rel(r, 'all') for r in g]
            sdb = max(OR.sd(100 * (r['rep'][str(k)]['body']['val'] / r['normal'] - 1) for k in range(R)) for r in g)
            L.append(f"| {NNAME[n]} | {KNAME[kind]} | {OR.mean(r['normal'] for r in g):.4f} | {OR.mean(b):+.2f} ({min(b):+.2f} .. {max(b):+.2f}) | "
                     f"{sum(x > 0 for x in b)}/12 | {OR.mean(nn_):+.2f} ({min(nn_):+.2f} .. {max(nn_):+.2f}) | {OR.mean(al):+.1f} | {sdb:.2f} |")
            out['ett'][f'{n} {kind}'] = dict(body=b, neo=nn_, all=al, normal=[r['normal'] for r in g])
    L += ['', '### ETT body r % per cell (mean of 3 seeds; seeds 42 / 7 / 2026 in brackets)', '',
          '| neuron | model | ' + ' | '.join(f'{ds} H{pl}' for ds, pl in CELLS) + ' |', '|---|---|' + '---|' * len(CELLS)]
    for n in NEU:
        for kind in KINDS:
            cell = [NNAME[n], KNAME[kind]]
            for ds, pl in CELLS:
                g = sorted([r for r in ett if r['neuron'] == n and r['kind'] == kind and r['ds'] == ds and r['pl'] == pl],
                           key=lambda r: SEEDS.index(r['seed']))
                v = [rel(r, 'body') for r in g]
                cell.append(f"{OR.mean(v):+.2f} ({' / '.join(f'{x:+.2f}' for x in v)})")
            L.append('| ' + ' | '.join(cell) + ' |')
    L += ['', '## MIT-BIH (macro-F1 %; drop = shuffled - normal, %p, mean of the R repeats per seed; changed = share of test beats whose '
          'prediction changes, %)', '',
          '| neuron | model | normal | body drop %p (seed 42 / 7 / 2026) | mean | changed % | neo drop %p (seeds) | mean | changed % |',
          '|---|---|---|---|---|---|---|---|---|']
    for n in NEU:
        for kind in KINDS:
            g = sorted([r for r in mb if r['neuron'] == n and r['kind'] == kind], key=lambda r: SEEDS.index(r['seed']))
            if len(g) != 3:
                raise RuntimeError(f'mb {n} {kind}: {len(g)} runs')
            b = [drop(r, 'body') for r in g]
            nn_ = [drop(r, 'neo') for r in g]
            L.append(f"| {NNAME[n]} | {KNAME[kind]} | {100 * OR.mean(r['normal'] for r in g):.2f} | {' / '.join(f'{x:+.2f}' for x in b)} | "
                     f"{OR.mean(b):+.2f} | {OR.mean(changed(r, 'body') for r in g):.2f} | {' / '.join(f'{x:+.2f}' for x in nn_)} | "
                     f"{OR.mean(nn_):+.2f} | {OR.mean(changed(r, 'neo') for r in g):.2f} |")
            out['mb'][f'{n} {kind}'] = dict(body=b, neo=nn_, normal=[r['normal'] for r in g])
    L += ['', f"run time per model: ETT {min(r['time_s'] for r in ett):.0f}-{max(r['time_s'] for r in ett):.0f} s, "
          f"MIT-BIH {min(r['time_s'] for r in mb):.0f}-{max(r['time_s'] for r in mb):.0f} s"]
    json.dump(out, open(os.path.join(RES, 'flif-v1-nc-order.json'), 'w'), indent=1)
    open(os.path.join(RES, 'flif-v1-nc-order.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('all checks passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=('ett', 'mb'))
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='one (나) and one base run per task (ETTh2 H96 / MIT-BIH, pop LIF, seed 42), R = 1')
    ag = ap.parse_args()
    if ag.summarize:
        summarize()
    elif ag.smoke:
        R = 1
        for t in ('ett', 'mb'):
            for x in runs(t):
                if x[0] in ('ncna', 'base') and x[1] == 'lif' and x[4] == 42 and x[2] in (None, 'ETTh2') and x[3] in (None, 96):
                    row = one_run(t, *x)
                    b = row['rep'][0]
                    print(f"smoke {t} {x[0]}: normal {row['normal']:.6f} body {b['body']['val']:.6f} neo {b['neo']['val']:.6f} "
                          f"all {b['all']['val']:.6f}; checks passed, {row['time_s']} s", flush=True)
    else:
        if ag.task is None:
            raise SystemExit('--task ett|mb, or --summarize')
        shard_main(ag.task, ag.shard)
