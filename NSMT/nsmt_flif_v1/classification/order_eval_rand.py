"""MODI 2026-11-20 (user request 11-20, "순서 섞는 실험을 ETT예측에 대해서 해야하는 것 아니야? - 순열로 섞는게 아니라 "무작위로" 섞는
실험으로 해줘."): test-time token-order shuffle of the stored models, with a NEW RANDOM ORDER FOR EVERY ROW (no training).

Models: the SpikeDE-basis runs of the positional-encoding test -- bias on = flif-v1-{ett,mitbih}-sde, bias off = flif-v1-{ett,mitbih}-nope;
Spikformer(DE) spkall_lif and the base nsmt4q_pop_<lif|flif>_msn_sde; ETT ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026 (72 runs),
MIT-BIH seeds 42 / 7 / 2026 (18 runs). Weights = the tested ones (ETT model_state/checkpoint.pth = best validation; MIT-BIH
model_state/best+model.pt = last epoch).

Random order: every row of a batch (ETT: one channel of one window, B*C rows; MIT-BIH: one beat) gets its own permutation of the N tokens,
torch.argsort(torch.rand(rows, N)) from a CPU generator seeded 1000 + r for repeat r (r = 0 .. R-1, R = 5). The test loaders do not
shuffle, so in repeat r every variant and every model sees the same permutations.
  normal  true order; must reproduce the logged test metric (ETT MSE and MAE |diff| <= 2e-6, MIT-BIH macro-F1 <= 1e-6), else stop
  body    the same permutation for the hippocampus input [T_h, B', N, P] and the Neocortex input [N, B', 1, P]; the hippocampus output
          [T_h, B', N, D] is put back at the original token positions before the head -> how much the body (attention + Neocortex)
          uses the token order. The head reads the tokens at their true positions.
  all     as body, but the output is NOT put back -> ETT: the flat head also reads the tokens at the wrong positions (reference);
          MIT-BIH: the head averages the tokens, so all = body (check)
  neo     (base only) the model's own Neocortex-order control (neo_shuffle) with these random permutations: the Neocortex reads the
          tokens in the random order and its outputs are put back; the hippocampus keeps the true order
Checks (fail-closed, any failure stops): normal reproduces the log; the arm matches the run (pe=0 + _nope + ssa_pe 0 for off; none for on);
Spikformer(DE) without the bias is blind to the order in 'body' (ETT |MSE ratio - 1| <= 1e-4 in every repeat, MIT-BIH identical
predictions); bias-off base: 'body' = 'neo' (ETT |MSE ratio - 1| <= 1e-4, MIT-BIH >= 99.99 % identical predictions); MIT-BIH 'all' = 'body'
(>= 99.99 % identical predictions); the normal metric is restored after the variants; every value finite; complete and unique rows.
usage (classification folder):
  python order_eval_rand.py --task ett --shard 0/4      (one process per GPU; CUDA_VISIBLE_DEVICES picks the GPU)
  python order_eval_rand.py --task mb  --shard 0/1
  python order_eval_rand.py --summarize                 -> results/flif-v1-order-rand.{json,md}
Rows go to results/flif-v1-order-rand/<task>-shard<i>of<k>.jsonl (one line per run; an existing file is refused).
"""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, os.getcwd())
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'forecasting'))
from spikingjelly.clock_driven import functional

RES = os.path.join(HERE, 'results')
FC = os.path.join(HERE, '..', 'forecasting', 'results')
OUTD = os.path.join(RES, 'flif-v1-order-rand')
CONDS = ('spkall_lif', 'nsmt4q_pop_lif_msn_sde', 'nsmt4q_pop_flif_msn_sde')
NAMES = {'spkall_lif': 'Spikformer(DE)', 'nsmt4q_pop_lif_msn_sde': 'base pop LIF', 'nsmt4q_pop_flif_msn_sde': 'base pop f-LIF'}
SEEDS = (42, 7, 2026)
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
ARMS = {'ett': (('on', os.path.join(FC, 'flif-v1-ett-sde')), ('off', os.path.join(FC, 'flif-v1-ett-nope'))),
        'mb': (('on', os.path.join(RES, 'flif-v1-mitbih-sde')), ('off', os.path.join(RES, 'flif-v1-mitbih-nope')))}
R = 5
TOL_R = 1e-4          # relative MSE tolerance of the exactness checks (float summation order inside the attention)
TOL_AGREE = 0.9999    # share of identical MIT-BIH predictions required by the exactness checks


def runs(task):
    """[(arm, cond, ds, pl, seed, RESULT dict)] in a fixed order; exactly one row per key."""
    out = []
    for arm, suite in ARMS[task]:
        got = {}
        for ln in open(os.path.join(suite, 'raw.txt')):
            if not ln.startswith('RESULT'):
                continue
            d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
            if d['cond'] not in CONDS:
                continue
            k = (d['cond'], d.get('ds'), int(d['pl']) if 'pl' in d else None, int(d['seed']))
            if k in got:
                raise RuntimeError(f'{suite}: duplicate {k}')
            got[k] = d
        keys = [(c, ds, pl, s) for c in CONDS for (ds, pl) in (CELLS if task == 'ett' else ((None, None),)) for s in SEEDS]
        miss = [k for k in keys if k not in got]
        if miss or len(got) != len(keys):
            raise RuntimeError(f'{suite}: {len(got)} rows, missing {miss}')
        for k in keys:
            d = got[k]
            off = arm == 'off'
            if off != (d.get('pe') == '0') or off != ('_nope' in os.path.basename(d['dir'])):
                raise RuntimeError(f'{suite} {k}: bias arm does not match the run (pe={d.get("pe")}, {os.path.basename(d["dir"])})')
            out.append((arm,) + k + (d,))
    return out


def draw(gen, rows, n):
    return torch.argsort(torch.rand(rows, n, generator=gen), dim=1)


class RandPerms:
    """stands in for token_order.TokenShuffle as the model's neo_shuffle: per-row random permutations, also at evaluation."""
    def __init__(self, n, gen):
        self.n, self.gen = n, gen

    def perms(self, B, device, training):
        return draw(self.gen, B, self.n).to(device)


def install(body, mode, gen):
    """mode 'body' / 'all': per-row random token order for both paths ('body' puts the hippocampus output back); returns undo()."""
    from token_order import permute_tokens
    orig_tok, orig_hb = body._tokens_mc, body.hippo_blocks
    state = {}

    def tok(x):
        h, n = orig_tok(x)                                           # [T_h, B', N, P], [N, B', 1, P]
        perm = draw(gen, h.shape[1], h.shape[2]).to(h.device)        # [B', N]; new position t holds original token perm[b, t]
        state['inv'] = torch.argsort(perm, dim=1)
        return permute_tokens(h, 2, perm), (permute_tokens(n, 0, perm) if n is not None else None)

    def hb(xh, xn):
        out = orig_hb(xh, xn)                                        # [T_h, B', N, D] in the shuffled order
        return permute_tokens(out, 2, state.pop('inv'))              # back to the original token positions

    body._tokens_mc = tok
    if mode == 'body':
        body.hippo_blocks = hb

    def undo():
        for k in ('_tokens_mc', 'hippo_blocks'):
            body.__dict__.pop(k, None)
    return undo


def build(task, d):
    from load_model import LOAD_MODEL
    cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda'))
    torch.manual_seed(0)
    body = LOAD_MODEL[a.model](a, True)
    if task == 'ett':
        from forecast_wrap import Forecaster
        m = Forecaster(body, a.embed_dim, a.pred_len, bias=a.bias)
        w = torch.load(os.path.join(d, 'model_state', 'checkpoint.pth'), map_location='cpu')
        m.load_state_dict(w)                                         # strict: the forecaster state of the run
    else:
        m = body
        w = torch.load(os.path.join(d, 'model_state', 'best+model.pt'), map_location='cpu')
        r = m.load_state_dict(w, strict=False)
        if r.missing_keys or not all('weak_decoder' in k for k in r.unexpected_keys):
            raise RuntimeError(f'{d}: weights do not fit (missing {r.missing_keys[:3]}, unexpected {r.unexpected_keys[:3]})')
    m = m.cuda().eval()
    m.train_mode = 'testing'
    return m, body, a


_LOADERS = {}


def loader(task, a):
    key = (task, getattr(a, 'data', None), getattr(a, 'pred_len', None))
    if key not in _LOADERS:
        if task == 'ett':
            from train_forecast import data_args
            from data_provider.data_factory import data_provider
            _LOADERS[key] = data_provider(data_args(a, a), 'test')[1]
        else:
            from dataloader import create_loader
            _LOADERS[key] = create_loader(train=False, num_classes=a.num_classes, batch_size=a.batch_size,
                                          data_root=a.root_path + '/5class_mitbih_test_3_187.csv', data_window_size=187, num_workers=4)[0]
    return _LOADERS[key]


@torch.no_grad()
def evaluate(task, m, a):
    """ETT: (mse, mae, None); MIT-BIH: (macro-F1, None, predictions)."""
    dl = loader(task, a)
    functional.reset_net(m)
    if task == 'ett':
        se = ae = 0.
        cnt = 0
        for x, y, _, _ in dl:
            x = x.float().cuda(); y = y.float().cuda()[:, -a.pred_len:, :]
            o = m(x)
            functional.reset_net(m)
            dd = (o - y).double()
            se += float((dd ** 2).sum()); ae += float(dd.abs().sum()); cnt += dd.numel()
        return se / cnt, ae / cnt, None
    from torcheval.metrics import MulticlassF1Score
    from utils import get_pred
    f1 = MulticlassF1Score(num_classes=a.num_classes, average='macro').to('cuda')
    preds = []
    for batch in dl:
        x = batch[0].float().cuda(); y = batch[1].long().squeeze(-1).cuda()
        o = m(x)
        o = o[0] if isinstance(o, tuple) else o
        functional.reset_net(m)
        p = get_pred(o)
        f1.update(p, y); preds.append(p.cpu())
    return f1.compute().item(), None, torch.cat(preds)


def one_run(task, arm, cond, ds, pl, seed, d):
    t0 = time.time()
    m, body, a = build(task, d['dir'])
    if int(a.ssa_pe) != (0 if arm == 'off' else 1):
        raise RuntimeError(f'{d["dir"]}: ssa_pe {a.ssa_pe} in the {arm} arm')
    key = f'{task} {arm} {cond} {ds} {pl} {seed}'
    n_tok = int(body.num_patches)
    v0, mae0, p0 = evaluate(task, m, a)
    if task == 'ett':
        if not (abs(v0 - float(d['mse'])) <= 2e-6 and abs(mae0 - float(d['mae'])) <= 2e-6):
            raise RuntimeError(f'{key}: normal mse {v0:.6f} / mae {mae0:.6f} != logged {d["mse"]} / {d["mae"]}')
    elif not abs(v0 - float(d['f1'])) <= 1e-6:
        raise RuntimeError(f'{key}: normal f1 {v0:.6f} != logged {d["f1"]}')
    row = dict(task=task, arm=arm, cond=cond, ds=ds, pl=pl, seed=seed, n_tokens=n_tok, normal=v0, normal_mae=mae0, sha=d.get('sha'), rep={})
    variants = ('body', 'all') + (('neo',) if cond != 'spkall_lif' else ())
    for r in range(R):
        rep = {}
        for v in variants:
            gen = torch.Generator().manual_seed(1000 + r)
            if v == 'neo':
                if getattr(body, 'neo_shuffle', 'missing') is not None:
                    raise RuntimeError(f'{key}: the base must be built without neo_shuffle')
                body.neo_shuffle = RandPerms(n_tok, gen)
                val, mae, p = evaluate(task, m, a)
                body.neo_shuffle = None
            else:
                undo = install(body, v, gen)
                val, mae, p = evaluate(task, m, a)
                undo()
            rep[v] = dict(val=val, mae=mae)
            if p is not None:
                rep[v]['changed'] = float((p != p0).float().mean())
                rep[v]['_p'] = p
        # exactness checks
        if task == 'ett':
            if arm == 'off' and cond == 'spkall_lif' and not abs(rep['body']['val'] / v0 - 1) <= TOL_R:
                raise RuntimeError(f'{key} r{r}: Spikformer(DE) without the bias changed MSE by {rep["body"]["val"] / v0 - 1:.2e} under body')
            if arm == 'off' and cond != 'spkall_lif' and not abs(rep['body']['val'] / rep['neo']['val'] - 1) <= TOL_R:
                raise RuntimeError(f'{key} r{r}: bias-off base body {rep["body"]["val"]:.6f} != neo {rep["neo"]["val"]:.6f}')
        else:
            if arm == 'off' and cond == 'spkall_lif' and not torch.equal(rep['body']['_p'], p0):
                raise RuntimeError(f'{key} r{r}: Spikformer(DE) without the bias changed {rep["body"]["changed"]:.4%} predictions under body')
            if arm == 'off' and cond != 'spkall_lif':
                ag = float((rep['body']['_p'] == rep['neo']['_p']).float().mean())
                if not ag >= TOL_AGREE:
                    raise RuntimeError(f'{key} r{r}: bias-off base body / neo agree on {ag:.4%} only')
            ag = float((rep['body']['_p'] == rep['all']['_p']).float().mean())
            if not ag >= TOL_AGREE:
                raise RuntimeError(f'{key} r{r}: MIT-BIH all / body agree on {ag:.4%} only')
            for v in rep:
                rep[v].pop('_p')
        for v in rep:
            for k, x in rep[v].items():
                if x is not None and not math.isfinite(x):
                    raise RuntimeError(f'{key} r{r} {v}: non-finite {k}')
        row['rep'][r] = rep
    vb, maeb, pb = evaluate(task, m, a)
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
    if os.path.exists(path):
        raise RuntimeError(f'{path} exists (refusing to overwrite)')
    with open(path + '.part', 'w') as f:
        for arm, cond, ds, pl, seed, d in todo:
            row = one_run(task, arm, cond, ds, pl, seed, d)
            f.write(json.dumps(row) + '\n'); f.flush()
            print(f"done {task} {arm} {cond} {ds} {pl} {seed} normal {row['normal']:.6f} body r0 {row['rep'][0]['body']['val']:.6f} "
                  f"({row['time_s']} s)", flush=True)
    os.rename(path + '.part', path)
    print(f'shard {i}/{k} of {task}: {len(todo)} runs -> {path}; all checks passed', flush=True)


def mean(xs):
    xs = list(xs)
    if not xs or any(not math.isfinite(x) for x in xs):
        raise RuntimeError('mean of nothing / non-finite')
    return sum(xs) / len(xs)


def sd(xs):
    xs = list(xs)
    mu = mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.


def summarize():
    rows = []
    for fn in sorted(os.listdir(OUTD)):
        if fn.endswith('.part'):
            raise RuntimeError(f'unfinished shard {fn}')
        if fn.endswith('.jsonl'):
            rows += [json.loads(ln) for ln in open(os.path.join(OUTD, fn))]
    for task in ('ett', 'mb'):
        want = {(arm, c, ds, pl, s) for arm, c, ds, pl, s, _ in runs(task)}
        got = [(r['arm'], r['cond'], r['ds'], r['pl'], r['seed']) for r in rows if r['task'] == task]
        if len(got) != len(set(got)) or set(got) != want:
            raise RuntimeError(f'{task}: {len(got)} rows ({len(set(got))} unique), {len(want)} expected')
        for r in rows:
            if r['task'] == task and sorted(int(x) for x in r['rep']) != list(range(R)):
                raise RuntimeError(f'{task} {r["cond"]} {r["seed"]}: repeats {sorted(r["rep"])}')
    shas = sorted({r['sha'] for r in rows})

    def rel(r, v):        # ETT: mean over repeats of (shuffled MSE / normal MSE - 1), %
        return mean(100 * (r['rep'][str(k)][v]['val'] / r['normal'] - 1) for k in range(R))

    def rep_sd(r, v):
        return sd(100 * (r['rep'][str(k)][v]['val'] / r['normal'] - 1) for k in range(R))

    def drop(r, v):       # MIT-BIH: mean over repeats of (shuffled - normal) macro-F1, %p
        return mean(100 * (r['rep'][str(k)][v]['val'] - r['normal']) for k in range(R))

    def changed(r, v):
        return mean(100 * r['rep'][str(k)][v]['changed'] for k in range(R))

    L = ['# Test-time token-order shuffle with a random order per row (stored models, no training)', '',
         'See the docstring of order_eval_rand.py. User request 11-20. Bias on = flif-v1-*-sde, bias off = flif-v1-*-nope (the default since '
         f'11-20). R = {R} repeats (generator seed 1000 + r); each row of a batch has its own random token order. Run shas: {", ".join(shas)}.', '',
         '- body: tokens shuffled for both paths, the hippocampus output put back before the head (does the body use the order?)',
         '- neo: only the Neocortex reads the tokens in the random order (base only)',
         '- all: as body, not put back (ETT: the flat head also reads the tokens at the wrong positions)', '',
         '## ETT (r = shuffled test MSE / normal test MSE - 1, %; mean of the R repeats per run, then over the 12 runs = 4 cells x 3 seeds)', '',
         '| model | bias | normal MSE (mean of 12) | body r % (min .. max of 12) | runs r > 0 | neo r % (min .. max) | all r % | largest repeat sd of body r |',
         '|---|---|---|---|---|---|---|---|']
    ett = [r for r in rows if r['task'] == 'ett']
    out = {'ett': {}, 'mb': {}, 'shas': shas, 'R': R}
    for c in CONDS:
        for arm in ('off', 'on'):
            g = [r for r in ett if r['cond'] == c and r['arm'] == arm]
            if len(g) != 12:
                raise RuntimeError(f'ett {c} {arm}: {len(g)} runs')
            b = [rel(r, 'body') for r in g]
            cell = [NAMES[c], arm, f"{mean(r['normal'] for r in g):.4f}", f"{mean(b):+.2f} ({min(b):+.2f} .. {max(b):+.2f})",
                    f"{sum(x > 0 for x in b)}/12"]
            if c == 'spkall_lif':
                cell.append('—')
                nn_ = None
            else:
                nn_ = [rel(r, 'neo') for r in g]
                cell.append(f"{mean(nn_):+.2f} ({min(nn_):+.2f} .. {max(nn_):+.2f})")
            al = [rel(r, 'all') for r in g]
            cell += [f"{mean(al):+.1f}", f"{max(rep_sd(r, 'body') for r in g):.2f}"]
            L.append('| ' + ' | '.join(cell) + ' |')
            out['ett'][f'{c} {arm}'] = dict(body=b, neo=nn_, all=al, normal=[r['normal'] for r in g])
    L += ['', '### ETT body r % per cell (mean of 3 seeds; seeds in brackets)', '',
          '| model | bias | ' + ' | '.join(f'{ds} H{pl}' for ds, pl in CELLS) + ' |', '|---|---|' + '---|' * len(CELLS)]
    for c in CONDS:
        for arm in ('off', 'on'):
            cell = [NAMES[c], arm]
            for ds, pl in CELLS:
                g = sorted([r for r in ett if r['cond'] == c and r['arm'] == arm and r['ds'] == ds and r['pl'] == pl], key=lambda r: SEEDS.index(r['seed']))
                v = [rel(r, 'body') for r in g]
                cell.append(f"{mean(v):+.2f} ({' / '.join(f'{x:+.2f}' for x in v)})")
            L.append('| ' + ' | '.join(cell) + ' |')
    L += ['', '## MIT-BIH (macro-F1 %; drop = shuffled - normal, %p, mean of the R repeats per seed; changed = share of test beats whose '
          'prediction changes, %)', '',
          '| model | bias | normal | body drop %p (seed 42 / 7 / 2026) | mean | changed % | neo drop %p (seeds) | mean | changed % |',
          '|---|---|---|---|---|---|---|---|---|']
    mb = [r for r in rows if r['task'] == 'mb']
    for c in CONDS:
        for arm in ('off', 'on'):
            g = sorted([r for r in mb if r['cond'] == c and r['arm'] == arm], key=lambda r: SEEDS.index(r['seed']))
            if len(g) != 3:
                raise RuntimeError(f'mb {c} {arm}: {len(g)} runs')
            b = [drop(r, 'body') for r in g]
            cell = [NAMES[c], arm, f"{100 * mean(r['normal'] for r in g):.2f}", ' / '.join(f'{x:+.2f}' for x in b), f'{mean(b):+.2f}',
                    f"{mean(changed(r, 'body') for r in g):.2f}"]
            if c == 'spkall_lif':
                cell += ['—', '—', '—']
                nn_ = None
            else:
                nn_ = [drop(r, 'neo') for r in g]
                cell += [' / '.join(f'{x:+.2f}' for x in nn_), f'{mean(nn_):+.2f}', f"{mean(changed(r, 'neo') for r in g):.2f}"]
            L.append('| ' + ' | '.join(cell) + ' |')
            out['mb'][f'{c} {arm}'] = dict(body=b, neo=nn_, normal=[r['normal'] for r in g])
    L += ['', f"run time per model: ETT {min(r['time_s'] for r in ett):.0f}-{max(r['time_s'] for r in ett):.0f} s, "
          f"MIT-BIH {min(r['time_s'] for r in mb):.0f}-{max(r['time_s'] for r in mb):.0f} s"]
    json.dump(out, open(os.path.join(RES, 'flif-v1-order-rand.json'), 'w'), indent=1)
    open(os.path.join(RES, 'flif-v1-order-rand.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('all checks passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=('ett', 'mb'))
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='one run (ETT: base pop LIF off ETTh2 H96 seed 42), checks and time only')
    ag = ap.parse_args()
    if ag.summarize:
        summarize()
    elif ag.smoke:
        R = 1
        for t in ('ett', 'mb'):
            sel = [x for x in runs(t) if x[0] == 'off' and x[1] == 'nsmt4q_pop_lif_msn_sde' and x[4] == 42 and x[2] in (None, 'ETTh2') and x[3] in (None, 96)]
            sel += [x for x in runs(t) if x[0] == 'off' and x[1] == 'spkall_lif' and x[4] == 42 and x[2] in (None, 'ETTh2') and x[3] in (None, 96)]
            for x in sel:
                row = one_run(t, *x)
                print(f'smoke {t} {x[1]}: checks passed, {row["time_s"]} s', flush=True)
    else:
        if ag.task is None:
            raise SystemExit('--task ett|mb, or --summarize')
        shard_main(ag.task, ag.shard)
