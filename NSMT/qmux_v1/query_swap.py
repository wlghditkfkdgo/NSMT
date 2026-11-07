"""Test-time replacement of the Neocortex query in a trained plain query model (2026-11-07, exp/f-lif-pop-v3; PROJECT_LOG query-swap pre-registration).
No training and no selection: one checkpoint, the weights of its reported test metric, the test split, the model code of this folder (read-only access to the checkpoint).
The query form takes its attention queries from the Neocortex output o [N, B', 1, D] and its keys / values from the Hippocampus tokens. Conditions (only the query input of the blocks changes):
  clean     o itself; must reproduce the logged metric
  other     o of another window, same channel. ETT: a random window at least seq_len + pred_len away (no shared input or target step); MIT-BIH: a random other beat. 3 draws
  perm      o with its tokens in a random order, independent per row (spike counts kept, order destroyed). 3 draws
  zero      o replaced by zeros. This is NOT the attention path off: q_bn has running statistics and q_lif may still fire, and the relative position bias stays
  attn_off  the attention term of the blocks is removed, x = x * (1 - 0); the spike-deleting MLP term stays
Fail closed (exit 1, no JSON): not a plain query model; the split forward differs from the model forward; the Neocortex forward is not repeatable after a reset (the path `other` uses);
a non-binary Neocortex output; `other` closer than allowed; `perm` changes a spike count or changes nothing; a clean metric that does not reproduce the log; a non-finite metric.
Report-only numbers in the JSON: firing rates of the query neuron, the attention term and the Hippocampus output per condition, the Neocortex output rate per token step, the bit agreement of o
with the `other` output against chance from the (token, channel) rates.
usage: python -B query_swap.py --task ett|mitbih --raw <raw.txt> --run_dir <run> --name <name> --out_dir <dir> [--conds clean] [--max_batches K]
       --max_batches K (K > 0): checks only, no metric check and no metric in the JSON (smoke, so the pre-registration is written before any replaced metric is seen)
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
from spikingjelly.clock_driven import functional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_model import LOAD_MODEL                                     # noqa: E402

N_CLASSES = 5
DRAWS = 3
SEED = 20261107
CONDS = ('clean', 'zero', 'attn_off', 'other', 'perm')


def stop(msg):
    print('FAIL:', msg, flush=True)
    sys.exit(1)


def ref_result(raw, run_dir):
    hit = [ln for ln in open(raw) if ln.startswith('RESULT') and ln.rstrip().endswith('dir=' + run_dir.rstrip('/'))]
    if len(hit) != 1:
        stop(f'{len(hit)} RESULT lines for {run_dir} in {raw}')
    return dict(re.findall(r'(\w+)=(\S+)', hit[0]))


def load(run, task, device):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    args = argparse.Namespace(**cfg)
    args.device = device
    if task == 'mitbih':
        model = LOAD_MODEL[args.model](args, True)
        model.load_state_dict(torch.load(os.path.join(run, 'model_state', 'best+model.pt'), map_location='cpu'))
        inner = model
    else:
        from forecast_wrap import Forecaster
        model = Forecaster(LOAD_MODEL[args.model](args, True), args.embed_dim, args.pred_len, bias=args.bias)
        model.load_state_dict(torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu'))
        inner = model.inner
        if model.kind != 'dual' or model.head.bias is not None:
            stop('expected a bias-free dual-path forecaster')
    model = model.to(device).eval()
    model.train_mode = 'testing'
    functional.reset_net(model)
    m = inner
    bad = [k for k, ok in (('hippo_xattn', m.hippo_xattn), ('query', m.hippo_xattn_mode == 'query'), ('hippo_ssa', not m.hippo_ssa), ('hippo_only', not m.hippo_only),
                           ('no_neo', not m.no_neo), ('roll_mode', m.roll_mode == 'none'), ('neo_memoryless', not m.neo_memoryless), ('neo_shuffle', m.neo_shuffle is None),
                           ('token_shuffle', m.token_shuffle is None), ('neo_bidir', not m.neo_bidir), ('neo_mux', getattr(args, 'neo_mux', 'none') == 'none')) if not ok]
    if bad:
        stop(f'expected the plain query form; offending: {bad}')
    return args, model, inner


def test_tensors(args, task):
    """The whole test split in loader order, on the CPU."""
    xs, ys = [], []
    if task == 'mitbih':
        from dataloader import create_loader
        dl, _ = create_loader(train=False, batch_size=args.batch_size, num_classes=N_CLASSES,
                              data_root=os.path.join(args.root_path, '5class_mitbih_test_3_187.csv'), data_window_size=187, num_workers=0)
        for x, y in dl:
            xs.append(x.float()); ys.append(y.long().reshape(-1))
    else:
        from data_provider.data_factory import data_provider
        from train_forecast import data_args
        _, dl = data_provider(data_args(args, args), 'test')
        for x, y, _, _ in dl:
            xs.append(x.float()); ys.append(y.float()[:, -args.pred_len:, :])
    return torch.cat(xs), torch.cat(ys)


def prep_ett(x):
    """As forecast_wrap.Forecaster.forward: window normalisation, channel-independent rows."""
    B, L, C = x.shape
    means = x.mean(1, keepdim=True)
    xc = x - means
    stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return (xc / stdev).permute(0, 2, 1).reshape(B * C, 1, L), (B, C, means, stdev)


class Tap:
    """Firing rates inside the blocks; `off` removes the attention term (the forward hook on blk.ssa returns zeros)."""
    def __init__(self, blocks):
        self.off = False
        self.reset()
        for b in blocks:
            b.ssa.q_lif.register_forward_hook(self._q)
            b.ssa.register_forward_hook(self._a)

    def reset(self):
        self.q = self.a = 0.
        self.nq = self.na = 0

    def _q(self, mod, inp, out):
        self.q += float(out.double().sum()); self.nq += out.numel()

    def _a(self, mod, inp, out):
        self.na += out.numel()
        if self.off:
            return torch.zeros_like(out)
        self.a += float(out.double().sum())
        return None


class Score:
    def __init__(self, task):
        self.task = task
        self.se = self.ae = 0.
        self.cnt = 0
        self.preds, self.ys = [], []
        self.h = 0.
        self.nh = 0
        self.q = self.a = 0.
        self.nq = self.na = 0

    def add(self, out, y, h, tap):
        if self.task == 'mitbih':
            self.preds.append(out.argmax(1).cpu()); self.ys.append(y)
        else:
            d = (out - y.to(out.device)).double()
            self.se += float((d ** 2).sum()); self.ae += float(d.abs().sum()); self.cnt += d.numel()
        self.h += float(h.double().sum()); self.nh += h.numel()
        self.q += tap.q; self.nq += tap.nq; self.a += tap.a; self.na += tap.na

    def result(self):
        r = dict(rate_query=self.q / self.nq, rate_attn=self.a / self.na, rate_hippo=self.h / self.nh)
        if self.task == 'mitbih':
            p, y = torch.cat(self.preds), torch.cat(self.ys)
            f = []
            for c in range(N_CLASSES):
                tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum()); fn = int(((p != c) & (y == c)).sum())
                f.append(0. if tp == 0 else 2 * tp / (2 * tp + fp + fn))
            r.update(f1=float(np.mean(f)), f1_cls=f, acc=float((p == y).float().mean()), n=len(y))
        else:
            r.update(mse=self.se / self.cnt, mae=self.ae / self.cnt, n=self.cnt)
        return r


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=('ett', 'mitbih'), required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--conds', default=','.join(CONDS))
    ap.add_argument('--max_batches', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    conds = a.conds.split(',')
    if any(c not in CONDS for c in conds) or 'clean' not in conds:
        stop(f'bad --conds {a.conds}')
    ref = ref_result(a.raw, a.run_dir)
    args, model, m = load(a.run_dir, a.task, a.device)
    X, Y = test_tensors(args, a.task)
    n = len(X)
    bs = int(args.batch_size)
    if a.task == 'ett':
        gap = int(args.seq_len) + int(args.pred_len)
        if n < 2 * gap + 1:
            stop(f'test split of {n} windows is too short for a gap of {gap}')
        lo, hi = gap, n - gap
    else:
        gap = 1
        lo, hi = 1, n - 1
    offs = {d: torch.randint(lo, hi + 1, (n,), generator=torch.Generator().manual_seed(SEED + d)) for d in range(DRAWS)}
    perm_g = {d: torch.Generator().manual_seed(SEED + 100 + d) for d in range(DRAWS)}
    for d, o in offs.items():
        dist = torch.minimum(o, n - o)
        if int(dist.min()) < gap:
            stop(f'draw {d}: another window at distance {int(dist.min())} < {gap}')
    tap = Tap(m.data_block)
    plan = [(c, 0) for c in conds if c in ('clean', 'zero', 'attn_off')] + [(c, d) for c in conds if c in ('other', 'perm') for d in range(DRAWS)]
    score = {k: Score(a.task) for k in plan}
    chk = dict(split_max_abs=0., neo_repeat_equal=None, other_min_distance=min(int(torch.minimum(o, n - o).min()) for o in offs.values()), gap=gap,
               perm_rows_changed=0, perm_rows=0, zero_all_zero=True, other_bits_changed=0)
    tok_sum = None
    pos_sum = None
    n_rows = 0
    agree = agree_n = 0.
    t0 = time.time()
    n_batches = 0

    def head(h, ctx):
        if a.task == 'mitbih':
            return m.head(h.mean(2)).mean(0)
        B, C, means, stdev = ctx
        return model.head(h.mean(0).reshape(B * C, -1)).reshape(B, C, args.pred_len).permute(0, 2, 1) * stdev + means

    def prep(x):
        return (x, None) if a.task == 'mitbih' else prep_ett(x)

    def neo(z):
        functional.reset_net(model)
        _, neo_in = m._tokens_mc(z)
        return m.neo_forward(neo_in, z.clone().detach())[1]

    for s in range(0, n, bs):
        if a.max_batches and n_batches >= a.max_batches:
            break
        idx = torch.arange(s, min(s + bs, n))
        xb = X[idx].to(a.device)
        z, ctx = prep(xb)
        functional.reset_net(model)
        patched, neo_in = m._tokens_mc(z)
        e = m.encoding(patched)
        o = m.neo_forward(neo_in, z.clone().detach())[1]
        if not bool(((o == 0) | (o == 1)).all()):
            stop('the Neocortex output is not binary')
        if o.shape[2] != 1 or o.shape[1] != z.shape[0]:
            stop(f'unexpected Neocortex output shape {tuple(o.shape)}')
        tok_sum = o.double().sum((1, 2, 3)) if tok_sum is None else tok_sum + o.double().sum((1, 2, 3))
        pos_sum = o.double().sum((1, 2)) if pos_sum is None else pos_sum + o.double().sum((1, 2))
        n_rows += o.shape[1]
        if n_batches < 2:                                              # the split forward and the repeat of the Neocortex path
            functional.reset_net(model)
            ref_out = model(xb)
            ref_out = ref_out[0] if a.task == 'mitbih' else ref_out
            functional.reset_net(model)
            h = e
            for blk in m.data_block:
                h = blk(h, o, self_kv=m.hippo_only)
            chk['split_max_abs'] = max(chk['split_max_abs'], float((head(h, ctx) - ref_out).abs().max()))
            if chk['split_max_abs'] > 1e-5:
                stop(f"the split forward differs from the model ({chk['split_max_abs']:.3g})")
            same = torch.equal(neo(z), o)
            chk['neo_repeat_equal'] = bool(same) if chk['neo_repeat_equal'] is None else (chk['neo_repeat_equal'] and bool(same))
            if not same:
                stop('the Neocortex forward is not repeatable after a reset')
        for k in plan:
            c, d = k
            if c in ('clean', 'attn_off'):
                mx = o
            elif c == 'zero':
                mx = torch.zeros_like(o)
                chk['zero_all_zero'] = chk['zero_all_zero'] and not bool(mx.any())
            elif c == 'other':
                zo, _ = prep(X[(idx + offs[d][idx]) % n].to(a.device))
                mx = neo(zo)
                if not bool(((mx == 0) | (mx == 1)).all()):
                    stop('the other-window Neocortex output is not binary')
                chk['other_bits_changed'] += int((mx != o).sum())
                if d == 0:
                    agree += float((mx == o).double().sum()); agree_n += o.numel()
            else:
                N, R = o.shape[0], o.shape[1]
                p = torch.rand(N, R, generator=perm_g[d]).argsort(0).to(o.device)
                mx = o.gather(0, p[:, :, None, None].expand_as(o))
                if not torch.equal(mx.sum((0, 2, 3)), o.sum((0, 2, 3))):
                    stop('perm changed a spike count')
                chk['perm_rows_changed'] += int((mx != o).any(0).any(1).any(1).sum()); chk['perm_rows'] += R
            tap.reset()
            tap.off = c == 'attn_off'
            functional.reset_net(model)
            h = e
            for blk in m.data_block:
                h = blk(h, mx, self_kv=m.hippo_only)
            tap.off = False
            score[k].add(head(h, ctx), Y[idx], h, tap)
        functional.reset_net(model)
        n_batches += 1
    if 'other' in conds and chk['other_bits_changed'] <= 0:
        stop('other never changed the query')
    if 'perm' in conds and chk['perm_rows_changed'] <= 0:
        stop('perm never changed the query')
    if 'zero' in conds and not chk['zero_all_zero']:
        stop('zero is not all zero')
    res = dict(name=a.name, task=a.task, raw=a.raw, run_dir=a.run_dir, cond=ref['cond'], seed=int(ref['seed']), sha_run=ref.get('sha'), n_windows=n, batch_size=bs,
               N=int(o.shape[0]), D=int(o.shape[3]), T_h=int(e.shape[0]), n_blocks=len(m.data_block), ssa_pe=int(bool(m.data_block[0].ssa.pe)), batches=n_batches,
               time_s=round(time.time() - t0, 1), checks=chk)
    if a.task == 'ett':
        res.update(ds=ref['ds'], pl=int(ref['pl']))
    rate_tok = (tok_sum / (n_rows * o.shape[3])).tolist()
    p = pos_sum / n_rows
    chance = float((p ** 2 + (1 - p) ** 2).mean())
    res['neo'] = dict(rate=float(pos_sum.sum() / (n_rows * o.shape[0] * o.shape[3])), rate_by_token=rate_tok, agree_other=(agree / agree_n if agree_n else None), agree_chance=chance)
    if a.max_batches:
        res['smoke'] = True                                            # checks only: no metric is written
    else:
        out = {}
        for (c, d), sc in score.items():
            r = sc.result()
            for key in ('f1', 'acc') if a.task == 'mitbih' else ('mse', 'mae'):
                if not math.isfinite(r[key]):
                    stop(f'{c} draw {d}: {key} = {r[key]}')
            if c in ('other', 'perm'):
                out.setdefault(c, []).append(r)
            else:
                out[c] = r
        clean = out['clean']
        if a.task == 'mitbih':
            if abs(clean['f1'] - float(ref['f1'])) > 1e-6 or clean['n'] != int(ref['n_test']):
                stop(f"clean macro-F1 {clean['f1']:.6f} (n {clean['n']}) vs logged {ref['f1']} (n {ref['n_test']})")
        else:
            if abs(clean['mse'] - float(ref['mse'])) / float(ref['mse']) > 1e-5 or abs(clean['mae'] - float(ref['mae'])) / float(ref['mae']) > 1e-5 or clean['n'] != int(ref['n']):
                stop(f"clean mse {clean['mse']:.6f} mae {clean['mae']:.6f} (n {clean['n']}) vs logged {ref['mse']} {ref['mae']} (n {ref['n']})")
        res['metrics'] = out
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, a.name + '.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print(f"ok {a.name}: {n_batches} batches, {res['time_s']} s; split {chk['split_max_abs']:.2e}, other distance >= {chk['other_min_distance']} (gap {gap}); -> {path}", flush=True)


if __name__ == '__main__':
    main()
