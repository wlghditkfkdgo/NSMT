"""MODI 2026-11-20 (hyperparameter range 1, user decision 11-20): fail-closed checks of --wd_exclude_1d / --val_per_epoch.
Run from nsmt_flif_v1/classification. The default path (both flags off) is checked elsewhere: 1-epoch default smokes before / after
the change give bit-identical weights and RESULT lines, and the runners' command lines are unchanged (scratchpad harness).
  unit (no argument):
    r1  make_adamw(exclude_1d=False) is torch.optim.AdamW(model.parameters(), lr, weight_decay): one group, the same parameter
        objects in the same order, the same optimizer state_dict
    r2  exclude_1d=True (ETT: Spikformer(DE), base pop LIF / pop f-LIF; MIT-BIH: the same three, PE off): the no-decay group is
        exactly the parameters with ndim <= 1 or a name ending in '.bias', every one of them belongs to a BatchNorm module, the two
        groups are disjoint and cover every trainable parameter; prints the counts
    r3  one AdamW step with zero gradients: no-decay parameters bit-identical, decayed ones = p * (1 - lr * wd) (allclose)
    r4  train_epoch(hook, marks): the hook runs after exactly the marked steps (1-based), training mode is restored after it, and
        hook None runs no hook (plain and aux branches)
    r5  train.py / train_forecast.py build the optimizer only through make_adamw with the flag; no direct torch.optim.AdamW left;
        train.py refuses --val_per_epoch != 1
  --smoke <raw.txt> ...: every RESULT line of the given smoke suites
    s1  config.pt wd_exclude_1d / val_per_epoch match the RESULT wdx= / vpe= fields and the run folder suffix _wdx / _vpe<K>
    s2  (vpe) val_inner.csv: K rows per epoch at the steps ceil(n j / K), j = 1 .. K - 1, and n; best_epoch.txt has 4 fields;
        its best value is the minimum validation L1 over all rows up to the stop, at the recorded (epoch, step)
    s3  (vpe) early stopping replayed from val_inner.csv (an epoch with any improvement resets the wait; patience from config)
        stops after the same number of epochs as epochs_run
    s4  (vpe) the stored best checkpoint re-evaluated on the validation split (unshuffled) gives the val MSE recorded at the best
        (epoch, step) within 2e-6 (MSE does not depend on the batch order; the L1 is a mean of per-batch means and does)
"""
import argparse
import csv
import inspect
import math
import os
import re
import sys

import torch

sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional

from forecast_wrap import Forecaster
from load_model import LOAD_MODEL
from utils import make_adamw

HERE = os.getcwd()
FAIL = []


def ok(name, cond, msg=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {msg}' if msg else ''), flush=True)
    if not cond:
        FAIL.append(name)


def find(path, **kv):
    for ln in open(path):
        if ln.startswith('RESULT') and all(f' {k}={v} ' in ln for k, v in kv.items()):
            return dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)['dir']
    raise RuntimeError((path, kv))


def ett_model(cond, ds='ETTh2'):
    run = find(os.path.join(HERE, '..', 'forecasting', 'results', 'flif-v1-ett-nope', 'raw.txt'), cond=cond, ds=ds, pl=96, seed=42)
    a = argparse.Namespace(**dict(torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu'), device='cuda'))
    torch.manual_seed(0)
    return Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda(), a


def mb_model(cond):
    run = find(os.path.join(HERE, 'results', 'flif-v1-mitbih-nope', 'raw.txt'), cond=cond, seed=42)
    a = argparse.Namespace(**dict(torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu'), device='cuda'))
    torch.manual_seed(0)
    return LOAD_MODEL[a.model](a, True).cuda(), a


def unit():
    conds = ('spkall_lif', 'nsmt4q_pop_lif_msn_sde', 'nsmt4q_pop_flif_msn_sde')
    models = [(f'ETT {c}', *ett_model(c)) for c in conds] + [(f'MIT-BIH {c}', *mb_model(c)) for c in conds]
    for name, m, a in models:
        # r1
        ref = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        got = make_adamw(m, a.lr, a.weight_decay, False)
        same_params = (len(got.param_groups) == 1 and len(got.param_groups[0]['params']) == len(ref.param_groups[0]['params'])
                       and all(p is q for p, q in zip(got.param_groups[0]['params'], ref.param_groups[0]['params'])))
        ok(f'r1 {name}: exclude off = plain AdamW', type(got) is type(ref) and same_params and got.state_dict() == ref.state_dict())
        # r2
        opt = make_adamw(m, a.lr, a.weight_decay, True)
        g_dec, g_no = opt.param_groups
        bn_names = {f'{mn}.{pn}' for mn, mod in m.named_modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
                    for pn, _ in mod.named_parameters(recurse=False)}
        named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        want_no = [p for n, p in named if p.ndim <= 1 or n.endswith('.bias')]
        no_names = [n for n, p in named if p.ndim <= 1 or n.endswith('.bias')]
        ids_dec, ids_no = {id(p) for p in g_dec['params']}, {id(p) for p in g_no['params']}
        cover = ids_dec | ids_no == {id(p) for _, p in named} and not (ids_dec & ids_no)
        ok(f'r2 {name}: no-decay group = ndim <= 1 / .bias, all BatchNorm, groups disjoint and complete',
           ids_no == {id(p) for p in want_no} and cover and set(no_names) <= bn_names
           and g_no['weight_decay'] == 0. and g_dec['weight_decay'] == a.weight_decay,
           f'{sum(p.numel() for p in want_no)} of {sum(p.numel() for _, p in named)} values in {len(want_no)} tensors without decay; '
           f'wd {a.weight_decay}')
        # r3
        before = {n: p.detach().clone() for n, p in named}
        for _, p in named:
            p.grad = torch.zeros_like(p)
        opt.step()
        nod = all(torch.equal(p, before[n]) for n, p in named if n in set(no_names))
        dec = all(torch.allclose(p, before[n] * (1 - a.lr * a.weight_decay), rtol=0, atol=1e-7) for n, p in named if n not in set(no_names))
        ok(f'r3 {name}: zero-gradient step decays only the >= 2-D tensors', nod and dec)
    # r4
    import train_forecast as TF
    m, a = ett_model('nsmt4q_pop_lif_msn_sde')
    g = torch.Generator().manual_seed(1)
    dl = [(torch.randn(4, a.seq_len, 7, generator=g), torch.randn(4, 48 + a.pred_len, 7, generator=g), None, None) for _ in range(5)]
    for aux in (False, True):
        mm, aa = ett_model('nsmt4q_pop_lif_msn_sde') if not aux else ett_model('nsmt4q_pop_lif_msn_sde')
        if aux:
            mm.aux = True
        opt = torch.optim.AdamW(mm.parameters(), lr=1e-6)
        calls = []

        def hook(n):
            calls.append((n, mm.training)); mm.eval(); mm.train_mode = 'testing'
        TF.train_epoch(mm, dl, opt, aa.pred_len, 'cuda', r=0.1 if aux else 0.0, hook=hook, marks={2, 4})
        ok(f'r4 train_epoch {"aux" if aux else "plain"}: hook after steps 2, 4 only, training mode restored',
           [c[0] for c in calls] == [2, 4] and all(c[1] for c in calls) and mm.training and mm.train_mode == 'training', str(calls))
        calls.clear()
        TF.train_epoch(mm, dl, opt, aa.pred_len, 'cuda', r=0.1 if aux else 0.0)
        ok(f'r4 train_epoch {"aux" if aux else "plain"}: no hook by default', calls == [])
    # r5
    for f in ('train.py', 'train_forecast.py'):
        src = open(os.path.join(HERE, f)).read()
        ok(f'r5 {f}: optimizer only via make_adamw(..., wd_exclude_1d)', 'torch.optim.AdamW(' not in src
           and len(re.findall(r"make_adamw\(model, args\.lr, args\.weight_decay, bool\(int\(getattr\(args, 'wd_exclude_1d', 0\)\)\)\)", src)) == 1)
    src = open(os.path.join(HERE, 'train.py')).read()
    ok('r5 train.py refuses --val_per_epoch != 1', "if int(getattr(args, 'val_per_epoch', 1)) != 1:" in src and 'raise NotImplementedError' in src)


def smoke(paths):
    from torch.utils.data import DataLoader
    import train_forecast as TF
    from data_provider.data_factory import data_provider
    n_lines = 0
    for path in paths:
        for ln in open(path):
            if not ln.startswith('RESULT'):
                continue
            n_lines += 1
            f = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
            d = f['dir']
            cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
            wdx, vpe = int(f.get('wdx', 0)), int(f.get('vpe', 1))
            leaf = os.path.basename(d)
            tag = f"{f['cond']} {f.get('ds', 'MIT-BIH')} wdx={wdx} vpe={vpe}"
            ok(f's1 {tag}: config / RESULT / folder agree',
               int(cfg.get('wd_exclude_1d', -1)) == wdx and int(cfg.get('val_per_epoch', -1)) == vpe
               and (('_wdx' in leaf) == bool(wdx)) and ((f'_vpe{vpe}' in leaf) == (vpe > 1)))
            if vpe == 1:
                continue
            K = vpe
            rows = list(csv.DictReader(open(os.path.join(d, 'log', 'val_inner.csv'))))
            n = int(rows[0]['steps_per_epoch'])
            marks = sorted({int(math.ceil(n * j / K)) for j in range(1, K)} | {n})
            be = open(os.path.join(d, 'log', 'best_epoch.txt')).read().split()
            b_ep, b_val, ep_run, b_step = int(be[0]), float(be[1]), int(be[2]), be[3]
            per_ep = {}
            for r in rows:
                per_ep.setdefault(int(r['epoch']), []).append(r)
            ok(f's2 {tag}: {K} validations per epoch at {marks}', len(be) == 4 and sorted(per_ep) == list(range(ep_run))
               and all([int(r['step']) for r in per_ep[e]] == marks for e in per_ep))
            best = min(rows, key=lambda r: float(r['val_loss']))
            ok(f's2 {tag}: best = minimum validation L1 at the recorded (epoch, step)',
               abs(float(best['val_loss']) - b_val) <= 1e-6 and int(best['epoch']) == b_ep and f"{best['step']}/{n}" == b_step,
               f'best {b_val} @ {b_ep} {b_step}; epochs run {ep_run}')
            # s3 early stopping replay
            pat = int(cfg['patience']); bst = float('inf'); wait = 0; stop = None
            for e in sorted(per_ep):
                imp = False
                for r in per_ep[e]:
                    if float(r['val_loss']) < bst:
                        bst = float(r['val_loss']); imp = True
                wait = 0 if imp else wait + 1
                if wait >= pat:
                    stop = e + 1; break
            ok(f's3 {tag}: early stopping replay', (stop or int(cfg['epoch'])) == ep_run, f'replay {stop or cfg["epoch"]} vs run {ep_run}, patience {pat}')
            # s4 checkpoint re-evaluation
            a = argparse.Namespace(**dict(cfg, device='cuda'))
            torch.manual_seed(0)
            m = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()
            m.load_state_dict(torch.load(os.path.join(d, 'model_state', 'checkpoint.pth'), map_location='cuda'))
            dl = DataLoader(data_provider(TF.data_args(a, a), 'val')[0], batch_size=a.batch_size, shuffle=False, num_workers=2)
            v = TF.evaluate(m, dl, a.pred_len, 'cuda')
            ok(f's4 {tag}: checkpoint reproduces the best val MSE', abs(v['mse'] - float(best['val_mse'])) <= 2e-6,
               f"{v['mse']:.6f} vs {best['val_mse']}")
    ok('smoke: RESULT lines found', n_lines > 0, str(n_lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', nargs='+')
    x = ap.parse_args()
    smoke(x.smoke) if x.smoke else unit()
    print(f'{len(FAIL)} failed' if FAIL else 'all checks passed')
    sys.exit(1 if FAIL else 0)
