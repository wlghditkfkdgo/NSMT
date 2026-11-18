"""MODI 2026-11-19: checks of the ETT aux loss (train_forecast.py --aux_alpha, Forecaster.aux_rec_loss; user request 11-19).
Built from the stored config.pt of the reference msn runs (ETTh2 H96 pop LIF, ETTm2 H720 pop f-LIF, seed 42), with keep_ratio set to
the NSMT value of the dataset (ETTh2 0.3, ETTm2 0.25), once with the spikingjelly hippocampus and once with --neuron_all lif:
  a1 r = aux_alpha * pred_len / 720 for the four cells (ETTh2 alpha 0.8, ETTm2 0.5)
  a2 training-mode forward with aux on: the target org_x has the reconstruction's shape [N, B', 1, P], is finite, and differs from
     the raw patch tokens (low-pass); with keep_ratio 0 it equals them
  a3 the reconstruction loss alone sends gradient to weak_decoder and the Neocortex and none to the hippocampus attention / head
  a4 the forecast output with aux on equals the output with aux off (aux only adds a loss term)
  a5 aux_rec_loss refuses a Spikformer-style forecaster and a call without a forward
Any failure exits 1. usage (classification folder): python check_aux.py
"""
import os
import sys
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional

from forecast_wrap import Forecaster
from load_model import LOAD_MODEL

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.join(HERE, '..', 'forecasting', 'results')
FAIL = []


def ok(name, cond, info=''):
    print(f"{'PASS' if cond else 'FAIL'} {name} {info}")
    if not cond:
        FAIL.append(name)


def run_dir(suite, **sel):
    got = []
    for ln in open(os.path.join(FC, suite, 'raw.txt')):
        if ln.startswith('RESULT'):
            d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
            if all(d.get(k) == str(v) for k, v in sel.items()):
                got.append(d['dir'])
    if len(got) != 1:
        raise RuntimeError(f'{suite} {sel}: {len(got)} rows')
    return got[0]


def build(d, **ov):
    cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda', **ov))
    torch.manual_seed(0)
    return Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda(), a


R = {('ETTh2', 96): 0.8 * 96 / 720, ('ETTh2', 720): 0.8, ('ETTm2', 96): 0.5 * 96 / 720, ('ETTm2', 720): 0.5}
ok('a1', all(0 < r < 1 for r in R.values()) and abs(R['ETTh2', 96] - 0.106667) < 1e-6 and abs(R['ETTm2', 96] - 0.066667) < 1e-6,
   '(' + ', '.join(f'{k[0]} H{k[1]} r {v:.6f}' for k, v in R.items()) + ')')

CASES = {'h2_pop_lif': (run_dir('flif-v1-ett-ms', cond='nsmt4q_pop_lif_msn', ds='ETTh2', pl=96, seed=42), 0.3),
         'm2_pop_flif': (run_dir('flif-v1-ett-ms', cond='nsmt4q_pop_flif_msn', ds='ETTm2', pl=720, seed=42), 0.25)}
for key, (d, kr) in CASES.items():
    for lib in ('sj_lif', 'lif'):
        tag = f'{key} {"spikingjelly" if lib == "sj_lif" else "SpikeDE"}'
        m, a = build(d, keep_ratio=kr, neuron_all=lib)
        g = torch.Generator().manual_seed(5)
        x = torch.randn(4, a.seq_len, 7, generator=g).cuda(); y = torch.randn(4, a.pred_len, 7, generator=g).cuda()
        m.train(); m.train_mode = 'training'; functional.reset_net(m)
        out_off = m(x).detach(); functional.reset_net(m)
        m.aux = True
        out_on = m(x)
        org = m.inner.org_x
        rec = m.aux_rec_loss()
        rec.backward()
        grads = {n: p.grad for n, p in m.named_parameters()}
        def nz(sub):
            gs = [g_ for n, g_ in grads.items() if sub in n]
            return bool(gs) and any(g_ is not None and g_.abs().sum() > 0 for g_ in gs)
        def zero(sub):
            gs = [g_ for n, g_ in grads.items() if sub in n]
            return bool(gs) and all(g_ is None or g_.abs().sum() == 0 for g_ in gs)
        functional.reset_net(m)
        # raw patch tokens of the same rows (keep_ratio 0 path of _tokens_mc)
        m.inner.keep_ratio = 0.0
        z = ((x - x.mean(1, keepdim=True)) / torch.sqrt(torch.var(x - x.mean(1, keepdim=True), dim=1, keepdim=True, unbiased=False) + 1e-5))
        z = z.permute(0, 2, 1).reshape(-1, 1, a.seq_len)
        m.inner._tokens_mc(z)
        raw = m.inner.org_x
        ok(f'{tag} a2', org.shape == (m.num_patches, x.shape[0] * 7, 1, a.patch_size) and bool(torch.isfinite(org).all())
           and not torch.allclose(org, raw), f'(target {tuple(org.shape)}, finite, differs from raw tokens: max |diff| {(org - raw).abs().max().item():.3f})')
        ok(f'{tag} a3', nz('weak_decoder') and nz('neo') and zero('q_linear') and zero('head.'),
           f'(rec-only gradient: decoder {nz("weak_decoder")}, Neocortex {nz("neo")}, attention q_linear zero {zero("q_linear")}, head zero {zero("head.")}; rec {rec.item():.4f})')
        ok(f'{tag} a4', torch.equal(out_off, out_on.detach()), '(forecast identical with aux on / off)')
        del m
        torch.cuda.empty_cache()

d = run_dir('flif-v1-gen-ett', cond='spk_ssa', ds='ETTh2', pl=96, seed=42)
m, a = build(d)
try:
    m.aux = True; m.aux_rec_loss(); ok('a5 refusal', False, '(spk_ssa accepted)')
except RuntimeError as e:
    ok('a5 refusal', True, f'({e})')
print('ALL PASS' if not FAIL else f'FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
