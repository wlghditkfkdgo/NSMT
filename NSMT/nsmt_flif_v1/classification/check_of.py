"""MODI 2026-11-19: checks of the head dropout (overfitting test) and the full-signal aux target (keep_ratio 0).
Built from the stored config.pt of reference runs (ETTh2 H96 seed 42: the SpikeDE msn base pop LIF and Spikformer(DE) spkall_lif):
  d1 head_dropout 0 builds no dropout module; the state_dict keys and every tensor equal those of a model with p = 0.3 (same seed)
  d2 eval mode: the output with p = 0.3 equals the output with p = 0 (dropout is training-only)
  d3 training mode: the output with p = 0.3 differs from p = 0 on the same input and seed, and is finite
  d4 refusals: p = 1 and p > 0 with the h6 head raise
  k1 keep_ratio 0: the aux target org_x equals the raw patch tokens of the normalised rows; keep_ratio 0.3 gives a different target
Any failure exits 1. usage (classification folder): python check_of.py
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


def build(d, p=0., head_kind='flat', **ov):
    cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda', **ov))
    torch.manual_seed(0)
    return Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias, head_kind=head_kind, seq_len=a.seq_len,
                      head_dropout=p).cuda(), a


for key, d in (('base_sde_pop_lif', run_dir('flif-v1-ett-sde', cond='nsmt4q_pop_lif_msn_sde', ds='ETTh2', pl=96, seed=42)),
               ('spkall_lif', run_dir('flif-v1-ett-sde', cond='spkall_lif', ds='ETTh2', pl=96, seed=42))):
    m0, a = build(d, 0.)
    m3, _ = build(d, 0.3)
    s0, s3 = m0.state_dict(), m3.state_dict()
    ok(f'{key} d1', m0.head_drop is None and isinstance(m3.head_drop, torch.nn.Dropout) and list(s0) == list(s3)
       and all(torch.equal(s0[k], s3[k]) for k in s0), f'(p 0: no module; p 0.3: Dropout; {len(s0)} identical state_dict entries)')
    x = torch.randn(4, a.seq_len, 7, generator=torch.Generator().manual_seed(5)).cuda()
    with torch.no_grad():
        for m in (m0, m3):
            m.eval(); m.train_mode = 'testing'; functional.reset_net(m)
        e0 = m0(x); functional.reset_net(m0); e3 = m3(x); functional.reset_net(m3)
    ok(f'{key} d2', torch.equal(e0, e3), '(eval output identical with p 0.3 / 0)')
    outs = []
    for m in (m0, m3):
        m.train(); m.train_mode = 'training'; functional.reset_net(m)
        torch.manual_seed(1)
        with torch.no_grad():
            outs.append(m(x))
        functional.reset_net(m)
    ok(f'{key} d3', bool(torch.isfinite(outs[1]).all()) and not torch.equal(outs[0], outs[1]), '(training output differs with p 0.3, finite)')
    del m0, m3
    torch.cuda.empty_cache()

for bad, kw in (('p 1', dict(p=1.0)), ('h6 head', dict(p=0.1, head_kind='h6'))):
    try:
        build(run_dir('flif-v1-ett-sde', cond='spkall_lif', ds='ETTh2', pl=96, seed=42), **kw)
        ok(f'd4 refusal {bad}', False, '(built)')
    except (ValueError, NotImplementedError) as e:
        ok(f'd4 refusal {bad}', True, f'({e})')

d = run_dir('flif-v1-ett-sde', cond='nsmt4q_pop_lif_msn_sde', ds='ETTh2', pl=96, seed=42)
x = torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(5)).cuda()
xc = x - x.mean(1, keepdim=True)
z = (xc / torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5)).permute(0, 2, 1).reshape(-1, 1, 96)
tg = {}
for kr in (0.0, 0.3):
    m, a = build(d, 0., keep_ratio=kr)
    m.train(); m.train_mode = 'training'; m.aux = True; functional.reset_net(m)
    m(x); tg[kr] = m.inner.org_x.clone(); m.aux_rec_loss(); functional.reset_net(m)
    if kr == 0.0:
        p = m.inner.padding_patch_layer(z).unfold(dimension=-1, size=m.inner.patch_size, step=m.inner.stride)     # [B', 1, N, P]
        raw = p.permute(0, 2, 1, 3).reshape(z.shape[0], p.shape[2], m.inner.patch_size).transpose(0, 1).unsqueeze(2)
    del m
ok('k1', torch.allclose(tg[0.0], raw, atol=1e-6) and not torch.allclose(tg[0.3], raw, atol=1e-3),
   f'(keep 0: target = raw patches, max |diff| {(tg[0.0] - raw).abs().max().item():.1e}; keep 0.3 differs by {(tg[0.3] - raw).abs().max().item():.3f})')
print('ALL PASS' if not FAIL else f'FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
