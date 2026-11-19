"""MODI 2026-11-19: checks of the positional-encoding-off test (--ssa_pe 0; user request 11-19 "relative positional encoding을 끄고
실험을 해줘. (Spikformer 도 마찬가지)"). Models are built from the stored config.pt of the SpikeDE-basis reference runs (seed 42):
ETT ETTh2 H96 (flif-v1-ett-sde) and MIT-BIH (flif-v1-mitbih-sde); spkall_lif = Spikformer(DE), nsmt4q_pop_<lif|flif>_msn_sde = the base.
  n0 the reference configs have ssa_pe 1 and pos_emb False (the absolute position is already off; the relative bias is the only PE)
  n1 ssa_pe 0 removes exactly the relative bias table (+ its index buffer) of every attention module; every other state_dict entry
     equals the ssa_pe 1 model built with the same seed
  n2 at initialisation (table = 0) the training-mode output of ssa_pe 0 equals that of ssa_pe 1 (bitwise)
  n3 order (MIT-BIH, trained last-epoch weights, eval mode, the same token permutation for the hippocampus and the Neocortex input):
     Spikformer(DE) without the table is blind to the token order (logits unchanged up to float rounding, no changed prediction);
     with its trained table it is not; the base without the table is not (the Neocortex reads the tokens in order)
  n4 order (ETT, trained best-validation weights, table removed): the output changes with the token order for every model (the flat
     head reads the tokens by position), so on ETT no model is order-blind without the table
Any failure exits 1. usage (classification folder): python check_nope.py
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
CL = os.path.join(HERE, 'results')
CONDS = ('spkall_lif', 'nsmt4q_pop_lif_msn_sde', 'nsmt4q_pop_flif_msn_sde')
PE_KEYS = ('relative_bias_table', 'rel_pos_idx')
FAIL = []


def ok(name, cond, info=''):
    print(f"{'PASS' if cond else 'FAIL'} {name} {info}", flush=True)
    if not cond:
        FAIL.append(name)


def run_dir(path, **sel):
    got = []
    for ln in open(path):
        if ln.startswith('RESULT'):
            d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
            if all(d.get(k) == str(v) for k, v in sel.items()):
                got.append(d['dir'])
    if len(got) != 1:
        raise RuntimeError(f'{path} {sel}: {len(got)} rows')
    return got[0]


def build(kind, d, **ov):
    cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda', **ov))
    torch.manual_seed(0)
    body = LOAD_MODEL[a.model](a, True)
    m = Forecaster(body, a.embed_dim, a.pred_len, bias=a.bias, seq_len=a.seq_len) if kind == 'ett' else body
    return m.cuda(), body, a, cfg


def inputs(kind, a):
    g = torch.Generator().manual_seed(5)
    return torch.randn(4, a.seq_len, 7, generator=g).cuda() if kind == 'ett' else torch.randn(64, 1, 187, generator=g).cuda()


def out_of(m, x, mode):
    functional.reset_net(m)
    if mode == 'training':
        m.train(); m.train_mode = 'training'
    else:
        m.eval(); m.train_mode = 'testing'
    with torch.no_grad():
        o = m(x)
    functional.reset_net(m)
    return (o[0] if isinstance(o, tuple) else o).detach()


def permuted(body, perm):
    """wraps body._tokens_mc: the same token permutation for the hippocampus input [T_h, B, N, P] and the Neocortex input [N, B, 1, P]."""
    orig = body._tokens_mc

    def f(x):
        h, n = orig(x)
        return h[:, :, perm].contiguous(), (n[perm].contiguous() if n is not None else None)
    body._tokens_mc = f
    return orig


def is_pe(k):
    return any(p in k for p in PE_KEYS)


DIRS = {('ett', c): run_dir(os.path.join(FC, 'flif-v1-ett-sde', 'raw.txt'), cond=c, ds='ETTh2', pl=96, seed=42) for c in CONDS}
DIRS.update({('mb', c): run_dir(os.path.join(CL, 'flif-v1-mitbih-sde', 'raw.txt'), cond=c, seed=42) for c in CONDS})

for (kind, c), d in DIRS.items():
    key = f'{kind} {c}'
    on, _, a, cfg = build(kind, d)
    ok(f'{key} n0', int(cfg.get('ssa_pe', 1)) == 1 and not cfg.get('pos_emb', False), f"(reference config: ssa_pe {cfg.get('ssa_pe', 1)}, pos_emb {cfg.get('pos_emb', False)})")
    off, _, _, _ = build(kind, d, ssa_pe=0)
    s1, s0 = on.state_dict(), off.state_dict()
    gone = [k for k in s1 if k not in s0]
    ok(f'{key} n1', len(gone) > 0 and all(is_pe(k) for k in gone) and not any(is_pe(k) for k in s0) and [k for k in s1 if k in s0] == list(s0)
       and all(torch.equal(s1[k], s0[k]) for k in s0), f'(removed {len(gone)} entries: {sorted(set(k.split(".")[-1] for k in gone))}; {len(s0)} identical entries)')
    x = inputs(kind, a)
    ok(f'{key} n2', torch.equal(out_of(on, x, 'training'), out_of(off, x, 'training')), '(initial training-mode output identical)')
    # trained weights: MIT-BIH = last epoch (best+model.pt, the tested weights); ETT = best validation (checkpoint.pth, the tested weights)
    w = torch.load(os.path.join(d, 'model_state', 'best+model.pt' if kind == 'mb' else 'checkpoint.pth'), map_location='cuda')
    tab = [v for k, v in w.items() if 'relative_bias_table' in k]
    miss_on = on.load_state_dict(w, strict=False)
    miss_off = off.load_state_dict(w, strict=False)
    if miss_off.missing_keys or not all(is_pe(k) or 'weak_decoder' in k for k in miss_off.unexpected_keys) or miss_on.missing_keys:
        raise RuntimeError(f'{key}: weights do not fit (missing {miss_off.missing_keys[:3]}, unexpected {miss_off.unexpected_keys[:3]})')
    N = on.num_patches if kind == 'ett' else on.num_patches
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(3)).cuda()
    if bool((perm == torch.arange(N, device=perm.device)).all()):
        raise RuntimeError('identity permutation')
    res = {}
    for name, m in (('trained table', on), ('table removed', off)):
        body = m.inner if kind == 'ett' else m
        o = out_of(m, x, 'testing')
        orig = permuted(body, perm)
        p = out_of(m, x, 'testing')
        body._tokens_mc = orig
        res[name] = ((o - p).abs().max().item(), (int((o.argmax(-1) != p.argmax(-1)).sum().item()) if kind == 'mb' else None), o.abs().mean().item())
    info = '; '.join(f'{k}: max |diff| {v[0]:.2e}' + (f', {v[1]}/64 predictions change' if v[1] is not None else '') + f' (mean |out| {v[2]:.2f})' for k, v in res.items())
    info += f"; trained table mean |b| {float(torch.cat([t.flatten() for t in tab]).abs().mean()):.3f}, {N} tokens"
    off_d, on_d = res['table removed'], res['trained table']
    if kind == 'mb' and c == 'spkall_lif':
        ok(f'{key} n3', off_d[0] < 1e-3 and off_d[1] == 0 and on_d[0] > 1e-2, f'(order-blind without the table, not with it; {info})')
    elif kind == 'mb':
        ok(f'{key} n3', off_d[0] > 1e-2, f'(order still matters without the table; {info})')
    else:
        ok(f'{key} n4', off_d[0] > 1e-2, f'(ETT flat head: order matters without the table; {info})')
    del on, off
    torch.cuda.empty_cache()
print('ALL PASS' if not FAIL else f'FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
