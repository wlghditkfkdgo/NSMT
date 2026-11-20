"""MODI 2026-11-20 (aux-target memory check, user decision 11-20): fail-closed checks of the --aux_shift change. Run from
nsmt_flif_v1/classification after `fw_identity_xnor.py compare` / `fw_identity_auxs.py compare` (x0: the default and aux paths are
bit-identical with aux_shift 0).
  a1  aux_pair on index tensors: s > 0 pairs output i with token i - s, s < 0 with token i + |s|, 0 is the identity (same objects)
  a2  ETT: Forecaster.aux_rec_loss() with aux_shift s equals the MSE computed by hand from the kept Neocortex output and the
      target (weak_decoder(x_neo)[s:] vs org_x[:-s], ...), for s in {2, 4, -2}, normal / memoryless / shuffled-order Neocortex,
      pop LIF / pop f-LIF (shuffled: the outputs are back at the original positions before the pairing)
  a3  MIT-BIH: the train-mode tuple of myModel carries the shifted pair, shapes [N - |s|, B, 1, P], and equals the hand pairing
  a4  the normal Neocortex is causal over the tokens (eval mode): changing the Neocortex input of token j leaves the outputs of
      tokens < j bit-identical and changes some output at a token >= j; the memoryless Neocortex is token-local: only token j
      changes. So a future target (s < 0) cannot be read from the token's own past, and a memoryless model sees token i only.
  a5  guards: |s| >= N tokens and the Spikformer-style model (no Neocortex) raise ValueError
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional

from forecast_wrap import Forecaster
from load_model import LOAD_MODEL

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


def cfg_of(run, **over):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    cfg = dict(cfg, device='cuda', ssa_pe=0)
    cfg.update(over)
    return argparse.Namespace(**cfg)


def ett_run(n):
    return find(os.path.join(HERE, '..', 'forecasting', 'results', 'flif-v1-ett-auxf', 'raw.txt'),
                cond=f'nsmt4q_pop_{n}_msn_sde_auxf', ds='ETTh2', pl=96, seed=42)


def mb_run(n):
    return find(os.path.join(HERE, 'results', 'flif-v1-mitbih-auxf', 'raw.txt'), cond=f'nsmt4q_pop_{n}_msn_sde_auxf', seed=42)


# a1
torch.manual_seed(0)
a = argparse.Namespace(**dict(torch.load(os.path.join(ett_run('lif'), 'model_state', 'config.pt'), map_location='cpu'), device='cuda'))
for s in (2, 4, -2, 0):
    m = LOAD_MODEL[a.model](argparse.Namespace(**dict(vars(a), aux_shift=s)), True)
    N = m.num_patches
    idx = torch.arange(N, dtype=torch.float64).view(N, 1, 1, 1).expand(N, 3, 1, 2)
    r, o = m.aux_pair(idx.clone(), idx.clone())
    if s == 0:
        rr, oo = idx, idx
        r2, o2 = m.aux_pair(rr, oo)
        ok(f'a1 s=0 identity', r2 is rr and o2 is oo)
    else:
        ok(f'a1 s={s} pairing', r.shape[0] == N - abs(s) and bool(((r - o) == s).all()), f'{N} tokens -> {r.shape[0]} pairs')

# a2 (ETT)
x = torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(5)).cuda()
for n in ('lif', 'flif'):
    for var in ('mem', 'nomem', 'shuf'):
        for s in (2, 4, -2):
            a = cfg_of(ett_run(n), aux_shift=s, neo_memoryless=(var == 'nomem'), neo_shuffle=(var == 'shuf'))
            torch.manual_seed(0)
            m = Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()
            m.aux = True; m.train(); m.train_mode = 'training'; functional.reset_net(m)
            m(x)
            xn, org = m._x_neo, m.inner.org_x
            rec = m.inner.weak_decoder(xn)
            hand = ((org[:-s] - rec[s:]) ** 2).mean() if s > 0 else ((org[-s:] - rec[:s]) ** 2).mean()
            got = m.aux_rec_loss()
            ok(f'a2 ETT {n} {var} s={s}', torch.equal(got, hand), f'{got.item():.6f} vs {hand.item():.6f}')
            functional.reset_net(m)

# a3 (MIT-BIH)
xb = torch.rand(4, 1, 187, generator=torch.Generator().manual_seed(7)).cuda()
for n in ('lif', 'flif'):
    for s in (2, -2, 4):
        a = cfg_of(mb_run(n), aux_shift=s, neo_memoryless=(s == 4), mitbih_val=True)
        torch.manual_seed(0)
        m = LOAD_MODEL[a.model](a, True).cuda(); m.train(); m.train_mode = 'training'; functional.reset_net(m)
        cap = {}
        h = m.weak_decoder.register_forward_hook(lambda mod, inp, out: cap.__setitem__('rec', out))
        _, _, _, org_x, rec_x = m(xb)
        h.remove()
        full_org, full_rec = m.org_x, cap['rec']
        N = m.num_patches
        exp_r, exp_o = (full_rec[s:], full_org[:-s]) if s > 0 else (full_rec[:s], full_org[-s:])
        ok(f'a3 MIT-BIH {n} s={s}{" nomem" if s == 4 else ""}', tuple(org_x.shape) == (N - abs(s), 4, 1, full_org.shape[-1])
           and torch.equal(org_x, exp_o) and torch.equal(rec_x, exp_r), f'shape {tuple(org_x.shape)}')
        functional.reset_net(m)

# a4 causality / locality of the Neocortex (eval mode, the trained weights of the stored aux runs)
for task in ('ett', 'mb'):
    for n in ('lif', 'flif'):
        for mem in (False, True):
            run = ett_run(n) if task == 'ett' else mb_run(n)
            a = cfg_of(run, neo_memoryless=mem)
            torch.manual_seed(0)
            inner = LOAD_MODEL[a.model](a, True).cuda()
            w = torch.load(os.path.join(run, 'model_state', 'checkpoint.pth' if task == 'ett' else 'best+model.pt'), map_location='cuda')
            if task == 'ett':
                w = {k[len('inner.'):]: v for k, v in w.items() if k.startswith('inner.')}
            rr = inner.load_state_dict(w, strict=False)
            if [k for k in rr.missing_keys if not (task == 'ett' and k.startswith('head.'))]:   # ETT: the unused inner head
                raise RuntimeError(f'{run}: missing {rr.missing_keys[:3]}')
            inner.eval()
            N = inner.num_patches
            z = (torch.randn(64, 1, 96, generator=torch.Generator().manual_seed(3)) if task == 'ett'
                 else torch.rand(64, 1, 187, generator=torch.Generator().manual_seed(3))).cuda()
            with torch.no_grad():
                _, neo_in = inner._tokens_mc(z)
                functional.reset_net(inner); o0 = inner.neo_forward(neo_in, z)[1]; functional.reset_net(inner)
                good = True; later = True
                for j in (1, N // 2, N - 2):
                    # 5.0: the memory-trained weights barely fire when run memoryless (rate ~1e-5), a 0.5 change moves no spike
                    pert = neo_in.clone(); pert[j] += 5.0 * torch.randn_like(pert[j])
                    o1 = inner.neo_forward(pert, z)[1]; functional.reset_net(inner)
                    diff = (o1 != o0).flatten(1).any(1)                    # [N] which token outputs changed
                    if mem:
                        good &= not bool(diff[:j].any()) and not bool(diff[j + 1:].any())
                    else:
                        good &= not bool(diff[:j].any())
                    later &= bool(diff[j:].any())
            ok(f'a4 {task} {n} {"memoryless: token-local" if mem else "normal: causal"}', good and later)

# a5 guards
a = argparse.Namespace(**dict(torch.load(os.path.join(ett_run('lif'), 'model_state', 'config.pt'), map_location='cpu'), device='cuda'))
for s in (24, -24, 30):
    try:
        LOAD_MODEL[a.model](argparse.Namespace(**dict(vars(a), aux_shift=s)), True); ok(f'a5 |s|={abs(s)} raises', False)
    except ValueError:
        ok(f'a5 s={s} raises', True)
spk = find(os.path.join(HERE, '..', 'forecasting', 'results', 'flif-v1-ett-nope', 'raw.txt'), cond='spkall_lif', ds='ETTh2', pl=96, seed=42)
b = argparse.Namespace(**dict(torch.load(os.path.join(spk, 'model_state', 'config.pt'), map_location='cpu'), device='cuda', aux_shift=2))
try:
    LOAD_MODEL[b.model](b, True); ok('a5 Spikformer(DE) with aux_shift raises', False)
except ValueError:
    ok('a5 Spikformer(DE) with aux_shift raises', True)

print(f'{len(FAIL)} failed' if FAIL else 'all checks passed')
sys.exit(1 if FAIL else 0)
