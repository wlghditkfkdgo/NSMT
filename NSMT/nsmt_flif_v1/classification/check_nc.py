"""MODI 2026-11-21 (4-2 coupling placement, user decision 11-20 "(가) + (나)"): fail-closed unit checks of --neo_current / --hippo_attn.
usage (classification folder, one GPU): python check_nc.py
Models are built from the stored configs of the range-1 reference runs (flif-v1-ett-r1 ETTh2 H96 seed 42 -> forecaster; flif-v1-mitbih-r1
seed 42 -> classifier), base pop LIF and pop f-LIF (nsmt4q_pop_<n>_msn_sde), with the flags changed on the Namespace; every model is built
after torch.manual_seed(0).
  n1  (가) with alpha set to 0 = the base: initial state = the base's + neo_alpha_c only; bit-identical train-mode output and gradient of
      every shared parameter (ETT L1, MIT-BIH CE)
  n2  (가) alpha = 1 (the init): the input current of the hippocampus embedding neuron = the base model's + the Neocortex output
      (exact), the same added current at every T_h step; the Neocortex output is binary; --neo_current_init 0.5 gives alpha = 0.5
  n3  (나): no attention module left in the hippocampus blocks; parameters = base - attention + D; every remaining initial tensor equals
      the base's; each block maps x to x (1 - MLP(x)) (exact)
  n4  token mixing (eval mode, the TRAINED base weights loaded into every variant -- missing keys: alpha only; (나) unused keys: the
      attention tensors only -- so the neurons fire as in a trained model; 32 random windows / beats; token j = N // 2 of the
      hippocampus and Neocortex inputs + 10; hippocampus output per token; the number of changed entries is printed):
        base and (가): some token i < j changes (the attention mixes tokens; sensitivity of the test)
        (나): no token i < j changes, token j and some token i > j change (only the forward-reading Neocortex mixes)
        (나) with --neo_memoryless: only token j changes (no path mixes tokens)
      (first version: 4 windows, + 3 -- the trained ETT base attention is so quiet that no earlier token changed; see the log)
  n5  train mode: nonzero gradient on alpha and on the Neocortex parameters in (가) and (나)
  n6  make_adamw(exclude_1d=True) puts neo_alpha_c in the weight-decay-0 group
  n7  forecast_wrap.Forecaster.step_features = the hippocampus output of myModel.forward on the same rows, (가) and (나)
  n8  invalid settings stop before training: hippo_attn 0 without neo_current; neo_current with hippo_xattn_mode replace; with res_mode ms
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
HERE = os.path.dirname(os.path.abspath(__file__))
from spikingjelly.clock_driven import functional
from load_model import LOAD_MODEL
from forecast_wrap import Forecaster
from utils import make_adamw

N_OK, N_FAIL = 0, 0


def ok(cond, msg):
    global N_OK, N_FAIL
    print(('PASS ' if cond else 'FAIL ') + msg)
    if cond:
        N_OK += 1
    else:
        N_FAIL += 1


def find(path, **kv):
    hits = [ln for ln in open(path) if ln.startswith('RESULT') and all(f' {k}={v} ' in ln for k, v in kv.items())]
    if len(hits) != 1:
        raise RuntimeError(f'{len(hits)} RESULT lines for {kv} in {path}')
    return dict(t.split('=', 1) for t in hits[0].split()[1:] if '=' in t)['dir']


def run_dir(task, cond):
    if task == 'ett':
        return find(os.path.join(HERE, '..', 'forecasting', 'results', 'flif-v1-ett-r1', 'raw.txt'), cond=cond, ds='ETTh2', pl=96, seed=42)
    return find(os.path.join(HERE, 'results', 'flif-v1-mitbih-r1', 'raw.txt'), cond=cond, seed=42)


def config(task, cond):
    c = torch.load(os.path.join(run_dir(task, cond), 'model_state', 'config.pt'), map_location='cpu')
    return argparse.Namespace(**dict(c, device='cuda'))


def trained(task, A, cond, **over):
    """a variant with the trained base weights (ETT checkpoint.pth = best validation; MIT-BIH best+model.pt = last epoch); returns
    (model, missing keys, unexpected keys)."""
    m = build(task, A, **over)
    f = 'checkpoint.pth' if task == 'ett' else 'best+model.pt'
    r = m.load_state_dict(torch.load(os.path.join(run_dir(task, cond), 'model_state', f), map_location='cpu'), strict=False)
    return m, sorted(strip(k) for k in r.missing_keys), sorted(strip(k) for k in r.unexpected_keys)


def build(task, a, **over):
    b = argparse.Namespace(**{**vars(a), **over})
    torch.manual_seed(0)
    inner = LOAD_MODEL[b.model](b, True)
    m = Forecaster(inner, b.embed_dim, b.pred_len, bias=b.bias) if task == 'ett' else inner
    return m.cuda()


def inner_of(m):
    return m.inner if isinstance(m, Forecaster) else m


def data(task, a):
    if task == 'ett':
        x = torch.randn(4, a.seq_len, 7, generator=torch.Generator().manual_seed(5)).cuda()
        y = torch.randn(4, a.pred_len, 7, generator=torch.Generator().manual_seed(6)).cuda()
        return x, y
    x = torch.rand(4, 1, 187, generator=torch.Generator().manual_seed(7)).cuda()
    return x, torch.tensor([0, 1, 2, 4]).cuda()


def train_step(task, m, x, y):
    m.train(); m.train_mode = 'training'; functional.reset_net(m)
    m.zero_grad(set_to_none=True)
    if task == 'ett':
        o = m(x); loss = (o - y).abs().mean()
    else:
        o = m(x)[0]; loss = torch.nn.functional.cross_entropy(o, y)
    loss.backward()
    functional.reset_net(m)
    return o.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


def strip(n):
    return n[len('inner.'):] if n.startswith('inner.') else n


def is_attn(n):
    parts = strip(n).split('.')
    return len(parts) > 2 and parts[0] == 'data_block' and parts[2] == 'ssa'


def is_neo(n):
    return strip(n).startswith(('encoding_neo.', 'time_block.'))


def capture_hippo(m, x, perturb_j=None, delta=10.0):
    """eval-mode forward; returns the hippocampus output [T_h, B', N, D] (wrapping _tokens_mc / hippo_blocks on the instance)."""
    inn = inner_of(m)
    tok, hb, got = inn._tokens_mc, inn.hippo_blocks, {}

    def t(x):
        p, q = tok(x)
        if perturb_j is not None:
            p = p.clone(); q = q.clone()
            p[:, :, perturb_j] += delta
            q[perturb_j] += delta
        return p, q

    def h(xh, xn):
        out = hb(xh, xn); got['h'] = out.detach().clone(); return out
    inn._tokens_mc, inn.hippo_blocks = t, h
    try:
        m.eval(); m.train_mode = 'testing'
        with torch.no_grad():
            functional.reset_net(m); m(x); functional.reset_net(m)
    finally:
        del inn._tokens_mc, inn.hippo_blocks
    return got['h']


for task in ('ett', 'mb'):
    for n in ('lif', 'flif'):
        cond = f'nsmt4q_pop_{n}_msn_sde'
        tag = f'{task} pop ' + ('f-LIF' if n == 'flif' else 'LIF')
        A = config(task, cond)
        X = data(task, A)
        base = build(task, A)
        D = int(A.embed_dim)

        # n1
        v = build(task, A, neo_current=1)
        inner_of(v).neo_alpha_c.data.zero_()
        sb, sv = base.state_dict(), v.state_dict()
        extra = sorted(set(sv) - set(sb))
        ok(extra in (['neo_alpha_c'], ['inner.neo_alpha_c']) and not (set(sb) - set(sv))
           and all(torch.equal(sb[k], sv[k]) for k in sb), f'n1 {tag}: (가) initial state = base + neo_alpha_c ({extra})')
        ob, gb = train_step(task, base, *X)
        ov, gv = train_step(task, v, *X)
        shared = [k for k in gb if k in gv]
        ok(torch.equal(ob, ov) and len(shared) == len(gb) and all(torch.equal(gb[k], gv[k]) for k in shared),
           f'n1 {tag}: (가) alpha 0 = base, train output and {len(shared)} gradients bit-identical')

        # n2
        v = build(task, A, neo_current=1)
        alpha = inner_of(v).neo_alpha_c
        ok(alpha.shape == (D,) and bool((alpha == 1).all()), f'n2 {tag}: alpha init 1, shape [{D}]')
        cur = {}
        for key, mm in (('b', base), ('v', v)):
            inn = inner_of(mm)
            h1 = inn.encoding.emb_lif.register_forward_pre_hook(lambda mod, inp, key=key: cur.__setitem__(key, inp[0].detach().clone()))
            nf = inn.neo_forward

            def wrap(*args, key=key, nf=nf, **kw):
                out = nf(*args, **kw); cur['neo' + key] = out[1].detach().clone(); return out
            inn.neo_forward = wrap
            train_step(task, mm, *X)
            h1.remove(); del inn.neo_forward
        mx = cur['neov']
        add = mx.permute(2, 1, 3, 0)                     # [1, B', D, N]
        ok(torch.equal(cur['neob'], cur['neov']), f'n2 {tag}: the Neocortex output is the same in base and (가)')
        ok(bool(((mx == 0) | (mx == 1)).all()), f'n2 {tag}: the Neocortex output is binary (rate {mx.mean().item():.4f})')
        ok(cur['v'].shape[0] == int(A.hippo_T) and torch.equal(cur['v'], cur['b'] + add),
           f'n2 {tag}: embedding neuron input = base input + Neocortex spikes, every one of the {cur["v"].shape[0]} T_h steps')
        ok(bool((cur['v'] != cur['b']).any()), f'n2 {tag}: the added current is not empty ({int((cur["v"] != cur["b"]).sum())} entries)')
        v5 = build(task, A, neo_current=1, neo_current_init=0.5)
        ok(bool((inner_of(v5).neo_alpha_c == 0.5).all()), f'n2 {tag}: --neo_current_init 0.5 gives alpha 0.5')

        # n3
        w = build(task, A, neo_current=1, hippo_attn=0)
        wi = inner_of(w)
        left = [nm for nm, _ in wi.data_block.named_modules() if '.ssa' in nm or nm.startswith('ssa')]
        ok(not left and all(getattr(b, 'no_attn', False) for b in wi.data_block) and len(wi.data_block) >= 1,
           f'n3 {tag}: (나) no attention module in {len(wi.data_block)} hippocampus block(s)')
        sw = w.state_dict()
        sb0 = build(task, A).state_dict()                # a fresh base (sb holds the live tensors of the trained-one-step base)
        attn_keys = [k for k in sb0 if is_attn(k)]
        ok(sorted(set(sb0) - set(sw)) == sorted(attn_keys) and len(attn_keys) > 0 and sorted(set(sw) - set(sb0)) == sorted(extra)
           and all(torch.equal(sb0[k], sw[k]) for k in sw if k in sb0), f'n3 {tag}: (나) state = base - {len(attn_keys)} attention tensors + alpha; the rest equal')
        pb = sum(p.numel() for p in base.parameters()); pw = sum(p.numel() for p in w.parameters())
        pa = sum(p.numel() for nm, p in base.named_parameters() if is_attn(nm))
        ok(pw == pb - pa + D, f'n3 {tag}: parameters {pw} = {pb} - {pa} (attention) + {D}')
        rec = []
        hooks = []
        for b in wi.data_block:
            st = {}
            hooks.append(b.mlp.register_forward_hook(lambda mod, inp, out, st=st: st.__setitem__('m', out.detach().clone())))
            hooks.append(b.register_forward_hook(lambda mod, inp, out, st=st: rec.append((inp[0].detach().clone(), st['m'], out.detach().clone()))))
        train_step(task, w, *X)
        for hk in hooks:
            hk.remove()
        ok(len(rec) == len(wi.data_block) and all(torch.equal(o, xx * (1. - mo)) for xx, mo, o in rec),
           f'n3 {tag}: every (나) block output = x (1 - MLP(x)) exactly ({len(rec)} block calls)')

        # n4
        N = int(inner_of(base).num_patches); j = N // 2
        g4 = torch.Generator().manual_seed(11)
        x4 = (torch.randn(32, A.seq_len, 7, generator=g4) if task == 'ett' else torch.rand(32, 1, 187, generator=g4)).cuda()
        attn_short = sorted(strip(k) for k in attn_keys)
        for lab, over, expect, miss, unexp in (('base', {}, 'mix', [], []),
                                               ('(가)', dict(neo_current=1), 'mix', ['neo_alpha_c'], []),
                                               ('(나)', dict(neo_current=1, hippo_attn=0), 'causal', ['neo_alpha_c'], attn_short),
                                               ('(나) memoryless', dict(neo_current=1, hippo_attn=0, neo_memoryless=True), 'alone', ['neo_alpha_c'], attn_short)):
            mm, mk, uk = trained(task, A, cond, **over)
            ok(mk == miss and uk == unexp, f'n4 {tag}: {lab}: trained base weights load (missing {mk}, unused {len(uk)} attention tensors)')
            r0, r1 = capture_hippo(mm, x4), capture_hippo(mm, x4, perturb_j=j)
            cnt = [int((r0[:, :, i] != r1[:, :, i]).sum()) for i in range(N)]
            ch = [c > 0 for c in cnt]
            before, after = any(ch[:j]), any(ch[j + 1:])
            if expect == 'mix':
                good = before and ch[j]
            elif expect == 'causal':
                good = (not before) and ch[j] and after
            else:
                good = ch[j] and not before and not after
            ok(good, f'n4 {tag}: {lab}: token {j} of {N} changed -> changed entries before {sum(cnt[:j])}, at j {cnt[j]}, '
                     f'after {sum(cnt[j + 1:])} (expect {expect})')

        # n5
        for lab, mm in (('(가)', build(task, A, neo_current=1)), ('(나)', w)):
            _, g = train_step(task, mm, *X)
            ga = [t for k, t in g.items() if strip(k) == 'neo_alpha_c']
            gn = [t for k, t in g.items() if is_neo(k)]
            ok(len(ga) == 1 and bool((ga[0] != 0).any()) and len(gn) > 0 and any(bool((t != 0).any()) for t in gn),
               f'n5 {tag}: {lab}: nonzero gradient on alpha and on {sum(bool((t != 0).any()) for t in gn)}/{len(gn)} Neocortex tensors')

        # n6
        opt = make_adamw(w, 5e-4, 0.06, True)
        nd = [g for g in opt.param_groups if g['weight_decay'] == 0.]
        ok(len(nd) == 1 and any(p is inner_of(w).neo_alpha_c for p in nd[0]['params']), f'n6 {tag}: alpha is in the weight-decay-0 group')

        # n7
        if task == 'ett':
            for lab, mm in (('(가)', build(task, A, neo_current=1)), ('(나)', w)):
                inn = inner_of(mm)
                B, L, C = X[0].shape
                z = X[0].permute(0, 2, 1).reshape(B * C, 1, L)
                mm.eval(); mm.train_mode = 'testing'
                got = {}
                hb = inn.hippo_blocks

                def h(xh, xn, hb=hb):
                    out = hb(xh, xn); got['h'] = out.detach().clone(); return out
                with torch.no_grad():
                    functional.reset_net(mm); s1 = mm.step_features(z).detach().clone(); functional.reset_net(mm)
                    inn.hippo_blocks = h
                    try:
                        inn(z)
                    finally:
                        del inn.hippo_blocks
                    functional.reset_net(mm)
                ok(torch.equal(s1, got['h']), f'n7 {tag}: {lab}: Forecaster.step_features = myModel.forward hippocampus output')

    # n8 (once per task)
    A = config(task, 'nsmt4q_pop_lif_msn_sde')
    for lab, over in (('hippo_attn 0 without neo_current', dict(hippo_attn=0)),
                      ('neo_current with hippo_xattn_mode replace', dict(neo_current=1, hippo_xattn_mode='replace', neuron_all='sj_lif')),
                      ('neo_current with res_mode ms', dict(neo_current=1, res_mode='ms', neuron_all='sj_lif'))):
        try:
            build(task, A, **over)
            ok(False, f'n8 {task}: {lab} was accepted')
        except (ValueError, NotImplementedError) as e:
            ok('neo_current' in str(e) or 'hippo_attn' in str(e), f'n8 {task}: {lab} stops: {str(e)[:90]}')

print(f'{N_OK} passed, {N_FAIL} failed')
sys.exit(1 if N_FAIL else 0)
