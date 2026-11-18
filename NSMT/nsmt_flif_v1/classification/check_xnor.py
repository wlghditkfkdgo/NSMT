"""MODI 2026-11-18: fail-closed checks of the XNOR + Log-PE score (--attn_score xnor_log, layers.SSA_rel_scl score='xnor_log';
Lv et al. NeurIPS 2025, official code microsoft/SeqSNN spikformer_xnor_log.py) for the coupling attention test
(docs/MODI_PROJECT_LOG.md 2026-11-18). x0 (default path bit-identical) is fw_identity_xnor.py.
  x1 Log-PE buffer: R_ij == max(0, ceil(log2((N-1)/(|i-j|+1)))) checked with exact rational arithmetic, symmetric, a function of
     |i-j| only; ranges for N = 24 / 84 / 180 (inputs 96 / 336 / 720) printed
  x2 self-attention module: the input current of the attention neuron == brute force sum_j [(#channels with q_i == k_j) + R_ij]
     * sigmoid(s) * v_j per head (q, k, v captured from their neurons), max |diff| <= 1e-5
  x3 the same in query mode (q_from_kv): q from kv (the Neocortex input), k / v from x
  x4 module setup: attention neuron threshold 0.5 (xnor_log) / 1.0 (dot), scale init -4, no learned bias table; the parameter
     names differ from the dot module only by +xl_scale / -relative_bias_table
  x5 gradients reach xl_scale and the q / k / v Linear weights and are finite
  x6 full ETT models (spk_ssa and the msn base, pop LIF / pop f-LIF, ETTh2 and ETTm2 H96 configs): with the same seed every
     shared parameter / buffer has the dot model's init; the only differences are xl_scale, log_pe vs relative_bias_table,
     rel_pos_idx; q / k / v identical, the attention-neuron input differs; the init attention firing of both is printed (at init
     both attention neurons can be silent, then the outputs are equal); with sigmoid(s) = 0.5 the xnor attention neuron fires and
     the output moves
  x7 batch invariance (eval, fresh xnor model and a randomly perturbed one): 4 windows at once == each window alone, max |diff| <= 1e-6
  x8 refusals: xnor_log with kv_cat, attn_score xnor_log without hippo_ssa / hippo_xattn
any failed comparison raises (exit code 1).
usage (classification folder): CUDA_VISIBLE_DEVICES=0 python check_xnor.py
"""
import os
import sys
from fractions import Fraction
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional                        # noqa: E402
from layers import SSA_rel_scl                                           # noqa: E402
from load_model import LOAD_MODEL                                        # noqa: E402
from forecast_wrap import Forecaster                                     # noqa: E402

FAIL = []
FC = os.path.join(os.getcwd(), '..', 'forecasting', 'results')


def ok(cond, what):
    print(('pass ' if cond else 'FAIL ') + what)
    if not cond:
        FAIL.append(what)


def logpe_exact(N):
    R = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            q = Fraction(N - 1, abs(i - j) + 1)
            r = 0
            while Fraction(2) ** r < q:
                r += 1
            R[i, j] = r                                  # ceil(log2(q)) for q > 1, 0 for q <= 1 (clip at 0)
    return R


def x1():
    for N in (24, 84, 180):
        m = SSA_rel_scl(dim=64, seq_len=N, num_heads=8, score='xnor_log')
        R = m.log_pe
        toep = all(torch.equal(R.diagonal(o), R.diagonal(o)[:1].expand(N - abs(o))) for o in range(-(N - 1), N))
        ok(torch.equal(R, logpe_exact(N)) and torch.equal(R, R.t()) and toep,
           f'x1 Log-PE N={N}: exact formula, symmetric, depends on |i-j| only; range {int(R.min())}..{int(R.max())}, '
           f'distance 0 / 1 / 5 / 11 -> {[int(R[0, d]) for d in (0, 1, 5, 11)]}')


def capture(m):
    cap = {}
    hs = [m.q_lif.register_forward_hook(lambda mod, i, o: cap.__setitem__('q', o.detach())),
          m.k_lif.register_forward_hook(lambda mod, i, o: cap.__setitem__('k', o.detach())),
          m.v_lif.register_forward_hook(lambda mod, i, o: cap.__setitem__('v', o.detach())),
          m.attn_lif.register_forward_pre_hook(lambda mod, i: cap.__setitem__('a_in', i[0].detach()))]
    return cap, hs


def brute(cap, H, R, s):
    T, B, N, C = cap['q'].shape
    d = C // H
    q = cap['q'].reshape(T, B, N, H, d).permute(0, 1, 3, 2, 4)
    k = cap['k'].reshape(T, B, N, H, d).permute(0, 1, 3, 2, 4)
    v = cap['v'].reshape(T, B, N, H, d).permute(0, 1, 3, 2, 4)
    match = (q.unsqueeze(-2) == k.unsqueeze(-3)).float().sum(-1)          # [T, B, H, N, N]: channels where q_i == k_j
    out = ((match + R) * torch.sigmoid(s)) @ v
    return out.transpose(2, 3).reshape(T, B, N, C), q


def x2_x3():
    torch.manual_seed(1)
    for mode in ('self', 'query'):
        for H in (8, 16):
            m = SSA_rel_scl(dim=64, seq_len=24, num_heads=H, score='xnor_log', q_from_kv=(mode == 'query')).cuda().train()
            with torch.no_grad():
                m.xl_scale.fill_(-1.3)
            cap, hs = capture(m)
            x = (torch.rand(4, 3, 24, 64, device='cuda') < 0.3).float()
            kv = (torch.rand(4, 3, 24, 64, device='cuda') < 0.2).float() if mode == 'query' else None
            src_q = kv if mode == 'query' else x
            m(x, kv=kv)
            functional.reset_net(m)
            exp, q = brute(cap, H, m.log_pe, m.xl_scale.detach())
            # q really comes from the intended source: rerun the q branch by hand
            qq = m.q_lif(m.q_bn(m.q_linear(src_q.flatten(0, 1)).transpose(-1, -2)).transpose(-1, -2).reshape(4, 3, 24, 64).contiguous())
            functional.reset_net(m)
            diff = float((exp - cap['a_in']).abs().max())
            spk = all(bool(((cap[t] == 0) | (cap[t] == 1)).all()) for t in 'qkv')
            ok(diff <= 1e-5 and spk and torch.equal(qq, cap['q']) and float(cap['q'].mean()) > 0 and float(cap['k'].mean()) > 0,
               f'x{2 if mode == "self" else 3} {mode} H={H}: attention-neuron input == brute force (max |diff| {diff:.2e}); q from '
               f'{"kv" if mode == "query" else "x"}; q / k / v 0/1, rates {float(cap["q"].mean()):.3f} / {float(cap["k"].mean()):.3f} / {float(cap["v"].mean()):.3f}')
            for h in hs:
                h.remove()


def x4_x5():
    a = SSA_rel_scl(dim=64, seq_len=24, num_heads=8, score='dot')
    b = SSA_rel_scl(dim=64, seq_len=24, num_heads=8, score='xnor_log')
    pa, pb = set(n for n, _ in a.named_parameters()), set(n for n, _ in b.named_parameters())
    ok(a.attn_lif.v_threshold == 1.0 and b.attn_lif.v_threshold == 0.5 and float(b.xl_scale) == -4.0
       and not hasattr(b, 'relative_bias_table') and pb - pa == {'xl_scale'} and pa - pb == {'relative_bias_table'},
       f'x4 thresholds dot {a.attn_lif.v_threshold} / xnor {b.attn_lif.v_threshold}; scale init {float(b.xl_scale)}; '
       f'params added {sorted(pb - pa)}, removed {sorted(pa - pb)}')
    torch.manual_seed(2)
    b = b.cuda().train()
    with torch.no_grad():
        b.xl_scale.fill_(0.)                             # sigmoid 0.5: the attention neuron fires, so gradients flow
    x = (torch.rand(4, 3, 24, 64, device='cuda') < 0.3).float()
    b(x).sum().backward()
    functional.reset_net(b)
    g = {n: p.grad for n, p in b.named_parameters()}
    need = ('xl_scale', 'q_linear.weight', 'k_linear.weight', 'v_linear.weight')
    ok(all(g[n] is not None and bool(torch.isfinite(g[n]).all()) and float(g[n].abs().sum()) > 0 for n in need),
       f'x5 gradients non-zero and finite: {[(n, float(g[n].abs().sum()) if g[n] is not None else None) for n in need]}')


def find(suite, cond, ds, pl, seed):
    for ln in open(os.path.join(FC, suite, 'raw.txt')):
        if ln.startswith('RESULT') and f' cond={cond} ' in ln and f' ds={ds} ' in ln and f' pl={pl} ' in ln and f' seed={seed} ' in ln:
            return dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)['dir']
    raise RuntimeError((suite, cond, ds, pl, seed))


def build(run, ov, seed=0):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda', **ov))
    torch.manual_seed(seed)
    return a, Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, a.pred_len, bias=a.bias).cuda()


def x6_x7():
    runs = [('flif-v1-gen-ett', 'spk_ssa', 'ETTh2'), ('flif-v1-gen-ett', 'spk_ssa', 'ETTm2'),
            ('flif-v1-ett-ms', 'nsmt4q_pop_lif_msn', 'ETTh2'), ('flif-v1-ett-ms', 'nsmt4q_pop_flif_msn', 'ETTm2')]
    for suite, cond, ds in runs:
        run = find(suite, cond, ds, 96, 42)
        a, d = build(run, {})
        _, x = build(run, dict(attn_score='xnor_log'))
        sd, sx = d.state_dict(), x.state_dict()
        only_d, only_x = sorted(set(sd) - set(sx)), sorted(set(sx) - set(sd))
        same = all(torch.equal(sd[k], sx[k]) for k in set(sd) & set(sx))
        exp_d = all(k.endswith(('relative_bias_table', 'rel_pos_idx')) for k in only_d) and len(only_d) == 2
        exp_x = all(k.endswith(('xl_scale', 'log_pe')) for k in only_x) and len(only_x) == 2
        inp = torch.randn(4, a.seq_len, 7, generator=torch.Generator().manual_seed(5)).cuda()
        d.train(); x.train(); d.train_mode = x.train_mode = 'training'   # batch statistics: a fresh model's v fires only in training mode
        cd, hd = capture(d.inner.data_block[0].ssa)
        cx, hx = capture(x.inner.data_block[0].ssa)
        ad, ax = [], []
        h2 = [d.inner.data_block[0].ssa.attn_lif.register_forward_hook(lambda mod, i, o: ad.append(float(o.mean()))),
              x.inner.data_block[0].ssa.attn_lif.register_forward_hook(lambda mod, i, o: ax.append(float(o.mean())))]
        with torch.no_grad():
            torch.manual_seed(7); functional.reset_net(d); od = d(inp); functional.reset_net(d)
            torch.manual_seed(7); functional.reset_net(x); ox = x(inp); functional.reset_net(x)
            in_differ = not torch.equal(cd['a_in'], cx['a_in']); mx_d, mx_x = float(cd['a_in'].max()), float(cx['a_in'].max())
            for mod in x.modules():                      # sigmoid(0) = 0.5: the xnor attention neuron fires -> the output must move
                if hasattr(mod, 'xl_scale'):
                    mod.xl_scale.fill_(0.)
            ax_hot = []
            hh = x.inner.data_block[0].ssa.attn_lif.register_forward_hook(lambda mod, i, o: ax_hot.append(float(o.mean())))
            torch.manual_seed(7); functional.reset_net(x); ox_hot = x(inp); functional.reset_net(x); hh.remove()
            for mod in x.modules():
                if hasattr(mod, 'xl_scale'):
                    mod.xl_scale.fill_(-4.)
        x.eval(); x.train_mode = 'testing'
        for h in hd + hx + h2:
            h.remove()
        same_qkv = all(torch.equal(cd[t], cx[t]) for t in 'qkv')
        moved = not torch.equal(od, ox) if (ad[0] > 0 or ax[0] > 0) else not torch.equal(od, ox_hot)
        ok(same and exp_d and exp_x and same_qkv and in_differ and moved and ax_hot[0] > 0,
           f'x6 {cond} {ds}: shared init identical {same}; dot-only {only_d}; xnor-only {only_x}; same q/k/v {same_qkv}; '
           f'attention-neuron input differs {in_differ}; init attention firing dot {ad[0]:.4f} / xnor {ax[0]:.4f} '
           f'(max input dot {mx_d:.3f} / xnor {mx_x:.3f}); with sigmoid(s) = 0.5 xnor fires {ax_hot[0]:.4f} and the output moves {moved}')
        for tag in ('fresh', 'perturbed'):
            if tag == 'perturbed':
                torch.manual_seed(3)
                with torch.no_grad():
                    for p in x.parameters():
                        p.add_(0.05 * torch.randn_like(p))
                    for mod in x.modules():
                        if hasattr(mod, 'xl_scale'):
                            mod.xl_scale.fill_(0.)
            with torch.no_grad():
                functional.reset_net(x); full = x(inp); functional.reset_net(x)
                rows = []
                for i in range(4):
                    rows.append(x(inp[i:i + 1])); functional.reset_net(x)
                perm = torch.tensor([2, 0, 3, 1])
                pm = x(inp[perm]); functional.reset_net(x)
            dmax = max(float((full - torch.cat(rows)).abs().max()), float((full[perm] - pm).abs().max()))
            ok(dmax <= 1e-6 and bool(torch.isfinite(full).all()),
               f'x7 {cond} {ds} {tag}: batch 4 == batch 1 and shuffled batch (max |diff| {dmax:.2e})')


def x8():
    refused = []
    try:
        SSA_rel_scl(dim=8, seq_len=4, num_heads=2, kv_cat=True, score='xnor_log'); refused.append(False)
    except ValueError:
        refused.append(True)
    run = find('flif-v1-gen-ett', 'spk_ssa', 'ETTh2', 96, 42)
    try:
        build(run, dict(hippo_ssa=False, attn_score='xnor_log')); refused.append(False)
    except ValueError:
        refused.append(True)
    ok(all(refused), f'x8 refusals (kv_cat, no hippo_ssa / hippo_xattn): {refused}')


if __name__ == '__main__':
    x1(); x2_x3(); x4_x5(); x6_x7(); x8()
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED')
    sys.exit(1 if FAIL else 0)
