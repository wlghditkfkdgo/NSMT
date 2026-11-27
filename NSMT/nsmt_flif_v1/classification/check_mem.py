"""MODI 2026-11-27: checks for --neo_readout mem (Neocortex soma membrane to the queries, q_linear once) and --aux_past rand (past
reconstruction aux) on the Sleep-EDF query form. Fail closed.
usage (classification folder): python check_mem.py
"""
import sys

import torch
from spikingjelly.clock_driven import functional

from config import Config, parse_arguments
from load_model import LOAD_MODEL

FAILS = []
BASE = ("--data SLEEPEDF --seed 42 --log_dir /tmp/claude-1001/check_mem_cfg --gating attn -s -nd 0 -emb 384 -nh 8 --mlp_ratios 1 --layers 1 "
        "--time_layers 2 --keep_ratio 0.15 --alpha 0 --tau 2.0 --no-bias --patch_size 1 --token_step 1 --head_pool last --ssa_pe 0 --fold 0 "
        "--ctx_epochs 16 --model myModel --neo_where both --hippo_T 4 --hippo_xattn --neo_grad 1.0 --neo_res ms --neuron_all lif "
        "--neo_neuron pop_lif --pop_soma_init 0.5 --hippo_xattn_mode query")


def ok(name, cond, info=''):
    print(f'{"PASS" if cond else "FAIL"} {name} {info}', flush=True)
    if not cond:
        FAILS.append(name)


def build(extra=''):
    sys.argv = ['x'] + (BASE + ' ' + extra).split()
    a = Config(); a.set_args(parse_arguments()); a.num_classes, a.num_channels, a.seq_len = 5, 60, 16
    torch.manual_seed(0)
    return LOAD_MODEL['myModel'](a, True).to(a.device), a.device


def run(m, x, mode):
    m.train(mode == 'training'); m.train_mode = mode; functional.reset_net(m)
    with torch.no_grad():
        o = m(x)
    functional.reset_net(m)
    return o


def main():
    torch.manual_seed(1)
    x = torch.randn(8, 60, 16).cuda()
    m0, dev = build()
    mm, _ = build('--neo_readout mem')
    ma, _ = build('--aux_past rand --alpha 0.1')
    mma, _ = build('--neo_readout mem --aux_past rand --alpha 0.1')
    # a. initialisation: the new options add no RNG draw for the shared parameters
    s0 = m0.state_dict()
    ok('a1 neo_readout mem: same parameters as the default model (none added)', s0.keys() == mm.state_dict().keys()
       and all(torch.equal(s0[k], mm.state_dict()[k]) for k in s0))
    sa = ma.state_dict()
    extra = sorted(set(sa) - set(s0))
    ok('a2 aux_past: only the decoder is added, every other parameter equal', all(k.startswith('aux_past_dec') for k in extra) and extra
       and all(torch.equal(s0[k], sa[k]) for k in s0), str(extra))
    # b. pre-reset soma membrane: spikes = membrane >= threshold; after-reset = membrane - threshold * spike
    lay = mm.time_block.Mem[-1]
    nrn = lay.lif.neuron
    cur = torch.randn(16, 5, 384).cuda() * 2
    with torch.no_grad():
        s, aux = nrn(cur, analog='pre'); _, aux2 = nrn(cur, analog='analog')
    pre, post = aux['analog'], aux2['analog']
    th = nrn.soma.threshold
    ok('b1 spike = 1 exactly where the pre-reset membrane >= threshold', torch.equal(s.bool(), pre >= th), f'spike rate {s.mean().item():.3f}')
    ok('b2 after-reset membrane = pre - threshold * spike', torch.allclose(post, pre - th * s, atol=1e-6))
    # c. readout mem: the Neocortex output is the last layer's membrane (real-valued), spikes elsewhere unchanged
    # MODI 2026-11-27 (correction): training mode; in eval mode the last layer is silent at init and c2 compared nothing (rate 0)
    mm.train(); mm.train_mode = 'training'; functional.reset_net(mm)
    with torch.no_grad():
        patched, neo_in = mm._tokens_mc(x)
        _, xn = mm.neo_forward(neo_in)
        mm.neo_readout = 'spike'; functional.reset_net(mm); _, xs = mm.neo_forward(neo_in); mm.neo_readout = 'mem'
    nonbin = ((xn != 0) & (xn != 1)).float().mean().item()
    ok('c1 mem readout: real-valued output [N, B, 1, D]', tuple(xn.shape) == (16, 8, 1, 384) and nonbin > 0.5, f'non-binary share {nonbin:.3f}')
    ok('c2 mem readout spikes (>= threshold) = the spike readout (and the layer fires)', torch.equal((xn >= th).float(), xs) and xs.mean().item() > 0.01,
       f'spike rate {xs.mean().item():.3f}')
    # d. q_once gives the same queries as computing q_linear at every T_h step
    for name, mdl in (('spike model', m0), ('mem model', mm)):
        outs = []
        for flag in (False, True):
            for blk in mdl.data_block:
                blk.ssa.q_once = flag
            outs.append(run(mdl, x, 'testing')[0] if isinstance(run(mdl, x, 'testing'), tuple) else run(mdl, x, 'testing'))
        ok(f'd1 q_linear once = per step ({name}, eval)', torch.allclose(outs[0], outs[1], atol=1e-5), f'max diff {(outs[0] - outs[1]).abs().max().item():.2e}')
    for blk in m0.data_block:
        blk.ssa.q_once = False
    ok('d2 q_once is set only by neo_readout mem', all(blk.ssa.q_once for blk in mm.data_block) and not any(getattr(b.ssa, 'q_once', False) for b in ma.data_block))
    # e. past aux: anchors t = 1..N-1, lag d in 1..t, target = raw token t - d; decoder unused at test time
    ma.train(); ma.train_mode = 'training'; functional.reset_net(ma)
    out = ma(x)
    t, d, src = ma._aux_past_last
    tgt, rec = out[3], out[4]
    raw = ma._neo_in[:, :, 0]
    okd = bool(((d >= 1) & (d <= t)).all()) and torch.equal(src, t - d)
    okt = all(torch.equal(tgt[i, b], raw[src[i, b], b]) for i in range(15) for b in range(8))
    ok('e1 lag d in 1..t, target = the raw token t - d', okd and okt, f'lags used {sorted(set(d.flatten().tolist()))[:6]}...')
    ok('e2 reconstruction [N-1, B, tok_dim], finite loss', tuple(rec.shape) == (15, 8, 60) and bool(torch.isfinite(((tgt - rec) ** 2).mean())))
    calls = []
    h = ma.aux_past_dec.register_forward_hook(lambda *_: calls.append(1))
    run(ma, x, 'testing'); h.remove()
    ok('e3 decoder not called at test time (no inference energy)', not calls)
    out2 = mma.train() or None
    mma.train_mode = 'training'; functional.reset_net(mma); o2 = mma(x)
    ok('e4 mem + aux: the aux reads the membrane output', ((o2[2] != 0) & (o2[2] != 1)).float().mean().item() > 0.5)
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
