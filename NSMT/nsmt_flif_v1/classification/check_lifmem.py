"""MODI 2026-11-27 (user decision 11-27: "(b)" + the Sleep-EDF changes re-run with a standard LIF neuron): checks for --neo_readout mem
and --aux_past rand with a plain spikeDE LIF / f-LIF Neocortex (neurons.py analog 'pre' without a soma = the membrane the spike is decided
on, look = v_t + (I_t - v_t) / tau). The population-neuron checks stay in check_mem.py. Fail closed.
usage (classification folder): python check_lifmem.py
"""
import sys

import torch
from spikingjelly.clock_driven import functional

from config import Config, parse_arguments
from load_model import LOAD_MODEL
from neurons import FLIFNeuron, LIFNeuron, PopulationFLIF

FAILS = []
BASE = ("--data SLEEPEDF --seed 42 --log_dir /tmp/claude-1001/check_lifmem_cfg --gating attn -s -nd 0 -emb 384 -nh 8 --mlp_ratios 1 --layers 1 "
        "--time_layers 2 --keep_ratio 0.15 --alpha 0 --tau 2.0 --no-bias --patch_size 1 --token_step 1 --head_pool last --ssa_pe 0 --fold 0 "
        "--ctx_epochs 16 --model myModel --neo_where both --hippo_T 4 --hippo_xattn --neo_grad 1.0 --neo_res ms --neuron_all lif "
        "--hippo_xattn_mode query")
NRN = {'lif': '--neo_neuron lif', 'flif': '--neo_neuron flif --neo_alpha 0.5'}


def ok(name, cond, info=''):
    print(f'{"PASS" if cond else "FAIL"} {name} {info}', flush=True)
    if not cond:
        FAILS.append(name)


def build(extra=''):
    sys.argv = ['x'] + (BASE + ' ' + extra).split()
    a = Config(); a.set_args(parse_arguments()); a.num_classes, a.num_channels, a.seq_len = 5, 60, 16
    torch.manual_seed(0)
    return LOAD_MODEL['myModel'](a, True).cuda()


def run(m, x, mode):
    m.train(mode == 'training'); m.train_mode = mode; functional.reset_net(m)
    with torch.no_grad():
        o = m(x)
    functional.reset_net(m)
    return o[0] if isinstance(o, tuple) else o


def main():
    torch.manual_seed(1)
    x = torch.randn(8, 60, 16).cuda()
    y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]).cuda()
    # a. neuron level: the pre-reset membrane of a single self-firing neuron
    cur = (torch.randn(16, 5, 32) * 2 + 0.8).cuda()
    for name, nrn in (('LIF', LIFNeuron(32, tau=2., max_length=16)), ('f-LIF', FLIFNeuron(32, alpha=.5, tau=2., max_length=16))):
        nrn = nrn.cuda()
        with torch.no_grad():
            s, aux = nrn(cur, analog='pre', return_aux=True)
            s0 = nrn(cur)
        pre, st = aux['analog'], aux['state'][..., 0]                       # state = v_{t+1} after the step (and reset)
        prev = torch.cat([torch.zeros_like(st[:1]), st[:-1]])               # v_t
        ok(f'a1 {name}: spikes unchanged by the readout, spike = 1 exactly where pre >= threshold', torch.equal(s, s0)
           and torch.equal(s.bool(), pre >= nrn.threshold), f'spike rate {s.mean().item():.3f}')
        ok(f'a2 {name}: pre_t = v_t + (I_t - v_t) / tau', torch.allclose(pre, prev + (cur - prev) / 2., atol=1e-5),
           f'max diff {(pre - prev - (cur - prev) / 2.).abs().max().item():.1e}')
        if name == 'LIF':
            ok('a3 LIF (spikeDE reset, h = 1): v_{t+1} = pre_t - theta s_t / tau', torch.allclose(st, pre - s / 2., atol=1e-5))
        xg = cur.clone().requires_grad_(True)
        _, ag = nrn(xg, analog='pre'); ag['analog'].sum().backward()
        ok(f'a4 {name}: the readout carries a gradient to the input current', bool(torch.isfinite(xg.grad).all()) and xg.grad.abs().sum().item() > 0)
    refused = 0
    for bad in (lambda: LIFNeuron(32, max_length=16).cuda()(cur, analog='analog'),
                lambda: PopulationFLIF(32, soma='none', max_length=16).cuda()(cur, analog='pre')):
        try:
            bad()
        except ValueError:
            refused += 1
    ok('a5 still refused: analog (after reset) on a soma-less LIF, pre on a soma-less K = 4 population', refused == 2)
    for n, flag in NRN.items():
        m0, mm = build(flag), build(flag + ' --neo_readout mem')
        ma, mma = build(flag + ' --aux_past rand --alpha 0.1'), build(flag + ' --neo_readout mem --aux_past rand --alpha 0.1')
        s0 = m0.state_dict()
        ok(f'b1 {n}: neo_readout mem adds no parameter, every initial value equal', s0.keys() == mm.state_dict().keys()
           and all(torch.equal(s0[k], mm.state_dict()[k]) for k in s0))
        sa = ma.state_dict(); extra = sorted(set(sa) - set(s0))
        ok(f'b2 {n}: aux_past adds only the decoder, every other initial value equal', bool(extra) and all(k.startswith('aux_past_dec') for k in extra)
           and all(torch.equal(s0[k], sa[k]) for k in s0))
        lay = mm.time_block.Mem[-1]
        ok(f'b3 {n}: the last NeoLayer neuron is the plain class', type(lay.lif.neuron) is (LIFNeuron if n == 'lif' else FLIFNeuron))
        mm.train(); mm.train_mode = 'training'; functional.reset_net(mm)   # training mode: at init the last layer is silent in eval mode
        with torch.no_grad():
            _, neo_in = mm._tokens_mc(x)
            _, xn = mm.neo_forward(neo_in)
            mm.neo_readout = 'spike'; functional.reset_net(mm); _, xs = mm.neo_forward(neo_in); mm.neo_readout = 'mem'
        nonbin = ((xn != 0) & (xn != 1)).float().mean().item()
        th = lay.lif.neuron.threshold
        ok(f'c1 {n}: mem readout real-valued [N, B, 1, D]', tuple(xn.shape) == (16, 8, 1, 384) and nonbin > 0.5, f'non-binary share {nonbin:.3f}')
        ok(f'c2 {n}: mem readout >= threshold = the spike readout (and the layer fires)', torch.equal((xn >= th).float(), xs) and xs.mean().item() > 0.01,
           f'spike rate {xs.mean().item():.3f}')
        for mode in ('training', 'testing'):
            outs = []
            for flag_once in (False, True):
                for blk in mm.data_block:
                    blk.ssa.q_once = flag_once
                outs.append(run(mm, x, mode))
            ok(f'c3 {n}: q_linear once = per step (mem model, {mode})', torch.allclose(outs[0], outs[1], atol=1e-5),
               f'max diff {(outs[0] - outs[1]).abs().max().item():.1e}')
        ma.train(); ma.train_mode = 'training'; functional.reset_net(ma)
        out = ma(x)
        t, d, src = ma._aux_past_last
        raw = ma._neo_in[:, :, 0]
        okt = bool(((d >= 1) & (d <= t)).all()) and torch.equal(src, t - d) and all(torch.equal(out[3][i, b], raw[src[i, b], b]) for i in range(15) for b in range(8))
        ok(f'd1 {n}: aux lag d in 1..t, target = raw token t - d', okt)
        mma.train(); mma.train_mode = 'training'; functional.reset_net(mma); o2 = mma(x)
        ok(f'd2 {n}: mem + aux, the aux reads the membrane', ((o2[2] != 0) & (o2[2] != 1)).float().mean().item() > 0.5)
        for tag, mdl in (('mem', mm), ('mem+aux', mma)):
            mdl.zero_grad(); mdl.train(); mdl.train_mode = 'training'; functional.reset_net(mdl)
            o = mdl(x)
            loss = torch.nn.functional.cross_entropy(o[0], y) + (0.1 * ((o[3] - o[4]) ** 2).mean() if tag == 'mem+aux' else 0)
            loss.backward(); functional.reset_net(mdl)
            gs = [p.grad for p in mdl.parameters() if p.grad is not None]
            ok(f'e1 {n} {tag}: training step finite, gradients finite', bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in gs))
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
