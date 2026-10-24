"""Layer-wise rate-matched BN gains for the NSMT neocortex made of K soma-less population streams aligned with the
hippocampus steps (MODI_PROJECT_LOG 2026-10-23; the rule of calibrate_groups.py, MODI 2026-10-22).

Model: --model myModel --neo_neuron popk_lif | popk_flif --hippo_T 4 --hippo_xattn: the neocortex embedding and its two
layers are K = 4 streams (tau 4 / 8 / 16 / 32, every member fires itself, spikeDE LIF or f-LIF alpha 0.5) over the token
axis; stream k is the key / value of hippocampus step k. For each of the three neocortex neuron layers, in forward order,
the BN weight of every stream is set so that the stream's initial firing on real CinC data equals the firing of the same
layer in the reference model, the NSMT T_h = 4 model with the single spikeDE neuron of the same family (nsmt_lif_t4 /
an f-LIF version of it) -- calibrated on the input the layer actually receives after the upstream layers were calibrated
(bisection on the gain, the stream neuron alone, 25 steps). Output: calib/tau_gains_myModel_<neo_neuron>_xaT4.json, which
model.myModel loads (layers.apply_tau_calibration).
Data: CinC 2017 training windows 0..31 of rhythm_data.build(fold 0, seed 42), model seed 0, training-mode BN.
usage: python calibrate_nsmt_streams.py [--only popk_lif ...]
"""
import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from types import SimpleNamespace as NS                                   # noqa: E402
from spikingjelly.clock_driven import functional                          # noqa: E402
from config import set_random_seed                                        # noqa: E402
from load_model import LOAD_MODEL                                         # noqa: E402
from layers import NeoNeuron                                              # noqa: E402
from stream_mixer import StreamNeurons                                    # noqa: E402
import rhythm_data as rd                                                  # noqa: E402

KW = dict(model='myModel', keep_ratio=0.15, gating='attn', embed_dim=384, num_heads=8, mlp_ratios=1.0, num_layers=1, time_num_layers=2,
          bias=False, tau=2.0, spk_encoding=False, neo_alpha=0.5, pop_taus='4,8,16,32', neo_tau=2.0, fde_method='pred', fde_step=1.0,
          fde_memory=0, neo_where='both', hippo_T=4, hippo_xattn=True, tau_calib=False)
STREAMS = {'popk_lif': 'lif', 'popk_flif': 'flif'}                        # stream model -> reference single neuron


@torch.no_grad()
def forward(m, x):
    m.train(); m.train_mode = 'testing'
    m(x); functional.reset_net(m)


def neo_layers(m, cls):
    """names of the neocortex neuron modules of type cls (encoding_neo / time_block), in forward-call order."""
    order, hs = [], []
    for name, mod in m.named_modules():
        if isinstance(mod, cls) and name.startswith(('encoding_neo', 'time_block')):
            hs.append(mod.register_forward_pre_hook(lambda md, i, n=name: order.append(n) if n not in order else None))
    return order, hs


@torch.no_grad()
def ref_rates(fam, x, kw):
    set_random_seed(0)
    ref = LOAD_MODEL['myModel'](NS(**dict(KW, neo_neuron=fam, **kw)), True).cuda()
    order, hs = neo_layers(ref, NeoNeuron)
    rates = {}
    mods = dict(ref.named_modules())
    hs += [mods[n].register_forward_hook(lambda md, i, o, n=n: rates.__setitem__(n, float(o.float().mean())))
           for n, mod in mods.items() if isinstance(mod, NeoNeuron) and n.startswith(('encoding_neo', 'time_block'))]
    forward(ref, x)
    for h in hs: h.remove()
    del ref; torch.cuda.empty_cache()
    return [rates[n] for n in order], order


@torch.no_grad()
def part_rate(part, inp):
    o = part(inp.contiguous()); functional.reset_net(part)
    return float(o.float().mean())


def slot(name):
    """0 = embedding, 1 / 2 = first / second layer -- the same slot in the forward and the reverse (bimx) stack."""
    if name.startswith('encoding_neo'):
        return 0
    return 1 + int(name.split('layers.')[1].split('.')[0])


def calibrate(neo, x, kw, steps=25, extra=None):
    target, ref_order = ref_rates(STREAMS[neo], x, kw)
    set_random_seed(0)
    m = LOAD_MODEL['myModel'](NS(**dict(KW, neo_neuron=neo, **kw, **(extra or {}))), True).cuda()
    mods = dict(m.named_modules())
    order, hs = neo_layers(m, StreamNeurons)
    forward(m, x)
    for h in hs: h.remove()
    if len(target) != 3 or len(order) not in (3, 6):
        raise SystemExit(f'{neo}: {len(order)} stream layers vs {len(target)} reference layers ({order} / {ref_order})')
    table, report = {}, []
    for name in order:
        tgt, rname = target[slot(name)], ref_order[slot(name)]
        mod = mods[name]
        bn_name = name.rsplit('.', 1)[0] + '.bn'
        bn = mods[bn_name]
        caps = {}
        hs = [p.register_forward_pre_hook(lambda md, i, k=k: caps.__setitem__(k, i[0].detach().clone())) for k, p in enumerate(mod.parts)]
        forward(m, x)
        for h in hs: h.remove()
        K = len(mod.parts); G = bn.weight.numel() // K
        gains = []
        for k, part in enumerate(mod.parts):
            g0 = float(bn.weight[k * G])
            z = caps[k] / g0                                               # BN bias is 0 at init: input = g0 * z
            lo, hi = math.log(0.05), math.log(128.)
            for _ in range(steps):
                mid = (lo + hi) / 2
                if part_rate(part, math.exp(mid) * z) < tgt:
                    lo = mid
                else:
                    hi = mid
            g = math.exp((lo + hi) / 2)
            gains.append(g)
            with torch.no_grad():
                bn.weight[k * G:(k + 1) * G] = g
        table[bn_name] = [round(g, 6) for g in gains]
        report.append(f'{bn_name} (reference {rname}): target {tgt:.4f} gains ' + '/'.join(f'{g:.3f}' for g in gains))
    # the calibrated model's own per-stream firing, for the record
    got = {}
    hs = [mods[n].parts[k].register_forward_hook(lambda md, i, o, n=n, k=k: got.__setitem__((n, k), float(o.float().mean())))
          for n in order for k in range(len(mods[n].parts))]
    forward(m, x)
    for h in hs: h.remove()
    report.append('calibrated firing per layer and stream: ' + '; '.join(
        f"{n}: " + '/'.join(f"{got[(n, k)]:.4f}" for k in range(len(mods[n].parts))) for n in order))
    del m; torch.cuda.empty_cache()
    return table, report, {n: target[slot(n)] for n in order}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', nargs='*', default=None); ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--bimx', action='store_true', help='MODI 2026-10-24: the bidirectional stream neocortex fused per stream (6 stream layers)')
    a = ap.parse_args()
    extra = dict(neo_bidir=True, neo_stream_fuse='mx') if a.bimx else {}
    d = rd.build(NS(data='CINC2017', crop_tokens=0, patch_size=64, fs_target=125., fold=0, val_frac=0.1, win_s=30., hop_s=15.), 42)
    x = torch.stack([d['train'][i][0] for i in range(a.n)]).float().cuda()
    kw = dict(patch_size=64, seq_len=d['seq_len'], num_channels=d['num_channels'], num_classes=d['num_classes'])
    os.makedirs(os.path.join(HERE, 'calib'), exist_ok=True)
    for neo in STREAMS:
        if a.only and neo not in a.only:
            continue
        table, report, target = calibrate(neo, x, kw, extra=extra)
        path = os.path.join(HERE, 'calib', f'tau_gains_myModel_{neo}_xaT4' + ('_bimx' if a.bimx else '') + '.json')
        with open(path, 'x') as handle:
            json.dump(dict(model='myModel', neo_neuron=neo, hippo_T=4, variant=('bimx' if a.bimx else 'plain'),
                           reference=f'myModel neo_neuron {STREAMS[neo]} hippo_T 4 hippo_xattn (the reverse stack takes the targets of the same slot)',
                           data='CINC2017 fold 0 seed 42 train windows 0..%d' % (a.n - 1), model_seed=0, targets=target, gains=table),
                      handle, indent=1)
        print(f'== {neo} -> {os.path.relpath(path, HERE)}'); print('\n'.join('   ' + r for r in report), flush=True)


if __name__ == '__main__':
    main()
