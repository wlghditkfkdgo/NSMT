"""Layer-wise rate-matched BN gains for the soma-less population models (MODI_PROJECT_LOG 2026-10-22).

For each grouped neuron layer of a popg (NeoMixer, tau groups) or popk (StreamMixer, streams) model, in forward order, set the
BN weight of every tau group so that the group's initial firing on real CinC data equals the firing of the same layer in the
reference model (NeoMixer with the spikeDE LIF / f-LIF mixer of the same family and the same combination). Each layer is
calibrated on the input it actually receives after the upstream layers were calibrated (bisection on the gain, the neuron
alone, 25 steps). The result is a fixed table calib/tau_gains_<model>_<mixer>_<comb>.json {BN module name: [gain per group]}
that the model constructors load (layers.apply_tau_calibration), so every run starts from the same values.
Data: CinC 2017 training windows 0..31 of rhythm_data.build(fold 0, seed 42), model seed 0, training-mode BN.
usage: python calibrate_groups.py [--only popg_lif:or ...]
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
from layers import GroupNeuron                                            # noqa: E402
from stream_mixer import StreamNeurons                                    # noqa: E402
import rhythm_data as rd                                                  # noqa: E402

KW = dict(keep_ratio=0.15, gating='attn', embed_dim=384, num_heads=8, mlp_ratios=1.0, num_layers=1, time_num_layers=2, bias=False,
          tau=2.0, spk_encoding=False, neo_alpha=0.5, pop_taus='4,8,16,32', neo_tau=2.0, fde_method='pred', fde_step=1.0, fde_memory=0,
          mixer_bidir=True, pos_emb=False, tau_calib=False)                    # build uncalibrated, then calibrate
MODELS = [('neoMixer', 'popg_lif'), ('neoMixer', 'popg_flif'), ('streamMixer', 'popk_lif'), ('streamMixer', 'popk_flif')]
BN_OF = {'emb_lif': 'emb_bn', 'lif': 'bn', 'neuron': 'bn', 'attn_neuron': 'attn_bn'}


def layer_kind(name):
    for k in ('encoding', 'mixer_rev.Mem.0', 'mixer_rev.layers.0', 'mixer.Mem.0', 'mixer.layers.0', 'mixer_rev.Mem.1', 'mixer_rev.layers.1',
              'mixer.Mem.1', 'mixer.layers.1', 'xattn.q', 'xattn.k', 'xattn.v', 'xattn.attn', 'xattn.proj', 'fc1', 'fc2'):
        if k in name:
            return k.replace('layers.', 'Mem.')                         # stream model 'layers.i' = NeoMixer 'Mem.i'
    raise KeyError(name)


def grouped_layers(m):
    """[(group-neuron module name, module, BN module name)] for every GroupNeuron / StreamNeurons with K > 1 parts."""
    out = []
    for name, mod in m.named_modules():
        if isinstance(mod, (GroupNeuron, StreamNeurons)) and len(mod.parts) > 1:
            parent, leaf = name.rsplit('.', 1)
            out.append((name, mod, f'{parent}.{BN_OF[leaf]}'))
    return out


@torch.no_grad()
def forward(m, x):
    m.train(); m.train_mode = 'testing'
    m(x); functional.reset_net(m)


@torch.no_grad()
def ref_rates(model_kind, fam, comb, x, kw):
    """mean firing of each layer kind in the reference model (one neuron module per kind there)."""
    set_random_seed(0)
    ref = LOAD_MODEL['neoMixer'](NS(**dict(KW, model='neoMixer', mixer=fam, mixer_comb=comb, **kw)), True).cuda()
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    from layers import NeoNeuron
    rates, hs = {}, []
    for name, mod in ref.named_modules():
        if isinstance(mod, (MultiStepLIFNode, NeoNeuron)) and not any(isinstance(c, (MultiStepLIFNode, NeoNeuron)) for c in mod.modules() if c is not mod):
            hs.append(mod.register_forward_hook(lambda md, i, o, k=layer_kind(name): rates.__setitem__(k, float((o[0] if isinstance(o, tuple) else o).float().mean()))))
    forward(ref, x)
    for h in hs: h.remove()
    del ref; torch.cuda.empty_cache()
    return rates


@torch.no_grad()
def part_rate(part, inp):
    o = part(inp.contiguous()); functional.reset_net(part)
    return float((o[0] if isinstance(o, tuple) else o).float().mean())


def calibrate(model, mixer, comb, x, kw, steps=25):
    fam = mixer.split('_', 1)[1]
    target = ref_rates(model, fam, comb, x, kw)
    set_random_seed(0)
    m = LOAD_MODEL[model](NS(**dict(KW, model=model, mixer=mixer, mixer_comb=comb, **kw)), True).cuda()
    mods = dict(m.named_modules())
    layers = grouped_layers(m)
    order = []                                                             # forward order of the grouped layers
    hs = [mod.register_forward_pre_hook(lambda md, i, n=name: order.append(n) if n not in order else None) for name, mod, _ in layers]
    forward(m, x)
    for h in hs: h.remove()
    info = {name: (mod, bn) for name, mod, bn in layers}
    table, report = {}, []
    for name in order:
        mod, bn_name = info[name]
        bn = mods[bn_name]
        caps = {}
        hs = [p.register_forward_pre_hook(lambda md, i, k=k: caps.__setitem__(k, i[0].detach().clone())) for k, p in enumerate(mod.parts)]
        forward(m, x)
        for h in hs: h.remove()
        K = len(mod.parts); C = bn.weight.numel(); G = C // K
        tgt = target[layer_kind(name)]
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
        report.append(f'{bn_name}: target {tgt:.4f} gains ' + '/'.join(f'{g:.3f}' for g in gains))
    del m; torch.cuda.empty_cache()
    return table, report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', nargs='*', default=None); ap.add_argument('--n', type=int, default=32)
    a = ap.parse_args()
    d = rd.build(NS(data='CINC2017', crop_tokens=0, patch_size=64, fs_target=125., fold=0, val_frac=0.1, win_s=30., hop_s=15.), 42)
    x = torch.stack([d['train'][i][0] for i in range(a.n)]).float().cuda()
    kw = dict(patch_size=64, seq_len=d['seq_len'], num_channels=d['num_channels'], num_classes=d['num_classes'])
    os.makedirs(os.path.join(HERE, 'calib'), exist_ok=True)
    for model, mixer in MODELS:
        for comb in ('or', 'xattn'):
            if a.only and f'{mixer}:{comb}' not in a.only:
                continue
            table, report = calibrate(model, mixer, comb, x, kw)
            path = os.path.join(HERE, 'calib', f'tau_gains_{model}_{mixer}_{comb}.json')
            json.dump(dict(model=model, mixer=mixer, comb=comb, data='CINC2017 fold 0 seed 42 train windows 0..%d' % (a.n - 1), model_seed=0,
                           gains=table), open(path, 'w'), indent=1)
            print(f'== {mixer}/{comb} -> {os.path.relpath(path, HERE)}'); print('\n'.join('   ' + r for r in report), flush=True)


if __name__ == '__main__':
    main()
