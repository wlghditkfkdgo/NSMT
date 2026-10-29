"""Layer-wise rate-matched BN gains for the SpkMixer population models (2026-10-29, PROJECT_LOG, exp/f-lif-pop-v3).

Generalises calibrate_groups.py (MODI 2026-10-22) to the whole-model neuron conditions of spk_mixer.SpkMixer: for every
calibratable neuron layer -- a GroupNeuron with K tau groups (popg_lif / popg_flif: one gain per group) or a NeoNeuron wrapping
a PopulationFLIF (pop_lif / pop_flif: one gain for the layer) -- in forward order, set the BN weight(s) in front of it so that
its initial firing on real CinC data equals the firing of the same layer in the reference model (SpkMixer with the single
spikeDE neuron of the same family: lif for pop_lif / popg_lif, flif for pop_flif / popg_flif; same fusion and residual). Each
layer is calibrated on the input it actually receives after the upstream layers were calibrated (bisection on the gain, the
neuron alone, 25 steps). Tables: calib/gains_spkMixer_<mixer>_<comb>_<connect>.json {BN module name: [gain per slice]}, loaded by
SpkMixer (spk_mixer.apply_gain_table). Existing tables are never overwritten.
Data: CinC 2017 training windows 0..31 of rhythm_data.build(fold 0, seed 42), model seed 0, training-mode BN.
usage: python calibrate_spk.py [--only pop_lif:sdor:IAND ...] [--combs ...] [--connects ...]
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
from layers import GroupNeuron, NeoNeuron                                 # noqa: E402
from neurons import PopulationFLIF                                        # noqa: E402
from spk_mixer import COMBS, CONNECTS, gain_table_path                    # noqa: E402
import rhythm_data as rd                                                  # noqa: E402

KW = dict(keep_ratio=0.15, gating='attn', embed_dim=384, num_heads=8, mlp_ratios=1.0, num_layers=1, time_num_layers=2, bias=False,
          tau=2.0, neo_alpha=0.5, pop_taus='4,8,16,32', neo_tau=2.0, fde_method='pred', fde_step=1.0, fde_memory=0, hippo_T=4,
          tau_calib=False)                                                   # build uncalibrated, then calibrate
SOMA_INIT = {'pop_lif': 0.5, 'pop_flif': 1.0}                               # as every CinC population run since stage 9
MIXERS = ('pop_lif', 'pop_flif', 'popg_lif', 'popg_flif')
BN_OF = {'emb_lif': 'emb_bn', 'lif': 'bn', 'neuron': 'bn', 'sd_neuron': 'sd_bn'}
KINDS = ('encoding', 'mixer_rev.Mem.0', 'mixer_rev.Mem.1', 'mixer.Mem.0', 'mixer.Mem.1', 'sd_neuron', 'fuse', 'fc1', 'fc2')


def layer_kind(name):
    for k in KINDS:
        if k in name:
            return k
    raise KeyError(name)


def calib_layers(m):
    """[(neuron module name, parts, BN module name)]: GroupNeuron (K parts) or NeoNeuron(PopulationFLIF) (the module itself)."""
    out = []
    for name, mod in m.named_modules():
        if isinstance(mod, GroupNeuron) and len(mod.parts) > 1:
            parts = list(mod.parts)
        elif isinstance(mod, NeoNeuron) and isinstance(mod.neuron, PopulationFLIF):
            parts = [mod]
        else:
            continue
        parent, leaf = name.rsplit('.', 1)
        out.append((name, parts, f'{parent}.{BN_OF[leaf]}'))
    return out


@torch.no_grad()
def forward(m, x):
    m.train(); m.train_mode = 'testing'
    m(x); functional.reset_net(m)


def build(mixer, comb, connect, kw):
    set_random_seed(0)
    a = NS(**dict(KW, model='spkMixer', mixer=mixer, mixer_comb=comb, connect_f=connect, pop_soma_init=SOMA_INIT.get(mixer), **kw))
    return LOAD_MODEL['spkMixer'](a, True).cuda()


@torch.no_grad()
def ref_rates(fam, comb, connect, x, kw):
    """mean firing of each layer kind in the reference model (NeoNeuron leaves, one per layer)."""
    ref = build(fam, comb, connect, kw)
    rates, hs = {}, []
    for name, mod in ref.named_modules():
        if isinstance(mod, NeoNeuron):
            hs.append(mod.register_forward_hook(lambda md, i, o, k=layer_kind(name): rates.__setitem__(k, float((o[0] if isinstance(o, tuple) else o).float().mean()))))
    forward(ref, x)
    for h in hs: h.remove()
    del ref; torch.cuda.empty_cache()
    return rates


@torch.no_grad()
def part_rate(part, inp):
    o = part(inp.contiguous()); functional.reset_net(part)
    return float((o[0] if isinstance(o, tuple) else o).float().mean())


def calibrate(mixer, comb, connect, x, kw, steps=25):
    fam = mixer.split('_', 1)[1]
    target = ref_rates(fam, comb, connect, x, kw)
    m = build(mixer, comb, connect, kw)
    mods = dict(m.named_modules())
    layers = calib_layers(m)
    order = []                                                             # forward order of the calibratable layers
    hs = [mods[name].register_forward_pre_hook(lambda md, i, n=name: order.append(n) if n not in order else None) for name, _, _ in layers]
    forward(m, x)
    for h in hs: h.remove()
    if len(order) != len(layers):
        raise RuntimeError(f'{len(layers)} calibratable layers, {len(order)} ran')
    info = {name: (parts, bn) for name, parts, bn in layers}
    table, report = {}, []
    for name in order:
        parts, bn_name = info[name]
        bn = mods[bn_name]
        caps = {}
        hs = [p.register_forward_pre_hook(lambda md, i, k=k: caps.__setitem__(k, i[0].detach().clone())) for k, p in enumerate(parts)]
        forward(m, x)
        for h in hs: h.remove()
        K = len(parts); C = bn.weight.numel(); G = C // K
        tgt = target[layer_kind(name)]
        if not (tgt > 0.):
            raise RuntimeError(f'reference layer {layer_kind(name)} is silent ({tgt}); cannot calibrate')
        gains = []
        for k, part in enumerate(parts):
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
    ap.add_argument('--combs', nargs='*', default=list(COMBS)); ap.add_argument('--connects', nargs='*', default=list(CONNECTS))
    ap.add_argument('--mixers', nargs='*', default=list(MIXERS))
    a = ap.parse_args()
    d = rd.build(NS(data='CINC2017', crop_tokens=0, patch_size=64, fs_target=125., fold=0, val_frac=0.1, win_s=30., hop_s=15.), 42)
    x = torch.stack([d['train'][i][0] for i in range(a.n)]).float().cuda()
    kw = dict(patch_size=64, seq_len=d['seq_len'], num_channels=d['num_channels'], num_classes=d['num_classes'])
    os.makedirs(os.path.join(HERE, 'calib'), exist_ok=True)
    for mixer in a.mixers:
        for comb in a.combs:
            for connect in a.connects:
                if a.only and f'{mixer}:{comb}:{connect}' not in a.only:
                    continue
                path = gain_table_path(mixer, comb, connect)
                if os.path.exists(path):                                   # never overwrite a table runs were trained with
                    print(f'== {mixer}/{comb}/{connect}: exists, skipped'); continue
                table, report = calibrate(mixer, comb, connect, x, kw)
                json.dump(dict(model='spkMixer', mixer=mixer, comb=comb, connect=connect, soma_init=SOMA_INIT.get(mixer),
                               data='CINC2017 fold 0 seed 42 train windows 0..%d' % (a.n - 1), model_seed=0, gains=table), open(path, 'w'), indent=1)
                print(f'== {mixer}/{comb}/{connect} -> {os.path.relpath(path, HERE)}'); print('\n'.join('   ' + r for r in report), flush=True)


if __name__ == '__main__':
    main()
