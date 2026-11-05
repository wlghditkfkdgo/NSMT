"""Which axes does |H_f - H_r| cover? Runs the real mix() with hooks on sd_neuron; CPU, untrained weights, no files written."""
import sys
import torch

folder = sys.argv[1]
kind = sys.argv[2]
sys.path.insert(0, folder)
from layers import build_neo_neuron

torch.manual_seed(0)
N, B, D, K = 12, 2, 8, 4
p = 6
x = (torch.rand(N, B, K, D) < .3).float()
x2 = x.clone()
x2[p] = 1. - x2[p]

if folder.endswith('qmux_v1'):
    from neo_mux import NeoMuxBlock
    fn = lambda: build_neo_neuron(kind, D, N, 'ND')
    blk = NeoMuxBlock(D, fn, 'muxlin', 'OR')
    blk.build_fusion(fn)
    mix = lambda z: blk.mix(z)
    charge = lambda z: (blk.mixer.forward_charge(z)[1], blk.mixer_rev.forward_charge(z.flip(0))[1].flip(0))
else:
    from spk_mixer import SpkBlock
    make = lambda d, T, layout: build_neo_neuron(kind, d, T, layout)
    blk = SpkBlock(D, kind, 1, 1., make, N, K, 'muxlin', 'OR')
    mix = lambda z: blk.mix(z.permute(2, 1, 0, 3)).permute(2, 1, 0, 3)     # mix() takes [T_h, B, N, D]
    charge = lambda z: (blk.mixer.forward_charge(z)[1], blk.mixer_rev.forward_charge(z.flip(0))[1].flip(0))

blk.eval()
seen = {}
blk.sd_neuron.register_forward_pre_hook(lambda m, a: seen.__setitem__('in', a[0].detach().clone()))
blk.sd_neuron.register_forward_hook(lambda m, a, o: seen.__setitem__('out', o.detach().clone()))

with torch.no_grad():
    out = mix(x)
    print(f'[{folder.split("/")[-1]} {kind}] input to s_d neuron {tuple(seen["in"].shape)}  s_d spikes {tuple(seen["out"].shape)}  mix output {tuple(out.shape)}  (N={N}, B={B}, K={K}, D={D})')
    d_in0 = seen['in'].clone()
    hf, hr = charge(x)
    hf2, hr2 = charge(x2)
    if hf.shape != (N, B, K, D) or hr.shape != (N, B, K, D):
        raise SystemExit(f'unexpected charge shape {tuple(hf.shape)}')
    df = (hf2 - hf).abs().amax(dim=(1, 2, 3))
    dr = (hr2 - hr).abs().amax(dim=(1, 2, 3))
    print(f'  token {p} changed -> H_f moved at steps {[t for t in range(N) if df[t] > 0]}')
    print(f'  token {p} changed -> H_r moved at steps {[t for t in range(N) if dr[t] > 0]}')
    print('  |dH_f| by step', ' '.join(f'{v:.1e}' for v in df.tolist()))
    print('  |dH_r| by step', ' '.join(f'{v:.1e}' for v in dr.tolist()))
    x3 = x.clone()
    x3[:, :, 0] = 1. - x3[:, :, 0]                                          # only stream / repeat 0 changes
    hf3, _ = charge(x3)
    moved = [k for k in range(K) if (hf3[:, :, k] - hf[:, :, k]).abs().amax() > 0]
    print(f'  repeat 0 changed -> H_f moved in repeats {moved}')
