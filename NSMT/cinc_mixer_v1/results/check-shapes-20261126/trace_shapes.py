"""Shape trace of the forecaster in use (2026-11-26, exp/f-lif-pop-v3; user request "neo-mixer 연산 방식을 tensor 차원과 함께 설명";
no training). Builds Forecaster(SpkMixer) exactly as load_spk_mixer / train_forecast.py do for the LIF SpkMixer (mixer lif,
mixer_comb muxlin, connect_f IAND, no bias, tau 2, T_h 4) on CPU with deliberately distinct sizes so every axis can be told apart
in the printed shapes: B 2, C 3 (B' = B*C = 6), L 96, P 16 (stride 8 -> N 12), D 40, MLP ratio 0.5 (D_h 20), H 24, mixer layers
per direction M 2, blocks 2. Prints every module's input / output shape in call order and checks (fail closed):
  c1 every neuron output is 0/1;  c2 the block output is a subset of the embedding spikes (IAND: x * (1 - m) can only delete);
  c3 the head input (T mean of the spikes) takes values k / T;  c4 muxlin: m == s_d ? L : min(F, R) recomputed from hooks;
  c5 output shape [B, H, C]; c6 N = (L - P) / S + 2; c7 the embedding, mixer output, s_d, F, R and block output all have spikes
  (so c1-c4 are not vacuous); c8 the spikeDE LIF equals the hand recursion H_t = u_t + (I_t - u_t) / tau, s_t = [H_t >= theta],
  u_{t+1} = H_t - theta s_t / tau. Run in train mode under no_grad (BN batch statistics).
usage (from cinc_mixer_v1/): python -B results/check-shapes-20261126/trace_shapes.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from forecast_wrap import Forecaster          # noqa: E402
from spk_mixer import SpkMixer                # noqa: E402

torch.manual_seed(0)
B, C, L, P, D, MLPR, H, M, BLK, T = 2, 3, 96, 16, 40, 0.5, 24, 2, 2, 4
inner = SpkMixer(seq_len=L, data_patch_size=P, embed_dim=D, num_classes=1, depths=BLK, mlp_ratios=MLPR, mixer='lif', mixer_layers=M,
                 T_h=T, neo_tau=2.0, bias=False, tau=2.0, mixer_comb='muxlin', connect_f='IAND')
model = Forecaster(inner, D, H, bias=False).train()   # train mode: BN uses batch statistics, so the random-init model fires
# (v1 ran .eval(): BN running stats 1 / 0 with std-0.02 weights gave no spike at all, and c1-c4 passed vacuously; kept as
# trace_shapes_v1_eval_silent.txt). c7 now fails if a checked tensor has no spike.
N = inner.num_patches
fails = 0


def check(label, c):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}')


rec, outs = [], {}
names = {m: n for n, m in model.named_modules()}


def hook(m, i, o):
    n = names.get(m, type(m).__name__)
    si = tuple(i[0].shape) if i and torch.is_tensor(i[0]) else None
    so = tuple(o.shape) if torch.is_tensor(o) else tuple(tuple(z.shape) for z in o)
    rec.append((n, type(m).__name__, si, so))
    outs.setdefault(n, []).append(o if torch.is_tensor(o) else o[0])


for n, m in model.named_modules():
    if n and type(m).__name__ in ('Linear', 'BatchNorm1d', 'NeoNeuron', 'MixerStack', 'NeoLayer', 'FuseLinear', 'SpikLinearLayer',
                                  'Embedding', 'SpkBlock'):
        m.register_forward_hook(hook)
x = torch.randn(B, L, C)
with torch.no_grad():
    e = None
    y = model(x)
print(f'sizes: B {B}, C {C}, B\' {B * C}, L {L}, P {P}, S {P // 2}, N {N}, T {T}, D {D}, D_h {int(D * MLPR)}, H {H}, M {M}, blocks {BLK}')
for n, k, si, so in rec:
    print(f'  {n:34s} {k:16s} in {si}  out {so}')
check('c6 N = (L - P) / S + 2', N == (L - P) // (P // 2) + 2)
check('c5 output [B, H, C]', tuple(y.shape) == (B, H, C))
neu = [n for n, k, _, _ in rec if k == 'NeoNeuron']
check(f'c1 every neuron output is 0/1 ({len(set(neu))} neuron modules)', all(((o == 0) | (o == 1)).all() for n in set(neu) for o in outs[n]))
with torch.no_grad():
    z = x.permute(0, 2, 1).reshape(B * C, 1, L)
    z = (z - z.mean(-1, keepdim=True)) / torch.sqrt(z.var(-1, keepdim=True, unbiased=False) + 1e-5)
    _, emb, m0, _, h = inner.forward_parts(z)
check('c2 block output is a subset of the embedding spikes (h <= e)', bool((h <= emb).all()) and tuple(h.shape) == (T, B * C, N, D))
f = h.mean(0)
check('c3 head input values are k / T', bool(torch.isclose(f * T, (f * T).round()).all()))
blk = inner.blocks[0]
with torch.no_grad():
    xm = emb.permute(2, 1, 0, 3).contiguous()
    F_, Hf = blk.mixer.forward_charge(xm)
    R_, Hr = blk.mixer_rev.forward_charge(xm.flip(0))
    R_, Hr = R_.flip(0), Hr.flip(0)
    sd = blk.sd_neuron(blk.sd_bn((Hf - Hr).abs().reshape(-1, D)).reshape(N, B * C, T, D).contiguous())
    Lf = blk.fuse(torch.cat([torch.maximum(F_, R_), sd], -1))
    mm = torch.where(sd > 0, Lf, torch.minimum(F_, R_)).permute(2, 1, 0, 3)
check('c4 muxlin: m == s_d ? L : min(F, R)', torch.equal(mm, m0))
rates = {'embedding e': emb, 'forward F': F_, 'reverse R': R_, 's_d': sd, 'fusion L': Lf, 'mixer output m': m0, 'block output h': h}
print('  spike rates (random init, train-mode BN): ' + ', '.join(f'{k} {v.float().mean():.3f}' for k, v in rates.items()))
check('c7 every checked tensor has spikes (c1-c4 not vacuous)', all(v.sum() > 0 for v in rates.values()))
check('c7b h is a strict subset somewhere (IAND deleted some embedding spikes)', bool((h < emb).any()))
from layers import build_neo_neuron     # noqa: E402
nn_ = build_neo_neuron('lif', 5, 10, 'ND', tau=2.0)                    # c8: the LIF recursion written in the explanation
cur = torch.randn(10, 3, 1, 5) * 1.5
with torch.no_grad():
    s_mod, h_mod = nn_.forward_charge(cur)
    u, S_, Hm = torch.zeros(3, 1, 5), [], []
    for t in range(10):
        Ht = u + (cur[t] - u) / 2.0                                       # H_t = u_t + (I_t - u_t) / tau
        st = (Ht - 1.0 >= 0).float()                                      # s_t = Heaviside(H_t - theta)
        u = Ht - 1.0 * st / 2.0                                           # u_{t+1} = H_t - theta * s_t / tau
        S_.append(st)
        Hm.append(Ht - 1.0)
check('c8 LIF: spikes and H - theta equal the hand recursion (tau 2, theta 1)',
      torch.equal(s_mod, torch.stack(S_)) and torch.allclose(h_mod, torch.stack(Hm), atol=1e-6) and 0 < s_mod.mean() < 1)
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
