"""Bidirectional Neocortex for the query form (2026-11-05, exp/f-lif-pop-v3; PROJECT_LOG; user 11-04: in [query pop LIF NSMT T=4] the
Neocortex path becomes the bidirectional Neo-mixer, each direction 1 layer, fused by muxlin, stacked twice; user answers: OR and IAND
residuals both tested, the four neurons, a bidirectional OR-fusion control). It replaces TemporalBlock as model.myModel.time_block
(`--neo_mux`); the input is the shared encoding_neo output and the output goes to the hippocampus query unchanged in shape.

    x = e [N, B, 1, D]                                  embedding spikes (token axis = spiking time axis)
    for each block:
        F, H_f = NeoLayer_f(x)                          Linear -> BN -> neuron over the token axis
        R, H_r = NeoLayer_r(x reversed), flipped back   its own weights
        muxlin: s_d = neuron_d(BN_d(|H_f - H_r|))
                L   = neuron(BN(W [F v R; s_d]))        W: 2D -> D, 0/1 input
                m   = s_d ? L : F ^ R                   training arithmetic s * L + (1 - s) * min(F, R), as spk_mixer.SpkBlock.mix
        or:     m   = F v R                             control: bidirectional without the gate
        x = max(x, m) (residual OR)  |  x (1 - m) (residual IAND)
    output x [N, B, 1, D]

Every neuron is the chosen Neocortex neuron (myModel's mlp_fn, as in TemporalBlock). All direction layers are built before any fusion
part, so at one seed the direction weights are the same across fusions and residuals.
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from model import NeoLayer

__all__ = ['NeoMuxStack', 'FUSIONS', 'RESIDUALS']

FUSIONS = ('muxlin', 'or')
RESIDUALS = ('OR', 'IAND')


class FuseLinear(nn.Module):
    """Linear(din -> dout) -> BN(dout) over every (token, batch, stream) row -> neuron over the token axis, on [N, B, K, din]."""
    def __init__(self, din, dout, neuron, bias=False):
        super().__init__()
        self.linear = nn.Linear(din, dout, bias=bias)
        self.bn = nn.BatchNorm1d(dout)
        self.neuron = neuron
        self.dout = dout

    def forward(self, x):
        N, B, K, _ = x.shape
        y = self.bn(self.linear(x).reshape(-1, self.dout)).reshape(N, B, K, self.dout)
        return self.neuron(y.contiguous())


def _init_linear(m):
    trunc_normal_(m.weight, std=.02)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)


class NeoMuxBlock(nn.Module):
    def __init__(self, dim, neuron_fn, fusion, residual, bias=False):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f'neo_mux fusion must be one of {FUSIONS}')
        if residual not in RESIDUALS:
            raise ValueError(f'neo_mux residual must be one of {RESIDUALS}')
        self.fusion, self.residual, self.dim = fusion, residual, dim
        self.mixer = NeoLayer(dim, bias=bias, neuron=neuron_fn())
        self.mixer_rev = NeoLayer(dim, bias=bias, neuron=neuron_fn())
        _init_linear(self.mixer.w_in)
        _init_linear(self.mixer_rev.w_in)

    def build_fusion(self, neuron_fn, bias=False):
        if self.fusion == 'muxlin':
            self.sd_bn = nn.BatchNorm1d(self.dim)
            self.sd_neuron = neuron_fn()
            self.fuse = FuseLinear(2 * self.dim, self.dim, neuron_fn(), bias)
            _init_linear(self.fuse.linear)

    def mix(self, x):
        """x [N, B, 1, D] -> m [N, B, 1, D]."""
        if self.fusion == 'or':
            return torch.maximum(self.mixer(x), self.mixer_rev(x.flip(0)).flip(0))
        f, hf = self.mixer.forward_charge(x)
        r, hr = self.mixer_rev.forward_charge(x.flip(0))
        r, hr = r.flip(0), hr.flip(0)
        d = (hf - hr).abs()
        N, B, K, D = d.shape
        sd = self.sd_neuron(self.sd_bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous())
        L = self.fuse(torch.cat([torch.maximum(f, r), sd], dim=-1))
        return sd * L + (1. - sd) * torch.minimum(f, r)

    def connect(self, x, m):
        return torch.maximum(x, m) if self.residual == 'OR' else x * (1. - m)

    def forward(self, x):
        return self.connect(x, self.mix(x))


class NeoMuxStack(nn.Module):
    def __init__(self, dim, neuron_fn, fusion, residual, blocks=2, bias=False):
        super().__init__()
        if int(blocks) < 1:
            raise ValueError('neo_mux needs at least one block')
        self.blocks = nn.ModuleList([NeoMuxBlock(dim, neuron_fn, fusion, residual, bias) for _ in range(int(blocks))])
        for b in self.blocks:                                               # after every direction layer (see the module docstring)
            b.build_fusion(neuron_fn, bias)

    def forward(self, x, org_data=None):
        for b in self.blocks:
            x = b(x)
        return x
