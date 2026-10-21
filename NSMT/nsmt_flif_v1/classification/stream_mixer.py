"""Soma-less population, separate streams (MODI_PROJECT_LOG 2026-10-21; the user's design "population axis from input to
output, merged in the head").

The Neo-mixer network is run as K parallel streams that differ only in the neuron time constant tau_k (pop_taus, K = 4:
4 / 8 / 16 / 32): every Linear is shared across the streams, every BN has K*D channels (per-stream statistics, weight init
layers.tau_gain(tau_k)), every neuron layer has one neuron module per stream. Tensors are [N, B, K, D] (N tokens = spiking
steps, T = N everywhere). The streams meet only in the head: token mean per stream, concatenated, Linear(K*D -> classes).

    tokens [N, B, 1, P*C] -> Linear -> copy to K streams -> BN(K*D) -> spikingjelly LIF(tau_k)          (embedding)
    block: forward / reverse neocortex stacks: 2 x [Linear -> BN(K*D) -> spikeDE LIF or f-LIF(tau_k)]
           comb 'or': m = max(F, R)   |   'xattn' / 'xattn_ch': m = bix.TokenXAttn(F, R) (stream-wise)
           x = x * (1 - m);  x = x * (1 - MLP(x)),  MLP = 2 x [Linear -> BN(K*D) -> spikingjelly LIF(tau_k)]
    head: Linear(K*D -> classes) on concat_k mean_t x[t, :, k]

mixer 'popk_lif' = spikeDE LIF in the neocortex stacks, 'popk_flif' = spikeDE f-LIF (alpha); the other layers use the
spikingjelly LIF as in every other model of this folder. With K = 1 and tau = 2 the BN calls reproduce NeoMixer's exactly
(check c28: bitwise equal logits to NeoMixer(mixer='flif') with copied weights).
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from layers import NeoNeuron, sj_lif, tau_gain
from neurons import LIFNeuron, FLIFNeuron
from bix import TokenXAttn, StreamSpikeLinear

__all__ = ['StreamMixer', 'StreamNeurons']


class StreamNeurons(nn.Module):
    """One neuron module per stream on [N, B, K, D]: kind 'sj' = spikingjelly LIF(tau_k), 'lif' / 'flif' = spikeDE."""
    def __init__(self, kind, taus, size, T, alpha=.5):
        super().__init__()
        def mk(t):
            if kind == 'sj':
                return sj_lif(t)
            common = dict(tau=float(t), threshold=1., surrogate_scale=5., step=1., max_length=int(T))
            n = LIFNeuron(size, **common) if kind == 'lif' else FLIFNeuron(size, alpha=alpha, method='pred', memory=None, **common)
            return NeoNeuron(n, 'ND')
        self.parts = nn.ModuleList([mk(t) for t in taus])

    def forward(self, x):
        if len(self.parts) == 1:
            return self.parts[0](x)
        return torch.cat([p(x[:, :, k:k + 1].contiguous()) for k, p in enumerate(self.parts)], dim=2)


class _Stack(nn.Module):
    def __init__(self, dim, n, K, neuron_fn, gamma, bias):
        super().__init__()
        self.layers = nn.ModuleList([StreamSpikeLinear(dim, dim, K, neuron_fn(), bias=bias, bn_layout='flat', gamma=gamma) for _ in range(n)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class _Embed(StreamSpikeLinear):
    """Linear on the shared token, copied to the K streams before the per-stream BN ('ncl' = Embedding's BN call)."""
    def forward(self, tok):
        y = self.linear(tok)                                                 # [N, B, 1, D]
        if self.K > 1:
            y = y.expand(-1, -1, self.K, -1)
        return self.neuron(self.bn_apply(y.contiguous()).contiguous())


class StreamBlock(nn.Module):
    def __init__(self, dim, K, taus, T, kind, alpha, mixer_layers, mlp_ratio, comb, heads, bias, gamma):
        super().__init__()
        self.comb, self.mixer_kind, self.bidir = comb, kind, True
        nfn = lambda: StreamNeurons(kind, taus, dim, T, alpha)
        self.mixer = _Stack(dim, mixer_layers, K, nfn, gamma, bias)
        self.mixer_rev = _Stack(dim, mixer_layers, K, nfn, gamma, bias)
        if comb != 'or':
            self.xattn = TokenXAttn(dim, T, K=K, heads=heads, form=('head' if comb == 'xattn' else 'channel'),
                                    neuron_fn=lambda: StreamNeurons('sj', taus, dim, T), bias=bias, gamma=gamma)
        hidden = int(dim * mlp_ratio)
        g_h = torch.cat([torch.full((hidden,), tau_gain(t)) for t in taus])
        self.fc1 = StreamSpikeLinear(dim, hidden, K, StreamNeurons('sj', taus, hidden, T), bias=bias, bn_layout='ncl', gamma=g_h)
        self.fc2 = StreamSpikeLinear(hidden, dim, K, StreamNeurons('sj', taus, dim, T), bias=bias, bn_layout='ncl', gamma=gamma)

    def mix(self, x):
        m = self.mixer(x)
        r = self.mixer_rev(x.flip(0)).flip(0)
        return torch.maximum(m, r) if self.comb == 'or' else self.xattn(m, r)

    def forward(self, x):
        x = x * (1. - self.mix(x))
        return x * (1. - self.fc2(self.fc1(x)))


class StreamMixer(nn.Module):
    def __init__(self, seq_len=187, data_patch_size=32, embed_dim=384, num_classes=5, depths=1, mlp_ratios=1.0,
                 mixer='popk_flif', mixer_bidir=True, mixer_layers=2, neo_alpha=0.5, pop_taus=(4., 8., 16., 32.), bias=False,
                 tau=2.0, train_mode='training', in_channels=1, mixer_comb='or', heads=8, **kwargs):
        super().__init__()
        if mixer not in ('popk_lif', 'popk_flif'):
            raise ValueError("StreamMixer mixer must be 'popk_lif' or 'popk_flif'")
        if not mixer_bidir:
            raise ValueError('StreamMixer is bidirectional')
        self.train_mode = train_mode
        self.patch_size, self.stride = data_patch_size, data_patch_size // 2
        self.num_patches = int((seq_len - data_patch_size) / self.stride + 1) + 1
        self.T = self.num_patches
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.taus = tuple(float(t) for t in pop_taus)
        self.K = len(self.taus)
        self.mixer_kind, self.mixer_bidir, self.mixer_comb, self.depths = mixer, True, mixer_comb, int(depths)
        self.no_neo = self.hippo_only = False
        self.in_channels = int(in_channels or 1)
        kind = mixer.split('_', 1)[1]
        gamma = torch.cat([torch.full((embed_dim,), tau_gain(t)) for t in self.taus])
        self.encoding = _Embed(data_patch_size * self.in_channels, embed_dim, self.K, StreamNeurons('sj', self.taus, embed_dim, self.T),
                               bias=bias, bn_layout='ncl', gamma=gamma)
        self.blocks = nn.ModuleList([StreamBlock(embed_dim, self.K, self.taus, self.T, kind, neo_alpha, mixer_layers, mlp_ratios,
                                                 mixer_comb, heads, bias, gamma) for _ in range(self.depths)])
        self.head = nn.Linear(self.K * embed_dim, num_classes, bias=bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def tokens(self, x):                                                     # as NeoMixer.tokens -> [N, B, 1, C*P]
        if x.dim() != 3:
            x = x.unsqueeze(1)
        B = x.shape[0]
        if self.in_channels == 1:
            xp = self.padding_patch_layer(x[:, 0]).unfold(-1, self.patch_size, self.stride)
            return xp.transpose(0, 1).unsqueeze(2).contiguous()
        C = x.shape[1]
        p = self.padding_patch_layer(x).unfold(-1, self.patch_size, self.stride)
        p = p.permute(0, 2, 1, 3).reshape(B, p.shape[2], C * self.patch_size)
        return p.transpose(0, 1).unsqueeze(2).contiguous()

    def forward_parts(self, x):
        e = self.encoding(self.tokens(x))                                    # [N, B, K, D]
        h, m0 = e, None
        for i, blk in enumerate(self.blocks):
            if i == 0:
                m0 = blk.mix(h)
                h = h * (1. - m0)
                h = h * (1. - blk.fc2(blk.fc1(h)))
            else:
                h = blk(h)
        feat = h.mean(0).reshape(h.shape[1], -1)                             # concat_k mean_t -> [B, K*D]
        return self.head(feat), e, m0, feat, h

    def forward(self, x):
        z, _, _, _, h = self.forward_parts(x)
        if self.train_mode == 'training':
            return z
        return z, h.detach()
