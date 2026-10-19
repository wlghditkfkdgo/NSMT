"""Stage 7 (MODI_PROJECT_LOG 2026-10-19): the cross-attention module replaced by the neocortex network.

Patch tokens, T = N alignment everywhere: the 11 patches (32 samples, stride 16, end padding -- as the ECG model) are the
11 spiking time steps of every neuron; there is no repetition axis and no second (hippocampus) path.

    beat [B, 1, 187] -> patches [N, B, 1, P] (token axis first)
    embedding: Linear(P -> D) -> BN -> spikingjelly LIF over the token axis           (fixed across conditions)
    L blocks:  x = x * (1 - mixer(x))        mixer = the neocortex network: `mixer_layers` NeoLayers (Linear -> BN ->
                                              neuron over the token axis); bidirectional = a second stack over the
                                              reversed token axis, flipped back and fused by OR (as stage 6);
                                              mixer 'none' = no cross-token mixing (control)
               x = x * (1 - MLP(x))          two SpikLinearLayers with spikingjelly LIF over the token axis (fixed)
    head: Linear(D -> classes) on the token mean (the ECG model's readout); loss = CAL-weighted CE only (no auxiliary loss)

Cross-token information can only travel through the mixer's neuron state (and the final mean).
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate

from layers import SpikLinearLayer, build_neo_neuron
from model import Embedding, NeoLayer

__all__ = ['NeoMixer']

MIXERS = ('none', 'lif', 'flif', 'pop_flif')


class MixerStack(nn.Module):
    """`n` NeoLayers over the token axis with the chosen spikeDE neuron (built by layers.build_neo_neuron)."""
    def __init__(self, dim, n, neuron_fn, bias=False):
        super().__init__()
        self.Mem = nn.ModuleList([NeoLayer(dim, bias=bias, neuron=neuron_fn()) for _ in range(n)])

    def forward(self, x):
        for l in self.Mem:
            x = l(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, mixer, mixer_bidir, mixer_layers, mlp_ratio, neo_kw, T, bias=False, tau=2.0):
        super().__init__()
        self.mixer_kind, self.bidir = mixer, bool(mixer_bidir)
        if mixer != 'none':
            fn = lambda: build_neo_neuron(mixer, dim, T, 'ND', **neo_kw)
            self.mixer = MixerStack(dim, mixer_layers, fn, bias)
            if self.bidir:
                self.mixer_rev = MixerStack(dim, mixer_layers, fn, bias)
        hidden = int(dim * mlp_ratio)
        self.fc1 = SpikLinearLayer(dim, hidden, tau=tau, lif_bias=bias)       # spikingjelly LIF over the token axis
        self.fc2 = SpikLinearLayer(hidden, dim, tau=tau, lif_bias=bias)

    def mix(self, x):
        if self.mixer_kind == 'none':
            return None
        m = self.mixer(x)
        if self.bidir:
            m = torch.maximum(m, self.mixer_rev(x.flip(0)).flip(0))
        return m

    def forward(self, x, mix_fn=None):
        m = self.mix(x)
        if mix_fn is not None:
            m = mix_fn(m)
        if m is not None:
            x = x * (1. - m)
        x = x * (1. - self.fc2(self.fc1(x)))
        return x


class NeoMixer(nn.Module):
    def __init__(self, seq_len=187, data_patch_size=32, embed_dim=384, num_classes=5, depths=1, mlp_ratios=1.0,
                 mixer='pop_flif', mixer_bidir=True, mixer_layers=2, neo_tau=2.0, neo_alpha=0.5, fde_method='pred',
                 fde_step=1.0, fde_memory=0, pop_taus=(4., 8., 16., 32.), bias=False, tau=2.0, train_mode='training',
                 in_channels=1, pop_soma_init=None, **kwargs):
        super().__init__()
        if mixer not in MIXERS:
            raise ValueError(f'mixer must be one of {MIXERS}')
        if mixer == 'none' and mixer_bidir:
            raise ValueError("mixer 'none' has no direction")
        self.train_mode = train_mode
        self.patch_size, self.stride = data_patch_size, data_patch_size // 2
        self.num_patches = int((seq_len - data_patch_size) / self.stride + 1) + 1
        self.T = self.num_patches                                          # T = N
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.mixer_kind, self.mixer_bidir, self.mixer_layers, self.depths = mixer, bool(mixer_bidir), int(mixer_layers), int(depths)
        self.no_neo = self.hippo_only = False                              # for the shared evaluators
        neo_kw = dict(tau=neo_tau, alpha=neo_alpha, method=fde_method, step=fde_step, memory=fde_memory, pop_taus=tuple(pop_taus),
                      pop_soma_init=pop_soma_init)
        self.in_channels = int(in_channels or 1)                           # stage 9: a token = P steps x C channels
        self.encoding = Embedding(num_patches=self.num_patches, patch_size=data_patch_size * self.in_channels, embed_dim=embed_dim, stride=self.stride,
                                  pe=False, bias=bias, tau=tau)            # spikingjelly LIF over the token axis
        self.blocks = nn.ModuleList([Block(embed_dim, mixer, mixer_bidir, mixer_layers, mlp_ratios, neo_kw, self.T, bias, tau)
                                     for _ in range(self.depths)])
        self.head = nn.Linear(embed_dim, num_classes, bias=bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def tokens(self, x):
        if x.dim() != 3:
            x = x.unsqueeze(1)
        B = x.shape[0]
        if self.in_channels == 1:                                          # unchanged single-channel path (stage 7)
            xp = self.padding_patch_layer(x[:, 0]).unfold(-1, self.patch_size, self.stride)   # [B, N, P]
            return xp.transpose(0, 1).unsqueeze(2).contiguous()                              # [N, B, 1, P]
        C = x.shape[1]                                                     # stage 9: [B, C, L] -> [N, B, 1, C*P]
        p = self.padding_patch_layer(x).unfold(-1, self.patch_size, self.stride)              # [B, C, N, P]
        p = p.permute(0, 2, 1, 3).reshape(B, p.shape[2], C * self.patch_size)
        return p.transpose(0, 1).unsqueeze(2).contiguous()

    def forward_parts(self, x, tok_fn=None, mix_fns=None):
        """Returns logits, embedding spikes [N, B, 1, D], first block's mixer output (None for 'none'), head input [B, D],
        and the last block's token spikes [N, B, 1, D]. tok_fn edits the token tensor; mix_fns[i] edits block i's mixer output."""
        tok = self.tokens(x)
        if tok_fn is not None:
            tok = tok_fn(tok)
        e = self.encoding(tok)                                             # [N, B, 1, D]
        h, m0 = e, None
        for i, blk in enumerate(self.blocks):
            fn = mix_fns[i] if mix_fns is not None else None
            if i == 0:
                m0 = blk.mix(h)
                if fn is not None:
                    m0 = fn(m0)
                if m0 is not None:
                    h = h * (1. - m0)
                h = h * (1. - blk.fc2(blk.fc1(h)))
            else:
                h = blk(h, fn)
        feat = h.mean(2).mean(0)                                           # token mean -> [B, D]
        return self.head(feat), e, m0, feat, h

    def forward(self, x):
        z, _, _, _, h = self.forward_parts(x)
        if self.train_mode == 'training':
            return z                                                       # plain logits: train.py uses CE only
        return z, h.detach()                                               # test.py's optional t-SNE feature
