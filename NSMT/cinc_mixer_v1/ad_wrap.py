"""2026-11-02 (exp/f-lif-pop-v3; PROJECT_LOG AD pre-registration): the SpkMixer / Spikformer-style models of this folder as
reconstruction anomaly detectors.

Reconstructor(inner) keeps the token-mixing body unchanged and swaps only the two ends, like forecast_wrap.Forecaster:

    x [B, L, C]  ->  window normalisation (mean / std over L per channel; not learned, as Forecaster and TimesNet's AD)
                 ->  channel-mixing input [B, C, L]: one token holds the patch of every channel (the models' in_channels path,
                     token layout [B, N, C*P] channel-major); P = 8, stride 4, end padding -> N = 25 tokens for L = 100
    body         Forecaster.step_features: 'spkmix' = the SpkMixer block output, 'ssa' = the Spikformer-style SSA blocks,
                 both [T_h, B, N, D], averaged over T_h (the head input of every earlier task)
    head         per-token Linear(D -> S*C) (S = stride): token i rebuilds the steps [S*i, S*(i+1)); N*S >= L, the tail cut;
                 output [B, L, C] after undoing the window normalisation.

The channel-independent rows of the ETT forecaster would multiply the work by C (25-55); the AD models of TSLib (TimesNet,
Anomaly Transformer) mix channels too.
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from forecast_wrap import Forecaster
from model import myModel
from spk_mixer import SpkMixer
from head_mem import drop_classification_term, zero_linear              # 2026-11-03 H1

__all__ = ['Reconstructor']


class Reconstructor(Forecaster):
    def __init__(self, inner, embed_dim, seq_len, n_channels, bias=False):
        nn.Module.__init__(self)
        if isinstance(inner, SpkMixer):
            self.kind = 'spkmix'
        elif isinstance(inner, myModel) and inner.hippo_ssa:
            self.kind = 'ssa'
        else:
            raise NotImplementedError('Reconstructor wraps spkMixer or myModel --hippo_ssa only')
        if int(getattr(inner, 'in_channels', 1)) != int(n_channels):
            raise ValueError(f'inner model built for {inner.in_channels} channels, data has {n_channels}')
        self.inner = inner
        self.inner.head = nn.Identity()
        self.num_patches, self.seq_len, self.C = inner.num_patches, int(seq_len), int(n_channels)
        self.stride = inner.stride
        if self.num_patches * self.stride < self.seq_len:
            raise ValueError(f'{self.num_patches} tokens x stride {self.stride} < L {self.seq_len}')
        self.T = getattr(inner, 'T', None)
        self.head = nn.Linear(embed_dim, self.stride * self.C, bias=bias)
        trunc_normal_(self.head.weight, std=.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)
        if getattr(inner, '_pool_taps', None) is not None:                  # 2026-11-03 head input forms: classification only
            raise NotImplementedError('--head_pool is for the classification head only (head_pool.py)')
        self.tap = drop_classification_term(inner)                         # 2026-11-03 H1 (head_mem.py): None without --head_mem
        if self.tap is not None:                                           # + per-token Linear(D -> S*C) on the charge, zero init
            self.mem_readout = zero_linear(embed_dim, self.stride * self.C)

    def norm(self, x):
        means = x.mean(1, keepdim=True).detach()
        xc = x - means
        stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return xc / stdev, means, stdev

    def decode(self, f, c=None):
        """per-token features [B, N, D] (H1: and the per-token charge c [B, N, D]) -> normalised reconstruction [B, L, C]."""
        B, N, _ = f.shape
        y = self.head(f)
        if self.tap is not None:                                           # 2026-11-03 H1
            if c is None:
                raise ValueError('head_mem model: decode needs the charge')
            y = y + self.mem_readout(c)
        return y.reshape(B, N * self.stride, self.C)[:, :self.seq_len]

    def charge(self):
        """2026-11-03 H1: the last forward's per-token charge [B, N, D] (T_h mean), None without --head_mem."""
        return None if self.tap is None else self.tap.charge.mean(0)

    def forward(self, x):
        xn, means, stdev = self.norm(x)
        f = self.features(xn.permute(0, 2, 1))                             # [B, N, D]
        return self.decode(f, self.charge()) * stdev + means
