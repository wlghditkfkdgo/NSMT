"""MODI 2026-10-27: the CinC models as ETT forecasters (generalization batch).

Forecaster(inner) keeps the token-mixing body of a classification model of this folder unchanged and swaps only the input
and output ends for the NSMT forecasting ones (nsmt_flif_v1/forecasting/ours.py):

    x [B, L, C]  ->  window normalisation (mean / std over L per channel, as ours.myModel; not learned)
                 ->  channel-independent rows [B*C, 1, L]  ->  the inner model's patching (P, stride P/2, end padding: N tokens)
    body         neoMixer: embedding + blocks over the token axis (T = N); token spikes h [N, B*C, 1, D]
                 myModel + hippo_ssa (Spikformer-style): each patch repeated T_h times, SSA blocks; spikes [T_h, B*C, N, D],
                 averaged over T_h (the classification readout also averages over T_h)
    head         Linear(N * D -> H) on the flattened per-token spikes (the NSMT forecasting head); output [B, H, C] after
                 undoing the window normalisation.

The inner classification head is replaced by nn.Identity (unused, no parameters). Loss / metrics are in train_forecast.py.
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from neo_mixer import NeoMixer
from model import myModel

__all__ = ['Forecaster']


class Forecaster(nn.Module):
    def __init__(self, inner, embed_dim, pred_len, bias=False):
        super().__init__()
        if isinstance(inner, NeoMixer):
            self.kind = 'mixer'
        elif isinstance(inner, myModel) and inner.hippo_ssa:
            self.kind = 'ssa'
        else:
            raise NotImplementedError('Forecaster wraps neoMixer or myModel --hippo_ssa only')
        self.inner = inner
        self.inner.head = nn.Identity()
        self.num_patches, self.pred_len = inner.num_patches, int(pred_len)
        self.T = getattr(inner, 'T', None)
        self.head = nn.Linear(embed_dim * self.num_patches, self.pred_len, bias=bias)
        trunc_normal_(self.head.weight, std=.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    @property
    def train_mode(self):
        return self.inner.train_mode

    @train_mode.setter
    def train_mode(self, v):
        self.inner.train_mode = v

    def features(self, z):
        """z [B', 1, L] -> per-token spikes [B', N, D]."""
        if self.kind == 'mixer':
            h = self.inner.forward_parts(z)[4]                              # [N, B', 1, D]
            return h[:, :, 0].transpose(0, 1)
        m = self.inner
        patched, _ = m._tokens_mc(z)                                       # [T_h, B', N, P]
        xh = m.encoding(patched)
        for blk in m.data_block:
            xh = blk(xh, None, self_kv=m.hippo_only)
        return xh.mean(0)                                                  # [B', N, D]

    def forward(self, x):
        B, L, C = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev
        z = x.permute(0, 2, 1).reshape(B * C, 1, L)
        f = self.features(z)
        y = self.head(f.reshape(B * C, -1)).reshape(B, C, self.pred_len).permute(0, 2, 1)   # [B, H, C]
        return y * stdev + means
