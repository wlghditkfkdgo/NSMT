"""MODI 2026-10-27: the CinC models as ETT forecasters (generalization batch).

Forecaster(inner) keeps the token-mixing body of a classification model of this folder unchanged and swaps only the input
and output ends for the NSMT forecasting ones (nsmt_flif_v1/forecasting/ours.py):

    x [B, L, C]  ->  window normalisation (mean / std over L per channel, as ours.myModel; not learned)
                 ->  channel-independent rows [B*C, 1, L]  ->  the inner model's patching (P, stride P/2, end padding: N tokens)
    body         neoMixer: embedding + blocks over the token axis (T = N); token spikes h [N, B*C, 1, D]
                 myModel + hippo_ssa (Spikformer-style): each patch repeated T_h times, SSA blocks; spikes [T_h, B*C, N, D],
                 averaged over T_h (the classification readout also averages over T_h)
                 myModel + hippo_xattn (dual path, MODI 2026-10-29): the same hippocampus with the T = N neocortex output as
                 keys / values ('replace' = [NSMT hippocampus T=4], 'cat' = B form), averaged over T_h like 'ssa'
    head         Linear(N * D -> H) on the flattened per-token spikes (the NSMT forecasting head); output [B, H, C] after
                 undoing the window normalisation.

The inner classification head is replaced by nn.Identity (unused, no parameters). Loss / metrics are in train_forecast.py.
MODI 2026-11-04 (H6, head test): head_kind 'h6' = a linear shortcut Linear(L -> H) on the normalised input row plus a small head
on the body: per-token Linear(D -> d) applied to every binary T_h step (identical to applying it to the T_h mean: linear, no bias
here; syops then counts it as spike-driven additions), averaged over T_h, flattened, Linear(N * d -> H). body_off = True drops the
body term (shortcut alone) to measure the body's share of a trained model. Default 'flat': unchanged.
The dual-path auxiliary decoder (weak_decoder) is never called here (plain forecasting loss), so it gets no gradient.
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from neo_mixer import NeoMixer
from model import myModel

__all__ = ['Forecaster']


class Forecaster(nn.Module):
    def __init__(self, inner, embed_dim, pred_len, bias=False, head_kind='flat', head_dim=8, seq_len=None):
        super().__init__()
        if isinstance(inner, NeoMixer):
            self.kind = 'mixer'
        elif isinstance(inner, myModel) and inner.hippo_ssa:
            self.kind = 'ssa'
        elif isinstance(inner, myModel) and inner.hippo_xattn and not (inner.no_neo or inner.hippo_only):
            self.kind = 'dual'                                             # MODI 2026-10-29
        else:
            raise NotImplementedError('Forecaster wraps neoMixer, myModel --hippo_ssa or myModel --hippo_xattn only')
        self.inner = inner
        self.inner.head = nn.Identity()
        self.num_patches, self.pred_len = inner.num_patches, int(pred_len)
        self.T = getattr(inner, 'T', None)
        self.head_kind, self.body_off = head_kind, False
        if head_kind == 'flat':
            self.head = nn.Linear(embed_dim * self.num_patches, self.pred_len, bias=bias)
            heads = [self.head]
        elif head_kind == 'h6':                                            # MODI 2026-11-04
            self.proj = nn.Linear(embed_dim, head_dim, bias=bias)
            self.head = nn.Linear(head_dim * self.num_patches, self.pred_len, bias=bias)
            self.shortcut = nn.Linear(int(seq_len), self.pred_len, bias=bias)
            heads = [self.proj, self.head, self.shortcut]
        else:
            raise ValueError('head_kind in {flat, h6}')
        for h in heads:
            trunc_normal_(h.weight, std=.02)
            if h.bias is not None:
                nn.init.constant_(h.bias, 0)

    @property
    def train_mode(self):
        return self.inner.train_mode

    @train_mode.setter
    def train_mode(self, v):
        self.inner.train_mode = v

    def step_features(self, z):
        """'ssa' / 'dual': z [B', 1, L] -> per-step hippocampus spikes [T_h, B', N, D] (the head input before the T_h mean)."""
        m = self.inner
        patched, neo_in = m._tokens_mc(z)                                  # [T_h, B', N, P], [N, B', 1, P]
        xh = m.encoding(patched, return_pre=(getattr(m, 'res_mode', 'iand') == 'ms'))   # MODI 2026-11-08: (spikes, stream) with ms
        x_neo = m.neo_forward(neo_in, z.clone().detach())[1] if self.kind == 'dual' else None   # [N, B', 1, D]
        return m.hippo_blocks(xh, x_neo)

    def features(self, z):
        """z [B', 1, L] -> per-token spikes [B', N, D]."""
        if self.kind == 'mixer':
            h = self.inner.forward_parts(z)[4]                              # [N, B', 1, D]
            return h[:, :, 0].transpose(0, 1)
        return self.step_features(z).mean(0)                               # [B', N, D]

    def forward(self, x):
        B, L, C = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev
        z = x.permute(0, 2, 1).reshape(B * C, 1, L)
        if self.head_kind == 'h6':
            y = self.shortcut(z[:, 0])
            if not self.body_off:
                p = self.proj(self.features(z)) if self.kind == 'mixer' else self.proj(self.step_features(z)).mean(0)   # [B', N, d]
                y = y + self.head(p.reshape(B * C, -1))
            y = y.reshape(B, C, self.pred_len).permute(0, 2, 1)
            return y * stdev + means
        f = self.features(z)
        y = self.head(f.reshape(B * C, -1)).reshape(B, C, self.pred_len).permute(0, 2, 1)   # [B, H, C]
        return y * stdev + means
