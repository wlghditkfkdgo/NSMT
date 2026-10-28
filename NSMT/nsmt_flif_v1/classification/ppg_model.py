"""MODI 2026-10-28: the CinC models of this folder as streaming (causal) heart-rate regressors for PPG-DaLiA (ppg_data.py).

x [B, 49920, 6] -> [B, 6, L] -> the inner model's multichannel patching (P = 256 samples = 4 s, stride 128 = 2 s, end padding):
N = 390 tokens; token t covers samples [128 t, 128 t + 256) and ends at 128 (t + 2). Label k (mean heart rate of the 8 s ending
at sample 128 (k + 1)) is read from token k - 1, the last token that ends there; label 0 from token 0 (2 s look-ahead for that
one label of 390). Every model only looks back:
  kind 'ssa'   myModel --hippo_ssa (Spikformer-style, T_h repeats): token attention restricted to the causal band attn_window
               (W tokens: token i sees j in [i - W + 1, i]; W = 1 sees itself only; -1 = all past) in every SSA module
  kind 'b'     myModel --hippo_xattn --hippo_xattn_mode cat (Spikformer-style + neocortex keys / values; the B form): the same
               band on the own and the neocortex terms; the neocortex runs forward over the tokens (T = N) or memoryless
  kind 'mixer' neoMixer without --mixer_bidir (forward T = N stacks only)
Head: Linear(D -> 1, no bias) on the binary token spikes of every repeat step, averaged over the T_h steps (= the head of the
averaged spikes, but the head input stays 0/1 so the energy counter books it as AC; the ETT defect of 10-27).
GRURegressor: non-spiking reference, one GRU layer over the same 390 tokens (256 x 6 inputs each), Linear(384 -> 1) per token.
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from neo_mixer import NeoMixer
from model import myModel

__all__ = ['TokenRegressor', 'GRURegressor', 'label_tokens']


def label_tokens(n_tokens, n_labels=390):
    idx = torch.arange(n_labels) - 1
    idx[0] = 0
    if int(idx[-1]) >= n_tokens:
        raise ValueError(f'{n_labels} labels need {int(idx[-1]) + 1} tokens, got {n_tokens}')
    return idx


class TokenRegressor(nn.Module):
    def __init__(self, inner, embed_dim, attn_window=0, n_labels=390):
        super().__init__()
        if isinstance(inner, NeoMixer):
            if inner.mixer_bidir:
                raise ValueError('PPG streaming: neoMixer without --mixer_bidir')
            self.kind = 'mixer'
        elif isinstance(inner, myModel) and inner.hippo_ssa:
            self.kind = 'ssa'
        elif isinstance(inner, myModel) and inner.hippo_xattn and inner.hippo_xattn_mode == 'cat' and not inner.neo_bidir:
            self.kind = 'b'
        else:
            raise NotImplementedError('TokenRegressor wraps neoMixer (forward), myModel --hippo_ssa or the B form')
        if self.kind == 'mixer' and attn_window:
            raise ValueError('attn_window is for the attention models')
        if self.kind != 'mixer' and not attn_window:
            raise ValueError('the attention models need attn_window (W > 0 or -1): PPG runs are causal')
        self.inner = inner
        self.inner.head = nn.Identity()
        self.attn_window = int(attn_window)
        for mod in inner.modules():
            if hasattr(mod, 'relative_bias_table') or mod.__class__.__name__ == 'SSA_rel_scl':
                mod.attn_window = self.attn_window
        self.num_patches = inner.num_patches
        self.T = getattr(inner, 'T', None)
        self.register_buffer('lab_idx', label_tokens(self.num_patches, n_labels), persistent=False)
        self.head = nn.Linear(embed_dim, 1, bias=False)
        trunc_normal_(self.head.weight, std=.02)

    @property
    def train_mode(self):
        return self.inner.train_mode

    @train_mode.setter
    def train_mode(self, v):
        self.inner.train_mode = v

    def spikes(self, z):
        """z [B, C, L] -> token spikes [S, B, N, D] (S = T_h repeat steps, 1 for the mixer)."""
        m = self.inner
        if self.kind == 'mixer':
            h = m.forward_parts(z)[4]                                     # [N, B, 1, D]
            return h[:, :, 0].unsqueeze(0).transpose(1, 2)               # [1, B, N, D]
        patched, neo_in = m._tokens_mc(z)                                 # [T_h, B, N, C*P], [N, B, 1, C*P]
        xh = m.encoding(patched)
        x_neo = m.neo_forward(neo_in, z)[1] if self.kind == 'b' else None
        for blk in m.data_block:
            xh = blk(xh, x_neo, self_kv=False)
        return xh                                                         # [T_h, B, N, D]

    def forward(self, x):
        s = self.spikes(x.permute(0, 2, 1).contiguous())
        y = self.head(s).squeeze(-1).mean(0)                              # [B, N]
        return y[:, self.lab_idx]                                         # [B, 390]


class GRURegressor(nn.Module):
    def __init__(self, patch_size=256, channels=6, hidden=384, n_labels=390, seq_len=49920):
        super().__init__()
        self.patch_size, self.stride = patch_size, patch_size // 2
        self.pad = nn.ReplicationPad1d((0, self.stride))
        self.num_patches = int((seq_len - patch_size) / self.stride + 1) + 1
        self.gru = nn.GRU(patch_size * channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.register_buffer('lab_idx', label_tokens(self.num_patches, n_labels), persistent=False)
        self.train_mode = 'training'

    def forward(self, x):
        z = self.pad(x.permute(0, 2, 1)).unfold(-1, self.patch_size, self.stride)        # [B, C, N, P]
        B, C, N, P = z.shape
        h, _ = self.gru(z.permute(0, 2, 1, 3).reshape(B, N, C * P))
        return self.head(h).squeeze(-1)[:, self.lab_idx]
