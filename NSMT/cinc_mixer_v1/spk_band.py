"""STFT band input for the SpkMixer (2026-11-16, PROJECT_LOG, exp/f-lif-pop-v3; user request 11-13 / 11-16).

The frames are the model's patches (window = patch length P, hop = stride, so frame n holds the samples of token n). Each frame:
periodic Hann window -> real FFT (orthonormal) -> P // 2 + 1 bins -> 4 bands with the edges ceil(nb / 8), ceil(nb / 4), ceil(nb / 2)
(band 0 = the lowest bins, including bin 0). A band's values are the real parts of its bins followed by their imaginary parts;
the imaginary parts of bin 0 and of the Nyquist bin are always 0 and left out, so the band sizes add up to P (P even).
The same definition as the other session's test of 11-13 (exp/nsmt-flif-neo 330d43b9b, model.stft_bands); re-implemented here.

  A  GroupBandEmbedding : the D channels are 4 groups of D / 4; group k reads band k through its own Linear(dim_k -> D / 4); the
                          BN (D channels) and the neuron layer (D units, each fires by itself) are those of the default embedding.
  B  MemberBandEmbedding: every unit keeps its 4 members; band i goes through its own Linear(dim_i -> D) to the member with the
                          i-th largest time constant (lowest band -> slowest member); BN over the D * 4 (unit, member) channels;
                          one soma per unit fires (PopulationFLIF with per-member currents).
Both take a list of 4 tensors [T, B, N', dim_i] and return spikes [T, B, N', D] (T = the spiking axis of the caller).
"""
import math

import torch
import torch.nn as nn

__all__ = ['band_edges', 'stft_band_dims', 'stft_values', 'member_order', 'GroupBandEmbedding', 'MemberBandEmbedding', 'BANDS']

BANDS = ('none', 'group', 'member')


def band_edges(n, n_bands=4):
    return [0] + [int(math.ceil(n / 2 ** (n_bands - 1 - i))) for i in range(n_bands - 1)] + [n]


def _band_index(patch_size, n_bands=4):
    """per band: (bins whose real part is used, bins whose imaginary part is used)."""
    if patch_size % 2 or patch_size < 2 * n_bands:
        raise ValueError(f'STFT bands need an even patch size >= {2 * n_bands}, got {patch_size}')
    e = band_edges(patch_size // 2 + 1, n_bands)
    out = []
    for i in range(n_bands):
        ks = list(range(e[i], e[i + 1]))
        out.append((ks, [k for k in ks if k not in (0, patch_size // 2)]))
    return out


def stft_band_dims(patch_size, n_bands=4):
    return [len(re) + len(im) for re, im in _band_index(patch_size, n_bands)]


def stft_values(tok, n_bands=4):
    """tok [..., P] (one frame per position, single input channel) -> list of n_bands tensors [..., dim_i]."""
    P = tok.shape[-1]
    idx = _band_index(P, n_bands)
    win = torch.hann_window(P, periodic=True, device=tok.device, dtype=torch.float32)
    X = torch.fft.rfft(tok.to(torch.float32) * win, dim=-1, norm='ortho')
    return [torch.cat([X[..., re].real, X[..., im].imag], dim=-1).contiguous() for re, im in idx]


def member_order(taus):
    """member that reads band i (i = 0 lowest): members by decreasing time constant (stable for equal values)."""
    t = [float(v) for v in taus]
    return sorted(range(len(t)), key=lambda k: -t[k])


def _check(xs, dims):
    if len(xs) != len(dims) or any(x.ndim != 4 or x.shape[-1] != d for x, d in zip(xs, dims)):
        raise ValueError(f'band embedding built for sizes {dims}, got {[tuple(x.shape) for x in xs]}')


class GroupBandEmbedding(nn.Module):
    def __init__(self, in_dims, embed_dim, bn, neuron, bias=False):
        super().__init__()
        K = len(in_dims)
        if embed_dim % K:
            raise ValueError(f'embed_dim {embed_dim} is not divisible by {K} groups')
        self.in_dims, self.embed_dim, self.group = [int(d) for d in in_dims], embed_dim, embed_dim // K
        self.emb_linear = nn.ModuleList([nn.Linear(d, self.group, bias=bias) for d in self.in_dims])
        self.emb_bn = bn                                                    # BatchNorm1d(embed_dim)
        self.emb_lif = neuron                                               # layout 'DN': [T, B, D, N'] in and out

    def current(self, xs):
        _check(xs, self.in_dims)
        T, B, N1 = xs[0].shape[:3]
        cur = torch.cat([lin(x) for lin, x in zip(self.emb_linear, xs)], dim=-1)             # [T, B, N', D], group-major
        return self.emb_bn(cur.reshape(T * B * N1, self.embed_dim)).reshape(T, B, N1, self.embed_dim)

    def forward(self, xs):
        cur = self.current(xs)
        return self.emb_lif(cur.permute(0, 1, 3, 2).contiguous()).transpose(-1, -2).contiguous()


class MemberBandEmbedding(nn.Module):
    def __init__(self, in_dims, embed_dim, neuron, order, bias=False):
        super().__init__()
        self.in_dims, self.embed_dim, self.order = [int(d) for d in in_dims], embed_dim, [int(k) for k in order]
        K = len(self.in_dims)
        if sorted(self.order) != list(range(K)):
            raise ValueError(f'member order {order} is not a permutation of {K} members')
        self.emb_linear = nn.ModuleList([nn.Linear(d, embed_dim, bias=bias) for d in self.in_dims])    # index = band
        self.emb_bn = nn.BatchNorm1d(embed_dim * K)                          # channel = unit * K + member
        self.emb_lif = neuron                                               # NeoNeuron around a PopulationFLIF with a soma

    def current(self, xs):
        _check(xs, self.in_dims)
        T, B, N1 = xs[0].shape[:3]
        K = len(xs)
        per = [None] * K
        for i in range(K):
            per[self.order[i]] = self.emb_linear[i](xs[i])
        cur = torch.stack(per, dim=-1)                                       # [T, B, N', D, K], member on the last axis
        return self.emb_bn(cur.reshape(T * B * N1, self.embed_dim * K)).reshape(T, B, N1, self.embed_dim, K)

    def forward(self, xs):
        return self.emb_lif(self.current(xs).contiguous())                   # [T, B, N', D]
