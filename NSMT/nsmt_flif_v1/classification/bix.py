"""BiX (MODI_PROJECT_LOG 2026-10-21): token-axis cross-attention between the forward and the reverse neocortex streams.

The user's idea: run the token sequence through the neocortex network forwards (F) and backwards (R), then attend
between the two streams along the token axis, then MLP and head. Every neuron keeps T = N (spiking step = token).

Tensors are [N, B, K, D]: N tokens (= spiking steps), K streams (1 for the Neo-mixer; 4 for the soma-less stream model,
stream_mixer.py), D channels. Queries come from F only, keys / values from R only.

    q = neuron(BN(Linear(F)))      k, v = neuron(BN(Linear(R)))          (Linear shared across streams, BN per stream)
    form 'head'   : scores_h[t, t'] = q_h[t] . k_h[t'] * D^-0.5 + relative bias_h(t' - t)        (heads, as SSA_rel_scl)
    form 'channel': scores_d[t, t'] = q[t, d] k[t', d] * D^-0.5   (one T x T' map per channel d, no bias: the user's
                    original form; with 0/1 spikes it reduces to q[t, d] * sum_t' k[t', d] v[t', d] -- check c26)
    z = scores @ v -> BN -> attention neuron -> proj: neuron(BN(Linear(.)))

BN in front of the attention neuron makes every neuron a (Linear or attention) + BN + neuron set (BN audit 10-21: without
it the Spikformer-style attention neuron fired 0.07-0.4 % at init); the Spikformer control spk_ssa_abn gets the same BN.
"""
import torch
import torch.nn as nn

from utils import RelBias1DDeterministicFn

__all__ = ['TokenXAttn', 'StreamSpikeLinear']


class StreamSpikeLinear(nn.Module):
    """Linear (shared across the K streams) -> BN over K*D channels (per-stream statistics) -> neuron, on [N, B, K, Din].
    bn_layout 'flat' = BatchNorm1d on [N*B*K... reshaped to (-1, K*D)] (NeoLayer's call), 'ncl' = on (N*B, K*D, 1)
    (SpikLinearLayer / Embedding's call); with K = 1 each reproduces the corresponding module call exactly (check c28).
    gamma: optional [K*Dout] init of the BN weight (tau calibration)."""
    def __init__(self, din, dout, K, neuron, bias=False, bn_layout='flat', gamma=None):
        super().__init__()
        self.K, self.dout, self.bn_layout = int(K), int(dout), bn_layout
        self.linear = nn.Linear(din, dout, bias=bias)
        self.bn = nn.BatchNorm1d(self.K * self.dout)
        if gamma is not None:
            with torch.no_grad():
                self.bn.weight.copy_(gamma)
        self.neuron = neuron

    def bn_apply(self, y):                                                  # y [N, B, K, D]
        N, B = y.shape[:2]
        C = self.K * self.dout
        if self.bn_layout == 'flat':
            return self.bn(y.reshape(-1, C)).reshape(N, B, self.K, self.dout)
        return self.bn(y.reshape(N * B, C, 1)).reshape(N, B, self.K, self.dout)

    def forward(self, x):
        return self.neuron(self.bn_apply(self.linear(x)).contiguous())


class TokenXAttn(nn.Module):
    """Cross-attention along the token (= spiking time) axis: queries from stream F, keys / values from stream R.
    neuron_fn() -> a module mapping [N, B, K, D] to spikes of the same shape (one call per neuron layer);
    gamma: optional [K*D] BN weight init used for every BN in the module (tau calibration)."""
    def __init__(self, dim, n_tokens, K=1, heads=8, form='head', neuron_fn=None, bias=False, gamma=None):
        super().__init__()
        if form not in ('head', 'channel'):
            raise ValueError("form must be 'head' or 'channel'")
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        self.dim, self.N, self.K, self.heads, self.form = int(dim), int(n_tokens), int(K), int(heads), form
        self.scale = dim ** -0.5                                            # as SSA_rel_scl
        mk = lambda: StreamSpikeLinear(dim, dim, K, neuron_fn(), bias=bias, bn_layout='flat', gamma=gamma)
        self.q, self.k, self.v = mk(), mk(), mk()
        self.attn_bn = nn.BatchNorm1d(self.K * dim)
        if gamma is not None:
            with torch.no_grad():
                self.attn_bn.weight.copy_(gamma)
        self.attn_neuron = neuron_fn()
        self.proj = mk()
        if form == 'head':                                                  # relative position bias, zero init (as SSA_rel_scl)
            self.relative_bias_table = nn.Parameter(torch.zeros(2 * self.N - 1, self.heads))
            pos = torch.arange(self.N)
            self.register_buffer('rel_pos_idx', (pos[None, :] - pos[:, None] + self.N - 1).long())
        self.last_scores = None                                             # set when keep_scores (analysis / checks)
        self.keep_scores = False

    def scores_and_z(self, q, k, v):
        """q, k, v [N, B, K, D] spikes -> z [N, B, K, D] (before the attention BN)."""
        N, B, K, D = q.shape
        if self.form == 'head':
            h, dh = self.heads, D // self.heads
            qh = q.permute(1, 2, 0, 3).reshape(B, K, N, h, dh).transpose(2, 3)      # [B, K, h, N, dh]
            kh = k.permute(1, 2, 0, 3).reshape(B, K, N, h, dh).transpose(2, 3)
            vh = v.permute(1, 2, 0, 3).reshape(B, K, N, h, dh).transpose(2, 3)
            s = (qh @ kh.transpose(-1, -2)) * self.scale                            # [B, K, h, N, N']
            rb = RelBias1DDeterministicFn.apply(self.relative_bias_table, self.rel_pos_idx)   # [N, N', h]
            s = s + rb.permute(2, 0, 1)[None, None]
            z = (s @ vh).transpose(2, 3).reshape(B, K, N, D).permute(2, 0, 1, 3)   # [N, B, K, D]
        else:                                                               # one T x T' map per channel: [B, K, N, N', D]
            qc, kc, vc = (t.permute(1, 2, 0, 3) for t in (q, k, v))                # [B, K, N, D]
            s = torch.einsum('bktd,bksd->bktsd', qc, kc) * self.scale
            z = torch.einsum('bktsd,bksd->bktd', s, vc).permute(2, 0, 1, 3)        # [N, B, K, D]
        if self.keep_scores:
            self.last_scores = s.detach()
        return z

    def forward(self, f, r):
        q, k, v = self.q(f), self.k(r), self.v(r)
        z = self.scores_and_z(q, k, v).contiguous()
        N, B = z.shape[:2]
        z = self.attn_bn(z.reshape(-1, self.K * self.dim)).reshape(N, B, self.K, self.dim)
        return self.proj(self.attn_neuron(z.contiguous()))
