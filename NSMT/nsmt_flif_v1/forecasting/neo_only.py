"""Stage 4 (MODI_PROJECT_LOG 2026-10-18): the neocortex path alone as a forecaster, and a linear baseline.

NeoForecaster -- no hippocampus, no auxiliary decoder; the L1 task loss trains the neocortex directly.

    window normalisation (as myModel)  ->  channel-independent rows [B*C, L]
    tokens:  unit 'step'  one raw time step per token (P = 1, L tokens)
             unit 'patch' patches of P with stride P//2 and end padding (as myModel, N tokens)
    neocortex: ours.Embedding (Linear -> BN -> neuron) + ours.TemporalBlock (Spiking MLP x M), neuron spikeDE LIF / f-LIF
      align 'on'   the token axis IS the spiking time axis: the neuron state runs along series time (T = tokens)
      align 'off'  control: tokens folded into the batch, each token presented T_s times (the hippocampus repetition
                   scheme), the repetition axis averaged; no neuron state crosses tokens -- time order reaches the
                   model only through the head's position-specific weights
    readout:  'flat'  Linear(tokens * D -> H) on the per-token spike outputs (the NSMT head)
              'last'  Linear(D -> H) on the last layer's membrane potential after the LAST step (align 'on' only): the
                      forecast must come from what the neuron dynamics carried forward
    output [B, H, C] after undoing the window normalisation (train.py / test.py take 3-D outputs).

LinearForecaster -- window normalisation + channel-independent Linear(L -> H) + undo.
"""
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import trunc_normal_

from ours import Embedding, TemporalBlock
from layers import NeoNeuron
from neurons import LIFNeuron, FLIFNeuron

__all__ = ['NeoForecaster', 'LinearForecaster']


class _KeepLastState:
    """Keeps the membrane state returned by `integrate` at the latest step, WITH its graph (the neurons.py aux
    'state' is detached). After a forward over T steps, `last_state` is the membrane after step T-1: [B, D, K]."""
    def integrate(self, t, u, F, x, history, oracle_p=None):
        out = super().integrate(t, u, F, x, history, oracle_p)
        self.last_state = out[0]
        return out


class LIFKeep(_KeepLastState, LIFNeuron):
    pass


class FLIFKeep(_KeepLastState, FLIFNeuron):
    pass


def keep_neuron(kind, size, T, layout, tau=2., alpha=.5, method='pred', step=1., memory=0):
    """layers.build_neo_neuron with the state-keeping classes (same computation, bitwise)."""
    common = dict(tau=tau, threshold=1., surrogate_scale=5., step=step, max_length=int(T))
    if kind == 'lif':
        n = LIFKeep(size, **common)
    elif kind == 'flif':
        n = FLIFKeep(size, alpha=alpha, method=method, memory=(int(memory) if memory else None), **common)
    else:
        raise ValueError(f"neoOnly needs --neo_neuron lif or flif, got {kind!r}")
    return NeoNeuron(n, layout)


def normalise(x):
    means = x.mean(1, keepdim=True).detach()
    x = x - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    return x / stdev, means, stdev


class NeoForecaster(nn.Module):
    def __init__(self, seq_len, pred_len, neo_unit='step', align='on', neo_readout='flat', align_Ts=4, patch_size=8,
                 embed_dim=64, time_num_layers=2, bias=False, tau=2.0, neo_neuron='flif', neo_tau=2.0, neo_alpha=0.5,
                 fde_method='pred', fde_step=1.0, fde_memory=0, **kwargs):
        super().__init__()
        if neo_unit not in ('step', 'patch') or align not in ('on', 'off') or neo_readout not in ('flat', 'last'):
            raise ValueError('neo_unit in {step, patch}, align in {on, off}, neo_readout in {flat, last}')
        if align == 'off' and neo_readout == 'last':
            raise ValueError("readout 'last' needs align 'on' (an independent last token sees only itself)")
        self.unit, self.align, self.readout, self.align_Ts = neo_unit, align, neo_readout, int(align_Ts)
        self.seq_len, self.pred_len, self.D = seq_len, pred_len, embed_dim
        self.hippo_only = False                                    # for the shared evaluators
        self.train_mode = 'training'
        if neo_unit == 'step':
            self.P, self.stride, self.pad = 1, 1, None
            self.tokens = seq_len
        else:
            self.P, self.stride = patch_size, patch_size // 2
            self.pad = nn.ReplicationPad1d((0, self.stride))
            self.tokens = int((seq_len - patch_size) / self.stride + 1) + 1
        self.T = self.tokens                                       # neocortex time steps when aligned (memory probe)
        steps = self.tokens if align == 'on' else self.align_Ts
        nk = dict(tau=neo_tau, alpha=neo_alpha, method=fde_method, step=fde_step, memory=fde_memory)
        self.encoding_neo = Embedding(num_patches=self.tokens, patch_size=self.P, embed_dim=embed_dim, stride=self.stride,
                                      pe=False, bias=bias, tau=tau, neuron=keep_neuron(neo_neuron, embed_dim, steps, 'DN', **nk))
        self.time_block = TemporalBlock(T=steps, num_layers=time_num_layers, patch_size=self.P, embed_dim=embed_dim,
                                        lif_bias=bias, tau=tau, mutual=False,
                                        neuron_fn=lambda: keep_neuron(neo_neuron, embed_dim, steps, 'ND', **nk))
        self.head = nn.Linear(self.tokens * embed_dim if neo_readout == 'flat' else embed_dim, pred_len, bias=bias)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):                                          # as myModel._init_weights
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _last_neuron(self):
        mods = list(self.time_block.Mem) if len(self.time_block.Mem) else []
        return (mods[-1].spk_neuron if mods else self.encoding_neo.emb_lif).neuron

    def tokenize(self, x):
        #  x: [B, L, C] normalised  ->  [B*C, tokens, P]
        xc = rearrange(x, 'b l c -> (b c) l')
        if self.unit == 'step':
            return xc.unsqueeze(-1)
        xc = self.pad(xc.unsqueeze(1))[:, 0]
        return xc.unfold(-1, self.P, self.stride)

    def forward(self, x):
        B, L, C = x.shape
        x, self.means, self.stdev = normalise(x)
        tok = self.tokenize(x)                                     # [BC, N, P]
        BC, N, P = tok.shape
        if self.align == 'on':
            inp = tok.transpose(0, 1).unsqueeze(2)                 # [N, BC, 1, P]: token axis = spiking time
            out = self.time_block(self.encoding_neo(inp))          # [N, BC, 1, D]
            feats = out[:, :, 0, :].transpose(0, 1)                # [BC, N, D]
        else:
            inp = tok.reshape(1, BC * N, 1, P).repeat(self.align_Ts, 1, 1, 1)   # [T_s, BC*N, 1, P]: tokens independent
            out = self.time_block(self.encoding_neo(inp))          # [T_s, BC*N, 1, D]
            feats = out[:, :, 0, :].mean(0).reshape(BC, N, -1)     # [BC, N, D]
        if self.readout == 'flat':
            z = self.head(feats.reshape(BC, -1))
        else:
            z = self.head(self._last_neuron().last_state[:, :, 0])  # membrane after the last step, [BC, D]
        z = rearrange(z, '(b c) h -> b h c', b=B)
        return z * self.stdev + self.means


class LinearForecaster(nn.Module):
    def __init__(self, seq_len, pred_len, **kwargs):
        super().__init__()
        self.lin = nn.Linear(seq_len, pred_len)
        self.seq_len, self.pred_len = seq_len, pred_len                 # train.py reads model.pred_len
        self.hippo_only = True                                     # no neocortex: the memory probe skips it
        self.train_mode = 'training'
        self.T = 1

    def forward(self, x):
        x, means, stdev = normalise(x)
        z = self.lin(x.transpose(1, 2)).transpose(1, 2)            # [B, H, C]
        return z * stdev + means
