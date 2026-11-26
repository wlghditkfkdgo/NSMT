"""Spike cross-attention fusion of the SpkMixer's two directional stacks (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG pre-registration
"neo-mixer fusion redesign, stage 1"; user decisions 11-26 / 11-27).

In a SpkBlock the forward stack gives spikes F and charges H_f, the reverse stack R and H_r, all [N, B', T, D] (token axis first,
the T repeats batch-like). Instead of the MUX fusion this module returns

    Q, K, V = F, R, R                                  ('direct', no parameters)
            = P_q(F), P_k(R), P_v(R)                   ('proj', P = Linear(D -> D) -> BN -> neuron over the token axis, Spikformer-like)
    A_s[b, t, n, n'] = sum_d Q[n, b, t, d] K[n', b, t, d]                 spike map, values 0..D           [B', T, N, N]
    M[b, n, n']     = (1 + rho[b, n, n']) / 2                             ('pc' only)                       [B', N, N]
        rho = Pearson correlation over D between the T-mean charges Hbar_f[n] and Hbar_r[n'] (one mask for the T repeats)
    Z = (A_s * M) V * D^-1/2  (A_s V without the mask)                                                     [N, B', T, D]
    out = neuron(BN(Z))     neuron over the token axis (N steps), 0/1                                       [N, B', T, D]

AttnMath is a parameter-free LEAF module that does the products and keeps an operation ledger `_ops = (AC, MAC)` of its last call
(totals over the batch), because syops and energy.SynOpRecorder only see nn.Linear / BatchNorm / neuron modules. syops does not
descend into a module that has its own hook, so the projections, BN and neuron live outside AttnMath (in SpkXAttn) and are counted
by their usual hooks. Ledger rules (Horowitz 45 nm: AC 0.9 pJ = one real addition, MAC 4.6 pJ = multiplication + addition):
    Q K^T with 0/1 Q, K       nnz(Q) * N  AC     (a query spike adds the key column over the N keys)
    P V with 0/1 V            nnz(V) * N  AC     (a value spike adds the map column into the N outputs)
    Pearson mask ('pc')       AC: 2 T N B' D (T means) + 4 N B' D (centring) + 2 B' N^2 (eps, 1 + rho)
                              MAC: 2 N B' D (1 / T) + 2 N B' D + 2 N B' (norms, sqrt) + B' N^2 D (dot products)
                                   + 3 B' N^2 (norm product, division, / 2) + B' T N^2 (mask x map)
    division and square root are counted as one MAC each (assumption); the D^-1/2 scale folds into the BN after it (not counted).
"""
import torch
import torch.nn as nn

__all__ = ['AttnMath', 'SpkXAttn', 'attn_syops_hook', 'MASKS']

MASKS = ('none', 'pc')


class AttnMath(nn.Module):
    """Parameter-free leaf: (Q, K, V, H_f, H_r) [N, B', T, D] -> Z [N, B', T, D] (pre-BN), with the op ledger."""
    def __init__(self, dim, mask='none'):
        super().__init__()
        if mask not in MASKS:
            raise ValueError(f'mask must be one of {MASKS}')
        self.dim, self.mask = int(dim), mask
        self.scale = self.dim ** -0.5
        self._ops = (0., 0.)
        self.last_mask = None                                                  # [B', N, N] of the last call (checks / probes)

    @staticmethod
    def pearson_mask(hf, hr, eps=1e-6):
        """hf, hr [N, B', T, D] -> (1 + rho) / 2 [B', N, N]; rho over D between the T-mean charges."""
        mf = hf.mean(2).transpose(0, 1)                                        # [B', N, D]
        mr = hr.mean(2).transpose(0, 1)
        cf = mf - mf.mean(-1, keepdim=True)
        cr = mr - mr.mean(-1, keepdim=True)
        num = cf @ cr.transpose(-1, -2)                                        # [B', N, N]
        den = cf.norm(dim=-1).unsqueeze(-1) * cr.norm(dim=-1).unsqueeze(-2) + eps
        return (1. + num / den) / 2.

    def forward(self, q, k, v, hf=None, hr=None):
        N, B, T, D = q.shape
        a = torch.einsum('nbtd,mbtd->btnm', q, k)                              # [B', T, N, N]
        mac = 0.
        ac = float(q.detach().count_nonzero()) * N + float(v.detach().count_nonzero()) * N
        if self.mask == 'pc':
            m = self.pearson_mask(hf, hr)
            self.last_mask = m.detach()
            a = a * m.unsqueeze(1)
            ac += 2. * T * N * B * D + 4. * N * B * D + 2. * B * N * N
            mac += 2. * N * B * D + 2. * N * B * D + 2. * N * B + float(B) * N * N * D + 3. * B * N * N + float(B) * T * N * N
        z = torch.einsum('btnm,mbtd->nbtd', a, v) * self.scale                 # [N, B', T, D]
        self._ops = (ac, mac)
        return z


def attn_syops_hook(module, inp, out):
    """syops custom hook for AttnMath: adds the ledger of the call (AC to [1], MAC to [2], both to [0])."""
    ac, mac = module._ops
    module.__syops__[0] += int(round(ac + mac))
    module.__syops__[1] += ac
    module.__syops__[2] += mac
    q = inp[0]
    module.__syops__[3] += float(q.detach().count_nonzero()) / max(1, q.numel()) * 100


class _Proj(nn.Module):
    """Linear(din -> dout) -> BN(dout) over every (token, batch, repeat) row -> neuron over the token axis, on [N, B', T, din]
    (the same computation as spk_mixer.FuseLinear; a copy here so that this file does not import spk_mixer)."""
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


class SpkXAttn(nn.Module):
    """F, R, H_f, H_r [N, B', T, D] -> spikes [N, B', T, D]. make_neuron() -> a token-axis neuron module (one per call)."""
    def __init__(self, dim, mask='none', proj=False, make_neuron=None, bias=False):
        super().__init__()
        self.proj = bool(proj)
        if self.proj:
            self.q = _Proj(dim, dim, make_neuron(), bias)
            self.k = _Proj(dim, dim, make_neuron(), bias)
            self.v = _Proj(dim, dim, make_neuron(), bias)
        self.core = AttnMath(dim, mask)
        self.bn = nn.BatchNorm1d(dim)
        self.neuron = make_neuron()
        self.dim = int(dim)

    def linears(self):
        return [p.linear for p in (self.q, self.k, self.v)] if self.proj else []

    def forward(self, f, r, hf, hr):
        q, k, v = (self.q(f), self.k(r), self.v(r)) if self.proj else (f, r, r)
        z = self.core(q, k, v, hf, hr)
        N, B, T, D = z.shape
        z = self.bn(z.reshape(-1, D)).reshape(N, B, T, D)
        return self.neuron(z.contiguous())
