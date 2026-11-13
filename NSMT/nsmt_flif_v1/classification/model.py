import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate
from spikingjelly.clock_driven import neuron as sj_neuron, functional as sj_functional

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg 
from timm.models.layers import trunc_normal_
# from timm.optim.optim_factory import create_optimizer_v2

# from timm.models import create_model
from timm.utils import *

# from encoder import Embedding, Block, TemporalBlock
from layers import SpkEncoder, SpikLinearLayer, SpikLinearMaxLayer, MLP, SSA_rel_scl, MutualCrossAttention
from layers import build_neo_neuron, NeoNeuron, apply_tau_calibration, swap_sj_neurons   # nsmt_flif_v1; swap: MODI 2026-10-29
from positional import tAPE
from token_order import TokenShuffle, permute_tokens, permute_tokens_det   # MODI 2026-10-28 (shuffle-retrain); 2026-11-02 (neo_shuffle)
from utils import random_masking_3D, cyclic_shift_batch_dim1
from einops import rearrange

import math


__all__ = ['mymodel']

def dct_matrix(T: int, device=None, dtype=torch.float32):
    device = device or 'cpu'
    n = torch.arange(T, device=device, dtype=dtype).reshape(1, T)
    k = torch.arange(T, device=device, dtype=dtype).reshape(T, 1)
    C = torch.cos(math.pi * (n + 0.5) * k / T) * math.sqrt(2.0 / T)
    C[0, :] *= 1 / math.sqrt(2.0)
    return C  # (T,T)

def dct_time(x: torch.Tensor, C: torch.Tensor, time_dim: int = 1):
    # (B,T,D) 가정 버전
    if time_dim != 1:
        perm = list(range(x.dim()))
        perm[1], perm[time_dim] = perm[time_dim], perm[1]
        x = x.permute(*perm).contiguous()
        need_inv = (perm != list(range(len(perm))))
    else:
        need_inv = False

    Y = torch.matmul(x.transpose(1, 2).contiguous(), C.t()).transpose(1, 2)

    if need_inv:
        inv = list(range(len(perm)))
        inv[1], inv[time_dim] = inv[time_dim], inv[1]
        Y = Y.permute(*inv).contiguous()
    return Y

def idct_time(xc: torch.Tensor, C: torch.Tensor, time_dim: int = 1):
    if time_dim != 1:
        perm = list(range(xc.dim()))
        perm[1], perm[time_dim] = perm[time_dim], perm[1]
        xc = xc.permute(*perm).contiguous()
        need_inv = (perm != list(range(len(perm))))
    else:
        need_inv = False

    Y = torch.matmul(xc.transpose(1, 2).contiguous(), C).transpose(1, 2)

    if need_inv:
        inv = list(range(len(perm)))
        inv[1], inv[time_dim] = inv[time_dim], inv[1]
        Y = Y.permute(*inv).contiguous()
    return Y

def lowpass_memory(x_bin: torch.Tensor,
                         keep_ratio: float = 0.25,
                         time_dim: int = 1,
                         center: bool = True,
                         rebinarize: bool = False,
                         thresh: float = 0.5):
    """
    x_bin: (B,T,D) 바이너리 스파이크(0/1) 또는 {0,1}로 해석 가능한 텐서
    return: x_lp (연속 또는 재이산화), coeff_lp (저주파만 남긴 DCT 계수)
    """
    x = x_bin.to(dtype=torch.float32)
    T = x.shape[time_dim]
    C = dct_matrix(T, device=x.device, dtype=x.dtype)

    # (선택) 평균 제거: DC가 너무 크면 저역만 남길 때 정보가 평준화됨
    if center:
        mean = x.mean(dim=time_dim, keepdim=True)
        x = x - mean

    # DCT
    Xc = dct_time(x, C, time_dim=time_dim)

    # 저주파 유지
    k = max(1, int(math.ceil(T * keep_ratio)))
    mask = torch.zeros(T, device=x.device, dtype=x.dtype)
    mask[:k] = 1
    shp = [1] * x.dim()
    shp[time_dim] = T
    mask = mask.view(*shp)
    Xc_lp = Xc * mask

    # 역변환
    x_lp = idct_time(Xc_lp, C, time_dim=time_dim)

    # (선택) 평균 복원: 장기 추세를 살리고 싶다면
    if center:
        x_lp = x_lp + mean

    # (선택) 재이산화: 추론용/스파이크 유지가 필요할 때만
    if rebinarize:
        # 주의: 역전파 필요하면 STE 등으로 처리
        x_lp = (x_lp > thresh).to(x.dtype)

    return x_lp, Xc_lp, mask



class Decoder(nn.Module):
    def __init__(self, embed_dim, d_out=None, d_ff=2, tau=2.0, bias=False) -> None:
        super().__init__()
        
        # self.replay = Consolidation(dim=embed_dim, out_seq=True, lif_bias=bias, tau=tau)

        d_ff = embed_dim
        
        d_out = d_out or embed_dim
        
        # self.recon1 = nn.Linear(d_ff, 8, bias=bias)
        # self.recon2 = SpikLinearLayer(8, d_out, lif_bias=bias)
        # self.recon2 = nn.Linear(8, d_out, bias=bias)
        self.recon2 = nn.Linear(d_ff, d_out, bias=bias)
        
    def forward(self, x, tr_mx=None):
        """
        [original] x: N x L x C(embed_dim)
        [MyModel] x: T x B x D
        
        out: reconstructed output -> N x L x c_out
        if expand is True: out's shape becomes [B X L]
        """

        # tr_mx = self.replay(x, tr_mx)
        # x = x * (1. - tr_mx)
        # x = self.dropout(x)
        # x = self.recon2(self.recon1(x))
        T, B, N, D = x.shape
        x = self.recon2(x)
        # x = torch.where(x>0, torch.ones_like(x), torch.ones_like(x) * -1.)
        rec_x = x.reshape(T, B, N, -1)
        
        return rec_x
    
class Embedding(nn.Module):
    def __init__(self, num_patches, pe=False, patch_size=63, stride=2, embed_dim=128, dropout=0, bias=False, tau=2.0,\
                 rpe_kernel_size: int = 3,
                 rpe_depthwise: bool = True,
                 rpe_rebinarize: bool = True,
                 neuron=None, pos_emb=False) -> None:
        super().__init__()
        
        self.num_patches = num_patches
        # self.patch_size = patch_size
        # in_channel = patch_size
        self.rpe_rebinarize = rpe_rebinarize
        self.stride = stride
        self.embed_dim = embed_dim
        self.pe = pe
        
        self.emb_linear = nn.Linear(patch_size, embed_dim, bias=bias)
        # MODI 2026-10-21 (1-B): learned absolute position [1, N, D] added to the real-valued pre-activation (before BN and
        # the neuron); every repetition step receives the same value. pos_emb=False: no module, output unchanged.
        self.pos = None
        if pos_emb:
            self.pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos, std=.02)
        self.emb_bn = nn.BatchNorm1d(embed_dim)
        # nsmt_flif_v1: `neuron` (NeoNeuron, layout 'DN') replaces the LIF; None keeps the spikingjelly LIF.
        # For encoding_neo the input is [N, B, 1, P], so this neuron runs over the token axis (T = N).
        self.emb_lif = neuron if neuron is not None else MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        # positional encoding
        # Residual dropout
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor, pe=True, return_pre=False):
        T, B, _,_ = x.shape

        x = self.emb_linear(x.flatten(0, 1))                      # [T*B, N', D]
        if self.pos is not None:                                  # MODI 2026-10-21 (1-B): position of each token
            if x.shape[1] == self.num_patches:                    # hippocampus layout: tokens on dim 1 (T = repetitions)
                x = x + self.pos
            elif x.shape[1] == 1 and T == self.num_patches:       # token-axis layout (Neo-mixer / neocortex): tokens on dim 0
                x = (x.reshape(T, B, 1, -1) + self.pos.reshape(T, 1, 1, -1)).reshape(T * B, 1, -1)
            else:
                raise RuntimeError(f'pos_emb: cannot place {tuple(self.pos.shape)} on {tuple(x.shape)} (T={T})')
        x = self.emb_bn(x.transpose(-1, -2).contiguous())
        x = x.reshape(T, B, self.embed_dim, -1).contiguous() # [T B N D]
        x = x.flatten(3).contiguous() # [T B D N] 
        self.org_x = x.clone().detach().transpose(-1, -2).contiguous()
        # MODI 2026-11-08 (membrane shortcut): return_pre also returns the neuron's input current (the BN output, graph kept), [T B N D]
        pre = x.transpose(-1, -2).contiguous() if return_pre else None


        x = self.emb_lif(x.reshape(T, B, self.embed_dim, -1).contiguous())
        x = x.transpose(-1, -2).contiguous()  # [T B N D]

        return (x, pre) if return_pre else x


def band_split(x, n_bands=4, time_dim=-1):
    """MODI 2026-11-13 (frequency bands, user request 11-13): split x along time into n_bands signals by DCT coefficient
    range, halving the frequency range each time (n_bands = 4, n coefficients: [0, n/8), [n/8, n/4), [n/4, n/2), [n/2, n);
    ceil at the edges). Band 0 holds the lowest frequencies (and the mean). The bands add up to x (the DCT is orthonormal).
    Returns [n_bands, *x.shape]."""
    x = x.to(torch.float32)
    time_dim = time_dim % x.dim()
    n = x.shape[time_dim]
    if n < 2 ** (n_bands - 1):
        raise ValueError(f'band_split: {n} samples are too few for {n_bands} bands')
    C = dct_matrix(n, device=x.device, dtype=torch.float64).to(x.dtype)   # built in float64: the float32 table is only orthonormal to ~1e-5
    Xc = dct_time(x, C, time_dim=time_dim)
    edges = [0] + [int(math.ceil(n / 2 ** (n_bands - 1 - i))) for i in range(n_bands - 1)] + [n]
    out = []
    for i in range(n_bands):
        mask = torch.zeros(n, device=x.device, dtype=x.dtype)
        mask[edges[i]:edges[i + 1]] = 1
        shp = [1] * x.dim(); shp[time_dim] = n
        out.append(idct_time(Xc * mask.view(*shp), C, time_dim=time_dim))
    return torch.stack(out)


class BandEmbedding(nn.Module):
    """MODI 2026-11-13 (frequency bands): Neocortex embedding whose population neuron receives one current per member.
    x [N, B, 1, P, K] (token axis first; slot k = the signal meant for member k) -> K Linear(P -> D) without sharing ->
    BatchNorm over the D * K (unit, member) channels -> the population neuron with currents [N, B, 1, D, K].
    Returns the soma spikes [N, B, 1, D] (and the member currents [N, B, 1, D, K] with return_pre, graph kept)."""
    def __init__(self, patch_size, embed_dim, n_members, neuron, bias=False):
        super().__init__()
        self.embed_dim, self.n_members = embed_dim, n_members
        self.emb_linear = nn.ModuleList([nn.Linear(patch_size, embed_dim, bias=bias) for _ in range(n_members)])
        self.emb_bn = nn.BatchNorm1d(embed_dim * n_members)
        self.emb_lif = neuron                                         # NeoNeuron around a PopulationFLIF with a soma

    def forward(self, x, pe=True, return_pre=False):
        T, B, N1, P, K = x.shape
        if K != self.n_members:
            raise ValueError(f'BandEmbedding built for {self.n_members} members, got {K} input slots')
        cur = torch.stack([self.emb_linear[k](x[..., k]) for k in range(K)], dim=-1)        # [T, B, N', D, K]
        cur = self.emb_bn(cur.reshape(T * B * N1, self.embed_dim * K)).reshape(T, B, N1, self.embed_dim, K)
        s = self.emb_lif(cur)                                         # [T, B, N', D] (NeoNeuron.forward_branches)
        return (s, cur) if return_pre else s


class HighFreqAmp(nn.Module):
    def __init__(self, dim, tau=2.0, bias=False):
        super().__init__()
        self.down1 = SpikLinearMaxLayer(dim, dim, kernel_size=2, tau=tau, lif_bias=bias) #[D//8]
        self.down2 = SpikLinearMaxLayer(dim // 2, dim // 2, kernel_size=2, tau=tau, lif_bias=bias) #[D//4]
        self.down3 = SpikLinearMaxLayer(dim // 4, dim // 4, kernel_size=2, tau=tau, lif_bias=bias) #[D//2]
        self.skip = SpikLinearLayer(dim, dim, tau=tau, lif_bias=bias)
       
        self.up1 = SpikLinearLayer(dim//8, dim, tau=tau, lif_bias=bias) #[D//1]self.max1 = SpikLinearLayer(dim, dim, tau=tau, lif_bias=bias) #[D//8]
        # self.up2 = SpikLinearLayer(dim//4, dim, tau=tau, lif_bias=bias) #[D//4]
        # self.up3 = SpikLinearLayer(dim//2, dim, tau=tau, lif_bias=bias) #[D//2]

        # self.agg = SpikLinearLayer(dim // 8 + dim // 4 + dim // 2 + dim, dim, tau=tau, lif_bias=bias)
        # self.agg = SpikLinearLayer(dim * 3, dim, tau=tau, lif_bias=bias)

        
    def forward(self, x):
        x_skip = self.skip(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        return x_skip * self.up1(x)

        # return self.agg(torch.cat([x1, x2, x3], dim=-1))
        # return x1 * x2 * x3
    

class Block(nn.Module):
    def __init__(self, T, dim, seq_len, num_heads, mlp_ratios=2., max_ratio=2, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.1,
                drop_path=0.1, norm_layer=nn.LayerNorm, sr_ratio=1, lif_bias=False, tau=2.0, attn=False,
                time_num_layers=1, patch_size=16, ssa=False, xattn=False, ssa_pe=True, attn_bn=False, xattn_mode='replace'):
        super().__init__()
        
        topk_ratio = None
        self.use_ssa = bool(ssa)                         # stage 9 Hippo-only: Spikformer-style token self-attention (N x N)
        self.use_xattn = bool(xattn)                     # MODI 2026-10-21 (1-C): token cross-attention, q = hippocampus, k/v = neocortex
        
        # self.high_freq = HighFreqAmp(dim) if not attn else SSA_rel_scl(dim=dim, seq_len=seq_len, num_heads=num_heads, pe=True, lif_bias=lif_bias, tau=tau, drop=attn_drop)
        if self.use_ssa or self.use_xattn:
            self.ssa = SSA_rel_scl(dim=dim, seq_len=seq_len, num_heads=num_heads, pe=bool(ssa_pe), lif_bias=lif_bias, tau=tau, drop=attn_drop,
                                   attn_bn=bool(attn_bn),              # MODI 2026-10-21: BN before the attention neuron (spk_ssa_abn)
                                   kv_cat=(self.use_xattn and xattn_mode == 'cat'),   # MODI 2026-10-26 (B): own + neocortex keys / values
                                   q_from_kv=(self.use_xattn and xattn_mode == 'query'))   # MODI 2026-10-30: queries from the neocortex
        else:
            self.mca = MutualCrossAttention(dim=dim, out_seq=False, lif_bias=lif_bias, num_heads=num_heads, tau=tau, drop=drop)
        mlp_hidden_dim = int(dim * mlp_ratios)
        mlp_input_dim = dim
        self.mlp = MLP(in_features=mlp_input_dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop, lif_bias=lif_bias, tau=tau)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.shuffle_replay = False

    def forward(self, x: torch.Tensor, mx=None, self_kv=False, u=None):
        T, B, N, D = x.shape
        # MODI 2026-11-08 (membrane shortcut): with the stream u (real-valued, [T B N D]) the block returns (spikes, u) and replaces the
        # IAND x (1 - f(x)) by u <- u + BN(W_proj a); s = LIF_proj(u); u <- u + BN(W_fc2 h); s = LIF_fc2(u) (same modules, no new ones)
        if u is not None and not (self.use_ssa or self.use_xattn):
            raise NotImplementedError('membrane shortcut: hippo_ssa / hippo_xattn blocks only')

        if self.use_ssa:                                # stage 9: x = x (1 - SSA(x)); same IAND structure as the other blocks
            if u is not None:
                u, x = self.ssa(x, ms_u=u)
                u, x = self.mlp(x, ms_u=u)
                return x, u
            x = x * (1. - self.ssa(x))
            x = x * (1. - self.mlp(x))
            return x

        if self.use_xattn:                              # MODI 2026-10-21 (1-C): hippocampus T_h repeats attend over the N neocortex tokens
            g = getattr(self, 'neo_grad', 0.5)          # MODI 2026-10-26: task-gradient share to the neocortex (default 0.5)
            mx = (g * mx + (1 - g) * mx.detach())       # same gradient share as the memory-replay path
            if mx.shape[2] not in (1, x.shape[0]):      # MODI 2026-10-23: K neocortex streams must equal the T_h hippocampus steps
                raise ValueError(f'{mx.shape[2]} neocortex streams for {x.shape[0]} hippocampus steps')
            if getattr(self, 'stream_rev', False):      # MODI 2026-10-24: order control, stream K-1-t -> step t
                mx = mx.flip(2)
            kv = mx.transpose(0, 2).expand(x.shape[0], -1, -1, -1).contiguous()   # [N, B, 1, D] -> [T_h, B, N, D]; [N, B, K, D] -> [K, B, N, D]
            if u is not None:                           # MODI 2026-11-08 (membrane shortcut)
                u, x = self.ssa(x, kv=kv, ms_u=u)
                u, x = self.mlp(x, ms_u=u)
                return x, u
            x = x * (1. - self.ssa(x, kv=kv))
            x = x * (1. - self.mlp(x))
            return x

        if self_kv:                                     # nsmt_flif_v1 hippo_only (MODI decision 7): key/value = hippocampus
            x = x * (1. - self.mca(x, x))
            x = x * (1. - self.mlp(x))
            return x

        if mx is None:                                  # no_neo ablation: skip Memory-Replay fusion
            x = x * (1. - self.mlp(x))
            return x

        if (not self.training) and self.shuffle_replay and mx is not None and B > 1:
            mx = cyclic_shift_batch_dim1(mx)

        g = getattr(self, 'neo_grad', 0.5)              # MODI 2026-10-26 (default 0.5 = the ECG code)
        mx = (g * mx + (1 - g) * mx.detach()) # [1 B N D]
        mx = mx.transpose(0, 2).contiguous() * x

        x = x * (1. - self.mca(x, mx))
        x = x * (1. - self.mlp(x))
        
        return x
  
class RecurrentNeoLayer(nn.Module):
    """Zipser 1993 re-entrant recurrent spiking memory (ported from forecasting/ours.py).
    V(t+1)=decay*V(t)+W_in*in(t)+W_rec*s(t-1); W_rec feedback -> self-sustaining attractor (long-term hold)."""
    def __init__(self, dim, tau=2.0, bias=False, rec_scale=0.1, rec_gate=False):
        super().__init__()
        self.dim = dim
        self.w_in  = nn.Linear(dim, dim, bias=bias)
        self.w_rec = nn.Linear(dim, dim, bias=bias)
        nn.init.uniform_(self.w_rec.weight, -rec_scale / dim**0.5, rec_scale / dim**0.5)
        self.w_rec._skip_global_init = True   # review 6.3: preserve small W_rec init from model-wide re-init
        self.bn    = nn.BatchNorm1d(dim)
        self.lif   = sj_neuron.LIFNode(tau=tau, detach_reset=True, surrogate_function=surrogate.Sigmoid())
        # learnable recurrence gate g (init 0): pre = w_in + g*w_rec. g=0 => stable feedforward (=off baseline);
        # training grows g only if recurrence helps -> recurrent neo becomes a SAFE superset of feedforward.
        self.rec_gate = rec_gate
        if rec_gate:
            self.g = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        sj_functional.reset_net(self.lif)
        T, B, N, D = x.shape
        s = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            rec = (self.g * self.w_rec(s)) if self.rec_gate else self.w_rec(s)
            pre = self.w_in(x[t]) + rec
            pre = self.bn(pre.reshape(-1, D)).reshape(B, N, D)
            s = self.lif(pre)
            outs.append(s)
        return torch.stack(outs, dim=0)


class NeoLayer(nn.Module):
    """Exact-passive superset Neocortex layer (review 8.1). Passive and active share the SAME
    preactivation (BN over the full T sequence) + single-step LIF loop. Active adds a ZERO-INIT
    gated recurrent residual pre_t = u_t + tanh(g)*W_rec s_{t-1}, g init 0 -> gamma 0 -> at init
    (and always for passive) active == passive EXACTLY. On MIT-BIH the gate can learn back to 0,
    recovering passive-level F1 (test of the 'self-gating fixes classification collapse' claim)."""
    def __init__(self, dim, tau=2.0, bias=False, recurrent=False, rec_scale=0.1, neuron=None):
        super().__init__()
        self.dim = dim
        self.recurrent = recurrent
        self.w_in = nn.Linear(dim, dim, bias=bias)
        self.bn   = nn.BatchNorm1d(dim)
        # nsmt_flif_v1: `neuron` (NeoNeuron, layout 'ND', spikeDE LIF / f-LIF) replaces the single-step LIF loop;
        # it integrates the same BN'd pre-activation u over the token axis T. Passive layers only.
        if neuron is not None and recurrent:
            raise ValueError('a spikeDE neocortex neuron is implemented for the passive NeoLayer only')
        self.spikede = neuron is not None
        self.lif  = neuron if neuron is not None else sj_neuron.LIFNode(tau=tau, detach_reset=True, surrogate_function=surrogate.Sigmoid())
        if recurrent:
            self.w_rec = nn.Linear(dim, dim, bias=bias)
            nn.init.uniform_(self.w_rec.weight, -rec_scale / dim**0.5, rec_scale / dim**0.5)
            self.w_rec._skip_global_init = True
            self.g = nn.Parameter(torch.zeros(1))            # zero-init gate -> exact passive at start
        self.store_v_seq = False
        self.v_seq = None

    def forward_charge(self, x):                             # MODI 2026-10-23: (spikes, H - theta), spikeDE neurons only
        if not self.spikede:
            raise ValueError('charge capture needs a spikeDE neocortex neuron')
        T, B, N, D = x.shape
        u = self.bn(self.w_in(x).reshape(-1, D)).reshape(T, B, N, D)
        return self.lif.forward_charge(u)

    def forward_ms(self, x, u):                              # MODI 2026-11-08 (Neocortex membrane shortcut)
        """u <- u + BN(W x); this layer's neuron fires on the stream u instead of BN(W x) alone. Returns (u, spikes)."""
        if not self.spikede or self.recurrent:
            raise ValueError('the Neocortex membrane shortcut needs a spikeDE neuron in a passive NeoLayer')
        T, B, N, D = x.shape
        cur = self.bn(self.w_in(x).reshape(-1, D)).reshape(T, B, N, D)
        if u.ndim == 5:                                      # MODI 2026-11-13 (frequency bands): one stream per member [T, B, N, D, K];
            u = u + cur.unsqueeze(-1)                        # every member's stream receives this layer's current
            return u, self.lif(u)                            # NeoNeuron.forward_branches
        u = u + cur
        return u, self.lif(u)

    def forward(self, x):                                    # x: [T, B, N, D]
        T, B, N, D = x.shape
        u = self.bn(self.w_in(x).reshape(-1, D)).reshape(T, B, N, D)
        if self.spikede:                                 # nsmt_flif_v1: whole sequence at once (state kept inside)
            return self.lif(u)
        sj_functional.reset_net(self.lif)
        gamma = torch.tanh(self.g) if self.recurrent else None
        s = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        outs, vs = [], []
        for t in range(T):
            pre = u[t]
            if self.recurrent:
                pre = pre + gamma * self.w_rec(s)
            s = self.lif(pre)
            outs.append(s)
            if self.store_v_seq:
                vs.append(self.lif.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(vs, 0)
        return torch.stack(outs, dim=0)


class TemporalBlock(nn.Module):
    def __init__(self, T, num_layers=2, patch_size=[16, 32], embed_dim=[64, 128, 256], ratio=2, lif_bias=False, tau=2.0, num_heads=8, mutual=False, topk:int | None=None, topk_temperature:float = 1.0, recurrent=False, rec_gate=False, exact=True,
                 neuron_fn=None):
        super().__init__()
        
        # self.T = T  # time step

        self.patch_size = patch_size
        self.T = T
        self.num_layers = int(num_layers)
        # hidden_dim = int(num_layers*embed_dim)

        # if (T < 11) and (embed_dim > 24):
        #     hidden_dim = 32
        # elif (11 < T < 40) or (embed_dim < 24):
        #     hidden_dim = 64
        # else:
        #     hidden_dim = 128
        # dpr = [x.item() for x in torch.linspace(0, 0.1, num_layers)]  # stochastic depth decay rule
        
        # self.in_layer = 
        self.exact = exact
        if neuron_fn is not None and (not exact or recurrent):          # nsmt_flif_v1
            raise ValueError('spikeDE neocortex neurons need the exact passive NeoLayer (no --neo_legacy / --neo_recurrent)')
        if exact:
            self.Mem = nn.ModuleList([NeoLayer(embed_dim, tau=tau, bias=lif_bias, recurrent=recurrent,
                                               neuron=(neuron_fn() if neuron_fn is not None else None))   # nsmt_flif_v1
                                      for l in range(int(num_layers))])
        elif recurrent:
            self.Mem = nn.ModuleList([RecurrentNeoLayer(embed_dim, tau=tau, bias=lif_bias, rec_gate=rec_gate) for l in range(int(num_layers))])
        else:
            self.Mem = nn.ModuleList(
                [SpikLinearLayer(embed_dim, embed_dim, lif_bias=lif_bias, tau=tau)
                for l in range(int(num_layers))])
        # self.Mem = SpikLinearLayer(hidden_dim, embed_dim, lif_bias=lif_bias, tau=tau, spk='lif')
        # self.mem = nn.parameter.Parameter(torch.randn((1, 1, hidden_dim, embed_dim)))
        # self.memory = MemoryBank(hidden_dim, embed_dim)
        # self.dropout = nn.ModuleList(
        #     [nn.Dropout(dpr[l])
        #      for l in range(num_layers)]
        # )
        # self.out_bn = nn.BatchNorm1d(embed_dim)
        # self.bias = nn.Linear(embed_dim, embed_dim, bias=lif_bias)
        # self.out = SpikLinearLayer(embed_dim, embed_dim, lif_bias=lif_bias, tau=tau, spk='lif')
        # self.out = MultiStepLIFNode(tau=2.0, backend='cupy', surrogate_function=surrogate.Sigmoid())


    def forward(self, x, org_data=None, u0=None):
        T, B, N, D = x.shape
        if u0 is not None:                               # MODI 2026-11-08 (Neocortex membrane shortcut): the stream starts at the
            u = u0                                       # embedding neuron's input current; layer i: u <- u + BN_i(W_i s), s = n_i(u)
            for i in range(self.num_layers):
                u, x = self.Mem[i].forward_ms(x, u)
            return x
        # x_in = x
        # x = self.out_layer(x)
        # x = self.memory(x)
        # x = self.out_bn(x.flatten(0, 1).transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        # x = x.view(T, B, -1, D).contiguous()
        # x = self.out(x)
        # x_in = x.transpose(0, 2).contiguous()
        for i in range(self.num_layers):
            x = self.Mem[i](x)
        # x = x + self.bias(x)
        # x = self.out(x)

        return x



class myModel(nn.Module):
    def __init__(self, gating=['original', 'ablation'], 
                 train_mode=['pretraining', 'training', 'testing', 'visual'], 
                 seq_len=187,
                data_patch_size=63, 
                num_classes=2, 
                time_num_layers=2,
                embed_dim=[64, 128, 256], 
                num_heads=[1, 2, 4], 
                mlp_ratios=1, 
                qkv_bias=False, 
                qk_scale=None,
                drop_rate=0.1, 
                attn_drop_rate=0., 
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm,
                depths=[6, 8, 6], 
                sr_ratios=[8, 4, 2], 
                T = 4, 
                num_channels=3,
                data_patching_stride=2, 
                padding_patches=None, 
                lif_bias=False, 
                tau=2.0, 
                spk_encoding=False,
                attn=['SSA', 'MSSA'],
                keep_ratio=0.25,
                pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                **kwargs,
                ):
        super().__init__()
        
        self.train_mode = train_mode
        self.gating = gating
        self.keep_ratio = keep_ratio
        
        # self.T = T  # time step
        self.spk_encoding = spk_encoding
        self.patch_size = data_patch_size
        self.stride = data_patch_size // 2
        self.num_channels = num_channels
        
        self.num_patches = int((seq_len - data_patch_size) / (self.stride) + 1)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.num_patches = self.num_patches + 1
        self.T = self.num_patches
        # stage 9: multichannel tokens (a token = P time steps x C channels, flattened) and a Spikformer-style Hippo-only
        # with its own number of repetitions T_h (SSA does not need T = N). Defaults keep the ECG model unchanged.
        self.in_channels = int(kwargs.get('in_channels', 1) or 1)
        self.tok_dim = self.patch_size * self.in_channels
        self.hippo_ssa = bool(kwargs.get('hippo_ssa', False))
        # MODI 2026-10-21: hippo_xattn (1-C) = hippocampus with its own T_h and token cross-attention over the neocortex tokens;
        # pos_emb (1-B) = learned absolute position in the hippocampus embedding; ssa_pe (1-B) = relative bias in SSA / xattn
        self.hippo_xattn = bool(kwargs.get('hippo_xattn', False))
        self.pos_emb = bool(kwargs.get('pos_emb', False))
        self.ssa_pe = bool(int(kwargs.get('ssa_pe', 1)))
        self.attn_bn = bool(kwargs.get('attn_bn', False))
        if self.hippo_xattn and self.hippo_ssa:
            raise ValueError('choose one of hippo_ssa and hippo_xattn')
        self.T_h = int(kwargs.get('hippo_T', 0) or 0) or self.T
        if self.T_h != self.T and not (self.hippo_ssa or self.hippo_xattn):
            raise ValueError('hippo_T != number of tokens is only allowed with hippo_ssa / hippo_xattn (the memory-replay cross-attention needs T = N)')
        # MODI 2026-10-26 (B): hippo_xattn_mode 'cat' = the hippocampus keeps its own keys / values (Spikformer-style self-attention)
        # and the neocortex tokens add a second key / value pair; neo_grad = task-gradient share to the neocortex through the
        # cross-attention / memory replay (0.5 = the ECG code); neo_memoryless = every token runs through the neocortex as its own
        # 1-step sequence (no state crosses tokens; the control for the T = N memory)
        self.hippo_xattn_mode = str(kwargs.get('hippo_xattn_mode', 'replace'))
        self.neo_grad = float(kwargs.get('neo_grad', 0.5))
        self.neo_memoryless = bool(kwargs.get('neo_memoryless', False))
        if self.hippo_xattn_mode not in ('replace', 'cat', 'query') or (self.hippo_xattn_mode in ('cat', 'query') and not self.hippo_xattn):
            raise ValueError("hippo_xattn_mode in {replace, cat, query}; cat / query need hippo_xattn")   # MODI 2026-10-30: + query
        if not 0. <= self.neo_grad <= 1.:
            raise ValueError('neo_grad must be in [0, 1]')

        # sine spike-encoding: roll the Hippocampus patch sequence per spiking step T with a
        # raised-cosine causal shift schedule (APE-like, ported from forecasting ours_roll).
        self.roll_mode = str(kwargs.get('roll_mode', 'none'))   # 'none'(repeat) | 'sine' | 'rand'
        self.neo_tau = float(kwargs.get('neo_tau', tau))            # P0/P1: Neo memory time-constant
        self.neo_recurrent = bool(kwargs.get('neo_recurrent', False))  # P1: re-entrant recurrent Neo (Zipser attractor)
        self.neo_rec_gate = bool(kwargs.get('neo_rec_gate', False))  # learnable recurrence gate (init 0): stable superset
        self.neo_exact = bool(kwargs.get('neo_exact', True))           # review 8.1: exact-passive superset NeoLayer
        self.no_neo = bool(kwargs.get('no_neo', False))             # ablation: Hippocampus only (no Neocortex / Memory Replay / low-pass aux)
        # nsmt_flif_v1: neocortex-path neuron replacement. 'both' = encoding_neo LIF + time_block LIFs, 'mlp' = time_block only;
        # both run over the token axis (T = N). The hippocampus path keeps the spikingjelly LIF in every condition.
        neo_neuron = str(kwargs.get('neo_neuron', 'sj_lif')); neo_where = str(kwargs.get('neo_where', 'both'))
        self.hippo_only = bool(kwargs.get('hippo_only', False))     # MODI decision 7: no neocortex, MCA key/value = hippocampus
        # stage 6: bidirectional neocortex -- a second embedding + time_block run over the REVERSED token axis; the two
        # spike outputs are fused by OR (max) after flipping the reverse one back, so the fused output stays 0/1
        self.neo_bidir = bool(kwargs.get('neo_bidir', False))
        # MODI 2026-10-23: 'pop_lif' = population LIF with soma (alpha 1); 'popk_lif' / 'popk_flif' = soma-less population streams
        # (the K members fire themselves, as stream_mixer.StreamNeurons) whose K = len(pop_taus) streams become the K = T_h steps
        # of the hippocampus cross-attention (stream k is the key / value of hippocampus step k; requires hippo_xattn, T_h = K)
        if neo_neuron not in ('sj_lif', 'lif', 'flif', 'pop_flif', 'pop_lif', 'popk_lif', 'popk_flif') or neo_where not in ('both', 'mlp'):
            raise ValueError("neo_neuron in {sj_lif, lif, flif, pop_flif, pop_lif, popk_lif, popk_flif}, neo_where in {both, mlp}")
        self.neo_streams = neo_neuron.startswith('popk_')
        # MODI 2026-10-24: stream_rev = stream K-1-t feeds hippocampus step t (order control); neo_stream_fuse 'mx' = a second
        # (reverse-token) stream neocortex with its own parameters, fused per stream as neuron(BN(W [H_f; H_r; |H_f - H_r|]))
        self.stream_rev = bool(kwargs.get('stream_rev', False))
        self.neo_stream_fuse = str(kwargs.get('neo_stream_fuse', 'none'))
        if self.neo_stream_fuse not in ('none', 'mx') or (self.neo_stream_fuse == 'mx') != (self.neo_streams and self.neo_bidir):
            raise ValueError("neo_stream_fuse 'mx' goes with popk streams and neo_bidir (and only with them)")
        if (self.stream_rev or self.neo_stream_fuse != 'none') and not self.neo_streams:
            raise ValueError('stream_rev / neo_stream_fuse are for popk neocortex streams')
        if self.neo_streams and (not self.hippo_xattn or neo_where != 'both'
                                 or self.T_h != len(tuple(kwargs.get('pop_taus', (4., 8., 16., 32.))))):
            raise ValueError('popk neocortex streams need hippo_xattn, neo_where both and hippo_T = number of streams')
        if self.neo_bidir and (self.no_neo or self.hippo_only):
            raise ValueError('neo_bidir needs the neocortex path')
        if (self.hippo_xattn_mode in ('cat', 'query') and self.neo_streams) or (self.neo_memoryless and (self.neo_bidir or self.neo_streams or self.no_neo
                                                                                             or self.hippo_only or self.hippo_ssa)):
            raise ValueError('hippo_xattn_mode cat / query: not with popk streams; neo_memoryless: unidirectional single-stream neocortex only')
        if neo_neuron != 'sj_lif' and (self.no_neo or self.hippo_only):
            raise ValueError('no_neo / hippo_only have no neocortex path; do not combine them with --neo_neuron')
        if self.no_neo and self.hippo_only:
            raise ValueError('choose one of no_neo and hippo_only')
        if self.hippo_ssa and (self.no_neo or self.hippo_only or self.neo_bidir or neo_neuron != 'sj_lif'):
            raise ValueError('hippo_ssa has no neocortex path; do not combine it with no_neo / hippo_only / neo options')
        if self.hippo_xattn and (self.no_neo or self.hippo_only):
            raise ValueError('hippo_xattn needs the neocortex path')
        # MODI 2026-10-28: token_shuffle = shuffle-retrain diagnostic (token_order.TokenShuffle); Spikformer-style only (no aux target)
        self.token_shuffle = TokenShuffle(self.num_patches, kwargs.get('shuffle_seed', 0)) if kwargs.get('token_shuffle', False) else None
        if self.token_shuffle is not None and not self.hippo_ssa:
            raise ValueError('token_shuffle is implemented for hippo_ssa (Spikformer-style) only')
        # MODI 2026-11-02: neo_shuffle = Neocortex-only order control. The Neocortex runs the N tokens in a shuffled order (TokenShuffle:
        # per-sample random in training, the fixed default_rng(0) permutation in evaluation) and its outputs are put back at the
        # original token positions, so the Neocortex output of token i still meets hippocampus token i and the hippocampus (keys /
        # values, relative position bias) keeps the true order; state still crosses tokens, but the history it carries is not the past
        self.neo_shuffle = TokenShuffle(self.num_patches, kwargs.get('shuffle_seed', 0)) if kwargs.get('neo_shuffle', False) else None
        if self.neo_shuffle is not None and (not self.hippo_xattn or self.neo_memoryless or self.neo_bidir or self.neo_streams
                                             or self.token_shuffle is not None):
            raise ValueError('neo_shuffle: hippo_xattn with a unidirectional single-stream neocortex that keeps its memory '
                             '(not with neo_memoryless / neo_bidir / popk streams / token_shuffle)')
        self.neo_neuron, self.neo_where = neo_neuron, neo_where
        # MODI 2026-11-13 (frequency bands, user request 11-13): the Neocortex input is split into K = 4 signals and member k of
        # every population neuron of the Neocortex embedding receives its own signal through its own Linear(P -> D).
        #   'freq': the DCT bands, lowest band -> slowest member (largest tau)      'rev': lowest band -> fastest member (control)
        #   'same': the unsplit signal in all K slots (control: the K separate embeddings without the band split)
        # The member currents are the start of the membrane shortcut stream, one stream per member (needs --neo_res ms).
        self.neo_band = str(kwargs.get('neo_band', 'none'))
        if self.neo_band not in ('none', 'freq', 'same', 'rev'):
            raise ValueError("neo_band in {none, freq, same, rev}")
        if self.neo_band != 'none' and (neo_neuron not in ('pop_lif', 'pop_flif') or neo_where != 'both' or not self.hippo_xattn
                                        or str(kwargs.get('neo_res', 'none')) != 'ms' or self.token_shuffle is not None
                                        or len(tuple(kwargs.get('pop_taus', (4., 8., 16., 32.)))) != 4):
            raise NotImplementedError('neo_band: query / cross-attention model with a 4-member population Neocortex (soma), '
                                      'neo_where both and --neo_res ms only')
        _neo = dict(tau=self.neo_tau, alpha=float(kwargs.get('neo_alpha', 0.5)), method=str(kwargs.get('fde_method', 'pred')),
                    step=float(kwargs.get('fde_step', 1.0)), memory=int(kwargs.get('fde_memory', 0)),
                    pop_taus=tuple(kwargs.get('pop_taus', (4., 8., 16., 32.))), pop_soma_init=kwargs.get('pop_soma_init', None),
                    reset=str(kwargs.get('neo_reset', 'spikede')))                       # MODI 2026-10-31
        emb_neuron = (build_neo_neuron(neo_neuron, embed_dim, self.T, 'DN', **_neo)
                      if (neo_neuron != 'sj_lif' and neo_where == 'both' and not self.neo_streams) else None)
        mlp_fn = ((lambda: build_neo_neuron(neo_neuron, embed_dim, self.T, 'ND', **_neo))
                  if (neo_neuron != 'sj_lif' and not self.neo_streams) else None)
        _T = self.T
        _t = torch.arange(_T, dtype=torch.float32)
        if self.roll_mode == 'rand':
            _sched = torch.randint(0, _T, (_T,)).sort(descending=True).values.long()  # random, time-ordered
            _sched[-1] = 0
        else:                                                    # sine (raised-cosine)
            _sched = torch.round((_T - 1) * 0.5 * (1.0 + torch.cos(math.pi * _t / max(_T - 1, 1)))).long()
        self.register_buffer('shift_sched', _sched.clamp_(0, _T - 1), persistent=False)

        print(f"len of tokens in sequence >> {self.num_patches}")


        self.spk_encoder = SpkEncoder(T) if spk_encoding else None
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.encoding = Embedding(num_patches=self.num_patches,
                                    patch_size=self.tok_dim,             # stage 9: P x C (= P for the ECG model)
                                    embed_dim=embed_dim,
                                    stride=self.stride,
                                    pe=False,
                                    bias=lif_bias,
                                    tau=tau,
                                    pos_emb=self.pos_emb              # MODI 2026-10-21 (1-B)
                                    )
        if self.neo_streams:                                           # MODI 2026-10-23: K soma-less streams over the token axis
          from stream_mixer import StreamNeurons, _Embed, _Stack
          from layers import tau_gain
          kind, taus = neo_neuron.split('_', 1)[1], tuple(float(t) for t in _neo['pop_taus'])
          K = len(taus)
          g = torch.cat([torch.full((embed_dim,), tau_gain(t, kind)) for t in taus])   # per-family table, then the layer-wise table below
          self.encoding_neo = _Embed(self.tok_dim, embed_dim, K, StreamNeurons(kind, taus, embed_dim, self.T, _neo['alpha']),
                                     bias=lif_bias, bn_layout='ncl', gamma=g)
        elif self.neo_band != 'none':                                  # MODI 2026-11-13: one current per population member
          self.encoding_neo = BandEmbedding(self.tok_dim, embed_dim, len(_neo['pop_taus']), emb_neuron, bias=lif_bias)
        elif not self.hippo_ssa:                                       # stage 9: Hippo-only (SSA) builds no neocortex
          self.encoding_neo = Embedding(num_patches=self.num_patches,
                                    patch_size=self.tok_dim,
                                    embed_dim=embed_dim,
                                    stride=self.stride,
                                    pe=False,
                                    bias=lif_bias,
                                    tau=tau,
                                    neuron=emb_neuron                  # nsmt_flif_v1
                                    )
        if self.neo_bidir and self.neo_streams:                        # MODI 2026-10-24: reverse-token streams + per-stream fusion
            self.encoding_neo_rev = _Embed(self.tok_dim, embed_dim, K, StreamNeurons(kind, taus, embed_dim, self.T, _neo['alpha']),
                                           bias=lif_bias, bn_layout='ncl', gamma=g)
        elif self.neo_bidir:                                           # stage 6: its own parameters and neurons
            emb_neuron_rev = (build_neo_neuron(neo_neuron, embed_dim, self.T, 'DN', **_neo)
                              if (neo_neuron != 'sj_lif' and neo_where == 'both') else None)
            self.encoding_neo_rev = Embedding(num_patches=self.num_patches, patch_size=self.tok_dim, embed_dim=embed_dim,
                                              stride=self.stride, pe=False, bias=lif_bias, tau=tau, neuron=emb_neuron_rev)

        self.data_block = nn.ModuleList([Block(T=T,
                    dim=embed_dim, seq_len=self.num_patches, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                    norm_layer=norm_layer, sr_ratio=sr_ratios, lif_bias=lif_bias, tau=tau, attn=(gating=='original'),
                    ssa=self.hippo_ssa, xattn=self.hippo_xattn, ssa_pe=self.ssa_pe, attn_bn=self.attn_bn,
                    xattn_mode=self.hippo_xattn_mode)                  # MODI 2026-10-26
                    for j in range(depths)])

        if self.neo_streams:
          self.time_block = _Stack(embed_dim, time_num_layers, K, lambda: StreamNeurons(kind, taus, embed_dim, self.T, _neo['alpha']),
                                   g, lif_bias)
        elif not self.hippo_ssa:
          self.time_block = TemporalBlock(T=self.T,
                                        num_layers=time_num_layers,
                                        patch_size=data_patch_size,
                                        embed_dim=embed_dim,
                                        lif_bias=lif_bias,
                                        num_heads=num_heads,
                                        tau=self.neo_tau,
                                        recurrent=self.neo_recurrent,
                                        rec_gate=self.neo_rec_gate,
                                        exact=self.neo_exact,
                                        mutual=False,
                                        neuron_fn=mlp_fn)              # nsmt_flif_v1
        if self.neo_bidir and self.neo_streams:                        # MODI 2026-10-24
            from layers import sj_lif
            self.time_block_rev = _Stack(embed_dim, time_num_layers, K, lambda: StreamNeurons(kind, taus, embed_dim, self.T, _neo['alpha']),
                                         g, lif_bias)
            from bix import StreamSpikeLinear
            self.neo_fuse = StreamSpikeLinear(3 * embed_dim, embed_dim, K, sj_lif(tau), bias=lif_bias, bn_layout='flat')
        elif self.neo_bidir:                                           # stage 6
            self.time_block_rev = TemporalBlock(T=self.T, num_layers=time_num_layers, patch_size=data_patch_size,
                                                embed_dim=embed_dim, lif_bias=lif_bias, num_heads=num_heads, tau=self.neo_tau,
                                                recurrent=self.neo_recurrent, rec_gate=self.neo_rec_gate, exact=self.neo_exact,
                                                mutual=False, neuron_fn=mlp_fn)
            
        # init_weight = torch.Tensor(self.T, embed_dim)
        # self.memory_slot = nn.parameter.Parameter(init_weight)
        # stdv = 1. / math.sqrt(self.memory_slot.size(1))
        # self.memory_slot.data.uniform_(-stdv, stdv)
        
        if ((self.train_mode == 'training') or (self.train_mode == 'pre_training')) and not self.hippo_ssa : 
            self.weak_decoder = Decoder(embed_dim=embed_dim, d_out=self.tok_dim, tau=tau, bias=lif_bias)
            
        self.head = nn.Linear(embed_dim, num_classes, bias=lif_bias) if num_classes > 0 else nn.Identity()
        self.dropout = nn.Dropout(drop_rate)
        if gating == 'ablation':
            self._init_ablation()

        self.apply(self._init_weights)
        self.tau_calibrated = False
        if self.neo_streams and kwargs.get('tau_calib', True):         # MODI 2026-10-23: layer-wise rate-matched gains (calibrate_nsmt_streams.py)
            self.tau_calibrated = apply_tau_calibration(self, 'myModel', neo_neuron,
                                                        f'xaT{self.T_h}' + ('_bimx' if self.neo_stream_fuse == 'mx' else ''), taus)
        for blk in self.data_block:                                    # MODI 2026-10-24: alignment order of the streams
            blk.stream_rev = self.stream_rev
            blk.neo_grad = self.neo_grad                               # MODI 2026-10-26
        # MODI 2026-10-29 (--neuron_all): the hippocampus neurons (embedding, SSA, MLP; T_h repeat steps) become the spikeDE
        # neuron too, so no spikingjelly LIF is left (Spikformer-style; the B form with a spikeDE neocortex). Default: unchanged.
        self.neuron_all = str(kwargs.get('neuron_all', 'sj_lif'))
        if self.neuron_all != 'sj_lif':
            if self.neuron_all not in ('lif', 'flif', 'pop_lif', 'pop_flif'):
                raise ValueError('neuron_all in {sj_lif, lif, flif, pop_lif, pop_flif}')
            if not (self.hippo_ssa or (self.hippo_xattn and self.hippo_xattn_mode == 'cat' and neo_neuron != 'sj_lif' and not self.neo_streams)):
                raise NotImplementedError('neuron_all: Spikformer-style (hippo_ssa) or the B form with a spikeDE neocortex only')
            nh = swap_sj_neurons(self.encoding, self.neuron_all, self.T_h, **_neo) + swap_sj_neurons(self.data_block, self.neuron_all, self.T_h, **_neo)
            left = [n for n, m_ in self.named_modules() if isinstance(m_, MultiStepLIFNode)]
            if left:
                raise NotImplementedError(f'neuron_all: spikingjelly LIF left in {left}')
            self.n_swapped = nh

        # MODI 2026-11-08 (membrane shortcut, user request 11-08). res_mode 'ms': the hippocampus keeps a real-valued stream u that
        # starts at the embedding neuron's input current; each block adds BN(W_proj a) and BN(W_fc2 h) to it, and the proj / fc2
        # neurons fire on u (no IAND). neo_res 'ms': the Neocortex stream starts at its embedding neuron's input current and layer i
        # adds BN_i(W_i s) to it; layer i's neuron fires on u. Both reuse the existing modules (no new parameters or neurons);
        # 'iand' / 'none' = unchanged.
        self.res_mode, self.neo_res = str(kwargs.get('res_mode', 'iand')), str(kwargs.get('neo_res', 'none'))
        if self.res_mode not in ('iand', 'ms') or self.neo_res not in ('none', 'ms'):
            raise ValueError("res_mode in {iand, ms}, neo_res in {none, ms}")
        if self.res_mode == 'ms' and (not (self.hippo_ssa or self.hippo_xattn) or self.roll_mode != 'none' or self.neuron_all != 'sj_lif'):
            raise NotImplementedError('res_mode ms: hippo_ssa / hippo_xattn with roll_mode none and the spikingjelly hippocampus only')
        if self.neo_res == 'ms' and (self.hippo_ssa or self.no_neo or self.hippo_only or self.neo_streams or self.neo_bidir
                                     or self.neo_memoryless or self.neo_shuffle is not None or neo_neuron == 'sj_lif'
                                     or not self.neo_exact or self.neo_recurrent):
            raise NotImplementedError('neo_res ms: a unidirectional single-stream Neocortex of passive NeoLayers with a spikeDE neuron only')

        # MODI 2026-11-04 (H2, head test): attention pooling over the token axis in front of the head. A learned query q (score
        # layer Linear(D -> 1), no bias) scores every token of every T_h step, a = softmax_N(q . x_n / sqrt(D)), and the head
        # reads sum_n a_n x_n instead of the token mean. q starts at 0 (a = 1/N: exactly the mean head at init) and is built
        # after every other module with the CPU RNG forked, so the rest of the model is initialised as without it.
        self.head_pool = str(kwargs.get('head_pool', 'mean'))
        if self.head_pool not in ('mean', 'attn'):
            raise ValueError('head_pool in {mean, attn}')
        if self.head_pool == 'attn':
            with torch.random.fork_rng(devices=[]):
                self.head_score = nn.Linear(embed_dim, 1, bias=False)
            nn.init.zeros_(self.head_score.weight)

        if ((self.train_mode != 'training') and (self.train_mode != 'pre_training')):
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()

    def _pool(self, x):
        """x [T, B, N, D] -> [T, B, D]: token mean (default) or the H2 attention pooling."""
        if self.head_pool == 'mean':
            return x.mean(2)
        a = torch.softmax(self.head_score(x).squeeze(-1) / x.shape[-1] ** 0.5, dim=2)   # [T, B, N]
        return (a.unsqueeze(-1) * x).sum(2)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, N):
        if N == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(N), mode="bilinear").reshape(1, -1, N).permute(0, 2, 1)

    def _init_weights(self, m):

        if getattr(m, '_skip_global_init', False):   # review 6.3: preserve RecurrentNeoLayer small W_rec init
            return

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _gen_mask(self, x, mask_ratio=0.6) -> tuple:
        """Generating boolean mask which have same dimension with `x`

        Args:
            x (torch.tensor): embedding matrix after PE ([T B N D])
            

        - Return:
            x_data (torch.tensor): input for hippo.
            temporal_emb (torch.tensor): input for neocortix.
            
        """
        T, B, N, D = x.shape
        
        x_masked, _, self.mask, _ = random_masking_3D(x.flatten(0, 1), mask_ratio=mask_ratio)
        self.mask = self.mask.float().reshape(T, B, -1).contiguous().unsqueeze(-1) #[TB N]
        x_masked = x_masked.reshape(T, B, N, -1).contiguous()
        
        # x_t = x.transpose(0, 2).contiguous().mean(2, keepdim=True) #[N B T D]
        # x_t_masked, _, self.time_mask, _ = random_masking_3D(x_t.transpose(0, 2).contiguous().flatten(0, 1), mask_ratio=mask_ratio)
        # self.time_mask = self.time_mask.float().reshape(-1, B, N).transpose(0, 2).contiguous().unsqueeze(-1) #[T B 1 1]
        # x_t_masked = x_t_masked.reshape(1, B, -1, D).transpose(0, 2).contiguous()
        x_t_masked = x_masked.transpose(0, 2).contiguous().mean(2, keepdim=True) #[N B T D]
        
        return x_masked, x_t_masked

    def _roll_T(self, base):
        # base [BC, N, P] -> [T, BC, N, P]. rolled[t] = roll(base, shift_sched[t]) along N (dim1).
        outs = [torch.roll(base, shifts=int(self.shift_sched[t]), dims=1) for t in range(self.T)]
        return torch.stack(outs, dim=0).contiguous()

    def _unroll(self, emb):
        # emb [T, BC, N, D] -> undo each step's shift (torch.roll = deterministic).
        T = emb.shape[0]
        outs = [torch.roll(emb[t], shifts=-int(self.shift_sched[t]), dims=1) for t in range(T)]
        return torch.stack(outs, dim=0)

    def forward(self, x):
        # [B, L, C]
        if x.dim() != 3 :
            x = x.unsqueeze(1)
        self.B, self.M, L = x.shape
        org_seq = x.clone().detach()
        # x = x.permute(0, 2, 1) #[B M L]

        # if self.keep_ratio > 0:
        #     low_freq_x, _, _ = lowpass_memory(x, keep_ratio=self.keep_ratio, time_dim=2, center=True, rebinarize=False)
            
        if self.in_channels > 1 or self.hippo_ssa or self.hippo_xattn:   # stage 9 path (multichannel and/or T_h != N); exact copies, no means
            patched_x, neo_in = self._tokens_mc(x)
        else:
          if self.spk_encoding:
            x = self.spk_encoder(x)
          else: 
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1) 
            
          # x = rearrange(x, 't b l c -> t b c l')
          x = rearrange(x, 't b c l -> t (b c) l', c=1) # [T BC N P] 
          x = self.padding_patch_layer(x)
          patched_x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) 
          self.org_x = patched_x.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True)
          neo_in = None
        
          if (self.train_mode == 'training') and (self.keep_ratio > 0) : 
            keep_ratio=self.keep_ratio
            low_freq_x, _, _ = lowpass_memory(x, keep_ratio=keep_ratio, time_dim=2, center=True, rebinarize=False) #[T BC D]
            patched_low_freq_x = low_freq_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            self.org_x = patched_low_freq_x.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True)
            
        if self.roll_mode in ('sine', 'rand'):
            base = patched_x[0]                              # [BC, N, P] (all T identical after repeat)
            x_hippo = self._unroll(self.encoding(self._roll_T(base)))  # sine/rand roll, encode, realign
        else:
            x_hippo = self.encoding(patched_x, return_pre=(self.res_mode == 'ms'))   # MODI 2026-11-08: (spikes, stream) with ms

        if self.no_neo or self.hippo_only or self.hippo_ssa:   # nsmt_flif_v1: hippo_only / hippo_ssa have no neocortex path
            x_neo = None                                # ablation: no Neocortex path
        else:
            if neo_in is None:
                neo_in = patched_x.transpose(0, 2).contiguous().mean(2, keepdim=True)
            _, x_neo = self.neo_forward(neo_in, org_seq)   # [N B 1 D]

        x_hippo = self.hippo_blocks(x_hippo, x_neo)

        if self.train_mode == 'training':
            
            return self._training(x_hippo, x_neo) 
            
        elif self.train_mode == 'testing':

            return self._testing(x_hippo)
        
        elif self.train_mode == 'visual':

            return self._visualization(x_hippo, x_neo)
        
        else:
            raise ValueError
        
        
    def hippo_blocks(self, x_hippo, x_neo):
        """The hippocampus blocks on the embedding output; returns the last block's spikes [T_h, B, N, D].
        MODI 2026-11-08: res_mode 'ms' takes x_hippo = (spikes, stream) from encoding(..., return_pre=True)."""
        if getattr(self, 'res_mode', 'iand') == 'ms':
            x, u = x_hippo
            for dblk in self.data_block:
                x, u = dblk(x, x_neo, self_kv=self.hippo_only, u=u)
            return x
        for dblk in self.data_block:
            x_hippo = dblk(x_hippo, x_neo, self_kv=self.hippo_only)
        return x_hippo

    def _tokens_mc(self, x):
        """stage 9: x [B, C, L] -> hippocampus input [T_h, B, N, P*C] (exact repeats), neocortex input [N, B, 1, P*C] and the
        auxiliary target self.org_x [N, B, 1, P*C] (low-pass of each channel, patched the same way)."""
        B, C, L = x.shape
        xp = self.padding_patch_layer(x)                                       # [B, C, L + S]
        def patch(z):
            p = z.unfold(dimension=-1, size=self.patch_size, step=self.stride)    # [B, C, N, P]
            return p.permute(0, 2, 1, 3).reshape(B, p.shape[2], C * self.patch_size)   # [B, N, C*P] (channel-major)
        tok = patch(xp)
        if tok.shape[1] != self.num_patches:
            raise RuntimeError(f'{tok.shape[1]} tokens but the model was built for {self.num_patches}')
        self.org_x = tok.detach().transpose(0, 1).unsqueeze(2).contiguous()
        if (self.train_mode == 'training') and (self.keep_ratio > 0) and not self.hippo_ssa:
            low, _, _ = lowpass_memory(xp, keep_ratio=self.keep_ratio, time_dim=2, center=True, rebinarize=False)
            self.org_x = patch(low).detach().transpose(0, 1).unsqueeze(2).contiguous()
        hippo = tok.unsqueeze(0).expand(self.T_h, -1, -1, -1).contiguous()     # [T_h, B, N, C*P]
        neo = tok.transpose(0, 1).unsqueeze(2).contiguous()                    # [N, B, 1, C*P]
        band = getattr(self, 'neo_band', 'none')
        if band != 'none':                                                     # MODI 2026-11-13: [N, B, 1, C*P, K], slot k -> member k
            taus = self.encoding_neo.emb_lif.neuron.tau                        # member time constants, in member order
            K = taus.numel()
            if band == 'same':
                neo = neo.unsqueeze(-1).expand(*neo.shape, K).contiguous()
            else:
                bands = band_split(xp, K, time_dim=2)                          # [K, B, C, L + S], band 0 = lowest frequencies
                sig = torch.stack([patch(bands[i]) for i in range(K)], dim=-1).transpose(0, 1).unsqueeze(2)   # [N, B, 1, C*P, K]
                order = torch.argsort(taus, descending=(band == 'freq'))       # freq: band 0 -> largest tau; rev: band 0 -> smallest tau
                neo = torch.empty_like(sig)
                neo[..., order] = sig                                          # member order[i] receives band i
                neo = neo.contiguous()
        if getattr(self, 'token_shuffle', None) is not None:                   # MODI 2026-10-28: same order for both paths
            perm = self.token_shuffle.perms(B, tok.device, self.training)
            hippo, neo = permute_tokens(hippo, 2, perm), permute_tokens(neo, 0, perm)
        return hippo, neo

    def neo_forward(self, neo_in, org_seq=None):
        """Neocortex path on neo_in [N, B, 1, P] (token axis first). Returns (embedding spikes, output spikes), both
        [N, B, 1, D]. stage 6 bidirectional: the reversed token sequence goes through its own embedding + time_block,
        the result is flipped back and fused with the forward output by OR (torch.maximum)."""
        if getattr(self, 'neo_memoryless', False):                     # MODI 2026-10-26 (B control): each token alone, 1 step
            N, B = neo_in.shape[0], neo_in.shape[1]
            e = self.encoding_neo(neo_in.reshape(1, N * B, *neo_in.shape[2:]))
            o = self.time_block(e, org_seq)
            return e.reshape(N, B, *e.shape[2:]), o.reshape(N, B, *o.shape[2:])
        if getattr(self, 'neo_shuffle', None) is not None:                  # MODI 2026-11-02: Neocortex-only order control
            perm = self.neo_shuffle.perms(neo_in.shape[1], neo_in.device, self.training)
            e = self.encoding_neo(permute_tokens_det(neo_in, perm))
            o = self.time_block(e, org_seq)
            inv = torch.argsort(perm, dim=1)                                 # back to the original token positions
            return permute_tokens_det(e, inv), permute_tokens_det(o, inv)
        if getattr(self, 'neo_res', 'none') == 'ms':                        # MODI 2026-11-08: Neocortex membrane shortcut
            e, u0 = self.encoding_neo(neo_in, return_pre=True)               # u0 = the embedding neuron's input current [N, B, 1, D]
            return e, self.time_block(e, org_seq, u0=u0)
        e = self.encoding_neo(neo_in)
        if getattr(self, 'neo_streams', False):                        # MODI 2026-10-23: [N, B, K, D], stream k -> hippocampus step k
            if getattr(self, 'neo_stream_fuse', 'none') == 'mx':       # MODI 2026-10-24: both token directions, fused per stream
                _, hf = self.time_block.forward_charge(e)
                _, hr = self.time_block_rev.forward_charge(self.encoding_neo_rev(neo_in.flip(0)))
                hr = hr.flip(0)
                return e, self.neo_fuse(torch.cat([hf, hr, (hf - hr).abs()], dim=-1))
            return e, self.time_block(e)
        o = self.time_block(e, org_seq)
        if getattr(self, 'neo_bidir', False):
            e_r = self.encoding_neo_rev(neo_in.flip(0))
            o_r = self.time_block_rev(e_r, org_seq).flip(0)
            o = torch.maximum(o, o_r)
        return e, o

    def _training(self, x_hippo, x_neo):
        x_data = x_hippo.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True)
        if x_neo is None:                               # no_neo: no reconstruction aux, no distill
            return self.head(self._pool(x_hippo)).mean(0), x_data, x_data, self.org_x, self.org_x
        rec_x = self.weak_decoder(x_neo, x_hippo) #[T BxC 1 P]
        org_x = self.org_x
        if getattr(self, 'neo_streams', False):                        # MODI 2026-10-23: every stream reconstructs the same target
            org_x = org_x.expand_as(rec_x)

        assert org_x.shape == rec_x.shape, f"original x_time' shape is {org_x.shape}, and reconstructed x_time' shape is {rec_x.shape}"

        z = self.head(self._pool(x_hippo)).mean(0)

        return z, x_data, x_neo, org_x, rec_x

    
    def _testing(self, x):
        
        if self.gating == 'ablation':
            return self.head(self._pool(x)).mean(0)
        else:
            return self.head(self._pool(x)).mean(0), x.clone().detach()
    

    # TODO
    def _init_ablation(self):
        
        delattr(self, 'time_block')
        # delattr(self, 'gate_attn')
        if hasattr(self, 'weak_decoder'): delattr(self, 'weak_decoder')
        if hasattr(self, 'replay') : delattr(self, 'replay')
    

@register_model
def mymodel(pretrained=False, **kwargs):
    model = myModel(pretrained=pretrained, **kwargs)
    model.default_cfg = _cfg()
    return model