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
from layers import build_neo_neuron, NeoNeuron                        # nsmt_flif_v1
from positional import tAPE
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
                 neuron=None) -> None:
        super().__init__()
        
        self.num_patches = num_patches
        # self.patch_size = patch_size
        # in_channel = patch_size
        self.rpe_rebinarize = rpe_rebinarize
        self.stride = stride
        self.embed_dim = embed_dim
        self.pe = pe
        
        self.emb_linear = nn.Linear(patch_size, embed_dim, bias=bias)
        self.emb_bn = nn.BatchNorm1d(embed_dim)
        # nsmt_flif_v1: `neuron` (NeoNeuron, layout 'DN') replaces the LIF; None keeps the spikingjelly LIF.
        # For encoding_neo the input is [N, B, 1, P], so this neuron runs over the token axis (T = N).
        self.emb_lif = neuron if neuron is not None else MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        # positional encoding
        # Residual dropout
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor, pe=True): 
        T, B, _,_ = x.shape

        x = self.emb_linear(x.flatten(0, 1))
        x = self.emb_bn(x.transpose(-1, -2).contiguous())
        x = x.reshape(T, B, self.embed_dim, -1).contiguous() # [T B N D]
        x = x.flatten(3).contiguous() # [T B D N] 
        self.org_x = x.clone().detach().transpose(-1, -2).contiguous() 

        
        x = self.emb_lif(x.reshape(T, B, self.embed_dim, -1).contiguous())
        x = x.transpose(-1, -2).contiguous()  # [T B N D] 
        
        return x


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
                time_num_layers=1, patch_size=16):
        super().__init__()
        
        topk_ratio = None
        
        # self.high_freq = HighFreqAmp(dim) if not attn else SSA_rel_scl(dim=dim, seq_len=seq_len, num_heads=num_heads, pe=True, lif_bias=lif_bias, tau=tau, drop=attn_drop)
        self.mca = MutualCrossAttention(dim=dim, out_seq=False, lif_bias=lif_bias, num_heads=num_heads, tau=tau, drop=drop)
        mlp_hidden_dim = int(dim * mlp_ratios)
        mlp_input_dim = dim
        self.mlp = MLP(in_features=mlp_input_dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop, lif_bias=lif_bias, tau=tau)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.shuffle_replay = False

    def forward(self, x: torch.Tensor, mx=None, self_kv=False):
        T, B, N, D = x.shape

        if self_kv:                                     # nsmt_flif_v1 hippo_only (MODI decision 7): key/value = hippocampus
            x = x * (1. - self.mca(x, x))
            x = x * (1. - self.mlp(x))
            return x

        if mx is None:                                  # no_neo ablation: skip Memory-Replay fusion
            x = x * (1. - self.mlp(x))
            return x

        if (not self.training) and self.shuffle_replay and mx is not None and B > 1:
            mx = cyclic_shift_batch_dim1(mx)

        mx = (0.5 * mx + (1 - 0.5) * mx.detach()) # [1 B N D]
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


    def forward(self, x, org_data=None):
        T, B, N, D = x.shape
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
        if neo_neuron not in ('sj_lif', 'lif', 'flif') or neo_where not in ('both', 'mlp'):
            raise ValueError("neo_neuron in {sj_lif, lif, flif}, neo_where in {both, mlp}")
        if neo_neuron != 'sj_lif' and (self.no_neo or self.hippo_only):
            raise ValueError('no_neo / hippo_only have no neocortex path; do not combine them with --neo_neuron')
        if self.no_neo and self.hippo_only:
            raise ValueError('choose one of no_neo and hippo_only')
        self.neo_neuron, self.neo_where = neo_neuron, neo_where
        _neo = dict(tau=self.neo_tau, alpha=float(kwargs.get('neo_alpha', 0.5)), method=str(kwargs.get('fde_method', 'pred')),
                    step=float(kwargs.get('fde_step', 1.0)), memory=int(kwargs.get('fde_memory', 0)))
        emb_neuron = (build_neo_neuron(neo_neuron, embed_dim, self.T, 'DN', **_neo)
                      if (neo_neuron != 'sj_lif' and neo_where == 'both') else None)
        mlp_fn = ((lambda: build_neo_neuron(neo_neuron, embed_dim, self.T, 'ND', **_neo))
                  if neo_neuron != 'sj_lif' else None)
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
                                    patch_size=self.patch_size,
                                    embed_dim=embed_dim,
                                    stride=self.stride,
                                    pe=False,
                                    bias=lif_bias,
                                    tau=tau
                                    )
        self.encoding_neo = Embedding(num_patches=self.num_patches,
                                    patch_size=self.patch_size,
                                    embed_dim=embed_dim,
                                    stride=self.stride,
                                    pe=False,
                                    bias=lif_bias,
                                    tau=tau,
                                    neuron=emb_neuron                  # nsmt_flif_v1
                                    )

        self.data_block = nn.ModuleList([Block(T=T,
                    dim=embed_dim, seq_len=self.num_patches, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                    norm_layer=norm_layer, sr_ratio=sr_ratios, lif_bias=lif_bias, tau=tau, attn=(gating=='original'))
                    for j in range(depths)])

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
            
        # init_weight = torch.Tensor(self.T, embed_dim)
        # self.memory_slot = nn.parameter.Parameter(init_weight)
        # stdv = 1. / math.sqrt(self.memory_slot.size(1))
        # self.memory_slot.data.uniform_(-stdv, stdv)
        
        if ((self.train_mode == 'training') or (self.train_mode == 'pre_training')) : 
            self.weak_decoder = Decoder(embed_dim=embed_dim, d_out=self.patch_size, tau=tau, bias=lif_bias)
            
        self.head = nn.Linear(embed_dim, num_classes, bias=lif_bias) if num_classes > 0 else nn.Identity()
        self.dropout = nn.Dropout(drop_rate)
        if gating == 'ablation':
            self._init_ablation()

        self.apply(self._init_weights)
        
        if ((self.train_mode != 'training') and (self.train_mode != 'pre_training')):
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()

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
            
        if self.spk_encoding:
            x = self.spk_encoder(x)
        else: 
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1) 
            
        # x = rearrange(x, 't b l c -> t b c l')
        x = rearrange(x, 't b c l -> t (b c) l', c=1) # [T BC N P] 
        x = self.padding_patch_layer(x)
        patched_x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) 
        self.org_x = patched_x.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True)
        
        if (self.train_mode == 'training') and (self.keep_ratio > 0) : 
            keep_ratio=self.keep_ratio
            low_freq_x, _, _ = lowpass_memory(x, keep_ratio=keep_ratio, time_dim=2, center=True, rebinarize=False) #[T BC D]
            patched_low_freq_x = low_freq_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            self.org_x = patched_low_freq_x.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True)
            
        if self.roll_mode in ('sine', 'rand'):
            base = patched_x[0]                              # [BC, N, P] (all T identical after repeat)
            x_hippo = self._unroll(self.encoding(self._roll_T(base)))  # sine/rand roll, encode, realign
        else:
            x_hippo = self.encoding(patched_x)

        if self.no_neo or self.hippo_only:              # nsmt_flif_v1: hippo_only also has no neocortex path
            x_neo = None                                # ablation: no Neocortex path
        else:
            x_neo_emb = self.encoding_neo(patched_x.transpose(0, 2).contiguous().mean(2, keepdim=True)) #[T B N D]
            x_neo = self.time_block(x_neo_emb, org_seq)
        
        for dblk in self.data_block:
            x_hippo = dblk(x_hippo, x_neo, self_kv=self.hippo_only)
        
        if self.train_mode == 'training':
            
            return self._training(x_hippo, x_neo) 
            
        elif self.train_mode == 'testing':

            return self._testing(x_hippo)
        
        elif self.train_mode == 'visual':

            return self._visualization(x_hippo, x_neo)
        
        else:
            raise ValueError
        
        
    def _training(self, x_hippo, x_neo):
        x_data = x_hippo.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True)
        if x_neo is None:                               # no_neo: no reconstruction aux, no distill
            return self.head(x_hippo.mean(2)).mean(0), x_data, x_data, self.org_x, self.org_x
        rec_x = self.weak_decoder(x_neo, x_hippo) #[T BxC 1 P]

        assert self.org_x.shape == rec_x.shape, f"original x_time' shape is {self.org_x.shape}, and reconstructed x_time' shape is {rec_x.shape}"

        z = self.head(x_hippo.mean(2)).mean(0)

        return z, x_data, x_neo, self.org_x, rec_x

    
    def _testing(self, x):
        
        if self.gating == 'ablation':
            return self.head(x.mean(2)).mean(0)
        else:
            return self.head(x.mean(2)).mean(0), x.clone().detach()
    

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