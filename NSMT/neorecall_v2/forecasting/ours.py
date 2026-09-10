import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate
from spikingjelly.clock_driven import neuron as sj_neuron, functional as sj_functional

from timm.models import register_model
from timm.models.vision_transformer import _cfg 
from timm.layers import trunc_normal_
from timm.utils import *
from einops import rearrange

from scipy.linalg import hadamard

import matplotlib.pyplot as plt
import numpy as np
import math

from layers import SpkEncoder, SpikLinearLayer
from utils import RelBias1DDeterministicFn
from positional import tAPE
from layers import SSA_rel_scl, MLP, MutualCrossAttention, SpikLinearLayer, SpikLinearMaxLayer, SpikTimeLinearLayer
from neo_bank import NeoMemoryBank

__all__ = ['myModel']

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
                 rpe_rebinarize: bool = True) -> None:
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
        self.emb_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
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



class Block(nn.Module):
    def __init__(self, T, dim, seq_len, num_heads, mlp_ratios=2., max_ratio=2, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.1,
                drop_path=0.1, norm_layer=nn.LayerNorm, sr_ratio=1, lif_bias=False, tau=2.0, attn=False,
                time_num_layers=1, patch_size=16, neo_full_grad=False, neo_fuse='gate_attn',
                hippo_mixer='attn', hf_layers=2, hf_kernel=3, res_kw=None):
        super().__init__()

        topk_ratio = None
        self.neo_full_grad = neo_full_grad      # True -> remove the 0.05 grad throttle to the Neocortex
        self.neo_fuse = neo_fuse                # 'xattn' (condition B, --neo_recall) | 'gate_attn' (baseline reference)

        # 'attn' = the original spiking cross/self-attention. 'max' = Max-Former token mixer.
        # Pure module swap: the IAND residual around it is unchanged, so the contrast isolates the
        # mixer and nothing else.
        self.hippo_mixer = hippo_mixer
        self.mixer_analog = (hippo_mixer == 'reservoir' and (res_kw or {}).get('readout') == 'analog')
        self.mca = (ReservoirMixer(dim, tau_out=tau, bias=lif_bias, **(res_kw or {}))
                    if hippo_mixer == 'reservoir' else
                    MaxMixer(dim, layers=hf_layers, kernel_size=hf_kernel, tau=tau, bias=lif_bias)
                    if hippo_mixer == 'max' else
                    MutualCrossAttention(seq_len=seq_len, dim=dim, pe=False, out_seq=False,
                                         lif_bias=lif_bias, num_heads=num_heads, tau=tau, drop=drop))
        # (self.time_recon was built here and never used in any wiring -- 4224 dead parameters)
        mlp_hidden_dim = int(dim * mlp_ratios)
        mlp_input_dim = dim
        self.mlp = MLP(in_features=mlp_input_dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop, lif_bias=lif_bias, tau=tau)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mx=None):
        T, B, N, D = x.shape

        if self.hippo_mixer in ('max', 'reservoir'):      # self-contained mixers ignore `mx`
            r = self.mca(x)
            x = (x + r) if self.mixer_analog else (x * (1. - r))
            x = x * (1. - self.mlp(x))
            return x

        if mx is None:                                    # no_neo ablation: skip Memory-Replay fusion
            x = x * (1. - self.mlp(x))
            return x

        # gradient to the Neocortex: full (new) or 5%-throttled (original).
        if not self.neo_full_grad:
            mx = (0.05 * mx + (1 - 0.05) * mx.detach())   # [T B 1 D]

        if self.neo_fuse == 'xattn':                      # CONDITION B (--neo_recall): the caller passes
            # mx = x_hippo, so mca(x, mx) is plain SELF-attention and the Neocortex does not enter
            # the Block at all -- it is read back once, additively, after the block loop.
            x = x * (1. - self.mca(x, mx))
            x = x * (1. - self.mlp(x))
            return x

        if self.neo_fuse == 'xattn_hq':                    # q = Hippo, k/v = Neo. TRUE cross-attention:
            # the Neocortex is the thing being read, not a gate on Hippo's own signal. `mx` is the
            # Neocortex [N,BC,1,D]; transposing lifts its patch axis into the token slot so the
            # query's N tokens attend over the Neocortex's N tokens, broadcast across T.
            kv = mx.transpose(0, 2).contiguous().repeat(T, 1, 1, 1)   # [T, BC, N, D]
            x = x * (1. - self.mca(x, kv))
            x = x * (1. - self.mlp(x))
            return x

        # 'gate_attn' -- the original Memory-Replay fusion, kept only as the baseline reference arm:
        # Neo gates Hippo's own signal into a key/value, then an IAND residual (multiplicative).
        g = x * mx.transpose(0, 2).contiguous()
        x = x * (1. - self.mca(x, g))
        x = x * (1. - self.mlp(x))

        return x

class MaxLayer(nn.Module):
    """Linear -> BN -> sliding max over the PATCH axis -> LIF."""

    def __init__(self, dim, kernel_size=3, tau=2.0, bias=False):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=bias)
        self.bn = nn.BatchNorm1d(dim)
        self.pool = nn.MaxPool1d(kernel_size, stride=1, padding=kernel_size // 2)
        self.lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy',
                                    surrogate_function=surrogate.Sigmoid())

    def forward(self, x):                                    # [T, BC, N, D]
        T, B, N, D = x.shape
        y = self.bn(self.linear(x.reshape(-1, D))).reshape(T, B, N, D)
        y = self.pool(y.permute(0, 1, 3, 2).reshape(-1, D, N))       # sliding max over N
        y = y.reshape(T, B, D, N).permute(0, 1, 3, 2).contiguous()
        return self.lif(y)


class MaxMixer(nn.Module):
    """Max-Former token mixer (arXiv 2505.18608), ported to the 1-D patch axis.

    Spiking neurons low-pass, and our own measurement shows it: the Hippo representation sits at a
    DCT low-frequency fraction of 0.38 (ETTh1) / 0.68 (ETTm1) where white is 0.25 -- Hippo is
    currently carrying the SLOW component, not the detail. A sliding max over the token axis is the
    high-pass counterpart to attention's smoothing.

    Two departures from a naive port, both forced by our architecture:
      - The reference model is membrane-domain (LIF at the START of each sub-block, additive
        residual), so its max acts on ANALOG values. Ours is spike-domain, and a max over binary
        spikes is a logical OR -- a morphological dilation, which SMOOTHS rather than sharpens and
        would inverts the paper's intent. The max therefore sits after Linear+BN and before the LIF,
        matching the reference `Max_Embed` order (Conv -> BN -> MaxPool).
      - The pool is over the PATCH axis with stride 1 and 'same' padding, as in the reference
        `Max_Mixer` (kernel 3, stride 1, padding 1): a sliding max, NOT a downsample. The repo's
        pre-existing SpikLinearMaxLayer pooled the FEATURE axis with stride=kernel (64 -> 32), which
        has no temporal-frequency meaning at all.

    Two stacked layers give real depth because the LIF between them is nonlinear; a stack of bare
    max-pools would collapse to one wider pool.
    """

    def __init__(self, dim, layers=2, kernel_size=3, tau=2.0, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([MaxLayer(dim, kernel_size, tau, bias) for _ in range(layers)])

    def forward(self, x, mx=None):
        for b in self.blocks:
            x = b(x)
        return x


class ReservoirMixer(nn.Module):
    """The frozen liquid used as the Block's TOKEN MIXER, replacing attention entirely.

    The input keeps its spiking axis: [T, BC, N, D] -> the liquid scans the PATCH axis N once per
    spiking step, with T folded into the batch. Two consequences, both deliberate:
      - T survives the block, so the head still reads all T steps. The previous serial design pooled
        T away before the head and that alone cost +1.22% (8.19.2) -- the single largest loss in that
        sweep. This wiring avoids it.
      - The Neocortex is no longer a side pathway that can be ignored or co-adapted to (8.17); it IS
        the mixing operation, so there is no gate, no auxiliary loss and no fusion position to pick.

    Residual pairing is forced by the readout, not chosen freely:
      spike  -> IAND `x * (1 - r)`, the existing convention; needs r binary.
      analog -> additive `x + r`; r is unit-variance so `1 - r` would be meaningless, and additive
                is the membrane-domain convention the Max-Former reference itself uses.
    """

    def __init__(self, dim, readout='spike', **kw):
        super().__init__()
        self.res = SpikingReservoir(in_dim=dim, out_dim=dim, readout=readout, **kw)
        self.analog = (readout == 'analog')

    def forward(self, x, mx=None):                          # [T, BC, N, D]
        T, B, N, D = x.shape
        e = x.permute(2, 0, 1, 3).reshape(N, T * B, 1, D)   # patch axis first, T folded into batch
        y = self.res(e)                                     # [N, T*BC, 1, D]
        return y.reshape(N, T, B, D).permute(1, 2, 0, 3).contiguous()


class NeoLayer(nn.Module):
    """Exact-passive superset Neocortex layer (review 8.1). Passive and active share the SAME
    preactivation (BN over the full T sequence) + the SAME single-step LIF loop. Active adds a
    ZERO-INIT gated recurrent residual:
        pre_t = u_t + gamma * W_rec s_{t-1},   gamma = tanh(g),  g init 0  -> gamma init 0.
    At g=0 (init, and always for passive) pre_t == u_t, so active's output/membrane are IDENTICAL
    to passive; the recurrence is learned strictly on top (ReZero-style). This guarantees active is
    never worse than passive at start and gives a clean W_rec=0 == passive control (review 6.4)."""
    def __init__(self, dim, tau=2.0, bias=False, recurrent=False, rec_scale=0.1, plif=False, wrec_mode='full'):
        super().__init__()
        self.dim = dim
        self.recurrent = recurrent
        self.w_in = nn.Linear(dim, dim, bias=bias)
        self.bn   = nn.BatchNorm1d(dim)
        if plif:
            self.lif = sj_neuron.ParametricLIFNode(init_tau=tau, detach_reset=True, surrogate_function=surrogate.Sigmoid())
        else:
            self.lif = sj_neuron.LIFNode(tau=tau, detach_reset=True, surrogate_function=surrogate.Sigmoid())
        if recurrent:
            self.w_rec = nn.Linear(dim, dim, bias=bias)
            nn.init.uniform_(self.w_rec.weight, -rec_scale / dim**0.5, rec_scale / dim**0.5)
            self.w_rec._skip_global_init = True
            self.g = nn.Parameter(torch.zeros(1))            # zero-init gate -> gamma=tanh(0)=0 -> exact passive at start
            self.wrec_mode = wrec_mode
            if wrec_mode == 'frozen':
                self.w_rec.weight.requires_grad_(False)
            self._wrec_shuf = None
        self.store_v_seq = False
        self.v_seq = None

    def _rec(self, s):
        if self.wrec_mode == 'zero':
            return None
        if self.wrec_mode == 'shuffle':
            if self._wrec_shuf is None:
                W = self.w_rec.weight.data
                gen = torch.Generator(device='cpu').manual_seed(1234)
                perm = torch.randperm(W.numel(), generator=gen).to(W.device)
                self._wrec_shuf = W.flatten()[perm].reshape(W.shape)
            return torch.nn.functional.linear(s, self._wrec_shuf)
        return self.w_rec(s)                                 # full | frozen

    def forward(self, x):                                    # x: [T, B, N, D]
        T, B, N, D = x.shape
        u = self.bn(self.w_in(x).reshape(-1, D)).reshape(T, B, N, D)   # shared preactivation, full-T BN (== passive)
        sj_functional.reset_net(self.lif)
        gamma = torch.tanh(self.g) if self.recurrent else None
        s = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        outs, vs = [], []
        for t in range(T):
            pre = u[t]
            if self.recurrent:
                rec = self._rec(s)
                if rec is not None:
                    pre = pre + gamma * rec                  # gated re-entrant residual (0 at init)
            s = self.lif(pre)                                # single-step LIF (== MultiStepLIF over u at gamma=0)
            outs.append(s)
            if self.store_v_seq:
                vs.append(self.lif.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(vs, 0)
        return torch.stack(outs, 0)


class TemporalBlock(nn.Module):
    def __init__(self, T, num_layers=2, patch_size=[16, 32], embed_dim=[64, 128, 256], ratio=2, lif_bias=False, tau=2.0, num_heads=8, mutual=False, topk:int | None=None, topk_temperature:float = 1.0, recurrent=False, plif=False, wrec_mode='full', exact=True):
        super().__init__()
        self.recurrent = recurrent
        
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
        # The Neocortex stack: `num_layers` LIF layers on the patch axis (T=N convention).
        # `recurrent` stays a NeoLayer knob (zero-init gate == passive at start).
        self.Mem = nn.ModuleList(
            [NeoLayer(embed_dim, tau=tau, bias=lif_bias, recurrent=recurrent, plif=plif, wrec_mode=wrec_mode)
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
        # x = x.transpose(0, 2).contiguous()
        for i in range(self.num_layers):
            x = self.Mem[i](x)
        # x = x + self.bias(x)
        # x = self.out(x)

        return x

    @torch.no_grad()
    def visualize_mem_heatmap(self, save_dir="./mem_viz", filename="mem_heatmap.png",
                              title="self.mem heatmap", normalize=False):
        """
        self.mem: (1, 1, hidden_dim, embed_dim) -> heatmap: (hidden_dim, embed_dim)
        """
        import os
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        mem2d = self.mem.detach().float().squeeze(0).squeeze(0)  # [hidden_dim, embed_dim]
        if normalize:
            # 행/열 정규화 등 원하는 방식으로 바꿔도 됨 (예: 전체 표준화)
            m = mem2d.mean()
            s = mem2d.std().clamp_min(1e-8)
            mem2d = (mem2d - m) / s

        mem_np = mem2d.cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.imshow(mem_np, aspect="auto")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("embed_dim")
        plt.ylabel("hidden_dim")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    


class AnalogReadout(nn.Module):
    """Linear + BN with NO spiking nonlinearity.

    Measured on a trained x0 checkpoint, the liquid state phi holds 98-99% of its DCT energy in the
    lowest quarter of the patch axis (autocorr 0.98) -- exactly the slow trend the Neocortex is meant
    to carry. Linear and BatchNorm preserve that (0.97), and the readout LIF then collapses it to
    0.24 (autocorr ~0), i.e. white. The cause is the axis it integrates over: the readout LIF's time
    axis IS the patch axis, and at the ~2% firing rate the task loss settles on, each unit emits 0-1
    spikes across the 24 patches -- an impulse, whose spectrum is flat by construction.

    Dropping it keeps the trend intact. The cost is that the Neocortex output reaching the Block is
    no longer binary, so the attention's key/value Linear is a MAC rather than an AC (reservoir_summary
    10.3: allowed, but the AC-only energy claim must then not be made).
    """

    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        s = x.shape
        y = self.bn(self.linear(x.reshape(-1, s[-1])))
        return y.reshape(*s[:-1], -1)


class SpikingReservoir(nn.Module):
    """Fixed random recurrent spiking liquid, used as a drop-in replacement for the Neocortex core.

    The defining property of reservoir computing is that the DYNAMICS and the READOUT are learned
    separately: W_in / W_rec / tau / threshold are frozen buffers and the scan runs under no_grad,
    so the only trainable part is the projection that maps the reservoir feature back onto the
    Neocortex embedding contract. Condition B already hands the Neocortex zero task gradient
    (--neo_recall_grad stop), so the auxiliary reconstruction loss trains that projection alone --
    a gradient-descent stand-in for the ridge readout of the reference design.

    What this deliberately is NOT: the scan axis is the PATCH axis N supplied by the caller, there
    is no micro-simulation inside a patch (S_micro = 1), and no state survives a window because
    train.py still calls functional.reset_net every batch. That makes this WITHIN-WINDOW fading
    memory only -- it cannot support a cross-window or episodic memory claim.

    Two frozen contrasts share one topology seed, so `wrec='zero'` is the exact parameter-matched
    control that isolates recurrence from the rest of the swap.
    """

    def __init__(self, in_dim, out_dim, size=256, rec_density=0.10, in_density=0.10,
                 ei_ratio=0.80, rho=0.90, in_scale=1.0, n_traces=2, wrec='full',
                 seed=0, tau_mem=(4.0, 16.0), tau_syn=(2.0, 8.0), tau_out=8.0, bias=False,
                 readout='spike'):
        super().__init__()
        R = int(size)
        self.R, self.n_traces, self.out_dim, self.wrec = R, int(n_traces), out_dim, wrec
        self.feat_dim = (2 + self.n_traces) * R          # [spike, membrane, traces...]

        # Topology RNG is a LOCAL generator on purpose: drawing from the global stream would shift
        # every module built afterwards and break the name-seeded init contract (--init_order_fix).
        g = torch.Generator().manual_seed(int(seed))

        # Dale's law: a neuron is excitatory or inhibitory for ALL of its outgoing connections.
        sign = torch.ones(R)
        n_inh = int(round(R * (1.0 - float(ei_ratio))))
        sign[torch.randperm(R, generator=g)[:n_inh]] = -1.0

        W_rec = torch.rand(R, R, generator=g) * (torch.rand(R, R, generator=g) < rec_density)
        W_rec.fill_diagonal_(0.)
        W_rec = W_rec * sign.unsqueeze(0)                # column j carries source neuron j's sign
        ev = torch.linalg.eigvals(W_rec.to(torch.complex64)).abs().max().item()
        if ev > 1e-8:
            W_rec = W_rec * (float(rho) / ev)            # scale to a target spectral radius
        if wrec == 'zero':
            W_rec = torch.zeros_like(W_rec)              # CONTROL: liquid without recurrence

        W_in = (torch.rand(in_dim, R, generator=g) * 2.0 - 1.0) \
               * (torch.rand(in_dim, R, generator=g) < in_density)
        W_in = W_in * (float(in_scale) / max(1e-8, (in_dim * in_density) ** 0.5))

        # heterogeneous, log-spaced time constants (a single tau makes the liquid one filter)
        tm = torch.exp(torch.linspace(math.log(tau_mem[0]), math.log(tau_mem[1]), R))
        ts = torch.exp(torch.linspace(math.log(tau_syn[0]), math.log(tau_syn[1]), R))
        lam = torch.tensor([0.5 + 0.45 * k / max(1, self.n_traces - 1) for k in range(self.n_traces)]) \
            if self.n_traces > 1 else torch.tensor([0.9])

        for n, t in (('W_in', W_in), ('W_rec', W_rec), ('a_m', 1.0 - 1.0 / tm),
                     ('a_s', 1.0 - 1.0 / ts), ('lam', lam), ('sign', sign)):
            self.register_buffer(n, t)                   # buffers, NOT Parameters -> no optimizer, no grad

        # the only trainable part: reservoir feature -> Neocortex embedding.
        # 'spike' keeps the original contract (binary out, AC accounting) but whitens the liquid's
        # slow structure; 'analog' preserves it. See AnalogReadout for the measurement.
        self.readout = str(readout)
        self.proj = (AnalogReadout(self.feat_dim, out_dim, bias=bias) if self.readout == 'analog'
                     else SpikLinearLayer(self.feat_dim, out_dim, tau=tau_out, lif_bias=bias))
        self.last_stats = {}
        self.tap_phi = False          # diagnostics only
        self.last_phi = None

    def forward(self, e, org_seq=None):
        """e: [N, BC, 1, in_dim] (patch axis first) -> [N, BC, 1, out_dim] spikes."""
        N, B = e.shape[0], e.shape[1]
        with torch.no_grad():
            u_in = e.detach().reshape(N, B, -1)
            i = torch.zeros(B, self.R, device=e.device, dtype=e.dtype)
            u = torch.zeros_like(i)
            s = torch.zeros_like(i)
            q = torch.zeros(self.n_traces, B, self.R, device=e.device, dtype=e.dtype)
            lam = self.lam.view(-1, 1, 1)
            outs = []
            for t in range(N):
                i = self.a_s * i + u_in[t] @ self.W_in + s @ self.W_rec
                u = self.a_m * u + (1.0 - self.a_m) * i
                s = (u >= 1.0).to(u.dtype)
                u = u - s                                # subtract-threshold reset (vth = 1)
                q = lam * q + (1.0 - lam) * s
                outs.append(torch.cat([s, u] + list(q), dim=-1))
            phi = torch.stack(outs, 0)                   # [N, B, feat_dim]
            rate = phi[..., :self.R]
            per_neuron = rate.mean(dim=(0, 1))
            self.last_stats = {'rate': rate.mean().item(),
                               'silent': (per_neuron < 1e-6).float().mean().item(),
                               'saturated': (per_neuron > 0.9).float().mean().item()}
            if self.tap_phi:                 # diagnostics: liquid state BEFORE the readout
                self.last_phi = phi
        return self.proj(phi.detach().unsqueeze(2))      # [N, B, 1, out_dim]


class myModel(nn.Module):
    def __init__(self, gating=['original', 'ablation', 'attn'],
                 train_mode=['pretraining', 'training', 'testing', 'visual'], 
                patch_size=63, 
                pred_len=0, 
                seq_len=0,
                time_num_layers=2,
                embed_dim=[64, 128, 256], 
                num_heads=[1, 2, 4], 
                mlp_ratios=1, 
                max_ratio=2,
                qkv_bias=False, 
                qk_scale=None,
                keep_ratio=0.25,
                drop_rate=0.1, 
                attn_drop_rate=0., 
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm,
                depths=[6, 8, 6], 
                sr_ratios=[8, 4, 2], 
                bias=False, 
                tau=2.0, 
                spk_encoding=False,
                pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                **kwargs,
                ):
        super().__init__()
        
        self.train_mode = train_mode
        self.gating = gating
        self.pred_len = pred_len
        self.keep_ratio = keep_ratio
        self.seq_len = seq_len
        
        # self.T = T  # time step
        self.spk_encoding = spk_encoding
        self.patch_size = patch_size
        # review 7.5: honor --no_overlap. Previously stride was hardcoded to patch_size//2, so any
        # experiment passing --no_overlap silently kept 50% overlap. Now stride = patch_size when
        # no_overlap is set (non-overlapping patches), else patch_size//2 (default 50% overlap).
        self.no_overlap = bool(kwargs.get('no_overlap', False))
        self.stride = patch_size if self.no_overlap else patch_size // 2

        self.num_patches = int((seq_len - patch_size) / (self.stride) + 1)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.num_patches = self.num_patches + 1
        self.T = self.num_patches
        self.dual_stream = bool(kwargs.get('dual_stream', False))   # two single-layer nets, mutual x-attn
        self.dual_dim = int(kwargs.get('dual_dim', 0)) or embed_dim   # per-stream width (capacity knob)
        self.dual_attn = str(kwargs.get('dual_attn', 'cross'))  # 'cross' (streams fused) | 'self' (independent)
        # --- Neo-as-recall: Hippo predicts; Neo reconstructs the window FROM Hippo's own
        # pre-prediction representation and is read back through a gate. Both directions are
        # stop-gradient, so each network is trained solely by its own objective.
        self.neo_recall      = bool(kwargs.get('neo_recall', False))
        self.neo_recall_gate = str(kwargs.get('neo_recall_gate', 'alpha'))   # 'alpha' | 'mul'
        self.neo_recall_tgt  = str(kwargs.get('neo_recall_tgt', 'raw'))      # 'raw' | 'lowfreq'
        self.neo_recall_grad = str(kwargs.get('neo_recall_grad', 'stop'))    # 'stop'|'full'|'to_neo'|'to_hippo'
        self.neo_mask_ratio  = float(kwargs.get('neo_mask_ratio', 0.0))      # >0: masked reconstruction aux (exp A)
        self.neo_backend     = str(kwargs.get('neo_backend', 'neolayer'))    # 'neolayer' (trained) | 'reservoir' (frozen)
        self.hippo_mixer     = str(kwargs.get('hippo_mixer', 'attn'))        # 'attn' | 'max' (Max-Former)
        self.serial_mode     = str(kwargs.get('serial_mode', 'off'))         # off | fold | skip
        self.neo_xattn       = bool(kwargs.get('neo_xattn', False))          # Block does q=Hippo, kv=Neo cross-attn
        self.no_aux          = bool(kwargs.get('no_aux', False))             # auxiliary reconstruction path does not exist
        # With the aux loss off the Neocortex pathway is UNREACHABLE whenever it is not read back
        # into the prediction -- either the gate is 'none', or the mixer is self-contained and
        # ignores it. It was still being built and run: 8,961 parameters with no gradient plus a
        # wasted forward pass every step.
        self._neo_path_off   = (self.no_aux and not bool(kwargs.get('neo_out_residual', False))
                                and (self.hippo_mixer in ('max', 'reservoir')
                                     or self.neo_recall_gate == 'none'))
        self.refine_steps    = int(kwargs.get('refine_steps', 1))            # extra passes after the bootstrap pass
        self.aux_mask = None                                                 # per-forward held-out mask (exp A)
        self.dual_head = str(kwargs.get('dual_head', 'f'))   # 'f'=stream-1 only (old) | 'sum' | 'cat'
        self.dual_mode = str(kwargs.get('dual_mode', 'ft'))   # 'ft' proposal | 'tt' | 'ff' (identical params, input-only controls)

        
        print(f"len of tokens in sequence >> {self.num_patches}")

        self.spk_encoder = SpkEncoder(self.T) if spk_encoding else None

        # --- Neocortex input/fusion controls (default = original NSMT behaviour) ---
        self.neo_tau = float(kwargs.get('neo_tau', tau))               # P0: slow-leak Neo (long-term memory). default=hippo tau
        self.aux_membrane = bool(kwargs.get('aux_membrane', False))    # low-pass aux from Neo MEMBRANE (leaky integrator=low-pass) not spikes
        self.neo_recurrent = bool(kwargs.get('neo_recurrent', False))  # P1: re-entrant recurrent Neo (Zipser attractor, long-term hold)
        self.neo_exact = bool(kwargs.get('neo_exact', True))           # review 8.1: exact-passive superset NeoLayer (zero-init gate); active==passive at start
        self.no_neo = bool(kwargs.get('no_neo', False))                # ablation: Hippocampus only (no Neocortex/Memory Replay/low-pass aux)
        self.neo_plif = bool(kwargs.get('neo_plif', False))            # learnable membrane tau (auto-tune attractor memory timescale)
        self.neo_full_grad = bool(kwargs.get('neo_full_grad', False))   # full gradient to Neocortex (remove 5% throttle)
        self.neo_wrec_mode = kwargs.get('wrec_mode', 'full')            # R1 ablation: full|zero|shuffle|frozen W_rec
        self.mask_past = int(kwargs.get('mask_past', 0))
        self.probe_neo = bool(kwargs.get('probe_neo', False))          # mechanism probe: log x_neo smoothness stats                # task-relevant memory probe: blank first-k look-back steps
        self.neo_membrane_out = bool(kwargs.get('neo_membrane_out', False))
        # --- Interpretation-B wavelet encoding: SCALE axis -> spiking simulation axis ---
        # Detail bands become the T axis (Hippocampus); the approximation (lowest-frequency)
        # band is routed to the Neocortex, matching its stated slow-trend role analytically.
        self.wavelet_enc     = bool(kwargs.get('wavelet_enc', False))
        self.wavelet_levels  = int(kwargs.get('wavelet_levels', 4))
        self.wavelet_name    = str(kwargs.get('wavelet_name', 'b3'))
        self.wavelet_mode    = str(kwargs.get('wavelet_mode', 'wavelet'))   # 'wavelet' | 'rand' (control)
        self.wavelet_shuffle = bool(kwargs.get('wavelet_shuffle', False))   # control: scramble scale order
        self.wavelet_neo_appr = bool(kwargs.get('wavelet_neo_appr', True))  # route cA_J to the Neocortex
        if self.wavelet_enc:
            from wavelet_enc import SWTEncoder
            self.swt = SWTEncoder(levels=self.wavelet_levels, name=self.wavelet_name,
                                  mode=self.wavelet_mode)                   # buffers only, 0 params
            self.register_buffer('wav_perm', torch.randperm(self.wavelet_levels))  # #1: feed LIF membrane (continuous trend) to Memory Replay, not spikes
        self.neo_predcode = bool(kwargs.get('neo_predcode', False))    # #2: latent predictive-coding aux (neo[t] predicts next-patch embedding[t+1])
        self.neo_fuse = str(kwargs.get('neo_fuse', 'gate_attn'))        # cross-attn/gate fusion variant
        self.neo_seq = str(kwargs.get('neo_seq', 'identity'))          # neo input composition along its time axis N
        self.neo_aux_next = bool(kwargs.get('neo_aux_next', False))     # reconstruction aux -> NEXT-patch (1-step) prediction
        # aux loss form. Previously hard-wired to neo_aux_next, which confounded "next-patch target"
        # with "MAE instead of MSE"; it is its own flag so the target can be swapped in isolation.
        self.aux_l1 = bool(kwargs.get('aux_l1', False))                 # train.py: MAE aux instead of MSE
        # fixed causal sinusoidal gather index for neo_seq='sine' (APE-like position-structured selection of past tokens)
        N = self.num_patches
        n = torch.arange(N, dtype=torch.float32)
        s = 0.5 * (1.0 + torch.sin(2.0 * math.pi * n / max(N, 1)))      # [0,1] sinusoid of absolute position
        idx = torch.round(n * s).long().clamp_(0, N - 1)               # causal (<= n) position-dependent past index
        self.register_buffer('neo_sine_idx', idx, persistent=False)

        # Hippocampus spike-encoding over the (otherwise-wasted, settled) spiking T axis: replace
        # the identical repeat with a per-step cyclic shift schedule (docs/24,26). parameter-free.
        self.hippo_enc = str(kwargs.get('hippo_enc', 'repeat'))         # 'repeat'(orig) | 'sine' | 'rand'
        self.hippo_boundary = str(kwargs.get('hippo_boundary', 'wrap')) # wrap(cyclic) | zero | edge
        self.hippo_unroll = str(kwargs.get('hippo_unroll', 'pre'))      # pre(unroll before fusion) | post(after data_block; fuse in rolled frame)
        _T = self.T
        _t = torch.arange(_T, dtype=torch.float32)
        if self.hippo_enc == 'rand':
            _sched = torch.randint(0, _T, (_T,)).sort(descending=True).values.long(); _sched[-1] = 0
        elif self.hippo_enc == 'sine':
            _sched = torch.round((_T - 1) * 0.5 * (1.0 + torch.cos(math.pi * _t / max(_T - 1, 1)))).long()
        else:
            _sched = (_T - 1 - _t).long()                              # unused for repeat
        self.register_buffer('hippo_shift', _sched.clamp_(0, _T - 1), persistent=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.encoding = None if self.dual_stream else Embedding(num_patches=self.num_patches,
                                    patch_size=self.patch_size,
                                    embed_dim=embed_dim,
                                    stride=self.stride,
                                    pe=False,
                                    bias=bias,
                                    tau=tau
                                    )
        # condition B builds the Neocortex input from Hippo's representation, not from raw patches,
        # so this patch embedding has no consumer there.
        self.encoding_neo = None if (self.dual_stream or self.neo_recall) else Embedding(num_patches=self.num_patches,
                                    patch_size=self.patch_size,
                                    embed_dim=embed_dim,
                                    stride=self.stride,
                                    pe=False,
                                    bias=bias,
                                    tau=self.neo_tau                   # P0: slow-leak Neo (long-term memory)
                                    )


        # self.encoding_neo = SpikLinearLayer(patch_size, embed_dim, tau=tau, lif_bias=bias)
        self.data_block = nn.ModuleList([] if self.dual_stream else [Block(
                    T=self.T, dim=embed_dim, seq_len=self.num_patches, num_heads=num_heads, mlp_ratios=mlp_ratios, max_ratio=max_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                    norm_layer=norm_layer, sr_ratio=sr_ratios, lif_bias=bias, tau=tau, attn=(self.gating == 'attn'),
                    time_num_layers=time_num_layers, patch_size=self.patch_size,
                    neo_full_grad=self.neo_full_grad,
                    hippo_mixer=self.hippo_mixer,
                    hf_layers=int(kwargs.get('hf_layers', 2)), hf_kernel=int(kwargs.get('hf_kernel', 3)),
                    res_kw=(dict(size=int(kwargs.get('res_size', 256)),
                                 rec_density=float(kwargs.get('res_rec_density', 0.10)),
                                 in_density=float(kwargs.get('res_in_density', 0.10)),
                                 ei_ratio=float(kwargs.get('res_ei_ratio', 0.80)),
                                 rho=float(kwargs.get('res_rho', 0.90)),
                                 in_scale=float(kwargs.get('res_in_scale', 30.0)),
                                 n_traces=int(kwargs.get('res_traces', 2)),
                                 wrec=str(kwargs.get('res_wrec', 'full')),
                                 seed=int(kwargs.get('res_seed', 0)),
                                 readout=str(kwargs.get('res_readout', 'spike')))
                            if self.hippo_mixer == 'reservoir' else None),
                    neo_fuse=('xattn_hq' if self.neo_xattn
                              else 'xattn' if bool(kwargs.get('neo_recall', False))
                              else ('stream' if bool(kwargs.get('neo_stream', False)) else self.neo_fuse)))
                    for j in range(depths)])
        
        # --- two-stream Neocortex: SAME spiking-transformer architecture as the Hippocampus,
        # fed the TIME-domain (direct-coded) signal while the Hippocampus gets the wavelet bands.
        self.neo_stream = bool(kwargs.get('neo_stream', False))
        self.neo_stream_input = str(kwargs.get('neo_stream_input', 'time'))   # 'time' | 'freq'
        if self.dual_stream:
            _dd = self.dual_dim
            self.enc_freq = Embedding(num_patches=self.num_patches, patch_size=self.patch_size,
                                      embed_dim=_dd, stride=self.stride, pe=False, bias=bias, tau=tau)
            self.enc_time = Embedding(num_patches=self.num_patches, patch_size=self.patch_size,
                                      embed_dim=_dd, stride=self.stride, pe=False, bias=bias, tau=tau)
            _mk = lambda: Block(T=self.T, dim=_dd, seq_len=self.num_patches, num_heads=num_heads,
                    mlp_ratios=mlp_ratios, max_ratio=max_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
                    sr_ratio=sr_ratios, lif_bias=bias, tau=tau, attn=(self.gating == 'attn'),
                    time_num_layers=time_num_layers, patch_size=self.patch_size,
                    neo_full_grad=True, neo_fuse='xattn')
            self.blk_freq = _mk()          # single layer, q=freq  k/v=time
            self.blk_time = _mk()          # single layer, q=time  k/v=freq
            _hin = _dd * 2 if self.dual_head == 'cat' else _dd
            if self.dual_head == 'attnpool':
                # Learned query attends over the tokens of BOTH streams (2N of them) and pools
                # them into one vector -> the head shrinks from D*N to D, and BOTH streams reach
                # the forecast by construction (no dead branch possible).
                self.pool_q = nn.Parameter(torch.randn(_dd) * 0.02)
                self.head_dual = nn.Linear(_dd, pred_len, bias=bias)
            else:
                self.head_dual = nn.Linear(_hin * self.num_patches, pred_len, bias=bias)
        if self.neo_recall:
            if self.serial_mode == 'off' and not self._neo_path_off:
                self.recall_alpha = nn.Parameter(torch.tensor(4.0))   # sigmoid(4)=0.982 -> starts ~pure Hippo
        if self.neo_stream:
            self.encoding_stream = Embedding(num_patches=self.num_patches, patch_size=self.patch_size,
                                             embed_dim=embed_dim, stride=self.stride, pe=False,
                                             bias=bias, tau=tau)
            self.neo_blocks = nn.ModuleList([Block(
                    T=self.T, dim=embed_dim, seq_len=self.num_patches, num_heads=num_heads,
                    mlp_ratios=mlp_ratios, max_ratio=max_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer,
                    sr_ratio=sr_ratios, lif_bias=bias, tau=tau, attn=(self.gating == 'attn'),
                    time_num_layers=time_num_layers, patch_size=self.patch_size,
                    neo_full_grad=True, neo_fuse='stream')
                    for j in range(depths)])
        if self.serial_mode == 'skip':
            self.serial_gamma = nn.Parameter(torch.zeros(1))   # zero-init -> starts exactly at the `fold` control
        if self.neo_backend == 'reservoir':
            if not (self.neo_recall or self.serial_mode == 'skip'):
                raise ValueError("--neo_backend reservoir is wired for condition B only (needs --neo_recall)")
            self.reservoir = SpikingReservoir(
                in_dim=embed_dim, out_dim=embed_dim,
                size=int(kwargs.get('res_size', 256)),
                rec_density=float(kwargs.get('res_rec_density', 0.10)),
                in_density=float(kwargs.get('res_in_density', 0.10)),
                ei_ratio=float(kwargs.get('res_ei_ratio', 0.80)),
                rho=float(kwargs.get('res_rho', 0.90)),
                in_scale=float(kwargs.get('res_in_scale', 1.0)),
                n_traces=int(kwargs.get('res_traces', 2)),
                wrec=str(kwargs.get('res_wrec', 'full')),
                seed=int(kwargs.get('res_seed', 0)),
                readout=str(kwargs.get('res_readout', 'spike')),
                tau_out=self.neo_tau, bias=bias)
        else:
            self.reservoir = None
        self.time_block = None if (self.dual_stream or self.reservoir is not None
                                   or self.serial_mode != 'off' or self._neo_path_off) else TemporalBlock(T=self.T,
                                        num_layers=time_num_layers,
                                        patch_size=patch_size,
                                        embed_dim=embed_dim,
                                        lif_bias=bias,
                                        num_heads=num_heads,
                                        tau=self.neo_tau,              # P0: slow-leak Neo (long-term memory)
                                        mutual=False,
                                        topk=16,
                                        recurrent=self.neo_recurrent,  # P1: re-entrant recurrent Neo
                                        plif=self.neo_plif,            # learnable tau
                                        wrec_mode=self.neo_wrec_mode,  # R1 W_rec isolation
                                        exact=self.neo_exact)          # review 8.1: exact-passive superset
        if (self.time_block is not None) and \
                (self.aux_membrane or getattr(self, 'probe_neo', False) or self.neo_membrane_out):
            # store membrane of the Neo's last LIF (aux/probe/#1 readout). The reservoir backend has
            # no NeoLayer stack; its membrane is already part of the readout feature.
            if self.neo_exact or self.neo_recurrent:
                self.time_block.Mem[-1].store_v_seq = True             # NeoLayer + RecurrentNeoLayer expose .v_seq directly
            else:
                self.time_block.Mem[-1].spk_neuron.store_v_seq = True
        
        if (self.train_mode == 'training') and self.serial_mode == 'off' and not self.no_aux:
            self.weak_decoder = Decoder(embed_dim=self.dual_dim, d_out=self.patch_size, tau=tau, bias=bias)
            # self.weak_decoder = Decoder(embed_dim=embed_dim, d_out=None, tau=tau, bias=bias)
        if self.neo_predcode:                                          # #2 latent predictive-coding head (D -> D)
            self.predcode_head = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.head = None if self.dual_stream else nn.Linear(embed_dim * self.num_patches, pred_len, bias=bias)

        # review 8.7: prediction-space TREND RESIDUAL. The head reads only x_hippo, so the Neocortex
        # can merely *inhibit* hippocampal spikes (IAND gate) -- a good trend state can never be ADDED
        # to the forecast. Here the Neocortex predicts a trend component that is added to the detail
        # forecast, mirroring the trend+seasonal decomposition that works well in DLinear/Autoformer:
        #     y = head(x_hippo) + tanh(b) * trend_head(x_neo),   b init 0  -> starts EXACTLY at baseline
        # (zero-init: the residual can only be switched on by training if it helps; never worse at init.)
        self.neo_out_residual = bool(kwargs.get('neo_out_residual', False))
        if self.neo_out_residual:
            self.trend_head = nn.Linear(embed_dim, pred_len, bias=bias)
            self.out_beta = nn.Parameter(torch.zeros(1))

        # Optional CLS-inspired teacher->student distillation.
        # Stage-1: the Hippocampus pathway (head on x_hippo) is the Teacher;
        # a lightweight Student head reads the Neocortex output x_neo and is
        # trained to (a) match the Teacher forecast (KD) and (b) solve the task.
        # Inference still uses the Teacher by default; set ``model.eval_student``
        # to evaluate the Student-only path. Disabled unless --use_kd is passed.
        self.use_kd = bool(kwargs.get("use_kd", False))
        self.eval_student = False
        # student head capacity:
        #   'pooled' (default): Linear(D -> pred_len) applied per token, averaged
        #   'joint'           : Linear(N*D -> pred_len) over all tokens jointly
        #                       (mirrors the Teacher head; Option-1 in the ablation)
        self.kd_head_mode = kwargs.get("kd_head_mode", "pooled")
        if self.use_kd:
            if self.kd_head_mode == "joint":
                self.neo_head = nn.Linear(embed_dim * self.num_patches, pred_len, bias=bias)
            else:
                self.neo_head = nn.Linear(embed_dim, pred_len, bias=bias)

        # Optional CLS-inspired EMA neocortex memory bank.
        # Defaults preserve original behaviour (bank disabled) when the
        # extra kwargs are absent, so existing call sites do not change.
        self.use_neo_bank = bool(kwargs.get("use_neo_bank", False))
        if self.use_neo_bank:
            self.neo_bank = NeoMemoryBank(
                T=self.T, embed_dim=embed_dim,
                decay=float(kwargs.get("neo_bank_decay", 0.99)),
                alpha=float(kwargs.get("neo_bank_alpha", 0.1)),
                thresh=float(kwargs.get("neo_bank_thresh", 0.10)),
            )

        if gating == 'ablation':
            self._init_ablation()

        # ---- reproducibility fix: ORDER-INDEPENDENT initialisation -------------------
        # `self.apply(_init_weights)` walks modules in registration order and every
        # trunc_normal_ draws sequentially from the global RNG. Therefore merely ADDING a
        # module -- even one that is never called in forward (e.g. a deeper TemporalBlock
        # under --no_neo) -- shifts the weights of every module initialised after it.
        # Measured: two FUNCTIONALLY IDENTICAL no-Neo models (time_layers 1 vs 2, same seed)
        # differ in 11/56 tensors and by ~1.4% test MSE on ETTh2 p96 -- the same magnitude as
        # the effects we report. Fix: derive a deterministic RNG stream per module NAME, so a
        # module's init depends only on (seed, name), never on what else exists.
        if bool(kwargs.get('init_order_fix', False)):
            self._init_weights_ordered()
        else:
            self.apply(self._init_weights)

        if ((self.train_mode != 'training') and (self.train_mode != 'pre_training')):
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()

        if self.wavelet_enc:
            # Runtime spiking steps = detail-band count. Set AFTER module construction so every
            # module is still built with num_patches -> parameter shapes identical to baseline.
            self.T = self.wavelet_levels
    
    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, N):
        if N == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(N), mode="bilinear").reshape(1, -1, N).permute(0, 2, 1)

    def _patchify(self, z):
        """[B,L,C] or [T,B,L,C] -> padded [T, BC, L'] ready for unfold (same path as the main input)."""
        if z.dim() == 3:
            z = z.unsqueeze(0)
        z = rearrange(z, 't b l c -> t b c l')
        z = rearrange(z, 't b c l -> t (b c) l')
        return self.padding_patch_layer(z)

    def _init_weights_ordered(self):
        """Name-seeded initialisation: weights depend only on (seed, module name)."""
        import zlib
        base = torch.initial_seed() % (2 ** 31)
        for name, m in self.named_modules():
            if getattr(m, '_skip_global_init', False):
                continue
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                torch.manual_seed((base * 1000003 + zlib.crc32(name.encode())) % (2 ** 31))
                self._init_weights(m)
        # leave the global RNG in a state that does not depend on which modules were built
        torch.manual_seed(base)

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
            
        
    def forward(self, x):

        self.B, L, self.M = x.shape
        # task-relevant memory probe: blank the earliest `mask_past` steps of the look-back
        # window (replace with the mean of the remaining steps) -> tests reliance on far-past.
        _mp = getattr(self, 'mask_past', 0)
        if _mp and 0 < _mp < L:
            x = x.clone()
            x[:, :_mp, :] = x[:, _mp:, :].mean(1, keepdim=True)
        org_seq = x.clone().detach() #[B L C]
        org_seq_norm = None   # set after instance-norm (below): raw time-domain input for the stream
        
        self.means = x.mean(1, keepdim=True).detach()
        x = x - self.means
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= self.stdev
        org_seq_norm = x.clone()          # normalised time-domain signal (two-stream source)
        
        _wav_appr = None; _wav_raw = None
        if self.wavelet_enc:
            det, appr = self.swt(x)                       # [J,B,L,C] details (fine->coarse), [B,L,C] approx
            if self.wavelet_shuffle:
                det = det[self.wav_perm]                  # CONTROL: scramble the scale ordering along T
            _wav_appr, _wav_raw = appr, x                 # cA_J -> Neocortex ; raw signal -> aux target
            x = det                                       # T axis is now the SCALE axis
        elif self.spk_encoding:
            x = self.spk_encoder(x)
        else:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1) 
            
        x = rearrange(x, 't b l c -> t b c l')
        x = rearrange(x, 't b c l -> t (b c) l') # [T BC N P] 
        x = self.padding_patch_layer(x)
        patched_x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) 
        _aux_x = self._patchify(_wav_raw) if self.wavelet_enc else x   # aux targets the SIGNAL, not the bands
        _aux_patched = _aux_x.unfold(dimension=-1, size=self.patch_size, step=self.stride) if self.wavelet_enc else patched_x
        self.org_x = _aux_patched.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True)
        
            
        if (self.train_mode == 'training') and (self.keep_ratio > 0) and (not self.neo_aux_next) \
                and not (self.neo_recall and self.neo_recall_tgt == 'raw'):
            # low-freq(DCT) reconstruction target. When neo_aux_next: keep RAW patch target
            # so the aux becomes next-RAW-token (1-step) prediction along the patch sequence.
            keep_ratio=self.keep_ratio
            low_freq_x, _, _ = lowpass_memory(_aux_x, keep_ratio=keep_ratio, time_dim=2, center=True, rebinarize=False) #[T BC D]
            patched_low_freq_x = low_freq_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            self.org_x = patched_low_freq_x.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True)
            
        # vis(x, low_freq_x, M=self.M, keep_ratio=keep_ratio)
        rolled_hippo = self.hippo_enc in ('sine', 'rand')
        if rolled_hippo and self.wavelet_enc:
            # patched_x[0] assumes an identical repeat along T; under wavelet encoding T is the
            # SCALE axis, so this would silently discard bands d_2..d_J. Refuse instead.
            raise ValueError("hippo_enc='sine'/'rand' is incompatible with --wavelet_enc "
                             "(T is the scale axis; patched_x[0] would drop bands d_2..d_J)")
        if rolled_hippo:
            base = patched_x[0]                                  # [BC, N, P] (all T identical after repeat)
            x_hippo = self.encoding(self._hippo_roll(base))      # shift-encode the wasted T axis
            if self.hippo_unroll == 'pre':
                x_hippo = self._hippo_unroll(x_hippo)            # realign before fusion (Exp A)
        else:
            x_hippo = None if self.dual_stream else self.encoding(patched_x)

        # x_neo = x_hippo.transpose(0, 2).contiguous().mean(2, keepdim=True)
        # if self.perm and self.training:
        #     idx = torch.randperm(self.T, device=x.device)
        #     patched_x = patched_x.index_select(2, idx)  
        

        if self.dual_stream:
            # DUAL single-layer networks. Hippocampus reads the FREQUENCY domain (wavelet bands
            # on the spiking T axis), Neocortex reads the TIME domain (ordinary direct coding).
            # Their self-attention is REPLACED by mutual cross-attention: each stream's query
            # attends to the other stream's key/value, so the only interaction is the exchange.
            _t = _wav_raw if (self.wavelet_enc and _wav_raw is not None) else org_seq_norm
            _tp = self._patchify(_t.unsqueeze(0).repeat(patched_x.shape[0], 1, 1, 1))
            _tp = _tp.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # [T,BC,N,P]
            _f_in = patched_x if self.dual_mode in ('ft', 'ff') else _tp   # stream-1 input
            _t_in = _tp if self.dual_mode in ('ft', 'tt') else patched_x   # stream-2 input
            xf = self.enc_freq(_f_in)                     # [T,BC,N,D]
            xt = self.enc_time(_t_in)                     # [T,BC,N,D]
            if self.dual_attn == 'self':
                hf = self.blk_freq(xf, xf)                # independent: self-attention only
                ht = self.blk_time(xt, xt)
            else:
                hf = self.blk_freq(xf, xt)                # q=freq, k/v=time (mutual cross-attn)
                ht = self.blk_time(xt, xf)                # q=time, k/v=freq
            if self.dual_head == 'attnpool':
                _h = torch.cat([hf, ht], dim=2)                       # [T,BC,2N,D]
                _w = torch.softmax((_h * self.pool_q).sum(-1), dim=-1)  # [T,BC,2N]
                x_hippo = (_h * _w.unsqueeze(-1)).sum(2, keepdim=True)  # [T,BC,1,D]
            elif self.dual_head == 'sum':
                x_hippo = hf + ht                       # additive: BOTH streams reach the head
            elif self.dual_head == 'cat':
                x_hippo = torch.cat([hf, ht], dim=-1)   # concat along D (head input doubled)
            else:
                x_hippo = hf                            # 'f': stream-1 only (the original wiring;
                                                        # blk_time was a dead branch under this)
            x_neo = ht
            self._dual = True
        elif self.neo_stream:
            # TWO-STREAM. Hippocampus already holds the frequency-domain bands (patched_x).
            # The Neocortex stream gets the TIME-domain signal under ordinary direct coding,
            # repeated to the same T so both streams share [T, BC, N, D] exactly.
            _src = _wav_raw if (self.wavelet_enc and _wav_raw is not None) else None
            if _src is None:                       # no wavelet -> reuse the raw normalised input
                _src = org_seq_norm
            _rep = self._patchify(_src.unsqueeze(0).repeat(x_hippo.shape[0], 1, 1, 1)
                                  if _src.dim() == 3 else _src)
            _rep = _rep.unfold(dimension=-1, size=self.patch_size, step=self.stride)   # [T,BC,N,P]
            x_stream = self.encoding_stream(_rep)                                      # [T,BC,N,D]
            if self.neo_stream_input == 'freq':     # CONTROL: both streams frequency-domain
                x_stream = self.encoding_stream(patched_x)
            for nblk in self.neo_blocks:
                x_stream = nblk(x_stream, x_stream)                                    # self-attention
            x_neo = x_stream
        elif self.no_neo:
            x_neo = None                                                       # ablation: no Neocortex path
        elif self._neo_path_off:
            x_neo = None                       # pathway removed: unreachable with --no_aux
        elif self.neo_recall:
            # CONDITION B: the Neocortex reads Hippo's POST-block representation, so x_neo is built
            # in the recall block further down. Running the raw-patch Neocortex pass here would only
            # be overwritten there -- it was wasted compute that still charged energy and nudged the
            # Neocortex BatchNorm running statistics. Skipped; `_neo_post` runs in the recall block.
            x_neo = None
        else:
            if self.wavelet_enc and self.wavelet_neo_appr:
                _ap = self._patchify(_wav_appr).unfold(dimension=-1, size=self.patch_size, step=self.stride)
                neo_in = _ap.transpose(0, 2).contiguous().mean(2, keepdim=True)     # [N BC 1 P] from cA_J
            else:
                neo_in = patched_x.transpose(0, 2).contiguous().mean(2, keepdim=True)  # [N BC 1 P] (N = neo time axis)
            neo_in = self._neo_seq(neo_in)
            x_neo_emb = self.encoding_neo(neo_in) #[T B N D]
            x_neo = self.time_block(x_neo_emb, org_seq) #[T BC 1 D]
            self._neo_emb = x_neo_emb                                   # #2 predcode target (next-patch embedding)
            if self.neo_membrane_out and self.neo_recurrent:           # #1: use continuous membrane (holds smooth trend) as neo output
                mv = self.time_block.Mem[-1].v_seq
                if mv is not None:
                    x_neo = mv
        if self.neo_xattn:
            # TRUE cross-attention mode. The Block reads the Neocortex as key/value, so the
            # Neocortex must exist BEFORE the Block -- but its input is defined as the Block's
            # OUTPUT. That circularity is resolved inside the forward, not across epochs: pass 0
            # bootstraps the Neocortex from the patch embedding, and every later pass rebuilds it
            # from the previous pass's Block output. Training and inference run the identical
            # number of passes, so there is no train/test protocol mismatch.
            # The query input stays `x_emb` in every pass, so this refines the key/value rather
            # than stacking depth -- model depth is unchanged against the baseline.
            _core = self.reservoir if self.reservoir is not None else self.time_block
            x_emb = x_hippo
            _nin = x_emb.transpose(0, 2).contiguous().mean(2, keepdim=True)     # bootstrap: [N,BC,1,D]
            for _p in range(self.refine_steps + 1):
                if _p > 0:
                    # the Block and the readout LIF carry membrane state across calls; without this
                    # reset pass p would start from pass p-1's membranes and the passes would not be
                    # the same function.
                    sj_functional.reset_net(self.data_block)
                    if self.reservoir is not None:
                        sj_functional.reset_net(self.reservoir)
                x_neo = _core(self._apply_neo_mask(_nin), org_seq)
                h = x_emb
                for dblk in self.data_block:
                    h = dblk(h, x_neo)                                          # q=Hippo, kv=Neo
                _nin = h.transpose(0, 2).contiguous().mean(2, keepdim=True)     # == `_nin` of condition B
            x_hippo = h
            # The SAME final representation that feeds the head must also feed the Neocortex. The
            # loop leaves `_nin` = pool(final Block output) unused, so the Neocortex that the aux
            # path and the diagnostics see would otherwise be built from the FIRST Block output.
            # One more core call closes that gap. It is outside the prediction path -- the forecast
            # is already fixed by `h` -- so this changes what we can measure, not what we predict.
            if self.reservoir is not None:
                sj_functional.reset_net(self.reservoir)
            x_neo = _core(self._apply_neo_mask(_nin), org_seq)
            x_neo = self._neo_post(x_neo)
            return self._finish(x_hippo, x_neo, rolled_hippo)

        if not self.neo_recall:
            x_neo = self._neo_post(x_neo)          # condition B calls this from the recall block

        for dblk in ([] if getattr(self, '_dual', False) else self.data_block):
            # neo_recall: Hippo runs SELF-attention + MLP first (Neo has not spoken yet).
            x_hippo = dblk(x_hippo, x_hippo if self.neo_recall else x_neo)

        if self.serial_mode in ('fold', 'skip'):
            # SINGLE PATH. The Block's final representation is folded over the spiking axis and goes
            # to the head; under 'skip' it first passes through the reservoir, whose slow component
            # is added back through a zero-init gate. There is no auxiliary loss and no second
            # pathway -- the task loss is the only training signal, so the frozen liquid plus a
            # task-trained readout is exactly the reservoir-computing setup.
            # The skip matters: the liquid sits at DCT low-frequency 0.98, so routing EVERYTHING
            # through it would delete the high-frequency detail that the Max mixer exists to build.
            hbar = x_hippo.transpose(0, 2).contiguous().mean(2, keepdim=True)      # [N, BC, 1, D]
            if self.serial_mode == 'skip':
                x_neo = self._neo_post(self.reservoir(hbar, org_seq))
                fused = hbar + torch.tanh(self.serial_gamma) * x_neo
            else:
                x_neo, fused = None, hbar                                          # CONTROL: no reservoir
            z = self.head(fused.transpose(0, 2).contiguous().reshape(1, self.B * self.M, -1))
            z = rearrange(z, 't (b c) l -> t b l c', b=self.B)
            z = z * self.stdev + self.means.repeat(1, 1, 1, 1)
            return z.expand(self.T, -1, -1, -1)          # keep the [T, B, pred_len, C] interface

        if self.neo_recall and not self._neo_path_off:
            _h = x_hippo                                              # pre-prediction representation
            # Hippo -> Neo : STOP-GRADIENT. Neo is trained only by its reconstruction loss.
            # --neo_recall_grad selects which of the two detaches stay. 'stop' (default) keeps both,
            # i.e. the original fully-insulated design.
            _cut_h2n = self.neo_recall_grad in ('stop', 'to_neo')       # detach Hippo before Neo reads it
            _cut_n2h = self.neo_recall_grad in ('stop', 'to_hippo')     # detach Neo before Hippo reads it
            _hin = _h.detach() if _cut_h2n else _h
            # transpose(0,2) puts the PATCH axis first (Neo's LIF integrates over patches) and the
            # spiking-timestep axis into the token slot, which is then averaged away. Keeping those
            # T tokens instead was tried (--neo_recall_tokens full) and was exactly null, so the
            # original pooling is the wiring.
            _nin = _hin.transpose(0, 2).contiguous().mean(2, keepdim=True)          # [N,BC,1,D] (T=N)
            _nin = self._apply_neo_mask(_nin)                          # exp A: masked reconstruction
            _core = self.reservoir if self.reservoir is not None else self.time_block
            x_neo = _core(_nin, org_seq)                               # LIF aligned with the patch axis
            x_neo = self._neo_post(x_neo)
            # Neo -> Hippo gate. alpha starts at ~1 so Neo begins with no influence
            # and can only be opted into; the learned alpha is a direct readout of Neo's usefulness.
            _gsrc = x_neo.detach() if _cut_n2h else x_neo
            _g = _gsrc.transpose(0, 2).contiguous()                    # [1,BC,N,D], broadcast over T
            if self.neo_recall_gate == 'none':
                x_hippo = _h          # CONTROL: Neo still trains on reconstruction but is NOT read
                                      # back -> isolates the gate from the self-attention change
            elif self.neo_recall_gate == 'mul':
                x_hippo = _h * _g
            else:
                _a = torch.sigmoid(self.recall_alpha)
                x_hippo = _a * _h + (1.0 - _a) * _g
            # x_hippo = dblk(x_hippo, self.memory_slot(x_neo))

        return self._finish(x_hippo, x_neo, rolled_hippo)

    def _finish(self, x_hippo, x_neo, rolled_hippo):
        """Shared tail of forward: realign, then dispatch to the train or test head."""
        if rolled_hippo and self.hippo_unroll == 'post':
            x_hippo = self._hippo_unroll(x_hippo)               # fuse in rolled frame, realign before head (Exp2)

        if self.train_mode == 'training':

            return self._training(x_hippo, x_neo)

        elif self.train_mode == 'testing':

            # Stage-1 default keeps the Teacher (Hippocampus) at inference.
            # eval_student=True routes to the Student-only (Neocortex) path.
            if getattr(self, 'eval_student', False) and self.use_kd:
                return self._testing_student(x_neo)
            return self._testing(x_hippo, x_neo)



    def _apply_neo_mask(self, nin):
        """exp A -- masked reconstruction. Zero out a random subset of the Neocortex's input
        patches; train.py then scores the reconstruction on those held-out patches ONLY. Without
        this the aux is an autoencoding whose answer is already in the input, which is the leading
        explanation for why the Neocortex learns nothing Hippo does not already hold.

        Masking is train-only, so the gate sees an unmasked Neocortex at test time. The scale shift
        that implies is absorbed by the BatchNorm in front of the Neocortex LIF, so no inverted-
        dropout rescale is applied."""
        self.aux_mask = None
        if not (self.neo_mask_ratio > 0 and self.training and self.train_mode == 'training'):
            return nin
        Np, BC = nin.shape[0], nin.shape[1]
        m = (torch.rand(Np, BC, 1, 1, device=nin.device) < self.neo_mask_ratio).to(nin.dtype)
        self.aux_mask = m                                    # 1 = held out -> scored
        return nin * (1.0 - m)

    def _neo_post(self, x_neo):
        """Diagnostics and optional taps that must see the FINAL x_neo. Condition A finalises the
        Neocortex before the Block loop; condition B finalises it in the recall block, so this is
        called from whichever point is last in the active wiring."""
        if x_neo is None or self.time_block is None:
            return x_neo                     # reservoir backend: membrane/probe taps do not apply
        if getattr(self, 'probe_neo', False):
            # mechanism probe (R2): is x_neo a smoother/low-freq trend signal? (not memory)
            xn = x_neo.detach().float().squeeze(2)                     # [T, BC, D]
            T_ = xn.shape[0]; fr = xn.mean().item()
            a = xn[:-1].reshape(-1); b = xn[1:].reshape(-1)
            ac = (((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-8)).item()
            C = dct_matrix(T_, device=xn.device, dtype=xn.dtype)
            Xc = torch.einsum('kt,tbd->kbd', C, xn); p = (Xc ** 2).mean(dim=(1, 2))
            lf = (p[:max(1, T_ // 4)].sum() / (p.sum() + 1e-8)).item()
            # causal confirm: does the recurrence make x_neo reconstruct the low-freq trend target better?
            aux = float('nan')
            if hasattr(self, 'weak_decoder') and (self.org_x is not None):
                try:
                    rec = self.weak_decoder(x_neo).mean(2, keepdim=True)
                    aux = ((rec - self.org_x) ** 2).mean().item()
                except Exception:
                    pass
            # membrane-storage probe: is the slow trend held in the LIF membrane potential V?
            mlf = float('nan'); m_ac = float('nan')
            try:
                v = self.time_block.Mem[-1].v_seq            # [T,B,N,D] continuous membrane
                if v is not None:
                    vv = v.detach().float().reshape(v.shape[0], -1, v.shape[-1])  # [T,*,D]
                    Cv = dct_matrix(vv.shape[0], device=vv.device, dtype=vv.dtype)
                    Vc = torch.einsum('kt,tbd->kbd', Cv, vv); pv = (Vc ** 2).mean(dim=(1, 2))
                    mlf = (pv[:max(1, vv.shape[0] // 4)].sum() / (pv.sum() + 1e-8)).item()
                    va = vv[:-1].reshape(-1); vb = vv[1:].reshape(-1)
                    m_ac = (((va - va.mean()) * (vb - vb.mean())).mean() / (va.std() * vb.std() + 1e-8)).item()
            except Exception:
                pass
            print(f"[probe_neo] wrec={self.neo_wrec_mode} fr={fr:.4f} spk_lowfreq={lf:.4f} spk_autocorr={ac:.4f} "
                  f"MEM_lowfreq={mlf:.4f} MEM_autocorr={m_ac:.4f} aux_recon_mse={aux:.6f}")
        if (not self.no_neo) and self.aux_membrane and self.train_mode == 'training':
            self._neo_mem = (self.time_block.Mem[-1].v_seq if (self.neo_exact or self.neo_recurrent)
                             else self.time_block.Mem[-1].spk_neuron.v_seq)  # [T BC 1 D] continuous membrane

        if self.use_neo_bank:
            # update bank only during real training, then inject as residual prior
            if self.training and (self.train_mode in ('training', 'pre_training')):
                self.neo_bank.update(x_neo)
            x_neo = self.neo_bank(x_neo) #[T BC 1 D] - same shape
        return x_neo

    def _trend_residual_norm(self, x_neo, T_out=None):
        """review 8.7: Neocortex -> trend component of the forecast, in NORMALIZED space
        (denorm happens once, jointly with the detail forecast). x_neo: [T_neo(=N_p), BC, 1, D].

        The two pathways do NOT share a first axis in wavelet mode: the Hippocampus axis is
        SCALE (J bands) while the Neocortex axis is patch-time (N_p). Pool the patch-time axis
        into a single trend forecast and broadcast it over scales, so the approximation band
        contributes ADDITIVELY (x = cA_J + sum_j d_j) instead of only gating."""
        xj = x_neo.squeeze(2)                                    # [T_neo, BC, D]
        z = self.trend_head(xj)                                  # [T_neo, BC, pred_len]
        z = rearrange(z, 't (b c) l -> t b l c', b=self.B)       # [T_neo, B, pred_len, M]
        if (T_out is not None) and (z.shape[0] != T_out):
            z = z.mean(0, keepdim=True).expand(T_out, -1, -1, -1)
        return z

    def _student_forecast(self, x_neo):
        # x_neo: [N(=T), BC, 1, D]. Returns denormalized z_neo: [T, B, pred_len, M].
        if self.kd_head_mode == "joint":
            # Use all N neocortical tokens jointly (mirrors the Teacher head).
            xj = x_neo.squeeze(2).permute(1, 0, 2).contiguous()      # [BC, N, D]
            xj = xj.reshape(self.B * self.M, -1)                     # [BC, N*D]
            z = self.neo_head(xj)                                    # [BC, pred_len]
            z = rearrange(z, '(b c) l -> b l c', b=self.B)           # [B, pred_len, M]
            z = z * self.stdev + self.means                         # denorm (broadcast [B,1,M])
            z = z.unsqueeze(0).repeat(x_neo.shape[0], 1, 1, 1)      # [T_neo, B, pred_len, M]
            return z
        # pooled (default): per-token forecast, later averaged over T at eval
        _Tn = x_neo.shape[0]                                     # Neocortex axis (N_p), NOT self.T
        z = self.neo_head(x_neo.reshape(_Tn, self.B * self.M, -1)) #[T_neo, BC, pred_len]
        z = rearrange(z, 't (b c) l -> t b l c', b=self.B)
        z = z * self.stdev
        z = z + self.means.repeat(_Tn, 1, 1, 1)
        return z #[T, B, pred_len, M]

    def _hippo_roll(self, base):
        # base [BC, N, P] -> [T, BC, N, P] via per-step cyclic shift schedule (hippo_shift).
        T = self.T
        outs = []
        for t in range(T):
            s = int(self.hippo_shift[t])
            r = torch.roll(base, shifts=s, dims=1)
            if s > 0 and self.hippo_boundary != 'wrap':
                r = r.clone()
                if self.hippo_boundary == 'zero':
                    r[:, :s, :] = 0.0
                elif self.hippo_boundary == 'edge':
                    r[:, :s, :] = base[:, 0:1, :]
            outs.append(r)
        return torch.stack(outs, dim=0).contiguous()

    def _hippo_unroll(self, emb):
        # emb [T, BC, N, D] -> realign column n == patch n (undo per-step shift, deterministic).
        T = emb.shape[0]
        outs = [torch.roll(emb[t], shifts=-int(self.hippo_shift[t]), dims=1) for t in range(T)]
        return torch.stack(outs, dim=0)

    def _neo_seq(self, t):
        # compose the Neocortex input sequence along its time axis N (dim0). t: [N, BC, 1, P]
        if self.neo_seq == 'identity':
            return t
        if self.neo_seq in ('delta', 'shift'):
            nxt = torch.cat([t[1:], t[-1:]], dim=0)         # patch_{n+1} (last replicated)
            return (nxt - t) if self.neo_seq == 'delta' else nxt
        if self.neo_seq == 'sine':                          # cosine/sine-based causal selection of past tokens (APE-like)
            return t.index_select(0, self.neo_sine_idx.to(t.device))
        return t

    def _training(self, x_hippo, x_neo):
        if self.no_aux:
            z = self.head(x_hippo.reshape(self.T, self.B * self.M, -1))
            z = rearrange(z, 't (b c) l -> t b l c', b=self.B)
            if self.neo_out_residual and (x_neo is not None):     # prediction-level additive trend
                z = z + torch.tanh(self.out_beta) * self._trend_residual_norm(x_neo, T_out=z.shape[0])
            return z * self.stdev + self.means.repeat(self.T, 1, 1, 1)
        if self.no_neo:                                               # ablation: no reconstruction aux (rec=0)
            self.org_x = rearrange(self.org_x, 't (b c) l p -> b c l t p', b=self.B, c=self.M)
            z = self.head(x_hippo.reshape(self.T, self.B * self.M, -1))
            z = rearrange(z, 't (b c) l -> t b l c', b=self.B)
            z = z * self.stdev + self.means.repeat(self.T, 1, 1, 1)
            x_data = x_hippo.clone().transpose(0, 2).contiguous().mean(2, keepdim=True)
            return z, x_data, x_data, self.org_x, self.org_x
        if self.neo_predcode:                                          # #2: latent predictive coding (neo[t] -> next-patch embedding[t+1])
            pred = self.predcode_head(x_neo)                           # [T, BC, 1, D]
            tgt = self._neo_emb.detach()                              # [T, BC, 1, D]
            rec_x = pred[:-1]; org_tgt = tgt[1:]                       # shift: predict next
            z = self.head(x_hippo.reshape(self.T, self.B * self.M, -1))
            z = rearrange(z, 't (b c) l -> t b l c', b=self.B)
            z = z * self.stdev + self.means.repeat(self.T, 1, 1, 1)
            x_data = x_hippo.clone().transpose(0, 2).contiguous().mean(2, keepdim=True)
            return z, x_data, x_neo, org_tgt, rec_x
        if getattr(self, '_dual', False):
            # Both streams are [T,BC,N,D]. Collapse the scale axis and lift tokens to dim 0 so
            # BOTH the reconstruction target (per-patch) and the distill term are well defined.
            _p = lambda v: v.mean(0).permute(1, 0, 2).unsqueeze(2).contiguous()   # -> [N,BC,1,D]
            z = self.head_dual(x_hippo.reshape(x_hippo.shape[0], self.B * self.M, -1))
            z = rearrange(z, 't (b c) l -> t b l c', b=self.B)
            z = z * self.stdev + self.means.repeat(z.shape[0], 1, 1, 1)
            self.org_x = rearrange(self.org_x, 't (b c) l p -> b c l t p', b=self.B, c=self.M)
            _xn = _p(x_neo)
            rec_x = self.weak_decoder(_xn, x_hippo)
            rec_x = rearrange(rec_x.mean(2, keepdim=True), 't (b c) l p -> b c l t p', b=self.B, c=self.M)
            return z, _p(x_hippo), _xn, self.org_x, rec_x
        neo_for_rec = self._neo_mem if self.aux_membrane else x_neo    # low-pass aux from membrane (continuous) or spikes
        if neo_for_rec is not None and neo_for_rec.shape[2] > 1:
            # two-stream: x_neo is [T=J, BC, N, D] -- the scale axis is T and the token axis is
            # alive, so the T=N convention the aux target relies on no longer holds. Pool the
            # scale axis and lift tokens back to dim 0 -> [N, BC, 1, D], matching org_x (per-patch).
            neo_for_rec = neo_for_rec.mean(0).permute(1, 0, 2).unsqueeze(2).contiguous()
        rec_x = self.weak_decoder(neo_for_rec, x_hippo) #[T B 1 P]]
        rec_x = rec_x.mean(2, keepdim=True)
        rec_x = rearrange(rec_x, 't (b c) l p -> b c l t p', b=self.B, c=self.M)
        self.org_x  = rearrange(self.org_x, 't (b c) l p -> b c l t p', b=self.B, c=self.M)

        if self.aux_mask is not None:                                  # exp A: score held-out patches only
            self.aux_mask = rearrange(self.aux_mask, 't (b c) l p -> b c l t p', b=self.B, c=self.M)

        if self.neo_aux_next:
            # NEXT-token (1-step) prediction: neo at patch n predicts the patch at n+1.
            rec_x = rec_x[:, :, :, :-1, :]
            self.org_x = self.org_x[:, :, :, 1:, :]
            if self.aux_mask is not None:
                self.aux_mask = self.aux_mask[:, :, :, :-1, :]

        assert self.org_x.shape == rec_x.shape, f"original x_time' shape is {self.org_x.shape}, and reconstructed x_time' shape is {rec_x.shape}"

        z = self.head(x_hippo.reshape(self.T, self.B * self.M, -1))
        z = rearrange(z, 't (b c) l -> t b l c', b=self.B)

        if self.neo_out_residual and (x_neo is not None):        # review 8.7: + trend from Neocortex
            z = z + torch.tanh(self.out_beta) * self._trend_residual_norm(x_neo, T_out=z.shape[0])

        z = z * self.stdev
        z = z + self.means.repeat(self.T, 1, 1, 1)

        x_data = x_hippo.clone().transpose(0, 2).contiguous().mean(2, keepdim=True)
        if self.use_kd:
            # Teacher z (from Hippocampus), Student z_neo (from Neocortex).
            z_neo = self._student_forecast(x_neo)  #[T, B, pred_len, M]
            return z, x_data, x_neo, self.org_x, rec_x, z_neo
        return z, x_data, x_neo, self.org_x, rec_x
        # return z

    def _testing(self, x, x_neo=None):

        if getattr(self, '_dual', False):
            z = self.head_dual(x.reshape(x.shape[0], self.B * self.M, -1))
            z = rearrange(z, 't (b c) l -> t b l c', b=self.B).contiguous()
            return z * self.stdev + self.means.repeat(z.shape[0], 1, 1, 1)
        z = self.head(x.reshape(self.T, self.B * self.M, -1))

        z = rearrange(z, 't (b c) l -> t b l c', b=self.B).contiguous()

        if self.neo_out_residual and (x_neo is not None):        # review 8.7: same trend residual at inference
            z = z + torch.tanh(self.out_beta) * self._trend_residual_norm(x_neo, T_out=z.shape[0])

        z = z * self.stdev
        z = z + self.means.repeat(self.T, 1, 1, 1)
        return z

    def _testing_student(self, x_neo):
        # Student-only inference path (Neocortex). Same denorm as the teacher.
        return self._student_forecast(x_neo).contiguous()
    

    # TODO
    def _init_ablation(self):
        
        if hasattr(self, 'time_block') : delattr(self, 'time_block')
        if hasattr(self, 'encoding_neo') : delattr(self, 'encoding_neo')
        # delattr(self, 'gate_attn')
        if hasattr(self, 'weak_decoder'): delattr(self, 'weak_decoder')
        if hasattr(self, 'replay') : delattr(self, 'replay')
        
    def init_testing(self):
        
        if hasattr(self, 'weak_decoder'): delattr(self, 'weak_decoder')
        if hasattr(self, 'replay') : delattr(self, 'replay')

@register_model
def mymodel(pretrained=False, **kwargs):
    model = myModel(pretrained=pretrained, **kwargs)
    model.default_cfg = _cfg()
    return model
