import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg 
from timm.models.layers import trunc_normal_
# from timm.optim.optim_factory import create_optimizer_v2

# from timm.models import create_model
from timm.utils import *

# from encoder import Embedding, Block, TemporalBlock
from layers import SpkEncoder, SpikLinearLayer, SpikLinearMaxLayer, MLP, SSA_rel_scl, MutualCrossAttention
from positional import tAPE
from utils import random_masking_3D
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


class Embedding(nn.Module):
    def __init__(self, num_patches, pe=False, patch_size=63, stride=2, embed_dim=128, in_channel=3, dropout=0, bias=False, tau=2.0) -> None:
        super().__init__()
        
        self.num_patches = num_patches
        # self.patch_size = patch_size
        # in_channel = patch_size
        
        self.stride = stride
        self.embed_dim = embed_dim
        self.pe = pe
        
        self.emb_conv = nn.Conv1d(in_channel, embed_dim, kernel_size=patch_size, stride=stride, bias=bias)
        self.emb_bn = nn.BatchNorm1d(embed_dim)
        # self.emb_lif_neo = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        self.emb_lif_hippo = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        
        # positional encoding
        if pe:
            self.tape = tAPE(embed_dim, max_len=num_patches)
            self.ape_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor, pe=True): 
        T, B, _,_ = x.shape
        
        x = self.emb_conv(x.flatten(0, 1))
        x = self.emb_bn(x)
        x = x.reshape(T, B, self.embed_dim, -1).contiguous() # [T B N D]
        # self.org_x = x.clone().detach().transpose(-1, -2).transpose(0, 2)
        x = x.flatten(3).contiguous() # [T B D N] 
        
        # x_neo = self.emb_lif_neo(x.reshape(T, B, self.embed_dim, -1))
        # # x_neo = x_neo.transpose(0, 3).contiguous()
        # x_neo = x_neo.transpose(-1, -2).contiguous()  # [N B T D] 
        # # x_neo = self.emb_spkLinear_neo(x_neo)
        
        x_hippo = x
        if self.pe : x_hippo = self.tape(x_hippo)
        x_hippo = self.emb_lif_hippo(x_hippo.reshape(T, B, self.embed_dim, -1).contiguous())
        x_hippo = x_hippo.transpose(-1, -2).contiguous()  # [T B N D] 
        # x_hippo = self.emb_spkLinear_hippo(x_hippo)
        
        return x_hippo
    

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

    def forward(self, x: torch.Tensor, mx=None):
        T, B, N, D = x.shape

        x = x * (1. - self.mca(x, x))
        x = x * (1. - self.mlp(x))
        
        return x


class myModelAB2(nn.Module):
    def __init__(self, gating=['original', 'ablation'], 
                 train_mode=['pretraining', 'training', 'testing', 'visual'], 
                 seq_len=192,
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
        self.T = self.num_patches
        
        print(f"len of tokens in sequence >> {self.num_patches}")


        self.spk_encoder = SpkEncoder(T) if spk_encoding else None
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.encoding = Embedding(num_patches=self.num_patches,
                                    patch_size=self.patch_size,
                                    embed_dim=embed_dim,
                                    stride=self.stride,
                                    in_channel=num_channels,
                                    pe=False,
                                    bias=lif_bias,
                                    tau=tau)
     
        self.data_block = nn.ModuleList([Block(T=T,
                    dim=embed_dim, seq_len=self.num_patches, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                    norm_layer=norm_layer, sr_ratio=sr_ratios, lif_bias=lif_bias, tau=tau, attn=(gating=='original'))
                    for j in range(depths)])

     
        self.head = nn.Linear(embed_dim, num_classes, bias=lif_bias) if num_classes > 0 else nn.Identity()
        self.dropout = nn.Dropout(drop_rate)

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

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # [B, L, C]
        self.B, L, self.M = x.shape
        org_data = x.clone().detach()
        x = x.permute(0, 2, 1) #[B M L]

        
        if self.keep_ratio > 0:
            low_freq_x, _, _ = lowpass_memory(x, keep_ratio=self.keep_ratio, time_dim=2, center=True, rebinarize=False)
            
        if self.spk_encoding:
            x = self.spk_encoder(x)
        else: 
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1) 
            
        # x = rearrange(x, 't b l c -> t b c l')
        # patched_x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) 
        # patched_x = rearrange(patched_x, 't b c n p -> t (b c) n p') # [T BC N P] 
        # self.org_x = patched_x.clone().detach() # [T B N P]
        
        x = self.encoding(x) # [T B N D]
        
        for dblk in self.data_block:
            x = dblk(x)
        
        return self.head(x.mean(2)).mean(0)
        
    # TODO
    def _init_ablation(self):
        
        if hasattr(self, 'time_block'): delattr(self, 'time_block')
        if hasattr(self, 'weak_decoder'): delattr(self, 'weak_decoder')
        if hasattr(self, 'replay') : delattr(self, 'replay')
    

@register_model
def mymodel_ab2(pretrained=False, **kwargs):
    model = myModelAB2(pretrained=pretrained, **kwargs)
    model.default_cfg = _cfg()
    return model