import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate

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
from positional import tAPE
from layers import SSA_rel_scl, MLP, MutualCrossAttention, SpikLinearLayer, SpikLinearMaxLayer, SSA_max

from ours import HighFreqAmp, Embedding

__all__ = ['myModelAB1']

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
    def __init__(self, embed_dim, d_out=None, d_ff=2, tau=2.0, bias=True) -> None:
        super().__init__()
        
        # self.replay = Consolidation(dim=embed_dim, out_seq=True, lif_bias=bias, tau=tau)

        d_ff = embed_dim
        
        d_out = d_out or embed_dim
        
        self.recon2 = nn.Linear(d_ff, d_out, bias=bias)
        # self.recon2 = SpikLinearLayer(d_ff, d_out, lif_bias=bias)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, tr_mx=None):
        """
        [original] x: N x L x C(embed_dim)
        [MyModel] x: T x B x D
        
        out: reconstructed output -> N x L x c_out
        if expand is True: out's shape becomes [B X L]
        """
        
        T, B, N, D = x.shape

        # tr_mx = self.replay(x, tr_mx)
        # x = x * (1. - tr_mx)
        x = self.recon2(x.flatten(0, 1))
        x = self.dropout(x)
        # x = self.recon2(x)
        # x = torch.where(x>0, torch.ones_like(x), torch.ones_like(x) * -1.)
        rec_x = x.reshape(T, B, N, -1).contiguous()
        
        return rec_x
    
class Block(nn.Module):
    def __init__(self, T, dim, seq_len, num_heads, mlp_ratio=2.,max_ratio=2, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.1,
                drop_path=0.1, norm_layer=nn.LayerNorm, sr_ratio=1, lif_bias=False, tau=2.0, mutual=False,
                time_num_layers=1, patch_size=16):
        super().__init__()
        
        topk_ratio = None
        
        self.high_freq = HighFreqAmp(dim, ratio=max_ratio)
        # self.high_freq = SSA_rel_scl(dim=dim, seq_len=seq_len, num_heads=num_heads, pe=True, lif_bias=lif_bias, tau=tau, drop=attn_drop)
        # self.high_freq = SSA_max(dim=dim, seq_len=seq_len, num_heads=num_heads, pe=True, lif_bias=lif_bias, tau=tau, drop=attn_drop)
   
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_input_dim = dim
        self.mlp = MLP(in_features=mlp_input_dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop, lif_bias=lif_bias, tau=tau)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mx=None):
        T, B, N, D = x.shape
        
        x1 =  x * (1. - self.high_freq(x))
        x_out = x * (1. - self.mlp(x1))
        
        return x_out
  
class myModelAB1(nn.Module):
    def __init__(self, gating=['original', 'ablation'], 
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
                drop_rate=0., 
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
        
        # self.T = T  # time step
        self.spk_encoding = spk_encoding
        self.patch_size = patch_size
        self.stride = patch_size // 2
        
        self.num_patches = int((seq_len - patch_size) / (self.stride) + 1)
        self.T = self.num_patches
        
        print(f"len of tokens in sequence >> {self.num_patches}")

        self.spk_encoder = SpkEncoder(self.T) if spk_encoding else None
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.encoding = Embedding(num_patches=self.num_patches,
                                    patch_size=self.patch_size,
                                    embed_dim=embed_dim,
                                    stride=self.stride,
                                    pe=True,
                                    bias=bias,
                                    tau=tau
                                    )
        
        self.data_block = nn.ModuleList([Block(
                    T=self.T, dim=embed_dim, seq_len=self.num_patches, num_heads=num_heads, mlp_ratio=mlp_ratios, max_ratio=max_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                    norm_layer=norm_layer, sr_ratio=sr_ratios, lif_bias=bias, tau=tau, mutual=(self.gating == 'original'),
                    time_num_layers=time_num_layers, patch_size=self.patch_size)
                    for j in range(depths)])
        
        self.head = nn.Linear(embed_dim * self.num_patches, pred_len, bias=bias)

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
        
        self.B, L, self.M = x.shape
        org_seq = x.clone().detach() #[B L C]
        
        self.means = x.mean(1, keepdim=True).detach()
        x = x - self.means
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= self.stdev
        
        if self.spk_encoding:
            x = self.spk_encoder(x)
        else: 
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1) 
            
        x = rearrange(x, 't b l c -> t b c l')
        x = rearrange(x, 't b c l -> t (b c) l') # [T BC N P] 
        patched_x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) 
        self.org_x = patched_x.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True)
        
            
        if (self.gating == 'original') and (self.train_mode == 'training') and(self.keep_ratio > 0) : 
            keep_ratio=self.keep_ratio
            low_freq_x, _, _ = lowpass_memory(x, keep_ratio=keep_ratio, time_dim=2, center=True, rebinarize=False) #[T BC D]
            patched_low_freq_x = low_freq_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            patched_low_freq_x = patched_low_freq_x.transpose(0, 2).contiguous().mean(2, keepdim=True)
            self.org_x = patched_low_freq_x.clone().detach()
            
        x_hippo = self.encoding(patched_x)

        for dblk in self.data_block:
            # x_hippo = dblk(x_hippo, self.time_block(x_hippo.transpose(0, 2).contiguous().mean(2, keepdim=True).clone().detach()))
            x_hippo = dblk(x_hippo)
            # x_hippo = dblk(x_hippo, self.memory_slot(x_neo))
            

        z = self.head(x_hippo.reshape(self.T, self.B * self.M, -1))

        z = rearrange(z, 't (b c) l -> t b l c', b=self.B)

        z = z * self.stdev
        z = z + self.means.repeat(self.T, 1, 1, 1)
        return z
    

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
def mymodel_ab1(pretrained=False, **kwargs):
    model = myModelAB1(pretrained=pretrained, **kwargs)
    model.default_cfg = _cfg()
    return model
