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
# from utils import random_masking_3D, vis
from positional import tAPE
from layers import SSA_rel_scl, MLP, MutualCrossAttention, SpikLinearLayer, SpikLinearMaxLayer

__all__ = ['Spikformer']
    
class Embedding(nn.Module):
    def __init__(self, num_patches, pe=False, patch_size=63, stride=2, embed_dim=128, dropout=0, bias=False, tau=2.0) -> None:
        super().__init__()
        
        self.num_patches = num_patches
        # self.patch_size = patch_size
        # in_channel = patch_size
        
        self.stride = stride
        self.embed_dim = embed_dim
        self.pe = pe
        
        self.emb_linear = nn.Linear(patch_size, embed_dim, bias=bias)
        self.emb_bn = nn.BatchNorm1d(embed_dim)
        self.emb_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        # positional encoding
        if pe:
            self.tape = tAPE(embed_dim, max_len=num_patches)
            self.ape_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        
        # Residual dropout
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor, pe=True): 
        T, B, _,_ = x.shape

        x = self.emb_linear(x.flatten(0, 1))
        x = self.emb_bn(x.transpose(-1, -2).contiguous())
        x = x.reshape(T, B, self.embed_dim, -1).contiguous() # [T B N D]
        x = x.flatten(3).contiguous() # [T B D N] 
        self.org_x = x.clone().detach().transpose(-1, -2).contiguous() 
        # x_neo = self.emb_lif(x.reshape(T, B, self.embed_dim, -1).contiguous())
        # x_neo = x_neo.transpose(-1, -2).contiguous()  # [T B N D] 
        
        x = self.tape(x)
        x = self.emb_lif(x.reshape(T, B, self.embed_dim, -1).contiguous())
        x = x.transpose(-1, -2).contiguous()  # [T B N D] 
        
        return x

    
class Block(nn.Module):
    def __init__(self, T, dim, seq_len, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0.1, norm_layer=nn.LayerNorm, sr_ratio=1, lif_bias=False, tau=2.0, 
                patch_size=16):
        super().__init__()
        
        topk_ratio = None
        self.attn = SSA_rel_scl(dim, seq_len, num_heads, lif_bias=lif_bias, tau=tau, topk_ratio=topk_ratio, drop=drop)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop, lif_bias=lif_bias, tau=tau)
        
    def forward(self, x: torch.Tensor):
        
        x = x + self.attn(x)
        x = x + self.mlp(x)
        
        return x


class Spikformer(nn.Module):
    def __init__(self, 
                patch_size=63, 
                pred_len=0, 
                seq_len=0,
                embed_dim=[64, 128, 256], 
                num_heads=[1, 2, 4], 
                mlp_ratios=1, 
                qkv_bias=False, 
                qk_scale=None,
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
        
        self.pred_len = pred_len
        
        # self.T = T  # time step
        self.spk_encoding = spk_encoding
        self.patch_size = patch_size
        self.stride = patch_size // 2
        
        self.num_patches = int((seq_len - patch_size) / (self.stride) + 1)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.num_patches = self.num_patches + 1
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
                    T=self.T, dim=embed_dim, seq_len=self.num_patches, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                    norm_layer=norm_layer, sr_ratio=sr_ratios, lif_bias=bias, tau=tau,
                    patch_size=self.patch_size)
                    for j in range(depths)])
        
        
        self.head = nn.Linear(embed_dim * self.num_patches, pred_len, bias=bias)
        self.apply(self._init_weights)

    
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
        x = self.padding_patch_layer(x)
        patched_x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) 
        
        x = self.encoding(patched_x)

        for dblk in self.data_block:
            x = dblk(x)

        z = self.head(x.reshape(self.T, self.B * self.M, -1))

        z = rearrange(z, 't (b c) l -> t b l c', b=self.B)

        z = z * self.stdev
        z = z + self.means.repeat(self.T, 1, 1, 1)
        return z
      
      
      
# @register_model
# def spikformer(pretrained=False, **kwargs):
#     model = Spikformer(pretrained=pretrained, **kwargs)
#     model.default_cfg = _cfg()
#     return model
