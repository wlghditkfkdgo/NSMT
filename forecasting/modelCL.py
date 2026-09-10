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

from layers import SpkEncoder, Consolidation, SpikLinearLayer
from utils import random_masking_3D, vis
from positional import tAPE
from layers import SSA_rel_scl, MLP, MutualCrossAttention, SpikLinearLayer, SpikLinearMaxLayer

__all__ = ['spikformer']

def walsh_ordering(H):
    # 각 행의 부호 변화 개수를 계산
    sign_changes = np.sum(np.abs(np.diff(H < 0, axis=1)), axis=1)
    # 부호 변화 수에 따라 정렬
    order = np.argsort(sign_changes)
    return H[order]

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
        # x = self.recon2(x)
        # x = torch.where(x>0, torch.ones_like(x), torch.ones_like(x) * -1.)
        rec_x = x.reshape(T, B, N, -1).contiguous()
        
        return rec_x
    
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
        x = self.emb_bn(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        x = x.reshape(T, B, self.embed_dim, -1).contiguous() # [T B N D]
        x = x.flatten(3).contiguous() # [T B D N] 
        self.org_x = x.clone().detach().transpose(-1, -2).contiguous() 
        # x_neo = self.emb_lif(x.reshape(T, B, self.embed_dim, -1).contiguous())
        # x_neo = x_neo.transpose(-1, -2).contiguous()  # [T B N D] 
        
        x_hippo = self.tape(x)
        x_hippo = self.emb_lif(x_hippo.reshape(T, B, self.embed_dim, -1).contiguous())
        x_hippo = x_hippo.transpose(-1, -2).contiguous()  # [T B N D] 
        
        return x_hippo

    
class Block(nn.Module):
    def __init__(self, T, dim, seq_len, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0.1, norm_layer=nn.LayerNorm, sr_ratio=1, lif_bias=False, tau=2.0, mutual=False,
                time_num_layers=1, patch_size=16):
        super().__init__()
        
        self.attn = SSA_rel_scl(dim, seq_len, num_heads, lif_bias=lif_bias, tau=tau, drop=drop) 
        if mutual: self.mca = MutualCrossAttention(dim=dim, lif_bias=lif_bias, num_heads=num_heads, tau=tau, drop=drop)
        # self.gate = SpikLinearLayer(in_dim=dim * 2, out_dim=dim, spk='lif', tau=tau, lif_bias=lif_bias)
        # self.gate = nn.Linear(dim * 2, 2, bias=lif_bias)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_input_dim = dim * 2 if mutual else dim 
        self.mlp = MLP(in_features=mlp_input_dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop, lif_bias=lif_bias, tau=tau)
        
        # self.dropout1 = nn.Dropout(drop_path)
        self.dropout2 = nn.Dropout(drop_path)
        self.dropout3 = nn.Dropout(drop_path)
        
    def forward(self, x: torch.Tensor, mx=None):
        T, _, _, _ = x.shape
        
        attn_x = x * (1. - self.attn(x))
        if mx is not None: 
            # x_mca = x * (1. - self.mca(x, mx)) 
            mca_x = mx * (1. - self.mca(x, mx))
            if mca_x.sum() == 0 :  print("Warning!")
            cat_x = torch.cat([attn_x, mca_x], dim=-1)
            x = x * (1. - self.mlp(cat_x))
            # cat_x = torch.cat([attn_x, mca_x], dim=-1)
        else:
            x = attn_x * (1. - self.mlp(attn_x))
        
        return x
  
class TemporalBlock(nn.Module):
    def __init__(self, T, num_layers, patch_size=[16, 32], embed_dim=[64, 128, 256], ratio=8, lif_bias=False, tau=2.0, num_heads=8, mutual=False):
        super().__init__()
        
        # self.T = T  # time step

        self.patch_size = patch_size
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList(
            # [RecurrentSpikLinearLayer(T, embed_dim, embed_dim, bias=lif_bias, tau=tau, spk='custom')
            [SpikLinearLayer(embed_dim, embed_dim, lif_bias=lif_bias, tau=tau, spk='lif')
            for l in range(num_layers)])
        self.dropout = nn.ModuleList(
            [nn.Dropout(0.1)
             for l in range(num_layers)]
        )

    def visualize_embedding(self, x, original_data=None, channel_idx=0):
        # x: [T, B, 1, D]

        # Ensure x has shape [T, B, 1, D] before squeezing
        if x.dim() == 4 and x.shape[2] == 1:
            x_mean = x.mean(dim=-1).squeeze(2)  # [T, B]
        else:
            x_mean = x.mean(dim=-1)  # fallback if shape is [T, B, D]

        batch_idx = 0  # 첫 번째 배치만 시각화

        plt.figure(figsize=(10, 5))
        plt.plot(range(x_mean.shape[0]), 
                x_mean[:, batch_idx].cpu().detach().numpy(), 
                label='Embedding Mean', linewidth=2)

        # === 추가: positional embedding 시각화 ===
        # pos_np = self.pos.detach().cpu().numpy()
        # # pos_scaled = (pos_np - pos_np.min()) / (pos_np.max() - pos_np.min() + 1e-8)
        # plt.plot(range(len(pos_scaled)), pos_scaled, 
        #         label='Positional Embedding', linestyle='--', color='orange', alpha=0.8)

        if original_data is not None:
            # original_data: [B, L, C]
            if original_data.dim() == 3:
                orig = original_data[batch_idx, :].mean(-1)  # [L]
            elif original_data.dim() == 2:
                orig = original_data[batch_idx, :]  # [L]
            else:
                orig = original_data.flatten()

            orig_min = orig.min()
            orig_max = orig.max()
            if orig_max > orig_min:
                orig_scaled = (orig - orig_min) / (orig_max - orig_min)
            else:
                orig_scaled = orig - orig_min  # all zeros if no variation

            T = x_mean.shape[0]
            L = orig_scaled.shape[0]
            orig_resampled = np.interp(
                np.linspace(0, L - 1, T), np.arange(L), orig_scaled.cpu().detach().numpy()
            )
            plt.plot(range(T), orig_resampled, label=f'Original Data (channel {channel_idx})', alpha=0.7)

        plt.xlabel('Token Index (T)')
        plt.ylabel('Value (Normalized)')
        plt.title('Embedding, Positional Encoding, and Original Data (Aligned)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('embedding_visualization.png')
        plt.close()

    def forward(self, x, org_data=None):
        # x: [T, B, 1, D]

        for l in range(self.num_layers):
            x = self.layers[l](x)
            x = self.dropout[l](x)
        # self.visualize_embedding(x, org_data)
        
        return x
    

class SpikformerEncoder(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        
        self.train_mode = args.train_mode
        self.gating = args.gating
        self.pred_len = args.pred_len
        
        self.spk_encoding = args.spk_encoding
        self.patch_size = args.patch_size
        self.stride = self.patch_size // 2
        
        mlp_ratios = 4
        
        self.num_patches = int((args.seq_len - self.patch_size) / (self.stride) + 1)
        self.T = self.num_patches
        
        
        print(f"Num of tokens in sequence >> {self.num_patches}")
        
        self.spk_encoder = SpkEncoder(self.T) if self.spk_encoding else None
        
        dpr = [x.item() for x in torch.linspace(0, args.drop_path_rate, args.depths)]
        
        self.encoding = Embedding(num_patches=self.num_patches,
                                    patch_size=self.patch_size,
                                    embed_dim=args.embed_dim,
                                    stride=self.stride,
                                    pe=True,
                                    bias=args.bias,
                                    tau=args.tau
                                    )
        # self.encoding_neo = RecurrentSpikLinearLayer(self.T, patch_size, embed_dim, bias=bias, tau=tau, spk='custom')
        self.encoding_neo = SpikLinearLayer(self.patch_size, args.embed_dim, lif_bias=args.bias, tau=args.tau, spk='lif')

        self.data_block = nn.ModuleList([Block(
                    T=self.T, dim=args.embed_dim, seq_len=self.num_patches, num_heads=args.num_heads, mlp_ratio=mlp_ratios,
                    drop=0.1, drop_path=dpr[j],
                    lif_bias=args.bias, tau=args.tau, mutual=(self.gating == 'original'),
                    time_num_layers=args.time_num_layers, patch_size=self.patch_size)
                    for j in range(args.depths)])
        
        self.time_block = TemporalBlock(T=self.T,
                                        num_layers=args.time_num_layers,
                                        patch_size=self.patch_size,
                                        embed_dim=args.embed_dim,
                                        lif_bias=args.bias,
                                        num_heads=args.num_heads,
                                        tau=args.tau,
                                        mutual=False)
        
        
        self.apply(self._init_weights)
        
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
        
        keep_ratio=0.25
        low_freq_x, _, _ = lowpass_memory(x, keep_ratio=keep_ratio, time_dim=2, center=True, rebinarize=False) #[T BC D]
        patched_x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) 
        patched_low_freq_x = low_freq_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        patched_low_freq_x = patched_low_freq_x.transpose(0, 2).contiguous().mean(2, keepdim=True)
        self.org_x = patched_low_freq_x.clone()
        
        x_hippo = self.encoding(patched_x)
        x_neo = self.encoding_neo(patched_x.transpose(0, 2).contiguous().mean(2, keepdim=True)) if (self.gating != 'ablation') else None
        x_neo = self.time_block(x_neo) if (self.gating != 'ablation') else None
        
        
        for dblk in self.data_block:
            x_hippo = dblk(x_hippo, x_neo)
        
        return x_hippo, x_neo
    
    
class Spikformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.patch_size = args.patch_size
        self.encoder = SpikformerEncoder(args)
        self.rec = Decoder(embed_dim=args.embed_dim, d_out=self.patch_szie, tau=args.tau, bias=args.bias)
        self.head = nn.Linear(args.embed_dim * self.num_patches, self.pred_len, bias=args.bias)
        
    
    def forward(self, x):
        B, *_ = x.shape
        
        x_hippo, x_neo = self.encoder(x)
        
        rec_x_neo = self.rec(x)
        rec_x_neo = rearrange(rec_x_neo, 't (b c) l p -> t b l c p', b=B, l=1)
        org_x  = rearrange(self.encoder.org_x, 't (b c) l p -> t b l c p', b=B, l=1)
        
        assert org_x.shape == rec_x_neo.shape, f"original x_time' shape is {org_x.shape}, and reconstructed x_time' shape is {rec_x_neo.shape}"
        
        z = self.head(x_hippo.reshape(self.T, self.B * self.M, -1))
        z = rearrange(z, 't (b c) l -> t b l c', b=self.B)

        z = z * self.stdev
        z = z + self.means.repeat(self.T, 1, 1, 1)
        
        return z, rec_x_neo, org_x
        
    # def store_grad(self):
    #     for name, layer in self.encoder.named_modules():
    #         if 'time_block' in type(layer).__name__:
    #             layer.store_grad()