import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg 
from timm.models.layers import trunc_normal_
from timm.utils import *
from einops import rearrange

from layers import SpkEncoder, Consolidation, SpikLinearLayer, SpikLinearMaxLayer
from utils import random_masking_3D
from positional import tAPE
from layers import SSA_rel_scl, MLP, MutualCrossAttention, SpikLinearLayer
import numpy as np 

import math


__all__ = ['spikformer']

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
    def __init__(self, embed_dim, patch_size=None, d_ff=2, tau=2.0, bias=True) -> None:
        super().__init__()
        
        # self.replay = Consolidation(dim=embed_dim, out_seq=True, lif_bias=bias, tau=tau)

        d_ff = embed_dim
        
        patch_size = patch_size or embed_dim
        
        self.dropout = nn.Dropout(0.1)
        self.recon2 = nn.Linear(d_ff, patch_size, bias=bias)
        
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
        x = self.dropout(x)
        x = self.recon2(x.flatten(0, 1))
        rec_x = x.reshape(T, B, N, -1).contiguous()
        
        return rec_x
    
# class HighFreqAmp(nn.Module):
#     def __init__(self, dim, tau=2.0, bias=False):
#         super().__init__()
#         self.max1 = SpikLinearMaxLayer(dim, dim * 2, tau=tau, lif_bias=bias)
        
#         self.max3 = SpikLinearLayer(dim, dim, tau=tau, lif_bias=bias)
        
#         self.concat = SpikLinearLayer(dim * 2, dim, tau=tau, lif_bias=bias)
        
#     def forward(self, x):
#         return self.concat(torch.cat([self.max1(x), self.max3(x)], dim=-1))
        

class HighFreqAmp(nn.Module):
    def __init__(self, dim, ratio=2, tau=2.0, bias=False):
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
        self.emb_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
        # positional encoding
        if pe:
            self.tape = tAPE(embed_dim, max_len=num_patches)
            self.ape_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
        # Residual dropout
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor, pe=True): 
        T, B, _,_ = x.shape
        
        x = self.emb_linear(x.flatten(0, 1))
        x = self.emb_bn(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        x = x.reshape(T, B, self.embed_dim, -1).contiguous() # [T B N D]
        x = x.flatten(3).contiguous() # [T B D N] 
        
        # x_neo = self.emb_lif(x.reshape(T, B, self.embed_dim, -1).contiguous())
        # x_neo = x_neo.transpose(-1, -2).contiguous()  # [T B N D] 
        
        # x = self.tape(x)
        x = self.emb_lif(x.reshape(T, B, self.embed_dim, -1).contiguous())
        x = x.transpose(-1, -2).contiguous()  # [T B N D] 
        
        return x

    
class Block(nn.Module):
    def __init__(self, dim, seq_len, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0.1, norm_layer=nn.LayerNorm, sr_ratio=1, lif_bias=False, tau=2.0, mutual=False):
        super().__init__()
        
        topk_ratio= None
        # self.attn = SSA_rel_scl(dim, seq_len, num_heads, lif_bias=lif_bias, tau=tau, drop=drop, topk_ratio=topk_ratio) 
        self.attn = HighFreqAmp(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # if mutual: self.mca = MutualCrossAttention(dim=dim, lif_bias=lif_bias, num_heads=num_heads, tau=tau, drop=drop)
        self.time_rec = SpikLinearLayer(dim, dim)
        mlp_input_dim = dim
        # mlp_input_dim = dim
        self.mlp = MLP(in_features=mlp_input_dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop, lif_bias=lif_bias, tau=tau)
        
        # self.gate = nn.Linear(seq_len * 2, seq_len, bias=lif_bias)
        # self.gate_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
        # self.dropout = nn.Dropout(drop_path)

        self.dropout1 = nn.Dropout(drop_path)
        self.dropout2 = nn.Dropout(drop_path)

    def forward(self, x: torch.Tensor, mx=None):
        
        if mx is not None: 
            attn_x = x * (1. - self.attn(x))
            # mca_x = x * (1. - self.mca(x, mx)) 
            mca_x = mx.transpose(0, 2).contiguous() * (1. - self.time_rec(mx.transpose(0, 2).contiguous()))
            if mca_x.sum() == 0 :  print("Warning!")
            
            # cat_x = torch.cat([attn_x, mca_x], dim=-1)
            cat_x = attn_x * mca_x
            x = x * (1. - self.mlp(cat_x)) 
        else:
            x1 = x * (1. - self.attn(x))
            x = x * (1. - self.mlp(x1)) 
    
        return x
  
# class TemporalBlock(nn.Module):
#     def __init__(self, num_layers, T, patch_size=[16, 32], embed_dim=[64, 128, 256], ratio=8, lif_bias=False, tau=2.0, num_heads=8, mutual=False):
#         super().__init__()
        
#         self.T = T  # time step

#         self.patch_size = patch_size
#         self.num_layers = num_layers
        
#         self.in_layer = SpikLinearLayer(embed_dim, embed_dim * 2, lif_bias=lif_bias, tau=tau)
#         self.layers = nn.ModuleList([SpikLinearLayer(embed_dim * 2, embed_dim * 2, lif_bias=lif_bias, tau=tau) for l in range(num_layers)])
#         self.out_layer = SpikLinearLayer(embed_dim * 2, embed_dim, lif_bias=lif_bias, tau=tau)

#     def forward(self, x):
        
#         x = self.in_layer(x)
#         for l in range(self.num_layers):
#             x = x * (1. - self.layers[l](x))
#         x = self.out_layer(x)

#         return x

class TemporalBlock(nn.Module):
    def __init__(self, T, num_layers=2, patch_size=[16, 32], embed_dim=[64, 128, 256], ratio=2, lif_bias=False, tau=2.0, num_heads=8, mutual=False):
        super().__init__()
        
        self.patch_size = patch_size
        hidden_ratio = ratio
        self.num_layers = num_layers
        
        self.in_layer = SpikLinearLayer(embed_dim, embed_dim*hidden_ratio, lif_bias=lif_bias, tau=tau)
        self.Mem = nn.ModuleList(
            [SpikLinearLayer(embed_dim * hidden_ratio, embed_dim * hidden_ratio, lif_bias=lif_bias, tau=tau)
            for l in range(int(num_layers))])
        self.out_layer = SpikLinearLayer(embed_dim*hidden_ratio, embed_dim, lif_bias=lif_bias, tau=tau)

    def forward(self, x, org_data=None):
        
        
        x = self.in_layer(x)
        for l in range(self.num_layers):
            x = self.Mem[l](x)
        x = self.out_layer(x)
        return x, x


class Spikformer(nn.Module):
    def __init__(self, gating=['original', 'ablation'], 
                 train_mode=['pretraining', 'training', 'testing', 'visual'], 
                patch_size=63, 
                c_out=0, 
                seq_len=0,
                time_num_layers=2,
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
                features='M',
                spk_encoding=False,
                pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                **kwargs,
                ):
        super().__init__()
        
        self.train_mode = train_mode
        self.gating = gating
        self.c_out = c_out

        self.seq_len = seq_len  # Added to fix undefined attribute error
        
        # self.T = T  # time step
        self.spk_encoding = spk_encoding
        self.patch_size = patch_size
        self.stride = patch_size // 2
        self.features = features  # 'MS' or 'MC' or 'M'
        # hidden_dim = embed_dim * 4
        
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
        if gating == 'original':
            self.encoding_neo = SpikLinearLayer(patch_size, embed_dim, lif_bias=bias, tau=tau)

        self.data_block = nn.ModuleList([Block(
                    dim=embed_dim, seq_len=self.num_patches, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                    norm_layer=norm_layer, sr_ratio=sr_ratios, lif_bias=bias, tau=tau, mutual=(gating == 'original'))
                    for j in range(depths)])
        
        if gating == 'original':
            self.time_block = TemporalBlock(T=self.T,
                                        num_layers=time_num_layers,
                                        patch_size=patch_size,
                                        embed_dim=embed_dim,
                                        lif_bias=bias,
                                        num_heads=num_heads,
                                        tau=tau,
                                        mutual=False)
        
        if ((self.train_mode == 'training') or (self.train_mode == 'pre_training')) and (gating == 'original') : 
            self.weak_decoder = Decoder(embed_dim=embed_dim, patch_size=self.patch_size, tau=tau, bias=bias)
            
        self.head = nn.Linear(embed_dim * self.num_patches, seq_len, bias=bias)
        self.dropout = nn.Dropout(0.1)
        
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

    def forward(self, x):
        self.B, L, self.M = x.shape
        org_seq = x.clone().detach() #[B L C]
        
        self.means = x.mean(1, keepdim=True).detach()
        x = x - self.means
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= self.stdev
        
        if self.spk_encoding:
            x = self.spk_encoder(x)
        else: 
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1) 
            
        x = rearrange(x, 't b l c -> t b c l')
        x = rearrange(x, 't b c l -> t (b c) l') # [T BC N P] 
        
        patched_x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) 
        
        if self.gating == 'original' :
            keep_ratio=0.2
            low_freq_x, _, _ = lowpass_memory(x, keep_ratio=keep_ratio, time_dim=2, center=True, rebinarize=False) #[T BC D]
            patched_low_freq_x = low_freq_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            patched_low_freq_x = patched_low_freq_x.transpose(0, 2).contiguous().mean(2, keepdim=True)
            self.org_x = patched_low_freq_x.clone().detach()
            
        x_hippo = self.encoding(patched_x)

        if self.gating == 'original' : 
            x_neo_emb = self.encoding_neo(patched_x.transpose(0, 2).contiguous().mean(2, keepdim=True))
            x_neo, mem = self.time_block(x_neo_emb, org_seq)
        else:
            x_neo, mem = None, None
        
        for dblk in self.data_block:
            x_hippo = dblk(x_hippo, mem)
        
        if (self.gating == 'original') and (self.train_mode == 'training'):
            
            return self._training(x_hippo, x_neo_emb * (1. - x_neo)) 
            
        elif (self.train_mode == 'testing') or (self.gating == 'ablation'):

            return self._testing(x_hippo)

        
    def _training(self, x_hippo, x_neo):
        rec_x = self.weak_decoder(x_neo, x_hippo) #[T B 1 P]
        
        assert self.org_x.shape == rec_x.shape, f"original x_time' shape is {self.org_x.shape}, and reconstructed x_time' shape is {rec_x.shape}"

        z = x_hippo.reshape(self.T, -1, self.M, x_hippo.shape[-2], x_hippo.shape[-1]).contiguous()
        z = z.permute(0, 1, 2, 4, 3).contiguous().mean(0) # [B C D N]
        z = self.head(z.flatten(-2, -1))  # [B C DxN] -> [B C c_out]
        z = self.dropout(z)
        z = z.transpose(-1, -2).contiguous()  # [B c_out C]
        # z = rearrange(z, '(b c) l -> b l c', b=self.B)

        z = z * (self.stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        z = z + (self.means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
       
        return z, x_hippo.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True), x_neo, self.org_x, rec_x

        # return z
    
    def _testing(self, x):
        
        z = x.reshape(self.T, -1, self.M, x.shape[-2], x.shape[-1]).contiguous()
        z = z.permute(0, 1, 2, 4, 3).contiguous().mean(0) # [B C D N]
        z = self.head(z.flatten(-2, -1))  # [B C DxN] -> [B C c_out]
        # z = self.dropout(z)
        z = z.transpose(-1, -2).contiguous()  # [B c_out C]
        # z = rearrange(z, '(b c) l -> b l c', b=self.B)

        z = z * (self.stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        z = z + (self.means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return z
    

    # TODO
    def _init_ablation(self):
        
        if hasattr(self, 'time_block') : delattr(self, 'time_block')
        # delattr(self, 'gate_attn')
        if hasattr(self, 'encoding_neo') : delattr(self, 'encoding_neo')
        if hasattr(self, 'weak_decoder'): delattr(self, 'weak_decoder')
        if hasattr(self, 'replay') : delattr(self, 'replay')
    

@register_model
def spikformer(pretrained=False, **kwargs):
    model = Spikformer(pretrained=pretrained, **kwargs)
    model.default_cfg = _cfg()
    return model