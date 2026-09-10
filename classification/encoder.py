import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate

from positional import tAPE
from layers import SSA_rel_scl, MLP, MutualCrossAttention, SpikLinearLayer
import numpy as np

__all__ = ['spikformer']


class Embedding(nn.Module):
    def __init__(self, num_patches, pe=False, patch_size=63, stride=2, embed_dim=128, in_channel=3, dropout=0, bias=False, tau=2.0) -> None:
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        # in_channel = patch_size
        
        self.stride = stride
        self.embed_dim = embed_dim
        self.pe = pe
        
        # self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.emb_conv = nn.Conv1d(in_channel, embed_dim, kernel_size=patch_size, stride=stride, bias=bias)
        self.emb_bn = nn.BatchNorm1d(embed_dim)
        self.emb_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        # self.linear = SpikLinearLayer(patch_size, embed_dim, tau=tau, lif_bias=bias)
        
        # positional encoding
        if pe:
            self.tape = tAPE(embed_dim, max_len=num_patches)
            self.ape_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor, pe=True): 
        # T, B, C, L = x.shape
        # # -> T, BxC, N, D = x.shape
        # n_vars = x.shape[2]
        # x = self.padding_patch_layer(x.view(T * B, C, -1)).view(T, B, C, -1)
        # x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        # self.org_x = x.clone().detach() #[T, BxC, N, P]
        # x = self.linear(x)
        #  # [T B*C N D]
        # if pe:
        #     x = self.tape(x.transpose(-1, -2).contiguous())
        #     x = x.transpose(-1, -2).contiguous()
            
        # return x, n_vars
        T, B, _, _ = x.shape
        # x = x.transpose(-1, -2).contiguous()
        x = self.emb_conv(x.flatten(0, 1)) # have some fire value [TB C N1 N2]
        # x = self.emb1_linear(x.flatten(0, 1))
        x = self.emb_bn(x)
        x = x.reshape(T, B, self.embed_dim, -1).contiguous()
        self.org_x = x.clone().detach().transpose(-1, -2) # [T B N D]
        x = x.flatten(3).contiguous() # [T B D N] 

        if pe:
            x = self.tape(x)
            
        
        x = self.emb_lif(x.reshape(T, B, self.embed_dim, -1).contiguous())
        x = x.transpose(-1, -2).contiguous()  # [T B N D]
        
        return x

# class Embedding(nn.Module):
#     def __init__(self, num_patches, pe=False, patch_size=63, stride=2, embed_dim=128, dropout=0, bias=False, tau=2.0) -> None:
#         super().__init__()
        
#         self.num_patches = num_patches
#         # self.patch_size = patch_size
#         # in_channel = patch_size
        
#         self.stride = stride
#         self.embed_dim = embed_dim
#         self.pe = pe
        
#         self.emb_linear = nn.Linear(patch_size, embed_dim, bias=bias)
#         self.emb_bn = nn.BatchNorm1d(embed_dim)
#         self.emb_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
#         # positional encoding
#         if pe:
#             self.tape = tAPE(embed_dim, max_len=num_patches)
#             self.ape_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
#         # Residual dropout
#         # self.dropout = nn.Dropout(dropout)

#     def forward(self, x:torch.tensor, pe=True): 
#         T, B, _,_ = x.shape
        
#         x = self.emb_linear(x.flatten(0, 1))
#         x = self.emb_bn(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
#         x = x.reshape(T, B, self.embed_dim, -1).contiguous() # [T B N D]
#         x = x.flatten(3).contiguous() # [T B D N] 

#         if pe:
#             x = self.tape(x)
            
        
#         x = self.emb_lif(x.reshape(T, B, self.embed_dim, -1).contiguous())
#         x = x.transpose(-1, -2).contiguous()  # [T B N D]
        
#         return x
    
class Block(nn.Module):
    def __init__(self, dim, seq_len, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0.1, norm_layer=nn.LayerNorm, sr_ratio=1, lif_bias=False, tau=2.0, mutual=False, attn='MSSA'):
        super().__init__()
        
        self.attn = SSA_rel_scl(dim, seq_len, num_heads, lif_bias=lif_bias, tau=tau, attn=attn, drop=drop) 
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mutual: self.mca = MutualCrossAttention(dim=dim, lif_bias=lif_bias, num_heads=num_heads, tau=tau, attn=attn, drop=drop)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, lif_bias=lif_bias, tau=tau)
        
        self.gate = nn.Linear(seq_len * 2, seq_len, bias=lif_bias)
        self.gate_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
        # self.dropout = nn.Dropout(drop_path)

        
    def forward(self, x: torch.Tensor, mx=None, time_block:nn.Module=None):
        T, B, N, D = x.shape

        x = x * (1. - self.attn(x))
        if mx is not None: x = x * (1. - self.mca(x, mx)) 
        x = x * (1. - self.mlp(x)) # channel mixer
        
        return x
  
class TemporalBlock(nn.Module):
    def __init__(self, num_layers, T, patch_size=[16, 32], embed_dim=[64, 128, 256], ratio=8, lif_bias=False, tau=2.0, num_heads=8, mutual=False):
        super().__init__()
        
        self.T = T  # time step

        self.patch_size = patch_size
        self.num_layers = num_layers
        
        self.in_layer = SpikLinearLayer(embed_dim, embed_dim * 2, lif_bias=lif_bias, tau=tau)
        # self.layers = nn.ModuleList([nn.Sequential(
        #     SpikLinearLayer(embed_dim * 2, embed_dim // 2, lif_bias=lif_bias, tau=tau),
        #     SpikLinearLayer(embed_dim // 2, embed_dim * 2, lif_bias=lif_bias, tau=tau),
        #     ) for l in range(num_layers)])
        self.layers = nn.ModuleList(nn.Sequential(
            SpikLinearLayer(embed_dim * 2, embed_dim * 4, lif_bias=lif_bias, tau=tau),
            SpikLinearLayer(embed_dim * 4, embed_dim * 2, lif_bias=lif_bias, tau=tau),
            ) for l in range(num_layers))
        self.out_layer = SpikLinearLayer(embed_dim * 2, embed_dim, lif_bias=lif_bias, tau=tau)

    def visualize_embedding(self, x, original_data=None, channel_idx=0):
        # x: [T, B, 1, D]
        import matplotlib.pyplot as plt

        # Ensure x has shape [T, B, 1, D] before squeezing
        if x.dim() == 4 and x.shape[2] == 1:
            x_mean = x.mean(dim=-1).squeeze(2)  # [T, B]
        else:
            x_mean = x.mean(dim=-1)  # fallback if shape is [T, B, D]

        batch_idx = 0  # 첫 번째 배치만 시각화

        plt.figure(figsize=(10, 5))
        plt.plot(range(x_mean.shape[0]), x_mean[:, batch_idx].cpu().detach().numpy(), label='Embedding Mean')

        if original_data is not None:
            # original_data: [B, L, C] or [B, L]
            # 첫 번째 배치의 지정된 채널만 시각화
            if original_data.dim() == 3:
                # c = min(channel_idx, original_data.shape[2] - 1)
                orig = original_data[batch_idx, :, :].mean(-1)  # [L]
            elif original_data.dim() == 2:
                orig = original_data[batch_idx, :]  # [L]
            else:
                orig = original_data.flatten()
            # Scale to [0, 1]
            orig_min = orig.min()
            orig_max = orig.max()
            if orig_max > orig_min:
                orig_scaled = (orig - orig_min) / (orig_max - orig_min)
            else:
                orig_scaled = orig - orig_min  # all zeros if no variation

            # x의 토큰 개수(T)와 original_data의 길이(L) 맞추기
            T = x_mean.shape[0]
            L = orig_scaled.shape[0]
            # L → T로 리샘플링
            orig_resampled = np.interp(np.linspace(0, L - 1, T), np.arange(L), orig_scaled.cpu().detach().numpy())
            plt.plot(range(T), orig_resampled, label=f'Original Data (channel {channel_idx})', alpha=0.7)

        plt.xlabel('Token Index (T)')
        plt.ylabel('Value')
        plt.title('Embedding & Original Data Visualization (Aligned)')
        plt.legend()
        plt.savefig('embedding_visualization.png')
        plt.close()

    def forward(self, x, org_data=None):
        # x: [T, B, 1, D]

        x = self.in_layer(x)
        for l in range(self.num_layers):
            x = x * (1. - self.layers[l](x))
        x = self.out_layer(x)

        # self.visualize_embedding(x, org_data)
        return x
