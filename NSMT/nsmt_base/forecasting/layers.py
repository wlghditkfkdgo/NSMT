import torch
import torch.nn as nn
import torch.nn.functional as F
# snn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate, MultiStepParametricLIFNode
from einops import rearrange
from spikingjelly.activation_based.encoding import PoissonEncoder
from spikingjelly.clock_driven import functional as sjF
from spikingjelly.activation_based.auto_cuda.neuron_kernel import LIFNodeBPTTKernel, LIFNodeATGF, LIFNodeFPTTKernel
from spikingjelly.activation_based import base
from spikingjelly.activation_based import surrogate as act_surrogate

from utils import create_temporal_proximity_mask
from copy import deepcopy
from typing import Callable
from spikingjelly.activation_based import neuron

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import RelBias1DDeterministicFn


__all__ = ['spikformer']

def make_look_ahead_mask(x):
    T, B, N, D = x.shape
    device = x.device
    mask = torch.triu(torch.ones((N, N))).reshape(1, 1, 1, N, N).to(device)
    
    return mask

class SpikLinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, spk='lif', detach_reset=True, bn=True, tau=2.0, lif_bias=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        
        self.linear = nn.Linear(in_dim, self.out_dim, bias=lif_bias)
        if bn : self.bn = nn.BatchNorm1d(self.out_dim)
        if spk == 'lif' : 
            self.spk_neuron = MultiStepLIFNode(tau=tau, detach_reset=detach_reset, backend='cupy', surrogate_function=surrogate.Sigmoid())
        elif spk == 'plif' : 
            self.spk_neuron = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        else:
            raise ValueError
        
    def forward(self, x):
        
        x_shape = x.shape
        
        x = x.view(-1, x_shape[-1]) if x.is_contiguous() else x.reshape(-1, x_shape[-1])
        x = self.linear(x)
        x = self.bn(x)
        if len(x_shape) == 4:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1).contiguous()
        elif len(x_shape) == 3:
            x = x.view(x_shape[0], x_shape[1], -1).contiguous()
        x = self.spk_neuron(x)
        
        return x
    
class SpikTimeLinearLayer(nn.Module):
    def __init__(self, num_patches, out_dim=None, spk='lif', detach_reset=True, bn=True, tau=2.0, lif_bias=False):
        super().__init__()
        
        self.in_dim = num_patches
        self.out_dim = out_dim if out_dim is not None else num_patches
        
        self.linear = nn.Linear(num_patches, self.out_dim, bias=lif_bias)
        if bn : self.bn = nn.BatchNorm1d(self.out_dim)
        if spk == 'lif' : 
            self.spk_neuron = MultiStepLIFNode(tau=tau, detach_reset=detach_reset, backend='cupy', surrogate_function=surrogate.Sigmoid())
        elif spk == 'plif' : 
            self.spk_neuron = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        else:
            raise ValueError
        
    def forward(self, x):
        
        x = x.transpose(0, 3).contiguous()
        x_shape = x.shape
        
        x = x.view(-1, x_shape[-1]) if x.is_contiguous() else x.reshape(-1, x_shape[-1])
        x = self.linear(x)
        x = self.bn(x)
        if len(x_shape) == 4:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1).contiguous()
        elif len(x_shape) == 3:
            x = x.view(x_shape[0], x_shape[1], -1).contiguous()
        x = x.transpose(0, 3).contiguous()
        x = self.spk_neuron(x)
        
        return x
    
class SpikLinearMaxLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, kernel_size=2, spk='lif', tau=2.0, lif_bias=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.kernel_size = kernel_size
        
        self.linear = nn.Linear(in_dim, self.out_dim, bias=lif_bias)
        self.bn = nn.BatchNorm1d(self.out_dim)
        self.max_pool = nn.MaxPool1d(kernel_size, stride=kernel_size)
        if spk == 'lif' : 
            self.spk_neuron = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        elif spk == 'plif' : 
            self.spk_neuron = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        else:
            raise ValueError
        
    def forward(self, x):
        
        x_shape = x.shape
        
        x = x.view(-1, x_shape[-1]) if x.is_contiguous() else x.reshape(-1, x_shape[-1])
        x = self.linear(x)
        x = self.bn(x)
        x = self.max_pool(x)
        x = self.spk_neuron(x.reshape(x_shape[0], x_shape[1], x_shape[2], -1).contiguous())
        
        return x
    
class SpikResLinearLayer(nn.Module):
    def __init__(self, T, in_dim, out_dim=None, spk='lif', tau=2.0, lif_bias=False):
        super().__init__()
        
        
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        
        self.linear = nn.Linear(in_dim, self.out_dim, bias=lif_bias)
        self.bn = nn.BatchNorm1d(self.out_dim)
        if spk == 'lif' : 
            self.spk_neuron = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        elif spk == 'plif' : 
            self.spk_neuron = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        else:
            raise ValueError
        
        self.memory = nn.Linear(T, T, bias=lif_bias)
        
        mask = torch.ones(T, T)
        mask = torch.tril(mask)
        
        self.register_buffer("mem_mask", mask)
        
    def forward(self, x):
        
        x_shape = x.shape # T B N D
        x = x.view(-1, x_shape[-1]) if x.is_contiguous() else x.reshape(-1, x_shape[-1])
        x = self.linear(x)
        x = self.bn(x)
        
        if len(x_shape) == 4:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1).contiguous()
        elif len(x_shape) == 3:
            x = x.view(x_shape[0], x_shape[1], -1).contiguous()
            
        x_ = x.permute(1, 2, 3, 0).contiguous().view(-1, x_shape[0])
        W = self.memory.weight * self.mem_mask.T
        y = F.linear(x_, W)
        y = y.view(x_shape[1], 1, self.out_dim, x_shape[0]).permute(3, 0, 1, 2).contiguous()
        x = self.spk_neuron(x + y)

        return x

    
class RecurrentSpikLinearLayer(nn.Module):
    def __init__(self, T, in_dim, out_dim=None, spk='lif', tau=2.0, bias=False):
        super().__init__()
        
        self.out_dim = out_dim if out_dim is not None else in_dim
        
        self.linear = nn.Linear(in_dim, self.out_dim, bias=bias)
        self.bn = nn.BatchNorm1d(self.out_dim)
        if spk == 'lif' : 
            self.spk_neuron = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
            
        elif spk == 'plif' : 
            self.spk_neuron = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        
        self.memory = nn.parameter.Parameter(torch.randn(1, 1, out_dim), requires_grad=True)
        
    def forward(self, x):
        
        x_shape = x.shape
        
        x = x.view(-1, x_shape[-1]) 
        x = self.linear(x)
        x = self.bn(x)
        
        x = x.reshape(x_shape[0], x_shape[1], x_shape[2], -1).contiguous()
        x = self.spk_neuron(x)
        
        return x

# class RecurrentSpikLinearLayer(nn.Module):
#     """
#     GRU-like gated recurrence on membrane potential U, vectorized over full T.

#     Inputs
#       x:    [T, B, ..., in_dim]
#       U_in: Optional[Tensor] = None
#             - If provided with shape [T, B, ..., out_dim], it's treated as the
#               membrane trajectory from a previous/other layer and will be used
#               to build prev_U via time-shift: prev_U[t] = U_in[t-1], prev_U[0]=0.
#             - If None, fall back to (stateful ? shift(self.U_carry) : zeros).

#     Returns
#       s:     [T, B, ..., out_dim]  (spike sequence)
#       U_new: [T, B, ..., out_dim]  (membrane trajectory computed this forward)
#     """
#     def __init__(self, in_dim, out_dim=None, spk='lif', tau=2.0, bias=False, stateful=True):
#         super().__init__()
#         self.out_dim = out_dim if out_dim is not None else in_dim
#         self.stateful = stateful

#         # projection
#         self.linear = nn.Linear(in_dim, self.out_dim, bias=bias)
#         # 선택 사용: BN/LN 필요시 외부에서 감싸거나 교체 가능
#         self.bn = nn.BatchNorm1d(self.out_dim)

#         # spiking neuron (multi-step)
#         if spk == 'lif':
#             self.spk_neuron = MultiStepLIFNode(
#                 tau=tau, detach_reset=True, backend='cupy',
#                 surrogate_function=surrogate.Sigmoid()
#             )
#         elif spk == 'plif':
#             self.spk_neuron = MultiStepParametricLIFNode(
#                 init_tau=tau, detach_reset=True, backend='cupy',
#                 surrogate_function=surrogate.Sigmoid()
#             )
#         else:
#             raise ValueError(f"Unknown spk type: {spk}")

#         # GRU-like gates on [x_t, U_{t-1}]
#         self.w_z = nn.Linear(self.out_dim * 2, self.out_dim, bias=bias)  # update gate
#         self.w_r = nn.Linear(self.out_dim * 2, self.out_dim, bias=bias)  # reset gate
#         self.w_h = nn.Linear(self.out_dim * 2, self.out_dim, bias=bias)  # candidate

#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU()

#         # cached membrane from previous forward (same layer)
#         self.U_carry = None  # [T, B, ..., out_dim]

#         # (학습 안정화 팁) z 바이어스 음수 초기화 권장
#         if self.w_z.bias is not None:
#             nn.init.constant_(self.w_z.bias, -2.0)

#     @torch.no_grad()
#     def _shift_time(self, U_seq_like):
#         """prev[t] = U_seq_like[t-1]; prev[0] = 0"""
#         zero_t = torch.zeros_like(U_seq_like[:1])
#         return torch.cat([zero_t, U_seq_like[:-1]], dim=0)

#     @torch.no_grad()
#     def _build_prev_U(self, like_x, U_in=None):
#         """
#         Decide prev_U with priority:
#           1) If U_in (sequence) is given and shape matches T,B,...,out_dim -> shift(U_in)
#           2) Else if stateful & self.U_carry matches shape -> shift(self.U_carry)
#           3) Else zeros
#         """
#         if U_in is not None:
#             # U_in must be a full sequence to preserve causality via shifting
#             if U_in.shape == like_x.shape:
#                 return self._shift_time(U_in)
#             else:
#                 # shape가 다르면 사용하지 않고 fall back
#                 pass

#         if self.stateful and (self.U_carry is not None) and (self.U_carry.shape == like_x.shape):
#             return self._shift_time(self.U_carry)

#         return torch.zeros_like(like_x)

#     def _gru_membrane(self, x_proj, U_in=None):
#         """
#         x_proj: [T, B, ..., out_dim]
#         U_in:  Optional [T, B, ..., out_dim]  (see _build_prev_U)
#         returns: new_U [T, B, ..., out_dim]
#         """
#         prev_U = self._build_prev_U(x_proj, U_in=U_in)

#         cat_xu = torch.cat([x_proj, prev_U], dim=-1)  # [..., 2*out_dim]
#         Z = self.sigmoid(self.w_z(cat_xu))            # update
#         R = self.sigmoid(self.w_r(cat_xu))            # reset
#         H_hat = self.relu(self.w_h(torch.cat([R * prev_U, x_proj], dim=-1)))  # candidate

#         new_U = (1.0 - Z) * prev_U + Z * H_hat
#         return new_U

#     def forward(self, x, U=None, return_U=True):
#         """
#         x: [T, B, ..., in_dim]
#         U: Optional[Tensor] = [T, B, ..., out_dim] or None
#            - If provided, used only to build prev_U via time-shift.
#         return_U: whether to return membrane trajectory for stacking
#         """
#         assert x.shape[-1] == self.linear.in_features, \
#             f"Expected last dim {self.linear.in_features}, got {x.shape[-1]}"

#         orig_shape = x.shape  # (T, B, ..., in_dim)

#         # Linear (and optional norm outside if needed)
#         x_flat = x.reshape(-1, orig_shape[-1])                # [T*B*..., in_dim]
#         x_flat = self.linear(x_flat)                          # -> [*, out_dim]
#         x_flat = self.bn(x_flat)
#         x_proj = x_flat.view(*orig_shape[:-1], self.out_dim).contiguous()  # [T,B,...,out_dim]

#         # GRU-like membrane update (can use U from caller)
#         U_new = self._gru_membrane(x_proj, U_in=U)            # [T,B,...,out_dim]

#         # Spiking over full sequence
#         s = self.spk_neuron(x_proj + U_new)                   # [T,B,...,out_dim]

#         # cache for stateful mode
#         self.U_carry = U_new.detach() if self.stateful else None

#         if return_U:
#             return s, U_new
#         else:
#             return s

#     def reset_states(self, keep_carry=False):
#         self.spk_neuron.reset()
#         if not keep_carry:
#             self.U_carry = None


class SpkEncoder(nn.Module):
    def __init__(self, time_steps) -> None:
        super().__init__()
        
        self.T = time_steps
        self.encoder = PoissonEncoder()
    
    def forward(self, x):
        
        device = x.device
        
        spk_encoding = torch.zeros((self.T, x.shape[0], x.shape[1]), device=device)
        
        for t in range(self.T):
            spk_encoding[t] = self.encoder(x)
            
        return spk_encoding
    
    

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1, lif_bias=False, tau=2.0):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.spk_linear1 = SpikLinearLayer(in_features, hidden_features, lif_bias=lif_bias, tau=tau)
        self.spk_linear2 = SpikLinearLayer(hidden_features, out_features, lif_bias=lif_bias, tau=tau)

        self.c_hidden = hidden_features
        self.c_output = out_features
        
        self.dropout = nn.Dropout(drop)
        

    def forward(self, x):

        x = self.spk_linear1(x)
        x = self.spk_linear2(x)

        return x
    


    
class SSA_rel_scl(nn.Module):
    def __init__(self, dim, seq_len, num_heads=8, pe=True, lif_bias=False, tau=2.0, topk_ratio=None, drop=0.1) -> None:
        super().__init__()
        
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.pe = pe
        self.topk = int(topk_ratio * seq_len) if topk_ratio is not None else None
        
        self.q_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())

        self.k_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        
        self.v_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        
        self.attn_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())

        self.proj_linear = SpikLinearLayer(in_dim=dim, out_dim=dim, spk='lif', tau=tau, lif_bias=lif_bias)

        if pe:
            self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len -1), num_heads))
            
            # coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)), indexing='ij')
            # coords = torch.flatten(torch.stack(coords), 1)
            
            # relative_coords = coords[:, :, None] - coords[:, None, :]
            # relative_coords[1] += self.seq_len - 1
            # relative_coords = rearrange(relative_coords, 'c h w -> h w c')
            # relative_coords = relative_coords.contiguous()
            
            # relative_idx = relative_coords.sum(-1).flatten().unsqueeze(1)
            # self.register_buffer("relative_idx", relative_idx)

            pos = torch.arange(self.seq_len)
            rel_pos_idx = pos[None, :] - pos[:, None]           # (N, N), offset = j - i
            rel_pos_idx = rel_pos_idx + (self.seq_len - 1)              # shift to [0, 2N-2]
            self.register_buffer("rel_pos_idx", rel_pos_idx.long())
        
        
        # self.dropout = nn.Dropout(drop)
    def forward(self, x, mask=None):
        T,B,N,C = x.shape
        
        # T, batch_size, seq_len, _ = x.shape 
        x_for_qkv = x.flatten(0, 1)  # TB, N, D
        
        k = self.k_linear(x_for_qkv)
        k = self.k_lif(self.k_bn(k.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous())
        k = k.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v = self.v_linear(x_for_qkv)
        v = self.v_lif(self.v_bn(v.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, N, C).contiguous())
        v = v.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        
        q = self.q_linear(x_for_qkv)
        q = self.q_lif(self.q_bn(q.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous())
        q = q.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # attn shape = [T B head N N]
        
        if mask is not None:
            attn_scores = attn_scores + mask * float('-inf')
           
        if self.pe:
            # relative_bias = self.relative_bias_table.gather(0, self.relative_idx.repeat(1, self.num_heads))
            # relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1*self.seq_len, w=1*self.seq_len)
            relative_bias = RelBias1DDeterministicFn.apply(self.relative_bias_table, self.rel_pos_idx)
            # -> (1, 1, H, N, N)로 브로드캐스트되게
            relative_bias = relative_bias.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            
            attn_scores = attn_scores + relative_bias
        
        if self.topk is not None and self.topk > 0:
            T_, B_, H_, Nq, Nk = attn_scores.shape
            k_keep = min(self.topk, Nk)

            # [T*B*H*N_q, N_k]
            attn_flat = attn_scores.reshape(-1, Nk)

            # 각 row마다 top-k index
            _, topk_idx = torch.topk(attn_flat, k=k_keep, dim=-1)

            # 같은 shape의 boolean mask 생성
            topk_mask = torch.zeros_like(attn_flat, dtype=torch.bool)
            topk_mask.scatter_(1, topk_idx, True)

            # 원래 shape로 복원
            topk_mask = topk_mask.view(T_, B_, H_, Nq, Nk)

            # top-k가 아닌 위치는 0으로 날려버림
            attn_scores = attn_scores.masked_fill(~topk_mask, 0.0)
        
        x = attn_scores @ v

        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x) 
        x = self.proj_linear(x)
        return x
    
    

class SSA_max(nn.Module):
    def __init__(self, dim, seq_len, num_heads=8, pe=True, lif_bias=False, tau=2.0, topk_ratio=None, drop=0.1) -> None:
        super().__init__()
        
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.pe = pe
        self.topk = int(topk_ratio * seq_len) if topk_ratio is not None else None
        
        self.q_linear = SpikLinearLayer(dim, dim, lif_bias=lif_bias, tau=tau)
        self.k_linear = SpikLinearLayer(dim, dim, lif_bias=lif_bias, tau=tau)

        
        self.v_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        
        self.attn_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())

        self.proj_linear = SpikLinearLayer(in_dim=dim, out_dim=dim, spk='lif', tau=tau, lif_bias=lif_bias)

        if pe:
            self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len -1), num_heads))
            
            # coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)), indexing='ij')
            # coords = torch.flatten(torch.stack(coords), 1)
            
            # relative_coords = coords[:, :, None] - coords[:, None, :]
            # relative_coords[1] += self.seq_len - 1
            # relative_coords = rearrange(relative_coords, 'c h w -> h w c')
            # relative_coords = relative_coords.contiguous()
            
            # relative_idx = relative_coords.sum(-1).flatten().unsqueeze(1)
            # self.register_buffer("relative_idx", relative_idx)

            pos = torch.arange(self.seq_len)
            rel = pos[None, :] - pos[:, None]           # (N, N), offset = j - i
            rel = rel + (self.seq_len - 1)              # shift to [0, 2N-2]
            self.register_buffer("rel_pos_idx", rel.long())
        
        
        # self.dropout = nn.Dropout(drop)
    def forward(self, x, mask=None):
        T,B,N,D = x.shape
        
        # T, batch_size, seq_len, _ = x.shape 
        
        q = self.q_linear(x).reshape(T, B, N, self.num_heads, D//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        k = self.k_linear(x).reshape(T, B, N, self.num_heads, D//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
                                   
        v = self.v_linear(x.flatten(0, 1) )
        v = self.v_lif(self.v_bn(v.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, N, D).contiguous())
        v = v.reshape(T, B, N, self.num_heads, D//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # attn shape = [T B head N N]
        
        if mask is not None:
            attn_scores = attn_scores + mask * float('-inf')
           
        if self.pe:
            # relative_bias = self.relative_bias_table.gather(0, self.relative_idx.repeat(1, self.num_heads))
            # relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1*self.seq_len, w=1*self.seq_len)
            relative_bias = RelBias1DDeterministicFn.apply(self.relative_bias_table, self.rel_pos_idx)
            # -> (1, 1, H, N, N)로 브로드캐스트되게
            relative_bias = relative_bias.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            
            attn_scores = attn_scores + relative_bias
        
        x = attn_scores @ v

        x = x.transpose(2, 3).reshape(T, B, N, D).contiguous()
        x = self.attn_lif(x) 
        weight = self.gating(x)
        x = x * weight[0] + weight[1]
        x = self.proj_linear(x)
        return x
    
    
class MLP4weight(nn.Module):
    def __init__(self, seq_len, ):
        super().__init__()
        
        self.fc = nn.Linear(seq_len, 2 * seq_len)
        self._init_weights(seq_len)
        
    def _init_weights(self, seq_len):
        with torch.no_grad():
            self.fc.weight.data.fill_(0)
            self.fc.bias.data[:seq_len].fill_(1)
            self.fc.bias.data[seq_len:].fill_(0)
            
    def forward(self, x):
        x = x.permute(2, 1, 3, 0).contiguous() #[1, B, D, T]
        x = self.fc(x) #[1, B, D, 2*T]
        x = x.permute(3, 1, 0, 2).contiguous()
        return x
            
    
class MutualCrossAttention(nn.Module):
    def __init__(self, seq_len, dim, out_seq=True, pe=True, lif_bias=False, num_heads=8, tau=2.0, drop=0.1) -> None:
        super().__init__()
        
        self.scale = dim ** -0.5
        self.num_heads = num_heads
        self.out_seq = out_seq
        self.seq_len = seq_len
        self.pe = pe
        # matrix decomposition

        # self.q_lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        self.q_linear2 = nn.Linear(dim, dim, bias=lif_bias)
        self.q_bn2 = nn.BatchNorm1d(dim)
        self.q_lif2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
    
        self.k_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        # self.k_lif = RecurrentSpikLinearLayer(tau=tau, detach_reset=True, backend='cupy', spk='plif')

        self.v_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        # self.v_lif = RecurrentSpikLinearLayer(tau=tau, detach_reset=True, backend='cupy', spk='plif')
        
        self.attn_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        
        self.proj_linear = SpikLinearLayer(in_dim=dim, out_dim=dim, spk='lif', tau=tau, lif_bias=lif_bias)
        self.dropout = nn.Dropout(drop)

    def _gen_mask(self, T, device):
        
        self.mask = create_temporal_proximity_mask(T).to(device)
        self.mask = self.mask.unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, 1, 1, T, T]
        
    def visualize_attn_scores(self, attn_scores, title=None):
        # --- attn_scores 시각화 코드 (디버깅용) ---
        # 예시: 첫 번째 batch, head, query에 대한 attention map 시각화
        attn_slice = attn_scores.mean(0).mean(0).mean(0).detach().cpu().numpy()  # shape: [N, T']
        plt.imshow(attn_slice, cmap='viridis', aspect='auto')
        plt.title('Attention Scores (first batch/head/query)')
        plt.xlabel('Key indices')
        plt.ylabel('Query indices')
        plt.colorbar()
        title = title if title is not None else "Attention Scores Visualization"    
        plt.savefig(title)
        plt.close()
        # ----------------------------
        
    def forward(self, q:torch.Tensor, kv:torch.Tensor):
        T, B, Nq, D = q.shape
        T, B, Nkv, D = kv.shape
        
        q = self.q_linear2(q.flatten(0, 1))
        q = self.q_lif2(self.q_bn2(q.transpose(-1,-2)).transpose(-1, -2).reshape(T, B, Nq, self.num_heads, D//self.num_heads).contiguous())
        q = q.transpose(-2, -3).contiguous() #[T B h N D//h]
        
        k = self.k_linear(kv.flatten(0, 1))
        k = self.k_lif(self.k_bn(k.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, Nkv, self.num_heads, D//self.num_heads).contiguous())

        k = k.transpose(-2, -3).contiguous()
        
        v = self.v_linear(kv.flatten(0, 1))
        v = self.v_lif(self.v_bn(v.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, Nkv, self.num_heads, D//self.num_heads).contiguous())
        v = v.transpose(-2, -3).contiguous()
        # v = kv.reshape(T, B, Nkv, self.num_heads, D//self.num_heads).contiguous()
        # v = v.transpose(-2, -3).contiguous()
        
        # k = k.transpose(0, 3).contiguous() #[1 B h T' D//h]
        # v = v.transpose(0, 3).contiguous()
        
        attn_scores = (q @ k.transpose(-1, -2)) * self.scale # [T B h N T']
    

        # self.attn_scores = self.attn_scores * self.mask
        z = attn_scores @ v # [T B h N D//h]
        # z = z.permute(3, 1, 0, 2, 4).contiguous() #[N, B, T, h, D//h]
        z = z.transpose(2, 3).reshape(T, B, -1, D).contiguous()
        z = self.attn_lif(z) 
        z = self.proj_linear(z)
        return z
    
    
# class Consolidation(nn.Module):
#     def __init__(self, dim, out_seq=True, lif_bias=False, tau=2.0) -> None:
#         super().__init__()
        
#         self.scale = dim ** -0.5
#         self.out_seq = out_seq
#         self.count = 0

#         self.gate_linear = nn.Linear(dim, dim, bias=lif_bias)
#         self.gate_bn = nn.BatchNorm1d(dim)
#         self.gate_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
#         self.proj = SpikLinearLayer(dim, dim, tau=tau, lif_bias=lif_bias)
        
#     def forward(self, q:torch.Tensor, kv:torch.Tensor):
#         T, B, Nq, D = q.shape
#         T, B, Nkv, D = kv.shape
        
#         topk = int(T * 0.5)
#         g = self.gate_linear(kv.flatten(0, 1)) # [T B N D]
#         g = g.reshape(T, B, Nkv, D).contiguous()
#         # g = kv
#         A = q.transpose(0, 2).contiguous() @ g.transpose(-1, -2).contiguous() # [T B T T']
#         A = A * self.scale
#         topk_val, topk_idx = A.topk(topk, dim=-1)
        
#         mask = torch.full_like(A, 1e-8)
#         mask.scatter_(-1, topk_idx, 1)
#         A = mask * A

#         # Call visualization function for debugging
#         # self.visualize_mask(mask.clone().detach())

#         update = A @ g #[T B T' D]
#         update = update.transpose(0, 2).contiguous()
#         # update = self.gate_bn(update.flatten(0, 1).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, -1, D).contiguous()
#         # update = self.gate_lif(update)
#         update = self.proj(update).mean(2, keepdim=True) #[T B 1 D] 
#         return update
    



class PatchMerging1D_TBND(nn.Module):
    """
    [T, B, N, D] -> [T, B, N/2, D_out]  (기본: D_out = 2*D)
    - 시간축 T는 보존
    - 토큰축 N을 2칸씩 묶어 채널로 결합(짝수/홀수), 이후 LN + Linear
    - N이 홀수면 pad_mode에 따라 1토큰 패딩
    """
    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        norm_layer=nn.LayerNorm,
        pad_mode: str = "zero"  # "zero" | "repeat" | "none"
    ):
        super().__init__()
        self.in_dim = dim
        self.out_dim = out_dim if out_dim is not None else 2 * dim
        self.norm = norm_layer(2 * dim)
        self.reduction = nn.Linear(2 * dim, self.out_dim, bias=False)
        assert pad_mode in {"zero", "repeat", "none"}
        self.pad_mode = pad_mode

    def _pad_one_token(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, D], N is odd -> pad to even
        T, B, N, D = x.shape
        if self.pad_mode == "none":
            raise AssertionError("N must be even when pad_mode='none'.")
        if self.pad_mode == "zero":
            pad_tok = torch.zeros((T, B, 1, D), device=x.device, dtype=x.dtype)
        else:  # "repeat"
            pad_tok = x[:, :, -1:, :].clone()
        return torch.cat([x, pad_tok], dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, D]
        T, B, N, D = x.shape
        if N % 2 == 1:
            x = self._pad_one_token(x)
            N = N + 1

        x_even = x[:, :, 0::2, :]   # [T, B, N/2, D]
        x_odd  = x[:, :, 1::2, :]   # [T, B, N/2, D]
        x_cat  = torch.cat([x_even, x_odd], dim=-1)  # [T, B, N/2, 2D]
        x_cat  = self.norm(x_cat)                    # LN on feature dim
        y      = self.reduction(x_cat)               # [T, B, N/2, out_dim]
        return y


class PatchExpanding1D_TBND(nn.Module):
    """
    [T, B, N, D] -> [T, B, 2N, D_out]  (기본: D_out = D)
    - 시간축 T는 보존
    - Linear로 채널을 2배로 확장 후, 픽셀셔플(1D)로 N을 2배로 펼침
    """
    def __init__(self, dim: int, dim_scale: int = 2, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim_scale == 2, "이 버전은 x2 업샘플 전용입니다."
        self.in_dim = dim
        self.expand = nn.Linear(dim, dim * dim_scale, bias=False)  # D -> 2D
        self.norm = norm_layer(dim // dim_scale)  # 최종 D_out = D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, D]
        x = self.expand(x)                            # [T, B, N, 2D]
        y = rearrange(x, 't b n (p d) -> t b (n p) d', p=2)  # [T, B, 2N, D]
        y = self.norm(y)
        return y