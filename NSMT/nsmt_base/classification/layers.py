import torch
import torch.nn as nn

# snn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate
# from spikingjelly.clock_driven import functional

from einops import rearrange
from spikingjelly.activation_based.encoding import PoissonEncoder
from utils import create_temporal_proximity_mask

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
    def __init__(self, in_dim, out_dim=None, tau=2.0, lif_bias=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        
        self.linear = nn.Linear(in_dim, self.out_dim, bias=lif_bias)
        self.bn = nn.BatchNorm1d(self.out_dim)
        self.lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
    def forward(self, x):
        
        x_shape = x.shape
        
        x = self.linear(x.flatten(0, 1))
        x = self.bn(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        x = self.lif(x.reshape(x_shape[0], x_shape[1], x_shape[2], -1).contiguous())
        
        return x
    
class SpikLinearMaxLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, kernel_size=2, spk='lif', tau=2.0, lif_bias=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.kernel_size = kernel_size
        
        self.linear = nn.Linear(in_dim, self.out_dim, bias=lif_bias)
        self.bn = nn.BatchNorm1d(self.out_dim)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.spk_neuron = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
      
    def forward(self, x):
        
        x_shape = x.shape
        
        x = x.view(-1, x_shape[-1]) if x.is_contiguous() else x.reshape(-1, x_shape[-1])
        x = self.linear(x)
        x = self.bn(x)
        x = self.max_pool(x)
        x = self.spk_neuron(x.reshape(x_shape[0], x_shape[1], x_shape[2], -1).contiguous())
        
        return x
    
    
class SpkEncoder(nn.Module):
    def __init__(self, time_steps) -> None:
        super().__init__()
        
        self.T = time_steps
        # self.batch_size = batch_size
        # self.seq_len = seq_len
        # self.device = device
        
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
        
        
        self.fc1 = SpikLinearLayer(in_features, hidden_features)
        self.fc2 = SpikLinearLayer(hidden_features, out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features
        
        self.dropout = nn.Dropout(drop)
        
        

    def forward(self, x):
        T, B, N, C = x.shape
        x = self.fc1(x)
        x = self.fc2(x)

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

        self.proj_linear = SpikLinearLayer(in_dim=dim, out_dim=dim, tau=tau, lif_bias=lif_bias)

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
        
        # if self.topk is not None and self.topk > 0:
        #     T_, B_, H_, Nq, Nk = attn_scores.shape
        #     k_keep = min(self.topk, Nk)

        #     # [T*B*H*N_q, N_k]
        #     attn_flat = attn_scores.reshape(-1, Nk)

        #     # 각 row마다 top-k index
        #     _, topk_idx = torch.topk(attn_flat, k=k_keep, dim=-1)

        #     # 같은 shape의 boolean mask 생성
        #     topk_mask = torch.zeros_like(attn_flat, dtype=torch.bool)
        #     topk_mask.scatter_(1, topk_idx, True)

        #     # 원래 shape로 복원
        #     topk_mask = topk_mask.view(T_, B_, H_, Nq, Nk)

        #     # top-k가 아닌 위치는 0으로 날려버림
        #     attn_scores = attn_scores.masked_fill(~topk_mask, 0.0)
        
        x = attn_scores @ v

        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x) 
        x = self.proj_linear(x)
        return x
    
    
    
    
class MutualCrossAttention(nn.Module):
    def __init__(self, dim, out_seq=True, lif_bias=False, num_heads=8, tau=2.0, attn='MSSA', drop=0.1) -> None:
        super().__init__()
        
        self.scale = dim ** -0.5
        self.num_heads = num_heads
        self.out_seq = out_seq
        
        self.attn = attn
        # matrix decomposition

        # self.q_lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        self.q_linear2 = nn.Linear(dim, dim, bias=lif_bias)
        self.q_bn2 = nn.BatchNorm1d(dim)
        self.q_lif2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())

        self.k_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())

        self.v_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        
        self.attn_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        
        self.proj_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())
        
        self.dropout = nn.Dropout(drop)
        
    def visualize_attn_scores(self, attn_scores, title=None):
        # --- attn_scores 시각화 코드 (디버깅용) ---
        # 예시: 첫 번째 batch, head, query에 대한 attention map 시각화
        attn_slice = attn_scores[0, 0, 0].detach().cpu().numpy()  # shape: [N, T']
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
        
        # mask = create_temporal_proximity_mask(T).to(q.device)
        # mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, 1, 1, T, T]
        
        q = self.q_linear2(q.flatten(0, 1))
        q = self.q_lif2(self.q_bn2(q.transpose(-1,-2)).transpose(-1, -2).reshape(T, B, Nq, self.num_heads, D//self.num_heads).contiguous())
        q = q.transpose(-2, -3).contiguous() #[T B h N D//h]
        
        k = self.k_linear(kv.flatten(0, 1))
        k = self.k_lif(self.k_bn(k.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, Nkv, self.num_heads, D//self.num_heads).contiguous())

        k = k.transpose(-2, -3).contiguous()
        
        v = self.v_linear(kv.flatten(0, 1))
        v = self.v_lif(self.v_bn(v.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, Nkv, self.num_heads, D//self.num_heads).contiguous())
        v = v.transpose(-2, -3).contiguous()
        
        k = k.transpose(0, 3).contiguous() #[1 B h T' D//h]
        v = v.transpose(0, 3).contiguous()
        
        self.attn_scores = (q @ k.transpose(-1, -2)) # [T B h N T']
        
        # self.visualize_attn_scores(self.attn_scores.clone().detach())
        # self.visualize_attn_scores(mask.clone().detach(), title="Temporal_Proximity_Mask.png") 
        
        # self.attn_scores = self.attn_scores * mask
        z = (self.attn_scores @ v) * self.scale # [T B h N D//h]
        
        z = z.permute(0, 1, 3, 2, 4).contiguous()
        z = z.reshape(T, B, -1, D).contiguous()
        z = self.attn_lif(z.reshape(T, B, -1, D).contiguous())
        z = z.flatten(0, 1)
        z = self.proj_lif(self.proj_bn(self.proj_linear(z).transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous().reshape(T, B, -1, D).contiguous())
            
        return self.dropout(z)
    
    
class MemoryUpdate(nn.Module):
    def __init__(self, dim, out_seq=True, lif_bias=False, tau=2.0) -> None:
        super().__init__()
        
        self.scale = dim ** -0.5
        self.out_seq = out_seq
        self.count = 0
        # self.topk = dim // 2

        # self.gate = nn.Linear(dim, dim, bias=lif_bias)
        # self.gate = SpikLinearLayer(dim, dim, tau=tau, lif_bias=lif_bias)
        self.gate_linear = nn.Linear(dim, dim, bias=lif_bias)
        self.gate_bn = nn.BatchNorm1d(dim)
        self.gate_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
        self.proj = SpikLinearLayer(dim, dim, tau=tau, lif_bias=lif_bias)
        # self.proj_linear = nn.Linear(dim, dim, bias=lif_bias)
        # self.proj_bn = nn.BatchNorm1d(dim)
        # self.proj_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')  
        # self.gate_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
    def forward(self, q:torch.Tensor, kv:torch.Tensor):
        T, B, Nq, D = q.shape
        T, B, Nkv, D = kv.shape
        
        topk = int(T * 0.5)
        g = self.gate_linear(kv.flatten(0, 1)) # [T B N D]
        g = g.reshape(T, B, Nkv, D).contiguous()
        # g = kv
        A = q.transpose(0, 2).contiguous() @ g.transpose(-1, -2).contiguous() # [T B T T']
        A = A * self.scale
        topk_val, topk_idx = A.topk(topk, dim=-1)
        
        mask = torch.full_like(A, 1e-8)
        mask.scatter_(-1, topk_idx, 1)
        A = mask * A

        # Call visualization function for debugging
        # self.visualize_mask(mask.clone().detach())

        update = A @ g #[T B T' D]
        update = update.transpose(0, 2).contiguous()
        # update = self.gate_bn(update.flatten(0, 1).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, -1, D).contiguous()
        # update = self.gate_lif(update)
        update = self.proj(update).mean(2, keepdim=True) #[T B 1 D] 
        return update

    def visualize_mask(self, mask):
        # --- mask 시각화 코드 (디버깅용) ---
        mask_slice = mask[0, 0, 0].cpu().numpy()
        if mask_slice.ndim == 2:
            plt.imshow(mask_slice, cmap='gray')
            plt.title('Mask Visualization (first batch/head/query)')
            plt.xlabel('Top-k indices')
            plt.ylabel('Query indices')
            plt.colorbar()
            plt.savefig("mask_visualization.png")
            plt.close()
        elif mask_slice.ndim == 1:
            # Reshape 1D array to 2D for visualization
            plt.imshow(mask_slice.reshape(1, -1), cmap='gray', aspect='auto')
            plt.title('Mask Visualization (reshaped 1D array)')
            plt.xlabel('Top-k indices')
            plt.ylabel('Query indices')
            plt.colorbar()
            plt.savefig("mask_visualization.png")
            plt.close()
        else:
            print(f"Cannot plot mask: expected 1D or 2D array, got shape {mask_slice.shape}")
            plt.close()
        # ----------------------------
    
        # g = self.gate(kv)
        
