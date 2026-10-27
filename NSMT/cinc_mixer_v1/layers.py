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
from neurons import LIFNeuron, FLIFNeuron, PopulationFLIF   # nsmt_flif_v1: spikeDE LIF / f-LIF / population f-LIF for the neocortex path


__all__ = ['spikformer']

def make_look_ahead_mask(x):
    T, B, N, D = x.shape
    device = x.device
    mask = torch.triu(torch.ones((N, N))).reshape(1, 1, 1, N, N).to(device)
    
    return mask

class SpikLinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, tau=2.0, lif_bias=False, neuron=None):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        
        self.linear = nn.Linear(in_dim, self.out_dim, bias=lif_bias)
        self.bn = nn.BatchNorm1d(self.out_dim)
        # nsmt_flif_v1: `neuron` = a ready-made neocortex neuron (NeoNeuron); None keeps the paper's spikingjelly LIF
        self.lif = neuron if neuron is not None else MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
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
    def __init__(self, dim, seq_len, num_heads=8, pe=True, lif_bias=False, tau=2.0, topk_ratio=None, drop=0.1, attn_bn=False) -> None:
        super().__init__()
        
        self.seq_len = seq_len
        # MODI 2026-10-21 (BN audit): optional BN in front of the attention neuron (the only neuron without Linear+BN in front)
        self.attn_bn = nn.BatchNorm1d(dim) if attn_bn else None
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
    def forward(self, x, mask=None, kv=None):
        T,B,N,C = x.shape
        
        # T, batch_size, seq_len, _ = x.shape 
        x_for_qkv = x.flatten(0, 1)  # TB, N, D
        # MODI 2026-10-21 (1-C): with kv [T, B, N, C] the keys / values come from kv and only the queries from x
        x_for_kv = x_for_qkv if kv is None else kv.flatten(0, 1)
        
        k = self.k_linear(x_for_kv)
        k = self.k_lif(self.k_bn(k.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous())
        k = k.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v = self.v_linear(x_for_kv)
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
        if self.attn_bn is not None:
            x = self.attn_bn(x.flatten(0, 1).transpose(-1, -2).contiguous()).transpose(-1, -2).reshape(T, B, N, C).contiguous()
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
        


# ---------------------------------------------------------------------------------------------------------------
# nsmt_flif_v1: neocortex-path neurons (same adapter as nsmt_flif_v1/forecasting/layers.py)
class NeoNeuron(nn.Module):
    """Adapter: NSMT's 4-D spiking tensors -> neurons.py [T, B, D] current -> same 4-D layout.

    layout 'DN': x is [T, B, D, N'] (model.Embedding feeds the neuron after BN in this order)
    layout 'ND': x is [T, B, N', D] (SpikLinearLayer feeds the neuron in this order)
    dim 0 is the spiking time axis. In the classification model it is the token axis (T = N) inside
    time_block, but the repetition axis (the same patch shown T times) inside encoding_neo.
    The neuron keeps its state inside forward(), so spikingjelly's reset_net is a no-op for it.
    """
    def __init__(self, neuron, layout):
        super().__init__()
        if layout not in ('DN', 'ND'):
            raise ValueError("layout must be 'DN' or 'ND'")
        self.neuron = neuron
        self.layout = layout

    def forward(self, x):
        if getattr(self, '_capture', False):                          # MODI 2026-10-24: forward_charge through __call__ (hooks fire)
            s, self._charge = self._spikes_and_charge(x)
            return s
        if x.ndim != 4:
            raise ValueError(f'NeoNeuron expects a 4-D tensor, got {tuple(x.shape)}')
        T, B = x.shape[0], x.shape[1]
        if self.layout == 'DN':
            D, N1 = x.shape[2], x.shape[3]
            cur = x.permute(0, 1, 3, 2).reshape(T, B * N1, D)
            s = self.neuron(cur)                                      # [T, B*N', D]
            return s.reshape(T, B, N1, D).permute(0, 1, 3, 2).contiguous()
        N1, D = x.shape[2], x.shape[3]
        s = self.neuron(x.reshape(T, B * N1, D))
        return s.reshape(T, B, N1, D)

    def forward_charge(self, x):
        """MODI 2026-10-23: (spikes, H - theta) in the layout of forward()'s output (layers.spikes_and_charge). Runs through
        the module call (MODI 2026-10-24) so that forward hooks on this module see it, as for forward()."""
        self._capture = True
        try:
            s = self(x)
        finally:
            self._capture = False
        h, self._charge = self._charge, None
        return s, h

    def _spikes_and_charge(self, x):
        if x.ndim != 4:
            raise ValueError(f'NeoNeuron expects a 4-D tensor, got {tuple(x.shape)}')
        T, B = x.shape[0], x.shape[1]
        if self.layout == 'DN':
            D, N1 = x.shape[2], x.shape[3]
            s, h = spikes_and_charge(self.neuron, x.permute(0, 1, 3, 2).reshape(T, B * N1, D))
            back = lambda z: z.reshape(T, B, N1, D).permute(0, 1, 3, 2).contiguous()
            return back(s), back(h)
        N1, D = x.shape[2], x.shape[3]
        s, h = spikes_and_charge(self.neuron, x.reshape(T, B * N1, D))
        return s.reshape(T, B, N1, D), h.reshape(T, B, N1, D)


NEO_NEURONS = ('sj_lif', 'lif', 'flif', 'pop_flif')


def spikes_and_charge(n, x):
    """MODI 2026-10-23 (membrane fusion): run a neurons.py neuron (LIFNeuron / FLIFNeuron / PopulationFLIF, members firing
    or one soma) on the current x [T, B, D] through the SAME rhs / integrate / emit calls as its forward(), and also return,
    with the graph attached, the value its threshold compares at every step minus the threshold:
        members fire: H_t = u_t + (I_t - u_t) / tau (rhs's look-ahead),   soma: H_t = V_{t-1} + (a_t - V_{t-1}) / tau_s,
    so spikes == (H - theta >= 0) exactly (arctan_spike's forward). -> (spikes [T, B, out_features], H - theta, same shape)"""
    if x.ndim != 3:
        raise ValueError('Expected [T, B, D] current')
    T, B, D = x.shape
    n._check_length(T)
    u = x.new_zeros(B, D, n.num_members)
    soma = n.initial_soma(x)
    history = n.new_history()
    out, charge = [], []
    for t in range(T):
        F, s = n.rhs(u, x[t])
        nxt, _, _ = n.integrate(t, u, F, x[t], history, None)
        if n.has_soma:
            prev = soma
            o, soma, drive = n.emit(nxt, s, soma)
            h = prev + (drive - prev) / n.soma.tau_s - n.soma.threshold
        else:
            o, soma, _ = n.emit(nxt, s, soma)
            h = (u + (x[t].unsqueeze(-1) - u) / n.tau - n.threshold).reshape(B, -1)
        out.append(o)
        charge.append(h)
        u = nxt
    out, charge = torch.stack(out), torch.stack(charge)
    if n._forward_pre_hooks:
        raise NotImplementedError('forward pre-hooks on a neuron run through spikes_and_charge')
    for hook in list(n._forward_hooks.values()):                      # the energy / syops recorders see the call as forward()
        hook(n, (x,), out)
    return out, charge


def build_neo_neuron(kind, size, T, layout, tau=2., alpha=.5, method='pred', step=1., memory=0, pop_taus=(4., 8., 16., 32.),
                     pop_soma_init=None):
    """'lif' / 'flif': neurons.py classes wrapped in NeoNeuron; `T` = spiking steps (sizes the kernel table).
    'sj_lif' is handled by the callers (they keep their own spikingjelly LIF, constructed exactly as in nsmt_base)."""
    common = dict(tau=tau, threshold=1., surrogate_scale=5., step=step, max_length=int(T))
    if kind == 'lif':
        n = LIFNeuron(size, **common)
    elif kind == 'flif':
        n = FLIFNeuron(size, alpha=alpha, method=method, memory=(int(memory) if memory else None), **common)
    elif kind in ('pop_flif', 'pop_lif', 'pop_rec'):
        # f_lif_pop_v3 population neuron: K non-resetting f-LIF branches (time constants pop_taus) mixed by learnable
        # weights into one LIF soma (time constant `tau`); same alpha / solver as 'flif'; output = soma spikes [T, B, D]
        # MODI 2026-10-21: 'pop_lif' = the same class with alpha 1 (pred + full history = exact Euler LIF branches);
        # 'pop_rec' = population f-LIF with a gated recurrent current tanh(g) W_rec s_{t-1} on the soma spikes (g init 0)
        common.pop('tau')
        cls = RecurrentPopulationFLIF if kind == 'pop_rec' else PopulationFLIF
        n = cls(size, tau=tuple(float(t) for t in pop_taus), soma='lif', tau_s=tau, heterogeneous=True,
                alpha=(1. if kind == 'pop_lif' else alpha), method=method, memory=(int(memory) if memory else None), **common)
        if pop_soma_init is not None:                                 # stage 9 calibration: soma mix weights 1/K -> value
            with torch.no_grad():
                n.soma.weight.fill_(float(pop_soma_init))
    else:
        raise ValueError(f"build_neo_neuron needs 'lif', 'flif', 'pop_flif', 'pop_lif' or 'pop_rec', got {kind!r}")
    return NeoNeuron(n, layout)


# ---- MODI 2026-10-21: soma-less population (tau groups / streams) and recurrent population ---------------------------------
# Rate-matched BN gamma init (MODI 2026-10-22): for each neuron family and tau, the gain whose initial firing on real CinC
# BN pre-activations (embedding input for spikingjelly LIF, first-mixer-layer input for spikeDE neurons; seed 0, 32 training
# windows) equals that of the tau = 2 neuron of the same family at gain 1 (0.0274 / 0.0369 / 0.0384), found by bisection.
# Replaces the 10-21 rule sqrt((2 tau - 1) / 3) x 2, which over-drove the spikeDE mixer neurons 3-9x (smoke, 10-22).
TAU_GAIN = {'sj': {4.: 1.901, 8.: 3.689, 16.: 7.652, 32.: 17.088},
            'lif': {4.: 1.632, 8.: 2.424, 16.: 3.432, 32.: 4.775},
            'flif': {4.: 1.758, 8.: 3.089, 16.: 5.547, 32.: 10.215}}


def tau_gain(tau, family='sj'):
    """BN gamma init for a neuron of `family` (sj = spikingjelly LIF, lif / flif = spikeDE) with time constant tau (table above;
    tau = 2 -> 1, so the default model is unchanged)."""
    t = float(tau)
    if t == 2.:
        return 1.
    if t not in TAU_GAIN[family]:
        raise ValueError(f'no calibrated gain for tau {t} ({family})')
    return TAU_GAIN[family][t]


def sj_lif(tau=2.):
    """the spikingjelly LIF used everywhere else in these models (embedding, MLP, attention)."""
    return MultiStepLIFNode(tau=float(tau), detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())


class GroupNeuron(nn.Module):
    """K groups of D/K units along one axis, group k with time constant taus[k]; every unit fires and resets itself
    (no soma). kind 'sj' = spikingjelly LIF (any input layout, `axis` = channel axis); 'lif' / 'flif' = spikeDE LIFNeuron /
    FLIFNeuron (alpha) on a 4-D [T, B, N', D] input (channel axis last), as NeoNeuron('ND')."""
    def __init__(self, size, taus, kind='sj', T=64, alpha=.5, axis=-1):
        super().__init__()
        self.taus, self.kind, self.axis = tuple(float(t) for t in taus), kind, axis
        K = len(self.taus)
        if size % K:
            raise ValueError(f'size {size} not divisible by {K} groups')
        self.G = size // K
        if kind == 'sj':
            self.parts = nn.ModuleList([sj_lif(t) for t in self.taus])
        elif kind in ('lif', 'flif'):
            mk = (lambda t: LIFNeuron(self.G, tau=t, threshold=1., surrogate_scale=5., max_length=int(T))) if kind == 'lif' else \
                 (lambda t: FLIFNeuron(self.G, alpha=alpha, method='pred', memory=None, tau=t, threshold=1., surrogate_scale=5., max_length=int(T)))
            self.parts = nn.ModuleList([NeoNeuron(mk(t), 'ND') for t in self.taus])
        else:
            raise ValueError(f'GroupNeuron kind {kind!r}')

    def forward(self, x):
        xs = torch.split(x, self.G, dim=self.axis)
        if getattr(self, '_capture', False):                          # MODI 2026-10-24: forward_charge through __call__
            parts = [p.forward_charge(c.contiguous()) for p, c in zip(self.parts, xs)]
            self._charge = torch.cat([b for _, b in parts], dim=self.axis)
            return torch.cat([a for a, _ in parts], dim=self.axis)
        return torch.cat([p(c.contiguous()) for p, c in zip(self.parts, xs)], dim=self.axis)

    def forward_charge(self, x):
        """MODI 2026-10-23: (spikes, H - theta) of the spikeDE groups (kind 'lif' / 'flif'), concatenated like forward(); through
        the module call (MODI 2026-10-24) so that hooks on this module and on every group see it."""
        if self.kind == 'sj':
            raise ValueError('charge capture is defined for the spikeDE groups')
        self._capture = True
        try:
            s = self(x)
        finally:
            self._capture = False
        h, self._charge = self._charge, None
        return s, h


def group_gamma(size, taus, family='sj'):
    """BN weight vector [size] for a GroupNeuron: group k gets tau_gain(taus[k], family)."""
    G = size // len(taus)
    return torch.cat([torch.full((G,), tau_gain(t, family)) for t in taus])


class RecurrentPopulationFLIF(PopulationFLIF):
    """Population f-LIF (soma='lif') whose input current at step t gets tanh(g) W_rec s_{t-1} from its own soma spikes;
    g = 0 at init gives exactly PopulationFLIF (the same rhs / integrate / emit calls). neurons.py is untouched."""
    def __init__(self, size, rec_scale=.1, **kwargs):
        super().__init__(size, **kwargs)
        self.w_rec = nn.Linear(size, size, bias=False)
        nn.init.uniform_(self.w_rec.weight, -rec_scale / size ** .5, rec_scale / size ** .5)
        self.w_rec._skip_global_init = True
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, mode=None, oracle_p=None, return_aux=False, analog=False):
        if return_aux or analog:
            raise ValueError('RecurrentPopulationFLIF: plain spike output only')
        T, B, D = x.shape
        self._check_length(T)
        u = x.new_zeros(B, D, self.num_members)
        soma = self.initial_soma(x)
        history = self.new_history()
        gamma = torch.tanh(self.g)
        s_prev = x.new_zeros(B, D)
        spikes = []
        for t in range(T):
            xt = x[t] + gamma * self.w_rec(s_prev)
            F, s = self.rhs(u, xt)
            nxt, _, _ = self.integrate(t, u, F, xt, history, oracle_p)
            out, soma, _ = self.emit(nxt, s, soma)
            spikes.append(out)
            s_prev, u = out, nxt
        return torch.stack(spikes)


CALIB_TAUS = (4., 8., 16., 32.)


def apply_tau_calibration(model, model_name, mixer, comb, taus):
    """Set the BN weights of a soma-less population model from calib/tau_gains_<model>_<mixer>_<comb>.json (calibrate_groups.py,
    MODI 2026-10-22). Only for the calibrated taus 4/8/16/32; a missing table or a grouped BN without an entry raises."""
    import json, os
    if tuple(float(t) for t in taus) != CALIB_TAUS:
        return False
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calib', f'tau_gains_{model_name}_{mixer}_{comb}.json')
    table = json.load(open(path))['gains']                              # FileNotFoundError = fail closed
    mods = dict(model.named_modules())
    for name, mod in mods.items():
        parts = getattr(mod, 'parts', None)
        if parts is not None and len(parts) > 1 and type(mod).__name__ in ('GroupNeuron', 'StreamNeurons'):
            parent, leaf = name.rsplit('.', 1)
            bn_name = parent + '.' + {'emb_lif': 'emb_bn', 'attn_neuron': 'attn_bn', 'sd_neuron': 'sd_bn'}.get(leaf, 'bn')   # 10-27: linsd
            if bn_name not in table:
                raise KeyError(f'{path}: no calibrated gains for {bn_name}')
    with torch.no_grad():
        for bn_name, gains in table.items():
            bn = mods[bn_name]
            G = bn.weight.numel() // len(gains)
            for k, g in enumerate(gains):
                bn.weight[k * G:(k + 1) * G] = float(g)
    return True
