"""Causal population Spike-TCN adaptation; chronological patch time and current residuals."""
import torch
import torch.nn as nn

from .layers import Embedding, TemporalBlock

__all__ = ['myModel']


class myModel(nn.Module):
    def __init__(self, seq_len=336, pred_len=96, patch_size=8, embed_dim=32,
                 num_population=4, head_dim=32, head_mode='flatten',
                 heterogeneous=True, retrieval=True, tau_min=2., tau_max=16.,
                 threshold=1., memory_strength=0.05, temperature=0.25,
                 input_scale=2., tcn_layers=2, tcn_kernel=3):
        super().__init__()
        if seq_len < patch_size or seq_len % patch_size:
            raise ValueError('Require complete non-overlapping patches')
        if tcn_layers < 1 or tcn_kernel < 1:
            raise ValueError('TCN depth and kernel must be positive')
        if min(pred_len, embed_dim, head_dim) < 1 or head_mode not in ('flatten', 'last'):
            raise ValueError('Invalid forecast dimensions or head_mode')
        self.seq_len, self.pred_len = seq_len, pred_len
        self.patch_size, self.head_mode = patch_size, head_mode
        self.num_patches = seq_len // patch_size
        self.embedding = Embedding(patch_size, embed_dim, input_scale,
                                   num_population=num_population,
                                   heterogeneous=heterogeneous, retrieval=retrieval,
                                   tau_min=tau_min, tau_max=tau_max, threshold=threshold,
                                   memory_strength=memory_strength, temperature=temperature)
        self.blocks = nn.ModuleList([
            TemporalBlock(embed_dim, num_population, tcn_kernel, 2 ** i, input_scale,
                          heterogeneous=heterogeneous, retrieval=retrieval,
                          tau_min=tau_min, tau_max=tau_max, threshold=threshold,
                          memory_strength=memory_strength, temperature=temperature)
            for i in range(tcn_layers)])
        self.head_compress = nn.Linear(embed_dim * num_population, head_dim)
        self.head = nn.Linear(head_dim * (self.num_patches if head_mode == 'flatten' else 1), pred_len)

    def forward(self, x, return_aux=False, memory_mode=None):
        # [B,L,C] -> [BC,T,P]; T follows chronological patches, C shares weights.
        if x.ndim != 3 or x.shape[1] != self.seq_len:
            raise ValueError('Expected [B, seq_len, C] input')
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B * C, L)
        x = x.unfold(-1, self.patch_size, self.patch_size).permute(1, 0, 2).contiguous()
        current = self.embedding.proj(x) * self.embedding.input_scale
        z = self.embedding.lif(current, return_aux=return_aux, memory_mode=memory_mode)
        if return_aux:
            z, aux = z
            layer_aux = {'embedding': aux}
        for i, block in enumerate(self.blocks):
            result = block(current, z, return_aux=return_aux, memory_mode=memory_mode)
            if return_aux:
                current, z, aux = result
                layer_aux['block' + str(i)] = aux
            else:
                current, z = result
        if return_aux:
            # Existing diagnostic names describe the final population layer.
            aux = dict(aux, layers=layer_aux)
        # K spikes remain separate up to this readout.
        z = self.head_compress(z.flatten(2))            # [T,BC,head_dim]
        if self.head_mode == 'flatten':
            z = z.transpose(0, 1).reshape(B * C, -1)
        else:
            z = z[-1]
        out = self.head(z).reshape(B, C, self.pred_len).transpose(1, 2)
        return (out, aux) if return_aux else out
