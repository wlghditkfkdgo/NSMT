"""One common PopulationLIF and forecast head across the four staged backbones."""
import torch.nn as nn

from .layers import Embedding
from .backbones import TemporalBlock, AttentionBlock, ChannelBlock, TokenBlock

__all__ = ['myModel']


class myModel(nn.Module):
    def __init__(self, seq_len=336, pred_len=96, patch_size=8, embed_dim=32,
                 num_population=4, head_dim=32, head_mode='flatten',
                 heterogeneous=True, retrieval=True, tau_min=2., tau_max=16.,
                 threshold=1., memory_strength=.05, temperature=.25, input_scale=2.,
                 architecture='patch', read_mode='sparse', null_logit_init=-1., num_heads=4):
        super().__init__()
        if seq_len < patch_size or seq_len % patch_size:
            raise ValueError('Require complete chronological non-overlapping patches')
        if min(pred_len, embed_dim, head_dim) < 1 or head_mode not in ['flatten', 'last']:
            raise ValueError('Invalid forecast dimensions or readout')
        self.seq_len, self.pred_len = seq_len, pred_len
        self.patch_size, self.head_mode = patch_size, head_mode
        self.num_patches = seq_len // patch_size
        neuron_args = dict(heterogeneous=heterogeneous, retrieval=retrieval,
                           tau_min=tau_min, tau_max=tau_max, threshold=threshold,
                           memory_strength=memory_strength, temperature=temperature,
                           read_mode=read_mode, null_logit_init=null_logit_init)
        self.embedding = Embedding(patch_size, embed_dim, input_scale, num_population=num_population, **neuron_args)
        # Construct the common readout first so its initialization also matches across backbones.
        self.head_compress = nn.Linear(embed_dim * num_population, head_dim)
        self.head = nn.Linear(head_dim * (self.num_patches if head_mode == 'flatten' else 1), pred_len)
        block_args = dict(embed_dim=embed_dim, num_population=num_population, input_scale=input_scale, **neuron_args)
        if architecture == 'patch':
            blocks = []
        elif architecture == 'tcn':
            blocks = [TemporalBlock(dilation=d, **block_args) for d in [1, 2]]
        elif architecture == 'patchtst':
            blocks = [AttentionBlock(num_patches=self.num_patches, num_heads=num_heads, **block_args), ChannelBlock(**block_args)]
        elif architecture == 'tsmixer':
            blocks = [TokenBlock(num_patches=self.num_patches, **block_args), ChannelBlock(**block_args)]
        else:
            raise ValueError('Unknown architecture')
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, return_aux=False, memory_mode=None):
        if x.ndim != 3 or x.shape[1] != self.seq_len:
            raise ValueError('Expected [B, seq_len, C] input')
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B * C, L)
        x = x.unfold(-1, self.patch_size, self.patch_size).permute(1, 0, 2).contiguous()
        current = self.embedding.proj(x) * self.embedding.input_scale
        result = self.embedding.lif(current, return_aux=return_aux, memory_mode=memory_mode)
        if return_aux:
            spikes, aux = result
            layers = {'embedding': aux}
        else:
            spikes = result
        for i, block in enumerate(self.blocks):
            result = block(current, spikes, return_aux=return_aux, memory_mode=memory_mode)
            if return_aux:
                current, spikes, aux = result
                layers['block' + str(i)] = aux
            else:
                current, spikes = result
        z = self.head_compress(spikes.flatten(2))
        z = z.transpose(0, 1).reshape(B * C, -1) if self.head_mode == 'flatten' else z[-1]
        output = self.head(z).reshape(B, C, self.pred_len).transpose(1, 2)
        return (output, dict(aux, layers=layers)) if return_aux else output
