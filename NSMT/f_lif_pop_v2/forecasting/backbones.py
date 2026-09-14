"""Small causal backbone adaptations; keep chronological population states in every block."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PopulationLIF


class CurrentBlock(nn.Module):
    def forward(self, current, spikes, return_aux=False, memory_mode=None):
        current = current + self.input_scale * self.mix(spikes.flatten(2))
        result = self.lif(current, return_aux=return_aux, memory_mode=memory_mode)
        if return_aux:
            spikes, aux = result
            return current, spikes, aux
        return current, result


class TemporalBlock(CurrentBlock):
    def __init__(self, embed_dim, num_population, dilation, input_scale=2., **neuron_args):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim * num_population, embed_dim, 3, dilation=dilation)
        self.left_padding = 2 * dilation
        self.input_scale = input_scale
        self.lif = PopulationLIF(num_population=num_population, **neuron_args)

    def mix(self, x):
        x = F.pad(x.permute(1, 2, 0).contiguous(), (self.left_padding, 0))
        return self.conv(x).permute(2, 0, 1)


class AttentionBlock(CurrentBlock):
    def __init__(self, embed_dim, num_population, num_patches, num_heads=4,
                 input_scale=2., **neuron_args):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError('Embedding width must be divisible by attention heads')
        self.proj = nn.Linear(embed_dim * num_population, embed_dim)
        self.position = nn.Parameter(torch.zeros(num_patches, 1, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.)
        self.register_buffer('future_mask', torch.ones(num_patches, num_patches, dtype=torch.bool).triu(1), persistent=False)
        self.input_scale = input_scale
        self.lif = PopulationLIF(num_population=num_population, **neuron_args)

    def mix(self, x):
        x = self.proj(x) + self.position
        return self.attention(x, x, x, attn_mask=self.future_mask, need_weights=False)[0]


class ChannelBlock(CurrentBlock):
    def __init__(self, embed_dim, num_population, input_scale=2., **neuron_args):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embed_dim * num_population, 2 * embed_dim),
                                 nn.GELU(), nn.Linear(2 * embed_dim, embed_dim))
        self.input_scale = input_scale
        self.lif = PopulationLIF(num_population=num_population, **neuron_args)

    def mix(self, x):
        return self.mlp(x)


class CausalLinear(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.linear = nn.Linear(steps, steps)
        self.register_buffer('past_mask', torch.ones(steps, steps).tril(), persistent=False)

    def forward(self, x):
        return F.linear(x, self.linear.weight * self.past_mask, self.linear.bias)


class TokenBlock(CurrentBlock):
    def __init__(self, embed_dim, num_population, num_patches, input_scale=2., **neuron_args):
        super().__init__()
        self.proj = nn.Linear(embed_dim * num_population, embed_dim)
        self.time_mlp = nn.Sequential(CausalLinear(num_patches), nn.GELU(), CausalLinear(num_patches))
        self.input_scale = input_scale
        self.lif = PopulationLIF(num_population=num_population, **neuron_args)

    def mix(self, x):
        x = self.proj(x).permute(1, 2, 0)
        return self.time_mlp(x).permute(2, 0, 1)
