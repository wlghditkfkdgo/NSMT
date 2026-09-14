"""Population membrane memory with amplitude-sensitive scores and explicit read support."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate


class Sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score):
        # Martins & Astudillo (2016), Alg.1: project onto the probability simplex.
        score = score - score.max(dim=-1, keepdim=True).values
        ordered = score.sort(dim=-1, descending=True).values
        cumulative = ordered.cumsum(dim=-1)
        ranks = torch.arange(1, score.shape[-1] + 1, device=score.device, dtype=score.dtype)
        support_size = (1 + ranks * ordered > cumulative).sum(dim=-1, keepdim=True)
        threshold = (cumulative.gather(-1, support_size - 1) - 1) / support_size
        output = (score - threshold).clamp_min(0.)
        ctx.save_for_backward(output > 0)
        return output

    @staticmethod
    def backward(ctx, gradient):
        # Eq.14 avoids sort/gather's nondeterministic CUDA scatter backward in torch1.12.
        support, = ctx.saved_tensors
        selected = gradient * support
        mean = selected.sum(dim=-1, keepdim=True) / support.sum(dim=-1, keepdim=True)
        return support * (gradient - mean)


def sparsemax(score):
    return Sparsemax.apply(score)


class PopulationLIF(nn.Module):
    def __init__(self, num_population=4, heterogeneous=True, retrieval=True,
                 tau_min=2., tau_max=16., threshold=1., memory_strength=.05,
                 temperature=.25, read_mode='sparse', null_logit_init=-1.):
        super().__init__()
        if num_population < 2 or not 1 < tau_min <= tau_max:
            raise ValueError('Require K >= 2 and 1 < tau_min <= tau_max')
        if threshold <= 0 or temperature <= 0 or read_mode not in ['dense', 'sparse']:
            raise ValueError('Invalid neuron or retrieval configuration')
        tau = torch.logspace(math.log10(tau_min), math.log10(tau_max), num_population)
        beta = torch.exp(-1. / tau)
        if not heterogeneous:
            beta = beta.mean().repeat(num_population)
        if not 0 <= memory_strength < 1 - beta.max().item():
            raise ValueError('Require memory_strength < 1 - max(beta)')
        self.register_buffer('beta', beta)
        self.num_population, self.retrieval = num_population, retrieval
        self.threshold, self.memory_strength = threshold, memory_strength
        self.temperature, self.read_mode = temperature, read_mode
        # Off/dense/sparse 모두 같은 파라미터와 초기값을 보존한다.
        self.query = nn.Linear(num_population, num_population, bias=False)
        self.key = nn.Linear(num_population, num_population, bias=False)
        self.gate = nn.Linear(2 * num_population + 2, 1)
        self.null_logit = nn.Parameter(torch.tensor(float(null_logit_init)))
        nn.init.eye_(self.query.weight)
        nn.init.eye_(self.key.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.spike = surrogate.Sigmoid(alpha=4.)

    def read_memory(self, charged, history, memory_mode=None, keys=None):
        """[BC,D,K], [BC,D,J,K] -> normalized read, support, and read-strength gate.

        All history passed here must already be in the past. This method never
        deletes history; forward appends every post-reset state independently.
        """
        if history.shape[-2] == 0:
            raise ValueError('Empty history is handled before calling read_memory')
        q = self.query(charged)
        k = self.key(history) if keys is None else keys
        # No L2 normalization: states that differ only in amplitude remain distinct.
        score = -(q.unsqueeze(-2) - k).square().mean(dim=-1) / self.temperature
        logits = torch.cat([score, self.null_logit.expand_as(score[..., :1])], dim=-1)
        probability = sparsemax(logits) if self.read_mode == 'sparse' else logits.softmax(dim=-1)
        real_probability = probability[..., :-1]
        real_mass = real_probability.sum(dim=-1, keepdim=True)
        weight = real_probability / real_mass.clamp_min(1e-12)
        if memory_mode == 'uniform':
            weight = torch.ones_like(weight) / history.shape[-2]
            real_mass = torch.ones_like(real_mass)
        elif memory_mode == 'recent':
            weight = torch.zeros_like(weight)
            weight[..., -1] = 1.
            real_mass = torch.ones_like(real_mass)
        elif memory_mode not in (None, 'off'):
            raise ValueError('Unknown memory intervention')
        memory = (weight.unsqueeze(-1) * history).sum(dim=-2)
        margin = score.max(dim=-1, keepdim=True).values - self.null_logit
        gate_input = torch.cat([charged, memory, margin, real_mass], dim=-1)
        gate = real_mass * self.gate(gate_input).sigmoid()
        if memory_mode == 'off':
            weight = torch.zeros_like(weight)
            memory, gate = torch.zeros_like(memory), torch.zeros_like(gate)
            real_mass = torch.zeros_like(real_mass)
        return memory, {'weight': weight, 'mask': weight > 0, 'gate': gate,
                        'real_mass': real_mass, 'score': score}

    def forward(self, x, return_aux=False, memory_mode=None):
        if x.ndim != 3 or x.shape[0] < 1:
            raise ValueError('Expected nonempty [T,BC,D] current')
        if memory_mode not in (None, 'off', 'uniform', 'recent'):
            raise ValueError('Unknown memory intervention')
        use_memory = self.retrieval and memory_mode != 'off'
        state = x.new_zeros(*x.shape[1:], self.num_population)
        states, keys, spikes = [], [], []
        weights_log, gate_log, charge_log, evidence_log, mass_log = [], [], [], [], []
        for t in range(x.shape[0]):
            # K constituents share current; each retains its own beta and membrane.
            charged = self.beta * state + (1. - self.beta) * x[t].unsqueeze(-1)
            evidence = torch.zeros_like(charged)
            gate = charged.new_zeros(*charged.shape[:-1], 1)
            real_mass = torch.zeros_like(gate)
            weight = charged.new_zeros(*charged.shape[:-1], t)
            if use_memory and t:
                history = torch.stack(states, dim=-2)
                memory, read = self.read_memory(charged, history, memory_mode, torch.stack(keys, dim=-2))
                weight, gate, real_mass = read['weight'], read['gate'], read['real_mass']
                evidence = self.memory_strength * gate * memory
            voltage = charged + evidence
            spike = self.spike(voltage - self.threshold)
            state = voltage - self.threshold * spike.detach()
            spikes.append(spike)
            if use_memory:
                states.append(state)
                keys.append(self.key(state))
            elif return_aux:
                states.append(state)
            if return_aux:
                weights_log.append(F.pad(weight.detach(), (0, x.shape[0] - t)))
                gate_log.append(gate.detach())
                charge_log.append(charged.detach())
                evidence_log.append(evidence.detach())
                mass_log.append(real_mass.detach())
        output = torch.stack(spikes)
        if return_aux:
            attention = torch.stack(weights_log)
            return output, {'membrane': torch.stack(states).detach(), 'spikes': output.detach(),
                            'attention': attention, 'mask': attention > 0,
                            'gate': torch.stack(gate_log), 'charge': torch.stack(charge_log),
                            'evidence': torch.stack(evidence_log), 'real_mass': torch.stack(mass_log)}
        return output


class Embedding(nn.Module):
    def __init__(self, patch_size, embed_dim, input_scale=2., **neuron_args):
        super().__init__()
        self.proj = nn.Linear(patch_size, embed_dim, bias=True)
        self.input_scale = input_scale
        self.lif = PopulationLIF(**neuron_args)

    def forward(self, x, return_aux=False, memory_mode=None):
        return self.lif(self.proj(x) * self.input_scale, return_aux, memory_mode)
