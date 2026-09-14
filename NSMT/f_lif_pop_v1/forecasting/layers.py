"""Population membrane memory. T is chronological patch time, never repetition."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate


class PopulationLIF(nn.Module):
    def __init__(self, num_population=4, heterogeneous=True, retrieval=True,
                 tau_min=2., tau_max=16., threshold=1., memory_strength=0.05,
                 temperature=0.25):
        super().__init__()
        if num_population < 2 or not 1 < tau_min <= tau_max:
            raise ValueError('Require K >= 2 and 1 < tau_min <= tau_max')
        if threshold <= 0 or temperature <= 0:
            raise ValueError('Threshold and temperature must be positive')
        tau = torch.logspace(math.log10(tau_min), math.log10(tau_max), num_population)
        beta = torch.exp(-1. / tau)
        # 균일 대조군: 집단의 평균 leak factor를 맞춘다.
        if not heterogeneous:
            beta = beta.mean().repeat(num_population)
        if not 0 <= memory_strength < 1 - beta.max().item():
            raise ValueError('Use memory_strength < 1 - max(beta) for this fixed-tau prototype')
        self.register_buffer('beta', beta)
        self.num_population = num_population
        self.retrieval = retrieval
        self.threshold = threshold
        self.memory_strength = memory_strength
        self.temperature = temperature
        # Q/K와 gate는 logical neuron 사이에 공유하며, K축만 처리한다.
        # Retrieval off에도 모듈을 보존하여 초기 파라미터/개수를 맞춘다.
        self.query = nn.Linear(num_population, num_population, bias=False)
        self.key = nn.Linear(num_population, num_population, bias=False)
        self.gate = nn.Linear(num_population, 1)
        nn.init.eye_(self.query.weight)
        nn.init.eye_(self.key.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.spike = surrogate.Sigmoid(alpha=4.)

    def forward(self, x, return_aux=False, memory_mode=None):
        """x: [T, BC, D] current -> spikes: [T, BC, D, K].

        Store post-reset membranes. Full BPTT through memory; detach reset only.
        State/history belong to this forward (independent windows cannot leak).
        memory_mode is an evaluation intervention: off/uniform/recent.
        """
        if x.ndim != 3 or x.shape[0] < 1:
            raise ValueError('Expected nonempty [T, BC, D] input')
        if memory_mode not in (None, 'off', 'uniform', 'recent'):
            raise ValueError('Unknown memory intervention')
        use_memory = self.retrieval and memory_mode != 'off'
        u = x.new_zeros(*x.shape[1:], self.num_population)
        states, keys, spikes = [], [], []
        weights_log, gate_log, charge_log, evidence_log = [], [], [], []
        for t in range(x.shape[0]):
            # 1. 현재 patch를 충전한다. 같은 logical neuron의 K개 LIF는 입력 공유.
            u_bar = self.beta * u + (1. - self.beta) * x[t].unsqueeze(-1)
            evidence = torch.zeros_like(u_bar)
            gate = u_bar.new_zeros(*u_bar.shape[:-1], 1)
            weight = u_bar.new_zeros(*u_bar.shape[:-1], t)
            if use_memory and t > 0:
                # 2. 현재 query로 j<t의 post-reset population 상태를 검색한다.
                history = torch.stack(states, dim=-2)       # [BC,D,t,K]
                q = F.normalize(self.query(u_bar), dim=-1, eps=1e-6)
                k = torch.stack(keys, dim=-2)
                score = (q.unsqueeze(-2) * k).sum(-1) / self.temperature
                weight = score.softmax(dim=-1)              # [BC,D,t], K개가 공유
                if memory_mode == 'uniform':
                    weight = torch.ones_like(weight) / t
                elif memory_mode == 'recent':
                    weight = torch.zeros_like(weight)
                    weight[..., -1] = 1.
                memory = (weight.unsqueeze(-1) * history).sum(-2)
                # 3. 증거 보강: 현재 상태 계수는 1, 기억을 가산한다.
                gate = self.gate(u_bar).sigmoid()
                evidence = self.memory_strength * gate * memory
            v = u_bar + evidence
            # 4. 발화 후 subtractive reset. 이미 발화한 증거의 잔여 상태를 저장.
            s = self.spike(v - self.threshold)
            u = v - self.threshold * s.detach()
            spikes.append(s)
            if use_memory:
                states.append(u)
                keys.append(F.normalize(self.key(u), dim=-1, eps=1e-6))
            elif return_aux:
                states.append(u)
            if return_aux:
                weights_log.append(F.pad(weight.detach(), (0, x.shape[0] - t)))
                gate_log.append(gate.detach())
                charge_log.append(u_bar.detach())
                evidence_log.append(evidence.detach())
        out = torch.stack(spikes)
        if return_aux:
            return out, {'membrane': torch.stack(states).detach(),
                         'spikes': out.detach(), 'attention': torch.stack(weights_log),
                         'gate': torch.stack(gate_log), 'charge': torch.stack(charge_log),
                         'evidence': torch.stack(evidence_log)}
        return out


class Embedding(nn.Module):
    def __init__(self, patch_size, embed_dim, input_scale=2., **neuron_args):
        super().__init__()
        self.proj = nn.Linear(patch_size, embed_dim, bias=True)
        self.input_scale = input_scale
        # Train-only input scaling is supplied by Dataset_ETT_hour.
        # No time/window normalization: keep amplitude and causal patch states.
        self.lif = PopulationLIF(**neuron_args)

    def forward(self, x, return_aux=False, memory_mode=None):
        return self.lif(self.proj(x) * self.input_scale, return_aux, memory_mode)
