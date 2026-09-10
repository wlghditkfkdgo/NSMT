"""Population-coded, channel-independent Spikformer prototype (no replay).

L denotes the original length; T=L for raw data, T=patch count otherwise.
There is NO independent simulation/repetition axis: every LIF runs along T.

    raw:   [B,L,C] -> [BC,L] -> Gaussian [BC,L,K] -> [L,BC,K,1]
    patch: [B,L,C] -> [BC,T,P] -> Gaussian [BC,T,K,P] -> [T,BC,K,P]
    both:  direct current / Bernoulli -> Linear(1 or P,D)-BN-LIF
           -> [T,BC,K,D] -> IAND(SSA), IAND(MLP) -> head -> [B,pred_len,C]
Default experiment: patch + direct, temporal SSA, scale=1, two-stage head.
Temporal SSA forms [BC,K,heads,T,T] ONLY after Q/K/V LIF along T, and
restores [T,BC,K,D] before the attention/output LIFs. Population SSA and
no-SSA controls are selectable. All observed patches can attend each other.
Optional learned [K,D] population identity is added AFTER embedding BN and
BEFORE LIF, initialized N(0,0.01); Gaussian receptive fields remain fixed.

Raw uses a shared 1->D projection PER population neuron. A K->D projection
would remove the K token axis and cannot produce K-by-K attention.
Gaussian coding is a continuous tuning response, not itself a spike code:
    r_k(z) = exp(-0.5 * ((clip(z, low, high) - mu_k) / sigma)**2).
Centers are evenly spaced; sigma = width * center spacing (fixed buffers).
With normalization enabled, low/high/sigma are in input-window std units.
Clipping saturates out-of-range amplitudes. No target statistics are used.

direct: feed r into the embedding; its LIF produces the first binary spikes.
rate: one Bernoulli(r) draw per observed time/patch and neuron, also in eval.
      No repeated observation window, no empirical per-value rate estimate,
      and no gradient through the fixed stochastic encoder. Use manual_seed
      for repeatable draws; evaluation averaging is the caller's decision.

Adapted from model_v1/forecasting/ours.py (Embedding, Block) and layers.py
(SpikLinearLayer, MLP, SSA_rel_scl). Keeps Linear-BN-LIF, softmax-free SSA,
and residual x*(1-branch); attn_scale=None restores reference dim**-0.5.
Removes Neocortex/replay/aux loss
and temporal repetition. Minimal components are local to avoid the existing
modules' CUDA-only backend and unrelated imports. Default backend is torch;
cupy is opt-in on CUDA. Uses PyTorch's default Linear initialization.

BN aggregates T, BC and K in training, as in the reference. Combined with
input-window normalization, this is an offline-window forecaster, NOT a
strictly causal streaming encoder. State resets at EVERY model forward.
Population SSA mixes populations at each time; temporal SSA mixes times
within each population. IAND can only suppress existing spikes. No temporal
positional bias is added. K identity remains accessible to the ordered head.
Shared raw 1->D embedding gives SSA no explicit center identity: equal tuning
responses have equal initial features. Center/learned population embeddings
would be a separate experiment. At seed 7 with default K=16,D=64,H=8, initial
SSA output was silent at the reference scale 0.125; attn_scale=1.0 produced
spikes. Inspect aux['ssa_spikes']; neither scale has forecasting validation.

readout='two_stage': shared Linear(K*D,head_dim) per T, then flatten T and
Linear(T*head_dim,pred_len). No activation/LIF in this continuous bottleneck.
readout='flatten' preserves T,K,D (head parameters scale with T*K*D*pred_len).
'mean' averages T, 'last' selects the final T; both preserve K,D. None return
the legacy Hippo auxiliary tuple: this standalone file is not trainer-wired.

References:
  Population tuning: https://doi.org/10.1073/pnas.2305853120
  Spikformer SSA: https://arxiv.org/abs/2209.15425
  SEW IAND: https://proceedings.neurips.cc/paper/2021/hash/afe434653a898da20044041262b3ac74-Abstract.html

Example (from NSMT):
    from forecasting.simple_test_model_v1 import SimpleTestModelV1
    model = SimpleTestModelV1(seq_len=96, pred_len=24, num_population=16)
    prediction, aux = model(torch.randn(2, 96, 7), return_aux=True)
    # aux['attention'][0]: [14, 16, num_heads, 12, 12], scaled counts,
    # NOT probabilities or binary spikes. Aux tensors are detached.

Run this file with --smoke-test for synthetic forward/backward checks only.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.clock_driven import functional as sj_functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate

__all__ = ["SimpleTestModelV1", "Model"]


class GaussianPopulationCoding(nn.Module):
    """[BC,T,P] -> [BC,T,K,P]; each scalar has K overlapping receptive fields."""

    def __init__(self, num_population=16, low=-3.0, high=3.0, width=1.0):
        super().__init__()
        if num_population < 2:
            raise ValueError("num_population must be at least 2")
        if not all(math.isfinite(v) for v in (low, high, width)) or not low < high or width <= 0:
            raise ValueError("Require finite low < high and width > 0")
        self.register_buffer("centers", torch.linspace(low, high, num_population))
        self.register_buffer("sigma", torch.tensor(width * (high - low) / (num_population - 1)))

    def forward(self, x):
        x = x.clamp(self.centers[0], self.centers[-1])
        distance = (x.unsqueeze(-2) - self.centers.view(1, 1, -1, 1)) / self.sigma
        return torch.exp(-0.5 * distance.square())


def _lif(tau, threshold, backend):
    return MultiStepLIFNode(
        tau=float(tau), v_threshold=float(threshold), detach_reset=True,
        surrogate_function=surrogate.Sigmoid(), backend=backend,
    )


class SpikingLinear(nn.Module):
    """Reference Linear-BN-LIF; only the last axis is projected."""

    def __init__(self, in_dim, out_dim, tau, threshold, backend, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.bn = nn.BatchNorm1d(out_dim)
        self.lif = _lif(tau, threshold, backend)

    def forward(self, x, population_embedding=None):
        y = self.linear(x)
        y = self.bn(y.reshape(-1, y.shape[-1])).reshape(y.shape)
        if population_embedding is not None:
            y = y + population_embedding  # [K,D], shared over N and BC; before LIF.
        return self.lif(y.contiguous())  # [T, BC, K, D]; T is always time.


class PopulationSSA(nn.Module):
    def __init__(self, d_model, num_heads, tau, threshold, backend, bias, attn_scale, axis="population"):
        super().__init__()
        self.axis = axis
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Preserve the existing Hippo/SSA scale; this is not a softmax temperature.
        self.scale = d_model ** -0.5 if attn_scale is None else attn_scale
        self.q = SpikingLinear(d_model, d_model, tau, threshold, backend, bias)
        self.k = SpikingLinear(d_model, d_model, tau, threshold, backend, bias)
        self.v = SpikingLinear(d_model, d_model, tau, threshold, backend, bias)
        self.attn_lif = _lif(tau, threshold, backend)
        self.proj = SpikingLinear(d_model, d_model, tau, threshold, backend, bias)

    def forward(self, x, return_attention=False):
        t, bc, k_tokens, d_model = x.shape

        def split_heads(z):
            return z.reshape(t, bc, k_tokens, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        q, key, value = (split_heads(layer(x)) for layer in (self.q, self.k, self.v))
        if self.axis == "temporal":
            # Q/K/V LIFs have ALREADY run along original time, above.
            # Reorder only the matmul; never run LIF with K or BC as time.
            q, key, value = (s.permute(1, 3, 2, 0, 4) for s in (q, key, value))
            scores = (q @ key.transpose(-2, -1)) * self.scale  # [BC,K,H,T,T]
            z = (scores @ value).permute(3, 0, 1, 2, 4).reshape(t, bc, k_tokens, d_model)
        else:
            scores = (q @ key.transpose(-2, -1)) * self.scale  # [T,BC,H,K,K]
            z = (scores @ value).transpose(2, 3).reshape(t, bc, k_tokens, d_model)
        z = self.proj(self.attn_lif(z.contiguous()))
        return z, scores.detach() if return_attention else None


class IANDBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio, tau, threshold, backend, bias, attn_scale, attention_axis="population"):
        super().__init__()
        self.attn = None if attention_axis == "none" else PopulationSSA(
            d_model, num_heads, tau, threshold, backend, bias, attn_scale, attention_axis,
        )
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            SpikingLinear(d_model, hidden, tau, threshold, backend, bias),
            SpikingLinear(hidden, d_model, tau, threshold, backend, bias),
        )

    def forward(self, x, return_attention=False):
        branch, attention = None, None
        if self.attn is not None:
            branch, attention = self.attn(x, return_attention)
            x = x * (1.0 - branch)
        x = x * (1.0 - self.mlp(x))
        return x, attention, branch.detach() if return_attention and branch is not None else None


class SimpleTestModelV1(nn.Module):
    def __init__(
        self, seq_len=96, pred_len=96, num_population=16, d_model=64,
        num_heads=8, depth=2, mlp_ratio=2.0, input_mode="patch", patch_size=8,
        stride=None, encoding="direct", population_low=-3.0, population_high=3.0,
        population_width=1.0, normalize=True, readout="two_stage", tau=2.0,
        threshold=1.0, backend="torch", bias=False, attn_scale=1.0,
        attention_axis="temporal", learn_population_embedding=False, head_dim=64,
    ):
        super().__init__()
        if min(seq_len, pred_len, d_model, num_heads, depth) < 1:
            raise ValueError("Lengths, dimensions, heads and depth must be positive")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads (K need not be)")
        if not math.isfinite(mlp_ratio) or int(d_model * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce at least one hidden feature")
        if not math.isfinite(tau) or not math.isfinite(threshold) or tau <= 1 or threshold <= 0:
            raise ValueError("Require finite tau > 1 and threshold > 0")
        if attn_scale is not None and (not math.isfinite(attn_scale) or attn_scale <= 0):
            raise ValueError("attn_scale must be finite and positive")
        if input_mode not in ("raw", "patch") or encoding not in ("direct", "rate"):
            raise ValueError("Use input_mode='raw'/'patch' and encoding='direct'/'rate'")
        if readout not in ("two_stage", "flatten", "mean", "last") or backend not in ("torch", "cupy"):
            raise ValueError("Invalid readout or backend")
        if attention_axis not in ("population", "temporal", "none") or head_dim < 1:
            raise ValueError("Use attention_axis='population'/'temporal'/'none' and positive head_dim")
        self.seq_len, self.pred_len = seq_len, pred_len
        self.input_mode, self.encoding = input_mode, encoding
        self.normalize, self.readout, self.backend = normalize, readout, backend
        self.attention_axis = attention_axis
        self.patch_size = 1 if input_mode == "raw" else patch_size
        # Non-overlapping by default; an explicit smaller stride enables overlap.
        self.stride = 1 if input_mode == "raw" else (patch_size if stride is None else stride)
        if not 1 <= self.stride <= self.patch_size:
            raise ValueError("Require 1 <= stride <= patch_size to avoid omitted observations")
        self.num_steps = 1 + max(0, math.ceil((seq_len - self.patch_size) / self.stride))
        self.pad_right = (self.num_steps - 1) * self.stride + self.patch_size - seq_len
        self.population = GaussianPopulationCoding(
            num_population, population_low, population_high, population_width,
        )
        self.embedding = SpikingLinear(self.patch_size, d_model, tau, threshold, backend, bias)
        self.blocks = nn.ModuleList([
            IANDBlock(d_model, num_heads, mlp_ratio, tau, threshold, backend, bias, attn_scale, attention_axis)
            for _ in range(depth)
        ])
        self.head_compress = nn.Linear(num_population * d_model, head_dim, bias=bias) if readout == "two_stage" else None
        head_in = self.num_steps * head_dim if readout == "two_stage" else (
            (self.num_steps if readout == "flatten" else 1) * num_population * d_model
        )
        self.head = nn.Linear(head_in, pred_len, bias=bias)
        # Initialize AFTER shared layers, so E=0 leaves paired model weights identical.
        self.population_embedding = nn.Parameter(torch.empty(num_population, d_model)) if learn_population_embedding else None
        if self.population_embedding is not None:
            nn.init.normal_(self.population_embedding, std=0.01)

    def reset_state(self):
        """Clear all membrane state; independent windows never share state."""
        sj_functional.reset_net(self)

    def forward(self, x, return_aux=False):
        if x.ndim != 3 or x.shape[1] != self.seq_len or min(x.shape) < 1:
            raise ValueError(f"Expected nonempty [B, {self.seq_len}, C], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Input must be floating point")
        if self.backend == "cupy" and not x.is_cuda:
            raise ValueError("backend='cupy' requires model and input on CUDA")
        self.reset_state()
        b, length, c = x.shape
        if self.normalize:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
            x = (x - mean) / std
        else:
            mean, std = 0.0, 1.0
        series = x.transpose(1, 2).reshape(b * c, length)
        # Repeat the final observation to retain an incomplete last patch.
        # Replicated values are features of that patch, not extra simulation steps.
        if self.pad_right:
            series = F.pad(series.unsqueeze(1), (0, self.pad_right), mode="replicate").squeeze(1)
        patches = series.unfold(-1, self.patch_size, self.stride)  # [BC,T,P]
        population = self.population(patches)  # [BC,T,K,P]
        encoded = torch.bernoulli(population) if self.encoding == "rate" else population
        z = self.embedding(encoded.permute(1, 0, 2, 3).contiguous(), self.population_embedding)
        aux = {}
        if return_aux:
            aux = {
                "population": (population.squeeze(-1) if self.input_mode == "raw" else population).detach(),
                "encoded": (encoded.squeeze(-1) if self.input_mode == "raw" else encoded).detach(),
                "embedding_spikes": z.detach(), "attention": [], "ssa_spikes": [], "block_spikes": [],
                "num_steps": self.num_steps, "pad_right": self.pad_right,
            }
        for block in self.blocks:
            z, attention, ssa_spikes = block(z, return_aux)
            if return_aux:
                aux["attention"].append(attention)
                aux["ssa_spikes"].append(ssa_spikes)
                aux["block_spikes"].append(z.detach())
        if self.readout == "two_stage":
            features = self.head_compress(z.flatten(2)).transpose(0, 1).reshape(b * c, -1)
        elif self.readout == "flatten":
            features = z.permute(1, 0, 2, 3).reshape(b * c, -1)
        elif self.readout == "mean":
            features = z.mean(dim=0).reshape(b * c, -1)
        else:
            features = z[-1].reshape(b * c, -1)
        prediction = self.head(features).reshape(b, c, self.pred_len).transpose(1, 2)
        prediction = prediction * std + mean
        return (prediction, aux) if return_aux else prediction


Model = SimpleTestModelV1


def _smoke_test(device="cpu", backend="torch"):
    """Synthetic contract checks, including two optimizer steps; no dataset run."""
    torch.set_num_threads(2)
    reports = []
    for mode in ("raw", "patch"):
        for encoding in ("direct", "rate"):
            for readout in ("flatten", "mean", "last"):
                torch.manual_seed(7)
                model = SimpleTestModelV1(
                    seq_len=13, pred_len=5, num_population=7, d_model=16,
                    num_heads=4, depth=2, patch_size=4, input_mode=mode,
                    encoding=encoding, readout=readout, backend=backend,
                    attention_axis="population", attn_scale=None,
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                x = torch.randn(2, 13, 3, device=device)
                target = torch.randn(2, 5, 3, device=device)
                steps = 13 if mode == "raw" else 4
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    prediction, aux = model(x, return_aux=True)
                    assert prediction.shape == target.shape and torch.isfinite(prediction).all()
                    assert aux["embedding_spikes"].shape == (steps, 6, 7, 16)
                    assert aux["population"].shape == ((6, 13, 7) if mode == "raw" else (6, 4, 7, 4))
                    previous = aux["embedding_spikes"]
                    assert ((previous == 0) | (previous == 1)).all() and previous.any()
                    for spikes, attention in zip(aux["block_spikes"], aux["attention"]):
                        assert ((spikes == 0) | (spikes == 1)).all()
                        assert (spikes <= previous).all()  # IAND only suppresses.
                        assert attention.shape == (steps, 6, 4, 7, 7)
                        assert torch.isfinite(attention).all() and (attention >= 0).all()
                        previous = spikes
                    if encoding == "rate":
                        assert ((aux["encoded"] == 0) | (aux["encoded"] == 1)).all()
                    loss = F.mse_loss(prediction, target)
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                    assert model.embedding.linear.weight.grad.abs().sum() > 0
                    optimizer.step()
                with torch.no_grad():
                    # Training BN keeps this check on active spikes, even before
                    # running statistics have converged for evaluation.
                    torch.manual_seed(23)
                    expected, first = model(x, return_aux=True)
                    assert first["embedding_spikes"].any()
                    model(torch.randn(1, 13, 2, device=device))  # Different BC, unrelated window.
                    torch.manual_seed(23)
                    actual, second = model(x, return_aux=True)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(first["embedding_spikes"], second["embedding_spikes"], rtol=0, atol=0)
                    model.eval()
                    assert torch.isfinite(model(torch.ones(1, 13, 1, device=device))).all()
                    if encoding == "direct":
                        expected = model(x)
                        permuted = model(x[:, :, [2, 0, 1]])
                        torch.testing.assert_close(permuted, expected[:, :, [2, 0, 1]])
                reports.append({"mode": mode, "encoding": encoding, "readout": readout,
                                "steps": steps, "parameters": sum(p.numel() for p in model.parameters()),
                                "synthetic_loss_after_second_forward": float(loss.detach()),
                                "embedding_firing_rate": float(aux["embedding_spikes"].mean()),
                                "ssa_firing_rates": [float(s.mean()) for s in aux["ssa_spikes"]],
                                "last_block_firing_rate": float(aux["block_spikes"][-1].mean())})
    # Check receptive-field identity and endpoint saturation with known values.
    coder = GaussianPopulationCoding(3, -1.0, 1.0, 1.0).to(device)
    responses = coder(torch.tensor([[[-1.0, 0.0, 1.0]]], device=device))[0, 0]
    torch.testing.assert_close(responses.diagonal(), torch.ones(3, device=device))
    assert torch.equal(responses.argmax(dim=0), torch.arange(3, device=device))
    clipped = coder(torch.tensor([[[-5.0, 5.0]]], device=device))
    torch.testing.assert_close(clipped[0, 0], responses[:, [0, 2]])
    # LIF must integrate successive observations, and not treat BC as time.
    lif = _lif(2.0, 1.0, backend).to(device)
    current = torch.zeros(3, 2, 1, 1, device=device)
    current[:, 0] = 1.5
    spikes = lif(current)
    torch.testing.assert_close(spikes[:, 0, 0, 0], torch.tensor([0., 1., 0.], device=device))
    assert not spikes[:, 1].any()
    # Patch coverage, including overlap and seq_len < patch_size.
    for length, patch_size, stride, count, pad in ((13, 4, 3, 4, 0), (3, 8, 8, 1, 5)):
        edge = SimpleTestModelV1(seq_len=length, pred_len=2, input_mode="patch", patch_size=patch_size,
                                 stride=stride, d_model=8, num_heads=2, depth=1, backend=backend).to(device)
        output = edge(torch.randn(1, length, 1, device=device))
        assert (edge.num_steps, edge.pad_right) == (count, pad) and torch.isfinite(output).all()
    return reports


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("torch", "cupy"), default="torch")
    args = parser.parse_args()
    if not args.smoke_test:
        parser.error("Import SimpleTestModelV1 or pass --smoke-test")
    print(json.dumps({"device": args.device, "backend": args.backend,
                      "checks": _smoke_test(args.device, args.backend), "status": "passed"}, indent=2))
