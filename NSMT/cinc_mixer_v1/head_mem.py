"""H1 head: the head also reads the membrane potential of the last neuron layer (2026-11-03, exp/f-lif-pop-v3; PROJECT_LOG H1
pre-registration; user, 11-03: "모델의 가장 마지막 레이어 (head)가 우리가 학습한 정보를 잃게 하는 것 같아" -> "H1 을 우선 수행해보자").

The head input of every task is the last block's output spikes averaged over the T_h = 4 repetition steps (5 levels, mostly 0).
H1 keeps that head unchanged and ADDS a readout of the last neuron layer's charge:
    last neuron layer = the last block's MLP fc2 neuron (the last firing layer before the head in both models)
        spkMixer (spk_mixer.SpkMixer): blocks[-1].fc2.lif, a layers.NeoNeuron -> its own _capture path (layers.spikes_and_charge:
                                       the same rhs / integrate / emit calls as forward(), H - theta with the graph attached)
        myModel --hippo_ssa (Spikformer-style): data_block[-1].mlp.fc2.lif, spikingjelly MultiStepLIFNode (tau 2, threshold 1,
                                       reset 0, decay_input) -> forward hook recomputing H_t = v + (x_t - v) / tau, s = [H >= 1],
                                       v <- H (1 - s) from the neuron input; fail closed unless [H - 1 >= 0] equals its output spikes
    charge = H - theta, [T_h, B, N, D], pooled exactly as the spikes are (T_h mean; token mean for classification)
    mem_readout = Linear(no bias) built with zero weights without touching the global RNG streams (torch.random.fork_rng), so a run
        with H1 starts bit-identical to the run without it (weights, outputs, gradients, data order); the difference grows only
        through the charge term during training.
Classification: a forward hook on `head` adds mem_readout(token-mean charge) to head(token-mean spikes) (state_dict key `head`
kept, new key `mem_readout`). forecast_wrap.Forecaster / ad_wrap.Reconstructor drop the classification term and build their own
per-token mem_readout (they replace `head`).
Energy: mem_readout takes a real input -> MAC (syops counts it as such); reading the membrane itself is not counted.
"""
import torch
import torch.nn as nn

from layers import NeoNeuron

__all__ = ['ChargeTap', 'zero_linear', 'enable_head_mem', 'drop_classification_term']


def zero_linear(n_in, n_out):
    """nn.Linear(n_in, n_out, bias=False) with zero weights; the CPU RNG state is restored after construction (the CUDA streams
    are not used by a CPU-built Linear)."""
    with torch.random.fork_rng(devices=[]):
        lin = nn.Linear(n_in, n_out, bias=False)
    with torch.no_grad():
        lin.weight.zero_()
    return lin


class ChargeTap:
    """Keeps the charge H - theta [T_h, B, N, D] of the last neuron layer from the latest forward (graph attached)."""
    def __init__(self, model):
        from model import myModel
        from spk_mixer import SpkMixer
        self._sj = None
        if isinstance(model, SpkMixer):
            self.kind, lif = 'spkmix', model.blocks[-1].fc2.lif
            if not isinstance(lif, NeoNeuron):
                raise NotImplementedError(f'head_mem: the last neuron of this spkMixer is {type(lif).__name__}, not a NeoNeuron')
            lif._capture = True                                          # NeoNeuron.forward then stores _charge
            self.lif = lif
        elif isinstance(model, myModel) and model.hippo_ssa:
            self.kind, lif = 'ssa', model.data_block[-1].mlp.fc2.lif
            tau, vth, vr = float(lif.tau), float(lif.v_threshold), lif.v_reset
            if not (getattr(lif, 'decay_input', True) and vr is not None and float(vr) == 0. and vth == 1.):
                raise NotImplementedError('head_mem: expected a spikingjelly LIF with decay_input, v_reset 0, threshold 1')
            self.tau, self.vth, self.lif = tau, vth, lif
            lif.register_forward_hook(self._sj_hook)
        else:
            raise NotImplementedError('head_mem: spkMixer or myModel --hippo_ssa only')

    def _sj_hook(self, mod, inp, out):
        x = inp[0]
        v = torch.zeros_like(x[0])
        hs = []
        for t in range(x.shape[0]):
            h = v + (x[t] - v) / self.tau
            s = (h >= self.vth).to(h.dtype)
            hs.append(h - self.vth)
            v = h * (1. - s)
        c = torch.stack(hs)
        if not torch.equal((c >= 0).to(out.dtype), out.detach()):
            n = int(((c >= 0).to(out.dtype) != out.detach()).sum())
            raise RuntimeError(f'head_mem: recomputed charge disagrees with the spikingjelly spikes at {n} positions')
        self._sj = c

    @property
    def charge(self):
        c = self.lif._charge if self.kind == 'spkmix' else self._sj
        if c is None:
            raise RuntimeError('head_mem: no charge captured (forward not run?)')
        return c


def enable_head_mem(model):
    """Attach the tap and the classification membrane term to a freshly built model (before any checkpoint is loaded)."""
    if getattr(model, '_charge_tap', None) is not None:
        raise RuntimeError('head_mem already enabled')
    tap = ChargeTap(model)
    model._charge_tap = tap
    head = model.head
    if not isinstance(head, nn.Linear):
        raise NotImplementedError(f'head_mem: head is {type(head).__name__}')
    model.mem_readout = zero_linear(head.in_features, head.out_features)

    def add_mem(mod, inp, out):                                          # head(x.mean(2)): x.mean(2) is [T_h, B, D]
        return out + model.mem_readout(tap.charge.mean(2))
    model._mem_hook = head.register_forward_hook(add_mem)
    return model


def drop_classification_term(inner):
    """For the forecasting / reconstruction wrappers, which replace inner.head: remove the classification membrane term."""
    if getattr(inner, '_charge_tap', None) is None:
        return None
    inner._mem_hook.remove()
    del inner.mem_readout
    return inner._charge_tap
