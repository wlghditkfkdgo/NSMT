"""Neuron-intrinsic energy accounting (nsmt_flif_v1).

syops counts only the synaptic side (nn.Linear / BatchNorm hooks: AC if the input is 0/1, else MAC;
E_AC = 0.9 pJ, E_MAC = 4.6 pJ, 45 nm). The neurons' own updates are not counted. We add them the way
the f-LIF paper does (Ge et al., ICLR 2026, Appendix D.3, eqs. 55-67), with kappa = E_MAC / E_AC:

    LIF   (eq. 60)   E = T * [(2 + 2 rho) kappa + (2 + rho)] * E_AC            per neuron, T steps
    f-LIF (eq. 64)   E = T * [(kappa + 1) log2 T + 1 + rho (2 kappa + 1)] * E_AC  (FFT convolution)
    total (eq. 67)   E_total = E_synaptic + sum over neurons

`e_flif_direct` is our addition: the neurons.py f-LIF re-sums the stored increments directly (step t
costs t multiply-accumulates; T(T-1)/2 for the full history, 276 at T = 24), so we report that
convention next to the paper's FFT one. The LIF-like per-step terms of eq. 60 are kept in it.

FiringRateRecorder hooks every spiking neuron module (spikingjelly MultiStepLIFNode and the
neurons.py LIFNeuron / FLIFNeuron; not the NeoNeuron adapter, to avoid double counting) and turns
the recorded firing rates into per-layer and per-model energies per input window.
nsmt_flif_v1/classification: the ECG model's NeoLayer calls a single-step spikingjelly LIFNode once
per token; those calls are recorded too (T = calls per forward), so the ECG baseline's time_block
neurons are counted like the spikeDE neurons that replace them.
"""
import csv
import math

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, LIFNode

from neurons import LIFNeuron, FLIFNeuron

__all__ = ['E_AC', 'E_MAC', 'KAPPA', 'e_lif', 'e_flif_fft', 'e_flif_direct', 'FiringRateRecorder', 'COLUMNS']

E_AC, E_MAC = 0.9e-12, 4.6e-12
KAPPA = E_MAC / E_AC                                                 # 5.111...
COLUMNS = ('neuron_lif_uJ', 'neuron_model_fft_uJ', 'neuron_model_direct_uJ', 'total_fft_uJ',
           'total_direct_uJ', 'rho_neo', 'rho_hippo', 'n_spiking_layers', 'kappa')
NEO_PREFIXES = ('encoding_neo', 'time_block', 'blocks.0.mixer', 'blocks.1.mixer')   # also *_rev (stage 6); stage 7 mixer stacks


def e_lif(T, rho):
    """Eq. 60, in units of E_AC: one hard-reset LIF neuron over T steps at firing rate rho."""
    return T * ((2. + 2. * rho) * KAPPA + (2. + rho))


def e_flif_fft(T, rho):
    """Eq. 64, in units of E_AC: one f-LIF neuron over T steps, history by FFT convolution."""
    return T * ((KAPPA + 1.) * math.log2(T) + 1. + rho * (2. * KAPPA + 1.))


def e_flif_direct(T, rho, memory=None):
    """Our direct-sum convention, in units of E_AC: step t re-sums min(t, W) stored increments
    (W = memory - 1 for a short-memory window, T - 1 for the full history), one MAC each, on top of
    the LIF-like per-step terms of eq. 60."""
    W = T - 1 if not memory else min(int(memory) - 1, T - 1)
    hist = sum(min(t, W) for t in range(T))
    return hist * (KAPPA + 1.) + e_lif(T, rho)


def _kind(module):
    if isinstance(module, FLIFNeuron):
        if getattr(module, 'num_members', 1) > 1 and getattr(module, 'has_soma', False):
            return 'pop_flif' if module.alpha < 1. else 'pop_lif'      # MODI 2026-10-21: alpha 1 = Euler LIF branches, no history
        return 'flif' if module.alpha < 1. else 'lif'                # alpha = 1 is the Euler LIF
    return 'lif'


def e_pop_flif(T, rho, K, memory=None, direct=False):
    """Population f-LIF (stage 6; our convention, not in the f-LIF paper): K non-resetting fractional branches, each
    paying the history term of eq. 64 (FFT) or the direct sum, plus the per-step LIF-like terms without reset
    (rho-independent part of eq. 60), plus K MACs per step for the learned soma mix, plus one LIF soma (eq. 60)."""
    if direct:
        W = T - 1 if not memory else min(int(memory) - 1, T - 1)
        hist = sum(min(t, W) for t in range(T)) * (KAPPA + 1.)
    else:
        hist = T * ((KAPPA + 1.) * math.log2(T) + 1.)
    branch = hist + T * (2. * KAPPA + 2.)
    return K * branch + T * K * KAPPA + e_lif(T, rho)


class FiringRateRecorder:
    """Forward hooks on every spiking neuron module; call add_samples(batch) once per forward."""
    TYPES = (MultiStepLIFNode, LIFNeuron, FLIFNeuron, LIFNode)

    def __init__(self, model):
        self.stats, self.handles, self.n_samples, self.n_forward = {}, [], 0, 0
        for name, m in model.named_modules():
            if isinstance(m, self.TYPES):
                single = type(m) is LIFNode                          # one call per time step (ECG NeoLayer)
                self.stats[name] = dict(kind=_kind(m), path='neo' if name.startswith(NEO_PREFIXES) else 'hippo',
                                        T=0, spikes=0., elems=0, calls=0, rate_sum=0., single=single,
                                        memory=getattr(m, 'memory', None), alpha=getattr(m, 'alpha', 1.),
                                        members=int(getattr(m, 'num_members', 1)))
                self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o = o.detach()
            st = self.stats[name]
            st['T'] = int(o.shape[0])
            st['spikes'] += float(o.sum())
            st['elems'] += int(o.numel())
            st['calls'] += 1
            st['rate_sum'] += float(o.sum()) / max(1, o.numel())     # syops' per-call spike_rate for 0/1 outputs
        return fn

    def add_samples(self, n):
        """Call once per forward pass with the batch size."""
        self.n_samples += int(n)
        self.n_forward += 1

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def layers(self):
        """Per-layer rows: energies in J per input window."""
        rows = []
        for name, st in self.stats.items():
            if st['calls'] == 0 or self.n_samples == 0:
                continue
            if st['single']:                                         # steps per forward = calls / forwards
                if st['calls'] % max(1, self.n_forward):
                    raise RuntimeError(f"{name}: {st['calls']} single-step calls over {self.n_forward} forwards")
                st['T'] = st['calls'] // max(1, self.n_forward)
            T, rho = st['T'], st['spikes'] / max(1, st['elems'])
            per_sample = st['elems'] / (T * self.n_samples)          # neurons per window
            lif, fft, direct = e_lif(T, rho), e_lif(T, rho), e_lif(T, rho)
            if st['kind'] == 'flif':
                fft, direct = e_flif_fft(T, rho), e_flif_direct(T, rho, st['memory'])
            elif st['kind'] == 'pop_flif':
                K = st['members']
                fft, direct = e_pop_flif(T, rho, K, st['memory']), e_pop_flif(T, rho, K, st['memory'], direct=True)
            elif st['kind'] == 'pop_lif':                              # MODI 2026-10-21: K Euler branches (no history term) + mix + soma
                K = st['members']
                fft = direct = K * T * (2. * KAPPA + 2.) + T * K * KAPPA + e_lif(T, rho)
            rows.append(dict(module=name, kind=st['kind'], path=st['path'], T=T, neurons_per_sample=per_sample,
                             rho=rho, rho_syops=st['rate_sum'] / st['calls'], alpha=st['alpha'],
                             E_lif_J=per_sample * lif * E_AC, E_fft_J=per_sample * fft * E_AC,
                             E_direct_J=per_sample * direct * E_AC))
        return rows

    def summary(self, syn_energy_J):
        rows = self.layers()
        lif = sum(r['E_lif_J'] for r in rows); fft = sum(r['E_fft_J'] for r in rows); direct = sum(r['E_direct_J'] for r in rows)

        def mean_rho(path):
            sel = [r for r in rows if r['path'] == path]
            return None if not sel else sum(r['rho'] * r['neurons_per_sample'] * r['T'] for r in sel) / sum(r['neurons_per_sample'] * r['T'] for r in sel)
        values = {'neuron_lif_uJ': f'{lif * 1e6:.6f}', 'neuron_model_fft_uJ': f'{fft * 1e6:.6f}',
                  'neuron_model_direct_uJ': f'{direct * 1e6:.6f}', 'total_fft_uJ': f'{(syn_energy_J + fft) * 1e6:.6f}',
                  'total_direct_uJ': f'{(syn_energy_J + direct) * 1e6:.6f}',
                  'rho_neo': 'None' if mean_rho('neo') is None else f"{mean_rho('neo'):.6f}",
                  'rho_hippo': 'None' if mean_rho('hippo') is None else f"{mean_rho('hippo'):.6f}",
                  'n_spiking_layers': str(len(rows)), 'kappa': f'{KAPPA:.6f}'}
        return {'columns': list(COLUMNS), 'values': [values[c] for c in COLUMNS], 'rows': rows}

    def write_layers(self, path):
        rows = self.layers()
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['module'])
            w.writeheader()
            for r in rows:
                w.writerow(r)
