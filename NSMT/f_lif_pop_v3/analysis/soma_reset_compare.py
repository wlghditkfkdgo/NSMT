"""v3 layers.Soma vs the original spikeDE LIFNeuron + euler solver driven by the same soma input a_k
(decision 10-17: the soma resets by theta/tau_s as spikeDE does).
usage: python soma_reset_compare.py <spikeDE root> <f_lif_pop_v3/forecasting>   (snn_recall, CPU)"""
import sys, torch
torch.compile = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from spikeDE.neuron import LIFNeuron
from spikeDE.solver import euler_integrate_tuple
import layers
torch.set_default_dtype(torch.float64)
T, B, D, tau_s, th = 42, 64, 8, 2., 1.
a = 2. * torch.randn(T, B, D, generator=torch.Generator().manual_seed(0)) + .8      # soma input a_k
def v3(reset_scale):                       # layers.Soma with the reset theta*s scaled
    soma = layers.Soma(D, 1, tau_s, th, 5.); v = torch.zeros(B, D); out = []
    for k in range(T):
        charge = v + (a[k] - v) / tau_s
        s = layers.arctan_spike(charge - th, 5.)
        v = charge - reset_scale * th * s; out.append(s)
    return torch.stack(out)
neuron = LIFNeuron(tau=tau_s, threshold=th, surrogate_grad_scale=5.)
def rhs(t, y):
    k = int(round(float(t)))
    return neuron(y[0], a[k] if k < T else torch.zeros(B, D))
hist = euler_integrate_tuple(rhs, (torch.zeros(B, D), torch.zeros(B, D)), torch.arange(T + 1.), 1)
ref = torch.stack(hist[-1][:]).double()[:T] if len(hist[-1]) > T else torch.stack(hist[-1]).double()
print('spikeDE spikes', tuple(ref.shape), int(ref.sum()))
for scale, name in ((1., 'v3 Soma (reset theta)'), (1. / tau_s, 'v3 Soma with reset theta/tau_s')):
    s = v3(scale)
    print(f'{name:32s}: spikes {int(s.sum())}, equal to spikeDE LIF+Euler: {torch.equal(s, ref)}')
