"""CPU reference audit, no installation/training. Args: spikeDE root, NSMT snapshot, outdir.

Execute only against the reviewed upstream commit fcd743befe504b1a471fa81887e6af7d6789da2e.
Uses upstream eager public solver and neuron without patching them. This is a scalar
reference check, not compiled/full-wrapper, GPU, training, or v3 architecture parity.
"""
import contextlib
import csv
import hashlib
import io
import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

ref, nsmt, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ref))
sys.path.insert(0, str(nsmt / 'f_lif_pop_v3/forecasting'))
from spikeDE.neuron import LIFNeuron
from spikeDE.solver import pred_integrate_tuple
from spikeDE.surrogate import arctan_surrogate
import layers

torch.set_num_threads(2)
torch.manual_seed(7)
np_rng = np.random.default_rng(7)
rows, traces = [], []


def upstream(current, alpha, tau, threshold=1.):
    neuron = LIFNeuron(tau=tau, threshold=threshold).double()
    state, spikes = [], []

    def rhs(t, values):
        n = int(t.item())
        state.append(values[0].detach().clone())
        drive = values[0].new_tensor([current[n] if n < len(current) else 0.])
        deriv, spike = neuron(values[0], drive)
        spikes.append(spike.detach().clone())
        return deriv, spike

    # One extra RHS observation reads U_T; its derivative cannot affect earlier states.
    with torch.no_grad():
        pred_integrate_tuple(rhs, (torch.zeros(1, dtype=torch.float64),
                                  torch.zeros(1, dtype=torch.float64)), [alpha],
                             torch.arange(len(current) + 2, dtype=torch.float64))
    return torch.stack(state[1:]).flatten().numpy(), torch.stack(spikes[:-1]).flatten().numpy()


def local_definition(current, alpha, tau):
    u = np.zeros(len(current) + 1)
    f = np.zeros(len(current))
    s = np.zeros(len(current))
    for n, value in enumerate(current):
        d = (value - u[n]) / tau
        s[n] = float(u[n] + d >= 1.)
        f[n] = d - s[n] / tau
        u[n+1] = sum((((n-j+1)**alpha - (n-j)**alpha) / math.gamma(alpha+1)) * f[j]
                     for j in range(n+1))
    return u[1:], s


currents = {'constant': np.full(40, 1.5), 'pulse': np.r_[8., np.zeros(39)],
            'random': np_rng.normal(1., 2., 40)}
for alpha in (.3, .5, .7, 1.):
    for tau in (4., 8.):
        for name, current in currents.items():
            u, s = upstream(current, alpha, tau)
            local_u, local_s = local_definition(current, alpha, tau)
            rows.append({'alpha': alpha, 'tau': tau, 'input': name,
                         'state_max_error': float(np.max(np.abs(u-local_u))),
                         'spikes_equal': bool(np.array_equal(s, local_s)),
                         'spike_count': int(s.sum())})
            traces.extend([name, alpha, tau, n, current[n], u[n], s[n], local_u[n], local_s[n]]
                          for n in range(len(current)))

with contextlib.redirect_stdout(io.StringIO()):
    port = runpy.run_path(str(nsmt / 'f_lif_pop_v3/analysis/reset_conventions.py'))
u, s = upstream(port['I'], port['ALPHA'], port['TAU'])
port_u, port_s = port['code_convention']()

x = torch.linspace(-2., 2., 101, dtype=torch.float64, requires_grad=True)
xx = x.detach().clone().requires_grad_(True)
y = arctan_surrogate(x, 5.)
yy = layers.arctan_spike(xx, 5.)
y.sum().backward()
yy.sum().backward()

with (out / 'scalar_trajectories.csv').open('w') as handle:
    writer = csv.writer(handle)
    writer.writerow(['input', 'alpha', 'tau', 'n', 'current', 'upstream_U_next',
                     'upstream_spike', 'independent_U_next', 'independent_spike'])
    writer.writerows(traces)
result = {'reference_commit': 'fcd743befe504b1a471fa81887e6af7d6789da2e',
          'torch': torch.__version__, 'device': 'cpu', 'dtype': 'float64', 'seed': 7,
          'reference_root': str(ref), 'nsmt_snapshot': str(nsmt), 'cases': rows,
          'existing_reset_conventions_port': {'state_max_error': float(np.max(np.abs(u-port_u))),
                                            'spikes_equal': bool(np.array_equal(s, port_s)),
                                            'spike_count': int(s.sum())},
          'surrogate': {'spikes_equal': bool(torch.equal(y, yy)),
                        'gradient_max_error': float((x.grad-xx.grad).abs().max())},
          'source_manifest': json.loads((ref/'source_manifest.json').read_text()),
          'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'trajectory_sha256': hashlib.sha256((out/'scalar_trajectories.csv').read_bytes()).hexdigest(),
          'limitations': 'Scalar eager public solver+LIF only. Full SNNWrapper/FX, compiled execution, '
                         'adjoint backward, GPU, and training not run. No claim of v3 soma equivalence.'}
(out/'reference_results.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['cases','source_manifest']}, indent=2))
print('cases', len(rows), 'worst state error', max(r['state_max_error'] for r in rows),
      'all spikes equal', all(r['spikes_equal'] for r in rows))
