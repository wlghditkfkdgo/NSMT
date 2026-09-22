"""Read-only follow-up probes; argv[1] is the frozen NSMT source root."""
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(sys.argv[1]) / 'f_lif_pop_v3/forecasting'))
import layers
import calibrate
from config import neuron_kwargs
from data_provider.synthetic import make_sequence, rng_codes, task_stats

torch.set_num_threads(2)
torch.manual_seed(7)
out = {'source_root': sys.argv[1], 'torch': torch.__version__, 'seed': 7}

# Check the reported chance on precisely the same sequences, with sequence weighting.
rows = []
for nk in (3, 5, 8):
    reported = task_stats(n_seq=300, seed=20260921, n_keys=nk, cue_mode='code')
    rng = np.random.default_rng(20260921)
    slot, kernel, full_coverage = [], [], []
    b = layers.fractional_coefficients(.7, 42, torch.float64).numpy()
    for _ in range(300):
        x, y, truth, recall = make_sequence(rng, n_keys=nk, cue_mode='code')
        q = np.flatnonzero(recall)
        slot.append(np.mean([truth[n].sum() / n for n in q]))
        kernel.append(np.mean([b[n - np.flatnonzero(truth[n])].sum() / b[1:n+1].sum()
                               for n in q]))
        key = np.linalg.norm(x.reshape(42, 8)[:, :4, None] - rng_codes(nk, 4).T[None], axis=1).argmin(-1)
        full_coverage.append(len(np.unique(key[recall])) == nk)
    rows.append({'n_keys': nk, 'reported': reported,
                 'independent_slot_chance': float(np.mean(slot)),
                 'independent_kernel_mass': float(np.mean(kernel)),
                 'fraction_sequences_querying_every_key': float(np.mean(full_coverage))})
out['task_stats'] = rows
out['codes_16_unique'] = len(np.unique(rng_codes(16, 4), axis=0))
try:
    rng_codes(17, 4)
except ValueError as e:
    out['codes_17_rejected'] = str(e)

# Test actual config-to-constructor behavior, plus state serialization under the same config.
cfg = SimpleNamespace(num_population=4, alpha=.7, tau=[4.,8.,16.,32.], heterogeneous=True,
                      num_patches=42, theta=1., eta_init=-4., tau_s=2., threshold=1.,
                      surrogate_scale=5., cap=False, eta_fixed=1.)
kw = neuron_kwargs(cfg)
model = layers.PopulationNeuron(2, **kw)
copy = layers.PopulationNeuron(2, **kw)
buf = io.BytesIO()
torch.save(model.state_dict(), buf)
buf.seek(0)
copy.load_state_dict(torch.load(buf, map_location='cpu'))
x = torch.randn(42, 2, 2)
with torch.no_grad():
    s, a = model(x, return_aux=True)
    s2, a2 = copy(x, return_aux=True)
out['config_roundtrip'] = {'eta': model.selector.eta.item(), 'cap': model.selector.cap,
                           'eta_trainable': model.selector.eta_hat.requires_grad,
                           'spikes_equal': torch.equal(s, s2),
                           'states_equal': torch.equal(a['state'], a2['state']),
                           'no_cap_reported_cap_rate_max': a['cap_rate'].max().item()}

# Check calibration values against the actual tensor, and selection under a violated bound.
rng = np.random.default_rng(20260921)
data = torch.from_numpy(np.stack([make_sequence(rng)[0] for _ in range(64)]))
patch = layers.to_patches(data, 8)
emb = layers.Embedding(8, 32, 8., input_norm='frozen', max_length=42)
emb.fit_norm(patch)
measured = calibrate.probe(emb, [patch[:, :32], patch[:, 32:]], torch.device('cpu'), 8.)
with torch.no_grad():
    _, aux = emb(patch, mode='full', return_aux=True)
actual = aux['state'].abs().amax(dim=(0, 1, 2))
bad = dict(measured, firing_rate=.2, finite=True, max_abs_state=1001., max_abs_current=1.,
           branch_abs_mean=[1., 1., 1., 1.])
if 'within_bound' in bad:
    bad.update(within_bound=False, declared_bound=10.)
picked, reason = calibrate.choose([bad])
out['calibration'] = {'reported': measured, 'actual_branch_abs_max': actual.tolist(),
                      'bound_violating_candidate_accepted': picked is not None,
                      'candidate_state': 1001., 'candidate_declared_bound': 10., 'reason': reason}
print(json.dumps(out, ensure_ascii=False, indent=2))
