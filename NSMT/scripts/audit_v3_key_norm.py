"""CPU-only audit of frozen selector normalization; no training or optimizer."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

oldroot, root, output = map(lambda p: Path(p).resolve(), sys.argv[1:4])
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
old = load('old_layers', oldroot/'f_lif_pop_v3/forecasting/layers.py')
new = load('new_layers', root/'f_lif_pop_v3/forecasting/layers.py')
torch.manual_seed(7)
torch.set_num_threads(2)
a = old.PopulationNeuron(embed_dim=3, max_length=12).double()
b = new.PopulationNeuron(embed_dim=3, max_length=12).double()
# Explicitly copy only matching tensors for identity parity; do not alter legacy checkpoint.
newstate = b.state_dict()
newstate.update(a.state_dict())
b.load_state_dict(newstate, strict=True)
x = torch.randn(12, 2, 3, dtype=torch.float64)*3+1
xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
ya, aa = a(xa, return_aux=True, analog=True)
yb, ab = b(xb, return_aux=True, analog=True)
aa['analog'].square().mean().backward()
ab['analog'].square().mean().backward()
result = {'torch':torch.__version__, 'seed':7, 'shape':list(x.shape), 'dtype':'float64',
          'identity_spike_max_error':float((ya-yb).abs().max()),
          'identity_state_max_error':float((aa['state']-ab['state']).abs().max()),
          'identity_input_gradient_max_error':float((xa.grad-xb.grad).abs().max()),
          'default_buffers_identity':bool((b.selector.key_mean == 0).all() and (b.selector.key_std == 1).all())}
# Compute xi directly from full recurrence before fitting and compare to stored statistics.
with torch.no_grad():
    _, full = b(x, mode='full', return_aux=True)
    previous = torch.cat([torch.zeros_like(full['state'][:1]), full['state'][:-1]], dim=0)
    xi = torch.cat([previous, x.unsqueeze(-1)],dim=-1)
    expected_mean = xi.mean((0,1,2))
    expected_std = xi.std((0,1,2)).clamp_min(1e-6)
b.fit_key_norm(x)
first = {k:v.clone() for k,v in b.state_dict().items()}
result['fit_mean_max_error'] = float((b.selector.key_mean-expected_mean).abs().max())
result['fit_std_max_error'] = float((b.selector.key_std-expected_std).abs().max())
result['fitted_mean'] = b.selector.key_mean.tolist()
result['fitted_std'] = b.selector.key_std.tolist()
b.fit_key_norm(x)
result['same_full_sample_repeat_max_error'] = max(float((v-first[k]).abs().max()) for k,v in b.state_dict().items() if v.is_floating_point() and torch.isfinite(v).all())
with torch.no_grad():
    base, aux = b(x, return_aux=True, analog=True)
    moved = x.clone(); moved[6:] += 10
    _, moved_aux = b(moved, return_aux=True, analog=True)
    c = new.PopulationNeuron(embed_dim=3, max_length=12).double()
    c.load_state_dict(b.state_dict(), strict=True)
    _, restored = c(x, return_aux=True, analog=True)
    result['fitted_causal_analog_max_error'] = float((aux['analog'][:6]-moved_aux['analog'][:6]).abs().max())
    result['new_schema_reload_analog_max_error'] = float((aux['analog']-restored['analog']).abs().max())
    result['fitted_finite'] = bool(torch.isfinite(aux['state']).all() and torch.isfinite(aux['analog']).all())
# Exercise the real strict loader against a copied historical checkpoint.
sys.path.insert(0,str(root/'f_lif_pop_v3/forecasting'))
from model import LOAD_MODEL
state_dir = next(root.glob('f_lif_pop_v3/forecasting/log/pilot-eta-001742/**/model_state'))
args = SimpleNamespace(**torch.load(state_dir/'config.pt', map_location='cpu'))
args.device = torch.device('cpu'); args.save_model_state_path = str(state_dir)
try:
    LOAD_MODEL[args.model](args, train=False)
    result['historical_checkpoint_load'] = {'success':True}
except Exception as exc:
    result['historical_checkpoint_load'] = {'success':False,'type':type(exc).__name__,'detail':str(exc)}
result['scope'] = 'Library normalization and old-checkpoint compatibility only; no fitting with real train data, training, performance or gradient-explosion claim tested.'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(output.read_text())
