"""MODI 2026-11-18: checks of the SpikeDE-basis comparison (user decision 11-18 "(가)": Spikformer-style and the msn base both built
from SpikeDE neurons only). --neuron_all lif swaps every spikingjelly MultiStepLIFNode (embedding, SSA, MLP) for the SpikeDE LIF
(NeoNeuron(LIFNeuron), tau 2, threshold 1, reset spikede); the msn base keeps its SpikeDE population Neocortex.
Models are built from the stored config.pt of the reference runs (ETT: ETTh2 H96 and ETTm2 H720 seed 42; MIT-BIH seed 42), once with
the stored neuron_all (sj_lif) and once with neuron_all lif, same seed:
  s1 no spikingjelly LIF is left; the number swapped = the number of spikingjelly LIF in the reference
  s2 the Neocortex neurons are untouched (same count of NeoNeuron(PopulationFLIF), no other SpikeDE neuron added there)
  s3 init: every reference state_dict entry exists with an identical tensor; the only new entries are the swapped neurons' buffers
     (tau, b); the parameter count is unchanged
  s4 a training-mode forward / backward on a fixed input is finite and reaches the hippocampus attention, the Neocortex embedding
     (base) and the output head; the output differs from the reference (the swap is active)
  s5 the swapped neurons are LIFNeuron(tau 2, threshold 1, reset spikede, max_length T_h = 4)
  s6 refusal: --neuron_all lif on the query form with a spikingjelly Neocortex raises
Any failure exits 1. usage (classification folder): python check_sde.py
"""
import os
import sys
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, os.getcwd())
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from forecast_wrap import Forecaster
from layers import NeoNeuron
from load_model import LOAD_MODEL
from neurons import LIFNeuron, PopulationFLIF

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.join(HERE, '..', 'forecasting', 'results')
CL = os.path.join(HERE, 'results')
FAIL = []


def ok(name, cond, info=''):
    print(f"{'PASS' if cond else 'FAIL'} {name} {info}")
    if not cond:
        FAIL.append(name)


def run_dir(path, **sel):
    got = []
    for ln in open(path):
        if ln.startswith('RESULT'):
            d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
            if all(d.get(k) == str(v) for k, v in sel.items()):
                got.append(d['dir'])
    if len(got) != 1:
        raise RuntimeError(f'{path} {sel}: {len(got)} rows')
    return got[0]


CASES = {
    'ett_h2_base_lif': ('ett', run_dir(os.path.join(FC, 'flif-v1-ett-ms', 'raw.txt'), cond='nsmt4q_pop_lif_msn', ds='ETTh2', pl=96, seed=42)),
    'ett_m2_base_flif': ('ett', run_dir(os.path.join(FC, 'flif-v1-ett-ms', 'raw.txt'), cond='nsmt4q_pop_flif_msn', ds='ETTm2', pl=720, seed=42)),
    'ett_h2_spk': ('ett', run_dir(os.path.join(FC, 'flif-v1-gen-ett', 'raw.txt'), cond='spk_ssa', ds='ETTh2', pl=96, seed=42)),
    'ett_m2_spk': ('ett', run_dir(os.path.join(FC, 'flif-v1-gen-ett', 'raw.txt'), cond='spk_ssa', ds='ETTm2', pl=720, seed=42)),
    'mb_base_lif': ('mb', run_dir(os.path.join(CL, 'flif-v1-mitbih-ms', 'raw.txt'), cond='nsmt4q_pop_lif_msn', seed=42)),
    'mb_base_flif': ('mb', run_dir(os.path.join(CL, 'flif-v1-mitbih-ms', 'raw.txt'), cond='nsmt4q_pop_flif_msn', seed=42)),
    'mb_spk': ('mb', run_dir(os.path.join(CL, 'flif-v1-gen-mitbih', 'raw.txt'), cond='spk_ssa', seed=42)),
}


def build(kind, d, **ov):
    cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, device='cuda', **ov))
    torch.manual_seed(0)
    body = LOAD_MODEL[a.model](a, True)
    m = Forecaster(body, a.embed_dim, a.pred_len, bias=a.bias) if kind == 'ett' else body
    return m.cuda(), body, a


def neo_kinds(m):
    pop = sum(isinstance(x, NeoNeuron) and isinstance(x.neuron, PopulationFLIF) for x in m.modules())
    lif = [n for n, x in m.named_modules() if isinstance(x, NeoNeuron) and isinstance(x.neuron, LIFNeuron)]
    return pop, lif


def fwd(kind, m, a):
    g = torch.Generator().manual_seed(5)
    m.train(); m.train_mode = 'training'; functional.reset_net(m)
    if kind == 'ett':
        x = torch.randn(4, a.seq_len, 7, generator=g).cuda(); y = torch.randn(4, a.pred_len, 7, generator=g).cuda()
        o = m(x); loss = torch.nn.functional.mse_loss(o, y)
    else:
        x = torch.randn(8, 1, 187, generator=g).cuda(); y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]).cuda()
        o = m(x); o = o[0] if isinstance(o, tuple) else o; loss = torch.nn.functional.cross_entropy(o, y)
    loss.backward()
    out = o.detach().cpu()
    functional.reset_net(m)
    return out, {n: p.grad for n, p in m.named_parameters()}


for key, (kind, d) in CASES.items():
    ref, rbody, a = build(kind, d)
    if getattr(a, 'neuron_all', 'sj_lif') != 'sj_lif':
        raise RuntimeError(f'{key}: reference already has neuron_all {a.neuron_all}')
    new, nbody, _ = build(kind, d, neuron_all='lif')
    n_sj = sum(isinstance(x, MultiStepLIFNode) for x in ref.modules())
    left = sum(isinstance(x, MultiStepLIFNode) for x in new.modules())
    ok(f'{key} s1', left == 0 and getattr(nbody, 'n_swapped', -1) == n_sj and n_sj > 0, f'(spikingjelly LIF {n_sj} -> left {left}, swapped {getattr(nbody, "n_swapped", None)})')
    rp, rl = neo_kinds(ref); np_, nl = neo_kinds(new)
    ok(f'{key} s2', rp == np_ and len(nl) == len(rl) + n_sj, f'(population Neocortex neurons {rp} -> {np_}; SpikeDE LIF {len(rl)} -> {len(nl)})')
    rs, ns = ref.state_dict(), new.state_dict()
    same = all(k in ns and torch.equal(rs[k].cpu(), ns[k].cpu()) for k in rs)
    extra = [k for k in ns if k not in rs]
    swapped = {n for n in nl}
    extra_ok = all(k.rsplit('.', 2)[0] in swapped and k.rsplit('.', 1)[1] in ('tau', 'b') for k in extra)
    pr, pn = sum(p.numel() for p in ref.parameters()), sum(p.numel() for p in new.parameters())
    ok(f'{key} s3', same and extra_ok and pr == pn, f'(reference entries identical {same}; new entries {len(extra)} all swapped tau / b {extra_ok}; params {pr} / {pn})')
    o_ref, _ = fwd(kind, ref, a)
    o_new, gr = fwd(kind, new, a)
    def reached(sub):
        gs = [g for n, g in gr.items() if sub in n]
        return bool(gs) and all(g is not None and torch.isfinite(g).all() for g in gs) and any(g.abs().sum() > 0 for g in gs)
    need = ['q_linear', 'head' if kind == 'ett' else 'head'] + (['neo'] if 'base' in key else [])
    rch = {s: reached(s) for s in need}
    ok(f'{key} s4', bool(torch.isfinite(o_new).all()) and all(rch.values()) and not torch.equal(o_ref, o_new),
       f'(finite {bool(torch.isfinite(o_new).all())}; gradient reaches {rch}; output differs from reference {not torch.equal(o_ref, o_new)})')
    cfgs = {(float(x.neuron.tau.flatten()[0]), x.neuron.threshold, x.neuron.reset_mode, x.neuron.max_length)
            for n, x in new.named_modules() if n in swapped and n not in rl}
    ok(f'{key} s5', cfgs == {(2.0, 1.0, 'spikede', int(a.hippo_T))}, f'(swapped neuron settings {cfgs})')
    del ref, new
    torch.cuda.empty_cache()

d = CASES['ett_h2_base_lif'][1]
try:
    build('ett', d, neuron_all='lif', neo_neuron='sj_lif', neo_res='none')
    ok('s6 refusal', False, '(built)')
except NotImplementedError as e:
    ok('s6 refusal', True, f'({e})')

print('ALL PASS' if not FAIL else f'FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
