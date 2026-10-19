"""Checks for nsmt_flif_v1/classification (MIT-BIH, MODI_PROJECT_LOG 2026-10-19). Fail-closed: a check that
cannot run, or whose test is vacuous (no spikes), FAILs.

  c1  defaults = the ECG reference (nsmt_base/classification/ECG): same parameters, outputs and parameters after
      AdamW steps, bitwise (the reference runs in a subprocess from its own folder)
  c2  the debug switches dropped from train.py (anomaly detection, CUDA_LAUNCH_BLOCKING) change nothing: parameters
      after training steps and the test logits are bitwise equal with and without them
  c3  spikeDE LIF = f-LIF(alpha = 1, pred, full history) inside the model: neocortex spikes bitwise, logits allclose
  c4  placement: hippocampus neurons are the spikingjelly LIF in every condition; neocortex neurons per condition
  c5  token-axis causality of the neocortex (all neuron kinds): changing token j changes no output before token j and
      does change token j or later; firing rate in [0.02, 0.98] at every layer (otherwise vacuous -> FAIL)
  c6  no_neo / hippo_only: neocortex modules receive no gradient; hippo_only uses the cross-attention, no_neo not
  c7  data: ECG create_loader on the 70:30 files -> shapes, class counts, CAL weights
  c8  neurons.py is byte-identical to nsmt_flif_v1/forecasting/neurons.py
  c9  energy recorder: every neuron recorded with T = 11 (incl. the single-step LIFNode of the ECG NeoLayer)
  c10 run directories: distinct for the 7 conditions; the baseline name equals the ECG naming
  c11 (stage 6) population f-LIF in the neocortex: K = 4 branches with time constants 4/8/16/32, one LIF soma, output 0/1 of
      shape [N, B, 1, D], causal along the token axis, fires (rate in [0.02, 0.98]); the energy recorder reports it as pop_flif
  c12 (stage 6) bidirectional neocortex: fused output == max(forward, flipped reverse) computed from the two branches
      separately; changing token j changes the forward branch only at tokens >= j and the reverse branch only at <= j;
      the reverse modules have their own parameters; run-directory names of the 4 stage-6 conditions are distinct
  c13 (stage 7) NeoMixer: patch tokens N = 11 and every spiking neuron runs T = 11 steps (energy recorder), no attention
      module; 'none' mixer: with T = N every neuron state is causal along tokens, so token j changes tokens >= j only
      (the embedding / MLP LIFs carry state too -- the control is 'aligned without a mixer', not 'token-independent');
      bidirectional mixer fed a changed embedding token j: forward stack changes >= j only, reverse stack <= j only,
      fused = max; training forward returns plain logits so
      train.py uses CE only; testing forward returns (logits, token spikes); run-directory names distinct
  c14 (stage 9) multichannel tokens: forcing the stage-9 token path with C = 1 reproduces the original path (inputs equal,
      logits allclose); with C = 700 every token equals a manual slice x[b, :, nS : nS+P] flattened channel-major (myModel
      and NeoMixer), including the replicated end padding
  c15 (stage 9) Hippo-only SSA: no neocortex / cross-attention / decoder modules, T_h = 4, the SSA sees [4, B, N, D] and
      its relative-bias table is [2N-1, heads]; the training loss path is plain CE (no time_block attribute)
  c16 (stage 9) population calibration: every pop_flif soma weight = 1.0 with --pop_soma_init 1.0 (0.25 without);
      on real CinC beats, with batch-statistics normalisation as in training (eval-mode BN at init passes ~0.02-scale
      currents and silences every neuron, first run), the calibrated neocortex fires at init and fires more than the
      uncalibrated one; the spikeDE LIF rate is reported as reference
  c17 (stage 9) CinC 2017 data: 8,528 records, label counts = REFERENCE-v3, 5 folds disjoint and covering all records,
      train / val / test disjoint; record-level aggregation of predict() = manual mean of window softmaxes
  c18 (stage 9) SHD data: 8,156 / 2,264 samples, 20 classes, binned counts = spikes kept by the converter, validation
      split disjoint from training and containing every class
usage: python check_cls.py [--out results/checks/check_cls.txt]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
NSMT = os.path.abspath(os.path.join(HERE, '..', '..'))
ECG = os.path.join(NSMT, 'nsmt_base', 'classification', 'ECG')
DATA = os.path.join(ECG, 'data')
sys.path.insert(0, HERE)
os.chdir(HERE)

import torch                                                          # noqa: E402

HP = dict(keep_ratio=0.15, gating='attn', patch_size=32, embed_dim=384, num_heads=8, num_classes=5, num_channels=1,
          seq_len=187, mlp_ratios=1.0, num_layers=1, time_num_layers=2, bias=False, tau=2.0, spk_encoding=False,
          model='myModel')
CONDS = {'base': {}, 'neo_lif_both': dict(neo_neuron='lif'), 'neo_flif_both': dict(neo_neuron='flif'),
         'neo_lif_mlp': dict(neo_neuron='lif', neo_where='mlp'), 'neo_flif_mlp': dict(neo_neuron='flif', neo_where='mlp'),
         'no_neo': dict(no_neo=True), 'hippo_only': dict(hippo_only=True),
         'neo_pop_both': dict(neo_neuron='pop_flif'), 'bidir_lif_both': dict(neo_neuron='lif', neo_bidir=True),            # stage 6
         'bidir_flif_both': dict(neo_neuron='flif', neo_bidir=True), 'bidir_pop_both': dict(neo_neuron='pop_flif', neo_bidir=True)}

# helper run in a subprocess (own sys.path / cwd): builds the model, runs AdamW steps on a fixed batch, saves tensors
HELPER = r'''
import os, sys, json, torch
d, out, kw, debug = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4] == '1'
if debug:
    os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    torch.autograd.set_detect_anomaly(True)
sys.path.insert(0, d); os.chdir(d)
from types import SimpleNamespace as NS
from config import set_random_seed
set_random_seed(0)
from spikingjelly.clock_driven import functional
from load_model import LOAD_MODEL
m = LOAD_MODEL['myModel'](NS(**kw), True).cuda()
init = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
g = torch.Generator().manual_seed(1)
x = torch.rand(64, 1, 187, generator=g).cuda(); y = torch.randint(0, 5, (64,), generator=g).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
m.train(); m.train_mode = 'training'
outs = []
for i in range(3):
    o = m(x)
    z = o[0]; mse = ((o[3] - o[4]) ** 2).mean()
    loss = torch.nn.functional.cross_entropy(z, y) * 0.9 + mse * 0.1
    opt.zero_grad(); loss.backward(); opt.step(); functional.reset_net(m)
    outs.append(z.detach().cpu())
m.eval(); m.train_mode = 'testing'
with torch.no_grad():
    zt = m(x)[0].detach().cpu(); functional.reset_net(m)
torch.save(dict(init=init, final={k: v.detach().cpu() for k, v in m.state_dict().items()}, train_z=outs, test_z=zt), out)
'''


def run_helper(d, kw, out, debug=False):
    PY = sys.executable
    r = subprocess.run([PY, '-c', HELPER, d, out, json.dumps(kw), '1' if debug else '0'], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'helper failed in {d}:\n{r.stderr[-2000:]}')
    return torch.load(out)


def same_tensors(a, b):
    if set(a) != set(b):
        return False, f'keys differ: {sorted(set(a) ^ set(b))[:5]}'
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    return (not bad), (f'{len(bad)} differ, e.g. {bad[:3]}' if bad else f'{len(a)} tensors equal')


RESULTS = []


def report(name, ok, msg):
    RESULTS.append((name, bool(ok), msg))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}", flush=True)


def check(name):
    def deco(fn):
        def run():
            try:
                fn()
            except Exception as e:                                    # fail closed
                report(name, False, f'could not run: {type(e).__name__}: {e}')
        run.__name__ = name
        return run
    return deco


@check('c1')
def c1():
    tmp = tempfile.mkdtemp(dir=os.environ.get('CHECK_TMP'))
    ref = run_helper(ECG, HP, os.path.join(tmp, 'ref.pt'))
    new = run_helper(HERE, HP, os.path.join(tmp, 'new.pt'))
    ok1, m1 = same_tensors(ref['init'], new['init'])
    ok2, m2 = same_tensors(ref['final'], new['final'])
    ok3 = all(torch.equal(a, b) for a, b in zip(ref['train_z'], new['train_z'])) and torch.equal(ref['test_z'], new['test_z'])
    report('c1', ok1 and ok2 and ok3, f'init: {m1}; after 3 AdamW steps: {m2}; train/test logits bitwise: {ok3}')


@check('c2')
def c2():
    tmp = tempfile.mkdtemp(dir=os.environ.get('CHECK_TMP'))
    msgs, oks = [], []
    for c in ('base', 'neo_flif_both'):
        kw = dict(HP, **CONDS[c])
        a = run_helper(HERE, kw, os.path.join(tmp, f'{c}_dbg.pt'), debug=True)
        b = run_helper(HERE, kw, os.path.join(tmp, f'{c}_nodbg.pt'), debug=False)
        ok, m = same_tensors(a['final'], b['final'])
        ok = ok and torch.equal(a['test_z'], b['test_z'])
        oks.append(ok); msgs.append(f'{c}: {m}, test logits equal {torch.equal(a["test_z"], b["test_z"])}')
    report('c2', all(oks), '; '.join(msgs))


def build(cond, seed=0, **extra):
    from config import set_random_seed
    from load_model import LOAD_MODEL
    set_random_seed(seed)
    kw = dict(HP, **CONDS[cond]); kw.update(extra)
    return LOAD_MODEL['myModel'](NS(**kw), True).cuda()


def make_fire(m, seed=7, emb_std=2.0, mlp_std=5.0):
    """At init the neocortex is silent (vacuous tests); re-draw its input weights from a fixed generator so it fires."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for emb, tb in ((m.encoding_neo, m.time_block), (getattr(m, 'encoding_neo_rev', None), getattr(m, 'time_block_rev', None))):
            if emb is None:
                continue
            w = emb.emb_linear.weight
            w.copy_(torch.randn(w.shape, generator=g) * emb_std / w.shape[1] ** .5)
            for l in tb.Mem:
                l.w_in.weight.copy_(torch.randn(l.w_in.weight.shape, generator=g) * mlp_std / l.w_in.weight.shape[1] ** .5)


def neo_spikes(m, x):
    from spikingjelly.clock_driven import functional
    store = {}
    h = m.time_block.register_forward_hook(lambda mod, i, o: store.__setitem__('s', o.detach()))
    with torch.no_grad():
        z = m(x)[0]
    h.remove(); functional.reset_net(m)
    return z, store['s']


@check('c3')
def c3():
    g = torch.Generator().manual_seed(2)
    x = (torch.randn(64, 1, 187, generator=g) * 3).cuda()
    msgs, oks = [], []
    for where in ('both', 'mlp'):
        a = build('neo_lif_' + where); b = build('neo_flif_' + where, neo_alpha=1.0)
        for m in (a, b):
            make_fire(m); m.eval(); m.train_mode = 'testing'
        za, sa = neo_spikes(a, x); zb, sb = neo_spikes(b, x)
        rate = sa.float().mean().item()
        ok = torch.equal(sa, sb) and torch.allclose(za, zb, atol=1e-5, rtol=1e-4) and 0.02 <= rate <= 0.98
        oks.append(ok); msgs.append(f'{where}: spikes equal {torch.equal(sa, sb)} (rate {rate:.3f}), max |dz| {(za - zb).abs().max().item():.2e}')
    report('c3', all(oks), '; '.join(msgs))


@check('c4')
def c4():
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode, LIFNode
    from layers import NeoNeuron
    from neurons import LIFNeuron, FLIFNeuron
    expect = {'base': ('MultiStepLIFNode', 'LIFNode'), 'neo_lif_both': ('LIFNeuron', 'LIFNeuron'),
              'neo_flif_both': ('FLIFNeuron', 'FLIFNeuron'), 'neo_lif_mlp': ('MultiStepLIFNode', 'LIFNeuron'),
              'neo_flif_mlp': ('MultiStepLIFNode', 'FLIFNeuron'), 'no_neo': ('MultiStepLIFNode', 'LIFNode'),
              'hippo_only': ('MultiStepLIFNode', 'LIFNode')}
    msgs, oks = [], []
    for c, (emb, mlp) in expect.items():
        m = build(c)
        name = lambda mod: type(mod.neuron).__name__ if isinstance(mod, NeoNeuron) else type(mod).__name__
        got_emb = name(m.encoding_neo.emb_lif); got_mlp = {name(l.lif) for l in m.time_block.Mem}
        hippo = {type(mod).__name__ for n, mod in m.named_modules()
                 if isinstance(mod, (MultiStepLIFNode, LIFNode, LIFNeuron, FLIFNeuron, NeoNeuron))
                 and not n.startswith(('encoding_neo', 'time_block'))}
        alpha = {round(l.lif.neuron.alpha, 3) for l in m.time_block.Mem if isinstance(l.lif, NeoNeuron) and hasattr(l.lif.neuron, 'alpha')}
        ok = got_emb == emb and got_mlp == {mlp} and hippo == {'MultiStepLIFNode'} and (alpha == {0.5} if 'flif' in c else True)
        oks.append(ok); msgs.append(f'{c}: emb {got_emb}, mlp {sorted(got_mlp)}, hippo {sorted(hippo)}' + (f', alpha {sorted(alpha)}' if alpha else ''))
    report('c4', all(oks), '; '.join(msgs))


@check('c5')
def c5():
    from spikingjelly.clock_driven import functional
    msgs, oks = [], []
    for c in ('base', 'neo_lif_both', 'neo_flif_both'):
        m = build(c); m.eval()
        make_fire(m)                                                   # at init the neocortex is silent
        g = torch.Generator().manual_seed(3)
        tok = torch.randn(11, 32, 1, 32, generator=g).cuda()             # [N, B, 1, P] neocortex input
        def run(t):
            with torch.no_grad():
                e = m.encoding_neo(t); o = m.time_block(e)
            functional.reset_net(m)
            return e, o
        e0, o0 = run(tok)
        rates = [e0.mean().item(), o0.mean().item()]
        sub = []
        for j in (2, 5, 8):
            t2 = tok.clone(); t2[j] = torch.randn(32, 1, 32, generator=g).cuda()
            e1, o1 = run(t2)
            before = torch.equal(e1[:j], e0[:j]) and torch.equal(o1[:j], o0[:j])
            after = not torch.equal(o1[j:], o0[j:])
            sub.append(before and after)
        ok = all(sub) and all(0.02 <= r <= 0.98 for r in rates)
        oks.append(ok); msgs.append(f'{c}: causal {sub}, rates emb {rates[0]:.3f} out {rates[1]:.3f}')
    report('c5', all(oks), '; '.join(msgs))


@check('c6')
def c6():
    from spikingjelly.clock_driven import functional
    g = torch.Generator().manual_seed(4)
    x = torch.rand(32, 1, 187, generator=g).cuda(); y = torch.randint(0, 5, (32,), generator=g).cuda()
    calls, zs, msgs, oks = {}, {}, [], []
    for c in ('no_neo', 'hippo_only'):
        m = build(c); m.train(); m.train_mode = 'training'
        n = [0]
        h = m.data_block[0].mca.register_forward_hook(lambda mod, i, o: n.__setitem__(0, n[0] + 1))
        o = m(x); loss = torch.nn.functional.cross_entropy(o[0], y); loss.backward(); functional.reset_net(m); h.remove()
        neo_grads = [p.grad for nme, p in m.named_parameters() if nme.startswith(('encoding_neo', 'time_block', 'weak_decoder'))]
        mca_grads = [p.grad for nme, p in m.named_parameters() if '.mca.' in nme]
        calls[c] = n[0]; zs[c] = all(gr is not None for gr in mca_grads)
        ok = all(gr is None for gr in neo_grads)
        oks.append(ok); msgs.append(f'{c}: neocortex grads all None {ok}, mca calls {n[0]}, mca receives gradient {zs[c]}')
    ok = calls['no_neo'] == 0 and calls['hippo_only'] == 1 and zs['hippo_only'] and not zs['no_neo']
    report('c6', all(oks) and ok, '; '.join(msgs))


@check('c7')
def c7():
    import numpy as np
    from dataloader import create_loader
    from utils import get_class_weights
    tr, ntr = create_loader(train=True, batch_size=512, num_classes=5, data_root=DATA + '/5class_mitbih_train_7_187.csv',
                            train_val_ratio=None, data_window_size=187, sampling='None', num_workers=0)
    te, nte = create_loader(train=False, batch_size=512, num_classes=5, data_root=DATA + '/5class_mitbih_test_3_187.csv',
                            data_window_size=187, num_workers=0)
    xb, yb = next(iter(tr))
    w = get_class_weights(ntr)
    exp = torch.clamp(torch.log(ntr.max().double() / ntr.double()), min=1.0)
    ok = (list(xb.shape) == [512, 1, 187] and ntr.tolist() == [63412, 1946, 5065, 562, 5627]
          and nte.tolist() == [27176, 833, 2171, 241, 2412] and len(tr.dataset) == 76612 and len(te.dataset) == 32833
          and torch.allclose(w.double(), exp, atol=1e-6) and not torch.isnan(xb).any())
    report('c7', ok, f'batch {list(xb.shape)}, train counts {ntr.tolist()}, test counts {nte.tolist()}, CAL weights {[round(v, 4) for v in w.tolist()]}')


@check('c8')
def c8():
    h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    a, b = h(os.path.join(HERE, 'neurons.py')), h(os.path.join(NSMT, 'nsmt_flif_v1', 'forecasting', 'neurons.py'))
    report('c8', a == b, f'sha256 {a[:16]} vs forecasting {b[:16]}')


@check('c9')
def c9():
    from spikingjelly.clock_driven import functional
    from energy import FiringRateRecorder
    g = torch.Generator().manual_seed(5)
    x = torch.rand(64, 1, 187, generator=g).cuda()
    msgs, oks = [], []
    for c in ('base', 'neo_flif_both', 'neo_lif_mlp'):
        m = build(c); m.eval(); m.train_mode = 'testing'
        rec = FiringRateRecorder(m)
        with torch.no_grad():
            for _ in range(2):
                m(x); rec.add_samples(x.shape[0]); functional.reset_net(m)
        rec.detach()
        rows = rec.layers()
        neo = [r for r in rows if r['path'] == 'neo']; hip = [r for r in rows if r['path'] == 'hippo']
        ok = len(neo) == 3 and len(hip) == 8 and all(r['T'] == 11 for r in rows)
        kinds = sorted({r['kind'] for r in neo})
        oks.append(ok); msgs.append(f'{c}: neo {len(neo)} ({kinds}), hippo {len(hip)}, T {sorted({r["T"] for r in rows})}')
    report('c9', all(oks), '; '.join(msgs))


@check('c10')
def c10():
    import config as cfg
    tmp = tempfile.mkdtemp(dir=os.environ.get('CHECK_TMP'))
    flags = {'base': [], 'neo_lif_both': ['--neo_neuron', 'lif'], 'neo_flif_both': ['--neo_neuron', 'flif'],
             'neo_lif_mlp': ['--neo_neuron', 'lif', '--neo_where', 'mlp'], 'neo_flif_mlp': ['--neo_neuron', 'flif', '--neo_where', 'mlp'],
             'no_neo': ['--no_neo'], 'hippo_only': ['--hippo_only']}
    names = {}
    argv0 = sys.argv
    try:
        for c, f in flags.items():
            sys.argv = ['train.py', '--log_dir', tmp, '--seed', '42', '-s'] + f
            a = cfg.Config(); a.set_args(cfg.parse_arguments())
            names[c] = os.path.basename(a.save_result_path)
    finally:
        sys.argv = argv0
    ok = len(set(names.values())) == len(names) and names['base'] == 'seed42_rec0_tau2.0_wmfull'
    report('c10', ok, ', '.join(f'{c}={n}' for c, n in names.items()))


@check('c11')
def c11():
    from spikingjelly.clock_driven import functional
    from energy import FiringRateRecorder
    from neurons import PopulationFLIF
    m = build('neo_pop_both'); m.eval()
    # the population soma is far less excitable than a single f-LIF (slow branches 4-32, 1/K mix): with re-drawn input
    # weights it stays silent (out rate 0.000 at 3x and 7x the c5 scale), so the test uses the 1-epoch smoke checkpoint
    # of neo_pop_both (its neurons fire at ~1%), as stage 4's c10 did for its causality test
    import glob
    ck = sorted(glob.glob(os.path.join(HERE, 'log', os.environ.get('SMOKE_SUITE', 'smoke-cls2-*'), 'MITBIH', '*', '*+neo_pop_both+*',
                                       '*_neopop_flif-both-a0.5', 'model_state', 'best+model.pt')))
    if not ck:
        raise FileNotFoundError('no smoke checkpoint of neo_pop_both (run make_queue_cls.sh smoke-cls2-<date> 1 first)')
    m.load_state_dict(torch.load(ck[-1], map_location='cuda'))
    mods = [m.encoding_neo.emb_lif.neuron] + [l.lif.neuron for l in m.time_block.Mem]
    ok_type = all(isinstance(n, PopulationFLIF) and n.num_members == 4 and n.has_soma and n.tau.tolist() == [4., 8., 16., 32.]
                  and abs(float(n.soma.tau_s) - 2.0) < 1e-9 for n in mods)
    g = torch.Generator().manual_seed(11)
    from dataloader import create_loader
    te, _ = create_loader(train=False, batch_size=64, num_classes=5, data_root=DATA + '/5class_mitbih_test_3_187.csv', data_window_size=187, num_workers=0)
    xb = next(iter(te))[0].float().cuda()[:32]                          # [32, 1, 187] real beats
    xp = m.padding_patch_layer(xb[:, 0]).unfold(-1, m.patch_size, m.stride)   # [32, 11, 32]
    tok = xp.transpose(0, 1).unsqueeze(2).contiguous()                  # [11, 32, 1, 32]
    def run(t):
        with torch.no_grad():
            e, o = m.neo_forward(t)
        functional.reset_net(m)
        return e, o
    e0, o0 = run(tok)
    binary = bool(((o0 == 0) | (o0 == 1)).all() and ((e0 == 0) | (e0 == 1)).all())
    rates = [e0.mean().item(), o0.mean().item()]
    causal = []
    for j in (2, 5, 8):
        t2 = tok.clone(); t2[j] = tok[j].roll(1, dims=0)                  # token j of another beat
        e1, o1 = run(t2)
        causal.append(torch.equal(o1[:j], o0[:j]) and torch.equal(e1[:j], e0[:j]) and not torch.equal(o1[j:], o0[j:]))
    rec = FiringRateRecorder(m)
    with torch.no_grad():
        m.train_mode = 'testing'; m(torch.rand(16, 1, 187).cuda()); rec.add_samples(16); functional.reset_net(m)
    rec.detach()
    kinds = {r['kind'] for r in rec.layers() if r['path'] == 'neo'}
    # non-vacuous = spikes at tokens >= j actually changed (in `causal`); the rate bounds only reject a dead layer
    ok = ok_type and binary and all(causal) and 0.001 <= rates[0] <= 0.98 and 0.005 <= rates[1] <= 0.98 and tuple(o0.shape) == (11, 32, 1, 384) and kinds == {'pop_flif'}
    report('c11', ok, f'smoke ckpt {os.path.relpath(ck[-1], HERE)[:60]}...; types/K/taus/soma ok {ok_type}, binary {binary}, shape {tuple(o0.shape)}, causal {causal}, rates emb {rates[0]:.4f} out {rates[1]:.4f}, energy kinds {sorted(kinds)}')


@check('c12')
def c12():
    from spikingjelly.clock_driven import functional
    import config as cfg
    msgs, oks = [], []
    for c in ('bidir_lif_both', 'bidir_flif_both'):
        m = build(c); m.eval(); make_fire(m)
        own = (m.encoding_neo_rev.emb_linear.weight.data_ptr() != m.encoding_neo.emb_linear.weight.data_ptr()
               and all(a.lif is not b.lif for a, b in zip(m.time_block.Mem, m.time_block_rev.Mem))
               and not torch.equal(m.time_block.Mem[0].w_in.weight, m.time_block_rev.Mem[0].w_in.weight))   # BN params are equal at init by design
        g = torch.Generator().manual_seed(12)
        tok = torch.randn(11, 32, 1, 32, generator=g).cuda()
        def branches(t):
            with torch.no_grad():
                f = m.time_block(m.encoding_neo(t)); r = m.time_block_rev(m.encoding_neo_rev(t.flip(0))).flip(0)
                _, o = m.neo_forward(t)
            functional.reset_net(m)
            return f, r, o
        f0, r0, o0 = branches(tok)
        fused = torch.equal(o0, torch.maximum(f0, r0))
        both = bool(((f0 == 1) & (r0 == 1)).any())                          # OR differs from sum somewhere
        causal = []
        for j in (2, 5, 8):
            t2 = tok.clone(); t2[j] = torch.randn(32, 1, 32, generator=g).cuda()
            f1, r1, _ = branches(t2)
            causal.append(torch.equal(f1[:j], f0[:j]) and not torch.equal(f1[j:], f0[j:])
                          and torch.equal(r1[j + 1:], r0[j + 1:]) and not torch.equal(r1[:j + 1], r0[:j + 1]))
        rates = [f0.mean().item(), r0.mean().item()]
        ok = own and fused and all(causal) and all(0.02 <= x <= 0.98 for x in rates)
        oks.append(ok); msgs.append(f'{c}: own params {own}, fused==max {fused}, overlap exists {both}, fwd>=j & rev<=j {causal}, rates {rates[0]:.3f}/{rates[1]:.3f}')
    tmp = tempfile.mkdtemp(dir=os.environ.get('CHECK_TMP'))
    flags = {'neo_pop_both': ['--neo_neuron', 'pop_flif'], 'bidir_lif_both': ['--neo_bidir', '--neo_neuron', 'lif'],
             'bidir_flif_both': ['--neo_bidir', '--neo_neuron', 'flif'], 'bidir_pop_both': ['--neo_bidir', '--neo_neuron', 'pop_flif'],
             'neo_flif_both': ['--neo_neuron', 'flif']}
    names, argv0 = {}, sys.argv
    try:
        for c, f in flags.items():
            sys.argv = ['train.py', '--log_dir', tmp, '--seed', '42', '-s'] + f
            a = cfg.Config(); a.set_args(cfg.parse_arguments()); names[c] = os.path.basename(a.save_result_path)
    finally:
        sys.argv = argv0
    distinct = len(set(names.values())) == len(names) and names['neo_flif_both'] == 'seed42_rec0_tau2.0_wmfull_neoflif-both-a0.5'
    report('c12', all(oks) and distinct, '; '.join(msgs) + f"; dirs distinct {distinct}: " + ', '.join(f'{k}={v}' for k, v in names.items() if k != 'neo_flif_both'))


@check('c13')
def c13():
    from spikingjelly.clock_driven import functional
    from energy import FiringRateRecorder
    from load_model import LOAD_MODEL
    from config import set_random_seed
    import config as cfg
    HPm = dict(HP, model='neoMixer')
    def bm(**kw):
        set_random_seed(0)
        return LOAD_MODEL['neoMixer'](NS(**{**HPm, **kw}), True).cuda()
    g = torch.Generator().manual_seed(13)
    msgs, oks = [], []
    # (a) structure and axes
    m = bm(mixer='pop_flif', mixer_bidir=True); m.eval()
    no_attn = not any('attn' in type(mod).__name__.lower() or 'attention' in type(mod).__name__.lower() for mod in m.modules())
    rec = FiringRateRecorder(m)
    with torch.no_grad():
        m.train_mode = 'testing'; out = m(torch.rand(16, 1, 187).cuda()); rec.add_samples(16); functional.reset_net(m)
    rec.detach()
    rows = rec.layers()
    axes_ok = m.num_patches == 11 and m.T == 11 and all(r['T'] == 11 for r in rows) and len(rows) == 7     # emb + 2x2 mixer + fc1 + fc2
    test_tuple = isinstance(out, tuple) and len(out) == 2 and tuple(out[1].shape) == (11, 16, 1, 384)
    m.train_mode = 'training'
    with torch.no_grad():
        tr = m(torch.rand(4, 1, 187).cuda()); functional.reset_net(m)
    train_plain = torch.is_tensor(tr) and tuple(tr.shape) == (4, 5)
    oks.append(no_attn and axes_ok and test_tuple and train_plain and not hasattr(m, 'time_block'))
    msgs.append(f'no attention {no_attn}, N=T=11 & 7 spiking layers at T=11 {axes_ok}, test (z, h) {test_tuple}, train plain logits {train_plain}')
    # (b) 'none' mixer: tokens independent before the mean
    m0 = bm(mixer='none'); m0.eval()
    with torch.no_grad():                                                  # make the embedding fire in eval (BN running stats = identity)
        w = m0.encoding.emb_linear.weight; w.copy_(torch.randn(w.shape, generator=g) * 2.0 / w.shape[1] ** .5)
        for blk in m0.blocks:
            for l in (blk.fc1, blk.fc2):
                l.linear.weight.copy_(torch.randn(l.linear.weight.shape, generator=g) * 3.0 / l.linear.weight.shape[1] ** .5)
    x = torch.rand(32, 1, 187, generator=g).cuda()
    def parts(mm, t_fn=None):
        with torch.no_grad():
            z, e, mo, f, h = mm.forward_parts(x, tok_fn=t_fn)
        functional.reset_net(mm)
        return z, e, mo, f, h
    _, e0, _, _, h0 = parts(m0)
    indep = []
    for j in (2, 5, 8):
        def tf(t, j=j):
            t = t.clone(); t[j] = t[j].roll(1, dims=0); return t
        _, e1, _, _, h1 = parts(m0, tf)
        indep.append(torch.equal(h1[:j], h0[:j]) and not torch.equal(h1[j], h0[j]) and not torch.equal(h1[j + 1:], h0[j + 1:]))
    rate0 = h0.mean().item()
    oks.append(all(indep) and 0.02 <= rate0 <= 0.98)
    msgs.append(f"'none' (aligned, no mixer): token j changes tokens >= j only, incl. later ones via the LIF state {indep} (rate {rate0:.3f})")
    # (c) bidirectional mixer causality per stack, fused = max
    m2 = bm(mixer='flif', mixer_bidir=True); m2.eval()
    with torch.no_grad():
        w = m2.encoding.emb_linear.weight; w.copy_(torch.randn(w.shape, generator=g) * 2.0 / w.shape[1] ** .5)
        for l in m2.blocks[0].mixer.Mem + m2.blocks[0].mixer_rev.Mem:
            l.w_in.weight.copy_(torch.randn(l.w_in.weight.shape, generator=g) * 5.0 / l.w_in.weight.shape[1] ** .5)
    blk = m2.blocks[0]
    with torch.no_grad():
        e0 = m2.encoding(m2.tokens(x))
    functional.reset_net(m2)
    def stacks(e):                                                         # the mixer stacks on a given embedding
        with torch.no_grad():
            f = blk.mixer(e); r = blk.mixer_rev(e.flip(0)).flip(0); fused = blk.mix(e)
        functional.reset_net(m2)
        return f, r, fused
    f0, r0, fu0 = stacks(e0)
    fused_ok = torch.equal(fu0, torch.maximum(f0, r0))
    caus = []
    for j in (2, 5, 8):
        e1 = e0.clone(); e1[j] = e0[j].roll(1, dims=0)                      # embedding token j of another beat
        f1, r1, _ = stacks(e1)
        caus.append(torch.equal(f1[:j], f0[:j]) and not torch.equal(f1[j:], f0[j:])
                    and torch.equal(r1[j + 1:], r0[j + 1:]) and not torch.equal(r1[:j + 1], r0[:j + 1]))
    rates = [f0.mean().item(), r0.mean().item()]
    oks.append(fused_ok and all(caus) and all(0.02 <= v <= 0.98 for v in rates))
    msgs.append(f'bidir f-LIF mixer: fused==max {fused_ok}, fwd>=j & rev<=j {caus}, rates {rates[0]:.3f}/{rates[1]:.3f}')
    # (d) run-directory names
    tmp = tempfile.mkdtemp(dir=os.environ.get('CHECK_TMP'))
    flags = {'mix_none_fwd': ['--mixer', 'none'], 'mix_pop_bidir': ['--mixer', 'pop_flif', '--mixer_bidir'], 'mix_pop_fwd': ['--mixer', 'pop_flif'],
             'mix_flif_bidir': ['--mixer', 'flif', '--mixer_bidir'], 'mix_lif_bidir': ['--mixer', 'lif', '--mixer_bidir'],
             'mix_pop_bidir_L2': ['--mixer', 'pop_flif', '--mixer_bidir', '--layers', '2']}
    names, argv0 = {}, sys.argv
    try:
        for c, f in flags.items():
            sys.argv = ['train.py', '--model', 'neoMixer', '--log_dir', tmp, '--seed', '42', '-s'] + f
            a = cfg.Config(); a.set_args(cfg.parse_arguments()); names[c] = os.path.basename(a.save_result_path)
    finally:
        sys.argv = argv0
    oks.append(len(set(names.values())) == len(names))
    msgs.append('dirs ' + ', '.join(f'{k}={v}' for k, v in names.items()))
    report('c13', all(oks), '; '.join(msgs))


def _rd_args(**kw):
    base = dict(data='CINC2017', crop_tokens=0, patch_size=64, fs_target=125., fold=0, val_frac=0.1, win_s=30., hop_s=15.)
    base.update(kw)
    return NS(**base)


@check('c14')
def c14():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    msgs, oks = [], []
    # (a) C = 1: stage-9 token path vs original path
    set_random_seed(0)
    a = NS(**dict(HP, patch_size=32, seq_len=187, num_channels=1))
    m = LOAD_MODEL['myModel'](a, True).cuda().eval(); m.train_mode = 'testing'
    g = torch.Generator().manual_seed(14)
    x = torch.rand(8, 1, 187, generator=g).cuda()
    from spikingjelly.clock_driven import functional
    with torch.no_grad():
        z0 = m(x)[0]; functional.reset_net(m)
        hip, neo = m._tokens_mc(x)
        xr = m.padding_patch_layer(x.unsqueeze(0).repeat(m.T, 1, 1, 1).flatten(0, 1).reshape(m.T, 8, -1)).unfold(-1, m.patch_size, m.stride)
        m.in_channels_force = True
        orig_forward_flag = m.in_channels; m.in_channels = 2                     # route through _tokens_mc (C is still 1 in x)
        z1 = m(x)[0]; functional.reset_net(m); m.in_channels = orig_forward_flag
    same_inputs = torch.equal(hip, xr)
    ok_a = same_inputs and torch.allclose(z0, z1, atol=1e-5)
    oks.append(ok_a); msgs.append(f'C=1: token tensors equal {same_inputs}, logits max diff {(z0 - z1).abs().max().item():.1e}')
    # (b) C = 700, myModel and NeoMixer
    for name in ('myModel', 'neoMixer'):
        set_random_seed(0)
        kw = dict(HP, patch_size=4, seq_len=50, num_channels=7, model=name, mixer='lif', mixer_bidir=True, neo_alpha=0.5, pop_taus='4,8,16,32')
        mm = LOAD_MODEL[name](NS(**kw), True).cuda()
        xb = torch.rand(3, 7, 50, generator=g).cuda()
        tok = (mm._tokens_mc(xb)[1] if name == 'myModel' else mm.tokens(xb))[:, :, 0]      # [N, B, C*P]
        P, S = mm.patch_size, mm.stride
        xpad = torch.cat([xb, xb[:, :, -1:].repeat(1, 1, S)], dim=2)
        good = True
        for n in range(tok.shape[0]):
            manual = xpad[:, :, n * S:n * S + P].reshape(3, -1)                          # channel-major
            good &= torch.equal(tok[n], manual)
        oks.append(good and tok.shape[0] == mm.num_patches)
        msgs.append(f'{name} C=7: {tok.shape[0]} tokens, all equal manual slices {bool(good)}')
    report('c14', all(oks), '; '.join(msgs))


@check('c15')
def c15():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    set_random_seed(0)
    m = LOAD_MODEL['myModel'](NS(**dict(HP, patch_size=64, seq_len=3750, num_channels=1, hippo_ssa=True, hippo_T=4)), True).cuda()
    seen = {}
    h = m.data_block[0].ssa.register_forward_hook(lambda mod, i, o: seen.__setitem__('in', tuple(i[0].shape)))
    from spikingjelly.clock_driven import functional
    with torch.no_grad():
        out = m(torch.rand(2, 1, 3750).cuda()); functional.reset_net(m)
    h.remove()
    N = m.num_patches
    absent = not any(hasattr(m, a) for a in ('encoding_neo', 'time_block', 'weak_decoder')) and not hasattr(m.data_block[0], 'mca')
    ok = absent and m.T_h == 4 and seen.get('in') == (4, 2, N, 384) and tuple(m.data_block[0].ssa.relative_bias_table.shape) == (2 * N - 1, 8) \
        and isinstance(out, tuple) and not hasattr(m, 'time_block')
    report('c15', ok, f'no neocortex/mca/decoder {absent}, T_h {m.T_h}, SSA input {seen.get("in")}, rel-bias {tuple(m.data_block[0].ssa.relative_bias_table.shape)}, N {N}')


@check('c16')
def c16():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from layers import NeoNeuron
    from neurons import PopulationFLIF
    import rhythm_data as rd
    from spikingjelly.clock_driven import functional
    d = rd.build(_rd_args(), 42)
    xb = torch.stack([d['train'][i][0] for i in range(16)]).cuda()
    rates, wts = {}, {}
    for cal in (None, 1.0, 'lif'):
        for name in ('myModel', 'neoMixer'):
            set_random_seed(0)
            kind = 'lif' if cal == 'lif' else 'pop_flif'
            kw = dict(HP, patch_size=64, seq_len=3750, num_channels=1, model=name, neo_neuron=kind, neo_alpha=0.5, pop_taus='4,8,16,32',
                      mixer=kind, mixer_bidir=True, pop_soma_init=(None if cal == 'lif' else cal))
            if name == 'neoMixer':
                kw.pop('neo_neuron')
            m = LOAD_MODEL[name](NS(**kw), True).cuda().train(); m.train_mode = 'training'   # BN with batch statistics
            pops = [mod.neuron for mod in m.modules() if isinstance(mod, NeoNeuron) and isinstance(mod.neuron, PopulationFLIF)]
            wts[(cal, name)] = sorted({round(float(v), 4) for p in pops for v in p.soma.weight.flatten().unique().tolist()})
            with torch.no_grad():
                if name == 'myModel':
                    _, neo = m._tokens_mc(xb) if False else (None, m.padding_patch_layer(xb[:, 0]).unfold(-1, 64, 32).transpose(0, 1).unsqueeze(2))
                    _, o = m.neo_forward(neo)
                else:
                    e = m.encoding(m.tokens(xb)); o = m.blocks[0].mix(e)
            functional.reset_net(m)
            rates[(cal, name)] = o.float().mean().item()
    ok = all(wts[(1.0, n)] == [1.0] for n in ('myModel', 'neoMixer')) and all(wts[(None, n)] == [0.25] for n in ('myModel', 'neoMixer')) \
        and all(rates[(1.0, n)] > 0 and rates[(1.0, n)] > rates[(None, n)] for n in ('myModel', 'neoMixer'))
    report('c16', ok, 'soma weights ' + ', '.join(f'{k}={v}' for k, v in wts.items()) + '; init firing on real beats ' +
           ', '.join(f'{k}={v:.4f}' for k, v in rates.items()))


@check('c17')
def c17():
    import numpy as np
    import rhythm_data as rd
    import rhythm_eval as rev
    sig, off, lab, names = rd.cinc_cache(125.)
    ref = [l.strip().split(',') for l in open(os.path.join(rd.CINC_DIR, 'REFERENCE-v3.csv')) if l.strip()]
    ref_counts = np.bincount([rd.CINC_LABELS[r[1]] for r in ref], minlength=4)
    tests, ok_split = [], True
    for f in range(5):
        tr, va, te = rd.cinc_split(lab, f)
        ok_split &= not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te)) and len(tr) + len(va) + len(te) == len(lab)
        tests.append(set(te))
    cover = set().union(*tests) == set(range(len(lab))) and sum(len(t) for t in tests) == len(lab)
    # aggregation: dummy model, logits = [mean, std, max, min] of the window
    class Dummy(torch.nn.Module):
        def forward(self, x):
            v = x[:, 0]
            return torch.stack([v.mean(1), v.std(1), v.max(1).values, v.min(1).values], 1)
    d = rd.build(_rd_args(), 42)
    sub = torch.utils.data.Subset(d['test'], list(range(40)))
    P, Y, U = rev.predict(Dummy().cuda(), sub, 'cuda', bs=7)
    man = {}
    for i in range(40):
        x, y, r = d['test'][i]
        man.setdefault(r, []).append(torch.softmax(Dummy()(x.unsqueeze(0).float()), 1)[0].numpy())
    agg_ok = all(np.allclose(P[k], np.mean(man[u], 0), atol=1e-6) for k, u in enumerate(U)) and len(U) == len(man)
    ok = len(lab) == 8528 and (np.bincount(lab, minlength=4) == ref_counts).all() and ok_split and cover and agg_ok
    report('c17', ok, f'records {len(lab)}, counts {np.bincount(lab, minlength=4).tolist()} = v3 {ref_counts.tolist()}, folds disjoint {ok_split}, '
           f'test folds cover all {cover}, window aggregation matches manual {agg_ok} ({len(U)} records from 40 windows)')


@check('c18')
def c18():
    import numpy as np
    import rhythm_data as rd
    z = np.load(rd.SHD_NPZ)
    sums_tr = z['x_train'].reshape(len(z['y_train']), -1).sum(1, dtype=np.int64)
    sums_te = z['x_test'].reshape(len(z['y_test']), -1).sum(1, dtype=np.int64)
    counts_ok = (sums_tr == z['n_spikes_train']).all() and (sums_te == z['n_spikes_test']).all()
    oks = []
    for seed in (42, 7):
        d = rd.build(_rd_args(data='SHD', patch_size=2), seed)
        tr, va = set(d['train'].idx.tolist()), set(d['val'].idx.tolist())
        oks.append(not (tr & va) and len(tr) + len(va) == 8156 and len(np.unique(z['y_train'][list(va)])) == 20)
    ok = len(z['y_train']) == 8156 and len(z['y_test']) == 2264 and len(np.unique(z['y_train'])) == 20 and counts_ok and all(oks)
    report('c18', ok, f'train {len(z["y_train"])}, test {len(z["y_test"])}, classes {len(np.unique(z["y_train"]))}, binned sums = kept spikes {bool(counts_ok)}, '
           f'val disjoint & all classes (seeds 42, 7) {oks}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'checks', 'check_cls.txt'))
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    for fn in (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18):
        if a.only and fn.__name__ not in a.only:
            continue
        fn()
    n_ok = sum(ok for _, ok, _ in RESULTS)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        for name, ok, msg in RESULTS:
            f.write(f"{'PASS' if ok else 'FAIL'} {name}: {msg}\n")
        f.write(f'{n_ok}/{len(RESULTS)} passed\n')
    print(f'{n_ok}/{len(RESULTS)} passed -> {a.out}')
    sys.exit(0 if n_ok == len(RESULTS) else 1)
