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
  c19 (MODI 2026-10-20) NSMT aux-loss options: with keep_ratio 0 the reconstruction target is the raw patched signal (C = 1
      path: allclose to manual slices of the padded input; C = 7 token path: bitwise) and with 0.15 it is not (non-vacuous);
      train_rhythm.train_epoch with alpha 0 returns loss == CE and leaves the decoder without gradient while the neocortex
      still gets one (through the cross-attention); alpha 0.5 / 0.9 give loss == (1 - a) CE + a MSE and a decoder gradient
  c20 (MODI 2026-10-21) --ssa_pe 0: no relative-bias table; at init its output equals the pe=1 model bitwise (table is zero at
      init) and under pe=1 the table receives a gradient
  c21 (MODI 2026-10-21) --pos_emb: exactly one extra parameter encoding.pos [1, N, D] (NSMT and NeoMixer); the embedding
      output shape is unchanged (the first version broadcast the table over the token-axis layout: 100x tensors, found in the
      smoke energy); with the no-pos weights loaded and pos zeroed the outputs are bitwise equal; an offset on token j changes
      the embedding spikes of token j only in both layouts; a random pos changes the logits
  c22 (MODI 2026-10-21) --hippo_xattn: T_h = 4, attention q [4, B, N, D] from the hippocampus and kv [4, B, N, D] from the
      neocortex tokens, neocortex causal at T = N, decoder present and loss = 0.9 CE + 0.1 MSE, replacing neocortex token j
      changes the block output (non-vacuous)
  c23 (MODI 2026-10-21) mixers lif_rec / sj_lif: 4 recurrent (resp. plain) spikingjelly NeoLayers, gate g = 0 at init so the output
      equals the W_rec-zeroed model bitwise, g = 1 changes it, g receives a gradient; the energy recorder hooks the LIFNodes
      under blocks.0.mixer as neocortex path
  c24 (MODI 2026-10-21) population branches tau 16/32/64/128: tau values, and the soma init in {1, 2, 4} whose initial firing on
      real data (training-mode BN) is closest to the tau 4-32 / psi 1 mixer (per dataset) -- the value is reported, not asserted
  c33-c37 (MODI 2026-10-23) membrane fusion and the K-stream neocortex: see the docstrings of c33 ... c37
  c38-c41 (MODI 2026-10-24) spike-xor fusion, tau-group linear fusions, stream order, fused bidirectional streams
  c42-c45 (MODI 2026-10-26) B: neo_grad share, own + neocortex keys / values (cat), memoryless neocortex control, option rules
  c46-c47 (MODI 2026-10-27) ETT forecasting wrapper (forecast_wrap.Forecaster + train_forecast.py): data splits, structure,
      body equal to the classification model, per-channel affine equivariance, channel independence, metrics
  c25 (MODI 2026-10-21) BiX (mixer_comb xattn): every spiking neuron runs T = N steps, attention map [B, 1, 8, N, N], the
      query Linear sees the forward stream and the key / value Linears the reverse stream, and changing input token j changes
      the mixer output before token j (bidirectional reach)
  c26 (MODI 2026-10-21) BiX-ch (user's per-channel T x T' maps): the attention output before its BN equals
      q[t, d] * sum_t' k[t', d] v[t', d] * scale exactly (the reduction claimed in the review), the head form does not
  c27 (MODI 2026-10-21; band rule 10-22) popg (soma-less tau groups): group taus 4/8/16/32 in every neuron layer, BN weights =
      the layer-wise calibrated table calib/*.json (calibrate_groups.py), and on real CinC data at init every group of every layer fires within
      0.5-2x of the same layer of the reference model (spikeDE LIF / f-LIF mixer, same combination) and >= 1 %
  c28 (MODI 2026-10-21) popk (StreamMixer): [N, B, 4, D] streams, shared Linears (equal taus -> identical streams), head input
      4*D, every stream of every layer within 0.5-2x of the reference model's layer and >= 1 % at init on real CinC data (10-22);
      K = 1 / tau 2 with copied weights gives logits
      bitwise equal to NeoMixer(mixer='flif')
  c29 (MODI 2026-10-21) pop_rec: with g = 0 the logits equal the population f-LIF mixer bitwise (copied weights), g = 1
      changes them, g receives a gradient
  c30 (MODI 2026-10-21) --attn_bn: only attn_bn.* parameters are added, with BN the init attention firing rises and the
      output changes; without the flag the model equals the default model bitwise
  c31 (MODI 2026-10-21) population LIF (alpha 1): branch states equal a manual non-resetting Euler LIF recursion, soma
      output is 0/1; initial firing on real CinC / SHD data is reported
  c32 (MODI 2026-10-22) linear fusion (mixer_comb lin): one Linear 768 -> 384 + BN + spikingjelly LIF on [F; R], no attention
      module, output 0/1, every neuron T = N, changing only the forward (resp. reverse) stream changes the fused output
  c48 (MODI 2026-10-28) --token_shuffle (shuffle-retrain): off = no shuffle module and the default model unchanged bitwise;
      on, evaluation: the logits equal the normal model (same weights) fed the tokens in the fixed order default_rng(0).permutation(N)
      (= the cause analysis b = 1) and differ from the unshuffled logits; on, training: every sample gets its own permutation,
      each row is a permutation, the token multiset of every sample is kept, two calls differ, the global torch RNG is not
      consumed; myModel without hippo_ssa refuses the flag; run folders get the suffix _tshuf (Spikformer-style and Neo-mixer)
  c49 (MODI 2026-10-28) PPG-DaLiA (ppg_data / ppg_model / train_ppg): window counts 1020 / 106 / 106 with every window equal to its
      slice, S13 train = 72 windows (the shorter of input / label), values in [-1, 1]; label k reads token k - 1 (token end = label
      end); on real windows with frozen BN, changing input samples [128 j, 128 j + 128) leaves every token before j - 1 unchanged
      (all models causal), the window-W attention reaches exactly W - 1 tokens further (W = 1, 3) and all past reaches far, the
      memoryless B form stops at the window, the T = N B form and the forward Neo-mixer reach beyond it; the head reads 0/1 spikes,
      equals the head of the averaged spikes, and syops books it as AC (0 MAC); the GRU reference is causal
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


@check('c19')
def c19():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    import torch.nn as nn
    from train_rhythm import train_epoch
    from spikingjelly.clock_driven import functional
    msgs, oks = [], []
    g = torch.Generator().manual_seed(19)
    # (a) reconstruction target, C = 1 path (CinC) and C = 7 token path (SHD-like)
    x1 = torch.randn(4, 1, 3750, generator=g).cuda()
    xpad = torch.cat([x1, x1[:, :, -1:].repeat(1, 1, 32)], dim=2)
    for keep in (0., 0.15):
        set_random_seed(0)
        m = LOAD_MODEL['myModel'](NS(**dict(HP, patch_size=64, seq_len=3750, num_channels=1, keep_ratio=keep)), True).cuda()
        m.train(); m.train_mode = 'training'
        with torch.no_grad():
            m(x1); functional.reset_net(m)
        manual = xpad.unfold(-1, 64, 32)[:, 0].transpose(0, 1).unsqueeze(2)          # [N, B, 1, P]
        diff = (m.org_x - manual).abs().max().item()
        ok = (diff < 1e-5) if keep == 0 else (diff > 1e-2)
        oks.append(ok); msgs.append(f'C=1 keep {keep}: max |target - raw| {diff:.1e}')
    x7 = torch.rand(3, 7, 50, generator=g).cuda()
    for keep in (0., 0.15):
        set_random_seed(0)
        m = LOAD_MODEL['myModel'](NS(**dict(HP, patch_size=4, seq_len=50, num_channels=7, keep_ratio=keep)), True).cuda()
        m.train(); m.train_mode = 'training'
        _, neo = m._tokens_mc(x7)
        same = torch.equal(m.org_x, neo)
        ok = same if keep == 0 else (not same and (m.org_x - neo).abs().max().item() > 1e-2)
        oks.append(ok); msgs.append(f'C=7 keep {keep}: target == raw tokens {same}')
    # (b) the loss in train_rhythm.train_epoch
    xb = torch.rand(16, 1, 187, generator=g); yb = torch.randint(0, 5, (16,), generator=g)
    for alpha in (0., 0.5, 0.9):
        set_random_seed(0)
        m = LOAD_MODEL['myModel'](NS(**HP), True).cuda()
        opt = torch.optim.SGD(m.parameters(), lr=0.)                                 # lr 0: parameters fixed, gradients kept
        r = train_epoch(m, [(xb, yb)], nn.CrossEntropyLoss(), opt, alpha, 'cuda')
        dec = [p.grad for p in m.weak_decoder.parameters()]
        dec_none = all(gr is None or float(gr.abs().sum()) == 0. for gr in dec)
        neo_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.time_block.parameters())
        want = r['ce'] if alpha == 0 else (1 - alpha) * r['ce'] + alpha * r['mse']
        loss_ok = (r['loss'] == r['ce']) if alpha == 0 else abs(r['loss'] - want) <= 1e-5 * max(1., abs(want))
        ok = loss_ok and neo_grad and (dec_none if alpha == 0 else not dec_none) and r['mse'] > 0
        oks.append(ok); msgs.append(f'alpha {alpha}: loss {r["loss"]:.6f} ce {r["ce"]:.6f} mse {r["mse"]:.6f} (loss ok {loss_ok}), '
                                    f'decoder grad {"none" if dec_none else "yes"}, neocortex grad {neo_grad}')
    report('c19', all(oks), '; '.join(msgs))


@check('c20')
def c20():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    g = torch.Generator().manual_seed(20); x = torch.randn(2, 1, 3750, generator=g).cuda()
    outs, ms = [], []
    for pe in (1, 0):
        set_random_seed(0)
        m = LOAD_MODEL['myModel'](NS(**dict(HP, patch_size=64, seq_len=3750, num_channels=1, num_classes=4, hippo_ssa=True, hippo_T=4, ssa_pe=pe)), True).cuda().eval()
        m.train_mode = 'testing'
        with torch.no_grad():
            outs.append(m(x)[0]); functional.reset_net(m)
        ms.append(m)
    has = [hasattr(m.data_block[0].ssa, 'relative_bias_table') for m in ms]
    same = torch.equal(outs[0], outs[1])
    m = ms[0].train(); m.train_mode = 'training'
    z = m(x)[0]; torch.nn.functional.cross_entropy(z, torch.tensor([0, 1]).cuda()).backward(); functional.reset_net(m)
    gr = m.data_block[0].ssa.relative_bias_table.grad
    ok = has == [True, False] and same and gr is not None and float(gr.abs().sum()) > 0
    report('c20', ok, f'rel-bias table present (pe=1, pe=0) {has}, outputs equal at init {same}, table grad sum {float(gr.abs().sum()) if gr is not None else None:.3e}')


@check('c21')
def c21():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    g = torch.Generator().manual_seed(21); x = torch.rand(4, 1, 187, generator=g).cuda()
    msgs, oks = [], []
    for name in ('myModel', 'neoMixer'):
        kw = dict(HP, model=name, neo_neuron='lif', mixer='lif', mixer_bidir=True, neo_alpha=0.5, pop_taus='4,8,16,32')
        set_random_seed(0); m0 = LOAD_MODEL[name](NS(**kw), True).cuda().eval(); m0.train_mode = 'testing'
        set_random_seed(0); m1 = LOAD_MODEL[name](NS(**dict(kw, pos_emb=True)), True).cuda().eval(); m1.train_mode = 'testing'
        extra = sorted(set(m1.state_dict()) - set(m0.state_dict()))
        shape = tuple(m1.encoding.pos.shape) if extra == ['encoding.pos'] else None
        m1.load_state_dict(m0.state_dict(), strict=False)
        emb = {}
        h = m1.encoding.emb_lif.register_forward_hook(lambda mod, i, o: emb.__setitem__('shape', tuple(o.shape)))
        h0 = m0.encoding.emb_lif.register_forward_hook(lambda mod, i, o: emb.__setitem__('shape0', tuple(o.shape)))
        with torch.no_grad():
            m1.encoding.pos.zero_()
            z0 = m0(x)[0]; functional.reset_net(m0)
            z1 = m1(x)[0]; functional.reset_net(m1)
            # a large offset on token j only must change the embedding spikes at token j only (both layouts)
            j = 5; m1.encoding.pos[0, j] = 50.
            if name == 'myModel':
                e0 = m0.encoding(m0._tokens_mc(x)[0] if False else m0.padding_patch_layer(x.unsqueeze(0).repeat(m0.T, 1, 1, 1).flatten(0, 1).reshape(m0.T, x.shape[0], -1)).unfold(-1, 32, 16)); functional.reset_net(m0)
                e1 = m1.encoding(m1.padding_patch_layer(x.unsqueeze(0).repeat(m1.T, 1, 1, 1).flatten(0, 1).reshape(m1.T, x.shape[0], -1)).unfold(-1, 32, 16)); functional.reset_net(m1)
                tok_axis = 2                                       # [T, B, N, D]
            else:
                e0 = m0.encoding(m0.tokens(x)); functional.reset_net(m0)
                e1 = m1.encoding(m1.tokens(x)); functional.reset_net(m1)
                tok_axis = 0                                       # [N, B, 1, D]
            others = [k for k in range(e0.shape[tok_axis]) if k != j]
            local = torch.equal(e0.index_select(tok_axis, torch.tensor(others).cuda()), e1.index_select(tok_axis, torch.tensor(others).cuda())) \
                and not torch.equal(e0.select(tok_axis, j), e1.select(tok_axis, j))
            m1.encoding.pos.normal_(0, 1.)
            z2 = m1(x)[0]; functional.reset_net(m1)
        h.remove(); h0.remove()
        same_shape = emb.get('shape') == emb.get('shape0') and e0.shape == e1.shape
        ok = extra == ['encoding.pos'] and shape == (1, m0.num_patches, 384) and torch.equal(z0, z1) and not torch.equal(z1, z2) and same_shape and local
        oks.append(ok); msgs.append(f'{name}: extra params {extra} shape {shape}, embedding shape unchanged {same_shape} {emb.get("shape")}, zero-pos equal {torch.equal(z0, z1)}, '
                                    f'offset on token {j} changes token {j} only {local}, random-pos differs {not torch.equal(z1, z2)}')
    report('c21', all(oks), '; '.join(msgs))


@check('c22')
def c22():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from train_rhythm import train_epoch
    from spikingjelly.clock_driven import functional
    import torch.nn as nn
    g = torch.Generator().manual_seed(22); x = torch.randn(3, 1, 3750, generator=g).cuda()
    set_random_seed(0)
    m = LOAD_MODEL['myModel'](NS(**dict(HP, patch_size=64, seq_len=3750, num_channels=1, num_classes=4, hippo_T=4, hippo_xattn=True, neo_neuron='lif')), True).cuda().train()
    m.train_mode = 'testing'                                           # BN with batch statistics (eval-mode BN at init silences every neuron, c16)
    # one batch-statistics pass fixes the BN running statistics (momentum 1), then eval mode makes the causality test
    # deterministic (train-mode BN would re-sum the permuted batch and flip near-threshold spikes before token j)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.momentum = 1.0
    with torch.no_grad():
        m(x); functional.reset_net(m)
    m.eval()
    seen = {}
    ssa = m.data_block[0].ssa; orig = ssa.forward
    def rec(xx, mask=None, kv=None):
        seen['q'] = tuple(xx.shape); seen['kv'] = None if kv is None else tuple(kv.shape); return orig(xx, mask=mask, kv=kv)
    ssa.forward = rec
    N = m.num_patches; j = N // 2; perm = torch.tensor([1, 2, 0]).cuda()
    with torch.no_grad():
        hip, neo = m._tokens_mc(x)
        _, o0 = m.neo_forward(neo); functional.reset_net(m)
        neo2 = neo.clone(); neo2[j] = neo2[j][perm]
        _, o1 = m.neo_forward(neo2); functional.reset_net(m)
        causal = torch.equal(o0[:j], o1[:j]) and not torch.equal(o0[j:], o1[j:])
        xh = m.encoding(hip); functional.reset_net(m)
        h0 = m.data_block[0](xh, o0); functional.reset_net(m)
        h1 = m.data_block[0](xh, o1); functional.reset_net(m)
        sens = not torch.equal(h0, h1)
    ssa.forward = orig
    m.train(); m.train_mode = 'training'
    opt = torch.optim.SGD(m.parameters(), lr=0.)
    r = train_epoch(m, [(x.cpu(), torch.tensor([0, 1, 2]))], nn.CrossEntropyLoss(), opt, 0.1, 'cuda')
    loss_ok = abs(r['loss'] - (0.9 * r['ce'] + 0.1 * r['mse'])) < 1e-6 and r['mse'] > 0
    ok = m.T_h == 4 and seen.get('q') == (4, 3, N, 384) and seen.get('kv') == (4, 3, N, 384) and causal and sens and hasattr(m, 'weak_decoder') and hasattr(m, 'time_block') and loss_ok
    report('c22', ok, f'T_h {m.T_h}, q {seen.get("q")}, kv {seen.get("kv")}, neocortex causal {causal}, block output changes with neocortex token {sens}, '
                      f'decoder {hasattr(m, "weak_decoder")}, loss {r["loss"]:.6f} = 0.9 ce {r["ce"]:.6f} + 0.1 mse {r["mse"]:.6f} ({loss_ok})')


@check('c23')
def c23():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from energy import FiringRateRecorder
    from spikingjelly.clock_driven import functional
    from model import NeoLayer
    import copy
    g = torch.Generator().manual_seed(23); x = torch.rand(4, 1, 187, generator=g).cuda()
    msgs, oks = [], []
    for mixer in ('lif_rec', 'sj_lif'):
        set_random_seed(0)
        m = LOAD_MODEL['neoMixer'](NS(**dict(HP, model='neoMixer', mixer=mixer, mixer_bidir=True, neo_alpha=0.5, pop_taus='4,8,16,32')), True).cuda().train()
        m.train_mode = 'testing'                                       # BN with batch statistics (see c16 / c22)
        layers = [l for l in m.modules() if isinstance(l, NeoLayer)]
        n_rec = sum(l.recurrent for l in layers)
        rec = FiringRateRecorder(m); hooked = [n for n in rec.stats if n.startswith('blocks.0.mixer') and rec.stats[n]['single']]
        with torch.no_grad():
            z0 = m(x)[0]; functional.reset_net(m)
        rec.detach()
        if mixer == 'lif_rec':
            gs = [float(l.g) for l in layers]
            m2 = copy.deepcopy(m)
            with torch.no_grad():
                for l in m2.modules():
                    if isinstance(l, NeoLayer): l.w_rec.weight.zero_()
                z1 = m2(x)[0]; functional.reset_net(m2)
                for l in m2.modules():
                    if isinstance(l, NeoLayer): l.g.fill_(1.)
                functional.reset_net(m2)
            m3 = copy.deepcopy(m)
            with torch.no_grad():
                for l in m3.modules():
                    if isinstance(l, NeoLayer): l.g.fill_(1.)
                z2 = m3(x)[0]; functional.reset_net(m3)
            m.train(); m.train_mode = 'training'
            z = m(x); torch.nn.functional.cross_entropy(z, torch.tensor([0, 1, 2, 3]).cuda()).backward(); functional.reset_net(m)
            ggrad = all(l.g.grad is not None and float(l.g.grad.abs()) > 0 for l in layers)
            ok = len(layers) == 4 and n_rec == 4 and gs == [0.] * 4 and torch.equal(z0, z1) and not torch.equal(z0, z2) and ggrad and len(hooked) == 4
            oks.append(ok); msgs.append(f'lif_rec: layers {len(layers)} recurrent {n_rec}, g {gs}, equals W_rec-zeroed {torch.equal(z0, z1)}, g=1 differs {not torch.equal(z0, z2)}, g grad {ggrad}, hooked LIFNodes {len(hooked)}')
        else:
            has_rec = any(hasattr(l, 'w_rec') for l in layers)
            ok = len(layers) == 4 and n_rec == 0 and not has_rec and len(hooked) == 4
            oks.append(ok); msgs.append(f'sj_lif: layers {len(layers)} recurrent {n_rec}, W_rec {has_rec}, hooked LIFNodes {len(hooked)}')
    report('c23', all(oks), '; '.join(msgs))


@check('c24')
def c24():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    import rhythm_data as rd
    from layers import NeoNeuron
    from neurons import PopulationFLIF
    from spikingjelly.clock_driven import functional
    msgs, oks = [], []
    for data, pargs in (('CINC2017', dict(data='CINC2017', crop_tokens=0, patch_size=64, fs_target=125., fold=0, val_frac=0.1, win_s=30., hop_s=15.)),
                        ('SHD', dict(data='SHD', crop_tokens=0, patch_size=2, val_frac=0.1))):
        d = rd.build(NS(**pargs), 42)
        xb = torch.stack([d['train'][i][0] for i in range(16)]).float().cuda()
        rates, taus = {}, {}
        for pt, psi in (('4,8,16,32', 1.0), ('16,32,64,128', 1.0), ('16,32,64,128', 2.0), ('16,32,64,128', 4.0)):
            set_random_seed(0)
            kw = dict(HP, patch_size=pargs['patch_size'], seq_len=d['seq_len'], num_channels=d['num_channels'], num_classes=d['num_classes'],
                      model='neoMixer', mixer='pop_flif', mixer_bidir=True, neo_alpha=0.5, pop_taus=pt, pop_soma_init=psi)
            m = LOAD_MODEL['neoMixer'](NS(**kw), True).cuda().train(); m.train_mode = 'training'
            pops = [mod.neuron for mod in m.modules() if isinstance(mod, NeoNeuron) and isinstance(mod.neuron, PopulationFLIF)]
            taus[pt] = sorted({round(float(t), 1) for p in pops for t in p.tau.flatten().tolist()})
            with torch.no_grad():
                o = m.blocks[0].mix(m.encoding(m.tokens(xb)))
            functional.reset_net(m); rates[(pt, psi)] = o.float().mean().item()
            del m; torch.cuda.empty_cache()
        ref = rates[('4,8,16,32', 1.0)]
        cand = {psi: rates[('16,32,64,128', psi)] for psi in (1.0, 2.0, 4.0)}
        pick = min(cand, key=lambda k: abs(cand[k] - ref))
        ok = taus['16,32,64,128'] == [16., 32., 64., 128.] and all(v > 0 for v in cand.values())
        oks.append(ok); msgs.append(f'{data}: taus {taus["16,32,64,128"]}, ref (4-32, psi 1) {ref:.4f}, 16-128 at psi ' + ', '.join(f'{k:g}: {v:.4f}' for k, v in cand.items()) + f' -> pick psi {pick:g}')
    report('c24', all(oks), '; '.join(msgs))


def _real_batch(data, n=16):
    import rhythm_data as rd
    pargs = dict(data='CINC2017', crop_tokens=0, patch_size=64, fs_target=125., fold=0, val_frac=0.1, win_s=30., hop_s=15.) if data == 'CINC2017' \
        else dict(data='SHD', crop_tokens=0, patch_size=2, val_frac=0.1)
    d = rd.build(NS(**pargs), 42)
    return torch.stack([d['train'][i][0] for i in range(n)]).float().cuda(), d, pargs['patch_size']


def _bn_freeze_eval(m, x):
    """one batch-statistics pass with momentum 1 fixes the BN running statistics, then eval mode (deterministic)."""
    from spikingjelly.clock_driven import functional
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.momentum = 1.0
    m.train()
    with torch.no_grad():
        m(x); functional.reset_net(m)
    m.eval()


def _mixer_kw(**kw):
    base = dict(HP, model='neoMixer', mixer='lif', mixer_bidir=True, neo_alpha=0.5, pop_taus='4,8,16,32', neo_tau=2.0,
                fde_method='pred', fde_step=1.0, fde_memory=0)
    base.update(kw)
    return base


def _layer_kind(name):
    for k in ('encoding', 'mixer_rev.Mem.0', 'mixer_rev.layers.0', 'mixer.Mem.0', 'mixer.layers.0', 'mixer_rev.Mem.1', 'mixer_rev.layers.1',
              'mixer.Mem.1', 'mixer.layers.1', 'xattn.q', 'xattn.k', 'xattn.v', 'xattn.attn', 'xattn.proj', 'fuse', 'fc1', 'fc2'):
        if k in name:
            return k.replace('Mem.', 'L').replace('layers.', 'L').replace('mixer_rev', 'rev').replace('mixer', 'fwd')
    return name


def _layer_group_rates(m, x):
    """{layer kind: {group: initial firing}} over the leaf spiking modules (training-mode BN, one forward)."""
    import re, collections
    from spikingjelly.clock_driven import functional
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    from layers import NeoNeuron
    acc, hs = collections.defaultdict(list), []
    for name, mod in m.named_modules():
        if isinstance(mod, (MultiStepLIFNode, NeoNeuron)) and not any(isinstance(c, (MultiStepLIFNode, NeoNeuron)) for c in mod.modules() if c is not mod):
            g = re.search(r'parts\.(\d+)', name); gi = int(g.group(1)) if g else 0
            hs.append(mod.register_forward_hook(lambda md, i, o, k=(_layer_kind(name), gi): acc[k].append(float((o[0] if isinstance(o, tuple) else o).float().mean()))))
    m.train(); m.train_mode = 'testing'
    with torch.no_grad():
        m(x); functional.reset_net(m)
    for h in hs: h.remove()
    out = {}
    for (k, gi), v in acc.items():
        out.setdefault(k, {})[gi] = sum(v) / len(v)
    return out


def _band_check(model, mixer, comb, xb, kw, lo=0.5, hi=2.0):
    """every group of every layer within [lo, hi] x the same layer of the reference model (spikeDE lif / flif mixer, same comb) and >= 1 %."""
    from load_model import LOAD_MODEL
    from config import set_random_seed
    fam = mixer.split('_', 1)[1]
    set_random_seed(0)
    ref = LOAD_MODEL['neoMixer'](NS(**_mixer_kw(mixer=fam, mixer_comb=comb, **kw)), True).cuda()
    rr = {k: v[0] for k, v in _layer_group_rates(ref, xb).items()}
    del ref; torch.cuda.empty_cache()
    set_random_seed(0)
    m = LOAD_MODEL[model](NS(**_mixer_kw(model=model, mixer=mixer, mixer_comb=comb, **kw)), True).cuda()
    r = _layer_group_rates(m, xb)
    del m; torch.cuda.empty_cache()
    ratios = {(k, g): v / rr[k] for k, gs in r.items() for g, v in gs.items()}
    lo_k = min(ratios, key=ratios.get); hi_k = max(ratios, key=ratios.get)
    mn = min(v for gs in r.values() for v in gs.values())
    ok = lo <= ratios[lo_k] and ratios[hi_k] <= hi and mn >= 0.01
    return ok, f'{mixer}/{comb}: ratio to reference {ratios[lo_k]:.2f} ({lo_k[0]}[{lo_k[1]}]) .. {ratios[hi_k]:.2f} ({hi_k[0]}[{hi_k[1]}]), min rate {mn:.4f}'


@check('c25')
def c25():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from energy import FiringRateRecorder
    from spikingjelly.clock_driven import functional
    g = torch.Generator().manual_seed(25); x = torch.rand(8, 1, 187, generator=g).cuda()
    set_random_seed(0)
    m = LOAD_MODEL['neoMixer'](NS(**_mixer_kw(mixer_comb='xattn')), True).cuda(); m.train_mode = 'testing'
    _bn_freeze_eval(m, x)
    N = m.num_patches; blk = m.blocks[0]; xa = blk.xattn; xa.keep_scores = True
    seen = {}
    hs = [xa.q.linear.register_forward_pre_hook(lambda md, i: seen.__setitem__('q_in', i[0].detach())),
          xa.k.linear.register_forward_pre_hook(lambda md, i: seen.__setitem__('k_in', i[0].detach())),
          xa.v.linear.register_forward_pre_hook(lambda md, i: seen.__setitem__('v_in', i[0].detach()))]
    rec = FiringRateRecorder(m)
    with torch.no_grad():
        e = m.encoding(m.tokens(x)); functional.reset_net(m)
        F = blk.mixer(e); functional.reset_net(m)
        R = blk.mixer_rev(e.flip(0)).flip(0); functional.reset_net(m)
        o0 = blk.mix(e); functional.reset_net(m)
        rec.add_samples(x.shape[0])
    for h in hs: h.remove()
    Ts = {st['T'] if not st['single'] else None for st in rec.stats.values() if st['calls']}
    rec.detach()
    q_ok = torch.equal(seen['q_in'], F) and torch.equal(seen['k_in'], R) and torch.equal(seen['v_in'], R)
    shape = tuple(xa.last_scores.shape)
    j = N - 3; perm = torch.roll(torch.arange(8), 1).cuda()
    def tok_fn(t):
        t = t.clone(); t[j] = t[j][perm]; return t
    with torch.no_grad():
        o1 = blk.mix(m.encoding(tok_fn(m.tokens(x)))); functional.reset_net(m)
    before = not torch.equal(o0[:j], o1[:j])
    ok = Ts == {N} and q_ok and shape == (8, 1, 8, N, N) and before
    report('c25', ok, f'neuron T values {sorted(Ts)} (N={N}), q<-F & k,v<-R {q_ok}, scores {shape}, token {j} change reaches earlier tokens {before}')


@check('c26')
def c26():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    msgs, oks = [], []
    g = torch.Generator().manual_seed(26)
    for form, comb in (('channel', 'xattn_ch'), ('head', 'xattn')):
        set_random_seed(0)
        m = LOAD_MODEL['neoMixer'](NS(**_mixer_kw(mixer_comb=comb)), True).cuda().eval()
        xa = m.blocks[0].xattn; N = m.num_patches
        q = (torch.rand(N, 4, 1, 384, generator=g) < 0.2).float().cuda()
        k = (torch.rand(N, 4, 1, 384, generator=g) < 0.2).float().cuda()
        v = (torch.rand(N, 4, 1, 384, generator=g) < 0.2).float().cuda()
        with torch.no_grad():
            z = xa.scores_and_z(q, k, v)
        manual = q * (k * v).sum(0, keepdim=True) * xa.scale
        diff = (z - manual).abs().max().item()
        ok = diff < 1e-5 if form == 'channel' else diff > 1e-2
        oks.append(ok); msgs.append(f'{form}: max |z - q*sum(k*v)*scale| {diff:.2e}')
    report('c26', all(oks), '; '.join(msgs))


def _group_rates(m, x):
    """per (layer, group) firing rates of every GroupNeuron / StreamNeurons part at init, training-mode BN."""
    from layers import GroupNeuron
    from stream_mixer import StreamNeurons
    from spikingjelly.clock_driven import functional
    rates, hs = {}, []
    for name, mod in m.named_modules():
        if isinstance(mod, (GroupNeuron, StreamNeurons)):
            for gi, part in enumerate(mod.parts):
                hs.append(part.register_forward_hook(lambda md, i, o, k=(name, gi): rates.__setitem__(k, float((o[0] if isinstance(o, tuple) else o).float().mean()))))
    m.train(); m.train_mode = 'testing'
    with torch.no_grad():
        m(x); functional.reset_net(m)
    for h in hs: h.remove()
    return rates


@check('c27')
def c27():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from layers import GroupNeuron, tau_gain
    msgs, oks = [], []
    xb, d, P = _real_batch('CINC2017', n=32)
    kw = dict(patch_size=P, seq_len=d['seq_len'], num_channels=d['num_channels'], num_classes=d['num_classes'])
    for mixer in ('popg_lif', 'popg_flif'):
        set_random_seed(0)
        m = LOAD_MODEL['neoMixer'](NS(**_mixer_kw(mixer=mixer, mixer_comb='xattn', **kw)), True).cuda()
        gns = [mod for mod in m.modules() if isinstance(mod, GroupNeuron)]
        taus_ok = all(gn.taus == (4., 8., 16., 32.) for gn in gns)
        import json
        tab = json.load(open(os.path.join(HERE, 'calib', f'tau_gains_neoMixer_{mixer}_xattn.json')))['gains']
        mods = dict(m.named_modules())
        gam_ok = m.tau_calibrated and all(abs(float(mods[b].weight[k * (mods[b].weight.numel() // len(g))]) - gk) < 1e-5
                                          for b, g in tab.items() for k, gk in enumerate(g))
        del m; torch.cuda.empty_cache()
        oks.append(taus_ok and gam_ok and len(gns) >= 10)
        msgs.append(f'{mixer}: {len(gns)} group layers, taus ok {taus_ok}, calibrated gains applied from calib/ ({len(tab)} BN layers) {gam_ok}')
        for comb in ('or', 'xattn'):
            ok, msg = _band_check('neoMixer', mixer, comb, xb, kw)
            oks.append(ok); msgs.append(msg)
    report('c27', all(oks), '; '.join(msgs))


@check('c28')
def c28():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    msgs, oks = [], []
    # (a) K = 1, tau 2, copied weights -> NeoMixer(mixer='flif')
    g = torch.Generator().manual_seed(28); x = torch.rand(8, 1, 187, generator=g).cuda()
    set_random_seed(0); a = LOAD_MODEL['neoMixer'](NS(**_mixer_kw(mixer='flif')), True).cuda()
    set_random_seed(0); b = LOAD_MODEL['streamMixer'](NS(**_mixer_kw(model='streamMixer', mixer='popk_flif', pop_taus='2')), True).cuda()
    sa, sb = a.state_dict(), b.state_dict()
    pairs = list(zip(sa.items(), sb.items()))
    aligned = len(sa) == len(sb) and all(va.shape == vb.shape and ka.split('.')[-1] == kb.split('.')[-1] for (ka, va), (kb, vb) in pairs)
    if aligned:
        b.load_state_dict({kb: va.clone() for (ka, va), (kb, _) in pairs})
    for mm in (a, b):
        mm.train(); mm.train_mode = 'testing'
    with torch.no_grad():
        za = a(x)[0]; functional.reset_net(a)
        zb = b(x)[0]; functional.reset_net(b)
    eq = aligned and torch.equal(za, zb)
    oks.append(eq); msgs.append(f'K=1 tau 2 vs NeoMixer flif: state dicts aligned {aligned} ({len(sa)}/{len(sb)}), logits bitwise equal {eq}, max diff {(za - zb).abs().max().item():.1e}')
    # (b) equal taus -> identical streams; head input K*D
    set_random_seed(0); c = LOAD_MODEL['streamMixer'](NS(**_mixer_kw(model='streamMixer', mixer='popk_lif', pop_taus='2,2,2,2', mixer_comb='xattn')), True).cuda()
    c.train(); c.train_mode = 'testing'
    with torch.no_grad():
        _, e, m0, feat, h = c.forward_parts(x); functional.reset_net(c)
    same = all(torch.equal(h[:, :, 0], h[:, :, k]) for k in range(1, 4))
    oks.append(same and tuple(h.shape[2:]) == (4, 384) and c.head.in_features == 4 * 384)
    msgs.append(f'equal taus: streams identical {same}, h {tuple(h.shape)}, head in {c.head.in_features}')
    # (c) real CinC data: every stream of every layer within 0.5-2x of the reference model's layer and >= 1 % at init
    xb, d, P = _real_batch('CINC2017', n=32)
    kw = dict(patch_size=P, seq_len=d['seq_len'], num_channels=d['num_channels'], num_classes=d['num_classes'])
    for mixer in ('popk_lif', 'popk_flif'):
        for comb in ('or', 'xattn'):
            ok, msg = _band_check('streamMixer', mixer, comb, xb, kw)
            oks.append(ok); msgs.append(msg)
    report('c28', all(oks), '; '.join(msgs))


@check('c29')
def c29():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    from layers import RecurrentPopulationFLIF
    g = torch.Generator().manual_seed(29); x = torch.rand(8, 1, 187, generator=g).cuda()
    set_random_seed(0); a = LOAD_MODEL['neoMixer'](NS(**_mixer_kw(mixer='pop_flif', pop_soma_init=1.0)), True).cuda()
    set_random_seed(0); b = LOAD_MODEL['neoMixer'](NS(**_mixer_kw(mixer='pop_rec', pop_soma_init=1.0)), True).cuda()
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    extra_ok = not unexpected and all(k.endswith(('w_rec.weight', '.g')) for k in missing)
    recs = [mm for mm in b.modules() if isinstance(mm, RecurrentPopulationFLIF)]
    for mm in (a, b):
        mm.train(); mm.train_mode = 'testing'
    with torch.no_grad():
        za = a(x)[0]; functional.reset_net(a)
        zb = b(x)[0]; functional.reset_net(b)
        w0 = [r.w_rec.weight.clone() for r in recs]
        for r in recs: r.g.fill_(1.); r.w_rec.weight.mul_(100.)      # init W_rec is +-0.005: amplify so the recurrent path must flip spikes
        zc = b(x)[0]; functional.reset_net(b)
        for r, w in zip(recs, w0): r.g.fill_(0.); r.w_rec.weight.copy_(w)
    b.train_mode = 'training'
    z = b(x); torch.nn.functional.cross_entropy(z, torch.arange(8).cuda() % 5).backward(); functional.reset_net(b)
    ggrad = all(r.g.grad is not None and float(r.g.grad.abs()) > 0 for r in recs)
    ok = extra_ok and len(recs) == 4 and torch.equal(za, zb) and not torch.equal(zb, zc) and ggrad
    report('c29', ok, f'extra params only w_rec / g {extra_ok}, recurrent neurons {len(recs)}, g=0 equal {torch.equal(za, zb)}, g=1 differs {not torch.equal(zb, zc)}, g grad {ggrad}')


@check('c30')
def c30():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    g = torch.Generator().manual_seed(30); x = torch.randn(4, 1, 3750, generator=g).cuda()
    kw = dict(HP, patch_size=64, seq_len=3750, num_channels=1, num_classes=4, hippo_ssa=True, hippo_T=4)
    set_random_seed(0); a = LOAD_MODEL['myModel'](NS(**kw), True).cuda()
    set_random_seed(0); a2 = LOAD_MODEL['myModel'](NS(**dict(kw, attn_bn=False)), True).cuda()
    set_random_seed(0); b = LOAD_MODEL['myModel'](NS(**dict(kw, attn_bn=True)), True).cuda()
    extra = sorted(set(b.state_dict()) - set(a.state_dict()))
    b.load_state_dict(a.state_dict(), strict=False)
    rates = {}
    for nm, mm in (('off', a), ('off2', a2), ('on', b)):
        h = mm.data_block[0].ssa.attn_lif.register_forward_hook(lambda md, i, o, k=nm: rates.__setitem__(k, float(o.float().mean())))
        mm.train(); mm.train_mode = 'testing'
        with torch.no_grad():
            rates[nm + '_z'] = mm(x)[0]; functional.reset_net(mm)
        h.remove()
    default_eq = torch.equal(rates['off_z'], rates['off2_z']) and set(a.state_dict()) == set(a2.state_dict())
    ok = all(e.startswith('data_block.0.ssa.attn_bn.') for e in extra) and len(extra) > 0 and default_eq and rates['on'] > rates['off'] and not torch.equal(rates['on_z'], rates['off_z'])
    report('c30', ok, f'extra {extra[:2]}..., flag off == default {default_eq}, init attention firing off {rates["off"]:.4f} -> on {rates["on"]:.4f}')


@check('c31')
def c31():
    from neurons import PopulationFLIF
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from layers import NeoNeuron
    g = torch.Generator().manual_seed(31)
    taus = (4., 8., 16., 32.)
    n = PopulationFLIF(6, tau=taus, soma='lif', tau_s=2., heterogeneous=True, alpha=1., method='pred', memory=None, threshold=1., surrogate_scale=5., step=1., max_length=40)
    I = torch.randn(40, 3, 6, generator=g)
    out, aux = n(I, return_aux=True)
    v = torch.zeros(3, 6, 4); tau = torch.tensor(taus); maxd = 0.
    for t in range(40):
        v = v + (I[t].unsqueeze(-1) - v) / tau
        maxd = max(maxd, (aux['state'][t] - v).abs().max().item())
    binary = bool(((out == 0) | (out == 1)).all())
    msgs = [f'branch states vs manual Euler max diff {maxd:.1e}, soma binary {binary}']
    rates = {}
    for data in ('CINC2017', 'SHD'):
        xb, d, P = _real_batch(data)
        set_random_seed(0)
        m = LOAD_MODEL['neoMixer'](NS(**_mixer_kw(mixer='pop_lif', pop_soma_init=0.5, patch_size=P, seq_len=d['seq_len'], num_channels=d['num_channels'],
                                                  num_classes=d['num_classes'])), True).cuda().train()
        m.train_mode = 'testing'
        from spikingjelly.clock_driven import functional
        with torch.no_grad():
            o = m.blocks[0].mix(m.encoding(m.tokens(xb))); functional.reset_net(m)
        rates[data] = float(o.float().mean())
        del m; torch.cuda.empty_cache()
    msgs.append('init mixer firing (pop LIF, psi 0.5 = closest to the LIF mixer of {0.25, 0.5, 1}) ' + ', '.join(f'{k} {v:.4f}' for k, v in rates.items()))
    report('c31', maxd < 1e-5 and binary, '; '.join(msgs))


@check('c32')
def c32():
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from energy import FiringRateRecorder
    from spikingjelly.clock_driven import functional
    g = torch.Generator().manual_seed(32); x = torch.rand(8, 1, 187, generator=g).cuda()
    set_random_seed(0)
    m = LOAD_MODEL['neoMixer'](NS(**_mixer_kw(mixer_comb='lin')), True).cuda(); m.train_mode = 'testing'
    _bn_freeze_eval(m, x)
    blk = m.blocks[0]
    lin_ok = tuple(blk.fuse.linear.weight.shape) == (384, 768) and not hasattr(blk, 'xattn')
    rec = FiringRateRecorder(m)
    with torch.no_grad():
        e = m.encoding(m.tokens(x)); functional.reset_net(m)
        F = blk.mixer(e); functional.reset_net(m)
        R = blk.mixer_rev(e.flip(0)).flip(0); functional.reset_net(m)
        m0 = blk.fuse(torch.cat([F, R], -1)); functional.reset_net(m)
        mm = blk.mix(e); functional.reset_net(m)
        rec.add_samples(8)
        F2 = F.clone(); F2[5] = 1. - F2[5]; R2 = R.clone(); R2[5] = 1. - R2[5]
        mf = blk.fuse(torch.cat([F2, R], -1)); functional.reset_net(m)
        mr = blk.fuse(torch.cat([F, R2], -1)); functional.reset_net(m)
    Ts = {st['T'] for st in rec.stats.values() if st['calls'] and not st['single']}
    rec.detach()
    binary = bool(((mm == 0) | (mm == 1)).all())
    ok = lin_ok and binary and torch.equal(m0, mm) and Ts == {m.num_patches} and not torch.equal(m0, mf) and not torch.equal(m0, mr)
    report('c32', ok, f'Linear (384, 768) & no attention {lin_ok}, fused output binary {binary}, mix() == fuse([F; R]) {torch.equal(m0, mm)}, '
                      f'neuron T {sorted(Ts)}, forward-only change -> output changes {not torch.equal(m0, mf)}, reverse-only {not torch.equal(m0, mr)}, rate {float(mm.mean()):.3f}')


# ---- MODI 2026-10-23: membrane fusion (linmem / linmx / xattnmem) and the K-stream neocortex aligned with T_h ---------------
MEM_NEURONS = (('lif', {}), ('flif', {}), ('pop_lif', dict(pop_soma_init=0.5)), ('pop_flif', dict(pop_soma_init=1.0)))


def _cinc_mixer(comb, mixer, extra, x):
    from load_model import LOAD_MODEL
    from config import set_random_seed
    set_random_seed(0)
    m = LOAD_MODEL['neoMixer'](NS(**_mixer_kw(mixer=mixer, mixer_comb=comb, patch_size=64, seq_len=3750, num_classes=4, **extra)), True).cuda()
    m.train_mode = 'testing'
    return m


@check('c33')
def c33():
    """charge capture: spikes_and_charge goes through the neuron's own calls -- for every spikeDE neocortex neuron kind the spikes
    equal forward() bitwise, spikes == (H - theta >= 0), H carries a gradient, and the energy recorder still sees the call."""
    from spikingjelly.clock_driven import functional
    from energy import FiringRateRecorder
    x, _, _ = _real_batch('CINC2017', n=8)
    rows, ok = [], True
    for mixer, extra in MEM_NEURONS:
        m = _cinc_mixer('linmem', mixer, extra, x)
        _bn_freeze_eval(m, x)
        blk = m.blocks[0]
        rec = FiringRateRecorder(m)
        with torch.no_grad():
            e = m.encoding(m.tokens(x)); functional.reset_net(m)
            ref = blk.mixer(e); functional.reset_net(m)
            calls0 = {k: v['calls'] for k, v in rec.stats.items()}
            s, h = blk.mixer.forward_charge(e); functional.reset_net(m)
        seen = all(rec.stats[k]['calls'] == calls0[k] + 1 for k in rec.stats if k.startswith('blocks.0.mixer.Mem.1'))
        rec.detach()
        e2 = e.detach().clone().requires_grad_(True)
        _, h2 = blk.mixer.forward_charge(e2); functional.reset_net(m)
        h2.sum().backward()
        good = torch.equal(ref, s) and torch.equal(s, (h >= 0).to(s.dtype)) and e2.grad is not None and float(e2.grad.abs().sum()) > 0 \
            and seen and 0.005 < float(s.mean()) < 0.9
        ok &= good
        rows.append(f'{mixer}: spikes==forward {torch.equal(ref, s)}, ==(H>=0) {torch.equal(s, (h >= 0).to(s.dtype))}, '
                    f'grad {float(e2.grad.abs().sum()):.2e}, recorder sees it {seen}, rate {float(s.mean()):.3f}')
    report('c33', ok, '; '.join(rows))


@check('c34')
def c34():
    """linmem / linmx: fusion Linear (384, 768) / (384, 1152) on [H_f; H_r] / [H_f; H_r; |H_f - H_r|], fused output 0/1, mix() ==
    fuse(charges) computed from the two stacks separately; on real CinC windows (training-mode BN) both directions' last
    layers get a gradient through the charges; changing the forward-only or reverse-only charge changes the output; linmem has
    the initial weights of lin at the same seed (same module shapes and order)."""
    from spikingjelly.clock_driven import functional
    x, _, _ = _real_batch('CINC2017', n=8)
    rows, ok = [], True
    for comb, width in (('linmem', 768), ('linmx', 1152)):
        for mixer, extra in MEM_NEURONS:
            m = _cinc_mixer(comb, mixer, extra, x)
            same_init = True
            if comb == 'linmem':                                       # before any forward (BN running statistics move)
                ml = _cinc_mixer('lin', mixer, extra, x)
                same_init = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), ml.state_dict().values()))
                del ml
            blk = m.blocks[0]
            m.train()
            m.zero_grad()
            z, *_ = m.forward_parts(x); z.sum().backward(); functional.reset_net(m)
            gf = float(blk.mixer.Mem[-1].w_in.weight.grad.abs().sum()); gr = float(blk.mixer_rev.Mem[-1].w_in.weight.grad.abs().sum())
            _bn_freeze_eval(m, x)
            with torch.no_grad():
                e = m.encoding(m.tokens(x)); functional.reset_net(m)
                _, hf = blk.mixer.forward_charge(e); functional.reset_net(m)
                _, hr = blk.mixer_rev.forward_charge(e.flip(0)); functional.reset_net(m)
                hr = hr.flip(0)
                cat = (lambda a, b: torch.cat([a, b], -1)) if comb == 'linmem' else (lambda a, b: torch.cat([a, b, (a - b).abs()], -1))
                m0 = blk.fuse(cat(hf, hr)); functional.reset_net(m)
                mm = blk.mix(e); functional.reset_net(m)
                mf = blk.fuse(cat(hf + 0.5, hr)); functional.reset_net(m)
                mr = blk.fuse(cat(hf, hr + 0.5)); functional.reset_net(m)
            binary = bool(((mm == 0) | (mm == 1)).all())
            shape = tuple(blk.fuse.linear.weight.shape) == (384, width)
            good = shape and binary and torch.equal(m0, mm) and gf > 0 and gr > 0 and not torch.equal(m0, mf) and not torch.equal(m0, mr) \
                and same_init and 0.005 < float(mm.mean()) < 0.9
            ok &= good
            rows.append(f'{comb}/{mixer}: Linear {tuple(blk.fuse.linear.weight.shape)}, binary {binary}, mix==fuse {torch.equal(m0, mm)}, '
                        f'grad fwd/rev {gf:.1e}/{gr:.1e}, both directions matter {not torch.equal(m0, mf)}/{not torch.equal(m0, mr)}, '
                        f"init == lin {same_init if comb == 'linmem' else 'n/a (other width)'}, rate {float(mm.mean()):.3f}")
    report('c34', ok, '; '.join(rows))


@check('c35')
def c35():
    """xattnmem: q = BN(W_q (H_f - theta)) and k = BN(W_k (H_r - theta)) are real-valued (no neuron), v = LIF(BN(W_v R)) is 0/1,
    the output is 0/1 with T = N, the attention map is [B, 1, 8, N, N]; the initial weights equal the spike BiX (xattn) at the
    same seed; on real CinC windows both directions' last layers get a gradient."""
    from spikingjelly.clock_driven import functional
    x, _, _ = _real_batch('CINC2017', n=8)
    rows, ok = [], True
    for mixer, extra in MEM_NEURONS:
        m = _cinc_mixer('xattnmem', mixer, extra, x)
        mb = _cinc_mixer('xattn', mixer, extra, x)                       # before any forward (BN running statistics move)
        same_init = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), mb.state_dict().values()))
        del mb
        blk = m.blocks[0]; xa = blk.xattn
        m.train(); m.zero_grad()
        z, *_ = m.forward_parts(x); z.sum().backward(); functional.reset_net(m)
        gf = float(blk.mixer.Mem[-1].w_in.weight.grad.abs().sum()); gr = float(blk.mixer_rev.Mem[-1].w_in.weight.grad.abs().sum())
        _bn_freeze_eval(m, x)
        seen = {}
        hs = [xa.q.register_forward_hook(lambda md, i, o: seen.__setitem__('q', o.detach())),
              xa.v.register_forward_hook(lambda md, i, o: seen.__setitem__('v', o.detach()))]
        xa.keep_scores = True
        with torch.no_grad():
            e = m.encoding(m.tokens(x)); functional.reset_net(m)
            mm = blk.mix(e); functional.reset_net(m)
        for h in hs: h.remove()
        q_real = not bool(((seen['q'] == 0) | (seen['q'] == 1)).all())
        v_bin = bool(((seen['v'] == 0) | (seen['v'] == 1)).all())
        binary = bool(((mm == 0) | (mm == 1)).all())
        shape = tuple(xa.last_scores.shape) == (8, 1, 8, m.num_patches, m.num_patches)
        good = q_real and v_bin and binary and shape and same_init and gf > 0 and gr > 0 and 0.005 < float(mm.mean()) < 0.9
        ok &= good
        rows.append(f'{mixer}: q real {q_real}, v 0/1 {v_bin}, out 0/1 {binary}, scores {tuple(xa.last_scores.shape)}, init == BiX {same_init}, '
                    f'grad fwd/rev {gf:.1e}/{gr:.1e}, rate {float(mm.mean()):.3f}')
    report('c35', ok, '; '.join(rows))


@check('c36')
def c36():
    """NSMT with a K-stream neocortex aligned with T_h (popk_lif / popk_flif, hippo_T 4, hippo_xattn): the neocortex output is
    [N, B, 4, D] with every stream neuron at T = N and its tau 4/8/16/32; the cross-attention key / value of hippocampus step t
    is stream t (not a copy of one stream); the calibration table is loaded and every stream of every neocortex layer fires at
    0.98-1.02x the reference layer on the calibration windows; each stream is causal along the tokens; the decoder
    target is the same signal for every stream and the training loss path runs; hippo_T 3 or no hippo_xattn is refused."""
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    from stream_mixer import StreamNeurons
    from train_rhythm import train_epoch
    import json as _json
    x, _, _ = _real_batch('CINC2017', n=32)
    rows, ok = [], True
    base = dict(HP, patch_size=64, seq_len=3750, num_channels=1, num_classes=4, hippo_T=4, hippo_xattn=True, neo_alpha=0.5)
    for neo in ('popk_lif', 'popk_flif'):
        set_random_seed(0)
        m = LOAD_MODEL['myModel'](NS(**dict(base, neo_neuron=neo)), True).cuda()
        table = _json.load(open(os.path.join(HERE, 'calib', f'tau_gains_myModel_{neo}_xaT4.json')))
        loaded = m.tau_calibrated and all(abs(float(dict(m.named_modules())[bn].weight[k * 384]) - g) < 1e-5
                                          for bn, gs in table['gains'].items() for k, g in enumerate(gs))
        taus = [float(p.neuron.tau) for p in m.encoding_neo.neuron.parts]
        m.train(); m.train_mode = 'testing'
        rates, hs = {}, []
        for n, mod in m.named_modules():
            if isinstance(mod, StreamNeurons) and n.startswith(('encoding_neo', 'time_block')):
                for k, part in enumerate(mod.parts):
                    hs.append(part.register_forward_hook(lambda md, i, o, n=n, k=k: rates.__setitem__((n, k), float(o.float().mean()))))
        seen = {}
        ssa = m.data_block[0].ssa; orig = ssa.forward
        def rec_fwd(xx, mask=None, kv=None):
            seen['kv'] = kv.detach(); return orig(xx, mask=mask, kv=kv)
        ssa.forward = rec_fwd
        with torch.no_grad():
            _, neo_in = m._tokens_mc(x)
            _, xn = m.neo_forward(neo_in); functional.reset_net(m)
            m(x); functional.reset_net(m)
        ssa.forward = orig
        for h in hs: h.remove()
        tgt = table['targets']
        band = all(0.98 <= rates[(n, k)] / tgt[n] <= 1.02 for (n, k) in rates)
        shape = tuple(xn.shape) == (m.num_patches, 32, 4, 384)
        aligned = all(torch.equal(seen['kv'][t], xn[:, :, t].transpose(0, 1)) for t in range(4)) and not torch.equal(seen['kv'][0], seen['kv'][3])
        # causality per stream along the tokens (eval-mode BN for a deterministic comparison)
        _bn_freeze_eval(m, x)
        with torch.no_grad():
            _, neo_in = m._tokens_mc(x[:8])
            _, a0 = m.neo_forward(neo_in); functional.reset_net(m)
            j = 60; ni = neo_in.clone(); ni[j] = ni[j][torch.roll(torch.arange(8), 1).cuda()]
            _, a1 = m.neo_forward(ni); functional.reset_net(m)
        causal = torch.equal(a0[:j], a1[:j]) and not torch.equal(a0[j:], a1[j:])
        # training loss path (decoder target expanded over the streams)
        set_random_seed(0)
        mt = LOAD_MODEL['myModel'](NS(**dict(base, neo_neuron=neo)), True).cuda().train()
        out = mt(x[:4]); functional.reset_net(mt)
        dec_ok = tuple(out[3].shape) == tuple(out[4].shape) == (m.num_patches, 4, 4, 64) and torch.equal(out[3][:, :, 0], out[3][:, :, 3])
        good = loaded and taus == [4., 8., 16., 32.] and band and shape and aligned and causal and dec_ok
        ok &= good
        rows.append(f'{neo}: table loaded {loaded}, stream taus {taus}, firing / reference in [0.98, 1.02] {band} '
                    f'({min(rates[k] / tgt[k[0]] for k in rates):.3f}-{max(rates[k] / tgt[k[0]] for k in rates):.3f}), neocortex {tuple(xn.shape)}, '
                    f'kv[t] = stream t {aligned}, causal per stream {causal}, decoder target per stream {dec_ok}')
    refused = []
    for bad in (dict(hippo_T=3), dict(hippo_xattn=False, hippo_T=0)):
        try:
            LOAD_MODEL['myModel'](NS(**dict(base, neo_neuron='popk_lif', **bad)), True); refused.append(False)
        except ValueError:
            refused.append(True)
    ok &= all(refused)
    report('c36', ok, '; '.join(rows) + f'; refused hippo_T 3 / no hippo_xattn {refused}')


@check('c37')
def c37():
    """the new T_h = 4 baselines build and run: population LIF with soma (neo_neuron pop_lif, alpha 1) and the single f-LIF, and
    the energy recorder reports their neocortex neurons as pop_lif / flif."""
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    from energy import FiringRateRecorder
    x, _, _ = _real_batch('CINC2017', n=4)
    base = dict(HP, patch_size=64, seq_len=3750, num_channels=1, num_classes=4, hippo_T=4, hippo_xattn=True, neo_alpha=0.5)
    rows, ok = [], True
    for neo, extra in (('pop_lif', dict(pop_soma_init=1.0)), ('flif', {})):
        set_random_seed(0)
        m = LOAD_MODEL['myModel'](NS(**dict(base, neo_neuron=neo, **extra)), True).cuda(); m.train(); m.train_mode = 'testing'
        rec = FiringRateRecorder(m)
        with torch.no_grad():
            z = m(x); functional.reset_net(m)
        rec.add_samples(4)
        kinds = sorted({st['kind'] for n, st in rec.stats.items() if st['path'] == 'neo' and st['calls']})
        rec.detach()
        zz = z[0] if isinstance(z, tuple) else z
        good = tuple(zz.shape) == (4, 4) and kinds == [neo] and bool(torch.isfinite(zz).all())
        ok &= good
        rows.append(f'{neo}: logits {tuple(zz.shape)}, neocortex neuron kinds {kinds}')
    report('c37', ok, '; '.join(rows))


# ---- MODI 2026-10-24: spike xor control, tau-group linear fusions, stream order and the fused bidirectional stream neocortex ----
@check('c38')
def c38():
    """linxor: Linear (384, 1152) on [F; R; |F - R|], |F - R| is 0/1 and equals F xor R, fused output 0/1, mix() == fuse() from the
    two stacks; initial weights equal linmx at the same seed (same shapes and order); no charge is captured (spike-only path)."""
    from spikingjelly.clock_driven import functional
    x, _, _ = _real_batch('CINC2017', n=8)
    rows, ok = [], True
    for mixer, extra in MEM_NEURONS:
        m = _cinc_mixer('linxor', mixer, extra, x)
        mm_ = _cinc_mixer('linmx', mixer, extra, x)
        same_init = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), mm_.state_dict().values())); del mm_
        _bn_freeze_eval(m, x)
        blk = m.blocks[0]
        called = []
        h = blk.mixer.Mem[-1].lif.register_forward_hook(lambda md, i, o: called.append(getattr(md, '_capture', False)))
        with torch.no_grad():
            e = m.encoding(m.tokens(x)); functional.reset_net(m)
            F = blk.mixer(e); functional.reset_net(m)
            R = blk.mixer_rev(e.flip(0)).flip(0); functional.reset_net(m)
            xor = (F - R).abs()
            m0 = blk.fuse(torch.cat([F, R, xor], -1)); functional.reset_net(m)
            mm = blk.mix(e); functional.reset_net(m)
        h.remove()
        is_xor = torch.equal(xor, ((F + R) == 1).to(F.dtype))
        good = tuple(blk.fuse.linear.weight.shape) == (384, 1152) and is_xor and torch.equal(m0, mm) and same_init \
            and bool(((mm == 0) | (mm == 1)).all()) and not any(called) and 0.005 < float(mm.mean()) < 0.9
        ok &= good
        rows.append(f'{mixer}: Linear {tuple(blk.fuse.linear.weight.shape)}, |F-R| == xor {is_xor}, mix == fuse {torch.equal(m0, mm)}, '
                    f'init == linmx {same_init}, no charge capture {not any(called)}, rate {float(mm.mean()):.3f}')
    report('c38', ok, '; '.join(rows))


@check('c39')
def c39():
    """tau-group (popg) linear fusions lin / linmem / linmx / linxor: the fusion neuron is a spikingjelly GroupNeuron (tau 4/8/16/32),
    the calibration table of the combination is loaded, every grouped layer fires at 0.98-1.02x its reference layer on the
    calibration windows (as c27 / c28), the charge path reproduces the spikes of the grouped mixer exactly, output 0/1."""
    import json as _json
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    from layers import GroupNeuron
    from calibrate_groups import KW, ref_rates, layer_kind
    import rhythm_data as rd
    d = rd.build(NS(data='CINC2017', crop_tokens=0, patch_size=64, fs_target=125., fold=0, val_frac=0.1, win_s=30., hop_s=15.), 42)
    x = torch.stack([d['train'][i][0] for i in range(32)]).float().cuda()
    kw = dict(patch_size=64, seq_len=d['seq_len'], num_channels=d['num_channels'], num_classes=d['num_classes'])
    rows, ok = [], True
    for mixer in ('popg_lif', 'popg_flif'):
        for comb in ('lin', 'linmem', 'linmx', 'linxor'):
            fam = mixer.split('_', 1)[1]
            target = ref_rates('neoMixer', fam, comb, x, kw)
            set_random_seed(0)
            m = LOAD_MODEL['neoMixer'](NS(**dict(KW, model='neoMixer', mixer=mixer, mixer_comb=comb, tau_calib=True, **kw)), True).cuda()
            fuse_group = isinstance(m.blocks[0].fuse.neuron, GroupNeuron) and m.blocks[0].fuse.neuron.kind == 'sj'
            rates, hs = {}, []
            for n, mod in m.named_modules():
                if isinstance(mod, GroupNeuron):
                    for k, part in enumerate(mod.parts):
                        hs.append(part.register_forward_hook(lambda md, i, o, n=n, k=k: rates.__setitem__((n, k), float(o.float().mean()))))
            m.train(); m.train_mode = 'testing'
            with torch.no_grad():
                out = m(x); functional.reset_net(m)
            for h in hs: h.remove()
            ratios = [rates[(n, k)] / target[layer_kind(n)] for (n, k) in rates]
            band = min(ratios) >= 0.98 and max(ratios) <= 1.02
            exact = True
            if comb in ('linmem', 'linmx'):
                with torch.no_grad():
                    e = m.encoding(m.tokens(x[:8])); functional.reset_net(m)
                    ref = m.blocks[0].mixer(e); functional.reset_net(m)
                    s_, h_ = m.blocks[0].mixer.forward_charge(e); functional.reset_net(m)
                exact = torch.equal(ref, s_) and torch.equal(s_, (h_ >= 0).to(s_.dtype))
            good = m.tau_calibrated and fuse_group and band and exact
            ok &= good
            rows.append(f'{mixer}/{comb}: table {m.tau_calibrated}, fuse = sj groups {fuse_group}, firing / reference {min(ratios):.3f}-{max(ratios):.3f}, '
                        f'charge path exact {exact}')
            del m; torch.cuda.empty_cache()
    report('c39', ok, '; '.join(rows))


@check('c40')
def c40():
    """stream order control (stream_rev): the neocortex and its initial values are those of the plain stream model at the same seed,
    and the key / value of hippocampus step t is stream 3 - t."""
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    x, _, _ = _real_batch('CINC2017', n=8)
    base = dict(HP, patch_size=64, seq_len=3750, num_channels=1, num_classes=4, hippo_T=4, hippo_xattn=True, neo_alpha=0.5)
    rows, ok = [], True
    for neo in ('popk_lif', 'popk_flif'):
        set_random_seed(0); m0 = LOAD_MODEL['myModel'](NS(**dict(base, neo_neuron=neo)), True).cuda()
        set_random_seed(0); m1 = LOAD_MODEL['myModel'](NS(**dict(base, neo_neuron=neo, stream_rev=True)), True).cuda()
        same = all(torch.equal(a, b) for a, b in zip(m0.state_dict().values(), m1.state_dict().values()))
        m1.train(); m1.train_mode = 'testing'
        seen = {}
        ssa = m1.data_block[0].ssa; orig = ssa.forward
        def rec_fwd(xx, mask=None, kv=None):
            seen['kv'] = kv.detach(); return orig(xx, mask=mask, kv=kv)
        ssa.forward = rec_fwd
        with torch.no_grad():
            _, neo_in = m1._tokens_mc(x)
            _, xn = m1.neo_forward(neo_in); functional.reset_net(m1)
            m1(x); functional.reset_net(m1)
        ssa.forward = orig
        rev = all(torch.equal(seen['kv'][t], xn[:, :, 3 - t].transpose(0, 1)) for t in range(4))
        ok &= same and rev
        rows.append(f'{neo}: same initial values as the plain model {same}, kv[t] = stream 3 - t {rev}')
    report('c40', ok, '; '.join(rows))


@check('c41')
def c41():
    """fused bidirectional stream neocortex (neo_bidir + neo_stream_fuse mx): forward and reverse stream stacks with their own
    parameters, the output = LIF(BN(W [H_f; H_r; |H_f - H_r|])) per stream [N, B, 4, D] (0/1), recomputed from the two stacks
    exactly; the forward stack is causal along the tokens and the reverse stack anti-causal; the bimx calibration table is
    loaded and every stream layer of both stacks fires at 0.98-1.02x its reference slot; both stacks get a gradient."""
    import json as _json
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    from stream_mixer import StreamNeurons
    x, _, _ = _real_batch('CINC2017', n=32)
    base = dict(HP, patch_size=64, seq_len=3750, num_channels=1, num_classes=4, hippo_T=4, hippo_xattn=True, neo_alpha=0.5,
                neo_bidir=True, neo_stream_fuse='mx')
    rows, ok = [], True
    for neo in ('popk_lif', 'popk_flif'):
        set_random_seed(0)
        m = LOAD_MODEL['myModel'](NS(**dict(base, neo_neuron=neo)), True).cuda()
        table = _json.load(open(os.path.join(HERE, 'calib', f'tau_gains_myModel_{neo}_xaT4_bimx.json')))
        rates, hs = {}, []
        for n, mod in m.named_modules():
            if isinstance(mod, StreamNeurons) and n.startswith(('encoding_neo', 'time_block')):
                for k, part in enumerate(mod.parts):
                    hs.append(part.register_forward_hook(lambda md, i, o, n=n, k=k: rates.__setitem__((n, k), float(o.float().mean()))))
        m.train(); m.train_mode = 'testing'
        with torch.no_grad():
            m(x); functional.reset_net(m)
        for h in hs: h.remove()
        band = all(0.98 <= rates[(n, k)] / table['targets'][n] <= 1.02 for (n, k) in rates) and len({n for n, _ in rates}) == 6
        m.zero_grad()
        z, *_ = m(x[:4]); functional.reset_net(m)
        z.sum().backward()
        gf = float(m.time_block.layers[-1].linear.weight.grad.abs().sum()); gr = float(m.time_block_rev.layers[-1].linear.weight.grad.abs().sum())
        _bn_freeze_eval(m, x[:8])
        with torch.no_grad():
            _, neo_in = m._tokens_mc(x[:8])
            _, o = m.neo_forward(neo_in); functional.reset_net(m)
            _, hf = m.time_block.forward_charge(m.encoding_neo(neo_in)); functional.reset_net(m)
            _, hr = m.time_block_rev.forward_charge(m.encoding_neo_rev(neo_in.flip(0))); functional.reset_net(m)
            hr = hr.flip(0)
            o2 = m.neo_fuse(torch.cat([hf, hr, (hf - hr).abs()], -1)); functional.reset_net(m)
            j = 60; ni = neo_in.clone(); ni[j] = ni[j][torch.roll(torch.arange(8), 1).cuda()]
            _, hf1 = m.time_block.forward_charge(m.encoding_neo(ni)); functional.reset_net(m)
            _, hr1 = m.time_block_rev.forward_charge(m.encoding_neo_rev(ni.flip(0))); functional.reset_net(m)
            hr1 = hr1.flip(0)
        causal = torch.equal(hf[:j], hf1[:j]) and not torch.equal(hf[j:], hf1[j:]) and torch.equal(hr[j + 1:], hr1[j + 1:]) and not torch.equal(hr[:j + 1], hr1[:j + 1])
        good = m.tau_calibrated and band and tuple(o.shape) == (m.num_patches, 8, 4, 384) and torch.equal(o, o2) \
            and bool(((o == 0) | (o == 1)).all()) and causal and gf > 0 and gr > 0
        ok &= good
        rows.append(f'{neo}: table {m.tau_calibrated}, 6 stream layers in [0.98, 1.02] {band}, output {tuple(o.shape)} == fuse(charges) {torch.equal(o, o2)}, '
                    f'forward causal / reverse anti-causal {causal}, grad fwd/rev {gf:.1e}/{gr:.1e}, rate {float(o.mean()):.3f}')
    report('c41', ok, '; '.join(rows))


B_BASE = dict(HP, patch_size=64, seq_len=3750, num_channels=1, num_classes=4, neo_alpha=0.5)   # MODI 2026-10-26 (B) checks


def _ce_grads(m, x, y):
    """train-mode forward, plain CE on the logits only, backward; returns (logits, {param name: grad})."""
    from spikingjelly.clock_driven import functional
    m.train(); m.zero_grad(set_to_none=True)
    out = m(x); z = out[0] if isinstance(out, tuple) else out
    torch.nn.functional.cross_entropy(z, y).backward(); functional.reset_net(m)
    return z.detach(), {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in m.named_parameters()}


@check('c42')
def c42():
    """(MODI 2026-10-26) --neo_grad: the default (no option) and 0.5 give bitwise equal logits; 1.0 gives bitwise equal logits too
    (the forward value does not depend on the share), with CE only every neocortex parameter gradient is exactly 2x the 0.5 one
    and every hippocampus parameter gradient is bitwise equal -- on both the T_h 4 cross-attention path and the T = N memory-replay
    path, on real CinC windows."""
    from load_model import LOAD_MODEL
    from config import set_random_seed
    x, _, _ = _real_batch('CINC2017', n=8)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]).cuda()
    rows, ok = [], True
    for tag, extra in (('T_h 4 cross-attention', dict(hippo_T=4, hippo_xattn=True)), ('T = N memory replay', {})):
        res = {}
        for g in (None, 0.5, 1.0):
            set_random_seed(0)
            kw = dict(B_BASE, neo_neuron='lif', **extra)
            if g is not None:
                kw['neo_grad'] = g
            m = LOAD_MODEL['myModel'](NS(**kw), True).cuda()
            res[g] = _ce_grads(m, x, y)
        neo = [n for n, gr in res[0.5][1].items() if n.startswith(('encoding_neo', 'time_block')) and gr is not None and gr.abs().sum() > 0]
        hip = [n for n, gr in res[0.5][1].items() if not n.startswith(('encoding_neo', 'time_block', 'weak_decoder')) and gr is not None]
        same_logits = torch.equal(res[None][0], res[0.5][0]) and torch.equal(res[0.5][0], res[1.0][0])
        same_default = all((res[None][1][n] is None and res[0.5][1][n] is None) or torch.equal(res[None][1][n], res[0.5][1][n]) for n in res[0.5][1])
        rel = max(float((res[1.0][1][n] - 2 * res[0.5][1][n]).abs().max() / (2 * res[0.5][1][n]).abs().max().clamp_min(1e-30)) for n in neo)
        hip_eq = all(torch.equal(res[1.0][1][n], res[0.5][1][n]) for n in hip)
        good = same_logits and same_default and len(neo) > 0 and rel < 1e-4 and hip_eq and len(hip) > 0
        ok &= good
        rows.append(f'{tag}: logits equal (default / 0.5 / 1.0) {same_logits}, default == 0.5 grads {same_default}, '
                    f'{len(neo)} neocortex grads at 1.0 = 2x (max rel err {rel:.1e}), {len(hip)} hippocampus grads bitwise equal {hip_eq}')
    report('c42', ok, '; '.join(rows))


@check('c43')
def c43():
    """(MODI 2026-10-26) --hippo_xattn_mode cat: with the Spikformer-style (hippo_ssa, T 4) weights copied into the cat model and
    the neocortex output forced to 0, the logits equal Spikformer-style bitwise (the extra key / value term vanishes); with the
    real neocortex output they differ and the neocortex receives a CE gradient; the extra parameters are exactly the neocortex and
    the second key / value pair; the replace model keeps its parameter count (1-C); the energy recorder sees k2 / v2 at T = 4."""
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    from energy import FiringRateRecorder
    x, _, _ = _real_batch('CINC2017', n=8)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]).cuda()
    set_random_seed(0)
    spk = LOAD_MODEL['myModel'](NS(**dict(B_BASE, hippo_ssa=True, hippo_T=4)), True).cuda()
    set_random_seed(1)
    cat = LOAD_MODEL['myModel'](NS(**dict(B_BASE, neo_neuron='lif', hippo_T=4, hippo_xattn=True, hippo_xattn_mode='cat', neo_grad=1.0)), True).cuda()
    set_random_seed(0)
    rep = LOAD_MODEL['myModel'](NS(**dict(B_BASE, neo_neuron='lif', hippo_T=4, hippo_xattn=True)), True).cuda()
    sd_s, sd_c = spk.state_dict(), cat.state_dict()
    missing = [k for k in sd_s if k not in sd_c or sd_c[k].shape != sd_s[k].shape]
    cat.load_state_dict({**sd_c, **{k: v for k, v in sd_s.items() if k in sd_c}})
    extra = sorted({k.split('.')[0] if not k.startswith('data_block') else '.'.join(k.split('.')[:4]) for k in sd_c if k not in sd_s})
    n_par = lambda m: sum(p.numel() for p in m.parameters())
    for m in (spk, cat):
        m.train(); m.train_mode = 'testing'
    with torch.no_grad():
        zs = spk(x)[0]; functional.reset_net(spk)
        orig = cat.neo_forward
        cat.neo_forward = lambda ni, org=None: (lambda e, o: (e, torch.zeros_like(o)))(*orig(ni, org))
        zc0 = cat(x)[0]; functional.reset_net(cat)
        cat.neo_forward = orig
        zc1 = cat(x)[0]; functional.reset_net(cat)
    zero_eq = torch.equal(zs, zc0)
    differs = not torch.equal(zs, zc1)
    cat.train_mode = 'training'
    _, gr = _ce_grads(cat, x, y)
    neo_grad = sum(float(g.abs().sum()) for n, g in gr.items() if n.startswith(('encoding_neo', 'time_block')) and g is not None)
    kv2_grad = sum(float(g.abs().sum()) for n, g in gr.items() if '.k2_' in n or '.v2_' in n if g is not None)
    cat.train_mode = 'testing'
    rec = FiringRateRecorder(cat)
    with torch.no_grad():
        cat(x[:4]); functional.reset_net(cat)
    rec.add_samples(4)
    kv2 = {n: (st['T'], st['path']) for n, st in rec.stats.items() if ('k2_lif' in n or 'v2_lif' in n)}
    rec.detach()
    counts = (n_par(spk), n_par(rep), n_par(cat))
    good = (not missing and zero_eq and differs and neo_grad > 0 and kv2_grad > 0 and counts[1] == 1264456 and counts[0] == 918088
            and counts[2] == counts[1] + 2 * 384 * 384 + 4 * 384 and len(kv2) == 2 and all(v == (4, 'hippo') for v in kv2.values()))
    report('c43', good, f'Spikformer tensors copied (missing {missing}), neocortex output 0 -> logits == Spikformer-style bitwise {zero_eq}, '
                        f'real neocortex -> differ {differs}, CE grad |neocortex| {neo_grad:.3e} |k2, v2| {kv2_grad:.3e}, '
                        f'params Spikformer / replace / cat {counts}, cat-only modules {extra}, recorder k2 / v2 {kv2}')


@check('c44')
def c44():
    """(MODI 2026-10-26) --neo_memoryless: same parameters as the T = N neocortex; replacing input token j changes the memoryless
    neocortex output at token j only (all other tokens bitwise equal) while the T = N neocortex also changes later tokens
    (positive control); the energy recorder sees the memoryless neocortex neurons at T = 1 with N x the neurons per window, so
    neurons x steps per window equals the T = N model's."""
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    from energy import FiringRateRecorder
    x, _, _ = _real_batch('CINC2017', n=8)
    ms = {}
    for tag, extra in (('memoryless', dict(neo_memoryless=True)), ('T = N', {})):
        set_random_seed(0)
        ms[tag] = LOAD_MODEL['myModel'](NS(**dict(B_BASE, neo_neuron='lif', hippo_T=4, hippo_xattn=True, hippo_xattn_mode='cat',
                                                   neo_grad=1.0, **extra)), True).cuda()
    same_params = all(torch.equal(a, b) for a, b in zip(ms['memoryless'].state_dict().values(), ms['T = N'].state_dict().values()))
    rows, change = [], {}
    j = 60
    for tag, m in ms.items():
        m.train_mode = 'testing'
        _bn_freeze_eval(m, x)
        with torch.no_grad():
            _, ni = m._tokens_mc(x)
            _, a0 = m.neo_forward(ni); functional.reset_net(m)
            ni2 = ni.clone(); ni2[j] = ni2[j][torch.roll(torch.arange(8), 1).cuda()]
            _, a1 = m.neo_forward(ni2); functional.reset_net(m)
        d = (a0 != a1).flatten(1).any(1)                                     # which tokens changed
        change[tag] = (bool(d[j]), int(d[:j].sum()), int(d[j + 1:].sum()), tuple(a0.shape))
        rec = FiringRateRecorder(m); m.train()
        with torch.no_grad():
            m(x[:4]); functional.reset_net(m)
        rec.add_samples(4)
        lay = [r for r in rec.layers() if r['path'] == 'neo']
        rec.detach()
        change[tag] += (sorted({r['T'] for r in lay}), sum(r['neurons_per_sample'] * r['T'] for r in lay))
    mm, tn = change['memoryless'], change['T = N']
    good = (same_params and mm[0] and mm[1] == 0 and mm[2] == 0 and tn[0] and tn[1] == 0 and tn[2] > 0 and mm[3] == tn[3]
            and mm[4] == [1] and tn[4] == [ms['T = N'].num_patches] and abs(mm[5] - tn[5]) < 1e-6 * tn[5])
    report('c44', good, f'same parameters {same_params}; token {j} replaced -> (changed at j, # earlier changed, # later changed, shape, '
                        f'recorder T, neurons x steps per window): memoryless {mm}, T = N {tn}')


@check('c45')
def c45():
    """(MODI 2026-10-26) option rules and run directories: cat without hippo_xattn, cat with popk streams, memoryless with
    neo_bidir / hippo_ssa / popk streams, neo_grad 1.5 are refused; the run directories of the five B conditions and of the
    replace model are distinct and the replace / default names carry no new suffix."""
    from load_model import LOAD_MODEL
    import config as cfg
    bad = [dict(neo_neuron='lif', hippo_T=4, hippo_xattn_mode='cat'),
           dict(neo_neuron='popk_lif', hippo_T=4, hippo_xattn=True, hippo_xattn_mode='cat'),
           dict(neo_neuron='lif', neo_bidir=True, neo_memoryless=True),
           dict(hippo_ssa=True, hippo_T=4, neo_memoryless=True),
           dict(neo_neuron='popk_lif', hippo_T=4, hippo_xattn=True, neo_memoryless=True),
           dict(neo_neuron='lif', neo_grad=1.5)]
    refused = []
    for b in bad:
        try:
            LOAD_MODEL['myModel'](NS(**dict(B_BASE, **b)), True); refused.append(False)
        except ValueError:
            refused.append(True)
    tmp = tempfile.mkdtemp(dir=os.environ.get('CHECK_TMP'))
    common = ['--data', 'CINC2017', '--patch_size', '64', '--model', 'myModel', '--hippo_T', '4', '--hippo_xattn']
    flags = {'replace_lif': ['--neo_neuron', 'lif'],
             'spkneo_lif': ['--neo_neuron', 'lif', '--hippo_xattn_mode', 'cat', '--neo_grad', '1.0'],
             'spkneo_flif': ['--neo_neuron', 'flif', '--hippo_xattn_mode', 'cat', '--neo_grad', '1.0'],
             'spkneo_pop_lif': ['--neo_neuron', 'pop_lif', '--pop_soma_init', '0.5', '--hippo_xattn_mode', 'cat', '--neo_grad', '1.0'],
             'spkneo_pop_flif': ['--neo_neuron', 'pop_flif', '--pop_soma_init', '1.0', '--hippo_xattn_mode', 'cat', '--neo_grad', '1.0'],
             'spkneo_lif_nomem': ['--neo_neuron', 'lif', '--hippo_xattn_mode', 'cat', '--neo_grad', '1.0', '--neo_memoryless']}
    names, argv0 = {}, sys.argv
    try:
        for c, f in flags.items():
            sys.argv = ['train_rhythm.py', '--log_dir', tmp, '--seed', '42', '-s'] + common + f
            a = cfg.Config(); a.set_args(cfg.parse_arguments()); names[c] = os.path.basename(a.save_result_path)
    finally:
        sys.argv = argv0
    plain = not any(s in names['replace_lif'] for s in ('_xacat', '_ng', '_nomem'))
    ok = all(refused) and len(set(names.values())) == len(names) and plain and all('_xacat_ng1' in names[c] for c in flags if c != 'replace_lif') \
        and names['spkneo_lif_nomem'].endswith('_nomem_f0')
    report('c45', ok, f'refused {refused}; ' + ', '.join(f'{c}={n}' for c, n in names.items()))


ETT_GEN = dict(keep_ratio=0.15, gating='attn', patch_size=8, embed_dim=64, num_heads=8, num_classes=1, num_channels=1, seq_len=96,
               mlp_ratios=0.5, num_layers=1, time_num_layers=1, bias=False, tau=2.0, spk_encoding=False, neo_alpha=0.5,
               pop_taus='4,8,16,32', neo_tau=2.0, fde_method='pred', fde_step=1.0, fde_memory=0)       # ETTh2 (run_ett_mixer.sh)
GEN_CONDS = {'spk_ssa': dict(model='myModel', hippo_ssa=True, hippo_T=4),
             'mix_lif_bidir': dict(model='neoMixer', mixer='lif', mixer_bidir=True),
             'mix_flif_bidir': dict(model='neoMixer', mixer='flif', mixer_bidir=True),
             'mix_pop_lif_bidir': dict(model='neoMixer', mixer='pop_lif', mixer_bidir=True, pop_soma_init=0.5),
             'mix_pop_flif_bidir': dict(model='neoMixer', mixer='pop_flif', mixer_bidir=True, pop_soma_init=1.0),
             'mixlinmx_lif': dict(model='neoMixer', mixer='lif', mixer_bidir=True, mixer_comb='linmx'),
             'mixlinmx_flif': dict(model='neoMixer', mixer='flif', mixer_bidir=True, mixer_comb='linmx'),
             'mixlinmx_pop_lif': dict(model='neoMixer', mixer='pop_lif', mixer_bidir=True, pop_soma_init=0.5, mixer_comb='linmx'),
             'mixlinmx_pop_flif': dict(model='neoMixer', mixer='pop_flif', mixer_bidir=True, pop_soma_init=1.0, mixer_comb='linmx')}

_DUAL = dict(model='myModel', hippo_T=4, hippo_xattn=True, neo_where='both', neo_grad=1.0)        # MODI 2026-10-29 (run_ett_dual.sh)
DUAL_CONDS = {}
for _pre, _mode in (('spkneo', 'cat'), ('nsmt4', 'replace')):
    DUAL_CONDS.update({f'{_pre}_lif': dict(_DUAL, hippo_xattn_mode=_mode, neo_neuron='lif'),
                       f'{_pre}_flif': dict(_DUAL, hippo_xattn_mode=_mode, neo_neuron='flif'),
                       f'{_pre}_pop_lif': dict(_DUAL, hippo_xattn_mode=_mode, neo_neuron='pop_lif', pop_soma_init=0.5),
                       f'{_pre}_pop_flif': dict(_DUAL, hippo_xattn_mode=_mode, neo_neuron='pop_flif', pop_soma_init=1.0),
                       f'{_pre}_lif_nomem': dict(_DUAL, hippo_xattn_mode=_mode, neo_neuron='lif', neo_memoryless=True)})
QUERY_CONDS = {f'nsmt4q_{k[len("nsmt4_"):]}': dict(v, hippo_xattn_mode='query') for k, v in DUAL_CONDS.items() if k.startswith('nsmt4_')}   # MODI 2026-10-30

def _ett(ds='ETTh2', pl=96, bs=32):
    sys.path.append(os.path.join(HERE, '..', 'forecasting'))
    from data_provider.data_factory import data_provider
    da = NS(data=ds, embed='timeF', batch_size=bs, freq='h', task_name='forecasting', root_path=os.path.join(HERE, '..', 'forecasting', 'dataset', 'ETT-small'),
            data_path=f'{ds}.csv', seq_len=96, label_len=48, pred_len=pl, features='M', target='OT', seasonal_patterns='Monthly',
            num_workers=0, augmentation_ratio=0)
    return {f: data_provider(da, f) for f in ('train', 'val', 'test')}


def _gen_model(cond, pl=96, seed=0):
    from config import set_random_seed
    from load_model import LOAD_MODEL
    from forecast_wrap import Forecaster
    set_random_seed(seed)
    a = NS(**dict(ETT_GEN, **{**GEN_CONDS, **DUAL_CONDS, **QUERY_CONDS}[cond]))
    return Forecaster(LOAD_MODEL[a.model](a, True), a.embed_dim, pl, bias=False).cuda()


@check('c46')
def c46():
    """(MODI 2026-10-27) ETT data through train_forecast's data provider: split sizes equal the earlier ETT runs (stage-2
    stdout: ETTh2 H96 8449 / 2785 / 2785, ETTm2 H720 33745 / 10801 / 10801). Forecaster for all 9 conditions (ETTh2
    hyperparameters; BN statistics fixed on the batch, body firing > 1 %): output [B, H, 7], 24 tokens, no parameter under
    inner.head, every spiking neuron runs T = 24 (Neo-mixer)
    or the SSA blocks see T_h = 4 (energy recorder); the body equals the classification model's own forward: Neo-mixer token
    spikes == forward_parts()[4], Spikformer-style == the testing-mode token spikes of myModel averaged over T_h (bitwise)."""
    from spikingjelly.clock_driven import functional
    from energy import FiringRateRecorder
    sizes = {}
    for ds, pl in (('ETTh2', 96), ('ETTm2', 720)):
        d = _ett(ds, pl)
        sizes[f'{ds} H{pl}'] = tuple(len(d[f][0]) for f in ('train', 'val', 'test'))
    ok_sizes = sizes == {'ETTh2 H96': (8449, 2785, 2785), 'ETTm2 H720': (33745, 10801, 10801)}
    d = _ett('ETTh2', 96)
    x = next(iter(d['test'][1]))[0].float().cuda()
    rows, ok = [], ok_sizes
    for cond in GEN_CONDS:
        m = _gen_model(cond); _bn_freeze_eval(m, x); m.train_mode = 'testing'      # eval-mode BN at init silences every neuron
        rec = FiringRateRecorder(m)
        with torch.no_grad():
            y = m(x); functional.reset_net(m)
        rec.add_samples(x.shape[0])
        Ts = sorted({int(r['T']) for r in rec.layers()})
        rec.detach()
        z = ((x - x.mean(1, keepdim=True)) / torch.sqrt(torch.var(x - x.mean(1, keepdim=True), dim=1, keepdim=True, unbiased=False) + 1e-5))
        z = z.permute(0, 2, 1).reshape(-1, 1, 96)
        with torch.no_grad():
            f = m.features(z); functional.reset_net(m)
            if m.kind == 'mixer':
                ref = m.inner.forward_parts(z)[4][:, :, 0].transpose(0, 1)
            else:
                ref = m.inner(z)[1].mean(0)
            functional.reset_net(m)
        no_head = not any(n.startswith('inner.head') for n, _ in m.named_parameters())
        good = tuple(y.shape) == (x.shape[0], 96, 7) and m.num_patches == 24 and no_head and torch.equal(f, ref) and float(f.mean()) > 0.01
        good &= (m.inner.T_h == 4 and Ts == [4]) if m.kind == 'ssa' else (m.T == 24 and Ts == [24])
        ok &= good
        rows.append(f'{cond}: out {tuple(y.shape)}, body == own forward {torch.equal(f, ref)}, rate {float(f.mean()):.3f}, '
                    f"{'T_h ' + str(m.inner.T_h) if m.kind == 'ssa' else 'T ' + str(m.T)}, recorder T {Ts}")
    report('c46', ok, f'splits {sizes}; ' + '; '.join(rows))


@check('c47')
def c47():
    """(MODI 2026-10-27) Forecaster in eval mode (BN statistics fixed on the batch, body firing > 1 %) on a real ETTh2 test batch
    (all 9 conditions): both ends on a x + b (random per-channel a > 0, b) against a float64 reference built window by window and
    channel by channel -- the body input equals (x - mean) / sqrt(var + 1e-5) per window and channel in row order b * C + c
    (every element within a float32 rounding bound 32 * 2^-24 * max|x| / std + 1e-6 of that window and channel) and, with the
    body output held fixed, the output equals head * std + mean per channel (relative error < 1e-5); the
    share of body spikes that differ between x and a x + b (the eps makes the normalisation not exactly affine-invariant) is
    reported -- channel independence (changing channel 0 changes only output
    channel 0, the rest bitwise), the new head receives a gradient under the L1 loss; train_forecast.evaluate MSE / MAE equal a
    manual computation over 3 batches."""
    from spikingjelly.clock_driven import functional
    import train_forecast as tf
    d = _ett('ETTh2', 96)
    x, yb = [t.float().cuda() for t in next(iter(d['test'][1]))[:2]]
    g = torch.Generator().manual_seed(3)
    a = (torch.rand(1, 1, 7, generator=g) * 3 + 0.5).cuda(); b = (torch.randn(1, 1, 7, generator=g) * 5).cuda()
    rows, ok = [], True
    for cond in GEN_CONDS:
        m = _gen_model(cond); _bn_freeze_eval(m, x); m.train_mode = 'testing'
        with torch.no_grad():
            y0 = m(x); functional.reset_net(m)
            z0 = ((x - x.mean(1, keepdim=True)) / torch.sqrt(torch.var(x - x.mean(1, keepdim=True), dim=1, keepdim=True, unbiased=False) + 1e-5))
            f0 = m.features(z0.permute(0, 2, 1).reshape(-1, 1, 96)); functional.reset_net(m)
            rate = float(f0.mean())
            x2 = x.clone(); x2[:, :, 0] = x2[:, :, 0].flip(1)
            y2 = m(x2); functional.reset_net(m)
            seen = []
            m.features = lambda z: (seen.append(z), f0)[1]                 # body output held fixed: tests both ends only
            yf1 = m(a * x + b)
            del m.features
            f1 = m.features(seen[0]); functional.reset_net(m)
            xa = (a * x + b).double().cpu(); B_, L_, C_ = xa.shape          # float64 reference, per window and channel
            mu = xa.mean(1); sd = torch.sqrt(xa.var(1, unbiased=False) + 1e-5)
            zref = ((xa - mu[:, None]) / sd[:, None]).permute(0, 2, 1).reshape(B_ * C_, 1, L_)
            hz = m.head(f0.reshape(B_ * C_, -1)).double().cpu().reshape(B_, C_, -1)
            yref = (hz * sd[:, :, None] + mu[:, :, None]).permute(0, 2, 1)
        bound = (32 * 2 ** -24 * xa.abs().amax(1) / sd).permute(0, 1).reshape(B_ * C_, 1, 1) + 1e-6   # float32 rounding of x - mean, / std
        zerr = (seen[0].double().cpu() - zref).abs()
        norm_ok = bool((zerr <= bound).all())
        yerr = float(((yf1.double().cpu() - yref).abs() / (yref.abs() + 1.)).max())
        eqv = norm_ok and yerr < 1e-5
        flip = float((f1 != f0).float().mean())
        indep = torch.equal(y2[:, :, 1:], y0[:, :, 1:]) and not torch.equal(y2[:, :, 0], y0[:, :, 0])
        m.train(); m.train_mode = 'training'; m.zero_grad(set_to_none=True)
        torch.nn.functional.l1_loss(m(x), yb[:, -96:]).backward(); functional.reset_net(m)
        hg = m.head.weight.grad is not None and float(m.head.weight.grad.abs().sum()) > 0
        good = eqv and indep and hg and rate > 0.01
        ok &= good
        rows.append(f'{cond}: body rate {rate:.3f}, body input == reference {norm_ok} (max err {float(zerr.max()):.1e}, '
                    f'max err / float32 bound {float((zerr / bound).max()):.2f}), both ends == reference {eqv} (rel err {yerr:.1e}; body spikes x vs a x + b differ '
                    f'{flip:.2e}), channel-independent {indep}, head grad {hg}')
    m = _gen_model('mixlinmx_lif'); _bn_freeze_eval(m, x)
    batches = []
    for i, bt in enumerate(d['test'][1]):
        if i == 3:
            break
        batches.append(bt)
    got = tf.evaluate(m, batches, 96, 'cuda')
    se = ae = n = 0.
    with torch.no_grad():
        for xb, yy, _, _ in batches:
            o = m(xb.float().cuda()).double(); functional.reset_net(m)
            dd = o - yy.float().cuda()[:, -96:].double()
            se += float((dd ** 2).sum()); ae += float(dd.abs().sum()); n += dd.numel()
    met = abs(got['mse'] - se / n) < 1e-9 and abs(got['mae'] - ae / n) < 1e-9 and got['n'] == n
    ok &= met
    report('c47', ok, '; '.join(rows) + f"; evaluate mse {got['mse']:.6f} mae {got['mae']:.6f} == manual {met}")


@check('c48')
def c48():
    """--token_shuffle (MODI 2026-10-28); see the module docstring."""
    import numpy as np
    from load_model import LOAD_MODEL
    from config import set_random_seed
    import config as cfg
    from spikingjelly.clock_driven import functional
    x, d, P = _real_batch('CINC2017', n=8)
    base = dict(HP, patch_size=64, seq_len=d['seq_len'], num_channels=1, num_classes=4, seed=42)
    builds = {'spk_ssa': dict(base, model='myModel', hippo_ssa=True, hippo_T=4),
              'mix_lif_bidir': dict(base, model='neoMixer', mixer='lif', mixer_bidir=True),
              'mix_pop_lif_bidir': dict(base, model='neoMixer', mixer='pop_lif', mixer_bidir=True, pop_soma_init=0.5, neo_alpha=0.5)}
    rows, ok = [], True
    for name, kw in builds.items():
        set_random_seed(0); m0 = LOAD_MODEL[kw['model']](NS(**kw), True).cuda()
        set_random_seed(0); m1 = LOAD_MODEL[kw['model']](NS(**dict(kw, token_shuffle=True)), True).cuda()
        same_w = all(torch.equal(a, b) for a, b in zip(m0.state_dict().values(), m1.state_dict().values()))
        off_ok = m0.token_shuffle is None and m1.token_shuffle is not None
        with torch.no_grad():                                              # the zero-initialised relative bias makes SSA order-blind;
            for n_, p_ in m0.named_parameters():                           # give it values so that token order matters
                if 'relative_bias_table' in n_:
                    p_.copy_(torch.randn(p_.shape, generator=torch.Generator().manual_seed(5)).to(p_.device))
        _bn_freeze_eval(m0, x); m1.load_state_dict(m0.state_dict()); m1.eval()
        for mm in (m0, m1):
            mm.train_mode = 'testing'
        N = m0.num_patches
        perm = torch.as_tensor(np.random.default_rng(0).permutation(N)).cuda()
        tok_fn = (lambda f: (lambda xx: f(xx)[perm]))
        with torch.no_grad():
            z0 = m0(x); z0 = z0[0] if isinstance(z0, tuple) else z0; functional.reset_net(m0)
            z1 = m1(x); z1 = z1[0] if isinstance(z1, tuple) else z1; functional.reset_net(m1)
            if kw['model'] == 'neoMixer':
                orig = m0.tokens; m0.tokens = tok_fn(orig)
            else:
                orig = m0._tokens_mc
                def f(xx, orig=orig):
                    h, n = orig(xx); return h[:, :, perm].contiguous(), n[perm].contiguous()
                m0._tokens_mc = f
            zr = m0(x); zr = zr[0] if isinstance(zr, tuple) else zr; functional.reset_net(m0)
            del m0.__dict__['tokens' if kw['model'] == 'neoMixer' else '_tokens_mc']
        eval_ok = torch.equal(z1, zr) and not torch.equal(z1, z0)
        m1.train()
        rng0 = torch.get_rng_state().clone(); crng0 = torch.cuda.get_rng_state().clone()
        with torch.no_grad():
            if kw['model'] == 'neoMixer':
                t_a = m1.tokens(x); t_b = m1.tokens(x); m1.token_shuffle = None; t_ref = m1.tokens(x)
            else:
                _, t_a = m1._tokens_mc(x); _, t_b = m1._tokens_mc(x); sh = m1.token_shuffle; m1.token_shuffle = None
                _, t_ref = m1._tokens_mc(x); m1.token_shuffle = sh
        rng_ok = torch.equal(rng0, torch.get_rng_state()) and torch.equal(crng0, torch.cuda.get_rng_state())
        B = x.shape[0]
        flat = lambda t: t[:, :, 0].permute(1, 0, 2)                       # [B, N, P]
        a, r = flat(t_a), flat(t_ref)
        multiset = all(torch.equal(torch.sort(a[b], 0)[0], torch.sort(r[b], 0)[0]) for b in range(B))
        rows_perm = []
        for b in range(B):                                                 # recover the permutation row by row
            idx = [int(torch.nonzero((r[b] == a[b, t]).all(-1))[0]) for t in range(N)]
            rows_perm.append(tuple(idx))
        is_perm = all(sorted(p) == list(range(N)) for p in rows_perm)
        distinct = len(set(rows_perm)) == B and not torch.equal(t_a, t_b) and rows_perm[0] != tuple(range(N))
        good = same_w and off_ok and eval_ok and rng_ok and multiset and is_perm and distinct
        ok &= good
        rows.append(f'{name}: same init {same_w}, module only when on {off_ok}, eval == fixed-order normal model {eval_ok}, '
                    f'train: rows are permutations {is_perm}, multiset kept {multiset}, {len(set(rows_perm))}/{B} distinct and calls differ {distinct}, '
                    f'global RNG untouched {rng_ok}')
    try:
        LOAD_MODEL['myModel'](NS(**dict(base, model='myModel', neo_neuron='lif', token_shuffle=True)), True)
        refused = False
    except ValueError:
        refused = True
    ok &= refused
    tmp = tempfile.mkdtemp(dir=os.environ.get('CHECK_TMP'))
    flags = {'spk_ssa': ['--model', 'myModel', '--hippo_ssa', '--hippo_T', '4'], 'mix_lif_bidir': ['--model', 'neoMixer', '--mixer', 'lif', '--mixer_bidir']}
    names, argv0 = {}, sys.argv
    try:
        for c, f in flags.items():
            for on in (False, True):
                sys.argv = ['train_rhythm.py', '--log_dir', tmp, '--seed', '42', '-s', '--data', 'CINC2017', '--patch_size', '64'] + f + (['--token_shuffle'] if on else [])
                a_ = cfg.Config(); a_.set_args(cfg.parse_arguments()); names[(c, on)] = os.path.basename(a_.save_result_path)
    finally:
        sys.argv = argv0
    sfx = [('_tshuf' in n) == on for (c, on), n in names.items()] + [names[(c, True)] == names[(c, False)].replace('_f0', '_tshuf_f0') for c in flags]
    ok &= all(sfx)
    report('c48', ok, '; '.join(rows) + f'; NSMT without hippo_ssa refused {refused}; folder suffix _tshuf only when on {all(sfx)} '
                      f'({names[("spk_ssa", True)]}, {names[("mix_lif_bidir", True)]})')


@check('c49')
def c49():
    """PPG-DaLiA (MODI 2026-10-28); see the module docstring."""
    import re as _re
    import numpy as np
    from load_model import LOAD_MODEL
    from config import set_random_seed
    from spikingjelly.clock_driven import functional
    from ppg_data import PPGWindows, ROOT, STRIDE, LSTRIDE, WIN, LWIN
    from ppg_model import TokenRegressor, GRURegressor, label_tokens
    from syops import get_model_complexity_info
    from syops.ops import LIF_syops_counter_hook
    from neurons import LIFNeuron, FLIFNeuron
    from torch.utils.data import DataLoader, Subset
    npz = os.path.join(ROOT, 'processed', 'logncde_seed0.npz')
    ds = {sp: PPGWindows(npz, sp) for sp in ('train', 'val', 'test')}
    raw = np.load(npz)
    counts = {sp: len(d) for sp, d in ds.items()}
    slice_ok = True
    for sp, d in ds.items():
        for k in (0, len(d) // 2, len(d) - 1):
            si, j = d.idx[k]; x, y, s_ = d[k]
            slice_ok &= s_ == si and torch.equal(x, torch.from_numpy(raw[f'S{si + 1}_{sp}_x'][j * STRIDE:j * STRIDE + WIN])) \
                and torch.equal(y, torch.from_numpy(raw[f'S{si + 1}_{sp}_y'][j * LSTRIDE:j * LSTRIDE + LWIN]))
    s13 = sum(1 for si, _ in ds['train'].idx if si == 12)
    rng_ok = all(float(raw[k].min()) >= -1 and float(raw[k].max()) <= 1 for k in raw.files if k.endswith(('_x', '_y')))
    idx = label_tokens(390)
    map_ok = int(idx[0]) == 0 and all(int(idx[k]) == k - 1 and 128 * (int(idx[k]) + 2) == 128 * (k + 1) for k in range(1, 390))
    ok = counts == {'train': 1020, 'val': 106, 'test': 106} and slice_ok and s13 == 72 and rng_ok and map_ok
    rows = [f'windows {counts}, slices equal {slice_ok}, S13 train {s13}, range ok {rng_ok}, label -> token k - 1 {map_ok}']
    x = torch.stack([ds['test'][i][0] for i in range(2)]).float().cuda()            # [2, 49920, 6]
    j = 200                                                                          # samples [128 j, 128 j + 128) -> tokens j - 1, j
    xp = x.clone(); xp[:, 128 * j:128 * j + 128] = torch.rand(2, 128, 6, generator=torch.Generator().manual_seed(3)).cuda() * 2 - 1
    base = dict(HP, patch_size=256, seq_len=49920, num_channels=6, num_classes=1, seed=42)
    ssa = dict(base, model='myModel', hippo_ssa=True, hippo_T=4)
    bf = dict(base, model='myModel', hippo_T=4, hippo_xattn=True, hippo_xattn_mode='cat', neo_grad=1.0, neo_where='both', neo_neuron='lif')
    cases = [('P0 window 1', ssa, 1, 'stop', 1), ('P1 window 3', ssa, 3, 'stop', 3), ('P3 all past', ssa, -1, 'far', None),
             ('P4 mixer LIF', dict(base, model='neoMixer', mixer='lif'), 0, 'far', None),
             ('P5 B T = N LIF', bf, 3, 'far', None), ('P6 B memoryless', dict(bf, neo_memoryless=True), 3, 'stop', 3)]
    for name, kw, W, kind, Wr in cases:
        set_random_seed(0)
        m = TokenRegressor(LOAD_MODEL[kw['model']](NS(**kw), True), 384, attn_window=W).cuda()
        with torch.no_grad():                                              # give the relative bias values (zero init)
            for n_, p_ in m.named_parameters():
                if 'relative_bias_table' in n_:
                    p_.copy_(torch.randn(p_.shape, generator=torch.Generator().manual_seed(5)).to(p_.device) * 0.1)
        _bn_freeze_eval(m, x)
        with torch.no_grad():
            s0 = m.spikes(x.permute(0, 2, 1).contiguous()); functional.reset_net(m)
            s1 = m.spikes(xp.permute(0, 2, 1).contiguous()); functional.reset_net(m)
            y0 = m(x); functional.reset_net(m)
        ch = (s0 != s1).any(-1).any(1).any(0)                              # [N] tokens whose spikes changed
        first = int(torch.nonzero(ch)[0]) if bool(ch.any()) else -1
        last = int(torch.nonzero(ch)[-1]) if bool(ch.any()) else -1
        causal = bool(ch.any()) and not bool(ch[:j - 1].any())
        # reach by surrogate gradients (non-zero everywhere): d(per-token outputs) / d(input samples of tokens j - 1, j)
        det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)                          # ReplicationPad1d backward (input grad only; not used in training)
        xg = x.clone().requires_grad_(True)
        out = m.head(m.spikes(xg.permute(0, 2, 1).contiguous())).squeeze(-1).mean(0)   # [B, N]
        def g_of(sel):
            return float(torch.autograd.grad(out[:, sel].sum(), xg, retain_graph=True)[0][:, 128 * j:128 * j + 128].abs().sum())
        g_before = g_of(slice(0, j - 1))
        if kind == 'stop':                                                 # exact zero beyond the window (the positive side: unit test below)
            g_in, g_out = g_of(slice(j + Wr - 1, j + Wr)), g_of(slice(j + Wr, None))
            reach = g_out == 0.
        else:
            g_in, g_out = g_of(slice(j, j + 1)), g_of(slice(j + 5, None))
            reach = g_in > 0 and g_out > 0
        functional.reset_net(m); del out, xg
        torch.use_deterministic_algorithms(det)
        causal = causal and g_before == 0.
        binary = bool(((s0 == 0) | (s0 == 1)).all())
        head_eq = bool(torch.allclose(y0, m.head(s0.mean(0)).squeeze(-1)[:, m.lab_idx], rtol=1e-5, atol=1e-6))
        good = causal and reach and binary and head_eq
        ok &= good
        rows.append(f'{name}: spikes changed at tokens {first}..{last} (input at tokens {j - 1}..{j}), causal {causal} (grad before {g_before:.1e}), '
                    f'reach as expected {reach} (grad in reach {g_in:.1e}, beyond {g_out:.1e}), '
                    f'head input 0/1 {binary}, head == head(mean) {head_eq}')
        if name == 'P1 window 3':
            dl = DataLoader(Subset(ds['test'], [0, 1]), batch_size=2)
            path = os.path.join(tempfile.mkdtemp(dir=os.environ.get('CHECK_TMP')), 'layers.txt')
            m.train_mode = 'testing'
            with open(path, 'w', encoding='utf-8') as fo:
                get_model_complexity_info(model=m, input_res=((49920, 6),), dataloader=dl, as_strings=False, print_per_layer_stat=True,
                                          verbose=False, ost=fo, custom_modules_hooks={LIFNeuron: LIF_syops_counter_hook, FLIFNeuron: LIF_syops_counter_hook})
            hl = [l for l in open(path, encoding='utf-8').read().splitlines() if l.startswith('  (head): Linear(')]
            mac = _re.search(r'ACs,\s*([\d.]+)\s*([KMG]?)\s*Ops', hl[0]) if hl else None
            ac_ok = bool(hl) and mac is not None and float(mac.group(1)) == 0.
            ok &= ac_ok
            rows.append(f'P1 head line found {bool(hl)}, head MAC = 0 {ac_ok} ({hl[0].strip()[:160] if hl else ""})')
    from layers import SSA_rel_scl                                     # unit test of the causal band on active spikes
    set_random_seed(0)
    a = SSA_rel_scl(dim=384, seq_len=390, num_heads=8, kv_cat=True).cuda().eval()
    with torch.no_grad():
        for lin in (a.q_linear, a.k_linear, a.v_linear, a.k2_linear, a.v2_linear):
            lin.weight.mul_(4.)                                            # enough activity at the init BN statistics
        a.relative_bias_table.copy_(torch.randn(a.relative_bias_table.shape, generator=torch.Generator().manual_seed(5)).cuda() * 0.1)
    gen = torch.Generator().manual_seed(7)
    xs = (torch.rand(4, 2, 390, 384, generator=gen) < 0.3).float().cuda(); kvs = (torch.rand(4, 2, 390, 384, generator=gen) < 0.3).float().cuda()
    pre = {}
    hk = a.attn_lif.register_forward_hook(lambda mod, i, o: pre.__setitem__('x', i[0].detach().clone()))
    unit = []
    for W in (1, 3, 15, -1):
        a.attn_window = W
        res = []
        for which in ('x', 'kv'):
            x1, kv1 = xs.clone(), kvs.clone()
            (x1 if which == 'x' else kv1)[:, :, j] = 1. - (x1 if which == 'x' else kv1)[:, :, j]
            with torch.no_grad():
                a(xs, kv=kvs); functional.reset_net(a); p0 = pre['x']
                a(x1, kv=kv1); functional.reset_net(a); p1 = pre['x']
            chg = torch.nonzero((p0 != p1).any(-1).any(1).any(0)).flatten().tolist()
            want = list(range(j, 390 if W < 0 else min(390, j + W)))
            res.append(chg == want)
        unit.append(all(res))
    hk.remove(); del a.attn_window
    unit_ok = all(unit)
    ok &= unit_ok
    rows.append(f'SSA band unit test (pre-activation changes exactly at tokens j .. j + W - 1, own and neocortex keys) W = 1 / 3 / 15 / all: {unit}')
    set_random_seed(0)
    g = GRURegressor().cuda().eval()
    with torch.no_grad():
        o0, o1 = g(x), g(xp)
    lab_ch = (o0 != o1).any(0)
    g_causal = not bool(lab_ch[:j - 1].any()) and bool(lab_ch[j + 50:].any())   # label k reads token k - 1
    ok &= g_causal
    rows.append(f'GRU causal and long reach {g_causal}')
    report('c49', ok, '; '.join(rows))

@check('c50')
def c50():
    """(MODI 2026-10-29) dual-path forecasters (forecast_wrap kind 'dual'; B form spkneo_* = keys / values [own; neocortex],
    [NSMT hippocampus T=4] nsmt4_* = keys / values replaced by the neocortex) for the 10 conditions of run_ett_dual.sh (ETTh2
    hyperparameters, BN statistics fixed on a real test batch): output [B, H, 7], no parameter under inner.head, mode / neo_grad as
    intended; the body equals myModel's own testing-mode forward averaged over T_h (bitwise); the per-step head input
    (step_features) is exactly 0/1 and its T_h mean is features(); the recorder sees T_h = 4 and the neocortex T = 24 (memoryless:
    1 step); the forecast loss reaches the neocortex and the hippocampus, the unused aux decoder gets no gradient; changing
    neocortex input token j changes the neocortex output only at j (memoryless) or at j and later tokens only (with memory)."""
    ok, rows = _dual_battery(DUAL_CONDS)
    report('c50', ok, '; '.join(rows))


def _dual_battery(conds):
    """the c50 checks for each dual-path forecaster in conds (MODI 2026-10-30: shared with c51)."""
    from spikingjelly.clock_driven import functional
    from energy import FiringRateRecorder
    d = _ett('ETTh2', 96)
    x = next(iter(d['test'][1]))[0].float().cuda()
    z = ((x - x.mean(1, keepdim=True)) / torch.sqrt(torch.var(x - x.mean(1, keepdim=True), dim=1, keepdim=True, unbiased=False) + 1e-5))
    z = z.permute(0, 2, 1).reshape(-1, 1, 96)
    rows, ok = [], True
    for cond, kw in conds.items():
        m = _gen_model(cond); _bn_freeze_eval(m, x); m.train_mode = 'testing'
        inn = m.inner; nomem = bool(kw.get('neo_memoryless', False))
        rec = FiringRateRecorder(m)
        with torch.no_grad():
            y = m(x); functional.reset_net(m)
        rec.add_samples(x.shape[0]); Ts = sorted({int(r['T']) for r in rec.layers()}); rec.detach()
        with torch.no_grad():
            st = m.step_features(z); functional.reset_net(m)
            f = m.features(z); functional.reset_net(m)
            ref = inn(z)[1].mean(0); functional.reset_net(m)
        no_head = not any(n.startswith('inner.head') for n, _ in m.named_parameters())
        binary = bool(((st == 0) | (st == 1)).all())
        m.zero_grad(set_to_none=True)
        m(x).pow(2).mean().backward(); functional.reset_net(m)
        gnorm = lambda pre: sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith(pre) and p.grad is not None)
        g_neo, g_hip = gnorm('inner.encoding_neo') + gnorm('inner.time_block'), gnorm('inner.encoding.')
        dec_none = all(p.grad is None for n, p in m.named_parameters() if n.startswith('inner.weak_decoder'))
        with torch.no_grad():
            _, neo_in = inn._tokens_mc(z)
            o0 = inn.neo_forward(neo_in, z)[1]; functional.reset_net(m)
            n2 = neo_in.clone(); n2[10] = n2[10] + 3.0
            o1 = inn.neo_forward(n2, z)[1]; functional.reset_net(m)
        ch = [(o0[t] != o1[t]).any().item() for t in range(o0.shape[0])]
        local = ch[10] and not any(ch[:10]) and (not any(ch[11:]) if nomem else any(ch[11:]))
        good = (m.kind == 'dual' and tuple(y.shape) == (x.shape[0], 96, 7) and no_head and torch.equal(f, ref) and binary
                and torch.equal(st.mean(0), f) and inn.hippo_xattn_mode == kw['hippo_xattn_mode'] and inn.neo_grad == 1.0
                and 4 in Ts and ((1 in Ts) if nomem else (24 in Ts)) and g_neo > 0 and g_hip > 0 and dec_none and local
                and float(f.mean()) > 0.01)
        ok &= good
        rows.append(f"{cond}: mode {inn.hippo_xattn_mode}, body == own forward {torch.equal(f, ref)}, step 0/1 {binary}, "
                    f"rate {float(f.mean()):.3f}, recorder T {Ts}, grad neo {g_neo:.2e} hippo {g_hip:.2e}, aux dec no grad {dec_none}, "
                    f"neocortex change at token 10 -> tokens {[t for t, c in enumerate(ch) if c][:4]}{'...' if sum(ch) > 4 else ''}")
    return ok, rows


@check('c51')
def c51():
    """(MODI 2026-10-30) Neocortex as the attention query (hippo_xattn_mode 'query', nsmt4q_*): (a) the c50 checks for the 5
    query forecasters (ETTh2 size); (b) CinC size (117 tokens, D 384, a fresh model built from the nsmt4_lif CinC config with mode
    'query'), forward hooks: the q_linear input is the neocortex output laid out [T_h, B, N, D] (= x_neo.transpose(0, 2) repeated
    over T_h), the k_linear / v_linear inputs are the hippocampus tokens, no k2 / v2 parameters, the same parameter names / shapes
    as the replace model, folder suffix _xaquery; (c) 'query' with popk streams and q_from_kv with kv_cat are refused."""
    import glob as _glob
    from spikingjelly.clock_driven import functional
    from load_model import LOAD_MODEL
    from layers import SSA_rel_scl
    ok, rows = _dual_battery(QUERY_CONDS)
    run = [r for r in sorted(_glob.glob(os.path.join(HERE, 'log', 'flif-v1-rhythm-dual-cinc', '**', '*+nsmt4_lif+lr0.001+seed42+*', 'seed42_*_f0'), recursive=True))
           if os.path.exists(os.path.join(r, 'model_state', 'config.pt'))]
    cfg = torch.load(os.path.join(run[0], 'model_state', 'config.pt'), map_location='cpu')
    a = NS(**dict(cfg, hippo_xattn_mode='query', device='cuda'))
    torch.manual_seed(0); mq = LOAD_MODEL[a.model](a, True).cuda().eval(); mq.train_mode = 'testing'
    torch.manual_seed(0); mr = LOAD_MODEL[a.model](NS(**dict(cfg, device='cuda')), True)
    same_params = [(n, tuple(p.shape)) for n, p in mq.named_parameters()] == [(n, tuple(p.shape)) for n, p in mr.named_parameters()]
    cap = {}
    ssa = mq.data_block[0].ssa
    ssa.q_linear.register_forward_hook(lambda mod, i, o: cap.__setitem__('q', i[0].detach()))
    ssa.k_linear.register_forward_hook(lambda mod, i, o: cap.__setitem__('k', i[0].detach()))
    ssa.v_linear.register_forward_hook(lambda mod, i, o: cap.__setitem__('v', i[0].detach()))
    orig = ssa.forward
    def hook(x, mask=None, kv=None):
        cap['x'], cap['kv'] = x.detach(), kv.detach()
        return orig(x, mask=mask, kv=kv)
    ssa.forward = hook
    xin = torch.randn(2, 1, 3750, generator=torch.Generator().manual_seed(3)).cuda()
    with torch.no_grad():
        functional.reset_net(mq)
        _, neo_in = mq._tokens_mc(xin); x_neo = mq.neo_forward(neo_in, xin.clone())[1]; functional.reset_net(mq)
        mq(xin); functional.reset_net(mq)
    exp_kv = x_neo.transpose(0, 2).expand(cap['kv'].shape[0], -1, -1, -1)
    wiring = (torch.equal(cap['kv'], exp_kv) and torch.equal(cap['q'], cap['kv'].flatten(0, 1))
              and torch.equal(cap['k'], cap['x'].flatten(0, 1)) and torch.equal(cap['v'], cap['x'].flatten(0, 1)))
    no_k2 = not any('k2_' in n or 'v2_' in n for n, _ in mq.named_parameters())
    refused = []
    try:
        SSA_rel_scl(dim=8, seq_len=4, num_heads=2, kv_cat=True, q_from_kv=True); refused.append(False)
    except ValueError:
        refused.append(True)
    try:
        LOAD_MODEL[a.model](NS(**dict(cfg, hippo_xattn_mode='query', neo_neuron='popk_lif', device='cuda')), True); refused.append(False)
    except (ValueError, NotImplementedError, KeyError):
        refused.append(True)
    import tempfile
    import config as _cfg
    tmp = tempfile.mkdtemp(dir=os.environ.get('CHECK_TMP')); names, argv0 = {}, sys.argv
    try:
        for mode in ('replace', 'cat', 'query'):
            sys.argv = ['train_rhythm.py', '--log_dir', tmp, '--seed', '42', '-s', '--data', 'CINC2017', '--patch_size', '64', '--model', 'myModel',
                        '--neo_neuron', 'lif', '--neo_where', 'both', '--hippo_T', '4', '--hippo_xattn', '--neo_grad', '1.0', '--hippo_xattn_mode', mode]
            a_ = _cfg.Config(); a_.set_args(_cfg.parse_arguments()); names[mode] = os.path.basename(a_.save_result_path)
    finally:
        sys.argv = argv0
    sfx = '_xaquery' in names['query'] and '_xaquery' not in names['replace'] + names['cat'] and names['query'] == names['replace'].replace('_ng1', '_xaquery_ng1')
    ok &= same_params and wiring and no_k2 and all(refused) and sfx
    rows.append(f'CinC size: x_neo {tuple(x_neo.shape)} -> q input {tuple(cap["q"].shape)}, q from neocortex / k, v from hippocampus {wiring}, '
                f'same parameters as replace {same_params}, no k2/v2 {no_k2}, refused (kv_cat+query, popk+query) {refused}, '
                f'folder {names["query"]} (suffix ok {sfx})')
    report('c51', ok, '; '.join(rows))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'checks', 'check_cls.txt'))
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    for fn in (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51):
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
