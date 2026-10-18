"""nsmt_flif_v1 checks c1-c5 (GPU). Run from this folder:

    python check_flif.py -nd 0 [--base_run DIR] [--out DIR]

c1  default model == nsmt_base: fresh init (seed 42) state dict, forward outputs (training tuple and
    testing tensor) and one backward's gradients are bitwise equal to the same computation done by the
    nsmt_base code (run in a subprocess with cwd = nsmt_base/forecasting via --dump-base); the saved
    nsmt_base ETTh1 H96 checkpoint loads and gives bitwise-equal test-mode outputs.
c2  --neo_neuron lif == --neo_neuron flif --neo_alpha 1 (state dict, outputs, gradients), both placements.
c3  --hippo_only: no encoding_neo / time_block / weak_decoder; parameter count = base minus those three;
    forward returns a plain 4-D tensor in both modes.
c4  FiringRateRecorder's per-call firing rate equals syops' per-layer "Spike Rate" for every spiking
    neuron module (same loader, same order).
c5  test() on the nsmt_base checkpoint reproduces nsmt_base's final+result.csv in its first 13 columns
    (0.378303 ...) and appends the new energy columns as numbers.
"""
import os
import re
import sys
import csv
import json
import argparse
import glob
import subprocess
from argparse import Namespace

import torch
from spikingjelly.clock_driven import functional
from neurons import LIFNeuron, FLIFNeuron

HERE = os.path.dirname(os.path.abspath(__file__))
NSMT = os.path.dirname(os.path.dirname(HERE))
BASE_DIR = os.path.join(NSMT, 'nsmt_base', 'forecasting')
BASE_RUN = os.path.join(BASE_DIR, 'log', 'ett-paper-check-20261014', 'ETTh1', '261014',
                        '261014+model+myModel+dataset+ETTh1+seq_len+96+pred_len+96+patch_size+8+keep_ratio+0.25'
                        '+embed_dim+32+num_heads+2+alpha+0.8+mlp_ratios+1.0+time_num_layers+2.0+gating+attn+lr+0.0005')
NEO_PARTS = ('encoding_neo', 'time_block', 'weak_decoder')


def build(run_dir, gpu, seed=42, train=True, **over):
    from config import Config, set_random_seed
    from model import LOAD_MODEL
    cfg = Config()
    cfg.load_args(run_dir, Namespace(saved_epoch=[1], num_device=gpu))
    for k, v in over.items():
        setattr(cfg, k, v)
    set_random_seed(seed)
    model = LOAD_MODEL['myModel'](cfg, train=train)
    return cfg, model.to(cfg.device)


def one_batch(cfg):
    from data_provider.data_factory import data_provider
    _, loader = data_provider(cfg, flag='test')
    x, y, _, _ = next(iter(loader))
    return x.float().to(cfg.device), y.float().to(cfg.device)


def forwards(model, x, backward=True, train_mode=False):
    """Outputs in 'training' (tuple) and 'testing' modes, plus gradients of one backward. eval() by default;
    train_mode=True uses BatchNorm batch statistics (no dropout is applied in these forwards), which is
    what makes a freshly initialised neocortex fire at all (eval-mode running stats give ~0 current)."""
    model.train() if train_mode else model.eval()
    functional.reset_net(model); model.train_mode = 'training'; out_tr = model(x)
    grads = {}
    if backward:
        model.zero_grad(set_to_none=True)
        loss = (out_tr[0] if isinstance(out_tr, tuple) else out_tr).float().mean()
        if isinstance(out_tr, tuple):
            loss = loss + out_tr[-1].float().mean()                   # reconstruction branch too
        loss.backward()
        grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
    functional.reset_net(model); model.train_mode = 'testing'; out_te = model(x)
    model.train_mode = 'training'
    tr = [o.detach().cpu() for o in (out_tr if isinstance(out_tr, tuple) else (out_tr,))]
    return tr, out_te.detach().cpu(), grads


def dump(run_dir, gpu, path):
    cfg, model = build(run_dir, gpu)
    x, y = one_batch(cfg)
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tr, te, grads = forwards(model, x)
    cfg2, loaded = build(run_dir, gpu, train=False)
    functional.reset_net(loaded); loaded.eval()
    with torch.no_grad():
        te_loaded = loaded(x).detach().cpu()
    torch.save({'state': sd, 'x': x.cpu(), 'tr': tr, 'te': te, 'grads': grads, 'te_loaded': te_loaded,
                'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'cwd': os.getcwd()}, path)
    print('dumped', path)


def close_dicts(a, b, rtol=1e-5, atol=1e-7):
    if a.keys() != b.keys():
        return False, f'keys differ: {sorted(set(a) ^ set(b))[:5]}'
    worst = max((a[k] - b[k]).abs().max().item() for k in a)
    bad = [k for k in a if not torch.allclose(a[k], b[k], rtol=rtol, atol=atol)]
    return (not bad), f'max |diff| = {worst:.3e}; ' + (f'{len(bad)} tensors outside tolerance, e.g. {bad[:3]}' if bad else 'all within tolerance')


def neo_spike_diff(m_a, m_b, x, train_mode=False):
    """Fraction of neocortex output spikes that differ between two models on one batch (testing mode)."""
    outs = []
    for m in (m_a, m_b):
        box = {}
        h = m.time_block.register_forward_hook(lambda mod, i, o: box.__setitem__('s', o.detach()))
        m.train() if train_mode else m.eval(); functional.reset_net(m); mode = m.train_mode; m.train_mode = 'testing'
        with torch.no_grad():
            m(x)
        m.train_mode = mode; h.remove(); outs.append(box['s'])
    return f"{(outs[0] != outs[1]).float().mean().item():.4f} (rates {outs[0].mean().item():.4f} vs {outs[1].mean().item():.4f})"


def equal_dicts(a, b):
    if a.keys() != b.keys():
        return False, f'keys differ: {sorted(set(a) ^ set(b))[:5]}'
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    return (not bad), (f'{len(bad)} tensors differ, e.g. {bad[:3]}' if bad else 'all equal')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-nd', type=int, default=0)
    ap.add_argument('--base_run', default=BASE_RUN)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'checks'))
    ap.add_argument('--dump-base', default=None, help='internal: write the reference dump and exit')
    a = ap.parse_args()
    if a.dump_base:
        dump(a.base_run, a.nd, a.dump_base); return
    os.makedirs(a.out, exist_ok=True)
    results = []
    def rec(name, ok, msg):
        results.append((name, bool(ok), msg)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")

    # ---- c1: reference from the nsmt_base code (subprocess, its own cwd / modules) ----------------
    ref_path = os.path.join(a.out, 'base_reference.pt')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.nd))
    r = subprocess.run([sys.executable, os.path.abspath(__file__), '-nd', '0', '--base_run', a.base_run, '--dump-base', ref_path],
                       cwd=BASE_DIR, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:]); rec('c1 reference dump', False, 'subprocess failed'); sys.exit(1)
    ref = torch.load(ref_path)
    cfg, model = build(a.base_run, a.nd)
    x, y = one_batch(cfg)
    rec('c1 batch', torch.equal(x.cpu(), ref['x']), 'same first test batch')
    ok, msg = equal_dicts({k: v.detach().cpu() for k, v in model.state_dict().items()}, ref['state'])
    rec('c1 init state dict', ok, msg)
    tr, te, grads = forwards(model, x)
    rec('c1 training outputs', len(tr) == len(ref['tr']) and all(torch.equal(p, q) for p, q in zip(tr, ref['tr'])), f'{len(tr)} tensors')
    rec('c1 testing output', torch.equal(te, ref['te']), 'tensor equal')
    ok, msg = equal_dicts(grads, ref['grads']); rec('c1 gradients', ok, msg)
    cfg_l, loaded = build(a.base_run, a.nd, train=False); loaded.eval(); functional.reset_net(loaded)
    with torch.no_grad():
        te_l = loaded(x).detach().cpu()
    rec('c1 checkpoint outputs', torch.equal(te_l, ref['te_loaded']), 'nsmt_base checkpoint, testing mode')
    n_base = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rec('c1 params', n_base == ref['n_params'], f'{n_base}')

    # ---- c2: lif == flif(alpha = 1) ------------------------------------------------------------------
    for where in ('both', 'mlp'):
        _, m_lif = build(a.base_run, a.nd, neo_neuron='lif', neo_where=where)
        _, m_fl = build(a.base_run, a.nd, neo_neuron='flif', neo_where=where, neo_alpha=1.0)
        ok, msg = equal_dicts({k: v.cpu() for k, v in m_lif.state_dict().items()}, {k: v.cpu() for k, v in m_fl.state_dict().items()})
        rec(f'c2 state dict ({where})', ok, msg)
        for tm in (False, True):                                     # eval (running BN stats) and train (batch BN stats)
            t1, e1, g1 = forwards(m_lif, x, train_mode=tm); t2, e2, g2 = forwards(m_fl, x, train_mode=tm)
            rec(f'c2 outputs ({where}, {"train" if tm else "eval"} BN)', all(torch.equal(p, q) for p, q in zip(t1, t2)) and torch.equal(e1, e2),
                f'training + testing; neocortex spike rate {neo_spike_diff(m_fl, m_fl, x, tm).split("rates ")[1]}')
        # Euler (u + F) and the alpha = 1 predictor (sum of all stored F_j) are the same forward values but
        # sum in a different order in backward -> gradients agree to floating-point precision, not bitwise.
        ok, msg = close_dicts(g1, g2); rec(f'c2 gradients ({where})', ok, msg)
        n_new = [n for n, m in m_fl.named_modules() if isinstance(m, (LIFNeuron, FLIFNeuron))]
        rec(f'c2 placement ({where})', (len(n_new) == (3 if where == 'both' else 2)), f'{n_new}')
        # info (not a pass condition): how different is the spikeDE LIF from the paper's spikingjelly LIF here?
        print('      info (train BN): lif testing output equals default sj_lif output?', torch.equal(e1, forwards(model, x, backward=False, train_mode=True)[1]),
              '| neocortex spikes differing sj_lif vs lif:', neo_spike_diff(model, m_lif, x, True))

    # ---- c3: hippo_only ------------------------------------------------------------------------------
    _, m_h = build(a.base_run, a.nd, hippo_only=True)
    missing = [p for p in NEO_PARTS if not hasattr(m_h, p)]
    n_neo = sum(p.numel() for n, p in model.named_parameters() if n.split('.')[0] in NEO_PARTS and p.requires_grad)
    n_h = sum(p.numel() for p in m_h.parameters() if p.requires_grad)
    rec('c3 modules removed', missing == list(NEO_PARTS), f'missing={missing}')
    rec('c3 params', n_h == n_base - n_neo, f'{n_h} = {n_base} - {n_neo}')
    th, eh, gh = forwards(m_h, x)
    rec('c3 outputs', len(th) == 1 and th[0].ndim == 4 and eh.ndim == 4 and torch.equal(th[0], eh), 'plain 4-D forecast in both modes')

    # ---- c6: --aux_lead shifts the auxiliary pairs by exactly `lead` patches (stage 2 A) -----------
    for nn_ in ('sj_lif', 'flif'):
        _, m0 = build(a.base_run, a.nd, neo_neuron=nn_)
        _, mL = build(a.base_run, a.nd, neo_neuron=nn_, aux_lead=2)
        ok, msg = equal_dicts({k: v.cpu() for k, v in m0.state_dict().items()}, {k: v.cpu() for k, v in mL.state_dict().items()})
        rec(f'c6 same init ({nn_})', ok, msg)
        t0, _, _ = forwards(m0, x, backward=False, train_mode=True)
        tL, _, _ = forwards(mL, x, backward=False, train_mode=True)
        z_ok = torch.equal(t0[0], tL[0]) and torch.equal(t0[2], tL[2])          # forecast and neocortex output unchanged
        org_ok = torch.equal(tL[3], t0[3][..., 2:, :])                           # target of patch n + 2 ...
        rec_ok = torch.equal(tL[4], t0[4][..., :-2, :])                          # ... paired with the output at step n
        rec(f'c6 shifted pairs ({nn_})', z_ok and org_ok and rec_ok,
            f'forecast/neo equal={z_ok}, org[n+2] equal={org_ok}, rec[n] equal={rec_ok}, shape {tuple(tL[4].shape)} vs {tuple(t0[4].shape)}')

    # ---- c7: --neo_grad scales only the task gradient reaching the neocortex (stage 2 B) ------------
    def task_grads(m):
        m.train(); functional.reset_net(m); m.train_mode = 'training'; m.zero_grad(set_to_none=True)
        out = m(x); out[0].float().mean().backward()                            # task branch only (no L_aux)
        return out[0].detach().cpu(), {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}
    for nn_ in ('sj_lif', 'flif'):
        _, ma = build(a.base_run, a.nd, neo_neuron=nn_)
        _, mb = build(a.base_run, a.nd, neo_neuron=nn_, neo_grad=1.0)
        za, ga = task_grads(ma); zb, gb = task_grads(mb)
        neo = [k for k in ga if k.split('.')[0] in ('encoding_neo', 'time_block')]
        hip = [k for k in ga if k not in neo]
        hip_ok = all(torch.equal(ga[k], gb[k]) for k in hip)
        ratio_ok = all(torch.allclose(gb[k], ga[k] / 0.05, rtol=1e-4, atol=1e-9) for k in neo)
        nz = sum(int(ga[k].abs().sum() > 0) for k in neo)
        worst = max(((gb[k] - ga[k] / 0.05).abs().max() / ga[k].abs().max().clamp_min(1e-30)).item() for k in neo)
        rec(f'c7 gradient share ({nn_})', torch.equal(za, zb) and hip_ok and ratio_ok and nz == len(neo),
            f'forecast equal={torch.equal(za, zb)}, hippocampus grads equal={hip_ok} ({len(hip)}), neocortex grads x20 '
            f'({len(neo)} tensors, {nz} nonzero, worst rel {worst:.2e})')

    # ---- c8: serial architecture (stage 3): the neocortex output is the backbone's spike embedding ----
    for nn_ in ('lif', 'flif'):
        for tb in (1, 0):
            _, ms = build(a.base_run, a.nd, arch='serial', neo_neuron=nn_, neo_where='both', backbone_T=tb)
            top = {n.split('.')[0] for n, _ in ms.named_modules() if n}
            ms.eval(); functional.reset_net(ms); ms.train_mode = 'training'
            with torch.no_grad():
                o = ms(x)
            steps = 1 if tb == 1 else ms.T
            rec(f'c8 serial structure ({nn_}, Tb={steps})',
                'encoding' not in top and 'weak_decoder' not in top and {'encoding_neo', 'time_block'} <= top
                and not isinstance(o, tuple) and o.ndim == 4 and o.shape[0] == steps,
                f'no hippocampus embedding / decoder, plain forecast with {tuple(o.shape)}')
    # causality on a TRAINED f-LIF neocortex (an untrained one barely fires, which would make the test vacuous):
    # the neocortex module is the same code in the parallel and the serial model.
    cands = sorted(glob.glob(os.path.join(HERE, 'log', 'flif-v1-s2p10', 'ETTh2', '*',
                                          '*+neo_flif_both+seed42+model+*pred_len+96+*')))
    if not cands:
        rec('c8 aligned embedding is causal', False, 'no trained f-LIF checkpoint found (log/flif-v1-s2p10) -- cannot test')
    else:
        from config import Config
        from model import LOAD_MODEL
        cfg_t = Config(); cfg_t.load_args(cands[0], Namespace(saved_epoch=[1], num_device=a.nd))
        mt = LOAD_MODEL['myModel'](cfg_t, False, ckpt='checkpoint.pth').to(cfg_t.device); mt.eval()
        xt = one_batch(cfg_t)[0]
        box = {}
        hk = mt.time_block.register_forward_hook(lambda mod, i, o: box.__setitem__('neo', o.detach()))
        def neo_out(j=None):
            h = None
            if j is not None:
                h = mt.encoding_neo.register_forward_pre_hook(
                    lambda mod, inp: (torch.cat([inp[0][:j], torch.randn_like(inp[0][j:]) * 3], 0),))
            functional.reset_net(mt)
            with torch.no_grad():
                mt(xt)
            if h is not None:
                h.remove()
            return box['neo'].clone()
        n0 = neo_out(); before, at = [], []
        for j in (5, 12, 20):
            d = (n0 != neo_out(j)).flatten(1).float().mean(1)
            before.append(bool(d[:j].sum() > 0)); at.append(float(d[j]))
        hk.remove()
        rate = n0.mean().item()
        rec('c8 aligned embedding is causal', (not any(before)) and rate >= 0.05 and min(at) > 0,
            f'trained f-LIF neocortex (rate {rate:.3f}): patches >= j never change outputs at tokens < j, and change '
            f'{["%.2f" % v for v in at]} of the spikes at token j (j = 5, 12, 20); fails if the neocortex is (nearly) silent')
    ms.train(); functional.reset_net(ms); ms.train_mode = 'training'; ms.zero_grad(set_to_none=True)
    ms(x).float().mean().backward()
    g = ms.encoding_neo.emb_linear.weight.grad
    rec('c8 task gradient reaches the embedding', g is not None and float(g.abs().sum()) > 0, 'serial: no 5 % cap, no decoder')

    # ---- c9: backbone_T equivalences ------------------------------------------------------------------
    _, h1 = build(a.base_run, a.nd, hippo_only=True, backbone_T=1)
    _, hd = build(a.base_run, a.nd, hippo_only=True)
    _, hN = build(a.base_run, a.nd, hippo_only=True, backbone_T=hd.T)
    ok_init = all(torch.equal(p_, q_) for p_, q_ in zip(h1.state_dict().values(), hd.state_dict().values()))
    _, z1, _ = forwards(h1, x, backward=False); _, zd, _ = forwards(hd, x, backward=False); _, zN, _ = forwards(hN, x, backward=False)
    rec('c9 hippo_only Tb=1 == first step of Tb=N', ok_init and torch.equal(z1[0], zd[0]) and z1.shape[0] == 1,
        'same init; the backbone is causal along its repetition axis')
    rec('c9 explicit Tb=N == default', torch.equal(zN, zd), 'hippo_only')
    _, pN = build(a.base_run, a.nd, backbone_T=model.T)
    tN, eN, _ = forwards(pN, x, backward=False)
    rec('c9 parallel explicit Tb=N == default', all(torch.equal(p_, q_) for p_, q_ in zip(tN, tr)) and torch.equal(eN, te),
        'paper model, training and testing outputs')

    # ---- c10: stage 4, the neocortex path alone (neo_only.NeoForecaster) ---------------------------------
    from model import LOAD_MODEL as _LM
    def neo_build(**over):
        cfg_n = Config(); cfg_n.load_args(a.base_run, Namespace(saved_epoch=[1], num_device=a.nd))
        for k_, v_ in dict(model='neoOnly', **over).items():
            setattr(cfg_n, k_, v_)
        set_random_seed(42)
        return cfg_n, _LM['neoOnly'](cfg_n, True).to(cfg_n.device)
    from config import Config, set_random_seed
    for u in ('step', 'patch'):
        for al, ro in (('on', 'flat'), ('off', 'flat'), ('on', 'last')):
            _, mn = neo_build(neo_unit=u, align=al, neo_readout=ro, neo_neuron='flif', align_Ts=4)
            top = {n_.split('.')[0] for n_, _ in mn.named_modules() if n_}
            mn.eval(); functional.reset_net(mn)
            with torch.no_grad():
                o = mn(x)
            want = {'encoding_neo', 'time_block', 'head'} | ({'pad'} if u == 'patch' else set())   # patch: end padding
            rec(f'c10 shape ({u}, {al}, {ro})', o.shape == (x.shape[0], mn.pred_len, x.shape[2]) and top == want,
                f'output {tuple(o.shape)}, modules {sorted(top)}')
    for u in ('step', 'patch'):
        for ro in ('flat', 'last'):
            _, ml = neo_build(neo_unit=u, align='on', neo_readout=ro, neo_neuron='lif')
            _, mf = neo_build(neo_unit=u, align='on', neo_readout=ro, neo_neuron='flif', neo_alpha=1.0)
            ml.train(); mf.train(); functional.reset_net(ml); functional.reset_net(mf)
            sp = {}
            hl = ml.time_block.register_forward_hook(lambda mod, i, o: sp.__setitem__('l', o.detach()))
            hf = mf.time_block.register_forward_hook(lambda mod, i, o: sp.__setitem__('f', o.detach()))
            ol, of_ = ml(x), mf(x); hl.remove(); hf.remove()
            (ol.mean()).backward(); (of_.mean()).backward()
            grads_ok = all(torch.allclose(p_.grad, q_.grad, rtol=1e-4, atol=1e-6)
                           for p_, q_ in zip(ml.parameters(), mf.parameters()) if p_.grad is not None)
            if ro == 'flat':      # spikes in, spikes out: bitwise
                ok, msg = torch.equal(ol, of_) and grads_ok, 'forecasts bitwise, gradients within 1e-4'
            else:                 # the membrane read by the head differs only by float summation order (Euler vs predictor sum)
                d_ = (ol - of_).abs().max().item()
                ok = torch.equal(sp['l'], sp['f']) and d_ < 1e-5 and grads_ok
                msg = f'spikes bitwise, forecast max |diff| {d_:.1e} (float32 summation order), gradients within 1e-4'
            rec(f'c10 lif == flif(alpha=1) ({u}, {ro})', ok, msg)
    smoke = os.path.join(HERE, 'log', 'smoke-s4-20261018', 'ETTh1')
    def smoke_model(cond):
        d = sorted(glob.glob(os.path.join(smoke, '*', f'*+{cond}+seed42+model+*')))
        if not d:
            return None, None, None
        cfg_s = Config(); cfg_s.load_args(d[0], Namespace(saved_epoch=[1], num_device=a.nd))
        m_ = _LM['neoOnly'](cfg_s, False, ckpt='checkpoint.pth').to(cfg_s.device); m_.eval()
        return cfg_s, m_, one_batch(cfg_s)[0]
    for u in ('step', 'patch'):                                       # causality of the aligned neocortex (trained 1 epoch)
        cfg_s, ms_, xs = smoke_model(f'neo_{u}_on_flat_flif')
        if ms_ is None:
            rec(f'c10 aligned causality ({u})', False, 'smoke checkpoint missing -- run scripts/make_queue_s4.sh smoke first'); continue
        box = {}
        hk = ms_.time_block.register_forward_hook(lambda mod, i, o: box.__setitem__('o', o.detach()))
        def run_(j=None):
            h = None
            if j is not None:
                h = ms_.encoding_neo.register_forward_pre_hook(
                    lambda mod, inp: (torch.cat([inp[0][:j], torch.randn_like(inp[0][j:]) * 3], 0),))
            functional.reset_net(ms_)
            with torch.no_grad():
                ms_(xs)
            if h is not None:
                h.remove()
            return box['o'].clone()
        o0 = run_(); before, at = [], []
        js = (10, 48, 80) if u == 'step' else (5, 12, 20)
        for j in js:
            dd = (o0 != run_(j)).flatten(1).float().mean(1)
            before.append(bool(dd[:j].sum() > 0)); at.append(float(dd[j]))
        hk.remove(); rate = o0.mean().item()
        rec(f'c10 aligned causality ({u})', (not any(before)) and rate >= 0.02 and min(at) > 0,
            f'rate {rate:.3f}; tokens < j unchanged for j = {js}; spikes changed at token j: {["%.2f" % v for v in at]} (fails if silent)')
    for u in ('step', 'patch'):                                       # independence of tokens when not aligned
        cfg_s, ms_, xs = smoke_model(f'neo_{u}_off_flat_flif')
        if ms_ is None:
            rec(f'c10 off independence ({u})', False, 'smoke checkpoint missing'); continue
        box = {}
        hk = ms_.time_block.register_forward_hook(lambda mod, i, o: box.__setitem__('o', o.detach()))
        N_ = ms_.tokens; j = N_ // 2
        def run_off(perturb):
            h = None
            if perturb:
                def pre(mod, inp):
                    z = inp[0].clone(); rows = torch.arange(z.shape[1], device=z.device) % N_ == j
                    z[:, rows] = torch.randn_like(z[:, rows]) * 3
                    return (z,)
                h = ms_.encoding_neo.register_forward_pre_hook(pre)
            functional.reset_net(ms_)
            with torch.no_grad():
                ms_(xs)
            if h is not None:
                h.remove()
            o = box['o']
            return o[:, :, 0, :].mean(0).reshape(-1, N_, o.shape[-1])
        f0, f1 = run_off(False), run_off(True)
        dd = (f0 != f1).flatten(2).any(2).float().mean(0)              # per token: fraction of rows changed
        others = torch.cat([dd[:j], dd[j + 1:]])
        hk.remove()
        rec(f'c10 off independence ({u})', float(others.sum()) == 0 and float(dd[j]) > 0,
            f'perturbing token {j} changes {float(dd[j]):.2f} of its rows and none of the other {N_ - 1} tokens')

    # ---- c4 + c5: test() on the nsmt_base checkpoint -------------------------------------------------
    from test import test
    scratch = os.path.join(a.out, 'c5_test_log'); os.makedirs(scratch, exist_ok=True)
    for f in ('final+result.csv', 'neuron_energy.csv', 'model+info+per+layer.txt'):
        if os.path.exists(os.path.join(scratch, f)):
            os.remove(os.path.join(scratch, f))
    cfg5 = build(a.base_run, a.nd, train=False)[0]
    cfg5.save_log_path = scratch
    test(cfg5)
    base_row = open(os.path.join(a.base_run, 'log', 'final+result.csv')).read().splitlines()
    new_row = open(os.path.join(scratch, 'final+result.csv')).read().splitlines()
    bh, bv = base_row[0].split(','), base_row[1].split(',')
    nh, nv = new_row[0].split(','), new_row[1].split(',')
    rec('c5 first 13 columns', nh[:13] == bh[:13] and nv[:13] == bv[:13], f'{nv[:13]}')
    from energy import COLUMNS
    extra = dict(zip(nh[13:], nv[13:]))
    numeric = all(extra[c] == 'None' or re.fullmatch(r'-?\d+(\.\d+)?', extra[c]) for c in COLUMNS)
    rec('c5 new columns', nh[13:] == list(COLUMNS) and numeric, json.dumps(extra))
    # c4: syops' "Spike Rate" of a Linear is the per-call mean rate of ITS INPUT, i.e. of the spiking neuron that
    # feeds it (syops' own LIF-node rate reads output[0] = the first time step only, so it is not comparable).
    # Pairs: time_block Linear #1 <- encoding_neo.emb_lif, time_block Linear #2 <- time_block.Mem.0.spk_neuron,
    #        mlp Linear #2 <- mlp.spk_linear1.spk_neuron.
    txt = open(os.path.join(scratch, 'model+info+per+layer.txt')).read().splitlines()
    def section_linear_rates(header):
        i0 = next(i for i, l in enumerate(txt) if header in l)
        indent = len(txt[i0]) - len(txt[i0].lstrip())
        out = []
        for l in txt[i0 + 1:]:
            if l.strip() and (len(l) - len(l.lstrip())) <= indent:
                break
            m = re.search(r'\): Linear\(.*?([\d.]+)% Spike Rate', l)
            if m:
                out.append(float(m.group(1)) / 100.)
        return out
    tb, mlp = section_linear_rates('(time_block): TemporalBlock('), section_linear_rates('(mlp): MLP(')
    rows = {r['module']: float(r['rho_syops']) for r in csv.DictReader(open(os.path.join(scratch, 'neuron_energy.csv')))}
    def row(prefix):
        return next(v for k, v in rows.items() if k.startswith(prefix))
    pairs = [('encoding_neo.emb_lif', tb[0]), ('time_block.Mem.0.spk_neuron', tb[1]), ('data_block.0.mlp.spk_linear1.spk_neuron', mlp[1])]
    diffs = {p: abs(row(p) - v) for p, v in pairs}
    rec('c4 firing rates', max(diffs.values()) < 6e-6, f'recorder vs syops consumer-Linear rate: {diffs}')

    with open(os.path.join(a.out, 'check_flif.txt'), 'w') as f:
        for n, ok, msg in results:
            f.write(f"{'PASS' if ok else 'FAIL'} {n}: {msg}\n")
        f.write(f"{sum(ok for _, ok, _ in results)}/{len(results)} passed\n")
    print(f"{sum(ok for _, ok, _ in results)}/{len(results)} passed -> {a.out}/check_flif.txt")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == '__main__':
    main()
