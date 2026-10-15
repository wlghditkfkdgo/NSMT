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
