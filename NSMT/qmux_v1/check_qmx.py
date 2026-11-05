"""Checks of the bidirectional Neocortex for the query form (2026-11-05, exp/f-lif-pop-v3; PROJECT_LOG; neo_mux.py). Fail closed: any failed
check -> exit 1. No training here; q4 (the rerun of source runs, the smoke runs) reads finished runs: --after <repro suite> <smoke suite>.
    q0  copy: every file marked "no" in QUERY_PROVENANCE.md has its source blob; the changed files are only the ones marked "yes".
    q1  default path: this folder's code with --neo_mux none loads the source's query checkpoints and reproduces their logged metrics
        (ETT: 4 neurons on ETTh2 H96 seed 42, population LIF on ETTm2 H96 seed 42; MIT-BIH: population LIF / f-LIF seed 42; 1e-6).
    q2  new model, 4 neurons x {muxlin, or} x {OR, IAND}, on ETT (D 64, N 24) and MIT-BIH (D 384, N 11):
          a  at one seed every parameter outside time_block and the random state after building equal the plain query model's; the direction
             weights are the same across fusions and residuals; the fusion weights are the same across residuals
          b  neurons: the Neocortex (embedding, both directions, s_d, fusion) all hold the chosen class, 5 (or) / 9 (muxlin) neuron layers; the
             hippocampus has no spikeDE neuron; 2 blocks of 1 layer per direction
          c  formula on a real batch: m = where(s_d, L, F ^ R) (muxlin), F v R (or); x = max(x, m) (OR), x (1 - m) (IAND); values 0 / 1;
             the input of the attention's q_linear is exactly that output, repeated over T_h
          d  direction: a change at token t reaches earlier and later tokens in the new Neocortex, only later ones in the plain one
          e  gradients: finite, and non-zero on both direction weights (and the fusion weights for muxlin)
    q3  initial firing, reported per layer on a real batch (training-mode BN). A layer below 0.1 % fails here: a silent gate cannot train.
usage: python check_qmx.py [--only q0,q1,q2,q3] [-nd gpu]
       python check_qmx.py --after <repro suite> <smoke suite>
"""
import argparse
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                          # noqa: E402  (before timm)
from load_model import LOAD_MODEL                                         # noqa: E402
from forecast_wrap import Forecaster                                      # noqa: E402
from summarize_rhythm import parse                                        # noqa: E402
from train_forecast import set_random_seed                                 # noqa: E402

MAIN = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1'
ETT_RES = [os.path.join(MAIN, 'forecasting', 'results', s, 'raw.txt') for s in ('flif-v1-ett-query', 'flif-v1-ett-query-m2')]
MIT_RES = os.path.join(MAIN, 'classification', 'results', 'flif-v1-mitbih-query', 'raw.txt')
MIT_DATA = '/home/yschoi/CLS_spiking_transformer/data'
NEURONS = {'lif': dict(neo_neuron='lif'), 'flif': dict(neo_neuron='flif', neo_alpha=0.5),
           'pop_lif': dict(neo_neuron='pop_lif', pop_soma_init=0.5), 'pop_flif': dict(neo_neuron='pop_flif', neo_alpha=0.5, pop_soma_init=1.0)}
FUSIONS, RESIDUALS = ('muxlin', 'or'), ('OR', 'IAND')
SILENT = 1e-3


def fail(msg):
    print(f'FAIL {msg}', flush=True)
    sys.exit(1)


def row(path_list, **kw):
    for p in path_list:
        for r in parse(p):
            if all(str(r.get(k)) == str(v) or r.get(k) == v for k, v in kw.items()):
                return r
    fail(f'no source row {kw}')


def load_cfg(run):
    return torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')


def ett_loader(args):
    from train_forecast import data_args
    from data_provider.data_factory import data_provider
    return data_provider(data_args(args, args), 'test')[1]


def mit_loader(args):
    from dataloader import create_loader
    return create_loader(train=False, num_classes=args.num_classes, batch_size=args.batch_size, data_root=os.path.join(MIT_DATA, '5class_mitbih_test_3_187.csv'),
                         data_window_size=187, num_workers=2)[0]


def build_ett(cfg, dev=None, **over):
    args = SimpleNamespace(**{**cfg, **over}); args.device = dev or torch.device('cpu')
    return Forecaster(LOAD_MODEL[args.model](args, True), args.embed_dim, args.pred_len, bias=args.bias), args


def build_mit(cfg, dev=None, **over):
    args = SimpleNamespace(**{**cfg, **over}); args.device = dev or torch.device('cpu')
    return LOAD_MODEL[args.model](args, True), args


def q0():
    root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=HERE, text=True).strip()
    bad, n = [], 0
    for line in open(os.path.join(HERE, 'QUERY_PROVENANCE.md')):
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) == 4 and re.fullmatch(r'[0-9a-f]{40}', c[2]):
            n += 1
            have = subprocess.check_output(['git', 'hash-object', os.path.join(HERE, c[0])], text=True).strip()
            if c[3] == 'no' and have != c[2]:
                bad.append(c[0])
            if c[3] != 'no' and have == c[2]:
                bad.append(f'{c[0]} marked changed but equals the source')
    if bad or n < 20:
        fail(f'q0: {n} rows, {bad}')
    print(f'q0 ok: {n} provenance rows match (root {os.path.basename(root)})', flush=True)


@torch.no_grad()
def q1(dev):
    for cond, ds in (('nsmt4q_lif', 'ETTh2'), ('nsmt4q_flif', 'ETTh2'), ('nsmt4q_pop_lif', 'ETTh2'), ('nsmt4q_pop_flif', 'ETTh2'), ('nsmt4q_pop_lif', 'ETTm2')):
        r = row(ETT_RES, cond=cond, ds=ds, pl=96.0, seed=42.0)
        cfg = load_cfg(r['dir'])
        m, args = build_ett(cfg, dev)
        m.load_state_dict(torch.load(os.path.join(r['dir'], 'model_state', 'checkpoint.pth'), map_location='cpu'))
        m = m.to(dev).eval(); m.train_mode = 'testing'
        se, cnt = 0., 0
        functional.reset_net(m)
        for x, y, _, _ in ett_loader(args):
            x = x.float().to(dev); y = y.float().to(dev)[:, -args.pred_len:, :]
            d = (m(x) - y).double(); functional.reset_net(m)
            se += float((d ** 2).sum()); cnt += d.numel()
        if abs(se / cnt - float(r['mse'])) > 1e-6:
            fail(f"q1 {cond} {ds}: mse {se / cnt:.6f} != {float(r['mse']):.6f}")
        print(f"q1 ok: {cond} {ds} H96 seed 42 mse {se / cnt:.6f}", flush=True)
    from torcheval.metrics import MulticlassF1Score
    for cond in ('nsmt4q_pop_lif', 'nsmt4q_pop_flif'):
        r = row([MIT_RES], cond=cond, seed=42.0)
        m, args = build_mit(load_cfg(r['dir']), dev)
        m.load_state_dict(torch.load(os.path.join(r['dir'], 'model_state', 'best+model.pt'), map_location='cpu'))
        m = m.to(dev).eval(); m.train_mode = 'testing'
        f1 = MulticlassF1Score(num_classes=args.num_classes, average='macro').to(dev)
        functional.reset_net(m)
        for b in mit_loader(args):
            out = m(b[0].float().to(dev)); out = out[0] if isinstance(out, tuple) else out
            functional.reset_net(m); f1.update(out.argmax(1), b[1].long().squeeze(-1).to(dev))
        if abs(float(f1.compute()) - float(r['f1'])) > 1e-6:
            fail(f"q1 {cond} MIT-BIH: f1 {float(f1.compute()):.6f} != {float(r['f1']):.6f}")
        print(f'q1 ok: {cond} MIT-BIH seed 42 f1 {float(f1.compute()):.6f}', flush=True)


def tasks():
    e = row(ETT_RES, cond='nsmt4q_pop_lif', ds='ETTh2', pl=96.0, seed=42.0)
    t = row([MIT_RES], cond='nsmt4q_pop_lif', seed=42.0)
    return (('ett', load_cfg(e['dir']), build_ett), ('mitbih', load_cfg(t['dir']), build_mit))


def batch(task, args, dev):
    if task == 'ett':
        x = next(iter(ett_loader(args)))[0].float().to(dev)
        B, L, C = x.shape
        m_ = x.mean(1, keepdim=True); xc = x - m_
        s = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return (xc / s).permute(0, 2, 1).reshape(B * C, 1, L)
    x = next(iter(mit_loader(args)))[0].float().to(dev)
    return x.unsqueeze(1) if x.dim() == 2 else x


def inner(model):
    return model.inner if hasattr(model, 'inner') else model


def set_momentum_one(model):
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.momentum = 1.0


def q2(dev):
    from neo_mux import NeoMuxStack
    from layers import NeoNeuron
    from neurons import LIFNeuron, FLIFNeuron
    for task, cfg, build in tasks():
        for nk, over in NEURONS.items():
            over = dict(over)
            set_random_seed(42)
            plain, args = build(cfg, **over, neo_mux='none')
            rng0 = torch.get_rng_state()
            sd0 = plain.state_dict()
            weights = {}
            for fu in FUSIONS:
                for cf in RESIDUALS:
                    tag = f'{task} {nk} {fu} {cf}'
                    set_random_seed(42)
                    model, _ = build(cfg, **over, neo_mux=fu, neo_mux_cf=cf, neo_mux_blocks=2)
                    # a: pairing
                    if not torch.equal(rng0, torch.get_rng_state()):
                        fail(f'q2a {tag}: random state after building differs from the plain query model')
                    sd = model.state_dict()
                    for k, v in sd.items():
                        if 'time_block' not in k and not torch.equal(v, sd0[k]):
                            fail(f'q2a {tag}: {k} differs from the plain query model')
                    dirw = {k: v for k, v in sd.items() if 'time_block' in k and ('.mixer.' in k or '.mixer_rev.' in k)}
                    fusw = {k: v for k, v in sd.items() if 'time_block' in k and 'fuse.linear' in k}
                    for name, w in (('dir', dirw), (f'fus_{fu}', fusw)):
                        if name in weights:
                            if set(weights[name]) != set(w) or any(not torch.equal(weights[name][k], w[k]) for k in w):
                                fail(f'q2a {tag}: {name} weights differ across conditions')
                        else:
                            weights[name] = w
                    # b: structure
                    m = inner(model)
                    tb = m.time_block
                    if not isinstance(tb, NeoMuxStack) or len(tb.blocks) != 2:
                        fail(f'q2b {tag}: time_block {type(tb).__name__}')
                    neos = [mod for name, mod in m.named_modules() if isinstance(mod, NeoNeuron)]
                    want = 9 if fu == 'muxlin' else 5
                    if len(neos) != want:
                        fail(f'q2b {tag}: {len(neos)} spikeDE neuron layers, expected {want}')
                    outside = [name for name, mod in m.named_modules() if isinstance(mod, NeoNeuron) and not name.startswith(('encoding_neo', 'time_block'))]
                    if outside:
                        fail(f'q2b {tag}: spikeDE neurons outside the Neocortex: {outside[:3]}')
                    cls = {type(n.neuron).__name__ for n in neos}
                    exp = {'lif': {'LIFNeuron'}, 'flif': {'FLIFNeuron'}, 'pop_lif': {'PopulationFLIF'}, 'pop_flif': {'PopulationFLIF'}}[nk]
                    if cls != exp:
                        fail(f'q2b {tag}: neuron classes {cls}, expected {exp}')
                    # c: formula on a real batch (eval mode, BN statistics set from this batch)
                    model = model.to(dev); m = inner(model); tb = m.time_block
                    args.device = dev
                    z = batch(task, args, dev)
                    set_momentum_one(model)
                    model.train(); m.train_mode = 'testing'
                    with torch.no_grad():
                        hip, neo = m._tokens_mc(z)
                        e = m.encoding_neo(neo)
                        x0 = e
                        for b in tb.blocks:
                            x0 = b(x0)
                        m.data_block[0](m.encoding(hip), x0, self_kv=False)      # the hippocampus BN statistics too
                        functional.reset_net(m)
                    model.eval()
                    with torch.no_grad():
                        hip, neo = m._tokens_mc(z)
                        e = m.encoding_neo(neo)
                        x = e
                        for b in tb.blocks:
                            f, hf = b.mixer.forward_charge(x)
                            r, hr = b.mixer_rev.forward_charge(x.flip(0))
                            r, hr = r.flip(0), hr.flip(0)
                            if fu == 'muxlin':
                                d = (hf - hr).abs(); N, B, K, D = d.shape
                                sd_ = b.sd_neuron(b.sd_bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous())
                                L = b.fuse(torch.cat([torch.maximum(f, r), sd_], dim=-1))
                                want_m = torch.where(sd_ > 0, L, torch.minimum(f, r))
                            else:
                                want_m = torch.maximum(f, r)
                            mm = b.mix(x)
                            if not torch.equal(mm, want_m) or not ((mm == 0) | (mm == 1)).all():
                                fail(f'q2c {tag}: fusion output differs from the formula or is not 0 / 1')
                            nx = torch.maximum(x, mm) if cf == 'OR' else x * (1. - mm)
                            if not torch.equal(b(x), nx):
                                fail(f'q2c {tag}: residual differs from the formula')
                            x = nx
                        o = m.neo_forward(neo, None)[1]
                        if not torch.equal(o, x) or o.shape != e.shape:
                            fail(f'q2c {tag}: neo_forward output differs from the stack')
                        # the query really is this output: the input of the attention's q_linear is the Neocortex output
                        seen = []
                        hd = m.data_block[0].ssa.q_linear.register_forward_pre_hook(lambda mod, inp: seen.append(inp[0].detach().clone()))
                        h = m.encoding(hip)
                        m.data_block[0](h, o, self_kv=False)
                        hd.remove(); functional.reset_net(m)
                        if not seen or not torch.equal(seen[0], o.transpose(0, 2).expand(h.shape[0], -1, -1, -1).contiguous().flatten(0, 1)):
                            fail(f'q2c {tag}: the query input is not the Neocortex output')
                        # d: direction
                        t = neo.shape[0] // 2
                        neo2 = neo.clone(); neo2[t] = neo2[t] + 3.
                        o2 = m.neo_forward(neo2, None)[1]
                        before, after = (o2[:t] != o[:t]).any().item(), (o2[t + 1:] != o[t + 1:]).any().item()
                        if not (before and after):
                            fail(f'q2d {tag}: a change at token {t} reaches earlier {before}, later {after} (both expected)')
                    # e: gradients
                    model.train(); m.train_mode = 'training'
                    model.zero_grad()
                    hip, neo = m._tokens_mc(z)
                    o = m.neo_forward(neo, None)[1]
                    xh = m.encoding(hip)
                    for blk in m.data_block:
                        xh = blk(xh, o, self_kv=False)
                    xh.mean().backward()
                    functional.reset_net(m)
                    for name, p in tb.named_parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            fail(f'q2e {tag}: non-finite gradient at {name}')
                    for i, b in enumerate(tb.blocks):
                        need = [b.mixer.w_in.weight, b.mixer_rev.w_in.weight] + ([b.fuse.linear.weight] if fu == 'muxlin' else [])
                        if any(p.grad is None or float(p.grad.abs().sum()) == 0. for p in need):
                            fail(f'q2e {tag}: zero gradient on a block {i} weight')
                    print(f'q2 ok: {tag}', flush=True)
                    del model; torch.cuda.empty_cache()
            # d (plain query): causal
            plain = plain.to(dev); m = inner(plain); args.device = dev
            z = batch(task, args, dev)
            set_momentum_one(plain)
            plain.train(); m.train_mode = 'testing'
            with torch.no_grad():
                hip, neo = m._tokens_mc(z)
                m.neo_forward(neo, None)
                plain.eval()
                o = m.neo_forward(neo, None)[1]
                t = neo.shape[0] // 2
                neo2 = neo.clone(); neo2[t] = neo2[t] + 3.
                o2 = m.neo_forward(neo2, None)[1]
            if (o2[:t] != o[:t]).any() or not (o2[t:] != o[t:]).any():
                fail(f'q2d {task} {nk} plain: expected only later tokens to change')
            del plain; torch.cuda.empty_cache()


@torch.no_grad()
def q3(dev):
    L = ['| task | neuron | fusion | residual | e % | block 1: F / R / s_d / L / m / x out % | block 2: F / R / s_d / L / m / x out % | q firing % |', '|---|---|---|---|---:|---|---|---:|']
    silent = []
    for task, cfg, build in tasks():
        for nk, over in NEURONS.items():
            for fu in FUSIONS:
                for cf in RESIDUALS:
                    set_random_seed(42)
                    model, args = build(cfg, **over, neo_mux=fu, neo_mux_cf=cf, neo_mux_blocks=2)
                    model = model.to(dev); m = inner(model); tb = m.time_block; args.device = dev
                    z = batch(task, args, dev)
                    model.train(); m.train_mode = 'testing'
                    hip, neo = m._tokens_mc(z)
                    e = m.encoding_neo(neo); x = e
                    cells = []
                    for bi, b in enumerate(tb.blocks):
                        f, hf = b.mixer.forward_charge(x)
                        r, hr = b.mixer_rev.forward_charge(x.flip(0)); r, hr = r.flip(0), hr.flip(0)
                        vals = {'F': f, 'R': r}
                        if fu == 'muxlin':
                            d = (hf - hr).abs(); N, B, K, D = d.shape
                            sd_ = b.sd_neuron(b.sd_bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous())
                            L_ = b.fuse(torch.cat([torch.maximum(f, r), sd_], dim=-1))
                            mm = sd_ * L_ + (1. - sd_) * torch.minimum(f, r)
                            vals.update(sd=sd_, L=L_)
                        else:
                            mm = torch.maximum(f, r)
                        x = b.connect(x, mm)
                        vals.update(m=mm, x=x)
                        for k, v in vals.items():
                            if k in ('F', 'R', 'sd', 'L') and float(v.mean()) < SILENT:
                                silent.append(f'{task} {nk} {fu} {cf} block {bi + 1} {k} {100 * float(v.mean()):.3f} %')
                        cells.append(' / '.join(f'{100 * float(vals[k].mean()):.2f}' if k in vals else '—' for k in ('F', 'R', 'sd', 'L', 'm', 'x')))
                    rates = []
                    h = m.encoding(hip)
                    def hook(mod, inp, out):
                        rates.append(float(out.mean()))
                    hd = m.data_block[0].ssa.q_lif.register_forward_hook(hook) if hasattr(m.data_block[0], 'ssa') and hasattr(m.data_block[0].ssa, 'q_lif') else None
                    m.data_block[0](h, x, self_kv=False)
                    if hd is not None:
                        hd.remove()
                    functional.reset_net(m)
                    L.append(f"| {task} | {nk} | {fu} | {cf} | {100 * float(e.mean()):.2f} | {cells[0]} | {cells[1]} | {100 * rates[0] if rates else float('nan'):.2f} |")
                    del model; torch.cuda.empty_cache()
    os.makedirs(os.path.join(HERE, 'results', 'checks'), exist_ok=True)
    open(os.path.join(HERE, 'results', 'checks', 'qmx_initial_firing.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)
    if silent:
        fail('q3 silent layers (< 0.1 %): ' + '; '.join(silent))
    print('q3 ok: no silent layer', flush=True)


def after(repro_suite, smoke_suite):
    """q4: the rerun of source runs equals their RESULT lines in every field but time, sha, dir, exp, and the smoke has every run with the expected neuron layers."""
    skip = ('time_s', 'sha', 'dir', 'exp', 'tag', 'val')
    p = os.path.join(HERE, 'results', repro_suite, 'raw.txt')
    rr = parse(p)
    if len(rr) != 2 or any(line.startswith('SKIP') for line in open(p)):
        fail(f'q4: {len(rr)} repro rows in {p}')
    for r in rr:
        src = row(ETT_RES, cond=r['cond'], ds=r['ds'], pl=r['pl'], seed=r['seed']) if 'ds' in r else row([MIT_RES], cond=r['cond'], seed=r['seed'])
        diff = {k: (r.get(k), src.get(k)) for k in set(r) | set(src) if k not in skip and r.get(k) != src.get(k)}
        if diff:
            fail(f"q4 {r['cond']}: differs from the source run: {sorted(diff.items())[:8]}")
        print(f"q4 ok: {r['cond']} {r.get('ds', 'MIT-BIH')} seed {int(r['seed'])} equals the source run in {len(set(r) - set(skip))} fields ({int(r['time_s'])} s)", flush=True)
    p = os.path.join(HERE, 'results', smoke_suite, 'raw.txt')
    sm = parse(p)
    if len(sm) != 36 or any(line.startswith('SKIP') for line in open(p)):
        fail(f'q4: {len(sm)} smoke rows in {p}')
    for r in sm:
        want = 8 + (9 if '_muxlin_' in r['cond'] else 5)                 # 8 hippocampus neuron layers + the Neocortex (embedding + 2 blocks)
        if 'n_spiking_layers' in r and int(r['n_spiking_layers']) != want:
            fail(f"q4 {r['cond']}: {r['n_spiking_layers']} neuron layers in the energy record, expected {want}")
    print(f'q4 ok: smoke {len(sm)} runs, no SKIP, neuron layers as expected', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='q0,q1,q2,q3'); ap.add_argument('-nd', type=int, default=0)
    ap.add_argument('--after', nargs=2, metavar=('REPRO_SUITE', 'SMOKE_SUITE'), default=None)
    a = ap.parse_args()
    if a.after:
        after(*a.after)
        print('all requested checks passed', flush=True)
        return
    dev = torch.device(f'cuda:{a.nd}')
    for q in a.only.split(','):
        {'q0': q0, 'q1': lambda: q1(dev), 'q2': lambda: q2(dev), 'q3': lambda: q3(dev)}[q]()
    print('all requested checks passed', flush=True)


if __name__ == '__main__':
    main()
