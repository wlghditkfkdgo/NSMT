#!/usr/bin/env python3
"""Gates of the prereg 2U model: the population axis kept, Linear + BN + spiking neuron blocks, one output
Linear. Reads no split beyond validation.

    python check_2u.py [--smoke <2U smoke suite> --calibration <its calibration suite>] [-nd <gpu>]

  U1  structure: Linear(8,32) -> BN -> population f-LIF (members fire, K=4) -> Linear(128,32, no bias) -> BN ->
      SpikingJelly LIF -> flatten [B*C, T*32] -> Linear(1344, 96); every Linear but the output one is followed
      by a BN and a spiking neuron; spike shapes [T, B*C, 128] and [T, B*C, 32]
  U2  the f-LIF input BN is per time step: in training each step is standardised by its own batch statistics
      (== a manual per-step BN, rel 1e-5) and differs from BN over the time axis folded in; the running
      statistics equal T sequential updates
  U3  gamma starts at input_scale (the D11 value) and set_scale moves it; at init the current is input_scale x
      the per-step standardised Linear output (the calibration's operating point)
  U4  the hidden block: BN over time x batch (== manual, rel 1e-5); SpikingJelly MultiStepLIFNode tau 2,
      detach_reset, sigmoid surrogate, v_th 1, v_reset 0; state reset per forward (repeat == first); cupy == torch
  U5  gradients reach every parameter the forecast uses, finite and nonzero; three AdamW steps stay finite and
      move BN gammas and the hidden block
  U6  evaluation does not depend on the batch: a window's output alone == inside a batch (eval, rel 1e-5)
  U7  refused: BN or a hidden block with the legacy neuron, a hidden block with the flatten head or an analog readout
  U8  the 2T path is unchanged: two 2T checkpoints reproduce their best validation MSE (1e-6) with this code
  U9  (--smoke) the 2U smoke runs: found by the 2U names, manifest fails only for the 1-epoch training,
      validation reproduces best_val_loss (1e-6), the rebuilt pearson masks equal the model's
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerun_common as rc                                       # noqa: E402
from config import parse_defaults, check_neuron_args, set_random_seed   # noqa: E402
from model import load_mymodel                                  # noqa: E402
from data_provider.data_factory import data_provider            # noqa: E402

RESULTS = []


def gate(name, ok, detail):
    RESULTS.append(bool(ok))
    print(f"[2u] {'PASS' if ok else 'FAIL'} | {name:<3} {detail}")


def rel(a, b):
    return float((a - b).abs().max() / max(1., float(b.abs().max())))


def cli(device='cpu', **kw):
    ns = parse_defaults()
    for key, value in {**rc.NEURON_FIXED, **rc.ETT_CONDITIONS_2U['pearson_R'], 'input_scale': 8.}.items():
        setattr(ns, key, value)
    for key, value in kw.items():
        setattr(ns, key, value)
    ns.task, ns.cpu = 'ett', device == 'cpu'
    ns.num_patches = ns.seq_len // ns.patch_size
    ns.device = torch.device(device)
    check_neuron_args(ns)
    return ns


def batch(seed=3, B=4, C=3, device='cpu'):
    g = torch.Generator().manual_seed(seed)
    x = torch.cumsum(torch.randn(B, 336, C, generator=g), dim=1)
    return x.to(device), (x[:, -1:, :] + torch.randn(B, 96, C, generator=g)).to(device)


def structure():
    ok, detail = {}, {}
    for cond in ('q1_R', 'pearson_R'):
        torch.manual_seed(7)
        m = load_mymodel(cli(**rc.ETT_CONDITIONS_2U[cond]))
        x, _ = batch()
        m.train()
        out, aux = m(x, mode=m.embedding.neuron.selective and 'hard' or 'full', return_aux=True)
        emb = m.embedding
        chain = [type(emb.emb_linear).__name__, type(emb.bn).__name__, type(emb.neuron).__name__,
                 type(m.hidden_linear).__name__, type(m.hidden_bn).__name__, type(m.hidden_neuron).__name__,
                 type(m.head_compress).__name__, type(m.head).__name__]
        want = ['Linear', 'BatchNorm1d', 'PopulationSelectiveFLIF' if cond.startswith('pearson') else 'PopulationFLIF',
                'Linear', 'BatchNorm1d', 'MultiStepLIFNode', 'Identity', 'Linear']
        shapes = (tuple(aux['spikes'].shape), tuple(aux['hidden_spikes'].shape), tuple(out.shape))
        ok[cond] = (chain == want and emb.neuron.soma is None and m.hidden_linear.bias is None
                    and (emb.emb_linear.in_features, emb.emb_linear.out_features) == (8, 32)
                    and (m.hidden_linear.in_features, m.hidden_linear.out_features) == (128, 32)
                    and (m.head.in_features, m.head.out_features) == (1344, 96)
                    and shapes == ((42, 12, 128), (42, 12, 32), (4, 96, 3))
                    and all(c.threshold == 1. for c in emb.neuron.cells))
        detail[cond] = f"{' -> '.join(chain)}; spikes {shapes[0]} -> {shapes[1]} -> out {shapes[2]}; " \
                       f"{sum(p.numel() for p in m.parameters())} parameters"
    gate('U1', all(ok.values()), f'{detail}')


@torch.no_grad()
def per_step_bn():
    torch.manual_seed(7)
    m = load_mymodel(cli())
    emb = m.embedding
    emb.train()
    x, _ = batch()
    from ours import window_norm
    import layers
    patch = layers.to_patches(window_norm(x)[0], 8)                 # [T, B*C, 8]
    z = emb.emb_linear(patch)
    mean0, var0 = emb.bn.running_mean.clone(), emb.bn.running_var.clone()
    got = emb.current(patch)
    manual = torch.stack([(z[t] - z[t].mean(0)) / (z[t].var(0, unbiased=False) + emb.bn.eps).sqrt() for t in range(z.shape[0])])
    manual = manual * emb.bn.weight + emb.bn.bias
    flat = (z - z.mean((0, 1))) / (z.var((0, 1), unbiased=False) + emb.bn.eps).sqrt() * emb.bn.weight + emb.bn.bias
    rm, rv = mean0, var0
    for t in range(z.shape[0]):                                     # T sequential running-stat updates
        n = z.shape[1]
        rm = (1 - emb.bn.momentum) * rm + emb.bn.momentum * z[t].mean(0)
        rv = (1 - emb.bn.momentum) * rv + emb.bn.momentum * z[t].var(0, unbiased=True) * 1.
    stats = rel(emb.bn.running_mean, rm) < 1e-5 and rel(emb.bn.running_var, rv) < 1e-5
    gate('U2', rel(got, manual) < 1e-5 and rel(got, flat) > 1e-3 and stats,
         f'per-step BN == manual rel {rel(got, manual):.1e}; differs from time-folded BN rel {rel(got, flat):.2f}; '
         f'running statistics == {z.shape[0]} sequential updates {stats}')
    return m


@torch.no_grad()
def gamma_start():
    m = load_mymodel(cli(input_scale=6.))
    emb = m.embedding
    start = bool((emb.bn.weight == 6.).all())
    emb.set_scale(10.)
    moved = bool((emb.bn.weight == 10.).all()) and emb.input_scale == 10.
    emb.set_scale(6.)
    emb.train()
    x, _ = batch()
    import layers
    from ours import window_norm
    patch = layers.to_patches(window_norm(x)[0], 8)
    z = emb.emb_linear(patch)
    standard = torch.stack([(z[t] - z[t].mean(0)) / (z[t].var(0, unbiased=False) + emb.bn.eps).sqrt() for t in range(z.shape[0])])
    err = rel(emb.current(patch), 6. * standard)
    gate('U3', start and moved and err < 1e-5,
         f'gamma starts at input_scale {start}; set_scale moves it {moved}; current == 6 x per-step standardised rel {err:.1e}')


def hidden_block(device):
    from spikingjelly.clock_driven import surrogate
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    torch.manual_seed(7)
    m = load_mymodel(cli())
    m.train()
    n = m.hidden_neuron
    kind = isinstance(n, MultiStepLIFNode) and n.tau == 2. and n.detach_reset and isinstance(n.surrogate_function, surrogate.Sigmoid) \
        and n.v_threshold == 1. and n.v_reset == 0.
    x, _ = batch()
    with torch.no_grad():
        spikes = m.embedding(__import__('layers').to_patches(__import__('ours').window_norm(x)[0], 8), 'hard')
        h = m.hidden_linear(spikes)
        T, R, _ = h.shape
        got = m.hidden_bn(h.reshape(T * R, -1)).reshape(T, R, -1)
        manual = (h - h.mean((0, 1))) / (h.var((0, 1), unbiased=False) + m.hidden_bn.eps).sqrt() * m.hidden_bn.weight + m.hidden_bn.bias
    m.eval()
    with torch.no_grad():
        a, b = m(x, mode='hard'), m(x, mode='hard')                 # the state is reset per forward
        small = m(x[:2], mode='hard')
    same_cpu_gpu = None
    if device.type == 'cuda':
        mg = load_mymodel(cli(device=str(device)))
        mg.load_state_dict(m.state_dict())
        mg.eval()
        with torch.no_grad():
            hg = torch.randn(42, 6, 32, device=device) * 2
            mg.hidden_neuron.reset(); mg.hidden_neuron.backend = 'cupy'
            s_cupy = mg.hidden_neuron(hg.contiguous())
            mg.hidden_neuron.reset(); mg.hidden_neuron.backend = 'torch'
            s_torch = mg.hidden_neuron(hg.contiguous())
            mg.hidden_neuron.reset()
        same_cpu_gpu = torch.equal(s_cupy, s_torch)
    gate('U4', kind and rel(got, manual) < 1e-5 and torch.equal(a, b) and small.shape[0] == 2 and same_cpu_gpu is not False,
         f'SpikingJelly MultiStepLIFNode tau 2 detach_reset sigmoid {kind}; BN over time x batch == manual rel '
         f'{rel(got, manual):.1e}; repeat forward identical {torch.equal(a, b)}; another batch size runs; '
         f'cupy == torch {same_cpu_gpu}')


def gradients():
    torch.manual_seed(7)
    ns = cli()
    m = load_mymodel(ns)
    m.train()
    x, y = batch()
    loss = torch.nn.functional.mse_loss(m(x, mode=ns.mode), y)
    loss.backward()
    used = {k: p for k, p in m.named_parameters() if not k.startswith('recall_head')}
    grads = {k: p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0 for k, p in used.items()}
    before = {k: p.detach().clone() for k, p in m.named_parameters()}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=.01)
    finite = True
    for _ in range(3):
        loss = torch.nn.functional.mse_loss(m(x, mode=ns.mode), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        finite = finite and bool(torch.isfinite(loss))
    moved = [k for k, p in m.named_parameters() if not torch.equal(p, before[k])]
    need = ['embedding.bn.weight', 'hidden_linear.weight', 'hidden_bn.weight', 'head.weight', 'embedding.emb_linear.weight']
    gate('U5', all(grads.values()) and finite and all(k in moved for k in need),
         f'nonzero finite gradients {sum(grads.values())}/{len(grads)} ({[k for k, v in grads.items() if not v]}); '
         f'3 AdamW steps finite {finite}; moved {len(moved)}/{len(before)}')


@torch.no_grad()
def batch_independence():
    torch.manual_seed(7)
    ns = cli()
    m = load_mymodel(ns)
    m.train()
    x, _ = batch(B=6)
    for _ in range(3):                                               # give the running statistics some values
        m(x, mode=ns.mode)
    m.eval()
    together = m(x, mode=ns.mode)
    alone = torch.cat([m(x[i:i + 1], mode=ns.mode) for i in range(x.shape[0])])
    gate('U6', rel(alone, together) < 1e-5, f'eval output alone vs in a batch rel {rel(alone, together):.1e}')


def refusals():
    import layers
    cases = {'bn + legacy': dict(neuron='legacy', soma=None, mode='full', input_norm='bn', hidden_dim=0),
             'hidden + legacy': dict(neuron='legacy', soma=None, mode='full', input_norm='frozen', hidden_dim=32),
             'hidden + flatten': dict(head_mode='flatten'),
             'hidden + analog': dict(readout='analog', soma='lif')}
    out = {}
    for name, kw in cases.items():
        try:
            cli(**kw)
            out[name] = False
        except ValueError:
            out[name] = True
    try:
        layers.Embedding(8, 32, 6., input_norm='bn')
        out['Embedding bn + legacy'] = False
    except ValueError:
        out['Embedding bn + legacy'] = True
    gate('U7', all(out.values()), f'refused {out}')


def two_t(device):
    worst, n = 0., 0
    for cond in ('q1_R', 'pearson_R'):
        status, run, result = rc.find_cell('etthard-spikede-20261021', 'ett', 'ETTh1', rc.ETT_CONDITIONS[cond], 7)
        model, args = rc.load_model(run, device)
        set_random_seed(args.seed)
        _, loader = data_provider(args, 'val')
        r = rc.one_pass(model, loader, args, ties=False)
        worst = max(worst, abs(r['mse'] - json.loads(Path(result).read_text())['train']['best_val_loss']))
        n += 1
    gate('U8', worst <= rc.REPRO_TOL, f'{n} 2T checkpoints (ETTh1 seed 7) reproduce their best validation MSE: worst |diff| {worst:.1e}')


def smoke(suite, calibration, device):
    cal = rc.calibration_manifest(calibration)
    detail, ok = {}, True
    for cond in ('q1_R', 'pearson_R'):
        spec = rc.ETT_CONDITIONS_2U[cond]
        status, run, result = rc.find_cell(suite, 'ett', 'ETTh1', spec, 7)
        model, args = rc.load_model(run, device)
        errs = rc.check_manifest(args, result, 'ett', spec, 7, cal[rc.calibration_label('ett', cond, 'ETTh1')],
                                 data='ETTh1', cond=cond)
        short = 'epoch: 1 != 50' in errs and all(e == 'epoch: 1 != 50' or 'without matching evidence' in e for e in errs)
        set_random_seed(args.seed)
        _, loader = data_provider(args, 'val')
        r = rc.one_pass(model, loader, args)
        best = json.loads(Path(result).read_text())['train']['best_val_loss']
        good = status == 'done' and short and abs(r['mse'] - best) <= rc.REPRO_TOL and r['safe'] \
            and (r['mask_mismatch'] == 0 if cond.startswith('pearson') else r['mask_mismatch'] is None)
        ok = ok and good
        detail[cond] = (f"val |diff| {abs(r['mse'] - best):.1e}, member rate {r['firing_rate']:.3f}, hidden rate "
                        f"{r['hidden_firing_rate']:.3f}, mismatch {r['mask_mismatch']}, manifest only short {short}")
    gate('U9', ok, f'2U smoke runs: {detail}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', default=None)
    ap.add_argument('--calibration', default=None)
    ap.add_argument('-nd', '--num_device', type=int, default=0)
    a = ap.parse_args()
    device = torch.device(f'cuda:{a.num_device}') if torch.cuda.is_available() else torch.device('cpu')
    structure()
    per_step_bn()
    gamma_start()
    hidden_block(device)
    gradients()
    batch_independence()
    refusals()
    if device.type == 'cuda':
        two_t(device)
    else:
        print('[2u] U8    not run (no GPU: the 2T runs were trained on CUDA)')
    if a.smoke:
        smoke(a.smoke, a.calibration, device)
    else:
        print('[2u] U9    not run (no --smoke suite)')
    print(f'[2u] {sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed')
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    raise SystemExit(main())
