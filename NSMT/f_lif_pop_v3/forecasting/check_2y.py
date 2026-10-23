#!/usr/bin/env python3
"""Gates of prereg 2Y: the output Linear reading only the last step, a shorter window, and the neuron without
state across patches, all on the 2W pick. Reads no split beyond validation.

    python check_2y.py [--smoke <2Y smoke suite> --calibration L96=<suite>,nomem=<suite>] [-nd <gpu>]

  Y1  the defaults are unchanged: the 2W pick's name and calibration tag are what they were, a model built with
      head_window 0 / neuron_memory full equals one built without the options (same initial values, same output),
      and (GPU) a 2W checkpoint reproduces its best validation MSE (1e-6) with this code
  Y2  neuron_memory none: a step's spikes depend on its own patch only (changing patch j changes step j and no
      other step, bit for bit), member k fires exactly when I_t / tau_k >= theta (one step from v = 0), and no
      gradient reaches an earlier step's current; with memory, changing patch j does change later steps and the
      gradient does reach the past
  Y3  the control removes only the state: same initial parameters and the same per-step BN current as the model
      with memory at the same seed
  Y4  head_window 1: the output Linear has 128 inputs and reads the last step only -- without memory, changing any
      earlier patch leaves the forecast bit-identical and changing the last patch does not; with memory an earlier
      patch reaches the forecast through the state alone
  Y5  seq_len 96: T = 12, the models build with the right head width, and the validation targets are the same
      rows as with seq_len 336 (same number of windows, identical y)
  Y6  refused: head_window with the flatten head, head_window > T, neuron_memory none with a selective class or the
      legacy neuron
  Y7  names: the 6 cells have 6 distinct run names, the window length and readout are in the name, the 3
      calibration families have 3 distinct tags, and a calibration of another family is never picked
  Y8  (--smoke) the smoke runs: found by their 2Y names, validation reproduces best_val_loss (1e-6)
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerun_common as rc                                       # noqa: E402
from config import parse_defaults, check_neuron_args, set_random_seed, calibration_tag, num_patches_of   # noqa: E402
from model import load_mymodel                                  # noqa: E402
from data_provider.data_factory import data_provider            # noqa: E402

RESULTS = []


def gate(name, ok, detail):
    RESULTS.append(bool(ok))
    print(f"[2y] {'PASS' if ok else 'FAIL'} | {name:<3} {detail}")


def cli(cell, device='cpu', **kw):
    ns = parse_defaults()
    for key, value in {**rc.NEURON_FIXED, **rc.cell_2y(cell), 'input_scale': 4.}.items():
        setattr(ns, key, value)
    for key, value in kw.items():
        setattr(ns, key, value)
    ns.task, ns.cpu = 'ett', device == 'cpu'
    ns.num_patches = num_patches_of(ns)
    ns.device = torch.device(device)
    check_neuron_args(ns)
    return ns


def build(cell, seed=7, **kw):
    ns = cli(cell, **kw)
    torch.manual_seed(seed)
    m = load_mymodel(ns)
    m.eval()
    return m, ns


def batch(L=336, seed=3, B=4, C=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.cumsum(torch.randn(B, L, C, generator=g), dim=1)
    return x


def spikes_of(m, x):
    with torch.no_grad():
        _, aux = m(x, mode='full', return_aux=True)
    return aux['spikes']                                        # [T, B*C, 128]


def bump(x, t, size=8):
    y = x.clone()
    y[:, t * size:(t + 1) * size, :] += 3.
    return y


def defaults(device):
    two_w = rc.cell_2w('q1_R-direct-0.0005-all')
    name_same = rc.variant_of('ett', rc.cell_2y('q1_R-mem-336-all')) == rc.variant_of('ett', two_w)
    ns_old = parse_defaults()
    for key, value in {**rc.NEURON_FIXED, **two_w, 'input_scale': 4.}.items():
        setattr(ns_old, key, value)
    ns_old.task = 'ett'
    check_neuron_args(ns_old)
    tag_same = calibration_tag(ns_old) == '_n-pop_flif-tau4-8-16-32-somanone'
    # an old saved config has neither option; Config.load_args fills them with the parser defaults (0, 'full')
    filled = (ns_old.head_window, ns_old.neuron_memory) == (0, 'full')
    ns_old.num_patches, ns_old.device = 42, torch.device('cpu')
    torch.manual_seed(7)
    m_old = load_mymodel(ns_old).eval()
    m_new, _ = build('q1_R-mem-336-all')
    x = batch()
    sd_same = all(torch.equal(a, b) for a, b in zip(m_old.state_dict().values(), m_new.state_dict().values())) \
        and list(m_old.state_dict()) == list(m_new.state_dict())
    with torch.no_grad():
        out_same = torch.equal(m_old(x, mode='full'), m_new(x, mode='full'))
    repro = None
    if device.type == 'cuda':
        status, run, result = rc.find_cell('etthard-2w-20261022', 'ett', 'ETTh2', two_w, 13)
        model, args = rc.load_model(run, device)
        set_random_seed(args.seed)
        _, loader = data_provider(args, 'val')
        r = rc.one_pass(model, loader, args, ties=False)
        repro = abs(r['mse'] - json.loads(Path(result).read_text())['train']['best_val_loss'])
    gate('Y1', filled and name_same and tag_same and sd_same and out_same and (repro is None or repro <= rc.REPRO_TOL),
         f'2W name unchanged {name_same}, calibration tag unchanged {tag_same}, same initial values {sd_same}, '
         f'same output {out_same}, 2W checkpoint (ETTh2 seed 13) val |diff| {repro}')


def memoryless():
    m0, _ = build('q1-nomem-336-all')                 # q1: no window normalisation, which would
    m1, _ = build('q1-mem-336-all')                  # spread a change of one patch over the whole window
    x = batch()
    j = 10
    s0, s0b = spikes_of(m0, x), spikes_of(m0, bump(x, j))
    other = [t for t in range(42) if t != j]
    only_own = torch.equal(s0[other], s0b[other]) and not torch.equal(s0[j], s0b[j])
    s1, s1b = spikes_of(m1, x), spikes_of(m1, bump(x, j))
    later = not torch.equal(s1[j + 1:], s1b[j + 1:]) and torch.equal(s1[:j], s1b[:j])
    # member k of unit d fires exactly when I / tau_k >= theta (one solver step from v = 0)
    import layers
    with torch.no_grad():
        cur = m0.embedding.current(layers.to_patches(x, 8, 8))   # [T, B*C, 32]
    tau = torch.tensor([4., 8., 16., 32.])
    want = ((cur.unsqueeze(-1) / tau) >= 1.).float().reshape(42, cur.shape[1], 128)
    exact = torch.equal(s0, want)
    # gradients: spikes of the last step w.r.t. the current of every step (surrogate path)
    grads = {}
    for name, m in (('nomem', m0), ('mem', m1)):
        c = m.embedding.current(layers.to_patches(x, 8, 8)).detach().requires_grad_(True)
        emb = m.embedding
        if emb.memoryless:
            T, B = c.shape[:2]
            s = emb.neuron(c.reshape(1, T * B, -1)).reshape(T, B, -1)
        else:
            s = emb.neuron(c)
        s[-1].sum().backward()
        g = c.grad.abs().sum(dim=(1, 2))
        grads[name] = (float(g[:-1].max()), float(g[-1]))
    no_past = grads['nomem'][0] == 0. and grads['nomem'][1] > 0.
    past = grads['mem'][0] > 0.
    gate('Y2', only_own and later and exact and no_past and past,
         f'nomem: changing patch {j} changes only step {j} {only_own}; spikes == (I/tau_k >= 1) {exact}; '
         f'grad to earlier steps / last step {grads["nomem"]}; mem: later steps change {later}, grad to the past '
         f'{grads["mem"][0]:.3g}')


def same_start():
    import layers
    m0, _ = build('q1-nomem-336-all')
    m1, _ = build('q1-mem-336-all')
    sd = all(torch.equal(a, b) for a, b in zip(m0.state_dict().values(), m1.state_dict().values()))
    x = batch()
    p = layers.to_patches(x, 8, 8)
    with torch.no_grad():
        same_current = torch.equal(m0.embedding.current(p), m1.embedding.current(p))
    m0.train(); m1.train()
    with torch.no_grad():
        same_train_current = torch.equal(m0.embedding.current(p), m1.embedding.current(p))
    gate('Y3', sd and same_current and same_train_current,
         f'same initial parameters {sd}; same per-step BN current (eval / train) {same_current} / {same_train_current}')


def head_window():
    m0, _ = build('q1-nomem-336-last1')
    m1, _ = build('q1-mem-336-last1')
    x = batch()
    with torch.no_grad():
        y0 = m0(x, mode='full')
        early = all(torch.equal(y0, m0(bump(x, t), mode='full')) for t in (0, 20, 40))
        last = not torch.equal(y0, m0(bump(x, 41), mode='full'))
        y1 = m1(x, mode='full')
        through_state = not torch.equal(y1, m1(bump(x, 20), mode='full'))
    width = (m0.head.in_features, m1.head.in_features)
    gate('Y4', early and last and through_state and width == (128, 128),
         f'head inputs {width}; nomem: earlier patches leave the forecast unchanged {early}, the last patch moves '
         f'it {last}; mem: patch 20 reaches the forecast through the state {through_state}')


def short_window(device):
    m, ns = build('q1_R-mem-96-all')
    ml, _ = build('q1_R-mem-96-last1')
    with torch.no_grad():
        out = m(batch(L=96), mode='full')
    shapes = (ns.num_patches, m.head.in_features, ml.head.in_features, tuple(out.shape))
    rows = None
    if device.type == 'cuda':
        ys = {}
        for L in (336, 96):
            args = cli('q1_R-mem-336-all', device=str(device))
            args.seq_len, args.data, args.data_path, args.root_path = L, 'ETTh1', 'ETTh1.csv', str(rc.DATA_ROOT)
            args.dataset = 'ETTh1'
            _, loader = data_provider(args, 'val')
            ys[L] = torch.cat([b[1] for b in loader])
        rows = (ys[336].shape, ys[96].shape, torch.equal(ys[336], ys[96]))
    ok = shapes == (12, 12 * 128, 128, (4, 96, 3)) and (rows is None or (rows[0] == rows[1] and rows[2]))
    gate('Y5', ok, f'T, head inputs (all / last1), output {shapes}; validation targets 336 vs 96 {rows}')


def refusals():
    cases = {'head_window with flatten': dict(head_mode='flatten', head_window=1),
             'head_window 13 > T 12': dict(seq_len=96, head_window=13),
             'nomem with pearson': dict(neuron='sel_pop_flif', mode='hard', hard_axis='shared', hard_stat='pearson',
                                        hard_q=.5, select_part='all', neuron_memory='none')}
    refused = {}
    for name, kw in cases.items():
        try:
            cli('q1-mem-336-all', **kw)
            refused[name] = False
        except ValueError:
            refused[name] = True
    import layers
    try:
        layers.Embedding(8, 32, 4., 'frozen', neuron='legacy', neuron_memory='none')
        refused['nomem with legacy'] = False
    except ValueError:
        refused['nomem with legacy'] = True
    gate('Y6', all(refused.values()), f'refused {refused}')


def names():
    import glob
    variants = {c: rc.variant_of('ett', rc.cell_2y(f'q1_R-{c}')) for c in rc.CELLS_2Y}
    tags = {c: calibration_tag(cli(f'q1_R-{c}')) for c in rc.CELLS_2Y}
    distinct = len(set(variants.values())) == 6
    families = len(set(tags.values())) == 3
    carried = all(('_L96' in v) == ('-96-' in c) and ('_hw1' in v) == c.endswith('last1') and ('-nomem' in v) == c.startswith('nomem')
                  for c, v in variants.items())
    # a run of each family finds only its own family's files (file pattern + content check in train.load_calibration)
    picked = {}
    for c in ('mem-336-all', 'mem-96-all', 'nomem-336-all'):
        ns = cli(f'q1_R-{c}')
        ns.dataset, ns.calibration_dir = 'ETTh1', 'calibration'
        stem = f"{ns.dataset}_a{ns.alpha}_norm-{ns.input_norm}_revin{calibration_tag(ns)}_seed*.json"
        found = sorted(glob.glob(str(rc.TASK / 'results' / 'calibration' / stem)))
        picked[c] = [Path(f).name for f in found]
    no_cross = all(('_L96' in f) == ('96' in c) and ('nomem' in f) == c.startswith('nomem')
                   for c, fs in picked.items() for f in fs)
    gate('Y7', distinct and families and carried and no_cross,
         f'6 distinct run names {distinct}; 3 calibration tags {families} {sorted(set(tags.values()))}; '
         f'length / readout / memory in the name {carried}; files a family can find {picked}')


def smoke(suite, calibration, device):
    cal = dict(kv.split('=') for kv in calibration.split(','))
    rows, ok = [], True
    for cell, fam in (('q1_R-mem-96-last1', 'L96'), ('q1-nomem-336-all', 'nomem'), ('q1_R-mem-336-last1', None)):
        spec = rc.cell_2y(cell)
        status, run, result = rc.find_cell(suite, 'ett', 'ETTh1', spec, 7)
        model, args = rc.load_model(run, device)
        set_random_seed(args.seed)
        _, loader = data_provider(args, 'val')
        r = rc.one_pass(model, loader, args, ties=False)
        best = json.loads(Path(result).read_text())['train']['best_val_loss']
        c = rc.calibration_manifest(cal[fam] if fam else 'rerun2u-cal-20261021')[rc.calibration_label('ett', cell.split('-')[0], 'ETTh1')]
        same_file = args.calibration_file == c['file']
        good = status == 'done' and abs(r['mse'] - best) <= rc.REPRO_TOL and r['safe'] and same_file
        ok &= good
        rows.append(f"{cell}: {status}, val |diff| {abs(r['mse'] - best):.2e}, safe {r['safe']}, rate {r['firing_rate']:.3f}, "
                    f"calibration {args.calibration_file} ({'its family' if same_file else 'WRONG'})")
    gate('Y8', ok, '; '.join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', default=None)
    ap.add_argument('--calibration', default=None)
    ap.add_argument('-nd', '--num_device', type=int, default=0)
    a = ap.parse_args()
    device = torch.device(f'cuda:{a.num_device}') if torch.cuda.is_available() else torch.device('cpu')
    defaults(device)
    memoryless()
    same_start()
    head_window()
    short_window(device)
    refusals()
    names()
    if a.smoke:
        smoke(a.smoke, a.calibration, device)
    else:
        print('[2y] Y8    not run (no --smoke suite)')
    print(f'[2y] {sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed')
    return 0 if all(RESULTS) else 1


if __name__ == '__main__':
    raise SystemExit(main())
