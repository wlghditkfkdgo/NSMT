"""Checks of the head input forms (2026-11-03, exp/f-lif-pop-v3; PROJECT_LOG head input form pre-registration; head_pool.py).
Fail closed: a missing value, NaN, or any mismatch fails the check.

  p0  the default path is unchanged: the 1-epoch smoke runs with the flag off (suite smoke-hp-off-<date>) give the same RESULT
      fields (all but time_s / sha / dir) as the flag-off smoke runs of the H1 checks (smoke-hm-after-20261103, the same default
      code: no code change since 4be7ceb08), for the CinC and MIT-BIH trainers x {muxlin, Spikformer-style}; the files changed
      against HEAD are only the listed ones
  p1  taps: e, x1 and h equal the model's own tensors bitwise (spkMixer: forward_parts' e and h, connect(e, m0); myModel
      --hippo_ssa: recomputed e (1 - SSA(e)) and x1 (1 - MLP(x1))); seg4 splits N tokens into 4 contiguous parts covering all tokens
  p2  zero start: for every form x model x task (CinC 117 tokens / 4 classes, MIT-BIH 11 tokens / 5 classes), with the same seed,
      the same RNG state after construction, the same weights (plus only the new zero parameter), bitwise the same output, loss and
      gradients of every shared parameter as the model without the form at the start of training; one AdamW step moves it
  p3  energy of the smoke runs with the forms (smoke-hp-on-<date>): CinC syops counts pool_readout of seg4 / layers (rate inputs) as
      MAC = inputs x outputs x T_h per window and of max (0/1 inputs) as ACs only, 0 < AC <= inputs x outputs x T_h (one AC per
      1-input; corrected 11-03 after the first run, which expected AC + MAC = the full count); MIT-BIH energy_recount_mitbih.py passes
      on them (recomputed macro-F1 = logged)
  smoke  every smoke run has a RESULT line and no SKIP (off 4, on 16)
usage: python check_hp.py --date 20261103
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
import torch                                                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                         # noqa: E402
from config import Config, set_random_seed                               # noqa: E402
from load_model import LOAD_MODEL                                        # noqa: E402
from head_pool import KINDS                                              # noqa: E402
from check_hm import result_rows, layer_line                             # noqa: E402

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRATCH = os.environ.get('CHECK_HP_SCRATCH', '/tmp/check_hp')
PY = sys.executable
CHANGED_OK = {'NSMT/cinc_mixer_v1/' + f for f in (
    'config.py', 'load_model.py', 'forecast_wrap.py', 'ad_wrap.py', 'scripts/run_rhythm_spk_mux.sh', 'scripts/run_mitbih_spk.sh')}
TASKS = {   # task -> (argv, num_classes, tokens, spkMixer residual as tuned)
    'cinc': (['--seq_len', '3750', '--patch_size', '64'], 4, 117, 'OR'),
    'mitbih': (['--seq_len', '187', '--patch_size', '32'], 5, 11, 'ADD'),
}
EXPS = ('nsmt_flif_v1_rhythm', 'cinc_mixer_v1_mitbih_spkmux')
DIMS = {'max': 1, 'seg4': 4, 'layers': 2}


def build(task, kind, form, seed=42):
    import config as cfgmod
    extra, ncls, _, res = TASKS[task]
    argv = ['x', '--data', 'X', '--seed', str(seed), '-s', '--no-bias', '--layers', '1', '--tau', '2.0', '-nh', '8',
            '--time_layers', '2', '--log_dir', SCRATCH, '-emb', '384', '--mlp_ratios', '1'] + extra
    argv += ['--model', 'spkMixer', '--mixer', 'lif', '--mixer_comb', 'muxlin', '--connect_f', res, '--hippo_T', '4'] if kind == 'spkmix' \
        else ['--model', 'myModel', '--hippo_ssa', '--hippo_T', '4']
    if form != 'none':
        argv += ['--head_pool', form]
    old = sys.argv; sys.argv = argv
    try:
        args = Config(); args.set_args(cfgmod.parse_arguments())
    finally:
        sys.argv = old
    args.num_classes, args.num_channels, args.device = ncls, 1, DEV
    set_random_seed(seed)
    model = LOAD_MODEL[args.model](args, True)
    rng = torch.get_rng_state()
    return model.to(DEV), rng


def inputs(task, seed=0):
    g = torch.Generator().manual_seed(seed)
    L, ncls = int(TASKS[task][0][1]), TASKS[task][1]
    return torch.randn(4, 1, L, generator=g).to(DEV), torch.randint(0, ncls, (4,), generator=g).to(DEV)


def run_once(model, x, y):
    model.train(); model.train_mode = 'training'
    torch.manual_seed(7); torch.cuda.manual_seed_all(7)                  # same dropout masks for both models
    functional.reset_net(model)
    out = model(x)
    out = out[0] if isinstance(out, tuple) else out
    loss = torch.nn.functional.cross_entropy(out, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return out.detach(), loss.detach()


@torch.no_grad()
def taps_match(model, x):
    """p1: the taps equal the model's own tensors (training-mode batch norm, dropout off, so the neurons fire as in training)."""
    model.train(); model.train_mode = 'training'
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    t = model._pool_taps
    functional.reset_net(model)
    if hasattr(model, 'blocks'):
        _, e, m0, _, h = model.forward_parts(x)
        x1 = model.blocks[-1].connect(e, m0)
    else:
        model(x)
        e = t.e.clone()
        blk = model.data_block[-1]
        functional.reset_net(model)
        x1 = e * (1. - blk.ssa(e))
        h = x1 * (1. - blk.mlp(x1))
        functional.reset_net(model)
        model(x)
    return torch.equal(t.e, e) and torch.equal(t.x1, x1) and torch.equal(t.h(), h), tuple(h.shape), \
        tuple(round(float(v.float().mean()), 4) for v in (e, x1, h))


def p1_p2():
    msgs, ok = [], True
    for task in TASKS:
        x, y = inputs(task)
        N = TASKS[task][2]
        sizes = [s.shape[0] for s in torch.tensor_split(torch.arange(N), 4)]
        cover = torch.equal(torch.cat(torch.tensor_split(torch.arange(N), 4)), torch.arange(N))
        msgs.append(f'{task}: seg4 sizes {sizes} cover all {N} tokens in order {cover}')
        ok &= cover and len(sizes) == 4 and min(sizes) > 0
        for kind in ('spkmix', 'ssa'):
            m0, r0 = build(task, kind, 'none')
            s0 = {k: v.clone() for k, v in m0.state_dict().items()}       # before training touches the batch-norm statistics
            o0, l0 = run_once(m0, x, y)
            g0 = {k: p.grad for k, p in m0.named_parameters()}
            for form in KINDS:
                m1, r1 = build(task, kind, form)
                s1 = m1.state_dict()
                new = 'pool_q' if form == 'attn' else 'pool_readout.weight'
                extra = sorted(set(s1) - set(s0))
                good = torch.equal(r0, r1) and set(s0) <= set(s1) and extra == [new] and all(torch.equal(s0[k], s1[k]) for k in s0) \
                    and float(s1[new].abs().sum()) == 0.
                o1, l1 = run_once(m1, x, y)
                g1 = {k: p.grad for k, p in m1.named_parameters()}
                same_grad = all((g0[k] is None and g1[k] is None) or (g0[k] is not None and g1[k] is not None and torch.equal(g0[k], g1[k]))
                                for k in g0)
                p2 = good and torch.equal(o0, o1) and torch.equal(l0, l1) and same_grad and bool(torch.isfinite(l1).all())
                opt = torch.optim.AdamW(m1.parameters(), lr=1e-3)
                opt.step()
                moved = float(m1.state_dict()[new].abs().sum()) > 0
                p1, shp, rate = taps_match(m1, x)
                if form != 'attn':
                    shape_ok = tuple(s1[new].shape) == (TASKS[task][1], DIMS[form] * 384)
                else:
                    shape_ok = tuple(s1[new].shape) == (384,)
                msgs.append(f'{task}/{kind}/{form}: p1 {p1} (h {shp}, mean e / x1 / h {rate}) p2 {p2} (rng/weights {good}, out {torch.equal(o0, o1)}, '
                            f'loss {torch.equal(l0, l1)}, grads {same_grad}) new {new} {tuple(s1[new].shape)} {shape_ok} moves {moved}')
                ok &= bool(p1 and p2 and moved and shape_ok)
                del m1
            del m0
            torch.cuda.empty_cache()
    return ok, msgs


def p0(date):
    b, sb = result_rows('smoke-hm-after-20261103')
    b = {k: v for k, v in b.items() if k[0] in EXPS}
    a, sa = result_rows(f'smoke-hp-off-{date}')
    msgs, ok = [], len(b) == 4 and len(a) == 4 and sb == 0 and sa == 0 and set(a) == set(b)
    msgs.append(f'reference rows {len(b)}, flag-off rows {len(a)}, same keys {set(a) == set(b)}')
    for k in sorted(b):
        if k not in a:
            continue
        diff = [f for f in sorted(set(a[k]) | set(b[k])) if f not in ('time_s', 'sha', 'dir', 'out') and a[k].get(f) != b[k].get(f)]
        msgs.append(f'{k[1]} {k[0]}: {len(b[k]) - 3} fields, differing {diff if diff else "none"}')
        ok &= not diff
    root = os.path.abspath(os.path.join(HERE, '..', '..'))
    ch = set(subprocess.run(['git', 'diff', '--name-only', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.split())
    msgs.append(f'changed files vs HEAD: {sorted(ch)}; outside the list: {sorted(ch - CHANGED_OK)}')
    ok &= bool(ch) and ch <= CHANGED_OK
    return ok, msgs


def p3(date):
    suite = f'smoke-hp-on-{date}'
    rows, sk = result_rows(suite)
    msgs, ok = [], len(rows) == 16 and sk == 0
    for k, r in sorted(rows.items()):
        form = next((f for f in KINDS if r.get('tag', '').endswith(f'_hp{f}')), None)
        if form is None:
            ok = False; msgs.append(f'{k}: tag without _hp<form>'); continue
        if r['exp'] == 'nsmt_flif_v1_rhythm':
            ll = layer_line(os.path.join(r['dir'], 'log'), 'pool_readout')
            if form == 'attn':
                good = ll is None
                msgs.append(f"CinC {r['cond']} {form}: no pool_readout line {good} (dot products / softmax not counted)")
            elif form == 'max':                                          # 0/1 input: syops counts an AC per 1-input only, no MAC
                n = 384 * 4 * 4
                good = ll is not None and ll[1] <= 1. and 0. < ll[0] <= n * 1.006
                msgs.append(f"CinC {r['cond']} {form}: pool_readout syops (AC, MAC) {ll}: MAC 0 and 0 < AC <= {n} -> {good}")
            else:                                                        # real input (rates): MAC = inputs x outputs x T_h
                n = DIMS[form] * 384 * 4 * 4
                good = ll is not None and ll[0] <= 1. and abs(ll[1] - n) <= max(1., 0.006 * n)
                msgs.append(f"CinC {r['cond']} {form}: pool_readout syops (AC, MAC) {ll} vs MAC {n} -> {good}")
            ok &= good
    raw = open(os.path.join(HERE, 'results', suite, 'raw.txt')).read().splitlines()
    sub = f'{suite}-mitbih'                                                # one task per recount
    os.makedirs(os.path.join(HERE, 'results', sub), exist_ok=True)
    with open(os.path.join(HERE, 'results', sub, 'raw.txt'), 'w') as f:
        f.write(''.join(l + '\n' for l in raw if l.startswith('RESULT') and ' exp=cinc_mixer_v1_mitbih_spkmux ' in l))
    p = subprocess.run([PY, '-u', 'energy_recount_mitbih.py', '--suite', sub], cwd=HERE, capture_output=True, text=True, env=dict(os.environ))
    tail = p.stdout.strip().splitlines()[-2:]                            # the script's own last lines (stderr holds library warnings)
    msgs.append(f'energy_recount_mitbih.py: rc {p.returncode}; ' + ' | '.join(tail))
    ok &= p.returncode == 0 and bool(tail) and tail[-1] == 'all checks passed'
    return ok, msgs


def smoke(date):
    ok, msgs = True, []
    for s, n in (('off', 4), ('on', 16)):
        rows, sk = result_rows(f'smoke-hp-{s}-{date}')
        msgs.append(f'smoke-hp-{s}: RESULT {len(rows)}, SKIP {sk}')
        ok &= len(rows) == n and sk == 0
    return ok, msgs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--date', required=True); ap.add_argument('--only', default='p0,p1p2,p3,smoke')
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    results = []
    for name, fn in (('p0', lambda: p0(a.date)), ('p1p2', p1_p2), ('p3', lambda: p3(a.date)), ('smoke', lambda: smoke(a.date))):
        if name not in a.only.split(','):
            continue
        try:
            ok, msgs = fn()
        except Exception as e:                                           # fail closed
            ok, msgs = False, [f'exception: {type(e).__name__}: {e}']
        results.append((name, ok))
        print(f'[{name}] {"pass" if ok else "FAIL"}', flush=True)
        for m in msgs:
            print('   ' + m, flush=True)
    print(f'{sum(ok for _, ok in results)}/{len(results)} pass')
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == '__main__':
    main()
