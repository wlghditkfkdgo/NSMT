"""Checks of the H1 head (2026-11-03, exp/f-lif-pop-v3; PROJECT_LOG H1 pre-registration; head_mem.py). Fail closed: a missing
value, NaN, or any mismatch fails the check.

  h0  the default path is unchanged: the 1-epoch smoke runs before the edit (suite smoke-hm-before-<date>, commit 783072bcf) and
      after it with the flag off (smoke-hm-after-<date>) give the same RESULT fields (all but time_s / sha / dir), for the four task
      trainers x {muxlin, Spikformer-style}; the files changed against HEAD are only the listed ones
  h1  the captured charge reproduces the last neuron layer's spikes exactly ([H - theta >= 0] == output spikes), both neuron kinds,
      classification / forecasting / reconstruction wrappers (AD on real MSL windows)
  h2  zero-initialised readout: with the same seed, the H1 model is built with the same RNG state after construction, the same
      weights (plus only mem_readout.weight = 0), and gives bitwise the same output, loss and gradients of every shared parameter
      as the model without H1 at the start of training; one AdamW step makes mem_readout non-zero
  h3  energy of the H1 smoke runs: AD energy_check ok with mem_MAC = N * D * S * C; ETT energy_recount_ett.py passes on the H1
      smoke suite with mem_MAC = N * D * H * C; MIT-BIH energy_recount_mitbih.py passes on it; CinC syops counts mem_readout as
      MAC only (T_h * D * classes per window)
  smoke  every smoke run of the three suites has a RESULT line and no SKIP
usage: python check_hm.py --date 20261103
"""
import argparse
import os
import re
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                         # noqa: E402
from config import Config, set_random_seed                               # noqa: E402
from load_model import LOAD_MODEL                                        # noqa: E402
from forecast_wrap import Forecaster                                     # noqa: E402
from ad_wrap import Reconstructor                                        # noqa: E402
from summarize_rhythm import parse                                       # noqa: E402

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRATCH = os.environ.get('CHECK_HM_SCRATCH', '/tmp/check_hm')
PY = sys.executable
CHANGED_OK = {'NSMT/cinc_mixer_v1/' + f for f in (
    'config.py', 'load_model.py', 'forecast_wrap.py', 'ad_wrap.py', 'train_ad.py', 'energy_recount_ett.py',
    'scripts/run_rhythm_spk_mux.sh', 'scripts/run_ett_spk.sh', 'scripts/run_mitbih_spk.sh', 'scripts/run_ad_spk.sh')}
TASKS = {   # task -> (argv, num_classes, channels)
    'cls': (['--seq_len', '3750', '--patch_size', '64', '-emb', '384', '--mlp_ratios', '1'], 4, 1),
    'ett': (['--seq_len', '96', '--patch_size', '8', '-emb', '64', '--mlp_ratios', '0.5'], 1, 1),
    'ad': (['--seq_len', '100', '--patch_size', '8', '-emb', '64', '--mlp_ratios', '1'], 1, 55),
}


def build(task, kind, head_mem, seed=42):
    import config as cfgmod
    extra, ncls, C = TASKS[task]
    argv = ['x', '--data', 'X', '--seed', str(seed), '-s', '--no-bias', '--layers', '1', '--tau', '2.0', '-nh', '8',
            '--time_layers', '2', '--log_dir', SCRATCH] + extra
    argv += ['--model', 'spkMixer', '--mixer', 'lif', '--mixer_comb', 'muxlin', '--connect_f', 'OR', '--hippo_T', '4'] if kind == 'spkmix' \
        else ['--model', 'myModel', '--hippo_ssa', '--hippo_T', '4']
    if head_mem:
        argv.append('--head_mem')
    old = sys.argv; sys.argv = argv
    try:
        args = Config(); args.set_args(cfgmod.parse_arguments())
    finally:
        sys.argv = old
    args.num_classes, args.num_channels, args.device = ncls, C, DEV
    set_random_seed(seed)
    inner = LOAD_MODEL[args.model](args, True)
    if task == 'cls':
        model = inner
    elif task == 'ett':
        model = Forecaster(inner, args.embed_dim, 96, bias=args.bias)
    else:
        model = Reconstructor(inner, args.embed_dim, 100, C, bias=args.bias)
    rng = torch.get_rng_state()
    return model.to(DEV), rng


def inputs(task, seed=0):
    g = torch.Generator().manual_seed(seed)
    if task == 'cls':
        return torch.randn(4, 1, 3750, generator=g).to(DEV), torch.randint(0, 4, (4,), generator=g).to(DEV)
    if task == 'ett':
        return torch.randn(2, 96, 7, generator=g).to(DEV), torch.randn(2, 96, 7, generator=g).to(DEV)
    import types
    from data_provider.data_factory import data_dict
    ds = data_dict['MSL'](types.SimpleNamespace(data='MSL'), os.path.join(
        '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/'
        'anomaly_detection/dataset', 'MSL'), 100, flag='test')
    x = torch.stack([torch.as_tensor(ds[i][0]) for i in (0, 5000, 20000, 60000)]).float()
    return x.to(DEV), x.to(DEV)


def run_once(model, task, x, y):
    model.train(); model.train_mode = 'training'
    torch.manual_seed(7); torch.cuda.manual_seed_all(7)                  # same dropout masks for both models
    functional.reset_net(model)
    out = model(x)
    if task == 'cls':
        out = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(out, y)
    else:
        loss = ((out - y) ** 2).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    return out.detach(), loss.detach()


def last_lif(model):
    inner = getattr(model, 'inner', model)
    return inner.blocks[-1].fc2.lif if hasattr(inner, 'blocks') else inner.data_block[-1].mlp.fc2.lif


def h1_h2():
    msgs, ok = [], True
    for task in ('cls', 'ett', 'ad'):
        x, y = inputs(task)
        for kind in ('spkmix', 'ssa'):
            m0, r0 = build(task, kind, False)
            m1, r1 = build(task, kind, True)
            tag = f'{task}/{kind}'
            s0, s1 = m0.state_dict(), m1.state_dict()
            extra = sorted(set(s1) - set(s0))
            good = torch.equal(r0, r1) and set(s0) <= set(s1) and extra == ['mem_readout.weight'] and \
                all(torch.equal(s0[k], s1[k]) for k in s0) and float(s1['mem_readout.weight'].abs().sum()) == 0.
            spk = {}
            hnd = last_lif(m1).register_forward_hook(lambda mod, i, o: spk.__setitem__('s', o.detach()))
            o0, l0 = run_once(m0, task, x, y)
            o1, l1 = run_once(m1, task, x, y)
            hnd.remove()
            tap = (m1.inner if hasattr(m1, 'inner') else m1)._charge_tap
            c = tap.charge.detach()
            h1 = 's' in spk and c.shape == spk['s'].shape and torch.equal((c >= 0).to(spk['s'].dtype), spk['s'])
            g0 = {k: p.grad for k, p in m0.named_parameters()}
            g1 = {k: p.grad for k, p in m1.named_parameters()}
            same_grad = all((g0[k] is None and g1[k] is None) or (g0[k] is not None and g1[k] is not None and torch.equal(g0[k], g1[k]))
                            for k in g0)
            h2 = good and torch.equal(o0, o1) and torch.equal(l0, l1) and same_grad and torch.isfinite(l1).all()
            opt = torch.optim.AdamW(m1.parameters(), lr=1e-3)
            opt.step()
            moved = float(m1.state_dict()['mem_readout.weight'].abs().sum()) > 0
            rate = float(spk['s'].float().mean()) if 's' in spk else float('nan')
            msgs.append(f'{tag}: h1 {h1} (charge {tuple(c.shape)}, rate {rate:.4f}) h2 {h2} (rng/weights {good}, out {torch.equal(o0, o1)}, '
                        f'loss {torch.equal(l0, l1)}, grads {same_grad}) readout moves after one step {moved}')
            ok &= bool(h1 and h2 and moved)
            del m0, m1
            torch.cuda.empty_cache()
    return ok, msgs


def result_rows(suite):
    p = os.path.join(HERE, 'results', suite, 'raw.txt')
    lines = open(p).read().splitlines() if os.path.exists(p) else []
    rows = {}
    for l in lines:
        if not l.startswith('RESULT'):
            continue
        kv = dict(x.split('=', 1) for x in l.split()[1:] if '=' in x)
        key = (kv['exp'], kv['cond'], kv.get('tag', ''), kv.get('ds', kv.get('dataset', '')), kv.get('pl', ''))
        rows[key] = kv
    return rows, sum(l.startswith('SKIP') for l in lines)


def h0(date):
    b, sb = result_rows(f'smoke-hm-before-{date}')
    a, sa = result_rows(f'smoke-hm-after-{date}')
    msgs, ok = [], len(b) == 8 and len(a) == 8 and sb == 0 and sa == 0 and set(a) == set(b)
    for k in sorted(b):
        if k not in a:
            continue
        diff = [f for f in sorted(set(a[k]) | set(b[k])) if f not in ('time_s', 'sha', 'dir', 'out') and a[k].get(f) != b[k].get(f)]
        msgs.append(f'{k[1]} {k[3]}: {len(b[k]) - 3} fields, differing {diff if diff else "none"}')
        ok &= not diff
    root = os.path.abspath(os.path.join(HERE, '..', '..'))
    ch = set(subprocess.run(['git', 'diff', '--name-only', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.split())
    msgs.append(f'changed files vs HEAD: {sorted(ch)}; outside the list: {sorted(ch - CHANGED_OK)}')
    ok &= bool(ch) and ch <= CHANGED_OK
    return ok, msgs


def layer_line(run_log, name):
    for line in open(os.path.join(run_log, 'model+info+per+layer.txt'), encoding='utf-8'):
        if line.startswith(f'  ({name}): Linear('):
            m = re.search(r'%\s*Params,\s*([\d.]+)\s*([KMG]?)\s*Ops,\s*[\d.]+%\s*ACs,\s*([\d.]+)\s*([KMG]?)\s*Ops', line)
            U = {'': 1., 'K': 1e3, 'M': 1e6, 'G': 1e9}
            if m:
                return float(m.group(1)) * U[m.group(2)], float(m.group(3)) * U[m.group(4)]
    return None


def h3(date):
    suite = f'smoke-hm-h1-{date}'
    rows, sk = result_rows(suite)
    msgs, ok = [], len(rows) == 8 and sk == 0
    for k, r in sorted(rows.items()):
        if '_hm' not in r.get('tag', ''):
            ok = False; msgs.append(f'{k}: tag without _hm'); continue
        if r['exp'] == 'cinc_mixer_v1_ad_spkmux':
            good = r.get('energy_check') == 'ok' and int(r.get('mem_MAC', -1)) == 25 * 64 * 4 * 55
            msgs.append(f"AD {r['cond']}: energy_check {r.get('energy_check')} mem_MAC {r.get('mem_MAC')} -> {good}")
            ok &= good
        elif r['exp'] == 'nsmt_flif_v1_rhythm':
            ll = layer_line(os.path.join(r['dir'], 'log'), 'mem_readout')
            good = ll is not None and ll[0] <= 1. and abs(ll[1] - 4 * 384 * 4) <= max(1., 0.006 * 4 * 384 * 4)
            msgs.append(f"CinC {r['cond']}: mem_readout syops (AC, MAC) {ll} vs MAC {4 * 384 * 4} -> {good}")
            ok &= good
    env = dict(os.environ)
    raw = open(os.path.join(HERE, 'results', suite, 'raw.txt')).read().splitlines()
    for sub, exp in (('ett', 'exp=cinc_mixer_v1_ett_spkmux'), ('mitbih', 'exp=cinc_mixer_v1_mitbih_spkmux')):   # one task per recount
        os.makedirs(os.path.join(HERE, 'results', f'{suite}-{sub}'), exist_ok=True)
        with open(os.path.join(HERE, 'results', f'{suite}-{sub}', 'raw.txt'), 'w') as f:
            f.write(''.join(l + '\n' for l in raw if l.startswith('RESULT') and f' {exp} ' in l))
    for script, extra in (('energy_recount_ett.py', ['--suite', f'{suite}-ett']), ('energy_recount_mitbih.py', ['--suite', f'{suite}-mitbih'])):
        p = subprocess.run([PY, '-u', script] + extra, cwd=HERE, capture_output=True, text=True, env=env)
        tail = p.stdout.strip().splitlines()[-2:]                        # the script's own last lines (stderr holds library warnings)
        msgs.append(f'{script}: rc {p.returncode}; ' + ' | '.join(tail))
        ok &= p.returncode == 0
    ep = os.path.join(HERE, 'results', f'{suite}-ett', 'energy_recount.csv')
    if os.path.exists(ep):
        import csv
        for e in csv.DictReader(open(ep)):
            if 'pl' in e and e.get('cond', '').startswith(('spkmx', 'spk_ssa')) and e.get('mem_MAC') not in (None, ''):
                good = int(float(e['mem_MAC'])) == int(float(e['head_MAC']))
                msgs.append(f"ETT {e['cond']}: mem_MAC {e['mem_MAC']} = head_MAC {e['head_MAC']} -> {good}")
                ok &= good
    return ok, msgs


def smoke(date):
    ok, msgs = True, []
    for s in ('before', 'after', 'h1'):
        rows, sk = result_rows(f'smoke-hm-{s}-{date}')
        msgs.append(f'smoke-hm-{s}: RESULT {len(rows)}, SKIP {sk}')
        ok &= len(rows) == 8 and sk == 0
    return ok, msgs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--date', required=True); ap.add_argument('--only', default='h0,h1h2,h3,smoke')
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    results = []
    for name, fn in (('h0', lambda: h0(a.date)), ('h1h2', h1_h2), ('h3', lambda: h3(a.date)), ('smoke', lambda: smoke(a.date))):
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
