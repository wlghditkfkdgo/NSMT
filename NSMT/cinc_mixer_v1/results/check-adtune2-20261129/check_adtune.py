"""Checks of the AD tuning code (2026-11-29, before training). Fails closed; prints PASS / FAIL and ALL CHECKS PASSED.
  c1 runner default path: the old run_ad_spk.sh (copy given by --old_dir) and the new one give the same train_ad.py arguments and the same
     tag for 6 earlier configurations (fake python that prints its arguments)
  c2 tag agreement: for every candidate configuration of every stage (each dataset, all values, also combined with non-default values of
     the other settings), the runner's --tag equals adtune.tag(); the runner passes exactly the requested settings; all tags of a dataset
     are distinct (run folder + stdout names are built from the tag)
  c3 Reconstructor: norm 'window' gives bit-identical outputs to the old ad_wrap.py with the same weights; 'mean' feeds x - mean(x) and
     adds the mean back; 'none' feeds x unchanged; a bad mode is refused
usage (from cinc_mixer_v1/): python -B results/check-adtune2-20261129/check_adtune.py --old_dir <copy of the old runner / ad_wrap.py> --tmp <dir>
"""
import argparse
import importlib.util
import itertools
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(os.path.dirname(HERE))
N = os.path.dirname(C)
sys.path.insert(0, C)
os.chdir(C)
import adtune as T                                                       # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--old_dir', required=True); ap.add_argument('--tmp', required=True); a = ap.parse_args()
os.makedirs(a.tmp, exist_ok=True)
bad = 0


def ok(c, m):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + m, flush=True)


fake = os.path.join(a.tmp, 'python')
open(fake, 'w').write('#!/bin/bash\n[ "$1" = "-" ] && { cat >/dev/null; exit 0; }\necho "ARGS: $*" >> "$FAKEOUT"\n')
os.chmod(fake, 0o755)


def runner(script, env, ds='MSL', seed='42'):
    out = os.path.join(a.tmp, 'args.txt')
    if os.path.exists(out):
        os.remove(out)
    e = dict(os.environ, FAKEOUT=out, PY=fake, RES=os.path.join(a.tmp, 'res'), LOG=os.path.join(a.tmp, 'log'), SUITE='x', **env)
    subprocess.run(['bash', script, ds, seed, '0'], cwd=N, env=e, capture_output=True, text=True)
    if not os.path.exists(out):
        return None
    args = open(out).read().split()[1:]
    args = [x for x in args if not x.startswith(a.tmp)]
    return args


def argval(args, k):
    return args[args.index(k) + 1] if k in args else None


NEW, OLD = os.path.join(C, 'scripts', 'run_ad_spk.sh'), os.path.join(a.old_dir, 'run_ad_spk.sh')
# c1
for env in (dict(COND='spkmx_lif_muxlin_or', LR='0.002', MLAYERS='1'), dict(COND='spkmx_lif_muxlin_iand', LR='0.002', MLAYERS='1'),
            dict(COND='spkmx_lif_muxlin_add', LR='0.001', MLAYERS='3'), dict(COND='spkmx_lif_muxlin_or', LR='0.0005', MLAYERS='1', HOLDOUT='1'),
            dict(COND='spk_ssa', NALL='lif', LR='0.002'), dict(COND='spkmx_lif_muxlin_or', LR='0.002', MLAYERS='2', MIXPOOL='mean')):
    o, n = runner(OLD, env), runner(NEW, env)
    ok(o is not None and o == n, f'c1 default path {env}: same arguments ({"tag " + argval(n, "--tag") if n else "no run"})')
# c2
bad0, tags = bad, {}
for ds in ('MSL', 'SMAP', 'PSM'):
    tags[ds] = set()
    vals = {k: sorted({v[i] for _, keys, vs in T.STAGES for v in vs for i, kk in enumerate(keys) if kk == k}, key=str) for k in ('norm', 'ep', 'emb', 'bl', 'ml', 'pt', 'mlp', 'lr', 'res')}
    combos = []
    for name, keys, vs in T.STAGES:                                     # every stage value on the start setting
        for v in vs:
            combos.append(T.apply(T.BASE[ds], keys, v))
    for norm, ep, emb in itertools.product(vals['norm'], vals['ep'], vals['emb']):   # cross of the first three stages
        combos.append(dict(T.BASE[ds], norm=norm, ep=ep, emb=emb))
    combos.append(dict(T.BASE[ds], norm='none', ep=30, emb=128, bl=2, ml=1, pt=16, mlp='1', lr='0.004', res='add'))
    combos.append(dict(T.BASE[ds], norm='mean', ep=1, emb=32, bl=1, ml=2, pt=8, mlp='2', lr='0.001', res='iand'))
    for c in combos:
        env = dict(x.split('=', 1) for x in T.line(ds, c, '42').split() if '=' in x and x.split('=', 1)[0].isupper())
        env.pop('SUITE')
        args = runner(NEW, env, ds)
        want = dict(lr=c['lr'], ml=str(c['ml']), bl=str(c['bl']), emb=str(c['emb']), pt=str(c['pt']), mlp=c['mlp'], ep=str(c['ep']))
        got = None if args is None else dict(lr=argval(args, '-lr'), ml=argval(args, '--time_layers'), bl=argval(args, '--layers'), emb=argval(args, '-emb'),
                                             pt=argval(args, '--patch_size'), mlp=argval(args, '--mlp_ratios'), ep=argval(args, '-e'))
        norm_ok = args is not None and ((argval(args, '--ad_norm') == c['norm']) if c['norm'] != 'window' else ('--ad_norm' not in args))
        cf = None if args is None else argval(args, '--connect_f')
        if args is None or argval(args, '--tag') != T.tag(c) or got != want or not norm_ok or cf != c['res'].upper():
            ok(False, f'c2 {ds} {c}: runner tag {None if args is None else argval(args, "--tag")} vs {T.tag(c)}, args {got} vs {want}, norm ok {norm_ok}, connect {cf}')
        tags[ds].add(T.tag(c))
    ok(len(tags[ds]) == len({json for json in map(lambda c: tuple(sorted(c.items())), combos)}), f'c2 {ds}: {len(tags[ds])} distinct tags for {len({tuple(sorted(c.items())) for c in combos})} distinct configurations')
ok(bad == bad0, 'c2 runner tag = adtune.tag() and runner arguments = requested settings for every candidate configuration')
# c3
spec = importlib.util.spec_from_file_location('ad_wrap_old', os.path.join(a.old_dir, 'ad_wrap.py'))
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
import ad_wrap as new                                                   # noqa: E402
from types import SimpleNamespace                                       # noqa: E402
from load_model import LOAD_MODEL                                       # noqa: E402
from spikingjelly.clock_driven import functional                        # noqa: E402
from config import Config, parse_arguments                              # noqa: E402
sys.argv = ['x', '--model', 'spkMixer', '--mixer', 'lif', '--mixer_comb', 'muxlin', '--connect_f', 'OR', '--hippo_T', '4', '--seq_len', '100',
            '--patch_size', '8', '-emb', '64', '-nh', '8', '--mlp_ratios', '1', '--time_layers', '1', '--layers', '1', '--no-bias', '-nd', '0', '--tau', '2.0']
cfg = parse_arguments(); args = Config(); args.set_args(cfg); args.num_classes, args.num_channels, args.device = 1, 25, 'cpu'
torch.manual_seed(0); r_old = old.Reconstructor(LOAD_MODEL['spkMixer'](args, True), 64, 100, 25, bias=False)
torch.manual_seed(0); r_new = new.Reconstructor(LOAD_MODEL['spkMixer'](args, True), 64, 100, 25, bias=False)
ok(all(torch.equal(p, q) for p, q in zip(r_old.state_dict().values(), r_new.state_dict().values())), 'c3 same seed -> identical weights (old / new class)')
x = torch.randn(6, 100, 25) * 2 + 3
for mode in ('train', 'eval'):
    for m in (r_old, r_new):
        (m.train() if mode == 'train' else m.eval()); m.train_mode = 'training' if mode == 'train' else 'testing'; functional.reset_net(m)
    with torch.no_grad():
        yo = r_old(x); functional.reset_net(r_old)
        yn = r_new(x); functional.reset_net(r_new)
    ok(torch.equal(yo, yn), f'c3 {mode} mode: norm window output bit-identical to the old class (max |d| {float((yo - yn).abs().max()):.1e})')
r_new.eval(); r_new.train_mode = 'testing'
xn, mu, sd = r_new.norm(x)
ok(torch.allclose(xn.mean(1), torch.zeros(6, 25), atol=1e-5) and torch.allclose(xn.std(1, unbiased=False), torch.ones(6, 25), atol=1e-3), 'c3 window: zero mean / unit std per window')
for mode in ('mean', 'none'):
    r_new.norm_mode = mode
    xn, mu, sd = r_new.norm(x)
    if mode == 'mean':
        good = torch.allclose(xn, x - x.mean(1, keepdim=True)) and torch.equal(sd, torch.ones_like(sd)) and torch.allclose(xn + mu, x)
    else:
        good = torch.equal(xn, x) and torch.equal(mu, torch.zeros_like(mu)) and torch.equal(sd, torch.ones_like(sd))
    seen = {}
    h = r_new.inner.register_forward_pre_hook(lambda mod, inp: seen.__setitem__('x', inp[0].detach().clone()))
    with torch.no_grad():
        y = r_new(x); functional.reset_net(r_new)
    h.remove()
    ok(good and torch.allclose(seen['x'], xn.permute(0, 2, 1)) if 'x' in seen else good, f'c3 {mode}: body input = ' + ('x - window mean, scale 1' if mode == 'mean' else 'x unchanged'))
r_new.norm_mode = 'window'
try:
    new.Reconstructor(LOAD_MODEL['spkMixer'](args, True), 64, 100, 25, norm='bad'); ok(False, 'c3 bad mode accepted')
except ValueError:
    ok(True, 'c3 bad norm mode refused')
print('ALL CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
sys.exit(0 if bad == 0 else 1)
