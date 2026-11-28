"""Checks before the MIT-BIH residual (IAND / MS lr search) and order-only (token_shuffle) runs (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG
pre-registration; user "1과 2를 진행해줘"). No training.
  --part runner  (stub python) without TSHUF the training arguments of run_mitbih_spk.sh equal the runner before this change (copy in
                 --old) for the ADD base, memoryless, SF-de PE off, IAND / MS validation runs; TSHUF=0 = unset; TSHUF=1 adds only
                 --token_shuffle and _tshuf; TSHUF=1 with spk_ssa is refused
  --part model   (CPU, the saved config of the MIT-BIH ADD base run, seed 42)
     k1 default path: the model built by the code before this change (--old copy, separate process) and by the current code at the
        same seed have bit-identical parameters and outputs (training mode forward, then eval mode forward)
     k2 token_shuffle on vs off at the same seed: identical parameters and state_dict keys; the global CPU random state after a
        training forward is identical (no global draw); the eval output differs (the shuffle acts); two eval passes, each after
        eval(), are identical; two training forwards use different orders
     k3 shuffle_tokens is a per-sample permutation of the tokens, the same for the T_h repeats
     k4 memoryless model (token_memory off, order-blind): eval logits with token_shuffle on = off (allclose 1e-5, same argmax)
     k5 the token_shuffle model survives copy.deepcopy and pickle (torch.save of the module)
     v2 (2026-11-28, after v1: at initialisation the eval-mode output did not depend on the token order, so two k2 checks failed and k4
        passed vacuously, check_model_v1_init_silent.txt): k2 - k5 load the TRAINED weights (k2, k3, k5: the ADD base seed 42
        checkpoint; k4: the memoryless seed 42 checkpoint), and k2 / k4 also require that the plain model's eval output changes when
        its input tokens are permuted (the check can see order use)
usage (from cinc_mixer_v1/): python -B results/check-tshuf-20261128/check_tshuf.py --part runner|model --old <dir: old cinc_mixer_v1 copy>
"""
import argparse
import copy
import io
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('--part', choices=['runner', 'model'], required=True)
ap.add_argument('--old', required=True)
a = ap.parse_args()
fails = 0


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}', flush=True)


def rows(path):
    return [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(path) if l.startswith('RESULT')]


if a.part == 'runner':
    tmp = tempfile.mkdtemp()
    stub = os.path.join(tmp, 'stub'); open(stub, 'w').write('#!/usr/bin/env bash\necho "ARGS $*"\nexit 3\n'); os.chmod(stub, 0o755)

    def args_of(runner, code, envs):
        res, log = tempfile.mkdtemp(), tempfile.mkdtemp()
        p = subprocess.run(['bash', runner, '42', '0'], capture_output=True, text=True, env=dict(os.environ, PY=stub, RES=res, LOG=log, SUITE='ts', **envs))
        txt = ''.join(open(os.path.join(dp, f)).read() for dp, _, fs in os.walk(log) for f in fs if f.endswith('.stdout'))
        m = re.search(r'^ARGS (.*)$', txt, re.M)
        shutil.rmtree(res); shutil.rmtree(log)
        return (m.group(1).replace(os.path.abspath(code), '<W>').replace(log, '<LOG>') if m else None), p.stdout + p.stderr

    old, new = os.path.join(a.old, 'scripts', 'run_mitbih_spk.sh'), os.path.join(C, 'scripts', 'run_mitbih_spk.sh')
    for kn in (dict(COND='spkmx_lif_muxlin_add', LR='0.02'), dict(COND='spkmx_lif_muxlin_add', LR='0.02', TMEM='off'),
               dict(COND='spk_ssa', NALL='lif', LR='0.01', SSAPE='0'), dict(COND='spkmx_lif_muxlin_iand', LR='0.005', VAL_FOLD='0'),
               dict(COND='spkmx_lif_muxlin_ms', LR='0.02', VAL_FOLD='0'), dict(COND='spkmx_lif_muxlin_ms', LR='0.0025')):
        o, _ = args_of(old, a.old, kn); n, _ = args_of(new, C, kn)
        check(f'{kn}: same arguments without TSHUF', o is not None and o == n, '' if o == n else f'\n  {o}\n  {n}')
        n0, _ = args_of(new, C, dict(kn, TSHUF='0'))
        check(f'{kn}: TSHUF=0 = unset', n0 == n)
        if kn['COND'].startswith('spkmx_lif_'):
            n1, _ = args_of(new, C, dict(kn, TSHUF='1'))
            ok = n1 is not None and n1.count(' --token_shuffle') == 1 and n1.count('_tshuf') == 1 and \
                n1.replace(' --token_shuffle', '', 1).replace('_tshuf', '', 1) == n
            check(f'{kn}: TSHUF=1 adds only --token_shuffle and _tshuf', ok, '' if ok else f'\n  {n}\n  {n1}')
        else:
            n1, out = args_of(new, C, dict(kn, TSHUF='1'))
            check(f'{kn}: TSHUF=1 refused for spk_ssa', n1 is None and 'LIF SpkMixer only' in out)
    shutil.rmtree(tmp)

else:
    import torch
    base = [r for r in rows(os.path.join(C, 'results', 'flif-v1-mitbih-spkmux-tune', 'raw.txt')) if r['tag'] == 'spkmx_lif_muxlin_add_lr0.02_L2' and r['seed'] == '42']
    check('source run (MIT-BIH ADD base, seed 42)', len(base) == 1)
    cfgp = os.path.join(base[0]['dir'], 'model_state', 'config.pt')
    tmp = tempfile.mkdtemp()
    BUILD = r'''
import os, sys, torch
from types import SimpleNamespace
code, cfgp, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, code); os.chdir(code)
from load_model import LOAD_MODEL
from spikingjelly.clock_driven import functional
args = SimpleNamespace(**torch.load(cfgp, map_location='cpu')); args.device = 'cpu'
torch.manual_seed(0)
m = LOAD_MODEL[args.model](args, True)
g = torch.Generator().manual_seed(5)
x = torch.randn(8, 187, generator=g)
m.train(); functional.reset_net(m); y_tr = m(x); functional.reset_net(m)
m.eval(); m.train_mode = 'testing'
with torch.no_grad():
    y_ev = m(x)[0]
functional.reset_net(m)
torch.save(dict(params={k: v.detach().clone() for k, v in m.state_dict().items()}, y_tr=y_tr.detach(), y_ev=y_ev, rng=torch.get_rng_state()), out)
'''
    bp = os.path.join(tmp, 'build.py'); open(bp, 'w').write(BUILD)
    res = {}
    for nm, code in (('old', a.old), ('new', C)):
        p = subprocess.run([sys.executable, '-B', '-W', 'ignore', bp, code, cfgp, os.path.join(tmp, nm + '.pt')], capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-2000:])
        res[nm] = torch.load(os.path.join(tmp, nm + '.pt')) if p.returncode == 0 else None
    o, n = res['old'], res['new']
    ok = o is not None and n is not None and o['params'].keys() == n['params'].keys() and all(torch.equal(o['params'][k], n['params'][k]) for k in o['params'])
    check(f'k1 default path: parameters bit-identical old vs new ({len(n["params"]) if n else 0} tensors)', ok)
    check('k1 default path: training-mode and eval-mode outputs bit-identical old vs new',
          ok and torch.equal(o['y_tr'], n['y_tr']) and torch.equal(o['y_ev'], n['y_ev']) and torch.equal(o['rng'], n['rng']))
    shutil.rmtree(tmp)

    sys.path.insert(0, C); os.chdir(C)
    from types import SimpleNamespace
    from load_model import LOAD_MODEL
    from spikingjelly.clock_driven import functional

    def build(ckdir=None, **over):
        args = SimpleNamespace(**torch.load(cfgp if ckdir is None else os.path.join(ckdir, 'model_state', 'config.pt'), map_location='cpu'))
        args.device = 'cpu'
        for k, v in over.items():
            setattr(args, k, v)
        torch.manual_seed(0)
        m = LOAD_MODEL[args.model](args, True)
        if ckdir is not None:                                     # v2: trained weights
            m.load_state_dict(torch.load(os.path.join(ckdir, 'model_state', 'best+model.pt'), map_location='cpu'))
        return m

    off = [r for r in rows(os.path.join(C, 'results', 'flif-v1-mitbih-memoff', 'raw.txt')) if r['seed'] == '42']
    check('source run (MIT-BIH memoryless, seed 42)', len(off) == 1)
    BD, OD = base[0]['dir'], off[0]['dir']

    def permuted(m, x, train=False):                              # the plain model on input tokens permuted per sample
        tok_fn = lambda t: torch.gather(t, 2, torch.argsort(torch.rand(t.shape[1], t.shape[2], generator=torch.Generator().manual_seed(9)), 1)
                                        .view(1, t.shape[1], t.shape[2], 1).expand_as(t))
        m.train(train); m.train_mode = 'testing'
        functional.reset_net(m)
        with torch.no_grad():
            y = m.forward_parts(x, tok_fn=tok_fn)[0]
        functional.reset_net(m)
        return y

    def fwd(m, x, train):
        m.train(train)
        m.train_mode = 'training' if train else 'testing'
        functional.reset_net(m)
        with torch.set_grad_enabled(train):
            y = m(x)
        functional.reset_net(m)
        y = y[0] if isinstance(y, tuple) else y
        return y.detach()

    import pandas as pd
    df = pd.read_csv(os.path.join(torch.load(cfgp, map_location='cpu')['root_path'], '5class_mitbih_test_3_187.csv'), header=None, nrows=64)
    x = torch.tensor(df.values[:, :187], dtype=torch.float32)     # v2: real test beats (the first 64 rows)
    m0, m1 = build(BD, token_shuffle=False), build(BD, token_shuffle=True)
    check('k2 token_shuffle on: model has the option, off has not', m1.token_shuffle and not getattr(m0, 'token_shuffle', False))
    s0, s1 = m0.state_dict(), m1.state_dict()
    check('k2 identical parameters and state_dict keys at the same seed', s0.keys() == s1.keys() and all(torch.equal(s0[k], s1[k]) for k in s0))
    torch.manual_seed(123); fwd(m0, x, True); r0 = torch.get_rng_state()
    torch.manual_seed(123); fwd(m1, x, True); r1 = torch.get_rng_state()
    check('k2 no global random draw: CPU random state after a training forward equal on / off', torch.equal(r0, r1))
    e0 = fwd(m0, x, False)
    check(f'k2 the trained plain model sees order (max |logit change| when its tokens are permuted {float((permuted(m0, x) - e0).abs().max()):.3f})',
          float((permuted(m0, x) - e0).abs().max()) > 1e-3)
    e1a = fwd(m1, x, False)                                       # fwd calls train(False) -> eval generator re-seeded
    e1b = fwd(m1, x, False)
    check('k2 eval output with the shuffle differs from without (the shuffle acts)', not torch.equal(e0, e1a))
    check('k2 two eval passes, each after eval(), identical', torch.equal(e1a, e1b))
    m1.eval(); m1.train_mode = 'testing'
    with torch.no_grad():
        functional.reset_net(m1); y1 = m1(x)[0]; functional.reset_net(m1); y2 = m1(x)[0]; functional.reset_net(m1)
    check('k2 within one eval pass consecutive batches get new orders (outputs of the same batch twice differ)', not torch.equal(y1, y2))
    t1, t2 = fwd(m1, x, True), fwd(m1, x, True)
    check('k2 two training forwards use different orders', not torch.equal(t1, t2))
    tok = m1.tokens(x)
    m1.train(False)
    sh = m1.shuffle_tokens(tok)
    perm_ok = torch.equal(torch.sort(sh, dim=2).values, torch.sort(tok, dim=2).values)
    same_rep = all(torch.equal(sh[i], sh[0]) for i in range(sh.shape[0]))
    moved = float((sh != tok).any(-1).float().mean())
    check(f'k3 per-sample token permutation, same for the T_h repeats (tokens moved {moved:.2f})', perm_ok and same_rep and moved > 0.5)
    mo0, mo1 = build(OD, token_memory='off', token_shuffle=False), build(OD, token_memory='off', token_shuffle=True)
    z0, z1 = fwd(mo0, x, False), fwd(mo1, x, False)
    check(f'k4 the trained memoryless model fires (logit spread {float(z0.std()):.3f})', float(z0.std()) > 1e-3)
    check(f'k4 memoryless (order-blind) model: shuffle on = off (max |diff| {float((z0 - z1).abs().max()):.2e}, argmax equal)',
          torch.allclose(z0, z1, rtol=1e-5, atol=1e-5) and torch.equal(z0.argmax(1), z1.argmax(1)))
    try:
        # v3 (v2 compared the copy with e1a, taken before the training forwards above changed the BN running statistics,
        # check_model_v2_stale_k5.txt): the reference eval output is taken right before the copy
        ref = fwd(m1, x, False)
        mc = copy.deepcopy(m1); buf = io.BytesIO(); torch.save(m1, buf); pickle.dumps(m1)
        ec = fwd(mc, x, False)
        check('k5 deepcopy / torch.save / pickle of the shuffle model work, the copy gives the same eval output', torch.equal(ec, ref))
    except Exception as ex:
        check(f'k5 deepcopy / pickle ({ex})', False)
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
