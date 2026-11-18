"""Checks of the neuron implementation / placement test (2026-11-18, PROJECT_LOG, exp/f-lif-pop-v3; user request 11-18). Fail closed: exit 1 on any failure.
  --compare A B   p0: snapshots of the earlier paths (results/check-pn-20261118/snap_pn.py; before / after the edit) are bitwise equal
  (no option)     p1 Spikformer-style with the spikeDE LIF (--neuron_all lif): every spikingjelly LIF swapped, none left; weights and random state
                     equal to the spikingjelly model at the same seed; each swapped neuron is the SpkMixer's LIF (class, tau, theta, surrogate,
                     reset, steps) and gives bitwise the same spikes / gradients as the SpkMixer's embedding LIF; finite training step, 0/1 spikes
                  p2 SpkMixer neuron_scope 'mixer': population exactly in the directional stacks + fusion, the family's single neuron in the
                     embedding / MLP; Linear / BN weights and random state equal to the 'all' and single-neuron models at the same seed
                  p3 gain tables of every grid combination: present, finite and positive, only population-layer BNs, applied by the runner's path
                  p4 f-LIF / population f-LIF settings (alpha 0.5, members tau 4 / 8 / 16 / 32, soma init 1.0 for pop_flif, 0.5 for pop_lif)
                  p5 refused combinations (model, loader, runners)
                  p6 flags, loader and folder suffixes
usage (from cinc_mixer_v1/): python -B check_pn.py [--compare A B]
"""
import argparse
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OK, BAD = [], []


def chk(name, cond, info=''):
    (OK if cond else BAD).append(name)
    print(('pass  ' if cond else 'FAIL  ') + name + (f'  [{info}]' if info else ''), flush=True)


def compare(a, b):
    A, B = torch.load(a), torch.load(b)
    chk('p0 same entries', set(A) == set(B) and len(A) > 0, f'{len(A)} / {len(B)}')
    diff = [k for k in A if k in B and not ((A[k] is None and B[k] is None) or (A[k] is not None and B[k] is not None and torch.equal(A[k], B[k])))]
    chk('p0 earlier paths bitwise equal (SpkMixer incl. 11-17 variants and pop LIF + table; Spikformer-style MIT-BIH / ETT forecaster: weights, '
        'random state, outputs, gradients; neuron spikes / charge)', not diff, f'{len(A)} entries, {len(diff)} differ {diff[:3]}')


def sf_build(task, extra=()):
    import check_ms as cm
    return cm.build(task, 'sf', False, extra=list(extra))


def p1():
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    from spikingjelly.clock_driven import functional
    from layers import NeoNeuron
    from neurons import LIFNeuron
    import stft_common as sc
    dev = torch.device('cuda')
    for task, sctask in (('mit', 'MITBIH'), ('ett', 'ETTh2')):
        a, _, ra = sf_build(task)
        b, args, rb = sf_build(task, ['--neuron_all', 'lif'])
        nsj = sum(isinstance(m, MultiStepLIFNode) for m in a.modules())
        left = sum(isinstance(m, MultiStepLIFNode) for m in b.modules())
        neo = [m.neuron for m in b.modules() if isinstance(m, NeoNeuron)]
        chk(f'p1 {task}: every spikingjelly LIF swapped, none left', nsj == 8 and b.n_swapped == nsj and left == 0 and len(neo) == nsj, f'{nsj} sj, {b.n_swapped} swapped, {left} left')
        sa, sb = a.state_dict(), b.state_dict()
        extra = [k for k in sb if k not in sa]
        chk(f'p1 {task}: weights / BN statistics equal to the spikingjelly model, random state equal',
            all(k in sb and torch.equal(sa[k], sb[k]) for k in sa) and all('neuron.' in k for k in extra) and torch.equal(ra, rb),
            f'{len(sa)} shared, {len(extra)} new neuron buffers')
        ok = all(type(n) is LIFNeuron and n.tau.numel() == 1 and float(n.tau) == 2. and n.threshold == 1. and n.surrogate_scale == 5.
                 and n.reset_mode == 'spikede' and n.max_length == 4 for n in neo)
        chk(f'p1 {task}: swapped neurons = spikeDE LIF, tau 2, theta 1, surrogate 5, reset spikede, 4 steps', ok)
        mx = sc.build(sctask, 'lif', seed=0)
        ref = mx.encoding.emb_lif.neuron
        emb = b.encoding.emb_lif.neuron
        D = int(a.encoding.embed_dim)
        x1 = (1.5 * torch.randn(4, 5, D, generator=torch.Generator().manual_seed(3))).requires_grad_(True)
        x2 = x1.detach().clone().requires_grad_(True)
        s1, s2 = ref(x1), emb(x2)
        s1.sum().backward(); s2.sum().backward()
        chk(f'p1 {task}: the Spikformer-style embedding LIF = the SpkMixer embedding LIF (spikes and input gradients bitwise)',
            torch.equal(s1, s2) and torch.equal(x1.grad, x2.grad), f'rate {float(s1.mean()):.4f}')
        b = b.to(dev); b.train()
        outs = []
        hs = [m.register_forward_hook(lambda md, i, o: outs.append(o.detach())) for m in b.modules() if isinstance(m, NeoNeuron)]
        if task == 'ett':
            from forecast_wrap import Forecaster
            f = Forecaster(b, args.embed_dim, 96, bias=args.bias).to(dev); y = f(torch.randn(8, 96, 7, generator=torch.Generator().manual_seed(4)).to(dev))
        else:
            y = b(torch.randn(8, 1, 187, generator=torch.Generator().manual_seed(4)).to(dev))
        y = y[0] if isinstance(y, tuple) else y
        y.float().pow(2).mean().backward(); functional.reset_net(b)
        for h in hs:
            h.remove()
        g = [p.grad for p in b.parameters()]
        chk(f'p1 {task}: training step finite, gradients reach every parameter, spikes 0/1',
            bool(torch.isfinite(y).all()) and all(q is not None and bool(torch.isfinite(q).all()) for q in g)
            and len(outs) == 8 and all(bool(((o == 0) | (o == 1)).all()) for o in outs), f'{len(g)} parameters, {len(outs)} neuron outputs')
        del a, b, mx; torch.cuda.empty_cache()


def p2():
    import stft_common as sc
    from layers import NeoNeuron
    from neurons import LIFNeuron, FLIFNeuron, PopulationFLIF
    POP = ('mixer.Mem', 'mixer_rev.Mem', 'sd_neuron', 'fuse.neuron')
    for task in ('ETTh2', 'MITBIH'):
        for mixer, single in (('pop_lif', LIFNeuron), ('pop_flif', FLIFNeuron)):
            m = sc.build(task, mixer, seed=42, tau_calib=False, neuron_scope='mixer'); r1 = torch.get_rng_state()
            a = sc.build(task, mixer, seed=42, tau_calib=False); r2 = torch.get_rng_state()
            s = sc.build(task, mixer.split('_', 1)[1], seed=42, tau_calib=False); r3 = torch.get_rng_state()
            kinds = {n: type(mod.neuron) for n, mod in m.named_modules() if isinstance(mod, NeoNeuron)}
            pop = [n for n in kinds if any(p in n for p in POP)]
            rest = [n for n in kinds if n not in pop]
            want_pop = 2 * sc.TASKS[task]['mixer_layers'] + 2
            chk(f'p2 {task} {mixer}: population exactly in the stacks + fusion ({want_pop}), the single neuron in embedding / MLP (3)',
                len(pop) == want_pop and len(rest) == 3 and all(kinds[n] is PopulationFLIF for n in pop) and all(kinds[n] is single for n in rest)
                and all(('encoding' in n or 'fc1' in n or 'fc2' in n) for n in rest), ', '.join(f'{n.replace("blocks.0.", "")}:{kinds[n].__name__}' for n in kinds))
            W = lambda x: {k: v for k, v in x.state_dict().items() if 'neuron' not in k and '.lif.' not in k and 'emb_lif' not in k}
            wm, wa, ws = W(m), W(a), W(s)
            chk(f'p2 {task} {mixer}: Linear / BN weights and random state equal to the all-population and the single-neuron model',
                set(wm) == set(wa) == set(ws) and all(torch.equal(wm[k], wa[k]) and torch.equal(wm[k], ws[k]) for k in wm)
                and torch.equal(r1, r2) and torch.equal(r1, r3), f'{len(wm)} tensors')
            m = m.cuda(); m.train(); m.train_mode = 'training'
            y = m(sc.batch(task, 8).cuda()); y.float().pow(2).mean().backward()
            chk(f'p2 {task} {mixer}: training step finite, gradients reach every parameter',
                bool(torch.isfinite(y).all()) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()))
            del m, a, s; torch.cuda.empty_cache()


def p3():
    import json
    import calibrate_pn as cp
    import stft_common as sc
    from layers import NeoNeuron
    from neurons import PopulationFLIF
    from spk_mixer import apply_gain_table
    n = 0
    for task in ('ETTh2', 'ETTm2', 'MITBIH'):
        for mixer in ('pop_lif', 'pop_flif'):
            for scope in ('all', 'mixer'):
                for pool in ('none', 'mean'):
                    path = cp.runner_table(task, mixer, scope, pool)
                    over = dict(**({'neuron_scope': scope} if scope != 'all' else {}), **({'mix_pool': pool} if pool != 'none' else {}))
                    m = sc.build(task, mixer, seed=0, tau_calib=False, **over)
                    popbn = {n_.rsplit('.', 1)[0] + '.' + {'emb_lif': 'emb_bn', 'lif': 'bn', 'neuron': 'bn', 'sd_neuron': 'sd_bn'}[n_.rsplit('.', 1)[1]]
                             for n_, mod in m.named_modules() if isinstance(mod, NeoNeuron) and isinstance(mod.neuron, PopulationFLIF)}
                    ok = os.path.exists(path)
                    if ok:
                        g = json.load(open(path))['gains']
                        vals = [v for x in g.values() for v in x]
                        ok = set(g) == popbn and all(v == v and 0 < v < 1e3 for v in vals) and apply_gain_table(m, path)
                    chk(f'p3 table {os.path.relpath(path, HERE)}', ok, f'{len(popbn)} population BNs')
                    n += 1
    for ds in ('ETTh2', 'ETTm2'):
        path = cp.runner_table(ds, 'pop_lif', 'mixer', 'none', 336)
        chk(f'p3 table {os.path.relpath(path, HERE)} (input 336)', os.path.exists(path) and path.endswith('_scmixer_mpnone_s336.json'))
    # the runners pick the same file (dry: parse the GT line of the runner for every combination)
    # (the file each smoke run actually used is checked from its logargs by results/check-pn-20261118/check_runs_pn.py)
    for runner, pat in (('run_ett_spk.sh', 'calib/pn/gains_${DS}_${PMX}_muxlin_${CFA}_sc${NS}_mp${MIXPOOL:-none}${STAG}.json'),
                        ('run_mitbih_spk.sh', 'calib/pn/gains_MITBIH_${PMX}_muxlin_${CFA}_sc${NS}_mp${MIXPOOL:-none}.json')):
        txt = open(os.path.join(HERE, 'scripts', runner)).read()
        chk(f'p3 {runner}: calib/pn path pattern = calibrate_pn.table_path', txt.count(pat) == 1, pat)


def p4():
    import stft_common as sc
    from layers import NeoNeuron
    from neurons import FLIFNeuron, PopulationFLIF
    for task in ('ETTh2', 'MITBIH'):
        f = sc.build(task, 'flif', seed=0)
        ns = [m.neuron for m in f.modules() if isinstance(m, NeoNeuron)]
        chk(f'p4 {task} f-LIF: every layer FLIFNeuron, alpha 0.5, tau 2', all(type(n) is FLIFNeuron and abs(float(n.alpha) - .5) < 1e-12 and float(n.tau) == 2. for n in ns), f'{len(ns)} layers')
        for mixer, soma, alpha in (('pop_flif', 1.0, .5), ('pop_lif', 0.5, 1.)):
            p = sc.build(task, mixer, seed=0, tau_calib=False)
            ps = [m.neuron for m in p.modules() if isinstance(m, NeoNeuron)]
            chk(f'p4 {task} {mixer}: members tau 4 / 8 / 16 / 32, alpha {alpha}, LIF soma tau 2, soma weights {soma}',
                all(type(n) is PopulationFLIF and [float(t) for t in n.tau] == [4., 8., 16., 32.] and abs(float(n.alpha) - alpha) < 1e-12 and n.has_soma
                    and bool((n.soma.weight == soma).all()) for n in ps), f'{len(ps)} layers')


def p5():
    import stft_common as sc
    for name, fn in (('scope mixer with lif', lambda: sc.build('ETTh2', 'lif', neuron_scope='mixer')),
                     ('scope mixer with flif', lambda: sc.build('ETTh2', 'flif', neuron_scope='mixer')),
                     ('scope mixer with band member', lambda: sc.build('ETTh2', 'pop_lif', band='member', neuron_scope='mixer')),
                     ('scope mixer with token axis', lambda: sc.build('ETTh2', 'pop_lif', taxis='token', neuron_scope='mixer')),
                     ('scope mixer with connect MS', lambda: sc.build('ETTh2', 'pop_lif', connect_f='MS', neuron_scope='mixer')),
                     ('scope mixer without a gain table', lambda: sc.build('ETTh2', 'pop_lif', neuron_scope='mixer', tau_calib=True)),
                     ('unknown scope', lambda: sc.build('ETTh2', 'pop_lif', neuron_scope='mlp', tau_calib=False)),
                     ('neuron_all with the membrane shortcut', lambda: sf_build('mit', ['--neuron_all', 'lif', '--ms_shortcut'])),
                     ('neuron_all with spkMixer', lambda: _spk_args(['--neuron_all', 'lif'])),
                     ('spk_neuron_scope with myModel', lambda: sf_build('mit', ['--spk_neuron_scope', 'mixer']))):
        try:
            fn(); refused = False
        except (ValueError, NotImplementedError, SystemExit):
            refused = True
        chk(f'p5 refused: {name}', refused)
    env0 = dict(os.environ, SUITE='check-pn-refuse', EPOCHS='1')
    for env, args, runner in (({'COND': 'spkmx_lif_muxlin_iand', 'NSCOPE': 'mixer'}, ['ETTh2', '96', '42', '9'], 'run_ett_spk.sh'),
                              ({'COND': 'spkmx_lif_muxlin_iand', 'NALL': 'lif'}, ['ETTh2', '96', '42', '9'], 'run_ett_spk.sh'),
                              ({'COND': 'spk_ssa', 'NALL': 'lif', 'MS': '1'}, ['ETTh2', '96', '42', '9'], 'run_ett_spk.sh'),
                              ({'COND': 'spkmx_pop_lif_muxlin_iand', 'MIXMS': '1'}, ['ETTh2', '96', '42', '9'], 'run_ett_spk.sh'),
                              ({'COND': 'spkmx_flif_muxlin_iand', 'MIXRESET': 'zero'}, ['ETTh2', '96', '42', '9'], 'run_ett_spk.sh'),
                              ({'COND': 'spkmx_pop_lif_muxlin_iand', 'NSCOPE': 'mixer', 'BAND': 'member'}, ['ETTh2', '96', '42', '9'], 'run_ett_spk.sh'),
                              ({'COND': 'spkmx_lif_muxlin_add', 'NSCOPE': 'mixer'}, ['42', '9'], 'run_mitbih_spk.sh'),
                              ({'COND': 'spkmx_pop_flif_muxlin_add', 'NSCOPE': 'mixer', 'MIXPOOL': 'max'}, ['42', '9'], 'run_mitbih_spk.sh')):
        p = subprocess.run(['bash', os.path.join(HERE, 'scripts', runner)] + args, env=dict(env0, **env), capture_output=True, text=True, timeout=60)
        chk(f'p5 runner refused: {runner} {env}', p.returncode != 0 and 'RESULT' not in p.stdout, (p.stdout + p.stderr).strip().splitlines()[-1][:90] if (p.stdout + p.stderr).strip() else '')


def _spk_args(extra):
    import config as cfgmod
    from config import Config
    from load_model import LOAD_MODEL
    argv = ['x', '--data', 'X', '--seed', '1', '-s', '--no-bias', '--layers', '1', '--tau', '2.0', '-nh', '8', '--log_dir', '/tmp/check_pn', '--seq_len', '96',
            '--patch_size', '8', '-emb', '64', '--mlp_ratios', '0.5', '--time_layers', '1', '--model', 'spkMixer', '--mixer', 'lif', '--mixer_comb', 'muxlin',
            '--connect_f', 'IAND', '--hippo_T', '4'] + list(extra)
    old = sys.argv; sys.argv = argv
    try:
        args = Config(); args.set_args(cfgmod.parse_arguments())
    finally:
        sys.argv = old
    args.num_classes, args.num_channels, args.device = 1, 1, torch.device('cuda')
    return LOAD_MODEL['spkMixer'](args, True), args


def p6():
    import check_ms as cm
    import load_model as lm
    from types import SimpleNamespace as NS
    _, a0, _ = sf_build('mit')
    _, a1, _ = sf_build('mit', ['--neuron_all', 'lif'])
    chk('p6 --neuron_all: folder suffix _nalif only when set', '_nalif' in a1.save_result_path and '_nalif' not in a0.save_result_path and a0.neuron_all == 'sj_lif')
    chk('p6 _neo_kwargs passes neuron_all only when set', 'neuron_all' not in lm._neo_kwargs(a0) and lm._neo_kwargs(a1)['neuron_all'] == 'lif')
    _, b0 = _spk_args([])
    chk('p6 --spk_neuron_scope default: nothing passed, no suffix', 'neuron_scope' not in lm._mixer_variants(b0) and '_ns' not in b0.save_result_path)
    b1 = NS(spk_mixpool='none', mixer_ms=False, mixer_reset='spikede', spk_neuron_scope='mixer')
    chk('p6 --spk_neuron_scope mixer: passed to the SpkMixer', lm._mixer_variants(b1) == {'neuron_scope': 'mixer'})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--compare', nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        for f in (p1, p2, p3, p4, p5, p6):
            f()
    print(f'{len(OK)} passed, {len(BAD)} failed')
    sys.exit(1 if BAD else 0)


if __name__ == '__main__':
    main()
