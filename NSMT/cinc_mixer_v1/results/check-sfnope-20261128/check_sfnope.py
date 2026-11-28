"""Checks before the SF-de position-information-off runs (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration). No training.
  --part runner  (stub python that prints its arguments) without SSAPE the training arguments of run_mitbih_spk.sh / run_ett_spk.sh
                 equal those of the runners before this change (copies in --old) for SF-de and SpkMixer conditions; SSAPE=1 = unset;
                 SSAPE=0 adds only --ssa_pe 0 and the tag suffix _pe0; SSAPE=0 with a SpkMixer condition is refused
  --part model   SF-de rebuilt from the saved config of an existing MIT-BIH and ETTh1 SF-de run: with ssa_pe 0 the SSA has no
                 relative_bias_table, the parameter count drops by exactly (2N - 1) x heads, and the SSA is permutation-equivariant
                 along the token axis (per-row random permutation of its input, output put back: the same spikes) in training mode
                 (batch BN statistics) and in eval mode, with spikes present; the same test with ssa_pe 1 and a random bias table
                 must FAIL (the test can detect order use)
                 v2 (2026-11-28, after v1 was vacuous: at initialisation the SSA output neuron never fired, rate 0.000, see
                 check_model_v1_vacuous.txt): the weights of the trained PE-on checkpoint (everything but relative_bias_table) are
                 loaded into the PE-off model, so the SSA fires; the trained PE-on model itself (its learned bias table) must be
                 detected as order-using
  --part smoke   the 1-epoch smoke runs (suites check-sfnope-smoke-mitbih / -ett): RESULT line, tag _pe0, saved config ssa_pe 0,
                 run folder contains _nope, params = PE-on params - (2N - 1) x heads
usage (from cinc_mixer_v1/): python -B results/check-sfnope-20261128/check_sfnope.py --part runner --old <dir with scripts/ copies>
                             CUDA_VISIBLE_DEVICES=<g> python -B results/check-sfnope-20261128/check_sfnope.py --part model | smoke
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('--part', choices=['runner', 'model', 'smoke'], required=True)
ap.add_argument('--old', default='')
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

    def args_of(runner, code, pos, envs):
        res, log = tempfile.mkdtemp(), tempfile.mkdtemp()
        p = subprocess.run(['bash', runner] + pos, capture_output=True, text=True, env=dict(os.environ, PY=stub, RES=res, LOG=log, SUITE='sn', **envs))
        txt = ''.join(open(os.path.join(dp, f)).read() for dp, _, fs in os.walk(log) for f in fs if f.endswith('.stdout'))
        m = re.search(r'^ARGS (.*)$', txt, re.M)
        shutil.rmtree(res); shutil.rmtree(log)
        return (m.group(1).replace(os.path.abspath(code), '<W>').replace(log, '<LOG>') if m else None), p.stdout + p.stderr

    for rn, pos, sets in [
        ('run_mitbih_spk.sh', ['42', '0'], [dict(COND='spk_ssa', NALL='lif', LR='0.01'), dict(COND='spkmx_lif_muxlin_add', LR='0.02'),
                                             dict(COND='spkmx_lif_muxlin_add', LR='0.02', TMEM='off')]),
        ('run_ett_spk.sh', ['ETTh1', '96', '42', '0'], [dict(COND='spk_ssa', NALL='lif', LR='0.0005'),
                                                         dict(COND='spkmx_lif_muxlin_iand', LR='0.00025', EMB='32', MLPR='4', WD='0.1')]),
        ('run_ett_spk.sh', ['ETTm1', '720', '7', '0'], [dict(COND='spk_ssa', NALL='lif', LR='0.0001', PATCH='16'),
                                                         dict(COND='spkmx_lif_muxlin_iand', LR='0.0005', LAYERS='2', PATCH='16', EMB='32', MLPR='0.5')]),
        ('run_ett_spk.sh', ['ETTh2', '96', '42', '0'], [dict(COND='spk_ssa'), dict(COND='spkmx_lif_muxlin_iand')])]:
        old, new = os.path.join(a.old, 'scripts', rn), os.path.join(C, 'scripts', rn)
        for kn in sets:
            o, _ = args_of(old, a.old, pos, kn); n, _ = args_of(new, C, pos, kn)
            check(f'{rn} {pos[0]} {kn}: same arguments without SSAPE', o is not None and o == n, '' if o == n else f'\n  {o}\n  {n}')
            n1, _ = args_of(new, C, pos, dict(kn, SSAPE='1'))
            check(f'{rn} {pos[0]} {kn}: SSAPE=1 = unset', n1 == n)
            if kn['COND'] == 'spk_ssa':
                n0, _ = args_of(new, C, pos, dict(kn, SSAPE='0'))
                ok = n0 is not None and n is not None and n0.count(' --ssa_pe 0') == 1 and n0.count('_pe0') == 1 and \
                    n0.replace(' --ssa_pe 0', '', 1).replace('_pe0', '', 1) == n
                check(f'{rn} {pos[0]} {kn}: SSAPE=0 adds only --ssa_pe 0 and _pe0', ok, '' if ok else f'\n  {n}\n  {n0}')
            else:
                n0, out = args_of(new, C, pos, dict(kn, SSAPE='0'))
                check(f'{rn} {pos[0]} {kn}: SSAPE=0 refused for the SpkMixer', n0 is None and 'spk_ssa only' in out)
    shutil.rmtree(tmp)

elif a.part == 'model':
    import torch
    sys.path.insert(0, C); os.chdir(C)
    from spikingjelly.clock_driven import functional
    from load_model import LOAD_MODEL
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    def take(x, idx):                                             # x [T, B, N, D], idx [B, N]: per-row permutation of the tokens
        return torch.gather(x, 2, idx.view(1, *idx.shape, 1).expand_as(x))

    STAGES = ('q_lif', 'k_lif', 'v_lif', 'attn_lif')

    def build(cfg_path, pe):
        args = SimpleNamespace(**torch.load(cfg_path, map_location='cpu')); args.device = dev; args.ssa_pe = pe
        torch.manual_seed(0)
        return LOAD_MODEL[args.model](args, True).to(dev)

    def equi(model, train, dens=0.3):
        ssa = model.data_block[0].ssa
        model.train(train)
        N, D = ssa.seq_len, ssa.q_linear.in_features
        g = torch.Generator().manual_seed(1)
        x = (torch.rand(4, 64, N, D, generator=g) < dens).float().to(dev)
        p = torch.argsort(torch.rand(64, N, generator=g), dim=1).to(dev); ip = torch.argsort(p, dim=1)
        cap, hs = {}, []
        for nm in STAGES:                                         # v4: every neuron inside the SSA, outputs [T, B, N, C]
            hs.append(getattr(ssa, nm).register_forward_hook(lambda mod, i, o, nm=nm: cap.setdefault(nm, []).append(o.reshape(x.shape))))
        try:
            with torch.no_grad():
                functional.reset_net(model); y0 = ssa(x)
                functional.reset_net(model); y1 = take(ssa(take(x, p)), ip)
                functional.reset_net(model)
        finally:
            for h in hs:
                h.remove()
        st = {nm: (float((cap[nm][0] != take(cap[nm][1], ip)).float().mean()), float(cap[nm][0].float().mean())) for nm in STAGES}
        st['output'] = (float((y0 != y1).float().mean()), float(y0.float().mean()))
        return st

    for task, suite, pick in [('MIT-BIH', 'flif-v1-mitbih-sfde', lambda r: r['tag'] == 'spk_ssa_lr0.01_L2_nalif' and r['seed'] == '42'),
                              ('ETTh1', 'flif-v1-ett-sfde-h1m1', lambda r: r['ds'] == 'ETTh1' and r['pl'] == '96' and r['seed'] == '42')]:
        r = [x for x in rows(os.path.join(C, 'results', suite, 'raw.txt')) if pick(x)]
        check(f'{task}: one source run', len(r) == 1)
        cp = os.path.join(r[0]['dir'], 'model_state', 'config.pt')
        m1, m0 = build(cp, 1), build(cp, 0)
        ck = torch.load(os.path.join(r[0]['dir'], 'model_state', 'best+model.pt' if task == 'MIT-BIH' else 'checkpoint.pth'), map_location='cpu')
        if task != 'MIT-BIH':                                     # Forecaster state: the backbone keys carry the prefix 'inner.'
            ck = {k[len('inner.'):]: v for k, v in ck.items() if k.startswith('inner.')}
        # allowed differences only: the ETT checkpoint has no inner classification head (the Forecaster replaces it, head.*), and the
        # PE-off model has no rel_pos_idx buffer (registered only with the bias table)
        miss1, unexp1 = m1.load_state_dict(ck, strict=False)
        miss0, unexp0 = m0.load_state_dict({k: v for k, v in ck.items() if 'relative_bias_table' not in k}, strict=False)
        hd = lambda ks: all(k.startswith('head.') for k in ks) and (task != 'MIT-BIH' or not ks)
        check(f'{task}: trained weights loaded (PE on: missing {miss1}, unexpected {unexp1}; PE off: missing {miss0}, unexpected {unexp0})',
              hd(miss1) and not unexp1 and hd(miss0) and all(k.endswith('ssa.rel_pos_idx') for k in unexp0) and len(unexp0) >= 1)
        s0, s1 = m0.data_block[0].ssa, m1.data_block[0].ssa
        N, H = s1.seq_len, s1.num_heads
        p1, p0 = sum(t.numel() for t in m1.parameters()), sum(t.numel() for t in m0.parameters())
        check(f'{task}: ssa_pe 0 has no relative_bias_table (pe {s0.pe}), ssa_pe 1 has one', not hasattr(s0, 'relative_bias_table') and hasattr(s1, 'relative_bias_table'))
        check(f'{task}: parameters {p1} - {p0} = (2N - 1) x heads = {(2 * N - 1) * H} (N {N}, heads {H})', p1 - p0 == (2 * N - 1) * H)
        check(f'{task}: every SSA in the model is off', all(not b.ssa.pe for b in m0.data_block if hasattr(b, 'ssa')))
        # v3: input spike densities 0.3 / 0.6 / 0.9 (v2: ETTh1 train silent at 0.3). v4 (v3: the ETTh1 output neuron stays silent in
        # train mode at every density): every neuron inside the SSA is compared too; pass = no mismatch at any stage and density, the
        # q / k / v neurons (right after the BN, the only part where train mode differs) fire, and in eval mode the output fires
        for train in (True, False):
            res = {dn: equi(m0, train, dn) for dn in (0.3, 0.6, 0.9)}
            # v5 (v4: MIT-BIH train mode, density 0.9, one q spike of 1.08 M differs, 9e-7 -- the batch-BN sums run in another order
            # after the permutation, a rounding flip at the threshold): train mode allows a mismatch fraction <= 1e-5 per stage
            # (100x below the 1e-3 the order-use detection needs); eval mode (no batch statistics) stays exact
            tol = 1e-5 if train else 0.
            ok = all(mi <= tol for st in res.values() for mi, _ in st.values()) and \
                all(any(res[dn][nm][1] > 0. for dn in res) for nm in ('q_lif', 'k_lif', 'v_lif')) and \
                (train or any(res[dn]['output'][1] > 0. for dn in res))
            check(f'{task} PE off, {"train" if train else "eval"} mode: permuted SSA = SSA at every stage and density', ok,
                  '; '.join(f'density {dn}: ' + ', '.join(f'{nm} {mi:.0e}/{ra:.3f}' for nm, (mi, ra) in st.items()) for dn, st in res.items())
                  + ' (stage mismatch / spike rate)')
        mis, rate = equi(m1, False)['output']
        check(f'{task} trained PE on (learned bias table), eval mode: the test detects order use (mismatch {mis:.2e}, rate {rate:.3f})', mis > 1e-3)

else:
    import torch
    for task, suite, tag_old, src in [('MIT-BIH', 'check-sfnope-smoke-mitbih', 'spk_ssa_lr0.01_L2_nalif', ('flif-v1-mitbih-sfde', None)),
                                      ('ETTh1 H96', 'check-sfnope-smoke-ett', 'spk_ssa_lr0.0005_L1_nalif', ('flif-v1-ett-sfde-h1m1', 'ETTh1'))]:
        p = os.path.join(C, 'results', suite, 'raw.txt')
        lines = open(p).read().splitlines() if os.path.exists(p) else []
        rr = rows(p) if lines else []
        check(f'{task} smoke: one RESULT, no SKIP', len(rr) == 1 and not any(l.startswith('SKIP') for l in lines))
        if len(rr) != 1:
            continue
        r = rr[0]
        cfg = torch.load(os.path.join(r['dir'], 'model_state', 'config.pt'), map_location='cpu')
        check(f'{task} smoke: tag {r["tag"]}', r['tag'] == tag_old + '_pe0')
        check(f'{task} smoke: saved config ssa_pe {cfg.get("ssa_pe")}', int(cfg.get('ssa_pe', 1)) == 0)
        check(f'{task} smoke: run folder has _nope', '_nope' in r['dir'])
        ref = [x for x in rows(os.path.join(C, 'results', src[0], 'raw.txt')) if x['tag'] == tag_old and (src[1] is None or x['ds'] == src[1])
               and (src[1] is None or x['pl'] == '96')]
        sys.path.insert(0, C); os.chdir(C)
        from load_model import LOAD_MODEL
        args = SimpleNamespace(**cfg); args.device = 'cpu'
        ssa = LOAD_MODEL[args.model](args, True).data_block[0].ssa
        want = (2 * ssa.seq_len - 1) * ssa.num_heads
        d = int(ref[0]['params']) - int(r['params'])
        check(f'{task} smoke: params {r["params"]} = PE-on params {ref[0]["params"]} - (2N - 1) x heads ({want}; N {ssa.seq_len})',
              len({x['params'] for x in ref}) == 1 and d == want)
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
