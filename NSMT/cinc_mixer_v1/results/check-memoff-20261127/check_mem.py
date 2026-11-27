"""Checks before the memoryless control (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG memoryless-control pre-registration). No training.
  m0 earlier conditions unchanged: the code before this change (`--ref`) and this code build 20 earlier conditions / arms with bitwise
     equal parameters / buffers and train / eval outputs (m0_build.py, a copy of check-sxa4's s4_build.py, in two processes)
  m1 the 'memoryless' LIF equals the hand recursion H_t = I_t / tau, s_t = [H_t >= theta] (no state carried), spikes and H - theta
  m2 no token-axis flow inside the mixer: with token_memory off, changing one token's mixer input changes the mixer output m at that
     token only (eval-mode BN warmed on inputs of the test distribution, exact); with token_memory on, other tokens change too; the
     mixer must fire (v1 of this check warmed BN on model inputs, eval mode did not fire and the check failed vacuously: kept as
     check_mem_v1_silent.txt)
  m3 the same seed gives the same parameters with token_memory on and off
  m4 runner: without TMEM the training arguments are the same as before (stub python); TMEM=off adds only --token_memory off and _tmoff
usage (from cinc_mixer_v1/): python -B results/check-memoff-20261127/check_mem.py --ref <copy of cinc_mixer_v1 before the change>
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, C)
os.chdir(C)
ap = argparse.ArgumentParser()
ap.add_argument('--ref', required=True)
a = ap.parse_args()
import torch                                                # noqa: E402
PY = sys.executable
fails = 0


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}', flush=True)


tmp = tempfile.mkdtemp(prefix='m0_')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4')
R = {}
for tag, code in (('ref', a.ref), ('new', C)):
    p = os.path.join(tmp, f'{tag}.pt')
    r = subprocess.run([PY, '-B', '-W', 'ignore', os.path.join(HERE, 'm0_build.py'), code, p], capture_output=True, text=True, env=env)
    check(f'm0 build ({tag})', r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else '')
    R[tag] = torch.load(p) if r.returncode == 0 else None
if R['ref'] and R['new']:
    same_all = all(list(R['ref'][k]['state']) == list(R['new'][k]['state']) and all(torch.equal(R['ref'][k]['state'][q], R['new'][k]['state'][q]) for q in R['ref'][k]['state'])
                   and torch.equal(R['ref'][k]['ytr'], R['new'][k]['ytr']) and torch.equal(R['ref'][k]['yev'], R['new'][k]['yev']) for k in R['ref'])
    check(f'm0 {len(R["ref"])} earlier conditions / arms bitwise equal', same_all)
from layers import build_neo_neuron                        # noqa: E402
n = build_neo_neuron('lif', 6, 9, 'ND', tau=2.0, reset='memoryless')
cur = torch.randn(9, 4, 1, 6) * 2.
with torch.no_grad():
    s, h = n.forward_charge(cur)
check('m1 memoryless LIF: spikes = [I / tau >= theta], H - theta = I / tau - theta (no carried state)',
      torch.equal(s, (cur / 2. - 1. >= 0).float()) and torch.allclose(h, cur / 2. - 1., atol=1e-6) and 0 < float(s.mean()) < 1)
from forecast_wrap import Forecaster                        # noqa: E402
from spk_mixer import SpkMixer                              # noqa: E402
from spikingjelly.clock_driven import functional            # noqa: E402


def build(tm, comb='muxlin', seed=42):
    torch.manual_seed(seed)
    inner = SpkMixer(seq_len=96, num_classes=1, depths=1, mixer='lif', T_h=4, neo_tau=2.0, bias=False, tau=2.0, mixer_comb=comb,
                     connect_f='IAND', data_patch_size=16, embed_dim=32, mlp_ratios=.5, mixer_layers=2, token_memory=tm)
    return Forecaster(inner, 32, 96, bias=False)


for comb in ('muxlin', 'msxdpca'):
    res = {}
    for tm in ('on', 'off'):
        mdl = build(tm, comb).train()
        blk = mdl.inner.blocks[0]
        g = torch.Generator().manual_seed(11)
        with torch.no_grad():                                   # BN running statistics from inputs of the test distribution (v1 warmed
            for _ in range(150):                                # the BN with model inputs instead, and eval mode did not fire: vacuous)
                blk.mix((torch.rand(4, 6, 12, 32, generator=g) < 0.3).float())
        mdl.eval()
        torch.manual_seed(5)
        x = (torch.rand(4, 6, 12, 32) < 0.3).float()           # [T, B', N, D] embedding spikes
        x2 = x.clone(); x2[:, :, 5] = 1. - x2[:, :, 5]          # change token 5 only
        with torch.no_grad():
            m1, m2 = blk.mix(x), blk.mix(x2)
        diff = (m1 != m2).any(-1).any(0).any(0)                 # [N] tokens whose output changed
        res[tm] = (diff, float(m1.mean()))
    off, on = res['off'][0], res['on'][0]
    if comb == 'muxlin':                                        # neuron state is the only token-axis path of muxlin
        ok = res['off'][1] > 0 and res['on'][1] > 0 and bool(off[5]) and not bool(off[torch.arange(12) != 5].any()) and bool(on[torch.arange(12) != 5].any())
        lab = 'token_memory off -> only token 5 changes; on -> other tokens change too (mixer fires)'
    else:                                                       # the attention mixes tokens whatever the neuron memory (v2 expected
        ok = res['off'][1] > 0 and bool(off[torch.arange(12) != 5].any())   # otherwise: wrong expectation, check_mem_v2_wrong_expectation.txt)
        lab = 'token_memory off -> other tokens still change through the attention (expected; the control is for muxlin)'
    check(f'm2 {comb}: {lab}', ok,
          f'off changed {off.nonzero().flatten().tolist()}, on changed {on.nonzero().flatten().tolist()}, mixer rate off {res["off"][1]:.3f} on {res["on"][1]:.3f}')
    P0, P1 = dict(build('on', comb).named_parameters()), dict(build('off', comb).named_parameters())
    check(f'm3 {comb}: same parameters with token_memory on / off at the same seed', list(P0) == list(P1) and all(torch.equal(P0[k], P1[k]) for k in P0))
stub = os.path.join(tmp, 'stubpy')
open(stub, 'w').write('#!/usr/bin/env bash\necho "ARGS $*"\nexit 3\n')
os.chmod(stub, 0o755)


def runner_args(runner, code_dir, envs, ds):
    res_, log = tempfile.mkdtemp(prefix='res_'), tempfile.mkdtemp(prefix='log_')
    subprocess.run(['bash', runner, ds, '96', '42', '0'], capture_output=True, text=True, env=dict(os.environ, PY=stub, RES=res_, LOG=log, SUITE='m4', **envs))
    txt = ''.join(open(os.path.join(log, f)).read() for f in os.listdir(log) if f.endswith('.stdout'))
    m = re.search(r'^ARGS (.*)$', txt, re.M)
    shutil.rmtree(res_); shutil.rmtree(log)
    return m.group(1).replace(os.path.abspath(code_dir), '<W>').replace(log, '<LOG>') if m else None


KN = [('ETTh1', dict(COND='spkmx_lif_muxlin_iand', LR='0.00025', LAYERS='1', EMB='32', MLPR='4', WD='0.1')),
      ('ETTm1', dict(COND='spkmx_lif_muxlin_iand', LR='0.0005', LAYERS='2', PATCH='16', EMB='32', MLPR='0.5')),
      ('ETTm1', dict(COND='spkmx_lif_msxdpca_iand', LR='0.0005', LAYERS='2', PATCH='16', EMB='32', MLPR='0.5', TH='8')),
      ('ETTh2', dict(COND='spk_ssa', LR='0.0005', PATCH='16'))]
for ds, kn in KN:
    o = runner_args(os.path.join(a.ref, 'scripts', 'run_ett_spk.sh'), a.ref, kn, ds)
    nn_ = runner_args(os.path.join(C, 'scripts', 'run_ett_spk.sh'), C, kn, ds)
    check(f'm4 {kn["COND"]} {ds}: same training arguments without TMEM', o is not None and o == nn_, '' if o == nn_ else f'\n  {o}\n  {nn_}')
n0 = runner_args(os.path.join(C, 'scripts', 'run_ett_spk.sh'), C, KN[0][1], 'ETTh1')
n1 = runner_args(os.path.join(C, 'scripts', 'run_ett_spk.sh'), C, dict(KN[0][1], TMEM='off'), 'ETTh1')
ok = n0 is not None and n1 is not None and n1.replace(' --token_memory off', '', 1).replace('_tmoff', '', 1) == n0
check('m4 TMEM=off adds only --token_memory off and _tmoff', ok, '' if ok else f'\n  {n0}\n  {n1}')
shutil.rmtree(tmp)
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
