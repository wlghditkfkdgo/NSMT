"""Checks before stage 1 of the fusion redesign (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG pre-registration). No training.
  k0a earlier conditions unchanged: an old copy of the code (HEAD f2df15b95, `--old`) and the worktree build the forecaster for
      muxlin IAND / ADD, or IAND, sdlin OR, muxlin IAND 3 blocks at one seed; every parameter / buffer and the outputs in train and
      eval mode are bitwise equal (k0_build.py in two processes)
  k0b runner unchanged: run_ett_spk.sh (old copy and worktree) with a stub python that prints its arguments; for earlier conditions
      and knob sets the training arguments are the same text (after replacing the two code directories); WSHARE=1 adds only
      --mixer_share and the tag suffix _ws
  k2  random pairing: at one seed, the 10 stage-1 arms (muxlin / sxd / sxdpc / sxp / sxppc x independent / shared, ADD) share every
      common parameter bitwise with arm 1 (muxlin independent): embedding, forward stack, MLP, head (and the reverse stack for the
      independent arms); shared arms have no reverse stack; parameter counts differ by exactly the reverse stack
  k3  sharing: with shared weights the reverse first-layer current equals the flipped forward current (eval-mode BN: exactly; train
      mode: within 1e-6, the BN reduction order differs); backward gives gradients to the shared stack from both directions
      (each direction alone gives a non-zero, different gradient and the joint gradient is their sum within 1e-6)
  k4  arithmetic: AttnMath equals a loop / numpy recomputation (A_s = Q K^T per (b, t), Pearson over D with numpy.corrcoef on the
      T-mean charges, (1 + rho) / 2 in [0, 1], Z = (A_s * M) V / sqrt(D)); in train mode the forward / reverse spikes, the attention
      output and the projections fire (non-vacuous)
  k5  ledger: AttnMath._ops equals the hand formula; syops with the AttnMath hook minus syops without it equals the ledger per
      window (AC and MAC); energy.SynOpRecorder adds the same ledger
usage (from cinc_mixer_v1/): python -B results/check-sxa-20261127/check_sxa.py --old <dir with the HEAD copy of cinc_mixer_v1>
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, C)
os.chdir(C)
ap = argparse.ArgumentParser()
ap.add_argument('--old', required=True)
a = ap.parse_args()
PY = sys.executable
fails = 0


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}', flush=True)


# ---- k0a -------------------------------------------------------------------------------------------------------------------
tmp = tempfile.mkdtemp(prefix='k0_')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4')
outs = {}
for tag, code in (('old', os.path.abspath(a.old)), ('new', C)):
    p = os.path.join(tmp, f'{tag}.pt')
    r = subprocess.run([PY, '-B', '-W', 'ignore', os.path.join(HERE, 'k0_build.py'), code, p], capture_output=True, text=True, env=env)
    check(f'k0a build in the {tag} code', r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else '')
    outs[tag] = torch.load(p) if r.returncode == 0 else None
if outs['old'] is not None and outs['new'] is not None:
    for k in outs['old']:
        o, n = outs['old'][k], outs['new'][k]
        same_keys = list(o['state']) == list(n['state'])
        same_vals = same_keys and all(torch.equal(o['state'][q], n['state'][q]) for q in o['state'])
        check(f'k0a {k}: same parameters / buffers ({len(o["state"])}) and outputs (train / eval) bitwise',
              same_vals and torch.equal(o['ytr'], n['ytr']) and torch.equal(o['yev'], n['yev']))

# ---- k0b -------------------------------------------------------------------------------------------------------------------
stub = os.path.join(tmp, 'stubpy')
open(stub, 'w').write('#!/usr/bin/env bash\necho "ARGS $*"\nexit 3\n')
os.chmod(stub, 0o755)


def runner_args(runner, code_dir, envs, ds='ETTm1', pl='96', seed='42'):
    res, log = tempfile.mkdtemp(prefix='res_'), tempfile.mkdtemp(prefix='log_')
    e = dict(os.environ, PY=stub, RES=res, LOG=log, SUITE='k0b', **envs)
    subprocess.run(['bash', runner, ds, pl, seed, '0'], capture_output=True, text=True, env=e)
    txt = ''
    for f in os.listdir(log):
        if f.endswith('.stdout'):
            txt = open(os.path.join(log, f)).read()
    m = re.search(r'^ARGS (.*)$', txt, re.M)
    shutil.rmtree(res); shutil.rmtree(log)
    if not m:
        return None
    return m.group(1).replace(os.path.abspath(code_dir), '<W>').replace(log, '<LOG>')


old_runner = os.path.join(os.path.abspath(a.old), 'scripts', 'run_ett_spk.sh')
new_runner = os.path.join(C, 'scripts', 'run_ett_spk.sh')
KN = [dict(COND='spkmx_lif_muxlin_iand', LR='0.0005', LAYERS='2', PATCH='16', EMB='32', MLPR='0.5'),
      dict(COND='spkmx_lif_muxlin_add', LR='0.0005', LAYERS='2', PATCH='16', EMB='32', MLPR='0.5'),
      dict(COND='spkmx_lif_muxlin_iand', LR='0.00025', LAYERS='1', EMB='32', MLPR='4', WD='0.1'),
      dict(COND='spkmx_lif_sdlin_or'),
      dict(COND='spk_ssa', LR='0.0005', PATCH='16')]
for kn in KN:
    ds = 'ETTh1' if kn.get('MLPR') == '4' else ('ETTh2' if kn['COND'] in ('spkmx_lif_sdlin_or', 'spk_ssa') else 'ETTm1')
    o = runner_args(old_runner, a.old, kn, ds)
    n = runner_args(new_runner, C, kn, ds)
    check(f'k0b runner {kn["COND"]} {ds} {"/".join(f"{k}={v}" for k, v in kn.items() if k != "COND")}: same training arguments',
          o is not None and o == n, '' if o == n else f'\n  old {o}\n  new {n}')
n0 = runner_args(new_runner, C, dict(KN[1], COND='spkmx_lif_sxdpc_add'))
n1 = runner_args(new_runner, C, dict(KN[1], COND='spkmx_lif_sxdpc_add', WSHARE='1'))
ok = n0 is not None and n1 is not None and n1.replace(' --mixer_share', '', 1).replace('_ws', '', 1) == n0 and ' --mixer_share' in n1
check('k0b WSHARE=1 adds only --mixer_share and _ws', ok, '' if ok else f'\n  {n0}\n  {n1}')
shutil.rmtree(tmp)

# ---- k2 / k3 / k4 ----------------------------------------------------------------------------------------------------------
from forecast_wrap import Forecaster           # noqa: E402
from spk_mixer import SpkMixer                 # noqa: E402
from sxa import AttnMath, attn_syops_hook      # noqa: E402

ARMS = [(c, s) for c in ('muxlin', 'sxd', 'sxdpc', 'sxp', 'sxppc') for s in (False, True)]
SHAPES = {'ETTh1': dict(data_patch_size=8, embed_dim=32, mlp_ratios=4., mixer_layers=1),
          'ETTm1': dict(data_patch_size=16, embed_dim=32, mlp_ratios=.5, mixer_layers=2)}


def build(comb, share, shape, seed=42):
    torch.manual_seed(seed)
    inner = SpkMixer(seq_len=96, num_classes=1, depths=1, mixer='lif', T_h=4, neo_tau=2.0, bias=False, tau=2.0, mixer_comb=comb,
                     connect_f='ADD', mixer_share=share, **shape)
    return Forecaster(inner, shape['embed_dim'], 96, bias=False)


for sn, shape in SHAPES.items():
    models = {arm: build(*arm, shape) for arm in ARMS}
    base = dict(models[('muxlin', False)].named_parameters())
    rev_n = sum(p.numel() for k, p in base.items() if '.mixer_rev.' in k)
    for arm, mdl in models.items():
        P = dict(mdl.named_parameters())
        common = [k for k in P if k in base]
        eq = all(torch.equal(P[k], base[k]) for k in common)
        must = [k for k in base if not k.startswith('inner.blocks.0.fuse') and not k.startswith('inner.blocks.0.sd_')
                and not (arm[1] and '.mixer_rev.' in k)]
        has_all = all(k in P for k in must)
        no_rev = (not arm[1]) or not any('.mixer_rev.' in k for k in P)
        own = sum(p.numel() for k, p in P.items() if k not in base)
        drop = sum(base[k].numel() for k in base if k not in P)
        cnt_ok = sum(p.numel() for p in P.values()) == sum(p.numel() for p in base.values()) + own - drop
        check(f'k2 {sn} {arm[0]} {"shared" if arm[1] else "indep."}: {len(common)} common parameters equal to muxlin indep. at the same '
              f'seed, embedding / stacks / MLP / head present, reverse stack {"absent" if arm[1] else "present"}',
              eq and has_all and no_rev and cnt_ok and (not arm[1] or drop >= rev_n),
              f'params {sum(p.numel() for p in P.values())} (new {own}, dropped {drop}, reverse stack {rev_n})')

# k3: sharing
mdl = build('sxdpc', True, SHAPES['ETTm1'])
blk = mdl.inner.blocks[0]
torch.manual_seed(0)
xm = (torch.rand(12, 6, 4, 32) < 0.3).float()
lay = blk.mixer.Mem[0]
for mode in ('eval', 'train'):
    lay.train(mode == 'train')
    with torch.no_grad():
        uf = lay.bn(lay.w_in(xm).reshape(-1, 32)).reshape(12, 6, 4, 32)
        ur = lay.bn(lay.w_in(xm.flip(0)).reshape(-1, 32)).reshape(12, 6, 4, 32)
    d = float((ur - uf.flip(0)).abs().max())
    check(f'k3 shared first layer ({mode} BN): reverse current = flipped forward current', d == 0. if mode == 'eval' else d <= 1e-6, f'max |diff| {d:.2g}')
mdl.train()
W = blk.mixer.Mem[0].w_in.weight
grads = {}
for part in ('fwd', 'rev', 'both'):
    mdl.zero_grad()
    f, hf = blk.stack(blk.mixer, xm, True)
    r, hr = blk.stack(blk.mixer, xm.flip(0), True)
    loss = {'fwd': (f * hf).sum(), 'rev': (r * hr).sum(), 'both': (f * hf).sum() + (r * hr).sum()}[part]
    loss.backward()
    grads[part] = W.grad.clone()
check('k3 shared stack gets gradients from both directions (each non-zero, different; joint = sum within 1e-6)',
      grads['fwd'].abs().sum() > 0 and grads['rev'].abs().sum() > 0 and not torch.allclose(grads['fwd'], grads['rev'])
      and torch.allclose(grads['both'], grads['fwd'] + grads['rev'], atol=1e-6))

# k4: arithmetic
torch.manual_seed(1)
N, B, T, D = 9, 3, 4, 16
q = (torch.rand(N, B, T, D) < .3).float(); k = (torch.rand(N, B, T, D) < .3).float(); v = (torch.rand(N, B, T, D) < .3).float()
hf, hr = torch.randn(N, B, T, D), torch.randn(N, B, T, D)
for mask in ('none', 'pc'):
    am = AttnMath(D, mask)
    z = am(q, k, v, hf, hr)
    zz = torch.zeros(N, B, T, D, dtype=torch.float64)
    M = np.ones((B, N, N))
    if mask == 'pc':
        mf, mr = hf.mean(2).double().numpy(), hr.mean(2).double().numpy()     # [N, B, D]
        for b in range(B):
            for n in range(N):
                for n2 in range(N):
                    M[b, n, n2] = (1 + np.corrcoef(mf[n, b], mr[n2, b])[0, 1]) / 2
        check('k4 Pearson mask equals numpy corrcoef (1 + rho) / 2 and lies in [0, 1]',
              np.allclose(am.last_mask.double().numpy(), M, atol=1e-5) and M.min() >= 0 and M.max() <= 1,
              f'max |diff| {np.abs(am.last_mask.double().numpy() - M).max():.2g}')
    for b in range(B):
        for t in range(T):
            A = q[:, b, t].double() @ k[:, b, t].double().T * torch.from_numpy(M[b])
            zz[:, b, t] = A @ v[:, b, t].double() / D ** .5
    check(f'k4 AttnMath ({mask}) equals the per-(b, t) loop', torch.allclose(z.double(), zz, atol=1e-5), f'max |diff| {float((z.double() - zz).abs().max()):.2g}')
    ac = float(q.sum()) * N + float(v.sum()) * N
    mac = 0.
    if mask == 'pc':
        ac += 2 * T * N * B * D + 4 * N * B * D + 2 * B * N * N
        mac += 4 * N * B * D + 2 * N * B + B * N * N * D + 3 * B * N * N + B * T * N * N
    check(f'k5 ledger ({mask}) equals the hand formula', am._ops == (ac, mac), f'{am._ops} vs {(ac, mac)}')

rates = {}
for comb in ('sxdpc', 'sxppc'):
    mdl = build(comb, False, SHAPES['ETTm1']).train()
    blk = mdl.inner.blocks[0]
    caught = {}
    h1 = blk.xattn.core.register_forward_hook(lambda m, i, o: caught.update(q=i[0], k=i[1], v=i[2]))
    h2 = blk.xattn.register_forward_hook(lambda m, i, o: caught.update(f=i[0], r=i[1], out=o))
    with torch.no_grad():
        mdl(torch.randn(8, 96, 7))
    h1.remove(); h2.remove()
    rates[comb] = {kk: float(vv.mean()) for kk, vv in caught.items()}
    check(f'k4 {comb} train mode fires: F, R, Q, K, V and the attention output all have spikes (non-vacuous)',
          all(vv.sum() > 0 for vv in caught.values()), ', '.join(f'{kk} {vv:.3f}' for kk, vv in rates[comb].items()))

# ---- k5: syops and the recorder ---------------------------------------------------------------------------------------------
from syops import get_model_complexity_info                   # noqa: E402
from syops.ops import LIF_syops_counter_hook                  # noqa: E402
from neurons import LIFNeuron, FLIFNeuron                     # noqa: E402
from energy import SynOpRecorder                              # noqa: E402
from spikingjelly.clock_driven import functional              # noqa: E402

for comb in ('sxdpc', 'sxppc', 'muxlin'):
    mdl = build(comb, True, SHAPES['ETTm1']).train()
    with torch.no_grad():
        for _ in range(120):                                 # BN running statistics, so that eval mode fires (v1 used 30: the running
            # variance still carried 4 % of its initial 1, eval mode did not fire and the spike terms of k5 were vacuous)
            mdl(torch.randn(16, 96, 7)); functional.reset_net(mdl)
    mdl.eval()
    g = torch.Generator().manual_seed(5)
    dl = [(torch.randn(4, 96, 7, generator=g), torch.zeros(1), 0, 0) for _ in range(3)]
    led = {'ac': 0., 'mac': 0., 'calls': 0, 'nnz': 0.}

    def tap(m, i, o):
        led['ac'] += m._ops[0]; led['mac'] += m._ops[1]; led['calls'] += 1
        led['nnz'] += float(i[0].count_nonzero()) + float(i[2].count_nonzero())
    hooks = [m.register_forward_hook(tap) for m in mdl.modules() if isinstance(m, AttnMath)]
    res = {}
    for nm, extra in (('with', {AttnMath: attn_syops_hook}), ('without', {})):
        led.update(ac=0., mac=0., calls=0, nnz=0.)
        with open(os.devnull, 'w') as fo:
            ops, _, _ = get_model_complexity_info(model=mdl, input_res=((96, 7),), dataloader=dl, as_strings=False, print_per_layer_stat=True,
                                                  verbose=False, ost=fo, custom_modules_hooks={LIFNeuron: LIF_syops_counter_hook,
                                                                                               FLIFNeuron: LIF_syops_counter_hook, **extra})
        res[nm] = (ops, dict(led))
    for h in hooks:
        h.remove()
    nwin = 12
    dac = res['with'][0][1] - res['without'][0][1]
    dmac = res['with'][0][2] - res['without'][0][2]
    lac, lmac = res['with'][1]['ac'] / nwin, res['with'][1]['mac'] / nwin
    nnz = res['with'][1]['nnz']
    check(f'k5 {comb}: syops with the hook - without = ledger per window (AC {dac:.1f} / {lac:.1f}, MAC {dmac:.1f} / {lmac:.1f}; '
          f'Q + V spikes in the syops pass {nnz:.0f})',
          abs(dac - lac) <= 1e-6 * max(1, lac) and abs(dmac - lmac) <= 1e-6 * max(1, lmac) and (comb == 'muxlin') == (lac == 0)
          and (comb == 'muxlin' or nnz > 0))
    rec = SynOpRecorder(mdl.inner)

    class NoAttn(SynOpRecorder):
        def _attn_hook(self, module, inp, out):
            pass
    rec0 = NoAttn(mdl.inner)
    led.update(ac=0., mac=0.)
    hooks = [m.register_forward_hook(tap) for m in mdl.modules() if isinstance(m, AttnMath)]
    with torch.no_grad():
        for x, _, _, _ in dl:
            mdl(x); functional.reset_net(mdl)
    for h in hooks:
        h.remove()
    rec.detach(); rec0.detach()
    ok = abs((rec.lin['ac'] - rec0.lin['ac']) - led['ac']) <= 1e-6 * max(1, led['ac']) and \
        abs((rec.lin['mac'] - rec0.lin['mac']) - led['mac']) <= 1e-6 * max(1, led['mac'])
    check(f'k5 {comb}: SynOpRecorder adds the same ledger (both rules)', ok and abs((rec.lin['ac_int'] - rec0.lin['ac_int']) - (rec.lin['ac'] - rec0.lin['ac'])) <= 1e-9 * max(1, led['ac']))
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
