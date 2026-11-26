"""Checks before stage 2 of the fusion redesign (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG stage-2 pre-registration). No training.
  s0 earlier conditions and the stage-1 arms unchanged: the stage-1 training code (`--ref`, commit 2fed8c580) and this code build
     muxlin IAND / ADD (indep., shared), or IAND, sdlin OR and sxd / sxdpc / sxp / sxppc x indep. / shared at one seed with bitwise
     equal parameters / buffers and train / eval outputs (s0_build.py in two processes)
  s1 OT plan: equals an independent float64 numpy Sinkhorn (plain domain, same cost, eps and iterations) within 1e-5; plan >= 0,
     sums to 1, every column sums to 1/N (1e-6), rows close to 1/N (reported); mask = N x plan
  s2 OT sanity: for two identical point sets with eps 0.01 x mean cost the plan puts > 90 % of its mass on the diagonal
  s3 MUX arms: the block's mixer output equals where(s_d, a, min(F, R)) recomputed from hooks, with s_d, a, F, R all firing
  s4 initial weights: each stage-2 arm (sxdot / sxpot / msxdpc / msxppc / msxdot / msxpot x indep. / shared) has every parameter
     of the matching stage-1 arm (same Q / K / V form and sharing) bitwise at the same seed; OT arms add nothing; MUX arms add
     only s_d's BN (weight 1 / bias 0)
  s5 ledger: the OT ledger equals the hand formula; syops with the AttnMath hook minus without = ledger per window, with spikes;
     SynOpRecorder adds the same
  s6 gradient through the OT mask: a loss through the mask alone gives the shared stack a non-zero gradient
usage: python -B check_s2.py --code <this code dir> --ref <worktree cinc_mixer_v1 at 2fed8c580>
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True)
ap.add_argument('--ref', required=True)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, a.code)
os.chdir(a.code)
import torch                                                # noqa: E402
PY = sys.executable
fails = 0


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}', flush=True)


# s0
tmp = tempfile.mkdtemp(prefix='s0_')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4')
R = {}
for tag, code in (('ref', a.ref), ('new', a.code)):
    p = os.path.join(tmp, f'{tag}.pt')
    r = subprocess.run([PY, '-B', '-W', 'ignore', os.path.join(HERE, 's0_build.py'), code, p], capture_output=True, text=True, env=env)
    check(f's0 build ({tag})', r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else '')
    R[tag] = torch.load(p) if r.returncode == 0 else None
if R['ref'] and R['new']:
    for k in R['ref']:
        o, n = R['ref'][k], R['new'][k]
        same = list(o['state']) == list(n['state']) and all(torch.equal(o['state'][q], n['state'][q]) for q in o['state'])
        check(f's0 {k}: parameters / buffers and outputs bitwise equal to the stage-1 code', same and torch.equal(o['ytr'], n['ytr']) and torch.equal(o['yev'], n['yev']))

from sxa import AttnMath, attn_syops_hook                   # noqa: E402

# s1 / s2
torch.manual_seed(3)
N, B, T, D = 11, 4, 4, 16
hf, hr = torch.randn(N, B, T, D), torch.randn(N, B, T, D) * 1.3 + 0.2
plan, eps = AttnMath.ot_plan(hf, hr, 20, 0.1)
x = hf.mean(2).transpose(0, 1).double().numpy(); y = hr.mean(2).transpose(0, 1).double().numpy()
ok_np = True
mx = 0.
for b in range(B):
    C = ((x[b][:, None, :] - y[b][None, :, :]) ** 2).sum(-1) / D
    e = 0.1 * C.mean()
    K = np.exp(-C / e)
    u, v = np.ones(N), np.ones(N)
    for _ in range(20):
        u = (1. / N) / (K @ v)
        v = (1. / N) / (K.T @ u)
    P = u[:, None] * K * v[None, :]
    mx = max(mx, np.abs(P - plan[b].double().numpy()).max())
check('s1 OT plan equals a float64 numpy Sinkhorn (plain domain)', mx < 1e-5, f'max |diff| {mx:.2g}')
cs = plan.sum(-2); rs = plan.sum(-1)
check('s1 plan >= 0, total 1, columns 1/N', bool((plan >= 0).all()) and torch.allclose(plan.sum((-1, -2)), torch.ones(B), atol=1e-5)
      and torch.allclose(cs, torch.full_like(cs, 1. / N), atol=1e-6), f'row sums deviate up to {float((rs - 1. / N).abs().max()):.2g} from 1/N = {1. / N:.4f}')
same = hf.clone()
p2, _ = AttnMath.ot_plan(same, same, 50, 0.01)
diag = float(torch.diagonal(p2, dim1=-2, dim2=-1).sum(-1).min())
check('s2 identical point sets, eps 0.01: diagonal mass > 0.9', diag > 0.9, f'min diagonal mass {diag:.3f}')

from forecast_wrap import Forecaster                        # noqa: E402
from spk_mixer import SpkMixer                              # noqa: E402
SH = dict(data_patch_size=16, embed_dim=32, mlp_ratios=.5, mixer_layers=2)


def build(comb, share, seed=42):
    torch.manual_seed(seed)
    inner = SpkMixer(seq_len=96, num_classes=1, depths=1, mixer='lif', T_h=4, neo_tau=2.0, bias=False, tau=2.0, mixer_comb=comb,
                     connect_f='ADD', mixer_share=share, **SH)
    return Forecaster(inner, 32, 96, bias=False)


# s3
for comb in ('msxdpc', 'msxpot'):
    mdl = build(comb, True).train()
    blk = mdl.inner.blocks[0]
    cap = {}
    hs = [blk.xattn.register_forward_hook(lambda m, i, o: cap.update(f=i[0], r=i[1], hf=i[2], hr=i[3], a=o)),
          blk.sd_neuron.register_forward_hook(lambda m, i, o: cap.update(sd=o))]
    with torch.no_grad():
        mdl(torch.randn(8, 96, 7))
        h = mdl.inner.encoding(mdl.inner.tokens(torch.randn(8 * 7, 1, 96)))
    for q in hs:
        q.remove()
    want = torch.where(cap['sd'] > 0, cap['a'], torch.minimum(cap['f'], cap['r']))
    got = cap['sd'] * cap['a'] + (1. - cap['sd']) * torch.minimum(cap['f'], cap['r'])
    fire = {k: float(cap[k].mean()) for k in ('f', 'r', 'a', 'sd')}
    check(f's3 {comb}: MUX = where(s_d, a, min(F, R)) and all four fire', torch.equal(want, got) and all(v > 0 for v in fire.values()),
          ', '.join(f'{k} {v:.3f}' for k, v in fire.items()))

# s4
S1 = {'d': 'sxdpc', 'p': 'sxppc'}
for comb in ('sxdot', 'sxpot', 'msxdpc', 'msxppc', 'msxdot', 'msxpot'):
    form = comb.lstrip('m')[2]
    for share in (False, True):
        P = dict(build(comb, share).named_parameters())
        Q = dict(build(S1[form], share).named_parameters())
        extra = sorted(set(P) - set(Q))
        miss = sorted(set(Q) - set(P))
        eq = all(torch.equal(P[k], Q[k]) for k in Q if k in P)
        want_extra = ['inner.blocks.0.sd_bn.bias', 'inner.blocks.0.sd_bn.weight'] if comb.startswith('m') else []
        bn_init = all(float(P[k].abs().sum()) == (P[k].numel() if k.endswith('weight') else 0.) for k in extra)
        check(f's4 {comb} {"shared" if share else "indep."}: every parameter of {S1[form]} equal; extra {extra or "none"}',
              eq and not miss and extra == want_extra and bn_init)

# s5
am = AttnMath(D, 'ot', 20, 0.1)
q = (torch.rand(N, B, T, D) < .3).float(); k = (torch.rand(N, B, T, D) < .3).float(); v = (torch.rand(N, B, T, D) < .3).float()
am(q, k, v, hf, hr)
L, BN2, BN = 20, float(B) * N * N, float(B) * N
ac = float(q.sum()) * N + float(v.sum()) * N + 2. * T * N * B * D + 2 * BN2 + BN2 + L * 2 * (3 * BN2 + 2 * BN) + 2 * BN2
mac = 4. * N * B * D + BN2 * D + 2 * BN2 + 2 * B + BN2 + L * 2 * (BN2 + BN) + 2 * BN2 + float(B) * T * N * N
check('s5 OT ledger equals the hand formula', am._ops == (ac, mac), f'{am._ops} vs {(ac, mac)}')
from syops import get_model_complexity_info                 # noqa: E402
from syops.ops import LIF_syops_counter_hook                # noqa: E402
from neurons import LIFNeuron, FLIFNeuron                   # noqa: E402
from energy import SynOpRecorder                            # noqa: E402
from spikingjelly.clock_driven import functional            # noqa: E402
for comb in ('sxdot', 'msxpot'):
    mdl = build(comb, True).train()
    with torch.no_grad():
        for _ in range(120):
            mdl(torch.randn(16, 96, 7)); functional.reset_net(mdl)
    mdl.eval()
    dl = [(torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(5 + i)), torch.zeros(1), 0, 0) for i in range(3)]
    led = dict(ac=0., mac=0., nnz=0.)

    def tap(m, i, o):
        led['ac'] += m._ops[0]; led['mac'] += m._ops[1]; led['nnz'] += float(i[0].count_nonzero()) + float(i[2].count_nonzero())
    hs = [m.register_forward_hook(tap) for m in mdl.modules() if isinstance(m, AttnMath)]
    res = {}
    for nm, extra in (('with', {AttnMath: attn_syops_hook}), ('without', {})):
        led.update(ac=0., mac=0., nnz=0.)
        with open(os.devnull, 'w') as fo:
            ops, _, _ = get_model_complexity_info(model=mdl, input_res=((96, 7),), dataloader=dl, as_strings=False, print_per_layer_stat=True,
                                                  verbose=False, ost=fo, custom_modules_hooks={LIFNeuron: LIF_syops_counter_hook,
                                                                                               FLIFNeuron: LIF_syops_counter_hook, **extra})
        res[nm] = (ops, dict(led))
    for h in hs:
        h.remove()
    dac, dmac = res['with'][0][1] - res['without'][0][1], res['with'][0][2] - res['without'][0][2]
    lac, lmac = res['with'][1]['ac'] / 12, res['with'][1]['mac'] / 12
    check(f's5 {comb}: syops with - without = ledger per window (AC {dac:.0f} / {lac:.0f}, MAC {dmac:.0f} / {lmac:.0f}; spikes {res["with"][1]["nnz"]:.0f})',
          abs(dac - lac) <= 1e-6 * lac and abs(dmac - lmac) <= 1e-6 * lmac and res['with'][1]['nnz'] > 0)
    rec = SynOpRecorder(mdl.inner)

    class NoAttn(SynOpRecorder):
        def _attn_hook(self, module, inp, out):
            pass
    rec0 = NoAttn(mdl.inner)
    led.update(ac=0., mac=0., nnz=0.)
    hs = [m.register_forward_hook(tap) for m in mdl.modules() if isinstance(m, AttnMath)]
    with torch.no_grad():
        for xb, _, _, _ in dl:
            mdl(xb); functional.reset_net(mdl)
    for h in hs:
        h.remove()
    rec.detach(); rec0.detach()
    check(f's5 {comb}: SynOpRecorder adds the same ledger', abs((rec.lin['ac'] - rec0.lin['ac']) - led['ac']) <= 1e-6 * led['ac']
          and abs((rec.lin['mac'] - rec0.lin['mac']) - led['mac']) <= 1e-6 * led['mac'])

# s6
mdl = build('sxdot', True).train()
blk = mdl.inner.blocks[0]
xm = (torch.rand(12, 6, 4, 32) < 0.3).float()
f, hf_ = blk.stack(blk.mixer, xm, True)
r, hr_ = blk.stack(blk.mixer, xm.flip(0), True)
p, _ = AttnMath.ot_plan(hf_, hr_.flip(0), 20, 0.1)
(p * torch.randn_like(p)).sum().backward()
g = blk.mixer.Mem[0].w_in.weight.grad
check('s6 a loss through the OT mask alone reaches the shared stack (non-zero gradient)', g is not None and float(g.abs().sum()) > 0)
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
