"""Checks before stage 4 of the fusion redesign (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG stage-4 pre-registration). No training.
  t0 earlier conditions and every stage-1 / 2 / 3a arm unchanged: the code before stage 4 (`--ref`, commit 3dc4a4888) and this code
     give bitwise equal parameters / buffers and train / eval outputs (s4_build.py in two processes)
  t1 additive mask: with alpha 0 AttnMath('pc' / 'ot', combine 'add') returns exactly the no-mask map's output; with alpha 0.7 it
     equals (A_s + 0.7 M) V / sqrt(D) recomputed by hand
  t2 initial weights: msxdpca / msxdota at a seed have every parameter of msxdpc (IAND) at that seed plus alpha = 0
  t3 ledger: the additive ledger equals the hand formula; syops with the AttnMath hook minus without = ledger per window (with spikes);
     SynOpRecorder adds the same
  t4 alpha learns: one backward pass of a train-mode forecaster gives alpha a non-zero gradient, and alpha is in model.parameters()
usage (from cinc_mixer_v1/): python -B results/check-sxa4-20261127/check_s4.py --ref <copy of cinc_mixer_v1 at 3dc4a4888>
"""
import argparse
import os
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


tmp = tempfile.mkdtemp(prefix='s4_')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4')
R = {}
for tag, code in (('ref', a.ref), ('new', C)):
    p = os.path.join(tmp, f'{tag}.pt')
    r = subprocess.run([PY, '-B', '-W', 'ignore', os.path.join(HERE, 's4_build.py'), code, p], capture_output=True, text=True, env=env)
    check(f't0 build ({tag})', r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else '')
    R[tag] = torch.load(p) if r.returncode == 0 else None
if R['ref'] and R['new']:
    for k in R['ref']:
        o, n = R['ref'][k], R['new'][k]
        same = list(o['state']) == list(n['state']) and all(torch.equal(o['state'][q], n['state'][q]) for q in o['state'])
        check(f't0 {k}: bitwise equal to the code before stage 4', same and torch.equal(o['ytr'], n['ytr']) and torch.equal(o['yev'], n['yev']))
from sxa import AttnMath, attn_syops_hook                   # noqa: E402
torch.manual_seed(2)
N, B, T, D = 10, 3, 4, 16
q = (torch.rand(N, B, T, D) < .3).float(); k = (torch.rand(N, B, T, D) < .3).float(); v = (torch.rand(N, B, T, D) < .3).float()
hf, hr = torch.randn(N, B, T, D), torch.randn(N, B, T, D)
z0 = AttnMath(D, 'none')(q, k, v, hf, hr)
for mask in ('pc', 'ot'):
    am = AttnMath(D, mask, combine='add')
    check(f't1 {mask} add, alpha 0: output equals the no-mask map exactly', torch.equal(am(q, k, v, hf, hr), z0) and float(am.alpha) == 0.)
    with torch.no_grad():
        am.alpha.fill_(0.7)
    z = am(q, k, v, hf, hr)
    M = am.last_mask
    want = torch.zeros_like(z)
    for b in range(B):
        for t in range(T):
            want[:, b, t] = (q[:, b, t] @ k[:, b, t].T + 0.7 * M[b]) @ v[:, b, t] / D ** .5
    check(f't1 {mask} add, alpha 0.7: equals (A_s + 0.7 M) V / sqrt(D)', torch.allclose(z, want, atol=1e-5), f'max |diff| {float((z - want).abs().max()):.2g}')
    ac = float(q.sum()) * N + float(v.sum()) * N
    BN2, BN = float(B) * N * N, float(B) * N
    if mask == 'pc':
        ac += 2. * T * N * B * D + 4. * N * B * D + 2. * BN2
        mac = 4. * N * B * D + 2. * N * B + BN2 * D + 3. * BN2
    else:
        L = 20
        ac += 2. * T * N * B * D + 2 * BN2 + BN2 + L * 2 * (3 * BN2 + 2 * BN) + 2 * BN2
        mac = 4. * N * B * D + BN2 * D + 2 * BN2 + 2 * B + BN2 + L * 2 * (BN2 + BN) + 2 * BN2
    ac += float(B) * T * N * N; mac += BN2
    check(f't3 {mask} add: ledger equals the hand formula', am._ops == (ac, mac), f'{am._ops} vs {(ac, mac)}')
from forecast_wrap import Forecaster                        # noqa: E402
from spk_mixer import SpkMixer                              # noqa: E402


def build(comb, seed=42):
    torch.manual_seed(seed)
    inner = SpkMixer(seq_len=96, num_classes=1, depths=1, mixer='lif', T_h=4, neo_tau=2.0, bias=False, tau=2.0, mixer_comb=comb,
                     connect_f='IAND', data_patch_size=16, embed_dim=32, mlp_ratios=.5, mixer_layers=2)
    return Forecaster(inner, 32, 96, bias=False)


Q = dict(build('msxdpc').named_parameters())
for comb in ('msxdpca', 'msxdota'):
    P = dict(build(comb).named_parameters())
    extra = sorted(set(P) - set(Q))
    check(f't2 {comb}: every msxdpc parameter equal at the same seed, extra = alpha (0)', not (set(Q) - set(P)) and all(torch.equal(P[x], Q[x]) for x in Q)
          and extra == ['inner.blocks.0.xattn.core.alpha'] and float(P[extra[0]]) == 0., f'extra {extra}')
from syops import get_model_complexity_info                 # noqa: E402
from syops.ops import LIF_syops_counter_hook                # noqa: E402
from neurons import LIFNeuron, FLIFNeuron                   # noqa: E402
from energy import SynOpRecorder                            # noqa: E402
from spikingjelly.clock_driven import functional            # noqa: E402
for comb in ('msxdpca', 'msxdota'):
    mdl = build(comb).train()
    with torch.no_grad():
        mdl.inner.blocks[0].xattn.core.alpha.fill_(0.5)
        for _ in range(120):
            mdl(torch.randn(16, 96, 7)); functional.reset_net(mdl)
    mdl.eval()
    dl = [(torch.randn(4, 96, 7, generator=torch.Generator().manual_seed(9 + i)), torch.zeros(1), 0, 0) for i in range(3)]
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
    check(f't3 {comb}: syops with - without = ledger per window (AC {dac:.0f} / {lac:.0f}, MAC {dmac:.0f} / {lmac:.0f}; spikes {res["with"][1]["nnz"]:.0f})',
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
    check(f't3 {comb}: SynOpRecorder adds the same ledger', abs((rec.lin['ac'] - rec0.lin['ac']) - led['ac']) <= 1e-6 * led['ac']
          and abs((rec.lin['mac'] - rec0.lin['mac']) - led['mac']) <= 1e-6 * led['mac'])
    mdl2 = build(comb).train()
    loss = mdl2(torch.randn(8, 96, 7)).pow(2).mean()
    loss.backward()
    al = mdl2.inner.blocks[0].xattn.core.alpha
    check(f't4 {comb}: alpha in model.parameters() and gets a non-zero gradient at alpha 0', any(p is al for p in mdl2.parameters())
          and al.grad is not None and float(al.grad.abs()) > 0, f'grad {float(al.grad) if al.grad is not None else None}')
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
