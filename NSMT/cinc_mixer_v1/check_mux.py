"""Checks for the fusion MUX of the SpkMixer (2026-10-30, PROJECT_LOG, exp/f-lif-pop-v3). Fail-closed: a check that cannot run, or
whose test is vacuous, FAILs.

  m0  default paths unchanged by the edit (spk_mixer.py, config.py): the 45 snapshot_spk.py snapshots (NeoMixer 6 x {or, linsd},
      Spikformer-style, SpkMixer 6 neurons x {or, sdor, sdlin, linmx} x OR, LIF x 4 fusions x {IAND, ADD}) are bitwise equal
      before and after
  m1  forms (LIF, real CinC batch, initial model): muxor m == where(s_d, F v R, F ^ R) == majority(F, R, s_d); muxlin m ==
      where(s_d, L, F ^ R) with L = fuse([F v R; s_d]) and the fuse input == [F v R; s_d]; values in {0, 1}; non-vacuous
      (s_d fires, m differs from the sdor / sdlin m of the same spikes)
  m2  same initial model: at seeds 0 / 42 / 7 / 2026 the whole state_dict of muxor == sdor and of muxlin == sdlin (bitwise)
  m3  gradients: (a) truth table of s max(F, R) + (1 - s) min(F, R) with autograd: dm/ds = |F - R|, dm/dF = s / (1 - s) /
      0.5 at ties, dm/da = s for a learned branch; torch.minimum / maximum split ties 0.5 / 0.5; (b) in the model (train mode)
      the gradient reaching s_d equals w (a - min(F, R)) and the one reaching L equals w s exactly; reach of F / R reported
      next to sdor / or
  m4  structure and energy: neuron layers muxor 8 / muxlin 9, all LIFNeuron, no spikingjelly neuron, T 4 on the repetition
      axis and 117 on the token axis; recorder AC / MAC == syops on 64 validation windows (0/1 inputs, OR residual)
usage: python check_mux.py --before before.pt --after after.pt [--only m1 m3]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                          # noqa: E402
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, LIFNode    # noqa: E402
import check_spk as C                                                     # noqa: E402  (data / build / warm / leaves)
from neurons import LIFNeuron, FLIFNeuron                                 # noqa: E402
from energy import FiringRateRecorder, SynOpRecorder                      # noqa: E402
import rhythm_eval as re_                                                 # noqa: E402

RESULTS = []


def report(name, ok, msg):
    RESULTS.append((name, bool(ok), msg)); print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}", flush=True)


def check(name):
    def deco(fn):
        def run():
            try:
                fn()
            except Exception as e:                                      # noqa: BLE001
                report(name, False, f'{type(e).__name__}: {e}')
        run.__name__ = name
        return run
    return deco


def parts(m, x):
    """the fusion inputs of block 0 on x, recomputed outside mix(): F, R, s_d [N, B, T_h, D] (model state reset after)."""
    blk = m.blocks[0]
    e = m.encoding(m.tokens(x)); functional.reset_net(m)
    xm = e.permute(2, 1, 0, 3).contiguous()
    f, hf = blk.mixer.forward_charge(xm); r, hr = blk.mixer_rev.forward_charge(xm.flip(0)); r, hr = r.flip(0), hr.flip(0)
    functional.reset_net(m)
    d = (hf - hr).abs(); N, B, K, D = d.shape
    sd = blk.sd_neuron(blk.sd_bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous()); functional.reset_net(m)
    return e, f, r, sd


@check('m0')
def m0():
    a = torch.load(ARGS.before); b = torch.load(ARGS.after)
    bad = [k for k in a if not (k in b and torch.equal(a[k]['z'], b[k]['z']) and torch.equal(a[k]['h'], b[k]['h'])
                                and a[k]['sd'].keys() == b[k]['sd'].keys() and all(torch.equal(a[k]['sd'][n], b[k]['sd'][n]) for n in a[k]['sd']))]
    report('m0', len(a) == 45 and len(b) == 45 and not bad, f'{len(a)} / {len(b)} snapshots, differing: {bad}')


@check('m1')
def m1():
    x = C.real(8)
    rows, ok = [], True
    for comb in ('muxor', 'muxlin'):
        m = C.build('lif', comb, 'OR'); C.warm(m, x)
        blk = m.blocks[0]
        seen = {}
        hs = [blk.fuse.linear.register_forward_pre_hook(lambda md, i: seen.__setitem__('fin', i[0].detach().clone()))] if comb == 'muxlin' else []
        with torch.no_grad():
            e, f, r, sd = parts(m, x)
            mo = blk.mix(e).permute(2, 1, 0, 3); functional.reset_net(m)
            fr, fa = torch.maximum(f, r), torch.minimum(f, r)
            if comb == 'muxor':
                exp = torch.where(sd.bool(), fr, fa)
                maj = ((f + r + sd) >= 2).float()
                other = torch.maximum(fr, sd)                                 # sdor of the same spikes
                good = torch.equal(mo, exp) and torch.equal(exp, maj)
                extra = f'm == majority {torch.equal(exp, maj)}'
            else:
                fin = torch.cat([fr, sd], -1)
                L = blk.fuse(fin); functional.reset_net(m)
                exp = torch.where(sd.bool(), L, fa)
                other = L                                                     # sdlin of the same spikes
                good = torch.equal(mo, exp) and torch.equal(seen['fin'], fin)
                extra = f'fuse input == [F v R; s_d] {torch.equal(seen["fin"], fin)}, L rate {float(L.mean()):.4f}, P(L | s_d) {float((L * sd).sum() / sd.sum()):.4f}'
        for h in hs: h.remove()
        vals = sorted(torch.unique(mo).tolist())
        sd_rate = float(sd.mean())
        differs = not torch.equal(mo, other)
        good = good and set(vals) <= {0., 1.} and sd_rate > 0. and differs
        ok &= good
        rows.append(f'{comb}: formula {"ok" if good else "BAD"} ({extra}; values {vals}; F {float(f.mean()):.4f} R {float(r.mean()):.4f} '
                    f'F^R {float((f * r).mean()):.4f} s_d {sd_rate:.4f} m {float(mo.mean()):.4f}; differs from {"sdor" if comb == "muxor" else "sdlin"} m: {differs})')
        del m
    report('m1', ok, ' | '.join(rows))


@check('m2')
def m2():
    rows, ok = [], True
    for seed in (0, 42, 7, 2026):
        for new, old in (('muxor', 'sdor'), ('muxlin', 'sdlin')):
            a = {k: v.cpu() for k, v in C.build('lif', new, 'OR', seed=seed).state_dict().items()}
            b = {k: v.cpu() for k, v in C.build('lif', old, 'OR', seed=seed).state_dict().items()}
            same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a) and len(a) > 20
            ok &= same
            rows.append(f'seed {seed} {new} == {old}: {same} ({len(a)} tensors)')
    report('m2', ok, ' | '.join(rows))


@check('m3')
def m3():
    # (a) truth table
    good_a, tab = True, []
    for fv in (0., 1.):
        for rv in (0., 1.):
            for sv in (0., 1.):
                f = torch.tensor(fv, requires_grad=True); r = torch.tensor(rv, requires_grad=True); s = torch.tensor(sv, requires_grad=True)
                a = torch.tensor(0.7, requires_grad=True)                     # a learned branch value, for dm/da
                m = s * torch.maximum(f, r) + (1. - s) * torch.minimum(f, r)
                gf, gr, gs = torch.autograd.grad(m, (f, r, s))
                ml = s * a + (1. - s) * torch.minimum(f, r)
                ga, gs2 = torch.autograd.grad(ml, (a, s))
                exp_f = 0.5 if fv == rv else (sv if fv > rv else 1. - sv)
                exp_r = 0.5 if fv == rv else (sv if rv > fv else 1. - sv)
                row_ok = (float(m) == float(fv + rv + sv >= 2) and float(gs) == abs(fv - rv) and float(gf) == exp_f and float(gr) == exp_r
                          and float(ga) == sv and abs(float(gs2) - (0.7 - min(fv, rv))) < 1e-7)
                good_a &= row_ok
                tab.append(f'F{fv:.0f}R{rv:.0f}s{sv:.0f}: m {float(m):.0f} dF {float(gf):.1f} dR {float(gr):.1f} ds {float(gs):.0f}')
    # (b) in the model, train mode
    x = C.real(8)
    rows, good_b = [], True
    for comb in ('muxor', 'muxlin'):
        m = C.build('lif', comb, 'OR'); C.warm(m, x)
        m.train(); m.train_mode = 'training'
        blk = m.blocks[0]
        keep = {}

        def grab(key):
            def hook(md, i, o):
                o.retain_grad(); keep[key] = o
            return hook
        hs = [blk.sd_neuron.register_forward_hook(grab('sd'))]
        if comb == 'muxlin':
            hs.append(blk.fuse.register_forward_hook(grab('L')))

            def grab_in(md, i):                                               # s_d also enters L: its share comes back via the fuse input
                i[0].retain_grad(); keep['fin'] = i[0]
            hs.append(blk.fuse.register_forward_pre_hook(grab_in))
        with torch.no_grad():
            e = m.encoding(m.tokens(x)); functional.reset_net(m)
        mo = blk.mix(e)                                                       # [T_h, B, N, D]
        w = torch.randn_like(mo)
        (mo * w).sum().backward()
        functional.reset_net(m)
        for h in hs: h.remove()
        with torch.no_grad():
            _, f, r, sd = parts(m, x)
        wn = w.permute(2, 1, 0, 3)
        fa = torch.minimum(f, r)
        a = torch.maximum(f, r) if comb == 'muxor' else keep['L'].detach()
        via_L = 0. if comb == 'muxor' else keep['fin'].grad[..., sd.shape[-1]:]
        g_sd_ok = torch.equal(sd, keep['sd'].detach()) and torch.allclose(keep['sd'].grad, wn * (a - fa) + via_L, atol=1e-5, rtol=0)
        g_L_ok = True if comb == 'muxor' else torch.allclose(keep['L'].grad, wn * sd, atol=1e-6, rtol=0)
        # reach of F (share of elements whose dm/dF != 0) by the formulas, next to sdor and or on the same spikes
        tie = (f == r).float()
        reach_mux = float(((tie + (1 - tie) * ((f > r).float() * sd + (f < r).float() * (1 - sd))) > 0).float().mean())
        fr = torch.maximum(f, r)
        reach_sdor = float((((f >= r) & (fr >= sd))).float().mean())
        reach_or = float((f >= r).float().mean())
        reach_L = float(sd.mean())
        good = g_sd_ok and g_L_ok and reach_mux >= reach_sdor
        good_b &= good
        rows.append(f'{comb}: grad s_d == w (a - min) {g_sd_ok}' + ('' if comb == 'muxor' else f', grad L == w s_d {g_L_ok} (L reached on {reach_L:.4f})')
                    + f'; F reach mux {reach_mux:.4f} / sdor {reach_sdor:.4f} / or {reach_or:.4f}')
        del m
    report('m3', good_a and good_b, ('truth table ok' if good_a else 'truth table BAD: ' + '; '.join(tab)) + ' | ' + ' | '.join(rows))


@check('m4')
def m4():
    from syops import get_model_complexity_info
    from syops.ops import LIF_syops_counter_hook
    x = C.real(4)
    d = C.data()
    sub = torch.utils.data.Subset(d['val'], list(range(64)))
    xw = C.real(16, 'val')
    rows, ok = [], True
    for comb, want in (('muxor', 8), ('muxlin', 9)):
        m = C.build('lif', comb, 'OR')
        sj = [n for n, md in m.named_modules() if isinstance(md, (MultiStepLIFNode, LIFNode))]
        lv = C.leaves(m)
        kinds = {type(md.neuron).__name__ for _, md in lv}
        C.warm(m, x)
        fr_rec = FiringRateRecorder(m)
        with torch.no_grad():
            m(x)
        fr_rec.add_samples(x.shape[0]); fr_rec.detach(); functional.reset_net(m)
        Ts = {}
        for r in fr_rec.layers():
            n = r['module']
            Ts.setdefault('rep' if ('encoding' in n or '.fc1.' in n or '.fc2.' in n) else 'tok', set()).add(int(r['T']))
        C.warm(m, xw)
        rec = SynOpRecorder(m)
        re_.predict(m, sub, 'cuda', bs=32, recorder=rec)
        rec.detach()
        s = rec.summary()
        m.train_mode = 'testing'; functional.reset_net(m)
        ops, params, _ = get_model_complexity_info(model=m, input_res=(tuple(sub[0][0].shape),), dataloader=re_.loader(sub, 32, False), as_strings=False,
                                                   print_per_layer_stat=True, verbose=False, ost=open(os.devnull, 'w'),
                                                   custom_modules_hooks={LIFNeuron: LIF_syops_counter_hook, FLIFNeuron: LIF_syops_counter_hook})
        ac_rel = abs(s['rec_AC_M'] * 1e6 - ops[1]) / max(1., ops[1]); mac_rel = abs(s['rec_MAC_M'] * 1e6 - ops[2]) / max(1., ops[2])
        good = (not sj and len(lv) == want and kinds == {'LIFNeuron'} and Ts.get('rep') == {4} and Ts.get('tok') == {117}
                and ac_rel < 1e-3 and mac_rel < 1e-3 and s['rec_AC_M'] > 0 and abs(s['syn_energy_int_uJ'] - s['rec_syn_uJ']) < 1e-6)
        ok &= good
        rows.append(f"{comb}: layers {len(lv)}, kinds {sorted(kinds)}, sj {len(sj)}, T {Ts}; recorder AC {s['rec_AC_M']:.3f} M / syops "
                    f"{ops[1] / 1e6:.3f} M (rel {ac_rel:.1e}), MAC {s['rec_MAC_M']:.3f} / {ops[2] / 1e6:.3f} (rel {mac_rel:.1e}); "
                    f"syn uJ {s['rec_syn_uJ']:.3f}; {'ok' if good else 'BAD'}")
        del m
    report('m4', ok, ' | '.join(rows))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'checks', 'check_mux.txt'))
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--before', default=None); ap.add_argument('--after', default=None)
    ARGS = ap.parse_args()
    C.ARGS = ARGS
    for fn in (m0, m1, m2, m3, m4):
        if ARGS.only and fn.__name__ not in ARGS.only:
            continue
        fn()
    n_ok = sum(ok for _, ok, _ in RESULTS)
    os.makedirs(os.path.dirname(ARGS.out), exist_ok=True)
    with open(ARGS.out, 'a' if ARGS.only else 'w') as f:
        for name, ok, msg in RESULTS:
            f.write(f"{'PASS' if ok else 'FAIL'} {name}: {msg}\n")
        f.write(f'{n_ok}/{len(RESULTS)} passed\n')
    print(f'{n_ok}/{len(RESULTS)} passed -> {ARGS.out}')
    sys.exit(0 if n_ok == len(RESULTS) and len(RESULTS) > 0 else 1)
