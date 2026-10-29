"""Checks for spk_mixer.SpkMixer (2026-10-29, PROJECT_LOG, exp/f-lif-pop-v3). Fail-closed: a check that cannot run, or
whose test is vacuous (no spikes / no change), FAILs.

  s0  default paths unchanged by the edits (layers.GroupNeuron axis handling, config, rhythm_eval): outputs and state_dicts of
      NeoMixer 6 neurons x {or, linsd} and the Spikformer-style model, saved before the edits, are bitwise equal after them
  s1  whole-model neuron: for 6 neurons x 4 fusions no spikingjelly neuron remains; every neuron layer is the chosen class
      (LIFNeuron / FLIFNeuron a=0.5 / PopulationFLIF a=1 / a=0.5 / tau-group LIF / f-LIF with taus 4/8/16/32); layer count =
      7 + {or 0, sdor 1, sdlin 2, linmx 1}; the energy recorder sees T = 4 for embedding / MLP and T = N = 117 for the mixer,
      s_d and fusion layers; training forward = plain logits, testing = (logits, [T_h, B, N, D])
  s2  layout [T_h, B, N, D] -> mixer [N, B, T_h, D]: flipping embedding spikes of repeat k at token j changes the forward stack
      only at (k, tokens >= j), the reverse stack only at (k, tokens <= j), other repeats never (non-vacuous: both change)
  s3  fusion forms (lif, real data): sdor m == max(F, R, s_d) and differs from max(F, R); sdlin fuse input == [F v R; s_d] and
      m == fuse(input); linmx fuse input == [H_f - th; H_r - th; |H_f - H_r|]; at the same seed embedding + mixer weights are
      identical across the four fusions, and the whole state_dict is identical across IAND / OR / ADD
  s4  residuals: block output = formula (x (1 - m) / max / +) recomputed from e, m and the MLP; values in {0, 1} for IAND / OR,
      in {0, 1, 2, 3} with some >= 2 for ADD (non-vacuous)
  s5  energy: on 64 validation windows the recorder's syops-rule AC / MAC equal syops' own totals (IAND, 0/1 everywhere); under
      ADD the integer rule gives fewer MACs and more ACs than the syops rule (fc1 sees values >= 2)
  s6  SpkMixer.tokens == myModel._tokens_mc hippocampus tokens (bitwise, hippo_ssa, T_h 4); embedding / MLP / head Linear shapes
      equal the Spikformer-style model's; no attention module
  s7  calibration (pop / popg x 4 fusions x 3 residuals = 48 tables): the table loads (tau_calibrated), and on real CinC data at
      init every calibrated layer fires within 0.5-2x of the same layer of its reference model (lif / flif, same fusion and
      residual) and >= 1 %; the reference layers fire >= 1 % too
usage: python check_spk.py [--only s1 s3] [--before before.pt --after after.pt]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from types import SimpleNamespace as NS                                   # noqa: E402
from spikingjelly.clock_driven import functional                          # noqa: E402
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, LIFNode    # noqa: E402
from config import set_random_seed                                        # noqa: E402
from load_model import LOAD_MODEL                                         # noqa: E402
from layers import GroupNeuron, NeoNeuron                                 # noqa: E402
from neurons import LIFNeuron, FLIFNeuron, PopulationFLIF                 # noqa: E402
from energy import FiringRateRecorder, SynOpRecorder                      # noqa: E402
from spk_mixer import COMBS, CONNECTS, NEURONS, CALIBRATED                # noqa: E402
import rhythm_data as rd                                                  # noqa: E402
import rhythm_eval as re_                                                 # noqa: E402

RESULTS = []
KW = dict(keep_ratio=0.15, gating='attn', embed_dim=384, num_heads=8, mlp_ratios=1.0, num_layers=1, time_num_layers=2, bias=False,
          tau=2.0, spk_encoding=False, neo_alpha=0.5, pop_taus='4,8,16,32', neo_tau=2.0, fde_method='pred', fde_step=1.0, fde_memory=0,
          mixer_bidir=True, pos_emb=False, tau_calib=False, patch_size=64, seq_len=3750, num_channels=1, num_classes=4,
          hippo_T=4, hippo_ssa=False, hippo_xattn=False, ssa_pe=1, attn_bn=False, neo_neuron='sj_lif', neo_where='both', neo_bidir=False,
          roll_mode='none', neo_recurrent=False, neo_rec_gate=False, neo_legacy=False, no_neo=False, hippo_only=False, grad_clip=0.,
          alpha=0.1, stream_rev=False, neo_stream_fuse='none', pop_soma_init=None, drop_rate=0.)
SOMA = {'pop_lif': 0.5, 'pop_flif': 1.0}
_DATA = {}


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


def data():
    if not _DATA:
        _DATA['d'] = rd.build(NS(data='CINC2017', crop_tokens=0, patch_size=64, fs_target=125., fold=0, val_frac=0.1, win_s=30., hop_s=15.), 42)
    return _DATA['d']


def real(n=8, split='train'):
    d = data()
    return torch.stack([d[split][i][0] for i in range(n)]).float().cuda()


def build(mixer, comb, connect, seed=0, tau_calib=False):
    set_random_seed(seed)
    return LOAD_MODEL['spkMixer'](NS(**dict(KW, model='spkMixer', mixer=mixer, mixer_comb=comb, connect_f=connect,
                                            pop_soma_init=SOMA.get(mixer), tau_calib=tau_calib)), True).cuda()


@torch.no_grad()
def warm(m, x, n=1):
    """set every BN's running statistics to the statistics of the real batch x (momentum 1, one training-mode pass) so that the
    eval-mode model reproduces the training-mode activity on x (c16 lesson: eval-mode BN at init silences everything), then eval."""
    for md in m.modules():
        if isinstance(md, torch.nn.BatchNorm1d):
            md.momentum = 1.0
    m.train(); m.train_mode = 'training'
    for _ in range(n):
        m(x); functional.reset_net(m)
    m.eval(); m.train_mode = 'testing'


def leaves(m):
    """[(name, neuron module)] = the neuron layers: NeoNeuron (single) or GroupNeuron (tau groups)."""
    out = []
    for name, mod in m.named_modules():
        if isinstance(mod, GroupNeuron):
            out.append((name, mod))
        elif isinstance(mod, NeoNeuron) and not any('.parts.' in name for _ in [0]) and '.parts.' not in name:
            out.append((name, mod))
    return out


@check('s0')
def s0():
    a = torch.load(ARGS.before); b = torch.load(ARGS.after)
    bad = [k for k in a if not (k in b and torch.equal(a[k]['z'], b[k]['z']) and torch.equal(a[k]['h'], b[k]['h'])
                                and a[k]['sd'].keys() == b[k]['sd'].keys() and all(torch.equal(a[k]['sd'][n], b[k]['sd'][n]) for n in a[k]['sd']))]
    report('s0', len(a) == 13 and not bad, f'{len(a)} snapshots, differing: {bad}')


@check('s1')
def s1():
    x = real(4)
    rows, ok = [], True
    exp_extra = {'or': 0, 'sdor': 1, 'sdlin': 2, 'linmx': 1}
    for mixer in NEURONS:
        for comb in COMBS:
            m = build(mixer, comb, 'IAND')
            sj = [n for n, md in m.named_modules() if isinstance(md, (MultiStepLIFNode, LIFNode))]
            lv = leaves(m)
            kinds = set()
            for n, md in lv:
                if isinstance(md, GroupNeuron):
                    inner = [p.neuron for p in md.parts]
                    taus = tuple(float(p.tau.item()) if p.tau.numel() == 1 else -1 for p in inner)
                    fam = 'popg_' + ('flif' if isinstance(inner[0], FLIFNeuron) and getattr(inner[0], 'alpha', 1.) < 1 else 'lif')
                    kinds.add((fam, taus, md.kind))
                else:
                    nn_ = md.neuron
                    if isinstance(nn_, PopulationFLIF):
                        kinds.add(('pop_flif' if nn_.alpha < 1 else 'pop_lif', int(nn_.num_members), bool(nn_.has_soma)))
                    elif isinstance(nn_, FLIFNeuron):
                        kinds.add(('flif', float(nn_.alpha)))
                    elif isinstance(nn_, LIFNeuron):
                        kinds.add(('lif',))
                    else:
                        kinds.add((type(nn_).__name__,))
            want = {'lif': {('lif',)}, 'flif': {('flif', 0.5)}, 'pop_lif': {('pop_lif', 4, True)}, 'pop_flif': {('pop_flif', 4, True)},
                    'popg_lif': {('popg_lif', (4., 8., 16., 32.), 'lif')}, 'popg_flif': {('popg_flif', (4., 8., 16., 32.), 'flif')}}[mixer]
            warm(m, x)
            rec = FiringRateRecorder(m)
            with torch.no_grad():
                z_test = m(x)
            rec.add_samples(x.shape[0]); rec.detach(); functional.reset_net(m)
            m.train(); m.train_mode = 'training'
            with torch.no_grad():
                z_train = m(x)
            functional.reset_net(m)
            Ts = {}
            for r in rec.layers():
                n = r['module']
                key = 'rep' if ('encoding' in n or '.fc1.' in n or '.fc2.' in n) else 'tok'
                Ts.setdefault(key, set()).add(int(r['T']))
            good = (not sj and kinds == want and len(lv) == 7 + exp_extra[comb] and Ts.get('rep') == {4} and Ts.get('tok') == {117}
                    and torch.is_tensor(z_train) and tuple(z_train.shape) == (4, 4) and isinstance(z_test, tuple)
                    and tuple(z_test[1].shape) == (4, 4, 117, 384))
            ok &= good
            rows.append(f'{mixer}/{comb}: sj {len(sj)}, layers {len(lv)}, kinds {sorted(kinds)}, T {Ts}, {"ok" if good else "BAD"}')
            del m
    report('s1', ok, ' | '.join(rows))


@check('s2')
def s2():
    x = real(8)
    m = build('lif', 'sdlin', 'IAND'); warm(m, x)
    blk = m.blocks[0]
    with torch.no_grad():
        e = m.encoding(m.tokens(x)); functional.reset_net(m)
        k, j = 1, 50
        e2 = e.clone(); e2[k, :, j, :] = 1. - e2[k, :, j, :]
        def run(em):
            xm = em.permute(2, 1, 0, 3).contiguous()
            f = blk.mixer(xm); r = blk.mixer_rev(xm.flip(0)).flip(0)
            functional.reset_net(m)
            return f, r
        f1, r1 = run(e); f2, r2 = run(e2)
    df = (f1 != f2).any(-1).any(1)                                        # [N, T_h]
    dr = (r1 != r2).any(-1).any(1)
    other = bool(df[:, [i for i in range(4) if i != k]].any() or dr[:, [i for i in range(4) if i != k]].any())
    fwd_ok = bool(df[j:, k].any()) and not bool(df[:j, k].any())
    rev_ok = bool(dr[:j + 1, k].any()) and not bool(dr[j + 1:, k].any())
    report('s2', fwd_ok and rev_ok and not other, f'forward changed tokens {df[:, k].nonzero().flatten().tolist()[:5]}..({int(df[:, k].sum())}), '
           f'reverse {int(dr[:, k].sum())} tokens <= {j}: {rev_ok}, other repeats changed: {other}')


@check('s3')
def s3():
    x = real(8)
    rows, ok = [], True
    for comb in ('sdor', 'sdlin', 'linmx'):
        m = build('lif', comb, 'IAND'); warm(m, x)
        blk = m.blocks[0]
        seen = {}
        hs = []
        if hasattr(blk, 'fuse'):
            hs.append(blk.fuse.linear.register_forward_pre_hook(lambda md, i: seen.__setitem__('fin', i[0].detach().clone())))
        with torch.no_grad():
            e = m.encoding(m.tokens(x)); functional.reset_net(m)
            mo = blk.mix(e); functional.reset_net(m)
            xm = e.permute(2, 1, 0, 3).contiguous()
            f, hf = blk.mixer.forward_charge(xm); r, hr = blk.mixer_rev.forward_charge(xm.flip(0)); r, hr = r.flip(0), hr.flip(0)
            functional.reset_net(m)
            fr = torch.maximum(f, r)
            if comb == 'linmx':
                exp_in = torch.cat([hf, hr, (hf - hr).abs()], -1)
                good = torch.equal(seen['fin'], exp_in) and torch.equal(mo.permute(2, 1, 0, 3), blk.fuse(exp_in))
                extra = f'real input {not bool(torch.equal(exp_in, exp_in.round()))}'
            else:
                d = (hf - hr).abs(); N, B, K, D = d.shape
                sd = blk.sd_neuron(blk.sd_bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous()); functional.reset_net(m)
                sd_rate = float(sd.mean())
                if comb == 'sdor':
                    exp = torch.maximum(fr, sd)
                    good = torch.equal(mo.permute(2, 1, 0, 3), exp) and not torch.equal(exp, fr) and sd_rate > 0.
                    extra = f's_d rate {sd_rate:.4f}, m != F v R: {not torch.equal(exp, fr)}'
                else:
                    exp_in = torch.cat([fr, sd], -1)
                    good = torch.equal(seen['fin'], exp_in) and torch.equal(mo.permute(2, 1, 0, 3), blk.fuse(exp_in)) and sd_rate > 0.
                    extra = f's_d rate {sd_rate:.4f}, fuse input 0/1 {bool(((exp_in == 0) | (exp_in == 1)).all())}'
        for h in hs: h.remove()
        ok &= good; rows.append(f'{comb}: {"ok" if good else "BAD"} ({extra}, F rate {float(f.mean()):.4f}, R rate {float(r.mean()):.4f})')
        del m
    sds = {comb: {k: v.cpu() for k, v in build('lif', comb, 'IAND').state_dict().items()} for comb in COMBS}
    shared = [k for k in sds['or'] if k.startswith(('encoding.', 'blocks.0.mixer.', 'blocks.0.mixer_rev.'))]
    same_init = all(torch.equal(sds['or'][k], sds[c][k]) for c in COMBS for k in shared) and len(shared) > 10
    sdc = {cf: {k: v.cpu() for k, v in build('lif', 'sdlin', cf).state_dict().items()} for cf in CONNECTS}
    same_cf = all(sdc['IAND'].keys() == sdc[cf].keys() and all(torch.equal(sdc['IAND'][k], sdc[cf][k]) for k in sdc['IAND']) for cf in CONNECTS)
    ok &= same_init and same_cf
    report('s3', ok, ' | '.join(rows) + f' | shared init across fusions ({len(shared)} tensors): {same_init}; identical across residuals: {same_cf}')


@check('s4')
def s4():
    x = real(8)
    rows, ok = [], True
    for cf in CONNECTS:
        m = build('lif', 'sdlin', cf); warm(m, x)
        blk = m.blocks[0]
        with torch.no_grad():
            z, e, m0, feat, h = m.forward_parts(x); functional.reset_net(m)
            fn = {'IAND': lambda a, b: a * (1. - b), 'OR': torch.maximum, 'ADD': lambda a, b: a + b}[cf]
            x1 = fn(e, m0)
            y = blk.fc2(blk.fc1(x1)); functional.reset_net(m)
            h2 = fn(x1, y)
        vals = sorted(torch.unique(h).tolist())
        good = torch.equal(h, h2) and (set(vals) <= {0., 1.} if cf != 'ADD' else (set(vals) <= {0., 1., 2., 3.} and max(vals) >= 2.))
        ok &= good; rows.append(f'{cf}: formula {torch.equal(h, h2)}, values {vals}, {"ok" if good else "BAD"}')
        del m
    report('s4', ok, ' | '.join(rows))


@check('s5')
def s5():
    from syops import get_model_complexity_info
    from syops.ops import LIF_syops_counter_hook
    d = data()
    sub = torch.utils.data.Subset(d['val'], list(range(64)))
    xw = real(16, 'val')
    rows, ok = [], True
    for cf in ('IAND', 'ADD'):
        m = build('lif', 'sdlin', cf); warm(m, xw)
        rec = SynOpRecorder(m)
        seen = {}
        h = m.blocks[0].fc1.linear.register_forward_pre_hook(lambda md, i: seen.__setitem__('fc1', float(i[0].max())))
        re_.predict(m, sub, 'cuda', bs=32, recorder=rec)
        rec.detach(); h.remove()
        s = rec.summary()
        m.train_mode = 'testing'; functional.reset_net(m)
        xb = sub[0][0]
        ops, params, fr = get_model_complexity_info(model=m, input_res=(tuple(xb.shape),), dataloader=re_.loader(sub, 32, False), as_strings=False,
                                                    print_per_layer_stat=True, verbose=False, ost=open(os.devnull, 'w'),   # as rhythm_eval.final_test
                                                    custom_modules_hooks={LIFNeuron: LIF_syops_counter_hook, FLIFNeuron: LIF_syops_counter_hook})
        ac_rel = abs(s['rec_AC_M'] * 1e6 - ops[1]) / max(1., ops[1]); mac_rel = abs(s['rec_MAC_M'] * 1e6 - ops[2]) / max(1., ops[2])
        if cf == 'IAND':
            good = ac_rel < 1e-3 and mac_rel < 1e-3 and abs(s['syn_energy_int_uJ'] - s['rec_syn_uJ']) < 1e-6 and s['rec_AC_M'] > 0
        else:
            good = seen['fc1'] >= 2. and s['int_MAC_M'] < s['rec_MAC_M'] and s['int_AC_M'] > s['rec_AC_M'] and s['syn_energy_int_uJ'] < s['rec_syn_uJ']
        ok &= good
        rows.append(f"{cf}: recorder AC {s['rec_AC_M']:.3f} M / syops {ops[1] / 1e6:.3f} M (rel {ac_rel:.1e}), MAC {s['rec_MAC_M']:.3f} / {ops[2] / 1e6:.3f} "
                    f"(rel {mac_rel:.1e}); integer rule AC {s['int_AC_M']:.3f} MAC {s['int_MAC_M']:.3f}; syn uJ syops-rule {s['rec_syn_uJ']:.3f} "
                    f"int-rule {s['syn_energy_int_uJ']:.3f}; fc1 max input {seen['fc1']:.0f}; {'ok' if good else 'BAD'}")
        del m
    report('s5', ok, ' | '.join(rows))


@check('s6')
def s6():
    x = real(4)
    m = build('lif', 'or', 'IAND')
    set_random_seed(0)
    mm = LOAD_MODEL['myModel'](NS(**dict(KW, model='myModel', hippo_ssa=True, hippo_T=4, mixer='lif', mixer_comb='or')), True).cuda()
    mm.train_mode = 'testing'
    with torch.no_grad():
        hip, _ = mm._tokens_mc(x)
        tok = m.tokens(x)
    same_tok = torch.equal(hip, tok) and tuple(tok.shape) == (4, 4, 117, 64)
    shapes = (m.encoding.emb_linear.weight.shape == mm.encoding.emb_linear.weight.shape and
              m.blocks[0].fc1.linear.weight.shape == mm.data_block[0].mlp.fc1.linear.weight.shape and
              m.blocks[0].fc2.linear.weight.shape == mm.data_block[0].mlp.fc2.linear.weight.shape and
              m.head.weight.shape == mm.head.weight.shape)
    no_attn = not any('ssa' in n or 'attn' in n for n, _ in m.named_modules())
    has_attn = any(n.endswith('.ssa') for n, _ in mm.named_modules())
    report('s6', same_tok and shapes and no_attn and has_attn, f'tokens bitwise {same_tok} {tuple(tok.shape)}, Linear shapes equal {shapes}, '
           f'spkMixer attention modules {not no_attn}, Spikformer-style has ssa {has_attn}')


@check('s7')
def s7():
    x = real(32)
    rows, ok, n_tab = [], True, 0
    for mixer in CALIBRATED:
        fam = mixer.split('_', 1)[1]
        for comb in COMBS:
            for cf in CONNECTS:
                ref = build(fam, comb, cf)
                rr, hs = {}, []
                for n, md in ref.named_modules():
                    if isinstance(md, NeoNeuron):
                        hs.append(md.register_forward_hook(lambda a, i, o, k=n: rr.__setitem__(k, float((o[0] if isinstance(o, tuple) else o).float().mean()))))
                ref.train(); ref.train_mode = 'testing'
                with torch.no_grad():
                    ref(x)
                functional.reset_net(ref)
                for h in hs: h.remove()
                del ref
                m = build(mixer, comb, cf, tau_calib=True); n_tab += int(m.tau_calibrated)
                rates, hs = {}, []
                for n, md in m.named_modules():
                    if isinstance(md, GroupNeuron):
                        for k, p in enumerate(md.parts):
                            hs.append(p.register_forward_hook(lambda a, i, o, key=(n, k): rates.__setitem__(key, float((o[0] if isinstance(o, tuple) else o).float().mean()))))
                    elif isinstance(md, NeoNeuron) and isinstance(md.neuron, PopulationFLIF):
                        hs.append(md.register_forward_hook(lambda a, i, o, key=(n, 0): rates.__setitem__(key, float((o[0] if isinstance(o, tuple) else o).float().mean()))))
                m.train(); m.train_mode = 'testing'
                with torch.no_grad():
                    m(x)
                functional.reset_net(m)
                for h in hs: h.remove()
                ratios = [(v, rr[n]) for (n, k), v in rates.items()]          # same module names in the reference (KeyError = FAIL)
                good = m.tau_calibrated and bool(ratios) and all(t >= 0.01 and v >= 0.01 and 0.5 <= v / t <= 2. for v, t in ratios)
                ok &= good
                rows.append(f'{mixer}/{comb}/{cf}: table {m.tau_calibrated}, layers {len(ratios)}, rate/ref '
                            f'{min(v / max(t, 1e-9) for v, t in ratios):.2f}-{max(v / max(t, 1e-9) for v, t in ratios):.2f}, '
                            f'min rate {min(v for v, _ in ratios):.3f}, min ref {min(t for _, t in ratios):.3f}, {"ok" if good else "BAD"}')
                del m
    report('s7', ok and n_tab == 48, f'{n_tab}/48 tables | ' + ' | '.join(rows))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'checks', 'check_spk.txt'))
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--before', default=None); ap.add_argument('--after', default=None)
    ARGS = ap.parse_args()
    for fn in (s0, s1, s2, s3, s4, s5, s6, s7):
        if ARGS.only and fn.__name__ not in ARGS.only:
            continue
        fn()
    n_ok = sum(ok for _, ok, _ in RESULTS)
    os.makedirs(os.path.dirname(ARGS.out), exist_ok=True)
    mode = 'a' if ARGS.only else 'w'
    with open(ARGS.out, mode) as f:
        for name, ok, msg in RESULTS:
            f.write(f"{'PASS' if ok else 'FAIL'} {name}: {msg}\n")
        f.write(f'{n_ok}/{len(RESULTS)} passed\n')
    print(f'{n_ok}/{len(RESULTS)} passed -> {ARGS.out}')
    sys.exit(0 if n_ok == len(RESULTS) else 1)
