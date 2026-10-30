"""Post-training probe of the fusion MUX (PROJECT_LOG 2026-10-30, pre-registered with the runs; no training, VALIDATION split only).

Checkpoints (LIF, residual OR, CinC fold 0, seeds 42 / 7 / 2026):
    muxor  (i)  m = s_d ? F v R : F ^ R      muxlin (ii) m = s_d ? L : F ^ R        new, results/flif-v1-rhythm-spk-mux-cinc
    sdor        m = F v R v s_d              sdlin       m = L                      B1 controls, results/flif-v1-rhythm-spk-cinc
F, R = last mixer layers' spikes, s_d = LIF(BN(|H_f - H_r|)), L = neuron(BN(W [F v R; s_d])). The instrumented model must reproduce
config['best_val_score'] within 1e-6 (fail closed); muxor m must equal majority(F, R, s_d) bitwise.

Statistics over all elements [tokens, windows, repeats, channels]:
    rates of F, R, F^R, F xor R, s_d, x (block input), m; P(s_d | F^R), P(s_d | F xor R), P(s_d | neither); L rate, P(L | s_d),
    P(L | not s_d) (sdlin / muxlin); what the OR residual adds (not x and m)
    gradient weight at the measured activations (formulas; torch.maximum / minimum split ties 0.5):
      to F  -- sdor [F>R] + .5[F=R] times [F v R > s_d] + .5[F v R = s_d]; sdlin [F>R] + .5[F=R] (into L);
               muxor s ([F>R] + .5[F=R]) + (1 - s) ([F<R] + .5[F=R]); muxlin the same split (the s branch into L)
      to L  -- sdlin 1, muxlin s
      select learns -- share of elements with dm/ds_d != 0 by the gate AND |H_sd - theta| <= 3 / (pi/2 * 5) (arctan surrogate >=
               10 % of its peak): muxor F xor R, muxlin L != min(F, R), sdor s_d >= F v R; sdlin and muxlin also reach s_d through L
    s_d per channel (quantiles, share below 0.1 %), by repeat index
usage: python mux_fuse_probe.py [-nd <gpu>] [--out results/mux-fuse-probe-<date>]
"""
import argparse
import json
import math
import os
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                     # noqa: E402,F401  (imported before timm, as in sd_probe.py)
from load_model import LOAD_MODEL                                    # noqa: E402
import rhythm_data as rd                                             # noqa: E402
import rhythm_eval as re_                                            # noqa: E402
from mux_probe import sha256                                         # noqa: E402

SEEDS = (42, 7, 2026)
COMBS = ('sdor', 'muxor', 'sdlin', 'muxlin')
BAND = 3. / (math.pi / 2. * 5.)
SUITES = {'sdor': 'flif-v1-rhythm-spk-cinc', 'sdlin': 'flif-v1-rhythm-spk-cinc',
          'muxor': 'flif-v1-rhythm-spk-mux-cinc', 'muxlin': 'flif-v1-rhythm-spk-mux-cinc'}


def runs(complete=True):
    found = {}
    for comb in COMBS:
        p = os.path.join(HERE, 'results', SUITES[comb], 'raw.txt')
        if not complete and not os.path.exists(p):
            continue
        for line in open(p):
            if not line.startswith('RESULT '):
                continue
            d = dict(re.findall(r'(\S+?)=(\S+)', line))
            if d['cond'] != f'spk_lif_{comb}_or':
                continue
            k = (comb, int(d['seed']))
            if k in found or int(d['fold']) != 0:
                raise SystemExit(f'{k}: twice or not fold 0')
            found[k] = d['dir']
    need = [(c, s) for c in COMBS for s in SEEDS]
    if complete and sorted(found) != sorted(need):
        raise SystemExit(f'runs found {sorted(found)}, need {sorted(need)}')
    return found


class Stats:
    def __init__(self, D, K, dev):
        self.c, self.n = {}, 0.
        self.chan_S = torch.zeros(D, dtype=torch.float64, device=dev)
        self.rep_S, self.rep_n = torch.zeros(K, dtype=torch.float64, device=dev), torch.zeros(K, dtype=torch.float64, device=dev)

    def acc(self, k, v):
        self.c[k] = self.c.get(k, 0.) + float(v)

    def add(self, comb, x, f, r, sd, hsd, L, m):
        n = float(f.numel()); self.n += n
        FR, FxR, FoR = f * r, (f != r).float(), torch.maximum(f, r)
        none = (1 - f) * (1 - r)
        for k, v in (('F', f), ('R', r), ('FR', FR), ('FxR', FxR), ('S', sd), ('x', x), ('m', m), ('add', (1 - x) * m), ('none', none)):
            self.acc(k, v.sum())
        self.acc('S_FR', (sd * FR).sum()); self.acc('S_FxR', (sd * FxR).sum()); self.acc('S_none', (sd * none).sum())
        tie, gt, lt = (f == r).float(), (f > r).float(), (f < r).float()
        band = (hsd.abs() <= BAND).float()
        s = sd
        if comb == 'sdor':
            wF = (gt + .5 * tie) * ((FoR > s).float() + .5 * (FoR == s).float())
            learn = ((s >= FoR).float()) * band
        elif comb == 'sdlin':
            wF = gt + .5 * tie
            learn = band
        else:
            wF = s * (gt + .5 * tie) + (1 - s) * (lt + .5 * tie)
            a = FoR if comb == 'muxor' else L
            learn = (a != torch.minimum(f, r)).float() * band
        self.acc('wF', wF.sum()); self.acc('learn', learn.sum()); self.acc('band', band.sum())
        if L is not None:
            self.acc('L', L.sum()); self.acc('L_S', (L * s).sum())
            self.acc('wL', (s if comb == 'muxlin' else torch.ones_like(s)).sum())
        if comb == 'muxor':
            self.acc('m_from_S', (s * m).sum()); self.acc('m_from_FR', ((1 - s) * m).sum())
        self.chan_S += sd.double().sum((0, 1, 2))
        self.rep_S += sd.double().sum((0, 1, 3)); self.rep_n += float(sd.shape[0] * sd.shape[1] * sd.shape[3])

    def result(self):
        c, n = self.c, self.n
        g = lambda k: c.get(k, float('nan'))
        chan = (self.chan_S / (n / self.chan_S.numel())).cpu().numpy()
        out = {k: g(k) / n for k in ('F', 'R', 'FR', 'FxR', 'S', 'x', 'm', 'add', 'wF', 'learn', 'band', 'L', 'wL')}
        out.update({'S_given_FR': g('S_FR') / g('FR'), 'S_given_FxR': g('S_FxR') / g('FxR'), 'S_given_none': g('S_none') / g('none'),
                    'L_given_S': g('L_S') / g('S'), 'L_given_notS': (g('L') - g('L_S')) / (n - g('S')),
                    'm_from_S_share': g('m_from_S') / g('m') if 'm_from_S' in c else float('nan'),
                    'chan_q10_50_90': [float(np.quantile(chan, q)) for q in (.1, .5, .9)], 'chan_dead': float((chan < 1e-3).mean()),
                    'by_repeat': (self.rep_S / self.rep_n).cpu().tolist()})
        return out


def instrument(blk, st, chk):
    def mix(x):
        xm = x.permute(2, 1, 0, 3).contiguous()
        f, hf = blk.mixer.forward_charge(xm)
        r, hr = blk.mixer_rev.forward_charge(xm.flip(0)); r, hr = r.flip(0), hr.flip(0)
        d = (hf - hr).abs(); N, B, K, D = d.shape
        sd, hsd = blk.sd_neuron.forward_charge(blk.sd_bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous())
        fr = torch.maximum(f, r)
        L = blk.fuse(torch.cat([fr, sd], dim=-1)) if blk.comb in ('sdlin', 'muxlin') else None
        if blk.comb == 'sdor':
            m = torch.maximum(fr, sd)
        elif blk.comb == 'sdlin':
            m = L
        else:
            m = sd * (fr if blk.comb == 'muxor' else L) + (1. - sd) * torch.minimum(f, r)
            if blk.comb == 'muxor':
                chk['muxor_eq_majority'] &= bool(torch.equal(m, ((f + r + sd) >= 2).to(m.dtype)))
        st.add(blk.comb, xm, f, r, sd, hsd, L, m)
        return m.permute(2, 1, 0, 3).contiguous()
    blk.mix = mix


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('-nd', '--num_device', type=int, default=0); ap.add_argument('--out', default=None)
    ap.add_argument('--only', default=None, help='<comb>/<seed>: one checkpoint (test run; no report)')
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, 'results', f'mux-fuse-probe-{time.strftime("%Y%m%d")}')
    os.makedirs(out, exist_ok=False)
    dev = torch.device(f'cuda:{a.num_device}')
    rec = {'script_sha256': sha256(__file__), 'band': BAND, 'runs': {}}
    todo = sorted(runs(complete=not a.only).items(), key=lambda kv: (COMBS.index(kv[0][0]), SEEDS.index(kv[0][1])))
    if a.only:
        c0, s0 = a.only.split('/'); todo = [kv for kv in todo if kv[0] == (c0, int(s0))]
        if not todo:
            raise SystemExit(f'{a.only}: not found')
    for (comb, seed), d in todo:
        t0 = time.time()
        cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
        args = SimpleNamespace(**cfg)
        if (args.model != 'spkMixer' or args.mixer != 'lif' or args.mixer_comb != comb or str(args.connect_f).lower() != 'or'
                or int(args.fold) != 0 or int(args.seed) != seed):
            raise SystemExit(f'{d}: config {args.model} {args.mixer} {args.mixer_comb} {args.connect_f} fold {args.fold} seed {args.seed}')
        args.device = dev
        m = LOAD_MODEL[args.model](args, True)
        ck = os.path.join(d, 'model_state', 'checkpoint.pth')
        m.load_state_dict(torch.load(ck, map_location='cpu'))
        m = m.to(dev).eval(); m.train_mode = 'testing'
        if len(m.blocks) != 1:
            raise SystemExit(f'{d}: {len(m.blocks)} blocks')
        blk = m.blocks[0]
        st = Stats(m.head.in_features, m.T_h, dev)
        chk = {'muxor_eq_majority': True}
        instrument(blk, st, chk)
        ds = rd.build(args, int(args.seed))['val']
        with torch.no_grad():
            P, Y, _ = re_.predict(m, ds, dev, bs=int(getattr(args, 'eval_bs', 32)))
        met, _ = re_.metrics(P, Y, args.data)
        diff = abs(met['score'] - float(cfg['best_val_score']))
        if diff > 1e-6 or not all(chk.values()):
            raise SystemExit(f'{comb} seed {seed}: score {met["score"]} vs stored {cfg["best_val_score"]} ({diff:.1e}); checks {chk}')
        s = st.result()
        rec['runs'][f'{comb}/s{seed}'] = {'run_dir': d, 'checkpoint_sha256': sha256(ck), 'val_score': met['score'], 'repro_abs_diff': diff,
                                          'checks': chk, 'stats': s, 'seconds': time.time() - t0}
        print(f"[mux-fuse-probe] {comb} seed {seed}: val {100 * met['score']:.2f} (repro {diff:.1e}) | F {100 * s['F']:.2f} R {100 * s['R']:.2f} "
              f"F&R {100 * s['FR']:.2f} s_d {100 * s['S']:.2f} m {100 * s['m']:.2f} | {time.time() - t0:.0f} s", flush=True)
        del m; torch.cuda.empty_cache()
    json.dump(rec, open(os.path.join(out, 'mux_fuse_probe.json'), 'x'), indent=1)
    if not a.only:
        open(os.path.join(out, 'mux_fuse_probe.md'), 'x').write(report(rec))
    print(f'wrote {out}')


def report(rec):
    R = rec['runs']
    mean = lambda c, k: float(np.mean([R[f'{c}/s{s}']['stats'][k] for s in SEEDS]))
    pct = lambda x: '—' if not math.isfinite(x) else f'{100 * x:.2f}'
    L = ['# Fusion MUX probe (validation, CinC fold 0, LIF, residual OR; mean of seeds 42 / 7 / 2026, %)', '',
         '## Firing', '', '| fusion | F | R | F&R | F xor R | s_d | P(s_d given F&R) | P(s_d given F xor R) | P(s_d given neither) | m | OR residual adds | L | P(L given s_d) | P(L given not s_d) |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for c in COMBS:
        L.append(f'| {c} | ' + ' | '.join(pct(mean(c, k)) for k in ('F', 'R', 'FR', 'FxR', 'S', 'S_given_FR', 'S_given_FxR', 'S_given_none', 'm', 'add', 'L', 'L_given_S', 'L_given_notS')) + ' |')
    L += ['', '## Gradient weight and where s_d fires', '', '| fusion | to F (mean weight) | to L | select learns | s_d channel q10 / q50 / q90 | channels < 0.1 % | s_d by repeat 1-4 | muxor: m from the s_d branch |',
          '|---|---|---|---|---|---|---|---|']
    for c in COMBS:
        q = np.mean([R[f'{c}/s{s}']['stats']['chan_q10_50_90'] for s in SEEDS], 0)
        rep = np.mean([R[f'{c}/s{s}']['stats']['by_repeat'] for s in SEEDS], 0)
        L.append(f"| {c} | {pct(mean(c, 'wF'))} | {pct(mean(c, 'wL'))} | {pct(mean(c, 'learn'))} | {' / '.join(f'{100 * v:.2f}' for v in q)} | "
                 f"{pct(mean(c, 'chan_dead'))} | {' '.join(f'{100 * v:.2f}' for v in rep)} | {pct(mean(c, 'm_from_S_share'))} |")
    L += ['', '## Per seed', '']
    for k, v in R.items():
        s = v['stats']
        L.append(f"- {k}: val {100 * v['val_score']:.2f} (repro {v['repro_abs_diff']:.1e}); s_d {100 * s['S']:.2f}, m {100 * s['m']:.2f}, "
                 f"select learns {100 * s['learn']:.2f}; {v['seconds']:.0f} s")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
