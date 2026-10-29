"""Post-hoc probe for a MUX residual (PROJECT_LOG 2026-10-29/30; not pre-registered, no training, VALIDATION split only).

Question (user, 10-29): on the B1 LIF checkpoints of the SpkMixer rerun, how often would the select signal of
    MUX(a):  out = s ? m : x = s * m + (1 - s) * x,   s = s_d, x = embedding spikes (block input), m = fusion output
fire, how does it overlap with the co-firing of the two directions (F and R = last mixer layers' spikes), and what effects /
problems can be expected. MUX(a) would replace the MIXER residual only (the MLP residual has no s_d).

Checkpoints: spk_lif_{or,sdor,sdlin}_<cf> x seeds 42 / 7 / 2026 of results/flif-v1-rhythm-spk-cinc/raw.txt, cf = the residual
selected in B1 (select_residual.json; fail closed if none). model_state/checkpoint.pth = the best-validation weights; the
unmodified model must reproduce config['best_val_score'] within 1e-6 (fail closed). s_d exists only in sdor / sdlin.

Note: in sdor, m = F v R v s_d, so s_d = 1 forces m = 1 and MUX(a) reduces to x v s_d (checked). The variant mux_fr uses
F v R as the data input instead (the "OR fusion + s_d as select only" design).

Statistics (full model; all elements [tokens N, windows B, repeats 4, channels D]):
    rates of F, R, F&R, F|R, F^R, s_d, x, m, m != x; joint table of (F, R, s_d); conditional rates
    MUX change: s & (m != x), split into add (x=0, m=1) and delete (x=1, m=0); same with m := F v R; the trained residual's own
    change (IAND delete x&m, OR add ~x&m, ADD adds m); share of elements where MUX output != trained residual output
    gradient reach (mean gradient weight, from the formulas at the measured activations): to m -- IAND x, OR [m>x] + 0.5[m=x]
    (torch.maximum splits ties), ADD 1, MUX s; to x -- IAND 1-m, OR [x>m] + 0.5[x=m], ADD 1, MUX 1-s; to the select s
    (MUX only): share with m != x AND |H_sd - theta| <= 3 / (pi/2 * 5) (arctan surrogate >= 10 % of its peak)
    s_d per channel (quantiles, share of channels below 0.1 %), by token decile, by repeat index, per window by class
Plug-in variants (the mixer residual replaced, MLP residual unchanged; out of the training distribution: reliance, not the
score of a retrained model):  full | mux (s = s_d, data m) | mux_fr (s = s_d, data F v R) | mux_rate (s ~ Bernoulli of each
(window, channel)'s s_d rate, generator seed 0) | keep_x (s = 0: mixer ignored) | take_m (s = 1: no residual).
keep_x / take_m also run on the or checkpoints.
usage: python mux_probe.py [-nd <gpu>] [--out results/mux-probe-<date>] [--only <comb>/<seed>]
"""
import argparse
import hashlib
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

COMBS = ('or', 'sdor', 'sdlin')
SEEDS = (42, 7, 2026)
CLASSES = ('N', 'AF', 'Other', 'Noisy')
BAND = 3. / (math.pi / 2. * 5.)                                       # arctan surrogate (scale 5) >= 10 % of its peak
VARIANTS_SD = ('full', 'mux', 'mux_fr', 'mux_rate', 'keep_x', 'take_m')
VARIANTS_OR = ('full', 'keep_x', 'take_m')
SUITE = os.path.join(HERE, 'results', 'flif-v1-rhythm-spk-cinc')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def selected_cf():
    sel = json.load(open(os.path.join(SUITE, 'select_residual.json')))
    cf = sel.get('selected')
    if not cf:
        raise SystemExit(f'no residual selected in select_residual.json (missing {sel.get("missing")})')
    return str(cf).lower()


def runs(cf, complete=True):
    found = {}
    for line in open(os.path.join(SUITE, 'raw.txt')):
        if not line.startswith('RESULT '):
            continue
        d = dict(re.findall(r'(\S+?)=(\S+)', line))
        m = re.fullmatch(rf'spk_lif_({"|".join(COMBS)})_{cf}', d['cond'])
        if not m:
            continue
        k = (m.group(1), int(d['seed']))
        if k in found or int(d['fold']) != 0:
            raise SystemExit(f'{k}: twice or not fold 0')
        found[k] = d['dir']
    need = [(c, s) for c in COMBS for s in SEEDS]
    if complete and sorted(found) != sorted(need):
        raise SystemExit(f'runs found {sorted(found)}, need {sorted(need)}')
    return found


class Stats:
    """float64 counters on the device; one instance per checkpoint (full variant only)."""
    def __init__(self, D, N, K, dev, cf, has_sd):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
        self.cf, self.has_sd, self.N, self.K = cf, has_sd, N, K
        self.n = 0.
        self.c = {k: 0. for k in ('F', 'R', 'FR', 'FoR', 'FxR', 'S', 'x', 'm', 'mnx', 'chg_add', 'chg_del', 'chgfr_add', 'chgfr_del',
                                  'trained_chg', 'mux_ne_trained', 'muxfr_ne_trained', 'gm_trained', 'gx_trained', 'gm_mux', 'gx_mux',
                                  'sel_learn', 'sel_band', 'S_given_FR', 'S_given_FxR', 'S_given_none', 'FR_and_S', 'FxR_and_S')}
        self.joint = z(8)
        self.chan_S = z(D)
        self.tok_S, self.tok_FR, self.tok_n = z(10), z(10), z(10)
        self.rep_S, self.rep_n = z(K), z(K)
        self.win_S = []
        self.dec = torch.div(torch.arange(N, device=dev) * 10, N, rounding_mode='floor')   # token -> decile

    def add(self, x, f, r, m, sd, hsd):
        """all [N, B, K, D] 0/1 (sd / hsd None without s_d); x = block input, m = the model's fusion output."""
        N, B, K, D = x.shape
        c = self.c
        n = float(x.numel()); self.n += n
        fr, fo, fx = f * r, torch.maximum(f, r), (f - r).abs()
        c['F'] += float(f.sum()); c['R'] += float(r.sum()); c['FR'] += float(fr.sum()); c['FoR'] += float(fo.sum()); c['FxR'] += float(fx.sum())
        c['x'] += float(x.sum()); c['m'] += float(m.sum())
        ne = (m != x).to(x.dtype); c['mnx'] += float(ne.sum())
        # trained residual: own change and gradient weights (mixer residual)
        if self.cf == 'iand':
            c['trained_chg'] += float((x * m).sum()); c['gm_trained'] += float(x.sum()); c['gx_trained'] += float((1 - m).sum())
            y_tr = x * (1 - m)
        elif self.cf == 'or':
            c['trained_chg'] += float(((1 - x) * m).sum())
            c['gm_trained'] += float((m > x).sum()) + .5 * float((m == x).sum())
            c['gx_trained'] += float((x > m).sum()) + .5 * float((x == m).sum())
            y_tr = torch.maximum(x, m)
        else:
            c['trained_chg'] += float(m.sum()); c['gm_trained'] += n; c['gx_trained'] += n
            y_tr = x + m
        tidx = self.dec.view(N, 1, 1, 1).expand(N, B, K, D)
        self.tok_FR += torch.zeros(10, dtype=torch.float64, device=x.device).index_add_(0, tidx.reshape(-1), fr.reshape(-1).double())
        self.tok_n += torch.zeros(10, dtype=torch.float64, device=x.device).index_add_(0, tidx.reshape(-1), torch.ones_like(fr).reshape(-1).double())
        if not self.has_sd:
            return
        s = sd
        c['S'] += float(s.sum())
        code = (f * 4 + r * 2 + s).long().reshape(-1)
        self.joint += torch.bincount(code, minlength=8).double()
        c['S_given_FR'] += float((s * fr).sum()); c['S_given_FxR'] += float((s * fx).sum())
        c['S_given_none'] += float((s * (1 - fo)).sum())
        c['FR_and_S'] += float((s * fr).sum()); c['FxR_and_S'] += float((s * fx).sum())
        c['chg_add'] += float((s * (1 - x) * m).sum()); c['chg_del'] += float((s * x * (1 - m)).sum())
        c['chgfr_add'] += float((s * (1 - x) * fo).sum()); c['chgfr_del'] += float((s * x * (1 - fo)).sum())
        y_mux = s * m + (1 - s) * x
        y_muxfr = s * fo + (1 - s) * x
        c['mux_ne_trained'] += float((y_mux != y_tr).sum()); c['muxfr_ne_trained'] += float((y_muxfr != y_tr).sum())
        c['gm_mux'] += float(s.sum()); c['gx_mux'] += float((1 - s).sum())
        band = (hsd.abs() <= BAND).to(x.dtype)
        c['sel_band'] += float(band.sum()); c['sel_learn'] += float((band * ne).sum())
        self.chan_S += s.sum((0, 1, 2)).double()
        self.tok_S += torch.zeros(10, dtype=torch.float64, device=x.device).index_add_(0, tidx.reshape(-1), s.reshape(-1).double())
        self.rep_S += s.sum((0, 1, 3)).double(); self.rep_n += float(N * B * D)
        self.win_S += s.sum((0, 2, 3)).double().cpu().tolist()

    def result(self, labels):
        c, n = self.c, self.n
        P = lambda k: c[k] / n
        out = dict(n_elements=n, rate_F=P('F'), rate_R=P('R'), rate_FandR=P('FR'), rate_ForR=P('FoR'), rate_FxorR=P('FxR'),
                   rate_x=P('x'), rate_m=P('m'), rate_m_ne_x=P('mnx'), trained_change=P('trained_chg'),
                   grad_m_trained=P('gm_trained'), grad_x_trained=P('gx_trained'),
                   tok_decile_FandR=(self.tok_FR / self.tok_n).cpu().tolist())
        if not self.has_sd:
            return out
        S = c['S']
        chan = (self.chan_S / (n / self.chan_S.numel())).cpu().numpy()
        win = np.array(self.win_S); per = n / len(win)                # elements per window
        if len(win) != len(labels):
            raise SystemExit(f'{len(win)} windows recorded but {len(labels)} labels')
        out.update(rate_sd=S / n, joint_FRS=(self.joint / n).cpu().tolist(),
                   sd_given_FandR=c['S_given_FR'] / max(c['FR'], 1), sd_given_FxorR=c['S_given_FxR'] / max(c['FxR'], 1),
                   sd_given_neither=c['S_given_none'] / max(n - c['FoR'], 1),
                   FandR_given_sd=c['FR_and_S'] / max(S, 1), FxorR_given_sd=c['FxR_and_S'] / max(S, 1),
                   mux_change_add=P('chg_add'), mux_change_del=P('chg_del'),
                   muxfr_change_add=P('chgfr_add'), muxfr_change_del=P('chgfr_del'),
                   mux_ne_trained=P('mux_ne_trained'), muxfr_ne_trained=P('muxfr_ne_trained'),
                   grad_m_mux=P('gm_mux'), grad_x_mux=P('gx_mux'), select_band=P('sel_band'), select_learn=P('sel_learn'),
                   sd_chan_quantiles=np.quantile(chan, [.1, .25, .5, .75, .9]).tolist(), sd_chan_dead=float((chan < 1e-3).mean()),
                   sd_chan_above_50=float((chan > .5).mean()),
                   tok_decile_sd=(self.tok_S / self.tok_n).cpu().tolist(), rep_sd=(self.rep_S / self.rep_n).cpu().tolist(),
                   win_sd_by_class={CLASSES[k]: dict(n=int((labels == k).sum()), mean_rate=float(win[labels == k].mean() / per) if (labels == k).any() else None,
                                                     median_rate=float(np.median(win[labels == k]) / per) if (labels == k).any() else None)
                                    for k in range(4)})
        if abs(self.joint.sum().item() - n) > .5 or abs(c['FoR'] - (c['F'] + c['R'] - c['FR'])) > .5:
            raise SystemExit('identity failed: joint table sum or P(F|R) = P(F) + P(R) - P(F&R)')
        return out


def instrument(blk, variant, st, gen, chk):
    """Replaces blk.mix (stats + the variant's mixer-residual output, stashed) and blk.connect (first call after mix returns the
    stash: the mixer residual; the MLP residual uses the original connect)."""
    orig_connect = blk.connect
    stash = {}

    def mix(x):                                                       # x [T_h, B, N, D]
        xm = x.permute(2, 1, 0, 3).contiguous()                       # [N, B, K, D]
        f, hf = blk.mixer.forward_charge(xm)
        r, hr = blk.mixer_rev.forward_charge(xm.flip(0))
        r, hr = r.flip(0), hr.flip(0)
        fo = torch.maximum(f, r)
        sd = hsd = None
        if blk.comb == 'or':
            m = fo
        else:
            d = (hf - hr).abs()
            N, B, K, D = d.shape
            sd, hsd = blk.sd_neuron.forward_charge(blk.sd_bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous())
            if blk.comb == 'sdor':
                m = torch.maximum(fo, sd)
            else:
                m = blk.fuse(torch.cat([fo, sd], dim=-1))
        if st is not None:
            st.add(xm, f, r, m, sd, hsd)
        y = None                                                      # the variant's mixer-residual output [N, B, K, D]
        if variant == 'keep_x':
            y = xm
            chk['keep_x_eq_m0'] &= bool(torch.equal(orig_connect(x, torch.zeros_like(x)), x))
        elif variant == 'take_m':
            y = m
        elif variant in ('mux', 'mux_fr', 'mux_rate'):
            s = sd
            if variant == 'mux_rate':
                p = sd.mean((0, 2))                                   # [B, D]: each (window, channel)'s s_d rate
                s = (torch.rand(sd.shape, generator=gen, device=sd.device) < p[None, :, None, :]).to(sd.dtype)
            data = fo if variant == 'mux_fr' else m
            y = s * data + (1 - s) * xm
            if variant == 'mux' and blk.comb == 'sdor':
                chk['sdor_mux_eq_x_or_sd'] &= bool(torch.equal(y, torch.maximum(xm, sd)))
        if blk.comb == 'sdor':
            chk['sdor_m_eq'] &= bool(torch.equal(m, torch.maximum(torch.maximum(f, r), sd)))
        if y is not None:
            stash['y'] = y.permute(2, 1, 0, 3).contiguous()
        return m.permute(2, 1, 0, 3).contiguous()

    def connect(x, m):
        if 'y' in stash:
            return stash.pop('y')
        return orig_connect(x, m)

    blk.mix, blk.connect = mix, connect


def window_labels(ds, bs):
    return np.concatenate([y.numpy() for _, y, _ in re_.loader(ds, bs, shuffle=False)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('-nd', '--num_device', type=int, default=0)
    ap.add_argument('--out', default=None); ap.add_argument('--only', default=None, help='<comb>/<seed>, e.g. sdor/42 (timing run)')
    ap.add_argument('--cf', default=None, help='timing run only (with --only): residual to use before B1 has selected one')
    a = ap.parse_args()
    if a.cf and not a.only:
        raise SystemExit('--cf is for timing runs (--only) only; the full probe reads the B1 selection')
    cf = a.cf.lower() if a.cf else selected_cf()
    out = a.out or os.path.join(HERE, 'results', f'mux-probe-{time.strftime("%Y%m%d")}')
    os.makedirs(out, exist_ok=False)
    dev = torch.device(f'cuda:{a.num_device}')
    rec = {'script_sha256': sha256(__file__), 'selected_residual': cf, 'band': BAND, 'runs': {}}
    todo = sorted(runs(cf, complete=not a.only).items(), key=lambda kv: (COMBS.index(kv[0][0]), SEEDS.index(kv[0][1])))
    if a.only:
        c0, s0 = a.only.split('/'); todo = [kv for kv in todo if kv[0] == (c0, int(s0))]
    for (comb, seed), d in todo:
        t0 = time.time()
        cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
        args = SimpleNamespace(**cfg)
        if (args.model != 'spkMixer' or args.mixer != 'lif' or args.mixer_comb != comb or str(args.connect_f).lower() != cf
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
        orig = (blk.mix, blk.connect)
        bs = int(getattr(args, 'eval_bs', 32))
        ds = rd.build(args, int(args.seed))['val']
        labels = window_labels(ds, bs)
        row = {'run_dir': d, 'checkpoint_sha256': sha256(ck), 'best_val_score_config': float(cfg['best_val_score']), 'variants': {}}
        has_sd = comb != 'or'
        chk = dict(keep_x_eq_m0=True, sdor_mux_eq_x_or_sd=True, sdor_m_eq=True)
        for v in (VARIANTS_SD if has_sd else VARIANTS_OR):
            st = Stats(m.head.in_features, m.num_patches, m.T_h, dev, cf, has_sd) if v == 'full' else None
            gen = torch.Generator(device=dev); gen.manual_seed(0)
            instrument(blk, v, st, gen, chk)
            tv = time.time()
            with torch.no_grad():
                P, Y, _ = re_.predict(m, ds, dev, bs=bs)
            blk.mix, blk.connect = orig
            met, _ = re_.metrics(P, Y, args.data)
            row['variants'][v] = {'score': met['score'], 'macro_f1': met['macro_f1'], 'f1': [met[f'f1_{i}'] for i in range(4)],
                                  'seconds': time.time() - tv}
            if v == 'full':
                row['stats'] = st.result(labels)
        row['checks'] = chk
        if not all(chk.values()):
            raise SystemExit(f'{comb} seed {seed}: identity check failed {chk}')
        diff = abs(row['variants']['full']['score'] - row['best_val_score_config'])
        row['repro_abs_diff'] = diff
        if diff > 1e-6:
            raise SystemExit(f'{comb} seed {seed}: full {row["variants"]["full"]["score"]} vs stored {row["best_val_score_config"]}')
        row['seconds'] = time.time() - t0
        row['peak_mem_GB'] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        rec['runs'][f'{comb}/s{seed}'] = row
        s = row['stats']
        print(f'[mux-probe] {comb} seed {seed}: ' + ' '.join(f"{v} {100 * x['score']:.2f}" for v, x in row['variants'].items())
              + f" | F&R {100 * s['rate_FandR']:.2f}%" + (f" s_d {100 * s['rate_sd']:.2f}%" if has_sd else '')
              + f" | {row['seconds']:.0f} s, {row['peak_mem_GB']:.2f} GB", flush=True)
        del m; torch.cuda.empty_cache()
    json.dump(rec, open(os.path.join(out, 'mux_probe.json'), 'x'), indent=1)
    if not a.only:
        open(os.path.join(out, 'mux_probe.md'), 'x').write(report(rec))
    print(f'wrote {out}')


def report(rec):
    R = rec['runs']
    pct = lambda x: f'{100 * x:.2f}'
    mean = lambda comb, f: float(np.mean([f(R[f'{comb}/s{s}']) for s in SEEDS]))
    L = [f"# MUX(a) select probe (validation, CinC fold 0, LIF, residual {rec['selected_residual'].upper()}; mean of seeds 42 / 7 / 2026, %)", '',
         '## Firing and overlap', '',
         '| fusion | F | R | F&R | F^R | s_d | P(s_d given F&R) | P(s_d given F^R) | P(s_d given neither) | P(F&R given s_d) | P(F^R given s_d) |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for c in COMBS:
        st = lambda k: (lambda r: r['stats'][k])
        row = [pct(mean(c, st(k))) for k in ('rate_F', 'rate_R', 'rate_FandR', 'rate_FxorR')]
        if c != 'or':
            row += [pct(mean(c, st(k))) for k in ('rate_sd', 'sd_given_FandR', 'sd_given_FxorR', 'sd_given_neither', 'FandR_given_sd', 'FxorR_given_sd')]
        else:
            row += ['—'] * 6
        L.append(f'| {c} | ' + ' | '.join(row) + ' |')
    L += ['', '## What MUX(a) would change (mixer residual)', '',
          '| fusion | x | m | m != x | trained residual change | MUX add / delete (data m) | MUX add / delete (data F v R) | MUX != trained (m) | MUX != trained (F v R) |',
          '|---|---|---|---|---|---|---|---|---|']
    for c in COMBS[1:]:
        g = lambda k: pct(mean(c, lambda r: r['stats'][k]))
        L.append(f"| {c} | {g('rate_x')} | {g('rate_m')} | {g('rate_m_ne_x')} | {g('trained_change')} | {g('mux_change_add')} / {g('mux_change_del')} | "
                 f"{g('muxfr_change_add')} / {g('muxfr_change_del')} | {g('mux_ne_trained')} | {g('muxfr_ne_trained')} |")
    L += ['', '## Gradient reach (mean gradient weight at the measured activations)', '',
          '| fusion | to m: trained / MUX | to x: trained / MUX | select near threshold | select learns (near threshold and m != x) |', '|---|---|---|---|---|']
    for c in COMBS[1:]:
        g = lambda k: pct(mean(c, lambda r: r['stats'][k]))
        L.append(f"| {c} | {g('grad_m_trained')} / {g('grad_m_mux')} | {g('grad_x_trained')} / {g('grad_x_mux')} | {g('select_band')} | {g('select_learn')} |")
    L += ['', '## Where s_d fires', '',
          '| fusion | channel rate q10 / q50 / q90 | dead channels (< 0.1 %) | > 50 % | by token decile (first ... last) | by repeat 1-4 | per window: N / AF / Other / Noisy |',
          '|---|---|---|---|---|---|---|']
    for c in COMBS[1:]:
        q = np.mean([R[f'{c}/s{s}']['stats']['sd_chan_quantiles'] for s in SEEDS], 0)
        tok = np.mean([R[f'{c}/s{s}']['stats']['tok_decile_sd'] for s in SEEDS], 0)
        rep = np.mean([R[f'{c}/s{s}']['stats']['rep_sd'] for s in SEEDS], 0)
        cls = [np.mean([R[f'{c}/s{s}']['stats']['win_sd_by_class'][k]['mean_rate'] for s in SEEDS]) for k in CLASSES]
        L.append(f"| {c} | {pct(q[0])} / {pct(q[2])} / {pct(q[4])} | {pct(mean(c, lambda r: r['stats']['sd_chan_dead']))} | "
                 f"{pct(mean(c, lambda r: r['stats']['sd_chan_above_50']))} | {' '.join(f'{100 * x:.1f}' for x in tok)} | "
                 f"{' '.join(f'{100 * x:.2f}' for x in rep)} | {' / '.join(f'{100 * x:.2f}' for x in cls)} |")
    L += ['', '## Plug-in validation score (mixer residual replaced, no retraining; out of distribution)', '',
          '| fusion | full | mux | mux_fr | mux_rate | keep_x (mixer ignored) | take_m (no residual) |', '|---|---|---|---|---|---|---|']
    for c in COMBS:
        vs = VARIANTS_SD if c != 'or' else VARIANTS_OR
        cells = []
        for v in VARIANTS_SD:
            if v not in vs:
                cells.append('—'); continue
            sc = [R[f'{c}/s{s}']['variants'][v]['score'] for s in SEEDS]
            full = [R[f'{c}/s{s}']['variants']['full']['score'] for s in SEEDS]
            cells.append(f'{100 * np.mean(sc):.2f}' if v == 'full' else f"{100 * np.mean(sc):.2f} ({100 * np.mean(np.array(sc) - full):+.2f}; "
                         + '/'.join(f'{100 * (x - y):+.1f}' for x, y in zip(sc, full)) + ')')
        L.append(f'| {c} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Per seed (F&R, s_d, select learns, %)', '']
    for k, r in R.items():
        s = r['stats']
        L.append(f"- {k}: F&R {pct(s['rate_FandR'])}" + (f", s_d {pct(s['rate_sd'])}, select learns {pct(s['select_learn'])}, MUX change "
                 f"{pct(s['mux_change_add'] + s['mux_change_del'])}" if 'rate_sd' in s else '') + f"; repro {r['repro_abs_diff']:.1e}; {r['seconds']:.0f} s")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
