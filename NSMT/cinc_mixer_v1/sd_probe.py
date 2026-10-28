"""Post-hoc probe of the trained spiking-difference Neo-mixers (PROJECT_LOG 2026-10-28; not pre-registered, no training,
validation split only): what does s_d = LIF_d(BN_d(|H_f - H_r|)) carry, and how much of the validation score depends on it?

Checkpoints: the 18 runs of results/flif-v1-rhythm-sd-cinc/raw.txt (mixlinsd_<n>, n in lif / flif / pop_lif / pop_flif / popg_lif /
popg_flif, CinC fold 0, seeds 42 / 7 / 2026); model_state/checkpoint.pth = the best-validation weights. The fusion input of block 0 is
[F; R; s_d] (F, R = last mixer layers' spikes of the two directions). Variants of s_d (the models were trained on the real s_d, so the
changed inputs are outside their training distribution: drops are pessimistic and show reliance, not what a retrained model would reach):
    full     s_d as trained (reproduces the stored best validation score, 1e-6, fail closed)
    zero     s_d = 0                                   (no difference spikes)
    perm     s_d with its token order shuffled in each window, the same order for all units (keeps every unit's spike count and the
             pattern across units, breaks the alignment with F / R in time)
    rate     s_d replaced by Bernoulli spikes at each (window, unit)'s own firing rate (keeps only how many spikes each unit sends)
    zero_fr  F = R = 0, s_d kept                        (only the difference spikes reach the fusion)
Random variants use a fixed generator (seed 0) per checkpoint and variant.
Statistics of the full model on the validation windows:
    input firing rates of F, R, s_d; per-unit s_d rate (share of units below 0.1%, above 50%)
    BN_d: gain gamma and offset beta; for gamma > 0 the constant-input firing point of d in units of its running std,
          z* = (1 - beta) / gamma (the spikingjelly LIF with decay_input reaches 1 only if its input is >= 1); share of gamma < 0
    s_d firing rate by z = (d - running mean) / running std, in bins (-inf, -1, 0, 1, 2, inf)
    fusion weight: share of sum_ij |W_ij| * rate_j carried by each input block (a proxy for how much input current each block gives)
usage: python sd_probe.py [-nd <gpu>] [--out results/sd-probe-20261028]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                     # noqa: E402,F401  (imported before timm, as in mask_probe.py)
from load_model import LOAD_MODEL                                    # noqa: E402
import rhythm_data as rd                                             # noqa: E402
import rhythm_eval as re_                                            # noqa: E402

VARIANTS = ('full', 'zero', 'perm', 'rate', 'zero_fr')
NS = ('lif', 'flif', 'pop_lif', 'pop_flif', 'popg_lif', 'popg_flif')
SEEDS = (42, 7, 2026)
ZBINS = (-1., 0., 1., 2.)


def runs():
    found = {}
    for line in open(os.path.join(HERE, 'results', 'flif-v1-rhythm-sd-cinc', 'raw.txt')):
        if not line.startswith('RESULT '):
            continue
        d = dict(re.findall(r'(\S+?)=(\S+)', line))
        k = (d['cond'].replace('mixlinsd_', ''), int(d['seed']))
        if k in found or int(d['fold']) != 0:
            raise SystemExit(f'{k}: twice or not fold 0')
        found[k] = d['dir']
    need = [(n, s) for n in NS for s in SEEDS]
    if sorted(found) != sorted(need):
        raise SystemExit(f'runs found {sorted(found)}')
    return found


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sd_mix(blk, variant, st, gen):
    bn = blk.sd_bn
    mu, sdv = bn.running_mean, torch.sqrt(bn.running_var + bn.eps)

    def mix(x):
        f, hf = blk.mixer.forward_charge(x)
        r, hr = blk.mixer_rev.forward_charge(x.flip(0))
        r, hr = r.flip(0), hr.flip(0)
        d = (hf - hr).abs()
        N, B, K, D = d.shape
        sd = blk.sd_neuron(bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous())
        if st is not None:                                           # full model: statistics
            st['n'] += N * B * K
            st['F'] += float(f.sum()); st['R'] += float(r.sum()); st['S'] += float(sd.sum())
            st['unit'] += sd.sum((0, 1, 2)).double()
            z = ((d - mu) / sdv).reshape(-1, D)
            b = torch.bucketize(z, torch.tensor(ZBINS, device=z.device))
            s = sd.reshape(-1, D)
            for i in range(len(ZBINS) + 1):
                m = b == i
                st['zcnt'][i] += float(m.sum()); st['zspk'][i] += float(s[m].sum())
            st['rate_FRS'] += torch.cat([f.sum((0, 1, 2)), r.sum((0, 1, 2)), sd.sum((0, 1, 2))]).double()
        if variant == 'zero':
            sd = torch.zeros_like(sd)
        elif variant == 'perm':
            idx = torch.argsort(torch.rand(B, N, generator=gen, device=d.device), dim=1).t()   # [N, B]
            sd = torch.gather(sd, 0, idx[:, :, None, None].expand_as(sd))
        elif variant == 'rate':
            p = sd.mean(0, keepdim=True).expand_as(sd)
            sd = (torch.rand(sd.shape, generator=gen, device=d.device) < p).to(sd.dtype)
        elif variant == 'zero_fr':
            f, r = torch.zeros_like(f), torch.zeros_like(r)
        return blk.fuse(torch.cat([f, r, sd], dim=-1))
    return mix


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('-nd', '--num_device', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'sd-probe-20261028'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    dev = torch.device(f'cuda:{a.num_device}')
    rec = {'script_sha256': sha256(__file__), 'runs': {}}
    for (n, seed), d in sorted(runs().items(), key=lambda kv: (NS.index(kv[0][0]), SEEDS.index(kv[0][1]))):
        t0 = time.time()
        cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
        args = SimpleNamespace(**cfg)
        if args.mixer != n or args.mixer_comb != 'linsd' or int(args.fold) != 0 or int(args.seed) != seed:
            raise SystemExit(f'{d}: config {args.mixer} {args.mixer_comb} fold {args.fold} seed {args.seed}')
        args.device = dev
        m = LOAD_MODEL[args.model](args, True)
        ck = os.path.join(d, 'model_state', 'checkpoint.pth')
        m.load_state_dict(torch.load(ck, map_location='cpu'))
        m = m.to(dev).eval(); m.train_mode = 'testing'
        if len(m.blocks) != 1:
            raise SystemExit(f'{d}: {len(m.blocks)} blocks')
        blk = m.blocks[0]
        orig = blk.mix
        ds = rd.build(args, int(args.seed))['val']
        row = {'run_dir': d, 'checkpoint_sha256': sha256(ck), 'best_val_score_config': float(cfg['best_val_score']), 'variants': {}}
        D = blk.sd_bn.num_features
        for v in VARIANTS:
            st = None
            if v == 'full':
                st = dict(n=0, F=0., R=0., S=0., unit=torch.zeros(D, dtype=torch.float64, device=dev),
                          zcnt=[0.] * (len(ZBINS) + 1), zspk=[0.] * (len(ZBINS) + 1), rate_FRS=torch.zeros(3 * D, dtype=torch.float64, device=dev))
            gen = torch.Generator(device=dev); gen.manual_seed(0)
            blk.mix = sd_mix(blk, v, st, gen)
            with torch.no_grad():
                P, Y, _ = re_.predict(m, ds, dev, bs=int(getattr(args, 'eval_bs', 32)))
            met, _ = re_.metrics(P, Y, args.data)
            row['variants'][v] = {'score': met['score'], 'macro_f1': met['macro_f1'], 'f1': [met[f'f1_{i}'] for i in range(4)]}
            if v == 'full':
                unit = (st['unit'] / st['n']).cpu().numpy()
                g = blk.sd_bn.weight.detach().cpu().numpy(); b = blk.sd_bn.bias.detach().cpu().numpy()
                zs = (1 - b[g > 0]) / g[g > 0]
                W = blk.fuse.linear.weight.detach().abs().double()                       # [D, 3D]
                mass = (W * (st['rate_FRS'] / st['n'])[None, :]).sum(0).cpu().numpy()
                row['stats'] = dict(rate_F=st['F'] / (st['n'] * D), rate_R=st['R'] / (st['n'] * D), rate_sd=st['S'] / (st['n'] * D),
                                    sd_units_below_0p1pct=float((unit < 1e-3).mean()), sd_units_above_50pct=float((unit > .5).mean()),
                                    sd_unit_rate_quantiles=np.quantile(unit, [.1, .25, .5, .75, .9]).tolist(),
                                    gamma_neg_share=float((g < 0).mean()), gamma_median=float(np.median(g)), beta_median=float(np.median(b)),
                                    zstar_quantiles=np.quantile(zs, [.1, .25, .5, .75, .9]).tolist() if len(zs) else None,
                                    rate_by_z=[st['zspk'][i] / max(st['zcnt'][i], 1) for i in range(len(ZBINS) + 1)],
                                    share_by_z=[st['zcnt'][i] / sum(st['zcnt']) for i in range(len(ZBINS) + 1)],
                                    weight_current_share=[float(mass[k * D:(k + 1) * D].sum() / mass.sum()) for k in range(3)])
        blk.mix = orig
        diff = abs(row['variants']['full']['score'] - row['best_val_score_config'])
        row['repro_abs_diff'] = diff
        if diff > 1e-6:
            raise SystemExit(f'{n} seed {seed}: full {row["variants"]["full"]["score"]} vs stored {row["best_val_score_config"]}')
        row['seconds'] = time.time() - t0
        rec['runs'][f'{n}/s{seed}'] = row
        s = row['stats']
        print(f'[sd-probe] {n} seed {seed}: ' + ' '.join(f"{v} {100 * row['variants'][v]['score']:.2f}" for v in VARIANTS)
              + f" | rates F {100 * s['rate_F']:.2f}% R {100 * s['rate_R']:.2f}% s_d {100 * s['rate_sd']:.2f}% | {row['seconds']:.0f} s", flush=True)
        del m; torch.cuda.empty_cache()
    summ = {}
    for n in NS:
        keys = [f'{n}/s{s}' for s in SEEDS]
        full = np.array([rec['runs'][k]['variants']['full']['score'] for k in keys])
        summ[n] = {v: dict(score_mean=float(np.mean([rec['runs'][k]['variants'][v]['score'] for k in keys])),
                           change_mean=float(np.mean([rec['runs'][k]['variants'][v]['score'] for k in keys] - full)),
                           change_per_seed=[float(rec['runs'][k]['variants'][v]['score'] - rec['runs'][k]['variants']['full']['score']) for k in keys])
                   for v in VARIANTS}
        st = [rec['runs'][k]['stats'] for k in keys]
        summ[n]['stats_mean'] = {k: (np.mean([x[k] for x in st], 0).tolist() if st[0][k] is not None else None) for k in st[0]}
    rec['summary'] = summ
    json.dump(rec, open(os.path.join(a.out, 'sd_probe.json'), 'x'), indent=1)
    L = ['# s_d probe (validation, CinC fold 0, mean of seeds 42 / 7 / 2026)', '',
         '| neuron | full val score | zero | perm | rate | zero_fr | (change vs full, %p) |', '|---|---|---|---|---|---|---|']
    for n in NS:
        L.append(f"| {n} | {100 * summ[n]['full']['score_mean']:.2f} | " + ' | '.join(f"{100 * summ[n][v]['change_mean']:+.2f}" for v in VARIANTS[1:]) + ' | |')
    L += ['', '| neuron | rate F / R / s_d % | s_d units <0.1% / >50% | s_d unit rate q10 / q50 / q90 % | gamma<0 share | z* q25 / q50 / q75 | s_d rate by z bin (<-1, -1..0, 0..1, 1..2, >2) % | share of z bins % | weight x rate share F / R / s_d % |',
          '|---|---|---|---|---|---|---|---|---|']
    for n in NS:
        s = summ[n]['stats_mean']
        zq = s['zstar_quantiles']
        L.append(f"| {n} | {100 * s['rate_F']:.2f} / {100 * s['rate_R']:.2f} / {100 * s['rate_sd']:.2f} | {100 * s['sd_units_below_0p1pct']:.1f} / {100 * s['sd_units_above_50pct']:.1f} | "
                 f"{100 * s['sd_unit_rate_quantiles'][0]:.2f} / {100 * s['sd_unit_rate_quantiles'][2]:.2f} / {100 * s['sd_unit_rate_quantiles'][4]:.2f} | {100 * s['gamma_neg_share']:.1f} | "
                 + (f"{zq[1]:.2f} / {zq[2]:.2f} / {zq[3]:.2f}" if zq else 'n/a') + ' | '
                 + ' / '.join(f'{100 * x:.1f}' for x in s['rate_by_z']) + ' | ' + ' / '.join(f'{100 * x:.1f}' for x in s['share_by_z']) + ' | '
                 + ' / '.join(f'{100 * x:.1f}' for x in s['weight_current_share']) + ' |')
    open(os.path.join(a.out, 'sd_probe.md'), 'x').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
