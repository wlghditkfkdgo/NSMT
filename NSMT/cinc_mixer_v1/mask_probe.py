"""Pre-check (no training, validation only): which part of the membrane difference does the trained LIF charge-difference
Neo-mixer (mixlinmx_lif, "③": m = LIF(BN(W [H_f - theta; H_r - theta; |H_f - H_r|]))) actually use? (PROJECT_LOG 2026-10-26)

Question behind it: a graded-spike fusion m = LIF(BN(W [F; R; |H_f - H_r| * g])) with g = XOR / AND / OR of the two directions'
spikes keeps the difference only where a spike event happened. Before training it, measure on the 15 trained ③ checkpoints
(CinC folds 0-4 x seeds 42 / 7 / 2026) how much of their validation score survives when the difference block d = |H_f - H_r|
is kept only in one region of the (F, R) pattern, the rest set to 0:
    full      d                    (reproduces the stored best validation score, 1e-6, fail closed)
    or        d * OR(F, R)         at least one direction fired
    xor       d * XOR(F, R)        exactly one fired
    and       d * AND(F, R)        both fired
    silent    d * (1 - OR)         neither fired (the sub-threshold differences)
    zero      0                    no difference block
    allor     [H_f * OR; H_r * OR; d * OR]   every analog input only where a spike happened
    rand_or   d * a random mask with the same share of entries as OR in that batch   (controls for HOW MUCH is kept)
    rand_sil  d * a random mask with the same share as the silent region             (fixed generator, seed 0)
F = (H_f - theta >= 0), R likewise -- the spikes of the last mixer layers, exactly as the model emits them. The models were trained
on the full d, so masked inputs are outside their training distribution: the drops are pessimistic and show which region the
trained fusion relies on, not what a model trained with the gate would reach.
Also reported: the share of (token, unit) entries and of the total |H_f - H_r| in each region.
Code: this folder is a snapshot of NSMT/nsmt_flif_v1/classification at commit 9cb3d3e95 (branch exp/nsmt-flif-neo); checkpoints and
the CinC cache are read from the main tree, nothing is written there.
usage: python mask_probe.py [-nd <gpu>] [--out results/mask-probe-20261026]
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
MAIN = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1/classification'
from spikingjelly.clock_driven import functional                     # noqa: E402
from load_model import LOAD_MODEL                                    # noqa: E402
import rhythm_data as rd                                             # noqa: E402
import rhythm_eval as re_                                            # noqa: E402

VARIANTS = ('full', 'or', 'xor', 'and', 'silent', 'zero', 'allor', 'rand_or', 'rand_sil')
SEEDS = (42, 7, 2026)
SUITES = {0: ['flif-v1-rhythm-mem-cinc/raw.txt'], 1: ['flif-v1-rhythm-folds-cinc/raw.txt', 'flif-v1-rhythm-folds-cinc/raw_recovered.txt']}


def runs():
    found = {}
    for key, files in SUITES.items():
        for name in files:
            for line in open(os.path.join(MAIN, 'results', name)):
                if not line.startswith('RESULT '):
                    continue
                d = dict(re.findall(r'(\S+?)=(\S+)', line))
                if d['cond'] != 'mixlinmx_lif' or (int(d['fold']) == 0) != (key == 0):
                    continue
                k = (int(d['fold']), int(d['seed']))
                if k in found:
                    raise SystemExit(f'{k} twice')
                found[k] = d['dir']
    need = [(f, s) for f in range(5) for s in SEEDS]
    if sorted(found) != sorted(need):
        raise SystemExit(f'runs found {sorted(found)}')
    return found


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def masked_mix(blk, variant, stats, gen=None):
    def mix(x):
        f, hf = blk.mixer.forward_charge(x)
        r, hr = blk.mixer_rev.forward_charge(x.flip(0))
        r, hr = r.flip(0), hr.flip(0)
        d = (hf - hr).abs()
        OR, AND = torch.maximum(f, r), f * r
        XOR = OR - AND
        silent = 1. - OR
        if stats is not None:                                        # entry shares and |d| mass per region
            tot = float(d.sum())
            for name, m in (('or', OR), ('xor', XOR), ('and', AND), ('silent', silent)):
                stats[name][0] += float(m.sum()); stats[name][1] += float((d * m).sum())
            stats['n'] += d.numel(); stats['mass'] += tot
        if variant == 'allor':
            return blk.fuse(torch.cat([hf * OR, hr * OR, d * OR], dim=-1))
        if variant in ('rand_or', 'rand_sil'):                       # same share of entries as the region, random positions
            p = float((OR if variant == 'rand_or' else silent).mean())
            g = (torch.rand(d.shape, generator=gen, device=d.device) < p).to(d.dtype)
            return blk.fuse(torch.cat([hf, hr, d * g], dim=-1))
        g = {'full': 1., 'or': OR, 'xor': XOR, 'and': AND, 'silent': silent, 'zero': 0.}[variant]
        return blk.fuse(torch.cat([hf, hr, d * g], dim=-1))
    return mix


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('-nd', '--num_device', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'mask-probe-20261026b'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    dev = torch.device(f'cuda:{a.num_device}')
    rec = {'script_sha256': sha256(__file__), 'snapshot_of': 'NSMT/nsmt_flif_v1/classification @ 9cb3d3e95', 'runs': {}}
    for (fold, seed), d in sorted(runs().items()):
        t0 = time.time()
        cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
        args = SimpleNamespace(**cfg)
        if args.mixer != 'lif' or args.mixer_comb != 'linmx' or int(args.fold) != fold or int(args.seed) != seed:
            raise SystemExit(f'{d}: config {args.mixer} {args.mixer_comb} fold {args.fold} seed {args.seed}')
        args.device = dev
        m = LOAD_MODEL[args.model](args, True)
        ck = os.path.join(d, 'model_state', 'checkpoint.pth')
        m.load_state_dict(torch.load(ck, map_location='cpu'))
        m = m.to(dev).eval(); m.train_mode = 'testing'
        blk = m.blocks[0]
        orig = blk.mix
        ds = rd.build(args, int(args.seed))['val']
        row = {'run_dir': d, 'checkpoint_sha256': sha256(ck), 'best_val_score_config': float(cfg['best_val_score']), 'variants': {}}
        stats = None
        for v in VARIANTS:
            stats = {k: [0., 0.] for k in ('or', 'xor', 'and', 'silent')} | {'n': 0, 'mass': 0.} if v == 'full' else None
            gen = torch.Generator(device=dev); gen.manual_seed(0)
            blk.mix = masked_mix(blk, v, stats, gen)
            with torch.no_grad():
                P, Y, units = re_.predict(m, ds, dev, bs=int(getattr(args, 'eval_bs', 32)))
            met, _ = re_.metrics(P, Y, args.data)
            row['variants'][v] = {'score': met['score'], 'macro_f1': met['macro_f1'], 'f1': [met[f'f1_{i}'] for i in range(4)]}
            if v == 'full':
                row['regions'] = {k: {'entry_share': stats[k][0] / stats['n'], 'abs_diff_share': stats[k][1] / stats['mass']}
                                  for k in ('or', 'xor', 'and', 'silent')}
        blk.mix = orig
        diff = abs(row['variants']['full']['score'] - row['best_val_score_config'])
        row['repro_abs_diff'] = diff
        if diff > 1e-6:
            raise SystemExit(f'fold {fold} seed {seed}: full variant {row["variants"]["full"]["score"]} vs stored {row["best_val_score_config"]}')
        row['seconds'] = time.time() - t0
        rec['runs'][f'f{fold}/s{seed}'] = row
        print(f'[mask] fold {fold} seed {seed}: ' + ' '.join(f"{v} {100 * row['variants'][v]['score']:.2f}" for v in VARIANTS)
              + f" | entries OR {100 * row['regions']['or']['entry_share']:.1f}% silent {100 * row['regions']['silent']['entry_share']:.1f}%"
              + f" | |d| in silent {100 * row['regions']['silent']['abs_diff_share']:.1f}% | {row['seconds']:.0f} s", flush=True)
        del m; torch.cuda.empty_cache()
    keys = sorted(rec['runs'])
    summ = {}
    for v in VARIANTS:
        sc = np.array([rec['runs'][k]['variants'][v]['score'] for k in keys]); mf = np.array([rec['runs'][k]['variants'][v]['macro_f1'] for k in keys])
        full = np.array([rec['runs'][k]['variants']['full']['score'] for k in keys])
        summ[v] = {'score_mean': float(sc.mean()), 'score_sd': float(sc.std(ddof=1)), 'drop_mean': float((sc - full).mean()),
                   'drop_sd': float((sc - full).std(ddof=1)), 'macro_f1_mean': float(mf.mean())}
    regions = {k: {s: float(np.mean([rec['runs'][r]['regions'][k][s] for r in keys])) for s in ('entry_share', 'abs_diff_share')}
               for k in ('or', 'xor', 'and', 'silent')}
    rec['summary'] = summ; rec['regions_mean'] = regions
    json.dump(rec, open(os.path.join(a.out, 'mask_probe.json'), 'x'), indent=1)
    lines = ['variant   val score (mean ± sd, 15 runs)   change vs full (mean ± sd)   macro-F1']
    for v in VARIANTS:
        s = summ[v]
        lines.append(f"{v:<8} {100 * s['score_mean']:6.2f} ± {100 * s['score_sd']:.2f}            {100 * s['drop_mean']:+6.2f} ± {100 * s['drop_sd']:.2f}            {100 * s['macro_f1_mean']:.2f}")
    lines.append('regions (mean over runs): ' + ' | '.join(f"{k}: entries {100 * v['entry_share']:.1f}%, |d| {100 * v['abs_diff_share']:.1f}%" for k, v in regions.items()))
    open(os.path.join(a.out, 'mask_probe.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
