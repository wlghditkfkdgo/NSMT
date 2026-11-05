"""Token-step profile of the membrane difference and the s_d gate (2026-11-05, PROJECT_LOG, exp/f-lif-pop-v3; user asked whether the
membrane difference is computed at every spiking step, then "진행" on measuring where s_d fires in trained checkpoints). No training,
report only, nothing here selects a model; no label is used by the profile itself (labels only reproduce the logged metric).

Why: at token step t the forward stream has seen t + 1 tokens and the backward stream N - t tokens, so |H_f - H_r| can reflect how many
tokens each direction has seen, not only what the tokens were; the s_d neuron also starts from rest at t = 0.
Prediction written before any result (PROJECT_LOG 11-05 message): under shuffled tokens the difference is larger at both ends than in
the middle, over at most 3 tokens per end for LIF and at least 8 tokens for the population neurons; s_d is low at the first step despite
a large difference (its own neuron starts from rest).

Checkpoints (seeds 42 / 7 / 2026; same selection, splits and reproduced metrics as bottleneck_probe.py):
    cinc    fold 0, best-validation weights, val split: muxlin LIF; sdlin LIF, f-LIF, population LIF, population f-LIF, soma-less population
            LIF, soma-less population f-LIF (sdlin has the same difference and s_d neuron; muxlin was trained with LIF only on these)
    ett     muxlin LIF tuned (IAND, 1 mixer layer), ETTh2 / ETTm2 x H96 / H720, test split
    mitbih  muxlin LIF tuned final (ADD, 2 mixer layers, whole training file), test file
Measured at every token step t, averaged over batch, repeats and channels, with the real weights and the model's own arithmetic:
    x      input spike rate of the mixer       f, r, or, and   F, R, F v R, F ^ R spike rates
    hf, hr mean H_f - theta, H_r - theta       d               mean |H_f - H_r|
    z      mean BN output fed to the s_d neuron                sd      s_d spike rate      m    mixer output spike rate
Conditions:
    orig   the real token order (every batch; this pass produces the model output, so the logged metric is reproduced with it)
    shuf   tokens permuted inside the mixer by a new permutation per batch (generator seed 0, its output discarded); values are indexed by
           position in the permuted sequence, so the token content is flat over positions and what remains is position-in-sequence alone
Fail-closed checks: logged metric reproduced (1e-6) with the probe in place, after the shuffled pass has run first on the same batches
(a state left by it would break this); first-batch output bitwise equal to the model's mix(); every step counted equally; finite values.
Summary quantities: ratios to the central half [N // 4, N - N // 4); end length = steps from an end until the value first comes within
10 % of the central-half mean.
usage: python sd_step_probe.py --task cinc|ett|mitbih --cond <cond>    (one part per call, device from -nd or cpu)
       python sd_step_probe.py --summarize                             -> results/sd-step-probe-20261105/summary.md
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottleneck_probe as bp                                              # noqa: E402  (imports spikingjelly before timm, then load_model)
from load_model import LOAD_MODEL                                          # noqa: E402
from mux_probe import sha256                                               # noqa: E402

OUT = os.environ.get('SD_STEP_OUT') or os.path.join(HERE, 'results', 'sd-step-probe-20261105')   # env: test runs only
bp.CINC.update({'pop_flif': ('flif-v1-rhythm-spk-cinc', 'spk_pop_flif_sdlin_or'), 'popg_lif': ('flif-v1-rhythm-spk-cinc', 'spk_popg_lif_sdlin_or'),
                'popg_flif': ('flif-v1-rhythm-spk-cinc', 'spk_popg_flif_sdlin_or')})
CINC_CONDS = ('muxlin', 'sdlin', 'flif', 'pop_lif', 'pop_flif', 'popg_lif', 'popg_flif')
NAMES = {'muxlin': 'muxlin LIF', 'sdlin': 'sdlin LIF', 'flif': 'sdlin f-LIF', 'pop_lif': 'sdlin population LIF', 'pop_flif': 'sdlin population f-LIF',
         'popg_lif': 'sdlin soma-less population LIF', 'popg_flif': 'sdlin soma-less population f-LIF'}
KEYS = ('x', 'f', 'r', 'or', 'and', 'hf', 'hr', 'd', 'z', 'sd', 'm')
SHUF_BATCHES = 60                                                           # ett / mitbih: shuffled pass on about this many spread batches


class Acc:
    def __init__(self, N):
        self.N, self.s, self.rows = N, {k: torch.zeros(N, dtype=torch.float64) for k in KEYS}, 0

    def add(self, vals):
        for k, v in vals.items():
            if v.shape[0] != self.N or v.ndim != 4:
                raise SystemExit(f'{k}: shape {tuple(v.shape)}, expected [{self.N}, B, K, D]')
            self.s[k] += v.double().sum(dim=(1, 2, 3)).cpu()
        v = vals['sd']
        self.rows += v.shape[1] * v.shape[2] * v.shape[3]

    def result(self):
        if self.rows == 0:
            raise SystemExit('nothing recorded')
        out = {k: (self.s[k] / self.rows).tolist() for k in KEYS}
        if not all(np.isfinite(out[k]).all() for k in KEYS):
            raise SystemExit('non-finite profile')
        return out


def measure(blk, xm):
    """The arithmetic of spk_mixer.SpkBlock.mix (muxlin / sdlin) on xm [N, B, K, D], with its intermediate values."""
    if blk.comb not in ('muxlin', 'sdlin'):
        raise SystemExit(f'comb {blk.comb}: no s_d gate to profile')
    f, hf = blk.mixer.forward_charge(xm)
    r, hr = blk.mixer_rev.forward_charge(xm.flip(0))
    r, hr = r.flip(0), hr.flip(0)
    if hf.shape != xm.shape or hr.shape != xm.shape:
        raise SystemExit(f'charge shape {tuple(hf.shape)} / {tuple(hr.shape)}, input {tuple(xm.shape)}')
    d = (hf - hr).abs()
    N, B, K, D = d.shape
    z = blk.sd_bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous()
    sd = blk.sd_neuron(z)
    fr, an = torch.maximum(f, r), torch.minimum(f, r)
    L = blk.fuse(torch.cat([fr, sd], dim=-1))
    m = sd * L + (1. - sd) * an if blk.comb == 'muxlin' else L
    return m, {'x': xm, 'f': f, 'r': r, 'or': fr, 'and': an, 'hf': hf, 'hr': hr, 'd': d, 'z': z, 'sd': sd, 'm': m}


def run_part(task, cond, dev):
    os.makedirs(os.path.join(OUT, 'parts'), exist_ok=True)
    path = os.path.join(OUT, 'parts', f'{task}_{cond}.json')
    if os.path.exists(path):
        raise SystemExit(f'{path} exists (not overwritten)')
    rows = bp.cinc_runs(cond) if task == 'cinc' else bp.ett_runs() if task == 'ett' else bp.mitbih_runs()
    if os.environ.get('SD_STEP_FIRST'):                                     # test runs only: the first checkpoint
        rows = rows[:1]
    rec = {'script_sha256': sha256(__file__), 'task': task, 'cond': cond, 'device': dev, 'runs': []}
    for r in rows:
        t0 = time.time()
        d = r['dir']
        cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
        args = argparse.Namespace(**cfg)
        args.device = dev
        inner = LOAD_MODEL[args.model](args, True)
        stride = 1
        if task == 'ett':
            from forecast_wrap import Forecaster
            from train_forecast import data_args
            from data_provider.data_factory import data_provider
            model = Forecaster(inner, args.embed_dim, args.pred_len, bias=args.bias)
            ck = os.path.join(d, 'model_state', 'checkpoint.pth')
            _, dl = data_provider(data_args(args, args), 'test')
            stride = max(1, len(dl) // SHUF_BATCHES)
            evaluate = lambda: bp.eval_ett(model, args, dev, dl)
            sp, key, want = 'test', 'mse', float(r['mse'])
        elif task == 'cinc':
            model = inner
            ck = os.path.join(d, 'model_state', 'checkpoint.pth')
            evaluate = lambda: bp.eval_cinc(model, args, r['seed'], dev, ('val',))
            sp, key, want = 'val', 'score', float(cfg['best_val_score'])
        else:
            from dataloader import create_loader
            model = inner
            ck = os.path.join(d, 'model_state', 'best+model.pt')
            loader, _ = create_loader(train=False, num_classes=args.num_classes, batch_size=args.batch_size,
                                      data_root=args.root_path + '/5class_mitbih_test_3_187.csv', data_window_size=187,
                                      num_workers=args.num_workers)
            stride = max(1, len(loader) // SHUF_BATCHES)
            evaluate = lambda: bp.eval_mitbih(model, args, dev, loader)
            sp, key, want = 'test', 'f1', float(r['f1'])
        model.load_state_dict(torch.load(ck, map_location='cpu'))
        model = model.to(dev).eval()
        model.train_mode = 'testing'
        m = model.inner if task == 'ett' else model
        if m.__class__.__name__ != 'SpkMixer' or len(m.blocks) != 1:
            raise SystemExit(f'{d}: not a one-block SpkMixer')
        blk, N = m.blocks[0], m.num_patches
        orig_mix = blk.mix
        acc = {'orig': Acc(N), 'shuf': Acc(N)}
        gen = torch.Generator().manual_seed(0)
        state = {'batch': 0, 'bitwise': None}

        def probe_mix(x):                                                   # x [T_h, B, N, D]
            if x.shape[2] != N:
                raise SystemExit(f'{x.shape[2]} tokens, expected {N}')
            if state['batch'] % stride == 0:                                # shuffled pass first: a state left by it would break the metric
                p = torch.randperm(N, generator=gen).to(x.device)
                if not torch.equal(p.sort().values, torch.arange(N, device=x.device)):
                    raise SystemExit('not a permutation')
                _, v = measure(blk, x[:, :, p].permute(2, 1, 0, 3).contiguous())
                acc['shuf'].add(v)
            mm, v = measure(blk, x.permute(2, 1, 0, 3).contiguous())
            acc['orig'].add(v)
            out = mm.permute(2, 1, 0, 3).contiguous()
            if state['batch'] == 0:
                state['bitwise'] = bool(torch.equal(out, orig_mix(x)))
            state['batch'] += 1
            return out

        blk.mix = probe_mix
        met = evaluate()
        blk.mix = orig_mix
        got = met[sp][key]
        if abs(got - want) > 1e-6:
            raise SystemExit(f'{d}: {sp} {key} {got:.6f} != logged {want:.6f} on {dev}')
        if state['bitwise'] is not True:
            raise SystemExit(f'{d}: first-batch output differs from mix()')
        rec['runs'].append({'dir': d, 'seed': int(r['seed']), 'ds': r.get('ds'), 'pl': int(r['pl']) if 'pl' in r else None, 'N': N, 'comb': blk.comb,
                            'checkpoint_sha256': sha256(ck), 'metric': {'split': sp, 'name': key, 'value': got, 'logged': want},
                            'batches': state['batch'], 'shuf_stride': stride, 'orig': acc['orig'].result(), 'shuf': acc['shuf'].result(),
                            'seconds': time.time() - t0})
        print(f"[sd-step] {task} {cond} {r.get('ds', '')} {r.get('pl', '')} seed {int(r['seed'])}: {key} {got:.6f} (logged {want:.6f}) "
              f"batches {state['batch']}, stride {stride} | {time.time() - t0:.0f} s", flush=True)
        del model, inner
    json.dump(rec, open(path, 'x'), indent=1)
    print(f'wrote {path}')


def central(N):
    return N // 4, N - N // 4


def ratio(v, a, b):
    lo, hi = central(len(v))
    mid = float(np.mean(v[lo:hi]))
    return float(np.mean(v[a:b])) / mid if mid else float('nan')


def end_length(v):
    """Steps from each end until the value first comes within 10 % of the central-half mean (left, right); capped at the central half."""
    lo, hi = central(len(v))
    mid = float(np.mean(v[lo:hi]))
    if mid == 0:
        return None, None
    close = np.abs(np.asarray(v) / mid - 1.) <= .1
    left = next((t for t in range(lo) if close[t]), lo)
    right = next((t for t in range(lo) if close[len(v) - 1 - t]), lo)
    return left, right


def fmt(x, n=2):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.{n}f}'


def group(parts):
    """{(label, N): [run, ...]} over the part files; ETT cells stay separate."""
    out = {}
    for task, cond, rec in parts:
        for run in rec['runs']:
            label = {'cinc': f'CinC {NAMES.get(cond, cond)}', 'mitbih': 'MIT-BIH muxlin LIF'}.get(task, f"{run['ds']} H{run['pl']} muxlin LIF")
            out.setdefault((label, run['N']), []).append(run)
    return out


def mean_profile(runs, cond, key):
    return np.mean([r[cond][key] for r in runs], axis=0)


def summarize():
    parts = []
    for p in sorted(glob.glob(os.path.join(OUT, 'parts', '*.json'))):
        rec = json.load(open(p))
        parts.append((rec['task'], rec['cond'], rec))
    if not parts:
        raise SystemExit('no parts')
    order = {'cinc': 0, 'ett': 1, 'mitbih': 2}
    parts.sort(key=lambda t: (order[t[0]], CINC_CONDS.index(t[1]) if t[0] == 'cinc' else 0))
    G = group(parts)
    for (label, N), runs in G.items():
        if len(runs) != 3:
            raise SystemExit(f'{label}: {len(runs)} checkpoints, expected 3')
    L = ['# s_d gate by token step (sd_step_probe.py)', '',
         'No training. Report only. Mean over 3 seeds; (a..b) = lowest and highest seed. `shuf` = tokens permuted inside the mixer (a new permutation per '
         'batch), so token content is flat over positions and what remains is position in the sequence alone. `orig` = real token order.',
         'Ratios are to the central half of the tokens. first k = the first k steps, last k = the last k. end length = steps from that end until the value '
         'first comes within 10 % of the central-half mean (capped at N // 4).', '',
         f"Checks passed in every part: logged metric reproduced (1e-6) with the shuffled pass running first; first-batch output bitwise equal to mix(). "
         f"Device: {', '.join(sorted({r['device'] for _, _, r in parts}))}.", '']
    for cond in ('shuf', 'orig'):
        for key, name in (('d', 'mean |H_f - H_r|'), ('sd', 's_d spike rate')):
            L += [f'## {name}, {cond}', '', f"| model | N | central half {'(%)' if key == 'sd' else ''} | first 1 | first 2 | first 4 | first 8 | last 1 | last 2 | last 4 | last 8 | end length first / last |",
                  '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
            for (label, N), runs in G.items():
                lo, hi = central(N)
                mids = [float(np.mean(r[cond][key][lo:hi])) for r in runs]
                scale = 100. if key == 'sd' else 1.
                row = [label, str(N), f'{np.mean(mids) * scale:.3f} ({min(mids) * scale:.3f}..{max(mids) * scale:.3f})']
                for side in ('first', 'last'):
                    for k in (1, 2, 4, 8):
                        if k > N // 4:
                            row.append('')
                            continue
                        a, b = (0, k) if side == 'first' else (N - k, N)
                        vals = [ratio(r[cond][key], a, b) for r in runs]
                        row.append(f'{np.mean(vals):.2f} ({min(vals):.2f}..{max(vals):.2f})')
                el = end_length(mean_profile(runs, cond, key))
                row.append(f'{el[0]} / {el[1]}')
                L.append('| ' + ' | '.join(row) + ' |')
            L.append('')
    L += ['## share of all s_d spikes in the two ends, against an even spread', '',
          '| model | N | k per end | share shuf (%) | share orig (%) | even (%) |', '|---|--:|--:|--:|--:|--:|']
    for (label, N), runs in G.items():
        for k in (1, 2, 4, 8):
            if k > N // 4:
                continue
            sh = {}
            for cond in ('shuf', 'orig'):
                v = mean_profile(runs, cond, 'sd')
                sh[cond] = 100. * (v[:k].sum() + v[N - k:].sum()) / v.sum() if v.sum() else float('nan')
            L.append(f"| {label} | {N} | {k} | {fmt(sh['shuf'], 1)} | {fmt(sh['orig'], 1)} | {200. * k / N:.1f} |")
    L += ['', '## profile by step (mean over seeds; steps 0..7, the central-half mean, steps N-8..N-1 for N > 24; every step otherwise)', '']
    for (label, N), runs in G.items():
        L += [f'### {label}, N = {N}', '']
        idx = list(range(N)) if N <= 24 else list(range(8)) + ['mid'] + list(range(N - 8, N))
        head = [str(i) for i in idx]
        L += ['| value | ' + ' | '.join(head) + ' |', '|---|' + '--:|' * len(idx)]
        for cond in ('shuf', 'orig'):
            for key, nm, sc, pr in (('x', 'input rate %', 100., 2), ('d', '|H_f-H_r|', 1., 3), ('sd', 's_d %', 100., 2), ('or', 'F v R %', 100., 2), ('and', 'F ^ R %', 100., 2), ('m', 'output %', 100., 2)):
                v = mean_profile(runs, cond, key) * sc
                lo, hi = central(N)
                cells = [f'{(v[i] if i != "mid" else v[lo:hi].mean()):.{pr}f}' for i in idx]
                L.append(f'| {cond} {nm} | ' + ' | '.join(cells) + ' |')
        L.append('')
    # sections below added after the parts were measured (the parts keep the sha of the measuring version); they only read the parts
    L += ['## warm-up under shuffled tokens: first step reaching half of the central-half mean (one value per seed; F, s_d, output counted from the first token, R from the last)', '',
          '| model | N | F | R | s_d | output | mean H_f - H_r at step 0, 8, N // 2, N - 9, N - 1 |', '|---|--:|--:|--:|--:|--:|--:|']
    for (label, N), runs in G.items():
        lo, hi = central(N)

        def t50(v):
            mid = float(np.mean(v[lo:hi]))
            return next((t for t in range(N) if v[t] >= .5 * mid), None) if mid > 0 else None
        seeds = lambda k, rev=False: ', '.join(str(t50(r['shuf'][k][::-1] if rev else r['shuf'][k])) for r in runs)
        sd = np.mean([np.asarray(r['shuf']['hf']) - np.asarray(r['shuf']['hr']) for r in runs], axis=0)
        L.append(f"| {label} | {N} | {seeds('f')} | {seeds('r', True)} | {seeds('sd')} | {seeds('m')} | "
                 + ' '.join(f'{sd[i]:+.2f}' for i in (0, 8, N // 2, N - 9, N - 1)) + ' |')
    L += ['', '## correlation over token positions between input rate and s_d rate, real order (mean of the seeds; position means only, not per sample)', '',
          '| model | N | r |', '|---|--:|--:|']
    for (label, N), runs in G.items():
        L.append(f"| {label} | {N} | {np.mean([np.corrcoef(r['orig']['x'], r['orig']['sd'])[0, 1] for r in runs]):+.3f} |")
    L += ['', '## context limit from bottleneck-probe-20261104, for reading against the warm-up (CinC validation score, difference from base in %p, mean of 3 seeds)', '',
          'ctxW assumed the restart transient falls inside the W discarded tokens. That holds where the warm-up above is shorter than W.', '',
          '| model | base | k16 | ctx4 | ctx16 | ctx64 |', '|---|--:|--:|--:|--:|--:|']
    for c, nm in (('muxlin', 'muxlin LIF'), ('sdlin', 'sdlin LIF'), ('flif', 'sdlin f-LIF'), ('pop_lif', 'sdlin population LIF')):
        runs = json.load(open(os.path.join(HERE, 'results', 'bottleneck-probe-20261104', 'parts', f'cinc_{c}.json')))['runs']
        val = lambda iv: 100. * float(np.mean([r['metrics'][iv]['val']['score'] for r in runs]))
        L.append(f"| {nm} | {val('base'):.2f} | " + ' | '.join(f"{val(iv) - val('base'):+.2f}" for iv in ('k16', 'ctx4', 'ctx16', 'ctx64')) + ' |')
    L.append('')
    t = sum(r['seconds'] for _, _, rec in parts for r in rec['runs'])
    L += [f'Checkpoints: {sum(len(r) for r in G.values())}; probe time {t / 60:.0f} min in total. Not run: AD, qmx checkpoints (still training), retraining with a changed gate.', '']
    path = os.path.join(OUT, 'summary.md')
    open(path, 'w').write('\n'.join(L))
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['cinc', 'ett', 'mitbih'])
    ap.add_argument('--cond')
    ap.add_argument('-nd', '--num_device', type=int, default=-1)
    ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    if a.task is None or a.cond is None:
        raise SystemExit('--task and --cond are required')
    dev = f'cuda:{a.num_device}' if a.num_device >= 0 else 'cpu'
    with torch.no_grad():
        run_part(a.task, a.cond, dev)


if __name__ == '__main__':
    main()
