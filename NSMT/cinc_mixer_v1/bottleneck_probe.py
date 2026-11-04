"""Bottleneck diagnostics of SpkMixer muxlin (2026-11-04, PROJECT_LOG, exp/f-lif-pop-v3; user: "선택된 세팅/모델에서 개선 가능성이
있거나 성능 문제의 원인인 설계 사항을 분석 ... 어떤 부분이 성능 개선의 병목인지 원인을 조사"). No training: interventions on block
0's token-axis mixer (SpkBlock.mix, the T = N part) of trained checkpoints; report only, nothing here selects a model.

Checkpoints (seeds 42 / 7 / 2026):
    cinc    fold 0, best-validation weights (model_state/checkpoint.pth); splits val (683 records) and test (1,706)
            muxlin        spk_lif_muxlin_or      results/flif-v1-rhythm-spk-mux-cinc   (the selected model; tuned = default settings)
            sdlin, or     spk_lif_{sdlin,or}_or  results/flif-v1-rhythm-spk-cinc       (fusion references, B1)
            pop_lif, flif spk_{pop_lif,flif}_sdlin_or                                  (neuron references, B2; longer-memory neurons)
    ett     muxlin tuned (summarize_ett_tune selections: IAND, 1 mixer layer, lr ETTh2 5e-4 / ETTm2 1e-4), ETTh2 / ETTm2 x H96 /
            H720, checkpoint.pth (best validation), test split, MSE on the scaled data (as RESULT mse)
    mitbih  muxlin tuned final (summarize_mitbih_tune selection: ADD, 2 layers, lr 0.02, whole training file), best+model.pt (last
            epoch), test file, macro-F1
Interventions on mix(x), x [T_h, B, N, D] -> m (the mixer output that the residual adds; N tokens = 117 / 24 / 11):
    base   unchanged (must reproduce the logged metrics: CinC val = config best_val_score, CinC test / ETT / MIT-BIH = RESULT, 1e-6)
    off    m = 0                                  (the residual passes the embedding unchanged; what the mixer contributes)
    shufS  a fixed token permutation (torch.randperm(N, seed S), S = 0, 1, 2) inside the mixer, outputs put back at their tokens
    rev    the token order reversed inside the mixer, outputs put back (the forward weights read the sequence backwards)
    kK     the whole mix() (both mixer directions, s_d neuron, fusion neuron) run on independent chunks of K tokens
           (K = 1, 4, 16, 64; K >= N skipped): every token-axis neuron restarts at each chunk -> the context the mixer can use
    ctxW   chunks of 4 tokens, each run on itself plus W tokens before and after (W = 4, 16, 64; W >= N skipped) and only its own
           4 outputs kept: every output token sees at most W tokens of context per direction, with the restart transients pushed
           into the discarded context (kK confounds 'no long context' with 'every chunk starts from rest')
    sd0 / sd1  (muxlin only) the MUX gate forced: m = F ^ R everywhere / m = L everywhere (L computed as usual)
Checks (fail closed): base reproduces; kN (one chunk) and ctxN (full context) are bitwise base; a second base call is bitwise the first (no state between
calls); the sd re-implementation with the true s_d is bitwise base.
usage: python bottleneck_probe.py --task cinc|ett|mitbih --cond <cond> [-nd gpu]    (one part file per call)
       python bottleneck_probe.py --summarize                                         ->  results/bottleneck-probe-20261104/summary.md
       python bottleneck_probe.py --energy                                            ->  .../energy.md (per-layer syops files)
"""
import argparse
import csv
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
sys.path.append(os.path.join(HERE, '..', 'forecasting'))
from spikingjelly.clock_driven import functional                          # noqa: E402  (before timm, as in mux_fuse_probe.py)
from load_model import LOAD_MODEL                                         # noqa: E402
from summarize_rhythm import parse                                        # noqa: E402
from mux_probe import sha256                                              # noqa: E402

OUT = os.environ.get('BOTTLENECK_OUT') or os.path.join(HERE, 'results', 'bottleneck-probe-20261104')   # env: test runs only
SEEDS = (42, 7, 2026)
KS = (1, 4, 16, 64)
WS, CTX_C = (4, 16, 64), 4
CINC = {'muxlin': ('flif-v1-rhythm-spk-mux-cinc', 'spk_lif_muxlin_or'), 'sdlin': ('flif-v1-rhythm-spk-cinc', 'spk_lif_sdlin_or'),
        'or': ('flif-v1-rhythm-spk-cinc', 'spk_lif_or_or'), 'pop_lif': ('flif-v1-rhythm-spk-cinc', 'spk_pop_lif_sdlin_or'),
        'flif': ('flif-v1-rhythm-spk-cinc', 'spk_flif_sdlin_or')}
PARTS = [('cinc', c) for c in CINC] + [('ett', 'muxlin'), ('mitbih', 'muxlin')]
LIGHT = ('pop_lif', 'flif')                     # slow neurons: validation split only, interventions LIGHT_IV
LIGHT_IV = ('base', 'base2', 'off', 'shuf0', 'rev', 'k1', 'k16', 'kN', 'ctx4', 'ctx16', 'ctx64', 'ctxN')


def cinc_runs(cond):
    suite, name = CINC[cond]
    rows = {int(r['seed']): r for r in parse(os.path.join(HERE, 'results', suite, 'raw.txt')) if r['cond'] == name and int(r['fold']) == 0}
    if sorted(rows) != sorted(SEEDS):
        raise SystemExit(f'cinc {cond}: seeds {sorted(rows)}')
    return [rows[s] for s in SEEDS]


def ett_runs():
    import summarize_ett_tune as et
    D = et.load()
    f = et.rows(D, et.MUX, 'iand', {'ETTh2': 0.0005, 'ETTm2': 0.0001}, 1)
    out = []
    for ds, pl in et.CELLS:
        v = f(ds, pl)
        if v is None:
            raise SystemExit(f'ett muxlin tuned {ds} {pl}: missing runs')
        out += v
    return out


def mitbih_runs():
    import summarize_mitbih_tune as sm
    sel = {s: json.load(open(os.path.join(sm.TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
    v = sm.seeds(sm.load(), sm.MUX, sm.tuned_cfg(sm.MUX, sel), -1)
    if v is None:
        raise SystemExit('mitbih muxlin tuned: missing runs')
    return v


def forced_mix(blk, val):
    """muxlin mix() with the gate forced (val 0 / 1) or, val None, the true s_d (re-implementation check)."""
    def mix(x):
        xm = x.permute(2, 1, 0, 3).contiguous()
        f, hf = blk.mixer.forward_charge(xm)
        r, hr = blk.mixer_rev.forward_charge(xm.flip(0))
        r, hr = r.flip(0), hr.flip(0)
        d = (hf - hr).abs(); N, B, K, D = d.shape
        sd = blk.sd_neuron(blk.sd_bn(d.reshape(-1, D)).reshape(N, B, K, D).contiguous())
        L = blk.fuse(torch.cat([torch.maximum(f, r), sd], dim=-1))
        if val is None:
            m = sd * L + (1. - sd) * torch.minimum(f, r)
        else:
            m = L if val == 1 else torch.minimum(f, r)
        return m.permute(2, 1, 0, 3).contiguous()
    return mix


def interventions(comb, N, light=False):
    if light:
        return [iv for iv in LIGHT_IV if not (iv[0] == 'k' and iv[1:].isdigit() and int(iv[1:]) >= N)]
    iv = ['base', 'base2', 'off', 'shuf0', 'shuf1', 'shuf2', 'rev'] + [f'k{k}' for k in KS if k < N] + ['kN']
    iv += [f'ctx{w}' for w in WS if w < N] + ['ctxN']
    return iv + (['sdtrue', 'sd0', 'sd1'] if comb == 'muxlin' else [])


def make_mix(blk, orig, iv, N):
    if iv in ('base', 'base2'):
        return orig
    if iv == 'off':
        return lambda x: torch.zeros_like(x)
    if iv.startswith('shuf') or iv == 'rev':
        p = torch.arange(N - 1, -1, -1) if iv == 'rev' else torch.randperm(N, generator=torch.Generator().manual_seed(int(iv[4:])))
        inv = torch.argsort(p)
        return lambda x: orig(x[:, :, p.to(x.device)])[:, :, inv.to(x.device)]
    if iv.startswith('k'):
        k = N if iv == 'kN' else int(iv[1:])
        return lambda x: torch.cat([orig(x[:, :, s:s + k]) for s in range(0, N, k)], dim=2)
    if iv.startswith('ctx'):                                               # chunks of CTX_C tokens, each run with W context tokens
        W = N if iv == 'ctxN' else int(iv[3:])                             # on both sides (forward warm-up before, backward after)
        def f(x):
            outs = []
            for s in range(0, N, CTX_C):
                e = min(N, s + CTX_C); a0, b0 = max(0, s - W), min(N, e + W)
                outs.append(orig(x[:, :, a0:b0])[:, :, s - a0:e - a0])
            return torch.cat(outs, dim=2)
        return f
    return forced_mix(blk, {'sdtrue': None, 'sd0': 0, 'sd1': 1}[iv])


@torch.no_grad()
def eval_cinc(model, args, seed, dev, splits=('val', 'test')):
    import rhythm_data as rd
    import rhythm_eval as re_
    b = rd.build(args, int(seed))
    out = {}
    for split in splits:
        P, Y, _ = re_.predict(model, b[split], dev, bs=int(getattr(args, 'eval_bs', 32)))
        met, _ = re_.metrics(P, Y, args.data)
        out[split] = {k: met[k] for k in ('score', 'macro_f1', 'f1_0', 'f1_1', 'f1_2', 'f1_3', 'acc')}
        out[split]['_P'] = P
    return out


@torch.no_grad()
def eval_ett(model, args, dev, dl):
    se, cnt = 0., 0
    functional.reset_net(model)
    for x, y, _, _ in dl:
        x = x.float().to(dev); y = y.float().to(dev)[:, -args.pred_len:, :]
        out = model(x)
        functional.reset_net(model)
        d = (out - y).double(); se += float((d ** 2).sum()); cnt += d.numel()
    return {'test': {'mse': se / cnt}}


@torch.no_grad()
def eval_mitbih(model, args, dev, loader):
    from torcheval.metrics import MulticlassF1Score
    f1 = MulticlassF1Score(num_classes=args.num_classes, average='macro').to(dev)
    f1c = MulticlassF1Score(num_classes=args.num_classes, average=None).to(dev)
    preds = []
    functional.reset_net(model)
    for batch in loader:
        x = batch[0].float().to(dev); y = batch[1].long().squeeze(-1).to(dev)
        out = model(x)
        out = out[0] if isinstance(out, tuple) else out
        functional.reset_net(model)
        f1.update(out.argmax(1), y); f1c.update(out.argmax(1), y); preds.append(out.argmax(1).cpu())
    per = f1c.compute().tolist()
    return {'test': {'f1': float(f1.compute()), **{f'f1_{i}': float(v) for i, v in enumerate(per)}, '_P': torch.cat(preds)}}


def run_part(task, cond, dev):
    os.makedirs(os.path.join(OUT, 'parts'), exist_ok=True)
    path = os.path.join(OUT, 'parts', f'{task}_{cond}.json')
    if os.path.exists(path):
        raise SystemExit(f'{path} exists (not overwritten)')
    rows = cinc_runs(cond) if task == 'cinc' else ett_runs() if task == 'ett' else mitbih_runs()
    if os.environ.get('BOTTLENECK_FIRST'):                                  # test runs only: the first checkpoint
        rows = rows[:1]
    rec ={'script_sha256': sha256(__file__), 'task': task, 'cond': cond, 'runs': []}
    for r in rows:
        t0 = time.time()
        d = r['dir']
        cfg = torch.load(os.path.join(d, 'model_state', 'config.pt'), map_location='cpu')
        args = SimpleNamespace(**cfg) if task != 'ett' else argparse.Namespace(**cfg)
        args.device = dev
        inner = LOAD_MODEL[args.model](args, True)
        if task == 'ett':
            from forecast_wrap import Forecaster
            from train_forecast import data_args
            from data_provider.data_factory import data_provider
            model = Forecaster(inner, args.embed_dim, args.pred_len, bias=args.bias)
            ck = os.path.join(d, 'model_state', 'checkpoint.pth')
            _, dl = data_provider(data_args(args, args), 'test')
            evaluate = lambda: eval_ett(model, args, dev, dl)
            ref = {'test': ('mse', float(r['mse']))}
        else:
            model = inner
            ck = os.path.join(d, 'model_state', 'checkpoint.pth' if task == 'cinc' else 'best+model.pt')
            if task == 'cinc':
                light = cond in LIGHT
                evaluate = lambda: eval_cinc(model, args, r['seed'], dev, ('val',) if light else ('val', 'test'))
                ref = {'val': ('score', float(cfg['best_val_score']))}
                if not light:
                    ref['test'] = ('score', float(r['score']))
            else:
                from dataloader import create_loader
                loader, _ = create_loader(train=False, num_classes=args.num_classes, batch_size=args.batch_size,
                                          data_root=args.root_path + '/5class_mitbih_test_3_187.csv', data_window_size=187,
                                          num_workers=args.num_workers)
                evaluate = lambda: eval_mitbih(model, args, dev, loader)
                ref = {'test': ('f1', float(r['f1']))}
        model.load_state_dict(torch.load(ck, map_location='cpu'))
        model = model.to(dev).eval(); model.train_mode = 'testing'
        m = model.inner if task == 'ett' else model
        if m.__class__.__name__ != 'SpkMixer' or len(m.blocks) != 1:
            raise SystemExit(f'{d}: not a one-block SpkMixer')
        blk, N = m.blocks[0], m.num_patches
        orig = blk.mix
        comb = blk.comb
        res, P0, first = {}, None, []
        def capture(x):                                                    # the first mixer input of the base pass
            if not first:
                first.append(x.detach().clone())
            return orig(x)
        for iv in interventions(comb, N, task == 'cinc' and cond in LIGHT):
            blk.mix = capture if iv == 'base' else make_mix(blk, orig, iv, N)
            met = evaluate()
            blk.mix = orig
            P = {s: v.pop('_P', None) for s, v in met.items()}
            if iv == 'base':
                P0 = P
                # causality of a restarted run (what kK / ctxW rely on): the forward mixer on the first half of a sequence gives
                # exactly the first half of its full-length output (a fresh start on a shorter sequence is the same dynamics)
                xm = first[0].permute(2, 1, 0, 3).contiguous()
                h = max(1, N // 2)
                with torch.no_grad():
                    full, pre = blk.mixer(xm)[:h], blk.mixer(xm[:h].contiguous())
                causal_mismatch = float((full != pre).float().mean())
                if causal_mismatch > 1e-4:
                    raise SystemExit(f'{d}: forward mixer prefix check failed (mismatch share {causal_mismatch:.2e})')
                for s, (k, want) in ref.items():
                    if abs(met[s][k] - want) > 1e-6:
                        raise SystemExit(f'{d}: base {s} {k} {met[s][k]:.6f} != logged {want:.6f}')
            elif iv in ('base2', 'kN', 'ctxN', 'sdtrue'):
                for s in P:
                    if P[s] is not None and not np.array_equal(np.asarray(P[s]), np.asarray(P0[s])):
                        raise SystemExit(f'{d}: {iv} differs from base on {s} (check failed)')
                    if P[s] is None and met[s] != res['base'][s]:
                        raise SystemExit(f'{d}: {iv} differs from base on {s} (check failed)')
            res[iv] = met
        rec['runs'].append({'dir': d, 'seed': int(r['seed']), 'ds': r.get('ds'), 'pl': int(r['pl']) if 'pl' in r else None, 'N': N,
                            'comb': comb, 'checkpoint_sha256': sha256(ck), 'causal_mismatch': causal_mismatch, 'metrics': res,
                            'seconds': time.time() - t0})
        key = 'score' if task == 'cinc' else 'mse' if task == 'ett' else 'f1'
        sp = 'val' if task == 'cinc' else 'test'
        print(f"[bottleneck] {task} {cond} {r.get('ds', '')} {r.get('pl', '')} seed {int(r['seed'])}: "
              + ' '.join(f"{iv} {res[iv][sp][key]:.4f}" for iv in res if iv not in ('base2', 'kN', 'ctxN', 'sdtrue'))
              + f' | {time.time() - t0:.0f} s', flush=True)
        del model, inner; torch.cuda.empty_cache()
    json.dump(rec, open(path, 'x'), indent=1)
    print(f'wrote {path}')


def summarize():
    recs = {}
    for task, cond in PARTS:
        p = os.path.join(OUT, 'parts', f'{task}_{cond}.json')
        if not os.path.exists(p):
            raise SystemExit(f'incomplete: {p} missing')
        recs[(task, cond)] = json.load(open(p))
    show = lambda c: [iv for iv in ['off', 'shuf0', 'shuf1', 'shuf2', 'rev'] + [f'k{k}' for k in KS] + ['sd0', 'sd1']
                      if iv in c]
    L = ['# Bottleneck diagnostics (no training; interventions on the token-axis mixer of trained checkpoints) — report only', '',
         'Δ = intervention − base, per seed (42, 7, 2026), then the mean. shuf = mean of the three fixed permutations. '
         'kK = mixer restarted every K tokens. Checks: base reproduces the logged metrics, kN / second base call / s_d '
         're-implementation bitwise equal to base (all passed, or the part file would not exist).', '']
    for split in ('val', 'test'):
        L += [f'## CinC fold 0, {split} split (CinC score %; class F1 N / AF / Other / Noisy)', '',
              '| model | base | off | shuf (mean of 3) | rev | k1 | k4 | k16 | k64 | ctx4 | ctx16 | ctx64 | sd0 | sd1 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for cond in CINC:
            runs = recs[('cinc', cond)]['runs']
            if split not in runs[0]['metrics']['base']:
                L.append(f'| {cond} | (validation only) |' + ' |' * 12)
                continue
            v = lambda iv, k='score': [100 * x['metrics'][iv][split][k] for x in runs] if iv in runs[0]['metrics'] else None
            base = v('base')
            def cell(iv):
                if iv == 'shuf':                                           # mean over the permutations that were run
                    perms = [f'shuf{j}' for j in range(3) if f'shuf{j}' in runs[0]['metrics']]
                    s = [np.mean([v(q)[i] for q in perms]) for i in range(len(runs))]
                else:
                    s = v(iv)
                if s is None:
                    return '—'
                d = [a - b for a, b in zip(s, base)]
                return f'{np.mean(d):+.2f} ({", ".join(f"{x:+.1f}" for x in d)})'
            L.append(f'| {cond} | {np.mean(base):.2f} | ' + ' | '.join(cell(iv) for iv in ('off', 'shuf', 'rev', 'k1', 'k4', 'k16', 'k64', 'ctx4', 'ctx16', 'ctx64', 'sd0', 'sd1')) + ' |')
        L += ['', f'Class F1 change (mean over seeds, %p; N / AF / Other / Noisy), {split}:', '',
              '| model | off | shuf0 | k4 | k16 | ctx4 | ctx16 |', '|---|---|---|---|---|---|---|']
        for cond in CINC:
            runs = recs[('cinc', cond)]['runs']
            def cls(iv):
                if iv not in runs[0]['metrics'] or split not in runs[0]['metrics'][iv]:
                    return '—'
                return ' / '.join(f"{np.mean([100 * (x['metrics'][iv][split][f'f1_{c}'] - x['metrics']['base'][split][f'f1_{c}']) for x in runs]):+.1f}" for c in range(4))
            L.append(f'| {cond} | ' + ' | '.join(cls(iv) for iv in ('off', 'shuf0', 'k4', 'k16', 'ctx4', 'ctx16')) + ' |')
        L.append('')
    L += ['Forward-mixer prefix check (share of spikes that differ between a restarted half-length run and the first half of the '
          'full run; must be <= 1e-4): ' + ', '.join(f"{t} {c} max {max(x['causal_mismatch'] for x in recs[(t, c)]['runs']):.1e}" for t, c in PARTS), '']
    L += ['## ETT, muxlin tuned (test MSE; Δ % = 100 (MSE_iv − MSE_base) / MSE_base, mean over seeds)', '',
          '| cell | base MSE | off | shuf (mean of 3) | rev | k1 | k4 | k16 | ctx4 | ctx16 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    runs = recs[('ett', 'muxlin')]['runs']
    for ds, pl in sorted({(x['ds'], x['pl']) for x in runs}, key=lambda c: (c[0], c[1])):
        rr = [x for x in runs if x['ds'] == ds and x['pl'] == pl]
        b = [x['metrics']['base']['test']['mse'] for x in rr]
        def cell(iv):
            if iv == 'shuf':
                s = [np.mean([x['metrics'][f'shuf{j}']['test']['mse'] for j in range(3)]) for x in rr]
            elif iv in rr[0]['metrics']:
                s = [x['metrics'][iv]['test']['mse'] for x in rr]
            else:
                return '—'
            return f'{np.mean([100 * (a - c) / c for a, c in zip(s, b)]):+.2f}'
        L.append(f'| {ds} H{pl} | {np.mean(b):.4f} | ' + ' | '.join(cell(iv) for iv in ('off', 'shuf', 'rev', 'k1', 'k4', 'k16', 'ctx4', 'ctx16')) + ' |')
    L += ['', '## MIT-BIH, muxlin tuned (test macro-F1 %; Δ %p, per seed)', '',
          '| base | off | shuf (mean of 3) | rev | k1 | k4 | ctx4 |', '|---:|---:|---:|---:|---:|---:|---:|']
    runs = recs[('mitbih', 'muxlin')]['runs']
    b = [100 * x['metrics']['base']['test']['f1'] for x in runs]
    def mcell(iv):
        if iv == 'shuf':
            s = [np.mean([100 * x['metrics'][f'shuf{j}']['test']['f1'] for j in range(3)]) for x in runs]
        elif iv in runs[0]['metrics']:
            s = [100 * x['metrics'][iv]['test']['f1'] for x in runs]
        else:
            return '—'
        d = [a - c for a, c in zip(s, b)]
        return f'{np.mean(d):+.2f} ({", ".join(f"{x:+.2f}" for x in d)})'
    L.append(f'| {np.mean(b):.2f} | ' + ' | '.join(mcell(iv) for iv in ('off', 'shuf', 'rev', 'k1', 'k4', 'ctx4')) + ' |')
    open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


UNIT = {'': 1., 'K': 1e3, 'M': 1e6, 'G': 1e9}
E_AC, E_MAC = 0.9e-6, 4.6e-6                                              # µJ per operation (utils.get_energy_consumption)


def layer_ops(run):
    """[(module path, AC ops, MAC ops)] of the Linear / BatchNorm / neuron lines of the syops per-layer file."""
    out, stack = [], []
    pat = re.compile(r'^(\s*)\(([^)]+)\): (\w+)\((.*)$')
    num = re.compile(r'([\d.]+)\s*([KMG]?)\s*Ops,\s*[\d.]+%\s*ACs,\s*([\d.]+)\s*([KMG]?)\s*Ops,\s*[\d.]+%\s*MACs')
    lines = open(os.path.join(run, 'log', 'model+info+per+layer.txt'), encoding='utf-8').read().splitlines()
    for i, line in enumerate(lines):
        m = pat.match(line)
        if not m:
            continue
        depth = len(m.group(1)) // 2
        stack = stack[:depth - 1] + [m.group(2)]
        n = num.search(m.group(4)) or (num.search(lines[i + 1]) if i + 1 < len(lines) and not pat.match(lines[i + 1]) else None)
        if n and m.group(3) in ('Linear', 'BatchNorm1d', 'LIFNeuron', 'MultiStepLIFNode', 'LIFNode', 'Conv1d'):
            out.append(('.'.join(stack), m.group(3), float(n.group(1)) * UNIT[n.group(2)], float(n.group(3)) * UNIT[n.group(4)]))
    return out


def energy():
    import summarize_cinc_tune as ct
    D = ct.load()
    groups = {'muxlin': ct.seeds(D, ct.MUX, ('or', 0.001, ct.L0)), 'Spikformer-style (lr 2e-3)': ct.seeds(D, ct.REF, ('-', 0.002, ct.L0))}
    L = ['# Energy by part (CinC fold 0, one test window; syops per-layer files, 0.9 pJ / AC, 4.6 pJ / MAC; neuron state updates from '
         'RESULT neuron_model_fft_uJ) — mean of seeds 42 / 7 / 2026, µJ', '']
    cats = ['embedding Linear (real input, MAC, x T_h)', 'BatchNorm (MAC)', 'token mixer / attention projections (AC)',
            'fusion Linear (AC)', 'MLP (AC)', 'head', 'neuron input ACs (syops)', 'neuron state updates']
    def cat(path, kind):
        if 'emb_linear' in path or path.endswith('encoding.linear'):
            return cats[0]
        if kind == 'BatchNorm1d':
            return cats[1]
        if kind in ('LIFNeuron', 'MultiStepLIFNode', 'LIFNode'):
            return cats[6]
        if 'fuse' in path:
            return cats[3]
        if 'fc1' in path or 'fc2' in path or 'mlp' in path:
            return cats[4]
        if path.startswith('head'):
            return cats[5]
        return cats[2]
    tab = {}
    for name, rows in groups.items():
        if rows is None:
            raise SystemExit(f'energy: {name} runs missing')
        per = []
        for r in rows:
            e = {c: 0. for c in cats}
            for path, kind, ac, mac in layer_ops(r['dir']):
                e[cat(path, kind)] += ac * E_AC + mac * E_MAC
            e[cats[7]] = float(r['neuron_model_fft_uJ'])
            syn = sum(v for k, v in e.items() if k != cats[7])
            if abs(syn - float(r['syn_energy_uJ'])) > 0.02 * float(r['syn_energy_uJ']):
                raise SystemExit(f"energy: {r['dir']}: per-layer sum {syn:.2f} vs logged syn {float(r['syn_energy_uJ']):.2f} µJ")
            e['_total'] = syn + e[cats[7]]; e['_logged'] = float(r['total_fft_uJ'])
            per.append(e)
        tab[name] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    names = list(tab)
    L += ['| part | ' + ' | '.join(names) + ' |', '|---|' + '---:|' * len(names)]
    for c in cats:
        L.append(f'| {c} | ' + ' | '.join(f"{tab[n][c]:.1f} ({100 * tab[n][c] / tab[n]['_total']:.0f} %)" for n in names) + ' |')
    L.append('| total (sum) / logged total_fft_uJ | ' + ' | '.join(f"{tab[n]['_total']:.1f} / {tab[n]['_logged']:.1f}" for n in names) + ' |')
    L += ['', 'What-if (same checkpoints, accounting only, not a new model):', '',
          '| accounting | ' + ' | '.join(names) + ' | ratio |', '|---|' + '---:|' * (len(names) + 1)]
    base = {n: tab[n]['_total'] for n in names}
    once = {n: base[n] - 0.75 * tab[n][cats[0]] for n in names}
    fold = {n: once[n] - tab[n][cats[1]] for n in names}
    for lab, v in (('as logged (embedding x T_h, BN counted)', base), ('embedding computed once (its T_h repeats are identical)', once),
                   ('+ BN folded into the preceding Linear', fold)):
        L.append(f'| {lab} | ' + ' | '.join(f'{v[n]:.1f}' for n in names) + f' | {v[names[0]] / v[names[1]]:.2f} |')
    open(os.path.join(OUT, 'energy.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['cinc', 'ett', 'mitbih']); ap.add_argument('--cond'); ap.add_argument('-nd', '--num_device', type=int, default=0)
    ap.add_argument('--summarize', action='store_true'); ap.add_argument('--energy', action='store_true')
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    if a.energy:
        os.makedirs(OUT, exist_ok=True)
        return energy()
    if (a.task, a.cond) not in PARTS:
        raise SystemExit(f'unknown part {a.task} {a.cond}; parts: {PARTS}')
    run_part(a.task, a.cond, torch.device(f'cuda:{a.num_device}'))


if __name__ == '__main__':
    main()
