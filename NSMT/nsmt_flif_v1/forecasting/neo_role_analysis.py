"""Phase A (MODI_PROJECT_LOG 2026-10-17): what does the neocortex path do in a trained NSMT model? No training.

For one trained run (best-validation weights by default) on the standard ETT test split:

  A1 substitution  replace the neocortex output x_neo (time_block output, [N, B*C, 1, D]) by
                   shuffle  the same channel of another window in the batch
                   ones     all ones  (the AND mask passes everything: key/value = hippocampus itself)
                   zeros    all zeros
                   permute  a fixed random permutation of the token axis (spike statistics kept, order destroyed)
                   reverse  the neocortex fed the time-reversed patch sequence, output flipped back to positions
                   and report test MSE of each against the clean forward.
  A2 mask stats    neocortex firing rate per token, fraction of hippocampus spikes that survive the AND,
                   agreement of x_neo between windows at the same token (vs the chance level 1 - 2p(1-p)).
  A3 probes        ridge regression (features standardised on train, penalty chosen per target on a held-out 20 %
                   of the train windows, refitted on train, R^2 on val windows) from {neo_all, neo_last, hip_all, hip_last, raw} to {future mean, future slope (window-
                   normalised space), mean of the patch d steps back, d = 0, 1, 4, 12, 23}.
  A4 repetition    the backbone is causal along its spiking (repetition) axis and the head is shared over steps, so
                   the mean of the first k step outputs is the model with T cut to k: test MSE for k = 1..T and of each
                   single step.
  A5 sensitivity   forward unchanged, the neocortex gradient share set to 1: L2 norm of d(test MSE)/d(neocortex input)
                   vs d(test MSE)/d(hippocampus input) (summed over the repetition axis), on --grad_windows windows.
A6 (energy split by path and the serial projection) reads the saved per-layer files only: --energy <suite>.

usage: python neo_role_analysis.py --run <run_dir> [--run ...] [-nd 0] [--out DIR] [--weights best|last]
       python neo_role_analysis.py --summarize --out DIR          (tables + summary.md over the per-run JSON)
       python neo_role_analysis.py --energy flif-v1-s2p10 --out DIR
The first clean forward must reproduce the test MSE stored by train.py for the same weights (checked, else exit 1).
"""
import os
import re
import csv
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import torch
from spikingjelly.clock_driven import functional

from data_provider.data_factory import data_provider
from analysis import extract_pred_true_for_metrics
from ett_future import load_run, cond_name

HERE = os.path.dirname(os.path.abspath(__file__))
SUBS = ('shuffle', 'ones', 'zeros', 'permute', 'reverse')
LAGS = (0, 1, 4, 12, 23)
KS = (1, 2, 4, 8, 12, 16, 24)


class Hooks:
    """One set of hooks; `mode` selects what they do on the next forward."""
    def __init__(self, model):
        self.m, self.mode, self.store = model, None, {}
        self.has_neo = not getattr(model, 'hippo_only', False)
        g = torch.Generator().manual_seed(0)
        self.perm = torch.randperm(model.T, generator=g)
        if self.has_neo:
            model.encoding_neo.register_forward_pre_hook(self._neo_in)
            model.time_block.register_forward_hook(self._neo_out)
            model.data_block[0].register_forward_pre_hook(self._block_in)
        model.encoding.register_forward_pre_hook(self._hip_in)
        model.encoding.register_forward_hook(self._hip_out)

    def _neo_in(self, mod, inp):
        x = inp[0]
        self.store['raw'] = x.detach()
        if self.mode == 'reverse':
            return (x.flip(0),)
        if self.mode == 'grad':
            leaf = x.detach().requires_grad_(True); self.store['neo_leaf'] = leaf
            return (leaf,)
        return None

    def _neo_out(self, mod, inp, out):
        if self.mode == 'shuffle':
            out = out.roll(self.m.M, dims=1)                      # same channel, previous window in the batch
        elif self.mode == 'ones':
            out = torch.ones_like(out)
        elif self.mode == 'zeros':
            out = torch.zeros_like(out)
        elif self.mode == 'permute':
            out = out[self.perm.to(out.device)]
        elif self.mode == 'reverse':
            out = out.flip(0)                                     # back to the original token positions
        self.store['neo'] = out.detach()
        return out

    def _block_in(self, mod, args):
        self.store['block_x'] = args[0].detach()                  # hippocampus embedding entering the block

    def _hip_in(self, mod, inp):
        if self.mode == 'grad':
            leaf = inp[0].detach().requires_grad_(True); self.store['hip_leaf'] = leaf
            return (leaf,)
        return None

    def _hip_out(self, mod, inp, out):
        self.store['hip'] = out.detach()


def forward(model, hooks, x, y, cfg, mode=None):
    hooks.mode = mode
    functional.reset_net(model)
    out = model(x)
    hooks.mode = None
    return out


def mse_of(preds, trues):
    p, t = torch.cat(preds), torch.cat(trues)
    return float(((p - t) ** 2).mean())


ALPHAS = (1., 10., 100., 1e3, 1e4, 1e5)


def _ridge_fit(A, Y, alpha):
    reg = alpha * torch.eye(A.shape[1], dtype=A.dtype, device=A.device); reg[-1, -1] = 0.
    return torch.linalg.solve(A.T @ A + reg, A.T @ Y)


def ridge_r2(Xtr, Ytr, Xva, Yva, dev):
    """Ridge R^2 on val per target. Features standardised on train; the penalty is chosen per target on the last 20 %
    of the train rows (fit on the first 80 %), then refitted on all train rows -- val and test never select anything.
    Runs on the run's GPU: CPU LAPACK/OpenMP in this process (after DataLoader workers forked) segfaulted (2026-10-17)."""
    Xtr, Ytr, Xva, Yva = (t.to(dev) for t in (Xtr, Ytr, Xva, Yva))
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    one = lambda Z: torch.cat([(Z - mu) / sd, torch.ones(len(Z), 1, dtype=Z.dtype, device=dev)], 1)
    A, B = one(Xtr), one(Xva)
    cut = int(0.8 * len(A))
    def r2(Yp, Y):
        return 1. - (Y - Yp).pow(2).sum(0) / (Y - Y.mean(0)).pow(2).sum(0)
    inner = torch.stack([r2(A[cut:] @ _ridge_fit(A[:cut], Ytr[:cut], al), Ytr[cut:]) for al in ALPHAS])   # [n_alpha, n_target]
    best = inner.argmax(0)
    out, chosen = [], []
    for j in range(Ytr.shape[1]):
        al = ALPHAS[int(best[j])]
        W = _ridge_fit(A, Ytr[:, j:j + 1], al)
        out.append(float(r2(B @ W, Yva[:, j:j + 1])[0])); chosen.append(al)
    return out, chosen


def probe_data(model, hooks, cfg, flag, max_windows):
    """Features and targets per window-channel from up to max_windows windows of a split."""
    set_seed = torch.manual_seed(1)
    _, loader = data_provider(cfg, flag=flag)
    feats, targs, n = defaultdict(list), [], 0
    N, H = model.T, cfg.pred_len
    tt = torch.arange(H, dtype=torch.float64); tt = (tt - tt.mean()) / tt.std()
    with torch.no_grad():
        for x, y, _, _ in loader:
            x, y = x.float().to(cfg.device), y.float().to(cfg.device)
            forward(model, hooks, x, y, cfg)
            B, C = x.shape[0], x.shape[2]
            yn = ((y[:, -H:, :] - model.means) / model.stdev).permute(0, 2, 1).reshape(B * C, H).double().cpu()   # [BC, H]
            raw = hooks.store['raw'] if hooks.has_neo else None
            hip = hooks.store['hip'].mean(0)                      # [BC, N, D] (repetition-axis mean)
            if raw is None:                                       # hippo_only: rebuild patches from the hippo input
                raw_bc = None
            else:
                raw_bc = raw[:, :, 0, :].permute(1, 0, 2).double().cpu()        # [BC, N, P]
            if raw_bc is None:
                patches = hooks_patches(model, x)
                raw_bc = patches
            t = [yn.mean(1), (yn * tt).mean(1)] + [raw_bc[:, N - 1 - d, :].mean(1) for d in LAGS]
            targs.append(torch.stack(t, 1))
            feats['raw'].append(raw_bc.reshape(B * C, -1))
            feats['hip_all'].append(hip.reshape(B * C, -1).double().cpu())
            feats['hip_last'].append(hip[:, N - 1, :].double().cpu())
            if hooks.has_neo:
                neo = hooks.store['neo'][:, :, 0, :].permute(1, 0, 2)          # [BC, N, D]
                feats['neo_all'].append(neo.reshape(B * C, -1).double().cpu())
                feats['neo_last'].append(neo[:, N - 1, :].double().cpu())
            n += B
            if n >= max_windows:
                break
    return {k: torch.cat(v) for k, v in feats.items()}, torch.cat(targs)


def hooks_patches(model, x):
    """Normalised, padded patches [BC, N, P] exactly as myModel.forward builds them (for hippo_only runs)."""
    from einops import rearrange
    xx = x.clone(); means = xx.mean(1, keepdim=True); xx = xx - means
    xx = xx / torch.sqrt(torch.var(xx, dim=1, keepdim=True, unbiased=False) + 1e-5)
    xx = rearrange(xx, 'b l c -> (b c) l').unsqueeze(1)
    xx = model.padding_patch_layer(xx)[:, 0]
    return xx.unfold(-1, model.patch_size, model.stride).double().cpu()


def analyse(run, gpu, out_dir, weights, grad_windows, probe_train, probe_val):
    cfg, model = load_run(run, gpu, weights)
    model.eval()
    hooks = Hooks(model)
    rec = {'run': run, 'data': cfg.data, 'pred_len': cfg.pred_len, 'cond': cond_name(cfg), 'seed': cfg.seed,
           'weights': weights, 'T': model.T, 'has_neo': hooks.has_neo}
    csvp = os.path.join(run, 'log', 'best' if weights == 'best' else '', 'final+result.csv')
    stored = float(open(csvp).read().splitlines()[1].split(',')[2])
    _, loader = data_provider(cfg, flag='test')
    modes = [None] + (list(SUBS) if hooks.has_neo else [])
    preds = {m: [] for m in modes}; trues = []; steps = []
    a2 = defaultdict(float); a2n = 0
    with torch.no_grad():
        for x, y, _, _ in loader:
            x, y = x.float().to(cfg.device), y.float().to(cfg.device)
            for m in modes:
                out = forward(model, hooks, x, y, cfg, m)
                p, t, _, _ = extract_pred_true_for_metrics(out, y, cfg.pred_len, long_pred_len=cfg.pred_len, reduce_T='mean')
                preds[m].append(p.cpu())
                if m is None:
                    trues.append(t.cpu())
                    steps.append(out[:, :, -cfg.pred_len:, :].cpu())           # [T, B, H, C] per-step forecasts
                    if hooks.has_neo:
                        neo = hooks.store['neo'][:, :, 0, :]                     # [N, BC, D]
                        hx = hooks.store['block_x']                              # [T, BC, N, D]
                        mask = neo.permute(1, 0, 2).unsqueeze(0)                 # [1, BC, N, D]
                        a2['neo_rate'] += float(neo.mean()); a2['hip_rate'] += float(hx.mean())
                        a2['survive'] += float((hx * mask).sum() / hx.sum().clamp_min(1.))
                        p_tok = neo.mean((1, 2))                                  # per token
                        a2['neo_rate_first'] += float(p_tok[0]); a2['neo_rate_last'] += float(p_tok[-1])
                        agree = (neo == neo.roll(model.M, dims=1)).float().mean()
                        pbar = neo.mean(); a2['agree'] += float(agree); a2['agree_chance'] += float(1 - 2 * pbar * (1 - pbar))
                        a2n += 1
    clean = mse_of(preds[None], trues)
    rec['stored_test_mse'] = stored; rec['clean_mse'] = clean
    if abs(clean - stored) > 2e-6:
        rec['error'] = f'clean forward {clean:.6f} != stored {stored:.6f}'
        return rec
    rec['a1'] = {m: mse_of(preds[m], trues) for m in modes if m is not None}
    rec['a2'] = {k: v / a2n for k, v in a2.items()} if a2n else None
    S = torch.cat(steps, 1); Tt = torch.cat(trues).permute(0, 1, 2)                 # S [T, Nw, H, C]
    rec['a4_prefix'] = {k: float(((S[:k].mean(0) - Tt) ** 2).mean()) for k in KS if k <= S.shape[0]}
    rec['a4_step'] = [float(((S[t] - Tt) ** 2).mean()) for t in range(S.shape[0])]
    rec['a4_step_spread'] = float(S.std(0).mean())
    if abs(rec['a4_prefix'][S.shape[0]] - clean) > 2e-6:
        rec['error'] = 'prefix-T check failed'; return rec
    # A5: functional sensitivity with the full gradient share
    if hooks.has_neo:
        saved = [b.neo_grad for b in model.data_block]
        for b in model.data_block:
            b.neo_grad = 1.0
        gn, gh, nw = 0., 0., 0
        for x, y, _, _ in loader:
            x, y = x.float().to(cfg.device), y.float().to(cfg.device)
            model.zero_grad(set_to_none=True)
            out = forward(model, hooks, x, y, cfg, 'grad')
            H = cfg.pred_len                                       # same as extract_pred_true_for_metrics (which runs
            p = out[:, :, -H:, :].mean(0); t = y[:, -H:, :]       # under no_grad): repetition-axis mean, last H steps
            ((p - t) ** 2).mean().backward()
            g_neo = hooks.store['neo_leaf'].grad                                   # [N, BC, 1, P]
            g_hip = hooks.store['hip_leaf'].grad.sum(0)                           # [BC, N, P] (repetition axis summed)
            gn += float(g_neo.pow(2).sum()); gh += float(g_hip.pow(2).sum()); nw += x.shape[0]
            if nw >= grad_windows:
                break
        for b, g in zip(model.data_block, saved):
            b.neo_grad = g
        gn, gh = gn ** .5, gh ** .5
        rec['a5'] = {'grad_neo': gn, 'grad_hip': gh, 'share_neo': gn / (gn + gh), 'windows': nw}
    # A3 probes
    Ftr, Ytr = probe_data(model, hooks, cfg, 'train', probe_train)
    Fva, Yva = probe_data(model, hooks, cfg, 'val', probe_val)
    names = ['fut_mean', 'fut_slope'] + [f'lag{d}' for d in LAGS]
    rec['a3'], rec['a3_alpha'] = {}, {}
    for f in Ftr:
        r2s, als = ridge_r2(Ftr[f], Ytr, Fva[f], Yva, cfg.device)
        rec['a3'][f] = dict(zip(names, r2s)); rec['a3_alpha'][f] = dict(zip(names, als))
    rec['a3_rows'] = {'train': int(len(Ytr)), 'val': int(len(Yva))}
    return rec


def path_energy(run, sub='best'):
    """Synaptic (syops per layer, 0.9/4.6 pJ) and neuron (eq. 60/64) energy by path, from saved files only."""
    d = os.path.join(run, 'log', sub)
    syn = defaultdict(float)
    path = []
    for l in open(os.path.join(d, 'model+info+per+layer.txt')).read().splitlines():
        m = re.match(r'(\s*)\((\w+)\): (\w+)\((.*)', l)
        if not m:
            continue
        ind = len(m.group(1)) // 2
        path = path[:ind - 1] + [m.group(2)]
        if m.group(3) in ('Linear', 'BatchNorm1d'):
            f = [x.strip() for x in m.group(4).rstrip(')').split(',')]
            def num(s):
                v, u = re.match(r'([\d.]+)\s*([KMG]?)', s).groups(); return float(v) * {'': 1, 'K': 1e3, 'M': 1e6, 'G': 1e9}[u]
            e = 0.9 * num(f[2].replace(' Ops', '')) + 4.6 * num(f[4].replace(' Ops', ''))   # pJ
            name = '.'.join(path)
            grp = ('neo' if name.startswith(('encoding_neo', 'time_block')) else 'hip_emb' if name.startswith('encoding')
                   else 'head' if name.startswith('head') else 'backbone')
            syn[grp] += e * 1e-6                                                   # uJ
    neu = defaultdict(float)
    for r in csv.DictReader(open(os.path.join(d, 'neuron_energy.csv'))):
        grp = 'neo' if r['path'] == 'neo' else ('hip_emb' if r['module'].startswith('encoding.') else 'backbone')
        neu[grp] += float(r['E_fft_J']) * 1e6
    return dict(syn), dict(neu)


def energy_table(suite, out_dir):
    rows = []
    for line in open(os.path.join(HERE, 'results', suite, 'raw.txt')):
        if not line.startswith('RESULT'):
            continue
        kv = dict(re.findall(r'(\w+)=(\S+)', line))
        cond = kv['cond'] if kv.get('regime', 'L0g0.05') == 'L0g0.05' else f"{kv['cond']}@{kv['regime']}"
        if cond not in ('base', 'hippo_only', 'neo_lif_both', 'neo_flif_both'):
            continue
        syn, neu = path_energy(kv['dir'])
        row = {'data': kv['dataset'], 'cond': cond, 'seed': kv['seed']}
        for g in ('hip_emb', 'backbone', 'head', 'neo'):
            row[f'syn_{g}'] = syn.get(g, 0.); row[f'neu_{g}'] = neu.get(g, 0.)
        tot = sum(syn.values()) + sum(neu.values()); row['total'] = tot
        T = 24.
        rep = row['syn_backbone'] + row['neu_backbone'] + row['syn_head']
        for tb in (1, 2, 4, 24):                                                    # firing rates held fixed
            row[f'serial_Tb{tb}'] = row['syn_neo'] + row['neu_neo'] + rep * tb / T if cond != 'hippo_only' else float('nan')
            row[f'hippo_Tb{tb}'] = (row['syn_hip_emb'] + row['neu_hip_emb'] + rep) * tb / T
        rows.append(row)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'a6_energy.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    print('a6 rows', len(rows))


def summarize(out_dir):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, 'runs', '*.json')))]
    bad = [r for r in recs if 'error' in r]
    good = [r for r in recs if 'error' not in r]
    def cell(r): return f"{r['data']}_H{r['pred_len']}"
    order = ['hippo_only', 'base', 'neo_lif_both', 'neo_flif_both', 'neo_lif_both@L2g0.05', 'neo_flif_both@L2g0.05']
    cells = sorted({cell(r) for r in good})
    md = [f'# Phase A — neocortex role ({len(good)} runs OK, {len(bad)} failed checks)', '']
    if bad:
        md += ['Failed: ' + '; '.join(f"{os.path.basename(r['run'])[:60]}: {r['error']}" for r in bad), '']
    def table(title, fn, fmt='{:+.2f}%'):
        out = [f'## {title}', '', '| cond | ' + ' | '.join(cells) + ' |', '|---|' + '---|' * len(cells)]
        for c in order:
            vals = []
            for ce in cells:
                v = [fn(r) for r in good if r['cond'] == c and cell(r) == ce]
                v = [x for x in v if x is not None]
                vals.append('–' if not v else fmt.format(float(np.mean(v))))
            out.append(f'| {c} | ' + ' | '.join(vals) + ' |')
        return out + ['']
    md += table('clean test MSE (best-validation weights)', lambda r: r['clean_mse'], '{:.6f}')
    for s in SUBS:
        md += table(f'A1 ΔMSE when the neocortex output is replaced: {s}', lambda r, s=s: None if not r['has_neo'] else 100 * (r['a1'][s] - r['clean_mse']) / r['clean_mse'])
    for k in ('neo_rate', 'hip_rate', 'survive', 'agree', 'agree_chance'):
        md += table(f'A2 {k}', lambda r, k=k: None if not r.get('a2') else r['a2'][k], '{:.3f}')
    for k in (1, 2, 4, 8, 12):
        md += table(f'A4 ΔMSE with the repetition axis cut to k = {k} (of 24)', lambda r, k=k: 100 * (r['a4_prefix'][str(k)] - r['clean_mse']) / r['clean_mse'])
    md += table('A5 neocortex share of the input-gradient norm', lambda r: None if not r.get('a5') else r['a5']['share_neo'], '{:.3f}')
    for feat in ('raw', 'hip_all', 'hip_last', 'neo_all', 'neo_last'):
        for tgt in ('fut_mean', 'fut_slope', 'lag1', 'lag4', 'lag12', 'lag23'):
            md += table(f'A3 R² (val) {feat} → {tgt}', lambda r, f=feat, t=tgt: r['a3'].get(f, {}).get(t), '{:.3f}')
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    flat = []
    for r in good:
        row = {k: r[k] for k in ('data', 'pred_len', 'cond', 'seed', 'clean_mse', 'stored_test_mse')}
        for s, v in (r.get('a1') or {}).items():
            row[f'a1_{s}'] = v
        for s, v in (r.get('a2') or {}).items():
            row[f'a2_{s}'] = v
        for s, v in r['a4_prefix'].items():
            row[f'a4_k{s}'] = v
        for s, v in (r.get('a5') or {}).items():
            row[f'a5_{s}'] = v
        for feat, d in r['a3'].items():
            for t, v in d.items():
                row[f'a3_{feat}_{t}'] = v
        flat.append(row)
    keys = sorted({k for r in flat for k in r}, key=lambda k: (not k[0].isalpha(), k))
    with open(os.path.join(out_dir, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in flat]
    print('summary', len(good), 'ok,', len(bad), 'failed ->', out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append'); ap.add_argument('-nd', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'neo-role-20261017'))
    ap.add_argument('--weights', choices=['best', 'last'], default='best')
    ap.add_argument('--grad_windows', type=int, default=512)
    ap.add_argument('--probe_train', type=int, default=3000); ap.add_argument('--probe_val', type=int, default=1500)
    ap.add_argument('--summarize', action='store_true'); ap.add_argument('--energy', default=None)
    a = ap.parse_args()
    if a.summarize:
        summarize(a.out); return
    if a.energy:
        energy_table(a.energy, a.out); return
    os.makedirs(os.path.join(a.out, 'runs'), exist_ok=True)
    for run in a.run:
        rec = analyse(os.path.abspath(run), a.nd, a.out, a.weights, a.grad_windows, a.probe_train, a.probe_val)
        name = f"{rec['data']}_H{rec['pred_len']}_{rec['cond']}_seed{rec['seed']}.json"
        json.dump(rec, open(os.path.join(a.out, 'runs', name), 'w'), indent=1)
        print('ROLE', name, 'ERROR ' + rec['error'] if 'error' in rec else f"clean {rec['clean_mse']:.6f} = stored {rec['stored_test_mse']:.6f}")
        if 'error' in rec:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
