"""MODI 2026-10-28: cause analysis without training (docs/MODI_PROJECT_LOG.md, 2026-10-28 pre-registration).

(a) Why is the T = N neocortex added to Spikformer-style attention (B, 90 epochs) worse than the memoryless neocortex?
(b) Why does the attention-free Neo-mixer not beat Spikformer-style on CinC?
Best-validation weights (model_state/checkpoint.pth = the weights of log/final+result.csv), CinC fold-0 test windows, except H3's
gradients (training crops of epoch 0).

--kind b   B models (Spikformer-style + T = N or memoryless neocortex keys / values)
  H1  neocortex output [N, W, D] on 768 fixed test windows: ridge readout (lambda 10, windows split 2:1) of the token's own input
      patch (lags 0..8), of R-peak presence in the token and of the last RR interval at the token; adjacent-token cosine
      similarity; effective number of neocortex keys per attention row (sum|s|)^2 / sum s^2 (first 256 windows).
  H2  output firing rate and own-patch readout by token position.
  H3  relative distance from the initial weights by module group (the init is rebuilt from the seed and verified on the decoder,
      which alpha 0 never updates); CE-gradient norms by group on 16 x 32 training crops (eval-mode BN).
  H4  test score with the neocortex output replaced: the next window of the batch / zeros / history scramble (the neocortex reads
      the tokens in a fixed random order, its outputs are put back at their own token positions); share of the neocortex term in
      the attention output, ||s2 v2|| / (||s v|| + ||s2 v2||) (first 256 windows).
--kind h5  Spikformer-style and Neo-mixer
  H5  test score when every window keeps only its last k tokens (tiled to the window length as short records are in training) and
      when the token order is shuffled in blocks of b tokens.
Checks (fail-closed, exit 1): the unmodified evaluation reproduces the logged test score and test_probs.npz; identity edits (k = N,
b = N, identity scramble) reproduce the logits bit for bit; the SSA replica equals the module; the memoryless history scramble
leaves the logits unchanged; the rebuilt init equals the never-updated decoder.
usage: python rhythm_cause.py --kind b|h5 --run_dir <dir> --name <cond>_seed<s> --out_dir <dir>
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional          # noqa: E402
from config import set_random_seed                        # noqa: E402
from layers import RelBias1DDeterministicFn               # noqa: E402
from load_model import LOAD_MODEL                         # noqa: E402
from train_rhythm import cal_weights                      # noqa: E402
import rhythm_data as rd                                   # noqa: E402
import rhythm_eval as re_                                  # noqa: E402

K_LIST, B_LIST = (4, 8, 16, 32, 64), (1, 4, 16)
LAM, N_WIN, N_ATT, LAGS = 10., 768, 256, 9
GROUPS = (('neocortex', ('encoding_neo.', 'time_block.')), ('neo_kv', ('.ssa.k2_', '.ssa.v2_')),
          ('attn_own', ('.ssa.q_', '.ssa.k_', '.ssa.v_', '.ssa.proj_')), ('hippo_rest', ('encoding.', '.mlp.', 'head.')),
          ('decoder', ('weak_decoder.',)))


def group_of(name):
    for g, keys in GROUPS:
        if any((name.startswith(k) if k.endswith('.') and not k.startswith('.') else k in name) for k in keys):
            return g
    return None


class Patch:
    """temporarily replace a bound method by an instance attribute."""
    def __init__(self, obj, name, fn):
        if name in obj.__dict__:
            raise RuntimeError(f'{name} is already patched')
        self.obj, self.name, self.fn = obj, name, fn

    def __enter__(self):
        setattr(self.obj, self.name, self.fn)

    def __exit__(self, *a):
        delattr(self.obj, self.name)


def load(run_dir, device):
    cfg = torch.load(os.path.join(run_dir, 'model_state', 'config.pt'), map_location='cpu')
    args = argparse.Namespace(**cfg)
    set_random_seed(int(args.seed))                                   # same RNG sequence as train_rhythm.main
    data = rd.build(args, int(args.seed))
    init = {k: v.detach().clone() for k, v in LOAD_MODEL[args.model](args, True).state_dict().items()}
    m = LOAD_MODEL[args.model](args, True)
    m.load_state_dict(torch.load(os.path.join(run_dir, 'model_state', 'checkpoint.pth'), map_location='cpu'))
    m = m.to(device).eval(); m.train_mode = 'testing'
    return m, args, data, init


@torch.no_grad()
def predict(m, ds, device, bs, x_fn=None):
    """rhythm_eval.predict with an optional input edit; also returns the window logits."""
    m.eval(); m.train_mode = 'testing'
    probs, units, labels, logits = [], [], [], []
    functional.reset_net(m)
    for x, y, u in rd.loader(ds, bs, shuffle=False):
        x = x.float().to(device)
        if x_fn is not None:
            x = x_fn(x)
        z = re_._logits(m, x)
        functional.reset_net(m)
        logits.append(z.float().cpu()); probs.append(torch.softmax(z.float(), dim=1).cpu()); units.append(u); labels.append(y)
    p, u, y = torch.cat(probs).numpy(), torch.cat(units).numpy(), torch.cat(labels).numpy()
    uniq, inv = np.unique(u, return_inverse=True)
    P = np.zeros((len(uniq), p.shape[1])); np.add.at(P, inv, p)
    P /= np.bincount(inv).astype(float)[:, None]
    Y = np.zeros(len(uniq), dtype=int); Y[inv] = y
    return P, Y, torch.cat(logits)


def score(P, Y):
    return re_.metrics(P, Y, 'CINC2017')[0]['score']


def block_perm(N, b, seed):
    blocks = [np.arange(i, min(i + b, N)) for i in range(0, N, b)]
    order = np.random.default_rng(seed).permutation(len(blocks))
    return torch.as_tensor(np.concatenate([blocks[i] for i in order]))


def tile_last(k, N, S):
    def f(x):                                                         # x [B, 1, L]; token t covers padded samples [S t, S t + P)
        seg = x[..., S * (N - k):]
        return seg.repeat(1, 1, math.ceil(x.shape[-1] / seg.shape[-1]))[..., :x.shape[-1]].contiguous()
    return f


def token_shuffle(m, perm):
    if hasattr(m, 'forward_parts'):                                   # Neo-mixer: tokens [N, B, 1, P]
        orig = m.tokens
        return Patch(m, 'tokens', lambda x: orig(x)[perm.to(x.device)])
    orig = m._tokens_mc                                               # Spikformer-style: hippocampus [T_h, B, N, P], neocortex [N, B, 1, P]
    def f(x):
        h, n = orig(x); p = perm.to(x.device)
        return h[:, :, p].contiguous(), n[p].contiguous()
    return Patch(m, '_tokens_mc', f)


def neo_replace(m, mode, perm=None, stats=None):
    orig = m.neo_forward
    def f(neo_in, org_seq=None):
        if mode == 'scramble':
            p = perm.to(neo_in.device)
            e, o = orig(neo_in[p].contiguous(), org_seq)
            inv = torch.argsort(p)
            return e[inv].contiguous(), o[inv].contiguous()
        e, o = orig(neo_in, org_seq)
        if mode == 'swap':
            B = o.shape[1]
            stats['identity_swaps'] += int(B == 1)
            r = torch.roll(torch.arange(B, device=o.device), 1)
            return e[:, r].contiguous(), o[:, r].contiguous()
        if mode == 'zero':
            return e, torch.zeros_like(o)
        raise ValueError(mode)
    return Patch(m, 'neo_forward', f)


def ssa_parts(a, x, kv):
    """SSA_rel_scl.forward (kv_cat) split into the own term s v and the neocortex term s2 v2."""
    T, B, N, C = x.shape; H = a.num_heads
    def branch(lin, bn, lif, inp):
        z = lif(bn(lin(inp).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous())
        return z.reshape(T, B, N, H, C // H).permute(0, 1, 3, 2, 4).contiguous()
    xf = x.flatten(0, 1)
    k = branch(a.k_linear, a.k_bn, a.k_lif, xf)
    v = branch(a.v_linear, a.v_bn, a.v_lif, xf)
    q = branch(a.q_linear, a.q_bn, a.q_lif, xf)
    s = (q @ k.transpose(-2, -1)) * a.scale
    rb = None
    if a.pe:
        rb = RelBias1DDeterministicFn.apply(a.relative_bias_table, a.rel_pos_idx).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        s = s + rb
    own = s @ v
    kv2 = kv.flatten(0, 1)
    k2 = branch(a.k2_linear, a.k2_bn, a.k2_lif, kv2)
    v2 = branch(a.v2_linear, a.v2_bn, a.v2_lif, kv2)
    s2 = (q @ k2.transpose(-2, -1)) * a.scale
    if rb is not None:
        s2 = s2 + rb
    add = s2 @ v2
    y = (own + add).transpose(2, 3).reshape(T, B, N, C).contiguous()
    if a.attn_bn is not None:
        y = a.attn_bn(y.flatten(0, 1).transpose(-1, -2).contiguous()).transpose(-1, -2).reshape(T, B, N, C).contiguous()
    return dict(s=s, s2=s2, own=own, add=add, out=a.proj_linear(a.attn_lif(y)))


def eff_keys(s):
    a = s.double().abs(); den = (s.double() ** 2).sum(-1)
    ok = den > 0
    return float(((a.sum(-1) ** 2)[ok] / den[ok]).mean()), float(1 - ok.double().mean())


def beats(x, fs):
    """R-peak sample indices of one z-scored window (polarity of the larger extreme, height 1.5, prominence 1, >= 0.3 s apart)."""
    s = x if np.percentile(x, 99.5) >= -np.percentile(x, 0.5) else -x
    p, _ = find_peaks(s, height=1.5, prominence=1.0, distance=int(0.3 * fs))
    return p


def token_labels(x, N, P, S, fs):
    """per token: R-peak inside the token's samples; the last RR interval (s) among peaks up to the token's last sample."""
    pk = beats(x, fs); L = len(x)
    peak = np.zeros(N, dtype=np.float32); rr = np.full(N, np.nan, dtype=np.float32)
    for t in range(N):
        a, b = S * t, min(S * t + P, L)
        peak[t] = float(((pk >= a) & (pk < b)).any())
        q = pk[pk < b]
        if len(q) >= 2:
            d = (q[-1] - q[-2]) / fs
            if 0.3 <= d <= 2.0:
                rr[t] = d
    return peak, rr


def ridge_r2(X, Y, tr, te, lam=LAM):
    """X [n, D], Y [n, k] on one device (float64); fit on rows tr, R^2 per output on rows te."""
    X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], 1)
    A = X[tr].T @ X[tr] + lam * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
    W = torch.linalg.solve(A, X[tr].T @ Y[tr])
    pred = X[te] @ W
    sst = ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
    r2 = 1 - ((Y[te] - pred) ** 2).sum(0) / sst
    return r2[sst > 0], pred


@torch.no_grad()
def part_h1_h2_h4share(m, args, data, device, checks):
    ds = data['test']
    idx = np.sort(np.random.default_rng(0).choice(len(ds), size=min(N_WIN, len(ds)), replace=False))
    N, P, S, fs = m.num_patches, m.patch_size, m.stride, float(args.fs_target)
    O, TOK, PEAK, RR = [], [], [], []
    att = dict(eff_add=[], eff_own=[], zero_add=[], n_own=0., n_add=0., n_win=0)
    for b0 in range(0, len(idx), 32):
        x = torch.stack([ds[int(i)][0] for i in idx[b0:b0 + 32]]).float().to(device)       # [B, 1, L]
        functional.reset_net(m)
        patched, neo_in = m._tokens_mc(x)
        _, o = m.neo_forward(neo_in, x)
        O.append(o[:, :, 0].to(torch.uint8).cpu()); TOK.append(neo_in[:, :, 0].float().cpu())
        for w in x[:, 0].cpu().numpy():
            pk, rr = token_labels(w, N, P, S, fs); PEAK.append(pk); RR.append(rr)
        if att['n_win'] < N_ATT:
            functional.reset_net(m)
            xh = m.encoding(patched)
            blk = m.data_block[0]
            g = getattr(blk, 'neo_grad', 0.5)
            mx = g * o + (1 - g) * o.detach()
            kv = mx.transpose(0, 2).expand(xh.shape[0], -1, -1, -1).contiguous()
            functional.reset_net(blk.ssa); ref = blk.ssa(xh, kv=kv)
            functional.reset_net(blk.ssa); pr = ssa_parts(blk.ssa, xh, kv)
            if not torch.equal(pr['out'], ref):
                checks.append(f'SSA replica differs from the module (max {float((pr["out"] - ref).abs().max()):.3g})')
            e, z = eff_keys(pr['s2']); att['eff_add'].append(e); att['zero_add'].append(z)
            att['eff_own'].append(eff_keys(pr['s'])[0])
            att['n_own'] += float((pr['own'].double() ** 2).sum()); att['n_add'] += float((pr['add'].double() ** 2).sum())
            att['n_win'] += x.shape[0]
        functional.reset_net(m)
    O = torch.cat(O, 1); TOK = torch.cat(TOK, 1)                                          # [N, W, D], [N, W, P]
    W_ = O.shape[1]
    PEAK = torch.as_tensor(np.stack(PEAK, 1)); RR = torch.as_tensor(np.stack(RR, 1))       # [N, W]
    dev = device
    X = O.double().to(dev); T_ = TOK.double().to(dev)
    win = torch.arange(W_, device=dev).expand(N, W_)
    is_tr = win < (2 * W_) // 3
    out = {}
    # H1 (i) own patch and lags
    lag_r2 = []
    for k in range(LAGS):
        Xk = X[k:].reshape(-1, X.shape[-1]); Yk = T_[:N - k].reshape(-1, T_.shape[-1]); tr = is_tr[k:].reshape(-1)
        r2, pred = ridge_r2(Xk, Yk, tr, ~tr)
        lag_r2.append(float(r2.mean()))
        if k == 0:                                                    # H2: own-patch R^2 by token position
            Yte = Yk[~tr].reshape(N, -1, Yk.shape[-1]); Pte = pred.reshape(N, -1, Yk.shape[-1])
            pos_r2 = (1 - ((Yte - Pte) ** 2).sum((1, 2)) / ((Yte - Yte.mean(1, keepdim=True)) ** 2).sum((1, 2))).cpu().numpy()
    out['readout_patch_lag'] = lag_r2
    Xf = X.reshape(-1, X.shape[-1]); trf = is_tr.reshape(-1)
    out['readout_peak'] = float(ridge_r2(Xf, PEAK.double().to(dev).reshape(-1, 1), trf, ~trf)[0].mean())
    v = ~torch.isnan(RR).reshape(-1).to(dev)
    out['readout_rr'] = float(ridge_r2(Xf[v], RR.double().to(dev).reshape(-1, 1)[v], trf[v], ~trf[v])[0].mean())
    out['rr_valid_share'] = float(v.double().mean()); out['peak_share'] = float(PEAK.double().mean())
    def adj_cos(Z):
        a, b = Z[:-1], Z[1:]
        num = (a * b).sum(-1); den = a.norm(dim=-1) * b.norm(dim=-1)
        ok = den > 0
        return float((num[ok] / den[ok]).mean())
    out['adj_cos_out'] = adj_cos(X); out['adj_cos_in'] = adj_cos(T_)
    out['eff_keys_neo'] = float(np.mean(att['eff_add'])); out['eff_keys_own'] = float(np.mean(att['eff_own']))
    out['zero_rows_neo'] = float(np.mean(att['zero_add']))
    out['neo_term_share'] = math.sqrt(att['n_add']) / (math.sqrt(att['n_own']) + math.sqrt(att['n_add']))
    # H2
    rate = X.mean((1, 2)).cpu().numpy()
    steady = rate[20:].mean()
    out['rate_mean'] = float(rate.mean()); out['rate_first5_over_steady'] = float(rate[:5].mean() / steady) if steady > 0 else float('nan')
    out['rate_slope_per100'] = float(np.polyfit(np.arange(20, N), rate[20:], 1)[0] * 100 / steady) if steady > 0 else float('nan')
    out['r2_first5_over_steady'] = float(pos_r2[:5].mean() / pos_r2[20:].mean())
    out['n_windows'] = int(W_)
    return out, dict(rate_by_pos=rate.tolist(), r2_by_pos=pos_r2.tolist())


def part_h3(m, args, data, init, device, checks):
    sd = m.state_dict()
    num, den, seen = {}, {}, set()
    for k, w0 in init.items():
        g = group_of(k)
        if g is None or w0.dim() < 2 or 'relative_bias' in k or not w0.dtype.is_floating_point:
            continue
        num[g] = num.get(g, 0.) + float(((sd[k].cpu().double() - w0.double()) ** 2).sum()); den[g] = den.get(g, 0.) + float((w0.double() ** 2).sum())
        seen.add(g)
    dist = {g: math.sqrt(num[g] / den[g]) for g in seen if den[g] > 0}
    if 'decoder' not in dist:
        checks.append('no decoder weights to verify the rebuilt init')
    elif dist['decoder'] != 0.:
        checks.append(f"rebuilt init differs from the never-updated decoder ({dist['decoder']:.3g})")
    w = cal_weights(data['counts']).to(device) if args.loss_kind == 'cal' else None
    crit = nn.CrossEntropyLoss(weight=w)
    data['train'].epoch = 0
    gsum = {}; nb = 0
    m.eval(); m.train_mode = 'testing'
    for x, y in rd.loader(data['train'], 32, shuffle=True, seed=int(args.seed), epoch=0):
        m.zero_grad(set_to_none=True); functional.reset_net(m)
        loss = crit(re_._logits(m, x.float().to(device)), y.long().to(device))
        loss.backward()
        for k, p in m.named_parameters():
            g = group_of(k)
            if g is None or p.grad is None or p.dim() < 2 or 'relative_bias' in k:
                continue
            gsum.setdefault(g, []).append(float((p.grad.double() ** 2).sum()))
        nb += 1
        if nb == 16:
            break
    m.zero_grad(set_to_none=True); functional.reset_net(m)
    gn = {g: float(np.mean(np.sqrt(np.array(v).reshape(nb, -1).sum(1)))) for g, v in gsum.items()}
    if 'neocortex' not in gn or 'attn_own' not in gn:
        checks.append('missing gradient groups')
    return dict(init_dist=dist, grad_norm=gn, grad_neo_over_own=gn.get('neocortex', float('nan')) / gn.get('attn_own', float('nan')),
                n_grad_batches=nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['b', 'h5'], required=True)
    ap.add_argument('--run_dir', required=True); ap.add_argument('--name', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--check_only', action='store_true', help='run the checks only (no hypothesis numbers are written)')
    a = ap.parse_args()
    dev = a.device
    m, args, data, init = load(a.run_dir, dev)
    checks = []
    ds, bs = data['test'], int(args.eval_bs)
    logged = dict(next(csv.DictReader(open(os.path.join(a.run_dir, 'log', 'final+result.csv')))))
    saved = np.load(os.path.join(a.run_dir, 'log', 'test_probs.npz'))
    P0, Y0, Z0 = predict(m, ds, dev, bs)
    s0 = score(P0, Y0)
    if not (abs(s0 - float(logged['score'])) < 1e-6 and np.array_equal(Y0, saved['Y']) and float(np.abs(P0 - saved['P']).max()) <= 1e-6):
        checks.append(f"base evaluation {s0:.6f} != logged {logged['score']} (max |P - saved| {float(np.abs(P0 - saved['P']).max()):.3g})")
    N = m.num_patches
    res = dict(name=a.name, run_dir=a.run_dir, kind=a.kind, cond=a.name.rsplit('_seed', 1)[0], seed=int(args.seed), tokens=N,
               score_base=s0, score_logged=float(logged['score']))
    extra = {}
    if a.kind == 'h5':
        S = int(m.stride)
        _, _, Z = predict(m, ds, dev, bs, tile_last(N, N, S))
        if not torch.equal(Z, Z0):
            checks.append('k = N tiling changed the logits')
        with token_shuffle(m, block_perm(N, N, 0)):
            _, _, Z = predict(m, ds, dev, bs)
        if not torch.equal(Z, Z0):
            checks.append('b = N shuffle changed the logits')
        if not a.check_only:
            for k in K_LIST:
                res[f'score_last{k}'] = score(*predict(m, ds, dev, bs, tile_last(k, N, S))[:2])
            for b in B_LIST:
                with token_shuffle(m, block_perm(N, b, 0)):
                    res[f'score_shuf{b}'] = score(*predict(m, ds, dev, bs)[:2])
    else:
        if not getattr(m, 'hippo_xattn', False) or getattr(m, 'hippo_xattn_mode', '') != 'cat':
            raise SystemExit('--kind b needs a B model (hippo_xattn, mode cat)')
        res['memoryless'] = bool(getattr(m, 'neo_memoryless', False))
        with neo_replace(m, 'scramble', torch.arange(N)):
            _, _, Z = predict(m, ds, dev, bs)
        if not torch.equal(Z, Z0):
            checks.append('identity scramble changed the logits')
        perm = torch.as_tensor(np.random.default_rng(1).permutation(N))
        with neo_replace(m, 'scramble', perm):
            Ps, Ys, Z = predict(m, ds, dev, bs)
        if res['memoryless'] and not torch.equal(Z, Z0):
            checks.append(f'memoryless history scramble changed the logits (max {float((Z - Z0).abs().max()):.3g})')
        if not res['memoryless'] and torch.equal(Z, Z0):
            checks.append('T = N history scramble left the logits unchanged')
        h3 = part_h3(m, args, data, init, dev, checks)
        h12, extra = part_h1_h2_h4share(m, args, data, dev, checks)
        if not a.check_only:
            res['score_scramble'] = score(Ps, Ys)
            stats = dict(identity_swaps=0)
            with neo_replace(m, 'swap', stats=stats):
                res['score_swap'] = score(*predict(m, ds, dev, bs)[:2])
            res['identity_swaps'] = stats['identity_swaps']
            with neo_replace(m, 'zero'):
                res['score_zero'] = score(*predict(m, ds, dev, bs)[:2])
            res.update(h12); res.update({f'h3_{k}': v for k, v in h3.items()})
        if res.get('identity_swaps', 0) > 1:
            checks.append(f"{res['identity_swaps']} batches of one window in the swap test")
    for k, v in list(res.items()):
        if isinstance(v, float) and not math.isfinite(v):
            checks.append(f'{k} is not finite')
    res['checks_failed'] = checks
    if not a.check_only:
        out = os.path.join(a.out_dir, a.name); os.makedirs(out, exist_ok=True)
        json.dump(res, open(os.path.join(out, 'cause.json'), 'w'), indent=1)
        if extra:
            json.dump(extra, open(os.path.join(out, 'by_position.json'), 'w'))
    print(json.dumps(dict(name=a.name, checks_failed=checks, n_keys=len(res))) if a.check_only else json.dumps(res))
    if checks:
        print('FAILED CHECKS:\n  ' + '\n  '.join(checks)); sys.exit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()
