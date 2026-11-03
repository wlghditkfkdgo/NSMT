"""MODI 2026-11-03: head readout diagnostic -- does the readout in front of the head throw away task information that the frozen
body already has? (docs/MODI_PROJECT_LOG.md 2026-11-03 15:36, pre-registered; user decision 11-03: diagnose first, ETT cells
ETTh2 / ETTm2 x H96 / H720, no new head on Spikformer-style.) No body training: one trained query-form checkpoint
(nsmt4q_pop_lif / nsmt4q_pop_flif) is frozen and only new linear readouts are fitted on features of the same forward.

Checkpoints: MIT-BIH model_state/best+model.pt of a --mitbih_val run (last epoch = the tested weights); probe train = the 80 %
part, selection = the 20 % part (create_loader's split, as in training), test = the test file. ETT model_state/checkpoint.pth
(best-validation weights = the reported ones); train / val / test = the ETT splits (every window, in order).
Features of the Hippocampus block (T_h steps, N tokens, D channels), from a split-open copy of the model forward:
  fin  final Hippocampus spikes [T_h, B, N, D] (the head input before the means)   emb  embedding spikes [T_h, B, N, D]
  neo  Neocortex output spikes [N, B, 1, D] (the attention query)
  hp / hf  charge H (membrane before the threshold) of the two spike-deleting neurons of the last block, ssa.proj_linear.lif /
       mlp.fc2.lif, rebuilt from their input as the spikingjelly cupy kernel computes it (decay_input, hard reset):
       H_t = v_{t-1} + (1 / tau) (I_t - v_{t-1} + v_reset), s_t = [H_t >= v_th], v_t = v_reset if s_t else H_t, v_0 = v_reset
Readouts (T_h means unless 'step'): R0 the head input (MIT-BIH token mean, ETT per token), R1 per token (MIT-BIH; for ETT R1 = R0),
R2 per T_h step, R3 R0 + emb, R4 R0 + neo, R5 R0 + hp + hf, R6 the raw input (report), R7 all five sources per token (report).
Probes:
  MIT-BIH  multinomial logistic regression on standardised features (80 % part statistics; zero-variance features unscaled),
           CAL-weighted CE as in training (utils.get_class_weights of the 80 % part) + lam ||W||^2 (bias free), lam in LAMS_CLS,
           full-batch L-BFGS from zero (strong Wolfe, at most 1000 iterations); lam by validation macro-F1 (ties -> larger lam);
           test only at the selected lam.
  ETT      ridge with an unpenalised intercept; rows = (window, channel), target = the window-normalised future, row weight =
           stdev^2, so the objective is the reported scaled-data squared error; weighted standardisation (train split);
           penalty lam * sum(w) * ||beta||^2 on the standardised coefficients, lam in LAMS_ETT; lam by validation MSE; solved from
           float64 sufficient statistics; test MSE / MAE recomputed directly in a second pass at the selected lam only.
  Grids (MODI 2026-11-03, set after the R0-only check runs picked the largest value of the first grids, before any other readout
  was fitted): while the best lam is the largest one tried, the grid grows by a factor 10 (up to CAP_CLS / CAP_ETT); a best lam at
  the cap is flagged lam_at_cap.
Fail-closed checks (any failure raises; no JSON is written):
  c54a the split-open forward equals the model's own output on every batch (MIT-BIH logits, ETT forecasts; max |diff| <= 1e-5)
  c54b the trained head reproduces the run's reported test metric (MIT-BIH macro-F1 |diff| <= 1e-6, n; ETT MSE / MAE <= 2e-6, n)
  c54c the spikes implied by the rebuilt charge equal each neuron's output exactly, every batch (and spikes occur)
  c54d MIT-BIH: the 80 / 20 indices are disjoint, cover the training file and give the sampler's class counts; ETT: rows seen =
       windows x channels in every split
  c54e ETT: the statistics-based test MSE of every readout (and of the trained head) equals the direct one (relative 1e-6)
usage: python head_probe.py --task mitbih|ett --run_dir <run> --raw <raw.txt holding the run> --name <name> [--out_dir <dir>]
       [--check_only: the checks with the R0 readout only]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spikingjelly.clock_driven import functional                    # noqa: E402
from load_model import LOAD_MODEL                                    # noqa: E402

LAMS_CLS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
LAMS_ETT = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
CAP_CLS, CAP_ETT = 1e2, 1e6


def lam_path(start, cap, score, better):
    """Evaluates score(lam) on start; while the best lam is the largest tried and below cap, appends 10x that lam.
    better(a, b) is True when score a beats score b (ties -> larger lam). Returns (lams, scores, best, at_cap)."""
    lams, sc = list(start), {}
    i = 0
    while i < len(lams):
        sc[lams[i]] = score(lams[i]); i += 1
        if i == len(lams):
            best = lams[0]
            for l in lams[1:]:
                if not better(sc[best], sc[l]):
                    best = l
            if best == lams[-1] and lams[-1] * 10 <= cap * (1 + 1e-9):
                lams.append(float(f'{lams[-1] * 10:g}'))
    return lams, sc, best, best == lams[-1] and lams[-1] * 10 > cap * (1 + 1e-9)
SRC = ('fin', 'emb', 'neo', 'hp', 'hf')
READ_CLS = {'R0': (('fin', 'mean'),), 'R1': (('fin', 'flat'),), 'R2': (('step', 'flat'),),
            'R3': (('fin', 'mean'), ('emb', 'mean')), 'R4': (('fin', 'mean'), ('neo', 'mean')),
            'R5': (('fin', 'mean'), ('hp', 'mean'), ('hf', 'mean')), 'R6': (('raw', 'flat'),),
            'R7': tuple((s, 'flat') for s in SRC)}
READ_ETT = {'R0': ('fin',), 'R2': ('step',), 'R3': ('fin', 'emb'), 'R4': ('fin', 'neo'), 'R5': ('fin', 'hp', 'hf'),
            'R6': ('raw',), 'R7': SRC}
N_CLASSES = 5


def ref_result(raw, run_dir):
    """The RESULT line of run_dir in raw (exactly one) as a dict."""
    hit = [ln for ln in open(raw) if ln.startswith('RESULT') and ln.rstrip().endswith('dir=' + run_dir.rstrip('/'))]
    if len(hit) != 1:
        raise RuntimeError(f'{len(hit)} RESULT lines for {run_dir} in {raw}')
    return dict(re.findall(r'(\w+)=(\S+)', hit[0]))


class Charge:
    """Forward hook on a spikingjelly MultiStepLIFNode: rebuilds the charge H [T, ...] from the neuron input (c54c)."""
    def __init__(self, lif):
        if type(lif).__name__ != 'MultiStepLIFNode' or lif.v_reset is None or not lif.decay_input:
            raise RuntimeError(f'expected a hard-reset, decay-input MultiStepLIFNode, got {lif}')
        self.on, self.H, self.calls, self.spikes = False, None, 0, 0
        lif.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if not self.on:
            return
        x = inp[0]
        r, vr, th = 1. / float(mod.tau), float(mod.v_reset), float(mod.v_threshold)
        v = torch.full_like(x[0], vr)
        H = torch.empty_like(x)
        for t in range(x.shape[0]):
            h = v + r * (x[t] - v + vr)
            H[t] = h
            v = torch.where(h >= th, torch.full_like(h, vr), h)
        if not torch.equal((H >= th).to(out.dtype), out):
            raise RuntimeError(f'c54c: rebuilt charge implies other spikes ({int(((H >= th).to(out.dtype) != out).sum())} differ)')
        self.H, self.calls, self.spikes = H, self.calls + 1, self.spikes + int(out.sum().item())

    def take(self):
        if self.calls != 1:
            raise RuntimeError(f'c54c: tapped neuron ran {self.calls} times in one forward')
        self.calls = 0
        return self.H


class Body:
    """Split-open copy of the query-form forward (model.myModel.forward, forecast_wrap.Forecaster.step_features)."""
    def __init__(self, m):
        bad = [k for k, ok in (('hippo_xattn', m.hippo_xattn), ('query', m.hippo_xattn_mode == 'query'), ('hippo_ssa', not m.hippo_ssa),
                               ('hippo_only', not m.hippo_only), ('no_neo', not m.no_neo), ('roll_mode', m.roll_mode == 'none'),
                               ('neo_memoryless', not m.neo_memoryless), ('neo_shuffle', m.neo_shuffle is None),
                               ('token_shuffle', m.token_shuffle is None)) if not ok]
        if bad:
            raise RuntimeError(f'expected the plain query form; offending: {bad}')
        self.m = m
        blk = m.data_block[-1]
        self.taps = {'hp': Charge(blk.ssa.proj_linear.lif), 'hf': Charge(blk.mlp.fc2.lif)}

    @torch.no_grad()
    def __call__(self, x):
        """x [B', 1, L] -> dict of [T_h, B', N, D] (fin, emb, hp, hf) and [B', N, D] (neo)."""
        m = self.m
        for t in self.taps.values():
            t.on = True
        patched, neo_in = m._tokens_mc(x)
        e = m.encoding(patched)
        o = m.neo_forward(neo_in, x.clone().detach())[1]
        h = e
        for blk in m.data_block:
            h = blk(h, o, self_kv=m.hippo_only)
        for t in self.taps.values():
            t.on = False
        return dict(fin=h, emb=e, neo=o[:, :, 0].transpose(0, 1), hp=self.taps['hp'].take(), hf=self.taps['hf'].take())


def load(run, task, device):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    args = argparse.Namespace(**cfg)
    args.device = device
    if task == 'mitbih':
        model = LOAD_MODEL[args.model](args, True)
        model.load_state_dict(torch.load(os.path.join(run, 'model_state', 'best+model.pt'), map_location='cpu'))
        inner = model
    else:
        from forecast_wrap import Forecaster
        model = Forecaster(LOAD_MODEL[args.model](args, True), args.embed_dim, args.pred_len, bias=args.bias)
        model.load_state_dict(torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu'))
        inner = model.inner
        if model.kind != 'dual' or model.head.bias is not None:
            raise RuntimeError('expected a bias-free dual-path forecaster')
    model = model.to(device).eval()
    model.train_mode = 'testing'
    functional.reset_net(model)
    return args, model, inner


def macro_f1(pred, y):
    f = []
    for c in range(N_CLASSES):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum()); fn = int(((pred != c) & (y == c)).sum())
        f.append(0. if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), f


# ---------------------------------------------------------------- MIT-BIH
@torch.no_grad()
def cls_features(model, body, loader, device, chk):
    """Per beat: fin / emb / neo / hp / hf [n, N, D] (T_h means), step [n, T_h, D] (token mean), raw [n, L]; labels; logits."""
    out = {k: [] for k in SRC + ('step', 'raw')}
    ys, zs = [], []
    for xb, yb in loader:
        x = xb.float().to(device)
        z_ref = model(x)[0]
        functional.reset_net(model)
        f = body(x)
        z = model.head(f['fin'].mean(2)).mean(0)
        functional.reset_net(model)
        chk['c54a_max_abs'] = max(chk.get('c54a_max_abs', 0.), float((z - z_ref).abs().max()))
        if chk['c54a_max_abs'] > 1e-5:
            raise RuntimeError(f"c54a: split forward differs from the model ({chk['c54a_max_abs']:.3g})")
        for k in ('fin', 'emb', 'hp', 'hf'):
            out[k].append(f[k].mean(0).cpu())
        out['neo'].append(f['neo'].cpu())
        out['step'].append(f['fin'].mean(2).transpose(0, 1).cpu())
        out['raw'].append(x[:, 0].cpu())
        ys.append(yb.long().reshape(-1)); zs.append(z.cpu())
    return {k: torch.cat(v) for k, v in out.items()}, torch.cat(ys), torch.cat(zs)


def cls_matrix(feat, name):
    parts = []
    for s, how in READ_CLS[name]:
        a = feat[s]
        parts.append(a.mean(1) if how == 'mean' else a.reshape(a.shape[0], -1))
    return torch.cat(parts, 1)


def fit_logreg(x, y, w_cls, lam, iters=1000):
    lin = torch.nn.Linear(x.shape[1], N_CLASSES).to(x.device)
    torch.nn.init.zeros_(lin.weight); torch.nn.init.zeros_(lin.bias)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1., max_iter=iters, history_size=20, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(x), y, weight=w_cls) + lam * lin.weight.square().sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        loss = F.cross_entropy(lin(x), y, weight=w_cls) + lam * lin.weight.square().sum()
    return lin, int(opt.state[opt._params[0]]['n_iter']), float(loss)


def run_mitbih(a, args, model, body, device, ref, chk):
    from dataloader import create_loader
    from utils import get_class_weights
    D = os.path.join(args.root_path, '5class_mitbih_train_7_187.csv')
    (tr, va), cns = create_loader(train=True, batch_size=args.batch_size, num_classes=N_CLASSES, data_root=D, train_val_ratio=[4, 1],
                                  data_window_size=187, sampling='None', num_workers=0)
    ti, vi = np.asarray(tr.dataset.indices), np.asarray(va.dataset.indices)
    base = tr.dataset.dataset
    yall = np.asarray(base.y).reshape(-1).astype(int)
    if len(np.intersect1d(ti, vi)) or len(np.union1d(ti, vi)) != len(base) or \
            not np.array_equal(np.bincount(yall[ti], minlength=N_CLASSES), cns.numpy()):
        raise RuntimeError('c54d: unexpected 80 / 20 split')
    chk['c54d'] = dict(train=len(ti), val=len(vi), file=len(base), class_counts=cns.tolist())
    test_dl, _ = create_loader(train=False, batch_size=args.batch_size, num_classes=N_CLASSES,
                               data_root=os.path.join(args.root_path, '5class_mitbih_test_3_187.csv'), data_window_size=187, num_workers=0)
    seq = lambda idx: DataLoader(Subset(base, idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    ftr, ytr, _ = cls_features(model, body, seq(ti), device, chk)
    fva, yva, _ = cls_features(model, body, seq(vi), device, chk)
    fte, yte, zte = cls_features(model, body, test_dl, device, chk)
    f1_head, per_head = macro_f1(zte.argmax(1), yte)
    if abs(f1_head - float(ref['f1'])) > 1e-6 or len(yte) != int(ref['n_test']):
        raise RuntimeError(f"c54b: trained head macro-F1 {f1_head:.6f} (n {len(yte)}) vs reported {ref['f1']} (n {ref['n_test']})")
    chk['c54b'] = dict(head_f1=f1_head, reported_f1=float(ref['f1']), n_test=len(yte))
    chk['c54c_spikes'] = {k: t.spikes for k, t in body.taps.items()}
    if min(chk['c54c_spikes'].values()) <= 0:
        raise RuntimeError('c54c: a tapped neuron never fired')
    w_cls = get_class_weights(class_num_samples=cns).float().to(device)
    res = dict(head=dict(test_f1=f1_head, test_f1_cls=per_head, test_acc=float((zte.argmax(1) == yte).float().mean())))
    ytr_d, yva_d = ytr.to(device), yva.to(device)
    for name in (('R0',) if a.check_only else READ_CLS):
        t0 = time.time()
        xtr = cls_matrix(ftr, name).to(device)
        mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True)
        sd = torch.where(sd > 1e-6, sd, torch.ones_like(sd))
        xtr = (xtr - mu) / sd
        xva = (cls_matrix(fva, name).to(device) - mu) / sd
        fits = {}

        def score(lam):
            fits[lam] = fit_logreg(xtr, ytr_d, w_cls, lam)
            with torch.no_grad():
                return macro_f1(fits[lam][0](xva).argmax(1).cpu(), yva)[0]
        lams, val, best, at_cap = lam_path(LAMS_CLS, CAP_CLS, score, lambda a, b: a > b)
        lin, it, loss = fits[best]
        with torch.no_grad():
            pte = lin((cls_matrix(fte, name).to(device) - mu) / sd).argmax(1).cpu()
        f1, per = macro_f1(pte, yte)
        res[name] = dict(dim=int(xtr.shape[1]), lam=best, lam_at_cap=at_cap, val_f1={str(l): v for l, v in val.items()}, test_f1=f1,
                         test_f1_cls=per, test_acc=float((pte == yte).float().mean()), n_iter={str(l): fits[l][1] for l in lams},
                         loss={str(l): fits[l][2] for l in lams}, time_s=round(time.time() - t0, 1))
        print(f"{name} dim {xtr.shape[1]} lam {best:g}{' (cap)' if at_cap else ''} val {val[best]:.4f} test {f1:.4f} "
              f"iters {[fits[l][1] for l in lams]}", flush=True)
        del xtr, xva, fits
        torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- ETT
class Stats:
    """float64 weighted sufficient statistics of [X - shift, 1] and Y. MODI 2026-11-03 (after the first full ETTh2 run): every
    feature is shifted by a fixed reference (the mean of the first training batch, shared by train / val / test) before it is
    accumulated. Centring makes the ridge solution shift-invariant; without the shift, nearly constant membrane features
    (var / mean^2 ~ 1e-9) lost their variance to cancellation in G - sw mu mu^T and the standardised system got negative
    eigenvalues (-3.4e-6 sw), so Cholesky failed at lam 1e-6 (ETTh2 H720 pop LIF seed 42, readouts R5 / R7)."""
    def __init__(self, d, H, device, shift):
        self.G = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=device)
        self.b = torch.zeros(d + 1, H, dtype=torch.float64, device=device)
        self.syy, self.n, self.shift = 0., 0, shift

    def add(self, X, Y, w):
        Xa = torch.cat([X.double() - self.shift, torch.ones(X.shape[0], 1, dtype=torch.float64, device=X.device)], 1)
        Wx = Xa * w[:, None]
        self.G += Wx.T @ Xa
        self.b += Wx.T @ Y
        self.syy += float((w[:, None] * Y.square()).sum())
        self.n += X.shape[0]

    def sse(self, B):
        return float((B * (self.G @ B)).sum() - 2 * (B * self.b).sum()) + self.syy


def ridge(st, lam):
    d = st.G.shape[0] - 1
    sw = st.G[d, d]
    mu = st.G[:d, d] / sw
    C = st.G[:d, :d] - sw * mu[:, None] * mu[None, :]
    var = torch.diagonal(C) / sw
    sd = torch.where(var > 1e-12, var.clamp(min=0).sqrt(), torch.ones_like(var))
    Sy = st.b[d]
    Cs = C / (sd[:, None] * sd[None, :])
    bc = (st.b[:d] - mu[:, None] * Sy[None, :]) / sd[:, None]
    L, info = torch.linalg.cholesky_ex(Cs + lam * sw * torch.eye(d, dtype=Cs.dtype, device=Cs.device))
    if int(info) != 0:
        raise RuntimeError(f'ridge system not positive definite (lam {lam})')
    beta = torch.cholesky_solve(bc, L) / sd[:, None]
    return torch.cat([beta, (Sy / sw - mu @ beta)[None]], 0)


@torch.no_grad()
def ett_batches(model, body, loader, H, device, chk):
    """Yields (sources [B*C, ...] flattened per row, target yn [B*C, H] float64, row stdev [B*C] float64, forecast, y)."""
    for x, y, _, _ in loader:
        x = x.float().to(device); y = y.float().to(device)[:, -H:, :]
        ref = model(x)
        functional.reset_net(model)
        B, L, C = x.shape
        means = x.mean(1, keepdim=True)
        xc = x - means
        stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        z = (xc / stdev).permute(0, 2, 1).reshape(B * C, 1, L)
        f = body(z)
        functional.reset_net(model)
        pn = model.head(f['fin'].mean(0).reshape(B * C, -1))
        out = pn.reshape(B, C, H).permute(0, 2, 1) * stdev + means
        chk['c54a_max_abs'] = max(chk.get('c54a_max_abs', 0.), float((out - ref).abs().max()))
        if chk['c54a_max_abs'] > 1e-5:
            raise RuntimeError(f"c54a: split forward differs from the model ({chk['c54a_max_abs']:.3g})")
        src = {k: f[k].mean(0).reshape(B * C, -1) for k in ('fin', 'emb', 'hp', 'hf')}
        src['neo'] = f['neo'].reshape(B * C, -1)
        src['step'] = f['fin'].transpose(0, 1).reshape(B * C, -1)
        src['raw'] = z[:, 0]
        yn = ((y - means) / stdev).permute(0, 2, 1).reshape(B * C, H).double()
        yield src, yn, stdev[:, 0, :].reshape(B * C).double(), ref, y


def run_ett(a, args, model, body, device, ref, chk):
    sys.path.append(os.path.join(HERE, '..', 'forecasting'))         # data_provider only, as train_forecast.py
    from train_forecast import data_args
    from data_provider.data_factory import data_provider
    H = int(args.pred_len)
    da = data_args(args, args)
    sets = {s: data_provider(da, s)[0] for s in ('train', 'val')}
    loaders = {s: DataLoader(sets[s], batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False) for s in sets}
    ds_te, loaders['test'] = data_provider(da, 'test')
    sets['test'] = ds_te
    reads = {r: READ_ETT[r] for r in (('R0',) if a.check_only else READ_ETT)}
    dims, stats, shift = None, {}, {}
    for split in ('train', 'val', 'test'):
        for src, yn, sd, _, _ in ett_batches(model, body, loaders[split], H, device, chk):
            if dims is None:
                dims = {r: sum(src[s].shape[1] for s in reads[r]) for r in reads}
                shift = {r: torch.cat([src[s] for s in reads[r]], 1).double().mean(0) for r in reads}   # first training batch
            for r in reads:
                st = stats.setdefault((split, r), Stats(dims[r], H, device, shift[r]))
                st.add(torch.cat([src[s] for s in reads[r]], 1), yn, sd.square())
        n_rows = stats[(split, 'R0')].n
        C = int(next(iter(loaders[split]))[0].shape[2])
        if n_rows != len(sets[split]) * C:
            raise RuntimeError(f'c54d: {split} rows {n_rows} != {len(sets[split])} windows x {C} channels')
        chk.setdefault('c54d', {})[split] = dict(windows=len(sets[split]), rows=n_rows)
    chk['c54c_spikes'] = {k: t.spikes for k, t in body.taps.items()}
    if min(chk['c54c_spikes'].values()) <= 0:
        raise RuntimeError('c54c: a tapped neuron never fired')
    sel, res = {}, {}
    for r in reads:
        Bs = {}

        def score(lam):
            Bs[lam] = ridge(stats[('train', r)], lam)
            st = stats[('val', r)]
            return st.sse(Bs[lam]) / (st.n * H)
        lams, val, best, at_cap = lam_path(LAMS_ETT, CAP_ETT, score, lambda a, b: a < b)
        sel[r] = Bs[best]
        st = stats[('test', r)]
        res[r] = dict(dim=dims[r], lam=best, lam_at_cap=at_cap, val_mse={str(l): v for l, v in val.items()},
                      test_mse_stats=st.sse(sel[r]) / (st.n * H))
    Wh = model.head.weight.detach().double().T
    B_head = torch.cat([Wh, (shift['R0'] @ Wh)[None]], 0)                # x W = (x - shift) W + shift W
    st = stats[('test', 'R0')]
    head_stats = st.sse(B_head) / (st.n * H)
    del stats
    torch.cuda.empty_cache()
    acc = {r: [0., 0.] for r in reads}
    hs = ha = 0.; cnt = 0
    for src, yn, sd, out, y in ett_batches(model, body, loaders['test'], H, device, chk):
        for r in reads:
            X = torch.cat([src[s] for s in reads[r]], 1).double() - shift[r]
            p = X @ sel[r][:-1] + sel[r][-1]
            e = sd[:, None] * (p - yn)
            acc[r][0] += float(e.square().sum()); acc[r][1] += float(e.abs().sum())
        d = (out - y).double()
        hs += float(d.square().sum()); ha += float(d.abs().sum()); cnt += d.numel()
    if cnt != int(ref['n']):
        raise RuntimeError(f"c54b: {cnt} test elements vs reported {ref['n']}")
    head_mse, head_mae = hs / cnt, ha / cnt
    if abs(head_mse - float(ref['mse'])) > 2e-6 or abs(head_mae - float(ref['mae'])) > 2e-6:
        raise RuntimeError(f"c54b: trained head test MSE / MAE {head_mse:.6f} / {head_mae:.6f} vs reported {ref['mse']} / {ref['mae']}")
    chk['c54b'] = dict(head_mse=head_mse, head_mae=head_mae, reported_mse=float(ref['mse']), reported_mae=float(ref['mae']), n=cnt)
    rel = abs(head_stats - head_mse) / head_mse
    chk['c54e_max_rel'] = rel
    for r in reads:
        res[r]['test_mse'], res[r]['test_mae'] = acc[r][0] / cnt, acc[r][1] / cnt
        rel = max(rel, abs(res[r]['test_mse_stats'] - res[r]['test_mse']) / res[r]['test_mse'])
        print(f"{r} dim {dims[r]} lam {res[r]['lam']:g} val {res[r]['val_mse'][str(res[r]['lam'])]:.6f} "
              f"test {res[r]['test_mse']:.6f} / {res[r]['test_mae']:.6f}", flush=True)
    chk['c54e_max_rel'] = rel
    if rel > 1e-6:
        raise RuntimeError(f'c54e: statistics-based and direct test MSE differ (relative {rel:.3g})')
    res['head'] = dict(test_mse=head_mse, test_mae=head_mae, test_mse_stats=head_stats)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=('mitbih', 'ett'), required=True)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'results', 'flif-v1-head-probe'))
    ap.add_argument('--check_only', action='store_true', help='checks c54a-e with the R0 readout only (no other readout is fitted)')
    a = ap.parse_args()
    t0 = time.time()
    device = 'cuda'
    run = a.run_dir.rstrip('/')
    ref = ref_result(a.raw, run)
    args, model, inner = load(run, a.task, device)
    body = Body(inner)
    chk = {}
    res = (run_mitbih if a.task == 'mitbih' else run_ett)(a, args, model, body, device, ref, chk)
    os.makedirs(a.out_dir, exist_ok=True)
    meta = dict(name=a.name, task=a.task, run_dir=run, raw=os.path.abspath(a.raw), cond=ref.get('cond'), seed=int(ref['seed']),
                data=ref.get('ds', ref.get('dataset')), pred_len=ref.get('pl'), checks=chk, readouts=res,
                lams_start=list(LAMS_CLS if a.task == 'mitbih' else LAMS_ETT), lam_cap=CAP_CLS if a.task == 'mitbih' else CAP_ETT,
                check_only=a.check_only,
                script_sha256=hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0), time_s=round(time.time() - t0, 1))
    json.dump(meta, open(os.path.join(a.out_dir, f'{a.name}.json'), 'w'), indent=1)
    print(f'checks {json.dumps(chk)}\ndone {a.name} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
