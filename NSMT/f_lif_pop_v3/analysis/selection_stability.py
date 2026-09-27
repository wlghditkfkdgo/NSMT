#!/usr/bin/env python3
"""Prereg 2Q (A) D-CA: why does statistical selection grow the branch state? (exploratory)

Train split only, CPU, float64, no training. Inputs are the 2P diagnostic's: ETTh1/ETTh2 train,
the first 8 batches shuffled at seed 7, window normalisation R on.

Currents come from three models per dataset: a fresh embedding at seed 7 for alpha=0.7 and for
alpha=1 (each at its own calibrated scale, frozen norm fitted on the 8 batches), and the trained
2O pearson_R (alpha=0.7) seed-7 checkpoint. The branch recursion is re-implemented here:
branches never reset and the selection reads only xi = [u, I], so given the currents the branch
states do not depend on the soma. It is checked against the model's own neuron first (D-CA).

Rules: q1 (no selection); q=0.5 pearson (shared axis), recent, random (mask seeds 0-7).

Measurements M1-M8 as registered in D-CA. Aggregations, fixed here before the first run:
  - "step-averaged" means the mean over steps t = 10..41 (selection is trivial for small t).
  - M4 fits, per branch and step, H_t = beta_u u_t + beta_f f_t (no intercept) over all
    sequence x unit cells by least squares; lambda = beta_u - (b_0 + beta_f) / tau.
  - Reading rule for H1 (applied to each dataset x model x alpha, pearson against q1):
      (1) in both fast branches (tau 4 and 8) the step-averaged beta_u is below q1's and the
          step-averaged beta_f is above q1's;
      (2) the first step with |lambda| > 1 occurs in a fast branch (tau 4 or 8) and no slow
          branch (16, 32) reaches |lambda| > 1 earlier; if no branch reaches it, (2) fails;
      (3) the cell with the largest |u| changes sign on at least half of the steps 20..40
          (u_{t+1} u_t < 0).
    All three -> "supported"; otherwise "not supported" with the failing items.
"""
import sys
import json
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers                                                   # noqa: E402
from config import TASK, neuron_kwargs                          # noqa: E402
from data_provider.data_factory import data_provider            # noqa: E402
from ours import window_norm                                    # noqa: E402
from alpha1_g11_diag import config_for, CAL                     # noqa: E402
import ett_alpha as A                                           # noqa: E402

DT = torch.float64
FAST, T0, SIGN_FROM = (0, 1), 10, 20
RANDOM_SEEDS = range(8)


# ---- the branch recursion ----------------------------------------------------------------------
def keep_of(q, J):
    return max(1, int(round(q * J)))


def mask_of(rule, q, xi, hist, gen=None):
    """[S, D, K+1], [S, D, J, K+1] -> [S, 1, J] in {0,1}; the same rules as layers.Selector.hard_mask."""
    S, D, J, F = hist.shape
    if q >= 1.:
        return hist.new_ones(S, 1, J)
    k = keep_of(q, J)
    if rule == 'recent':
        sim = torch.arange(J, dtype=DT).expand(S, 1, J)
    elif rule == 'random':
        sim = torch.rand(S, 1, J, generator=gen).to(DT)
    else:                                                        # pearson, shared axis
        a = xi.reshape(S, 1, 1, D * F)
        h = hist.permute(0, 2, 1, 3).reshape(S, 1, J, D * F)
        a, h = a - a.mean(-1, keepdim=True), h - h.mean(-1, keepdim=True)
        sim = (a * h).sum(-1) / (a.square().sum(-1).sqrt() * h.square().sum(-1).sqrt()).clamp_min(1e-12)
    idx = sim.topk(k, dim=-1).indices
    return torch.zeros(S, 1, J, dtype=DT).scatter_(-1, idx, 1.)


def run(I, b, tau, rule='pearson', q=.5, masks=None, leak_keep=False, gen=None, measure=False):
    """I: [T, S, D] currents. Returns states [T, S, D, K] (u(t+1) at index t), the masks used,
    and (measure=True) the per-step quantities of M1-M5."""
    T, S, D = I.shape
    K = tau.numel()
    b0 = b[0].item()
    u = torch.zeros(S, D, K, dtype=DT)
    incs, inps, leaks, keys, states, used, rows = [], [], [], [], [], [], []
    for t in range(T):
        inp = I[t].unsqueeze(-1) / tau                           # input part of the increment
        leak = -u / tau                                          # leak part
        f = inp + leak
        xi = torch.cat([u, I[t].unsqueeze(-1)], dim=-1)
        nxt = b0 * f
        row = None
        if t:
            b_hist = b[1:t + 1].flip(0)                          # oldest slot first
            m = masks[t] if masks is not None else mask_of(rule, q, xi, torch.stack(keys, dim=-2), gen)
            used.append(m)
            c = b_hist * m                                       # [S, 1, t]
            past = torch.stack(incs, dim=-2)                     # [S, D, t, K]
            if leak_keep:                                        # M8: select the input part only
                H = torch.einsum('sdj,sdjk->sdk', c.expand(S, D, t), torch.stack(inps, dim=-2)) \
                    + torch.einsum('j,sdjk->sdk', b_hist, torch.stack(leaks, dim=-2))
            else:
                H = torch.einsum('sdj,sdjk->sdk', c.expand(S, D, t), past)
            nxt = nxt + H
            if measure:
                row = measure_step(t, m, b_hist, c, past, torch.stack(inps, dim=-2), torch.stack(leaks, dim=-2),
                                   H, u, f, tau, b0)
        else:
            used.append(None)
        incs.append(f); inps.append(inp); leaks.append(leak); keys.append(xi)
        u = nxt
        states.append(u)
        rows.append(row)

    return torch.stack(states), used, rows


def measure_step(t, m, b_hist, c, past, inps, leaks, H, u, f, tau, b0):
    """M1-M5 at step t (H is the history term that builds u(t+1); u, f are u(t), f(t))."""
    S = m.shape[0]
    lag = torch.arange(t, 0, -1, dtype=DT)                       # slot j (oldest first) has lag t-j
    kept = m[:, 0, :]
    full = b_hist.expand(S, 1, t)
    H_in = torch.einsum('sdj,sdjk->sdk', c.expand(-1, past.shape[1], -1), inps)
    H_lk = torch.einsum('sdj,sdjk->sdk', c.expand(-1, past.shape[1], -1), leaks)
    F_in = torch.einsum('sdj,sdjk->sdk', full.expand(-1, past.shape[1], -1), inps)
    F_lk = torch.einsum('sdj,sdjk->sdk', full.expand(-1, past.shape[1], -1), leaks)
    excl = (b_hist - c).expand(-1, past.shape[1], -1)
    out = {'t': t, 'keep': int(kept[0].sum().item()),
           'mean_lag': ((kept * lag).sum(-1) / kept.sum(-1)).mean().item(),               # M1
           'kept_mass': ((kept * b_hist).sum(-1) / b_hist.sum()).mean().item(),          # M2
           'H_in': H_in.abs().mean().item(), 'H_leak': H_lk.abs().mean().item(),         # M3
           'full_in': F_in.abs().mean().item(), 'full_leak': F_lk.abs().mean().item(),
           'sum_kept': H.abs().mean().item(),
           'sum_excluded': torch.einsum('sdj,sdjk->sdk', excl, past).abs().mean().item(),
           'abs_kept': torch.einsum('sdj,sdjk->sdk', c.expand(-1, past.shape[1], -1), past.abs()).mean().item(),
           'abs_excluded': torch.einsum('sdj,sdjk->sdk', excl, past.abs()).mean().item(),
           'beta_u': [], 'beta_f': [], 'lambda': [], 'r2': [], 'cos_u': [], 'cos_f': []}
    for k in range(tau.numel()):                                 # M4, M5 per branch
        y = H[..., k].reshape(-1)
        X = torch.stack([u[..., k].reshape(-1), f[..., k].reshape(-1)], dim=1)
        beta = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1)
        res = y - X @ beta
        out['beta_u'].append(beta[0].item()); out['beta_f'].append(beta[1].item())
        out['lambda'].append(beta[0].item() - (b0 + beta[1].item()) / tau[k].item())
        out['r2'].append(1. - (res.square().sum() / (y - y.mean()).square().sum().clamp_min(1e-300)).item())
        cos = lambda a, v: (a @ v / (a.norm() * v.norm()).clamp_min(1e-300)).item()
        out['cos_u'].append(cos(y, X[:, 0])); out['cos_f'].append(cos(y, X[:, 1]))

    return out


# ---- summaries -------------------------------------------------------------------------------
def summarise(states, rows):
    peak = states.abs().max().item()
    T, S, D, K = states.shape
    flat = states.abs().reshape(T, -1).max(0)                    # the cell holding the peak
    cell = int(flat.values.argmax())
    s, rest = divmod(cell, D * K)
    d, k = divmod(rest, K)
    path = states[:, s, d, k]
    flips = [(path[t + 1] * path[t] < 0).item() for t in range(SIGN_FROM, T - 1)]
    steps = [r for r in rows if r is not None and r['t'] >= T0]
    avg = lambda key, i: sum(r[key][i] for r in steps) / len(steps)
    first = {}
    for kk in range(K):
        hit = [r['t'] for r in rows if r is not None and abs(r['lambda'][kk]) > 1.]
        first[kk] = hit[0] if hit else None
    return {'peak': peak, 'peak_cell': {'sequence': s, 'unit': d, 'branch': k, 'path': path.tolist(),
                                        'sign_flip_share_20_40': sum(flips) / len(flips)},
            'beta_u': [avg('beta_u', i) for i in range(K)], 'beta_f': [avg('beta_f', i) for i in range(K)],
            'lambda_avg': [avg('lambda', i) for i in range(K)],
            'lambda_max': [max(abs(r['lambda'][i]) for r in rows if r is not None) for i in range(K)],
            'first_lambda_gt1': first, 'r2': [avg('r2', i) for i in range(K)],
            'cos_u': [avg('cos_u', i) for i in range(K)], 'cos_f': [avg('cos_f', i) for i in range(K)],
            'mean_lag': sum(r['mean_lag'] for r in steps) / len(steps),
            'kept_mass': sum(r['kept_mass'] for r in steps) / len(steps),
            'H_in_ratio': sum(r['H_in'] for r in steps) / sum(r['full_in'] for r in steps),
            'H_leak_ratio': sum(r['H_leak'] for r in steps) / sum(r['full_leak'] for r in steps),
            'sum_kept': sum(r['sum_kept'] for r in steps) / len(steps),
            'sum_excluded': sum(r['sum_excluded'] for r in steps) / len(steps),
            'abs_kept': sum(r['abs_kept'] for r in steps) / len(steps),
            'abs_excluded': sum(r['abs_excluded'] for r in steps) / len(steps),
            'steps': rows}


def verdict(p, q1):
    one = all(p['beta_u'][i] < q1['beta_u'][i] and p['beta_f'][i] > q1['beta_f'][i] for i in FAST)
    hits = {k: v for k, v in p['first_lambda_gt1'].items() if v is not None}
    fast = [hits[f] for f in FAST if f in hits]
    slow = [hits[s] for s in (2, 3) if s in hits]
    two = bool(fast) and (not slow or min(fast) <= min(slow))   # a fast branch first; a tie is not "earlier"
    three = p['peak_cell']['sign_flip_share_20_40'] >= .5
    failing = [n for n, ok in (('(1) beta', one), ('(2) first |lambda|>1 in a fast branch', two),
                               ('(3) sign alternation', three)) if not ok]
    return 'supported' if not failing else 'not supported: ' + ', '.join(failing)


# ---- inputs ----------------------------------------------------------------------------------
def batches_of(data):
    c = config_for(data, .7, 1., torch.device('cpu'))
    _, loader = data_provider(c, 'train')
    out = []
    for i, batch in enumerate(loader):
        if i >= 8:
            break
        out.append(layers.to_patches(window_norm(batch[0].float())[0], c.patch_size))
    return torch.cat(out, dim=1)                                 # [42, S, 8]


def init_embedding(data, alpha, patches):
    cal = json.load(open(TASK / 'results' / 'calibration' / CAL[(data, alpha)]))['picked']
    c = config_for(data, alpha, 1., torch.device('cpu'))
    torch.manual_seed(7)
    emb = layers.Embedding(c.patch_size, c.embed_dim, cal['input_scale'], input_norm='frozen', **neuron_kwargs(c))
    emb.fit_norm(patches)
    return emb


@torch.no_grad()
def check_against_model(neuron, I, rule_q):
    """Model's own neuron (float64) vs the recursion here, for q1 and pearson q=0.5."""
    worst = 0.
    for q in rule_q:
        neuron.selector.hard_q = q
        st_model = neuron(I, mode='hard', return_aux=True)[1]['state']
        st_here, _, _ = run(I, neuron.b.to(DT), neuron.tau.to(DT), 'pearson', q)
        worst = max(worst, ((st_model - st_here).abs().max() / st_model.abs().max()).item())
    return worst


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    record, lines = {}, []
    t_start = time.time()
    for data in A.DATASETS:
        patches = batches_of(data).to(DT)
        print(f"[stab] {data}: {patches.shape[1]} sequences x {patches.shape[0]} steps")
        models = {}
        for alpha in (.7, 1.):
            emb = init_embedding(data, alpha, patches.float()).double()
            models[f'init a{alpha:g}'] = emb
        _, run_dir, _ = A.find_cell(data, 96, 'pearson_R', 7)
        trained, _ = A.load_model(run_dir, torch.device('cpu'))
        models['trained 2O pearson_R a0.7 seed7'] = trained.embedding.double()

        for name, emb in models.items():
            I = emb.current(patches)                             # [42, S, 32]
            neuron = emb.neuron
            b, tau = neuron.b.to(DT), neuron.tau.to(DT)
            rel = check_against_model(neuron, I, (1., .5))
            print(f"[stab] {data} {name}: recursion vs model state, max relative diff {rel:.2e}")
            if rel > 1e-4:
                raise SystemExit(f"[stab] recursion does not reproduce the model ({rel:.2e} > 1e-4); stop (D-CA)")
            res = {'check_rel_diff': rel}
            for rule, q in (('q1', 1.), ('pearson', .5), ('recent', .5)):
                st, used, rows = run(I, b, tau, 'pearson' if rule == 'q1' else rule, q, measure=True)
                res[rule] = summarise(st, rows)
                res[rule]['_masks'] = used
            rnd = []
            for seed in RANDOM_SEEDS:
                gen = torch.Generator().manual_seed(seed)
                st, _, rows = run(I, b, tau, 'random', .5, gen=gen, measure=True)
                rnd.append(summarise(st, rows))
            res['random'] = {'peaks': [r['peak'] for r in rnd], **{k: rnd[0][k] for k in ('beta_u', 'beta_f', 'lambda_avg')},
                             'lambda_max_over_seeds': [max(r['lambda_max'][i] for r in rnd) for i in range(4)]}
            res['verdict_H1'] = verdict(res['pearson'], res['q1'])
            # M8: leak kept for every past slot, input part selected
            st, _, rows = run(I, b, tau, 'pearson', .5, leak_keep=True, measure=True)
            res['M8_leak_kept_pearson'] = summarise(st, rows)
            record[f'{data} | {name}'] = res
            for key in ('q1', 'pearson', 'recent'):
                r = res[key]
                lines.append(f"{data:5s} {name:32s} {key:8s} peak {r['peak']:9.2f}  beta_u {fmt(r['beta_u'])}  "
                             f"beta_f {fmt(r['beta_f'])}  lambda_avg {fmt(r['lambda_avg'])}  |lambda|max {fmt(r['lambda_max'])}  "
                             f"first|l|>1 {r['first_lambda_gt1']}  cos_u {fmt(r['cos_u'])}  cos_f {fmt(r['cos_f'])}  "
                             f"lag {r['mean_lag']:.2f}  mass {r['kept_mass']:.3f}  in/full {r['H_in_ratio']:.3f}  "
                             f"leak/full {r['H_leak_ratio']:.3f}  sign-flip {r['peak_cell']['sign_flip_share_20_40']:.2f}")
            rr = res['random']
            lines.append(f"{data:5s} {name:32s} random   peaks {min(rr['peaks']):.2f}-{max(rr['peaks']):.2f} over mask seeds 0-7  "
                         f"|lambda|max {fmt(rr['lambda_max_over_seeds'])}")
            m8 = res['M8_leak_kept_pearson']
            lines.append(f"{data:5s} {name:32s} M8 leak-kept pearson peak {m8['peak']:9.2f}  lambda_avg {fmt(m8['lambda_avg'])}  "
                         f"|lambda|max {fmt(m8['lambda_max'])}")
            lines.append(f"{data:5s} {name:32s} H1 -> {res['verdict_H1']}")
            print('\n'.join(lines[-6:]))

        # M7: fixed-mask replay across alpha (init models)
        I7, I1 = models['init a0.7'].current(patches), models['init a1'].current(patches)
        n7, n1 = models['init a0.7'].neuron, models['init a1'].neuron
        m7 = record[f'{data} | init a0.7']['pearson']['_masks']
        m1 = record[f'{data} | init a1']['pearson']['_masks']
        st_1_with_7, _, _ = run(I1, n1.b.to(DT), n1.tau.to(DT), masks=m7)
        st_7_with_1, _, _ = run(I7, n7.b.to(DT), n7.tau.to(DT), masks=m1)
        rep = {'alpha1_dynamic': record[f'{data} | init a1']['pearson']['peak'],
               'alpha1_with_alpha0.7_masks': st_1_with_7.abs().max().item(),
               'alpha0.7_dynamic': record[f'{data} | init a0.7']['pearson']['peak'],
               'alpha0.7_with_alpha1_masks': st_7_with_1.abs().max().item()}
        record[f'{data} | M7 replay'] = rep
        lines.append(f"{data:5s} M7 replay: alpha=1 dynamic {rep['alpha1_dynamic']:.2f}, alpha=1 with alpha=0.7's masks "
                     f"{rep['alpha1_with_alpha0.7_masks']:.2f}; alpha=0.7 dynamic {rep['alpha0.7_dynamic']:.2f}, "
                     f"alpha=0.7 with alpha=1's masks {rep['alpha0.7_with_alpha1_masks']:.2f}")
        print(lines[-1])

    for v in record.values():                                    # masks are not written out
        for key in ('q1', 'pearson', 'recent'):
            if isinstance(v, dict) and key in v:
                v[key].pop('_masks', None)
    out = Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(record, indent=1, allow_nan=False))
    Path(__file__).with_suffix('.txt').write_text('\n'.join(lines) + '\n')
    print(f"[stab] done in {time.time() - t_start:.0f} s; {out.name}, {out.with_suffix('.txt').name}")


def fmt(v):
    return '[' + ' '.join(f'{x:+.3f}' for x in v) + ']'


if __name__ == '__main__':
    main()
