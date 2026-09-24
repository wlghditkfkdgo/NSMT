"""Screen the user's design: keep the fractional kernel, gate each past slot by a statistic.

The design (confirmed 2026-09-24) is

    u(n+1,k) = b_0 f(n,k) + sum_{j<n} m(n,j) * b(n-j) * f(j,k),     m(n,j) in {0, 1}

with m decided by a statistical relevance rule -- NOT multiplied into the coefficient the way
the current eta/rho/cap selector does. So m can only remove terms: c = m*b <= b <= b_0 and no
cap is needed for that bound, there is no eta to dilute, and there is no learned selector to
backpropagate through. Removing terms still changes every later increment, so the masked
trajectory is NOT bounded by the unmasked one (audit 35); max|u| is measured, not assumed.

This script trains nothing. It asks one question on the eta=0 checkpoint (whose trajectory IS
the plain f-LIF): does a statistic between the current descriptor and each past descriptor keep
the answer slots and drop the rest? Reported as the answer's share of the kept kernel mass,

    M_eff(hard) = sum_{j in A} m b(n-j) / sum_j m b(n-j)

against (i) the same quantity at m == 1, computed in this run on this sample (the plain kernel),
(ii) the oracle mask (1.0 by definition, a privileged ceiling, not a matched control), and (iii)
a random control that keeps the SAME number of slots per query, drawn uniformly from the slots
that exist at that query (audit A14-CONTROL).

Monte Carlo error (audit 37/38). The random control is a Monte Carlo estimate. Its error is the
spread of the FINAL aggregate across draws: every draw is carried through the full
unit -> query -> sequence aggregation, and `mc_se` is the standard deviation over draws of the
sequence mean, divided by sqrt(draws). It says how well the control is estimated, nothing about
sequence-to-sequence or seed-to-seed spread, and it is not a test statistic. The random stream
is keyed by the rule's name and the batch index, so the control does not depend on the order
in which rules are evaluated.

Ties (audit 36/37). Similarities can tie exactly (a constant descriptor falls back to 0;
identical input patches). The rule here is declared: descending stable sort, so ties are
broken toward the OLDER slot. `tie_frac` reports how often a tie touches the decision (top-q:
the k-th and (k+1)-th similarity are equal; ranks: any exact duplicate among the n slots). The
rank chance (n-m)/((m+1)(n-1)) assumes no ties and is only meaningful where tie_frac is small.
The model's own hard_mask uses torch.topk, whose tie order is implementation-defined; whether
that ever matters is exactly what tie_frac on the shared axis measures.

Three descriptor axes. The numbers 5 / 160 / 32 are COMPONENT COUNTS of the descriptor, not
independent sample sizes, so no t-test or df claim is made from them.

    unit    : one unit's xi = [u_1..u_K ; I], 5 components
    shared  : all D units' xi flattened, D*(K+1) = 160 components  (one decision for the population)
    input   : the embedded input patch, D = 32 components          (no state in the mask decision)

Reference distribution for the thresholds: on the TRAIN split, the current descriptor against
the past of a DIFFERENT sequence in the batch at the same positions, pooled over recall
queries. That is a cross-sequence reference conditioned on train's event labels, not a
label-free null and not a guaranteed exchangeable permutation null: sequences share the same
small cue alphabet. Quantiles 0.90/0.95/0.99, plus a top-q% rule per query. Validation only
for the measurements; the confirm splits are never loaded.

Two settings. `frozen` computes m on the eta=0 trajectory and only re-weights the kernel.
`closed` re-runs the recurrence with the mask applied at every step, so the descriptors
themselves come from the masked states -- the real closed loop. The closed-loop recurrence is
re-implemented here and checked against the model at m == 1. A rule with any non-finite
closed-loop batch is reported as NONFIN with no metrics: it is unevaluable, not averaged over
the batches that survived.
"""
import sys
import json
import math
import hashlib
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hypergeom

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'forecasting'))

from config import Config, parse_defaults                      # noqa: E402
from model import LOAD_MODEL                                   # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
import layers                                                  # noqa: E402

AXES = ('unit', 'shared', 'input')
STATS = ('pearson', 'cosine')
QUANTILES = (.90, .95, .99)
TOPQ = (.10, .25, .50)
FIELDS = ('meff', 'kept_frac', 'kept', 'empty', 'answer_kept', 'any_hit', 'any_hit_random', 'tie_frac')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream_seed(seed, key, batch):
    """Random stream keyed by the rule's NAME and the batch, not by evaluation order (audit 37)."""
    tag = hashlib.sha256(f'{"/".join(key)}|{batch}'.encode()).hexdigest()[:8]
    return (seed * 1000003 + int(tag, 16)) % (2 ** 63 - 1)


def centre(a, do):
    return a - a.mean(-1, keepdim=True) if do else a


def similarity(a, b, stat):
    """a: [..., 1, F] against b: [..., J, F] -> [..., J]. Pearson = cosine of centred vectors.

    A constant descriptor has zero centred norm; the clamp makes its similarity 0, not NaN.
    """
    a, b = centre(a, stat == 'pearson'), centre(b, stat == 'pearson')
    num = (a * b).sum(-1)
    den = (a.square().sum(-1).sqrt() * b.square().sum(-1).sqrt()).clamp_min(1e-12)
    return num / den


def ranks_desc(sim):
    """0 = most similar; ties broken toward the older slot (descending STABLE sort)."""
    order = torch.sort(sim, dim=-1, descending=True, stable=True).indices
    return order.argsort(-1)


def descriptors(state, current, n):
    """xi_n and the history xi_j (j < n) in the three axes. state[t] = u_{t+1}, u_0 = 0."""
    B, D, K = state.shape[1:]
    zero = torch.zeros_like(state[0])
    xi = torch.cat([state[n - 1], current[n].unsqueeze(-1)], dim=-1)                   # [B, D, K+1]
    hist = torch.stack([torch.cat([state[j - 1] if j else zero, current[j].unsqueeze(-1)], -1)
                        for j in range(n)], dim=-2)                                     # [B, D, n, K+1]
    return {
        'unit':   (xi.unsqueeze(-2), hist),
        'shared': (xi.reshape(B, 1, 1, D * (K + 1)), hist.permute(0, 2, 1, 3).reshape(B, 1, n, D * (K + 1))),
        'input':  (current[n].reshape(B, 1, 1, D), current[:n].permute(1, 0, 2).reshape(B, 1, n, D)),
    }


def queries(truth, kind, T):
    for n in range(1, T):
        answer = truth[:, n, :n]
        valid = (kind[:, n] > 0) & answer.any(-1)
        if valid.any():
            yield n, answer, valid


# ----------------------------------------------------------------------------- recurrence

def fractional_forward(x, tau, b, mask_fn=None):
    """Plain f-LIF recurrence with an optional hard mask. x: [T, B, D] -> states [T, B, D, K].

    mask_fn(t, u, x, states) -> ([B, D, t] in {0,1}, tie flag [B, D]). With mask_fn None this
    must reproduce PopulationNeuron at eta = 0 exactly; main() checks that first.
    """
    T, B, D = x.shape
    K = tau.numel()
    u = x.new_zeros(B, D, K)
    incs, states, kept, ties = [], [], [], []
    b0 = b[0]
    for t in range(T):
        f = (x[t].unsqueeze(-1) - u) / tau
        nxt = b0 * f
        if t:
            past = torch.stack(incs, dim=-2)                    # [B, D, t, K]
            b_hist = b[1:t + 1].flip(0)                         # b(n-j), oldest first
            c = b_hist.expand(B, D, t)
            if mask_fn is not None:
                m, tie = mask_fn(t, u, x, states)
                c = c * m
                kept.append(m)
                ties.append(tie)
            nxt = nxt + torch.einsum('bdj,bdjk->bdk', c, past)
        incs.append(f)
        u = nxt
        states.append(u)
    return torch.stack(states), kept, ties


# ----------------------------------------------------------------------------- calibration

@torch.no_grad()
def reference_thresholds(model, loader, args, batches):
    """Cross-sequence reference on TRAIN (see module docstring). Refuses a batch of one."""
    neuron = model.embedding.neuron
    pools = {(ax, st): [] for ax in AXES for st in STATS}
    n_pairs = 0
    for i, (x, y, truth, kind) in enumerate(loader):
        if i >= batches:
            break
        if x.shape[0] < 2:
            raise SystemExit('reference pairing needs at least two sequences per batch')
        x = x.float()
        current = model.embedding.current(layers.to_patches(x, args.patch_size))
        state = neuron(current, mode='full', return_aux=True)[1]['state']
        if not torch.isfinite(state).all():
            raise SystemExit(f'non-finite state on train batch {i}')
        perm = torch.roll(torch.arange(x.shape[0]), 1)          # partner = next sequence in the batch
        for n, answer, valid in queries(truth, kind, current.shape[0]):
            desc = descriptors(state, current, n)
            n_pairs += int(valid.sum()) * n
            for ax in AXES:
                q, k = desc[ax]
                for st in STATS:
                    pools[(ax, st)].append(similarity(q, k[perm], st)[valid].flatten())
    thr = {}
    for key, chunks in pools.items():
        v = torch.cat(chunks).double()
        thr[key] = {qt: float(torch.quantile(v, qt)) for qt in QUANTILES}
        thr[key]['n'] = int(v.numel())
    return thr, n_pairs


# ----------------------------------------------------------------------------- rules

def rule_mask(sim, rule):
    """Hard mask and a per-row flag saying whether a tie touched the decision."""
    kind, val = rule
    if kind == 'thr':
        return (sim >= val).to(sim.dtype), torch.zeros(sim.shape[:-1], dtype=torch.bool)
    n = sim.shape[-1]
    k = max(1, int(round(val * n)))
    srt = torch.sort(sim, dim=-1, descending=True, stable=True)
    m = torch.zeros_like(sim).scatter_(-1, srt.indices[..., :k], 1.)
    tie = (srt.values[..., k - 1] == srt.values[..., k]) if k < n else torch.zeros(sim.shape[:-1], dtype=torch.bool)
    return m, tie


def make_rules(thr):
    rules = {}
    for ax in AXES:
        for st in STATS:
            for qt in QUANTILES:
                rules[(ax, st, f'ref{qt:.2f}')] = ('thr', thr[(ax, st)][qt])
            for q in TOPQ:
                rules[(ax, st, f'top{int(q * 100)}%')] = ('topq', q)
    return rules


# ----------------------------------------------------------------------------- measurement

def per_query(m, tie, answer, b_hist, n, draws, gen):
    """FIELDS for one query plus the random control per draw. m: [B, Dm, n], answer: [B, n] bool.

    Rows without an answer are dropped by `valid` in accumulate(); the safe denominators here
    only keep them from producing NaN on the way (audit A17-AGGREGATION).
    """
    B, Dm, _ = m.shape
    a = answer.unsqueeze(1).expand(B, Dm, n).to(m.dtype)
    w = b_hist.expand(B, Dm, n)
    kept_mass = (m * w).sum(-1)
    empty = kept_mass <= 0
    meff = torch.where(empty, torch.zeros_like(kept_mass), (m * w * a).sum(-1) / kept_mass.clamp_min(1e-12))
    cnt = m.sum(-1)
    a_cnt = answer.sum(-1).to(m.dtype).clamp_min(1.).unsqueeze(1).expand(B, Dm)

    samples = []                                                # random control, one value per draw
    for _ in range(draws):
        rank = torch.rand(B, Dm, n, generator=gen).argsort(-1).argsort(-1).to(m.dtype)
        rm = (rank < cnt.unsqueeze(-1)).to(m.dtype)
        rmass = (rm * w).sum(-1)
        samples.append(torch.where(rmass <= 0, torch.zeros_like(rmass),
                                   (rm * w * a).sum(-1) / rmass.clamp_min(1e-12)))
    any_rand = torch.as_tensor(1. - hypergeom.pmf(0, n, a_cnt.cpu().numpy(), cnt.cpu().numpy()),
                               dtype=m.dtype)
    fields = {
        'meff': meff, 'kept_frac': cnt / n, 'kept': cnt, 'empty': empty.to(m.dtype),
        'answer_kept': (m * a).sum(-1) / a_cnt,
        'any_hit': ((m * a).sum(-1) > 0).to(m.dtype),
        'any_hit_random': any_rand,
        'tie_frac': tie.to(m.dtype).reshape(B, -1).expand(B, Dm),   # [B,1] (shared/input) or [B,D] (unit)
    }
    return fields, torch.stack(samples)                         # [draws, B, Dm]


def accumulate(acc, step, draws_step, valid):
    """Mean over units, then sum over queries (divided later by the query count)."""
    keep = valid.to(torch.float64)
    for k in FIELDS:
        v = step[k].mean(-1).double()
        if not torch.isfinite(v[valid]).all():
            raise SystemExit(f'non-finite {k} on a valid row')
        acc['tot'][k] += torch.where(valid, v, torch.zeros_like(v)) * keep
    d = draws_step.mean(-1).double()                            # [draws, B]
    acc['draws'] += torch.where(valid.unsqueeze(0), d, torch.zeros_like(d)) * keep
    acc['cnt'] += keep


def fresh(B, draws):
    return {'tot': {f: torch.zeros(B, dtype=torch.float64) for f in FIELDS},
            'draws': torch.zeros(draws, B, dtype=torch.float64),
            'cnt': torch.zeros(B, dtype=torch.float64)}


def finish(acc, out):
    ok = acc['cnt'] > 0
    for f in FIELDS:
        out[f].extend((acc['tot'][f][ok] / acc['cnt'][ok]).tolist())
    out['draws'].extend((acc['draws'][:, ok] / acc['cnt'][ok]).T.tolist())   # per sequence: [draws]


def summarise(seq):
    """Sequence means; the random control's mean and its Monte Carlo SE from per-draw aggregates."""
    r = {f: float(np.mean(seq[f])) for f in FIELDS}
    per_draw = np.asarray(seq['draws'], dtype=np.float64).mean(0)          # [draws]: sequence mean per draw
    r['meff_random'] = float(per_draw.mean())
    r['mc_se'] = float(per_draw.std(ddof=1) / math.sqrt(len(per_draw))) if len(per_draw) > 1 else float('nan')
    return r


@torch.no_grad()
def screen(model, loader, args, rules, batches, draws, seed):
    """Every rule in both settings plus the m == 1 baseline. Per-sequence lists, then means."""
    neuron = model.embedding.neuron
    tau, b = neuron.tau, neuron.b
    keys = list(rules)
    blank = lambda: {**{f: [] for f in FIELDS}, 'draws': []}                # noqa: E731
    out = {s: {k: blank() for k in keys} for s in ('frozen', 'closed')}
    out['kernel'] = blank()
    peak = {s: {k: 0. for k in keys} for s in ('frozen', 'closed')}
    nonfinite = {k: 0 for k in keys}
    base_peak = 0.
    n_seq = n_q = 0

    for i, (x, y, truth, kind) in enumerate(loader):
        if i >= batches:
            break
        x = x.float()
        B = x.shape[0]
        n_seq += B
        current = model.embedding.current(layers.to_patches(x, args.patch_size))
        T = current.shape[0]
        base_state = neuron(current, mode='full', return_aux=True)[1]['state']
        if not torch.isfinite(base_state).all():
            raise SystemExit(f'non-finite eta=0 state on validation batch {i}')
        base_peak = max(base_peak, base_state.abs().max().item())
        qs = list(queries(truth, kind, T))
        n_q += sum(int(v.sum()) for _, _, v in qs)

        # --- baseline: the plain kernel, m == 1, same sample, same aggregation
        acc = fresh(B, draws)
        gen = torch.Generator().manual_seed(stream_seed(seed, ('kernel',), i))
        for n, answer, valid in qs:
            m = torch.ones(B, 1, n)
            accumulate(acc, *per_query(m, torch.zeros(B, 1, dtype=torch.bool), answer, b[1:n + 1].flip(0), n, draws, gen), valid)
        finish(acc, out['kernel'])

        # --- frozen: statistic on the eta=0 trajectory, kernel re-weighted only
        for key in keys:
            ax, st, _ = key
            gen = torch.Generator().manual_seed(stream_seed(seed, ('frozen',) + key, i))
            acc = fresh(B, draws)
            for n, answer, valid in qs:
                q, k = descriptors(base_state, current, n)[ax]
                m, tie = rule_mask(similarity(q, k, st), rules[key])
                accumulate(acc, *per_query(m, tie, answer, b[1:n + 1].flip(0), n, draws, gen), valid)
            finish(acc, out['frozen'][key])
            peak['frozen'][key] = base_peak                     # the reference trajectory's own peak

        # --- closed: mask applied inside the recurrence, descriptors from the masked states
        for key in keys:
            ax, st, _ = key
            rule = rules[key]

            def mask_fn(t, u, xx, states, ax=ax, st=st, rule=rule):
                st_stack = torch.stack(states) if states else u.new_zeros(0, *u.shape)
                q, k = descriptors(torch.cat([st_stack, u.unsqueeze(0)]), xx, t)[ax]
                m, tie = rule_mask(similarity(q, k, st), rule)
                return m.expand(u.shape[0], u.shape[1], t), tie

            states, kept, ties = fractional_forward(current, tau, b, mask_fn)
            if not torch.isfinite(states).all():
                nonfinite[key] += 1                             # unevaluable; never averaged around
                continue
            peak['closed'][key] = max(peak['closed'][key], states.abs().max().item())
            gen = torch.Generator().manual_seed(stream_seed(seed, ('closed',) + key, i))
            acc = fresh(B, draws)
            for n, answer, valid in qs:
                accumulate(acc, *per_query(kept[n - 1], ties[n - 1], answer, b[1:n + 1].flip(0), n, draws, gen), valid)
            finish(acc, out['closed'][key])

    return out, peak, nonfinite, n_seq, n_q


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description='hard statistical mask on the fractional kernel')
    parser.add_argument('--run', required=True, help='checkpoint dir with model_state/config.pt (eta=0)')
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--ref_batches', type=int, default=4)
    parser.add_argument('--draws', type=int, default=32, help='Monte Carlo draws for the random control')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=str(HERE.with_suffix('.json')))
    cli = parser.parse_args()

    base = parse_defaults()
    base.cpu = True
    config = Config()
    config.load_args(cli.run, base)
    config.device = torch.device('cpu')
    config.max_eval_batches = 0
    model = LOAD_MODEL[config.model](config, train=False)
    neuron = model.embedding.neuron
    eta = float(neuron.selector.eta)
    if eta != 0.:
        raise SystemExit(f'need an eta=0 checkpoint (plain f-LIF trajectory), got eta={eta}')
    _, train = data_provider(config, 'train')
    _, val = data_provider(config, 'val')
    bound = getattr(config, 'g11_bound', None)

    run = Path(cli.run)
    prov = {
        'checkpoint_dir': str(run.resolve()), 'run_id': config.run_id,
        'config_pt_sha256': sha256(run / 'model_state' / 'config.pt'),
        'model_sha256': sha256(run / 'model_state' / 'best+model.pt') if (run / 'model_state' / 'best+model.pt').exists() else None,
        'script_sha256': sha256(HERE), 'layers_sha256': sha256(HERE.parents[1] / 'forecasting' / 'layers.py'),
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=HERE.parent).stdout.strip(),
        'command': ' '.join(sys.argv), 'torch': torch.__version__, 'numpy': np.__version__,
        'data_seed': getattr(config, 'data_seed', None), 'mc_seed': cli.seed, 'draws': cli.draws,
        'val_batches': cli.batches, 'ref_batches': cli.ref_batches, 'g11_bound': bound,
        'tie_rule': 'descending stable sort: ties broken toward the older slot (analysis only; '
                    'the model uses torch.topk)',
    }
    print(f'checkpoint: {config.run_id}')
    print(f'eta = {eta}, alpha = {config.alpha}, tau = {neuron.tau.tolist()}, G11 bound = {bound}')
    print(f'b_0 = {neuron.b[0].item():.4f}, B(T-1) = {neuron.b[1:config.num_patches].sum().item():.4f}')

    # gate: the local recurrence must reproduce the model at m == 1 before anything else
    x, *_ = next(iter(val))
    current = model.embedding.current(layers.to_patches(x.float(), config.patch_size))
    with torch.no_grad():
        ref = neuron(current, mode='full', return_aux=True)[1]['state']
        mine, _, _ = fractional_forward(current, neuron.tau, neuron.b)
    gap = (ref - mine).abs().max().item()
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(mine).all())
    verdict = 'FAIL (non-finite)' if not finite else 'OK' if gap == 0. else 'FAIL' if gap > 1e-6 else 'OK (fp)'
    print(f'[gate] local recurrence vs model at m == 1: finite = {finite}, max |diff| = {gap:.3e}  {verdict}')
    if verdict.startswith('FAIL'):
        raise SystemExit('recurrence gate failed; refusing to measure')
    prov['gate'] = {'finite': finite, 'max_abs_diff': gap}

    thr, n_pairs = reference_thresholds(model, train, config, cli.ref_batches)
    print(f'\n=== reference thresholds (TRAIN, cross-sequence partner, {n_pairs} pairs; see docstring) ===')
    print(f'{"axis":>7} {"stat":>8} {"n":>8} ' + ' '.join(f'{"q" + str(q):>8}' for q in QUANTILES))
    for ax in AXES:
        for st in STATS:
            t = thr[(ax, st)]
            print(f'{ax:>7} {st:>8} {t["n"]:>8} ' + ' '.join(f'{t[q]:>8.4f}' for q in QUANTILES))
    rules = make_rules(thr)

    # where does the similarity actually point: at the answer, or just at the previous slot?
    rec = {(ax, st): {'argmax_prev': [], 'answer_rank': [], 'rank_chance': [], 'tie_frac': []}
           for ax in AXES for st in STATS}
    with torch.no_grad():
        for i, (x, y, truth, kind) in enumerate(val):
            if i >= cli.batches:
                break
            cur = model.embedding.current(layers.to_patches(x.float(), config.patch_size))
            st_ = neuron(cur, mode='full', return_aux=True)[1]['state']
            for n, answer, valid in queries(truth, kind, cur.shape[0]):
                desc = descriptors(st_, cur, n)
                a = answer.unsqueeze(1)
                m_ans = answer.sum(-1).double()
                for ax in AXES:
                    q, k = desc[ax]
                    for st in STATS:
                        sim = similarity(q, k, st)               # [B, Dm, n]
                        order = ranks_desc(sim).double()         # 0 = most similar, ties -> older slot
                        prev = (order[..., n - 1] == 0).double().mean(-1)
                        best = torch.where(a.expand_as(order), order, order.new_full((), float('inf'))
                                           ).amin(-1).mean(-1) / max(n - 1, 1)
                        chance = (n - m_ans) / ((m_ans + 1) * max(n - 1, 1))
                        srt = torch.sort(sim, dim=-1).values
                        tie = (srt[..., 1:] == srt[..., :-1]).any(-1).double().mean(-1) if n > 1 else torch.zeros(sim.shape[0])
                        rec[(ax, st)]['argmax_prev'].extend(prev[valid].tolist())
                        rec[(ax, st)]['answer_rank'].extend(best[valid].tolist())
                        rec[(ax, st)]['rank_chance'].extend(chance[valid].tolist())
                        rec[(ax, st)]['tie_frac'].extend(tie[valid].tolist())
    print('\n=== what the similarity points at (validation, eta=0 trajectory, per query) ===')
    print(f'{"axis":>7} {"stat":>8} {"argmax = prev slot":>19} {"best answer rank":>17} {"chance":>8} {"tie frac":>9}')
    pointing = {}
    for (ax, st), d in rec.items():
        pointing[f'{ax}/{st}'] = {k: float(np.mean(v)) for k, v in d.items()}
        print(f'{ax:>7} {st:>8} {np.mean(d["argmax_prev"]):>19.4f} {np.mean(d["answer_rank"]):>17.4f} '
              f'{np.mean(d["rank_chance"]):>8.4f} {np.mean(d["tie_frac"]):>9.4f}')
    print('rank: 0 = most similar of the n past slots, normalised by n-1, ties -> older slot; '
          'chance = (n-m)/((m+1)(n-1)) assumes no ties; tie frac = queries with any exact duplicate similarity.')

    out, peak, nonfinite, n_seq, n_q = screen(model, val, config, rules, cli.batches, cli.draws, cli.seed)
    kernel = summarise(out['kernel'])
    print(f'\nvalidation: {n_seq} sequences, {n_q} recall queries, {cli.batches} batches, '
          f'random control = {cli.draws} draws')
    print(f'plain kernel (m == 1), same sample and aggregation: M_eff = {kernel["meff"]:.4f}, '
          f'random with all slots = {kernel["meff_random"]:.4f} (identical by construction)')
    head = (f'{"setting":>7} {"axis":>7} {"stat":>8} {"rule":>9} {"M_eff":>7} {"random":>7} {"mc_se":>7} '
            f'{"kept%":>6} {"kept":>5} {"empty%":>7} {"ans.kept":>8} {"any-hit":>7} {"rand":>7} '
            f'{"tie%":>6} {"max|u|":>7} {"G11":>6}')
    summary = {'kernel': kernel}
    for setting in ('frozen', 'closed'):
        print(f'\n=== {setting} ===')
        print(head)
        summary[setting] = {}
        for key in rules:
            nf = nonfinite[key] if setting == 'closed' else 0
            name = '/'.join(key)
            if nf:
                summary[setting][name] = {'nonfinite_batches': nf, 'g11': 'NONFIN'}
                print(f'{setting:>7} {key[0]:>7} {key[1]:>8} {key[2]:>9} {"unevaluable: non-finite closed-loop batches = " + str(nf):>80}')
                continue
            r = summarise(out[setting][key])
            mx = peak[setting][key]
            flag = '-' if bound is None else ('OK' if mx < bound else 'FAIL')
            summary[setting][name] = {**r, 'max_abs_state': mx, 'nonfinite_batches': 0, 'g11': flag}
            print(f'{setting:>7} {key[0]:>7} {key[1]:>8} {key[2]:>9} {r["meff"]:>7.4f} {r["meff_random"]:>7.4f} '
                  f'{r["mc_se"]:>7.4f} {100 * r["kept_frac"]:>6.1f} {r["kept"]:>5.2f} '
                  f'{100 * r["empty"]:>7.2f} {r["answer_kept"]:>8.4f} {r["any_hit"]:>7.4f} '
                  f'{r["any_hit_random"]:>7.4f} {100 * r["tie_frac"]:>6.2f} {mx:>7.2f} {flag:>6}')

    print('\nM_eff: answer share of the KEPT kernel mass; the m == 1 row above is the baseline from THIS run.')
    print('random: same kept count per query, uniform over existing slots (audit A14-CONTROL).')
    print('mc_se: Monte Carlo standard error of the random control\'s FINAL mean (sd over draws of the '
          'sequence mean / sqrt(draws)). Not a sampling CI, not a test statistic.')
    print('tie%: queries where a tie touched the top-q cut (frozen/closed) -- 0 for threshold rules.')
    print('oracle mask = 1.0 by definition: a privileged ceiling, not a matched control.')
    print('empty queries count as M_eff = 0; the fallback for an empty mask is a design decision.')
    print('frozen max|u| is the unmasked reference trajectory; safety of the masked system is the closed column.')
    print('closed thresholds were still calibrated on the unmasked train trajectory.')
    print('nothing here is a recall MSE: the trained readout expects unmasked states.')

    def slim(seq):
        """Per-sequence values for every field; for the random control, the per-DRAW sequence
        means (what mc_se is computed from) rather than the draws x sequences matrix."""
        d = {f: seq[f] for f in FIELDS}
        d['random_per_draw'] = np.asarray(seq['draws'], dtype=np.float64).mean(0).tolist() if seq['draws'] else []
        return d

    record = {'provenance': prov,
              'thresholds': {f'{ax}/{st}': v for (ax, st), v in thr.items()},
              'reference_pairs': n_pairs, 'pointing': pointing,
              'n_sequences': n_seq, 'n_queries': n_q, 'summary': summary,
              'per_sequence': {'kernel': slim(out['kernel']),
                               **{s: {'/'.join(k): slim(v) for k, v in out[s].items()} for s in ('frozen', 'closed')}}}
    Path(cli.out).write_text(json.dumps(record, indent=1))
    print(f'\nfull-precision record: {cli.out}')


if __name__ == '__main__':
    main()
