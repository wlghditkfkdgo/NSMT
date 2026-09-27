#!/usr/bin/env python3
"""Prereg 2Q D-CD: M9 perturbation test and the M4 identifiability report (exploratory).

Same inputs, models and recursion as selection_stability.py (imported). For each dataset x model
x rule (q1, pearson q=0.5, recent q=0.5):

  M4 identifiability: per step t >= 10 and branch, the singular values of X = [u(t), f(t)] over
  all cells; rank (s_min > 1e-12 s_max) and condition s_max / s_min. A step with rank < 2 or
  condition > 1e6 is "not identifiable" and its beta / lambda are not used (D-CD).

  M9: run once; at step n0 = 20 add to every cell's u a random direction (seed 0) of norm
  1e-6 * ||u(n0)||, and continue to the end
    (a) with the unperturbed run's masks (fixed: the linear response of the recursion),
    (b) re-selecting from the perturbed states (dynamic).
  Per branch, A_end = ||delta u_k at the last state|| / ||delta u_k at n0||, and the per-step
  geometric mean A_end ** (1 / 22).
  (2') "amplification in a fast branch": pearson (a) has A_end > 1 in tau 4 or tau 8, and that
  A_end exceeds A_end of both tau 16 and tau 32.
"""
import sys
import json
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection_stability as S                                 # noqa: E402

N0, REL = 20, 1e-6
DT = S.DT


def run_perturbed(I, b, tau, rule, q, base_masks, delta, fixed):
    """The recursion of selection_stability.run with delta added to u at step N0."""
    T, Sn, D = I.shape
    b0 = b[0].item()
    u = torch.zeros(Sn, D, tau.numel(), dtype=DT)
    incs, keys, states = [], [], []
    for t in range(T):
        if t == N0:
            u = u + delta
        f = (I[t].unsqueeze(-1) - u) / tau
        xi = torch.cat([u, I[t].unsqueeze(-1)], dim=-1)
        nxt = b0 * f
        if t:
            b_hist = b[1:t + 1].flip(0)
            m = base_masks[t] if fixed else S.mask_of(rule, q, xi, torch.stack(keys, dim=-2))
            c = (b_hist * m).expand(Sn, D, t)
            nxt = nxt + torch.einsum('sdj,sdjk->sdk', c, torch.stack(incs, dim=-2))
        incs.append(f); keys.append(xi)
        u = nxt
        states.append(u)
    return torch.stack(states)


def identifiability(I, states, tau):
    """Per branch: steps t >= 10 with rank < 2 or condition > 1e6, and the largest condition."""
    T = I.shape[0]
    out = []
    for k in range(tau.numel()):
        bad, worst = 0, 0.
        for t in range(S.T0, T):
            u_t = states[t - 1][..., k].reshape(-1)              # u(t)
            f_t = (I[t].reshape(-1) - u_t) / tau[k]
            sv = torch.linalg.svdvals(torch.stack([u_t, f_t], dim=1))
            cond = (sv[0] / sv[-1]).item() if sv[-1] > 1e-12 * sv[0] else float('inf')
            bad += int(cond > 1e6)
            worst = max(worst, cond)
        out.append({'not_identifiable_steps': bad, 'max_condition': worst if worst != float('inf') else 'rank<2'})
    return out


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    record, lines = {}, []
    t0 = time.time()
    for data in S.A.DATASETS:
        patches = S.batches_of(data).to(DT)
        models = {f'init a{a:g}': S.init_embedding(data, a, patches.float()).double() for a in (.7, 1.)}
        _, run_dir, _ = S.A.find_cell(data, 96, 'pearson_R', 7)
        trained, _ = S.A.load_model(run_dir, torch.device('cpu'))
        models['trained 2O pearson_R a0.7 seed7'] = trained.embedding.double()
        for name, emb in models.items():
            I = emb.current(patches)
            b, tau = emb.neuron.b.to(DT), emb.neuron.tau.to(DT)
            res = {}
            for rule, q in (('q1', 1.), ('pearson', .5), ('recent', .5)):
                stat = 'pearson' if rule == 'q1' else rule
                base, masks, _ = S.run(I, b, tau, stat, q)
                gen = torch.Generator().manual_seed(0)
                u_n0 = base[N0 - 1]                              # u(n0)
                direction = torch.randn(u_n0.shape, generator=gen, dtype=DT)
                delta = direction / direction.norm() * REL * u_n0.norm()
                entry = {'identifiability': identifiability(I, base, tau), 'peak': base.abs().max().item()}
                for mode, fixed in (('fixed', True), ('dynamic', False)):
                    pert = run_perturbed(I, b, tau, stat, q, masks, delta, fixed)
                    d_end = (pert[-1] - base[-1])
                    a_end = [(d_end[..., k].norm() / delta[..., k].norm()).item() for k in range(tau.numel())]
                    entry[mode] = {'A_end': a_end, 'per_step': [a ** (1. / (base.shape[0] - N0)) for a in a_end]}
                res[rule] = entry
                ident = entry['identifiability']
                lines.append(f"{data:5s} {name:32s} {rule:8s} peak {entry['peak']:9.2f}  "
                             f"A_end fixed {S.fmt(entry['fixed']['A_end'])}  per-step {S.fmt(entry['fixed']['per_step'])}  "
                             f"A_end dynamic {S.fmt(entry['dynamic']['A_end'])}  "
                             f"M4 not-identifiable steps {[x['not_identifiable_steps'] for x in ident]}")
            p = res['pearson']['fixed']['A_end']
            fast = max(p[0], p[1])
            res['verdict_2prime'] = ('amplification in a fast branch' if fast > 1 and fast > p[2] and fast > p[3]
                                     else 'no fast-branch amplification')
            lines.append(f"{data:5s} {name:32s} (2') -> {res['verdict_2prime']}")
            record[f'{data} | {name}'] = res
            print('\n'.join(lines[-4:]), flush=True)
    Path(__file__).with_suffix('.json').write_text(json.dumps(record, indent=1, allow_nan=False))
    Path(__file__).with_suffix('.txt').write_text('\n'.join(lines) + '\n')
    print(f"[perturb] done in {time.time() - t0:.0f} s", flush=True)


if __name__ == '__main__':
    main()
