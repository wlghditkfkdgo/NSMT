#!/usr/bin/env python3
"""Prereg 2Q D-CE (post-hoc supplement, added after the first M9 rows): M9b input perturbation.

M9 perturbs u(n0) only; in this recursion u(n+1) is re-summed from the stored increments, so that
change reaches later steps only through f(n0) and the selection keys. M9b instead adds a random
direction of norm 1e-6 * ||I(n0)|| (seed 0) to the input current at n0 = 20, which changes f(n0)
and every later history sum consistently. (a) masks of the unperturbed run, (b) re-selected.
Per branch A_end = ||delta u at the last state|| / ||delta u(n0 + 1)||. Not part of the H1 reading.
"""
import sys
import json
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection_stability as S                                 # noqa: E402
from selection_perturb import N0, REL, DT                        # noqa: E402


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
                d = torch.randn(I[N0].shape, generator=gen, dtype=DT)
                Ip = I.clone()
                Ip[N0] = Ip[N0] + d / d.norm() * REL * I[N0].norm()
                entry = {}
                for mode, fixed in (('fixed', True), ('dynamic', False)):
                    pert, _, _ = S.run(Ip, b, tau, stat, q, masks=masks if fixed else None)
                    first, last = pert[N0] - base[N0], pert[-1] - base[-1]          # u(n0+1), last state
                    a_end = [(last[..., k].norm() / first[..., k].norm()).item() for k in range(tau.numel())]
                    entry[mode] = {'A_end': a_end, 'per_step': [a ** (1. / (base.shape[0] - N0 - 1)) for a in a_end]}
                res[rule] = entry
                lines.append(f"{data:5s} {name:32s} {rule:8s} A_end fixed {S.fmt(entry['fixed']['A_end'])}  "
                             f"per-step {S.fmt(entry['fixed']['per_step'])}  A_end dynamic {S.fmt(entry['dynamic']['A_end'])}")
            p = res['pearson']['fixed']['A_end']
            fast = max(p[0], p[1])
            res['verdict_2prime'] = ('amplification in a fast branch' if fast > 1 and fast > p[2] and fast > p[3]
                                     else 'no fast-branch amplification')
            lines.append(f"{data:5s} {name:32s} (2') -> {res['verdict_2prime']}")
            record[f'{data} | {name}'] = res
            print('\n'.join(lines[-4:]), flush=True)
    Path(__file__).with_suffix('.json').write_text(json.dumps(record, indent=1, allow_nan=False))
    Path(__file__).with_suffix('.txt').write_text('\n'.join(lines) + '\n')
    print(f"[perturb-input] done in {time.time() - t0:.0f} s", flush=True)


if __name__ == '__main__':
    main()
