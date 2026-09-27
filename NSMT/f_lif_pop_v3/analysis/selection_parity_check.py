#!/usr/bin/env python3
"""Audit 59 A31-DIAG-FINITE: a fail-closed re-run of selection_stability.py's parity gate.

selection_stability.check_against_model kept max(worst, rel); max(0., nan) is 0., so a NaN state
would have passed. Here, for every dataset x model x q in {1, 0.5} used by the diagnostic:
  - the model's own neuron state and the re-implemented recursion's state must both be finite,
  - the reference norm must be > 0,
  - the relative difference must be <= 1e-4;
anything else is a failure, not a pass. A fixture shows the gate rejects an injected NaN and a
zero reference. Same inputs as the diagnostic (train split, first 8 batches at seed 7, R on).
"""
import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection_stability as S                                 # noqa: E402

DT = S.DT


def parity(model_state, here_state):
    """(ok, reason, relative difference) -- fail closed."""
    if not (torch.isfinite(model_state).all() and torch.isfinite(here_state).all()):
        return False, 'non-finite state', None
    ref = model_state.abs().max()
    if ref <= 0:
        return False, 'zero reference', None
    rel = ((model_state - here_state).abs().max() / ref).item()
    return rel <= 1e-4, ('ok' if rel <= 1e-4 else 'difference above 1e-4'), rel


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    report, ok_all = {}, True

    x = torch.randn(3, 4, 2, dtype=DT)                          # fixture: the gate must fail closed
    nan = x.clone(); nan[0, 0, 0] = float('nan')
    fixture = {'nan': parity(x, nan)[0] is False and parity(nan, x)[0] is False,
               'zero_reference': parity(torch.zeros_like(x), torch.zeros_like(x))[0] is False,
               'equal': parity(x, x.clone())[0] is True}
    report['fixture'] = fixture
    print(f"[parity] fixture: NaN rejected {fixture['nan']}, zero reference rejected {fixture['zero_reference']}, "
          f"equal accepted {fixture['equal']}")
    ok_all = ok_all and all(fixture.values())

    for data in S.A.DATASETS:
        patches = S.batches_of(data).to(DT)
        models = {f'init a{a:g}': S.init_embedding(data, a, patches.float()).double() for a in (.7, 1.)}
        _, run_dir, _ = S.A.find_cell(data, 96, 'pearson_R', 7)
        trained, _ = S.A.load_model(run_dir, torch.device('cpu'))
        models['trained 2O pearson_R a0.7 seed7'] = trained.embedding.double()
        for name, emb in models.items():
            I = emb.current(patches)
            neuron = emb.neuron
            for q in (1., .5):
                neuron.selector.hard_q = q
                st_model = neuron(I, mode='hard', return_aux=True)[1]['state']
                st_here, _, _ = S.run(I, neuron.b.to(DT), neuron.tau.to(DT), 'pearson', q)
                ok, reason, rel = parity(st_model, st_here)
                report[f'{data} | {name} | q{q:g}'] = {'ok': ok, 'reason': reason, 'relative_difference': rel,
                                                       'peak': st_model.abs().max().item()}
                ok_all = ok_all and ok
                print(f"[parity] {data} {name:32s} q {q:g}: {reason}, relative difference "
                      f"{'n/a' if rel is None else f'{rel:.2e}'}, peak {st_model.abs().max().item():.2f}", flush=True)
    report['all_pass'] = ok_all
    print(f"[parity] all parity checks pass (fail-closed): {ok_all}")
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=1, allow_nan=False))


if __name__ == '__main__':
    main()
