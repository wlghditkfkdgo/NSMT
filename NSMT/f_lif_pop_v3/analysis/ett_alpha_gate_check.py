#!/usr/bin/env python3
"""Check ett_alpha.py (prereg 2P) BEFORE its validation passes are run (audit 54 (c)2).

1. The gate on both datasets: every one of the 6 conditions x 8 seeds is either a finished run
   whose manifest matches, or a recorded G11 failure (D-BX). Nothing is evaluated.
2. Injected faults are refused: alpha labels swapped both ways, an alpha=1 run carrying the
   alpha=0.7 scale or bound, a finished cell that also has a G11 record, a G11 cell that also has
   a checkpoint, a missing cell, and G11 records with another bound, another run id, or a state
   below the bound.
3. Per-comparison blocking on synthetic rows (analyse): which contrasts are withheld for an unsafe
   cell, a reproduction failure and a condition that failed in training; the contrast arithmetic
   against an independent computation; strict JSON with J's relative change undefined.
No validation pass and no training; the only model loads are for the manifest (checkpoint hashes).
"""
import sys
import copy
import json
import math
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
import ett_alpha as A                                           # noqa: E402

OK = []


def expect(name, cond, detail=''):
    OK.append(bool(cond))
    print(f"[alpha-gate] {'PASS' if cond else 'FAIL'}  {name}{': ' + detail if detail else ''}")


def main():
    device = torch.device('cuda:0')

    # ---- 1 ---------------------------------------------------------------------------------
    for data in A.DATASETS:
        errors, runs, failures = A.gate(data, 96, device)
        fail_conds = sorted({c for c, _ in failures})
        expect(f'{data} gate', not errors and len(runs) == 40 and len(failures) == 8 and fail_conds == ['pearson_R_a1'],
               f'{len(errors)} errors, {len(runs)} finished runs, {len(failures)} recorded G11 failures {fail_conds}')
        for e in errors[:5]:
            print(f"[alpha-gate]   {e}")

    # ---- 2 ---------------------------------------------------------------------------------
    def manifest(cond, as_cond, edit=None):
        _, run, result = A.find_cell('ETTh1', 96, cond, 7)
        _, args = A.load_model(run, device)
        if edit:
            edit(args)
        return A.check_manifest(args, result, 'ETTh1', 96, as_cond, 7)

    for name, bad in {
        'alpha=1 run labelled alpha=0.7':      manifest('q1_R_a1', 'q1_R'),
        'alpha=0.7 run labelled alpha=1':      manifest('q1_R', 'q1_R_a1'),
        'alpha=1 run with the 0.7 scale 10':   manifest('q1_R_a1', 'q1_R_a1', lambda a: setattr(a, 'input_scale', 10.)),
        'alpha=1 run with the 0.7 bound':      manifest('q1_R_a1', 'q1_R_a1',
                                                        lambda a: setattr(a, 'g11_bound', 1334.6653747558594)),
    }.items():
        expect(f'refuse {name}', bool(bad), f'{len(bad)} mismatch(es), e.g. {bad[:1]}')

    _, done_run, _ = A.find_cell('ETTh1', 96, 'q1_R_a1', 7)
    _, g11_run, _ = A.find_cell('ETTh1', 96, 'pearson_R_a1', 7)
    real_exists = Path.exists

    def exists_also(target):
        def fake(self):
            return True if str(self) == str(target) else real_exists(self)
        return fake

    real_glob = A.glob.glob

    def missing(pattern, *a, **k):
        return [] if 'seed1024_' in pattern and 'q0.5_revin' in pattern and A.S2P in pattern else real_glob(pattern, *a, **k)

    for name, patcher, cell in (
            ('finished cell that also has a G11 record', mock.patch.object(Path, 'exists', exists_also(Path(done_run) / 'G11_violation.json')), 'q1_R_a1'),
            ('G11 cell that also has a checkpoint', mock.patch.object(Path, 'exists', exists_also(Path(g11_run) / 'model_state' / 'best+model.pt')), 'pearson_R_a1'),
            ('missing cell (pearson_R_a1 seed 1024)', mock.patch.object(A.glob, 'glob', missing), None)):
        with patcher:
            try:
                if cell:
                    A.find_cell('ETTh1', 96, cell, 7)
                else:
                    A.gate('ETTh1', 96, device)
                expect(f'refuse {name}', False, 'not refused')
            except SystemExit as stop:
                expect(f'refuse {name}', True, f'stopped: {stop}')

    tmp = Path(tempfile.mkdtemp(prefix='ett_alpha_gate_'))
    for name, edit in (('G11 record with another bound', lambda r: r.update(bound=r['bound'] + 1.)),
                       ('G11 record from another run', lambda r: r.update(run_id=r['run_id'].replace('seed7', 'seed13'))),
                       ('G11 record below its bound', lambda r: r.update(max_abs_state=r['bound'] - 1.))):
        run = tmp / name.replace(' ', '_')
        (run / 'model_state').mkdir(parents=True)
        shutil.copy(Path(g11_run) / 'model_state' / 'config.pt', run / 'model_state' / 'config.pt')
        record = json.load(open(Path(g11_run) / 'G11_violation.json'))
        edit(record)
        (run / 'G11_violation.json').write_text(json.dumps(record))
        bad, _ = A.check_failure(str(run), 'ETTh1', 96, 'pearson_R_a1', 7)
        expect(f'refuse {name}', bool(bad), f'{bad[:1]}')
    bad, _ = A.check_failure(g11_run, 'ETTh1', 96, 'pearson_R_a1', 7)
    expect('accept the real G11 record', not bad, f'{bad}')

    # ---- 3 ---------------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    base = [{'cond': c, 'seed': s, 'status': 'evaluated', 'safe': True, 'repro_ok': True,
             'mse': float(.5 + .05 * rng.standard_normal()), 'mse_per_channel': (.5 + .05 * rng.standard_normal(7)).tolist()}
            for s in A.SEEDS for c in A.CONDITIONS]
    names = list(A.CONTRASTS)

    def withheld(rows):
        return sorted(n for n, t in A.analyse(rows)[1].items() if t['status'] == 'withheld')

    rows = copy.deepcopy(base)
    blocked, contrasts, per_channel = A.analyse(rows)
    expect('no failure -> all 8 contrasts reported', not blocked and not withheld(rows) and per_channel is not None)
    a = np.array([r['mse'] for r in rows if r['cond'] == 'q1_R'])
    b = np.array([r['mse'] for r in rows if r['cond'] == 'q1_R_a1'])
    d = a - b
    half = stats.t.ppf(.975, 7) * d.std(ddof=1) / math.sqrt(8)
    t = contrasts[names[0]]
    expect('F_q arithmetic = independent paired t', abs(t['mean'] - d.mean()) < 1e-15
           and abs(t['ci'][1] - (d.mean() + half)) < 1e-12 and abs(t['relative'] - d.mean() / b.mean()) < 1e-15)
    j = contrasts['J = S_on(0.7) - S_on(1)']
    expect('J has no relative change and the record is strict JSON', j['relative'] is None
           and bool(json.dumps({'contrasts': contrasts, 'per_channel': per_channel}, allow_nan=False)))

    rows = copy.deepcopy(base)
    next(r for r in rows if r['cond'] == 'q1_R' and r['seed'] == 7).update(safe=False)
    want = sorted(['F_q = q1_R(0.7) - q1_R(1)', 'J = S_on(0.7) - S_on(1)', 'q1_R(0.7) - gru_R', 'q1_R(0.7) - linear_R'])
    expect('q1_R seed 7 unsafe -> only the 4 contrasts using q1_R withheld', withheld(rows) == want
           and A.analyse(rows)[2] is None, f'{withheld(rows)}')

    rows = copy.deepcopy(base)
    next(r for r in rows if r['cond'] == 'gru_R' and r['seed'] == 13).update(repro_ok=False)
    want = sorted(['q1_R(1) - gru_R', 'q1_R(0.7) - gru_R'])
    expect('gru_R seed 13 not reproduced -> only the 2 GRU contrasts withheld', withheld(rows) == want, f'{withheld(rows)}')

    rows = [r for r in copy.deepcopy(base) if r['cond'] != 'pearson_R_a1']
    rows += [{'cond': 'pearson_R_a1', 'seed': s, 'status': 'g11_failure_in_training', 'safe': False} for s in A.SEEDS]
    want = sorted(['F_p = pearson_R(0.7) - pearson_R(1)', 'S_on(1) = pearson_R(1) - q1_R(1)', 'J = S_on(0.7) - S_on(1)'])
    expect('pearson_R_a1 failed in training -> F_p, S_on(1), J withheld; F_q reported', withheld(rows) == want
           and A.analyse(rows)[1][names[0]]['status'] != 'withheld', f'{withheld(rows)}')

    print(f"[alpha-gate] {sum(OK)}/{len(OK)} checks pass")


if __name__ == '__main__':
    main()
