#!/usr/bin/env python3
"""Check ett_readout.py (prereg 2Q B) BEFORE its validation passes (audit 56 A29).

1. The gate over every dataset: 2 x 4 conditions x 8 seeds, all finished, manifest errors 0.
2. The CLI entry point itself, with the LAST cell of the OTHER dataset (ETTh2 pearson_R_lin 1024)
   removed, stops in the gate: data_provider and test are tripwires that raise if reached, and
   no output directory is created (A29-GATE-COVERAGE).
3. check_failure on the real 2P G11 failure (ETTh1 pearson_R_a1 seed 7, alpha=1 calibration):
   accepted as it is; refused when the record's run_uuid, the config's input_scale,
   calibration_file or no-test flag, or the record's epoch or peak are changed
   (A29-FAILURE-PROVENANCE). Edits go to a temporary copy; the real run is untouched.
4. analyse / head_decision on synthetic rows: only the contrasts that use a blocked cell are
   withheld; a withheld or clearly-worse H selects 'flatten', otherwise 'linear'.
"""
import sys
import copy
import json
import runpy
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'forecasting'))
import ett_readout as R                                         # noqa: E402
import ett_alpha as A                                           # noqa: E402

OK = []


def expect(name, cond, detail=''):
    OK.append(bool(cond))
    print(f"[readout-gate] {'PASS' if cond else 'FAIL'}  {name}{': ' + detail if detail else ''}")


def main():
    device = torch.device('cuda:0')

    # ---- 1 ---------------------------------------------------------------------------------
    errors, runs, failures = R.gate(96, device)
    expect('gate over both datasets', not errors and len(runs) == 64 and not failures,
           f'{len(errors)} errors, {len(runs)} finished runs, {len(failures)} G11 failures')
    for e in errors[:5]:
        print(f"[readout-gate]   {e}")

    # ---- 2 ---------------------------------------------------------------------------------
    real_glob = R.glob.glob
    fired = []

    def missing(pattern, *a, **k):
        drop = 'ETTh2' in pattern and 'seed1024' in pattern and R.S2Q in pattern and 'q0.5' in pattern
        return [] if drop else real_glob(pattern, *a, **k)

    def tripwire(name):
        def fire(*a, **k):
            fired.append(name)
            raise AssertionError(f'{name} reached before the gate refused')
        return fire

    out = R.TASK / 'results' / 'readout-gate-fixture-should-not-exist'
    import data_provider.data_factory as factory
    with mock.patch.object(R.glob, 'glob', missing), \
         mock.patch.object(factory, 'data_provider', tripwire('data_provider')), \
         mock.patch.object(sys, 'argv', ['ett_readout.py', '--out', out.name]):
        try:
            runpy.run_path(str(HERE.parents[1] / 'forecasting' / 'ett_readout.py'), run_name='__main__')
            stopped = 'not stopped'
        except SystemExit as stop:
            stopped = f'stopped: {stop}'
    expect('CLI stops in the gate when the other dataset\'s last cell is missing',
           stopped.startswith('stopped') and 'ETTh2 pearson_R_lin seed 1024' in stopped and not fired and not out.exists(),
           f"{stopped}; tripwires fired {fired}; output dir created {out.exists()}")

    # ---- 3 ---------------------------------------------------------------------------------
    spec = {**A.CONDITIONS['pearson_R_a1'], 'head_mode': 'flatten'}
    name = A.calibration_of('ETTh1', 1.)[0]
    calibration = (name, A.calibrated('ETTh1', 1.))
    _, real_run, _ = A.find_cell('ETTh1', 96, 'pearson_R_a1', 7)
    bad, _ = R.check_failure(real_run, 'ETTh1', 96, 'pearson_R_a1', 7, spec, calibration)
    expect('accept the real G11 failure', not bad, f'{bad}')

    tmp = Path(tempfile.mkdtemp(prefix='readout_gate_'))

    def variant(label, edit_record=None, edit_config=None):
        run = tmp / label.replace(' ', '_')
        (run / 'model_state').mkdir(parents=True)
        saved = torch.load(Path(real_run) / 'model_state' / 'config.pt', map_location='cpu')
        record = json.load(open(Path(real_run) / 'G11_violation.json'))
        if edit_config:
            edit_config(saved)
        if edit_record:
            edit_record(record)
        torch.save(saved, run / 'model_state' / 'config.pt')
        (run / 'G11_violation.json').write_text(json.dumps(record))
        bad, _ = R.check_failure(str(run), 'ETTh1', 96, 'pearson_R_a1', 7, spec, calibration)
        expect(f'refuse {label}', bool(bad), f'{bad[:1]}')

    variant('record run_uuid from another run', edit_record=lambda r: r.update(run_uuid='0' * 16))
    variant('config input_scale 999', edit_config=lambda c: c.update(input_scale=999.))
    variant('config calibration_file of alpha 0.7', edit_config=lambda c: c.update(calibration_file=R.CALIBRATION_R['ETTh1'][0]))
    variant('config without --no-test', edit_config=lambda c: c.update(test=True))
    variant('record epoch 1 (stdout says 0)', edit_record=lambda r: r.update(epoch=1))
    variant('record peak +1 (stdout disagrees)', edit_record=lambda r: r.update(max_abs_state=r['max_abs_state'] + 1.))

    # ---- 4 ---------------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    base = [{'cond': c, 'seed': s, 'status': 'evaluated', 'safe': True, 'repro_ok': True,
             'mse': float(.5 + .05 * rng.standard_normal())} for s in R.SEEDS for c in R.CONDITIONS]

    def withheld(rows):
        return sorted(n for n, t in R.analyse(rows)[1].items() if t['status'] == 'withheld')

    rows = copy.deepcopy(base)
    expect('no failure -> 4 contrasts reported', not withheld(rows))
    rows = copy.deepcopy(base)
    next(r for r in rows if r['cond'] == 'q1_R' and r['seed'] == 7).update(safe=False)
    expect('q1_R seed 7 unsafe -> H_q and S(flatten) only', withheld(rows) == sorted(
        ['H_q = q1_R(linear) - q1_R(flatten)', 'S(flatten) = pearson_R - q1_R']), f'{withheld(rows)}')
    rows = copy.deepcopy(base)
    next(r for r in rows if r['cond'] == 'pearson_R_lin' and r['seed'] == 13).update(repro_ok=False)
    expect('pearson_R_lin seed 13 not reproduced -> H_p and S(linear) only', withheld(rows) == sorted(
        ['H_p = pearson_R(linear) - pearson_R(flatten)', 'S(linear) = pearson_R - q1_R']), f'{withheld(rows)}')

    fine = {d: R.analyse(copy.deepcopy(base))[1] for d in R.DATASETS}
    expect('head rule: nothing clearly worse -> linear', R.head_decision(fine)[0] == 'linear', f'{R.head_decision(fine)}')
    worse = copy.deepcopy(base)
    for r in worse:
        if r['cond'] == 'pearson_R_lin':
            r['mse'] = next(x['mse'] for x in worse if x['cond'] == 'pearson_R' and x['seed'] == r['seed']) + .01 + .001 * (r['seed'] % 3)
    bad_rule = {'ETTh1': R.analyse(worse)[1], 'ETTh2': fine['ETTh2']}
    expect('head rule: H_p clearly worse on one dataset -> flatten', R.head_decision(bad_rule)[0] == 'flatten',
           f'{R.head_decision(bad_rule)}')
    held = copy.deepcopy(base)
    next(r for r in held if r['cond'] == 'q1_R_lin' and r['seed'] == 7).update(safe=False)
    held_rule = {'ETTh1': fine['ETTh1'], 'ETTh2': R.analyse(held)[1]}
    expect('head rule: a withheld H -> flatten', R.head_decision(held_rule)[0] == 'flatten', f'{R.head_decision(held_rule)}')
    expect('strict JSON of the contrasts', bool(json.dumps(fine, allow_nan=False)))

    print(f"[readout-gate] {sum(OK)}/{len(OK)} checks pass")


if __name__ == '__main__':
    main()
