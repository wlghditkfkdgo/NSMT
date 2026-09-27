#!/usr/bin/env python3
"""Check weather_test.py's opening gate WITHOUT opening the weather test period (prereg 2R D-CJ).

1. gate(): all 40 cells, manifest errors 0.
2. The CLI entry point with one fault injected at a time. The registry (open_once) and the data
   loader (data_provider) are tripwires that raise if reached; each fault must stop in the gate,
   before either, and no output directory may appear:
     last cell missing (linear_R 1024), alpha flag flipped on q1_R_a1 seed 7, alpha=1 calibration
     file changed, a checkpoint changed (pearson_R 13), weather.csv changed, early-stop evidence
     missing (gru_R 21).
3. A validation pass (writes nothing) for every condition at seed 7 reproduces the recorded best
   validation MSE within 1e-6.
Neither the registry nor any test row is touched.
"""
import sys
import json
import runpy
from pathlib import Path
from unittest import mock

import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'forecasting'))
import weather_test as W                                        # noqa: E402
import ett_test                                                 # noqa: E402
import split_registry                                           # noqa: E402
import data_provider.data_factory as factory                    # noqa: E402

OK = []


def expect(name, cond, detail=''):
    OK.append(bool(cond))
    print(f"[weather-gate] {'PASS' if cond else 'FAIL'}  {name}{': ' + detail if detail else ''}")


def main():
    device = torch.device('cuda:0')

    # ---- 1 ---------------------------------------------------------------------------------
    errors, runs, failures = W.gate(device)
    expect('gate over 40 cells', not errors and len(runs) + len(failures) == 40,
           f'{len(errors)} errors, {len(runs)} finished, {len(failures)} recorded G11 failures')
    for e in errors[:5]:
        print(f"[weather-gate]   {e}")

    # ---- 2 ---------------------------------------------------------------------------------
    _, run_a1, _ = W.find_cell('q1_R_a1', 7)
    _, run_p13, _ = W.find_cell('pearson_R', 13)
    _, run_g21, _ = W.find_cell('gru_R', 21)
    cal_a1 = W.TASK / 'results' / 'calibration' / W.CALIBRATION[1.][0]
    real_glob, real_load, real_sha = W.glob.glob, ett_test.load_model, ett_test.sha256

    def missing(pattern, *a, **k):
        return [] if 'Linear' in pattern and 'seed1024' in pattern else real_glob(pattern, *a, **k)

    def edit_on(run_dir, edit):
        def load(run, dev):
            model, args = real_load(run, dev)
            if str(run) == str(run_dir):
                edit(args)
            return model, args
        return load

    def tampered(target):
        def sha(path):
            return '0' * 64 if str(path) == str(target) else real_sha(path)
        return sha

    fired = []

    def tripwire(name):
        def fire(*a, **k):
            fired.append(name)
            raise AssertionError(f'{name} reached before the gate refused')
        return fire

    cases = {
        'last cell missing (linear_R 1024)': mock.patch.object(W.glob, 'glob', missing),
        'alpha flag flipped (q1_R_a1 7)':    mock.patch.object(ett_test, 'load_model', edit_on(run_a1, lambda a: setattr(a, 'alpha', .7))),
        'alpha=1 calibration changed':       mock.patch.object(ett_test, 'sha256', tampered(cal_a1)),
        'checkpoint changed (pearson_R 13)': mock.patch.object(ett_test, 'sha256', tampered(Path(run_p13) / 'model_state' / 'best+model.pt')),
        'weather.csv changed':               mock.patch.object(ett_test, 'sha256', tampered(W.DATA_ROOT / 'weather.csv')),
        'early-stop evidence missing (gru_R 21)': mock.patch.object(ett_test, 'load_model', edit_on(run_g21, lambda a: setattr(a, 'suite', 'no-such-suite'))),
    }
    out = W.TASK / 'results' / 'weather-gate-fixture-should-not-exist'
    for name, patcher in cases.items():
        fired.clear()
        with patcher, mock.patch.object(split_registry, 'open_once', tripwire('registry')), \
                mock.patch.object(factory, 'data_provider', tripwire('data_provider')), \
                mock.patch.object(sys, 'argv', ['weather_test.py', '--out', out.name]):
            try:
                runpy.run_path(str(HERE.parents[1] / 'forecasting' / 'weather_test.py'), run_name='__main__')
                outcome, stopped = 'not stopped', False
            except SystemExit as stop:
                outcome, stopped = f'stopped: {str(stop)[:110]}', True
            except AssertionError as trip:
                outcome, stopped = f'TRIPWIRE: {trip}', False
        expect(f'refuse {name} before the registry and the test rows', stopped and not fired and not out.exists(),
               f'{outcome}; tripwires {fired}; output dir {out.exists()}')

    # ---- 3 ---------------------------------------------------------------------------------
    worst = 0.
    for cond in W.CONDITIONS:
        status, run, result = W.find_cell(cond, 7)
        if status != 'done':
            print(f"[weather-gate] {cond} seed 7 is a recorded G11 failure; no validation pass")
            continue
        model, args = ett_test.load_model(run, device)
        r = W.test(args, model, cond, flag='val')
        stored = json.load(open(result))['train']['best_val_loss']
        worst = max(worst, abs(r['mse'] - stored))
        print(f"[weather-gate] {cond:>9} seed 7 val {r['mse']:.6f} vs recorded {stored:.6f}; safe {r['safe']}")
    expect('validation pass reproduces the recorded best MSE (seed 7, every condition)', worst <= 1e-6, f'max |diff| {worst:.1e}')

    print(f"[weather-gate] {sum(OK)}/{len(OK)} checks pass")


if __name__ == '__main__':
    main()
