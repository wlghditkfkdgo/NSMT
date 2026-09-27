"""Check ett_future.py's opening gate WITHOUT opening the future period (prereg 2O D-BM).

1. All 128 runs (48 reused from 2N, 80 new) pass check_manifest.
2. Faults specific to 2O are refused on copies of real records: an R-on spiking run carrying
   the R-off input scale, bound or calibration; a run whose revin flag does not match its
   condition; a Linear run labelled as GRU.
3. A validation pass (writes nothing) for every condition x dataset at seed 7 reproduces the
   recorded best validation MSE.
Neither the registry nor the future rows are touched.
"""
import sys
import copy
import json
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
import ett_future as F                                          # noqa: E402


def main():
    device = torch.device('cuda:0')

    # ---- 1 ---------------------------------------------------------------------------------
    errors, n = [], 0
    for data in F.DATASETS:
        for cond in F.CONDITIONS:
            for seed in F.SEEDS:
                run, result = F.find_run(data, 96, cond, seed)
                _, args = F.load_model(run, device)
                errors += [f'{data} {cond}/{seed}: {b}' for b in F.check_manifest(args, result, data, 96, cond, seed)]
                n += 1
    print(f'[gate] runs passing the 2O manifest: {n - len({e.split(":")[0] for e in errors})}/{n}')
    for e in errors:
        print(f'[gate]   {e}')

    # ---- 2 ---------------------------------------------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix='ett_future_gate_'))

    def attempt(name, cond, edit_payload=None, edit_args=None, as_cond=None):
        run, result = F.find_run('ETTh1', 96, cond, 7)
        _, args = F.load_model(run, device)
        payload = json.load(open(result))
        if edit_payload:
            edit_payload(payload)
        if edit_args:
            edit_args(args)
        path = tmp / f'{name}.json'
        path.write_text(json.dumps(payload))
        bad = F.check_manifest(args, path, 'ETTh1', 96, as_cond or cond, 7)
        print(f"[gate] inject {name:<28} -> {'REFUSED' if bad else 'PASSED (gate hole)'}: {bad[:1]}")
        return bool(bad)

    refused = [
        attempt('R_on_with_R_off_scale', 'pearson_R', edit_args=lambda a: setattr(a, 'input_scale', 6.)),
        attempt('R_on_with_R_off_bound', 'pearson_R', edit_args=lambda a: setattr(a, 'g11_bound', 507.55577087402344)),
        attempt('R_on_with_R_off_calib', 'q1_R',
                lambda p: p['provenance']['calibration'].update(file='ETTh1_a0.7_norm-frozen_seed7_260925-173548.json')),
        attempt('revin_flag_missing', 'pearson_R', edit_args=lambda a: setattr(a, 'revin', False)),
        attempt('R_off_run_as_R_on', 'pearson', as_cond='pearson_R'),
        attempt('linear_run_as_gru', 'linear', as_cond='gru'),
    ]
    print(f'[gate] injected 2O faults refused: {sum(refused)}/{len(refused)}')

    # ---- 3 ---------------------------------------------------------------------------------
    worst = 0.
    for data in F.DATASETS:
        for cond in F.CONDITIONS:
            run, result = F.find_run(data, 96, cond, 7)
            model, args = F.load_model(run, device)
            r = F.test(args, model, cond, flag='val')
            stored = json.load(open(result))['train']['best_val_loss']
            worst = max(worst, abs(r['mse'] - stored))
            print(f"[gate] {data} {cond:>9} seed 7 val {r['mse']:.6f} vs recorded {stored:.6f}; safe {r['safe']}; "
                  f"ties {r['ties']}; mismatch {r['mask_mismatch']}")
    print(f'[gate] validation reproduction, max |diff| over 16 = {worst:.1e}')

    gate_faults(device)


def gate_faults(device):
    """Audit 51 (c)2: faults at the level of the whole gate. Each must stop gate() before the
    future rows or the registry are touched -- both are replaced by tripwires that raise."""
    from unittest import mock

    touched = []

    def tripwire(name):
        def fire(*a, **k):
            touched.append(name)
            raise AssertionError(f'{name} reached before the gate refused')
        return fire

    target = ('ETTh2', 'linear_R', 1024)                         # the last cell of the last dataset
    orig_find, orig_load, orig_sha, orig_glob = F.find_run, F.load_model, F.sha256, F.glob.glob
    run_t, _ = orig_find(*target[:1], 96, target[1], target[2])
    run_m, _ = orig_find('ETTh2', 96, 'pearson_R', 1024)

    def on_cell(run_dir, edit):
        def load(run, dev):
            model, args = orig_load(run, dev)
            if run == run_dir:
                edit(args)
            return model, args
        return load

    def missing(data, pl, cond, seed):
        return orig_find(data, pl, cond, 99999 if (data, cond, seed) == target else seed)

    def duplicated(pattern, *a, **k):
        found = orig_glob(pattern, *a, **k)
        return found * 2 if 'Linear' in pattern and 'seed1024' in pattern and 'ETTh2' in pattern and 'revin' in pattern else found

    def tampered(path):
        return 'tampered' if str(path) == str(Path(run_t) / 'model_state' / 'best+model.pt') else orig_sha(path)

    cases = {
        'last cell missing':            mock.patch.object(F, 'find_run', missing),
        'last cell duplicated':         mock.patch.object(F.glob, 'glob', duplicated),
        'other seed in a config':       mock.patch.object(F, 'load_model', on_cell(run_t, lambda a: setattr(a, 'seed', 13))),
        'checkpoint content changed':   mock.patch.object(F, 'sha256', tampered),
        'data CSV changed':             mock.patch.dict(F.DATA_SHA256, {'ETTh2': '0' * 64}),
        'R calibration changed':        mock.patch.dict(F.CALIBRATION_R, {'ETTh2': (F.CALIBRATION_R['ETTh2'][0], '0' * 64)}),
        'early-stop evidence missing':  mock.patch.object(F, 'load_model', on_cell(run_t, lambda a: setattr(a, 'suite', 'no-such-suite'))),
        'safety bound changed':         mock.patch.object(F, 'load_model', on_cell(run_m, lambda a: setattr(a, 'g11_bound', a.g11_bound + 1.))),
        'R flag flipped':               mock.patch.object(F, 'load_model', on_cell(run_m, lambda a: setattr(a, 'revin', False))),
    }
    refused = 0
    with mock.patch.object(F, 'open_once', tripwire('registry')), \
         mock.patch.object(F, 'data_provider', tripwire('data_provider')):
        errors, _ = F.gate('ETTh1', 96, device)
        print(f"[gate] untouched gate: {len(errors)} errors, tripwires fired {touched}")
        for name, patcher in cases.items():
            with patcher:
                try:
                    errors, _ = F.gate('ETTh1', 96, device)
                    outcome = f'{len(errors)} error(s): {errors[:1]}' if errors else 'PASSED (gate hole)'
                    ok = bool(errors)
                except SystemExit as stop:
                    outcome, ok = f'stopped: {stop}', True
            refused += ok and not touched
            print(f"[gate] fault {name:<28} -> {'REFUSED' if ok else 'NOT REFUSED'} before any future access "
                  f"({'clean' if not touched else 'TRIPWIRE ' + str(touched)}): {outcome}")
    print(f'[gate] whole-gate faults refused before the future rows or the registry: {refused}/{len(cases)}')


if __name__ == '__main__':
    import sys as _sys
    if '--faults-only' in _sys.argv:
        gate_faults(torch.device('cuda:0'))
    else:
        main()
