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


if __name__ == '__main__':
    main()
