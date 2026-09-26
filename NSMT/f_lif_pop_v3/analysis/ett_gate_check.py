"""Check ett_test.py's opening gate WITHOUT opening any test split (audit 43 A21-ETT-OPEN-GATE).

1. The real 48 runs: every one must pass check_manifest (no false alarm).
2. Fault injection on copies of one real run's records: each fault audit 43 found the earlier
   gate letting through must now be refused.
3. A validation pass (flag='val', writes nothing) per dataset x condition at seed 7: the recorded
   best validation MSE must be reproduced, and the firing rate is now window x channel weighted.
The registry is never touched and data_provider is never asked for 'test'.
"""
import sys
import copy
import json
import shutil
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
import ett_test as E                                             # noqa: E402

SUITE = 'etthard-20260925'


def main():
    device = torch.device('cuda:0')
    bounds = {d: json.load(open(E.TASK / 'results' / 'calibration' / E.CALIBRATION[d][0]))['picked']['declared_bound']
              for d in E.DATASETS}

    # ---- 1. 실제 48 run ----------------------------------------------------------------------
    clean, errors = 0, []
    for data in E.DATASETS:
        for cond in E.CONDS:
            for seed in E.SEEDS:
                run, result = E.find_run(SUITE, data, 96, cond, seed)
                _, args = E.load_model(run, device)
                bad = E.check_manifest(args, result, data, 96, cond, seed, bounds[data])
                clean += not bad
                errors += [f'{data} {cond}/{seed}: {b}' for b in bad]
    print(f'[gate] real runs passing the manifest: {clean}/48')
    for e in errors:
        print(f'[gate]   {e}')

    # ---- 2. 결함 주입 (실제 run 한 개의 기록 사본) ---------------------------------------------
    run, result = E.find_run(SUITE, 'ETTh1', 96, 'pearson', 7)
    _, base_args = E.load_model(run, device)
    payload = json.load(open(result))
    tmp = Path(tempfile.mkdtemp(prefix='ett_gate_'))

    def attempt(name, edit_payload=None, edit_args=None):
        p, a = copy.deepcopy(payload), copy.copy(base_args)
        if edit_payload:
            edit_payload(p)
        if edit_args:
            edit_args(a)
        path = tmp / f'{name}.json'
        path.write_text(json.dumps(p))
        bad = E.check_manifest(a, path, 'ETTh1', 96, 'pearson', 7, bounds['ETTh1'])
        print(f"[gate] inject {name:<22} -> {'REFUSED' if bad else 'PASSED (gate hole)'}: {bad[:1]}")
        return bool(bad)

    def short_csv(a):                                            # CSV가 epochs_run보다 한 줄 짧다
        d = tmp / 'short_log'
        d.mkdir(exist_ok=True)
        lines = (Path(base_args.save_log_path) / 'best_log_0.csv').read_text().splitlines()
        (d / 'best_log_0.csv').write_text('\n'.join(lines[:-1]) + '\n')
        a.save_log_path = str(d)

    refused = [
        attempt('test_not_skipped', lambda p: p.update(test_skipped=False)),
        attempt('checkpoint_hash', lambda p: p['provenance'].update(checkpoint_sha256='wrong')),
        attempt('trained_on_cpu', lambda p: p['provenance'].update(device='cpu')),
        attempt('other_torch', lambda p: p['provenance'].update(torch='2.1.0')),
        attempt('other_calibration', lambda p: p['provenance']['calibration'].update(file='ETTh1_other.json')),
        attempt('epochs_vs_csv', lambda p: p['train'].update(epochs_run=p['train']['epochs_run'] + 1)),
        attempt('csv_truncated', edit_args=short_csv),
        attempt('best_val_mismatch', lambda p: p['train'].update(best_val_loss=p['train']['best_val_loss'] + 1e-3)),
        attempt('root_path', edit_args=lambda a: setattr(a, 'root_path', '/tmp/ETT-small')),
    ]
    print(f'[gate] injected faults refused: {sum(refused)}/{len(refused)}')
    shutil.rmtree(tmp)

    # ---- 3. validation 통과 (아무것도 쓰지 않는다) -----------------------------------------------
    for data in E.DATASETS:
        for cond in E.CONDS:
            run, result = E.find_run(SUITE, data, 96, cond, 7)
            model, args = E.load_model(run, device)
            r = E.test(args, model, cond, flag='val')
            stored = json.load(open(result))['train']['best_val_loss']
            print(f"[gate] {data} {cond:>8} seed 7 val: {r['mse']:.6f} vs recorded {stored:.6f} "
                  f"|diff| {abs(r['mse'] - stored):.1e}; safe {r['safe']}; firing {r['firing_rate']}; "
                  f"kept {r['kept_mass_frac']}; ties {r['ties']}; mismatch {r['mask_mismatch']}")


if __name__ == '__main__':
    main()
