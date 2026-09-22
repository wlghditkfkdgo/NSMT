"""Fault-injection audit of calibration decisions, not a training/numerical result.

Args: frozen NSMT root, output JSON. Synthetic probe rows isolate main()'s handling of
failed sparse checks. Temporary output directories never touch experiment artifacts.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
source = root / 'f_lif_pop_v3/forecasting'
sys.path.insert(0, str(source))
import calibrate

with patch.object(sys, 'argv', ['calibrate.py', '--cpu']):
    args = calibrate.parse_arguments()
args.input_norm = 'none'
args.task = 'recall'
row = dict(input_scale=8., firing_rate=.2, dead_frac=0., saturated_frac=0.,
           unit_rate_min=.2, unit_rate_max=.2, branch_abs_mean=[1.]*4,
           branch_abs_max=[2.]*4, max_abs_state=2., max_abs_current=1.,
           declared_bound=10., within_bound=True, finite=True)
cases = {'healthy': {}, 'sparse_over_bound': {'max_abs_state': 1001., 'within_bound': False},
         'sparse_nonfinite': {'finite': False}, 'sparse_out_of_band': {'firing_rate': .01}}
results = []
original_cwd = Path.cwd()
os.chdir(source)
try:
    for name, changes in cases.items():
        sparse = dict(row, **changes)

        def probe(embedding, batches, device, scale, mode='full'):
            return dict(sparse if mode == 'sparse' else row)

        with tempfile.TemporaryDirectory(prefix='nsmt_calibration_fault_') as tmp:
            stdout = io.StringIO()
            with patch.object(calibrate, 'parse_arguments', return_value=args), \
                 patch.object(calibrate, 'TASK', Path(tmp)), \
                 patch.object(calibrate, 'GRID', [8.]), \
                 patch.object(calibrate, 'data_provider', return_value=(None, [(torch.zeros(2, 336, 1),)])), \
                 patch.object(calibrate, 'probe', side_effect=probe), \
                 contextlib.redirect_stdout(stdout):
                try:
                    returned = calibrate.main()
                    # The reviewed CLI ends in raise SystemExit(main()).
                    exit_code = int(returned or 0)
                except SystemExit as exc:
                    returned = None
                    exit_code = exc.code
            artifact = next(Path(tmp).rglob('*.json'))
            payload = json.loads(artifact.read_text())
            results.append({'case': name, 'injected_sparse_fields': changes,
                            'calibration_accepted': payload['picked'] is not None,
                            'sparse_diagnostic_preserved': payload['sparse_at_init'] is not None,
                            'reason': payload['reason'], 'main_return': returned,
                            'process_exit_if_main_called_normally': exit_code})
finally:
    os.chdir(original_cwd)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'source_root': str(root), 'fault_injection_only': True,
                             'results': results}, indent=2))
print(output.read_text())
