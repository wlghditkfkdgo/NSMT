#!/usr/bin/env python3
"""Prereg 2R D-CI pre-checks for the weather loader, before any 2R calibration or training.

  1. split boundaries on a SYNTHETIC CSV of the same shape (52,696 x 21 + date, 'OT' present):
     row value = row index, so every window's first row, last input row and last target row can
     be read back. Expect train rows [0, 36887), val targets [36887, 42157), test targets
     [42157, 52696), 336 context rows before val/test, windows 36,456 / 5,175 / 10,444 at H96.
  2. the StandardScaler is fit on the train rows only: changing a test row does not change it.
  3. column order: every variable except 'date' and 'OT', then 'OT' last (model_v1).
  4. the real weather.csv: SHA256, shape and finiteness only -- nothing is read into a window,
     and no val/test statistic is computed.
No model, no real val/test rows.
"""
import sys
import json
import hashlib
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
from config import NSMT                                         # noqa: E402
from data_provider.data_loader import Dataset_Custom            # noqa: E402

WEATHER = NSMT / 'forecasting' / 'dataset' / 'weather' / 'weather.csv'
WEATHER_SHA256 = '34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63'


def synthetic(path, bump_test_row=False):
    n = 52696
    cols = [f'v{i}' for i in range(20)]
    data = {c: np.arange(n, dtype=np.float64) + i for i, c in enumerate(cols)}
    data['OT'] = np.arange(n, dtype=np.float64) * 2.
    frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=n, freq='10min').astype(str), **data})
    frame = frame[['date', 'v0', 'OT'] + cols[1:]]                 # OT deliberately not last in the file
    if bump_test_row:
        frame.loc[50000, 'v0'] = 1e9
    frame.to_csv(path, index=False)


def main():
    report = {}
    tmp = Path(tempfile.mkdtemp(prefix='weather_checks_'))
    synthetic(tmp / 'weather.csv')
    args = SimpleNamespace(seq_len=336, pred_len=96, root_path=str(tmp), data_path='weather.csv')

    # 1 --------------------------------------------------------------------------------------
    # (first input row, first target row, last target row, windows); audit 60 A32: the first training
    # target is row 336, not 0 -- a window's target follows its 336 input rows
    want = {'train': (0, 336, 36886, 36456), 'val': (36551, 36887, 42156, 5175), 'test': (41821, 42157, 52695, 10444)}
    ok1 = True
    for flag, (x_first, y_first, y_last, n) in want.items():
        ds = Dataset_Custom(args, flag)
        mean, std = ds.scaler.mean_[0], ds.scaler.scale_[0]
        raw = lambda v: int(round(float(v) * std + mean))            # back to the row index (column v0)
        x0, y0, _, _ = ds[0]                                         # the loader returns (x, y, mark, mark)
        _, yl, _, _ = ds[len(ds) - 1]
        got = (raw(x0[0, 0]), raw(y0[0, 0]), raw(yl[-1, 0]), len(ds))
        good = got == (x_first, y_first, y_last, n)
        ok1 = ok1 and good
        report[f'1_{flag}'] = {'first_input_row': got[0], 'first_target_row': got[1], 'last_target_row': got[2],
                               'windows': got[3], 'ok': good}
        print(f"[weather] 1 {flag:5s}: first input {got[0]}, targets [{got[1]}, {got[2]}], windows {got[3]} "
              f"(want {x_first}, [{y_first}, {y_last}], {n}) -> {good}")

    # 2 --------------------------------------------------------------------------------------
    synthetic(tmp / 'weather.csv', bump_test_row=True)
    bumped = Dataset_Custom(args, 'train').scaler
    synthetic(tmp / 'weather.csv')
    clean = Dataset_Custom(args, 'train').scaler
    ok2 = bool(np.array_equal(bumped.mean_, clean.mean_) and np.array_equal(bumped.scale_, clean.scale_))
    report['2_scaler_train_only'] = ok2
    print(f"[weather] 2 a change in a test row leaves the scaler unchanged: {ok2}")

    # 3 --------------------------------------------------------------------------------------
    cols = Dataset_Custom(args, 'train').columns
    ok3 = cols[-1] == 'OT' and 'date' not in cols and len(cols) == 21
    report['3_columns'] = {'last': cols[-1], 'count': len(cols), 'ok': ok3}
    print(f"[weather] 3 {len(cols)} variables, last '{cols[-1]}' -> {ok3}")

    # 4 --------------------------------------------------------------------------------------
    digest = hashlib.sha256(WEATHER.read_bytes()).hexdigest()
    frame = pd.read_csv(WEATHER)
    values = frame.drop(columns=['date']).to_numpy(dtype=np.float64)
    ok4 = digest == WEATHER_SHA256 and values.shape == (52696, 21) and bool(np.isfinite(values).all()) and 'OT' in frame.columns
    report['4_real_file'] = {'sha256': digest, 'shape': list(values.shape), 'ok': ok4}
    print(f"[weather] 4 weather.csv sha256 {digest[:12]}..., shape {values.shape}, finite, OT present -> {ok4}")

    report['all_pass'] = bool(ok1 and ok2 and ok3 and ok4)
    print(f"[weather] all pre-checks pass: {report['all_pass']}")
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
