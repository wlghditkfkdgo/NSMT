#!/usr/bin/env python3
"""Prereg 2R D-CJ pre-check 5: adding the weather loader leaves the ETT path unchanged -- 16 of the
2O R-on spiking checkpoints (q1_R and pearson_R, ETTh1/ETTh2, seeds 7, 13, 21, 42) reproduce their
stored best validation MSE within 1e-6 on the GPU they were trained on."""
import sys
import json
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
from data_provider.data_factory import data_provider            # noqa: E402
from test import evaluate                                       # noqa: E402
import ett_alpha as A                                           # noqa: E402


@torch.no_grad()
def main():
    device = torch.device('cuda:0')
    # audit 63 A32-ETT-PATH-FINITE: fail closed -- max(worst, nan) keeps worst, so every value is
    # checked for being a finite number before it is compared
    diffs, bad = [], []
    for data in A.DATASETS:
        for cond in ('q1_R', 'pearson_R'):
            for seed in (7, 13, 21, 42):
                _, run, result = A.find_cell(data, 96, cond, seed)
                model, args = A.load_model(run, device)
                _, val = data_provider(args, 'val')
                now = evaluate(model, val, args)[0]['mse']
                stored = json.load(open(result)).get('train', {}).get('best_val_loss')
                finite = all(isinstance(v, float) and math.isfinite(v) for v in (now, stored))
                d = abs(now - stored) if finite else None
                diffs.append(d)
                if d is None or not d <= 1e-6:
                    bad.append(f'{data} {cond}/{seed}: {now!r} vs {stored!r}')
    ok = len(diffs) == 16 and not bad
    worst = max(d for d in diffs if d is not None) if any(d is not None for d in diffs) else None
    print(f"[weather-ett] {len(diffs)} ETT checkpoints, failures {bad}, max |diff| {worst!r} -> {ok}")
    Path(__file__).with_suffix('.json').write_text(json.dumps({'runs': len(diffs), 'max_abs_diff': worst,
                                                              'failures': bad, 'ok': ok}))


if __name__ == '__main__':
    main()
