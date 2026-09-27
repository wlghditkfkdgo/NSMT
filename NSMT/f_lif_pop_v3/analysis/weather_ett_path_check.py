#!/usr/bin/env python3
"""Prereg 2R D-CJ pre-check 5: adding the weather loader leaves the ETT path unchanged -- 16 of the
2O R-on spiking checkpoints (q1_R and pearson_R, ETTh1/ETTh2, seeds 7, 13, 21, 42) reproduce their
stored best validation MSE within 1e-6 on the GPU they were trained on."""
import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
from data_provider.data_factory import data_provider            # noqa: E402
from test import evaluate                                       # noqa: E402
import ett_alpha as A                                           # noqa: E402


@torch.no_grad()
def main():
    device = torch.device('cuda:0')
    worst, n = 0., 0
    for data in A.DATASETS:
        for cond in ('q1_R', 'pearson_R'):
            for seed in (7, 13, 21, 42):
                _, run, result = A.find_cell(data, 96, cond, seed)
                model, args = A.load_model(run, device)
                _, val = data_provider(args, 'val')
                d = abs(evaluate(model, val, args)[0]['mse'] - json.load(open(result))['train']['best_val_loss'])
                worst, n = max(worst, d), n + 1
    print(f"[weather-ett] {n} ETT checkpoints reproduce best validation MSE, max |diff| {worst:.2e} -> {worst <= 1e-6}")
    Path(__file__).with_suffix('.json').write_text(json.dumps({'runs': n, 'max_abs_diff': worst, 'ok': worst <= 1e-6}))


if __name__ == '__main__':
    main()
