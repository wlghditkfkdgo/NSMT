"""Compare nsmt_base ETT runs (results/<suite>/raw.txt) with the paper's Table 13.

usage: python summarize_ett.py <suite> [--md results/<suite>/summary.md]
Reads every RESULT line (seed, mse, mae), averages over seeds per (dataset, pred_len) and prints
the difference to the paper value. Nothing is dropped: SKIP lines are listed as failures.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

PAPER = {  # Table 13, NSMT column (MSE, MAE)
    'ETTh1': {96: (.378, .386), 192: (.426, .417), 336: (.473, .441), 720: (.461, .454)},
    'ETTh2': {96: (.288, .334), 192: (.358, .380), 336: (.403, .412), 720: (.409, .429)},
    'ETTm1': {96: (.317, .345), 192: (.364, .371), 336: (.390, .391), 720: (.446, .425)},
    'ETTm2': {96: (.175, .251), 192: (.241, .297), 336: (.302, .335), 720: (.399, .393)},
}
HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('suite')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    raw = HERE / 'results' / a.suite / 'raw.txt'
    runs, fails = defaultdict(list), []
    for line in raw.read_text().splitlines():
        if line.startswith('SKIP'):
            fails.append(line)
            continue
        m = re.match(r'RESULT exp=nsmt_base dataset=(\w+?)_p(\d+) seed=(\d+) mse=([0-9.]+) mae=([0-9.]+)', line)
        if m:
            runs[(m.group(1), int(m.group(2)))].append((int(m.group(3)), float(m.group(4)), float(m.group(5))))
    lines = [f'# nsmt_base ETT check — suite `{a.suite}`', '',
             '| dataset | H | seeds | MSE (ours) | MSE (paper) | ΔMSE | MAE (ours) | MAE (paper) | ΔMAE |',
             '|---|---|---|---|---|---|---|---|---|']
    worst = 0.
    for ds, table in PAPER.items():
        for pl, (pm, pa) in table.items():
            rs = runs.get((ds, pl), [])
            if not rs:
                lines.append(f'| {ds} | {pl} | 0 | not run | {pm:.3f} | – | not run | {pa:.3f} | – |')
                continue
            mse = sum(r[1] for r in rs) / len(rs)
            mae = sum(r[2] for r in rs) / len(rs)
            worst = max(worst, abs(mse - pm), abs(mae - pa))
            seeds = ','.join(str(r[0]) for r in rs)
            lines.append(f'| {ds} | {pl} | {seeds} | {mse:.6f} | {pm:.3f} | {mse - pm:+.4f} | {mae:.6f} | {pa:.3f} | {mae - pa:+.4f} |')
    lines += ['', f'largest |difference| over the filled cells: {worst:.4f}',
              f'failed runs: {len(fails)}'] + [f'- {f}' for f in fails]
    text = '\n'.join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + '\n')


if __name__ == '__main__':
    main()
