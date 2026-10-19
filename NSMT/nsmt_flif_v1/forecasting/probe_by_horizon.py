"""Stage-4 memory probe re-aggregated by horizon (MODI_PROJECT_LOG 2026-10-19, user question: does H720 use memory?).

Reads results/<suite>/memory_probe_bestval/<DS>_H<H>_<cond>_seed<s>_*_probe.csv (written by neo_memory_probe.py) and
prints / writes the relative MSE increase when only the last 12 tokens are kept, per condition x dataset x horizon
(mean over seeds). No training, no new evaluation.
usage: python probe_by_horizon.py [--suite flif-v1-s4] [--keep 12] [--out results/<suite>/probe_by_horizon.csv]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-s4')
    ap.add_argument('--keep', type=int, default=12)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, 'results', a.suite, 'memory_probe_bestval', '*_probe.csv'))):
        m = re.match(r'.*/(ETT\w+?)_H(\d+)_(\S+?)_seed(\d+)_', f)
        d = pd.read_csv(f)
        k = d[(d.curve == 'keep_last') & (d.x == a.keep)]
        if len(k) != 1:
            raise RuntimeError(f'{f}: expected one keep_last row at x={a.keep}, got {len(k)}')
        r = k.iloc[0]
        rows.append(dict(data=m.group(1), H=int(m.group(2)), cond=m.group(3), seed=int(m.group(4)),
                         rel_dmse=float(r.dmse / r.clean_mse), clean_mse=float(r.clean_mse), windows=int(r.windows)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('no probe files')
    t = (df.groupby(['cond', 'data', 'H']).rel_dmse.agg(['mean', 'std', 'count']).reset_index())
    out = a.out or os.path.join(HERE, 'results', a.suite, f'probe_by_horizon_keep{a.keep}.csv')
    t.to_csv(out, index=False, float_format='%.6f')
    wide = t.pivot_table(index='cond', columns=['data', 'H'], values='mean') * 100
    print(f'relative MSE increase (%) when only the last {a.keep} tokens are kept, mean over seeds (n per cell = {sorted(set(t["count"]))})')
    print(wide.round(2).to_string())
    print(f'-> {out}')


if __name__ == '__main__':
    main()
