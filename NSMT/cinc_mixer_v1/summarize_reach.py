"""Stage 9 report (MODI_PROJECT_LOG 2026-10-20): token-axis reach table from results/<suite>/reach/<cond>_seed<s>/.

For every condition: the share (%) of the token-mixing network's output spikes that change at lag d after token j is
replaced (rhythm_analysis.py), mean ± sd over seeds, and the network's mean firing rate. Negative lags = earlier tokens
(reverse direction of the bidirectional mixers). A seed without reach_lag.csv makes the row incomplete (fail-closed).
usage: python summarize_reach.py --suite flif-v1-rhythm-cinc
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ('nsmt_lif', 'nsmt_flif', 'nsmt_pop', 'mix_lif_bidir', 'mix_flif_bidir', 'mix_pop_bidir')
LAGS = (-10, -5, -1, 0, 1, 2, 5, 10, 20, 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', required=True)
    a = ap.parse_args()
    R = os.path.join(HERE, 'results', a.suite)
    seeds = (42, 7, 2026) if 'cinc' in a.suite else (42, 7, 2026, 123, 1024)
    L = [f'# Stage 9 `{a.suite}` — token-axis reach', '',
         'Share (%) of the token-mixing network\'s output spikes that change at lag d after one input token is replaced by '
         'another sample\'s token (best-validation weights, first 512 test units); mean ± sd over seeds '
         f'{seeds}. NSMT: neocortex network; Neo-mixer: first block\'s mixer (both directions fused).', '',
         '| condition | n | firing rate | ' + ' | '.join(f'd={d}' for d in LAGS) + ' |', '|---' * (3 + len(LAGS)) + '|']
    for c in CONDS:
        tabs, rates = [], []
        for s in seeds:
            d = os.path.join(R, 'reach', f'{c}_seed{s}')
            if not os.path.exists(os.path.join(d, 'reach_lag.csv')):
                break
            tabs.append(pd.read_csv(os.path.join(d, 'reach_lag.csv')).set_index('lag')['change'])
            rates.append(json.load(open(os.path.join(d, 'meta.json')))['rate'])
        if len(tabs) != len(seeds):
            L.append(f'| {c} | {len(tabs)} | incomplete |' + ' |' * len(LAGS))
            continue
        cells = []
        for lag in LAGS:
            v = np.array([t.get(lag, np.nan) for t in tabs], dtype=float)
            cells.append('n/a' if not np.all(np.isfinite(v)) else f'{100 * v.mean():.2f} ± {100 * v.std(ddof=1):.2f}')
        r = np.array(rates)
        L.append(f'| {c} | {len(tabs)} | {r.mean():.4f} ± {r.std(ddof=1):.4f} | ' + ' | '.join(cells) + ' |')
    open(os.path.join(R, 'reach_summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
