"""Why the 'recent' control fails on this task: answer-slot coverage of the two content-free
same-budget rules, from the generator alone (no model), on the VALIDATION seed.

At query position n the mode=hard rule keeps k = max(1, round(q n)) of the n past slots.
'recent' keeps lags 1..k; 'random' keeps a uniform k-subset. For each recall query with a
answer slots, report the expected fraction of answer slots kept and the probability that at
least one is kept (random: exact hypergeometric). Pooled over queries, sequence structure
ignored -- a description of the task, not an inference.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
from data_provider.synthetic import make_sequence                       # noqa: E402

N_SEQ, T, SEED, Q = 400, 42, 20260921 + 10000, .5                      # validation split, 2K budget


def main():
    rng = np.random.default_rng(SEED)
    rec_frac, rec_any, rnd_frac, rnd_any, lags = [], [], [], [], []
    for _ in range(N_SEQ):
        _, _, truth, kind = make_sequence(rng, n_events=T)
        for n in np.flatnonzero(kind > 0):
            src = np.flatnonzero(truth[n])
            if not src.size:
                continue
            lag = n - src
            a, k = lag.size, max(1, int(round(Q * n)))
            kept = (lag <= k).sum()
            rec_frac.append(kept / a)
            rec_any.append(float(kept > 0))
            rnd_frac.append(k / n)
            rnd_any.append(1. - hypergeom.pmf(0, n, a, k))
            lags.extend(lag.tolist())
    print(f'validation seed {SEED}, {N_SEQ} sequences, {len(rec_frac)} recall queries, q = {Q}')
    print(f'answer lag: mean {np.mean(lags):.2f}, median {np.median(lags):.0f} (in events)')
    print(f"{'rule':>7} {'answer slots kept':>18} {'>=1 answer kept':>16}")
    print(f"{'recent':>7} {np.mean(rec_frac):>18.4f} {np.mean(rec_any):>16.4f}")
    print(f"{'random':>7} {np.mean(rnd_frac):>18.4f} {np.mean(rnd_any):>16.4f}")


if __name__ == '__main__':
    main()
