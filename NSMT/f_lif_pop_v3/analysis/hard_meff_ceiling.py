"""Post-hoc (after 2M): the highest M_eff ANY selector can reach under the top-q budget.

O7-1 asks M_eff >= 0.5, where M_eff is the answer share of the kept kernel mass. mode=hard at
q keeps k = max(1, round(q n)) of the n past slots, so even a perfect ranker must fill k - a
slots that are not answers. The ceiling below keeps every answer (or the k heaviest answers
when k < a) and fills the rest with the LIGHTEST non-answer slots, which maximises the share.
Aggregated like D-Q: recall queries within a sequence, then sequences. Generator only, no
model, VALIDATION seed. This does not change the 2M verdict; it says whether the threshold
was reachable at this budget (compare prereg 2G, the same question for the eta design).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
from data_provider.synthetic import make_sequence                       # noqa: E402
from layers import fractional_coefficients                              # noqa: E402

N_SEQ, T, SEED, ALPHA = 1000, 42, 20260921 + 10000, .7


def ceiling(lags, n, q, b):
    """lags: answer lags (1..n). Best achievable answer share of kept kernel mass."""
    w = b[np.arange(1, n + 1)]                           # w[d-1] = b(d), lag d
    ans = np.zeros(n, dtype=bool)
    ans[lags - 1] = True
    k = n if q >= 1 else max(1, int(round(q * n)))
    a = ans.sum()
    if k <= a:
        return 1.
    other = np.sort(w[~ans])[:k - a]                     # 가장 가벼운 비정답 칸으로 채운다
    return w[ans].sum() / (w[ans].sum() + other.sum())


def main():
    b = fractional_coefficients(ALPHA, T + 1).double().numpy()
    rng = np.random.default_rng(SEED)
    rows = {q: [] for q in (.1, .25, .5, 1.)}
    for _ in range(N_SEQ):
        _, _, truth, kind = make_sequence(rng, n_events=T)
        per = {q: [] for q in rows}
        for n in np.flatnonzero(kind > 0):
            src = np.flatnonzero(truth[n])
            if src.size:
                for q in rows:
                    per[q].append(ceiling(n - src, n, q, b))
        for q in rows:
            if per[q]:
                rows[q].append(np.mean(per[q]))
    print(f'validation seed {SEED}, {N_SEQ} sequences; M_eff ceiling for ANY selector at budget q (D-Q aggregation)')
    for q, v in rows.items():
        print(f'  q = {q:<4}  ceiling {np.mean(v):.4f}   (O7-1 threshold 0.5: {"reachable" if np.mean(v) >= .5 else "NOT reachable"})')


if __name__ == '__main__':
    main()
