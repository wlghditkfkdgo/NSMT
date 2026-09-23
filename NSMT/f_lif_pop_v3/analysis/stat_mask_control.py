"""Proper controls for the fixed-lag mask, after audit 31 (A14-UNIT / A14-CONTROL / A14-SPLIT).

`stat_mask_feasibility.txt` compared a fixed top-k lag set against `k/41`, the coverage a
random draw of k lags out of 41 would give. That control is wrong for the question we
actually care about. At query position n only lags 1..n exist, so a fixed top-k set leaves
c_n = #{lag in top-k : lag <= n} usable picks, not k. The honest control keeps c_n and draws
that many slots uniformly from the n that exist, which is a far stronger baseline than k/41.

Two coverage measures are reported because they answer different questions and the fixed set
wins one and loses the other:

  pooled   - of every correct slot in the corpus, what fraction does the mask keep
  any-hit  - of every query, what fraction keeps at least one correct slot

Both are exact expectations, not simulations: pooled uses c_n * a_n / n and any-hit uses the
hypergeometric 1 - C(n - a_n, c_n) / C(n, c_n).

Split (A14-SPLIT): validation seed only. The lags are chosen on the same 400 sequences the
coverage is then measured on, so these are in-sample numbers and not an estimate of what a
mask fitted on train would do on unseen data.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
from data_provider.synthetic import make_sequence                       # noqa: E402

N_SEQ, T, SEED = 400, 42, 20260921 + 10000                              # validation split
KS = (3, 5, 8, 12, 20)


def collect(n_seq=N_SEQ, seed=SEED):
    """Every recall query as (n, lags of its correct slots)."""
    rng = np.random.default_rng(seed)
    queries = []
    for _ in range(n_seq):
        _, _, truth, kind = make_sequence(rng, n_events=T)
        for n in np.flatnonzero(kind > 0):                              # kind 0 = copy, not a recall
            src = np.flatnonzero(truth[n])
            if src.size:
                queries.append((int(n), n - src))                       # lag = distance back
    return queries


def lag_frequency(queries, n_lags=T - 1):
    """Pooled count of correct slots at each lag -- what a fixed mask would be fitted on."""
    freq = np.zeros(n_lags + 1)
    for _, lags in queries:
        np.add.at(freq, lags, 1.)
    return freq[1:]                                                     # lag 0 is the query itself


def evaluate(queries, keep):
    """Coverage of a fixed lag set, and of the matched-budget random control."""
    keep = set(int(v) for v in keep)
    hit = tot = 0.
    any_fixed = any_ctrl = 0.
    pooled_ctrl = 0.
    budget = []
    for n, lags in queries:
        a = lags.size                                                   # correct slots for this query
        tot += a
        hit += sum(1 for L in lags if L in keep)
        c = sum(1 for L in keep if L <= n)                              # picks the mask can actually use
        budget.append(c)
        any_fixed += float(any(L in keep for L in lags))
        if n > 0 and c > 0:
            pooled_ctrl += c * a / n                                    # E[hits] drawing c of n slots
            any_ctrl += 1. - hypergeom.pmf(0, n, a, c)              # P(at least one correct slot)
    return dict(pooled=hit / tot, pooled_control=pooled_ctrl / tot,
                any_hit=any_fixed / len(queries), any_control=any_ctrl / len(queries),
                budget=float(np.mean(budget)))


def main():
    queries = collect()
    freq = lag_frequency(queries)
    order = np.argsort(-freq) + 1                                       # lag index is 1-based
    total_slots = sum(lags.size for _, lags in queries)

    print('=== split: validation seed %d, %d sequences (A14-SPLIT: NOT train) ===' % (SEED, N_SEQ))
    print('  recall queries = %d,  correct slots = %d,  slots per query = %.9f'
          % (len(queries), total_slots, total_slots / len(queries)))
    print('  top-8 lags by pooled frequency = %s' % list(order[:8]))

    print('\n=== A14-UNIT: what the old 0.908 actually was ===')
    per_query = freq / len(queries)                                     # the old, wrong denominator
    for k in KS:
        print('  top-%2d: sum of per-query rates = %.9f  <- expected COUNT, not a probability'
              % (k, per_query[order[:k] - 1].sum()))

    print('\n=== A14-CONTROL: fixed lag set vs matched-budget random ===')
    print('  %-6s %10s %10s %10s | %10s %10s %10s' %
          ('k', 'pooled', 'ctrl', 'k/41', 'any-hit', 'ctrl', 'picks'))
    for k in KS:
        r = evaluate(queries, order[:k])
        print('  top-%-2d %9.4f%% %9.4f%% %9.4f%% | %9.4f%% %9.4f%% %10.4f'
              % (k, 100 * r['pooled'], 100 * r['pooled_control'], 100 * k / (T - 1),
                 100 * r['any_hit'], 100 * r['any_control'], r['budget']))


if __name__ == '__main__':
    main()
