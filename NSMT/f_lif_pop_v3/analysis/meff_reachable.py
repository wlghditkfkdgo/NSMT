"""Reachability of O7-1's M_eff, for both the uniform-on-answer policy and the free bound.

Run:  python meff_reachable.py > meff_reachable.txt

Prereg 2H: the uniform-on-answer oracle is a DEFINED POLICY BASELINE, not an upper bound.
q_i = b_i p_i / sum_j b_j p_j is itself an arbitrary simplex point, so the adaptive mass
eta*B can be poured into whatever cap headroom the answer slots still have. At alpha=0.7,
history 41, answer lags {41, 1} and eta=0.2, uniform gives 0.1644974409 while the best p
gives 0.1744286355. Answer slots are placed contiguously from start = n - lag.
"""
import numpy as np
from math import lgamma, exp

T, ALPHA = 42, .7
_g = exp(lgamma(ALPHA + 1.))
B_COEF = np.array([((d + 1.) ** ALPHA - d ** ALPHA) / _g for d in range(T)])
B0, PAST = B_COEF[0], B_COEF[1:T][::-1]          # PAST[i] = b_{n-j}, 오래된 slot이 앞
MASS, N = PAST.sum(), T - 1


def m_eff(eta, p, answers):
    den = (PAST * p).sum()
    rho = (1 - eta) + (eta * MASS * p / den if den > 0 else 0.)
    c = np.minimum(PAST * rho, B0)

    return c[answers].sum() / c.sum()


def uniform_policy(eta, answers):
    p = np.zeros(N)
    p[answers] = 1. / len(answers)

    return m_eff(eta, p, answers)


def exact_bound(eta, answers):
    """Exact supremum of M_eff over every policy, in closed form (audit 17).

    With positive b_i capped at C, answer set A, B = sum b_i, B_A = sum_A b_i and m = |A|:

        S = min((1 - eta) B_A + eta B,  m C)
        U = S / [S + (1 - eta)(B - B_A)]

    S is the most the answer slots can hold: their own share of the neutral term plus all of
    the adaptive mass eta*B, but never more than the cap allows. The denominator adds the
    non-answer residual, which no policy can remove. A Dirichlet search over p is NOT this
    bound -- the audit found a constructed policy beating a 4,000-sample search by 0.0037 --
    so sampling is reported separately as `sampled_best`.
    """
    B_A = PAST[answers].sum()
    S = min((1 - eta) * B_A + eta * MASS, len(answers) * B0)

    return S / (S + (1 - eta) * (MASS - B_A))


def sampled_best(eta, answers, tries=4000, seed=0):
    """Best M_eff found by Dirichlet sampling. A lower bound on `exact_bound`, not a bound."""
    rng = np.random.default_rng(seed)
    best = uniform_policy(eta, answers)
    for _ in range(tries):
        p = np.zeros(N)
        p[answers] = rng.dirichlet(np.ones(len(answers)))
        best = max(best, m_eff(eta, p, answers))

    return best


def main():
    etas = (.2, .3, .5, .7, .9, 1.)
    print(f"alpha={ALPHA}, T={T}, b0={B0:.4f}, B={MASS:.4f}")
    print("uniform-on-answer POLICY (상한이 아님, prereg 2H)\n")
    print(f"{'정답 칸':>8} {'lag':>6} " + "".join(f"{f'eta={e}':>10}" for e in etas))
    for size in (1, 2, 3, 4, 6, 10):
        for lag in (5, 21, 35):
            idx = np.arange(max(0, N - lag), min(max(0, N - lag) + size, N))
            if len(idx) < size:
                continue
            print(f"{size:>8} {lag:>6} " + "".join(f"{uniform_policy(e, idx):>10.4f}" for e in etas))
        print()

    print("0.5에 닿는 eta (정답 칸을 lag 21부터 연속 배치)")
    print("  grid 열은 해상도 0.005에서의 최초 도달값이고, exact 열은 이분법으로 구한 연속 경계다.")
    print(f"{'정답 칸':>8} {'uniform(grid)':>14} {'bound(grid)':>12} {'bound(exact)':>13} {'sampled':>10}")
    for size in (1, 2, 3, 4, 6, 10):
        idx = np.arange(N - 21, N - 21 + size)
        grid = np.arange(.01, 1.001, .005)
        lo_u = next((e for e in grid if uniform_policy(e, idx) >= .5), None)
        lo_b = next((e for e in grid if exact_bound(e, idx) >= .5), None)
        lo, hi = 0., 1.                                   # exact_bound는 eta에 단조 증가
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if exact_bound(mid, idx) >= .5 else (mid, hi)
        sam = next((e for e in grid if sampled_best(e, idx) >= .5), None)
        fmt = lambda v: f'{v:.3f}' if v else '불가'
        print(f"{size:>8} {fmt(lo_u):>14} {fmt(lo_b):>12} {hi:>13.10f} {fmt(sam):>10}")


if __name__ == '__main__':
    main()
