"""G11 pre-check: is the R3 coefficient cap  c = min(b_d * rho, b_0)  stable under adversarial
selection policies?  Read-only scalar computation, g=0 (pi=1), no training."""
import math
import numpy as np
np.set_printoptions(precision=3, suppress=True)

A = 0.7
b = lambda d: ((d + 1) ** A - d ** A) / math.gamma(A + 1)
B = lambda n: sum(b(n - j) for j in range(n))
B0 = b(0)


def run(T, tau, I, policy, eta, cap):
    """u_{n+1} = b0 f_n + sum_{j<n} c_{n,j} f_j ;  c = min(b_{n-j}*rho, b0) if cap else b*rho"""
    u = np.zeros(T + 1); f = np.zeros(T)
    for n in range(T):
        f[n] = (I[n] - u[n]) / tau
        acc = B0 * f[n]
        if n > 0:
            p = policy(n, u, f)
            den = sum(b(n - j) * p.get(j, 0.) for j in range(n))
            Bn = B(n)
            for j in range(n):
                rt = Bn * p.get(j, 0.) / den if den > 0 else 1.
                c = b(n - j) * ((1 - eta) + eta * rt)
                acc += (min(c, B0) if cap else c) * f[j]
        u[n + 1] = acc
        if not np.isfinite(u[n + 1]) or abs(u[n + 1]) > 1e12:
            return u[1:n + 2]
    return u[1:]


def recent_k(k):
    def f(n, u, fh):
        idx = list(range(max(0, n - k), n))
        return {j: 1. / len(idx) for j in idx}
    return f


def far1(n, u, fh): return {0: 1.}
def uniform(n, u, fh): return {j: 1. / n for j in range(n)}


def rand_support(m, seed):
    rng = np.random.default_rng(seed)
    def f(n, u, fh):
        idx = rng.choice(n, size=min(m, n), replace=False)
        return {int(j): 1. / len(idx) for j in idx}
    return f


def greedy_adv(T, tau, I, eta, cap):
    """At each step choose the single past slot that maximises |u_{n+1}|. True worst single-support case."""
    u = np.zeros(T + 1); f = np.zeros(T)
    for n in range(T):
        f[n] = (I[n] - u[n]) / tau
        best = B0 * f[n]
        if n > 0:
            Bn = B(n); cand = []
            for jstar in range(n):
                acc = B0 * f[n]
                for j in range(n):
                    rt = Bn / b(n - jstar) if j == jstar else 0.
                    c = b(n - j) * ((1 - eta) + eta * rt)
                    acc += (min(c, B0) if cap else c) * f[j]
                cand.append(acc)
            best = max(cand, key=abs)
        u[n + 1] = best
        if not np.isfinite(u[n + 1]) or abs(u[n + 1]) > 1e12:
            return u[1:n + 2]
    return u[1:]


POLICIES = [("recent-1", recent_k(1)), ("recent-4", recent_k(4)), ("recent-12", recent_k(12)),
            ("far-1(j=0)", far1), ("uniform", uniform), ("random-4", rand_support(4, 7))]
rng = np.random.default_rng(0)
INPUTS = {"pulse": lambda T: np.r_[2., np.zeros(T - 1)],
          "const": lambda T: np.ones(T) * 1.5,
          "noise": lambda T: rng.standard_normal(T) * 2}

for T in (42, 84):
    print(f"\n{'='*96}\nT={T}   max|u| over the window.   'x' = diverged (>1e12)\n{'='*96}")
    print(f"{'input':6} {'tau':>4} {'eta':>6} | " + " | ".join(f"{nm:>10}" for nm, _ in POLICIES) + " | " + f"{'greedy-adv':>10}")
    for iname, ifn in INPUTS.items():
        I = ifn(T)
        for tau in (2., 16.):
            for eta in (0.3, 1.0):
                row = []
                for nm, pol in POLICIES:
                    for capped in (True,):
                        m = np.abs(run(T, tau, I, pol, eta, capped)).max()
                    row.append("x" if m > 1e11 else f"{m:10.3f}")
                g = np.abs(greedy_adv(T, tau, I, eta, True)).max()
                row.append("x" if g > 1e11 else f"{g:10.3f}")
                print(f"{iname:6} {tau:4.0f} {eta:6.2f} | " + " | ".join(f"{v:>10}" for v in row))

print(f"\n{'='*96}\nCap ON vs OFF, worst configuration (greedy adversarial, pulse input)\n{'='*96}")
for T in (42, 84):
    for tau in (2., 4., 16.):
        for eta in (0.3, 0.5, 1.0):
            I = np.r_[2., np.zeros(T - 1)]
            on = np.abs(greedy_adv(T, tau, I, eta, True)).max()
            off = np.abs(greedy_adv(T, tau, I, eta, False)).max()
            print(f"  T={T:3d} tau={tau:4.0f} eta={eta:4.2f} | cap ON {on:12.4f} | cap OFF "
                  + ("diverged" if off > 1e11 else f"{off:16.2f}"))

print(f"\n{'='*96}\nneutral limit + exact zeros under the cap\n{'='*96}")
I = np.r_[2., np.zeros(41)]
u0 = run(42, 2., I, uniform, 0.0, True); u1 = run(42, 2., I, uniform, 0.0, False)
print(f"  eta=0: cap ON vs OFF identical? {np.allclose(u0, u1, atol=0, rtol=0)}  (cap inactive since b_d <= b_0 for d>=1)")
print(f"  eta=1, p_j=0 -> rho=0 -> c=0 : exact zero preserved = {min(b(5)*0.0, B0) == 0.0}")
print(f"  max amplification allowed by cap: lag 1 -> {B0/b(1):.2f}x , lag 41 -> {B0/b(41):.2f}x")

print(f"\n{'='*96}\nFIX CANDIDATES for the fast branch (greedy adversarial, worst case)\n{'='*96}")


def greedy_adv_cap(T, tau, I, eta, capval):
    u = np.zeros(T + 1); f = np.zeros(T)
    for n in range(T):
        f[n] = (I[n] - u[n]) / tau
        best = B0 * f[n]
        if n > 0:
            Bn = B(n); cand = []
            for jstar in range(n):
                acc = B0 * f[n]
                for j in range(n):
                    rt = Bn / b(n - jstar) if j == jstar else 0.
                    acc += min(b(n - j) * ((1 - eta) + eta * rt), capval) * f[j]
                cand.append(acc)
            best = max(cand, key=abs)
        u[n + 1] = best
        if not np.isfinite(u[n + 1]) or abs(u[n + 1]) > 1e12: return u[1:n + 2]
    return u[1:]


rng2 = np.random.default_rng(0)
CASES = {"pulse": lambda T: np.r_[2., np.zeros(T - 1)], "const": lambda T: np.ones(T) * 1.5,
         "noise": lambda T: rng2.standard_normal(T) * 2}

print("\n(F1) shift the bank one octave slower: tau = [4,8,16,32], cap = b0")
print(f"{'input':6} {'tau':>4} | " + " | ".join(f"T={T},eta={e:<4}" for T in (42, 84) for e in (0.5, 1.0)))
for iname, ifn in CASES.items():
    for tau in (4., 8., 16., 32.):
        vals = [np.abs(greedy_adv_cap(T, tau, ifn(T), e, B0)).max() for T in (42, 84) for e in (0.5, 1.0)]
        print(f"{iname:6} {tau:4.0f} | " + " | ".join(f"{v:12.4f}" for v in vals))

print("\n(F2) keep tau=[2,...] but use a tau-scaled cap  c <= min(b0, lam*tau)")
print(f"{'input':6} {'lam':>5} | " + " | ".join(f"T={T},eta={e:<4}" for T in (42, 84) for e in (0.5, 1.0)))
for iname, ifn in CASES.items():
    for lam in (0.5, 0.35, 0.25):
        vals = [np.abs(greedy_adv_cap(T, 2., ifn(T), e, min(B0, lam * 2.))).max() for T in (42, 84) for e in (0.5, 1.0)]
        print(f"{iname:6} {lam:5.2f} | " + " | ".join(f"{v:12.4f}" for v in vals))
print(f"  note: at tau=2, lam=0.5 -> cap 1.000 ; lam=0.35 -> 0.700 ; b_1={b(1):.3f} so neutral limit (eta=0) needs cap >= b_1")

print("\n(F3) does F1 keep a fast enough response?  step response of the branch (I=1 from n=0)")
for tau in (2., 4.):
    u = run(42, tau, np.ones(42), uniform, 0.0, True)
    print(f"  tau={tau:2.0f}: u at n=1,2,4,8,42 = {u[[0,1,3,7,41]].round(3)}")
