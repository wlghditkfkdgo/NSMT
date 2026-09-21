"""Pre-registration numerics for the shared-soma population f-LIF (v3 design).

Read-only analysis, no training. Conventions: 1/tau inside the dynamics, h=1,
predictor coefficients b_d = [(d+1)^a - d^a]/Gamma(a+1); dendritic branches never reset;
one soma per logical neuron emits the single spike.
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)
TAUS = np.array([2., 4., 8., 16.])


def b(d, a):
    return ((d + 1) ** a - d ** a) / math.gamma(a + 1)


def branches(I, taus, a, rho=None):
    """u[t+1,k] = sum_{j<=t} b_{t-j} rho_{t,j} f_{j,k},  f_{j,k} = (I_j - u_{j,k})/tau_k. No reset."""
    T, K = len(I), len(taus)
    u = np.zeros((T + 1, K))
    f = np.zeros((T, K))
    for t in range(T):
        f[t] = (I[t] - u[t]) / taus
        w = np.array([b(t - j, a) * (1.0 if rho is None else rho(t, j)) for j in range(t + 1)])
        u[t + 1] = (w[:, None] * f[:t + 1]).sum(0)
    return u[1:], f


def soma(uk, w_k, tau_s=2.0, theta=1.0):
    """v_t = v_{t-1} + (a_t - v_{t-1})/tau_s - theta*s_{t-1};  s_t = H(v_t - theta). Euler, subtractive reset."""
    T = len(uk)
    v, s = np.zeros(T + 1), np.zeros(T + 1)
    for t in range(T):
        a_t = uk[t] @ w_k
        v[t + 1] = v[t] + (a_t - v[t]) / tau_s - theta * s[t]
        s[t + 1] = 1.0 if v[t + 1] >= theta else 0.0
    return v[1:], s[1:]


if __name__ == "__main__":
    print("=" * 78)
    print("B1. alpha=1, rho=1 : branch recursion must equal the ordinary leaky integrator")
    I = np.array([1.5, .2, .2, 1.5, .2, .9, .3, 1.2])
    u_frac, _ = branches(I, TAUS, 1.0)
    u_rec = np.zeros((len(I), 4)); prev = np.zeros(4)
    for t in range(len(I)):
        prev = prev + (I[t] - prev) / TAUS
        u_rec[t] = prev
    print("max|fractional-sum - Euler recursion| =", np.abs(u_frac - u_rec).max())

    print()
    print("B2. constant input I=1, no spiking: branch value after n steps (finite-horizon gain)")
    print(f"{'alpha':>6} | " + " | ".join(f"tau={t:<4.0f}" for t in TAUS))
    for a in (0.5, 0.7, 1.0):
        for n in (10, 42):
            u, _ = branches(np.ones(n), TAUS, a)
            print(f"{a:>6} | " + " | ".join(f"n={n:<2d} {v:.3f}" for v in u[-1]))

    print()
    print("B3. kernel-mass (attenuation) confound: multiply every rho by a constant c")
    for c in (1.0, 0.5, 0.25):
        u, _ = branches(np.ones(42), TAUS, 0.7, rho=lambda t, j, c=c: c if j < t else 1.0)
        print(f"  c={c:<5} branch values at n=42:", u[-1])

    print()
    print("B4. firing preview at init (standardised input x input_scale, w_k=1/K, tau_s=2, theta=1)")
    rng = np.random.default_rng(7)
    x = rng.standard_normal(42)
    for scale in (1.0, 2.0, 4.0):
        u, _ = branches(x * scale, TAUS, 0.7)
        v, s = soma(u, np.full(4, .25))
        print(f"  input_scale={scale:<4} mean|u_k|={np.abs(u).mean(0).round(3)}  soma firing rate={s.mean():.3f}")

    print()
    print("=" * 78)
    print("A. Worked selection example (alpha=0.7, K=4, identity Q/K, temperature 0.25)")
    I = np.array([1.5, .2, .2, 1.5, .2])          # step 3 repeats the step-0 context
    u, f = branches(I, TAUS, 0.7)
    t = 4                                          # current step; candidates j=0..3
    xi = np.hstack([u, I[:, None]])                # key context [u_1..u_K ; I]
    q, ks = xi[t], xi[:t]
    e = -((q - ks) ** 2).mean(1) / 0.25            # score = -mean squared distance / temperature
    print("  input I                :", I)
    print("  branch states u (t=4)  :", u[t].round(4))
    print("  scores e_{4,j}         :", e.round(3), "  (j=0,1,2,3)")


    def sparsemax(z):
        zs = np.sort(z)[::-1]
        css = np.cumsum(zs)
        k = np.arange(1, len(z) + 1)
        sup = k[(1 + k * zs) > css][-1]
        tau_thr = (css[sup - 1] - 1) / sup
        return np.maximum(z - tau_thr, 0)


    p_soft = np.exp(e - e.max()); p_soft /= p_soft.sum()
    p_sp = sparsemax(e)
    bd = np.array([b(t - j, 0.7) for j in range(t)])
    B = bd.sum()
    print("  softmax p              :", p_soft.round(4))
    print("  sparsemax p            :", p_sp.round(4))
    for name, p in (("softmax", p_soft), ("sparsemax", p_sp)):
        r_max = p / p.max()
        r_np = len(p) * p
        r_mass = B * p / (bd * p).sum()
        r_conv = 0.5 + 0.5 * r_mass                                  # eta = 0.5 convex blend
        print(f"  --- {name}")
        print("      rho = p/max p      :", r_max.round(3), " kernel mass ratio kappa =", round((bd * r_max).sum() / B, 3))
        print("      rho = n p          :", r_np.round(3), " kappa =", round((bd * r_np).sum() / B, 3))
        print("      rho = B p / sum(bp):", r_mass.round(3), " kappa =", round((bd * r_mass).sum() / B, 3))
        print("      rho = (1-e)+e*mass :", r_conv.round(3), " kappa =", round((bd * r_conv).sum() / B, 3))
    print("  fractional coefficients b_d (d=4-j):", bd.round(4), " total B =", round(B, 4))
    print("  final coefficients b*rho (sparsemax, convex eta=.5):", (bd * (0.5 + 0.5 * (B * p_sp / (bd * p_sp).sum()))).round(4))
