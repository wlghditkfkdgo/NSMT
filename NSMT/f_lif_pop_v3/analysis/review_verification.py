"""Independent re-check of the claims in Population_fLIF_v3A_Critical_Review_and_O7_Decisions_KO.md
and a test of candidate remedies. Read-only scalar/matrix computation, no training."""
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)

A, H = 0.7, 1.0
def b(d): return ((d + 1) ** A - d ** A) / math.gamma(A + 1)
B = lambda n: sum(b(n - j) for j in range(n))           # past mass at step n (lags 1..n)
sig = lambda x: 1 / (1 + math.exp(-x))
softplus = lambda x: math.log1p(math.exp(x))

print("=== 1. kappa with pi_k at the ORIGINAL initial g=softplus(-5), n=41, uniform p  (review 5.3) ===")
g0 = softplus(-5.0)
for tau in (2., 4., 8., 16.):
    num = sum(b(d) * math.exp(-d * g0 / tau) for d in range(1, 42))
    print(f"  tau={tau:4.0f}: kappa = {num / B(41):.10f}")

def branch(T, tau, I, policy, eta, cap=None, select_input_only=False):
    """policy(n) -> dict {j: p_j} over j<n.  Returns trajectory u_1..u_T."""
    u = np.zeros(T + 1); f = np.zeros(T); Iarr = np.array(I, dtype=float)
    for n in range(T):
        f[n] = (Iarr[n] - u[n]) / tau
        acc = b(0) * (Iarr[n] if select_input_only else f[n]) / (tau if select_input_only else 1.)
        if select_input_only:
            acc -= b(0) * u[n] / tau
        if n > 0:
            p = policy(n)
            Bn = B(n)
            denom = sum(b(n - j) * p.get(j, 0.) for j in range(n))
            for j in range(n):
                rho_t = Bn * p.get(j, 0.) / denom if denom > 0 else 1.
                rho = (1 - eta) + eta * rho_t
                c = b(n - j) * rho
                if cap is not None: c = min(c, cap)
                if select_input_only:
                    acc += (c * Iarr[j] - b(n - j) * u[j]) / tau     # leak keeps the fixed kernel
                else:
                    acc += c * f[j]
        u[n + 1] = acc
    return u[1:]

T = 42; pulse = [2.] + [0.] * (T - 1)
recent = lambda n: {n - 1: 1.}
far0 = lambda n: {0: 1.}

print()
print("=== 2. review 7.2: recent-slot policy, tau=2, alpha=0.7, I0=2 then 0, g=0 -> max|u| ===")
for eta in (0., sig(-4), 0.3, 0.5):
    print(f"  eta={eta:.4f}: max|u| = {np.abs(branch(T, 2., pulse, recent, eta)).max():,.4f}")

print()
print("=== 3. where does the loop gain eta*B_n/tau cross 1 ? (tau=2) ===")
for eta in (0.3, 0.5):
    n_cross = next((n for n in range(1, T) if eta * B(n) / 2. > 1), None)
    print(f"  eta={eta}: gain>1 from n={n_cross}   (B_41={B(41):.3f}, eta*B_41/2={eta*B(41)/2:.2f})")
print(f"  tau_min/B_41 = {2 / B(41):.4f}  -> eta below this keeps worst-case gain < 1 over T=42")

print()
print("=== 4. same recent policy on the SLOW branch tau=16 ===")
for eta in (0.3, 0.5, 1.0):
    print(f"  eta={eta}: max|u| = {np.abs(branch(T, 16., pulse, recent, eta)).max():.4f}")

print()
print("=== 5. moving 'recent' slot vs FIXED far slot (j=0), tau=2 ===")
for eta in (0.5, 1.0):
    print(f"  eta={eta}: recent -> {np.abs(branch(T, 2., pulse, recent, eta)).max():,.2f}   fixed j=0 -> {np.abs(branch(T, 2., pulse, far0, eta)).max():.4f}")

print()
print("=== 6. remedies on the worst case (recent policy, tau=2) ===")
for eta in (0.5, 1.0):
    r_bound = np.abs(branch(T, 2., pulse, recent, min(eta, 0.07))).max()
    r_cap = np.abs(branch(T, 2., pulse, recent, eta, cap=b(0))).max()
    r_in = np.abs(branch(T, 2., pulse, recent, eta, select_input_only=True)).max()
    r_in_far = np.abs(branch(T, 2., pulse, far0, eta, select_input_only=True)).max()
    print(f"  eta={eta}: (R1) eta clipped to 0.07 -> {r_bound:.4f} | (R3) coefficient cap c<=b0={b(0):.3f} -> {r_cap:.4f} | "
          f"(R4) input-only selection: recent -> {r_in:.4f}, fixed j=0 -> {r_in_far:.4f}")

print()
print("=== 7. kappa under the cap remedy (worst case, eta=1, recent policy, n=41) ===")
c_recent = min(b(1) * (B(41) / b(1)), b(0)); kappa = c_recent / B(41)
print(f"  single-support: capped coefficient {c_recent:.4f}, kappa = {kappa:.4f}  (mass not preserved, reported instead)")
print(f"  max amplification allowed by cap at lag 41: b0/b41 = {b(0)/b(41):.2f}x")

print()
print("=== 8. review 8: full-history fast path must include leak feedback  (tau=4, g=0) ===")
rng = np.random.default_rng(7); I = rng.standard_normal(T); tau = 4.
u_loop = branch(T, tau, I, lambda n: {}, 0.)
Bm = np.array([[b(r - c) if r >= c else 0. for c in range(T)] for r in range(T)])
J = np.eye(T, k=-1)
y = np.linalg.solve(np.eye(T) + Bm @ J / tau, Bm @ I / tau)          # (I + BJ/tau) y = B I / tau
y_naive = Bm @ I / tau                                                 # what IDEA_LOG 7.1 implied
print(f"  |loop - H_eff|_max = {np.abs(u_loop - y).max():.2e}     |loop - naive B@I/tau|_max = {np.abs(u_loop - y_naive).max():.3f}")

print()
print("=== 9. review 13.3: oracle effective mass with small eta (m0=0.15) ===")
for eta in (sig(-4), 0.07, 0.2, 0.41):
    print(f"  eta={eta:.4f}: M_eff = {(1-eta)*0.15 + eta:.4f}")
