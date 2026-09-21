"""Why does the paper Eq.(13) convention fire every step here? Print pre/post-reset values."""
import math
import numpy as np
np.set_printoptions(precision=3, suppress=True)
TAU, ALPHA, THETA, T = 4.0, 0.7, 1.0, 22
I = np.full(T, 1.5)
b = lambda d: ((d + 1) ** ALPHA - d ** ALPHA) / math.gamma(ALPHA + 1)

for mode in ("soft", "hard", "none"):
    U, pre, S = np.zeros(T + 1), np.zeros(T + 1), np.zeros(T)
    for k in range(1, T + 1):
        s = sum(b(k - 1 - j) / TAU * (-U[j] + I[min(j + 1, T - 1)]) for j in range(k))
        pre[k] = s
        if mode != "none" and s >= THETA:
            S[k - 1] = 1.0
            U[k] = s - THETA if mode == "soft" else 0.0
        else:
            U[k] = s
    print(f"reset={mode:5s} spikes={int(S.sum()):2d}")
    print("   pre-reset U :", pre[1:15])
    print("   stored   U :", U[1:15])
    print("   leak term -U_j is small after a reset -> next sum jumps back over theta")

print()
print("parameter sweep: does the every-step firing persist?  (40 steps, soft reset)")
print("  tau alpha    I  | spikes | longest consecutive run | max pre-reset U")
for tau in (2., 4., 8.):
    for a in (0.5, 0.7, 0.9):
        for Ic in (1.2, 1.5, 2.0):
            T2 = 40
            bb = lambda d, a=a: ((d + 1) ** a - d ** a) / math.gamma(a + 1)
            U, pre, S = np.zeros(T2 + 1), np.zeros(T2 + 1), np.zeros(T2)
            for k in range(1, T2 + 1):
                s = sum(bb(k - 1 - j) / tau * (-U[j] + Ic) for j in range(k))
                pre[k] = s
                if s >= 1.0:
                    S[k - 1], U[k] = 1.0, s - 1.0
                else:
                    U[k] = s
            run = mx = 0
            for v in S:
                run = run + 1 if v else 0
                mx = max(mx, run)
            print(f"  {tau:4.0f} {a:5} {Ic:4} | {int(S.sum()):6d} | {mx:23d} | {pre.max():14.3f}")
