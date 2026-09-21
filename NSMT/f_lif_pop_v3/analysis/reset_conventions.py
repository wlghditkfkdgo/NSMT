"""Same scalar neuron, same input, three reset conventions. Read-only comparison.
(a) paper Eq.(13): fractional charge -> spike from that state -> reset that state -> store post-reset U
(b) released spikeDE code: spike from U+D (local Euler trial), reset folded into F as -theta*S/tau
(c) v3-A (DH style): branch never resets; a separate soma spikes with subtractive reset
"""
import math
import numpy as np
np.set_printoptions(precision=3, suppress=True)

TAU, ALPHA, THETA, T = 4.0, 0.7, 1.0, 40
I = np.full(T, 1.5)


def b(d, a=ALPHA):
    return ((d + 1) ** a - d ** a) / math.gamma(a + 1)


def paper_eq13():
    U, S = np.zeros(T + 1), np.zeros(T)
    for k in range(1, T + 1):
        s = sum(b(k - 1 - j) / TAU * (-U[j] + I[min(j + 1, T - 1)]) for j in range(k))
        if s >= THETA:
            S[k - 1], U[k] = 1.0, s - THETA        # post-convolution reset, stored in history
        else:
            U[k] = s
    return U[1:], S


def code_convention():
    U, F, S = np.zeros(T + 1), np.zeros(T), np.zeros(T)
    for n in range(T):
        D = (-U[n] + I[n]) / TAU
        S[n] = 1.0 if U[n] + D >= THETA else 0.0   # spike from the local Euler trial
        F[n] = D - THETA * S[n] / TAU              # reset lives inside the dynamics term
        U[n + 1] = sum(b(n - j) * F[j] for j in range(n + 1))
    return U[1:], S


def v3a():
    u, f, v, s = np.zeros(T + 1), np.zeros(T), np.zeros(T + 1), np.zeros(T + 1)
    for n in range(T):
        f[n] = (I[n] - u[n]) / TAU                 # branch: no reset anywhere
        u[n + 1] = sum(b(n - j) * f[j] for j in range(n + 1))
        v[n + 1] = u[n + 1] - THETA * s[n]         # soma: tau_s=1, subtractive reset
        s[n + 1] = 1.0 if v[n + 1] >= THETA else 0.0
    return u[1:], s[1:]


for name, fn in (("(a) paper Eq.13 ", paper_eq13), ("(b) spikeDE code", code_convention), ("(c) v3-A DH style", v3a)):
    state, spikes = fn()
    idx = np.flatnonzero(spikes)
    print(f"{name}: spikes={int(spikes.sum()):2d}  first spike at n={idx[0] if len(idx) else '-'}  "
          f"ISIs={np.diff(idx)[:6]}  max|state|={np.abs(state).max():.3f}  state[-1]={state[-1]:.3f}")
print()
print("state trajectory, first 14 steps")
for name, fn in (("(a) paper", paper_eq13), ("(b) code ", code_convention), ("(c) v3-A ", v3a)):
    print(f"  {name}:", fn()[0][:14])
print()
print("what feeds the history in each case")
print("  (a) post-reset U_j        -> the reset changes every later state through the power-law kernel")
print("  (b) F_j containing -theta*S_j/tau -> the reset is a current, smeared over all later steps")
print("  (c) f_j with no reset term -> the reset never enters the fractional memory at all")
