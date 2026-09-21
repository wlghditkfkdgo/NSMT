"""L1 (Teka et al. 2014) form of the fractional LIF. Read-only numerical check.

Scheme (h=1):  tau * D^a V(t_N) = -V + I  with the L1 approximation
    D^a V(t_N) ~ 1/(Gamma(2-a) h^a) * sum_{k=0}^{N-1} w_{N,k} (V_{k+1}-V_k),
    w_{N,k} = (N-k)^(1-a) - (N-1-k)^(1-a),   w_{N,N-1} = 1  (the unknown increment)
so the explicit step is
    V_N = V_{N-1} + (Gamma(2-a) h^a / tau) (-V_{N-1} + I_{N-1}) - sum_{k<N-1} w_{N,k} (V_{k+1}-V_k).
NOTE: the k index of w must be taken w.r.t. the target time N, not N-1; that off-by-one
silently passes the alpha=1 test (all weights are 0 there) but oscillates for alpha<1.
"""
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)


def l1_flif(I, tau, a, theta=1.0, v_reset=0.0, h=1.0, spiking=True,
            refractory=0, trace_uses_reset=True):
    T = len(I)
    V = np.zeros(T + 1)
    dV = np.zeros(T + 1)
    S = np.zeros(T + 1)
    c = h ** a * math.gamma(2 - a) / tau
    ref = 0
    for N in range(T):                       # produces V[N+1] (formula index N+1)
        trace = sum(dV[k] * ((N + 1 - k) ** (1 - a) - (N - k) ** (1 - a)) for k in range(N))
        v_new = V[N] + c * (-V[N] + I[N]) - trace
        if ref > 0:                          # refractory: hold at reset, emit nothing
            V[N + 1], ref = v_reset, ref - 1
        elif spiking and v_new >= theta:
            S[N + 1], V[N + 1], ref = 1.0, v_reset, refractory
        else:
            V[N + 1] = v_new
        dV[N] = V[N + 1] - V[N] if (trace_uses_reset or S[N + 1] == 0) else v_new - V[N]
    return V[1:], S[1:]


print("1) alpha=1 must reduce exactly to Euler LIF (tau=4, no spikes)")
I = np.array([1.5, .2, .2, 1.5, .2, .9, .3, 1.2])
v_l1, _ = l1_flif(I, 4., 1.0, spiking=False)
v_eu = np.zeros(len(I)); p = 0.
for t in range(len(I)):
    p = p + (I[t] - p) / 4.; v_eu[t] = p
print("   max|L1(alpha=1) - Euler| =", np.abs(v_l1 - v_eu).max())

print()
print("2) subthreshold relaxation to a constant input (tau=4, I=1, no spikes)")
for a in (1.0, 0.8, 0.6):
    v, _ = l1_flif(np.ones(60), 4., a, spiking=False)
    print(f"   alpha={a}: V[0,1,2,5,10,30,59] = {v[[0,1,2,5,10,30,59]].round(4)}")

print()
print("3) constant drive with spiking: inter-spike intervals")
for a in (1.0, 0.8, 0.6):
    for ref in (0, 1):
        v, s = l1_flif(np.full(200, 1.6), 4., a, refractory=ref)
        idx = np.flatnonzero(s); isi = np.diff(idx)
        print(f"   alpha={a} refractory={ref}: spikes={len(idx):3d} first ISIs={isi[:6]} last ISIs={isi[-3:] if len(isi)>3 else isi}")

print()
print("4) does the reset increment belong in the memory trace? (alpha=0.6, refractory=1)")
for use in (True, False):
    v, s = l1_flif(np.full(200, 1.6), 4., 0.6, refractory=1, trace_uses_reset=use)
    idx = np.flatnonzero(s); isi = np.diff(idx)
    print(f"   trace_uses_reset={use!s:5}: spikes={len(idx):3d} ISIs={isi[:8]}")
