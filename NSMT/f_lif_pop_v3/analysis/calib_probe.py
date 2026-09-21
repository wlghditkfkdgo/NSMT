"""Where does the shared-soma population neuron actually fire? Init-time calibration probe."""
import math
import numpy as np
from prereg_numerics import branches, soma, TAUS   # reuse definitions

rng = np.random.default_rng(7)
X = rng.standard_normal((32, 42))          # 32 windows of standardised patch currents
print("soma_tau  w_k     scale | firing rate | mean max|v| | mean|u_k| (tau=2..16)")
for tau_s in (1.0, 2.0):
    for wname, w in (("1/K", np.full(4, .25)), ("1.0", np.ones(4))):
        for scale in (1., 2., 4., 8.):
            rates, vmax, umean = [], [], []
            for x in X:
                u, _ = branches(x * scale, TAUS, 0.7)
                v, s = soma(u, w, tau_s=tau_s)
                rates.append(s.mean()); vmax.append(v.max()); umean.append(np.abs(u).mean(0))
            print(f"{tau_s:>8} {wname:>5} {scale:>7} | {np.mean(rates):11.3f} | {np.mean(vmax):11.3f} | {np.mean(umean,0).round(3)}")
print()
print("Same, but alpha=1 branches (SOE-equivalent baseline), soma_tau=1, w=1.0")
for scale in (1., 2., 4.):
    rates = []
    for x in X:
        u, _ = branches(x * scale, TAUS, 1.0)
        v, s = soma(u, np.ones(4), tau_s=1.0)
        rates.append(s.mean())
    print(f"  scale={scale:<4} firing rate={np.mean(rates):.3f}")
