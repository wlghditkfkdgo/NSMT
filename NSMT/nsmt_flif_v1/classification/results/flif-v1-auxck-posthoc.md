# Aux-target memory check — POST HOC reading (not pre-registered; report only)

MODI 2026-11-20 POST HOC (not pre-registered) reading of the aux-target memory check (aux_check.py): written after the
pre-registered summary showed that the no-memory control reconstructs its OWN token better than the memory model in all 6 cells,
so the own-token calibration g(0) of the pre-registered rule is negative and ADDS to d. This script reports, per task / neuron /
target, what the verdict rests on without that calibration:
  raw gap         e(control, s) - e(memory, s)                     (no own-token calibration)
  g(0)            e(control, 0) - e(memory, 0)                     (the calibration term of the pre-registered rule)
  linear margin   e_lin(token i alone, s) - e(memory, s)           (how much the memory model beats a ridge fit on token i alone)
  control - lin   e(no memory, s) - e_lin(token i alone, s)        (does the no-memory control reach the linear no-memory bound?)
  rho             Neocortex firing rate of the no-memory run at s and at s = 0
ETT: per cell (ETTh2 / ETTm2) and the mean. Report only: the pre-registered verdict (results/flif-v1-auxck.md) is not changed.
usage (classification folder): python aux_check_posthoc.py   -> results/flif-v1-auxck-posthoc.md

| task | neuron | target | raw gap, no memory | raw gap, shuffled | g(0), no memory | linear margin of memory | no memory − linear | rho no memory (s / s = 0) |
|---|---|---|---|---|---|---|---|---|
| ETT | pop LIF | past, 2 back (s = 2) | +0.237 / +0.258 → +0.247 | +0.183 / +0.386 → +0.284 | -0.090 / -0.192 → -0.141 | +0.093 / +0.285 → +0.189 | +0.144 / -0.027 → +0.058 | 0.00101 / 0.01830 ; 0.04311 / 0.07497 |
| ETT | pop LIF | past, 4 back (s = 4) | +0.142 / +0.468 → +0.305 | +0.111 / +0.523 → +0.317 | -0.090 / -0.192 → -0.141 | +0.096 / +0.508 → +0.302 | +0.046 / -0.040 → +0.003 | 0.00026 / 0.01830 ; 0.02167 / 0.07497 |
| ETT | pop LIF | next patch (s = -2) | +0.131 / +0.059 → +0.095 | +0.074 / +0.195 → +0.134 | -0.090 / -0.192 → -0.141 | -0.016 / +0.089 → +0.037 | +0.147 / -0.030 → +0.059 | 0.00075 / 0.01830 ; 0.04393 / 0.07497 |
| ETT | pop f-LIF | past, 2 back (s = 2) | +0.148 / +0.252 → +0.200 | +0.146 / +0.251 → +0.198 | -0.273 / -0.072 → -0.173 | +0.144 / +0.281 → +0.213 | +0.003 / -0.030 → -0.013 | 0.03295 / 0.05602 ; 0.06419 / 0.08542 |
| ETT | pop f-LIF | past, 4 back (s = 4) | +0.069 / +0.423 → +0.246 | +0.035 / +0.412 → +0.224 | -0.273 / -0.072 → -0.173 | +0.045 / +0.471 → +0.258 | +0.024 / -0.048 → -0.012 | 0.01117 / 0.05602 ; 0.04508 / 0.08542 |
| ETT | pop f-LIF | next patch (s = -2) | +0.013 / +0.065 → +0.039 | +0.017 / +0.078 → +0.048 | -0.273 / -0.072 → -0.173 | +0.011 / +0.102 → +0.056 | +0.002 / -0.037 → -0.017 | 0.03491 / 0.05602 ; 0.06583 / 0.08542 |
| MIT-BIH | pop LIF | past, 2 back (s = 2) | +0.239 | +0.365 | -0.239 | +0.347 | -0.107 | 0.01832 / 0.02126 |
| MIT-BIH | pop LIF | past, 4 back (s = 4) | +0.471 | +0.369 | -0.239 | +0.592 | -0.121 | 0.02378 / 0.02126 |
| MIT-BIH | pop LIF | next patch (s = -2) | -0.125 | +0.278 | -0.239 | +0.068 | -0.193 | 0.01562 / 0.02126 |
| MIT-BIH | pop f-LIF | past, 2 back (s = 2) | +0.266 | +0.354 | -0.104 | +0.386 | -0.120 | 0.03435 / 0.03206 |
| MIT-BIH | pop f-LIF | past, 4 back (s = 4) | +0.448 | +0.360 | -0.104 | +0.578 | -0.131 | 0.02704 / 0.03206 |
| MIT-BIH | pop f-LIF | next patch (s = -2) | -0.058 | +0.051 | -0.104 | +0.149 | -0.208 | 0.02896 / 0.03206 |
