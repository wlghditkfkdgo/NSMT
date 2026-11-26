"""Rough energy of the proposed fusion (2026-11-26, exp/f-lif-pop-v3; user proposal "membrane OT / correlation mask on a spiking
attention between F and R", review only, no training). Per test window (one [1, L, C] input, C = 7 ETT variables), single block,
first-chain shapes (N, D per dataset; T 4), compared with the first chain's logged syops energy (judge.json, syn_energy_uJ):
  mask (Pearson over D between H_f(n) and H_r(n'), real x real)      T * C * N^2 * D  MAC (4.6 pJ)
  mask from the T-mean membrane (one N x N mask shared by the T copies)  C * N^2 * D  MAC
  spike map F R^T and map x V with V binary, upper bound (every element)   2 * T * C * N^2 * D  AC (0.9 pJ)
Not counted: centring / norms / division of the correlation, the elementwise mask product (T * C * N^2), a re-spiking neuron,
Q / K / V projections if added, and any Sinkhorn iterations (OT). ETTm1's first chain has 3 blocks; this is one block.
usage (from cinc_mixer_v1/): python -B results/check-attnmask-20261126/est_energy.py
"""
import json
import os

C_ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cells = json.load(open(os.path.join(C_, 'results', 'flif-v1-ett-spkmux-retune', 'judge.json')))['models']['first']['cells']
shape = {'ETTh1': (24, 32), 'ETTh2': (12, 64), 'ETTm1': (12, 32), 'ETTm2': (12, 64)}   # first chain (N, D)
T, C = 4, 7
print('cell        now uJ | mask MAC uJ (+%) | T-mean mask MAC (+%) | spike products ub uJ | mask + products ub (+%)')
for ds, (N, D) in shape.items():
    mac = T * C * N * N * D * 4.6e-6
    mac_t = C * N * N * D * 4.6e-6
    ac = 2 * T * C * N * N * D * 0.9e-6
    for p in ('96', '192', '336', '720'):
        e = cells[f'{ds} {p}']['syn_energy_uJ']
        print(f'{ds} H{p:<4s} {e:7.2f} | {mac:5.2f} (+{100 * mac / e:4.1f} %) | {mac_t:5.2f} (+{100 * mac_t / e:4.1f} %) | '
              f'{ac:5.2f} | +{100 * (mac + ac) / e:5.1f} %')
