"""Post hoc report of stage 4 (2026-11-27, exp/f-lif-pop-v3; written before the results, NOT part of the pre-registered verdicts).
  Q1 the learned alpha (best-validation weights, model_state/checkpoint.pth) per arm, cell and seed
  Q2 each new arm against the IAND reference and the multiplicative arm per cell (seed-mean MSE change, %)
usage (from cinc_mixer_v1/): python -B results/check-sxa4-20261127/post_s4.py
"""
import json
import os

import torch

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = json.load(open(os.path.join(C, 'results', 'flif-v1-ett-sxa-s4', 'summary.json')))
rows = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(os.path.join(C, 'results', 'flif-v1-ett-sxa-s4', 'raw.txt')) if l.startswith('RESULT')]
print('Q1 learned alpha (best-validation weights)')
for cond in ('spkmx_lif_msxdpca_iand', 'spkmx_lif_msxdota_iand'):
    for ds, pl in (('ETTh1', '96'), ('ETTh1', '720'), ('ETTm1', '96'), ('ETTm1', '720')):
        al = []
        for r in rows:
            if r['cond'] == cond and r['ds'] == ds and r['pl'] == pl:
                sd = torch.load(os.path.join(r['dir'], 'model_state', 'checkpoint.pth'), map_location='cpu')
                al.append((r['seed'], float(sd['inner.blocks.0.xattn.core.alpha']), r['best_epoch_val_epochs']))
        print(f'  {cond[10:-5]:8s} {ds} H{pl:<4s} ' + '  '.join(f'seed {s}: {a:+.4f} (best:val:run {b})' for s, a, b in sorted(al)))
print('Q2 per-cell seed-mean MSE change (%), ETTh1 96, 720, ETTm1 96, 720')
for c in S['comparisons']:
    print(f'  {c["A"]} vs {c["B"]}: ' + ' '.join(f'{x:+.2f}' for x in c['mse']['r_cells']) + f'  (R {c["mse"]["R_pct"]:+.2f})')
