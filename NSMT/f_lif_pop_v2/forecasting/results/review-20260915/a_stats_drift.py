"""Read-only review checks on completed v2 patch suites: paired statistics, diagnostics,
naive baselines, and how far retrieval parameters moved from their initial values."""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting'
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)

COMPARISONS = [
    ('heterogeneous_dense', 'heterogeneous_no_memory'),
    ('heterogeneous_sparse', 'heterogeneous_no_memory'),
    ('heterogeneous_sparse', 'heterogeneous_dense'),
    ('homogeneous_dense', 'homogeneous_no_memory'),
    ('homogeneous_sparse', 'homogeneous_no_memory'),
    ('homogeneous_sparse', 'homogeneous_dense'),
    ('heterogeneous_no_memory', 'homogeneous_no_memory'),
]

for H in [96, 720]:
    df = pd.read_csv(f'{ROOT}/results/selective-v2-20260914_patch_p{H}/per_run.csv')
    piv = df.pivot_table(index=['data', 'seed'], columns='variant', values='mse')
    rows = []
    for a, b in COMPARISONS:
        d6 = piv[a] - piv[b]
        rel6 = 100 * d6 / piv[b]
        m3 = piv[a].groupby(level='seed').mean() - piv[b].groupby(level='seed').mean()
        p6 = stats.ttest_1samp(d6, 0).pvalue
        p3 = stats.ttest_1samp(m3, 0).pvalue
        # paired-sample size for 80% power / two-sided 0.05 at a 1% relative effect, using n=6 SD
        delta = 0.01 * piv[b].mean()
        n_needed = ((stats.norm.ppf(.975) + stats.norm.ppf(.8)) * d6.std(ddof=1) / delta) ** 2
        rows.append(dict(comparison=f'{a} - {b}', mean_d=d6.mean(), rel_pct=rel6.mean(),
                         sd_d6=d6.std(ddof=1), p_n6=p6, a_better=f'{int((d6 < 0).sum())}/6',
                         ETTh1_rel=rel6.xs('ETTh1').mean(), ETTh2_rel=rel6.xs('ETTh2').mean(),
                         macro_d=m3.mean(), p_macro_n3=p3, pairs_for_1pct=n_needed))
    print(f'\n===== patch H{H}: paired MSE comparisons (a - b; negative = a better) =====')
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    sd = df.groupby(['data', 'variant']).mse.std(ddof=1).groupby('data').mean()
    print('mean seed SD of MSE per dataset:', sd.round(4).to_dict())

    diag = df.groupby('variant')[['best_epoch', 'epochs', 'support_density', 'empty_read_fraction',
                                  'mean_real_mass', 'mean_gate_after_first_patch',
                                  'evidence_to_charge_abs_ratio', 'off_minus_full_mse',
                                  'uniform_minus_full_mse', 'recent_minus_full_mse']].mean()
    print(f'\n----- patch H{H}: diagnostics mean per variant (first 8 test windows) -----')
    print(diag.to_string(float_format=lambda v: f'{v:.4g}'))
    best_epoch_by_data = df.groupby(['data', 'variant']).best_epoch.mean().unstack(0)
    print('\nmean best epoch (0-based):')
    print(best_epoch_by_data.to_string(float_format=lambda v: f'{v:.1f}'))

    base = {}
    for data in ['ETTh1', 'ETTh2']:
        run = json.load(open(glob.glob(f'{ROOT}/results/selective-v2-20260914_patch_p{H}/patch_{data}_*seed7.json')[0]))
        base[data] = {k: round(v['mse'], 4) for k, v in run['test']['baselines'].items()}
    print('naive baselines (test MSE):', base)

    drift = []
    for _, r in df[df.policy != 'off'].iterrows():
        state = torch.load(os.path.join(r.log_path, 'model_state/best+model.pt'), map_location='cpu')
        eye = torch.eye(4)
        q, k = state['embedding.lif.query.weight'], state['embedding.lif.key.weight']
        gw, gb = state['embedding.lif.gate.weight'], state['embedding.lif.gate.bias']
        drift.append(dict(variant=r.variant, data=r.data, seed=r.seed, best_epoch=r.best_epoch,
                          q_minus_I=(q - eye).norm().item(), k_minus_I=(k - eye).norm().item(),
                          q_minus_k=(q - k).norm().item(), gate_w_norm=gw.norm().item(),
                          gate_b=gb.item(), null_logit=state['embedding.lif.null_logit'].item(),
                          q_diag=np.round(q.diag().numpy(), 3).tolist(), k_diag=np.round(k.diag().numpy(), 3).tolist(),
                          gate_w=np.round(gw.flatten().numpy(), 3).tolist()))
    drift = pd.DataFrame(drift)
    print(f'\n----- patch H{H}: retrieval parameter drift from init (init: Q=K=I, gate w=b=0, null=-1; ||I||_F=2) -----')
    print(drift.groupby('variant')[['best_epoch', 'q_minus_I', 'k_minus_I', 'q_minus_k', 'gate_w_norm', 'gate_b', 'null_logit']]
          .agg(['mean', 'max']).to_string(float_format=lambda v: f'{v:.3f}'))
    print(drift.groupby(['variant', 'data'])[['q_minus_I', 'gate_w_norm', 'null_logit']].mean().to_string(float_format=lambda v: f'{v:.3f}'))
    for variant in ['heterogeneous_sparse', 'heterogeneous_dense']:
        sub = drift[drift.variant == variant]
        print(variant, 'Q diag mean (tau 2,4,8,16):', np.round(np.mean(sub.q_diag.tolist(), 0), 3).tolist(),
              'K diag mean:', np.round(np.mean(sub.k_diag.tolist(), 0), 3).tolist())
        print(variant, 'gate weight mean [charge x4, memory x4, margin, real_mass]:',
              np.round(np.mean(sub.gate_w.tolist(), 0), 3).tolist())
