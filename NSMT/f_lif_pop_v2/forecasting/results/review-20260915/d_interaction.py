"""Read-only: heterogeneity x retrieval interaction on completed v2 patch suites.

Contrast = (hetero read - hetero off) - (homo read - homo off); negative supports the
hypothesis that a heterogeneous population key makes retrieval more useful."""
import pandas as pd
from scipy import stats

ROOT = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting'

for H in [96, 720]:
    df = pd.read_csv(f'{ROOT}/results/selective-v2-20260914_patch_p{H}/per_run.csv')
    p = df.pivot_table(index=['data', 'seed'], columns='variant', values='mse')
    for read in ['dense', 'sparse']:
        d = (p[f'heterogeneous_{read}'] - p['heterogeneous_no_memory']) - (p[f'homogeneous_{read}'] - p['homogeneous_no_memory'])
        m3 = d.groupby(level='seed').mean()
        print(f'H{H} {read:6s} interaction mean {d.mean():+.5f}  sd(n=6) {d.std(ddof=1):.5f}  '
              f'p(n=6) {stats.ttest_1samp(d, 0).pvalue:.3f}  supports-synergy {int((d < 0).sum())}/6  '
              f'ETTh1 {d.xs("ETTh1").mean():+.5f}  ETTh2 {d.xs("ETTh2").mean():+.5f}  '
              f'p(seed-macro n=3) {stats.ttest_1samp(m3, 0).pvalue:.3f}')
