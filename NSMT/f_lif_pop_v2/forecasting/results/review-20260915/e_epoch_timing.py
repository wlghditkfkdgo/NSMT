"""Read-only: mean training+validation seconds per epoch by suite and variant, plus the v2 smoke runs."""
import glob
import json
import statistics
from collections import defaultdict

NSMT = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT'
V2 = NSMT + '/f_lif_pop_v2/forecasting/results/'

for label, pattern in [('v2 patch H96 (2 jobs/GPU)', V2 + 'selective-v2-20260914_patch_p96/patch_*.json'),
                       ('v2 patch H720 (2 jobs/GPU)', V2 + 'selective-v2-20260914_patch_p720/patch_*.json'),
                       ('v2 tcn H96 (2 jobs/GPU)', V2 + 'selective-v2-20260914_tcn_p96/tcn_*.json'),
                       ('v1 tcn H96 (1 job/GPU)', NSMT + '/f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETT*.json'),
                       ('v1 patch H96 (1 job/GPU)', NSMT + '/f_lif_pop_v1/forecasting/results/ett-first-20260914/ETT*flatten*.json')]:
    seconds = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        history = json.load(open(path)).get('history', [])
        if not history:
            continue
        name = path.split('/')[-1]
        policy = next((p for p in ['sparse', 'dense', 'retrieval'] if p in name), 'off')
        seconds[('hetero_' if 'heterogeneous' in name else 'homo_') + policy].append(statistics.mean(e['seconds'] for e in history))
    print(label, {k: (round(statistics.mean(v), 1), len(v)) for k, v in sorted(seconds.items())})

for path in sorted(glob.glob(V2 + 'smoke-v2-20260914-r2/*.json')):
    run = json.load(open(path))
    if 'history' in run:
        print(run['run_id'], 'train_batches', run['config']['max_train_batches'], 'eval_batches', run['config']['max_eval_batches'],
              'epoch_seconds', [round(e['seconds'], 1) for e in run['history']])
