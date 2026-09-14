"""Audit every run artifact and independently reconstruct paired statistics."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import TASK
from utils import write_json, sha256


def check(suite):
    root = TASK / 'results' / suite
    done = json.loads((root / 'completion.json').read_text())
    ids = [j['id'] for j in done['jobs']]
    assert len(ids) == len(set(ids)) == 32
    results = [json.loads((root / (name + '.json')).read_text()) for name in ids]
    horizons = {r['config']['pred_len'] for r in results}
    assert len(horizons) == 1
    pred_len = horizons.pop()
    assert pred_len in [96, 720] and done.get('pred_len', pred_len) == pred_len
    expected = {(head, data, seed, h, r) for head in ['flatten', 'last']
                for data, seed, h, r in product(['ETTh1', 'ETTh2'], [7, 13, 21] if head == 'flatten' else [7], [False, True], [False, True])}
    actual = {(r['config']['head_mode'], r['config']['data'], r['config']['seed'],
               r['config']['heterogeneous'], r['config']['retrieval']) for r in results}
    assert actual == expected
    source_sets = {json.dumps(r['source_sha256'], sort_keys=True) for r in results}
    assert len(source_sets) == 1 and len({r['git_commit'] for r in results}) == 1
    for filename, digest in results[0]['source_sha256'].items():
        assert sha256(TASK / filename) == digest, filename
    initial_hashes = {}
    parameter_counts = {}
    data_cache = {}
    tensor = np.empty((2, 3, 2, 2, 2))  # data,seed,heterogeneity,retrieval,metric (main only)
    metric_lookup = {}
    for r in results:
        c, history = r['config'], r['history']
        assert not r['protocol']['smoke_only'] and c['max_train_batches'] == c['max_eval_batches'] == 0
        assert c['seq_len'] == 336 and c['pred_len'] == pred_len and c['num_population'] == 4
        key = (c['head_mode'], c['seed'])
        initial_hashes.setdefault(key, set()).add(r['initial_parameter_sha256'])
        parameter_counts.setdefault(c['head_mode'], set()).add(r['parameters'])
        losses = [h['val']['mse'] for h in history]
        assert r['best_epoch'] == int(np.argmin(losses))
        assert r['best_val_mse'] == min(losses)
        assert abs(r['restored_val']['mse'] - min(losses)) < 1e-7
        assert r['test']['elements'] == (2880 - pred_len + 1) * pred_len * 7
        p = Path(c['save_result_path'])
        epoch = pd.read_csv(p / 'log/best_log_0.csv')
        assert len(epoch) == len(history)
        final = pd.read_csv(p / 'log/final+result.csv').iloc[0]
        for metric in ['loss', 'mse', 'mae']:
            assert abs(final[metric] - r['test'][metric]) < 5.01e-7
            for split in ['train', 'val']:
                expected_values = np.array([h[split][metric] for h in history])
                np.testing.assert_allclose(epoch[split + '_' + metric], expected_values, rtol=0, atol=5.01e-7)
                events = EventAccumulator(str(p / 'log' / (split + '_0'))).Reload().Scalars(split + '_0/' + metric)
                assert [e.step for e in events] == list(range(len(history)))
                np.testing.assert_allclose([e.value for e in events], expected_values, rtol=2e-7, atol=2e-7)
        horizon = pd.read_csv(p / 'log/horizon_metrics.csv')
        for metric in ['mse', 'mae']:
            np.testing.assert_allclose(horizon[metric].mean(), r['test'][metric], rtol=1e-10, atol=1e-10)
        saved = torch.load(p / 'model_state/config.pt', map_location='cpu')
        assert saved['best_epoch'] == r['best_epoch'] and saved['variant'] == c['variant'] and saved['seed'] == c['seed']
        state = torch.load(p / 'model_state/best+model.pt', map_location='cpu')
        assert all(torch.isfinite(v).all() for v in state.values())
        assert sum(v.numel() for name, v in state.items() if name != 'embedding.lif.beta') == r['parameters']
        assert sha256(p / 'model_state/best+model.pt') == r['checkpoint_sha256']
        assert (p / 'logargs.txt').is_file()
        if c['data'] not in data_cache:
            data = r['data']
            assert sha256(data['path']) == data['sha256']
            raw = pd.read_csv(data['path'])[data['columns']].to_numpy(dtype=np.float64)
            mean, scale = raw[:8640].mean(0), raw[:8640].std(0)
            np.testing.assert_allclose(mean, data['scaler_mean'], atol=1e-12)
            np.testing.assert_allclose(scale, data['scaler_scale'], atol=1e-12)
            data_cache[c['data']] = ((raw - mean) / scale).astype(np.float32)
        sample = json.loads((p / 'log/forecast_example.json').read_text())
        np.testing.assert_array_equal(np.array(sample['target'], dtype=np.float32), data_cache[c['data']][11520:11520 + pred_len])
        diag = r['test']['diagnostics']
        assert all(0 <= x <= 1 for x in diag['spike_rate_per_constituent'])
        if c['retrieval']:
            assert abs(sum(diag['lag_mass']) - 1.) < 1e-5
            assert set(r['test']['interventions']) == {'off', 'uniform', 'recent'}
            for value in r['test']['interventions'].values():
                assert value['elements'] == r['test']['elements']
        else:
            assert diag['mean_abs_evidence'] == 0 and not r['test']['interventions']
        if not c['heterogeneous']:
            assert diag['population_membrane_std'] < 1e-6
        values = [r['test']['mse'], r['test']['mae']]
        metric_lookup[(c['head_mode'], c['data'], c['seed'], c['heterogeneous'], c['retrieval'])] = np.array(values)
        if c['head_mode'] == 'flatten':
            tensor[['ETTh1', 'ETTh2'].index(c['data']), [7, 13, 21].index(c['seed']), int(c['heterogeneous']), int(c['retrieval'])] = values
    assert all(len(v) == 1 for v in initial_hashes.values())
    assert all(len(v) == 1 for v in parameter_counts.values())
    # Recalculate all tables from an independent NumPy tensor / raw run lookup.
    macro = pd.read_csv(root / 'macro.csv')
    for h, r in product([False, True], repeat=2):
        name = ('heterogeneous' if h else 'homogeneous') + ('_retrieval' if r else '_no_memory')
        row = macro[(macro['head'] == 'flatten') & (macro.variant == name)].iloc[0]
        by_seed = tensor[:, :, int(h), int(r)].mean(0)
        np.testing.assert_allclose([row.mse, row.mae], by_seed.mean(0), rtol=1e-12)
        np.testing.assert_allclose([row.mse_sd, row.mae_sd], by_seed.std(0, ddof=1), rtol=1e-12, atol=1e-14)
    for _, row in pd.read_csv(root / 'per_task.csv').iterrows():
        seeds = [7, 13, 21] if row['head'] == 'flatten' else [7]
        values = np.array([metric_lookup[(row['head'], row['data'], s, row.heterogeneous, row.retrieval)] for s in seeds])
        np.testing.assert_allclose([row.mse, row.mae], values.mean(0), rtol=1e-12)
        if len(seeds) > 1:
            np.testing.assert_allclose([row.mse_sd, row.mae_sd], values.std(0, ddof=1), rtol=1e-12, atol=1e-14)
    for _, row in pd.read_csv(root / 'paired.csv').iterrows():
        key = (row['head'], row['data'], row.seed, row.heterogeneous)
        on, off = metric_lookup[key + (True,)], metric_lookup[key + (False,)]
        np.testing.assert_allclose([row.delta_mse, row.delta_mae], on - off, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(row.relative_mse_pct, 100 * (on[0] / off[0] - 1), rtol=1e-10, atol=1e-12)
    for _, row in pd.read_csv(root / 'paired_macro_by_seed.csv').iterrows():
        deltas = []
        relative = []
        for data in ['ETTh1', 'ETTh2']:
            key = (row['head'], data, row.seed, row.heterogeneous)
            on, off = metric_lookup[key + (True,)], metric_lookup[key + (False,)]
            deltas.append(on - off)
            relative.append(100 * (on[0] / off[0] - 1))
        np.testing.assert_allclose([row.delta_mse, row.delta_mae], np.mean(deltas, 0), atol=1e-14, rtol=1e-10)
        np.testing.assert_allclose(row.relative_mse_pct, np.mean(relative), atol=1e-12, rtol=1e-10)
    audit = {'status': 'passed', 'runs': len(results), 'training_commit': results[0]['git_commit'],
             'source_identical_and_current': True, 'paired_initial_parameters_identical': True,
             'parameter_counts': {k: list(v)[0] for k, v in parameter_counts.items()},
             'all_run_checks': ['complete matrix', 'best validation selection/restoration',
                               'full test element count', 'CSV/history/TensorBoard', 'horizon average',
                               'config/checkpoint/hash/finite weights', 'train-only scaler',
                               'first target at canonical test boundary', 'population/memory diagnostics'],
             'independent_summary_checks': ['per-task mean/SD', 'dataset-first macro mean/SD',
                                            'paired retrieval on-off deltas', 'seed macro deltas']}
    write_json(root / 'check_summary.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='ett-first-20260914')
    check(parser.parse_args().suite)
