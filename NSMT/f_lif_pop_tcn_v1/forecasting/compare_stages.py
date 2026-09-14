"""Compare completed stage1/stage2 suites at each horizon without mixing tasks."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import TASK, NSMT
from utils import write_json


def compare(output):
    root = TASK / 'results' / output
    root.mkdir(parents=True, exist_ok=True)
    suites = {96: ('ett-first-20260914', 'ett-tcn-h96-20260914'),
              720: ('ett-first-h720-20260914', 'ett-tcn-h720-20260914')}
    directories = [NSMT / 'f_lif_pop_v1/forecasting/results', TASK / 'results']
    raw, frames, layer_rows = {}, [], []
    for horizon, names in suites.items():
        for architecture, directory, suite in zip(['patch_snn', 'spike_tcn'], directories, names):
            folder = directory / suite
            assert json.loads((folder / 'check_summary.json').read_text())['status'] == 'passed'
            frame = pd.read_csv(folder / 'per_run.csv')
            frame = frame[frame['head'] == 'flatten'].copy()
            assert len(frame) == 24
            frame['architecture'], frame['horizon'] = architecture, horizon
            frames.append(frame)
            for run_id in frame.run_id:
                run = json.loads((folder / (run_id + '.json')).read_text())
                c = run['config']
                raw[(architecture, horizon, c['data'], c['seed'], c['variant'])] = run
                if architecture == 'spike_tcn':
                    for layer, diag in run['test']['diagnostics']['layers'].items():
                        layer_rows.append({'horizon': horizon, 'data': c['data'], 'seed': c['seed'],
                                           'variant': c['variant'], 'layer': layer,
                                           'spike_rate': np.mean(diag['spike_rate_per_constituent']),
                                           'membrane_diversity': diag['population_membrane_std'],
                                           'evidence_ratio': diag['evidence_to_charge_abs_ratio'],
                                           'memory_entropy': diag['mean_normalized_entropy']})
    # Verify actual run provenance, not just matching table labels.
    variants = ['homogeneous_no_memory', 'homogeneous_retrieval',
                'heterogeneous_no_memory', 'heterogeneous_retrieval']
    matched = 0
    for horizon, data, seed, variant in product([96, 720], ['ETTh1', 'ETTh2'], [7, 13, 21], variants):
        a, b = [raw[(arch, horizon, data, seed, variant)] for arch in ['patch_snn', 'spike_tcn']]
        assert a['data'] == b['data']
        for key in ['seq_len', 'pred_len', 'patch_size', 'embed_dim', 'num_population',
                    'head_dim', 'head_mode', 'seed', 'heterogeneous', 'retrieval',
                    'lr', 'weight_decay', 'epoch', 'patience', 'batch_size', 'tau_min',
                    'tau_max', 'threshold', 'memory_strength', 'temperature', 'input_scale',
                    'max_train_batches', 'max_eval_batches', 'num_workers']:
            assert a['config'][key] == b['config'][key], key
        assert a['protocol'] == b['protocol']
        assert a['test']['elements'] == b['test']['elements']
        for baseline in ['persistence', 'window_mean']:
            for metric in ['mse', 'mae']:
                np.testing.assert_allclose(a['test']['baselines'][baseline][metric],
                                           b['test']['baselines'][baseline][metric], rtol=0, atol=1e-12)
        matched += 1
    runs = pd.concat(frames, ignore_index=True)
    runs.to_csv(root / 'per_run.csv', index=False)
    by_seed = runs.groupby(['architecture', 'horizon', 'variant', 'seed'])[['mse', 'mae']].mean().reset_index()
    macro = by_seed.groupby(['architecture', 'horizon', 'variant']).agg(
        mse=('mse', 'mean'), mse_sd=('mse', 'std'), mae=('mae', 'mean'), mae_sd=('mae', 'std')).reset_index()
    macro.to_csv(root / 'macro.csv', index=False)
    tasks = runs.groupby(['architecture', 'horizon', 'data', 'variant']).agg(
        mse=('mse', 'mean'), mse_sd=('mse', 'std'), mae=('mae', 'mean'), mae_sd=('mae', 'std')).reset_index()
    tasks.to_csv(root / 'per_task.csv', index=False)
    paired = []
    for (architecture, horizon, heterogeneous), group in runs.groupby(['architecture', 'horizon', 'heterogeneous']):
        for seed, frame in group.groupby('seed'):
            on = frame[frame.retrieval].set_index('data')
            off = frame[~frame.retrieval].set_index('data')
            paired.append({'architecture': architecture, 'horizon': horizon,
                           'heterogeneous': bool(heterogeneous), 'seed': seed,
                           'delta_mse': (on.mse - off.mse).mean(),
                           'delta_mae': (on.mae - off.mae).mean()})
    pd.DataFrame(paired).to_csv(root / 'paired_macro_by_seed.csv', index=False)
    pd.DataFrame(layer_rows).to_csv(root / 'tcn_layer_diagnostics.csv', index=False)
    selection = []
    for (arch, horizon, data, seed, variant), run in raw.items():
        history = run['history']
        selection.append({'architecture': arch, 'horizon': horizon, 'data': data, 'seed': seed,
                          'variant': variant, 'best_epoch_one_based': run['best_epoch'] + 1,
                          'trained_epochs': len(history), 'capped': len(history) == run['config']['epoch'],
                          'trained_with_reduced_lr': any(h['lr'] < run['config']['lr'] for h in history),
                          'best_val_mse': run['best_val_mse'], 'last_val_mse': history[-1]['val']['mse']})
    pd.DataFrame(selection).to_csv(root / 'training_selection.csv', index=False)
    lines = ['# 1차 patch SNN / 2차 causal population Spike-TCN', '',
             'ETTh1/ETTh2 각각의 전체 test MSE/MAE. 데이터셋 평균을 seed별 계산한 뒤3seed mean±sample SD.',
             '각 architecture/horizon은24 runs. Prediction length가 다른 결과는 합산하지 않는다.', '',
             '| Architecture | Horizon | Variant | MSE ± SD | MAE ± SD |', '|---|---:|---|---:|---:|']
    for _, row in macro.iterrows():
        lines.append(f'| {row.architecture} | {row.horizon} | {row.variant} | '
                     f'{row.mse:.6f} ± {row.mse_sd:.6f} | {row.mae:.6f} ± {row.mae_sd:.6f} |')
    lines += ['', '## 이질적 population에서 검색 추가', '',
              '| Architecture | Horizon | no-memory MSE | retrieval MSE | Relative change |',
              '|---|---:|---:|---:|---:|']
    for (arch, horizon), frame in macro.groupby(['architecture', 'horizon']):
        frame = frame.set_index('variant')
        off, on = frame.loc['heterogeneous_no_memory', 'mse'], frame.loc['heterogeneous_retrieval', 'mse']
        lines.append(f'| {arch} | {horizon} | {off:.6f} | {on:.6f} | {100 * (on / off - 1):+.3f}% |')
    lines += ['', '## 비교 조건과 제한', '',
              '- 48쌍의 실제 run에서 데이터 hash/전처리/splits, 입력/출력/patch/폭/head/seed, optimizer budget 및 뉴런 설정 일치를 검사했다.',
              '- 2차에는 두 convolution/population layer와 current residual이 추가된다. Parameter는24714개 더 많으며 초기 backbone weights 및 effective capacity는 architecture 간 같지 않다.',
              '- 각 architecture 내부의 네 조건은 같은 nominal parameters/초기값을 사용한다. Retrieval off의 Q/K/gate는 미사용이다.',
              '- 모든 최종 모델은 최소 validation MSE로 선택했다. Maximum10/early-stop3은 수렴 보장이 없으며 LR 감소 뒤 회복 시간이 짧을 수 있다.',
              '- Test off/uniform/recent 개입은 재학습 대조가 아니다. 이를 사용한 checkpoint/hyperparameter 재선택은 하지 않았다.',
              '- Layer diagnostics는 첫 test8windows뿐이다. 더 깊어진 backbone의 효과와 기억 선택의 효과를 정확도 하나로 동일시하지 않는다.',
              '- 2차는 chronological persistent state/current residual을 사용하는 Spike-TCN 변형이다. 논문 원본 재현, matched-capacity sweep, longer-budget convergence, synthetic recall, 에너지 측정: not run.',
              '- 각 horizon의 원본 REPORT/CSV/run JSON/audits와 canonical docs/PROJECT_LOG.md를 함께 읽는다.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, horizon in enumerate([96, 720]):
        for col, data in enumerate(['ETTh1', 'ETTh2']):
            ax = axes[row, col]
            frame = tasks[(tasks.horizon == horizon) & (tasks.data == data)].set_index(['architecture', 'variant'])
            keys = list(product(['patch_snn', 'spike_tcn'], ['heterogeneous_no_memory', 'heterogeneous_retrieval']))
            values = frame.loc[keys]
            ax.bar(range(4), values.mse, yerr=values.mse_sd, capsize=4,
                   color=['#9dbac8', '#4d819a', '#86b5a4', '#236650'])
            ax.set_xticks(range(4), ['Patch\nno memory', 'Patch\nretrieval', 'TCN\nno memory', 'TCN\nretrieval'])
            ax.set_title(f'{data} / horizon {horizon}')
            ax.set_ylabel('Test MSE (train-standardized)')
    fig.suptitle('Heterogeneous population: mean ± sample SD over 3 seeds')
    for ext in ['png', 'pdf']:
        fig.savefig(root / ('comparison.' + ext), dpi=180)
    plt.close(fig)
    write_json(root / 'check_comparison.json', {'status': 'passed', 'matched_run_pairs': matched,
               'runs': len(runs), 'per_horizon_not_pooled': True,
               'source_suites': {str(k): list(v) for k, v in suites.items()},
               'matched_capacity': False})
    print(macro.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='stage-comparison-20260914')
    compare(parser.parse_args().output)
