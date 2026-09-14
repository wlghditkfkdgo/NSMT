"""Paired first-experiment summary; primary 3 seeds and exploratory 1 seed separate."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import write_json


def summarize(suite):
    root = Path(__file__).resolve().parent / 'results' / suite
    completion = json.loads((root / 'completion.json').read_text())
    if len(completion['jobs']) != 32 or any(j['status'] != 'complete' for j in completion['jobs']):
        raise ValueError('Require all 32 completed runs')
    results = [json.loads((root / (j['id'] + '.json')).read_text()) for j in completion['jobs']]
    rows = []
    for r in results:
        c, t = r['config'], r['test']
        d = t['diagnostics']
        row = {'run_id': r['run_id'], 'data': c['data'], 'head': c['head_mode'], 'variant': c['variant'],
               'heterogeneous': c['heterogeneous'], 'retrieval': c['retrieval'], 'seed': c['seed'],
               'mse': t['mse'], 'mae': t['mae'], 'best_epoch': r['best_epoch'],
               'epochs': len(r['history']), 'parameters': r['parameters'], 'seconds': r['wall_seconds'],
               'initial_parameter_sha256': r['initial_parameter_sha256'],
               'log_path': c['save_result_path'], 'memory_gate': d['mean_gate_after_first_patch'],
               'memory_entropy': d['mean_normalized_entropy'], 'mean_lag_patches': d['mean_lag_in_patches'],
               'membrane_diversity': d['population_membrane_std'],
               'evidence_ratio': d['evidence_to_charge_abs_ratio'],
               'spike_rate': np.mean(d['spike_rate_per_constituent'])}
        for name, metric in t['baselines'].items():
            row[name + '_mse'] = metric['mse']
        for mode, metric in t['interventions'].items():
            row[mode + '_mse'] = metric['mse']
            row[mode + '_minus_full_mse'] = metric['mse'] - t['mse']
        rows.append(row)
    runs = pd.DataFrame(rows).sort_values(['head', 'data', 'variant', 'seed'])
    runs.to_csv(root / 'per_run.csv', index=False)
    group = ['head', 'data', 'variant', 'heterogeneous', 'retrieval']
    tasks = runs.groupby(group).agg(n=('seed', 'count'), mse=('mse', 'mean'), mse_sd=('mse', 'std'),
                                   mae=('mae', 'mean'), mae_sd=('mae', 'std')).reset_index()
    tasks.to_csv(root / 'per_task.csv', index=False)
    paired_rows = []
    for (head, data, hetero), frame in runs.groupby(['head', 'data', 'heterogeneous']):
        on = frame[frame.retrieval].set_index('seed')
        off = frame[~frame.retrieval].set_index('seed')
        for seed in sorted(on.index):
            paired_rows.append({'head': head, 'data': data, 'heterogeneous': bool(hetero), 'seed': int(seed),
                                'delta_mse': on.loc[seed, 'mse'] - off.loc[seed, 'mse'],
                                'delta_mae': on.loc[seed, 'mae'] - off.loc[seed, 'mae'],
                                'relative_mse_pct': 100 * (on.loc[seed, 'mse'] / off.loc[seed, 'mse'] - 1)})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(root / 'paired.csv', index=False)
    macro_seed = runs.groupby(['head', 'variant', 'seed'])[['mse', 'mae']].mean().reset_index()
    macro = macro_seed.groupby(['head', 'variant']).agg(n=('seed', 'count'), mse=('mse', 'mean'),
                   mse_sd=('mse', 'std'), mae=('mae', 'mean'), mae_sd=('mae', 'std')).reset_index()
    macro.to_csv(root / 'macro.csv', index=False)
    paired_macro = paired.groupby(['head', 'heterogeneous', 'seed'])[['delta_mse', 'delta_mae', 'relative_mse_pct']].mean().reset_index()
    paired_macro.to_csv(root / 'paired_macro_by_seed.csv', index=False)
    interaction = paired.pivot(index=['head', 'data', 'seed'], columns='heterogeneous', values='delta_mse')
    interaction['interaction'] = interaction[True] - interaction[False]
    interaction.reset_index().to_csv(root / 'interaction.csv', index=False)
    interventions = runs[runs.retrieval].groupby(['head', 'data', 'variant'])[
        ['off_minus_full_mse', 'uniform_minus_full_mse', 'recent_minus_full_mse',
         'memory_gate', 'memory_entropy', 'membrane_diversity', 'evidence_ratio']].mean().reset_index()
    interventions.to_csv(root / 'interventions.csv', index=False)
    def records(frame):
        return json.loads(frame.to_json(orient='records', double_precision=15))
    write_json(root / 'aggregate.json', {'runs': 32, 'macro': records(macro),
              'per_task': records(tasks), 'paired_macro_by_seed': records(paired_macro),
              'interventions': records(interventions), 'note': 'SD over seeds, ddof1; n1 SD null. Macro first averages datasets within seed.'})
    het_no = macro[(macro['head'] == 'flatten') & (macro.variant == 'heterogeneous_no_memory')].iloc[0]
    het_yes = macro[(macro['head'] == 'flatten') & (macro.variant == 'heterogeneous_retrieval')].iloc[0]
    changes = []
    for data in ['ETTh1', 'ETTh2']:
        frame = tasks[(tasks['head'] == 'flatten') & (tasks.data == data)].set_index('variant')
        changes.append(f'{data} ΔMSE={frame.loc["heterogeneous_retrieval", "mse"] - frame.loc["heterogeneous_no_memory", "mse"]:+.6f}')
    capped = int(((runs['head'] == 'flatten') & (runs.epochs == 10)).sum())
    lines = ['# f-LIF population: 1차 forecasting 결과', '',
             '입력336/예측96, patch8, D32/K4, train-only 표준화. 본 실험24개(3seed), 보조8개(1seed).',
             '모든 수치는 전체 test window/horizon/channel의 train-standardized MSE/MAE이다.', '',
             f'이질적 집단에 검색을 추가하면 macro MSE {het_no.mse:.6f} → {het_yes.mse:.6f} '
             f'({100 * (het_yes.mse / het_no.mse - 1):+.3f}%). ' + '; '.join(changes) + '.',
             f'본 실험24개 중 {capped}개가10epoch 상한에 도달했다. 아래 결과는 짧은 예산의 예비 검증이다.', '',
             '## 본 실험: 두 데이터셋 macro (seed별 평균 후 mean ± sample SD)', '',
             '| Variant | MSE | MAE |', '|---|---:|---:|']
    for _, row in macro[macro['head'] == 'flatten'].iterrows():
        lines.append(f'| {row.variant} | {row.mse:.6f} ± {row.mse_sd:.6f} | {row.mae:.6f} ± {row.mae_sd:.6f} |')
    lines += ['', '## 데이터셋별 결과 (flatten, 3seed)', '', '| Data | Variant | MSE ± SD | MAE ± SD |', '|---|---|---:|---:|']
    for _, row in tasks[tasks['head'] == 'flatten'].iterrows():
        lines.append(f'| {row.data} | {row.variant} | {row.mse:.6f} ± {row.mse_sd:.6f} | {row.mae:.6f} ± {row.mae_sd:.6f} |')
    lines += ['', '## 검색 효과: 같은 seed의 retrieval on − off', '',
              '음수는 검색 모델의 오차가 더 낮음을 뜻한다. 아래는 두 데이터셋의 seed별 macro다.', '',
              '| Population | ΔMSE mean ± SD | ΔMAE mean ± SD |', '|---|---:|---:|']
    for hetero, frame in paired_macro[paired_macro['head'] == 'flatten'].groupby('heterogeneous'):
        lines.append(f'| {"heterogeneous" if hetero else "homogeneous"} | {frame.delta_mse.mean():.6f} ± {frame.delta_mse.std():.6f} | {frame.delta_mae.mean():.6f} ± {frame.delta_mae.std():.6f} |')
    lines += ['', '## 마지막 상태 head (seed7만; 탐색적)', '', '| Data | Variant | MSE | MAE |', '|---|---|---:|---:|']
    for _, row in runs[runs['head'] == 'last'].iterrows():
        lines.append(f'| {row.data} | {row.variant} | {row.mse:.6f} | {row.mae:.6f} |')
    lines += ['', '## 같은 checkpoint의 memory 개입', '',
              '양수 Δ는 해당 개입이 full retrieval보다 오차를 높였음을 뜻한다. 각 개입은 전체 test에 적용했다.', '',
              '| Head | Data | Variant | off − full MSE | uniform − full | recent − full |', '|---|---|---|---:|---:|---:|']
    for _, row in interventions.iterrows():
        lines.append(f'| {row["head"]} | {row.data} | {row.variant} | {row.off_minus_full_mse:.6f} | {row.uniform_minus_full_mse:.6f} | {row.recent_minus_full_mse:.6f} |')
    lines += ['', '## 해석 범위 및 이어서 확인할 사항', '',
              '- 10epoch/early-stop3의 첫 확인 실험이다. 최적 성능/수렴/통계적 유의성을 주장하지 않는다.',
              '- 효과를 population 이질성, 검색 on/off, 두 요인의 interaction으로 나누어 읽는다.',
              '- 모든 조건의 nominal parameter 수를 맞췄지만 retrieval off의 Q/K/gate는 미사용이다.',
              '- Flatten head는 모든 과거 spike를 직접 읽는다. Last head는 용량과 정보 접근도 달라진다.',
              '- Internal diagnostics는 첫 test8window만이다. ETT에는 정답 memory slot label이 없다.',
              '- off/uniform/recent는 고정 checkpoint의 개입이다. 해당 방식으로 재학습한 대조와 다르다.',
              '- 강도가 작은 gamma=.05, fixed tau, pre-query/post-value convention의 한정된 검증이다.',
              '- 이전 population coding 실험은 입력96/window norm/Gaussian/deeper backbone이 달라 직접 비교할 수 없다.',
              '- 학습 source hash/데이터 hash/명령/환경/checkpoint 경로는 각 run JSON, 검증은 check_summary.json.',
              '- 장기학습, learned tau, pre-reset memory, fractional prior, hard selection, synthetic recall 학습, 에너지 측정: not run.',
              '- Canonical 진행 기록: repository root docs/PROJECT_LOG.md.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    # Standalone publication/export artifacts; no browser or generated illustrations.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    variants = ['homogeneous_no_memory', 'homogeneous_retrieval', 'heterogeneous_no_memory', 'heterogeneous_retrieval']
    for ax, data in zip(axes, ['ETTh1', 'ETTh2']):
        frame = tasks[(tasks['head'] == 'flatten') & (tasks.data == data)].set_index('variant').loc[variants]
        ax.bar(np.arange(4), frame.mse, yerr=frame.mse_sd, capsize=4,
               color=['#95a5a6', '#547a96', '#76b3a5', '#176b5b'])
        ax.set_xticks(np.arange(4), ['Homo\nno memory', 'Homo\nretrieval', 'Hetero\nno memory', 'Hetero\nretrieval'])
        ax.set_title(data + ' / input 336 / horizon 96')
        ax.set_ylabel('Test MSE (train-standardized)')
        ax.set_ylim(bottom=0)
    fig.suptitle('Population membrane memory: mean ± sample SD over 3 seeds')
    for ext in ['png', 'pdf']:
        fig.savefig(root / ('comparison.' + ext), dpi=180)
    plt.close(fig)
    print(macro.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='ett-first-20260914')
    summarize(parser.parse_args().suite)
