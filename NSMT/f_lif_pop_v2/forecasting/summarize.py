"""One horizon/backbone per suite; paired off/dense/sparse comparisons over three seeds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import TASK
from utils import write_json


def summarize(suite):
    root = TASK / 'results' / suite
    completion = json.loads((root / 'completion.json').read_text())
    if len(completion['jobs']) != 36 or any(j['status'] != 'complete' for j in completion['jobs']):
        raise ValueError('Require all 36 completed runs')
    results = [json.loads((root / (j['id'] + '.json')).read_text()) for j in completion['jobs']]
    rows, layers = [], []
    for r in results:
        c, t = r['config'], r['test']
        diag = t['diagnostics']
        row = dict(run_id=r['run_id'], architecture=c['architecture'], horizon=c['pred_len'],
                   data=c['data'], seed=c['seed'], heterogeneous=c['heterogeneous'], variant=c['variant'],
                   policy=c['read_mode'] if c['retrieval'] else 'off', mse=t['mse'], mae=t['mae'],
                   best_epoch=r['best_epoch'], epochs=len(r['history']), best_val_mse=r['best_val_mse'],
                   parameters=r['parameters'], seconds=r['wall_seconds'], log_path=c['save_result_path'])
        for key in ['support_density','empty_read_fraction','mean_real_mass','mean_gate_after_first_patch','mean_lag_in_patches','evidence_to_charge_abs_ratio']:
            row[key]=diag[key]
        for mode, metric in t['interventions'].items():
            row[mode+'_minus_full_mse']=metric['mse']-t['mse']
        rows.append(row)
        for name,value in diag['layers'].items():
            layers.append(dict(run_id=r['run_id'],layer=name,variant=c['variant'],data=c['data'],seed=c['seed'],
                               support_density=value['support_density'],empty_read_fraction=value['empty_read_fraction'],
                               mean_real_mass=value['mean_real_mass'],spike_rate=np.mean(value['spike_rate_per_constituent']),
                               membrane_diversity=value['population_membrane_std']))
    runs=pd.DataFrame(rows).sort_values(['data','variant','seed'])
    assert runs.architecture.nunique()==runs.horizon.nunique()==1
    runs.to_csv(root/'per_run.csv',index=False)
    pd.DataFrame(layers).to_csv(root/'layer_diagnostics.csv',index=False)
    tasks=runs.groupby(['data','variant']).agg(mse=('mse','mean'),mse_sd=('mse','std'),mae=('mae','mean'),mae_sd=('mae','std')).reset_index()
    tasks.to_csv(root/'per_task.csv',index=False)
    seed_macro=runs.groupby(['variant','seed'])[['mse','mae']].mean().reset_index()
    macro=seed_macro.groupby('variant').agg(mse=('mse','mean'),mse_sd=('mse','std'),mae=('mae','mean'),mae_sd=('mae','std')).reset_index()
    macro.to_csv(root/'macro.csv',index=False)
    paired=[]
    for (data,hetero,seed),frame in runs.groupby(['data','heterogeneous','seed']):
        frame=frame.set_index('policy')
        for on,off in [('dense','off'),('sparse','off'),('sparse','dense')]:
            paired.append(dict(data=data,heterogeneous=bool(hetero),seed=int(seed),comparison=on+'-'+off,
                               delta_mse=frame.loc[on,'mse']-frame.loc[off,'mse'],delta_mae=frame.loc[on,'mae']-frame.loc[off,'mae']))
    paired=pd.DataFrame(paired)
    paired.to_csv(root/'paired.csv',index=False)
    paired.groupby(['heterogeneous','seed','comparison'])[['delta_mse','delta_mae']].mean().reset_index().to_csv(root/'paired_macro_by_seed.csv',index=False)
    diagnostics=runs.groupby('variant')[['support_density','empty_read_fraction','mean_real_mass']].mean().reset_index()
    write_json(root/'aggregate.json',{'runs':len(runs),'architecture':runs.architecture.iloc[0],'horizon':int(runs.horizon.iloc[0]),
               'macro':macro.to_dict(orient='records'),
               'diagnostics':diagnostics.to_dict(orient='records'),
               'note':'Macro averages datasets within seed, then reports mean and sample SD over seeds.'})
    lines=[f'# PopulationLIF v2 — {runs.architecture.iloc[0]} / H{runs.horizon.iloc[0]}','',
           'ETTh1/ETTh2 × seeds7,13,21 × homogeneous/heterogeneous × off/dense/sparse =36 runs.',
           '동일 점수(-learned Q/K squared distance)·null 후보·retrieval-aware gate에서 softmax와 sparsemax를 비교한다.',
           '모든 최종 checkpoint는 validation MSE 최소값으로 선택했다. 수치는 전체 test의 train-standardized MSE/MAE다.','',
           '| Variant | MSE ± SD | MAE ± SD |','|---|---:|---:|']
    for _,r in macro.iterrows():lines.append(f'| {r.variant} | {r.mse:.6f} ± {r.mse_sd:.6f} | {r.mae:.6f} ± {r.mae_sd:.6f} |')
    lines+=['','| Data | Variant | MSE ± SD | MAE ± SD |','|---|---|---:|---:|']
    for _,r in tasks.iterrows():lines.append(f'| {r.data} | {r.variant} | {r.mse:.6f} ± {r.mse_sd:.6f} | {r.mae:.6f} ± {r.mae_sd:.6f} |')
    lines+=['','## 선택 동작 (첫 test8windows, 마지막 population layer)','',
            '| Variant | Selected/available slots | Empty read fraction | Real probability mass |','|---|---:|---:|---:|']
    for _,r in diagnostics.iterrows():lines.append(f'| {r.variant} | {r.support_density:.6f} | {r.empty_read_fraction:.6f} | {r.mean_real_mass:.6f} |')
    lines+=['','- Sparsemax는 일부 weight를 정확히0으로 만들 수 있지만 매 query의 sparsity를 보장하지 않는다. 실제 density/empty rate를 함께 읽는다.',
            '- Off/dense/sparse의 nominal parameter/초기값은 같고 off의 retrieval 파라미터는 미사용이다. Homogeneous는 redundant-state 대조다.',
            '- off/uniform/recent는 같은 checkpoint의 평가 개입이다. uniform/recent는 null을 우회하여 과거 후보를 강제 선택하고 gate를 재계산한다.',
            '- 서로 다른 backbone은 용량/연산이 다르다. PatchTST와 TSMixer는 causal population-spiking 변형이며 원 논문 재현이 아니다.',
            '- 알려진 slot의 operator 검사는 통과했지만 end-to-end synthetic recall 학습/에너지 측정/정답 ETT lag 검증은 not run.',
            '- 과거slot mask는 raw input의 후속 recurrent-state 영향을 삭제하지 않는다. Dense search와 sparse use는 별개다.',
            '- 이전 v1 결과는 scorer/gate/budget 등이 달라 sparsemax만의 효과를 비교하는 대조가 아니다. v2 dense와 sparse가 해당 대조다.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(macro.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True)
    summarize(parser.parse_args().suite)
