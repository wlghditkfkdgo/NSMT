"""Independent checks of execution provenance, checkpoints, metrics, and paired summaries."""
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
    root=TASK/'results'/suite
    done=json.loads((root/'completion.json').read_text())
    ids=[j['id'] for j in done['jobs']]
    assert len(ids)==len(set(ids))==36 and all(j['status']=='complete' for j in done['jobs'])
    runs=[json.loads((root/(name+'.json')).read_text()) for name in ids]
    expected=set(product(['ETTh1','ETTh2'],[7,13,21],[False,True],['off','dense','sparse']))
    actual={(r['config']['data'],r['config']['seed'],r['config']['heterogeneous'],r['config']['read_mode'] if r['config']['retrieval'] else 'off') for r in runs}
    assert actual==expected
    assert len({r['config']['architecture'] for r in runs})==len({r['config']['pred_len'] for r in runs})==1
    assert len({r['git_commit'] for r in runs})==1
    assert len({json.dumps(r['source_sha256'],sort_keys=True) for r in runs})==1
    for filename,digest in runs[0]['source_sha256'].items():assert sha256(TASK/filename)==digest,filename
    initial,counts,raw_cache,lookup={},{},{},{}
    for r in runs:
        c,h=r['config'],r['history']
        assert not r['protocol']['smoke_only'] and c['max_train_batches']==c['max_eval_batches']==0
        assert c['seq_len']==336 and c['pred_len'] in [96,720] and c['head_mode']=='flatten'
        initial.setdefault(c['seed'],set()).add(r['initial_parameter_sha256'])
        counts.setdefault(c['pred_len'],set()).add(r['parameters'])
        losses=[x['val']['mse'] for x in h]
        assert r['best_epoch']==int(np.argmin(losses)) and r['best_val_mse']==min(losses)
        assert abs(r['restored_val']['mse']-min(losses))<1e-7
        assert r['test']['elements']==(2880-c['pred_len']+1)*c['pred_len']*7
        path=Path(c['save_result_path'])
        epoch=pd.read_csv(path/'log/best_log_0.csv')
        assert len(epoch)==len(h)
        final=pd.read_csv(path/'log/final+result.csv').iloc[0]
        for metric in ['loss','mse','mae']:
            assert abs(final[metric]-r['test'][metric])<5.01e-7
            for split in ['train','val']:
                expected_values=[x[split][metric] for x in h]
                np.testing.assert_allclose(epoch[split+'_'+metric],expected_values,atol=5.01e-7,rtol=0)
                events=EventAccumulator(str(path/'log'/(split+'_0'))).Reload().Scalars(split+'_0/'+metric)
                assert [x.step for x in events]==list(range(len(h)))
                np.testing.assert_allclose([x.value for x in events],expected_values,atol=2e-7,rtol=2e-7)
        horizon=pd.read_csv(path/'log/horizon_metrics.csv')
        for metric in ['mse','mae']:
            np.testing.assert_allclose(horizon[metric].mean(),r['test'][metric],atol=1e-10,rtol=1e-10)
        saved=torch.load(path/'model_state/config.pt',map_location='cpu')
        assert saved['best_epoch']==r['best_epoch'] and saved['variant']==c['variant']
        state=torch.load(path/'model_state/best+model.pt',map_location='cpu')
        assert all(torch.isfinite(x).all() for x in state.values())
        assert sum(x.numel() for name,x in state.items() if not name.endswith('.beta'))==r['parameters']
        assert sha256(path/'model_state/best+model.pt')==r['checkpoint_sha256']
        if c['data'] not in raw_cache:
            data=r['data'];assert sha256(data['path'])==data['sha256']
            raw=pd.read_csv(data['path'])[data['columns']].to_numpy(dtype=np.float64)
            mean,scale=raw[:8640].mean(0),raw[:8640].std(0)
            np.testing.assert_allclose(mean,data['scaler_mean'],atol=1e-12)
            np.testing.assert_allclose(scale,data['scaler_scale'],atol=1e-12)
            raw_cache[c['data']]=((raw-mean)/scale).astype(np.float32)
        sample=json.loads((path/'log/forecast_example.json').read_text())
        np.testing.assert_array_equal(np.array(sample['target'],dtype=np.float32),raw_cache[c['data']][11520:11520+c['pred_len']])
        diag=r['test']['diagnostics']
        assert len(diag['layers'])==(1 if c['architecture']=='patch' else 3)
        for layer in diag['layers'].values():
            assert 0<=layer['support_density']<=1+1e-6 and 0<=layer['empty_read_fraction']<=1
            assert 0<=layer['mean_real_mass']<=1+1e-5
            assert all(0<=x<=1 for x in layer['spike_rate_per_constituent'])
            assert abs(sum(layer['lag_mass'])-(1-layer['empty_read_fraction']))<1e-5
            if not c['heterogeneous']:assert layer['population_membrane_std']<1e-6
            if not c['retrieval']:assert layer['mean_abs_evidence']==0 and layer['support_density']==0
        assert set(r['test']['interventions'])==({'off','uniform','recent'} if c['retrieval'] else set())
        for value in r['test']['interventions'].values():assert value['elements']==r['test']['elements']
        policy=c['read_mode'] if c['retrieval'] else 'off'
        lookup[(c['data'],c['seed'],c['heterogeneous'],policy)]=np.array([r['test']['mse'],r['test']['mae']])
    assert all(len(v)==1 for v in initial.values()) and all(len(v)==1 for v in counts.values())
    macro=pd.read_csv(root/'macro.csv').set_index('variant')
    for hetero,policy in product([False,True],['off','dense','sparse']):
        name=('heterogeneous' if hetero else 'homogeneous')+('_no_memory' if policy=='off' else '_'+policy)
        data=np.array([[lookup[(d,s,hetero,policy)] for d in ['ETTh1','ETTh2']] for s in [7,13,21]]).mean(1)
        row=macro.loc[name]
        np.testing.assert_allclose([row.mse,row.mae],data.mean(0),rtol=1e-12)
        np.testing.assert_allclose([row.mse_sd,row.mae_sd],data.std(0,ddof=1),rtol=1e-12,atol=1e-14)
    for _,row in pd.read_csv(root/'paired.csv').iterrows():
        on,off=row.comparison.split('-')
        delta=lookup[(row.data,row.seed,row.heterogeneous,on)]-lookup[(row.data,row.seed,row.heterogeneous,off)]
        np.testing.assert_allclose([row.delta_mse,row.delta_mae],delta,rtol=1e-10,atol=1e-14)
    result={'status':'passed','runs':36,'training_commit':runs[0]['git_commit'],
            'checks':['complete matrix','source and initial parameter hashes','minimum validation checkpoint',
                      'full evaluation counts','CSV/history/TensorBoard','checkpoint hash/finite weights/parameter count',
                      'train-only scaler and test target boundary','layer selection diagnostics','independent macro SD/paired deltas']}
    write_json(root/'check_summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True)
    check(parser.parse_args().suite)
