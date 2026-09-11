"""Independent NumPy audit of coding effects plus saved logs/checkpoints."""
import csv
import hashlib
import json
from pathlib import Path

import run_ett
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from summarize_population_ablation import TASK, NSMT, SUITE, SEEDS, TASKS, AXES, CODES


def main():
    torch.set_num_threads(2)
    root=TASK/'results'/SUITE
    aggregate=json.loads((root/'aggregate.json').read_text())
    assert aggregate['complete']==aggregate['expected']==96
    values=np.empty((3,8,2,2,2),dtype=np.float64)
    checked=0
    for si,seed in enumerate(SEEDS):
        for ti,(dataset,horizon) in enumerate(TASKS):
            for ai,axis in enumerate(AXES):
                for ci,code in enumerate(CODES):
                    name=f'{dataset}_p{horizon}_{axis}_code{code}_seed{seed}'
                    r=json.loads((root/(name+'.json')).read_text())
                    values[si,ti,ai,ci]=[r['test']['mse'],r['test']['mae']]
                    log=Path(r['save_log_path'])
                    config=torch.load(log.parent/'model_state/config.pt',map_location='cpu')
                    assert config['seed']==seed and config['population_code']==code
                    assert config['model_config']==r['model_config']
                    state=torch.load(r['checkpoint'],map_location='cpu')
                    assert 'state_dict' not in state and 'head.weight' in state
                    assert tuple(state['head.weight'].shape)==(horizon,12*64)
                    assert all(torch.isfinite(v).all() for v in state.values())
                    for split in ('train','val'):
                        accumulator=EventAccumulator(str(log/(split+'_0'))).Reload()
                        for metric in ('loss','mse','mae'):
                            events=accumulator.Scalars(f'{split}_0/{metric}')
                            assert [e.step for e in events]==list(range(1,r['epochs_run']+1))
                            key='mse' if metric=='loss' else metric
                            reference=[epoch['train_'+key] if split=='train' else epoch['validation'][key] for epoch in r['history']]
                            np.testing.assert_allclose([e.value for e in events],reference,rtol=1e-6,atol=1e-7)
                    checked+=1
    task_rows=list(csv.DictReader((root/'per_task.csv').open()))
    assert len(task_rows)==32
    for row in task_rows:
        ti=TASKS.index((row['dataset'],int(row['horizon'])))
        ai=AXES.index(row['axis']);ci=CODES.index(row['code'])
        for mi,metric in enumerate(('mse','mae')):
            x=values[:,ti,ai,ci,mi]
            np.testing.assert_allclose([float(row[metric+'_mean']),float(row[metric+'_std'])],
                                       [x.mean(),x.std(ddof=1)],rtol=1e-12,atol=1e-14)
    for ai,axis in enumerate(AXES):
        for ci,code in enumerate(CODES):
            for mi,metric in enumerate(('mse','mae')):
                x=values[:,:,ai,ci,mi].mean(axis=1)
                actual=aggregate['variants'][axis+'_'+code][metric]
                np.testing.assert_allclose(actual['values'],x,rtol=1e-12,atol=1e-14)
                np.testing.assert_allclose([actual['mean'],actual['std']],[x.mean(),x.std(ddof=1)],rtol=1e-12,atol=1e-14)
        for mi,metric in enumerate(('mse','mae')):
            delta=values[:,:,ai,0,mi]-values[:,:,ai,1,mi]
            percent=100*delta/values[:,:,ai,1,mi]
            actual=aggregate['coding_effect'][axis][metric]
            for key,matrix in (('macro_delta',delta),('macro_relative_percent',percent)):
                x=matrix.mean(axis=1)
                np.testing.assert_allclose([actual[key]['mean'],actual[key]['std']],
                                           [x.mean(),x.std(ddof=1)],rtol=1e-12,atol=1e-14)
            assert actual['seed_task_wins']==int((delta<0).sum())
            assert actual['task_mean_wins']==int((delta.mean(axis=0)<0).sum())
            assert actual['all_seed_task_wins']==int((delta<0).all(axis=0).sum())
    paired=list(csv.DictReader((root/'paired.csv').open()))
    assert len(paired)==32
    for row in paired:
        ti=TASKS.index((row['dataset'],int(row['horizon'])));ai=AXES.index(row['axis'])
        mi=('mse','mae').index(row['metric'])
        x=values[:,ti,ai,0,mi]-values[:,ti,ai,1,mi]
        percent=100*x/values[:,ti,ai,1,mi]
        np.testing.assert_allclose([float(row['delta_mean']),float(row['delta_std']),
                                   float(row['relative_percent_mean']),float(row['relative_percent_std'])],
                                  [x.mean(),x.std(ddof=1),percent.mean(),percent.std(ddof=1)],rtol=1e-12,atol=1e-14)
        assert int(row['wins'])==int((x<0).sum())
    for mi,metric in enumerate(('mse','mae')):
        x=((values[:,:,0,0,mi]-values[:,:,0,1,mi])-(values[:,:,1,0,mi]-values[:,:,1,1,mi])).mean(axis=1)
        actual=aggregate['interaction'][metric]
        np.testing.assert_allclose([actual['mean'],actual['std']],[x.mean(),x.std(ddof=1)],rtol=1e-12,atol=1e-14)
    provenance=json.loads((root/'provenance.json').read_text())
    assert len(provenance)==96 and len({p['run_id'] for p in provenance})==96
    for row in provenance:
        assert not row['reused']
        assert hashlib.sha256((NSMT/row['source']).read_bytes()).hexdigest()==row['sha256']
    return {'status':'passed','runs':96,'paired_initial_checks':aggregate['paired_initial_hash_checks'],
            'task_rows':32,'paired_rows':32,'tensorboard_and_config_checkpoint_runs':checked,
            'method':'independent NumPy [seed,task,axis,code,metric] tensor, sample SD ddof=1'}


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
