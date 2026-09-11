"""Independent NumPy audit of the final three-seed statistics and provenance."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from summarize_multiseed import TASK, NSMT, SEED_SUITES, TASKS, VARIANTS, PAIRS


def main():
    output=TASK/'results/ett-multiseed-20260911'
    aggregate=json.loads((output/'aggregate.json').read_text())
    assert aggregate['complete']==aggregate['expected']==120
    # Dense [seed, task, variant, metric], built directly from source run JSONs.
    values=np.empty((3,8,5,2),dtype=np.float64)
    for si,(seed,suite) in enumerate(SEED_SUITES.items()):
        for ti,(dataset,horizon) in enumerate(TASKS):
            for vi,variant in enumerate(VARIANTS):
                p=TASK/'results'/suite/f'{dataset}_p{horizon}_{variant}_seed{seed}.json'
                r=json.loads(p.read_text())
                values[si,ti,vi]=[r['test']['mse'],r['test']['mae']]
                assert Path(r['checkpoint']).is_file()
    per_task=list(csv.DictReader((output/'per_task.csv').open()))
    assert len(per_task)==40
    for row in per_task:
        ti=TASKS.index((row['dataset'],int(row['pred_len'])))
        vi=VARIANTS.index(row['variant'])
        for mi,metric in enumerate(('mse','mae')):
            x=values[:,ti,vi,mi]
            np.testing.assert_allclose([float(row[metric+'_mean']),float(row[metric+'_std'])],
                                       [x.mean(),x.std(ddof=1)],rtol=1e-12,atol=1e-14)
    for vi,variant in enumerate(VARIANTS):
        macro=values[:,:,vi,:].mean(axis=1)
        for mi,metric in enumerate(('mse','mae')):
            actual=aggregate['variants'][variant]['macro_'+metric]
            np.testing.assert_allclose(actual['values'],macro[:,mi],rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose([actual['mean'],actual['std']],
                                       [macro[:,mi].mean(),macro[:,mi].std(ddof=1)],rtol=1e-12,atol=1e-14)
    paired=list(csv.DictReader((output/'paired.csv').open()))
    assert len(paired)==32
    for tested,reference in PAIRS:
        a=values[:,:,VARIANTS.index(tested),0];b=values[:,:,VARIANTS.index(reference),0]
        delta=a-b;relative=100*delta/b
        stats=aggregate['comparisons'][tested+' vs '+reference]
        for field,matrix in (('macro_delta_mse',delta),('macro_relative_percent',relative)):
            per_seed=matrix.mean(axis=1)
            np.testing.assert_allclose([stats[field]['mean'],stats[field]['std']],
                                       [per_seed.mean(),per_seed.std(ddof=1)],rtol=1e-12,atol=1e-14)
        assert stats['individual_seed_task_wins']==int((delta<0).sum())
        assert stats['tasks_won_on_3seed_mean']==int((delta.mean(axis=0)<0).sum())
        assert stats['tasks_won_in_all_3_seeds']==int((delta<0).all(axis=0).sum())
        for ti,(dataset,horizon) in enumerate(TASKS):
            row=next(r for r in paired if (r['dataset'],int(r['pred_len']),r['tested'],r['reference'])==(dataset,horizon,tested,reference))
            np.testing.assert_allclose([float(row['delta_mse_mean']),float(row['delta_mse_std'])],
                                       [delta[:,ti].mean(),delta[:,ti].std(ddof=1)],rtol=1e-12,atol=1e-14)
    provenance=json.loads((output/'provenance.json').read_text())
    assert len(provenance)==120 and sum(r['reused'] for r in provenance)==40
    assert len({r['run_id'] for r in provenance})==120
    for row in provenance:
        assert hashlib.sha256((NSMT/row['source_result']).read_bytes()).hexdigest()==row['result_sha256']
    print(json.dumps({'status':'passed','source_runs':120,'checkpoints':120,'reused_runs':40,
                      'task_mean_sd_rows':40,'paired_rows':32,'macro_variants':5,
                      'method':'independent NumPy seed/task/variant/metric tensor; ddof=1'},indent=2))


if __name__=='__main__':
    main()
