"""Paired 2x2 coding/SSA ablation; strict final completeness and log validation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as stats

TASK = Path(__file__).resolve().parent
NSMT = TASK.parents[1]
SUITE = 'ett-population-ablation-20260911'
SEEDS = (7,13,21)
AXES = ('temporal','no_attention')
CODES = ('gaussian','repeat')
TASKS = tuple((d,h) for d in ('ETTh1','ETTh2','ETTm1','ETTm2') for h in (96,720))


def distribution(x):
    return {'n':len(x),'mean':stats.mean(x),'std':stats.stdev(x) if len(x)>1 else None,'values':list(x)}


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n')
        w.writeheader();w.writerows(rows)


def summarize(partial=False,plot=False):
    root=TASK/'results'/SUITE
    manifest=json.loads((root/'manifest.json').read_text())
    assert len(manifest['jobs'])==96
    if not partial:
        completed=json.loads((root/'completion.json').read_text())
        assert len(completed['jobs'])==96
        assert {j['id'] for j in manifest['jobs']}=={j['id'] for j in completed['jobs']}
        assert all(j['status']=='complete' and j['returncode']==0 for j in completed['jobs'])
    runs={};raw=[];provenance=[];legacy_deltas=[]
    expected={(d,h,a,c,s) for d,h in TASKS for a in AXES for c in CODES for s in SEEDS}
    sources=None;protocol=None;environment=None;common_config=None
    for job in manifest['jobs']:
        path=root/(job['id']+'.json')
        if not path.exists() and partial:
            continue
        r=json.loads(path.read_text());c=r['config']
        key=(c['dataset'],c['pred_len'],c['variant'],c['population_code'],c['seed'])
        assert key in expected and key not in runs
        assert r['run_id']==job['id'] and r['status']=='complete'
        assert not r['protocol']['quick_smoke_only'] and r['model_config']['population_code']==key[3]
        normalized={k:v for k,v in c.items() if k not in ('dataset','pred_len','variant','population_code','seed','device')}
        env={k:r['environment'][k] for k in ('python','torch','cuda','packages')}
        if sources is None:
            sources=r['source_sha256'];protocol=r['protocol'];environment=env;common_config=normalized
            for name, digest in sources.items():
                assert hashlib.sha256((NSMT/name).read_bytes()).hexdigest()==digest
        assert (r['source_sha256'],r['protocol'],env,normalized)==(sources,protocol,environment,common_config)
        for split,field in (('val','validation'),('test','test')):
            assert r[field]['elements']==r['data']['splits'][split]['windows']*key[1]*7
            assert all(0 <= r[field][m] < float('inf') for m in ('mse','mae'))
        assert abs(r['validation']['mse']-min(e['validation']['mse'] for e in r['history']))<1e-7
        assert r['epochs_run']==len(r['history'])<=c['epochs']
        assert r['best_epoch'] in [e['epoch'] for e in r['history'] if e['best']]
        log=Path(r['save_log_path'])
        # CSV text has six decimals; full precision remains in JSON.
        csv_epochs=list(csv.DictReader((log/'best_log_0.csv').open()))
        assert len(csv_epochs)==r['epochs_run']
        for line,epoch in zip(csv_epochs,r['history']):
            assert int(line['epoch'])==epoch['epoch']
            for column,value in (('train_loss',epoch['train_mse']),('train_mse',epoch['train_mse']),
                                 ('train_mae',epoch['train_mae']),('val_loss',epoch['validation']['mse']),
                                 ('val_mse',epoch['validation']['mse']),('val_mae',epoch['validation']['mae'])):
                assert abs(float(line[column])-value)<=5.01e-7
        final=list(csv.DictReader((log/'final+result.csv').open()))
        assert len(final)==1 and int(final[0]['seed'])==key[-1]
        for metric in ('mse','mae'):
            assert abs(float(final[0][metric])-r['test'][metric])<=5.01e-7
        assert Path(r['checkpoint'])==log.parent/'model_state/best+model.pt'
        assert Path(r['checkpoint']).is_file() and (log.parent/'model_state/config.pt').is_file()
        assert (log.parent/'logargs.txt').is_file()
        for split in ('train_0','val_0'):
            assert list((log/split).glob('events.out.tfevents.*'))
        runs[key]=r
        raw.append({'dataset':key[0],'horizon':key[1],'axis':key[2],'code':key[3],'seed':key[4],
                    'mse':r['test']['mse'],'mae':r['test']['mae'],'best_epoch':r['best_epoch'],
                    'epochs_run':r['epochs_run'],'parameters':r['parameters'],'log':str(log)})
        provenance.append({'run_id':r['run_id'],'source':str(path.relative_to(NSMT)),
                           'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'reused':False})
        if key[3]=='gaussian':
            previous='ett-quick-20260910' if key[4]==7 else f'ett-multiseed-20260911-seed{key[4]}'
            old=json.loads((TASK/'results'/previous/f'{key[0]}_p{key[1]}_{key[2]}_seed{key[4]}.json').read_text())
            legacy_deltas.append(abs(old['test']['mse']-r['test']['mse']))
    if not partial:
        assert set(runs)==expected
    paired_initial_checks=0
    for d,h,a,c,s in list(runs):
        if c!='gaussian' or (d,h,a,'repeat',s) not in runs:
            continue
        gauss,repeat=runs[d,h,a,c,s],runs[d,h,a,'repeat',s]
        assert gauss['data']==repeat['data']
        assert gauss['parameters']==repeat['parameters']
        assert gauss['initial_state_sha256']==repeat['initial_state_sha256']
        assert {k:v for k,v in gauss['model_config'].items() if k!='population_code'}=={
            k:v for k,v in repeat['model_config'].items() if k!='population_code'}
        paired_initial_checks+=1
    summary={'complete':len(runs),'expected':96,'seeds':list(SEEDS),'std':'sample SD across seeds, ddof=1',
             'paired_initial_hash_checks':paired_initial_checks,'variants':{},'coding_effect':{},'interaction':{},
             'legacy_gaussian_max_abs_test_mse_delta':max(legacy_deltas) if legacy_deltas else None}
    task_rows=[];paired_rows=[]
    for axis in AXES:
        for code in CODES:
            seeds=[s for s in SEEDS if all((d,h,axis,code,s) in runs for d,h in TASKS)]
            if seeds:
                summary['variants'][axis+'_'+code]={'seeds':seeds,**{
                    metric:distribution([stats.mean(runs[d,h,axis,code,s]['test'][metric] for d,h in TASKS) for s in seeds])
                    for metric in ('mse','mae')}}
            for d,h in TASKS:
                if not all((d,h,axis,code,s) in runs for s in SEEDS):
                    continue
                row={'dataset':d,'horizon':h,'axis':axis,'code':code}
                for metric in ('mse','mae'):
                    x=[runs[d,h,axis,code,s]['test'][metric] for s in SEEDS]
                    row.update({metric+'_mean':stats.mean(x),metric+'_std':stats.stdev(x)})
                task_rows.append(row)
        seeds=[s for s in SEEDS if all((d,h,axis,c,s) in runs for d,h in TASKS for c in CODES)]
        if not seeds:
            continue
        summary['coding_effect'][axis]={'seeds':seeds}
        for metric in ('mse','mae'):
            delta=lambda d,h,s:runs[d,h,axis,'gaussian',s]['test'][metric]-runs[d,h,axis,'repeat',s]['test'][metric]
            relative=lambda d,h,s:100*delta(d,h,s)/runs[d,h,axis,'repeat',s]['test'][metric]
            summary['coding_effect'][axis][metric]={
                'macro_delta':distribution([stats.mean(delta(d,h,s) for d,h in TASKS) for s in seeds]),
                'macro_relative_percent':distribution([stats.mean(relative(d,h,s) for d,h in TASKS) for s in seeds]),
                'seed_task_wins':sum(delta(d,h,s)<0 for d,h in TASKS for s in seeds),
                'seed_task_pairs':8*len(seeds),
                'task_mean_wins':sum(stats.mean(delta(d,h,s) for s in seeds)<0 for d,h in TASKS),
                'all_seed_task_wins':sum(all(delta(d,h,s)<0 for s in seeds) for d,h in TASKS)}
            if len(seeds)==3:
                for d,h in TASKS:
                    x=[delta(d,h,s) for s in seeds];pct=[relative(d,h,s) for s in seeds]
                    paired_rows.append({'dataset':d,'horizon':h,'axis':axis,'metric':metric,
                                        'delta_mean':stats.mean(x),'delta_std':stats.stdev(x),
                                        'relative_percent_mean':stats.mean(pct),'relative_percent_std':stats.stdev(pct),
                                        'wins':sum(v<0 for v in x)})
    if len(runs)==96:
        summary['training_diagnostics']={}
        for axis in AXES:
            for code in CODES:
                selected=[r for (d,h,a,c,s),r in runs.items() if (a,c)==(axis,code)]
                firing=[r['test_first_batch_spikes']['embedding.lif'] for r in selected]
                summary['training_diagnostics'][axis+'_'+code]={
                    'runs':len(selected),'epoch_min':min(r['epochs_run'] for r in selected),
                    'epoch_max':max(r['epochs_run'] for r in selected),
                    'epoch_cap_runs':sum(r['epochs_run']==r['config']['epochs'] for r in selected),
                    'best_at_epoch_cap_runs':sum(r['best_epoch']==r['config']['epochs'] for r in selected),
                    'test_first_batch_embedding_firing_min':min(firing),
                    'test_first_batch_embedding_firing_max':max(firing),
                    'test_metrics_equal_window_mean_runs':sum(all(abs(r['test'][m]-r['test']['window_mean'][m])<1e-12
                                                                for m in ('mse','mae')) for r in selected)}
        for metric in ('mse','mae'):
            per_seed=[]
            for s in SEEDS:
                effects=[]
                for d,h in TASKS:
                    on=runs[d,h,'temporal','gaussian',s]['test'][metric]-runs[d,h,'temporal','repeat',s]['test'][metric]
                    off=runs[d,h,'no_attention','gaussian',s]['test'][metric]-runs[d,h,'no_attention','repeat',s]['test'][metric]
                    effects.append(on-off)
                per_seed.append(stats.mean(effects))
            summary['interaction'][metric]=distribution(per_seed)
    save_csv(root/'per_run.csv',raw);save_csv(root/'per_task.csv',task_rows);save_csv(root/'paired.csv',paired_rows)
    (root/'aggregate.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if not partial:
        write_report(root,summary,task_rows,paired_rows,plot)
    return summary


def write_report(root,summary,task_rows,paired_rows,plot):
    fmt=lambda x:f"{x['mean']:.6f} ± {x['std']:.6f}"
    lines=['# Population tuning ablation — 96 new ETT runs','',
           'ETT 4 datasets × horizons 96/720 × seeds 7/13/21 × Gaussian/repeated scalar × temporal SSA/no SSA. '
           'Input 96, patch/stride 8, direct, K16/D64, two-stage head, max 10 epochs/patience 3. '
           'Select minimum validation MSE and evaluate all test windows.','',
           'Only the tuning response changes within each pair. Both clip to [-3,3]; repeat maps the scalar affinely to [0,1] '
           'and copies it across K slots. Shape, trainable parameter count and initial state hashes match in all 48 pairs. '
           'No population identity is used. Repeated slots do not offer independent feature diversity.','',
           '![Fixed input transforms](input_transforms.png)','',
           '## Macro metrics','',
           'Average eight tasks within each seed first; mean ± sample SD below is across three seed means.','',
           '| Variant | MSE mean ± SD | MAE mean ± SD |','|---|---:|---:|']
    for name,v in summary['variants'].items():
        lines.append(f"| {name} | {fmt(v['mse'])} | {fmt(v['mae'])} |")
    lines+=['','## Training budget and firing diagnostics','',
            '| Variant | Epoch range | Reached 10 epochs / 24 | Best at epoch 10 / 24 | Test first-batch embedding firing range |',
            '|---|---:|---:|---:|---:|']
    for name,v in summary['training_diagnostics'].items():
        lines.append(f"| {name} | {v['epoch_min']}–{v['epoch_max']} | {v['epoch_cap_runs']}/24 | {v['best_at_epoch_cap_runs']}/24 | "
                     f"{100*v['test_first_batch_embedding_firing_min']:.2f}–{100*v['test_first_batch_embedding_firing_max']:.2f}% |")
    lines+=['','Firing ranges describe only the first test batch of each run; they are not full-split rates or energy estimates. '
            'Reaching the epoch limit, especially with the best score at that limit, leaves convergence unresolved.']
    lines+=['','## Paired coding effect','',
            'Gaussian minus repeated scalar; negative means Gaussian is better. No p-values or convergence claim.','',
            '| SSA | Metric | Macro delta mean ± SD | Mean task-relative change (%) ± SD | Wins / 24 | Tasks won on mean / 8 | Tasks won in all seeds / 8 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for axis,v in summary['coding_effect'].items():
        for m in ('mse','mae'):
            x=v[m]
            lines.append(f"| {axis} | {m} | {fmt(x['macro_delta'])} | {fmt(x['macro_relative_percent'])} | {x['seed_task_wins']}/24 | {x['task_mean_wins']}/8 | {x['all_seed_task_wins']}/8 |")
    lines+=['','Relative changes are computed per matched task/seed before averaging; they are not ratios of the macro MSE values.']
    lines+=['','Interaction is (Gaussian−repeat with SSA) − (Gaussian−repeat without SSA). '
            'Negative means the coding advantage is larger with SSA.','']
    for metric,x in summary['interaction'].items():
        lines.append(f'- {metric.upper()}: {fmt(x)}')
    lines+=['','## Interpretation','']
    for axis in AXES:
        effect=summary['coding_effect'][axis]['mse']
        exceptions=[f"{r['dataset']}/{r['horizon']}" for r in paired_rows
                    if r['axis']==axis and r['metric']=='mse' and r['delta_mean']>=0]
        lines.append(f"- {axis}: Gaussian has lower MSE in {effect['seed_task_wins']}/24 seed-task pairs and "
                     f"all three seeds in {effect['all_seed_task_wins']}/8 tasks. Tasks without a mean MSE improvement: "+', '.join(exceptions)+'.')
    if all(summary['coding_effect'][a]['mse']['macro_delta']['mean']<0 for a in AXES):
        lines+=['','Fixed Gaussian tuning helps on average against this deletion control under the short fixed training budget, '
                'with and without SSA. This supports a coding contribution in the current prototype; '
                'it does not establish a general advantage over an independently optimized raw-input model. '
                'The repeat controls hitting the epoch cap leave the longer-training comparison open.']
    for metric in ('mse','mae'):
        lines+=['',f'## Per-task {metric.upper()}: mean ± sample SD','',
                '| Dataset | Horizon | SSA Gaussian | SSA repeat | No SSA Gaussian | No SSA repeat |','|---|---:|---:|---:|---:|---:|']
        for d,h in TASKS:
            cells=[]
            for axis in AXES:
                for code in CODES:
                    r=next(r for r in task_rows if (r['dataset'],r['horizon'],r['axis'],r['code'])==(d,h,axis,code))
                    cells.append(f"{r[metric+'_mean']:.4f} ± {r[metric+'_std']:.4f}")
            lines.append(f'| {d} | {h} | '+' | '.join(cells)+' |')
    lines+=['','## Logging and checks','',
            'Each run uses neorecall-style date/config/seed directories with log/best_log_0.csv, log/final+result.csv, '
            'TensorBoard train_0/val_0, logargs.txt and model_state/config.pt + best+model.pt. '
            'per_run.csv locates each log. CSV values have six decimals; JSON preserves full precision. '
            'Full split counts, checkpoints, CSV/history agreement, source/config/data equality and paired initialization are checked.',
            '',f"Maximum absolute MSE difference between newly trained Gaussian runs and their historical counterparts: {summary['legacy_gaussian_max_abs_test_mse_delta']:.12g}. Historical runs are not reused in these statistics.",
            '','## Limits','',
            'This estimates the effect of fixed Gaussian tuning against a clipped affine repeat control in the specified SNN. '
            'Same nominal parameter count does not mean the repeat control has independent K-slot features. '
            'Identical repeated features also couple the head gradients; this is a deletion ablation within this architecture, '
            'not proof that Gaussian coding is superior to every raw-input SNN. '
            'An independently optimized raw projection/K1 model, learned Gaussian centers/widths, longer training and energy measurements were not run. '
            'Three seeds share the same data splits; they measure initialization/order variability, not independent dataset replication. '
            'SSA on/off have different parameter sets; exact initialization matching applies within each coding pair.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
        for row,axis in enumerate(AXES):
            for col,metric in enumerate(('mse','mae')):
                ax=axes[row,col]
                selected=[next(r for r in paired_rows if (r['dataset'],r['horizon'],r['axis'],r['metric'])==(d,h,axis,metric)) for d,h in TASKS]
                y=[r['relative_percent_mean'] for r in selected];err=[r['relative_percent_std'] for r in selected]
                ax.bar(range(8),y,yerr=err,capsize=3,color=['#16806a' if v<0 else '#c25540' for v in y])
                ax.axhline(0,color='black',linewidth=.8);ax.set_title(axis+' / '+metric.upper())
                ax.set_ylabel('Gaussian vs repeat change (%)');ax.grid(axis='y',alpha=.2)
                ax.set_xticks(range(8),[d+'\n'+str(h) for d,h in TASKS],fontsize=8)
        fig.suptitle('Population tuning effect — paired mean ± sample SD across 3 seeds')
        fig.text(.5,.01,'Negative favors Gaussian. SD bars are not confidence intervals. Same dimensions and paired initial weights.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.03,1,.96));fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    print(json.dumps(summarize(args.allow_partial,args.plot),indent=2))
