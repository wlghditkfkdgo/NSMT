"""Aggregate unchanged ETT quick runs at seeds 7,13,21; sample SD over seeds.

The seed-7 artifacts are referenced in place, never copied/relabelled as new runs.
Each task mean/std uses three seeds. Macro statistics first average the eight
tasks within each seed, then average those three numbers. Paired differences
match dataset, horizon and seed. No p-values or independence assumptions across
datasets are used. --allow-partial is for progress only; final output requires
all 120 complete results and validates code/data/config/protocol consistency.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

TASK = Path(__file__).resolve().parent
NSMT = TASK.parents[1]
SEED_SUITES = {7:"ett-quick-20260910",13:"ett-multiseed-20260911-seed13",21:"ett-multiseed-20260911-seed21"}
VARIANTS = ("population","temporal","temporal_embedding","no_attention","linear")
TASKS = [(d,h) for d in ("ETTh1","ETTh2","ETTm1","ETTm2") for h in (96,720)]
PAIRS = (("temporal","population"),("temporal_embedding","temporal"),("temporal","no_attention"),("temporal","linear"))


def distribution(values):
    return {"n":len(values),"mean":stats.mean(values),"std":stats.stdev(values) if len(values)>1 else None,
            "values":values}


def csv_write(path,rows):
    if rows:
        with path.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0].keys(),lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def summarize(output_suite="ett-multiseed-20260911",allow_partial=False,plot=False):
    output=TASK/"results"/output_suite
    output.mkdir(parents=True,exist_ok=True)
    runs,provenance,raw_rows={},[],[]
    reference_source=None
    reference_config=None
    reference_environment=None
    reference_protocol=None
    expected_keys={(d,h,v,s) for d,h in TASKS for v in VARIANTS for s in SEED_SUITES}
    for seed,suite in SEED_SUITES.items():
        root=TASK/"results"/suite
        manifest_path=root/"manifest.json"
        if not manifest_path.exists():
            if allow_partial:continue
            raise FileNotFoundError(manifest_path)
        manifest=json.loads(manifest_path.read_text())
        assert len(manifest["jobs"])==40 and len({j['id'] for j in manifest['jobs']})==40
        if not allow_partial:
            completion=json.loads((root/"completion.json").read_text())
            assert {j['id'] for j in completion['jobs']}=={j['id'] for j in manifest['jobs']}
            assert all(j['status']=='complete' and j['returncode']==0 for j in completion['jobs'])
        for job in manifest["jobs"]:
            path=root/(job['id']+".json")
            if not path.exists():
                if allow_partial:continue
                raise FileNotFoundError(path)
            r=json.loads(path.read_text())
            c=r['config']; key=(c['dataset'],c['pred_len'],c['variant'],seed)
            assert key in expected_keys and key not in runs and c['seed']==seed
            assert r['run_id']==job['id']==f"{key[0]}_p{key[1]}_{key[2]}_seed{seed}"
            assert r['status']=='complete' and not r['protocol']['quick_smoke_only']
            for metric in ('mse','mae'):
                assert math.isfinite(r['test'][metric]) and math.isfinite(r['validation'][metric])
            for split,field in (('val','validation'),('test','test')):
                assert r[field]['elements']==r['data']['splits'][split]['windows']*key[1]*7
            assert abs(r['validation']['mse']-min(e['validation']['mse'] for e in r['history']))<1e-7
            assert r['best_epoch']<=r['epochs_run']<=c['epochs']
            normalized_config={k:v for k,v in c.items() if k not in ('seed','suite','dataset','pred_len','variant','device')}
            environment={k:r['environment'][k] for k in ('python','torch','cuda','packages')}
            if reference_source is None:
                reference_source=r['source_sha256'];reference_config=normalized_config
                reference_environment=environment;reference_protocol=r['protocol']
                for name,digest in reference_source.items():
                    assert hashlib.sha256((NSMT/name).read_bytes()).hexdigest()==digest
            assert reference_source==r['source_sha256']
            assert reference_config==normalized_config
            assert reference_environment==environment
            assert reference_protocol==r['protocol']
            original=runs.get((*key[:3],7))
            if original is not None:
                assert original['data']==r['data'] and original['model_config']==r['model_config']
            runs[key]=r
            provenance.append({"seed":seed,"source_result":str(path.relative_to(NSMT)),
                               "result_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"run_id":job['id'],
                               "source_commit":r['git_commit'],"reused":seed==7})
            raw_rows.append({"dataset":key[0],"pred_len":key[1],"variant":key[2],"seed":seed,
                             "mse":r['test']['mse'],"mae":r['test']['mae'],"val_mse":r['validation']['mse'],
                             "best_epoch":r['best_epoch'],"epochs_run":r['epochs_run'],"parameters":r['parameters'],
                             "seconds":r['seconds'],"checkpoint":r['checkpoint']})
    if not allow_partial:assert set(runs)==expected_keys
    summary={"complete":len(runs),"expected":120,"seeds":list(SEED_SUITES),"seed_suites":SEED_SUITES,
             "std_definition":"sample standard deviation over seeds (ddof=1)","variants":{},"comparisons":{}}
    task_rows=[]
    for d,h in TASKS:
        for variant in VARIANTS:
            available=[runs[(d,h,variant,s)] for s in SEED_SUITES if (d,h,variant,s) in runs]
            if len(available)!=3:continue
            row={"dataset":d,"pred_len":h,"variant":variant,"n_seeds":3}
            for metric in ('mse','mae'):
                values=[r['test'][metric] for r in available]
                row.update({metric+"_mean":stats.mean(values),metric+"_std":stats.stdev(values)})
            task_rows.append(row)
    for variant in VARIANTS:
        complete_seeds=[s for s in SEED_SUITES if all((d,h,variant,s) in runs for d,h in TASKS)]
        if not complete_seeds:continue
        summary['variants'][variant]={"complete_seeds":complete_seeds}
        for metric in ('mse','mae'):
            macro=[stats.mean(runs[(d,h,variant,s)]['test'][metric] for d,h in TASKS) for s in complete_seeds]
            summary['variants'][variant]["macro_"+metric]=distribution(macro)
    paired_rows=[]
    for tested,reference in PAIRS:
        complete_seeds=[s for s in SEED_SUITES if all((d,h,v,s) in runs for d,h in TASKS for v in (tested,reference))]
        if not complete_seeds:continue
        macro_deltas=[];macro_relative=[];raw_deltas=[];all_seed_wins=0;mean_wins=0
        for seed in complete_seeds:
            deltas=[];relative=[]
            for d,h in TASKS:
                a,b=runs[(d,h,tested,seed)],runs[(d,h,reference,seed)]
                assert a['data']==b['data']
                difference=a['test']['mse']-b['test']['mse']
                deltas.append(difference);relative.append(100*difference/b['test']['mse'])
            macro_deltas.append(stats.mean(deltas));macro_relative.append(stats.mean(relative));raw_deltas.extend(deltas)
        for d,h in TASKS:
            delta=[runs[(d,h,tested,s)]['test']['mse']-runs[(d,h,reference,s)]['test']['mse'] for s in complete_seeds]
            relative=[100*(runs[(d,h,tested,s)]['test']['mse']/runs[(d,h,reference,s)]['test']['mse']-1) for s in complete_seeds]
            if len(complete_seeds)==3:
                all_seed_wins+=all(value<0 for value in delta);mean_wins+=stats.mean(delta)<0
                paired_rows.append({"dataset":d,"pred_len":h,"tested":tested,"reference":reference,"n_seeds":3,
                                    "delta_mse_mean":stats.mean(delta),"delta_mse_std":stats.stdev(delta),
                                    "relative_percent_mean":stats.mean(relative),"relative_percent_std":stats.stdev(relative),
                                    "wins":sum(value<0 for value in delta)})
        summary['comparisons'][tested+" vs "+reference]={"complete_seeds":complete_seeds,
            "macro_delta_mse":distribution(macro_deltas),"macro_relative_percent":distribution(macro_relative),
            "individual_seed_task_wins":sum(value<0 for value in raw_deltas),"seed_task_pairs":len(raw_deltas),
            "tasks_won_on_3seed_mean":mean_wins if len(complete_seeds)==3 else None,
            "tasks_won_in_all_3_seeds":all_seed_wins if len(complete_seeds)==3 else None}
    csv_write(output/'per_run.csv',raw_rows);csv_write(output/'per_task.csv',task_rows);csv_write(output/'paired.csv',paired_rows)
    (output/'aggregate.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+"\n")
    if not allow_partial:
        fmt=lambda x:f"{x['mean']:.5f} ± {x['std']:.5f}"
        lines=["# ETT multi-seed replication — seeds 7, 13, 21","",
               "120 complete results: 40 existing seed-7 runs and 80 new seed-13/21 runs. "
               "Unchanged model, training protocol, data hashes and environment versions were verified.","",
               "Full ETT splits; input 96; prediction 96/720; patch/stride 8; max 10 epochs; patience 3; "
               "best validation MSE checkpoint. All models use train-only scaling and input-window normalization.","",
               "K-axis denotes population-token attention; N-axis denotes observed-patch attention. "
               "LIF simulation follows N in both variants. N-axis + identity adds a trainable [K,D] population identity; "
               "Gaussian centers and widths remain fixed in every population-coded variant. "
               "No SSA retains the population frontend, spiking MLP and two-stage head. Linear is the shared-channel normalized linear baseline.","",
               "## Seed-level macro statistics","",
               "Each seed averages eight dataset/horizon tasks first. Mean ± sample SD (ddof=1) below is over three such seed averages.","",
               "| Variant | Macro MSE mean ± SD | Macro MAE mean ± SD |","|---|---:|---:|"]
        for variant,v in summary['variants'].items():
            lines.append(f"| {variant} | {fmt(v['macro_mse'])} | {fmt(v['macro_mae'])} |")
        for metric in ('mse','mae'):
            lines.extend(["",f"## Per-task {metric.upper()}: mean ± sample SD across three seeds","",
                          "| Dataset | Horizon | K-axis | N-axis | N-axis + identity | No SSA | Linear |","|---|---:|---:|---:|---:|---:|---:|"])
            for d,h in TASKS:
                cells=[]
                for variant in VARIANTS:
                    row=next(r for r in task_rows if (r['dataset'],r['pred_len'],r['variant'])==(d,h,variant))
                    cells.append(f"{row[metric+'_mean']:.4f} ± {row[metric+'_std']:.4f}")
                lines.append(f"| {d} | {h} | "+" | ".join(cells)+" |")
        lines.extend(["","## Paired MSE comparisons","",
                      "Deltas are tested minus reference, matched by task and seed; negative means improvement. "
                      "Macro delta SD is over three seed-level means, not over 24 heterogeneous task/seed pairs.","",
                      "| Tested vs reference | Macro ΔMSE mean ± SD | Wins / 24 seed-task pairs | Tasks won on mean / 8 | Tasks won in all seeds / 8 |",
                      "|---|---:|---:|---:|---:|"])
        for name,v in summary['comparisons'].items():
            lines.append(f"| {name} | {fmt(v['macro_delta_mse'])} | {v['individual_seed_task_wins']}/24 | {v['tasks_won_on_3seed_mean']}/8 | {v['tasks_won_in_all_3_seeds']}/8 |")
        lines.extend(["","## Limits","",
                      "Three seeds describe initialization/training-order variability under this short fixed protocol. "
                      "They do not establish convergence or statistical significance. All seeds reuse the same data splits; "
                      "datasets/horizons are not independent replications of initialization. "
                      "The seed-7 result motivated this replication; model/settings were frozen before the two new seeds. "
                      "Pairing matches the random seed and task, but does not guarantee identical shared-layer initialization "
                      "across architectures with different parameter sets (including No SSA). "
                      "No-population SNN, flatten-head, longer-budget and separately tuned attention-scale comparisons were not run. "
                      "See per_run.csv, per_task.csv, paired.csv and provenance.json for all inputs and paired results.",""])
        (output/'REPORT.md').write_text("\n".join(lines))
        if plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
            for ax,(tested,reference) in zip(axes.flat,PAIRS):
                selected=[next(r for r in paired_rows if (r['dataset'],r['pred_len'],r['tested'],r['reference'])==(d,h,tested,reference)) for d,h in TASKS]
                mean=np.array([r['relative_percent_mean'] for r in selected]);sd=np.array([r['relative_percent_std'] for r in selected])
                ax.bar(range(8),mean,yerr=sd,capsize=3,color=['#16806a' if x<0 else '#c25540' for x in mean],alpha=.85)
                ax.axhline(0,color='black',linewidth=.8);ax.set_title(tested+" vs "+reference,fontsize=10)
                ax.set_ylabel('Relative test MSE change (%)');ax.set_xticks(range(8),[d+"\n"+str(h) for d,h in TASKS],fontsize=8)
                ax.grid(axis='y',alpha=.2)
            fig.suptitle('Paired differences across seeds 7, 13, 21 — mean ± sample SD',fontsize=12)
            fig.text(.5,.01,'Negative means improvement. Error bars show SD, not confidence intervals. Fixed short training protocol.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.035,1,.96));fig.savefig(output/'comparison.png',dpi=180);fig.savefig(output/'comparison.pdf');plt.close(fig)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-suite',default='ett-multiseed-20260911')
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    print(json.dumps(summarize(args.output_suite,args.allow_partial,args.plot),indent=2))
