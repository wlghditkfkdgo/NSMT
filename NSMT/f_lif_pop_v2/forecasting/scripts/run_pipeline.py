"""Ordered, detached-ready experiment queue; each completed stage is audited before advancing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import fcntl
from itertools import product
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from types import SimpleNamespace

TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK))
from config import Config, set_random_seed
from summarize import summarize
from check_summary import check
from test import test
from utils import write_json, sha256

REPOSITORY=TASK.parents[2]
SOURCE_FILES=['config.py','model.py','ours.py','layers.py','backbones.py','train.py','test.py','utils.py',
              'data_provider/data_factory.py','data_provider/data_loader.py']


def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPOSITORY,text=True).strip()


def verify_source(state):
    if git('branch','--show-current')!='exp/f-lif-pop-v2' or git('rev-parse','HEAD')!=state['expected_head']:
        raise RuntimeError('Branch/HEAD changed; stop pending work and preserve the repository')
    for name,digest in state['source_sha256'].items():
        if sha256(TASK/name)!=digest:raise RuntimeError('Training source changed: '+name)


def run_suite(args,state,architecture,horizon):
    verify_source(state)
    suite=f'{args.pipeline}_{architecture}_p{horizon}'
    root=TASK/'results'/suite
    if root.exists():
        if not args.resume:raise FileExistsError('Suite exists; do not overwrite '+suite)
        completion=json.loads((root/'completion.json').read_text())
        if len(completion['jobs'])!=36 or any(j['status']!='complete' for j in completion['jobs']):
            raise RuntimeError('Only fully trained suites can be resumed for audit: '+suite)
        for job in completion['jobs']:
            previous=json.loads((root/(job['id']+'.json')).read_text())
            if previous['source_sha256']!=state['source_sha256']:raise RuntimeError('Frozen source mismatch')
            for key,value in [('epoch',args.epochs),('patience',args.patience),('scheduler_patience',args.scheduler_patience),('batch_size',args.batch_size)]:
                if previous['config'][key]!=value:raise RuntimeError('Resume protocol mismatch: '+key)
        print('RECHECK COMPLETE SUITE',suite,flush=True)
        audit_suite(args,architecture,horizon,suite)
        return suite
    jobs=[]
    for seed,data,heterogeneous,policy in product([7,13,21],['ETTh1','ETTh2'],[False,True],['off','dense','sparse']):
        variant=('heterogeneous' if heterogeneous else 'homogeneous')+('_no_memory' if policy=='off' else '_'+policy)
        command=[sys.executable,str(TASK/'train.py'),'--suite',suite,'--architecture',architecture,
                 '--data',data,'--seed',str(seed),'--pred_len',str(horizon),
                 '--heterogeneous' if heterogeneous else '--no-heterogeneous',
                 '--no-retrieval' if policy=='off' else '--retrieval','--read_mode','dense' if policy=='dense' else 'sparse',
                 '--epoch',str(args.epochs),'--patience',str(args.patience),
                 '--scheduler_patience',str(args.scheduler_patience),'--batch_size',str(args.batch_size)]
        jobs.append({'id':f'{architecture}_{data}_p{horizon}_flatten_{variant}_seed{seed}',
                     'seed':seed,'command':command,'status':'pending'})
    manifest={'suite':suite,'architecture':architecture,'horizon':horizon,'created_utc':datetime.now(timezone.utc).isoformat(),
              'training_commit':state['expected_head'],'gpus':args.gpus,'workers_per_gpu':args.workers_per_gpu,'jobs':jobs}
    write_json(root/'manifest.json',manifest)
    status=TASK/'scripts/queues'/(suite+'.json')
    write_json(status,manifest)
    mutex=threading.Lock()

    def worker(gpu,worker_id):
        while True:
            with mutex:
                if any(j['status']=='failed' for j in jobs):return
                job=next((j for j in jobs if j['status']=='pending'),None)
                if job is None:return
                verify_source(state)
                job.update(status='running',gpu=gpu,worker=worker_id,started_utc=datetime.now(timezone.utc).isoformat())
                write_json(status,manifest)
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED=str(job['seed']),PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LD_LIBRARY_PATH=str(Path(sys.executable).resolve().parents[1]/'lib'))
            stdout=TASK/'log'/suite/(job['id']+'.stdout');stdout.parent.mkdir(parents=True,exist_ok=True)
            print('START',job['id'],'gpu',gpu,flush=True)
            with stdout.open('x') as handle:
                process=subprocess.Popen(job['command'],cwd=TASK,env=env,stdout=handle,stderr=subprocess.STDOUT)
                with mutex:
                    job.update(pid=process.pid,stdout=str(stdout));write_json(status,manifest)
                code=process.wait()
            with mutex:
                job.update(status='complete' if code==0 else 'failed',returncode=code,finished_utc=datetime.now(timezone.utc).isoformat())
                write_json(status,manifest)
            print('END',job['id'],'code',code,flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)*args.workers_per_gpu) as pool:
        futures=[pool.submit(worker,gpu,i) for gpu in args.gpus for i in range(args.workers_per_gpu)]
        for future in futures:future.result()
    manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
    write_json(root/'completion.json',manifest)
    if any(j['status']!='complete' for j in jobs):raise RuntimeError('Suite incomplete; later stages not launched: '+suite)
    audit_suite(args,architecture,horizon,suite)
    return suite


def audit_suite(args,architecture,horizon,suite):
    root=TASK/'results'/suite
    summarize(suite)
    check(suite)
    # Fresh object/checkpoint, whole test, no overwriting the original log files.
    import numpy as np
    original=json.loads((root/f'{architecture}_ETTh1_p{horizon}_flatten_heterogeneous_sparse_seed7.json').read_text())
    config=Config();config.load_args(original['config']['save_result_path'],SimpleNamespace(cpu=False,num_device=args.gpus[0]))
    set_random_seed(config.seed)
    actual=test(config,save=False)
    for metric in ['mse','mae']:
        np.testing.assert_allclose(actual[metric],original['test'][metric],rtol=0,atol=1e-12)
        for mode in ['off','uniform','recent']:
            np.testing.assert_allclose(actual['interventions'][mode][metric],original['test']['interventions'][mode][metric],rtol=0,atol=1e-12)
    write_json(root/'check_reload.json',{'status':'passed','run_id':original['run_id'],'checks':['whole test and interventions MSE/MAE'],'atol':1e-12})


def record_stage(args,state,architecture,suites,tag):
    import pandas as pd
    now=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
    lines=[f'\n## {now} — PopulationLIF v2 {architecture} 완료 (72 runs)\n',
           f'- Branch `exp/f-lif-pop-v2`; base `{state["base_commit"]}`; 각 horizon training commit은 completion.json의 training_commit에 기록한다 (후처리 복구 전후 commit이 다를 수 있으며 학습 source hashes는 동일). 완료 commit은 tag `{tag}`로 식별한다. 목적: 동일 점수·gate에서 dense vs sparse 선택 효과 및 population 이질성을 분리한다.',
           f'- ETTh1/ETTh2 × seeds7/13/21 × homo/hetero × off/dense/sparse × H96/720. Seq336,patch8,D32,K4,head32/flatten,tau2..16,gamma.05,temperature.25,null init−1. 최대{args.epochs}epochs,early-stop{args.patience},ReduceLROnPlateau factor.5/patience{args.scheduler_patience},AdamW lr.001/wd.01,batch{args.batch_size},clip1. 최소 validation MSE checkpoint를 복원해 평가했다.',
           '- 데이터와 환경: 기존 ETT-hour train[0,8640),val[8640,11520),test[11520,14400),train-only StandardScaler,7변수,stride1,context336. H96 windows8209/2785/2785, H7207585/2161/2161. Conda snn_recall/Py3.10/torch1.12.0+cu113,CPUthreads2,deterministic/TF32off. GPU/worker 배치와 모든 실행명령은 각 manifest 및 run JSON, source/data hashes/패키지 버전도 run JSON에 있다.',
           f'- 실행 pipeline command: `{state["command"]}`. Suites: '+', '.join('`'+x+'`' for x in suites)+'.',
           '\n| Horizon | Variant | MSE ± SD | MAE ± SD |\n|---|---|---:|---:|']
    for suite in suites:
        root=TASK/'results'/suite
        aggregate=json.loads((root/'aggregate.json').read_text())
        for row in aggregate['macro']:
            lines.append(f'| {aggregate["horizon"]} | {row["variant"]} | {row["mse"]:.6f} ± {row["mse_sd"]:.6f} | {row["mae"]:.6f} ± {row["mae_sd"]:.6f} |')
        runs=pd.read_csv(root/'per_run.csv')
        lines.append(f'\nH{aggregate["horizon"]}: {int((runs.epochs==args.epochs).sum())}/36 budget cap; {runs.parameters.iloc[0]} nominal parameters; individual run {runs.seconds.min():.1f}–{runs.seconds.max():.1f}s.')
        paired=pd.read_csv(root/'paired_macro_by_seed.csv')
        for hetero,frame in paired[paired.comparison=='sparse-dense'].groupby('heterogeneous'):
            delta=frame.delta_mse.mean()
            lines.append(f'- H{aggregate["horizon"]} {"heterogeneous" if hetero else "homogeneous"}: sparse−dense paired macro ΔMSE {delta:+.6f} (seed SD {frame.delta_mse.std():.6f}); 평균 MSE 기준 sparse가 {"낮음" if delta<0 else "높거나 같음"}. 통계적 유의성 주장은 하지 않는다.')
        for row in aggregate['diagnostics']:
            lines.append(f'- H{aggregate["horizon"]} {row["variant"]}: final-layer support density {row["support_density"]:.6f}, empty-read fraction {row["empty_read_fraction"]:.6f}, real mass {row["mean_real_mass"]:.6f} (첫8 test windows).')
    lines+=['\n각 실행 (best epoch은0-based):\n','| Run | MSE | MAE | Best epoch | Epochs |\n|---|---:|---:|---:|---:|']
    for suite in suites:
        for _,r in pd.read_csv(TASK/'results'/suite/'per_run.csv').iterrows():
            lines.append(f'| {r.run_id} | {r.mse:.6f} | {r.mae:.6f} | {r.best_epoch} | {r.epochs} |')
    lines+=['\n- 검증 통과: complete matrix, source/초기parameter hashes, minimum val checkpoint/복원, 전체 element수, CSV/history/TensorBoard, finite checkpoint/hash, train scaler/target boundary, 층별 support/null/lag diagnostics, independent macro/paired deltas. 각 horizon sparse ETTh1 seed7의 fresh checkpoint 및 off/uniform/recent 전체 MSE/MAE를 atol1e-12에서 재현했다.',
            '- Artifact는 `NSMT/f_lif_pop_v2/forecasting/results/<suite>/`의 REPORT,per_run/per_task/macro/paired/paired_macro_by_seed,layer_diagnostics,aggregate,manifest/completion/checks 및36raw result JSON. per_run.csv log_path가 task log/<suite>/<dataset>/<date>/<config>/seed+variant의 neorecall CSV/events/logargs/config.pt/best+model.pt를 가리킨다. Raw events/checkpoints/stdout은 local, 텍스트 결과는 Git.',
            '- 해석 제한: dense와 sparse는 score/nullable candidates/gate를 공유하지만 정규화 방식/지원집합/real probability mass가 함께 달라진다. Sparse 사용은 dense search 비용 절감을 보장하지 않는다. 실제 density/empty rate를 함께 보고 판단한다. Homogeneous는 redundant state 대조, backbone 간 용량은 다르다. 진단은 첫8 test windows; 전체 synthetic recall 학습/정답 ETT lag/에너지 측정/통계적 유의성 검정은 not run. 기존 v1과는 scorer/gate/budget이 달라 동일 실험으로 합산하지 않는다. 다음 단계는 성능 개선 여부로 선별하지 않고 실행/검증 통과 뒤 진행한다. Main 통합/push: not run.\n']
    with (REPOSITORY/'docs/PROJECT_LOG.md').open('a') as handle:handle.write('\n'.join(lines))


def main(args):
    if len(set(args.gpus))!=len(args.gpus) or not 1<=args.workers_per_gpu<=2:raise ValueError('Use unique GPUs and one or two workers per GPU')
    queue=TASK/'scripts/queues';queue.mkdir(parents=True,exist_ok=True)
    lock=(queue/(args.pipeline+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=TASK/'results'/args.pipeline
    if root.exists() and not args.resume:raise FileExistsError('Pipeline exists; use a new ID')
    if args.resume:
        state=json.loads((queue/(args.pipeline+'.json')).read_text())
        state.setdefault('resumptions',[]).append({'previous_head':state['expected_head'],'resume_head':git('rev-parse','HEAD'),
            'command':shlex.join([sys.executable,*sys.argv]),'utc':datetime.now(timezone.utc).isoformat()})
        state.update(expected_head=git('rev-parse','HEAD'),status='running')
    else:
        state={'pipeline':args.pipeline,'status':'running','command':shlex.join([sys.executable,*sys.argv]),
           'base_commit':args.base_commit,'initial_commit':git('rev-parse','HEAD'),'expected_head':git('rev-parse','HEAD'),
           'source_sha256':{name:sha256(TASK/name) for name in SOURCE_FILES},'stages':[]}
    verify_source(state)
    write_json(root/'pipeline.json',state)
    write_json(queue/(args.pipeline+'.json'),state)
    for architecture in ['patch','tcn','patchtst','tsmixer']:
        if any(s['architecture']==architecture and s['status']=='complete' for s in state['stages']):continue
        suites=[]
        for horizon in [96,720]:
            suites.append(run_suite(args,state,architecture,horizon))
        verify_source(state)
        date=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d')
        tag=f'exp/f-lif-pop-v2-{architecture}-{date}'
        if git('tag','--list',tag):raise RuntimeError('Tag exists; preserve '+tag)
        record_stage(args,state,architecture,suites,tag)
        stage={'architecture':architecture,'status':'complete','suites':suites,'tag':tag if args.finalize else None}
        state['stages'].append(stage)
        state['status']='complete' if architecture=='tsmixer' else 'running'
        archive={k:v for k,v in state.items() if k!='expected_head'}
        write_json(root/'pipeline.json',archive)
        if args.finalize:
            relative=str(TASK.relative_to(REPOSITORY))
            # This task's code/config/results/log and the canonical log only.
            paths=[relative,'NSMT/forecasting/f-LIF_pop_v2.py','docs/PROJECT_LOG.md']
            git('add','--',*paths);git('diff','--cached','--check','--',*paths)
            git('commit','-q','--only','-m',f'experiment: record selective population v2 {architecture} results','--',*paths)
            git('tag','-a',tag,'-m',f'Completed v2 {architecture} H96/H720: 72 runs, dense/sparse/null controls and artifact audits')
            state['expected_head']=git('rev-parse','HEAD')
        write_json(queue/(args.pipeline+'.json'),state)
        print('STAGE COMPLETE',architecture,tag,flush=True)
    state['status']='complete'
    write_json(queue/(args.pipeline+'.json'),state)
    print('PIPELINE COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pipeline',default='selective-v2-20260914')
    parser.add_argument('--base_commit',default='f8215f54106980bad7c782bf08acaef871175c34')
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3])
    parser.add_argument('--workers_per_gpu',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--patience',type=int,default=6)
    parser.add_argument('--scheduler_patience',type=int,default=2)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Re-audit fully trained suites only; preserve runs/checkpoints and refuse partial suites')
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        failure={'status':'failed','error':repr(error),'type':type(error).__name__,'utc':datetime.now(timezone.utc).isoformat()}
        write_json(TASK/'scripts/queues'/(args.pipeline+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json'),failure)
        status=TASK/'scripts/queues'/(args.pipeline+'.json')
        if status.exists():
            state=json.loads(status.read_text());state.update(status='failed',failure=failure);write_json(status,state)
        raise
