"""Wait for both suites, audit completed runs, and append canonical experiment records."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

import pandas as pd

from check_summary import check
from check_reload import check as check_reload
from compare_stages import compare
from summarize import summarize
from utils import write_json

TRAINING_COMMIT = '794a29693a97dc1246c97ccf42d37ff6fd0ac585'
COMPLETED_TAG = 'exp/f-lif-pop-tcn-v1-20260914'


def record(suite, horizon):
    root = TASK / 'results' / suite
    log = TASK.parents[2] / 'docs/PROJECT_LOG.md'
    marker = f'## 2026-09-14 — 2차 Spike-TCN H{horizon} 완료 (24 runs)'
    if marker in log.read_text():
        return
    runs = pd.read_csv(root / 'per_run.csv')
    macro = pd.read_csv(root / 'macro.csv')
    tasks = pd.read_csv(root / 'per_task.csv')
    interventions = pd.read_csv(root / 'interventions.csv')
    raw = [json.loads((root / (name + '.json')).read_text()) for name in runs.run_id]
    done = json.loads((root / 'completion.json').read_text())
    wall = (datetime.fromisoformat(done['finished_utc']) - datetime.fromisoformat(done['created_utc'])).total_seconds()
    values = macro.set_index('variant')
    off, on = values.loc['heterogeneous_no_memory', 'mse'], values.loc['heterogeneous_retrieval', 'mse']
    lines = ['', marker, '',
             f'- Branch `exp/f-lif-pop-tcn-v1`; base `ed5c8d16fb5e9c56234c58d6a8d870ffe20f4efc`; training snapshot `{raw[0]["git_commit"]}`. 24개 전부 exit0. 전체48개 완료 tag/commit은 두 horizon 검증 뒤 별도 기록한다.',
             '- 목적/코드/데이터/환경/seed/hyperparameters는 직전 2차 사전 기록과 동일하다. Convolution2개/kernel3/dilation1,2/current residual; population/optimizer/data 코드는 사전 동결했다. ETTh1/ETTh2×seeds7/13/21×네 조건, flatten head만 학습했다.',
             f'- Full test elements/run: {raw[0]["test"]["elements"]}; parameters/run: {raw[0]["parameters"]}. Launcher wall {wall:.1f}s; run {runs.seconds.min():.1f}–{runs.seconds.max():.1f}s; max GPU allocated {max(r["max_cuda_memory_bytes"] for r in raw)/1024**3:.3f}GiB.',
             f'- 24개 중 {sum(runs.epochs == 10)}개가10epoch 상한에 도달했다. {sum(any(h["lr"] < .001 for h in r["history"]) for r in raw)}개가 감소된 LR로 학습했다. 최소 validation MSE epoch는1-based {runs.best_epoch.min()+1}–{runs.best_epoch.max()+1}. 최종 checkpoint 복원 검증 통과.',
             f'- 이질적 population에서 retrieval 추가 macro MSE {off:.6f}→{on:.6f} ({100 * (on / off - 1):+.3f}%). Macro는 데이터셋 평균을 seed별 계산한 뒤3seed mean±sample SD이며 horizon 간 합산하지 않는다.',
             '', '| Variant | MSE ± SD | MAE ± SD |', '|---|---:|---:|']
    for _, row in macro.iterrows():
        lines.append(f'| {row.variant} | {row.mse:.6f} ± {row.mse_sd:.6f} | {row.mae:.6f} ± {row.mae_sd:.6f} |')
    lines += ['', '| Data | Variant | MSE ± SD | MAE ± SD |', '|---|---|---:|---:|']
    for _, row in tasks.iterrows():
        lines.append(f'| {row.data} | {row.variant} | {row.mse:.6f} ± {row.mse_sd:.6f} | {row.mae:.6f} ± {row.mae_sd:.6f} |')
    lines += ['', '각 실행: best epoch은0-based.', '',
              '| Data | Variant | Seed | MSE | MAE | Best epoch | Epochs |', '|---|---|---:|---:|---:|---:|---:|']
    for _, row in runs.iterrows():
        lines.append(f'| {row.data} | {row.variant} | {row.seed} | {row.mse:.6f} | {row.mae:.6f} | {row.best_epoch} | {row.epochs} |')
    lines += ['', '같은 checkpoint의 모든 층 memory 개입 (ΔMSE=개입−full, 재학습 아님):', '',
              '| Data | Variant | off | uniform | recent |', '|---|---|---:|---:|---:|']
    for _, row in interventions.iterrows():
        lines.append(f'| {row.data} | {row.variant} | {row.off_minus_full_mse:+.6f} | {row.uniform_minus_full_mse:+.6f} | {row.recent_minus_full_mse:+.6f} |')
    lines += ['',
              '- 검증 통과:24matrix/중복/누락, frozen source hash/학습 commit 일치, 조건별 초기parameter/count, minimum validation checkpoint/복원, 전체 test element수, CSV/history/TensorBoard 일치, horizon평균, 저장config/checkpoint hash/finite weight, train-only scaler와 test 첫 target 경계, 층별 population 진단, 독립 NumPy 기반 macro/paired 통계. `check_summary.json`.',
              '- 개별 fresh checkpoint 재평가는 이 horizon 집계 시점 not run. 두 horizon 학습이 모두 종료된 뒤 ETTh1/heterogeneous-retrieval/seed7의 전체 test MSE/MAE 및 off/uniform/recent를 검증한다(atol1e-12). 결과는 check_reload.json 및 전체 완료 기록에 남긴다.',
              f'- Exact launcher: `bash f_lif_pop_tcn_v1/forecasting/scripts/run_ett.sh --suite {suite} --pred_len {horizon}` (run_stage2.sh가 호출). 각 실제 subprocess명령은 manifest/completion 및 run JSON. 분석 호출: `summarize({suite!r}); check({suite!r})`; 모든 학습이 끝난 뒤 `check_reload({suite!r}, {horizon}, "cuda:0")`를 실행한다. 관리 script command: `{shlex.join([sys.executable, *sys.argv])}`; 환경 prefix는 사전 기록과 동일.',
              f'- Artifact: `NSMT/f_lif_pop_tcn_v1/forecasting/results/{suite}/`의24run JSON,manifest/completion,REPORT,per_run/per_task/macro/paired/paired_macro_by_seed/interaction/interventions CSV,aggregate,checks,comparison PNG/PDF. Log 위치는 per_run.csv의log_path; task `log/{suite}/` 하위에 neorecall CSV/TensorBoard/logargs/model_state 형식. Raw events/checkpoints/stdout은 local.',
              '- 제한: 최대10/early-stop3의 짧은 예산, 세seed/두dataset. 층별진단은 첫 test8window만, all-layer intervention은 어느 층이 원인인지 분리하지 않는다. 1차 대비 parameter가24714개 많고 current residual/depth가 추가되어 matched-capacity 비교가 아니다. 논문 원본 재현/longer-budget/last-head 학습/uniform-recent 재학습/synthetic recall/에너지측정: not run. Test로 재선택하지 않았다. Main 통합/push: not run.', '']
    with log.open('a') as handle:
        handle.write('\n'.join(lines))


def main(args):
    suites = [(96, 'ett-tcn-h96-20260914'), (720, 'ett-tcn-h720-20260914')]
    for horizon, suite in suites:
        completion = TASK / 'results' / suite / 'completion.json'
        while not completion.exists():
            if not args.wait:
                raise FileNotFoundError(completion)
            time.sleep(10)
        summarize(suite)
        check(suite)
        record(suite, horizon)
        print(f'Audited and recorded {suite}', flush=True)
    # Keep re-evaluation off the training GPUs until the two suites finish.
    for horizon, suite in suites:
        check_reload(suite, horizon, 'cuda:0')
    compare('stage-comparison-20260914')
    write_json(TASK / 'results/stage-comparison-20260914/postprocess.json',
               {'status': 'passed', 'command': shlex.join([sys.executable, *sys.argv]),
                'canonical_log_appended': True, 'target_completed_tag': COMPLETED_TAG,
                'finalization_status_file': str(TASK / 'scripts/queues/finalization.json')})
    if args.finalize:
        finalize()


def finalize():
    """Publish local experiment artifacts only, and stop if repository context changed."""
    repository = TASK.parents[2]

    def git(*command):
        return subprocess.check_output(['git', *command], cwd=repository, text=True).strip()

    if git('branch', '--show-current') != 'exp/f-lif-pop-tcn-v1' or git('rev-parse', 'HEAD') != TRAINING_COMMIT:
        raise RuntimeError('Metrics complete; branch/HEAD changed, so automatic commit/tag was not performed')
    if git('tag', '--list', COMPLETED_TAG):
        raise RuntimeError('Completed tag already exists; preserve it and review versioning manually')
    comparison = TASK / 'results/stage-comparison-20260914'
    for suite in ['ett-tcn-h96-20260914', 'ett-tcn-h720-20260914']:
        for filename in ['check_summary.json', 'check_reload.json']:
            assert json.loads((TASK / 'results' / suite / filename).read_text())['status'] == 'passed'
    assert json.loads((comparison / 'check_comparison.json').read_text())['status'] == 'passed'
    marker = '## 2026-09-14 — 2차 전체48개 검증 및 로컬 완료 기록'
    log = repository / 'docs/PROJECT_LOG.md'
    if marker not in log.read_text():
        with log.open('a') as handle:
            handle.write('\n' + marker + '\n\n'
                         '- H96/H720 각24개 학습 및 전체 artifact audit 완료. 두 horizon 모두 새 config/checkpoint 객체로 전체 test MSE/MAE와 모든 memory 개입을 atol1e-12에서 재현했다. 각 horizon 기록에서 pending이던 check_reload는 이제 passed다.\n'
                         '- 1차/2차 48쌍의 실제 데이터·전처리·학습 조건 일치 검사를 통과했다. `results/stage-comparison-20260914/{REPORT.md,macro.csv,per_task.csv,paired_macro_by_seed.csv,training_selection.csv,tcn_layer_diagnostics.csv,check_comparison.json,comparison.png,comparison.pdf}`에 비교를 보존한다. 각 horizon은 별개 task로 집계한다.\n'
                         '- 최종 artifact 생성 뒤 모델/학습 source는 바꾸지 않았다. 자동 후처리는 source/data/checkpoint/지표 검증을 수행했다. 자동 생성한 2차 그림의 사람/시각 검토는 not run; CSV 수치 검증은 passed.\n'
                         f'- 완료 commit은 annotated tag `{COMPLETED_TAG}`로 식별한다. 자동 finalization은 예상 branch/학습 HEAD를 확인하고 이 실험의 명시된 분석 코드·결과·canonical log만 commit한다. 다른 branch/HEAD로 바뀌면 commit/tag를 중지하고 local queue status에 기록한다. Main 통합/remote push: not run.\n'
                         '- 다음 검증 후보(이번에는 not run): 더 긴 공통 학습 budget; uniform/recent 재학습 및 정답 lag가 있는 synthetic recall; population 층별 retrieval 대조; matched-capacity backbone 비교. 이번 두 ETT-hour/3seed/짧은 예산만으로 보편적 이득이나 통계적 유의성을 주장하지 않는다.\n')
    relative = str(TASK.relative_to(repository))
    paths = [relative + '/compare_stages.py', relative + '/scripts/finish_stage2.py',
             relative + '/README.md', 'docs/PROJECT_LOG.md']
    for suite in ['ett-tcn-h96-20260914', 'ett-tcn-h720-20260914']:
        paths += [relative + '/results/' + suite, relative + '/log/' + suite]
    paths.append(relative + '/results/stage-comparison-20260914')
    git('add', '--', *paths)
    git('diff', '--cached', '--check', '--', *paths)
    # --only preserves independently staged changes outside these task paths.
    git('commit', '-q', '--only', '-m', 'experiment: record audited population Spike-TCN horizon 96 and 720 results', '--', *paths)
    git('tag', '-a', COMPLETED_TAG, '-m', 'Completed 48 causal population Spike-TCN adaptation runs; full artifact audits, checkpoint reloads, and stage comparisons')
    write_json(TASK / 'scripts/queues/finalization.json',
               {'status': 'complete', 'commit': git('rev-parse', 'HEAD'), 'tag': COMPLETED_TAG,
                'tag_commit': git('rev-parse', COMPLETED_TAG + '^{}'), 'pushed': False})
    print('Finalized local experiment:', COMPLETED_TAG, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--finalize', action='store_true', help='Commit/tag only after all audits and expected branch/HEAD checks')
    try:
        main(parser.parse_args())
    except Exception as error:
        write_json(TASK / 'scripts/queues/postprocess_failure.json',
                   {'status': 'failed', 'error': str(error), 'type': type(error).__name__})
        raise
