# 임시 인계 문서 — PopulationLIF forecasting 실험

최초 결과 스냅샷 시각: **2026-09-15 13:59:24 KST**. 당시 branch `exp/f-lif-pop-v2`, HEAD `8c3d13673ce628bff51e7ab858a67b5bc72584da`.

이 문서는 요청에 따라 만든 **한 시점의 요약·코드·결과 사본**이며 자동 갱신되지 않는다. 유일한 정식 append-only 이력은 [저장소 루트 PROJECT_LOG.md](../../../../docs/PROJECT_LOG.md)이다. 다음 세션에서는 이 문서와 함께 live queue 및 새 완료 기록을 읽어야 한다. 사용자 concept/review 원문과 진행 중 학습은 수정하지 않았다.

<a id="handoff-start"></a>
## 0. 새 세션·새 에이전트가 먼저 읽을 운영 인계

접근성/재개 절차 점검: **2026-09-15 15:10:28 KST**. 아래1–16절 수치·개별결과·코드 사본은 **최초 작성 시각(2026-09-15 13:59:24 KST)**의 스냅샷이다. 이번 점검에서는 TCN H96 **완료24/36, 실행8, 대기4, 실패0**였고 H720은 아직 미시작이었다. 이 문서는 이후에도 자동 갱신되지 않으므로 최신 상태는0.4절로 다시 확인한다.

### 0.1 문서 하나만 받은 에이전트의 첫 작업

1. **AGENTS.md → canonical PROJECT_LOG의 마지막 기록 → live pipeline queue → 최근 stdout** 순서로 읽는다. 전체 결과표/코드 사본을 먼저 정독할 필요는 없다.
2. 0.4절로 현재 branch/HEAD/source hashes, 실제 controller·학습 PID와 suite 상태를 확인한다. 문서에 저장된 PID나 `status=running` 하나만으로 생존을 판단하지 않는다.
3. 기존 controller/학습이 살아 있으면 관찰을 이어간다. 이미 승인된 범위는 네 구조288회 실행·검증·상세기록·실험 branch 로컬 commit/tag다. Main 통합/remote push는 포함되지 않는다.
4. 중단됐다면0.5절 상태별 기준으로 이어간다. `--resume`는 중간 epoch 학습 복원 기능이 아니다.
5. 실제 `.py` 파일과 이 문서의 코드 사본을 구분한다. 진행 중 source/HEAD guard와 사용자 원문/local 변경을 보존한다.

### 0.2 문서 내부 바로가기

- [기존 상태 스냅샷](#handoff-snapshot) · [아이디어](#handoff-concept) · [코드 지도](#handoff-code) · [프로토콜](#handoff-protocol)
- [v1 결과](#handoff-v1) · [v2 결과](#handoff-v2) · [2차 진행 스냅샷](#handoff-progress)
- [복구 이력](#handoff-recovery) · [향후 계획](#handoff-plan) · [저장 방식](#handoff-artifacts)
- [commit/tag/hash](#handoff-identity) · [개별run/원시JSON](#handoff-runs) · [핵심 코드 사본](#handoff-source)

### 0.3 핵심 절대경로와 접근 범위

| 용도 | 클릭 가능한 절대경로 |
|---|---|
| 저장소 규칙 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/AGENTS.md](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/AGENTS.md) |
| Git 저장소 루트 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning) |
| 명령 실행 cwd | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT) |
| 이 문서 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/results/TEMP_EXPERIMENT_HANDOFF_20260915.md](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/results/TEMP_EXPERIMENT_HANDOFF_20260915.md) |
| 정식 canonical PROJECT_LOG | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/docs/PROJECT_LOG.md](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/docs/PROJECT_LOG.md) |
| 같은 log의 NSMT 링크 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/docs/PROJECT_LOG.md](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/docs/PROJECT_LOG.md) |
| 사용자 concept | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/docs/population_selective_membrane_memory_snn_concept.md](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/docs/population_selective_membrane_memory_snn_concept.md) |
| 사용자 review | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/docs/PopulationLIF_implementation_review.md](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/docs/PopulationLIF_implementation_review.md) |
| v2 README | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/README.md](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/README.md) |
| PopulationLIF/Sparsemax 코드 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/layers.py](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/layers.py) |
| Backbone 코드 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/backbones.py](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/backbones.py) |
| 모델 조립/forward | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/ours.py](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/ours.py) |
| Config | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/config.py](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/config.py) |
| 학습/checkpoint 선택 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/train.py](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/train.py) |
| 평가/진단 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/test.py](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/test.py) |
| 검증 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/check_summary.py](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/check_summary.py) |
| 실행/재개 controller | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/run_pipeline.py](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/run_pipeline.py) |
| Shell launcher | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/run_pipeline.sh](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/run_pipeline.sh) |
| 최신 pipeline queue | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/queues/selective-v2-20260914.json](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/queues/selective-v2-20260914.json) |
| Suite/PID/실패/lock 파일 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/queues](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/queues) |
| 실제 재개 launcher 정보 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/queues/launcher-resume-20260915.json](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/queues/launcher-resume-20260915.json) |
| 실제 재개 stdout | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/log/selective-v2-20260915.resume.stdout](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/log/selective-v2-20260915.resume.stdout) |
| v2 전체 결과 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/results](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/results) |
| 데이터 디렉터리 | [/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/forecasting/dataset/ETT-small](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/forecasting/dataset/ETT-small) |
| 고정 Python | [/home/yschoi/.conda/envs/snn_recall/bin/python](/home/yschoi/.conda/envs/snn_recall/bin/python) |

본문 상대링크는 **이 Markdown의 results 디렉터리 기준**, 셸 명령은 **NSMT cwd 기준**이다. 렌더러가 상대링크를 지원하지 않으면 위 절대경로를 연다. 같은 서버/워크스페이스의 다른 세션에서는 그대로 사용할 수 있다.

다른 서버/checkout에서는 `git rev-parse --show-toplevel` 아래 `NSMT/`를 새 작업 루트로 매핑한다. **Git만으로 checkpoint/dataset/events/stdout/queues는 복원되지 않으며 concept/review 두 파일도 현재 untracked 원문이다.** 실제 local artifact의 존재를 확인해야 한다. 문서만 복사하거나 Git checkout만으로 전체 실행 상태가 복원된다고 가정하지 않는다.

### 0.4 복사해서 실행하는 읽기 전용 확인

<!-- handoff-status-code -->
```bash
cd '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT'
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python - <<'PY'
from pathlib import Path
from collections import Counter
import hashlib
import json
import subprocess

# NSMT cwd에서 실행. 파일을 쓰거나 학습을 시작하지 않는 조회 코드다.
task = Path.cwd() / 'f_lif_pop_v2/forecasting'
queue = task / 'scripts/queues'
pipeline_file = queue / 'selective-v2-20260914.json'
branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print('branch:', branch, 'HEAD:', head)
if not pipeline_file.exists():
    print('Local queue missing: consult archived pipeline.json and PROJECT_LOG; absence does not prove inactivity.')
else:
    pipeline = json.loads(pipeline_file.read_text())
    print('pipeline:', pipeline['status'], 'completed stages:', pipeline.get('stages', []))
    print('HEAD matches:', head == pipeline.get('expected_head'))
    mismatch = [name for name, digest in pipeline['source_sha256'].items()
                if not (task / name).exists() or hashlib.sha256((task / name).read_bytes()).hexdigest() != digest]
    print('frozen source mismatches:', mismatch)
    for path in sorted(queue.glob('selective-v2-20260914_*_p*.json')):
        suite = json.loads(path.read_text())
        if 'jobs' not in suite:
            continue
        print(path.name, dict(Counter(j['status'] for j in suite['jobs'])))
        for job in suite['jobs']:
            if job['status'] != 'running':
                continue
            proc = Path('/proc') / str(job.get('pid', -1)) / 'cmdline'
            try:
                command = proc.read_bytes().replace(b'\0', b' ').decode(errors='replace')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                command = ''
            matches = str(task / 'train.py') in command and suite['suite'] in command
            print(' ', job['id'], 'pid', job.get('pid'), 'expected process:', matches)
# launcher PID가 shell일 수 있으므로 실제 controller 명령도 확인한다.
controllers = []
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        command = proc.read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if command.split(' ', 1)[0].endswith('python') and str(task / 'scripts/run_pipeline.py') in command:
        controllers.append((proc.parent.name, command))
print('live controllers:', controllers)
PY
```

Suite명은 `selective-v2-20260914_<patch|tcn|patchtst|tsmixer>_p<96|720>`다. 위 명령은 현재 생성된 모든 suite를 조회하므로 TCN H96 이후에도 사용할 수 있다. 미생성 suite는 예정 상태다. GPU 사용은 `nvidia-smi`로 별도 확인한다.

### 0.5 관측 상태에 따른 다음 행동

| 관측 상태 | 다음 행동 | 현재 코드의 범위 |
|---|---|---|
| controller 또는 task 학습 PID 생존 | 새 launcher 없이 queue/stdout 관찰 | 중복 시작하지 않는다 |
| 해당 suite36회 학습 완료, 후처리 실패, controller·학습 모두 종료 | traceback 확인 → 후처리 수정/검증 → source/config 일치 확인 → 아래 detached resume | completion.json의36개 모두 complete인 suite만 재검증 |
| 일부 학습 완료 또는 학습 중 crash | 완료 JSON/checkpoint/실패 stdout 보존, 원인과 재실행 범위 기록 | **partial suite/중간 epoch 자동 resume 미구현**. 단순 --resume 반복으로 복구되지 않음 |
| source/branch/HEAD가 queue 기대값과 다름 | git diff/PROJECT_LOG/새 stage tag로 변경 원인 확인 | guard 통과를 위해 queue hash/HEAD를 임의로 바꾸거나 local 변경을 폐기하지 않는다 |
| 학습·audit 완료 후 commit/tag 실패 | git log/tag, pipeline.stages, canonical log의 정합성부터 복구 | 기존 tag를 덮어쓰지 않으며 이 상태의 자동 resume 성공은 보장하지 않음 |
| pipeline.complete와4개 stage 완료 | 각 suite audit/완료tag를 확인하고 전체 비교 작성 | 같은 ID로 재학습하지 않음 |

재개 전 기본 protocol(30epoch/early6/scheduler2/batch128),10개 training source hashes, branch를 확인한다. `--resume`는 새 시작 HEAD를 기록하므로 후처리만 변경한 commit인지 diff도 확인한다. 학습 source가 달라졌다면 기존 suite와 섞지 않는 새 실험 계획이 필요하다.

다음은 **재개 가능한 중단 상태임을 확인한 뒤에만** 사용하는 detached 명령이다. 현재 실행과 중복해서 실행하지 않는다. 새 stdout/launcher 파일명을 만들고 기존 run/checkpoint는 보존한다. 문서 점검에서는 이 명령을 실행하지 않았다.

<!-- handoff-resume-code -->
```bash
cd '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT'
/home/yschoi/.conda/envs/snn_recall/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess

# 0.5절 재개 조건 확인 후에만 실행한다. 현재 살아 있는 실행과 중복하지 않는다.
nsmt = Path.cwd()
task = nsmt / 'f_lif_pop_v2/forecasting'
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
command = ['bash', str(task / 'scripts/run_pipeline.sh'),
           '--pipeline', 'selective-v2-20260914', '--resume']
stdout = task / 'log' / ('resume-' + stamp + '.stdout')
metadata = task / 'scripts/queues' / ('launcher-resume-' + stamp + '.json')
env = os.environ.copy()
env.update(PYTHONUNBUFFERED='1', LD_LIBRARY_PATH='/home/yschoi/.conda/envs/snn_recall/lib')
env.pop('CUDA_VISIBLE_DEVICES', None)
with stdout.open('x') as handle:
    process = subprocess.Popen(command, cwd=nsmt, env=env, stdin=subprocess.DEVNULL,
                               stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
info = {'pid': process.pid, 'command': command, 'cwd': str(nsmt), 'stdout': str(stdout),
        'utc': stamp, 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
with metadata.open('x') as handle:
    json.dump(info, handle, indent=2)
    handle.write('\n')
print(json.dumps(info, indent=2))
PY
```

새 launcher JSON의 stdout과 suite queue로 후속 작업 시작을 확인하고 canonical log에 append한다. 자동 stage commit은 task 코드/config/text결과/canonical log를 포함한다. 진행 중 중간 commit은 controller HEAD guard에 영향을 준다.

### 0.6 한 run의 checkpoint/로그까지 찾아가기

[개별run 표](#handoff-runs)의 이름을 클릭하면 원시 JSON이 열린다. `config.save_result_path`가 실제 저장 디렉터리다. `config.save_model_state_path`, `config.save_log_path`, `checkpoint_sha256`, `command`, `git_commit`, `source_sha256`를 함께 확인한다.

- [대표 H720 run JSON](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/results/selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_heterogeneous_sparse_seed7.json)
- [저장 config](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/log/selective-v2-20260914_patch_p720/ETTh1/260915/260915+model+myModel+architecture+patch+seq_len+336+pred_len+720+patch_size+8+embed_dim+32+num_population+4+head_dim+32+lr+0.001/seed7_flatten_heterogeneous_sparse/model_state/config.pt)
- [최저 validation checkpoint](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/log/selective-v2-20260914_patch_p720/ETTh1/260915/260915+model+myModel+architecture+patch+seq_len+336+pred_len+720+patch_size+8+embed_dim+32+num_population+4+head_dim+32+lr+0.001/seed7_flatten_heterogeneous_sparse/model_state/best+model.pt)
- [Epoch별 CSV](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/log/selective-v2-20260914_patch_p720/ETTh1/260915/260915+model+myModel+architecture+patch+seq_len+336+pred_len+720+patch_size+8+embed_dim+32+num_population+4+head_dim+32+lr+0.001/seed7_flatten_heterogeneous_sparse/log/best_log_0.csv)
- [최종 test CSV](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/log/selective-v2-20260914_patch_p720/ETTh1/260915/260915+model+myModel+architecture+patch+seq_len+336+pred_len+720+patch_size+8+embed_dim+32+num_population+4+head_dim+32+lr+0.001/seed7_flatten_heterogeneous_sparse/log/final+result.csv)
- [Checkpoint 재평가 검증](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/results/selective-v2-20260914_patch_p720/check_reload.json)

`.pt`는 바이너리이므로 PyTorch loader로 읽는다. `check_summary.py --suite <suite>`와 controller의 `audit_suite`는 검증 artifact를 쓰거나 CPU/GPU 평가를 실행하므로 읽기 전용 상태조회와 구분한다. 정상 학습 중 필요 없이 재실행하지 않는다.

<a id="handoff-snapshot"></a>
## 1. 바로 확인할 현재 상태

| 구분 | 학습 완료 | 실행 | 대기/미시작 | 검증 |
| --- | --- | --- | --- | --- |
| v1 patch H96/H720 | 64 (main48+last-head16) | 0 | 0 | 당시 완료 기록/검증 보존 |
| v1 TCN H96/H720 | 48 | 0 | 0 | 당시 완료 기록/검증 보존 |
| v2 1차 patch H96/H720 | 72/72 | 0 | 0 | 두 horizon audit/fresh reload passed |
| v2 2차 TCN H96 | 16/36 | 8 | 12 | complete matrix audit 아직 not run |
| v2 2차 TCN H720 | 0/36 | 0 | 36 | not run |
| v2 3차 patchtst | 0/72 | 0 | 72 | not run |
| v2 후속 tsmixer | 0/72 | 0 | 72 | not run |

v2 계획288회 중 개별 학습 완료는 88회, 전체 조건 검증을 마친 것은 patch72회다. 현재 TCN 실패 0회. v1과 v2는 scorer/gate/학습 예산이 달라 동일 대조 실험으로 합산하지 않는다.

<a id="handoff-concept"></a>
## 2. 우리가 확인하려는 아이디어

[사용자 concept](../../../docs/population_selective_membrane_memory_snn_concept.md)와 [사용자 implementation review](../../../docs/PopulationLIF_implementation_review.md)가 출발점이다.

하나의 논리적 뉴런을 K개의 LIF 구성원으로 표현하고, 각 구성원이 서로 다른 시간상수로 동일 입력을 누적하게 한다. 시간에 따라 저장한 population 막전위 벡터 중 현재 상태에 관련 있는 기억을 골라 현재 막전위에 **증거 보강**으로 더한다. 과거 출력의 단순 교체나 외부 정답 검색은 아니다.

여기서 **이질성**은 데이터셋/입력 채널이 다르다는 뜻이 아니라, 같은 입력 전류를 받는 구성원들의 tau/beta가 다르다는 뜻이다. K=4, tau=[2,4,8,16], beta=exp(-1/tau)이며 tau는 고정이다. 동질적 대조는 beta의 평균을4번 반복한다. 같은 초기 상태/전류/threshold와 공유 read 때문에 동질적 구성원은 같은 상태로 남는다. 따라서 이는 redundant-state 대조이며 유효 표현 차원이 같은 비교는 아니다.

이번 구현은 concept의 Branch B(직접 막전위 검색)다. 정확한 fractional derivative, fractional temporal prior, cross-window 장기저장소는 구현하지 않았다. 과거 post-reset state를 현재 pre-retrieval state로 조회하며 모든 상태는 입력 window마다 초기화한다.

## 3. 진행 이력과 설계 수정

| 단계 | 수행 내용 | 확인한 점 |
| --- | --- | --- |
| 2026-09-11 설계 | 임시 f-LIF_pop entry와 concept 정리, memory 사용을 증거 보강으로 확정 | population과 검색 효과를 분리하기 위해 patch 우선 |
| 2026-09-14 v1 patch | H96/H720 각 main24+last-head8 | H96 retrieval 효과 작음, H720 retrieval 악화 |
| 2026-09-14 v1 TCN | H96/H720 각24 | 국소 causal conv 뒤 dense 검색의 추가 이득 뚜렷하지 않음 |
| implementation review | 코드와 원문 대조 및 작은 반례 | 과거 slot 명시적 제외 없음, cosine amplitude 구분 제한, query-only gate |
| v2 구현 | dense/sparse 공통 scorer/null/gate와 네 backbone | 정확한0/empty read/amplitude 구분 및 CPU/GPU 검증 |
| 2026-09-14~15 v2 patch | H96/H720 각36 | H96 이질성 이득, H720 이질성+검색 이득 미확인 |
| 후처리 복구 | read weight floor를 반영한 audit 및 completed-suite resume | 학습 source/checkpoint/raw result는 변경 없음 |
| 현재 | v2 TCN H96 진행 | 부분 결과를 최종 구조 비교로 해석하지 않음 |

v1 검토 근거는 [IMPLEMENTATION_REVIEW.md](../../../f_lif_pop_tcn_v1/forecasting/results/stage-comparison-20260914/IMPLEMENTATION_REVIEW.md)에 있다. v1의 softmax는 관련 없는 과거를 정확히 제외하는 mask가 아니었다. Bias-free Q/K를 L2 정규화하면 양의 배수인 상태는 같은 방향으로 취급하고, query-only gate는 같은 query의 다른 bank 품질을 직접 보지 못한다. 이런 표현 제한이 실제 성능 악화의 원인임을 입증한 것은 아니다.

## 4. 현재 v2 뉴런의 실제 수식과 의미

```text
charged = beta * state + (1 - beta) * current
score[j] = -mean_K((Wq(charged) - Wk(history[j]))**2) / temperature
p = sparsemax(concat(score, learned_null_logit))  # dense 대조는 softmax
real_mass = sum(p_real)
weight[j] = p_real[j] / max(real_mass, 1e-12)
mask[j] = weight[j] > 0
memory = sum_j(weight[j] * history[j])
margin = max(score) - learned_null_logit
gate = real_mass * sigmoid(Linear(concat(charged, memory, margin, real_mass)))
voltage = charged + gamma * gate * memory
spike = surrogate(voltage - threshold)
state = voltage - threshold * spike.detach()
```

Q/K는 K×K bias-free 선형변환, identity 초기값이며 L2 정규화하지 않는다. Gate Linear(2K+2,1)는 weight/bias0으로 시작한다. null logit 초기값−1, gamma.05,temperature.25,threshold1,input_scale2다. 모든 K가 같은 시간 slot mask/weights를 공유한다. Full BPTT이며 reset spike만 detach한다. Sparsemax forward와 backward는 layers.py에 구현했다.

**정규화 주의:** real_mass≥1e-12이면 weight 합≈1, tiny positive mass이면1보다 작고0이면0이다. Gate에도 real_mass가 곱해진다. null-only인 sparse read의 memory/gate/evidence는0이다. 실제 mask는 별도 binary parameter가 아니라 sparsemax positive support다. Dense→sparse 비교에서는 support와 상대 가중치/null mass가 함께 바뀐다. 모든 과거 score를 계산하므로 효율/에너지 절감은 주장하지 않는다. Raw input의 후속 recurrent-state 영향까지 mask가 지우지는 않는다.

<a id="handoff-code"></a>
## 5. 네 architecture와 코드 구성

입력 `[B,336,7]`을 채널별로 `[T=42,B*C,patch8]`로 바꾸고 `Linear(8,32) × input_scale2 → PopulationLIF`를 적용한다. 여기서 ×2는 **입력 전류 배율**이며 두 embedding 층이 있다는 뜻이 아니다. 출력 `[T,BC,D=32,K=4]` spike를 `Linear(DK,32) → flatten → Linear(42×32,H)`로 읽는다. 입력 window는 forecast origin 이전의 관측이며 모든 temporal block은 patch 수준에서 causal하다.

| architecture | 추가 block | 전체 population 층 | 해석 |
| --- | --- | --- | --- |
| patch | 없음 | 1 | population/검색 기여를 보기 위한 얕은 기준 |
| tcn | Conv1d(DK,D,k3,dilation1), 이어서 dilation2 | 3 | 왼쪽 padding/current residual; convolution만의 범위는7patch |
| patchtst | causal MultiheadAttention(D32,4heads)+feature MLP | 3 | spike 입력을 쓰는 hybrid softmax attention; all-spiking Spikformer 원본 아님 |
| tsmixer | lower-triangular temporal MLP+feature MLP | 3 | causal/channel-independent spiking 변형; ANN 원본 재현 아님 |

모든 조건의 공통 embedding/head를 먼저 생성해 seed/horizon별 초기값을 맞춘다. 같은 backbone 내 off/dense/sparse는 nominal parameter 수가 같지만 off의 검색 parameter는 미사용이다. Backbone 간 parameter 수와 연산량은 다르다. TSMixer의 위삼각 weight는 영구 mask된다.


| 파일 | 역할 |
| --- | --- |
| [config.py](../config.py) | Config/CLI/seed/device/path; max30/early6/scheduler2 |
| [model.py](../model.py) | LOAD_MODEL factory/checkpoint 로딩 |
| [ours.py](../ours.py) | myModel: common embedding/head와 backbone 선택 |
| [layers.py](../layers.py) | PopulationLIF, Sparsemax autograd, Embedding |
| [backbones.py](../backbones.py) | CurrentBlock/TCN/Attention/Channel/Token/CausalLinear |
| [train.py](../train.py) | train_one_epoch/val_one_epoch/AdamW/scheduler/min-val checkpoint |
| [test.py](../test.py) | 전체 MSE/MAE/개입/층별 diagnostics |
| [utils.py](../utils.py) | EpochLog/EarlyStopping/hash/JSON, v1과 동일 |
| [data_factory.py](../data_provider/data_factory.py) | DataLoader factory, v1과 동일 |
| [data_loader.py](../data_provider/data_loader.py) | train-only ETT scaler/splits, v1과 동일 |
| [check_model.py](../check_model.py) | operator/gradient/causality/paired initialization 검증 |
| [summarize.py](../summarize.py) | 36-run suite macro/paired/REPORT |
| [check_summary.py](../check_summary.py) | checkpoint/data/log/statistics/read-mass audit |
| [run_pipeline.py](../scripts/run_pipeline.py) | 단계 queue/재검증/완료 기록·commit·tag |
| [run_pipeline.sh](../scripts/run_pipeline.sh) | 고정 conda Python과 LD_LIBRARY_PATH launcher |


Entry: [f-LIF_pop_v2.py](../../../forecasting/f-LIF_pop_v2.py). 이전 구현: [f-LIF_pop_v1.py](../../../forecasting/f-LIF_pop_v1.py) 및 [v1 patch](../../../f_lif_pop_v1/forecasting/ours.py), [v1 TCN](../../../f_lif_pop_tcn_v1/forecasting/ours.py). 코드 스타일은 [model_v1 Config](../../../model_v1/forecasting/config.py)와 [neorecall logging](../../../neorecall_v1/forecasting/utils.py)의 Config/LOAD_MODEL/myModel/Embedding/train·val 함수/data_provider 관례를 따른다. 사용자 예시 경로 forecating은 현재 checkout에서 forecasting으로 확인된다.

<a id="handoff-protocol"></a>
## 6. 데이터·학습·평가 프로토콜

| 항목 | v2 공통 설정 |
| --- | --- |
| 데이터 | ETTh1, ETTh2; local forecasting/dataset/ETT-small;7변수 |
| 구간 | train[0,8640), validation targets[8640,11520), test targets[11520,14400) |
| 전처리 | train-only StandardScaler; preceding336 context; window stride1; drop_last=False |
| window 수 | H96 train/val/test8209/2785/2785; H7207585/2161/2161 |
| matrix | 각 구조2dataset×2horizon×3seed×2population×3read=72; 전체4구조288 |
| seed | 7,13,21; deterministic algorithms/cudnn;TF32off;CPUthreads2 |
| 모델 | seq336,patch8,D32,K4,head32/flatten,tau2..16,threshold1,gamma.05,temp.25 |
| optimizer | AdamW lr.001,weight_decay.01,batch128,clip1,MSE loss |
| 학습 종료 | 최대30epoch; early stopping patience6 |
| scheduler | ReduceLROnPlateau factor.5,patience2; validation MSE 추적 |
| checkpoint | 엄격한 최소 validation MSE; 복원/재검증 후 전체test |
| metric | train-standardized MSE/MAE; 모든 window×horizon×channel, float64 error accumulation |
| macro | 각 seed의2dataset 평균을 먼저 구한 뒤3seed mean/sample SD; horizon 별개 |
| paired | 같은 data/seed/population에서 sparse−dense, sparse−off, dense−off |
| 환경 | Python3.10.18,torch1.12.0+cu113,SpikingJelly0.0.0.0.14,numpy1.26.4,pandas2.3.1,sklearn1.7.1,TB2.19.0 |
| 장치 | RTX A6000 48GiB×4, GPU0–3/각2workers; shared GPU wall time은 효율 benchmark 아님 |

v1은 최대10epoch/early3인 짧은 예산이었다. v1 patch의 last-head 보조 실험은 seed7만 수행했으며 v2 main matrix에 포함하지 않는다. 실제 각 run의 명령/config/source·data SHA/환경은 아래 raw JSON 링크에 있다.

<a id="handoff-v1"></a>
## 7. 완료 결과 요약 — v1 (수정 전 dense pilot)

표는 flatten/3seed 본 실험만이며 last-head16회는 개별 결과 부록에 분리 표시했다. v1 결과를 sparse 원안의 성패로 해석하지 않는다.


### v1 patch H96

| Variant | MSE ± SD | MAE ± SD |
| --- | --- | --- |
| heterogeneous_no_memory | 0.385644 ± 0.021461 | 0.423791 ± 0.016197 |
| heterogeneous_retrieval | 0.382667 ± 0.018490 | 0.420998 ± 0.014005 |
| homogeneous_no_memory | 0.407324 ± 0.017954 | 0.438958 ± 0.011965 |
| homogeneous_retrieval | 0.397969 ± 0.005556 | 0.432702 ± 0.003111 |

원본: [v1 patch H96 REPORT](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/REPORT.md)

### v1 patch H720

| Variant | MSE ± SD | MAE ± SD |
| --- | --- | --- |
| heterogeneous_no_memory | 0.679887 ± 0.053368 | 0.580342 ± 0.022406 |
| heterogeneous_retrieval | 0.711945 ± 0.031368 | 0.592568 ± 0.014067 |
| homogeneous_no_memory | 0.717480 ± 0.033274 | 0.602915 ± 0.016036 |
| homogeneous_retrieval | 0.736275 ± 0.041440 | 0.611361 ± 0.018349 |

원본: [v1 patch H720 REPORT](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/REPORT.md)

### v1 TCN H96

| Variant | MSE ± SD | MAE ± SD |
| --- | --- | --- |
| heterogeneous_no_memory | 0.380509 ± 0.003947 | 0.417491 ± 0.004645 |
| heterogeneous_retrieval | 0.381907 ± 0.008823 | 0.419841 ± 0.007770 |
| homogeneous_no_memory | 0.397856 ± 0.024672 | 0.430200 ± 0.013963 |
| homogeneous_retrieval | 0.387563 ± 0.013150 | 0.425702 ± 0.009946 |

원본: [v1 TCN H96 REPORT](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/REPORT.md)

### v1 TCN H720

| Variant | MSE ± SD | MAE ± SD |
| --- | --- | --- |
| heterogeneous_no_memory | 0.800043 ± 0.046684 | 0.628503 ± 0.022646 |
| heterogeneous_retrieval | 0.809756 ± 0.031170 | 0.631460 ± 0.010588 |
| homogeneous_no_memory | 0.787103 ± 0.062192 | 0.624228 ± 0.019320 |
| homogeneous_retrieval | 0.811149 ± 0.040830 | 0.627735 ± 0.012471 |

원본: [v1 TCN H720 REPORT](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/REPORT.md)

<a id="handoff-v2"></a>
## 8. 완료 결과 요약 — v2 1차 patch

H96/H720 각각36회 학습과 artifact/read-mass audit 및 대표 checkpoint fresh 평가가 끝났다. 체크포인트 선택에 test metric을 사용하지 않았다.


### v2 patch H96

| Variant | MSE ± SD | MAE ± SD |
| --- | --- | --- |
| heterogeneous_dense | 0.377628 ± 0.008199 | 0.419095 ± 0.006241 |
| heterogeneous_no_memory | 0.378484 ± 0.011626 | 0.419706 ± 0.008360 |
| heterogeneous_sparse | 0.381363 ± 0.005930 | 0.421432 ± 0.004041 |
| homogeneous_dense | 0.396104 ± 0.008466 | 0.431852 ± 0.005043 |
| homogeneous_no_memory | 0.395326 ± 0.006549 | 0.431763 ± 0.003007 |
| homogeneous_sparse | 0.391533 ± 0.003132 | 0.427766 ± 0.001204 |

데이터셋별 상세 평균:

| Dataset | Variant | MSE ± SD | MAE ± SD |
| --- | --- | --- | --- |
| ETTh1 | heterogeneous_dense | 0.426393 ± 0.004418 | 0.445142 ± 0.004326 |
| ETTh1 | heterogeneous_no_memory | 0.423974 ± 0.005129 | 0.443024 ± 0.004525 |
| ETTh1 | heterogeneous_sparse | 0.426684 ± 0.005367 | 0.444722 ± 0.004513 |
| ETTh1 | homogeneous_dense | 0.449270 ± 0.003327 | 0.458933 ± 0.001549 |
| ETTh1 | homogeneous_no_memory | 0.447416 ± 0.001958 | 0.458227 ± 0.003037 |
| ETTh1 | homogeneous_sparse | 0.447584 ± 0.005594 | 0.456953 ± 0.003783 |
| ETTh2 | heterogeneous_dense | 0.328862 ± 0.012632 | 0.393047 ± 0.008857 |
| ETTh2 | heterogeneous_no_memory | 0.332993 ± 0.019259 | 0.396389 ± 0.013671 |
| ETTh2 | heterogeneous_sparse | 0.336041 ± 0.007756 | 0.398143 ± 0.004906 |
| ETTh2 | homogeneous_dense | 0.342938 ± 0.013875 | 0.404772 ± 0.008887 |
| ETTh2 | homogeneous_no_memory | 0.343235 ± 0.012793 | 0.405299 ± 0.007531 |
| ETTh2 | homogeneous_sparse | 0.335482 ± 0.000705 | 0.398578 ± 0.001514 |

선택 동작 (각 run의 첫8test windows, 최종 population 층):

| Variant | Support density | Empty read fraction | Real mass |
| --- | --- | --- | --- |
| heterogeneous_dense | 0.998776 | 0.000000 | 0.881238 |
| heterogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| heterogeneous_sparse | 0.326631 | 0.149941 | 0.772621 |
| homogeneous_dense | 0.998048 | 0.000000 | 0.838302 |
| homogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| homogeneous_sparse | 0.390572 | 0.107705 | 0.825785 |

같은 checkpoint의 개입 ΔMSE=개입−원래검색 (음수이면 개입이 개선):

| Variant | Off | Uniform | Recent |
| --- | --- | --- | --- |
| heterogeneous_dense | +0.005519 | +0.000316 | +0.000343 |
| heterogeneous_sparse | +0.006680 | -0.000742 | -0.000366 |
| homogeneous_dense | +0.007340 | +0.001834 | +0.003281 |
| homogeneous_sparse | +0.008036 | +0.001088 | +0.002993 |

최대30epoch 도달 1/36; best epoch(0-based) 1–29. 전체 정밀 통계: [REPORT.md](selective-v2-20260914_patch_p96/REPORT.md), [per_run.csv](selective-v2-20260914_patch_p96/per_run.csv), [paired_macro_by_seed.csv](selective-v2-20260914_patch_p96/paired_macro_by_seed.csv), [check_summary.json](selective-v2-20260914_patch_p96/check_summary.json), [check_read_mass.json](selective-v2-20260914_patch_p96/check_read_mass.json), [check_reload.json](selective-v2-20260914_patch_p96/check_reload.json).

### v2 patch H720

| Variant | MSE ± SD | MAE ± SD |
| --- | --- | --- |
| heterogeneous_dense | 0.706692 ± 0.056545 | 0.591711 ± 0.023375 |
| heterogeneous_no_memory | 0.685652 ± 0.053363 | 0.582620 ± 0.021507 |
| heterogeneous_sparse | 0.721476 ± 0.038874 | 0.597785 ± 0.016600 |
| homogeneous_dense | 0.713817 ± 0.042492 | 0.601710 ± 0.019110 |
| homogeneous_no_memory | 0.681632 ± 0.055167 | 0.588881 ± 0.020140 |
| homogeneous_sparse | 0.649910 ± 0.034544 | 0.575184 ± 0.013379 |

데이터셋별 상세 평균:

| Dataset | Variant | MSE ± SD | MAE ± SD |
| --- | --- | --- | --- |
| ETTh1 | heterogeneous_dense | 0.560415 ± 0.020543 | 0.550091 ± 0.011353 |
| ETTh1 | heterogeneous_no_memory | 0.553551 ± 0.023521 | 0.545229 ± 0.014516 |
| ETTh1 | heterogeneous_sparse | 0.560954 ± 0.019902 | 0.550680 ± 0.010581 |
| ETTh1 | homogeneous_dense | 0.583428 ± 0.015605 | 0.562644 ± 0.008291 |
| ETTh1 | homogeneous_no_memory | 0.581082 ± 0.031239 | 0.564492 ± 0.013720 |
| ETTh1 | homogeneous_sparse | 0.587875 ± 0.024077 | 0.565376 ± 0.013857 |
| ETTh2 | heterogeneous_dense | 0.852969 ± 0.109647 | 0.633332 ± 0.045942 |
| ETTh2 | heterogeneous_no_memory | 0.817753 ± 0.088790 | 0.620010 ± 0.034376 |
| ETTh2 | heterogeneous_sparse | 0.881998 ± 0.072022 | 0.644889 ± 0.031179 |
| ETTh2 | homogeneous_dense | 0.844206 ± 0.100586 | 0.640777 ± 0.046028 |
| ETTh2 | homogeneous_no_memory | 0.782183 ± 0.098968 | 0.613270 ± 0.036416 |
| ETTh2 | homogeneous_sparse | 0.711944 ± 0.045112 | 0.584991 ± 0.012949 |

선택 동작 (각 run의 첫8test windows, 최종 population 층):

| Variant | Support density | Empty read fraction | Real mass |
| --- | --- | --- | --- |
| heterogeneous_dense | 0.999274 | 0.000000 | 0.881230 |
| heterogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| heterogeneous_sparse | 0.385989 | 0.091833 | 0.866983 |
| homogeneous_dense | 0.997401 | 0.000000 | 0.898756 |
| homogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| homogeneous_sparse | 0.456939 | 0.057825 | 0.913030 |

같은 checkpoint의 개입 ΔMSE=개입−원래검색 (음수이면 개입이 개선):

| Variant | Off | Uniform | Recent |
| --- | --- | --- | --- |
| heterogeneous_dense | -0.013100 | +0.002695 | +0.016542 |
| heterogeneous_sparse | -0.008923 | -0.001135 | +0.008103 |
| homogeneous_dense | -0.002572 | +0.001208 | +0.009902 |
| homogeneous_sparse | -0.007048 | +0.002827 | +0.007722 |

최대30epoch 도달 2/36; best epoch(0-based) 0–29. 전체 정밀 통계: [REPORT.md](selective-v2-20260914_patch_p720/REPORT.md), [per_run.csv](selective-v2-20260914_patch_p720/per_run.csv), [paired_macro_by_seed.csv](selective-v2-20260914_patch_p720/paired_macro_by_seed.csv), [check_summary.json](selective-v2-20260914_patch_p720/check_summary.json), [check_read_mass.json](selective-v2-20260914_patch_p720/check_read_mass.json), [check_reload.json](selective-v2-20260914_patch_p720/check_reload.json).

### 현재까지의 해석

- H96에서 검색 없는 이질성은 MSE −4.26%지만 H720은 +.59%다. 이질성의 이득이 모든 horizon에서 유지된다는 결론은 아니다.
- H720 hetero dense는 off 대비 +3.07%, sparse는 +5.22%/dense 대비 +2.09%다. 이질성+선택적 검색의 추가 이득은 아직 확인되지 않았다.
- H720 homo sparse는 homo off 대비 −4.65%이며6조건 중 최저 macro MSE다. ETTh1은 .581082→.587875로 악화, ETTh2는 .782183→.711944로 개선해 dataset 의존적이다.
- H720 hetero sparse는 support38.60%/empty9.18%로 선택 기능을 사용한다. 하지만 같은 checkpoint의 검색-off ΔMSE−.008923은 이 조건의 검색이 도움이 되지 않는 경우가 있음을 보여준다. 반대로 H96에서는 같은 개입의 ΔMSE+.006680이다.
- 검색 학습 모델에서 검색을 끄는 개입과, 검색 없는 모델을 처음부터 별도로 학습하는 대조는 다르다. Uniform/recent는 null을 우회하여 real slot 선택을 강제하고 gate를 다시 계산한다. 이들은 별도 재학습 대조가 아니다.
- 표본2dataset/3seed; 통계적 유의성 검정은 not run. Support/lag 진단은 첫8windows만이므로 전체test의 검색 동작이나 정답 memory 선택을 보증하지 않는다.

<a id="handoff-progress"></a>
## 9. 현재 2차 v2 Spike-TCN 진척도

스냅샷 2026-09-15 13:59:24 KST: H96 완료16/36 (44.4%), 실행8, 대기12, 실패0. H96+H720 총72회 기준 완료율은 22.2%다. H720은 아직 not run.

| Run | PID | Physical GPU | PID alive | 마지막 stdout |
| --- | --- | --- | --- | --- |
| tcn_ETTh1_p96_flatten_heterogeneous_dense_seed13 | 1426930 | 1 | True | EarlyStopping counter: 5 out of 6 |
| tcn_ETTh1_p96_flatten_heterogeneous_sparse_seed13 | 1429826 | 0 | True | EarlyStopping counter: 4 out of 6 |
| tcn_ETTh2_p96_flatten_homogeneous_no_memory_seed13 | 1430025 | 0 | True | EarlyStopping counter: 5 out of 6 |
| tcn_ETTh2_p96_flatten_homogeneous_dense_seed13 | 1431173 | 3 | True | EarlyStopping counter: 4 out of 6 |
| tcn_ETTh2_p96_flatten_homogeneous_sparse_seed13 | 1432459 | 3 | True | EarlyStopping counter: 3 out of 6 |
| tcn_ETTh2_p96_flatten_heterogeneous_no_memory_seed13 | 1432651 | 1 | True | EarlyStopping counter: 3 out of 6 |
| tcn_ETTh2_p96_flatten_heterogeneous_dense_seed13 | 1433565 | 2 | True | EarlyStopping counter: 2 out of 6 |
| tcn_ETTh2_p96_flatten_heterogeneous_sparse_seed13 | 1433593 | 2 | True | EarlyStopping counter: 2 out of 6 |

완료된 TCN run의 wall time은 56.0–90.5분이었다. 실제 관측 시간이며 남은 실행은 조기종료/동시GPU 경합/평가시간에 따라 달라지므로 확정 ETA로 쓰지 않는다. 프로세스 생존과 GPU사용은 확인했지만 아직36조건이 모이지 않아 최종 macro/audit는 not run이다. 아래 부록의 TCN 수치는 개별 완료 결과이며 균형 잡힌 전체 구조 비교가 아니다.

Live queue: [selective-v2-20260914_tcn_p96.json](../scripts/queues/selective-v2-20260914_tcn_p96.json); pipeline: [selective-v2-20260914.json](../scripts/queues/selective-v2-20260914.json); stdout: [selective-v2-20260915.resume.stdout](../log/selective-v2-20260915.resume.stdout).

<a id="handoff-recovery"></a>
## 10. 발견한 문제와 복구 이력

1. **v1 선택 mask 누락:** dense all-history softmax였으므로 별도v2에서 sparsemax/null/amplitude-sensitive score/retrieval-aware gate를 구현했다. v1 artifact는 보존했다.
2. **torch1.12 deterministic CUDA backward:** 초기 sparsemax 자동미분 sort/gather의 scatter backward가 실패했다. Sparsemax 논문 Eq.14의 support-centered gradient를 custom backward로 구현한 뒤 CPU/GPU gradcheck와 네 구조 H720 smoke가 통과했다. 실패 smoke `smoke-v2-20260914`와 성공 smoke `smoke-v2-20260914-r2`는 모두 보존하고 성능 main matrix에서 제외했다.
3. **v2 H96 후처리 assertion:** weight 분모 floor 때문에 일부 tiny positive dense mass에서 weight 합<1인데, 초기 audit은 nonempty이면 합1을 가정했다.5개 dense run에서 실패하여 H720 시작 전 pipeline이 멈췄다. 모든36checkpoint에서 `sum(weight)=real_mass/max(real_mass,1e-12)`를 재검증하도록 수정했고 모든 audit/fresh reload가 통과했다. 학습/평가 source10개, checkpoint, raw result는 그대로다. Mean lag/entropy에는 floor 영향이 있으므로 README에 해석 제한을 정정했다.
4. **재개 및 상태:** 완전히 학습된 suite만 재검증하여 다음 미시작 suite로 넘어가는 `--resume`를 추가했다. Partial suite resume은 지원하지 않는다. Queue의 `failure` 항목은 복구 전 과거 이력으로 남아 있다. 현재 상태를 판단할 때 `status`, `stages`, 최근 stdout, 실제 PID를 함께 본다. 새 failure 기록은 시각별 파일로 보존하며 status를 failed로 기록한다.

검증 근거: [check_model_cpu.json](check_model_cpu.json), [check_model_cuda0.json](check_model_cuda0.json), [source_protocol_check.json](source_protocol_check.json)

<a id="handoff-plan"></a>
## 11. 앞으로 실행되는 것과 아직 결정하지 않은 후속 연구

### 이미 구현·설정된 실행 순서

1. TCN H96의36회 완료 → matrix/source/초기값/min-val/checkpoint/CSV/TB/data/진단/통계 검증 → 대표 sparse checkpoint 전체test/개입 재현.
2. TCN H72036회 → 동일 검증. 두 horizon72회 완료 후 canonical 기록/로컬 commit/annotated tag.
3. Hybrid spiking PatchTST H96/H72072회 → 동일 검증/기록.
4. Causal spiking TSMixer H96/H72072회 → 동일 검증/기록.

성능 개선 여부로 다음 구조를 선별하지 않는다. 현재 고정된 예산/seed/control을 유지한다. 모델별 효과는 먼저 같은 구조 안의 paired 비교로 판단하며, backbone 간 다른 용량을 주의한다.

### 필요성은 있지만 이번288회에는 포함하지 않은 후보 (not run)

- 정답 lag/relevant slot/distractor/context switch를 통제한 end-to-end synthetic recall forecasting 학습. 현재는 operator 수준 known-slot 검사만 통과했다.
- TCN의 어느 population 층이 검색 이득/손해를 만드는지 층별 on/off 개입과 별도훈련 대조.
- Uniform/recent read 정책의 재학습, gamma/temperature/null/gate 설정을 따로 분리하는 ablation. Test 결과로 튜닝하는 대신 별도 validation protocol을 먼저 고정해야 한다.
- Backbone matched-capacity/effective-parameter 대조, 더 많은 데이터/seed, 학습 budget 민감도.
- Fractional temporal prior/learned tau/독립 mask rule 등 concept의 미정 설계 비교.
- Sparse use가 실제 search/latency/energy 비용을 줄이는 구현 및 측정.

이 후보들은 앞으로 의논할 내용이며 현재 queue에 들어간 실행으로 오해하면 안 된다.

<a id="handoff-artifacts"></a>
## 12. 실행·저장·이어가기 방법

작업 cwd는 `/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT`이며 Python은 `/home/yschoi/.conda/envs/snn_recall/bin/python`을 사용한다. 기본 system Python으로 실행하지 않는다. 기존 실행 명령은 다음과 같다(현재 실행 중이므로 중복 실행하지 않는다).

```bash
bash f_lif_pop_v2/forecasting/scripts/run_pipeline.sh
# 후처리 복구 때 실제 사용한 명령
bash f_lif_pop_v2/forecasting/scripts/run_pipeline.sh --resume
```

읽기 전용 상태 확인:

```bash
cat f_lif_pop_v2/forecasting/scripts/queues/selective-v2-20260914.json
cat f_lif_pop_v2/forecasting/scripts/queues/selective-v2-20260914_tcn_p96.json
tail -40 f_lif_pop_v2/forecasting/log/selective-v2-20260915.resume.stdout
git status --short
```

학습 중 branch/HEAD와10개 frozen source를 바꾸면 queue가 후속 실행을 중단한다. 현재 기대 HEAD는 위 스냅샷 값이며 이후 stage 자동 commit이 갱신한다. 새 세션에서는 실제 queue 값을 다시 확인한다. 이 임시 문서는 task results 아래 두어 다음 stage 완료 commit에 포함되도록 했고, 중간 commit으로 HEAD를 바꾸지 않는다. Main 통합/remote push는 수행하지 않았다.

Artifact layout:

```text
f_lif_pop_v2/forecasting/
  log/<suite>/<dataset>/<date>/<config>/seed+variant/
    logargs.txt
    log/best_log_0.csv, final+result.csv
    log/train_0/, val_0/                  # TensorBoard
    log/history.json, provenance.json, horizon_metrics.csv, forecast_example.json
    model_state/config.pt, best+model.pt
  results/<suite>/
    <run_id>.json                        # config/command/hash/epoch history/test/diagnostics
    manifest.json, completion.json
    per_run.csv, per_task.csv, macro.csv, paired.csv, paired_macro_by_seed.csv
    layer_diagnostics.csv, aggregate.json, REPORT.md
    check_summary.json, check_read_mass.json, check_reload.json
  scripts/queues/                        # local PID/status/locks/failure records
```

코드/config/docs/text결과는 Git, checkpoint/dataset/events/stdout/cache/queues는 local이다. Raw artifact를 commit 준비 때문에 삭제하지 않는다. 각 run JSON의 config.save_result_path가 exact checkpoint/log 경로다. 프로젝트 이력은 repo-root docs/PROJECT_LOG.md 하나만 canonical이며 NSMT/docs/PROJECT_LOG.md는 같은 파일을 링크한다.

<a id="handoff-identity"></a>
## 13. 재현 식별자와 source snapshot

| Tag | Commit | 성격 |
| --- | --- | --- |
| exp/f-lif-pop-v1-20260911-snapshot | 386539e9f60e1f4cfaa0683a80f5d701e046d5eb | 설계/코드 snapshot; 전체학습 완료 아님 |
| exp/f-lif-pop-v1-20260911-snapshot-2 | 46ec681cff3acf56df5432098872cfce31da4332 | 설계/코드 snapshot; 전체학습 완료 아님 |
| exp/f-lif-pop-v1-20260914 | 277d18f548f23496788c319b4076df66c9f18be7 | 해당 범위 완료 결과 |
| exp/f-lif-pop-v1-h720-20260914 | ed5c8d16fb5e9c56234c58d6a8d870ffe20f4efc | 해당 범위 완료 결과 |
| exp/f-lif-pop-tcn-v1-20260914 | f8215f54106980bad7c782bf08acaef871175c34 | 해당 범위 완료 결과 |
| exp/f-lif-pop-v2-20260914-snapshot | a7d382e5ac540d12c2a6ad9056ac828739680442 | 설계/코드 snapshot; 전체학습 완료 아님 |
| exp/f-lif-pop-v2-patch-h96-20260915 | 9574d2286d26c4502c504da63f349e42a4cb0eaa | 해당 범위 완료 결과 |
| exp/f-lif-pop-v2-patch-20260915 | 8c3d13673ce628bff51e7ab858a67b5bc72584da | 해당 범위 완료 결과 |

v2 branch base `f8215f54106980bad7c782bf08acaef871175c34`; 첫 학습 snapshot `a7d382e5ac540d12c2a6ad9056ac828739680442`; H96 후처리 복구 `9574d2286d26c4502c504da63f349e42a4cb0eaa`. Horizon별 training commit은 각 completion.json에 있다. 아래10개 source hash는 재개 전후 동일하다.

| Training file | SHA256 |
| --- | --- |
| config.py | 1c382a084dd1fae40e88e43f797ba690e2824660937c4b7127ca4cead8ce8f18 |
| model.py | 74bd444c9352fd315769d37cc5d8262ba51d12b21e628d366bcf2933d98b2388 |
| ours.py | 32a1ee1a23b7b689b218c1a631e73b37e68d69449ac6697b4d405840c2f654be |
| layers.py | 81240bae1d02521f5ceb530370b99dc1d28b2323f67447a47fc6b08119b279cd |
| backbones.py | c7743de12a4f24acd0d43647f3643d69a5afd6088e669b11947be03df2018ce0 |
| train.py | 1b9428bbc505eadeaa86f0307235ff50a46b563e34592a46a2ae0dc8c335d2bd |
| test.py | cb3dd8140cb22ea858a42698630388cdc475292207e98717e730eab7d2c6f538 |
| utils.py | 01a5f85fc6f339a71e3555a7cdd2aa928f0640d723e2202e9cf874abd725c25a |
| data_provider/data_factory.py | a31983af3eb03f6f4957d6ecceebb8bddeeeeaed8f6ecd8675cddbdca5c9efcc |
| data_provider/data_loader.py | 8231657c1691e7f55c9867d24d4230d7a69b820f61b24ef399eaf82b6403689d |

사용자 원문 SHA256:

| 문서 | SHA256 |
| --- | --- |
| population_selective_membrane_memory_snn_concept.md | 01355e2a9cb0766632a52095d88a1676c845d4b4041695aea791b3eabbc01336 |
| PopulationLIF_implementation_review.md | aa0b6a52a65ca7a0a50d4d5019ab892527393759a822b29b26e3452fdb34f81d |

원문 두 파일은 기존 untracked 상태로 보존한다. 임시문서의 정확한 숫자는 raw JSON에서 읽었고, 아래 개별 표는 float를17자리 유효숫자로 출력한다. 요약표의6자리 반올림과 구별한다.

<a id="handoff-runs"></a>
## 14. 개별 실행 상세 결과

Best epoch은0-based이고 Epochs는 실제 학습 횟수다. Val MSE는 최저 validation MSE다. Run 링크는 전체 config/정확한 실행명령/학습history/환경/hash/log·checkpoint 경로/개입·진단을 가진 원시 JSON이다. v1 last-head는 run_id에 last로 표시되며 seed7만이다.


### v1 patch H96 — 32개 완료

| Run/raw JSON | Test MSE | Test MAE | Val MSE | Best epoch | Epochs | Parameters | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ETTh1_p96_flatten_heterogeneous_no_memory_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_heterogeneous_no_memory_seed13.json) | 0.43249315210069922 | 0.44880423435217076 | 0.71848516665321926 | 6 | 10 | 133573 | 21.757811546325684 |
| [ETTh1_p96_flatten_heterogeneous_no_memory_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_heterogeneous_no_memory_seed21.json) | 0.41896662047908523 | 0.43843203669528136 | 0.70573929538729352 | 6 | 10 | 133573 | 20.390916585922241 |
| [ETTh1_p96_flatten_heterogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_heterogeneous_no_memory_seed7.json) | 0.42129931812929211 | 0.44166549986403369 | 0.72040224395619656 | 7 | 10 | 133573 | 19.762920379638672 |
| [ETTh1_p96_flatten_heterogeneous_retrieval_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_heterogeneous_retrieval_seed13.json) | 0.43607927398963509 | 0.45069145257494986 | 0.72511812199936021 | 6 | 10 | 133573 | 54.351420164108276 |
| [ETTh1_p96_flatten_heterogeneous_retrieval_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_heterogeneous_retrieval_seed21.json) | 0.4193928821054736 | 0.43855622986649095 | 0.71341286972505402 | 6 | 10 | 133573 | 48.024216890335083 |
| [ETTh1_p96_flatten_heterogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_heterogeneous_retrieval_seed7.json) | 0.42291676222201197 | 0.44125944728019484 | 0.72492044631359764 | 7 | 10 | 133573 | 52.403419256210327 |
| [ETTh1_p96_flatten_homogeneous_no_memory_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_homogeneous_no_memory_seed13.json) | 0.45797998667735951 | 0.46348953588199698 | 0.78049984409542617 | 9 | 10 | 133573 | 21.214222431182861 |
| [ETTh1_p96_flatten_homogeneous_no_memory_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_homogeneous_no_memory_seed21.json) | 0.44922338766510739 | 0.45753000282844747 | 0.77282497496619962 | 9 | 10 | 133573 | 20.347854852676392 |
| [ETTh1_p96_flatten_homogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_homogeneous_no_memory_seed7.json) | 0.4554451509033754 | 0.46206950482159131 | 0.78212311543132718 | 8 | 10 | 133573 | 20.638721227645874 |
| [ETTh1_p96_flatten_homogeneous_retrieval_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_homogeneous_retrieval_seed13.json) | 0.46779018420403334 | 0.46882582148177782 | 0.80030075937622314 | 9 | 10 | 133573 | 53.616886138916016 |
| [ETTh1_p96_flatten_homogeneous_retrieval_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_homogeneous_retrieval_seed21.json) | 0.46402570697568096 | 0.4669981686393766 | 0.79924153237416784 | 9 | 10 | 133573 | 48.955310344696045 |
| [ETTh1_p96_flatten_homogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_flatten_homogeneous_retrieval_seed7.json) | 0.4481480426276051 | 0.45706599689176136 | 0.79161363226071024 | 8 | 10 | 133573 | 53.381699323654175 |
| [ETTh1_p96_last_heterogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_last_heterogeneous_no_memory_seed7.json) | 0.73058425237666402 | 0.60102374312039997 | 1.0386154030640313 | 6 | 10 | 7621 | 22.050432205200195 |
| [ETTh1_p96_last_heterogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_last_heterogeneous_retrieval_seed7.json) | 0.74573131232244527 | 0.6032364074999802 | 1.0533457921454519 | 3 | 7 | 7621 | 38.392600536346436 |
| [ETTh1_p96_last_homogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_last_homogeneous_no_memory_seed7.json) | 0.84658937817790036 | 0.65753616188778863 | 1.0964960586706369 | 0 | 4 | 7621 | 8.7693455219268799 |
| [ETTh1_p96_last_homogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh1_p96_last_homogeneous_retrieval_seed7.json) | 0.80753311315099618 | 0.63696598437373875 | 1.0909895749237217 | 1 | 5 | 7621 | 29.6194007396698 |
| [ETTh2_p96_flatten_heterogeneous_no_memory_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_heterogeneous_no_memory_seed13.json) | 0.3874704721294443 | 0.43490332130760656 | 0.2714063577670035 | 1 | 5 | 133573 | 12.611171007156372 |
| [ETTh2_p96_flatten_heterogeneous_no_memory_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_heterogeneous_no_memory_seed21.json) | 0.31989985047621161 | 0.38268254999163975 | 0.24780992367458449 | 1 | 5 | 133573 | 10.584928512573242 |
| [ETTh2_p96_flatten_heterogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_heterogeneous_no_memory_seed7.json) | 0.333736023498373 | 0.39625781682384575 | 0.25807071364810191 | 1 | 5 | 133573 | 11.379686117172241 |
| [ETTh2_p96_flatten_heterogeneous_retrieval_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_heterogeneous_retrieval_seed13.json) | 0.37088122261303319 | 0.42227404010630537 | 0.26467603797816397 | 1 | 5 | 133573 | 25.733312606811523 |
| [ETTh2_p96_flatten_heterogeneous_retrieval_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_heterogeneous_retrieval_seed21.json) | 0.31688138478526146 | 0.37988065759983336 | 0.24663174425531573 | 1 | 5 | 133573 | 28.188334226608276 |
| [ETTh2_p96_flatten_heterogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_heterogeneous_retrieval_seed7.json) | 0.32984962009361374 | 0.3933288297681467 | 0.25458003485636588 | 1 | 5 | 133573 | 28.585755825042725 |
| [ETTh2_p96_flatten_homogeneous_no_memory_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_homogeneous_no_memory_seed13.json) | 0.39654660208813736 | 0.44082114189262861 | 0.27584367202986904 | 1 | 5 | 133573 | 10.690802812576294 |
| [ETTh2_p96_flatten_homogeneous_no_memory_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_homogeneous_no_memory_seed21.json) | 0.33565318125389965 | 0.40010578274710745 | 0.26089040501277044 | 1 | 5 | 133573 | 10.749848127365112 |
| [ETTh2_p96_flatten_homogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_homogeneous_no_memory_seed7.json) | 0.34909344896782246 | 0.40973386095500985 | 0.26949853201838597 | 1 | 5 | 133573 | 10.389272451400757 |
| [ETTh2_p96_flatten_homogeneous_retrieval_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_homogeneous_retrieval_seed13.json) | 0.33597783160101136 | 0.40036999394004236 | 0.26180678462417084 | 7 | 10 | 133573 | 48.983968734741211 |
| [ETTh2_p96_flatten_homogeneous_retrieval_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_homogeneous_retrieval_seed21.json) | 0.33679915944794581 | 0.40179194282710434 | 0.2611510186752079 | 1 | 5 | 133573 | 26.038981914520264 |
| [ETTh2_p96_flatten_homogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_flatten_homogeneous_retrieval_seed7.json) | 0.33507196945757228 | 0.40115713161147437 | 0.2675639225604729 | 1 | 5 | 133573 | 26.481744527816772 |
| [ETTh2_p96_last_heterogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_last_heterogeneous_no_memory_seed7.json) | 0.50089805942468768 | 0.50687428770412435 | 0.31649669062542157 | 4 | 8 | 7621 | 17.239566802978516 |
| [ETTh2_p96_last_heterogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_last_heterogeneous_retrieval_seed7.json) | 0.4971457635745003 | 0.50437646115437629 | 0.30892700283063351 | 4 | 8 | 7621 | 39.042295455932617 |
| [ETTh2_p96_last_homogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_last_homogeneous_no_memory_seed7.json) | 0.56560441599479605 | 0.53810179642157741 | 0.33002770700017581 | 7 | 10 | 7621 | 21.103277444839478 |
| [ETTh2_p96_last_homogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-20260914/ETTh2_p96_last_homogeneous_retrieval_seed7.json) | 0.56484248115818381 | 0.53691771638592201 | 0.32868839930470695 | 7 | 10 | 7621 | 54.1298668384552 |

### v1 patch H720 — 32개 완료

| Run/raw JSON | Test MSE | Test MAE | Val MSE | Best epoch | Epochs | Parameters | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ETTh1_p720_flatten_heterogeneous_no_memory_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_heterogeneous_no_memory_seed13.json) | 0.57923668943689677 | 0.55849439806635393 | 1.2282757960850048 | 3 | 7 | 972853 | 13.001743793487549 |
| [ETTh1_p720_flatten_heterogeneous_no_memory_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_heterogeneous_no_memory_seed21.json) | 0.54461597890983759 | 0.538896955989003 | 1.1982406597380943 | 4 | 8 | 972853 | 16.425410747528076 |
| [ETTh1_p720_flatten_heterogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_heterogeneous_no_memory_seed7.json) | 0.55882511696474801 | 0.54523100894831678 | 1.2232909217603531 | 3 | 7 | 972853 | 14.742236137390137 |
| [ETTh1_p720_flatten_heterogeneous_retrieval_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_heterogeneous_retrieval_seed13.json) | 0.57785361598376506 | 0.55589353573102684 | 1.2300454966032499 | 4 | 8 | 972853 | 37.653014898300171 |
| [ETTh1_p720_flatten_heterogeneous_retrieval_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_heterogeneous_retrieval_seed21.json) | 0.54737313175552194 | 0.54020816842280805 | 1.1949511950458611 | 4 | 8 | 972853 | 39.792474269866943 |
| [ETTh1_p720_flatten_heterogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_heterogeneous_retrieval_seed7.json) | 0.59695190032828926 | 0.56389122620214427 | 1.2211276021613398 | 4 | 8 | 972853 | 37.953068733215332 |
| [ETTh1_p720_flatten_homogeneous_no_memory_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_homogeneous_no_memory_seed13.json) | 0.58235123460073424 | 0.56267169226488611 | 1.2948096856485429 | 8 | 10 | 972853 | 20.032006978988647 |
| [ETTh1_p720_flatten_homogeneous_no_memory_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_homogeneous_no_memory_seed21.json) | 0.5624945769882177 | 0.55652184083657585 | 1.2637677067684581 | 8 | 10 | 972853 | 19.019868612289429 |
| [ETTh1_p720_flatten_homogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_homogeneous_no_memory_seed7.json) | 0.550437147627721 | 0.5441097376774422 | 1.2753069354401829 | 9 | 10 | 972853 | 20.895761013031006 |
| [ETTh1_p720_flatten_homogeneous_retrieval_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_homogeneous_retrieval_seed13.json) | 0.59192284885529101 | 0.56992367210433947 | 1.3053387047662286 | 8 | 10 | 972853 | 45.210730791091919 |
| [ETTh1_p720_flatten_homogeneous_retrieval_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_homogeneous_retrieval_seed21.json) | 0.58221340429286617 | 0.56512332701464663 | 1.2756026829100715 | 6 | 10 | 972853 | 46.25495719909668 |
| [ETTh1_p720_flatten_homogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_flatten_homogeneous_retrieval_seed7.json) | 0.61603415983607601 | 0.57433568031994198 | 1.2742119296398406 | 7 | 10 | 972853 | 45.85838770866394 |
| [ETTh1_p720_last_heterogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_last_heterogeneous_no_memory_seed7.json) | 0.7761046797663308 | 0.65128798784808595 | 1.485355589048996 | 8 | 10 | 28213 | 19.57261061668396 |
| [ETTh1_p720_last_heterogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_last_heterogeneous_retrieval_seed7.json) | 0.77343736550923115 | 0.65111316716468159 | 1.4830121470207631 | 5 | 9 | 28213 | 43.546115159988403 |
| [ETTh1_p720_last_homogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_last_homogeneous_no_memory_seed7.json) | 0.8165548408604717 | 0.66977738040217571 | 1.5303278455926022 | 7 | 10 | 28213 | 19.843559503555298 |
| [ETTh1_p720_last_homogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh1_p720_last_homogeneous_retrieval_seed7.json) | 0.81525717654593854 | 0.66929426954664029 | 1.5278842302848064 | 7 | 10 | 28213 | 44.771973371505737 |
| [ETTh2_p720_flatten_heterogeneous_no_memory_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_heterogeneous_no_memory_seed13.json) | 0.84388682346250754 | 0.62268242087007586 | 0.66113677906126755 | 2 | 6 | 972853 | 12.573973178863525 |
| [ETTh2_p720_flatten_heterogeneous_no_memory_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_heterogeneous_no_memory_seed21.json) | 0.69192689298727394 | 0.57039297052863913 | 0.66797826675548788 | 2 | 6 | 972853 | 11.698179721832275 |
| [ETTh2_p720_flatten_heterogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_heterogeneous_no_memory_seed7.json) | 0.86083181302947298 | 0.64635640427225416 | 0.66636503796314317 | 1 | 5 | 972853 | 10.731648921966553 |
| [ETTh2_p720_flatten_heterogeneous_retrieval_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_heterogeneous_retrieval_seed13.json) | 0.87950155590710799 | 0.63360649943635161 | 0.66322668559697218 | 2 | 6 | 972853 | 31.742126703262329 |
| [ETTh2_p720_flatten_heterogeneous_retrieval_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_heterogeneous_retrieval_seed21.json) | 0.80414327484784387 | 0.61486656962773045 | 0.66648549702232651 | 2 | 6 | 972853 | 29.307544946670532 |
| [ETTh2_p720_flatten_heterogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_heterogeneous_retrieval_seed7.json) | 0.86584558919526144 | 0.64693942464407661 | 0.66136144311441281 | 1 | 5 | 972853 | 26.985006809234619 |
| [ETTh2_p720_flatten_homogeneous_no_memory_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_homogeneous_no_memory_seed13.json) | 0.77663605815209724 | 0.60837163824823304 | 0.66708013569749725 | 2 | 6 | 972853 | 12.693254470825195 |
| [ETTh2_p720_flatten_homogeneous_no_memory_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_homogeneous_no_memory_seed21.json) | 0.92044504518843884 | 0.6777029279599226 | 0.67424150009494133 | 0 | 4 | 972853 | 9.3934228420257568 |
| [ETTh2_p720_flatten_homogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_homogeneous_no_memory_seed7.json) | 0.91251608273392038 | 0.66810998860712922 | 0.67802123185111796 | 1 | 5 | 972853 | 11.402848720550537 |
| [ETTh2_p720_flatten_homogeneous_retrieval_seed13](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_homogeneous_retrieval_seed13.json) | 0.79021044445183697 | 0.61090836852742392 | 0.66995304870324524 | 2 | 6 | 972853 | 29.76710033416748 |
| [ETTh2_p720_flatten_homogeneous_retrieval_seed21](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_homogeneous_retrieval_seed21.json) | 0.90838405602815897 | 0.67301109874980103 | 0.67821647294916299 | 0 | 4 | 972853 | 20.905413150787354 |
| [ETTh2_p720_flatten_homogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_flatten_homogeneous_retrieval_seed7.json) | 0.92888405661438622 | 0.67486528417932268 | 0.67493514189021198 | 1 | 5 | 972853 | 25.34603476524353 |
| [ETTh2_p720_last_heterogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_last_heterogeneous_no_memory_seed7.json) | 1.0718776979692104 | 0.75934453390221035 | 0.72183546919034414 | 3 | 7 | 28213 | 14.297825336456299 |
| [ETTh2_p720_last_heterogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_last_heterogeneous_retrieval_seed7.json) | 1.0744148692763347 | 0.76107863659640151 | 0.72006077295600035 | 3 | 7 | 28213 | 39.277795076370239 |
| [ETTh2_p720_last_homogeneous_no_memory_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_last_homogeneous_no_memory_seed7.json) | 1.1244522318146319 | 0.77979921700722943 | 0.73651736644199417 | 3 | 7 | 28213 | 13.507673501968384 |
| [ETTh2_p720_last_homogeneous_retrieval_seed7](../../../f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/ETTh2_p720_last_homogeneous_retrieval_seed7.json) | 1.1234084359565666 | 0.78026027834222544 | 0.73985897916673082 | 3 | 7 | 28213 | 32.640618801116943 |

### v1 TCN H96 — 24개 완료

| Run/raw JSON | Test MSE | Test MAE | Val MSE | Best epoch | Epochs | Parameters | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ETTh1_p96_flatten_heterogeneous_no_memory_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_heterogeneous_no_memory_seed13.json) | 0.42796979154797032 | 0.44579800265987057 | 0.72496397477979069 | 2 | 6 | 158287 | 235.02734088897705 |
| [ETTh1_p96_flatten_heterogeneous_no_memory_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_heterogeneous_no_memory_seed21.json) | 0.42582709988904488 | 0.44582753983545009 | 0.69938738275598955 | 7 | 10 | 158287 | 327.01747035980225 |
| [ETTh1_p96_flatten_heterogeneous_no_memory_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_heterogeneous_no_memory_seed7.json) | 0.41819455567820096 | 0.44096140888115126 | 0.72691159537347383 | 1 | 5 | 158287 | 172.0588161945343 |
| [ETTh1_p96_flatten_heterogeneous_retrieval_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_heterogeneous_retrieval_seed13.json) | 0.42468791025718866 | 0.44455941407227184 | 0.72585718565469459 | 2 | 6 | 158287 | 302.16316699981689 |
| [ETTh1_p96_flatten_heterogeneous_retrieval_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_heterogeneous_retrieval_seed21.json) | 0.42964064373552296 | 0.45021614550697475 | 0.69516445903814705 | 7 | 10 | 158287 | 472.55181741714478 |
| [ETTh1_p96_flatten_heterogeneous_retrieval_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_heterogeneous_retrieval_seed7.json) | 0.43145043395147503 | 0.45076928866619442 | 0.72231713542794274 | 4 | 8 | 158287 | 407.99526596069336 |
| [ETTh1_p96_flatten_homogeneous_no_memory_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_homogeneous_no_memory_seed13.json) | 0.4357113181958493 | 0.44987745935171486 | 0.74757496235176379 | 4 | 8 | 158287 | 261.76126432418823 |
| [ETTh1_p96_flatten_homogeneous_no_memory_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_homogeneous_no_memory_seed21.json) | 0.43060500522767586 | 0.45076026967869282 | 0.72637157768602634 | 5 | 9 | 158287 | 349.74959540367126 |
| [ETTh1_p96_flatten_homogeneous_no_memory_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_homogeneous_no_memory_seed7.json) | 0.4237245374765003 | 0.44185972733279205 | 0.74502548606429364 | 5 | 9 | 158287 | 302.9955689907074 |
| [ETTh1_p96_flatten_homogeneous_retrieval_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_homogeneous_retrieval_seed13.json) | 0.43425104436799911 | 0.45065442162765584 | 0.75265341009884068 | 7 | 10 | 158287 | 471.45107436180115 |
| [ETTh1_p96_flatten_homogeneous_retrieval_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_homogeneous_retrieval_seed21.json) | 0.4486783566831597 | 0.46520280968059624 | 0.73747972883258939 | 6 | 10 | 158287 | 470.06192207336426 |
| [ETTh1_p96_flatten_homogeneous_retrieval_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh1_p96_flatten_homogeneous_retrieval_seed7.json) | 0.43354365263856254 | 0.45042311165595361 | 0.75552610453497171 | 2 | 6 | 158287 | 300.07315754890442 |
| [ETTh2_p96_flatten_heterogeneous_no_memory_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_heterogeneous_no_memory_seed13.json) | 0.32678023688332919 | 0.38103382211946085 | 0.24142033606428617 | 1 | 5 | 158287 | 179.49111795425415 |
| [ETTh2_p96_flatten_heterogeneous_no_memory_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_heterogeneous_no_memory_seed21.json) | 0.33259077987908808 | 0.38718935693225615 | 0.24799385762221016 | 1 | 5 | 158287 | 185.97861313819885 |
| [ETTh2_p96_flatten_heterogeneous_no_memory_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_heterogeneous_no_memory_seed7.json) | 0.35168856419514316 | 0.40413425018140314 | 0.26448790029841485 | 3 | 7 | 158287 | 263.33745980262756 |
| [ETTh2_p96_flatten_heterogeneous_retrieval_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_heterogeneous_retrieval_seed13.json) | 0.3233361077213544 | 0.38103130429476545 | 0.24132362623452325 | 1 | 5 | 158287 | 266.61133027076721 |
| [ETTh2_p96_flatten_heterogeneous_retrieval_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_heterogeneous_retrieval_seed21.json) | 0.33091295011859717 | 0.38688864803117373 | 0.24607383344903905 | 1 | 5 | 158287 | 268.80487608909607 |
| [ETTh2_p96_flatten_heterogeneous_retrieval_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_heterogeneous_retrieval_seed7.json) | 0.3514119875964154 | 0.40557879793151125 | 0.25948508092532829 | 4 | 8 | 158287 | 372.88064098358154 |
| [ETTh2_p96_flatten_homogeneous_no_memory_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_homogeneous_no_memory_seed13.json) | 0.41216492272327604 | 0.43954946785504795 | 0.27766648034355979 | 5 | 9 | 158287 | 320.31625008583069 |
| [ETTh2_p96_flatten_homogeneous_no_memory_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_homogeneous_no_memory_seed21.json) | 0.3191767531788976 | 0.38296380409820407 | 0.2557271261556312 | 0 | 4 | 158287 | 154.5576639175415 |
| [ETTh2_p96_flatten_homogeneous_no_memory_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_homogeneous_no_memory_seed7.json) | 0.36575497375030075 | 0.41619169548466051 | 0.26809492002305357 | 4 | 8 | 158287 | 262.28346109390259 |
| [ETTh2_p96_flatten_homogeneous_retrieval_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_homogeneous_retrieval_seed13.json) | 0.37011919702313484 | 0.42268139659088316 | 0.27775332865387137 | 1 | 5 | 158287 | 248.63726878166199 |
| [ETTh2_p96_flatten_homogeneous_retrieval_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_homogeneous_retrieval_seed21.json) | 0.3189085165218134 | 0.38114112492983265 | 0.24885430824876925 | 1 | 5 | 158287 | 244.6981520652771 |
| [ETTh2_p96_flatten_homogeneous_retrieval_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/ETTh2_p96_flatten_homogeneous_retrieval_seed7.json) | 0.31987480923912792 | 0.38410794245039737 | 0.25699450875332602 | 3 | 7 | 158287 | 375.87110543251038 |

### v1 TCN H720 — 24개 완료

| Run/raw JSON | Test MSE | Test MAE | Val MSE | Best epoch | Epochs | Parameters | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ETTh1_p720_flatten_heterogeneous_no_memory_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_heterogeneous_no_memory_seed13.json) | 0.58163420738738592 | 0.55448285259842489 | 1.1914716550427982 | 2 | 6 | 997567 | 202.21918296813965 |
| [ETTh1_p720_flatten_heterogeneous_no_memory_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_heterogeneous_no_memory_seed21.json) | 0.58626355384778805 | 0.55821179655161846 | 1.1733268867800346 | 1 | 5 | 997567 | 161.77007842063904 |
| [ETTh1_p720_flatten_heterogeneous_no_memory_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_heterogeneous_no_memory_seed7.json) | 0.53906189441799501 | 0.53039953214486046 | 1.1895879951891497 | 1 | 5 | 997567 | 164.04147410392761 |
| [ETTh1_p720_flatten_heterogeneous_retrieval_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_heterogeneous_retrieval_seed13.json) | 0.55661717489027507 | 0.53930237988313556 | 1.190409770151672 | 2 | 6 | 997567 | 272.49365139007568 |
| [ETTh1_p720_flatten_heterogeneous_retrieval_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_heterogeneous_retrieval_seed21.json) | 0.58931746390834561 | 0.5579044781254201 | 1.1799351431022047 | 1 | 5 | 997567 | 218.03066325187683 |
| [ETTh1_p720_flatten_heterogeneous_retrieval_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_heterogeneous_retrieval_seed7.json) | 0.56503614086446041 | 0.54461991307184432 | 1.1913382511322581 | 2 | 6 | 997567 | 260.63457560539246 |
| [ETTh1_p720_flatten_homogeneous_no_memory_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_homogeneous_no_memory_seed13.json) | 0.58268124487872275 | 0.55647245540402068 | 1.1915365511245981 | 6 | 10 | 997567 | 292.54760789871216 |
| [ETTh1_p720_flatten_homogeneous_no_memory_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_homogeneous_no_memory_seed21.json) | 0.57929908276350084 | 0.56361607088870846 | 1.2020894184908468 | 2 | 6 | 997567 | 185.65648150444031 |
| [ETTh1_p720_flatten_homogeneous_no_memory_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_homogeneous_no_memory_seed7.json) | 0.56101489731596488 | 0.54958770051779426 | 1.2150144408310601 | 3 | 7 | 997567 | 207.69422435760498 |
| [ETTh1_p720_flatten_homogeneous_retrieval_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_homogeneous_retrieval_seed13.json) | 0.54723254077543093 | 0.54503082949703296 | 1.2149633239203315 | 3 | 7 | 997567 | 333.73566842079163 |
| [ETTh1_p720_flatten_homogeneous_retrieval_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_homogeneous_retrieval_seed21.json) | 0.55902619670333209 | 0.54933805708077554 | 1.2137963663122979 | 6 | 10 | 997567 | 462.74274516105652 |
| [ETTh1_p720_flatten_homogeneous_retrieval_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh1_p720_flatten_homogeneous_retrieval_seed7.json) | 0.57038511539844294 | 0.55441090583626595 | 1.2089674391329817 | 4 | 8 | 997567 | 345.14401197433472 |
| [ETTh2_p720_flatten_heterogeneous_no_memory_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_heterogeneous_no_memory_seed13.json) | 1.0724367199753271 | 0.72665478338672063 | 0.68365315644247293 | 0 | 4 | 997567 | 142.34762406349182 |
| [ETTh2_p720_flatten_heterogeneous_no_memory_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_heterogeneous_no_memory_seed21.json) | 1.0676477168927974 | 0.72691099346425758 | 0.67704030572725271 | 0 | 4 | 997567 | 131.23300909996033 |
| [ETTh2_p720_flatten_heterogeneous_no_memory_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_heterogeneous_no_memory_seed7.json) | 0.95321250297333582 | 0.67436029333588432 | 0.67527179379134838 | 0 | 4 | 997567 | 145.1899209022522 |
| [ETTh2_p720_flatten_heterogeneous_retrieval_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_heterogeneous_retrieval_seed13.json) | 0.99229295736687229 | 0.6993896919303394 | 0.67505052228782814 | 0 | 4 | 997567 | 200.3692262172699 |
| [ETTh2_p720_flatten_heterogeneous_retrieval_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_heterogeneous_retrieval_seed21.json) | 1.0533326385296538 | 0.7199890340107844 | 0.67711743457428608 | 0 | 4 | 997567 | 218.88766026496887 |
| [ETTh2_p720_flatten_heterogeneous_retrieval_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_heterogeneous_retrieval_seed7.json) | 1.1019383991904972 | 0.7275564969567978 | 0.68989543960504329 | 0 | 4 | 997567 | 186.85013318061829 |
| [ETTh2_p720_flatten_homogeneous_no_memory_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_homogeneous_no_memory_seed13.json) | 1.1270760230424448 | 0.73567123835078707 | 0.69073441967242888 | 1 | 5 | 997567 | 163.43561768531799 |
| [ETTh2_p720_flatten_homogeneous_no_memory_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_homogeneous_no_memory_seed21.json) | 0.88601329455165501 | 0.65514998475254049 | 0.6798039353397588 | 2 | 6 | 997567 | 200.64794635772705 |
| [ETTh2_p720_flatten_homogeneous_no_memory_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_homogeneous_no_memory_seed7.json) | 0.98653585928736132 | 0.68487140844473982 | 0.70375971453994313 | 4 | 8 | 997567 | 232.9882493019104 |
| [ETTh2_p720_flatten_homogeneous_retrieval_seed13](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_homogeneous_retrieval_seed13.json) | 1.1689082130035275 | 0.73920369090885041 | 0.69487527530263504 | 2 | 6 | 997567 | 268.35082006454468 |
| [ETTh2_p720_flatten_homogeneous_retrieval_seed21](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_homogeneous_retrieval_seed21.json) | 1.0243159220836247 | 0.69301334235046186 | 0.6688790140630565 | 2 | 6 | 997567 | 267.94958925247192 |
| [ETTh2_p720_flatten_homogeneous_retrieval_seed7](../../../f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/ETTh2_p720_flatten_homogeneous_retrieval_seed7.json) | 0.99702384720484249 | 0.68541365177037827 | 0.68681302597244087 | 4 | 8 | 997567 | 361.11041450500488 |

### v2 patch H96 — 36개 완료

| Run/raw JSON | Test MSE | Test MAE | Val MSE | Best epoch | Epochs | Parameters | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [patch_ETTh1_p96_flatten_heterogeneous_dense_seed13](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_heterogeneous_dense_seed13.json) | 0.43052520004435219 | 0.44961429335869085 | 0.72557208519526217 | 6 | 13 | 133580 | 82.263892412185669 |
| [patch_ETTh1_p96_flatten_heterogeneous_dense_seed21](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_heterogeneous_dense_seed21.json) | 0.42691736727380519 | 0.44483274148567681 | 0.71895673012404326 | 6 | 13 | 133580 | 113.19566798210144 |
| [patch_ETTh1_p96_flatten_heterogeneous_dense_seed7](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_heterogeneous_dense_seed7.json) | 0.42173667768764217 | 0.44097960306155404 | 0.72928313890173124 | 7 | 14 | 133580 | 117.98689389228821 |
| [patch_ETTh1_p96_flatten_heterogeneous_no_memory_seed13](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_heterogeneous_no_memory_seed13.json) | 0.42926142724729743 | 0.44770004990206863 | 0.72288151852116667 | 6 | 13 | 133580 | 29.969054937362671 |
| [patch_ETTh1_p96_flatten_heterogeneous_no_memory_seed21](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_heterogeneous_no_memory_seed21.json) | 0.42364105768257454 | 0.44270382823070664 | 0.7150460197872508 | 6 | 13 | 133580 | 30.428729057312012 |
| [patch_ETTh1_p96_flatten_heterogeneous_no_memory_seed7](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_heterogeneous_no_memory_seed7.json) | 0.4190193954801964 | 0.43866681810009761 | 0.72279938922665965 | 8 | 15 | 133580 | 33.365219116210938 |
| [patch_ETTh1_p96_flatten_heterogeneous_sparse_seed13](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_heterogeneous_sparse_seed13.json) | 0.43187980224942907 | 0.44926442826423812 | 0.72216152216347695 | 6 | 13 | 133580 | 140.44902873039246 |
| [patch_ETTh1_p96_flatten_heterogeneous_sparse_seed21](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_heterogeneous_sparse_seed21.json) | 0.42701036894334393 | 0.4446609334923774 | 0.71797912322456459 | 6 | 13 | 133580 | 127.66362404823303 |
| [patch_ETTh1_p96_flatten_heterogeneous_sparse_seed7](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_heterogeneous_sparse_seed7.json) | 0.42116168368017443 | 0.44024000215028958 | 0.72840730335800896 | 7 | 14 | 133580 | 142.13387227058411 |
| [patch_ETTh1_p96_flatten_homogeneous_dense_seed13](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_homogeneous_dense_seed13.json) | 0.45275755566493364 | 0.46029327311127977 | 0.76614908690904793 | 13 | 20 | 133580 | 147.60286140441895 |
| [patch_ETTh1_p96_flatten_homogeneous_dense_seed21](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_homogeneous_dense_seed21.json) | 0.44891914400648264 | 0.45925921217927301 | 0.7694710788141963 | 15 | 22 | 133580 | 141.29745364189148 |
| [patch_ETTh1_p96_flatten_homogeneous_dense_seed7](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_homogeneous_dense_seed7.json) | 0.44613202886927711 | 0.45724645955313087 | 0.75925787657758848 | 29 | 30 | 133580 | 219.42174339294434 |
| [patch_ETTh1_p96_flatten_homogeneous_no_memory_seed13](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_homogeneous_no_memory_seed13.json) | 0.44821535082423969 | 0.45694674259577506 | 0.75758007880258948 | 13 | 20 | 133580 | 46.119028806686401 |
| [patch_ETTh1_p96_flatten_homogeneous_no_memory_seed21](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_homogeneous_no_memory_seed21.json) | 0.4451840499356593 | 0.45603944828440862 | 0.7786140827169068 | 15 | 22 | 133580 | 48.618356227874756 |
| [patch_ETTh1_p96_flatten_homogeneous_no_memory_seed7](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_homogeneous_no_memory_seed7.json) | 0.44884739307581362 | 0.46169455831212808 | 0.7591590763017888 | 15 | 22 | 133580 | 48.068639755249023 |
| [patch_ETTh1_p96_flatten_homogeneous_sparse_seed13](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_homogeneous_sparse_seed13.json) | 0.45287390155610913 | 0.46043378243894073 | 0.77689577729916193 | 13 | 20 | 133580 | 207.09512591362 |
| [patch_ETTh1_p96_flatten_homogeneous_sparse_seed21](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_homogeneous_sparse_seed21.json) | 0.448150637817542 | 0.45749798661250674 | 0.77470231651474297 | 15 | 22 | 133580 | 183.0152907371521 |
| [patch_ETTh1_p96_flatten_homogeneous_sparse_seed7](selective-v2-20260914_patch_p96/patch_ETTh1_p96_flatten_homogeneous_sparse_seed7.json) | 0.44172817589481422 | 0.45292777868196282 | 0.77453738307158038 | 14 | 21 | 133580 | 188.14642119407654 |
| [patch_ETTh2_p96_flatten_heterogeneous_dense_seed13](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_heterogeneous_dense_seed13.json) | 0.34344715790959957 | 0.40298606031504841 | 0.25563941546053565 | 1 | 8 | 133580 | 54.574846982955933 |
| [patch_ETTh2_p96_flatten_heterogeneous_dense_seed21](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_heterogeneous_dense_seed21.json) | 0.32146220411639703 | 0.3859908068754162 | 0.25328784537190763 | 5 | 12 | 133580 | 78.495340585708618 |
| [patch_ETTh2_p96_flatten_heterogeneous_dense_seed7](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_heterogeneous_dense_seed7.json) | 0.32167645203454348 | 0.39016380309317233 | 0.2516357932053474 | 1 | 8 | 133580 | 71.204339742660522 |
| [patch_ETTh2_p96_flatten_heterogeneous_no_memory_seed13](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_heterogeneous_no_memory_seed13.json) | 0.35429734818674158 | 0.41034820970902441 | 0.25692345502133107 | 1 | 8 | 133580 | 21.147143602371216 |
| [patch_ETTh2_p96_flatten_heterogeneous_no_memory_seed21](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_heterogeneous_no_memory_seed21.json) | 0.31681595568766652 | 0.38302495343343795 | 0.25093543500509868 | 5 | 12 | 133580 | 28.748282194137573 |
| [patch_ETTh2_p96_flatten_heterogeneous_no_memory_seed7](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_heterogeneous_no_memory_seed7.json) | 0.32786642931539267 | 0.39579366677810696 | 0.25626140632471234 | 1 | 8 | 133580 | 21.210817337036133 |
| [patch_ETTh2_p96_flatten_heterogeneous_sparse_seed13](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_heterogeneous_sparse_seed13.json) | 0.34453059878540909 | 0.40291788435268155 | 0.25435695589974094 | 1 | 8 | 133580 | 91.404720783233643 |
| [patch_ETTh2_p96_flatten_heterogeneous_sparse_seed21](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_heterogeneous_sparse_seed21.json) | 0.32932681871826169 | 0.39311554403242371 | 0.2525545519476744 | 1 | 8 | 133580 | 67.836817979812622 |
| [patch_ETTh2_p96_flatten_heterogeneous_sparse_seed7](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_heterogeneous_sparse_seed7.json) | 0.33426700340969029 | 0.39839418663666076 | 0.2596758369640213 | 1 | 8 | 133580 | 90.361396789550781 |
| [patch_ETTh2_p96_flatten_homogeneous_dense_seed13](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_homogeneous_dense_seed13.json) | 0.35895694515214699 | 0.41502835046030201 | 0.26522019504707467 | 5 | 12 | 133580 | 102.03583836555481 |
| [patch_ETTh2_p96_flatten_homogeneous_dense_seed21](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_homogeneous_dense_seed21.json) | 0.33467440190608605 | 0.39935224520979551 | 0.25851212332360302 | 1 | 8 | 133580 | 68.814688205718994 |
| [patch_ETTh2_p96_flatten_homogeneous_dense_seed7](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_homogeneous_dense_seed7.json) | 0.33518360649152157 | 0.39993413039110876 | 0.2582420289786419 | 7 | 14 | 133580 | 111.56990051269531 |
| [patch_ETTh2_p96_flatten_homogeneous_no_memory_seed13](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_homogeneous_no_memory_seed13.json) | 0.35743036294990022 | 0.41351280385558153 | 0.26323814134752888 | 5 | 12 | 133580 | 27.427499532699585 |
| [patch_ETTh2_p96_flatten_homogeneous_no_memory_seed21](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_homogeneous_no_memory_seed21.json) | 0.33968074140709531 | 0.4036670809372862 | 0.2619570312399111 | 1 | 8 | 133580 | 20.152692794799805 |
| [patch_ETTh2_p96_flatten_homogeneous_no_memory_seed7](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_homogeneous_no_memory_seed7.json) | 0.33259518413760891 | 0.39871861485547239 | 0.26410003873581284 | 1 | 8 | 133580 | 19.011906147003174 |
| [patch_ETTh2_p96_flatten_homogeneous_sparse_seed13](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_homogeneous_sparse_seed13.json) | 0.33595673829954059 | 0.39691213529956215 | 0.26218337238912998 | 13 | 20 | 133580 | 210.07762694358826 |
| [patch_ETTh2_p96_flatten_homogeneous_sparse_seed21](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_homogeneous_sparse_seed21.json) | 0.3358180547269658 | 0.39895168153729804 | 0.25895452777305289 | 1 | 8 | 133580 | 85.744422435760498 |
| [patch_ETTh2_p96_flatten_homogeneous_sparse_seed7](selective-v2-20260914_patch_p96/patch_ETTh2_p96_flatten_homogeneous_sparse_seed7.json) | 0.33467191984803674 | 0.39987090174270445 | 0.2607230579127105 | 7 | 14 | 133580 | 126.36534810066223 |

### v2 patch H720 — 36개 완료

| Run/raw JSON | Test MSE | Test MAE | Val MSE | Best epoch | Epochs | Parameters | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [patch_ETTh1_p720_flatten_heterogeneous_dense_seed13](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_heterogeneous_dense_seed13.json) | 0.58353379111702808 | 0.56291908095550247 | 1.2282031453514957 | 2 | 9 | 972860 | 53.454457521438599 |
| [patch_ETTh1_p720_flatten_heterogeneous_dense_seed21](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_heterogeneous_dense_seed21.json) | 0.54425608381609292 | 0.54133813391609875 | 1.1928336535440516 | 4 | 11 | 972860 | 86.828738927841187 |
| [patch_ETTh1_p720_flatten_heterogeneous_dense_seed7](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_heterogeneous_dense_seed7.json) | 0.55345638180339785 | 0.54601655674297911 | 1.2198640643061378 | 2 | 9 | 972860 | 73.090698719024658 |
| [patch_ETTh1_p720_flatten_heterogeneous_no_memory_seed13](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_heterogeneous_no_memory_seed13.json) | 0.57799019219717684 | 0.56032339849718127 | 1.2262087018307672 | 2 | 9 | 972860 | 19.498690128326416 |
| [patch_ETTh1_p720_flatten_heterogeneous_no_memory_seed21](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_heterogeneous_no_memory_seed21.json) | 0.53106955051132743 | 0.53137107905912517 | 1.1918141193970377 | 4 | 11 | 972860 | 25.394774913787842 |
| [patch_ETTh1_p720_flatten_heterogeneous_no_memory_seed7](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_heterogeneous_no_memory_seed7.json) | 0.55159450711763525 | 0.54399123409361216 | 1.2151415698822705 | 2 | 9 | 972860 | 20.508918285369873 |
| [patch_ETTh1_p720_flatten_heterogeneous_sparse_seed13](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_heterogeneous_sparse_seed13.json) | 0.58311318530097356 | 0.56251243991501099 | 1.2259626485067574 | 2 | 9 | 972860 | 76.636059284210205 |
| [patch_ETTh1_p720_flatten_heterogeneous_sparse_seed21](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_heterogeneous_sparse_seed21.json) | 0.54459976732527715 | 0.5421280679703836 | 1.1932866441709784 | 4 | 11 | 972860 | 98.303338766098022 |
| [patch_ETTh1_p720_flatten_heterogeneous_sparse_seed7](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_heterogeneous_sparse_seed7.json) | 0.55514909706848314 | 0.54740057305897249 | 1.2203697941710496 | 2 | 9 | 972860 | 83.673384428024292 |
| [patch_ETTh1_p720_flatten_homogeneous_dense_seed13](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_homogeneous_dense_seed13.json) | 0.600070691221118 | 0.56999850658927331 | 1.2662159342546988 | 16 | 23 | 972860 | 172.86797094345093 |
| [patch_ETTh1_p720_flatten_homogeneous_dense_seed21](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_homogeneous_dense_seed21.json) | 0.56912536239318456 | 0.55365812913621326 | 1.2474241617792836 | 29 | 30 | 972860 | 180.98575854301453 |
| [patch_ETTh1_p720_flatten_homogeneous_dense_seed7](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_homogeneous_dense_seed7.json) | 0.58108776387581662 | 0.56427449377165084 | 1.264800718076136 | 10 | 17 | 972860 | 112.28055500984192 |
| [patch_ETTh1_p720_flatten_homogeneous_no_memory_seed13](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_homogeneous_no_memory_seed13.json) | 0.61675731295130531 | 0.58027698606135947 | 1.274344023496073 | 16 | 23 | 972860 | 48.457937479019165 |
| [patch_ETTh1_p720_flatten_homogeneous_no_memory_seed21](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_homogeneous_no_memory_seed21.json) | 0.55862503668104668 | 0.55543514656586868 | 1.2699225147357156 | 5 | 12 | 972860 | 25.366487264633179 |
| [patch_ETTh1_p720_flatten_homogeneous_no_memory_seed7](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_homogeneous_no_memory_seed7.json) | 0.56786222290048116 | 0.55776283584597408 | 1.2623204056801161 | 10 | 17 | 972860 | 36.111533403396606 |
| [patch_ETTh1_p720_flatten_homogeneous_sparse_seed13](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_homogeneous_sparse_seed13.json) | 0.61287180337725622 | 0.57825582938048858 | 1.2640005192420791 | 9 | 16 | 972860 | 152.0752649307251 |
| [patch_ETTh1_p720_flatten_homogeneous_sparse_seed21](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_homogeneous_sparse_seed21.json) | 0.56483783817381283 | 0.55071403249404227 | 1.240990624075758 | 29 | 30 | 972860 | 214.33515596389771 |
| [patch_ETTh1_p720_flatten_homogeneous_sparse_seed7](selective-v2-20260914_patch_p720/patch_ETTh1_p720_flatten_homogeneous_sparse_seed7.json) | 0.5859156908025589 | 0.56715764599460794 | 1.2644866407037896 | 10 | 17 | 972860 | 137.88123679161072 |
| [patch_ETTh2_p720_flatten_heterogeneous_dense_seed13](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_heterogeneous_dense_seed13.json) | 0.83414391300499291 | 0.6197285451904273 | 0.65001778826491707 | 2 | 9 | 972860 | 74.112914562225342 |
| [patch_ETTh2_p720_flatten_heterogeneous_dense_seed21](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_heterogeneous_dense_seed21.json) | 0.75395401652186411 | 0.59572764741694073 | 0.66562811839089953 | 3 | 10 | 972860 | 79.343515157699585 |
| [patch_ETTh2_p720_flatten_heterogeneous_dense_seed7](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_heterogeneous_dense_seed7.json) | 0.97081006233238309 | 0.68453889113666433 | 0.67865587399298466 | 1 | 8 | 972860 | 65.614367723464966 |
| [patch_ETTh2_p720_flatten_heterogeneous_no_memory_seed13](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_heterogeneous_no_memory_seed13.json) | 0.85136144577658657 | 0.62107316790074751 | 0.65714529664331878 | 2 | 9 | 972860 | 20.877828598022461 |
| [patch_ETTh2_p720_flatten_heterogeneous_no_memory_seed21](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_heterogeneous_no_memory_seed21.json) | 0.71706496039432255 | 0.58511550136180845 | 0.65539692989040998 | 3 | 10 | 972860 | 23.470742702484131 |
| [patch_ETTh2_p720_flatten_heterogeneous_no_memory_seed7](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_heterogeneous_no_memory_seed7.json) | 0.884832213806657 | 0.65384267023779297 | 0.67970246468787798 | 1 | 8 | 972860 | 19.778042078018188 |
| [patch_ETTh2_p720_flatten_heterogeneous_sparse_seed13](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_heterogeneous_sparse_seed13.json) | 0.87315230597412508 | 0.63688098238146706 | 0.65145178737816412 | 2 | 9 | 972860 | 76.807239294052124 |
| [patch_ETTh2_p720_flatten_heterogeneous_sparse_seed21](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_heterogeneous_sparse_seed21.json) | 0.8148069591720033 | 0.61849548209431038 | 0.66742339657845395 | 2 | 9 | 972860 | 68.573480129241943 |
| [patch_ETTh2_p720_flatten_heterogeneous_sparse_seed7](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_heterogeneous_sparse_seed7.json) | 0.95803343067059643 | 0.67929178557743708 | 0.67534054845931657 | 1 | 8 | 972860 | 81.533246278762817 |
| [patch_ETTh2_p720_flatten_homogeneous_dense_seed13](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_homogeneous_dense_seed13.json) | 0.73764662197104092 | 0.59148415015966782 | 0.65791146099032682 | 2 | 9 | 972860 | 71.110562324523926 |
| [patch_ETTh2_p720_flatten_homogeneous_dense_seed21](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_homogeneous_dense_seed21.json) | 0.93750383054647013 | 0.6826348395110744 | 0.68301378246825406 | 0 | 7 | 972860 | 58.931501865386963 |
| [patch_ETTh2_p720_flatten_homogeneous_dense_seed7](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_homogeneous_dense_seed7.json) | 0.85746618329814284 | 0.64821069119368158 | 0.68410974838141425 | 1 | 8 | 972860 | 53.492494106292725 |
| [patch_ETTh2_p720_flatten_homogeneous_no_memory_seed13](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_homogeneous_no_memory_seed13.json) | 0.79137426539568856 | 0.61425554236620594 | 0.66722433693622674 | 2 | 9 | 972860 | 20.668715715408325 |
| [patch_ETTh2_p720_flatten_homogeneous_no_memory_seed21](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_homogeneous_no_memory_seed21.json) | 0.67894038039891635 | 0.57637016322289181 | 0.67203747064302455 | 2 | 9 | 972860 | 21.251629590988159 |
| [patch_ETTh2_p720_flatten_homogeneous_no_memory_seed7](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_homogeneous_no_memory_seed7.json) | 0.87623530235052449 | 0.64918307590796087 | 0.68839646628511886 | 1 | 8 | 972860 | 19.587193489074707 |
| [patch_ETTh2_p720_flatten_homogeneous_sparse_seed13](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_homogeneous_sparse_seed13.json) | 0.75588595295645788 | 0.59800657420153513 | 0.66228009202425675 | 2 | 9 | 972860 | 86.775252103805542 |
| [patch_ETTh2_p720_flatten_homogeneous_sparse_seed21](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_homogeneous_sparse_seed21.json) | 0.66574683986080607 | 0.57210992384322445 | 0.67771928644186052 | 2 | 9 | 972860 | 73.922946691513062 |
| [patch_ETTh2_p720_flatten_homogeneous_sparse_seed7](selective-v2-20260914_patch_p720/patch_ETTh2_p720_flatten_homogeneous_sparse_seed7.json) | 0.71419947347522594 | 0.58485769606994753 | 0.67910024845918682 | 4 | 11 | 972860 | 81.715526819229126 |

### v2 TCN H96 — 부분 완료 결과 (최종 matrix 검증 전)

| Run/raw JSON | Test MSE | Test MAE | Val MSE | Best epoch | Epochs | Parameters | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [tcn_ETTh1_p96_flatten_heterogeneous_dense_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh1_p96_flatten_heterogeneous_dense_seed7.json) | 0.41695444012223543 | 0.44091224031550952 | 0.7085855031023307 | 3 | 10 | 158308 | 4671.9829914569855 |
| [tcn_ETTh1_p96_flatten_heterogeneous_no_memory_seed13](selective-v2-20260914_tcn_p96/tcn_ETTh1_p96_flatten_heterogeneous_no_memory_seed13.json) | 0.42026050225207107 | 0.44231921536531976 | 0.7198527891426415 | 2 | 9 | 158308 | 3751.6587798595428 |
| [tcn_ETTh1_p96_flatten_heterogeneous_no_memory_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh1_p96_flatten_heterogeneous_no_memory_seed7.json) | 0.41608011427859876 | 0.44022902721801188 | 0.70497976263444584 | 3 | 10 | 158308 | 4273.8837900161743 |
| [tcn_ETTh1_p96_flatten_heterogeneous_sparse_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh1_p96_flatten_heterogeneous_sparse_seed7.json) | 0.41123267503819766 | 0.43416293109547349 | 0.71353794174306417 | 2 | 9 | 158308 | 4322.1543486118317 |
| [tcn_ETTh1_p96_flatten_homogeneous_dense_seed13](selective-v2-20260914_tcn_p96/tcn_ETTh1_p96_flatten_homogeneous_dense_seed13.json) | 0.44930886249833868 | 0.45839887076047592 | 0.75195213760896074 | 2 | 9 | 158308 | 4195.6938514709473 |
| [tcn_ETTh1_p96_flatten_homogeneous_dense_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh1_p96_flatten_homogeneous_dense_seed7.json) | 0.43445948527753159 | 0.45474455477461012 | 0.75155585889614307 | 3 | 10 | 158308 | 4571.2273163795471 |
| [tcn_ETTh1_p96_flatten_homogeneous_no_memory_seed13](selective-v2-20260914_tcn_p96/tcn_ETTh1_p96_flatten_homogeneous_no_memory_seed13.json) | 0.44221859357321663 | 0.45581601566498603 | 0.76737510017038812 | 4 | 11 | 158308 | 4555.4057068824768 |
| [tcn_ETTh1_p96_flatten_homogeneous_no_memory_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh1_p96_flatten_homogeneous_no_memory_seed7.json) | 0.42776988056435922 | 0.44575782595673524 | 0.74410620972826513 | 2 | 9 | 158308 | 3849.860184431076 |
| [tcn_ETTh1_p96_flatten_homogeneous_sparse_seed13](selective-v2-20260914_tcn_p96/tcn_ETTh1_p96_flatten_homogeneous_sparse_seed13.json) | 0.43742595613423757 | 0.4504641132503332 | 0.75455244610307071 | 3 | 10 | 158308 | 4754.8415327072144 |
| [tcn_ETTh1_p96_flatten_homogeneous_sparse_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh1_p96_flatten_homogeneous_sparse_seed7.json) | 0.43883643871668387 | 0.45844917368567634 | 0.74130034764069352 | 5 | 12 | 158308 | 5429.3908770084381 |
| [tcn_ETTh2_p96_flatten_heterogeneous_dense_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh2_p96_flatten_heterogeneous_dense_seed7.json) | 0.36546169003596413 | 0.41602123271542241 | 0.25839547211034641 | 4 | 11 | 158308 | 5104.2992653846741 |
| [tcn_ETTh2_p96_flatten_heterogeneous_no_memory_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh2_p96_flatten_heterogeneous_no_memory_seed7.json) | 0.34207546992104293 | 0.40110701286739109 | 0.25674812815287201 | 1 | 8 | 158308 | 3358.0367252826691 |
| [tcn_ETTh2_p96_flatten_heterogeneous_sparse_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh2_p96_flatten_heterogeneous_sparse_seed7.json) | 0.33326448412741255 | 0.39118694296517237 | 0.25522921786930602 | 1 | 8 | 158308 | 3918.3199336528778 |
| [tcn_ETTh2_p96_flatten_homogeneous_dense_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh2_p96_flatten_homogeneous_dense_seed7.json) | 0.3379121872069476 | 0.38933230366761712 | 0.26334123445917379 | 3 | 10 | 158308 | 4610.924973487854 |
| [tcn_ETTh2_p96_flatten_homogeneous_no_memory_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh2_p96_flatten_homogeneous_no_memory_seed7.json) | 0.35069658368568984 | 0.40147205898095772 | 0.26294343872926568 | 4 | 11 | 158308 | 4612.5851008892059 |
| [tcn_ETTh2_p96_flatten_homogeneous_sparse_seed7](selective-v2-20260914_tcn_p96/tcn_ETTh2_p96_flatten_homogeneous_sparse_seed7.json) | 0.33590857853943024 | 0.39661053891219716 | 0.26390781917877193 | 3 | 10 | 158308 | 4648.071417093277 |

<a id="handoff-source"></a>
## 15. 핵심 코드 원문 사본

아래는 작성 시점 파일의 원문이다. 다음 세션에서 수정할 때는 이 Markdown 사본이 아니라 실제 .py 파일을 편집한다. 설정/학습/평가/검증/queue 코드는5절 링크를 따른다. 파일별 SHA는13절과 대조할 수 있다.

### layers.py

원본: [layers.py](../layers.py)

```python
"""Population membrane memory with amplitude-sensitive scores and explicit read support."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate


class Sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score):
        # Martins & Astudillo (2016), Alg.1: project onto the probability simplex.
        score = score - score.max(dim=-1, keepdim=True).values
        ordered = score.sort(dim=-1, descending=True).values
        cumulative = ordered.cumsum(dim=-1)
        ranks = torch.arange(1, score.shape[-1] + 1, device=score.device, dtype=score.dtype)
        support_size = (1 + ranks * ordered > cumulative).sum(dim=-1, keepdim=True)
        threshold = (cumulative.gather(-1, support_size - 1) - 1) / support_size
        output = (score - threshold).clamp_min(0.)
        ctx.save_for_backward(output > 0)
        return output

    @staticmethod
    def backward(ctx, gradient):
        # Eq.14 avoids sort/gather's nondeterministic CUDA scatter backward in torch1.12.
        support, = ctx.saved_tensors
        selected = gradient * support
        mean = selected.sum(dim=-1, keepdim=True) / support.sum(dim=-1, keepdim=True)
        return support * (gradient - mean)


def sparsemax(score):
    return Sparsemax.apply(score)


class PopulationLIF(nn.Module):
    def __init__(self, num_population=4, heterogeneous=True, retrieval=True,
                 tau_min=2., tau_max=16., threshold=1., memory_strength=.05,
                 temperature=.25, read_mode='sparse', null_logit_init=-1.):
        super().__init__()
        if num_population < 2 or not 1 < tau_min <= tau_max:
            raise ValueError('Require K >= 2 and 1 < tau_min <= tau_max')
        if threshold <= 0 or temperature <= 0 or read_mode not in ['dense', 'sparse']:
            raise ValueError('Invalid neuron or retrieval configuration')
        tau = torch.logspace(math.log10(tau_min), math.log10(tau_max), num_population)
        beta = torch.exp(-1. / tau)
        if not heterogeneous:
            beta = beta.mean().repeat(num_population)
        if not 0 <= memory_strength < 1 - beta.max().item():
            raise ValueError('Require memory_strength < 1 - max(beta)')
        self.register_buffer('beta', beta)
        self.num_population, self.retrieval = num_population, retrieval
        self.threshold, self.memory_strength = threshold, memory_strength
        self.temperature, self.read_mode = temperature, read_mode
        # Off/dense/sparse 모두 같은 파라미터와 초기값을 보존한다.
        self.query = nn.Linear(num_population, num_population, bias=False)
        self.key = nn.Linear(num_population, num_population, bias=False)
        self.gate = nn.Linear(2 * num_population + 2, 1)
        self.null_logit = nn.Parameter(torch.tensor(float(null_logit_init)))
        nn.init.eye_(self.query.weight)
        nn.init.eye_(self.key.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.spike = surrogate.Sigmoid(alpha=4.)

    def read_memory(self, charged, history, memory_mode=None, keys=None):
        """[BC,D,K], [BC,D,J,K] -> normalized read, support, and read-strength gate.

        All history passed here must already be in the past. This method never
        deletes history; forward appends every post-reset state independently.
        """
        if history.shape[-2] == 0:
            raise ValueError('Empty history is handled before calling read_memory')
        q = self.query(charged)
        k = self.key(history) if keys is None else keys
        # No L2 normalization: states that differ only in amplitude remain distinct.
        score = -(q.unsqueeze(-2) - k).square().mean(dim=-1) / self.temperature
        logits = torch.cat([score, self.null_logit.expand_as(score[..., :1])], dim=-1)
        probability = sparsemax(logits) if self.read_mode == 'sparse' else logits.softmax(dim=-1)
        real_probability = probability[..., :-1]
        real_mass = real_probability.sum(dim=-1, keepdim=True)
        weight = real_probability / real_mass.clamp_min(1e-12)
        if memory_mode == 'uniform':
            weight = torch.ones_like(weight) / history.shape[-2]
            real_mass = torch.ones_like(real_mass)
        elif memory_mode == 'recent':
            weight = torch.zeros_like(weight)
            weight[..., -1] = 1.
            real_mass = torch.ones_like(real_mass)
        elif memory_mode not in (None, 'off'):
            raise ValueError('Unknown memory intervention')
        memory = (weight.unsqueeze(-1) * history).sum(dim=-2)
        margin = score.max(dim=-1, keepdim=True).values - self.null_logit
        gate_input = torch.cat([charged, memory, margin, real_mass], dim=-1)
        gate = real_mass * self.gate(gate_input).sigmoid()
        if memory_mode == 'off':
            weight = torch.zeros_like(weight)
            memory, gate = torch.zeros_like(memory), torch.zeros_like(gate)
            real_mass = torch.zeros_like(real_mass)
        return memory, {'weight': weight, 'mask': weight > 0, 'gate': gate,
                        'real_mass': real_mass, 'score': score}

    def forward(self, x, return_aux=False, memory_mode=None):
        if x.ndim != 3 or x.shape[0] < 1:
            raise ValueError('Expected nonempty [T,BC,D] current')
        if memory_mode not in (None, 'off', 'uniform', 'recent'):
            raise ValueError('Unknown memory intervention')
        use_memory = self.retrieval and memory_mode != 'off'
        state = x.new_zeros(*x.shape[1:], self.num_population)
        states, keys, spikes = [], [], []
        weights_log, gate_log, charge_log, evidence_log, mass_log = [], [], [], [], []
        for t in range(x.shape[0]):
            # K constituents share current; each retains its own beta and membrane.
            charged = self.beta * state + (1. - self.beta) * x[t].unsqueeze(-1)
            evidence = torch.zeros_like(charged)
            gate = charged.new_zeros(*charged.shape[:-1], 1)
            real_mass = torch.zeros_like(gate)
            weight = charged.new_zeros(*charged.shape[:-1], t)
            if use_memory and t:
                history = torch.stack(states, dim=-2)
                memory, read = self.read_memory(charged, history, memory_mode, torch.stack(keys, dim=-2))
                weight, gate, real_mass = read['weight'], read['gate'], read['real_mass']
                evidence = self.memory_strength * gate * memory
            voltage = charged + evidence
            spike = self.spike(voltage - self.threshold)
            state = voltage - self.threshold * spike.detach()
            spikes.append(spike)
            if use_memory:
                states.append(state)
                keys.append(self.key(state))
            elif return_aux:
                states.append(state)
            if return_aux:
                weights_log.append(F.pad(weight.detach(), (0, x.shape[0] - t)))
                gate_log.append(gate.detach())
                charge_log.append(charged.detach())
                evidence_log.append(evidence.detach())
                mass_log.append(real_mass.detach())
        output = torch.stack(spikes)
        if return_aux:
            attention = torch.stack(weights_log)
            return output, {'membrane': torch.stack(states).detach(), 'spikes': output.detach(),
                            'attention': attention, 'mask': attention > 0,
                            'gate': torch.stack(gate_log), 'charge': torch.stack(charge_log),
                            'evidence': torch.stack(evidence_log), 'real_mass': torch.stack(mass_log)}
        return output


class Embedding(nn.Module):
    def __init__(self, patch_size, embed_dim, input_scale=2., **neuron_args):
        super().__init__()
        self.proj = nn.Linear(patch_size, embed_dim, bias=True)
        self.input_scale = input_scale
        self.lif = PopulationLIF(**neuron_args)

    def forward(self, x, return_aux=False, memory_mode=None):
        return self.lif(self.proj(x) * self.input_scale, return_aux, memory_mode)
```

### backbones.py

원본: [backbones.py](../backbones.py)

```python
"""Small causal backbone adaptations; keep chronological population states in every block."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PopulationLIF


class CurrentBlock(nn.Module):
    def forward(self, current, spikes, return_aux=False, memory_mode=None):
        current = current + self.input_scale * self.mix(spikes.flatten(2))
        result = self.lif(current, return_aux=return_aux, memory_mode=memory_mode)
        if return_aux:
            spikes, aux = result
            return current, spikes, aux
        return current, result


class TemporalBlock(CurrentBlock):
    def __init__(self, embed_dim, num_population, dilation, input_scale=2., **neuron_args):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim * num_population, embed_dim, 3, dilation=dilation)
        self.left_padding = 2 * dilation
        self.input_scale = input_scale
        self.lif = PopulationLIF(num_population=num_population, **neuron_args)

    def mix(self, x):
        x = F.pad(x.permute(1, 2, 0).contiguous(), (self.left_padding, 0))
        return self.conv(x).permute(2, 0, 1)


class AttentionBlock(CurrentBlock):
    def __init__(self, embed_dim, num_population, num_patches, num_heads=4,
                 input_scale=2., **neuron_args):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError('Embedding width must be divisible by attention heads')
        self.proj = nn.Linear(embed_dim * num_population, embed_dim)
        self.position = nn.Parameter(torch.zeros(num_patches, 1, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.)
        self.register_buffer('future_mask', torch.ones(num_patches, num_patches, dtype=torch.bool).triu(1), persistent=False)
        self.input_scale = input_scale
        self.lif = PopulationLIF(num_population=num_population, **neuron_args)

    def mix(self, x):
        x = self.proj(x) + self.position
        return self.attention(x, x, x, attn_mask=self.future_mask, need_weights=False)[0]


class ChannelBlock(CurrentBlock):
    def __init__(self, embed_dim, num_population, input_scale=2., **neuron_args):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embed_dim * num_population, 2 * embed_dim),
                                 nn.GELU(), nn.Linear(2 * embed_dim, embed_dim))
        self.input_scale = input_scale
        self.lif = PopulationLIF(num_population=num_population, **neuron_args)

    def mix(self, x):
        return self.mlp(x)


class CausalLinear(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.linear = nn.Linear(steps, steps)
        self.register_buffer('past_mask', torch.ones(steps, steps).tril(), persistent=False)

    def forward(self, x):
        return F.linear(x, self.linear.weight * self.past_mask, self.linear.bias)


class TokenBlock(CurrentBlock):
    def __init__(self, embed_dim, num_population, num_patches, input_scale=2., **neuron_args):
        super().__init__()
        self.proj = nn.Linear(embed_dim * num_population, embed_dim)
        self.time_mlp = nn.Sequential(CausalLinear(num_patches), nn.GELU(), CausalLinear(num_patches))
        self.input_scale = input_scale
        self.lif = PopulationLIF(num_population=num_population, **neuron_args)

    def mix(self, x):
        x = self.proj(x).permute(1, 2, 0)
        return self.time_mlp(x).permute(2, 0, 1)
```

### ours.py

원본: [ours.py](../ours.py)

```python
"""One common PopulationLIF and forecast head across the four staged backbones."""
import torch.nn as nn

from .layers import Embedding
from .backbones import TemporalBlock, AttentionBlock, ChannelBlock, TokenBlock

__all__ = ['myModel']


class myModel(nn.Module):
    def __init__(self, seq_len=336, pred_len=96, patch_size=8, embed_dim=32,
                 num_population=4, head_dim=32, head_mode='flatten',
                 heterogeneous=True, retrieval=True, tau_min=2., tau_max=16.,
                 threshold=1., memory_strength=.05, temperature=.25, input_scale=2.,
                 architecture='patch', read_mode='sparse', null_logit_init=-1., num_heads=4):
        super().__init__()
        if seq_len < patch_size or seq_len % patch_size:
            raise ValueError('Require complete chronological non-overlapping patches')
        if min(pred_len, embed_dim, head_dim) < 1 or head_mode not in ['flatten', 'last']:
            raise ValueError('Invalid forecast dimensions or readout')
        self.seq_len, self.pred_len = seq_len, pred_len
        self.patch_size, self.head_mode = patch_size, head_mode
        self.num_patches = seq_len // patch_size
        neuron_args = dict(heterogeneous=heterogeneous, retrieval=retrieval,
                           tau_min=tau_min, tau_max=tau_max, threshold=threshold,
                           memory_strength=memory_strength, temperature=temperature,
                           read_mode=read_mode, null_logit_init=null_logit_init)
        self.embedding = Embedding(patch_size, embed_dim, input_scale, num_population=num_population, **neuron_args)
        # Construct the common readout first so its initialization also matches across backbones.
        self.head_compress = nn.Linear(embed_dim * num_population, head_dim)
        self.head = nn.Linear(head_dim * (self.num_patches if head_mode == 'flatten' else 1), pred_len)
        block_args = dict(embed_dim=embed_dim, num_population=num_population, input_scale=input_scale, **neuron_args)
        if architecture == 'patch':
            blocks = []
        elif architecture == 'tcn':
            blocks = [TemporalBlock(dilation=d, **block_args) for d in [1, 2]]
        elif architecture == 'patchtst':
            blocks = [AttentionBlock(num_patches=self.num_patches, num_heads=num_heads, **block_args), ChannelBlock(**block_args)]
        elif architecture == 'tsmixer':
            blocks = [TokenBlock(num_patches=self.num_patches, **block_args), ChannelBlock(**block_args)]
        else:
            raise ValueError('Unknown architecture')
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, return_aux=False, memory_mode=None):
        if x.ndim != 3 or x.shape[1] != self.seq_len:
            raise ValueError('Expected [B, seq_len, C] input')
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B * C, L)
        x = x.unfold(-1, self.patch_size, self.patch_size).permute(1, 0, 2).contiguous()
        current = self.embedding.proj(x) * self.embedding.input_scale
        result = self.embedding.lif(current, return_aux=return_aux, memory_mode=memory_mode)
        if return_aux:
            spikes, aux = result
            layers = {'embedding': aux}
        else:
            spikes = result
        for i, block in enumerate(self.blocks):
            result = block(current, spikes, return_aux=return_aux, memory_mode=memory_mode)
            if return_aux:
                current, spikes, aux = result
                layers['block' + str(i)] = aux
            else:
                current, spikes = result
        z = self.head_compress(spikes.flatten(2))
        z = z.transpose(0, 1).reshape(B * C, -1) if self.head_mode == 'flatten' else z[-1]
        output = self.head(z).reshape(B, C, self.pred_len).transpose(1, 2)
        return (output, dict(aux, layers=layers)) if return_aux else output
```

## 16. 기존 연구 연결점과 해석 범위

이 항목은 이전 조사/README의 출처를 인계하는 것이며 이번 문서 작성에서 새 문헌 검색을 수행한 것은 아니다. 신규성 확정이나 논문 원본 성능 재현을 의미하지 않는다.

- Sparsemax: [Martins & Astudillo 2016](https://proceedings.mlr.press/v48/martins16.pdf), 코드의 simplex projection/지원집합 backward 근거.
- [Spike-TCN](https://arxiv.org/html/2402.01533v2): causal convolution과 spiking forecasting 연결; 이번 persistent patch state/current residual은 원본과 다르다.
- [PatchTST](https://arxiv.org/abs/2211.14730): patch/channel-independent forecasting 비교 구조의 출발점. 이번 구조는 causal population-spiking adaptation이다.
- [TSMixer](https://arxiv.org/abs/2303.06053): temporal/feature MLP mixing 비교의 출발점.
- Fractional dynamics/neuronal heterogeneity 관련 상세 선행연구 맥락은 사용자 concept §32를 참조한다. 이 실험의 직접 검증 대상은 다양한 beta의 population과 내부 selective read의 결합이다.
