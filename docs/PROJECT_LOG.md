# PROJECT_LOG — NSMT 프로젝트·실험 통합 기록

이 파일은 실제 Git 저장소 루트의 `docs/PROJECT_LOG.md`이다. `NSMT/docs/PROJECT_LOG.md`는 이 파일을 가리키는 링크로, 별도의 기록을 만들지 않는다. 기존 실험 보고서는 참고 자료로 보존하며 앞으로의 변경·실험·정정은 이 파일 끝에 날짜별로 append한다.

## 운영 규칙 — 2026-09-10 도입

1. `main`은 재현 가능한 기준 코드로 유지한다. 실험 중에는 직접 커밋하거나 push하지 않는다. 실험을 검증하고 사용자가 통합을 요청한 경우에만 별도로 반영한다.
2. 새 실험은 `exp/<실험명>` 브랜치에서 시작한다. 보통 최신 `origin/main`에서 분기하며, 다른 실험을 기반으로 할 때는 기준 브랜치와 정확한 commit을 기록한다.
3. 실험 종료 시 코드·설정·이 문서의 결과 기록을 함께 커밋하고 annotated tag `exp/<실험명>-<YYYYMMDD>`를 만든다. 같은 날짜에 반복하면 접미사를 붙인다. 이미 공개한 태그는 이동하지 않는다.
4. 태그가 가리키는 커밋에 태그 이름을 기록하면 자기 자신의 commit hash를 본문에 넣을 수 없는 문제 없이 정확한 상태를 식별할 수 있다. 후속 기록에서 필요하면 해시도 append한다.
5. 실험명, 목적/가설, 기준 commit, 변경 파일·동작, 데이터셋·분할·전처리, 환경, 실행 명령, seed, 하이퍼파라미터, 지표, 산출물 위치, 검증/실패, 결론 및 commit/tag를 기록한다. 수행하지 않은 항목은 명시한다.
6. 브랜치와 해당 태그만 명시적으로 push한다. `--all`, `--mirror`, force push는 사용하지 않는다.
7. 데이터셋·체크포인트·캐시·압축본·원시 콘솔/TensorBoard 로그는 로컬에 보존한다. 코드·설정·문서·CSV/TXT/JSON 결과 표와 선정된 시각화는 Git에 보존한다. 대용량 산출물의 원격 보관이 필요하면 별도 저장 위치와 체크섬을 기록한다.
8. NSMT에서 로그와 결과는 `<모델>/<작업>/log/`, `results/`, 실행기는 `scripts/`, 대기열과 잠금 파일은 `scripts/queues/` 아래에 둔다.

새 실험 시작 예시(저장소 루트, 작업 트리가 정리된 상태):

```bash
git fetch origin
git switch -c exp/<experiment-name> origin/main
# 코드 수정·실험 수행 후 이 파일 끝에 실험 기록 추가
git add <code-and-config-paths> docs/PROJECT_LOG.md
git commit -m "experiment: <purpose and result>"
git tag -a exp/<experiment-name>-<YYYYMMDD> -m "<result summary>"
git push -u origin exp/<experiment-name>
git push origin refs/tags/exp/<experiment-name>-<YYYYMMDD>
```

---

## 2026-09-10 — 현재 로컬 상태 보존 및 브랜치 운영 도입

### 식별과 범위

- 구분: **코드·기존 결과 스냅샷**. 새로운 학습 실험 또는 검증 완료 기준 모델이 아니다.
- 브랜치: `exp/local-snapshot-20260910`
- 스냅샷 태그: `snapshot/local-state-20260910` (annotated)
- 시작 시 로컬 `main` / 기준 commit: `189bf5997f58a6a4ea1c765b81201d2b90353828`
- 시작 시 원격 `main`: `191f366c6b9dd3cbfaeeb28bb49e7800c8ab488e`
- 원격: `https://github.com/wlghditkfkdgo/NSMT.git`
- 원격 main은 로컬 main보다 5커밋 앞섰다. 이번에는 요청된 로컬 상태를 보존하기 위해 로컬 HEAD에서 분기했다. 원격 main의 추가 변경을 merge하지 않았으며 main으로 push하지 않는다.
- 실제 Git 루트는 `Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/`이고 `NSMT/`는 그 하위 폴더다. 이번 스냅샷은 NSMT만이 아니라 루트의 classification·forecasting·anomaly_detection 코드와 기존 결과까지 포함한다.

### 보존한 기존 변경

- NSMT의 `neorecall_v1`, `neorecall_v2`, `neorecall_ad_v1` 및 신규 `model_v1` 작업 코드를 처음 Git에 포함했다.
- `model_v1/forecasting`은 `neorecall_v1/forecasting`을 독립 복제한 상태다. Python 소스의 동일성을 확인했다. 데이터셋은 공유한다.
- 루트 classification의 로컬 변경에는 import 경로 조정, SMOTE 처리 제거, encoder의 embedding/temporal block 변경 및 시각화 코드가 포함되어 있다.
- 루트 forecasting의 로컬 변경에는 iTransformer 설정·모델 등록, 평가 시 replay를 batch 축으로 이동하는 옵션과 유틸리티, 평가 코드 및 데이터셋별 실행 옵션 변경이 포함되어 있다.
- 기존의 baseline·ablation·분석 코드, 노트북, 환경 정의, 텍스트 결과 및 보고서를 현재 내용으로 보존했다. 과거 변경의 실험 성능을 이번 작업에서 재검증한 것은 아니다.

### 2026-09-09~10 폴더 정리와 이번 운영 변경

- 기존 `logs_recall` / `results_recall`은 `NSMT/neorecall_v1/forecasting/log` / `results`로 이동했다.
- AD의 로그·결과는 `NSMT/neorecall_ad_v1/anomaly_detection/log` / `results`로 이동했다.
- 혼합된 tuning 기록은 pool 명령의 실행 모델로 분류하여 각 모델의 `forecasting/log/tune`, `forecasting/results/tune`으로 분리했다. 결과 표는 v1 228행, v2 1,045행이며 원문 순서를 복원해 누락이 없음을 확인했다. 이 행 수에는 중단 기록도 포함되므로 성공한 실험 수를 뜻하지 않는다.
- 모델별 실행기는 `<모델>/<작업>/scripts/`, 공용 실행기는 `NSMT/scripts/`로 이동했다. 상대 경로를 사용하는 호출과 기본 출력 경로를 수정했다.
- 사용 중이 아닌 빈 queue 파일 20개와 lock 파일 20개를 삭제했다. 앞으로는 작업별 `scripts/queues/`에 생성한다.
- 이번 커밋에서 네 모델의 절대 dataset 심볼릭 링크를 같은 대상을 가리키는 상대 링크로 바꿔 clone 위치에 의존하지 않도록 했다.
- 루트 `AGENTS.md`에 브랜치·태그·append-only 문서 규칙을 기록했다. `.gitignore`로 로컬 데이터·산출물 제외 범위를 명시하고, 이미 추적하던 Python 캐시 12개는 Git 인덱스에서만 제거했다. 실제 로컬 캐시와 데이터는 보존했다.
- 새 clone에서도 필요한 빈 `scripts/queues/` 및 신규 모델 출력 디렉터리가 유지되도록 `.gitkeep`을 추가했다.

### 데이터와 환경 / 재실행 준비

- 데이터셋과 체크포인트는 이번 Git 업로드에 포함되지 않는다. NSMT 공유 데이터는 `NSMT/forecasting/dataset/`, `NSMT/anomaly_detection/dataset/`, 루트 작업 데이터는 각 작업의 `dataset/`와 classification의 `dataset_pt/`에 로컬로 남아 있다.
- ETT 실행은 `NSMT/forecasting/dataset/ETT-small/<ETTh1|ETTh2|ETTm1|ETTm2>.csv`가 필요하다. AD는 `NSMT/anomaly_detection/dataset/<데이터셋>/`에 기존 전처리 데이터를 준비한다. 이번 작업에서 데이터 분할·전처리를 변경하지 않았다.
- 과거 학습의 완전한 재현에는 해당 데이터, 전처리, seed 및 환경이 별도로 필요하다. 이 스냅샷을 clone하는 것만으로 전체 과거 실험이 즉시 재현된다고 보장하지 않는다.
- 현재 NSMT 실행기의 기본 Python은 `/home/yschoi/.conda/envs/snn_recall/bin/python`이다. 다른 서버에서는 `PY`로 경로를 지정한다. 실행기는 해당 환경의 `lib`를 `LD_LIBRARY_PATH`에 추가한다.
- 관측한 Python 환경: 3.10.18. 설치 패키지 버전 목록은 [environments/snn_recall-20260910-packages.txt](environments/snn_recall-20260910-packages.txt). 이 목록은 현재 환경의 inventory이며 새 환경 설치 검증을 수행한 lockfile은 아니다. 기존 `env.yml`, 작업별 `environment.yml`도 보존했다.
- `NSMT/scripts/SETUP_ON_TARGET.sh`는 기존 데이터 디렉터리를 유지하고 준비된 데이터셋을 연결한다. 누락된 데이터 다운로드나 전처리를 대신하지 않는다.

실행 예시(저장소 루트에서 시작, **이번 작업에서는 학습 미실행**):

```bash
cd NSMT
PY=/path/to/snn_recall/bin/python \
  bash model_v1/forecasting/scripts/run_recall.sh ETTh1 96 baseline 7 0
PY=/path/to/snn_recall/bin/python \
  bash model_v1/forecasting/scripts/run_recall.sh ETTh1 96 recall_raw 7 0
```

이 예시의 데이터는 ETTh1, 예측 길이는 96, seed는 7, GPU는 0이다. 실행기의 기본 설정은 epoch 50, patience 3, warmup 0, batch size 64, seq_len 96, patch size 8, embedding 64, heads 8, learning rate 0.001이다. 나머지 설정과 조건별 플래그는 태그 시점의 실행기와 config를 기준으로 한다. 앞으로 수행할 실제 실험에서는 사용한 환경 변수 override까지 기록한다.

### 기존 결과 위치와 성능 기록

- v1 기존 요약: `NSMT/neorecall_v1/forecasting/results/raw.txt`
- v1 tuning: `NSMT/neorecall_v1/forecasting/results/tune/raw.txt`
- v2 tuning: `NSMT/neorecall_v2/forecasting/results/tune/raw.txt`
- AD 기존 요약: `NSMT/neorecall_ad_v1/anomaly_detection/results/raw.txt`
- 원시 로그·체크포인트는 해당 작업 `log/` 등에 로컬로 남는다. CSV/TXT/JSON 형태의 기존 지표·설정은 포함한다.
- 이 작업의 새 MSE/MAE/F1 결과: **없음 — 새 학습을 실행하지 않음**.
- 기존 상세 분석: [46_neorecall_results_and_next.md](../NSMT/docs/46_neorecall_results_and_next.md), [reservoir_summary.md](../NSMT/docs/reservoir_summary.md).
- 폴더 이동 및 기존 결과 행 번호: [reorganization_manifest.json](../NSMT/docs/reorganization_manifest.json). 파일 배치 설명: [project_layout.md](../NSMT/docs/project_layout.md).

### 검증과 한계

- 이전 폴더 정리에서 NSMT 셸 스크립트 28개 문법, 모든 NSMT Python 문법, 네 모델의 실제 config 기본/상대/절대 로그 경로 및 모의 학습 호출을 통한 실행기 경로 검사를 통과했다. 공용 pool의 작업 디렉터리와 setup의 데이터 보존/링크도 확인했다.
- v1과 model_v1 Python 소스 동일성, 이동 파일 수, tuning 결과 원문 복원 검사를 통과했다.
- 이번 저장소 전체 문법 점검에서 기존 미완성 파일 2개를 확인했다: `anomaly_detection/light_trainer.py:270`의 `class trainer(pl.)`, `forecasting/scripts/ETTh2_Spikformer_336.sh:63`의 여분 `done`. 현재 상태 보존을 위해 임의 수정하지 않았다. 이 파일들을 실행 가능한 기준 코드로 간주하면 안 된다.
- 이번 스냅샷은 검증된 NSMT 경로와 미완성된 과거 실험 파일이 공존한다. `main`으로 승격하지 않는다. GPU 학습, 전체 프로젝트 import 및 새 환경 설치는 수행하지 않았다.
- Git 포함 범위에서 95MiB 초과 파일 및 대표적인 private-key/token 패턴이 검출되지 않았다. 대용량 로컬 산출물 제외 여부를 확인했다.

### 결론

현재 로컬 코드와 기존 텍스트 결과를 실험 브랜치 및 스냅샷 태그로 식별 가능한 상태로 보존한다. 이후 새 아이디어는 별도 `exp/<실험명>`에서 수행하고 실행 내용·결과·commit/tag를 이 파일 끝에 추가한다. 이번 스냅샷의 기준은 `snapshot/local-state-20260910` 태그가 가리키는 commit이다.

### 2026-09-10 스냅샷 최종 사전 점검

- 전체 Python 219개·셸 72개를 문법 검사했다. 위에 기록한 기존 파일 2개 외에 문법 오류는 발견하지 않았다.
- Git 포함 파일은 41793개, 파일 크기 합계 약 169.40MiB이다(이 기록의 추가 길이는 제외). 데이터셋 안의 전처리 Python·압축 해제 셸 코드 4개는 데이터 제외 규칙의 예외로 포함했다.
- 데이터 본문·체크포인트·Python 캐시·압축본·stdout·잠금 파일이 포함되지 않았음을 확인했다. 통합 문서 링크와 v1/model_v1 Python 소스 동일성을 다시 확인했다.
