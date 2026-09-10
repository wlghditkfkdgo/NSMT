# NSMT 실험 폴더 구조

2026-09-09 기준으로 현재 작업 공간인 `NSMT/` 안의 파일을 정리했다.

```text
NSMT/
├── model_v1/forecasting/{코드, dataset, log/, results/, scripts/}
├── neorecall_v1/forecasting/{코드, dataset, log/, results/, scripts/}
├── neorecall_v2/forecasting/{코드, dataset, log/, results/, scripts/}
├── neorecall_ad_v1/anomaly_detection/{코드, dataset, log/, results/, scripts/}
├── forecasting/{dataset/, scripts/unzip_all.sh}
├── anomaly_detection/{dataset/, scripts/unzip_datasets.sh}
├── scripts/{gpu_pool2.sh, SETUP_ON_TARGET.sh}
└── docs/
```

`model_v1/forecasting`은 `neorecall_v1/forecasting`의 독립 복사본이다. 원본은 유지하며, 기존 실험 기록을 옮기기 전에 복제했다. 데이터셋 링크는 공유 데이터셋을 가리킨다. 두 프로젝트의 Python 소스는 동일하며 각자 수정할 수 있다.

| 기존 위치 | 현재 위치 |
| --- | --- |
| `logs_recall/*` | `neorecall_v1/forecasting/log/*` |
| `results_recall/*` | `neorecall_v1/forecasting/results/*` |
| `logs_recall_ad/*` | `neorecall_ad_v1/anomaly_detection/log/*` |
| `results_recall_ad/*` | `neorecall_ad_v1/anomaly_detection/results/*` |
| `logs_tune/*` | 실행 모델에 따라 `neorecall_v1/forecasting/log/tune/*` 또는 `neorecall_v2/forecasting/log/tune/*` |
| `results_tune/raw.txt` | 실행 모델에 따라 `neorecall_v1/forecasting/results/tune/raw.txt` 또는 `neorecall_v2/forecasting/results/tune/raw.txt` |
| 루트의 `run_recall.sh`, `run_recall_v2.sh`, `run_recall_ad.sh` | 각각 해당 모델·작업의 `scripts/` |
| 루트의 `pool_*.log` | 해당 forecasting 모델의 `log/` |
| AD 작업 폴더의 기존 실행 기록 및 `*.out` | `neorecall_ad_v1/anomaly_detection/log/` |

튜닝 기록의 소속은 보존된 pool 실행 명령의 모델, 데이터셋, 예측 길이, 조건, TAG로 판별했다. `raw.txt`는 v1 228행과 v2 1,045행으로 분리했으며 각 모델 내 원래 순서와 모든 행을 보존했다. `_pre`가 붙은 reservoir 대조군은 해당 v2 실행 기록에, 중단된 refine 실행 3행도 v2에 포함했다. v1의 `pool_p0p2.log`는 당시 함께 실행된 AD 명령도 담은 혼합 실행 기록으로 원문 그대로 보관했다.

파일별 이동 위치, 복제 당시 해시, 결과 분리 전 해시와 원래 행 번호는 [reorganization_manifest.json](reorganization_manifest.json)에 기록했다. 기존 stdout, 체크포인트 내부 설정 및 pool 로그의 과거 절대 경로는 기록 보존을 위해 바꾸지 않았다. 과거 체크포인트를 지정할 때는 이동된 현재 경로를 사용한다.

## 실행

아래 명령은 `NSMT/`에서 실행한다. 모델 실행기는 스크립트 위치로 작업 폴더를 찾으므로 다른 디렉터리에서도 절대 경로로 실행할 수 있다.

```bash
bash model_v1/forecasting/scripts/run_recall.sh ETTh1 96 baseline 7 0
bash neorecall_v1/forecasting/scripts/run_recall.sh ETTh1 96 recall_raw 7 0
bash neorecall_v2/forecasting/scripts/run_recall_v2.sh ETTh1 96 recall_raw 7 0
bash neorecall_ad_v1/anomaly_detection/scripts/run_recall_ad.sh SMD recall 7 0
```

모델 실행기의 기본 출력은 해당 작업의 `log/`와 `results/raw.txt`이다. `train.py`를 직접 실행할 때도 기본 로그 위치는 해당 코드 폴더의 `log/`이며 상대 `--log_dir`는 코드 폴더를 기준으로 해석한다. 기존 실행별 로그·체크포인트·평가 CSV 묶음의 내부 구조는 유지했다.

튜닝 기록을 따로 모으려면 해당 모델 아래 경로를 지정한다. 셸 실행기의 상대 `LOG`·`RES`는 명령을 호출한 디렉터리를 기준으로 절대 경로로 변환된다.

```bash
LOG="$PWD/model_v1/forecasting/log/tune" \
RES="$PWD/model_v1/forecasting/results/tune" \
TAG=trial1 bash model_v1/forecasting/scripts/run_recall.sh ETTh1 96 recall_raw 7 0
```

공용 GPU 실행기는 `bash scripts/gpu_pool2.sh <queue_file> <gpu_csv>`로 실행한다. 큐 명령은 `NSMT/`를 기준으로 평가하므로 `bash model_v1/forecasting/scripts/run_recall.sh ...`처럼 새 스크립트 경로를 사용한다. pool 출력도 해당 작업의 `log/`로 리다이렉트한다. 서버 설정 스크립트는 `bash scripts/SETUP_ON_TARGET.sh`이며 실제 데이터셋 디렉터리가 있으면 유지한다.

## 검증

셸 스크립트 28개의 `bash -n`, Python 문법 검사, v1과 model_v1 Python 소스 동일성, 이동 파일 수, 튜닝 결과의 원래 행 재구성 검사를 통과했다. 네 작업의 실제 설정 객체를 다른 작업 디렉터리에서 생성해 기본·상대·절대 로그 경로를 확인했다. 임시 디렉터리에서 학습 호출을 모의 처리하여 네 실행기의 기본·상대 출력 경로, 공용 GPU 실행기의 작업 디렉터리, 서버 설정 스크립트의 데이터셋 보존 및 링크 생성을 확인했다. 실제 모델 학습은 실행하지 않았다.

## 작업 대기열과 잠금 파일

`queue_*.txt`는 GPU 실행기가 읽는 명령 대기열이다. 작업을 가져올 때마다 해당 행을 삭제하므로 빈 파일은 대기 중인 명령이 없다는 뜻이다. 완료·성공 여부는 실행 로그와 결과로 확인한다. `<queue_file>.lock`은 여러 GPU worker의 대기열 접근을 직렬화하는 `flock`용 파일이며 내용이 비어 있는 것이 정상이다. 파일의 존재 자체가 실행 중임을 뜻하지 않는다.

2026-09-09 루트에 남아 있던 빈 대기열 20개와 잠금 파일 20개를 실행 중인 pool이 없고 잠금 획득이 가능한 것을 확인한 뒤 삭제했다. 앞으로 대기열과 잠금 파일은 각 작업의 `scripts/queues/`에 둔다. 실행기는 전달받은 대기열 옆에 잠금 파일을 생성한다.

```bash
mkdir -p model_v1/forecasting/scripts/queues
Q=model_v1/forecasting/scripts/queues/trial.txt
# 위 파일에 실행할 명령을 한 줄씩 작성한 뒤 실행한다.
bash scripts/gpu_pool2.sh "$Q" 0,1,2,3 \
  > model_v1/forecasting/log/pool_trial.log 2>&1
```

대기열 파일은 사용 전에 작성해야 한다. 실행 중인 대기열·잠금 파일은 이동하거나 삭제하지 않는다.
