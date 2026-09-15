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

---

## 2026-09-10 — Population coding + 시간축 LIF + IAND SSA v1 구현 스냅샷

### 목적, 기준과 식별

- 구분: **임시 모델 구현·합성 입력 검증 스냅샷**. 데이터셋 학습 완료 실험이 아니다.
- 목적: 원래 시계열 또는 patch 순서를 유일한 spiking simulation 축으로 사용하고, Gaussian population K축에서 multi-head SSA를 수행한다. Hippo의 Memory Replay/cross-attention을 제거하고 IAND self-attention + IAND MLP로 돌아간다.
- 브랜치: `exp/population-spikformer-v1`.
- 기준: `exp/local-snapshot-20260910`의 `c66ffcee91e39e42d88c47023e1d4974b71d5a6d`. 사용자가 말한 기존 Hippo와 `model_v1/forecasting`을 보존하기 위해 이 스냅샷을 기준으로 선택했다. 관측된 `origin/main`은 `191f366c6b9dd3cbfaeeb28bb49e7800c8ab488e`; 이번 작업에서 fetch/merge하지 않았다.
- 시작 작업 트리: clean. main 변경/통합 및 원격 push: not run.
- 이 항목과 코드의 commit 식별자: annotated snapshot tag `exp/population-spikformer-v1-20260910`이 가리키는 commit. 태그는 완료된 학습을 의미하지 않는다.

### Population coding 검토와 차원 결정

- Population coding은 하나의 값을 여러 뉴런의 tuning response 패턴으로 나타내는 방법이며, 그 자체가 rate/latency 등의 spike 생성 규칙은 아니다. 이번에는 겹치는 Gaussian receptive field를 사용한다: `r_k(z) = exp(-0.5 * ((clip(z, low, high) - mu_k) / sigma)^2)`.
- K개 중심을 `[low, high]`에 균등 배치하고 `sigma = population_width * (high-low)/(K-1)`로 둔다. 중심/폭은 학습하지 않는 buffer다. 큰 폭은 활성 population 수를 늘리고 작은 폭은 표현을 희소하게 만든다. 범위 밖 값은 경계에 clip되므로 큰 진폭 간 차이는 사라진다. 최적 K/범위/폭은 검증하지 않았다.
- `L`은 원래 시계열 길이, `T`는 실제 simulation 길이로 구분했다. raw에서는 `T=L`, patch에서는 `T=patch count`이다. 각 관측을 반복하는 별도 T축은 만들지 않는다.
- raw: `[B,L,C] -> [BC,L] -> [BC,L,K] -> [L,BC,K,1] -> Linear(1,D)-BN-LIF -> [L,BC,K,D]`.
- 사용자 설명의 raw `K->D` projection은 K를 소거하므로 K-by-K attention과 양립하지 않는다. 각 population scalar를 공유 `1->D` projection으로 올리는 해석을 적용했다. 같은 반응값을 가진 서로 다른 중심은 초기 SSA feature가 같으며, 명시적인 population identity embedding은 없다. Ordered flatten head가 K별 위치를 구분한다. Center embedding/learned population embedding은 후속 검토 대상이다.
- patch: 먼저 `[BC,T,P]`로 나누고 각 P 위치의 scalar에 population coding을 적용해 `[BC,T,K,P]`; `P->D` embedding을 거쳐 `[T,BC,K,D]`로 만든다. P축의 여러 값을 먼저 합쳐 하나의 population response로 만드는 방식이 아니다.
- direct: 연속 Gaussian response를 embedding 전류로 넣고 embedding LIF에서 첫 binary spike를 만든다. rate: 각 실제 시점/patch에서 `Bernoulli(r)` 한 번을 샘플링한 뒤 embedding한다. 평가 때도 stochastic이며 고정 encoder의 sampling에는 입력 gradient가 흐르지 않는다. 동일 값을 여러 번 관측하는 rate 추정과 구별해야 한다. Latency coding: not implemented.
- heads=H일 때 Q/key/V는 `[T,BC,H,K,D/H]`, attention은 `[T,BC,H,K,K]`이다. `D % H == 0`만 필요하고 K는 H로 나누어떨어질 필요가 없다. Attention은 softmax 확률이나 binary map이 아니라 scaled spike-coincidence count다. 시간 쌍의 attention map은 만들지 않으며 temporal memory는 LIF가 담당한다.
- SSA 및 MLP 잔차는 둘 다 `x * (1 - branch(x))`이다. Binary를 보존하지만 없던 spike를 만들 수 없으므로 층이 깊어질수록 희소성/침묵을 관찰해야 한다.

근거 원문:

- De & Chaudhuri, *Common population codes produce extremely nonlinear neural manifolds*: [PNAS](https://doi.org/10.1073/pnas.2305853120), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10523500/). Gaussian tuning response와 분산 population representation 참고. 본 구현은 이 논문의 전체 모델 재현이 아니다.
- Zhou et al., *Spikformer: When Spiking Neural Network Meets Transformer*: [논문](https://arxiv.org/pdf/2209.15425). Binary Q/K/V, softmax 없는 SSA, multi-head 확장, scale의 역할 참고.
- Fang et al., *Deep Residual Learning in Spiking Neural Networks*: [논문](https://proceedings.neurips.cc/paper/2021/file/afe434653a898da20044041262b3ac74-Paper.pdf). SEW IAND 잔차 참고.

### 코드, 설정, 출력

- 새 모델 소스: `NSMT/forecasting/simple_test_model_v1.py` 하나. 공개 클래스 `SimpleTestModelV1` 및 alias `Model`; `model(x)`는 `[B,pred_len,C]`, `return_aux=True`는 예측과 detached population/encoded/embedding/SSA map/SSA spikes/block spikes를 반환한다.
- 참고 구현: `NSMT/model_v1/forecasting/ours.py`의 Embedding/Block 및 `layers.py`의 SpikLinearLayer/MLP/SSA_rel_scl. Linear-BN-LIF, Sigmoid surrogate, detach_reset, IAND, 기존 `D**-0.5` scale을 유지했다. CUDA 전용 설정과 무관한 import를 피하도록 필요한 작은 모듈을 파일 안에 옮겨 구성했다.
- Neocortex, replay, 보조 reconstruction, temporal repeat, positional bias는 구성하지 않는다. 모델마다 매 forward 시작 시 모든 LIF state를 reset한다. 기존 train.py/model.py 등록 및 기존 Hippo의 auxiliary tuple 계약 연결: not run. 임시 독립 모델이다.
- 기본값: raw, direct, seq_len=96, pred_len=96, K=16, D=64, heads=8, depth=2, mlp_ratio=2, tau=2, threshold=1, detach_reset=True, bias=False, population range=[-3,3], width=1 (sigma=0.4), normalize=True, readout=flatten, backend=torch, attn_scale=None -> D**-0.5=0.125.
- 초기화는 PyTorch Linear 기본값을 사용했다. 기존 전체 Hippo의 timm trunc_normal 초기화를 복제하지 않았다.
- 각 입력 window/channel의 mean/std를 detach하여 normalize하고 예측을 역변환한다. target/미래 horizon 통계는 사용하지 않는다. 학습 BN은 T/BC/K를 함께 집계한다. 따라서 전체 관측 window를 사용하는 offline 예측기이며 strict causal streaming encoder가 아니다.
- patch 기본 P=8, stride=P (non-overlap). 기존 Hippo의 P/2 overlap 기본값과 다르다. `stride`로 overlap을 선택할 수 있고 관측값을 건너뛰는 stride>P는 거부한다. 마지막 불완전 patch는 마지막 관측을 replicate-pad한다. Raw/patch는 simulation step 수가 달라 동일 tau라도 원래 시간 단위 memory 길이가 달라진다.
- Head: flatten은 T,K,D를 전부 보존하여 선형 예측한다. `mean`은 T 평균, `last`는 마지막 T만 읽으며 둘 다 K,D를 보존한다. Flatten head weight 수는 `T*K*D*pred_len`; 기본 raw에서는 9,437,184개로 head가 클 수 있다. 세 readout의 성능 비교: not run.
- JSON 결과: `NSMT/simple_test_model_v1/forecasting/results/smoke-20260910-cpu.json`, `smoke-20260910-cuda-cupy.json`, `activity-20260910-cpu.json`. 모델 소스는 사용자 지정 `NSMT/forecasting/`에, 결과는 작업 규칙의 `<model>/<task>/results/`에 저장했다.
- 데이터셋/분할: 없음 — 합성 torch.randn 입력만 사용. 실제 데이터 전처리 변경, dataset training, MSE/MAE benchmark: **not run**. 체크포인트/원시 학습 로그: 생성하지 않음.

### 환경과 정확한 명령

- 작업 디렉터리: 저장소 루트 아래 `NSMT/`.
- Python `/home/yschoi/.conda/envs/snn_recall/bin/python`, 3.10.18; PyTorch 1.12.0+cu113; SpikingJelly 0.0.0.0.14; CuPy CUDA11x 13.5.1 (기존 inventory). CUDA 검증은 GPU 0, NVIDIA RTX A6000. 패키지 설치/환경 변경: not run.
- 합성 smoke: 매 조합 seed=7, repeat/state-reset 검사 seed=23. B=2,L=13,C=3,pred_len=5,K=7,D=16,H=4,depth=2,MLP ratio=2,P=4,stride=P; raw/patch × direct/rate × flatten/mean/last 12조합. Adam lr=0.001, 각 조합 2 optimizer steps; 나머지는 생성자 기본값. CPU와 GPU의 RNG 샘플/결과가 같다고 가정하지 않는다.
- 추가 boundary check: (L,P,stride)=(13,4,3),(3,8,8), pred_len=2,D=8,H=2,depth=1,K=16. Gaussian 중심/clip 및 별도 LIF sequence `[1.5,1.5,1.5]`의 spike `[0,1,0]` 검사도 내장했다.

```bash
git switch -c exp/population-spikformer-v1 c66ffcee91e39e42d88c47023e1d4974b71d5a6d
mkdir -p simple_test_model_v1/forecasting/results
/home/yschoi/.conda/envs/snn_recall/bin/python forecasting/simple_test_model_v1.py --smoke-test --device cpu --backend torch > simple_test_model_v1/forecasting/results/smoke-20260910-cpu.json
/home/yschoi/.conda/envs/snn_recall/bin/python forecasting/simple_test_model_v1.py --smoke-test --device cuda:0 --backend cupy > simple_test_model_v1/forecasting/results/smoke-20260910-cuda-cupy.json
/home/yschoi/.conda/envs/snn_recall/bin/python -m py_compile forecasting/simple_test_model_v1.py
git diff --check
```

초기 SSA activity 진단의 정확한 명령 (seed=7, 학습 모드 BN, gradient/optimizer 없음):

```bash
/home/yschoi/.conda/envs/snn_recall/bin/python - <<'PY'
import json
from pathlib import Path
import torch
from forecasting.simple_test_model_v1 import SimpleTestModelV1
torch.set_num_threads(2)
rows = []
for mode in ('raw', 'patch'):
    for scale in (None, 1.0):
        torch.manual_seed(7)
        model = SimpleTestModelV1(seq_len=96, pred_len=24, input_mode=mode, attn_scale=scale)
        with torch.no_grad():
            _, aux = model(torch.randn(2, 96, 3), return_aux=True)
        rows.append({'input_mode': mode, 'attn_scale': model.blocks[0].attn.scale,
                     'embedding_rate': float(aux['embedding_spikes'].mean()),
                     'ssa_rates': [float(s.mean()) for s in aux['ssa_spikes']],
                     'block_rates': [float(s.mean()) for s in aux['block_spikes']],
                     'attention_max': [float(s.max()) for s in aux['attention']]})
report = {'seed': 7, 'device': 'cpu', 'mode': 'train, no_grad', 'optimizer_steps': 0,
          'input_shape': [2, 96, 3], 'pred_len': 24, 'other_hyperparameters': 'constructor defaults', 'checks': rows}
path = Path('simple_test_model_v1/forecasting/results/activity-20260910-cpu.json')
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
```

### 검증 결과, 한계와 결론

- CPU/torch 및 CUDA/CuPy 각각 12조합 모두 통과. 예측/population/SSA shape, binary 보존, IAND 출력이 입력 spike를 초과하지 않음, finite output/gradient, embedding weight의 nonzero gradient, 2회 optimizer step, 활성 spike가 있는 training 모드의 반복 forward/state reset, 다른 BC 크기의 중간 window, 상수 입력, direct 평가의 channel permutation을 확인했다.
- K=7,H=4 조합으로 head 분할이 K가 아닌 D축임을 검증했다. LIF의 원래 시간 방향 누적과 BC 간 state 분리, Gaussian 중심/endpoint saturation, overlap 및 짧은 시계열 padding 검사도 통과했다. Python 문법 및 whitespace 검사 통과.
- Smoke의 embedding 발화율: CPU 약 2.88–4.32%, CUDA 약 2.72–4.20%; 마지막 block 약 0.82–2.86% / 0.82–2.57%. **두 backend 모두 이 작은 smoke 설정에서 SSA branch 출력 발화율은 0%였다.** Finite gradient/shape 통과가 모든 attention 경로의 유효 학습 또는 모델 성능을 보장하지 않는다.
- 추가 raw/direct 초기 probe에서 Q/K/V weight gradient는 nonzero, SSA projection weight gradient는 0인 경우를 확인했다. Attention LIF 출력이 없으면 projection weight가 받을 spike도 없다는 점을 점검하기 위해 아래 activity 진단을 수행했다.
- 위 gradient probe 명령은 다음과 같다 (CPU, optimizer 없음):

```bash
/home/yschoi/.conda/envs/snn_recall/bin/python - <<'PY'
import torch
from forecasting.simple_test_model_v1 import SimpleTestModelV1
torch.set_num_threads(2)
torch.manual_seed(7)
m = SimpleTestModelV1(seq_len=13, pred_len=5, num_population=7, d_model=16, num_heads=4)
y = m(torch.randn(2,13,3))
y.square().mean().backward()
print({n: float(p.grad.abs().sum()) for n,p in m.named_parameters() if '.attn.' in n and 'linear.weight' in n})
PY
```

- 기본 K=16,D=64,H=8,L=96을 사용한 activity 진단에서도 reference scale=0.125는 raw/patch의 두 SSA block 모두 0%였다. scale=1.0일 때 raw의 SSA 발화율은 3.39%, 2.48%; patch는 2.45%, 2.80%였다. 이는 단일 seed의 초기 합성 응답이며 `1.0`의 예측 성능 우위를 뜻하지 않는다. 기존 scale을 기본값으로 보존하고 `attn_scale=1.0`을 명시해 비교할 수 있다.
- 설계 검토 대상: Gaussian 중심/폭/범위, raw population identity embedding, finite Bernoulli noise와 eval seed/평균 정책, SSA scale와 threshold, IAND 깊이에 따른 침묵, readout 및 head 크기, patch stride/tau의 실제 시간 의미, offline BN과 엄격한 streaming 인과성.
- GPU/CPU 성능 수치 비교, 에너지 절감 검증, 긴 학습의 안정성, 실제 forecasting 정확도, 기존 trainer 통합: **not run**.
- 결론: 요청한 시간축/집단축 분리와 multi-head IAND SSA를 독립 파일로 구현하고 합성 동작을 검증했다. 현재 상태는 다음 설계 논의를 위한 구현 스냅샷이다. 학습 전 SSA 발화 설정의 재검토가 필요하다.

보존 명령 (NSMT 작업 디렉터리):

```bash
git add forecasting/simple_test_model_v1.py simple_test_model_v1/forecasting/results/smoke-20260910-cpu.json simple_test_model_v1/forecasting/results/smoke-20260910-cuda-cupy.json simple_test_model_v1/forecasting/results/activity-20260910-cpu.json ../docs/PROJECT_LOG.md
git commit -m "experiment: snapshot population-coded temporal Spikformer prototype"
git tag -a exp/population-spikformer-v1-20260910 -m "Implementation snapshot: population coding with natural-time LIF and IAND SSA; CPU/CUDA synthetic checks only, no dataset training"
```


---

## 2026-09-10 — ETT quick validation 실행 준비 스냅샷

- 목적: patch + direct population coding, N축 multi-head SSA, 선택적 학습 가능한 population identity, 사용자 제안 `KD -> D′ -> ND′ -> H` head를 실제 ETT forecasting에서 비교한다.
- 브랜치: `exp/population-spikformer-ett-quick`; 기준 commit: `4b9569065b491222648d2fb03e80652124c94e07` (`exp/population-spikformer-v1`). 시작 작업 트리 clean. main 수정/merge/push: not run.
- 구분/식별: **실행 준비 스냅샷**, tag `exp/population-spikformer-ett-quick-20260910-snapshot`. 본학습 결과는 후속 항목에 append한다.
- 소스: `NSMT/forecasting/simple_test_model_v1.py`. 기본 patch/direct, temporal SSA, scale=1.0, two_stage head. Q/K/V의 LIF는 `[N,BC,K,D]`에서 N을 따라 실행하고 matmul만 `[BC,K,h,N,D/h]`로 바꿔 `[BC,K,h,N,N]` map을 구한다. 출력은 다시 N-first로 복원하여 LIF 실행. K축 SSA 및 SSA 제거 대조군도 지원한다. IAND SSA/MLP를 유지한다.
- 학습 가능한 identity는 `[K,D]` parameter이며 embedding BN 뒤/LIF 앞에 더한다. Gaussian 중심/폭은 고정. 추가 embedding은 공통 레이어 뒤에 초기화하여 embedding 유무 비교의 공통 초기 weight를 유지한다. `Normal(0,0.01)` 초기화, BN/입력 window 정규화 유지, strict streaming 모델은 아니다.
- Head: 모든 patch가 공유하는 bias 없는 `Linear(KD,D′)` 후 시간 위치를 유지해 flatten하고 bias 없는 `Linear(ND′,H)`. 추가 활성화/LIF 없음. 기존 raw/rate/flatten/mean/last 경로는 명시적 옵션으로 남긴다.
- 실행기: `NSMT/simple_test_model_v1/forecasting/run_ett.py`; queue launcher는 `scripts/launch_ett.py`, 환경 wrapper는 `scripts/run_parallel.sh`. 한 GPU에 한 독립 프로세스, 종료되면 다음 작업. 실패는 기록하고 나머지 작업은 계속한다. mutable queue/lock/PID는 `scripts/queues/`, 원시 stdout/checkpoint는 `log/`, 구성/epoch history/지표/원본 CSV hash는 `results/`의 JSON으로 보존한다.
- 사전 정의 실험: ETTh1, ETTh2, ETTm1, ETTm2 × prediction length 96,720 × variant population(K축), temporal(N축), temporal_embedding(N축+identity), no_attention(embedding+IAND MLP), linear(공유 per-channel Linear(96,H)+동일 window 정규화) = **40개**, seed=7. 매 test에서 persistence와 window-mean 기준선도 측정한다. 이 행렬은 test 결과를 보고 고른 것이 아니다.
- 공통 SNN: seq_len=96, patch_size=stride=8, N=12, K=16, D=64, heads=8, depth=2, mlp_ratio=2, D′=64, tau=2, threshold=1, attn_scale=1, range=[-3,3], population_width=1, normalize=True, bias=False, backend=cupy, FP32/TF32 off, deterministic algorithms on.
- 학습: 전체 train windows, shuffle/drop_last=False, batch_size=128, 최대 10 epoch, early stopping patience=3, AdamW lr=0.001/weight_decay=0.01, MSE loss, gradient clip norm=1, ReduceLROnPlateau(val MSE, factor=0.5, patience=1), 최저 validation MSE checkpoint 복원 후 test. 기존 Hippo 실행기의 L1 loss/50 epoch와 다른 신속 검증 프로토콜이므로 과거 보고서와 통제된 직접 비교가 아니다. 다중 seed/긴 학습: not run.
- 데이터: `NSMT/forecasting/dataset/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv`, 7개 전체 변수(features=M). 기존 hour 12/4/4개월 경계 [8640,11520,14400], minute은 각 4배. Val/test context는 이전 split 끝의 96점을 포함하지만 target은 해당 split 내에 있다. StandardScaler는 train 구간에만 fit. 지표는 이 standardized scale에서 모든 window/horizon/channel 원소를 합산하고 마지막 partial batch도 포함한다. Window stride=1. GPU resident 작은 배열에서 indexing해 모든 window를 읽는다. 임의 데이터 subset은 본실험에 사용하지 않는다.
- 환경: `/home/yschoi/.conda/envs/snn_recall/bin/python` 3.10.18, torch 1.12.0+cu113, SpikingJelly 0.0.0.0.14, cupy-cuda11x 13.5.1, pandas 2.3.1, numpy 1.26.4, sklearn 1.7.1. GPU 0–3 NVIDIA RTX A6000 각 약 48GiB. 패키지 설치/환경 변경 없음.
- 검증: `check_experiment.py`의 CPU와 CUDA/CuPy 4가지 모델 조건 통과. 모든 LIF 입력의 first axis=N 확인, einsum 참조 attention/aggregation 일치, shared 2-stage head의 시간/K 순서 및 수동 결과 일치, finite gradient/identity gradient, state reset/BC 변경 확인. 네 ETT × 두 horizon × 세 split의 count 및 첫/마지막 window는 원래 Dataset_ETT_hour/minute와 float32 값까지 일치. 문법/whitespace 통과.
- 첫 check 실행은 torch를 먼저 import할 때 시스템 libstdc++의 GLIBCXX_3.4.29 누락으로 실패했다. 기존 실행기와 동일하게 conda lib를 LD_LIBRARY_PATH 앞에 둔 후 두 check 모두 통과했다. 학습 wrapper도 이 경로를 설정한다.
- GPU pilot: ETTh1/p96/temporal, seed7, 1 epoch, **train 4 batches/val-test 2 batches만**. checkpoint 복원 검증 통과, peak GPU memory 약 4.08GiB, 총 약3.12초. 첫 train batch SSA projection 발화율 약2.98%/2.98%; 4 optimizer step 이후 BN running statistics가 아직 부족한 validation은 대부분 침묵했다. Pilot의 부분 test MSE=0.8774057은 전체 benchmark 결과가 아니며 본실험 비교에서 제외한다.
- 사전 검증 결과: `NSMT/simple_test_model_v1/forecasting/results/check-ett-20260910-{cpu,cuda}.json`, pilot metadata/history/metrics는 `results/pilot-20260910/ETTh1_p96_temporal_seed7.json`. 본학습 MSE/MAE: **not run at this snapshot**.

정확한 검증 명령 (NSMT 작업 디렉터리):

```bash
/home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/run_ett.py --dataset ETTh1 --pred-len 96 --variant temporal --suite pilot-20260910 --epochs 1 --max-train-batches 4 --max-eval-batches 2 --device cuda:0
LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} /home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/check_experiment.py > simple_test_model_v1/forecasting/results/check-ett-20260910-cpu.json
LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} /home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/check_experiment.py --device cuda:0 --backend cupy --skip-parity > simple_test_model_v1/forecasting/results/check-ett-20260910-cuda.json
/home/yschoi/.conda/envs/snn_recall/bin/python -m py_compile forecasting/simple_test_model_v1.py simple_test_model_v1/forecasting/run_ett.py simple_test_model_v1/forecasting/check_experiment.py simple_test_model_v1/forecasting/scripts/launch_ett.py
git diff --check
```

스냅샷 이후 본실험 실행 명령 (실제 시작/완료는 후속 항목에 기록):

```bash
bash simple_test_model_v1/forecasting/scripts/run_parallel.sh --suite ett-quick-20260910 --gpus 0 1 2 3 --epochs 10 --patience 3 --batch-size 128 --seed 7
```


---

## 2026-09-10 — ETT quick validation 40개 GPU 병렬 실험 완료

### 식별, 실행 범위와 환경

- 브랜치: `exp/population-spikformer-ett-quick`. 학습에 사용한 고정 code commit은 `9fca022c79358ea705ae9940034a0cdbad0bc0ce`; 준비 snapshot tag는 `exp/population-spikformer-ett-quick-20260910-snapshot`. 이전 아이디어 코드 기준은 `4b9569065b491222648d2fb03e80652124c94e07`이다.
- 이번 결과/분석 commit은 annotated tag `exp/population-spikformer-ett-quick-20260910`으로 식별한다. **사전 정의한 40개 quick training/evaluation을 완료한 실험**이며, 긴 학습이나 기준 모델 승격을 의미하지 않는다. main merge 및 원격 push: not run.
- 실제 실행 시간: 2026-09-10 20:28:37–20:37:10 KST (11:28:37–11:37:10 UTC), launcher wall time 약 **513.07초 = 8분 33초**. 구현 및 사전 점검 시간은 제외했다.
- RTX A6000 GPU 0–3에 각 1개 독립 프로세스를 실행하고 다음 작업을 자동 할당했다. GPU별 완료 작업 수는 0:11, 1:7, 2:9, 3:13; 전체 40개 returncode=0. Peak allocated GPU memory의 최댓값은 약 4.38GiB. 완료 후 네 GPU의 학습 프로세스가 종료된 것을 확인했다. GPU runtime은 작업 배치와 모델/epoch 수의 영향을 받으며 별도 속도 benchmark가 아니다.
- 데이터/분할/환경/seed/하이퍼파라미터는 바로 위 실행 준비 항목과 각 run JSON에 기록한 값 그대로다. 네 ETT × horizon96/720 × 5 variants, seed7, full train/val/test, train-only StandardScaler, 입력96, patch8/stride8, fixed Gaussian K16, direct, D64/head8/depth2/MLP ratio2, two-stage D′64, tau2/threshold1/scale1, AdamW lr.001/wd.01, MSE, max10 epochs/patience3. Test 점수에 따른 설정 변경/추가 run 선택은 하지 않았다.
- 준비 항목의 `shuffle/drop_last=False` 표기를 명확히 정정한다: 실제는 **train shuffle=True, drop_last=False**다. Seed+1000의 별도 torch.Generator로 각 epoch 전체 train window를 permutation한다. Val/test는 순차 평가한다.
- 코드 변경은 준비 snapshot에서 끝냈으며 본학습 중 모델/훈련기 소스를 변경하지 않았다. 이후 추가한 `summarize_ett.py`는 결과 검증/CSV/Markdown/그림 생성 전용이다. 모든 run의 모델/훈련기 SHA-256과 최종 소스가 일치하고, 기록된 학습 code commit도 전부 위 hash로 일치한다.

### 실제 실행과 결과 검증 명령

NSMT 작업 디렉터리에서 실행했다. Launcher는 각 run의 Python command와 GPU 배정을 manifest/completion에 기록하며, 각 run JSON에는 환경변수 CUDA_VISIBLE_DEVICES와 전체 CLI/config가 포함되어 있다.

```bash
bash simple_test_model_v1/forecasting/scripts/run_parallel.sh --suite ett-quick-20260910 --gpus 0 1 2 3 --epochs 10 --patience 3 --batch-size 128 --seed 7
/home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/summarize_ett.py --suite ett-quick-20260910 --plot
```

- `summarize_ett.py`는 40개 manifest 항목이 모두 완료되었는지, test/validation element count가 `windows * horizon * 7`과 일치하는지, 복원 checkpoint의 val MSE가 기록된 epoch 중 최솟값인지, 비교 쌍의 데이터 metadata/code hash가 일치하는지 검사한다. 모두 통과했다. 모든 partial batch가 지표에 포함되었다.
- 학습기 자체에서도 checkpoint 복원 후 validation MSE를 다시 계산해 선택 점수의 재현을 확인했다. 첫 train batch의 gradient norm과 모든 LIF의 첫 train/val/test batch 발화율이 기록되어 있다. Population embedding의 gradient는 모든 8개 run에서 nonzero였다.
- 모든 SNN은 4–6 epoch에서 early stopping됐다. 선형 모델 중 6개는 10 epoch 제한에 도달했고 그중 일부는 마지막 epoch가 best였다. 모든 모델의 수렴을 확인한 실험은 아니다.
- N축 SSA의 선택 checkpoint에서 test 첫 batch SSA projection 발화율은 약 2.00–5.08%였다. 앞선 초기 scale=0.125의 완전 침묵은 이번 N축/scale1 설정에서 관찰되지 않았다. 이는 첫 batch의 진단이며 전체 split 평균 발화율/에너지 측정이 아니다.

### 지표

각 데이터셋/예측 길이에 같은 가중치를 둔 **8개 task macro 평균**이다. 개별 task의 MSE/MAE는 train-standardized scale에서 모든 window/horizon/channel 원소를 평균한 값이다.

| Variant | Macro MSE | Macro MAE | Parameters, H=96 / H=720 |
|---|---:|---:|---:|
| population | 0.381274 | 0.401924 | 207,232 / 686,464 |
| temporal | 0.381355 | 0.402974 | 207,232 / 686,464 |
| temporal_embedding | 0.380970 | 0.403555 | 208,256 / 687,488 |
| no_attention | 0.381260 | 0.403161 | 173,440 / 652,672 |
| linear | 0.374076 | 0.389852 | 9,216 / 69,120 |

N축 기본 모델(embedding 없음)의 개별 test 결과:

| Dataset | Horizon | MSE | MAE | Linear MSE | Linear MAE |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.391540 | 0.409145 | 0.387361 | 0.396180 |
| ETTh1 | 720 | 0.502549 | 0.487203 | 0.464809 | 0.458981 |
| ETTh2 | 96 | 0.309746 | 0.361636 | 0.292030 | 0.340833 |
| ETTh2 | 720 | 0.435166 | 0.453410 | 0.420357 | 0.439469 |
| ETTm1 | 96 | 0.350222 | 0.384385 | 0.353932 | 0.374141 |
| ETTm1 | 720 | 0.464247 | 0.447546 | 0.483544 | 0.445788 |
| ETTm2 | 96 | 0.186414 | 0.270170 | 0.182040 | 0.264575 |
| ETTm2 | 720 | 0.410960 | 0.410298 | 0.408532 | 0.398848 |

비교 지표는 task별 `(tested_MSE/reference_MSE - 1)*100`의 산술 평균이다. 음수가 개선이며 macro MSE 비율과는 다른 집계 방식이다.

| Tested vs reference | MSE wins | Mean relative MSE |
|---|---:|---:|
| temporal vs linear | 2/8 | +2.0932% |
| temporal vs no_attention | 3/8 | +0.1757% |
| temporal vs population | 4/8 | +0.0822% |
| temporal_embedding vs temporal | 4/8 | -0.2304% |

### 해석, 한계와 다음 판단

- **현재 quick protocol에서는 attention의 추가 성능 이점을 확인하지 못했다.** K축/N축/SSA 제거의 macro MSE는 모두 약0.3813이다. N축이 K축보다 낮은 MSE를 보인 task는4/8, SSA 제거보다 낮은 task는3/8이다. 근소한 차이를 우열이나 통계적 유의성으로 해석하지 않는다.
- 학습 가능한 population identity는 N축 대비4/8 개선, 평균 task별 상대 MSE -0.23%였다. Macro MAE는 오히려 조금 높다. Gradient가 흐르고 있으나 일관된 개선의 근거는 부족하다.
- N축 모델은 input-window mean 기준선보다8/8에서 MSE가 낮아 단순한 평균 출력만 하는 상태는 아니다. 하지만 선형 모델보다 MSE가 낮은 task는 ETTm1의96/720 두 조건뿐이고, MAE는8/8에서 선형 모델이 낮다. 현재 복잡한 SNN 구조가 이 선형 기준선을 전반적으로 앞선다고 주장할 수 없다.
- 단일 seed, max10 epochs의 screening이다. 동일 seed를 썼지만 attention 제거 조건의 공통 이후 레이어 초기 weight까지 모두 동일하게 맞춘 실험은 아니다. 여러 seed, 더 긴 학습, axis별 scale/threshold 최적화: not run.
- Gaussian population coding을 제거한 SNN 및 이전 flatten head와의 별도 대조 실험: not run. 따라서 population coding 자체의 효과나 two-stage head의 정확도 이점을 독립적으로 입증한 것은 아니다. Head의 차원/계산/gradient 계약 및 파라미터 감소는 검증했다.
- Window 정규화/BN과 양방향 observed-patch attention을 사용하는 offline forecaster다. Streaming 인과성, 실제 에너지, GPU/CPU 성능 benchmark: not run.
- 결론: 구현과 실제 GPU 학습/평가는 정상 완료했지만, 제안한 N축 attention과 identity embedding의 예측 이점은 이번 결과에서 지지되지 않는다. 다음 비교의 기준은 이번 SSA 제거/선형 결과로 삼고, 단순 축 교체나 identity 추가보다 표현 단계 또는 IAND 억제 방식의 영향을 분리해 검증하는 편이 타당하다. 후속 학습은 이번에 실행하지 않았다.

### 산출물과 보존

- 모델: `NSMT/forecasting/simple_test_model_v1.py`.
- 실행/검증/집계: `NSMT/simple_test_model_v1/forecasting/{run_ett.py,check_experiment.py,summarize_ett.py}`, `scripts/{run_parallel.sh,launch_ett.py}`.
- 전체 결과표: `NSMT/simple_test_model_v1/forecasting/results/ett-quick-20260910/REPORT.md`.
- CSV: 같은 폴더의 `summary.csv`, `comparisons.csv`; macro/비교 집계: `aggregate.json`.
- 사전 정의 행렬 및 실제 GPU/프로세스/종료 기록: `manifest.json`, `completion.json`; 각40개 run JSON은 CLI, config, 데이터/소스 hash, 패키지 환경, epoch history, 발화/gradient, 지표, checkpoint 경로를 포함한다.
- 비교 그림: 같은 폴더의 `comparison.png`, `comparison.pdf`. 표의 상대 MSE 차이로 생성하고 시각적으로 확인했다.
- 체크포인트: `NSMT/simple_test_model_v1/forecasting/log/ett-quick-20260910/<dataset>_p<horizon>_<variant>_seed7/best.pt`. 원시 stdout과 runtime copy는 같은 log 아래 로컬에 보존한다. 데이터/checkpoint/원시 콘솔 로그를 Git에 넣거나 삭제하지 않았다.

완료 보존 명령:

```bash
git add simple_test_model_v1/forecasting/summarize_ett.py simple_test_model_v1/forecasting/results/ett-quick-20260910 ../docs/PROJECT_LOG.md
git commit -m "experiment: record 40 parallel ETT quick-validation results"
git tag -a exp/population-spikformer-ett-quick-20260910 -m "Completed 40 ETT quick runs on four GPUs: horizons 96/720, seed 7; no consistent SSA or population-identity gain over controls"
```


---

## 2026-09-11 — ETT multi-seed 실행 계획과 기준 고정

- 목적: seed7 quick 결과의 작은 차이가 초기화/학습 순서 변화에도 유지되는지 확인한다. 기존 seed7의40개 결과를 재사용하고 seed13/21 각각40개를 추가하여 총120개 결과를 비교한다.
- 브랜치 `exp/population-spikformer-ett-multiseed`; base commit `6834f10174956bafb03605d629842298cc822b8c`. 시작 clean. 모델/훈련기/단일-seed launcher를 수정하지 않고 기존 프로토콜을 그대로 반복한다. main 통합 및 원격 push: not run.
- 실행 준비 snapshot tag: `exp/population-spikformer-ett-multiseed-20260911-snapshot`. 완료 학습 태그가 아니다. 새 추가 실행기는 `NSMT/simple_test_model_v1/forecasting/scripts/run_multiseed.sh`.
- 사전 고정 범위: ETTh1/ETTh2/ETTm1/ETTm2 × pred_len96/720 × population/temporal/temporal_embedding/no_attention/linear × seeds7,13,21. 동일 seed의 동일 task끼리 비교한다. seed7 결과를 보고 특정 조건만 선택하지 않고 전체40개 행렬을 반복한다.
- 공통 설정은 2026-09-10 quick protocol 그대로: seq_len96, P=stride8, Gaussian K16/D64/heads8/depth2/MLP ratio2, direct, two_stage head D′64, tau2/threshold1/scale1, biasFalse, normalizeTrue, cupy FP32. Gaussian range[-3,3]/width1 고정. embedding 조건만 학습 가능한 [K,D] identity를 사용한다.
- 데이터: `NSMT/forecasting/dataset/ETT-small/` 4 CSV의7개 변수. Hour 경계8640/11520/14400, minute은4배; val/test의96점 context 포함. Train-only StandardScaler; 모든 train/val/test windows, stride1, drop_lastFalse; train shuffleTrue. 각 결과 JSON에 CSV hash/정규화 statistics/분할 window 수를 보존한다.
- 학습: AdamW lr0.001/wd0.01, MSE loss, clip norm1, ReduceLROnPlateau(valMSE,factor.5,patience1), max10epochs/early stopping3, batch128. 최저 validation MSE checkpoint 복원 후 전체 test MSE/MAE. Seed별 초기화와 shuffle만 달라진다. Test 결과로 하이퍼파라미터를 바꾸지 않는다.
- 환경: 기존 `/home/yschoi/.conda/envs/snn_recall/bin/python` 환경과 RTX A6000 GPU0–3 사용. Wrapper가 conda lib를 LD_LIBRARY_PATH에 추가한다. 시작 시4개 GPU 모두 idle. 패키지 설치/환경 변경 not run. 정확한 Python/torch/cupy/CUDA와 패키지 버전은 각 run JSON에 자동 기록한다.
- 검증: 기존 seed7의40개 manifest 완료 상태/returncode와 모델/훈련기 SHA-256을 현재 파일과 대조해 재사용 가능함을 확인했다. 모델/훈련 코드 변경이 없어 같은 smoke training을 반복하지 않는다. 새 wrapper는 bash -n으로 검사한다.
- 집계 계획: task별3개 seed의 MSE/MAE 평균과 표본 표준편차(ddof=1), 같은 seed의 paired 차이/승패. 전체 macro는 seed마다8개 task를 먼저 평균하고 그3개 macro의 평균/표준편차를 계산한다. 서로 다른 dataset/horizon을 독립 seed 반복처럼 취급하지 않는다. n=3의 작은 반복만으로 유의성/수렴을 주장하지 않는다.
- 결과 재사용 경로: `results/ett-quick-20260910/`. 새 결과 경로: `results/ett-multiseed-20260911-seed13/`, `results/ett-multiseed-20260911-seed21/`; 종합 결과 예정 위치 `results/ett-multiseed-20260911/` (모두 `NSMT/simple_test_model_v1/forecasting/` 아래).
- Checkpoint/stdout은 같은 작업 `log/<suite>/`, mutable queue/lock/PID는 `scripts/queues/`에 로컬 보존. 데이터/checkpoint/원시 로그 업로드나 삭제 없음.
- 이 snapshot 시점의 seed13/21 학습/성능: **not run**. 완료 후 실제 명령, 시간, 지표와 결론을 append한다.

실행 명령 (NSMT):

```bash
bash -n simple_test_model_v1/forecasting/scripts/run_multiseed.sh
bash simple_test_model_v1/forecasting/scripts/run_multiseed.sh
```

Wrapper는 다음 두 명령을 순서대로 실행하며, 각 명령은4개의 GPU를 병렬 사용한다:

```bash
bash simple_test_model_v1/forecasting/scripts/run_parallel.sh --suite ett-multiseed-20260911-seed13 --gpus 0 1 2 3 --epochs 10 --patience 3 --batch-size 128 --seed 13
bash simple_test_model_v1/forecasting/scripts/run_parallel.sh --suite ett-multiseed-20260911-seed21 --gpus 0 1 2 3 --epochs 10 --patience 3 --batch-size 128 --seed 21
```


---

## 2026-09-11 — ETT 3-seed replication 완료 (120 results, 80 new runs)

### 목적과 기준

- Seed7의 구조 간 작은 차이가 seed13/21에서도 유지되는지 전체 행렬을 반복 검증했다. ETTh1/ETTh2/ETTm1/ETTm2 × H96/720 × 5 variants × seeds7/13/21 = 120개 결과이며, 기존 seed7의40개는 원본 위치에서 재사용하고80개만 새로 학습했다. 모든 새 작업 returncode0, 실패0.
- 브랜치 `exp/population-spikformer-ett-multiseed`, base `6834f10174956bafb03605d629842298cc822b8c`, 실행 준비 commit `4a28741bce8b38848e171014ce1825d3cea1d866`. 준비 snapshot tag는 앞선 계획 항목에 기록되어 있다. 이 완료 항목과 결과/집계 코드를 포함하는 commit은 annotated tag `exp/population-spikformer-ett-multiseed-20260911`로 식별한다.
- 모델/훈련기/단일-seed launcher/기존 seed7 결과는 변경하지 않았다. 새 코드는 `scripts/run_multiseed.sh`, `summarize_multiseed.py`, `check_multiseed_summary.py`다. Seed7의 훈련 당시 commit은 `9fca022c79358ea705ae9940034a0cdbad0bc0ce`, 새 훈련 commit은 위 준비 commit이며 모델/훈련기 파일 hash가 동일함을120개 전체에서 확인했다.
- 모델 SHA256: `50a7362cdde6efc625ee0efa25c2a434091d0abf154ad463b6d51b1c30795402`; 훈련기 SHA256: `8e3e13c8af8e6c9c34849fee920629c96f6bb9eae5b3273d9f4968440ff6b1ca`.

### 데이터, 설정, 실행 환경

- 데이터는 `NSMT/forecasting/dataset/ETT-small/`의4개 CSV,7개 변수, 표준12/4/4개월 분할이다. Hour 경계8640/11520/14400, minute은4배; validation/test에는 이전96점 context를 포함하되 예측 target은 해당 split 안에 있다. Train-only StandardScaler, 입력 window 정규화, 모든 window 사용, stride1/drop_lastFalse. 모든 seed에서 CSV hash, scaler, window 수, model config와 protocol metadata가 동일함을 검사했다.
- 입력96, 예측96/720, patch/stride8, K16/D64/heads8/depth2/MLP ratio2, direct coding, LIF time=N, two-stage head D′64, tau2/threshold1/attention scale1, biasFalse, normalizeTrue. Gaussian 중심[-3,3]/width1은 고정; temporal_embedding만 학습 가능한[K,D] identity를 추가한다. Gaussian 중심/폭 학습 실험이 아니다.
- AdamW lr0.001/wd0.01, MSE loss, clip norm1, ReduceLROnPlateau(valMSE,factor0.5,patience1), batch128, max10epochs/early stopping3. Validation MSE 최소 checkpoint 복원 후 validation 재현 검사 및 전체 test 평가. Test로 설정을 조정하지 않았다.
- Python3.10.18, torch1.12.0+cu113/CUDA11.3, SpikingJelly0.0.0.0.14, CuPy13.5.1, numpy1.26.4, pandas2.3.1, sklearn1.7.1. RTX A6000 GPU0–3, FP32/CuPy, deterministic algorithms, TF32 off. 환경 변경/패키지 설치 not run.
- 실제 실행: seed13 suite는2026-09-11 13:33:28.740–13:42:08.897 KST (520.157초), seed21은13:42:10.926–13:50:59.580 KST (528.653초). 두 suite는 순차 실행하고 각 suite에서4 GPU를 병렬 사용했다. 시작부터 최종 완료까지1050.839초, 약17분31초. 각 suite GPU별 작업 수0/1/2/3 = 9/10/11/10. 최대 torch allocated GPU memory는4.378GiB/run이었다.
- 새 SNN64개 모두 early stopping: seed13은4/5/8/9epoch = 16/14/1/1개, seed21은4/5/6epoch = 15/13/4개. 새 linear16개 중13개가10epoch 제한에 도달했다. 짧은 예산 아래 비교이며 수렴 확인은 아니다.

실제 실행/집계/검증 명령 (cwd NSMT):

```bash
bash -n simple_test_model_v1/forecasting/scripts/run_multiseed.sh
bash simple_test_model_v1/forecasting/scripts/run_multiseed.sh
/home/yschoi/.conda/envs/snn_recall/bin/python -m py_compile simple_test_model_v1/forecasting/summarize_multiseed.py simple_test_model_v1/forecasting/check_multiseed_summary.py
/home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/summarize_multiseed.py --allow-partial > simple_test_model_v1/forecasting/log/multiseed-summary.stdout
/home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/summarize_multiseed.py --plot > simple_test_model_v1/forecasting/log/multiseed-summary.stdout
/home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/check_multiseed_summary.py > simple_test_model_v1/forecasting/results/ett-multiseed-20260911/checks.json
```

Wrapper의 두 정확한 launcher command는 앞선 계획 entry와 각 suite의 manifest/completion에,120개 개별 훈련 CLI는 원본 result JSON에 보존되어 있다. Partial 집계는 진행 확인에만 사용했고 최종 파일은 strict 모드로 다시 생성했다.

### 결과와 검증

아래 macro는 seed마다8개 dataset/horizon task를 먼저 평균한 후, 그3개 평균의 평균 ± 표본 표준편차(ddof=1)다. 서로 다른 task 간 분산을 seed 분산으로 계산하지 않았다. 개별 task 지표는 train-standardized scale에서 전체 test window/horizon/channel 평균이다.

| Variant | Macro MSE mean ± SD | Macro MAE mean ± SD |
|---|---:|---:|
| population | 0.380846 ± 0.001601 | 0.402101 ± 0.001247 |
| temporal | 0.380714 ± 0.000697 | 0.402519 ± 0.000602 |
| temporal_embedding | 0.380848 ± 0.001050 | 0.402922 ± 0.000943 |
| no_attention | 0.381705 ± 0.001340 | 0.402975 ± 0.000896 |
| linear | 0.373984 ± 0.000097 | 0.389648 ± 0.000183 |

Paired ΔMSE = tested minus reference, 동일 task/seed로 짝지었다. 음수는 개선이다.

| Tested vs reference | Macro ΔMSE mean ± SD | Seed-task wins | 3-seed mean task wins | All-3-seed task wins |
|---|---:|---:|---:|---:|
| temporal vs population | -0.000132 ± 0.001151 | 12/24 | 5/8 | 1/8 |
| temporal_embedding vs temporal | 0.000134 ± 0.000769 | 14/24 | 5/8 | 2/8 |
| temporal vs no_attention | -0.000991 ± 0.001277 | 10/24 | 4/8 | 0/8 |
| temporal vs linear | 0.006730 ± 0.000674 | 6/24 | 2/8 | 1/8 |

- Strict 집계 검증 passed:120개 행렬 완전성, manifest/completion status 및 returncode, finite MSE/MAE, 전체 val/test element 수, best validation checkpoint, code/config/data/environment 일치.
- 별도 NumPy [seed,task,variant,metric] tensor로 원본 JSON을 읽어40개 task 평균/표준편차,5개 macro,32개 paired 결과 및 승패를 독립 재계산해 일치 확인했다.120개 checkpoint 존재와 provenance SHA256, 기존40개 재사용 여부도 통과했다. 결과는 `checks.json`에 기록했다.
- 생성 PNG를 시각적으로 확인했다. Error bar는 표본 SD이며 confidence interval이 아니다. 모델/훈련기 변경이 없으므로 앞선 CPU/GPU backbone/loader smoke checks를 반복하지 않았다 (이번 반복 실행 not run). 이번80개 실제 GPU 학습/평가 및 checkpoint 재평가는 완료했다.

### 해석과 한계

- N축 vs K축 평균 ΔMSE는-0.000132이지만 paired seed SD0.001151이며 seed별 macro 우열이 바뀐다. 전체12/24 승, 세 seed 모두 이기는 task는1/8이다. N축 선택의 일관된 이점은 확인되지 않았다.
- N축 vs SSA 제거는 평균-0.000991, paired SD0.001277이다. 평균상 이득은 있지만10/24 승, task 평균4/8 승, 세 seed 모두 이기는 task0/8이다. SSA 추가가 안정적으로 도움이 된다는 근거는 부족하다.
- Identity embedding은 seed7/13 macro에서 소폭 개선했으나 seed21에서는 악화했다. 합산 macro ΔMSE+0.000134, 세 seed 모두 이긴 task2/8이다. 학습 가능한 identity의 일반적 이점은 지지되지 않는다. Task별 상대 변화의 평균(-0.0503%)과 절대 macro ΔMSE(+0.000134)는 가중 방식이 달라 부호가 다를 수 있다.
- 선형 모델의 macro MSE/MAE가 모든 seed에서 더 낮다. N축 모델의 macro ΔMSE는+0.006730 ±0.000674이고, task별 상대 MSE 변화 평균은+1.8649%다. N축은3-seed task 평균상 ETTm1의96/720에서만 선형 모델보다 MSE가 낮고, ETTm1/720만 세 seed 모두 이긴다. MAE는24개 모든 seed-task 쌍에서 선형 모델이 낮다.
- 결론: multi-seed는 작은 단일-seed 개선을 구조의 효과로 오해하지 않도록 하는 데 유용했다. 현재 짧은 고정 프로토콜에서 attention 축 변경과 identity 추가의 안정적 우위는 확인하지 못했다. 추가 seed만 늘리기보다 population frontend 또는 IAND 억제 효과를 각각 분리한 대조 실험이 다음 설계 검토 대상이다. 이 후속 실험은 not run.
- n=3이고 seed7을 보고 반복 검증을 시작했다. 동일한 데이터 분할에서 초기화/학습 순서 변동을 본 것이며 데이터셋 일반화, 수렴, 통계적 유의성 검증은 아니다. Seed를 짝지어도 파라미터 집합이 다른 SSA 제거 구조의 공통 후속 레이어 초기화까지 동일한 것은 아니다.
- Population coding 없는 SNN, Gaussian 중심/폭 학습, flatten head 대조, 긴 학습/별도 scale 튜닝, streaming 인과성/실제 에너지/성능 benchmark: not run. 이번 결과만으로 population coding 자체나 head의 정확도 효과를 결론내리지 않는다.

### 산출물과 보존

- 종합 `NSMT/simple_test_model_v1/forecasting/results/ett-multiseed-20260911/`: `REPORT.md`, `per_run.csv`(120), `per_task.csv`(40), `paired.csv`(32), `aggregate.json`, `provenance.json`, `checks.json`, `comparison.png/pdf`.
- 새 원본 `results/ett-multiseed-20260911-seed13/`와 `results/ett-multiseed-20260911-seed21/`: 각40개 run JSON과 manifest/completion. Seed7 원본은 `results/ett-quick-20260910/`에 그대로 보존하며 provenance에 각 원본 경로/hash/commit/reuse 여부를 명시했다.
- Checkpoint/원시 stdout/runtime copies는 `NSMT/simple_test_model_v1/forecasting/log/<suite>/`, mutable queue/lock은 `scripts/queues/`에 로컬 보존. 데이터/checkpoint/원시 콘솔 로그를 Git에 추가하거나 삭제하지 않았다.
- 완료 code/log/text results 및 결과 그림은 실험 브랜치에 commit하고 annotated 완료 tag로 보존한다. main 통합/원격 push: not run.

완료 보존 명령:

```bash
git add simple_test_model_v1/forecasting/summarize_multiseed.py simple_test_model_v1/forecasting/check_multiseed_summary.py simple_test_model_v1/forecasting/results/ett-multiseed-20260911 simple_test_model_v1/forecasting/results/ett-multiseed-20260911-seed13 simple_test_model_v1/forecasting/results/ett-multiseed-20260911-seed21 ../docs/PROJECT_LOG.md
git diff --cached --check
git commit -m "experiment: record three-seed ETT replication results"
git tag -a exp/population-spikformer-ett-multiseed-20260911 -m "Completed ETT three-seed replication: 80 new GPU runs plus 40 reused seed-7 results; paired statistics independently verified"
```


---

## 2026-09-11 — Population coding 자체의 효과: 사전 고정 및 neorecall 로그 방식 적용

- 사용자 요청: 앞으로 neorecall_v1 Python과 neorecall_ad_v1/anomaly_detection/log의 저장 관례를 따르고 population coding 자체의 이득을 검증한다.
- 브랜치 `exp/population-coding-ablation`; base `e59ac8ab4646059fbfa94655e7b0000d91f13cff`. 시작 clean. 완료된 multi-seed 실험을 기준으로 새 branch 생성. main 통합/push not run.
- 기존4개 SNN 조건은 모두 population coding을 사용했으므로 그 결과만으로 coding 효과를 분리할 수 없었다. 이번은 Gaussian tuning vs clipped affine scalar repeat × temporal SSA vs no SSA의2×2이다. ETTh1/ETTh2/ETTm1/ETTm2 × H96/720 × seeds7/13/21 = **96개 모두 새 실행**; 기존 결과 재사용 없음.
- `population_code=gaussian`: 기존 fixed Gaussian K16, centers[-3,3], sigma0.4. `repeat`: 동일 범위로 clip한 scalar를 `(z+3)/6`으로[0,1]에 선형 매핑한 뒤K16번 복제. Population 선택성이 없는 대조군이다. 같은 차원/파라미터 수/초기 state_dict/clip 범위/current 범위를 유지한다. 동일 분포나 동일 발화율을 강제하지 않으며 tuning이 만드는 표현 다양성과 발화 변화가 처치의 일부다.
- K를1로 줄이지 않아 head 축소/파라미터 차이와 혼동하지 않는다. 다만 복제된 슬롯이 독립 feature를 가지지 않으므로 동일 nominal capacity가 동일 effective capacity를 의미하지 않는다. 일반적으로 최적화된 raw SNN과의 비교나 Gaussian 자체의 보편적 우위를 입증하는 실험은 아니다. Unclipped raw, learned raw projection/K1, Gaussian 중심/폭 학습: not run.
- 나머지 설정 고정: seq96, patch/stride8, direct, N=time, D64/heads8/depth2/MLP ratio2, two-stage head D′64, tau2/threshold1/scale1, normalizeTrue/biasFalse, identity embedding 없음. Gaussian vs repeat 쌍은 동일 seed/axis에서 모든 초기 state_dict hash가 같아야 한다.
- 데이터/분할/전처리: 기존 ETT-small4 CSV7 features,12/4/4개월 경계(hour8640/11520/14400, minute4배), 이전96점 val/test context, train-only StandardScaler, input-window 정규화. 모든 window, stride1/drop_lastFalse, shuffle generator seed+1000.
- 학습: AdamW lr.001/wd.01, MSE, clip1, ReduceLROnPlateau(valMSE factor.5 patience1), max10epoch/early-stop3, batch128, best valMSE checkpoint 복원 후 val 재현 검사 및 full test. GPU0–3 RTX A6000 병렬, 기존 snn_recall Python3.10.18/torch1.12.0+cu113/SpikingJelly0.0.0.0.14/CuPy13.5.1, FP32 deterministic/TF32 off. 환경 설치/변경 not run.
- `neorecall_v1/forecasting/config.py`의 Config, `utils.py`의 EpochLog/EarlyStopping, `test.py`의 final+result.csv 출력과 실제 AD CSV/logargs.txt를 읽고 adapter `experiment_logging.py`를 구현했다. 새 로그: `<task>/log/<suite>/<dataset>/<YYMMDD>/<date+config>/seed<seed>_<variant>_code<code>/` 아래 `logargs.txt`, `log/best_log_0.csv`, `log/final+result.csv`, TensorBoard `log/train_0`/`val_0`, `model_state/config.pt`와 raw state_dict `best+model.pt`.
- Epoch CSV는 epoch/train_loss/train_mse/train_mae/val_loss/val_mse/val_mae, 소수6자리. Final CSV는 측정한 forecasting loss/MSE/MAE와 parameters/best_epoch/seed만 기록한다. AD detection 지표나 측정하지 않은 energy/ops를 만들어 넣지 않는다. 원본 정밀도/config/provenance는 results JSON으로 유지한다. Text CSV/logargs는 Git, TensorBoard raw events/checkpoints/stdout은 로컬 보존. 미래 관례를 root AGENTS.md에 추가했다. 기존 기록 이동/삭제 없음.
- CPU 및 GPU/CuPy 검증 passed: 두 axis에서 gaussian/repeat/기존 base 모델 초기 state_dict 동일, 기존 Gaussian forward/gradient bitwise 동일, repeat endpoint/clip/slot 동일성, embedding gradient nonzero, N축 유지, forward 사이 state reset. 결과 `results/ett-population-ablation-20260911/check_model_cpu.json`, `check_model_gpu.json`.
- ETTh1/H96 Gaussian/repeat 각각1epoch, train/eval2batch smoke passed. CSV 반올림, TensorBoard train/val loss/mse/mae scalar, logargs/config/checkpoint 경로와 paired 초기 hash 확인. `check_logging.json`; smoke 원본은 `results/population-ablation-smoke-20260911/`. Smoke는 실제 예측 성능 판단에 사용하지 않는다.
- 집계 사전 계획: 각 axis에서 같은 task/seed의 Gaussian−repeat ΔMSE/MAE, task별3-seed 평균/표본 SD, seed마다8개 task macro 후3-seed 평균/SD. 추가로 ΔMSE의 SSA on−off 차이로 coding 효과의 attention 의존성을 살핀다. n=3/짧은 예산으로 유의성이나 수렴을 주장하지 않는다.
- 준비 tag `exp/population-coding-ablation-20260911-snapshot`은 코드/검증 snapshot이다. 이 시점의96개 full training은 **not run**. 실제 실행/완료 결과를 append한다.

검증/실행 명령 (cwd NSMT; 아래 Python 환경 lib를 LD_LIBRARY_PATH에 설정):

```bash
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/check_population_ablation.py
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/check_population_ablation.py --device cuda:0 --backend cupy
# Smoke commands are preserved verbatim in each smoke result JSON.
bash -n simple_test_model_v1/forecasting/scripts/run_population_ablation.sh
bash simple_test_model_v1/forecasting/scripts/run_population_ablation.sh
```

Wrapper의 정확한 launcher 인수:

```bash
bash simple_test_model_v1/forecasting/scripts/run_parallel.sh --suite ett-population-ablation-20260911 --gpus 0 1 2 3 --variants temporal no_attention --population-codes gaussian repeat --seeds 7 13 21 --epochs 10 --patience 3 --batch-size 128
```


---

## 2026-09-11 — Population coding 삭제 대조 실험 완료: 96 new runs

### 실행 및 보존 기준

- 앞선 계획의96개 full ETT 실험을 모두 새로 학습/평가했다. Gaussian/repeat × N축 SSA/no SSA × ETTh1/ETTh2/ETTm1/ETTm2 × H96/720 × seed7/13/21. 완료96, 실패0, 기존 결과 재사용0.
- 브랜치 `exp/population-coding-ablation`; base `e59ac8ab4646059fbfa94655e7b0000d91f13cff`. 모든 full run의 훈련 commit은 준비 snapshot commit `664c9ec247c2f7f55721ec57ac8239e3e6126128`. 완료 기록/분석 코드/결과를 포함하는 commit은 annotated tag `exp/population-coding-ablation-20260911`로 식별한다. `...-snapshot` 태그와 구분한다.
- 실행 중 모델/훈련기/logger/launcher를 수정하지 않았다. 집계/검증/그림 코드만 추가했다. Seed별 초기화와 shuffle 외 설정 변경 및 test 기반 튜닝 없음. main 통합/push not run.
- 훈련 설정은 계획 그대로: 입력96, H96/720, patch/stride8, K16/D64/heads8/depth2, direct, N=time, fixed Gaussian 또는 clipped affine repeat, identity 없음, two-stage head64, tau2/threshold1/scale1, normTrue/biasFalse. AdamW lr.001/wd.01, MSE/clip1, ReduceLROnPlateau(valMSE factor.5 patience1), batch128, 최대10epoch/early-stop3, 최소 validation MSE checkpoint 선택.
- 데이터는 ETT-small4 CSV7변수,12/4/4개월 분할 및 train-only StandardScaler, val/test 앞96점 context, 모든 window/전체 target 평가. 앞선 계획의 경계/전처리 동일. 전체96개 code/config/environment 일치 및48개 coding pair의 data/model config/초기 state hash/파라미터 수 일치를 검사했다.
- 환경은 기존 Python3.10.18, torch1.12.0+cu113/CUDA11.3, SpikingJelly0.0.0.0.14, CuPy13.5.1, numpy1.26.4/pandas2.3.1/sklearn1.7.1, RTX A6000 GPU0–3, FP32/deterministic/TF32 off. 패키지 설치/환경 변경 not run.
- GPU 병렬 wall time:2026-09-11 14:20:23.217–14:46:03.950 KST,1540.733초(**25분41초**). GPU0/1/2/3 작업 수27/17/24/28, 각GPU 동시에1개 process. 최대 torch allocated memory4.092GiB/run. CLI/시작종료/PID/GPU는 manifest/completion 및 개별 result JSON에 보존.
- 파라미터 수는 coding pair에서 정확히 같다: SSA사용 H96/720 =207232/686464, SSA제거 =173440/652672. SSA on/off 자체는 서로 다른 파라미터 수다.

### 기록 방식 적용과 검증

- 사용자 지정 neorecall 방식으로96개 모두 기록했다. `<task>/log/ett-population-ablation-20260911/<dataset>/260911/<date+config>/seed<seed>_<variant>_code<code>/` 아래 `logargs.txt`, `log/best_log_0.csv`, `log/final+result.csv`, `log/train_0`, `log/val_0`, `model_state/config.pt`, raw state_dict `model_state/best+model.pt`.
- Epoch CSV는 MSE loss에 맞춘 train/val loss/MSE/MAE,6자리 출력이다. 실제 값은 full-precision JSON에도 보존했다. Final CSV에는 측정한 loss/MSE/MAE,parameters,best_epoch,seed를 기록한다. AD 지표/측정하지 않은 energy/ops를 대신 채우지 않는다. Epoch 번호는 기존 standalone 훈련기의1-based 번호를 유지한다.
- 독립 NumPy [seed,task,axis,code,metric] tensor로32개 task mean/SD,32개 paired mean/SD와 상대변화/승패,4개 macro와 interaction을 재계산하여 통과했다. n=3 표본 SD(ddof1), macro는 각 seed 안에서8개 task를 먼저 평균한다.
- 96개 전체에서 CSV↔history/최종JSON(반올림 허용치5.01e-7), TensorBoard train/val loss/MSE/MAE 값/step 수, config.pt 설정, raw checkpoint key/shape/finite weights, provenance hash를 검사했다.48개 pair의 초기 state hash/파라미터 수 동일. 상태 passed (`check_summary.json`).
- 모델/훈련기 source hash는96개 모두 동일하고 현재 소스와 일치한다. Gaussian48개를 과거 해당 seed/axis/task 결과와 사후 대조했을 때 test MSE 차이는 모두 정확히0이었다. 로그 방식 변경이 보고 지표를 바꾸지 않았다는 재현 확인이며, 과거 결과를 현재 집계에 재사용한 것은 아니다.
- 입력 transform 그림과 paired 변화 그림의 PNG를 시각적으로 확인했다. Error bar는 seed 간 표본 SD이며 confidence interval이 아니다. 첫 test batch embedding은 모든 run에서 발화했고, test MSE/MAE가 window-mean baseline과 완전히 같은 run은0개였다. 전체 split 발화율/에너지 검증은 아니다.

실제 명령 (cwd NSMT):

```bash
bash simple_test_model_v1/forecasting/scripts/run_population_ablation.sh
/home/yschoi/.conda/envs/snn_recall/bin/python -m py_compile simple_test_model_v1/forecasting/summarize_population_ablation.py simple_test_model_v1/forecasting/check_population_summary.py
/home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/plot_population_control.py
/home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/summarize_population_ablation.py --allow-partial > simple_test_model_v1/forecasting/log/ett-population-ablation-20260911/summary.stdout
/home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/summarize_population_ablation.py --plot > simple_test_model_v1/forecasting/log/ett-population-ablation-20260911/summary.stdout
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/check_population_summary.py > simple_test_model_v1/forecasting/results/ett-population-ablation-20260911/check_summary.json
# Report wording/relative-percentage column was added after the audit; statistics were unchanged.
/home/yschoi/.conda/envs/snn_recall/bin/python simple_test_model_v1/forecasting/summarize_population_ablation.py > simple_test_model_v1/forecasting/log/ett-population-ablation-20260911/summary.stdout
```

### 지표

각 seed에서8개 task를 같은 가중치로 평균한 뒤,3개 seed의 평균±표본 SD다. Test metric은 train-standardized scale의 전체 window/horizon/channel 평균이다.

| Variant | Macro MSE mean ± SD | Macro MAE mean ± SD |
|---|---:|---:|
| temporal_gaussian | 0.380714 ± 0.000697 | 0.402519 ± 0.000602 |
| temporal_repeat | 0.395050 ± 0.001073 | 0.408129 ± 0.000651 |
| no_attention_gaussian | 0.381705 ± 0.001340 | 0.402975 ± 0.000896 |
| no_attention_repeat | 0.398110 ± 0.001580 | 0.409543 ± 0.001031 |

Coding 효과는 같은 axis/task/seed의 Gaussian−repeat다. 상대 변화는 task마다100×(Gaussian/repeat−1)을 계산한 후 평균한다 (macro MSE 비율과 다름).

| SSA | Metric | Macro Δ mean ± SD | Mean task-relative change (%) ± SD | Seed-task wins | All-3-seed task wins |
|---|---|---:|---:|---:|---:|
| temporal | mse | -0.014335 ± 0.001226 | -3.0518 ± 0.2856 | 19/24 | 6/8 |
| temporal | mae | -0.005610 ± 0.000580 | -1.1539 ± 0.1428 | 19/24 | 6/8 |
| no_attention | mse | -0.016405 ± 0.001385 | -3.4264 ± 0.3295 | 19/24 | 6/8 |
| no_attention | mae | -0.006568 ± 0.000613 | -1.3641 ± 0.1291 | 18/24 | 5/8 |

### 해석과 한계

- **이 고정된 SNN과 짧은 예산에서 Gaussian tuning을 유지하는 이득이 확인됐다.** N축 SSA 사용 시3개 seed 모두 macro MSE가 개선됐고 task별 상대 변화 평균은-3.0518%±0.2856%p, SSA 제거 시-3.4264%±0.3295%p였다. 각 axis에서19/24개 seed-task 승,6/8개 task는 세 seed 모두 개선됐다. 앞선 attention/identity 비교와 달리 coding 삭제 효과는 반복 seed에서 일관된 방향이었다.
- 양쪽 axis 모두 평균상 개선되지 않은 task는 ETTm1/H96와 ETTm2/H720다. SSA사용 기준 ETTh1/H720은 평균 상대 MSE-12.4305%로 가장 큰 개선, ETTh1/H96은-5.0193%였다. ETTm2/H720은+2.3944%로 세 seed 모두 악화했다. 따라서 모든 ETT 조건에 유리하다는 주장은 하지 않는다.
- Coding 효과의 SSA-on minus SSA-off interaction은 ΔMSE+0.002070±0.001679, ΔMAE+0.000958±0.000484다. 평균적으로 coding 이득은 SSA 없이도 나타나고 조금 더 크다. 이 숫자만으로 interaction의 통계적 유의성을 주장하지 않는다.
- Gaussian48개는 모두 early-stop(SSA4–6epoch, noSSA4–9epoch). Repeat48개 중14개는10epoch 상한에 도달(SSA9, noSSA5),6개는epoch10이 best였다(SSA5,noSSA1). **긴 학습 후 같은 격차가 남는지는 미확인**이다. 이번은 같은 최대 예산/early-stop 설정에서의 정확도와 최적화 결과를 함께 본 것이다.
- 이 대조군은 scalar를clip/affine 후K개 슬롯에 복제한다. 차원과 nominal parameter 수/초기화를 맞추지만, 독립 feature diversity와 head gradient 구조까지 같아지는 것은 아니다. 반복 feature는 head gradient도 결합시킨다. 따라서 고정 구조 안의 삭제 대조 결과이며 일반적으로 최적화된 raw SNN 전체에 대한 Gaussian 우위의 증명이 아니다.
- 입력 분포/발화율을 동일하게 강제하지 않았다. Gaussian의 비선형 tuning이 만드는 representation/발화/optimization 차이가 처치의 일부다. 실제 에너지/전체 발화율, 긴 학습, K1 또는 독립적으로 학습한 raw projection, Gaussian 중심/폭 학습, 별도 threshold/scale 튜닝: not run.
- n=3 seed는 같은 데이터 분할의 초기화/순서 변동이다. 독립 데이터셋 반복이나 통계적 유의성/수렴을 입증하지 않는다. 다음 검토 대상으로는 raw K1/learned-projection 대조와 충분한 학습 예산이 적절하나 이번96개 범위에는 추가 실행하지 않았다.

### 산출물 및 완료 보존

- `NSMT/simple_test_model_v1/forecasting/results/ett-population-ablation-20260911/`:96개 원본 result JSON, manifest/completion, `REPORT.md`, `per_run.csv`(96), `per_task.csv`(32), `paired.csv`(32), aggregate/provenance, CPU/GPU/logging/summary checks, `input_transforms.png/pdf`, `comparison.png/pdf`.
- `per_run.csv`는 각 neorecall 형식의 log 절대 경로를 제공한다. Epoch/final CSV 및 logargs.txt는 Git에 보존하고, TensorBoard events/config.pt/checkpoint/stdout/runtime copy는 로컬에 보존한다. Canonical log는 이 파일 하나이며 NSMT/docs/PROJECT_LOG.md의 기존 연결을 유지한다.
- 모델 변경: `forecasting/simple_test_model_v1.py`의 선택적 `population_code=repeat`; 기존 Gaussian 기본 동작 보존. 저장 방식/훈련기/launcher/새 실행기는 준비 commit에, 분석기 `summarize_population_ablation.py`, 독립 검사기 `check_population_summary.py`, 그림 생성기 `plot_population_control.py`는 완료 commit에 보존한다.
- 원시 artifact 삭제, 데이터/checkpoint/event upload, main 통합/원격 push: not run. 완료 annotated tag `exp/population-coding-ablation-20260911`.

완료 보존 명령:

```bash
git add ../docs/PROJECT_LOG.md simple_test_model_v1/forecasting/summarize_population_ablation.py simple_test_model_v1/forecasting/check_population_summary.py simple_test_model_v1/forecasting/plot_population_control.py simple_test_model_v1/forecasting/results/ett-population-ablation-20260911 simple_test_model_v1/forecasting/log/ett-population-ablation-20260911
git diff --cached --check
git commit -m "experiment: record matched population-coding ablation results"
git tag -a exp/population-coding-ablation-20260911 -m "Completed 96 matched ETT coding ablations: three seeds, Gaussian vs repeated scalar, SSA on/off; neorecall logs and independent audits passed"
```

---

## 2026-09-11 — Population-selective membrane memory: 임시 파일 및 개념 검토 snapshot

- 목적/사용자 요청: `NSMT/forecasting/f-LIF_pop_v1.py` 임시 모델 파일을 만들고, 사용자가 제공한 개념 문서를 읽어 이해 내용을 정리하며 학술 선행 연구에 근거해 구체화를 논의한다. 이번 산출물은 설계 논의용 placeholder이며 모델 구현/완료 학습 실험이 아니다.
- 브랜치 `exp/f-lif-pop-v1`; 선택한 base `7990abd8c1fdcd15eec73e355dd38c6556918f51` (`exp/population-coding-ablation`의 완료 기록 commit). 기존 forecasting 구조와 최신 실험 기록을 이어받기 위해 현재 HEAD에서 새 branch를 만들었다. 조회한 로컬 `origin/main`은 `191f366c6b9dd3cbfaeeb28bb49e7800c8ab488e`; fetch/main 통합/push는 not run.
- 시작 시 존재한 untracked `NSMT/docs/population_selective_membrane_memory_snn_concept.md`는 읽기만 하고 원문/추적 상태를 보존했다. SHA256 `01355e2a9cb0766632a52095d88a1676c845d4b4041695aea791b3eabbc01336`. 이 사용자 문서는 snapshot commit에 포함하지 않는다.
- 코드/config 변경: 요청한 파일 하나에 module docstring만 추가했다. Branch B(이질적 LIF population → 과거 막전위 벡터 → 현재 조건부 검색 → 현재 발화 동역학)를 기록하고 reset/storage, relevance, gate, soft/hard selection, output, fractional prior는 미확정으로 명시했다. 실행 가능한 model class/forward/trainer/config 변경은 없다.
- 이해/구체화 방향: population은 입력 Gaussian coding과 별개인 내부 시간 상태 표현이다. logical neuron 안의 K constituent는 과거 시점 선택을 공유한다. 과거 slot은 그 시점까지의 이력을 요약한 내부 상태이며 원시 관측 하나와 동일하지 않다. 실제 관측/순서 있는 patch 시간축, 같은 sequence의 j<t 읽기, retrieval 전 query 생성과 retrieval 후 기록 순서를 지켜야 한다. 직접 막전위 검색은 원래 f-LIF 이산화/이론을 그대로 구현하지 않는다.
- 공개 학술 색인/원문을 이용해 관련성을 검토했다(아래는 문헌 검토이며 이 모델의 검증 결과가 아니다): [Ge et al., ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/80b4df828ee59926a5f2422f1c072d88-Abstract-Conference.html)의 fractional history; [Perez-Nieves et al., Nature Communications 2021/PubMed](https://pubmed.ncbi.nlm.nih.gov/34608134/)의 membrane/synaptic time-constant heterogeneity; [Ramsauer et al., Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)의 content-addressed vector retrieval; [Limbacher, Özdenizci & Legenstein, arXiv 2022](https://arxiv.org/abs/2205.11276)의 Hebbian SNN associative memory; [SpikeTrack, 2026 원문](https://arxiv.org/html/2602.23963v1)의 spike-query memory retrieval; [PSN, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a834ac3dfdb90da54292c2c932c997cc-Abstract-Conference.html)의 masked temporal weighting; [Mamba](https://arxiv.org/abs/2312.00752)의 content-dependent state updates.
- 문헌에 따른 논의 쟁점: retrieval/heterogeneity/SNN attention 각각의 존재를 신규성으로 주장하지 않는다. 이질적 막전위 population을 logical-neuron별 historical slot 및 retrieval 표현으로 사용하는 결합의 효과가 검증 대상이다. Ge의 finite-mixture 비동등성은 고정된 유한 선형 LIF mode 혼합의 정확한 전체 시간 응답에 대한 범위이며 일반적인 finite nonlinear model 불가능성으로 확대하지 않는다.
- 수식상 추가 해석(본 검토의 추론): Branch B의 `p_alpha(d)=(d+1)^alpha-d^alpha`에서 alpha=1은 균일 시간 prior다. 정규화한 직접 state retrieval은 alpha=1만으로 ordinary LIF로 환원되지 않으며 retrieval strength=0 대조가 필요하다. 초기 논의안은 fixed heterogeneous time constants, population-shared soft retrieval, 별도 memory-use gate, K spike output, post-reset state 저장을 출발점으로 삼고 fractional prior와 hard selection의 추가 효과를 분리하는 것이다. 이 선택들을 코드로 확정하지 않았다.
- 제안 실험(모두 not run): current context로 정답 lag가 바뀌는 합성 recall/반복 regime 과제; 동일 물리 뉴런 수의 homogeneous/heterogeneous × retrieval on/off; fixed/random/recent retrieval 및 input/spike-state memory 대조. 예측 오차 외 정답 slot 접근, memory intervention, population redundancy를 관찰한다. 제안과 실행 결과를 구분한다.
- 데이터/분할/전처리/seed/학습 hyperparameter/metrics: 해당 없음; dataset loading, 학습, forecasting 평가, forward/backward, GPU, stability/efficiency benchmark 모두 not run. Checkpoint/원시 log/결과 지표 산출 없음. 산출물은 위 placeholder와 canonical log의 이 append entry다.
- 환경/검사: Bash/Linux, 기존 `/home/yschoi/.conda/envs/snn_recall/bin/python` 3.10.18 사용. 초기 기본 `python`은 2.7.18이라 pathlib import 검사 실패; 패키지 설치 없이 기존 Python 3.10.18로 재실행하여 AST parse/in-memory compile 통과. 의미 없는 학습/unit test는 추가하지 않았다. `git diff --check`와 staged whitespace 검사를 수행한다.
- 결론: 아이디어 이해와 최소 설계 논의 준비를 보존하는 snapshot이다. 구현/효과/학습 완료를 주장하지 않는다. Code/log snapshot commit은 annotated tag `exp/f-lif-pop-v1-20260911-snapshot`으로 식별한다.

실제 생성/검사/보존 명령 (cwd NSMT; 파일 및 이 기록은 apply_patch로 작성):

```bash
git switch -c exp/f-lif-pop-v1 7990abd8c1fdcd15eec73e355dd38c6556918f51
/home/yschoi/.conda/envs/snn_recall/bin/python --version
/home/yschoi/.conda/envs/snn_recall/bin/python - <<'PY'
import ast
import hashlib
from pathlib import Path
p = Path('forecasting/f-LIF_pop_v1.py')
s = p.read_text()
ast.parse(s)
compile(s, str(p), 'exec')
print('Placeholder syntax: passed')
concept = Path('docs/population_selective_membrane_memory_snn_concept.md')
print('Concept SHA256:', hashlib.sha256(concept.read_bytes()).hexdigest())
PY
git diff --check
git add forecasting/f-LIF_pop_v1.py ../docs/PROJECT_LOG.md
git diff --cached --check
git commit -m "experiment: scaffold population membrane memory design snapshot"
git tag -a exp/f-lif-pop-v1-20260911-snapshot -m "Design snapshot only: temporary model placeholder and literature discussion; no implementation or training run"
```

## 2026-09-11 — Memory 사용 의미 확정: 증거 보강 (design snapshot 2)

- 사용자 선택: 검색한 기억은 '증거보강'에 사용한다. 현재 충전 막전위에 검색 기억을 가산하고 그 결과로 발화/reset하는 의미를 `NSMT/forecasting/f-LIF_pop_v1.py`의 설계 메모에 반영했다. 후보식 `v_t = u_bar_t + gamma * g_t * memory_t`에서 현재 상태 계수는 1이며 기억 기여가 0이면 기존 population update가 된다. Gate의 구체식/강도/기억의 부호 처리/reset과 저장 상태는 여전히 미확정이다.
- 기존 브랜치 `exp/f-lif-pop-v1`에서 계속 진행; 실험 base `7990abd8c1fdcd15eec73e355dd38c6556918f51`, 직전 snapshot `386539e`. 이번 기록은 이전 항목을 수정하지 않고 사용자 결정을 추가한다. 사용자 원본 concept 문서의 내용과 untracked 상태는 보존한다.
- 코드/config: placeholder docstring만 변경; executable model/학습 설정 추가 없음. 데이터/분할/전처리/seed/hyperparameter/metrics: 해당 없음. 모델 구현/학습/forward-backward/안정성 및 성능 검증: not run. 새 artifact는 placeholder와 이 canonical log 기록뿐이다.
- 환경/검사: 기존 Python 3.10.18의 in-memory compile 및 Git whitespace 검사. 명령(cwd NSMT): `/home/yschoi/.conda/envs/snn_recall/bin/python -c 'from pathlib import Path; p = Path("forecasting/f-LIF_pop_v1.py"); compile(p.read_text(), str(p), "exec"); print("Placeholder syntax: passed")'`; `git diff --check`; `git add forecasting/f-LIF_pop_v1.py ../docs/PROJECT_LOG.md`; `git diff --cached --check`.
- 보존: `git commit -m "experiment: record additive memory evidence decision"`; `git tag -a exp/f-lif-pop-v1-20260911-snapshot-2 -m "Design snapshot only: user selected additive memory evidence; no model implementation or training"`. 해당 annotated tag가 이번 commit을 식별한다. 결론은 기억 사용 의미의 확정이며 완료 학습 결과가 아니다. main 통합/push: not run.


---

## 2026-09-14 — f-LIF population 1차 forecasting: 구현/검증 및 실행 전 snapshot

- 목적/사용자 승인: 앞서 추천한 얕은 patch SNN으로 1차 forecasting 실험을 수행한다. 기존 repository 코드를 살펴보고 사용자가 읽기 쉬운 model_v1/forecasting의 코드 스타일 및 기존 로그 저장 관례를 따른다. 사용자가 선택한 memory 의미는 증거 보강이다.
- 기존 `exp/f-lif-pop-v1`에서 계속 진행한다. 실험 base `7990abd8c1fdcd15eec73e355dd38c6556918f51`, 이번 구현 직전 commit `46ec681c`. main 통합/push는 not run. 원래 untracked concept 문서는 내용/추적 상태를 보존한다.
- 코드 조사: repository 전역 소스/config/script 314개,98,854줄,고유 내용202개를 읽어 AST 정의/import/관례와 중복을 조사했다. `NSMT/f_lif_pop_v1/forecasting/results/source_review.json`에 path/hash/구조를 보존한다. 이는 모든 줄에 대한 수동 의미 검증이 아니다. model_v1/forecasting 및 neorecall_v1의 config/model/ours/layers/train/test/utils, ETT loader, 기존 실험 logger/launcher와 AD의 실제 CSV/logargs를 상세히 참고했다. 전역 조사에서 기존 `anomaly_detection/light_trainer.py:270` 구문 오류를 관찰했으며 본 실험 의존성이 아니므로 수정하지 않았다.
- 파일 구조: `NSMT/f_lif_pop_v1/forecasting/{config.py,model.py,ours.py,layers.py,train.py,test.py,utils.py,data_provider/,scripts/,results/,log/}`. `Config`, `LOAD_MODEL`, `myModel`, `Embedding`, `train_one_epoch`, `val_one_epoch`, `test`, `EpochLog`, `EarlyStopping` 이름/역할과 명시적 tensor reshape 및 주석을 따른다. utils의 EpochLog/EarlyStopping는 neorecall 원문에서 필요한 import만 추출했다(동점은 strict minimum 선택, np.inf 사용). 원래 요청 파일 `NSMT/forecasting/f-LIF_pop_v1.py`는 실제 클래스의 import entry로 유지한다.
- 확정된 최소 설계: [B,336,C] → 비중첩 patch8 → [42,BC,8] → Linear(8,32,bias=True)*2 → PopulationLIF → [42,BC,32,4] spikes → Linear(128,32) → flatten(42*32) → Linear(1344,96). 1개 population layer, 별도 Gaussian coding/temporal attention/replay/보조 loss 없음. 시간축은 실제 patch이며 독립 window마다 상태/기억을 초기화한다. 입력 크기 정보 및 patch 인과성을 보존하기 위해 train-only scaling 외 window normalization/BN/LN을 사용하지 않는다. Bias와 고정 current scale2를 사용한다.
- Population: K4/tau[2,4,8,16], beta=exp(-1/tau) 고정. homogeneous는 동일한 평균 beta를4회 반복. 같은 logical neuron의 입력 공유. 공유 K×K Q/K identity 초기화, 정규화 cosine score/temperature.25 → j<t softmax, value=post-reset 막전위 원형. Gate=공유 Linear(K,1)의 sigmoid, 초기 weight/bias0(값.5). `v=u_bar+.05*gate*memory`를 발화 전 가산. Threshold1, subtractive reset `u=v-s.detach()`, sigmoid surrogate alpha4, memory full BPTT. 첫 step memory0. `max(beta)+gamma<1`의 보수적 크기 조건을 강제하며 full gradient stability/robustness theorem 주장은 하지 않는다. 시간 prior/top-k는 not run.
- 본 실험24개: ETTh1/ETTh2 × homogeneous/heterogeneous × retrieval on/off × seed7/13/21. 보조8개: 같은 data/variant의 last head(`Linear(128,32) → 마지막 patch → Linear(32,96)`) seed7만. 총32개 새 run, 과거 결과 재사용0. Head 차이와 seed 수를 구분해 집계한다. 같은 head/seed에서4조건의 초기 trainable parameter/hash 및 nominal parameter 수를 맞춘다. Retrieval off Q/K/gate는 미사용 파라미터이므로 effective capacity/실제 계산량까지 동일한 대조는 아니다.
- 데이터: 기존 로컬 ETT-small ETTh1/ETTh2 CSV7변수. Train[0,8640), val target[8640,11520), test target[11520,14400), val/test 앞336점 context. StandardScaler는 train8640행만 fit. Stride1 모든 window, train8209/val2785/test2785, drop_lastFalse. DataLoader train shuffle generator seed+1000, num_workers0, test/val 순서 고정. CSV hash/scaler/full source provenance는 각 result JSON.
- 학습: AdamW lr.001/wd.01, MSE, clip1, batch128, 최대10epoch/early-stop3, ReduceLROnPlateau(valMSE factor.5/patience1). 최소 validation MSE의 raw state_dict를 복원하고 val 재현 검사 후 전체 test MSE/MAE를 train-standardized 단위로 계산. CSV와 TensorBoard epoch는 기존처럼0-based. Test 기반 튜닝 없음.
- 진단: 전체 test의 persistence/window-mean baseline. Retrieval checkpoint는 off/uniform/recent 개입을 각각 전체 test에 수행하며 재학습 결과와 구분한다. 첫 test8window의 집단별 발화율, membrane diversity/max, gate, evidence크기, lag분포/entropy를 저장한다. ETT에서 정답 기억 위치는 알려져 있지 않으므로 retrieval correctness/인과적 입력 중요도를 주장하지 않는다. 합성 recall 학습/장기학습/다른 dataset/energy benchmark는 not run.
- 환경: 기존 snn_recall Python3.10.18, torch1.12.0+cu113, SpikingJelly 및 기존 NumPy/pandas/sklearn/TensorBoard. 설치 변경 없음. RTX A6000 GPU0–3 사용 전 모두 idle(12MiB/0%). FP32/deterministic/TF32 off, CPU thread2, CUDA_VISIBLE_DEVICES로 각GPU1 process.
- 검증: `check_model.py` CPU 및 cuda:0 모두 통과: 초기parameter 일치/평균beta 일치/이진spike/j<t mask/미래patch에 앞선state 불변/window reset/검색off 및 gamma0 baseline/동질집단동일성/이질집단다양성/독립 가산식 재구성/유한하고 nonzero Q,K,gate gradient/두head shape. Python compileall, bash -n, git diff --check 통과. Smoke `smoke-20260914`는 ETTh1 hetero+retrieval/flatten seed7,1epoch train2batch/eval2batch;2.9초 완료. CSV 반올림/TensorBoard train-val loss-mse-mae step0/config.pt/finite checkpoint/logargs/val 복원을 검사했다. `results/preflight.json`. Smoke는 예측 성능 판단에 사용하지 않는다.
- 저장: `log/<suite>/<dataset>/<YYMMDD>/<date+config>/seed<seed>_<head>_<variant>/{logargs.txt,log/best_log_0.csv,log/final+result.csv,log/train_0,log/val_0,model_state/config.pt,model_state/best+model.pt}`. CSV6자리, full precision은 history/provenance/result JSON; horizon CSV/forecast 예시와 diagnostics 추가. Checkpoint/raw events/stdout은 로컬, text결과/code/config/documentation는 Git. queue/lock은 scripts/queues/. 기존 artifact 삭제 없음.
- 이 snapshot에서32개 full training은 **not run**. 실행/완료는 아래에 append한다. 준비 commit은 annotated tag `exp/f-lif-pop-v1-20260914-snapshot`으로 식별한다.

검사/실행 명령(cwd NSMT; 기존 환경 lib를 LD_LIBRARY_PATH에 설정):

```bash
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python -m compileall -q f_lif_pop_v1/forecasting
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v1/forecasting/check_model.py
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v1/forecasting/check_model.py --device cuda:0
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v1/forecasting/train.py --suite smoke-20260914 --epoch 1 --max_train_batches 2 --max_eval_batches 2
bash -n f_lif_pop_v1/forecasting/scripts/run_ett.sh
git add forecasting/f-LIF_pop_v1.py f_lif_pop_v1/forecasting ../docs/PROJECT_LOG.md
git diff --cached --check
git commit -m "experiment: implement population membrane forecasting with reference-style training"
git tag -a exp/f-lif-pop-v1-20260914-snapshot -m "Implementation and smoke snapshot; full 32-run forecasting experiment not yet run"
bash f_lif_pop_v1/forecasting/scripts/run_ett.sh --suite ett-first-20260914
```


---

## 2026-09-14 — f-LIF population 1차 forecasting 완료: 32 runs

### 실행 및 재현 식별자

- 사용자 승인된 1차 실험을 모두 수행했다. 본 실험24개(ETTh1/ETTh2 × homogeneous/heterogeneous × retrieval on/off × seed7/13/21), 마지막 상태 head 보조8개(seed7). 완료32,실패0,과거 결과 재사용0.
- 브랜치 `exp/f-lif-pop-v1`; base `7990abd8c1fdcd15eec73e355dd38c6556918f51`. 모든 run의 학습 commit은 `e2f1cdbb207b4aefa7cf7da675535e0ae79236b8` (준비 tag `exp/f-lif-pop-v1-20260914-snapshot`). 이번 code/log/result 완료 commit은 annotated tag `exp/f-lif-pop-v1-20260914`로 식별한다. main 통합/원격 push: not run.
- 실행 중 모델/config/trainer/logger/data source를 수정하지 않았다. 실행9개 source hash가32개 모두 일치하며 검사 시 소스와 일치한다. 집계/검증 코드만 별도로 작성했다. 설계와 학습 조건은 위 사전 항목 그대로이며 test 기반 튜닝/추가 학습을 하지 않았다.
- 실제 GPU 병렬시간: 2026-09-14 19:52:13.881–19:56:34.827 KST (UTC10:52–10:56),260.946초(4분21초). GPU0/1/2/3별9/6/11/6개 run; GPU당 동시에1개 process, run당 최대 torch allocated1.07071GiB. 시각/PID/GPU/명령은 manifest/completion과 개별 JSON에 있다.
- 환경: Python3.10.18, torch1.12.0+cu113/CUDA11.3, SpikingJelly0.0.0.0.14, NumPy1.26.4, pandas2.3.1, sklearn1.7.1, TensorBoard2.19.0, RTX A6000×4. 패키지 설치/변경 없음. FP32/deterministic/TF32 off. 주 모델133573 parameters, last head7621; 동일 head/seed의4조건(그리고 두 dataset)의 초기 trainable parameter hash와 개수가 모두 동일했다. Beta buffer는 의도대로 달라지지만 평균 beta를 맞췄다.
- 데이터/분할: ETT-hour7변수, train[0,8640),val target[8640,11520),test target[11520,14400),336점 context,train-only StandardScaler. Train8209/val2785/test2785 window. 모든 test2785×96×7=1871520 element를 평가했다. Dataset CSV hash/scale mean,std/정확한 명령은 개별 JSON에 보존한다.
- 입력336/출력96/patch8/D32/K4/head32, fixed tau2/4/8/16 또는 평균beta의 homogeneous, additive gamma.05, gate sigmoid, cosine QK temperature.25, post-reset state, subtractive reset/full history BPTT. AdamW .001/wd.01, MSE/clip1, max10epoch/patience3/batch128, ReduceLROnPlateau(valMSE). 전체32개 중16개,주24개 중13개가10epoch 상한에 도달했다. 주 실험4개는epoch9(0-based,마지막)가 best였다. 수렴을 주장하지 않는다.

### 본 실험 macro 지표

두 데이터셋을 각 seed 안에서 먼저 평균한 뒤3seed의 평균±표본SD(ddof1)를 계산한다. 같은 표준화 단위의 window/horizon/channel 전체 MSE/MAE다. SD는 신뢰구간이 아니다.

| Variant | MSE mean ± SD | MAE mean ± SD |
|---|---:|---:|
| heterogeneous_no_memory | 0.385644 ± 0.021461 | 0.423791 ± 0.016197 |
| heterogeneous_retrieval | 0.382667 ± 0.018490 | 0.420998 ± 0.014005 |
| homogeneous_no_memory | 0.407324 ± 0.017954 | 0.438958 ± 0.011965 |
| homogeneous_retrieval | 0.397969 ± 0.005556 | 0.432702 ± 0.003111 |

검색 효과는 같은 dataset/head/seed의 retrieval on−off다. 음수는 검색 사용 모델의 오차가 낮다는 뜻이다.

| Population | Macro ΔMSE mean ± SD | Macro ΔMAE mean ± SD |
|---|---:|---:|
| homogeneous | -0.009355 ± 0.016715 | -0.006257 ± 0.011577 |
| heterogeneous | -0.002977 ± 0.003053 | -0.002792 ± 0.002239 |

### 각 run 결과 (epoch는0-based)

다음32개 각각의 full-precision 결과와 log 경로는 `NSMT/f_lif_pop_v1/forecasting/results/ett-first-20260914/per_run.csv`와 `<run_id>.json`에 있다. `flatten`은 본 실험, `last`는 seed7만의 보조 실험이다.

| Data | Head | Variant | Seed | Test MSE | Test MAE | Best epoch | Epochs run |
|---|---|---|---:|---:|---:|---:|---:|
| ETTh1 | flatten | heterogeneous_no_memory | 7 | 0.421299 | 0.441665 | 7 | 10 |
| ETTh1 | flatten | heterogeneous_no_memory | 13 | 0.432493 | 0.448804 | 6 | 10 |
| ETTh1 | flatten | heterogeneous_no_memory | 21 | 0.418967 | 0.438432 | 6 | 10 |
| ETTh1 | flatten | heterogeneous_retrieval | 7 | 0.422917 | 0.441259 | 7 | 10 |
| ETTh1 | flatten | heterogeneous_retrieval | 13 | 0.436079 | 0.450691 | 6 | 10 |
| ETTh1 | flatten | heterogeneous_retrieval | 21 | 0.419393 | 0.438556 | 6 | 10 |
| ETTh1 | flatten | homogeneous_no_memory | 7 | 0.455445 | 0.462070 | 8 | 10 |
| ETTh1 | flatten | homogeneous_no_memory | 13 | 0.457980 | 0.463490 | 9 | 10 |
| ETTh1 | flatten | homogeneous_no_memory | 21 | 0.449223 | 0.457530 | 9 | 10 |
| ETTh1 | flatten | homogeneous_retrieval | 7 | 0.448148 | 0.457066 | 8 | 10 |
| ETTh1 | flatten | homogeneous_retrieval | 13 | 0.467790 | 0.468826 | 9 | 10 |
| ETTh1 | flatten | homogeneous_retrieval | 21 | 0.464026 | 0.466998 | 9 | 10 |
| ETTh2 | flatten | heterogeneous_no_memory | 7 | 0.333736 | 0.396258 | 1 | 5 |
| ETTh2 | flatten | heterogeneous_no_memory | 13 | 0.387470 | 0.434903 | 1 | 5 |
| ETTh2 | flatten | heterogeneous_no_memory | 21 | 0.319900 | 0.382683 | 1 | 5 |
| ETTh2 | flatten | heterogeneous_retrieval | 7 | 0.329850 | 0.393329 | 1 | 5 |
| ETTh2 | flatten | heterogeneous_retrieval | 13 | 0.370881 | 0.422274 | 1 | 5 |
| ETTh2 | flatten | heterogeneous_retrieval | 21 | 0.316881 | 0.379881 | 1 | 5 |
| ETTh2 | flatten | homogeneous_no_memory | 7 | 0.349093 | 0.409734 | 1 | 5 |
| ETTh2 | flatten | homogeneous_no_memory | 13 | 0.396547 | 0.440821 | 1 | 5 |
| ETTh2 | flatten | homogeneous_no_memory | 21 | 0.335653 | 0.400106 | 1 | 5 |
| ETTh2 | flatten | homogeneous_retrieval | 7 | 0.335072 | 0.401157 | 1 | 5 |
| ETTh2 | flatten | homogeneous_retrieval | 13 | 0.335978 | 0.400370 | 7 | 10 |
| ETTh2 | flatten | homogeneous_retrieval | 21 | 0.336799 | 0.401792 | 1 | 5 |
| ETTh1 | last | heterogeneous_no_memory | 7 | 0.730584 | 0.601024 | 6 | 10 |
| ETTh1 | last | heterogeneous_retrieval | 7 | 0.745731 | 0.603236 | 3 | 7 |
| ETTh1 | last | homogeneous_no_memory | 7 | 0.846589 | 0.657536 | 0 | 4 |
| ETTh1 | last | homogeneous_retrieval | 7 | 0.807533 | 0.636966 | 1 | 5 |
| ETTh2 | last | heterogeneous_no_memory | 7 | 0.500898 | 0.506874 | 4 | 8 |
| ETTh2 | last | heterogeneous_retrieval | 7 | 0.497146 | 0.504376 | 4 | 8 |
| ETTh2 | last | homogeneous_no_memory | 7 | 0.565604 | 0.538102 | 7 | 10 |
| ETTh2 | last | homogeneous_retrieval | 7 | 0.564842 | 0.536918 | 7 | 10 |

### 해석과 제한

- **검색을 끈 대조에서 시간상수 이질성의 이득은 두 데이터셋의3seed 모두 확인됐다.** Homogeneous no-memory macro MSE .407324 → heterogeneous no-memory .385644. 다만 동일 effective capacity는 아니며 fixed-tau/짧은 예산의 관찰이다. 검색 사용 조건에서는 ETTh2 평균 MSE가 homogeneous(.335950)보다 heterogeneous(.339204)에서 약간 높으므로 이질성의 보편적 이득으로 확대하지 않는다.
- **이질적 집단에서 검색의 추가 이득은 작고 dataset 의존적이다.** Macro .385644 → .382667, 약0.772% 감소. Paired macro는3seed 모두 개선됐지만 ETTh1에서는 .424253 → .426130(세seed 모두 소폭 악화), ETTh2에서는 .347035 → .339204(세seed 모두 개선)다. 학습·seed변동 대비 작은 효과이며 통계적 유의성은 검정하지 않았다.
- Homogeneous 집단의 검색 효과는 macro 평균 개선이나 seed21에서는 악화했고 ETTh2 seed13 영향이 크다. 이질성에 따른 retrieval 효과 interaction(hetero Δ−homo Δ)의 seed macro 평균±SD는 +.006377±.014336이다. 이번 결과로 population이 검색 효용을 특별히 증폭한다고 주장하지 않는다.
- 같은 heterogeneous-retrieval checkpoint에서 memory를 끄면 test MSE가 평균 ETTh1+.003029,ETTh2+.013447 상승했다. **기억 기여가 예측에 실제로 사용됨**을 보여주지만, 검색 없이 새로 학습한 모델보다 항상 우수함을 뜻하지 않는다. ETTh1에서는 memory 사용 모델 자체가 no-memory 재학습 대조보다 나빴다.
- 같은 checkpoint에서 uniform으로 바꾸면 ETTh1+.000863,ETTh2+.000672; recent로 바꾸면+.001602,+.000188이다. 내용 조건부 가중치의 기여는 이 테스트에서 작다. 별도로 uniform/recent 방식으로 재학습한 공정한 비교는 not run.
- 내부 진단은 첫 test8window×전체변수/뉴런/patch만이다. Heterogeneous retrieval의 평균gate .5012, 정규화entropy .9262, 평균lag10.15patch, evidence/charge 절대평균비 .01789; 넓게 분산된 soft retrieval이었다. Homogeneous membrane diversity는0,heterogeneous retrieval의 평균 constituent std .1844. 모든 run의 진단 sample에서 발화가 있었다. 이 수치로 전체 split 발화율/energy/정답 memory 위치를 주장하지 않는다.
- Last-state head(seed7)의 heterogeneous no-memory macro MSE .615741 → retrieval .621439로 악화했다(ETTh1 .730584→.745731,ETTh2 .500898→.497146). 첫 구조에서 좋은 장기기억이 충분히 형성됐다는 증거는 약하다. 이 head는133573→7621 parameters로 줄고 접근 가능한 정보도 달라지므로 flatten과 정확도만 비교해 원인을 단정하지 않는다.
- ETT 정답 memory slot을 알 수 없고, 본 실험은10epoch의 작은 budget이다. Full model의 선택성/생물학적 동등성/fractional dynamics/energy efficiency를 증명하지 않는다. 오래된 population coding 실험과도 입력96vs336/window norm/Gaussian/depth 등이 달라 직접 정확도 비교 대상이 아니다.
- 다음 후보(이번에는 not run): 긴 학습에서 gap 유지 확인; 알려진 정답 lag를 가진 synthetic recall로 검색 검증; uniform/recent 재학습 대조; pre/post-reset storage, gate strength, learned tau 비교. 먼저 모델이 어떤 기억을 선택해야 하는지 분리 검증하는 것이 타당하다. 이 후속 항목은 별도 승인된 실행 범위가 아니며 지금은 제안이다.

### 검증과 보존

- `check_summary.py`가32개 실험 matrix/중복/누락, 학습commit/source hash동일성, 초기parameter 일치, 최소val checkpoint/복원, 전체test element수, CSV↔full precision history/final, TensorBoard loss/mse/mae와 epoch, horizon평균, config.pt/state_dict/checkpoint hash/finite weight를 검사하여 통과했다.
- 독립 NumPy tensor와 raw JSON lookup으로 per-task mean/SD, dataset-first macro mean/SD, paired ΔMSE/MAE/상대변화,seed macro를 재계산했다. Interaction도 별도 raw lookup으로 재계산해 일치했다. Train scaler는 원본CSV의 첫8640행 평균/std로 독립 검산하고 첫 test target[11520:11616]를 예시 저장값과 정확히 비교했다.
- `Config.load_args`+`LOAD_MODEL(train=False)`로 ETTh1/flatten/heterogeneous retrieval/seed7의 저장 config/checkpoint를 새로 로드했다. 전체 test MSE/MAE 및 off/uniform/recent MSE가 원본 결과와 atol1e-12에서 일치했다. 원래 hyphen filename의 import entry도 검사했다. 기존 결과 파일은 덮어쓰지 않았다. `check_reload.json`.
- 비교 그림 `comparison.png/pdf`는 mean±sampleSD이며 PNG를 직접 확인했다. Code/CSV/checkpoint 재검사 결과는 `check_summary.json`, 사전검사는 `results/preflight.json`. 원본utils에서 복사된 공백은 snapshot commit 전 staged whitespace검사에서 발견하여 정리했다; 동작 변경 없음.
- 결과 디렉터리: `NSMT/f_lif_pop_v1/forecasting/results/ett-first-20260914/`에32개 JSON,manifest/completion,REPORT.md,per_run/per_task/macro/paired/paired_macro_by_seed/interaction/interventions CSV,aggregate,checks,comparison.png/pdf. 같은 task `log/ett-first-20260914/`의 dataset/date/config/seed+variant별 기존 neorecall 형식 CSV/TensorBoard/logargs/model_state를 보존한다.
- Checkpoint/dataset/raw TensorBoard/stdout은 Git에 넣지 않고 로컬 유지한다. Source/config/documentation/CSV/JSON/결과 비교 그림을 실험 branch에 기록한다. 원래 untracked concept 문서는 내용과 상태 그대로 둔다. User artifact 삭제, main 통합, 원격 push: not run.

실제 실행/집계/검증 명령(cwd NSMT):

```bash
bash f_lif_pop_v1/forecasting/scripts/run_ett.sh --suite ett-first-20260914
/home/yschoi/.conda/envs/snn_recall/bin/python -m py_compile f_lif_pop_v1/forecasting/summarize.py f_lif_pop_v1/forecasting/check_summary.py
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v1/forecasting/summarize.py --suite ett-first-20260914
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v1/forecasting/check_summary.py --suite ett-first-20260914
# REPORT에 실제 결과 설명을 추가한 뒤 summarize.py를 재실행했다. 지표는 변하지 않았다.
# Standalone reload는 위 Config.load_args/test(save=False) 경로를 in-process로 검사했다.
git add f_lif_pop_v1/forecasting ../docs/PROJECT_LOG.md
git diff --cached --check
git commit -m "experiment: record first population membrane forecasting results"
git tag -a exp/f-lif-pop-v1-20260914 -m "Completed 32 first-stage forecasting runs; three-seed main and seed-7 last-head controls, reference logs and independent audits"
```

새 세션에서 이어가기: 이 완료 항목 → task README.md → results/ett-first-20260914/REPORT.md를 읽고 `per_run.csv`의 log_path로 config/checkpoint를 찾는다. `test.py --config <log_path>`는 학습 없이 재평가하고 기존 CSV를 덮어쓰지 않는다. 새로운 학습은 반드시 새 suite 이름으로 기존 artifact를 보존한다. 같은 실험을 계속하면 현재 branch를 사용하고, 독립적인 새 실험이면 선택한 기준 commit에서 새 exp branch와 base를 기록한다.

## 2026-09-14 — 1차 shallow patch SNN: H720 확장 사전 기록

- 사용자 요청: prediction length720을 먼저 완료한 뒤 2차 **Spike-TCN**으로 진행한다. 앞서 제안한 장기학습/합성 검색 대조는 이번의 2차 구조 실험과 구별한다.
- 기존 branch `exp/f-lif-pop-v1`에서 이어간다. H96 완료 기준 commit `277d18f548f23496788c319b4076df66c9f18be7`, 원래 experiment base `7990abd8c1fdcd15eec73e355dd38c6556918f51`. Snapshot tag `exp/f-lif-pop-v1-h720-20260914-snapshot`는 학습 완료 태그가 아니다.
- 변경: launcher에 `--pred_len 96|720` 추가, run ID/manifest에 horizon 반영, 집계·검증의 H96 상수를 해당 suite horizon에서 계산하도록 일반화. 모델/뉴런/trainer/data/utils의 9개 실행 source는 H96과 동일하다. 기본96과 기존 결과는 보존한다.
- 조건: ETTh1/ETTh2, seq336, pred720, patch8, D32/K4, flatten head32×3seed(7,13,21)×4조건=24 runs; last head seed7×4조건=8 runs. Homogeneous/heterogeneous × retrieval off/on. AdamW lr.001,wd.01,batch128,clip1, 최대10epoch, ReduceLROnPlateau factor.5/patience1, early stopping3, 최소 validation MSE checkpoint 복원 후 전체 test. 최대10epoch은 공통 탐색 예산이며 수렴을 뜻하지 않는다.
- 데이터: 기존 train[0,8640),val targets[8640,11520),test targets[11520,14400), train-only StandardScaler, stride1, drop_lastFalse. H720 train7585/val2161/test2161 windows, test10,891,440 elements. 기존 CSV hash와 scaler는 각 run에 저장된다.
- 환경: `/home/yschoi/.conda/envs/snn_recall/bin/python`, Python3.10.18, torch1.12.0+cu113, SpikingJelly0.0.0.0.14; RTX A6000 GPU0–3, GPU당1프로세스, CPU threads2, deterministic/TF32 off. 환경과 실행명령 전체는 run JSON에 기록한다.
- 저장: `NSMT/f_lif_pop_v1/forecasting/{results,log}/ett-first-h720-20260914/`, queue/lock은 같은 task `scripts/queues/`. 기존 neorecall CSV/TensorBoard/logargs/model_state 구조를 유지한다. Checkpoint/raw events/stdout은 local, 코드/설정/텍스트 결과는 Git. Untracked concept 문서를 보존한다.
- 사전 명령(cwd NSMT): `python -m py_compile`은 위 conda Python으로 launcher/summarize/check_summary에 실행하여 통과. `env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v1/forecasting/train.py --suite smoke-h720-20260914 --pred_len 720 --epoch 1 --max_train_batches 2 --max_eval_batches 2`로 H720 학습/검증/복원/평가 smoke를 실행한다. 축소 smoke는 성능 비교에 포함하지 않는다.
- 본 실행 명령: `bash f_lif_pop_v1/forecasting/scripts/run_ett.sh --suite ett-first-h720-20260914 --pred_len 720`. 본 학습/최종 검증/지표는 이 사전 항목 시점 not run; 완료 후 append한다. Main 통합/push는 not run.

## 2026-09-14 — 1차 H720 완료 (32 runs)

- 완료 tag `exp/f-lif-pop-v1-h720-20260914` (이 항목을 포함한 commit). Branch `exp/f-lif-pop-v1`, 학습 snapshot commit `a8677f85b350c7cf0fd1defedc86cdc37b70067d`. 데이터/환경/조건은 직전 사전 기록과 동일. H720 smoke 통과, 정식32개 모두 exit0.
- Launcher wall 237.5s; 개별 run 9.4–46.3s; 최대 GPU allocated 1.095GiB. Main24개 중 6개가10epoch 상한, 전체32개 중 30개에서 감소된 LR로 학습했다.
- 아래 MSE/MAE는 test2161windows×720×7 전체, train-standardized. Macro는 데이터셋 평균을 seed별 계산한 다음3seed mean±sample SD. Last head는 seed7만의 탐색 대조.

| Head | Variant | MSE | MAE |
|---|---|---:|---:|
| flatten | heterogeneous_no_memory | 0.679887 ± 0.053368 | 0.580342 ± 0.022406 |
| flatten | heterogeneous_retrieval | 0.711945 ± 0.031368 | 0.592568 ± 0.014067 |
| flatten | homogeneous_no_memory | 0.717480 ± 0.033274 | 0.602915 ± 0.016036 |
| flatten | homogeneous_retrieval | 0.736275 ± 0.041440 | 0.611361 ± 0.018349 |
| last | heterogeneous_no_memory | 0.923991 ± nan | 0.705316 ± nan |
| last | heterogeneous_retrieval | 0.923926 ± nan | 0.706096 ± nan |
| last | homogeneous_no_memory | 0.970504 ± nan | 0.724788 ± nan |
| last | homogeneous_retrieval | 0.969333 ± nan | 0.724777 ± nan |

### 각 실행 (best epoch은 0-based)

| Data | Head | Variant | Seed | MSE | MAE | Best epoch | Epochs |
|---|---|---|---:|---:|---:|---:|---:|
| ETTh1 | flatten | heterogeneous_no_memory | 7 | 0.558825 | 0.545231 | 3 | 7 |
| ETTh1 | flatten | heterogeneous_no_memory | 13 | 0.579237 | 0.558494 | 3 | 7 |
| ETTh1 | flatten | heterogeneous_no_memory | 21 | 0.544616 | 0.538897 | 4 | 8 |
| ETTh1 | flatten | heterogeneous_retrieval | 7 | 0.596952 | 0.563891 | 4 | 8 |
| ETTh1 | flatten | heterogeneous_retrieval | 13 | 0.577854 | 0.555894 | 4 | 8 |
| ETTh1 | flatten | heterogeneous_retrieval | 21 | 0.547373 | 0.540208 | 4 | 8 |
| ETTh1 | flatten | homogeneous_no_memory | 7 | 0.550437 | 0.544110 | 9 | 10 |
| ETTh1 | flatten | homogeneous_no_memory | 13 | 0.582351 | 0.562672 | 8 | 10 |
| ETTh1 | flatten | homogeneous_no_memory | 21 | 0.562495 | 0.556522 | 8 | 10 |
| ETTh1 | flatten | homogeneous_retrieval | 7 | 0.616034 | 0.574336 | 7 | 10 |
| ETTh1 | flatten | homogeneous_retrieval | 13 | 0.591923 | 0.569924 | 8 | 10 |
| ETTh1 | flatten | homogeneous_retrieval | 21 | 0.582213 | 0.565123 | 6 | 10 |
| ETTh2 | flatten | heterogeneous_no_memory | 7 | 0.860832 | 0.646356 | 1 | 5 |
| ETTh2 | flatten | heterogeneous_no_memory | 13 | 0.843887 | 0.622682 | 2 | 6 |
| ETTh2 | flatten | heterogeneous_no_memory | 21 | 0.691927 | 0.570393 | 2 | 6 |
| ETTh2 | flatten | heterogeneous_retrieval | 7 | 0.865846 | 0.646939 | 1 | 5 |
| ETTh2 | flatten | heterogeneous_retrieval | 13 | 0.879502 | 0.633606 | 2 | 6 |
| ETTh2 | flatten | heterogeneous_retrieval | 21 | 0.804143 | 0.614867 | 2 | 6 |
| ETTh2 | flatten | homogeneous_no_memory | 7 | 0.912516 | 0.668110 | 1 | 5 |
| ETTh2 | flatten | homogeneous_no_memory | 13 | 0.776636 | 0.608372 | 2 | 6 |
| ETTh2 | flatten | homogeneous_no_memory | 21 | 0.920445 | 0.677703 | 0 | 4 |
| ETTh2 | flatten | homogeneous_retrieval | 7 | 0.928884 | 0.674865 | 1 | 5 |
| ETTh2 | flatten | homogeneous_retrieval | 13 | 0.790210 | 0.610908 | 2 | 6 |
| ETTh2 | flatten | homogeneous_retrieval | 21 | 0.908384 | 0.673011 | 0 | 4 |
| ETTh1 | last | heterogeneous_no_memory | 7 | 0.776105 | 0.651288 | 8 | 10 |
| ETTh1 | last | heterogeneous_retrieval | 7 | 0.773437 | 0.651113 | 5 | 9 |
| ETTh1 | last | homogeneous_no_memory | 7 | 0.816555 | 0.669777 | 7 | 10 |
| ETTh1 | last | homogeneous_retrieval | 7 | 0.815257 | 0.669294 | 7 | 10 |
| ETTh2 | last | heterogeneous_no_memory | 7 | 1.071878 | 0.759345 | 3 | 7 |
| ETTh2 | last | heterogeneous_retrieval | 7 | 1.074415 | 0.761079 | 3 | 7 |
| ETTh2 | last | homogeneous_no_memory | 7 | 1.124452 | 0.779799 | 3 | 7 |
| ETTh2 | last | homogeneous_retrieval | 7 | 1.123408 | 0.780260 | 3 | 7 |

### 해석, 검증, 보존

- Heterogeneous retrieval 추가의 macro MSE는 .679887→.711945(약4.715% 악화), MAE .580342→.592568. ETTh1 .560893→.574060, ETTh2 .798882→.849830. H96의 작은 개선은 H720에서 재현되지 않았다. 이질성만 추가한 no-memory는 homogeneous .717480보다 낮지만 seed변동/짧은 예산의 제한이 있다.
- 같은 heterogeneous-retrieval checkpoint의 memory를 off로 바꾸면 ETTh1 ΔMSE−.017590, ETTh2−.026809로 개선된다. Uniform 역시−.001159/−.006319; recent는+.010669/+.007987로 악화. 이 H720 결과는 현재 soft content retrieval의 유익성을 뒷받침하지 않는다. Test 개입을 이용해 모델/강도를 재선택하거나 재학습하지 않았다.
- Last head heterogeneous macro .923991→.923926로 거의 차이가 없다. H720 flatten972853 vs last28213 parameters이며, head용량과 정보 접근이 다르다. Main의 모든 조건은 같은 nominal count/초기 trainable parameters; retrieval off의 Q/K/gate는 미사용이다.
- `check_summary.json`:32matrix, best-validation selection/restoration, CSV/history/TensorBoard, 전체 element수/각 horizon평균, 저장config/checkpoint/hash/finite weight, 독립 CSV train scaler 및 첫 test target[11520:12240], 모든 집계/paired deltas 통과. `check_reload.json`: 별도 config/checkpoint 로딩으로 ETTh1/heterogeneous-retrieval/seed7의 전체 test 및 off/uniform/recent MSE/MAE 재현(atol1e-12). PNG 직접 확인. `results/h720-preflight.json`은 기존H96 32개 회귀 검사를 과거파일 덮어쓰기 없이 기록한다.
- Exact analysis commands(cwd NSMT, prefix `env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python`): `f_lif_pop_v1/forecasting/summarize.py --suite ett-first-h720-20260914`; `f_lif_pop_v1/forecasting/check_summary.py --suite ett-first-h720-20260914`; `f_lif_pop_v1/forecasting/check_reload.py --suite ett-first-h720-20260914 --pred_len 720`. 실제 실행은 이 세 함수 summarize/check/reload_check를 같은 Python process에서 순차 호출했으며 reload는 새로운 model/config 객체를 생성했다.
- Artifacts: `NSMT/f_lif_pop_v1/forecasting/results/ett-first-h720-20260914/{REPORT.md,per_run.csv,per_task.csv,macro.csv,paired.csv,paired_macro_by_seed.csv,interaction.csv,interventions.csv,aggregate.json,manifest.json,completion.json,check_summary.json,check_reload.json,comparison.png,comparison.pdf}` 및32 full-precision run JSON. `per_run.csv:log_path`가 각 기존 neorecall 형식 CSV/TensorBoard/logargs/config.pt/best+model.pt를 가리킨다. Smoke는 별도 `smoke-h720-20260914`, 성능표 제외.
- 한계: 최대10/early-stop3 공통 budget, 3seed, 두 ETT-hour dataset뿐. Longer-budget convergence, uniform/recent 재학습, synthetic recall, 에너지측정: not run. 2차는 별도 branch에서 causal population Spike-TCN과 H96/H720을 같은 protocol로 비교한다. 사용자 untracked concept 유지, main 통합/push: not run.

## 2026-09-14 — 2차 causal population Spike-TCN 사전 기록

- 사용자 순서대로 1차 H720을 완료한 후 시작한다. 새 branch `exp/f-lif-pop-tcn-v1`, 선택한 base는 H720 완료 `ed5c8d16fb5e9c56234c58d6a8d870ffe20f4efc` (`exp/f-lif-pop-v1-h720-20260914`). Main에서 다시 분기하지 않은 이유는 검증된 1차 뉴런/학습 framework 및 H96/H720 결과를 직접 기준으로 삼기 위해서다. Snapshot tag `exp/f-lif-pop-tcn-v1-20260914-snapshot`는 전체 학습 완료가 아니다.
- 목적: causal convolution이 국소 패턴을 처리하는 backbone에서도 heterogeneous population/검색 on-off 효과가 유지되는가? 구조 비교는 H96/H720 모두 시행한다. 실험 결과에 따라 gamma/architecture/hyperparameter를 다시 고르지 않는다.
- 코드: `NSMT/f_lif_pop_tcn_v1/forecasting/`의 Config/LOAD_MODEL/myModel/Embedding/TemporalBlock/train_one_epoch/val_one_epoch/EpochLog/EarlyStopping/data_provider/scripts 구조. `f_lif_pop_v1`의 명시된 파일만 복사하고 독립 구현으로 보존한다. 기존 `model_v1/forecasting/ours.py` token conv는 centered padding이므로 그대로 사용하지 않고 left-only CausalConv1d로 작성했다. 원형 전체 repository 구조 검토는 1차의 source_review.json에 기록되어 있다.
- 구조: patch8 Linear8→32×2 및 PopulationLIF0 → 두 residual current block(kernel3,dilation1/2) → 마지막 population spike DK128→head32→flatten42patch→H. `I_l=I_(l-1)+2*Conv_d(flatten_DK(S_(l-1)))`, `S_l=PopulationLIF_l(I_l)`. Conv-only receptive field7patch=56관측치; recurrence/retrieval 및 flatten head는 그보다 긴 전체42patch에 접근한다. Analog current shortcut은 block 사이에 존재하나 최종 head에는 마지막 binary spike만 입력한다. No time/window normalization/dropout/extra simulation repeats.
- 선행 근거: Lv et al., *Efficient and Effective Time-Series Forecasting with Spiking Neural Networks*, https://arxiv.org/html/2402.01533v2 §3.3. 원 논문은 SEW shortcut/시계열 시점별 membrane reset을 사용한다. 여기서는 아이디어에 맞게 persistent chronological population state와 **current residual**로 바꿨으므로 original Spike-TCN의 재현/논문 metric 비교라고 주장하지 않는다.
- PopulationLIF 수식/구현은1차와 AST가 완전히 동일하다. 각 층은 독립 Q/K/gate/history, 모든 층 같은 K4/fixed tau2..16/gamma.05/temperature.25/threshold1, post-reset memory/full BPTT/detach reset/forward-local state. Uniform/recent/off test 개입은 모든 population layer에 동시 적용한다. Top-level diagnostics는 마지막층, `diagnostics.layers`는 embedding/block0/block1 별 값, 모두 첫 test8windows sample이다.
- Matrix: horizon96/720 각각 ETTh1/ETTh2×seeds7/13/21×homogeneous/heterogeneous×retrieval off/on=24, 총48. Flatten만 정식 학습; last head는 shape check만, training not run. 동일 architecture 네 조건 initial trainable parameters/hash/count 일치. H96 nominal158287, H720997567 parameters; 1차보다24714개 많으므로 matched-capacity architecture comparison은 아니다. Retrieval off Q/K/gate는 미사용이다.
- Data/preprocessing: 1차 CSV 및 train-only StandardScaler와 동일; train[0,8640),val[8640,11520),test[11520,14400),context336,stride1,7variables,drop_lastFalse. H96 windows8209/2785/2785 및 test1,871,520 elements; H7207585/2161/2161 및 test10,891,440 elements. 데이터 hash/columns/scalers/split은 run JSON.
- Training: AdamW lr.001/wd.01,batch128,clip1,최대10epochs,ReduceLROnPlateau factor.5/patience1,strict validation-MSE early stopping3,최소 validation checkpoint 복원/재검증 후 전체 test. 1차와 동일 budget; LR 감소 후 회복 epoch가 짧을 수 있는 제한을 유지하며 장기학습은 not run. Py3.10.18/torch1.12.0+cu113/SpikingJelly0.0.0.0.14/RTX A6000 GPU0–3,1process/GPU,CPUthreads2,deterministic/TF32off. Conda env 및 LD_LIBRARY_PATH는1차와 동일.
- 사전 검사: CPU/GPU `check_model.py` 통과—초기 parameter/평균 beta 일치, 모든 층 binary spikes/과거만 검색/미래 patch 교란 불변, cross-window reset, memory off와gamma0 baseline 일치, homogeneous identity, 이질성, additive update 독립 재구성, 각 층 Q/K/gate 및 convolution의 finite nonzero gradients, H96/H720·last head shape, dilation2 impulse support. `source_protocol_check.json`은 PopulationLIF/train_one_epoch/val_one_epoch AST동일, utils 및 두 data source bytes동일, 전체 Python compile 통과를 기록한다. Shell `bash -n` 통과. H720 1epoch/각2batches smoke는 저장/복원/전체 형식 평가까지 통과(성능표 제외); H96 smoke도 같은 조건으로 실행한다.
- 정확한 명령(cwd NSMT, Python prefix `env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python`): `f_lif_pop_tcn_v1/forecasting/check_model.py --device cpu --output f_lif_pop_tcn_v1/forecasting/results/check_model_cpu.json`; 같은 명령 device cuda:0/output check_model_cuda.json. Smoke: `f_lif_pop_tcn_v1/forecasting/train.py --suite smoke-tcn-20260914 --pred_len 720 --epoch 1 --max_train_batches 2 --max_eval_batches 2` 및 pred_len96. Source protocol check는 AST 비교 inline Python; 결과 JSON과 검사 항목을 보존한다.
- 정식 실행: `bash f_lif_pop_tcn_v1/forecasting/scripts/run_stage2.sh` → `run_ett.sh --suite ett-tcn-h96-20260914 --pred_len 96` 완료 후 `run_ett.sh --suite ett-tcn-h720-20260914 --pred_len 720`. 이 사전 기록 시점 정식 학습/지표/완료 audits는 not run; 완료 후 append.
- Artifacts는 해당 task `results/<suite>` 및 `log/<suite>/<data>/<date>/<config>/seed+head+variant`. 기존 best_log_0/final+result CSV, train_0/val_0 events,logargs,model_state config/best checkpoint 형식. Code/config/docs/text summaries versioned; rawevents/checkpoint/stdout local. Canonical log append-only, concept untracked 보존. Main integration/push: not run.

## 2026-09-14 21:27 KST — 2차 실행 중 및 후처리 인계

- 실제 학습은 snapshot `794a29693a97dc1246c97ccf42d37ff6fd0ac585` (`exp/f-lif-pop-tcn-v1-20260914-snapshot`)에서 시작했다. Branch `exp/f-lif-pop-tcn-v1`, 선택 base `ed5c8d16fb5e9c56234c58d6a8d870ffe20f4efc`. 이 항목은 완료 보고가 아니다. H96 24개 중8개 완료/4개 running/12개 pending이었던12:24:59 UTC 상태를 확인했다. H720 24개는 H96 뒤 자동 실행되며 아직 not run. 최종48개 성능/결론은 not available yet.
- H96/H720 두 실제 데이터 smoke 모두 통과했다. 정식 H96 초기 결과에서 각 population layer 발화를 확인했고, 조기 종료 및 최소 validation checkpoint 복원을 확인했다. H96 정식 epoch 시간은 약30–45초로 1차보다 길다. 완료된 일부 seed의 성능으로 설정을 바꾸지 않는다.
- 새 분석 코드 `compare_stages.py`는 horizon별 비교, 실제48쌍 데이터/전처리/하이퍼파라미터/naive baseline 일치, macro/paired 통계, training_selection.csv, 층별 diagnostic CSV와 PNG/PDF를 작성한다. `scripts/finish_stage2.py`가 각24개 suite 완료 후 summarize/check/PROJECT_LOG append, 두 horizon 완료 후 fresh checkpoint reload, 전체 비교를 수행한다. 스크립트 compile 통과; 전체 결과에 대한 실행 검증은 학습 완료 전이므로 not run.
- 후처리 프로세스는 학습과 독립된 detached process로 시작했다. 최초 managed monitor PID1224395는 대기 중임을 정확한 cmdline으로 확인한 뒤 종료하고, PID1226071의 detached monitor로 교체했다. 학습 프로세스는 중단하지 않았다.
- Detached command(cwd NSMT): `env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python /home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_tcn_v1/forecasting/scripts/finish_stage2.py --wait --finalize`. 실제 생성은 Python subprocess.Popen(start_new_session=True, stdin=DEVNULL, stdout/err=task log/stage2-postprocess-20260914.stdout)이며, 명령/PID/시작UTC를 `scripts/queues/postprocess.json`에 기록했다.
- 상태파일(task 아래): `scripts/queues/ett-tcn-h96-20260914.json`, H720 시작 후 `ett-tcn-h720-20260914.json`, `postprocess.json`. 학습 완료는 각 `results/<suite>/completion.json`; 최종 검증은 `check_summary.json`, `check_reload.json`, `results/stage-comparison-20260914/check_comparison.json`. 자동 versioning 성공은 `scripts/queues/finalization.json`, 예외는 `postprocess_failure.json` 및 local monitor stdout에 남긴다.
- 학습 중 실행 source9개와 HEAD를 동결한다. 새 run이 실제 HEAD를 provenance에 기록하므로 중간 commit/branch 전환은 audit 일치에 영향을 준다. 분석 코드/README/이 인계 기록은 완료 commit에 함께 보존하도록 작업 트리에 남긴다. 기존 untracked concept hash `01355e2a9cb0766632a52095d88a1676c845d4b4041695aea791b3eabbc01336`와 상태를 유지했다.
- `--finalize`는 전체 audits 통과 후 예상 branch/HEAD를 검사하고, 명시된 실험 분석 코드·각 suite log/results·canonical log만 `git add`/`git commit --only`한 뒤 annotated `exp/f-lif-pop-tcn-v1-20260914` tag를 만든다. 다른 staged paths는 그대로 둔다. Branch/HEAD가 바뀌었거나 완료 tag가 이미 있으면 자동 commit/tag를 하지 않고 상태파일에 이유를 기록한다. Main 통합/remote push: not run. 현재 완료 tag는 아직 생성되지 않았다.

## 2026-09-14 — 2차 Spike-TCN H96 완료 (24 runs)

- Branch `exp/f-lif-pop-tcn-v1`; base `ed5c8d16fb5e9c56234c58d6a8d870ffe20f4efc`; training snapshot `794a29693a97dc1246c97ccf42d37ff6fd0ac585`. 24개 전부 exit0. 전체48개 완료 tag/commit은 두 horizon 검증 뒤 별도 기록한다.
- 목적/코드/데이터/환경/seed/hyperparameters는 직전 2차 사전 기록과 동일하다. Convolution2개/kernel3/dilation1,2/current residual; population/optimizer/data 코드는 사전 동결했다. ETTh1/ETTh2×seeds7/13/21×네 조건, flatten head만 학습했다.
- Full test elements/run: 1871520; parameters/run: 158287. Launcher wall 1913.8s; run 154.6–472.6s; max GPU allocated 3.198GiB.
- 24개 중 4개가10epoch 상한에 도달했다. 24개가 감소된 LR로 학습했다. 최소 validation MSE epoch는1-based 1–8. 최종 checkpoint 복원 검증 통과.
- 이질적 population에서 retrieval 추가 macro MSE 0.380509→0.381907 (+0.367%). Macro는 데이터셋 평균을 seed별 계산한 뒤3seed mean±sample SD이며 horizon 간 합산하지 않는다.

| Variant | MSE ± SD | MAE ± SD |
|---|---:|---:|
| heterogeneous_no_memory | 0.380509 ± 0.003947 | 0.417491 ± 0.004645 |
| heterogeneous_retrieval | 0.381907 ± 0.008823 | 0.419841 ± 0.007770 |
| homogeneous_no_memory | 0.397856 ± 0.024672 | 0.430200 ± 0.013963 |
| homogeneous_retrieval | 0.387563 ± 0.013150 | 0.425702 ± 0.009946 |

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_no_memory | 0.423997 ± 0.005138 | 0.444196 ± 0.002801 |
| ETTh1 | heterogeneous_retrieval | 0.428593 ± 0.003501 | 0.448515 ± 0.003437 |
| ETTh1 | homogeneous_no_memory | 0.430014 ± 0.006015 | 0.447499 ± 0.004904 |
| ETTh1 | homogeneous_retrieval | 0.438824 ± 0.008541 | 0.455427 ± 0.008467 |
| ETTh2 | heterogeneous_no_memory | 0.337020 ± 0.013031 | 0.390786 ± 0.011963 |
| ETTh2 | heterogeneous_retrieval | 0.335220 ± 0.014525 | 0.391166 ± 0.012821 |
| ETTh2 | homogeneous_no_memory | 0.365699 ± 0.046494 | 0.412902 ± 0.028436 |
| ETTh2 | homogeneous_retrieval | 0.336301 ± 0.029292 | 0.395977 ± 0.023174 |

각 실행: best epoch은0-based.

| Data | Variant | Seed | MSE | MAE | Best epoch | Epochs |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | heterogeneous_no_memory | 7 | 0.418195 | 0.440961 | 1 | 5 |
| ETTh1 | heterogeneous_no_memory | 13 | 0.427970 | 0.445798 | 2 | 6 |
| ETTh1 | heterogeneous_no_memory | 21 | 0.425827 | 0.445828 | 7 | 10 |
| ETTh1 | heterogeneous_retrieval | 7 | 0.431450 | 0.450769 | 4 | 8 |
| ETTh1 | heterogeneous_retrieval | 13 | 0.424688 | 0.444559 | 2 | 6 |
| ETTh1 | heterogeneous_retrieval | 21 | 0.429641 | 0.450216 | 7 | 10 |
| ETTh1 | homogeneous_no_memory | 7 | 0.423725 | 0.441860 | 5 | 9 |
| ETTh1 | homogeneous_no_memory | 13 | 0.435711 | 0.449877 | 4 | 8 |
| ETTh1 | homogeneous_no_memory | 21 | 0.430605 | 0.450760 | 5 | 9 |
| ETTh1 | homogeneous_retrieval | 7 | 0.433544 | 0.450423 | 2 | 6 |
| ETTh1 | homogeneous_retrieval | 13 | 0.434251 | 0.450654 | 7 | 10 |
| ETTh1 | homogeneous_retrieval | 21 | 0.448678 | 0.465203 | 6 | 10 |
| ETTh2 | heterogeneous_no_memory | 7 | 0.351689 | 0.404134 | 3 | 7 |
| ETTh2 | heterogeneous_no_memory | 13 | 0.326780 | 0.381034 | 1 | 5 |
| ETTh2 | heterogeneous_no_memory | 21 | 0.332591 | 0.387189 | 1 | 5 |
| ETTh2 | heterogeneous_retrieval | 7 | 0.351412 | 0.405579 | 4 | 8 |
| ETTh2 | heterogeneous_retrieval | 13 | 0.323336 | 0.381031 | 1 | 5 |
| ETTh2 | heterogeneous_retrieval | 21 | 0.330913 | 0.386889 | 1 | 5 |
| ETTh2 | homogeneous_no_memory | 7 | 0.365755 | 0.416192 | 4 | 8 |
| ETTh2 | homogeneous_no_memory | 13 | 0.412165 | 0.439549 | 5 | 9 |
| ETTh2 | homogeneous_no_memory | 21 | 0.319177 | 0.382964 | 0 | 4 |
| ETTh2 | homogeneous_retrieval | 7 | 0.319875 | 0.384108 | 3 | 7 |
| ETTh2 | homogeneous_retrieval | 13 | 0.370119 | 0.422681 | 1 | 5 |
| ETTh2 | homogeneous_retrieval | 21 | 0.318909 | 0.381141 | 1 | 5 |

같은 checkpoint의 모든 층 memory 개입 (ΔMSE=개입−full, 재학습 아님):

| Data | Variant | off | uniform | recent |
|---|---|---:|---:|---:|
| ETTh1 | heterogeneous_retrieval | -0.005142 | -0.001652 | +0.002332 |
| ETTh1 | homogeneous_retrieval | +0.001797 | -0.000754 | +0.001767 |
| ETTh2 | heterogeneous_retrieval | -0.000616 | -0.001995 | +0.000231 |
| ETTh2 | homogeneous_retrieval | -0.006661 | -0.002792 | +0.001247 |

- 검증 통과:24matrix/중복/누락, frozen source hash/학습 commit 일치, 조건별 초기parameter/count, minimum validation checkpoint/복원, 전체 test element수, CSV/history/TensorBoard 일치, horizon평균, 저장config/checkpoint hash/finite weight, train-only scaler와 test 첫 target 경계, 층별 population 진단, 독립 NumPy 기반 macro/paired 통계. `check_summary.json`.
- 개별 fresh checkpoint 재평가는 이 horizon 집계 시점 not run. 두 horizon 학습이 모두 종료된 뒤 ETTh1/heterogeneous-retrieval/seed7의 전체 test MSE/MAE 및 off/uniform/recent를 검증한다(atol1e-12). 결과는 check_reload.json 및 전체 완료 기록에 남긴다.
- Exact launcher: `bash f_lif_pop_tcn_v1/forecasting/scripts/run_ett.sh --suite ett-tcn-h96-20260914 --pred_len 96` (run_stage2.sh가 호출). 각 실제 subprocess명령은 manifest/completion 및 run JSON. 분석 호출: `summarize('ett-tcn-h96-20260914'); check('ett-tcn-h96-20260914')`; 모든 학습이 끝난 뒤 `check_reload('ett-tcn-h96-20260914', 96, "cuda:0")`를 실행한다. 관리 script command: `/home/yschoi/.conda/envs/snn_recall/bin/python /home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_tcn_v1/forecasting/scripts/finish_stage2.py --wait --finalize`; 환경 prefix는 사전 기록과 동일.
- Artifact: `NSMT/f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h96-20260914/`의24run JSON,manifest/completion,REPORT,per_run/per_task/macro/paired/paired_macro_by_seed/interaction/interventions CSV,aggregate,checks,comparison PNG/PDF. Log 위치는 per_run.csv의log_path; task `log/ett-tcn-h96-20260914/` 하위에 neorecall CSV/TensorBoard/logargs/model_state 형식. Raw events/checkpoints/stdout은 local.
- 제한: 최대10/early-stop3의 짧은 예산, 세seed/두dataset. 층별진단은 첫 test8window만, all-layer intervention은 어느 층이 원인인지 분리하지 않는다. 1차 대비 parameter가24714개 많고 current residual/depth가 추가되어 matched-capacity 비교가 아니다. 논문 원본 재현/longer-budget/last-head 학습/uniform-recent 재학습/synthetic recall/에너지측정: not run. Test로 재선택하지 않았다. Main 통합/push: not run.

## 2026-09-14 — 사용자 PopulationLIF implementation review 대조

- 요청: `NSMT/docs/PopulationLIF_implementation_review.md`를 읽고 PopulationLIF를 검토한다. 리뷰 문서 전체, 원 concept의 selection/gate/soft-hard 미정/상태해석 부분, 1차 및2차 layers.py와 기존 check_model.py/README를 대조했다. 사용자 원문/리뷰 파일은 수정하지 않았다.
- 범위는 구현·개념 검토이며 새 학습 실험이 아니다. Branch `exp/f-lif-pop-tcn-v1`, training snapshot `794a29693a97dc1246c97ccf42d37ff6fd0ac585`, base `ed5c8d16fb5e9c56234c58d6a8d870ffe20f4efc`. 점검 시점2차H96 24개 complete, H7208개 complete/4개 running/12개 pending. 학습 source/HEAD 변경, 재학습, checkpoint 수정: not run.
- 핵심 판정: 사용자 리뷰에 동의한다. 현재 것은 **population-based content-adaptive dense membrane-memory retrieval**이며 explicit content-dependent read mask/sparse exclusion은 없다. `layers.py:64–76`의 모든 과거 softmax와 weighted sum, `:91`의 diagnostic-only future padding을 구별해야 한다. Recent override는 고정 recency test intervention이지 학습된 content mask가 아니다. Population, 이질적 fixed tau, shared input, current pre-query/past post-reset value, K 공유 temporal weight, additive signed evidence, separate gate, population spike output 및 window-local memory는 구현되어 있다. 두 PopulationLIF 클래스 AST가 동일함을 확인했다.
- 원문 §24.5는 soft/hard 선택을 미정으로 둔다. Dense pilot 자체를 무조건 구현 오류로 취급하지는 않지만, 원문 §10의 read selection/read strength 구분 및 irrelevant memory exclusion까지 검증한 것으로 설명하면 부정확하다. 기존 1차/2차 결과의 '검색'은 dense 조건부 가중 검색을 뜻한다. Fractional prior는 선택 사항이므로 누락 버그가 아니다.
- 추가 설계 제약1: bias-free linear Q/K + L2 normalization은 양의 비례 관계인 비영 상태의 절대 크기를 score에서 구별하지 못한다(normalization epsilon보다 큰 norm 기준). Value의 진폭은 보존된다. Homogeneous population은 K개 막전위가 동일하게 유지되므로 같은 부호의 과거 상태는 cosine에서 크기로 구별되지 않는다. Redundant-population 대조로 유효하나 effective representation capacity가 같다는 뜻은 아니다.
- 추가 설계 제약2: sigmoid gate는 u_bar만 보며 memory bank/score/선택된 후보 수를 직접 보지 않는다. 기억 기여를 학습으로 작게 만들 수 있지만 같은 u_bar에서 후보의 질만 바뀌면 gate는 같다. Softmax는 후보가 모두 낮은 score여도 총weight1이며, 기본 경로에는 null/empty-support 선택이 없다. 이는 문서상 gate가 '존재한다'는 판정을 뒤집는 버그가 아니라 후보 품질에 대한 적응의 제한이다.
- CPU 진단만 시행: seed7,threads2,Py3.10.18/torch1.12.0+cu113, conda Python 및 LD_LIBRARY_PATH는 기존과 동일. T8,BC2,D3 random input의 과거168weights 모두 양수(min6.967689e-5). Identity Q/K에서 [1,2,3,4]와10배 상태의 weight≈[.5,.5]. Homogeneous currents[.5,.4,-20.]의 마지막 query/past cosine=[−1,−1]인데 weights=[.5,.5],gate=.5,evidence 각+.00325379. 두 history의 현재 charge를 모두[.1,.1,.1,.1]로 맞추고 past state부호만 바꾸면 gate=.5로 같고 evidence±.00247737. Homogeneous random input의 구성원간 막전위 차이는0. 초기화 상태에서 보인 구조 반례이며 학습된 checkpoint의 실패율/성능 원인 측정이 아니다.
- 기존 check_model의 causal/future-perturbation 검사는 sparse/semantic selectivity 검사가 아니었다. H720 MSE악화(.679887→.711945)는 dense v1의 결과로 제한해야 한다. Masked/sparse 원안의 실패 또는 mask 추가 시 개선을 입증하지 않는다.
- 후속 구현 제안(not run): dense v1 보존, 별도버전의 K공유 mask; survivor normalization과 별도gate를 분리; empty support면M=0; top-k만으로 후보 없음이 해결되지 않는 점 확인; excluded-slot weight0/empty evidence0/direct-read 불변성/causality/known-lag synthetic distractor 검증. Soft mask도 모두 양수면 sparse가 아니며, dense scoring 후 sparse sum은 search 효율 개선을 자동 보장하지 않는다. Mask된 raw input의 영향이 이후 state에 남을 수 있으므로 직접slot제외와 과거입력 완전삭제를 구별한다.
- Artifact: `NSMT/f_lif_pop_tcn_v1/forecasting/results/stage-comparison-20260914/IMPLEMENTATION_REVIEW.md` 및 `implementation_review_probes.json`에 상세판정/수치/seed/device/reproduction recipe/source SHA256를 보존했다. 정확한 진단 실행은 cwd NSMT에서 `env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python -`의 inline Python이며, JSON의 reproduction 항목에 입력·검사식을 기록했다. 학습데이터 사용은 없으며 CPU toy input만 사용했다. 두 layers.py의 현재 hash와 기록 hash를 재확인한다.
- 이 검토 기록과 artifacts는 이미 대기 중인2차 후처리의 comparison디렉터리/canonical log 보존 범위에 포함된다. 학습 중 commit을 추가하지 않아 pending run provenance를 유지한다. Main 통합/push: not run.

## 2026-09-14 — 2차 Spike-TCN H720 완료 (24 runs)

- Branch `exp/f-lif-pop-tcn-v1`; base `ed5c8d16fb5e9c56234c58d6a8d870ffe20f4efc`; training snapshot `794a29693a97dc1246c97ccf42d37ff6fd0ac585`. 24개 전부 exit0. 전체48개 완료 tag/commit은 두 horizon 검증 뒤 별도 기록한다.
- 목적/코드/데이터/환경/seed/hyperparameters는 직전 2차 사전 기록과 동일하다. Convolution2개/kernel3/dilation1,2/current residual; population/optimizer/data 코드는 사전 동결했다. ETTh1/ETTh2×seeds7/13/21×네 조건, flatten head만 학습했다.
- Full test elements/run: 10891440; parameters/run: 997567. Launcher wall 1491.5s; run 131.2–462.7s; max GPU allocated 3.225GiB.
- 24개 중 2개가10epoch 상한에 도달했다. 24개가 감소된 LR로 학습했다. 최소 validation MSE epoch는1-based 1–7. 최종 checkpoint 복원 검증 통과.
- 이질적 population에서 retrieval 추가 macro MSE 0.800043→0.809756 (+1.214%). Macro는 데이터셋 평균을 seed별 계산한 뒤3seed mean±sample SD이며 horizon 간 합산하지 않는다.

| Variant | MSE ± SD | MAE ± SD |
|---|---:|---:|
| heterogeneous_no_memory | 0.800043 ± 0.046684 | 0.628503 ± 0.022646 |
| heterogeneous_retrieval | 0.809756 ± 0.031170 | 0.631460 ± 0.010588 |
| homogeneous_no_memory | 0.787103 ± 0.062192 | 0.624228 ± 0.019320 |
| homogeneous_retrieval | 0.811149 ± 0.040830 | 0.627735 ± 0.012471 |

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_no_memory | 0.568987 ± 0.026019 | 0.547698 ± 0.015097 |
| ETTh1 | heterogeneous_retrieval | 0.570324 ± 0.016979 | 0.547276 ± 0.009581 |
| ETTh1 | homogeneous_no_memory | 0.574332 ± 0.011656 | 0.556559 ± 0.007015 |
| ETTh1 | homogeneous_retrieval | 0.558881 ± 0.011577 | 0.549593 ± 0.004695 |
| ETTh2 | heterogeneous_no_memory | 1.031099 ± 0.067494 | 0.709309 ± 0.030266 |
| ETTh2 | heterogeneous_retrieval | 1.049188 ± 0.054940 | 0.715645 ± 0.014577 |
| ETTh2 | homogeneous_no_memory | 0.999875 ± 0.121084 | 0.691898 ± 0.040718 |
| ETTh2 | homogeneous_retrieval | 1.063416 ± 0.092372 | 0.705877 ± 0.029111 |

각 실행: best epoch은0-based.

| Data | Variant | Seed | MSE | MAE | Best epoch | Epochs |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | heterogeneous_no_memory | 7 | 0.539062 | 0.530400 | 1 | 5 |
| ETTh1 | heterogeneous_no_memory | 13 | 0.581634 | 0.554483 | 2 | 6 |
| ETTh1 | heterogeneous_no_memory | 21 | 0.586264 | 0.558212 | 1 | 5 |
| ETTh1 | heterogeneous_retrieval | 7 | 0.565036 | 0.544620 | 2 | 6 |
| ETTh1 | heterogeneous_retrieval | 13 | 0.556617 | 0.539302 | 2 | 6 |
| ETTh1 | heterogeneous_retrieval | 21 | 0.589317 | 0.557904 | 1 | 5 |
| ETTh1 | homogeneous_no_memory | 7 | 0.561015 | 0.549588 | 3 | 7 |
| ETTh1 | homogeneous_no_memory | 13 | 0.582681 | 0.556472 | 6 | 10 |
| ETTh1 | homogeneous_no_memory | 21 | 0.579299 | 0.563616 | 2 | 6 |
| ETTh1 | homogeneous_retrieval | 7 | 0.570385 | 0.554411 | 4 | 8 |
| ETTh1 | homogeneous_retrieval | 13 | 0.547233 | 0.545031 | 3 | 7 |
| ETTh1 | homogeneous_retrieval | 21 | 0.559026 | 0.549338 | 6 | 10 |
| ETTh2 | heterogeneous_no_memory | 7 | 0.953213 | 0.674360 | 0 | 4 |
| ETTh2 | heterogeneous_no_memory | 13 | 1.072437 | 0.726655 | 0 | 4 |
| ETTh2 | heterogeneous_no_memory | 21 | 1.067648 | 0.726911 | 0 | 4 |
| ETTh2 | heterogeneous_retrieval | 7 | 1.101938 | 0.727556 | 0 | 4 |
| ETTh2 | heterogeneous_retrieval | 13 | 0.992293 | 0.699390 | 0 | 4 |
| ETTh2 | heterogeneous_retrieval | 21 | 1.053333 | 0.719989 | 0 | 4 |
| ETTh2 | homogeneous_no_memory | 7 | 0.986536 | 0.684871 | 4 | 8 |
| ETTh2 | homogeneous_no_memory | 13 | 1.127076 | 0.735671 | 1 | 5 |
| ETTh2 | homogeneous_no_memory | 21 | 0.886013 | 0.655150 | 2 | 6 |
| ETTh2 | homogeneous_retrieval | 7 | 0.997024 | 0.685414 | 4 | 8 |
| ETTh2 | homogeneous_retrieval | 13 | 1.168908 | 0.739204 | 2 | 6 |
| ETTh2 | homogeneous_retrieval | 21 | 1.024316 | 0.693013 | 2 | 6 |

같은 checkpoint의 모든 층 memory 개입 (ΔMSE=개입−full, 재학습 아님):

| Data | Variant | off | uniform | recent |
|---|---|---:|---:|---:|
| ETTh1 | heterogeneous_retrieval | -0.019497 | -0.002820 | +0.007008 |
| ETTh1 | homogeneous_retrieval | -0.008102 | -0.000646 | +0.004169 |
| ETTh2 | heterogeneous_retrieval | -0.023199 | -0.000465 | +0.003628 |
| ETTh2 | homogeneous_retrieval | -0.043576 | -0.008340 | -0.003596 |

- 검증 통과:24matrix/중복/누락, frozen source hash/학습 commit 일치, 조건별 초기parameter/count, minimum validation checkpoint/복원, 전체 test element수, CSV/history/TensorBoard 일치, horizon평균, 저장config/checkpoint hash/finite weight, train-only scaler와 test 첫 target 경계, 층별 population 진단, 독립 NumPy 기반 macro/paired 통계. `check_summary.json`.
- 개별 fresh checkpoint 재평가는 이 horizon 집계 시점 not run. 두 horizon 학습이 모두 종료된 뒤 ETTh1/heterogeneous-retrieval/seed7의 전체 test MSE/MAE 및 off/uniform/recent를 검증한다(atol1e-12). 결과는 check_reload.json 및 전체 완료 기록에 남긴다.
- Exact launcher: `bash f_lif_pop_tcn_v1/forecasting/scripts/run_ett.sh --suite ett-tcn-h720-20260914 --pred_len 720` (run_stage2.sh가 호출). 각 실제 subprocess명령은 manifest/completion 및 run JSON. 분석 호출: `summarize('ett-tcn-h720-20260914'); check('ett-tcn-h720-20260914')`; 모든 학습이 끝난 뒤 `check_reload('ett-tcn-h720-20260914', 720, "cuda:0")`를 실행한다. 관리 script command: `/home/yschoi/.conda/envs/snn_recall/bin/python /home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_tcn_v1/forecasting/scripts/finish_stage2.py --wait --finalize`; 환경 prefix는 사전 기록과 동일.
- Artifact: `NSMT/f_lif_pop_tcn_v1/forecasting/results/ett-tcn-h720-20260914/`의24run JSON,manifest/completion,REPORT,per_run/per_task/macro/paired/paired_macro_by_seed/interaction/interventions CSV,aggregate,checks,comparison PNG/PDF. Log 위치는 per_run.csv의log_path; task `log/ett-tcn-h720-20260914/` 하위에 neorecall CSV/TensorBoard/logargs/model_state 형식. Raw events/checkpoints/stdout은 local.
- 제한: 최대10/early-stop3의 짧은 예산, 세seed/두dataset. 층별진단은 첫 test8window만, all-layer intervention은 어느 층이 원인인지 분리하지 않는다. 1차 대비 parameter가24714개 많고 current residual/depth가 추가되어 matched-capacity 비교가 아니다. 논문 원본 재현/longer-budget/last-head 학습/uniform-recent 재학습/synthetic recall/에너지측정: not run. Test로 재선택하지 않았다. Main 통합/push: not run.

## 2026-09-14 — 2차 전체48개 검증 및 로컬 완료 기록

- H96/H720 각24개 학습 및 전체 artifact audit 완료. 두 horizon 모두 새 config/checkpoint 객체로 전체 test MSE/MAE와 모든 memory 개입을 atol1e-12에서 재현했다. 각 horizon 기록에서 pending이던 check_reload는 이제 passed다.
- 1차/2차 48쌍의 실제 데이터·전처리·학습 조건 일치 검사를 통과했다. `results/stage-comparison-20260914/{REPORT.md,macro.csv,per_task.csv,paired_macro_by_seed.csv,training_selection.csv,tcn_layer_diagnostics.csv,check_comparison.json,comparison.png,comparison.pdf}`에 비교를 보존한다. 각 horizon은 별개 task로 집계한다.
- 최종 artifact 생성 뒤 모델/학습 source는 바꾸지 않았다. 자동 후처리는 source/data/checkpoint/지표 검증을 수행했다. 자동 생성한 2차 그림의 사람/시각 검토는 not run; CSV 수치 검증은 passed.
- 완료 commit은 annotated tag `exp/f-lif-pop-tcn-v1-20260914`로 식별한다. 자동 finalization은 예상 branch/학습 HEAD를 확인하고 이 실험의 명시된 분석 코드·결과·canonical log만 commit한다. 다른 branch/HEAD로 바뀌면 commit/tag를 중지하고 local queue status에 기록한다. Main 통합/remote push: not run.
- 다음 검증 후보(이번에는 not run): 더 긴 공통 학습 budget; uniform/recent 재학습 및 정답 lag가 있는 synthetic recall; population 층별 retrieval 대조; matched-capacity backbone 비교. 이번 두 ETT-hour/3seed/짧은 예산만으로 보편적 이득이나 통계적 유의성을 주장하지 않는다.

## 2026-09-14 22:31 KST — PopulationLIF v2 재실험 사전 검증 및 실행 계획 (snapshot)

- 목적: 사용자 concept 및 implementation review를 다시 대조하고, 누락되었던 명시적 read 제외와 필요 없는 검색의 비사용을 구현한다. Branch `exp/f-lif-pop-v2`, 선택한 base `f8215f54106980bad7c782bf08acaef871175c34` (완료된 v1 TCN). 이 기록의 commit은 annotated tag `exp/f-lif-pop-v2-20260914-snapshot`으로 식별한다. **학습 완료 tag가 아닌 코드/검증 snapshot**이다.
- 원문 재점검: concept §5/7/8/10–14/17/20/24/26–28/33과 구현의 대응은 `NSMT/f_lif_pop_v2/forecasting/README.md` 표에 기록했다. 원문은 변경하지 않았다. Concept SHA256 `01355e2a9cb0766632a52095d88a1676c845d4b4041695aea791b3eabbc01336`; review SHA256 `aa0b6a52a65ca7a0a50d4d5019ab892527393759a822b29b26e3452fdb34f81d`. 기존 untracked 두 사용자 문서는 상태를 보존하며 이번 commit에 포함하지 않는다.
- 구현: Branch B의 직접 막전위 검색. 동일 current를 받는 K4 LIF, fixed tau[2,4,8,16]의 beta=exp(-1/tau); homogeneous는 mean beta 반복. 과거 post-reset state를 현재 pre-retrieval state로 조회한다. L2 정규화 없는 learned Q/K negative mean squared distance로 amplitude를 구분하고, real 후보+learned null 후보에 sparsemax를 적용한다. Positive support가 명시적 mask이며 제외 weight는 정확히0이다. Real mass로 사용량을 조절하고 charged/retrieved memory/best-score margin/real mass를 함께 보는 sigmoid gate로 gamma=.05 evidence를 더한다. null-only이면 memory/gate/evidence=0. K전체에 같은 slot mask를 사용하고 state/history는 window마다 초기화한다. Full BPTT, subtractive reset만 detach. 정확한 fractional derivative 구현은 아니다.
- 선행연구 근거: Martins & Astudillo (2016), https://proceedings.mlr.press/v48/martins16.pdf 의 sparsemax Alg.1/Eq.14. 정해진 top-k 강제선택 없이 정확한0을 만들 수 있으나 매 query의 sparsity/효율을 보장하지 않는다. 모든 과거 score는 계산한다. 초기 자동미분 구현은 torch1.12 deterministic CUDA의 scatter_add backward에서 실패했고, 논문 Eq.14 support-centered gradient의 custom backward로 수정했다. 실패한 smoke artifact도 삭제하지 않았다.
- 코드 구성/스타일: `Config`, `LOAD_MODEL`, `myModel`, `Embedding`, train_one_epoch/val_one_epoch, data_provider, EpochLog/EarlyStopping 패턴을 유지했다. utils.py/data_factory.py/data_loader.py는 v1과 byte-identical. layers.py에 뉴런, backbones.py에 공통 current residual block, ours.py에서 구조 선택, `NSMT/forecasting/f-LIF_pop_v2.py`를 import entry로 제공한다. Dense는 동일 scorer/null/gate에서 softmax만 사용하고 off는 같은 nominal parameters를 예약하되 retrieval을 우회한다. 같은 seed/horizon에서 6조건 초기값과 backbone 간 공통 embedding/head 초기값을 맞췄다.
- 실행 순서: 얕은 patch SNN → Spike-TCN 변형 → hybrid spiking PatchTST → causal spiking TSMixer. Patch는 추가 block 없음, 나머지는2blocks/총3population층. TCN kernel3/dilation1,2; PatchTST causal softmax attention D32/4heads + feature MLP; TSMixer lower-triangular temporal MLP + feature MLP. 후자는 원 논문 재현이나 all-spiking Spikformer가 아니다. Backbone 간 parameter수가 달라 matched-capacity 비교는 아니다.
- 사전 고정 matrix: 각 구조 ETTh1/ETTh2 × H96/H720 × seeds7,13,21 × homogeneous/heterogeneous × off/dense/sparse =72회, 총288회 계획. Horizon별36개 complete suite를 검증하고 H96/720 둘 다 끝난 뒤 다음 구조로 간다. Test 개선 여부로 다음 구조를 선별하지 않는다. Suite명 `selective-v2-20260914_<architecture>_p<horizon>`. 기존 v1은 scorer/gate/budget이 달라 이번 sparse 효과의 직접 대조가 아니며 v2 dense가 직접 대조다.
- 학습 예산: 비동기 선택 질문에 별도 응답이 없어 고지한 기본값 최대30epochs/early stopping6/ReduceLROnPlateau factor.5 patience2를 사용한다. Scheduler와 checkpoint는 validation MSE를 추적하고 엄격한 최저 validation MSE checkpoint를 복원한다. 마지막 epoch/test로 고르지 않는다. AdamW lr.001,weight_decay.01,batch128,clip1; seq336,patch8,stride8,D32,K4,head32/flatten,threshold1,input_scale2,temperature.25,null_logit_init−1. 모든 학습 seed 공통 설정.
- 데이터: 기존 local `NSMT/forecasting/dataset/ETT-small/ETTh{1,2}.csv`,7변수, train[0,8640),val targets[8640,11520),test targets[11520,14400),preceding336 context. Train-only StandardScaler, stride1 windows, drop_lastFalse. H96 train/val/test windows8209/2785/2785; H7207585/2161/2161. 전체 test window×horizon×channel MSE/MAE를 float64 accumulation한다. 데이터 SHA256/scaler/환경패키지/source hashes/명령을 각 result JSON에 저장한다.
- 환경: `/home/yschoi/.conda/envs/snn_recall/bin/python`, Python3.10.18,torch1.12.0+cu113,SpikingJelly0.0.0.0.14,numpy1.26.4,pandas2.3.1,sklearn1.7.1,TensorBoard2.19. RTX A6000 48GiB ×4(GPU0–3),2workers/GPU,CPUthreads2/process,deterministic/TF32off. GPU shared wall time은 독립 속도 benchmark가 아니다.
- 검증 passed: CPU/GPU sparsemax closed forms/shift invariance/finite-difference Jacobian; known-slot/context-switch/amplitude distractor/excluded-slot direct-read/null-empty; 모든 구조의6조건 초기parameters와 공통embedding/head hash; 층별 binary spikes/causality/no future leakage/window reset/off 및 gamma0 exact equivalence/homogeneous identity; finite gradients/nonzero query-key-gate gradients/H96·H720 shape. 결과 `results/check_model_cpu.json`, `check_model_cuda0.json`, `source_protocol_check.json`. 전체 Python compile passed.
- Exact 검증 commands (cwd NSMT; prefix `env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib`): `/home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v2/forecasting/check_model.py --device cpu` 및 `--device cuda:0`. 수정 후 GPU H720 smoke commands/관측:

| Architecture | Exact command (동일 env prefix) | Seconds | Peak GiB | Parameters |
|---|---|---:|---:|---:|
| patch | `/home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v2/forecasting/train.py --suite smoke-v2-20260914-r2 --architecture patch --pred_len 720 --num_device 0 --epoch 1 --max_train_batches 2 --max_eval_batches 2` | 3.573 | 1.237 | 972860 |
| patchtst | `/home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v2/forecasting/train.py --suite smoke-v2-20260914-r2 --architecture patchtst --pred_len 720 --num_device 2 --epoch 1 --max_train_batches 2 --max_eval_batches 2` | 4.621 | 3.647 | 992980 |
| tcn | `/home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v2/forecasting/train.py --suite smoke-v2-20260914-r2 --architecture tcn --pred_len 720 --num_device 1 --epoch 1 --max_train_batches 2 --max_eval_batches 2` | 10.248 | 3.609 | 997588 |
| tsmixer | `/home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v2/forecasting/train.py --suite smoke-v2-20260914-r2 --architecture tsmixer --pred_len 720 --num_device 3 --epoch 1 --max_train_batches 2 --max_eval_batches 2` | 4.541 | 3.633 | 991024 |

- Smoke는1epoch/최대2train·eval batches의 실행/복원 검증이며 성능 결론이나288회 본실험에 포함하지 않는다. 초기 실패 `results/smoke-v2-20260914`, 수정 후 성공4회 `results/smoke-v2-20260914-r2`. stdout은 `log/preflight/` local. 수정 후 일부 과거 weight가0이고 null-only가 관측되었지만 최종 학습에서의 선택 동작은 다시 측정해야 한다.
- 본실험 예정 command (cwd NSMT): `bash f_lif_pop_v2/forecasting/scripts/run_pipeline.sh` (내부 conda python `scripts/run_pipeline.py --finalize`; 기본288matrix). Detached 실행 PID/명령/snapshot은 `scripts/queues/launcher.json`, live pipeline/suite 상태와 lock은 같은 queues 아래, 전체 stdout은 `log/selective-v2-20260914.pipeline.stdout`. 실행 시 branch/HEAD/source hashes를 고정 검증하고 실패하면 후속 단계 실행을 중단한다. 부분 suite 자동 resume은 미구현이며 기존 ID/결과를 덮어쓰지 않는다.
- 본실험 완료 시 자동 검증: complete matrix, source/초기parameter hashes, min-val checkpoint/복원, 전체 element수, CSV/history/TensorBoard, finite checkpoint/hash, train scaler/target boundary, 층별 support/empty/null/lag, 독립 macro SD 및 paired deltas. 각 horizon ETTh1 heterogeneous sparse seed7은 fresh 객체/checkpoint로 전체 test+off/uniform/recent를 atol1e-12에서 재평가한다. 각 구조72회 완료 시 canonical log에 모든run/bestepoch/mean±SD/ΔMSE/진단을 append하고 로컬 commit 및 annotated `exp/f-lif-pop-v2-<architecture>-<completion-date>` tag를 남긴다.
- 결과/로그: task `results/<suite>/`에 full-precision run JSON/CSV/REPORT/aggregate/audits/manifest/completion. `log/<suite>/<dataset>/<date>/<config>/seed+variant/`에 기존 neorecall 방식 log/best_log_0.csv,final+result.csv,train_0,val_0,logargs.txt,model_state/config.pt,best+model.pt. Checkpoints/events/stdout/queues는 local, 코드/config/text결과/canonical기록은Git. 진단은 첫8test windows에서 층별 support density/empty fraction/real mass/lag/membrane diversity/evidence scale. Off/uniform/recent는 같은 checkpoint의 평가개입이며 uniform/recent는 null을 우회하는 강제 real-read이다.
- 결론/범위: 수정된 선택 연산과 네 구조의 실행 가능성을 사전 검증했다. 이 snapshot 시점 본학습/전체288회 결과/audit: **not run**, 뒤의 완료 기록으로 갱신한다. End-to-end synthetic forecasting 학습, ETT 정답lag 검증, 에너지측정, 통계적 유의성 검정, matched-capacity backbone 비교: not run. Sparsemax inactive-region gradient 및 support/null collapse는 실제 진단으로 확인해야 한다. Read mask는 저장된 과거slot을 제외할 뿐 raw input의 이후 recurrent-state 영향까지 삭제하지 않는다. Main 통합/remote push: not run.

## 2026-09-14 22:32 KST — v2 본실험 실행 시작

- Snapshot commit `a7d382e5ac540d12c2a6ad9056ac828739680442`, tag `exp/f-lif-pop-v2-20260914-snapshot`을 고정하고 `bash f_lif_pop_v2/forecasting/scripts/run_pipeline.sh`를 detached process로 실행했다. Launcher PID `1246599`; exact absolute command/cwd/UTC는 `NSMT/f_lif_pop_v2/forecasting/scripts/queues/launcher.json`.
- 첫 suite `selective-v2-20260914_patch_p96`의8개 학습 subprocess가 GPU0–3에서 시작했음을 확인했다. 이 시점 완료 결과는 없으며, H720/TCN/PatchTST/TSMixer는 예정 상태다. Live per-job status는 suite queue JSON, 전체 진행은 `scripts/queues/selective-v2-20260914.json`과 `log/selective-v2-20260914.pipeline.stdout`.
- 본 기록은 canonical log에 즉시 append했으며 첫 구조 완료 자동 commit에 함께 포함된다. 학습 중 source/HEAD를 바꾸지 않기 위해 별도 중간 commit은 만들지 않는다. 이후 각 구조의 학습·audit·fresh checkpoint 평가가 끝나면 상세 결과/완료 commit/tag가 자동 기록된다.

### 22:33 KST 실행 시작 기록 보충

- 위의 “완료 결과는 없음”은 launcher 직후 상태를 뜻한다. 기록 작성 중 첫 실행이 종료되었으므로 queue를 다시 읽은 아래 상태로 보충한다. 최종 비교 결과는 전체 suite 검증 뒤 보고한다.
- 관측 상태: {'pending': 24, 'running': 8, 'complete': 4, 'failed': 0}.

| 완료 Run | Epochs | Best epoch (0-based) | Test MSE | Test MAE |
|---|---:|---:|---:|---:|
| patch_ETTh1_p96_flatten_homogeneous_no_memory_seed7 | 22 | 15 | 0.448847393076 | 0.461694558312 |
| patch_ETTh1_p96_flatten_heterogeneous_no_memory_seed7 | 15 | 8 | 0.419019395480 | 0.438666818100 |
| patch_ETTh2_p96_flatten_homogeneous_no_memory_seed7 | 8 | 1 | 0.332595184138 | 0.398718614855 |
| patch_ETTh2_p96_flatten_heterogeneous_no_memory_seed7 | 8 | 1 | 0.327866429315 | 0.395793666778 |

- 이들은 개별 완료 결과이며 complete-matrix 집계/audit는 아직 not run. 후속 단계는 queue에 설정된 순서로 실행된다.

## 2026-09-15 10:41 KST — v2 1차 H96 36회 완료 브리핑 및 후처리 복구

- 목적/branch/base/훈련환경/데이터/학습 hyperparameter는 2026-09-14 v2 사전 고정 기록과 같다. Branch `exp/f-lif-pop-v2`, base `f8215f54106980bad7c782bf08acaef871175c34`, H96 training commit `a7d382e5ac540d12c2a6ad9056ac828739680442`; 이번36회 완료 및 복구 commit은 annotated tag `exp/f-lif-pop-v2-patch-h96-20260915`로 식별한다. ETTh1/ETTh2 × seeds7/13/21 × homo/hetero × off/dense/sparse,seq336/H96/patch8/D32/K4/head32,AdamW lr.001 wd.01,batch128,max30/early6/scheduler2,factor.5,min validation MSE. Train8640/val2880/test2880의 train-only scaler와 모든 test window를 사용했다.
- 최초288회 pipeline은 H96 학습36회 종료 후 audit에서 중단되어 H720/이후 구조는 **not run** 상태였다. 기존 queue의 status=running은 stale였으며 이번에 failed 원인을 명시했다. stdout traceback/completion/original failure JSON을 보존한다. Model crash나 학습 실패는 아니었다.
- 원인: `weight=p/max(real_mass,1e-12)`인 frozen neuron에서 tiny positive dense mass이면 read 합이1보다 작다. 기존 sum(lag_mass)=nonempty fraction 검사가5개 dense run에서 실패했다. 이전 README와 사전 log의 단위합/conditional mean 설명을 이 기록으로 정정한다. 실제 invariant는 mean(real_mass/max(real_mass,1e-12)); raw mean_lag/entropy에도 floor 영향이 있으므로 순수한 조건부 통계로 읽지 않는다. 기존 metric/raw JSON/checkpoint와10개 training source SHA256는 모두 그대로 보존했다.
- 수정 코드: check_summary.py가36개 CPU checkpoint를 다시 읽어 weight합/real mass/저장lag/empty를 직접 검증하고 check_read_mass.json을 기록한다. 단순 tolerance 완화가 아니다. run_pipeline.py에 completed-suite만 재검증하는 --resume 및 실패 상태 기록을 추가했다. Source hash/epoch budget 일치 확인, partial suite 거부, 기존 run/checkpoint 덮어쓰기 금지를 유지한다. README/summarize에 정확한 floor 설명을 추가했다.
- 검증 commands: `env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v2/forecasting/check_summary.py --suite selective-v2-20260914_patch_p96`; 같은 환경에서 `scripts/run_pipeline.py`의 `audit_suite(SimpleNamespace(gpus=[0]),'patch',96,'selective-v2-20260914_patch_p96')`. CPU checkpoint36개 invariant, 전체 artifact/CSV/TensorBoard/data/hash/minval/paired macro checks 모두 passed. Representative sparse ETTh1 seed7 fresh GPU checkpoint의 전체 test+off/uniform/recent MSE/MAE 재현(atol1e-12) passed. 변경한 Python3개 compile passed. Raw stdout `log/preflight/{check_summary_recovery,audit_recovery}.stdout`.
- 결론: 이질성만 추가하면 MSE −4.26%; heterogeneous dense는 off 대비 −.23%, sparse는 off 대비 +.76% 및 dense 대비 +.99%. Sparse−dense macro ΔMSE는3seed 모두 양수. Homogeneous sparse는 off 대비 −.96%이나 ETTh2가 주된 기여. 이질성+선택적 검색의 추가 성능 개선은 이번 H96에서 입증되지 않았다. Hetero sparse density32.6631%, empty14.9941%(첫8test windows)로 선택/비사용 동작은 관측된다. Same checkpoint retrieval-off ΔMSE +.006680은 검색을 사용한다는 관찰이며 별도 훈련 off baseline보다 우수하다는 뜻은 아니다. 1/36은30epoch 도달, best epoch0-based1–29.
- Artifact: `NSMT/f_lif_pop_v2/forecasting/results/selective-v2-20260914_patch_p96/BRIEFING_20260915.md`, REPORT.md, per_run/per_task/macro/paired/paired_macro_by_seed/layer_diagnostics CSV, aggregate, check_summary/check_read_mass/check_reload JSON. 모든36개 세부metric/bestepoch/artifact path는 per_run.csv; 로그/checkpoint는 앞의 neorecall layout. Raw events/stdout/checkpoints는 local, text결과는Git.
- 후속 재개 예정 exact command (cwd NSMT): `bash f_lif_pop_v2/forecasting/scripts/run_pipeline.sh --resume`. H96 결과를 재검증하고 H720부터 기존 순서를 따른다. 이번 완료는 H96만이며 전체72/288회 완료를 뜻하지 않는다. H720/다음구조 최종 결과, synthetic recall 학습, 에너지, 통계적 유의성 검정: not run. Main 통합/push: not run.

| Variant | MSE ± SD | MAE ± SD |
|---|---:|---:|
| heterogeneous_dense | 0.377628 ± 0.008199 | 0.419095 ± 0.006241 |
| heterogeneous_no_memory | 0.378484 ± 0.011626 | 0.419706 ± 0.008360 |
| heterogeneous_sparse | 0.381363 ± 0.005930 | 0.421432 ± 0.004041 |
| homogeneous_dense | 0.396104 ± 0.008466 | 0.431852 ± 0.005043 |
| homogeneous_no_memory | 0.395326 ± 0.006549 | 0.431763 ± 0.003007 |
| homogeneous_sparse | 0.391533 ± 0.003132 | 0.427766 ± 0.001204 |

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
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


- 표의 SD는 dataset을 먼저 평균한 뒤 세 seed 사이에서 계산한 sample SD다.

### 2026-09-15 10:42 KST — H720 실제 재개 확인

- Recovery commit `9574d2286d26c4502c504da63f349e42a4cb0eaa` / tag `exp/f-lif-pop-v2-patch-h96-20260915`. `bash f_lif_pop_v2/forecasting/scripts/run_pipeline.sh --resume` detached launcher PID1392366가 H96 재검증을 통과하고 H720 suite의8개 GPU학습을 시작했다(완료0/실행8/대기28/실패0). H96 raw results/checkpoints와10개 training source는 그대로다.
- 재개 exact command/PID/UTC/commit은 `NSMT/f_lif_pop_v2/forecasting/scripts/queues/launcher-resume-20260915.json`, stdout은 `log/selective-v2-20260915.resume.stdout`. H720/TCN/PatchTST/TSMixer 결과는 아직 없으며 각 단계 완료 시 자동 검증/상세기록/commit/tag를 수행한다. 이번 실행확인 append는 다음 완료 commit에 포함한다.

## 2026-09-15 10:48 KST — PopulationLIF v2 patch 완료 (72 runs)

- Branch `exp/f-lif-pop-v2`; base `f8215f54106980bad7c782bf08acaef871175c34`; 각 horizon training commit은 completion.json의 training_commit에 기록한다 (후처리 복구 전후 commit이 다를 수 있으며 학습 source hashes는 동일). 완료 commit은 tag `exp/f-lif-pop-v2-patch-20260915`로 식별한다. 목적: 동일 점수·gate에서 dense vs sparse 선택 효과 및 population 이질성을 분리한다.
- ETTh1/ETTh2 × seeds7/13/21 × homo/hetero × off/dense/sparse × H96/720. Seq336,patch8,D32,K4,head32/flatten,tau2..16,gamma.05,temperature.25,null init−1. 최대30epochs,early-stop6,ReduceLROnPlateau factor.5/patience2,AdamW lr.001/wd.01,batch128,clip1. 최소 validation MSE checkpoint를 복원해 평가했다.
- 데이터와 환경: 기존 ETT-hour train[0,8640),val[8640,11520),test[11520,14400),train-only StandardScaler,7변수,stride1,context336. H96 windows8209/2785/2785, H7207585/2161/2161. Conda snn_recall/Py3.10/torch1.12.0+cu113,CPUthreads2,deterministic/TF32off. GPU/worker 배치와 모든 실행명령은 각 manifest 및 run JSON, source/data hashes/패키지 버전도 run JSON에 있다.
- 실행 pipeline command: `/home/yschoi/.conda/envs/snn_recall/bin/python /home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/run_pipeline.py --finalize`. Suites: `selective-v2-20260914_patch_p96`, `selective-v2-20260914_patch_p720`.

| Horizon | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| 96 | heterogeneous_dense | 0.377628 ± 0.008199 | 0.419095 ± 0.006241 |
| 96 | heterogeneous_no_memory | 0.378484 ± 0.011626 | 0.419706 ± 0.008360 |
| 96 | heterogeneous_sparse | 0.381363 ± 0.005930 | 0.421432 ± 0.004041 |
| 96 | homogeneous_dense | 0.396104 ± 0.008466 | 0.431852 ± 0.005043 |
| 96 | homogeneous_no_memory | 0.395326 ± 0.006549 | 0.431763 ± 0.003007 |
| 96 | homogeneous_sparse | 0.391533 ± 0.003132 | 0.427766 ± 0.001204 |

H96: 1/36 budget cap; 133580 nominal parameters; individual run 19.0–219.4s.
- H96 homogeneous: sparse−dense paired macro ΔMSE -0.004571 (seed SD 0.006096); 평균 MSE 기준 sparse가 낮음. 통계적 유의성 주장은 하지 않는다.
- H96 heterogeneous: sparse−dense paired macro ΔMSE +0.003735 (seed SD 0.002404); 평균 MSE 기준 sparse가 높거나 같음. 통계적 유의성 주장은 하지 않는다.
- H96 heterogeneous_dense: final-layer support density 0.998776, empty-read fraction 0.000000, real mass 0.881238 (첫8 test windows).
- H96 heterogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H96 heterogeneous_sparse: final-layer support density 0.326631, empty-read fraction 0.149941, real mass 0.772621 (첫8 test windows).
- H96 homogeneous_dense: final-layer support density 0.998048, empty-read fraction 0.000000, real mass 0.838302 (첫8 test windows).
- H96 homogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H96 homogeneous_sparse: final-layer support density 0.390572, empty-read fraction 0.107705, real mass 0.825785 (첫8 test windows).
| 720 | heterogeneous_dense | 0.706692 ± 0.056545 | 0.591711 ± 0.023375 |
| 720 | heterogeneous_no_memory | 0.685652 ± 0.053363 | 0.582620 ± 0.021507 |
| 720 | heterogeneous_sparse | 0.721476 ± 0.038874 | 0.597785 ± 0.016600 |
| 720 | homogeneous_dense | 0.713817 ± 0.042492 | 0.601710 ± 0.019110 |
| 720 | homogeneous_no_memory | 0.681632 ± 0.055167 | 0.588881 ± 0.020140 |
| 720 | homogeneous_sparse | 0.649910 ± 0.034544 | 0.575184 ± 0.013379 |

H720: 2/36 budget cap; 972860 nominal parameters; individual run 19.5–214.3s.
- H720 homogeneous: sparse−dense paired macro ΔMSE -0.063907 (seed SD 0.076909); 평균 MSE 기준 sparse가 낮음. 통계적 유의성 주장은 하지 않는다.
- H720 heterogeneous: sparse−dense paired macro ΔMSE +0.014783 (seed SD 0.018488); 평균 MSE 기준 sparse가 높거나 같음. 통계적 유의성 주장은 하지 않는다.
- H720 heterogeneous_dense: final-layer support density 0.999274, empty-read fraction 0.000000, real mass 0.881230 (첫8 test windows).
- H720 heterogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H720 heterogeneous_sparse: final-layer support density 0.385989, empty-read fraction 0.091833, real mass 0.866983 (첫8 test windows).
- H720 homogeneous_dense: final-layer support density 0.997401, empty-read fraction 0.000000, real mass 0.898756 (첫8 test windows).
- H720 homogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H720 homogeneous_sparse: final-layer support density 0.456939, empty-read fraction 0.057825, real mass 0.913030 (첫8 test windows).

각 실행 (best epoch은0-based):

| Run | MSE | MAE | Best epoch | Epochs |
|---|---:|---:|---:|---:|
| patch_ETTh1_p96_flatten_heterogeneous_dense_seed7 | 0.421737 | 0.440980 | 7 | 14 |
| patch_ETTh1_p96_flatten_heterogeneous_dense_seed13 | 0.430525 | 0.449614 | 6 | 13 |
| patch_ETTh1_p96_flatten_heterogeneous_dense_seed21 | 0.426917 | 0.444833 | 6 | 13 |
| patch_ETTh1_p96_flatten_heterogeneous_no_memory_seed7 | 0.419019 | 0.438667 | 8 | 15 |
| patch_ETTh1_p96_flatten_heterogeneous_no_memory_seed13 | 0.429261 | 0.447700 | 6 | 13 |
| patch_ETTh1_p96_flatten_heterogeneous_no_memory_seed21 | 0.423641 | 0.442704 | 6 | 13 |
| patch_ETTh1_p96_flatten_heterogeneous_sparse_seed7 | 0.421162 | 0.440240 | 7 | 14 |
| patch_ETTh1_p96_flatten_heterogeneous_sparse_seed13 | 0.431880 | 0.449264 | 6 | 13 |
| patch_ETTh1_p96_flatten_heterogeneous_sparse_seed21 | 0.427010 | 0.444661 | 6 | 13 |
| patch_ETTh1_p96_flatten_homogeneous_dense_seed7 | 0.446132 | 0.457246 | 29 | 30 |
| patch_ETTh1_p96_flatten_homogeneous_dense_seed13 | 0.452758 | 0.460293 | 13 | 20 |
| patch_ETTh1_p96_flatten_homogeneous_dense_seed21 | 0.448919 | 0.459259 | 15 | 22 |
| patch_ETTh1_p96_flatten_homogeneous_no_memory_seed7 | 0.448847 | 0.461695 | 15 | 22 |
| patch_ETTh1_p96_flatten_homogeneous_no_memory_seed13 | 0.448215 | 0.456947 | 13 | 20 |
| patch_ETTh1_p96_flatten_homogeneous_no_memory_seed21 | 0.445184 | 0.456039 | 15 | 22 |
| patch_ETTh1_p96_flatten_homogeneous_sparse_seed7 | 0.441728 | 0.452928 | 14 | 21 |
| patch_ETTh1_p96_flatten_homogeneous_sparse_seed13 | 0.452874 | 0.460434 | 13 | 20 |
| patch_ETTh1_p96_flatten_homogeneous_sparse_seed21 | 0.448151 | 0.457498 | 15 | 22 |
| patch_ETTh2_p96_flatten_heterogeneous_dense_seed7 | 0.321676 | 0.390164 | 1 | 8 |
| patch_ETTh2_p96_flatten_heterogeneous_dense_seed13 | 0.343447 | 0.402986 | 1 | 8 |
| patch_ETTh2_p96_flatten_heterogeneous_dense_seed21 | 0.321462 | 0.385991 | 5 | 12 |
| patch_ETTh2_p96_flatten_heterogeneous_no_memory_seed7 | 0.327866 | 0.395794 | 1 | 8 |
| patch_ETTh2_p96_flatten_heterogeneous_no_memory_seed13 | 0.354297 | 0.410348 | 1 | 8 |
| patch_ETTh2_p96_flatten_heterogeneous_no_memory_seed21 | 0.316816 | 0.383025 | 5 | 12 |
| patch_ETTh2_p96_flatten_heterogeneous_sparse_seed7 | 0.334267 | 0.398394 | 1 | 8 |
| patch_ETTh2_p96_flatten_heterogeneous_sparse_seed13 | 0.344531 | 0.402918 | 1 | 8 |
| patch_ETTh2_p96_flatten_heterogeneous_sparse_seed21 | 0.329327 | 0.393116 | 1 | 8 |
| patch_ETTh2_p96_flatten_homogeneous_dense_seed7 | 0.335184 | 0.399934 | 7 | 14 |
| patch_ETTh2_p96_flatten_homogeneous_dense_seed13 | 0.358957 | 0.415028 | 5 | 12 |
| patch_ETTh2_p96_flatten_homogeneous_dense_seed21 | 0.334674 | 0.399352 | 1 | 8 |
| patch_ETTh2_p96_flatten_homogeneous_no_memory_seed7 | 0.332595 | 0.398719 | 1 | 8 |
| patch_ETTh2_p96_flatten_homogeneous_no_memory_seed13 | 0.357430 | 0.413513 | 5 | 12 |
| patch_ETTh2_p96_flatten_homogeneous_no_memory_seed21 | 0.339681 | 0.403667 | 1 | 8 |
| patch_ETTh2_p96_flatten_homogeneous_sparse_seed7 | 0.334672 | 0.399871 | 7 | 14 |
| patch_ETTh2_p96_flatten_homogeneous_sparse_seed13 | 0.335957 | 0.396912 | 13 | 20 |
| patch_ETTh2_p96_flatten_homogeneous_sparse_seed21 | 0.335818 | 0.398952 | 1 | 8 |
| patch_ETTh1_p720_flatten_heterogeneous_dense_seed7 | 0.553456 | 0.546017 | 2 | 9 |
| patch_ETTh1_p720_flatten_heterogeneous_dense_seed13 | 0.583534 | 0.562919 | 2 | 9 |
| patch_ETTh1_p720_flatten_heterogeneous_dense_seed21 | 0.544256 | 0.541338 | 4 | 11 |
| patch_ETTh1_p720_flatten_heterogeneous_no_memory_seed7 | 0.551595 | 0.543991 | 2 | 9 |
| patch_ETTh1_p720_flatten_heterogeneous_no_memory_seed13 | 0.577990 | 0.560323 | 2 | 9 |
| patch_ETTh1_p720_flatten_heterogeneous_no_memory_seed21 | 0.531070 | 0.531371 | 4 | 11 |
| patch_ETTh1_p720_flatten_heterogeneous_sparse_seed7 | 0.555149 | 0.547401 | 2 | 9 |
| patch_ETTh1_p720_flatten_heterogeneous_sparse_seed13 | 0.583113 | 0.562512 | 2 | 9 |
| patch_ETTh1_p720_flatten_heterogeneous_sparse_seed21 | 0.544600 | 0.542128 | 4 | 11 |
| patch_ETTh1_p720_flatten_homogeneous_dense_seed7 | 0.581088 | 0.564274 | 10 | 17 |
| patch_ETTh1_p720_flatten_homogeneous_dense_seed13 | 0.600071 | 0.569999 | 16 | 23 |
| patch_ETTh1_p720_flatten_homogeneous_dense_seed21 | 0.569125 | 0.553658 | 29 | 30 |
| patch_ETTh1_p720_flatten_homogeneous_no_memory_seed7 | 0.567862 | 0.557763 | 10 | 17 |
| patch_ETTh1_p720_flatten_homogeneous_no_memory_seed13 | 0.616757 | 0.580277 | 16 | 23 |
| patch_ETTh1_p720_flatten_homogeneous_no_memory_seed21 | 0.558625 | 0.555435 | 5 | 12 |
| patch_ETTh1_p720_flatten_homogeneous_sparse_seed7 | 0.585916 | 0.567158 | 10 | 17 |
| patch_ETTh1_p720_flatten_homogeneous_sparse_seed13 | 0.612872 | 0.578256 | 9 | 16 |
| patch_ETTh1_p720_flatten_homogeneous_sparse_seed21 | 0.564838 | 0.550714 | 29 | 30 |
| patch_ETTh2_p720_flatten_heterogeneous_dense_seed7 | 0.970810 | 0.684539 | 1 | 8 |
| patch_ETTh2_p720_flatten_heterogeneous_dense_seed13 | 0.834144 | 0.619729 | 2 | 9 |
| patch_ETTh2_p720_flatten_heterogeneous_dense_seed21 | 0.753954 | 0.595728 | 3 | 10 |
| patch_ETTh2_p720_flatten_heterogeneous_no_memory_seed7 | 0.884832 | 0.653843 | 1 | 8 |
| patch_ETTh2_p720_flatten_heterogeneous_no_memory_seed13 | 0.851361 | 0.621073 | 2 | 9 |
| patch_ETTh2_p720_flatten_heterogeneous_no_memory_seed21 | 0.717065 | 0.585116 | 3 | 10 |
| patch_ETTh2_p720_flatten_heterogeneous_sparse_seed7 | 0.958033 | 0.679292 | 1 | 8 |
| patch_ETTh2_p720_flatten_heterogeneous_sparse_seed13 | 0.873152 | 0.636881 | 2 | 9 |
| patch_ETTh2_p720_flatten_heterogeneous_sparse_seed21 | 0.814807 | 0.618495 | 2 | 9 |
| patch_ETTh2_p720_flatten_homogeneous_dense_seed7 | 0.857466 | 0.648211 | 1 | 8 |
| patch_ETTh2_p720_flatten_homogeneous_dense_seed13 | 0.737647 | 0.591484 | 2 | 9 |
| patch_ETTh2_p720_flatten_homogeneous_dense_seed21 | 0.937504 | 0.682635 | 0 | 7 |
| patch_ETTh2_p720_flatten_homogeneous_no_memory_seed7 | 0.876235 | 0.649183 | 1 | 8 |
| patch_ETTh2_p720_flatten_homogeneous_no_memory_seed13 | 0.791374 | 0.614256 | 2 | 9 |
| patch_ETTh2_p720_flatten_homogeneous_no_memory_seed21 | 0.678940 | 0.576370 | 2 | 9 |
| patch_ETTh2_p720_flatten_homogeneous_sparse_seed7 | 0.714199 | 0.584858 | 4 | 11 |
| patch_ETTh2_p720_flatten_homogeneous_sparse_seed13 | 0.755886 | 0.598007 | 2 | 9 |
| patch_ETTh2_p720_flatten_homogeneous_sparse_seed21 | 0.665747 | 0.572110 | 2 | 9 |

- 검증 통과: complete matrix, source/초기parameter hashes, minimum val checkpoint/복원, 전체 element수, CSV/history/TensorBoard, finite checkpoint/hash, train scaler/target boundary, 층별 support/null/lag diagnostics, independent macro/paired deltas. 각 horizon sparse ETTh1 seed7의 fresh checkpoint 및 off/uniform/recent 전체 MSE/MAE를 atol1e-12에서 재현했다.
- Artifact는 `NSMT/f_lif_pop_v2/forecasting/results/<suite>/`의 REPORT,per_run/per_task/macro/paired/paired_macro_by_seed,layer_diagnostics,aggregate,manifest/completion/checks 및36raw result JSON. per_run.csv log_path가 task log/<suite>/<dataset>/<date>/<config>/seed+variant의 neorecall CSV/events/logargs/config.pt/best+model.pt를 가리킨다. Raw events/checkpoints/stdout은 local, 텍스트 결과는 Git.
- 해석 제한: dense와 sparse는 score/nullable candidates/gate를 공유하지만 정규화 방식/지원집합/real probability mass가 함께 달라진다. Sparse 사용은 dense search 비용 절감을 보장하지 않는다. 실제 density/empty rate를 함께 보고 판단한다. Homogeneous는 redundant state 대조, backbone 간 용량은 다르다. 진단은 첫8 test windows; 전체 synthetic recall 학습/정답 ETT lag/에너지 측정/통계적 유의성 검정은 not run. 기존 v1과는 scorer/gate/budget이 달라 동일 실험으로 합산하지 않는다. 다음 단계는 성능 개선 여부로 선별하지 않고 실행/검증 통과 뒤 진행한다. Main 통합/push: not run.

## 2026-09-15 13:39 KST — v2 진행 브리핑 (1차72회 검증 완료, 2차 학습 중)

- 1차 patch H96/H720 각36회 complete 및 check_summary/check_read_mass/check_reload passed. 완료 commit `8c3d13673ce628bff51e7ab858a67b5bc72584da`, annotated tag `exp/f-lif-pop-v2-patch-20260915`. 기존 실패는 복구된 과거 이력이며 현재 추가 실패는 관측되지 않았다. 2차 TCN H96 상태 {'complete': 16, 'running': 8, 'pending': 12, 'failed': 0}; 실행중8개 PID 생존/GPU 사용을 확인했다. TCN H720/PatchTST/TSMixer는 아직 not run. 부분 TCN 집계를 최종 구조 비교로 사용하지 않는다.
- H96에서 no-memory 이질성 MSE 이득 −4.26%였으나 H720은 homo .681632 vs hetero .685652로 +.59%다. H720 hetero dense .706692는 off 대비 +3.07%, sparse .721476는 off 대비 +5.22%/dense 대비 +2.09%. 이번 이질성+검색 결합의 추가 이득은 확인되지 않았다.
- H720 homo sparse .649910(MAE .575184)가6조건 중 최저 macro MSE로 homo off 대비 −4.65%. Dataset별로 ETTh1 .581082→.587875는 악화, ETTh2 .782183→.711944는 개선이므로 일반적 개선으로 해석하지 않는다. 수치는 두 dataset/세 seed 관측이며 통계적 유의성 검정은 not run.
- H720 hetero sparse 진단(first8 test windows): support density38.60%, empty-read9.18%. 같은 checkpoint의 retrieval-off 개입은 MSE −.008923, uniform −.001135, recent +.008103. 검색이 선택적으로 동작하지만 해당 장기예측에서 유용한 보강으로 작용한다는 증거는 부족하다. 개입은 checkpoint 재선택이나 재학습이 아니다.
- H720 best epoch0-based0–29,2/36 budget30 도달; 최저 validation MSE checkpoint 평가 유지. 상세 수치/SD/MAE는 각 suite REPORT.md/per_run.csv/paired_macro_by_seed.csv 및 앞의72회 완료 기록. 본 브리핑은 학습/설정 변경 없이 append했으며 다음 자동 완료 commit에 포함한다.

## 2026-09-15 13:59:24 KST — 요청한 임시 인계 Markdown 작성

- 사용자 요청에 따라 `NSMT/f_lif_pop_v2/forecasting/results/TEMP_EXPERIMENT_HANDOFF_20260915.md` 한 파일을 생성했다. 단일 canonical log를 대체하지 않는 시점별 인계 사본이며 자동 갱신되지 않는다. v1/v2 이력·개념·변경사유·코드지도·실제 핵심3파일 원문·고정프로토콜·184개 기존 완료run 및 TCN 부분완료16개 full-precision metric·상세artifact링크·복구·향후계획·명령·tag/source SHA를 포함한다.
- 관측한 v2 TCN H96 상태: {'complete': 16, 'running': 8, 'pending': 12, 'failed': 0}; 실행중8개 PID 생존 확인. H720 not run. Partial run 결과는 최종 비교와 구분했다. 문서의 모든 local link 및 코드 원문 사본/sourcehash/완료run수를 검사했다. 새 학습/모델수정/추가 성능검증은 not run.
- 문서 SHA256 `894caab712dc831b1a54d360d8e9e131b586a8edeadea7e871762216d54fcdff`. 현재 branch exp/f-lif-pop-v2/HEAD `8c3d13673ce628bff51e7ab858a67b5bc72584da`를 유지했고 기존 사용자 원문/untracked 상태를 보존했다. 진행중queue의 HEAD guard를 유지하기 위해 중간commit을 만들지 않았으며 이 task results와 canonical append는 다음 자동 stage 완료commit에 포함된다.

## 2026-09-15 15:10:28 KST — 임시 인계 문서 접근성/재개 절차 보강

- 지정 파일 `NSMT/f_lif_pop_v2/forecasting/results/TEMP_EXPERIMENT_HANDOFF_20260915.md`를 직접 수정했다. 기존 local 링크250개(고유247개)는 모두 존재했으나 운영 인계 첫 진입점/절대경로/최신상태 동적조회/중단상황별 절차를 보강할 필요가 있었다.0절에 핵심 절대경로25개, 내부목차, 읽기순서, 읽기전용 상태조회, 조건부 detached resume 예제, 대표 checkpoint/log 직접링크6개를 추가했다.
- 원래13:59 결과 스냅샷과 최신 점검상태를 구분했다. 현재 TCN H96 {'complete': 24, 'running': 8, 'pending': 4, 'failed': 0}; H720 not run. 다른서버/Git-only checkout에 local artifact와 untracked 사용자원문이 없을 수 있으며 partial suite/중간epoch resume이 미지원임을 명시했다. 학습code/checkpoint/source/HEAD는 변경하지 않았다.
- 파일링크/내부anchor 존재와 핵심 코드3개 사본 보존, 두 명령의 Python syntax compile을 확인했다. Resume 예제 실행/추가학습/성능 재검증은 not run. 읽기전용 상태조회는 이어서 실제 실행 점검한다. 문서 SHA 변경 전 `894caab712dc831b1a54d360d8e9e131b586a8edeadea7e871762216d54fcdff`, 변경 후 `74c1c0a8eebbf86b805e4a4cc88fef52e0fd6b1fb9bbcb7495c6e250302caa23`. 이전생성기록 SHA는 당시버전 값으로 보존한다.
- 실제 문서에서 읽기전용 Bash/Python 블록을 추출해 실행: passed. HEAD/source 일치, suite별현황,8개 학습 PID의 task·suite 명령 일치와controller 생존을 확인했다. 로컬링크281개/내부anchor링크14개 모두 검증했다. 상태조회·조건부재개 Bash 문법검사도 passed이며 재개블록은 실행하지 않았다.

## 2026-09-15 16:13 KST — v2 진행 비판적 검토: 완료 결과 재분석과 TCN GPU 공유 처리량 측정 (새 학습 없음)

- 목적: 인계 문서와 사용자 feasibility assessment(`NSMT/docs/PopulationLIF_research_feasibility_assessment.md`, untracked 원문 보존)를 검토하며, 완료된 v2 patch 72회와 진행 중 TCN 실행의 해석 근거를 확인한다. Branch `exp/f-lif-pop-v2`, HEAD `8c3d13673ce628bff51e7ab858a67b5bc72584da` 유지. 학습 source/config/queue/checkpoint 변경 없음. Controller PID1392373과 TCN H96 학습은 계속 진행 중이다(16:13 점검 complete35/running1/failed0).
- 수행: (1) patch H96/H720 per_run.csv의 대응 t검정(data×seed n=6, seed macro n=3)과 1% 상대효과를 80% power로 검출하는 데 필요한 쌍 수; (2) dense/sparse checkpoint 48개의 Q/K/gate/null 초기값 대비 이동량; (3) 같은 checkpoint를 CPU에서 전체 test 구간에 고르게 뽑은 96 window로 재실행해 read 분포의 고정 (t,j) kernel 설명력, lag, 학습 scorer와 초기 scorer(Q=K=I, null −1)의 read 차이 측정; (4) 같은 split/scaler/전체 test window에서 closed-form channel-independent ridge 선형 기준(λ는 validation MSE로 선택); (5) 이질성×검색 상호작용; (6) suite별 epoch 시간 비교와, 학습 job이 없던 물리 GPU0/GPU2에서 synthetic batch128 train-step probe(한 GPU 단독 vs 다른 GPU 2개 동시, warmup3+측정8 step, 학습 artifact 없음).
- Commands: cwd `NSMT/f_lif_pop_v2/forecasting/results/review-20260915`에서 `env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib OMP_NUM_THREADS=8 /home/yschoi/.conda/envs/snn_recall/bin/python <name>.py > <name>.txt` (a_stats_drift, b_retrieval_content, b2_learned_vs_init_read, c_ridge_baseline, d_interaction, e_epoch_timing). Probe는 `CUDA_VISIBLE_DEVICES=<idle GPU UUID> OMP_NUM_THREADS=2 python bench_tcn.py --arch tcn|patch --policy off|sparse --tag single|pair_a|pair_b`.
- 통계력: 검색 관련 12개 대응 비교는 모두 p>0.07이다. H96 hetero sparse−dense +1.15%(p=.14), sparse−off +0.84%(p=.38), dense−off −0.29%(p=.75). H720 hetero sparse는 hetero off보다 6쌍 모두 나빴다(+4.76%, p=.078). 1% 상대효과 검출에 필요한 쌍은 H96 7–48, H720 128–2177이며 현재 6쌍이다. Seed SD는 H96 ETTh1 .0043/ETTh2 .0112, H720 .0225/.0859. 견고한 효과는 H96 hetero−homo no-memory −4.12%(6/6, p=.012, seed macro p=.035)뿐이고 H720은 +0.02%(ETTh1 −4.7%, ETTh2 +4.7%, p=.81)다.
- 이질성×검색 상호작용 `(hetero read−off)−(homo read−off)`(음수가 가설 지지): H96 dense −.0016(4/6, p=.65), sparse +.0067(1/6, p=.068, seed macro p=.023); H720 dense −.011(p=.82), sparse +.068(1/6, p=.14). 보정 전 p값이며, heterogeneous key가 검색을 더 유용하게 만든다는 근거는 없고 H96 sparse는 반대 방향이다.
- 기준선 test MSE: ridge linear/last-value 정규화 ridge는 ETTh1 H96 .3702/.3696, H720 .4696/.4325, ETTh2 H96 .3010/.2719, H720 .7404/.3925. v2 patch 6조건의 dataset 평균은 ETTh1 H96 .424–.449, H720 .554–.588, ETTh2 H96 .329–.343, H720 .712–.882. ETTh2 H720은 6조건 모두 window mean(.4319)과 persistence(.5945)보다 나쁘다.
- 검색 경로: ||Q−I||_F, ||K−I||_F 조건 평균 .14–.37(||I||_F=2). Heterogeneous null logit 조건 평균 −1.04~−0.99(초기 −1), gate 평균 .41–.52(초기 .5×real mass), heterogeneous gate weight 평균 .03–.08로 입력 간 거의 균일, evidence/charge 1.3–1.9%. 학습 scorer와 초기 scorer의 read 차이 TV는 dense .10–.20, sparse .28–.45(support Jaccard .58–.71). 학습 read와 uniform read의 TV는 dense .22–.33, sparse .54–.65인데 기존 같은 checkpoint 개입의 uniform ΔMSE는 H96 hetero dense +.0003, sparse −.0007이다. 고정 per-neuron (t,j) kernel이 설명하지 못하는 read 분산이 74–86%라 고정 lag 붕괴는 아니다. lag-1 질량 .12–.20(uniform .083), top-1이 lag1인 비율 21–42%. 첫8 window 진단은 전체 평균과 비슷하나 ETTh1 hetero sparse H96 empty-read는 .104(96 windows) 대 .055(첫8)였다.
- 처리량: epoch 시간은 v2 TCN H96 377–418s(off 포함, GPU당 2 job), 구조가 같은 v1 TCN 33–44s(GPU당 1 job), v2 patch 2–10s. Probe step 시간은 TCN off 단독 .416s 대 공유 4.49/4.55s, TCN sparse .633s 대 4.51/4.84s, patch sparse .083s 대 .135/.134s. Default compute mode, MPS 없음, 공유 중 전력 114–146W/300W. TCN은 GPU당 2 job일 때 총 처리량이 3.7–5.4배 낮다. 원인(CUDA context time-slicing 추정)은 확인하지 않았다.
- 결론: 현재 ETT 행렬은 검색 효과를 판정할 검정력이 없다. 검색 경로는 약하고 거의 일정한 gain으로 쓰이며, 어떤 기억을 읽는지는 MSE에 거의 영향을 주지 않는다. H96 이질성 이득은 재현되지만 homogeneous는 같은 scalar LIF의 4복제이므로 다중 시간척도와 유효 상태 차원이 섞여 있다. 정답 slot을 아는 synthetic recall 검증을 backbone 행렬보다 우선하자는 assessment 방향에 동의하며, 설계에 oracle retrieval, flatten으로 우회할 수 없는 readout, 학습된 uniform/recent, capacity-matched 대조가 필요하다. `--workers_per_gpu 1` 재개와 PatchTST/TSMixer 보류는 사용자 결정 전이므로 실행하지 않았다.
- 한계: p값은 다중비교 보정 전이고 n=6/3이다. Ridge는 SNN과 용량·형식이 다른 참조값이다. Probe는 synthetic 입력의 train step만 쟀다. PatchTST/TSMixer의 GPU 공유 영향, synthetic recall 학습, capacity-matched 대조, memory 강도 공식 비교는 not run.
- Artifact: `NSMT/f_lif_pop_v2/forecasting/results/review-20260915/`(스크립트 7개, `b_retrieval_content.csv`, `b2_learned_vs_init_read.csv`, 출력 `*.txt` 7개). 이 entry와 artifact는 다음 자동 stage commit에 포함된다. 중간 commit은 만들지 않아 HEAD guard를 유지했고, 새 파일은 `git diff --check` 공백 검사를 통과했다.

### 2026-09-15 16:16 KST — 처리량 추가 확인 (실제 TCN 단독 epoch, PatchTST/TSMixer probe)

- 실제 학습 확인: `tcn_ETTh2_p96_flatten_heterogeneous_sparse_seed21`(GPU3)은 같은 GPU의 다른 job이 끝나기 전 epoch 415–419s였고, 혼자 남은 뒤 epoch 60s였다(전환 epoch 158s). 위 probe의 TCN sparse ×7.4와 일치한다.
- 학습 job이 없던 GPU1(단독)/GPU2(2개 동시)에서 같은 `bench_tcn.py --policy sparse` probe를 실행했다. PatchTST 단독 .257s 대 공유 .406/.356s, TSMixer 단독 .255s 대 공유 .398/.405s. 두 구조는 GPU당 2 job일 때 job당 1.4–1.6배 느리지만 총 처리량은 약 1.3배 높다. 따라서 측정 범위에서 GPU 공유 문제는 TCN에 한정되고, 영향받는 남은 suite는 TCN H720 36회다. 앞 entry 한계의 "PatchTST/TSMixer의 GPU 공유 영향 not run"을 이 기록으로 정정한다. 출력은 같은 artifact 폴더의 `bench_tcn.txt`에 추가한다.

## 2026-09-15 20:48 KST — PopulationLIF v2 tcn 완료 (72 runs)

- Branch `exp/f-lif-pop-v2`; base `f8215f54106980bad7c782bf08acaef871175c34`; 각 horizon training commit은 completion.json의 training_commit에 기록한다 (후처리 복구 전후 commit이 다를 수 있으며 학습 source hashes는 동일). 완료 commit은 tag `exp/f-lif-pop-v2-tcn-20260915`로 식별한다. 목적: 동일 점수·gate에서 dense vs sparse 선택 효과 및 population 이질성을 분리한다.
- ETTh1/ETTh2 × seeds7/13/21 × homo/hetero × off/dense/sparse × H96/720. Seq336,patch8,D32,K4,head32/flatten,tau2..16,gamma.05,temperature.25,null init−1. 최대30epochs,early-stop6,ReduceLROnPlateau factor.5/patience2,AdamW lr.001/wd.01,batch128,clip1. 최소 validation MSE checkpoint를 복원해 평가했다.
- 데이터와 환경: 기존 ETT-hour train[0,8640),val[8640,11520),test[11520,14400),train-only StandardScaler,7변수,stride1,context336. H96 windows8209/2785/2785, H7207585/2161/2161. Conda snn_recall/Py3.10/torch1.12.0+cu113,CPUthreads2,deterministic/TF32off. GPU/worker 배치와 모든 실행명령은 각 manifest 및 run JSON, source/data hashes/패키지 버전도 run JSON에 있다.
- 실행 pipeline command: `/home/yschoi/.conda/envs/snn_recall/bin/python /home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/run_pipeline.py --finalize`. Suites: `selective-v2-20260914_tcn_p96`, `selective-v2-20260914_tcn_p720`.

| Horizon | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| 96 | heterogeneous_dense | 0.384109 ± 0.006163 | 0.422137 ± 0.005505 |
| 96 | heterogeneous_no_memory | 0.373195 ± 0.005423 | 0.414063 ± 0.005773 |
| 96 | heterogeneous_sparse | 0.371949 ± 0.003504 | 0.412151 ± 0.000938 |
| 96 | homogeneous_dense | 0.391371 ± 0.015574 | 0.426185 ± 0.010558 |
| 96 | homogeneous_no_memory | 0.389710 ± 0.001491 | 0.425079 ± 0.001710 |
| 96 | homogeneous_sparse | 0.383846 ± 0.003422 | 0.422972 ± 0.004383 |

H96: 0/36 budget cap; 158308 nominal parameters; individual run 1478.4–6735.6s.
- H96 homogeneous: sparse−dense paired macro ΔMSE -0.007525 (seed SD 0.018103); 평균 MSE 기준 sparse가 낮음. 통계적 유의성 주장은 하지 않는다.
- H96 heterogeneous: sparse−dense paired macro ΔMSE -0.012161 (seed SD 0.007074); 평균 MSE 기준 sparse가 낮음. 통계적 유의성 주장은 하지 않는다.
- H96 heterogeneous_dense: final-layer support density 0.994616, empty-read fraction 0.000000, real mass 0.904684 (첫8 test windows).
- H96 heterogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H96 heterogeneous_sparse: final-layer support density 0.339157, empty-read fraction 0.056818, real mass 0.887815 (첫8 test windows).
- H96 homogeneous_dense: final-layer support density 0.997939, empty-read fraction 0.000000, real mass 0.908752 (첫8 test windows).
- H96 homogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H96 homogeneous_sparse: final-layer support density 0.326302, empty-read fraction 0.107284, real mass 0.822386 (첫8 test windows).
| 720 | heterogeneous_dense | 0.840394 ± 0.032524 | 0.642261 ± 0.013403 |
| 720 | heterogeneous_no_memory | 0.809613 ± 0.051103 | 0.627039 ± 0.022168 |
| 720 | heterogeneous_sparse | 0.854352 ± 0.075129 | 0.647211 ± 0.021696 |
| 720 | homogeneous_dense | 0.842025 ± 0.090161 | 0.643018 ± 0.023468 |
| 720 | homogeneous_no_memory | 0.842784 ± 0.083232 | 0.642835 ± 0.024398 |
| 720 | homogeneous_sparse | 0.825472 ± 0.021632 | 0.642080 ± 0.005551 |

H720: 0/36 budget cap; 997588 nominal parameters; individual run 721.3–5134.8s.
- H720 homogeneous: sparse−dense paired macro ΔMSE -0.016553 (seed SD 0.071668); 평균 MSE 기준 sparse가 낮음. 통계적 유의성 주장은 하지 않는다.
- H720 heterogeneous: sparse−dense paired macro ΔMSE +0.013958 (seed SD 0.062132); 평균 MSE 기준 sparse가 높거나 같음. 통계적 유의성 주장은 하지 않는다.
- H720 heterogeneous_dense: final-layer support density 0.994239, empty-read fraction 0.000000, real mass 0.897835 (첫8 test windows).
- H720 heterogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H720 heterogeneous_sparse: final-layer support density 0.358352, empty-read fraction 0.059007, real mass 0.897516 (첫8 test windows).
- H720 homogeneous_dense: final-layer support density 0.995947, empty-read fraction 0.000023, real mass 0.876122 (첫8 test windows).
- H720 homogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H720 homogeneous_sparse: final-layer support density 0.340796, empty-read fraction 0.100215, real mass 0.839178 (첫8 test windows).

각 실행 (best epoch은0-based):

| Run | MSE | MAE | Best epoch | Epochs |
|---|---:|---:|---:|---:|
| tcn_ETTh1_p96_flatten_heterogeneous_dense_seed7 | 0.416954 | 0.440912 | 3 | 10 |
| tcn_ETTh1_p96_flatten_heterogeneous_dense_seed13 | 0.425279 | 0.449004 | 3 | 10 |
| tcn_ETTh1_p96_flatten_heterogeneous_dense_seed21 | 0.429893 | 0.449854 | 6 | 13 |
| tcn_ETTh1_p96_flatten_heterogeneous_no_memory_seed7 | 0.416080 | 0.440229 | 3 | 10 |
| tcn_ETTh1_p96_flatten_heterogeneous_no_memory_seed13 | 0.420261 | 0.442319 | 2 | 9 |
| tcn_ETTh1_p96_flatten_heterogeneous_no_memory_seed21 | 0.426720 | 0.446360 | 2 | 9 |
| tcn_ETTh1_p96_flatten_heterogeneous_sparse_seed7 | 0.411233 | 0.434163 | 2 | 9 |
| tcn_ETTh1_p96_flatten_heterogeneous_sparse_seed13 | 0.424597 | 0.447472 | 2 | 9 |
| tcn_ETTh1_p96_flatten_heterogeneous_sparse_seed21 | 0.416373 | 0.436580 | 3 | 10 |
| tcn_ETTh1_p96_flatten_homogeneous_dense_seed7 | 0.434459 | 0.454745 | 3 | 10 |
| tcn_ETTh1_p96_flatten_homogeneous_dense_seed13 | 0.449309 | 0.458399 | 2 | 9 |
| tcn_ETTh1_p96_flatten_homogeneous_dense_seed21 | 0.436815 | 0.453640 | 8 | 15 |
| tcn_ETTh1_p96_flatten_homogeneous_no_memory_seed7 | 0.427770 | 0.445758 | 2 | 9 |
| tcn_ETTh1_p96_flatten_homogeneous_no_memory_seed13 | 0.442219 | 0.455816 | 4 | 11 |
| tcn_ETTh1_p96_flatten_homogeneous_no_memory_seed21 | 0.448692 | 0.462638 | 2 | 9 |
| tcn_ETTh1_p96_flatten_homogeneous_sparse_seed7 | 0.438836 | 0.458449 | 5 | 12 |
| tcn_ETTh1_p96_flatten_homogeneous_sparse_seed13 | 0.437426 | 0.450464 | 3 | 10 |
| tcn_ETTh1_p96_flatten_homogeneous_sparse_seed21 | 0.437340 | 0.455296 | 7 | 14 |
| tcn_ETTh2_p96_flatten_heterogeneous_dense_seed7 | 0.365462 | 0.416021 | 4 | 11 |
| tcn_ETTh2_p96_flatten_heterogeneous_dense_seed13 | 0.336695 | 0.389940 | 1 | 8 |
| tcn_ETTh2_p96_flatten_heterogeneous_dense_seed21 | 0.330372 | 0.387088 | 1 | 8 |
| tcn_ETTh2_p96_flatten_heterogeneous_no_memory_seed7 | 0.342075 | 0.401107 | 1 | 8 |
| tcn_ETTh2_p96_flatten_heterogeneous_no_memory_seed13 | 0.323966 | 0.380763 | 1 | 8 |
| tcn_ETTh2_p96_flatten_heterogeneous_no_memory_seed21 | 0.310069 | 0.373601 | 0 | 7 |
| tcn_ETTh2_p96_flatten_heterogeneous_sparse_seed7 | 0.333264 | 0.391187 | 1 | 8 |
| tcn_ETTh2_p96_flatten_heterogeneous_sparse_seed13 | 0.312012 | 0.374663 | 1 | 8 |
| tcn_ETTh2_p96_flatten_heterogeneous_sparse_seed21 | 0.334213 | 0.388841 | 1 | 8 |
| tcn_ETTh2_p96_flatten_homogeneous_dense_seed7 | 0.337912 | 0.389332 | 3 | 10 |
| tcn_ETTh2_p96_flatten_homogeneous_dense_seed13 | 0.368443 | 0.417974 | 1 | 8 |
| tcn_ETTh2_p96_flatten_homogeneous_dense_seed21 | 0.321285 | 0.383018 | 1 | 8 |
| tcn_ETTh2_p96_flatten_homogeneous_no_memory_seed7 | 0.350697 | 0.401472 | 4 | 11 |
| tcn_ETTh2_p96_flatten_homogeneous_no_memory_seed13 | 0.340545 | 0.398102 | 1 | 8 |
| tcn_ETTh2_p96_flatten_homogeneous_no_memory_seed21 | 0.328341 | 0.386686 | 5 | 12 |
| tcn_ETTh2_p96_flatten_homogeneous_sparse_seed7 | 0.335909 | 0.396611 | 3 | 10 |
| tcn_ETTh2_p96_flatten_homogeneous_sparse_seed13 | 0.323652 | 0.387110 | 1 | 8 |
| tcn_ETTh2_p96_flatten_homogeneous_sparse_seed21 | 0.329911 | 0.389902 | 1 | 8 |
| tcn_ETTh1_p720_flatten_heterogeneous_dense_seed7 | 0.555349 | 0.537490 | 1 | 8 |
| tcn_ETTh1_p720_flatten_heterogeneous_dense_seed13 | 0.572141 | 0.553762 | 2 | 9 |
| tcn_ETTh1_p720_flatten_heterogeneous_dense_seed21 | 0.567058 | 0.552042 | 1 | 8 |
| tcn_ETTh1_p720_flatten_heterogeneous_no_memory_seed7 | 0.551602 | 0.533579 | 1 | 8 |
| tcn_ETTh1_p720_flatten_heterogeneous_no_memory_seed13 | 0.561480 | 0.545840 | 2 | 9 |
| tcn_ETTh1_p720_flatten_heterogeneous_no_memory_seed21 | 0.561105 | 0.541160 | 1 | 8 |
| tcn_ETTh1_p720_flatten_heterogeneous_sparse_seed7 | 0.590922 | 0.559296 | 2 | 9 |
| tcn_ETTh1_p720_flatten_heterogeneous_sparse_seed13 | 0.575500 | 0.553798 | 2 | 9 |
| tcn_ETTh1_p720_flatten_heterogeneous_sparse_seed21 | 0.569072 | 0.553113 | 1 | 8 |
| tcn_ETTh1_p720_flatten_homogeneous_dense_seed7 | 0.587191 | 0.558381 | 1 | 8 |
| tcn_ETTh1_p720_flatten_homogeneous_dense_seed13 | 0.596574 | 0.571850 | 3 | 10 |
| tcn_ETTh1_p720_flatten_homogeneous_dense_seed21 | 0.549902 | 0.544313 | 3 | 10 |
| tcn_ETTh1_p720_flatten_homogeneous_no_memory_seed7 | 0.599493 | 0.568222 | 5 | 12 |
| tcn_ETTh1_p720_flatten_homogeneous_no_memory_seed13 | 0.588809 | 0.564815 | 3 | 10 |
| tcn_ETTh1_p720_flatten_homogeneous_no_memory_seed21 | 0.567380 | 0.553690 | 3 | 10 |
| tcn_ETTh1_p720_flatten_homogeneous_sparse_seed7 | 0.632927 | 0.584326 | 4 | 11 |
| tcn_ETTh1_p720_flatten_homogeneous_sparse_seed13 | 0.582379 | 0.562501 | 2 | 9 |
| tcn_ETTh1_p720_flatten_homogeneous_sparse_seed21 | 0.582404 | 0.561235 | 6 | 13 |
| tcn_ETTh2_p720_flatten_heterogeneous_dense_seed7 | 1.137724 | 0.739277 | 0 | 7 |
| tcn_ETTh2_p720_flatten_heterogeneous_dense_seed13 | 1.038333 | 0.708684 | 0 | 7 |
| tcn_ETTh2_p720_flatten_heterogeneous_dense_seed21 | 1.171760 | 0.762308 | 0 | 7 |
| tcn_ETTh2_p720_flatten_heterogeneous_no_memory_seed7 | 1.075361 | 0.713220 | 1 | 8 |
| tcn_ETTh2_p720_flatten_heterogeneous_no_memory_seed13 | 0.951893 | 0.667993 | 1 | 8 |
| tcn_ETTh2_p720_flatten_heterogeneous_no_memory_seed21 | 1.156239 | 0.760443 | 0 | 7 |
| tcn_ETTh2_p720_flatten_heterogeneous_sparse_seed7 | 1.273534 | 0.775225 | 1 | 8 |
| tcn_ETTh2_p720_flatten_heterogeneous_sparse_seed13 | 0.989122 | 0.694559 | 0 | 7 |
| tcn_ETTh2_p720_flatten_heterogeneous_sparse_seed21 | 1.127961 | 0.747276 | 0 | 7 |
| tcn_ETTh2_p720_flatten_homogeneous_dense_seed7 | 1.300151 | 0.780559 | 4 | 11 |
| tcn_ETTh2_p720_flatten_homogeneous_dense_seed13 | 0.946844 | 0.677538 | 0 | 7 |
| tcn_ETTh2_p720_flatten_homogeneous_dense_seed21 | 1.071489 | 0.725467 | 0 | 7 |
| tcn_ETTh2_p720_flatten_homogeneous_no_memory_seed7 | 1.278089 | 0.772421 | 4 | 11 |
| tcn_ETTh2_p720_flatten_homogeneous_no_memory_seed13 | 1.008415 | 0.704077 | 0 | 7 |
| tcn_ETTh2_p720_flatten_homogeneous_no_memory_seed21 | 1.014520 | 0.693787 | 2 | 9 |
| tcn_ETTh2_p720_flatten_homogeneous_sparse_seed7 | 1.056306 | 0.707301 | 3 | 10 |
| tcn_ETTh2_p720_flatten_homogeneous_sparse_seed13 | 1.021631 | 0.708901 | 0 | 7 |
| tcn_ETTh2_p720_flatten_homogeneous_sparse_seed21 | 1.077187 | 0.728216 | 0 | 7 |

- 검증 통과: complete matrix, source/초기parameter hashes, minimum val checkpoint/복원, 전체 element수, CSV/history/TensorBoard, finite checkpoint/hash, train scaler/target boundary, 층별 support/null/lag diagnostics, independent macro/paired deltas. 각 horizon sparse ETTh1 seed7의 fresh checkpoint 및 off/uniform/recent 전체 MSE/MAE를 atol1e-12에서 재현했다.
- Artifact는 `NSMT/f_lif_pop_v2/forecasting/results/<suite>/`의 REPORT,per_run/per_task/macro/paired/paired_macro_by_seed,layer_diagnostics,aggregate,manifest/completion/checks 및36raw result JSON. per_run.csv log_path가 task log/<suite>/<dataset>/<date>/<config>/seed+variant의 neorecall CSV/events/logargs/config.pt/best+model.pt를 가리킨다. Raw events/checkpoints/stdout은 local, 텍스트 결과는 Git.
- 해석 제한: dense와 sparse는 score/nullable candidates/gate를 공유하지만 정규화 방식/지원집합/real probability mass가 함께 달라진다. Sparse 사용은 dense search 비용 절감을 보장하지 않는다. 실제 density/empty rate를 함께 보고 판단한다. Homogeneous는 redundant state 대조, backbone 간 용량은 다르다. 진단은 첫8 test windows; 전체 synthetic recall 학습/정답 ETT lag/에너지 측정/통계적 유의성 검정은 not run. 기존 v1과는 scorer/gate/budget이 달라 동일 실험으로 합산하지 않는다. 다음 단계는 성능 개선 여부로 선별하지 않고 실행/검증 통과 뒤 진행한다. Main 통합/push: not run.

## 2026-09-15 21:18 KST — PopulationLIF v2 patchtst 완료 (72 runs)

- Branch `exp/f-lif-pop-v2`; base `f8215f54106980bad7c782bf08acaef871175c34`; 각 horizon training commit은 completion.json의 training_commit에 기록한다 (후처리 복구 전후 commit이 다를 수 있으며 학습 source hashes는 동일). 완료 commit은 tag `exp/f-lif-pop-v2-patchtst-20260915`로 식별한다. 목적: 동일 점수·gate에서 dense vs sparse 선택 효과 및 population 이질성을 분리한다.
- ETTh1/ETTh2 × seeds7/13/21 × homo/hetero × off/dense/sparse × H96/720. Seq336,patch8,D32,K4,head32/flatten,tau2..16,gamma.05,temperature.25,null init−1. 최대30epochs,early-stop6,ReduceLROnPlateau factor.5/patience2,AdamW lr.001/wd.01,batch128,clip1. 최소 validation MSE checkpoint를 복원해 평가했다.
- 데이터와 환경: 기존 ETT-hour train[0,8640),val[8640,11520),test[11520,14400),train-only StandardScaler,7변수,stride1,context336. H96 windows8209/2785/2785, H7207585/2161/2161. Conda snn_recall/Py3.10/torch1.12.0+cu113,CPUthreads2,deterministic/TF32off. GPU/worker 배치와 모든 실행명령은 각 manifest 및 run JSON, source/data hashes/패키지 버전도 run JSON에 있다.
- 실행 pipeline command: `/home/yschoi/.conda/envs/snn_recall/bin/python /home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/run_pipeline.py --finalize`. Suites: `selective-v2-20260914_patchtst_p96`, `selective-v2-20260914_patchtst_p720`.

| Horizon | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| 96 | heterogeneous_dense | 0.375810 ± 0.004424 | 0.413817 ± 0.002602 |
| 96 | heterogeneous_no_memory | 0.376346 ± 0.005147 | 0.414718 ± 0.004728 |
| 96 | heterogeneous_sparse | 0.376121 ± 0.006987 | 0.416054 ± 0.004805 |
| 96 | homogeneous_dense | 0.388525 ± 0.002644 | 0.424769 ± 0.001372 |
| 96 | homogeneous_no_memory | 0.386814 ± 0.001612 | 0.423532 ± 0.001645 |
| 96 | homogeneous_sparse | 0.387346 ± 0.007030 | 0.424883 ± 0.003459 |

H96: 0/36 budget cap; 153700 nominal parameters; individual run 34.4–405.1s.
- H96 homogeneous: sparse−dense paired macro ΔMSE -0.001179 (seed SD 0.005929); 평균 MSE 기준 sparse가 낮음. 통계적 유의성 주장은 하지 않는다.
- H96 heterogeneous: sparse−dense paired macro ΔMSE +0.000311 (seed SD 0.005218); 평균 MSE 기준 sparse가 높거나 같음. 통계적 유의성 주장은 하지 않는다.
- H96 heterogeneous_dense: final-layer support density 0.999670, empty-read fraction 0.000000, real mass 0.929329 (첫8 test windows).
- H96 heterogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H96 heterogeneous_sparse: final-layer support density 0.374973, empty-read fraction 0.042667, real mass 0.906463 (첫8 test windows).
- H96 homogeneous_dense: final-layer support density 0.999485, empty-read fraction 0.000000, real mass 0.908583 (첫8 test windows).
- H96 homogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H96 homogeneous_sparse: final-layer support density 0.410973, empty-read fraction 0.030231, real mass 0.929636 (첫8 test windows).
| 720 | heterogeneous_dense | 0.787864 ± 0.039247 | 0.622330 ± 0.016759 |
| 720 | heterogeneous_no_memory | 0.798103 ± 0.039595 | 0.626114 ± 0.016795 |
| 720 | heterogeneous_sparse | 0.779744 ± 0.038431 | 0.619437 ± 0.016433 |
| 720 | homogeneous_dense | 0.835611 ± 0.067934 | 0.642131 ± 0.022011 |
| 720 | homogeneous_no_memory | 0.848380 ± 0.106832 | 0.644202 ± 0.037498 |
| 720 | homogeneous_sparse | 0.836971 ± 0.076719 | 0.642083 ± 0.024944 |

H720: 0/36 budget cap; 992980 nominal parameters; individual run 30.5–284.2s.
- H720 homogeneous: sparse−dense paired macro ΔMSE +0.001359 (seed SD 0.009610); 평균 MSE 기준 sparse가 높거나 같음. 통계적 유의성 주장은 하지 않는다.
- H720 heterogeneous: sparse−dense paired macro ΔMSE -0.008119 (seed SD 0.004429); 평균 MSE 기준 sparse가 낮음. 통계적 유의성 주장은 하지 않는다.
- H720 heterogeneous_dense: final-layer support density 0.999344, empty-read fraction 0.000000, real mass 0.916438 (첫8 test windows).
- H720 heterogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H720 heterogeneous_sparse: final-layer support density 0.396958, empty-read fraction 0.033394, real mass 0.931737 (첫8 test windows).
- H720 homogeneous_dense: final-layer support density 0.997632, empty-read fraction 0.000000, real mass 0.893939 (첫8 test windows).
- H720 homogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H720 homogeneous_sparse: final-layer support density 0.418903, empty-read fraction 0.054458, real mass 0.919918 (첫8 test windows).

각 실행 (best epoch은0-based):

| Run | MSE | MAE | Best epoch | Epochs |
|---|---:|---:|---:|---:|
| patchtst_ETTh1_p96_flatten_heterogeneous_dense_seed7 | 0.431547 | 0.442888 | 8 | 15 |
| patchtst_ETTh1_p96_flatten_heterogeneous_dense_seed13 | 0.427277 | 0.444072 | 2 | 9 |
| patchtst_ETTh1_p96_flatten_heterogeneous_dense_seed21 | 0.420615 | 0.439924 | 5 | 12 |
| patchtst_ETTh1_p96_flatten_heterogeneous_no_memory_seed7 | 0.426249 | 0.439928 | 8 | 15 |
| patchtst_ETTh1_p96_flatten_heterogeneous_no_memory_seed13 | 0.427592 | 0.445132 | 2 | 9 |
| patchtst_ETTh1_p96_flatten_heterogeneous_no_memory_seed21 | 0.417471 | 0.437489 | 5 | 12 |
| patchtst_ETTh1_p96_flatten_heterogeneous_sparse_seed7 | 0.425966 | 0.444585 | 6 | 13 |
| patchtst_ETTh1_p96_flatten_heterogeneous_sparse_seed13 | 0.427754 | 0.445617 | 2 | 9 |
| patchtst_ETTh1_p96_flatten_heterogeneous_sparse_seed21 | 0.420501 | 0.440199 | 5 | 12 |
| patchtst_ETTh1_p96_flatten_homogeneous_dense_seed7 | 0.424570 | 0.440867 | 5 | 12 |
| patchtst_ETTh1_p96_flatten_homogeneous_dense_seed13 | 0.447126 | 0.455743 | 2 | 9 |
| patchtst_ETTh1_p96_flatten_homogeneous_dense_seed21 | 0.436254 | 0.449011 | 2 | 9 |
| patchtst_ETTh1_p96_flatten_homogeneous_no_memory_seed7 | 0.429160 | 0.447182 | 4 | 11 |
| patchtst_ETTh1_p96_flatten_homogeneous_no_memory_seed13 | 0.449751 | 0.457118 | 2 | 9 |
| patchtst_ETTh1_p96_flatten_homogeneous_no_memory_seed21 | 0.434360 | 0.449519 | 5 | 12 |
| patchtst_ETTh1_p96_flatten_homogeneous_sparse_seed7 | 0.433092 | 0.449894 | 4 | 11 |
| patchtst_ETTh1_p96_flatten_homogeneous_sparse_seed13 | 0.452737 | 0.459723 | 2 | 9 |
| patchtst_ETTh1_p96_flatten_homogeneous_sparse_seed21 | 0.433872 | 0.449917 | 5 | 12 |
| patchtst_ETTh2_p96_flatten_heterogeneous_dense_seed7 | 0.322836 | 0.384847 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_heterogeneous_dense_seed13 | 0.331480 | 0.388715 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_heterogeneous_dense_seed21 | 0.321107 | 0.382456 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_heterogeneous_no_memory_seed7 | 0.331356 | 0.393448 | 7 | 14 |
| patchtst_ETTh2_p96_flatten_heterogeneous_no_memory_seed13 | 0.332019 | 0.391155 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_heterogeneous_no_memory_seed21 | 0.323391 | 0.381159 | 5 | 12 |
| patchtst_ETTh2_p96_flatten_heterogeneous_sparse_seed7 | 0.317591 | 0.381312 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_heterogeneous_sparse_seed13 | 0.340609 | 0.397559 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_heterogeneous_sparse_seed21 | 0.324305 | 0.387052 | 5 | 12 |
| patchtst_ETTh2_p96_flatten_homogeneous_dense_seed7 | 0.346666 | 0.405953 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_homogeneous_dense_seed13 | 0.331219 | 0.393741 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_homogeneous_dense_seed21 | 0.345317 | 0.403297 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_homogeneous_no_memory_seed7 | 0.347673 | 0.403667 | 10 | 17 |
| patchtst_ETTh2_p96_flatten_homogeneous_no_memory_seed13 | 0.323917 | 0.387771 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_homogeneous_no_memory_seed21 | 0.336024 | 0.395936 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_homogeneous_sparse_seed7 | 0.328697 | 0.393611 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_homogeneous_sparse_seed13 | 0.336940 | 0.397471 | 1 | 8 |
| patchtst_ETTh2_p96_flatten_homogeneous_sparse_seed21 | 0.338738 | 0.398683 | 1 | 8 |
| patchtst_ETTh1_p720_flatten_heterogeneous_dense_seed7 | 0.548850 | 0.535492 | 2 | 9 |
| patchtst_ETTh1_p720_flatten_heterogeneous_dense_seed13 | 0.595444 | 0.565496 | 2 | 9 |
| patchtst_ETTh1_p720_flatten_heterogeneous_dense_seed21 | 0.542045 | 0.540850 | 1 | 8 |
| patchtst_ETTh1_p720_flatten_heterogeneous_no_memory_seed7 | 0.559541 | 0.543738 | 2 | 9 |
| patchtst_ETTh1_p720_flatten_heterogeneous_no_memory_seed13 | 0.584578 | 0.559806 | 2 | 9 |
| patchtst_ETTh1_p720_flatten_heterogeneous_no_memory_seed21 | 0.538540 | 0.538865 | 1 | 8 |
| patchtst_ETTh1_p720_flatten_heterogeneous_sparse_seed7 | 0.549001 | 0.535567 | 2 | 9 |
| patchtst_ETTh1_p720_flatten_heterogeneous_sparse_seed13 | 0.592119 | 0.562687 | 2 | 9 |
| patchtst_ETTh1_p720_flatten_heterogeneous_sparse_seed21 | 0.542460 | 0.540203 | 1 | 8 |
| patchtst_ETTh1_p720_flatten_homogeneous_dense_seed7 | 0.601105 | 0.570969 | 4 | 11 |
| patchtst_ETTh1_p720_flatten_homogeneous_dense_seed13 | 0.587542 | 0.567578 | 3 | 10 |
| patchtst_ETTh1_p720_flatten_homogeneous_dense_seed21 | 0.540230 | 0.540593 | 3 | 10 |
| patchtst_ETTh1_p720_flatten_homogeneous_no_memory_seed7 | 0.627486 | 0.581174 | 4 | 11 |
| patchtst_ETTh1_p720_flatten_homogeneous_no_memory_seed13 | 0.550241 | 0.545061 | 2 | 9 |
| patchtst_ETTh1_p720_flatten_homogeneous_no_memory_seed21 | 0.516829 | 0.524409 | 2 | 9 |
| patchtst_ETTh1_p720_flatten_homogeneous_sparse_seed7 | 0.584000 | 0.562825 | 3 | 10 |
| patchtst_ETTh1_p720_flatten_homogeneous_sparse_seed13 | 0.583258 | 0.565682 | 3 | 10 |
| patchtst_ETTh1_p720_flatten_homogeneous_sparse_seed21 | 0.530688 | 0.535120 | 3 | 10 |
| patchtst_ETTh2_p720_flatten_heterogeneous_dense_seed7 | 0.990507 | 0.691544 | 0 | 7 |
| patchtst_ETTh2_p720_flatten_heterogeneous_dense_seed13 | 0.926571 | 0.658134 | 1 | 8 |
| patchtst_ETTh2_p720_flatten_heterogeneous_dense_seed21 | 1.123766 | 0.742464 | 0 | 7 |
| patchtst_ETTh2_p720_flatten_heterogeneous_no_memory_seed7 | 1.003866 | 0.696686 | 0 | 7 |
| patchtst_ETTh2_p720_flatten_heterogeneous_no_memory_seed13 | 0.954107 | 0.666326 | 1 | 8 |
| patchtst_ETTh2_p720_flatten_heterogeneous_no_memory_seed21 | 1.147986 | 0.751261 | 0 | 7 |
| patchtst_ETTh2_p720_flatten_heterogeneous_sparse_seed7 | 0.984195 | 0.689759 | 0 | 7 |
| patchtst_ETTh2_p720_flatten_heterogeneous_sparse_seed13 | 0.907104 | 0.652261 | 1 | 8 |
| patchtst_ETTh2_p720_flatten_heterogeneous_sparse_seed21 | 1.103587 | 0.736145 | 0 | 7 |
| patchtst_ETTh2_p720_flatten_homogeneous_dense_seed7 | 1.221703 | 0.761215 | 1 | 8 |
| patchtst_ETTh2_p720_flatten_homogeneous_dense_seed13 | 0.972870 | 0.678040 | 1 | 8 |
| patchtst_ETTh2_p720_flatten_homogeneous_dense_seed21 | 1.090217 | 0.734391 | 0 | 7 |
| patchtst_ETTh2_p720_flatten_homogeneous_no_memory_seed7 | 1.315899 | 0.793616 | 1 | 8 |
| patchtst_ETTh2_p720_flatten_homogeneous_no_memory_seed13 | 1.017330 | 0.694919 | 1 | 8 |
| patchtst_ETTh2_p720_flatten_homogeneous_no_memory_seed21 | 1.062493 | 0.726031 | 0 | 7 |
| patchtst_ETTh2_p720_flatten_homogeneous_sparse_seed7 | 1.263344 | 0.777131 | 1 | 8 |
| patchtst_ETTh2_p720_flatten_homogeneous_sparse_seed13 | 0.972488 | 0.678160 | 1 | 8 |
| patchtst_ETTh2_p720_flatten_homogeneous_sparse_seed21 | 1.088046 | 0.733579 | 0 | 7 |

- 검증 통과: complete matrix, source/초기parameter hashes, minimum val checkpoint/복원, 전체 element수, CSV/history/TensorBoard, finite checkpoint/hash, train scaler/target boundary, 층별 support/null/lag diagnostics, independent macro/paired deltas. 각 horizon sparse ETTh1 seed7의 fresh checkpoint 및 off/uniform/recent 전체 MSE/MAE를 atol1e-12에서 재현했다.
- Artifact는 `NSMT/f_lif_pop_v2/forecasting/results/<suite>/`의 REPORT,per_run/per_task/macro/paired/paired_macro_by_seed,layer_diagnostics,aggregate,manifest/completion/checks 및36raw result JSON. per_run.csv log_path가 task log/<suite>/<dataset>/<date>/<config>/seed+variant의 neorecall CSV/events/logargs/config.pt/best+model.pt를 가리킨다. Raw events/checkpoints/stdout은 local, 텍스트 결과는 Git.
- 해석 제한: dense와 sparse는 score/nullable candidates/gate를 공유하지만 정규화 방식/지원집합/real probability mass가 함께 달라진다. Sparse 사용은 dense search 비용 절감을 보장하지 않는다. 실제 density/empty rate를 함께 보고 판단한다. Homogeneous는 redundant state 대조, backbone 간 용량은 다르다. 진단은 첫8 test windows; 전체 synthetic recall 학습/정답 ETT lag/에너지 측정/통계적 유의성 검정은 not run. 기존 v1과는 scorer/gate/budget이 달라 동일 실험으로 합산하지 않는다. 다음 단계는 성능 개선 여부로 선별하지 않고 실행/검증 통과 뒤 진행한다. Main 통합/push: not run.

## 2026-09-15 21:45 KST — PopulationLIF v2 tsmixer 완료 (72 runs)

- Branch `exp/f-lif-pop-v2`; base `f8215f54106980bad7c782bf08acaef871175c34`; 각 horizon training commit은 completion.json의 training_commit에 기록한다 (후처리 복구 전후 commit이 다를 수 있으며 학습 source hashes는 동일). 완료 commit은 tag `exp/f-lif-pop-v2-tsmixer-20260915`로 식별한다. 목적: 동일 점수·gate에서 dense vs sparse 선택 효과 및 population 이질성을 분리한다.
- ETTh1/ETTh2 × seeds7/13/21 × homo/hetero × off/dense/sparse × H96/720. Seq336,patch8,D32,K4,head32/flatten,tau2..16,gamma.05,temperature.25,null init−1. 최대30epochs,early-stop6,ReduceLROnPlateau factor.5/patience2,AdamW lr.001/wd.01,batch128,clip1. 최소 validation MSE checkpoint를 복원해 평가했다.
- 데이터와 환경: 기존 ETT-hour train[0,8640),val[8640,11520),test[11520,14400),train-only StandardScaler,7변수,stride1,context336. H96 windows8209/2785/2785, H7207585/2161/2161. Conda snn_recall/Py3.10/torch1.12.0+cu113,CPUthreads2,deterministic/TF32off. GPU/worker 배치와 모든 실행명령은 각 manifest 및 run JSON, source/data hashes/패키지 버전도 run JSON에 있다.
- 실행 pipeline command: `/home/yschoi/.conda/envs/snn_recall/bin/python /home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting/scripts/run_pipeline.py --finalize`. Suites: `selective-v2-20260914_tsmixer_p96`, `selective-v2-20260914_tsmixer_p720`.

| Horizon | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| 96 | heterogeneous_dense | 0.368598 ± 0.002025 | 0.409724 ± 0.000768 |
| 96 | heterogeneous_no_memory | 0.367226 ± 0.003632 | 0.409205 ± 0.001661 |
| 96 | heterogeneous_sparse | 0.368727 ± 0.005457 | 0.409290 ± 0.003321 |
| 96 | homogeneous_dense | 0.391448 ± 0.008942 | 0.428576 ± 0.008141 |
| 96 | homogeneous_no_memory | 0.387066 ± 0.000689 | 0.426148 ± 0.002534 |
| 96 | homogeneous_sparse | 0.391366 ± 0.006573 | 0.429257 ± 0.004431 |

H96: 0/36 budget cap; 151744 nominal parameters; individual run 35.3–412.6s.
- H96 homogeneous: sparse−dense paired macro ΔMSE -0.000082 (seed SD 0.007652); 평균 MSE 기준 sparse가 낮음. 통계적 유의성 주장은 하지 않는다.
- H96 heterogeneous: sparse−dense paired macro ΔMSE +0.000129 (seed SD 0.005175); 평균 MSE 기준 sparse가 높거나 같음. 통계적 유의성 주장은 하지 않는다.
- H96 heterogeneous_dense: final-layer support density 0.999590, empty-read fraction 0.000000, real mass 0.936090 (첫8 test windows).
- H96 heterogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H96 heterogeneous_sparse: final-layer support density 0.373709, empty-read fraction 0.033804, real mass 0.924830 (첫8 test windows).
- H96 homogeneous_dense: final-layer support density 0.999077, empty-read fraction 0.000000, real mass 0.936171 (첫8 test windows).
- H96 homogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H96 homogeneous_sparse: final-layer support density 0.390757, empty-read fraction 0.024619, real mass 0.943729 (첫8 test windows).
| 720 | heterogeneous_dense | 0.757781 ± 0.032110 | 0.607161 ± 0.012592 |
| 720 | heterogeneous_no_memory | 0.781466 ± 0.037816 | 0.614956 ± 0.018120 |
| 720 | heterogeneous_sparse | 0.756560 ± 0.023316 | 0.609909 ± 0.014726 |
| 720 | homogeneous_dense | 0.792932 ± 0.011242 | 0.628939 ± 0.009179 |
| 720 | homogeneous_no_memory | 0.802469 ± 0.037864 | 0.627306 ± 0.012955 |
| 720 | homogeneous_sparse | 0.787126 ± 0.019379 | 0.627744 ± 0.010980 |

H720: 0/36 budget cap; 991024 nominal parameters; individual run 27.8–261.3s.
- H720 homogeneous: sparse−dense paired macro ΔMSE -0.005806 (seed SD 0.008605); 평균 MSE 기준 sparse가 낮음. 통계적 유의성 주장은 하지 않는다.
- H720 heterogeneous: sparse−dense paired macro ΔMSE -0.001220 (seed SD 0.009893); 평균 MSE 기준 sparse가 낮음. 통계적 유의성 주장은 하지 않는다.
- H720 heterogeneous_dense: final-layer support density 0.998060, empty-read fraction 0.000000, real mass 0.904333 (첫8 test windows).
- H720 heterogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H720 heterogeneous_sparse: final-layer support density 0.385492, empty-read fraction 0.039090, real mass 0.927002 (첫8 test windows).
- H720 homogeneous_dense: final-layer support density 0.998450, empty-read fraction 0.000000, real mass 0.916003 (첫8 test windows).
- H720 homogeneous_no_memory: final-layer support density 0.000000, empty-read fraction 1.000000, real mass 0.000000 (첫8 test windows).
- H720 homogeneous_sparse: final-layer support density 0.438660, empty-read fraction 0.034274, real mass 0.944490 (첫8 test windows).

각 실행 (best epoch은0-based):

| Run | MSE | MAE | Best epoch | Epochs |
|---|---:|---:|---:|---:|
| tsmixer_ETTh1_p96_flatten_heterogeneous_dense_seed7 | 0.411471 | 0.432053 | 2 | 9 |
| tsmixer_ETTh1_p96_flatten_heterogeneous_dense_seed13 | 0.420912 | 0.439479 | 2 | 9 |
| tsmixer_ETTh1_p96_flatten_heterogeneous_dense_seed21 | 0.412837 | 0.431113 | 2 | 9 |
| tsmixer_ETTh1_p96_flatten_heterogeneous_no_memory_seed7 | 0.410098 | 0.431584 | 2 | 9 |
| tsmixer_ETTh1_p96_flatten_heterogeneous_no_memory_seed13 | 0.419749 | 0.439383 | 2 | 9 |
| tsmixer_ETTh1_p96_flatten_heterogeneous_no_memory_seed21 | 0.412262 | 0.431003 | 2 | 9 |
| tsmixer_ETTh1_p96_flatten_heterogeneous_sparse_seed7 | 0.411480 | 0.432206 | 2 | 9 |
| tsmixer_ETTh1_p96_flatten_heterogeneous_sparse_seed13 | 0.420607 | 0.438588 | 2 | 9 |
| tsmixer_ETTh1_p96_flatten_heterogeneous_sparse_seed21 | 0.412422 | 0.430664 | 2 | 9 |
| tsmixer_ETTh1_p96_flatten_homogeneous_dense_seed7 | 0.445045 | 0.458645 | 4 | 11 |
| tsmixer_ETTh1_p96_flatten_homogeneous_dense_seed13 | 0.438548 | 0.450145 | 3 | 10 |
| tsmixer_ETTh1_p96_flatten_homogeneous_dense_seed21 | 0.437836 | 0.451793 | 3 | 10 |
| tsmixer_ETTh1_p96_flatten_homogeneous_no_memory_seed7 | 0.441220 | 0.457550 | 7 | 14 |
| tsmixer_ETTh1_p96_flatten_homogeneous_no_memory_seed13 | 0.437576 | 0.449649 | 3 | 10 |
| tsmixer_ETTh1_p96_flatten_homogeneous_no_memory_seed21 | 0.436647 | 0.451446 | 3 | 10 |
| tsmixer_ETTh1_p96_flatten_homogeneous_sparse_seed7 | 0.446455 | 0.460520 | 4 | 11 |
| tsmixer_ETTh1_p96_flatten_homogeneous_sparse_seed13 | 0.436765 | 0.449097 | 3 | 10 |
| tsmixer_ETTh1_p96_flatten_homogeneous_sparse_seed21 | 0.435780 | 0.452469 | 10 | 17 |
| tsmixer_ETTh2_p96_flatten_heterogeneous_dense_seed7 | 0.322021 | 0.385672 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_heterogeneous_dense_seed13 | 0.315664 | 0.380468 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_heterogeneous_dense_seed21 | 0.328684 | 0.389557 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_heterogeneous_no_memory_seed7 | 0.315984 | 0.383188 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_heterogeneous_no_memory_seed13 | 0.319360 | 0.381900 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_heterogeneous_no_memory_seed21 | 0.325904 | 0.388172 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_heterogeneous_sparse_seed7 | 0.315482 | 0.380869 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_heterogeneous_sparse_seed13 | 0.328139 | 0.387368 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_heterogeneous_sparse_seed21 | 0.324233 | 0.386042 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_homogeneous_dense_seed7 | 0.358332 | 0.417305 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_homogeneous_dense_seed13 | 0.336389 | 0.397373 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_homogeneous_dense_seed21 | 0.332538 | 0.396195 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_homogeneous_no_memory_seed7 | 0.334357 | 0.400456 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_homogeneous_no_memory_seed13 | 0.336415 | 0.398686 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_homogeneous_no_memory_seed21 | 0.336183 | 0.399101 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_homogeneous_sparse_seed7 | 0.342935 | 0.405724 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_homogeneous_sparse_seed13 | 0.354453 | 0.411360 | 1 | 8 |
| tsmixer_ETTh2_p96_flatten_homogeneous_sparse_seed21 | 0.331811 | 0.396372 | 1 | 8 |
| tsmixer_ETTh1_p720_flatten_heterogeneous_dense_seed7 | 0.561517 | 0.543782 | 2 | 9 |
| tsmixer_ETTh1_p720_flatten_heterogeneous_dense_seed13 | 0.555390 | 0.541948 | 1 | 8 |
| tsmixer_ETTh1_p720_flatten_heterogeneous_dense_seed21 | 0.534672 | 0.537215 | 1 | 8 |
| tsmixer_ETTh1_p720_flatten_heterogeneous_no_memory_seed7 | 0.550300 | 0.538158 | 2 | 9 |
| tsmixer_ETTh1_p720_flatten_heterogeneous_no_memory_seed13 | 0.548660 | 0.537409 | 1 | 8 |
| tsmixer_ETTh1_p720_flatten_heterogeneous_no_memory_seed21 | 0.534450 | 0.537096 | 1 | 8 |
| tsmixer_ETTh1_p720_flatten_heterogeneous_sparse_seed7 | 0.558528 | 0.542387 | 2 | 9 |
| tsmixer_ETTh1_p720_flatten_heterogeneous_sparse_seed13 | 0.559805 | 0.544556 | 1 | 8 |
| tsmixer_ETTh1_p720_flatten_heterogeneous_sparse_seed21 | 0.534029 | 0.536767 | 1 | 8 |
| tsmixer_ETTh1_p720_flatten_homogeneous_dense_seed7 | 0.584003 | 0.563731 | 2 | 9 |
| tsmixer_ETTh1_p720_flatten_homogeneous_dense_seed13 | 0.585452 | 0.559697 | 2 | 9 |
| tsmixer_ETTh1_p720_flatten_homogeneous_dense_seed21 | 0.549865 | 0.547283 | 1 | 8 |
| tsmixer_ETTh1_p720_flatten_homogeneous_no_memory_seed7 | 0.580690 | 0.562544 | 2 | 9 |
| tsmixer_ETTh1_p720_flatten_homogeneous_no_memory_seed13 | 0.588955 | 0.560072 | 2 | 9 |
| tsmixer_ETTh1_p720_flatten_homogeneous_no_memory_seed21 | 0.528539 | 0.531101 | 2 | 9 |
| tsmixer_ETTh1_p720_flatten_homogeneous_sparse_seed7 | 0.583281 | 0.563900 | 2 | 9 |
| tsmixer_ETTh1_p720_flatten_homogeneous_sparse_seed13 | 0.588833 | 0.561851 | 2 | 9 |
| tsmixer_ETTh1_p720_flatten_homogeneous_sparse_seed21 | 0.548052 | 0.546231 | 1 | 8 |
| tsmixer_ETTh2_p720_flatten_heterogeneous_dense_seed7 | 0.919167 | 0.667085 | 0 | 7 |
| tsmixer_ETTh2_p720_flatten_heterogeneous_dense_seed13 | 0.920936 | 0.649097 | 1 | 8 |
| tsmixer_ETTh2_p720_flatten_heterogeneous_dense_seed21 | 1.055003 | 0.703840 | 1 | 8 |
| tsmixer_ETTh2_p720_flatten_heterogeneous_no_memory_seed7 | 1.071765 | 0.705093 | 1 | 8 |
| tsmixer_ETTh2_p720_flatten_heterogeneous_no_memory_seed13 | 0.929049 | 0.651483 | 1 | 8 |
| tsmixer_ETTh2_p720_flatten_heterogeneous_no_memory_seed21 | 1.054570 | 0.720494 | 0 | 7 |
| tsmixer_ETTh2_p720_flatten_heterogeneous_sparse_seed7 | 0.937457 | 0.673119 | 0 | 7 |
| tsmixer_ETTh2_p720_flatten_heterogeneous_sparse_seed13 | 0.917676 | 0.648203 | 1 | 8 |
| tsmixer_ETTh2_p720_flatten_heterogeneous_sparse_seed21 | 1.031868 | 0.714420 | 0 | 7 |
| tsmixer_ETTh2_p720_flatten_homogeneous_dense_seed7 | 1.010972 | 0.702325 | 0 | 7 |
| tsmixer_ETTh2_p720_flatten_homogeneous_dense_seed13 | 0.974803 | 0.677156 | 1 | 8 |
| tsmixer_ETTh2_p720_flatten_homogeneous_dense_seed21 | 1.052497 | 0.723445 | 0 | 7 |
| tsmixer_ETTh2_p720_flatten_homogeneous_no_memory_seed7 | 1.109666 | 0.718485 | 1 | 8 |
| tsmixer_ETTh2_p720_flatten_homogeneous_no_memory_seed13 | 0.957073 | 0.669170 | 1 | 8 |
| tsmixer_ETTh2_p720_flatten_homogeneous_no_memory_seed21 | 1.049888 | 0.722463 | 0 | 7 |
| tsmixer_ETTh2_p720_flatten_homogeneous_sparse_seed7 | 0.998492 | 0.697959 | 0 | 7 |
| tsmixer_ETTh2_p720_flatten_homogeneous_sparse_seed13 | 0.943451 | 0.669195 | 1 | 8 |
| tsmixer_ETTh2_p720_flatten_homogeneous_sparse_seed21 | 1.060648 | 0.727326 | 0 | 7 |

- 검증 통과: complete matrix, source/초기parameter hashes, minimum val checkpoint/복원, 전체 element수, CSV/history/TensorBoard, finite checkpoint/hash, train scaler/target boundary, 층별 support/null/lag diagnostics, independent macro/paired deltas. 각 horizon sparse ETTh1 seed7의 fresh checkpoint 및 off/uniform/recent 전체 MSE/MAE를 atol1e-12에서 재현했다.
- Artifact는 `NSMT/f_lif_pop_v2/forecasting/results/<suite>/`의 REPORT,per_run/per_task/macro/paired/paired_macro_by_seed,layer_diagnostics,aggregate,manifest/completion/checks 및36raw result JSON. per_run.csv log_path가 task log/<suite>/<dataset>/<date>/<config>/seed+variant의 neorecall CSV/events/logargs/config.pt/best+model.pt를 가리킨다. Raw events/checkpoints/stdout은 local, 텍스트 결과는 Git.
- 해석 제한: dense와 sparse는 score/nullable candidates/gate를 공유하지만 정규화 방식/지원집합/real probability mass가 함께 달라진다. Sparse 사용은 dense search 비용 절감을 보장하지 않는다. 실제 density/empty rate를 함께 보고 판단한다. Homogeneous는 redundant state 대조, backbone 간 용량은 다르다. 진단은 첫8 test windows; 전체 synthetic recall 학습/정답 ETT lag/에너지 측정/통계적 유의성 검정은 not run. 기존 v1과는 scorer/gate/budget이 달라 동일 실험으로 합산하지 않는다. 다음 단계는 성능 개선 여부로 선별하지 않고 실행/검증 통과 뒤 진행한다. Main 통합/push: not run.
