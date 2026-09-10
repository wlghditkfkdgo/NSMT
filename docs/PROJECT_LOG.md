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
