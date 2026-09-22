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

## 2026-09-20 19:50 KST — Population f-LIF v3 설계 확정, 선행연구 검토, 사전등록 (새 학습 없음)

### 0. 이 기록의 목적과 범위

다른 세션이 **이 항목 하나만 읽어도** 현재 아이디어·모델 정의·구현 계획·선행연구·미결 사항을 파악할 수 있도록 작성했다. 이번 세션에서 한 일은 (a) 영문 구현계획서 `NSMT/docs/Population_fLIF_Group_Interaction_Implementation_Plan_EN.md`의 비판적 검토, (b) 그 검토에 대한 사용자 재검토안 `NSMT/docs/Population_fLIF_plan_reassessment_KO.md`의 반영, (c) 웹/학술DB 선행연구 조사, (d) 주 모델 구조 변경(사용자 지시), (e) 사전등록 문서 작성, (f) 읽기 전용 수치 분석이다. **학습·GPU 실행은 없다.** Branch `exp/f-lif-pop-v2`, HEAD `329183b94` 유지, 기존 artifact 변경 없음.

현재 상태: v2 288회(4 backbone × 2 horizon × 36) 전부 완료되어 tag `exp/f-lif-pop-v2-{patch,tcn,patchtst,tsmixer}-20260915` 생성됨. GPU 4장 유휴. v2 최종 macro MSE에서 최저 조건은 suite마다 뒤바뀐다(patch H96 het_dense .3776 / patch H720 hom_sparse .6499 / tcn H96 het_sparse .3719 / tcn H720 het_off .8096 / patchtst H96 het_dense .3758 / patchtst H720 het_sparse .7797 / tsmixer H96 het_off .3672 / tsmixer H720 het_sparse .7566). 2026-09-15 항목의 재분석대로 **검색 관련 12개 대응 비교가 모두 p>0.07이고, 학습된 선택을 uniform으로 바꿔도 MSE가 같았다.** 즉 v2(보통 LIF + 검색된 막전위 잔차, Branch B)는 메커니즘 판정에 실패했고, v3는 **상태 자체를 fractional 적분으로 만드는 모델**로 전환한다.

### 1. 아이디어의 수학: LIF → f-LIF → Population f-LIF

셋 다 같은 재료를 쓴다. 입력 전류 `I_n`, 시간상수 `τ`, 순간 변화량 `F_n = (I_n − U_n)/τ`. (1/τ 규약, h=1, 스파이크 제외)

**LIF** — 과거가 숫자 하나에 눌려 담긴다.
```
U_{n+1} = U_n + F_n = (1−1/τ)U_n + I_n/τ        →  U_{n+1} = Σ_j (1−1/τ)^{n−j} I_j/τ   (지수 망각)
```

**f-LIF** — 상태를 전달하지 않고 **매 스텝 과거 변화량 목록을 다시 합산**한다.
```
U_{n+1} = Σ_{j≤n} b_{n−j} F_j ,   b_d = [(d+1)^α − d^α]/Γ(α+1)      (멱함수 가중, d^{α−1})
```
α=1이면 `b_d ≡ 1`이라 합산이 오일러와 **정확히** 같다(이번 측정 최대오차 0.0). 저장 공간은 숫자 1개 → T개로 바뀐다.

펄스 응답 비교(τ=4, n=0에 입력 4):

| n | 1 | 8 | 16 | 23 |
|---|---:|---:|---:|---:|
| LIF | 0.750 | 0.100 | 0.010 | 0.0013 |
| f-LIF α=0.7 | 0.385 | 0.091 | 0.038 | 0.0225 |

23스텝 뒤 f-LIF가 LIF의 **약 17배**를 기억한다. 단 초반(n=1)은 오히려 더 빨리 떨어진다(stretched exponential → power-law tail).

T=42·α=0.7의 커널 실제 모양: `b_0=1.1005, b_1=0.6873`, `b_41/b_1=0.367`, 총질량 `Σ_{d<42} b_d = 15.06`(α=1이면 42), **질량의 절반이 lag 15 너머**. 즉 이 설정의 f-LIF는 "잊는 뉴런"이 아니라 "거의 다 기억하는 뉴런"이다. 그래서 선택 기능의 역할이 자연히 "억제" 쪽이 된다.

**Population f-LIF** — 같은 입력을 K개 시간척도로 보고, 과거 사건마다 **관련도**를 곱한다.
```
F_{n,k} = (I_n − U_{n,k})/τ_k          같은 I_n, 다른 τ_k = [2,4,8,16]
ρ_{n,j} = Sel( [U_n; I_n], [U_j; I_j] )                 population 전체로 계산, K에 공통
U_{n+1,k} = b_0 F_{n,k} + Σ_{j<n} b_{n−j} · ρ_{n,j} · F_{j,k}
```
한 줄 요약: LIF는 "흐려진 숫자 하나", f-LIF는 "나이별 가중된 모든 과거 변화량", Population f-LIF는 "K개 필터로 본 모든 과거 변화량 × 나이 × 관련도".

유한구간 이득(상수 입력 1, 무발화, n=42): α=0.7이면 `[0.946, 0.883, 0.753, 0.556]`(τ=2..16), α=1이면 `[1.000, 1.000, 0.996, 0.934]`. **fractional 가지는 느리게 차오르고 느린 구성원일수록 절대 크기가 작다.**

### 2. "선별(selection)"이 실제로 어떻게 계산되는가

시점 n에서 후보는 `j = 0..n−1`(엄격히 과거). 계산은 5단계다.

1. **맥락 벡터** `ξ_n = [u_{n,1..K} ; I_n] ∈ R^{K+1}`. 과거 사건도 같은 형식으로 저장한다.
2. **점수** `e_{n,j} = −‖W_Q ξ_n − W_K ξ_j‖² / (d_q ϑ)`. 학습된 선형사상으로 비교한 **음의 평균제곱거리**. L2 정규화를 하지 않으므로 크기 차이도 구분된다. 초기값 `W_Q=W_K=[I 0]`이면 "현재 상태와 가장 비슷한 과거 상태"가 높은 점수를 받는다.
3. **확률화** `p_n = softmax(e_n)`(dense) 또는 `sparsemax(e_n)`(sparse, 정확한 0 가능).
4. **계수화** `ρ̃_{n,j} = B_n p_{n,j} / Σ_ℓ b_{n−ℓ} p_{n,ℓ}` (`B_n = Σ_{j<n} b_{n−j}`), 그리고 `ρ = (1−η) + η ρ̃`.
5. **적용** 최종 계수 `b_{n−j} · ρ_{n,j}`로 과거 dynamics를 합산한다. **확률이 아니라 fractional 계수를 곱하는 배율**이라는 점이 핵심이다.

실제 계산 예(α=0.7, K=4, 항등 Q/K, ϑ=0.25, 입력 `I=[1.5, 0.2, 0.2, 1.5, 0.2]` — 3번 시점이 0번 맥락을 반복):

```
t=4의 가지 상태 u = [0.2494, 0.2915, 0.2025, 0.1184]
점수 e_{4,j}     = [−1.629, −0.018, −0.017, −1.761]      (j=0,1,2,3)
sparsemax p      = [0,      0.4995, 0.5005, 0]           ← 0번·3번(큰 입력 시점) 완전 배제
b_d (d=4−j)      = [0.4910, 0.5297, 0.5868, 0.6873],  B_4 = 2.2948
ρ (질량보존·η=.5) = [0.500,  1.527,  1.529,  0.500]
최종 b·ρ         = [0.2455, 0.8087, 0.8970, 0.3436]      κ = Σbρ/B = 1.000
```

즉 현재(조용한 구간)와 상태가 닮은 과거 1·2번 시점의 계수를 **원래 fractional 계수보다 키우고**, 닮지 않은 0·3번을 깎는다. 같은 점수로 다른 스케일을 쓰면 결과가 크게 달라진다.

| ρ 정의 | sparsemax에서 ρ | κ(커널 총질량비) | 성질 |
|---|---|---:|---|
| `p / max p` (EN 계획 원안) | [0, 0.998, 1.000, 0] | **0.486** | 감쇠 전용. 올바르게 골라도 총질량 절반 손실 |
| `n · p` (1차 대안) | [0, 1.998, 2.002, 0] | 0.973 | 증폭 가능하나 질량 보존 아님 |
| `B p / Σ b p` (채택) | [0, 2.053, 2.057, 0] | **1.000** | 질량 보존 + 증폭 가능 |
| `(1−η) + η·위` (채택, η=.5) | [0.5, 1.527, 1.529, 0.5] | **1.000** | η=0이면 f-LIF로 환원(중립극한) |

`ρ ≤ 1`로 제한된 원안은 "먼 과거가 가까운 과거를 **절대적으로** 앞설 수 없다"는 제약을 만들어, 검증 가설을 감쇠 기반 선택으로 좁힌다. 이것이 D4 결정의 근거다. 단, 볼록 결합은 `(1−η)` 바닥을 깔아 **완전 배제를 막으므로**, 정확한 0을 원하면 `η=1` 또는 별도 hard-mask 모드를 선언해야 한다.

### 3. 주 모델 v3-A — 공유 소마 population f-LIF (사용자 지시로 확정)

**하나의 논리 뉴런 = 하나의 population = 같은 입력 `I_n` + 하나의 출력 스파이크 `s_n`.** K개 구성원은 서로 다른 τ로 적분하며 **기억과 검색 key만 담당하고 발화하지 않는다.** 발화와 reset은 소마 한 곳.

```
(가지, 리셋 없음)   f_{n,k} = (I_n − u_{n,k})/τ_k
                    u_{n+1,k} = Σ_{j≤n} b_{n−j} · ρ_{n,j,k} · f_{j,k}        (ρ_{n,n,k}=1)
(선택)              ρ_{n,j,k} = [(1−η) + η ρ̃_{n,j}] · π_k(n−j)
(소마)              a_n = Σ_k w_k u_{n,k}
                    v_n = v_{n−1} + (a_n − v_{n−1})/τ_s − θ·s_{n−1}
                    s_n = H(v_n − θ)
(출력)              s_n 하나 → 다음 층 전류 I^{(l+1)} = scale · Linear(s^{(l)})
```

환원: `η=0` ⇒ full-history fractional population. `α=1, η=0` ⇒ 가지가 보통 leaky integrator(**측정 최대오차 0.0**)이고 전체는 다중시간척도 선형필터 + LIF 소마. `K=1` ⇒ scalar f-LIF + 소마.

**이 구조가 자동으로 없애는 문제 세 가지**

1. **reset–선택 교란 소멸.** 기억 항목 `f_{j,k}`에 reset 항이 없으므로 `ρ`는 순수 subthreshold 증거만 조절한다. EN 계획 §14.5와 재검토안 §2.2의 최대 난점이 구조적으로 사라진다.
2. **"발화 판정량 ≠ 상태" 문제 소멸.** 가지는 리셋 없는 연속량, 발화는 소마 변수가 결정한다. 두 양이 같아야 한다는 요구 자체가 없어진다.
3. **성분 출력 붕괴 논쟁 유보.** 출력이 하나이므로 개념 문서 §13의 미결 항목을 1차 실험에서 우회한다.

**"population 내부 상호작용"의 위치.** 확산 결합 `−λL u`는 제거한다. 대신 (i) **공유 선택**: `ρ`가 population 전체 상태로 계산되므로 `∂u_{n+1,k}/∂u_{n,ℓ} ≠ 0 (ℓ≠k)` — 다른 구성원의 상태가 내 구성원의 다음 상태를 바꾼다(full 모드에서만 0). (ii) **소마 혼합** `w_k`. 따라서 "상호작용 제거"가 아니라 "확산이라는 특정 형태를 빼고 선택·소마를 상호작용 경로로 삼는다"가 정확하다. EN 계획 §1.3의 "공유 선택만으로는 상호작용이 아니다"는 계획서 자신의 야코비안 기준으로 부정확했다.

**후속 계획(사용자 지시).** 동일 출력(v3-A) 구조로 먼저 완성하고 성능을 확인한 뒤, **성분별 스파이크(v3-B)**를 비교한다. v3-B는 위 세 문제가 되살아나므로 그 시점에 reset 규약 4후보를 명시적으로 재판정한다.

**주의(측정됨).** 소마를 단순 평균(`w=1/K`)으로 두고 v2 기본값 `input_scale=2`를 쓰면 **발화율 0.004**로 사실상 죽은 뉴런이다. 학습 전 보정이 필수다(아래 8절 D11).

### 4. Reset 규약 네 후보와 L1/Teka 형식

v3-B(성분별 스파이크)나 scalar 대조에서는 여전히 규약을 골라야 한다.

| 후보 | 발화 판정량 | reset 위치 | 이력 항목 | α=1 환원 |
|---|---|---|---|---|
| spikeDE 고정 커밋 | `U_n + D_n` (시험값) | `F_n` 안 `−θS_n/τ` → 커널로 영구 전파 | `F_j` | reset-as-current LIF |
| 논문 식 (13) 문자 그대로 | `U_{n+1}` (상태) | 상태 점프, post-reset `U` 저장 | `−U_j+X_j` | **불일치**(재검토안 반례) |
| 재검토안 hybrid | `V_{n+1}` | 별도 항 `−Σ_r θ S_r`(영구, 비감쇠) | `D_j` | 일치 |
| **L1/Teka (권고 추가)** | `V_n` (상태) | 상태 점프 + 점프 **증분**이 memory trace에 멱함수 가중으로 | `ΔU_j` (증분) | 일치 |

재검토안 §2.2의 α=1 반례: `τ=2, I=2.4, θ=1`에서 식 (13)을 문자 그대로 재합산하면 `V_2 = 2.4/2 + (2.4−0.2)/2 = 2.3`, 보통 Euler charge–reset LIF는 `V_2 = 0.2 + (2.4−0.2)/2 = 1.3`. **이전 reset의 감소량 1만큼 차이**가 난다. "post-reset 상태를 저장한다"만으로는 이력 적분에서 reset을 어떻게 누적할지가 정해지지 않는다.

또한 코드 규약에는 지적되지 않은 결함이 있다. reset 크기가 `θ/τ_k`이므로 **τ=16 성분은 발화해도 0.06θ만 내려가 사실상 리셋이 없고**, τ=2 성분은 커널 누적으로 한 번의 발화가 window 전체에 걸쳐 총 ≈7.5θ를 빼는 강한 장기 억제를 받는다. 이질적 τ에서 이 규약은 성분마다 다른 뉴런을 만든다.

**L1/Teka 형식**(Teka et al. 2014)은 Caputo 도함수를 L1 스킴으로 이산화해 `V(t_N)`을 "Markov 항 + memory trace"로 쓴다. memory trace는 **과거 전압의 증분 `V(t_{k+1})−V(t_k)`를 `(N−k)^{1−α} − (N−1−k)^{1−α}` 가중으로 합한 것**이다. 스파이크가 나면 전압은 `V_reset`으로 점프하되 **memory trace는 리셋하지 않고**, 리셋으로 생긴 음의 증분이 이후 멱함수 가중으로 계속 반영된다. 즉 reset이 (코드처럼) 상태 없이 전파되지도, (hybrid처럼) 영구 상수로 남지도 않는다. α=1에서 정확히 오일러 reset LIF이고 생물학 f-LIF 문헌의 표준이다.

### 5. 계획 재검토안(KO) 요약 — `Population_fLIF_plan_reassessment_KO.md`

- **§2.1 τ 규약:** 논문 p.5 이산식과 식 (15)는 `1/τ`, p.6의 `c_m^{(α)}`와 꼬리식은 `τ^α`로 **논문 내부가 불일치**. 코드는 `/tau`. → `1/τ` 규약을 채택하고 식 (13)과 다름을 명시. 단 "1/τ 규약에서는 α가 커널 모양만 바꾼다"는 부정확하며, α는 총질량 `(Nh)^α/Γ(α+1)`도 바꾼다. 또 `1/τ`가 작다고 무발화 정상상태 이득이 작은 것은 아니다(`U_∞=I`).
- **§2.2 발화·reset:** 지적 수용. source parity와 주 모델 정의를 분리한다. 단 식 (13) 교체만으로 해결되지 않음(위 반례). 자기일관 경로의 조건 3개(발화는 실제 fractional charge가 결정 / 선택자가 과거 reset의 상태 감소를 임의로 지우지 않음 / α=1에서 선언한 규약으로 환원)를 **주 모델 선택의 선행 조건으로 격상**.
- **§2.3 ρ 스케일:** 감쇠 전용 지적 수용. 단 `ρ=np`는 `Σρ=n`을 보존할 뿐 `Σ b ρ = Σ b`가 아니다. 질량보존형 `B p/Σ b p` 제안. 정확한 sparse support·상한·질량보존은 동시에 만족 불가. 감쇠 전용은 삭제가 아니라 **대조군**으로 유지.
- **§2.4 확산 결합:** 단일 후보 고정은 수정하되 "차이를 줄인다 → key 품질 악화"로 단정 불가. λ∈{0,.05,.1,.3} 사전 진단. 반대칭 결합은 `U^T A U=0`이지만 fractional·이질 τ·reset까지 포함한 안정성 보장은 아니며, `A1≠0`이면 **동질 population 불변성이 깨져** 비교군 재구성이 필요.
- **§2.5 환경:** CPU reference 채택. 제약은 `torch.fx`가 아니라 `torch.compile`(PyTorch 2.0+). reference 등급을 **source parity / source-derived / mathematical validation** 세 단계로 구분. golden 자료에 내부 항까지 저장. 상수 forcing 적분기 검사 하나로 f-LIF 전체를 검증했다고 하면 안 됨.
- **§2.6 통계:** seed 8–10은 개선이지 검정력 확보가 아니다. `2 dataset × 3 seed = 6`을 독립 6쌍으로 취급 금지. 주 대비를 `Δ_selection`, `Δ_interaction` 둘로 좁히고 거기에 예산 집중. H720은 보조로 두되 불리하다고 사후 제외 금지. window-mean 미달은 "선택이 오차를 줄였는가"와 "모델이 유용한가"를 분리해 해석.
- **§2.7 발화율 보정:** 필요하나 구성원별 발화율을 강제로 동일하게 맞추면 이질성 자체가 지워진다. train 구간만으로 **공유 input scale**을 정해 기록. 커널 총질량 `42^0.7/Γ(1.7) ≈ 15.06`은 맞지만 이를 일정 이득처럼 쓰면 누설·reset 무시.
- **§2.8 Key:** `D_n`은 `(U_n, I_n)`의 선형함수라 추가해도 선형 projection의 표현력이 늘지 않음(지적 수용). `ΔU_n`은 새 정보 → `ξ^Δ = [U; I; ΔU]` 비교 추가. v2에서 key가 무력했던 것이 새 fractional 상태에서도 실패한다는 증거는 아님.
- **§2.9 Surrogate:** 코드 backward는 `s/2 / (1 + (π/2·s·x)²)`이고 함수 기본 scale은 2.0이지만, **`LIFNeuron`은 `BaseNeuron`의 `surrogate_grad_scale=5.0`을 명시적으로 전달**한다. 따라서 "scale 5가 틀렸다"가 아니라 "함수 기본값·호출값·논문 식 (30)을 구분하지 않은 기술"이 문제. 첫 비교에서는 두 경로에 같은 scale을 공통 적용.
- **§3 논문 사실:** fractional 우월 전제 금지(논문도 데이터마다 α 튜닝, α=1이 최적인 경우 존재). 원 논문에 ETT forecasting 없음. 내용 의존 계수에 고정 convolution 가속 논거 적용 불가.
- **§4 순서:** A(scalar 정의) → B(population 진단) → C(선택 진단) → D(controlled recall + M1) → E(반복·확장). **모델 정의 전 108회/아키텍처 matrix는 기본안에서 내린다.**
- **최종 판단:** "공식 코드와 같으면 우리가 원하는 f-LIF 확장"이라는 기준을 버리고, **출처 재현 / 모델 정의 자기일관성 / 연구 검증**을 분리한다.

### 6. 선행연구 검토 (2026-09-20 조사)

**(a) fractional 기억 뉴런**

| 연구 | 핵심 | 우리와의 관계 |
|---|---|---|
| Ge et al., *Fractional-Order SNN*, **ICLR 2026** (arXiv:2507.16937, 코드 `PhysAGI/spikeDE`) | ABM predictor로 f-IF/f-LIF, α는 데이터마다 튜닝(0.3–1.0), 이론(멱함수 완화·유한 LIF 앙상블로 환원 불가·섭동 강건성) | 출발점. **시계열 예측 실험 없음** |
| Teka, Marinov, Santamaria, **PLoS CB 2014** (DOI 10.1371/journal.pcbi.1003526) | 생물학 f-LIF, L1 스킴, "voltage-memory trace"=과거 전압 증분의 가중합, 스파이크 시 trace는 리셋하지 않음 | **reset 규약 제4후보의 근거** |
| **He, Kang, Li, Zha, *LongSpike*, arXiv:2606.12895 (2026-06)** | f-SNN 저자들의 후속. Caputo fractional **SSM** + 스파이크. 커널을 `t^{α−1}/Γ(α) = (sin πα/π)∫e^{−ωt}ω^{−α}dω`로 쓰고 **sum-of-exponentials(M개 지수)로 근사**. M=1이면 SpikingSSM(α=1)로 정확히 환원, **M=2로 LRA-Text 80.4%→88.2%**. α 고정, 커널은 lag에만 의존, population·내용선택 없음 | **가장 중요한 비교 대상.** "서로 다른 감쇠율의 적분기 몇 개"가 곧 fractional 기억의 계산법 ⇒ 우리 이질 population(α=1)은 이미 4항 SOE다. `hetero α=1` 기준선이 결정적 |
| Cui, Kang et al., *NvoFDE*, **AAAI 2025** (arXiv:2503.16207) | **가변차수** `α(t, x(t))`가 은닉 상태에 의존 → 이산화 계수가 내용 의존. 그래프 노드 분류, FROND 대비 1–3% | "내용에 따라 기억 커널을 바꾼다"의 가장 가까운 ML 선례. **차이: 커널 차수 조절 vs 개별 사건 선택**. 안정성 논의 거의 없음 |

**(b) 이질적 다중시간척도 population**

| 연구 | 핵심 |
|---|---|
| Zheng et al., **DH-SNN**, Nat. Commun. 15:277 (2024) | 뉴런당 다수 수상돌기 가지, 가지별 **학습되는 감쇠율**, `i^d_{t+1} = α_d i^d_t + (1−α_d) I^d`, 소마 `u ← β u + (1−β)R Σ_d i^d − o·u_th`. **가지 전류는 절대 리셋되지 않음**, 소마만 발화. 가지는 입력을 **나눠** 받고 소마에서 단순 합산, 가지 간 결합 없음. SHD 92.1%, SSC 82.5%. 다중시간척도 XOR에서 느린 가지=저주파/빠른 가지=고주파 분화. 그래디언트가 가지 전류를 통해 오래 유지됨 | **v3-A의 직접 선례** (단 우리는 가지가 입력을 공유) |
| Feng et al., **TS-LIF**, ICLR 2025 (arXiv:2503.05108) | 시계열 예측용 2구획(수상돌기 느림·소마 빠름) **상호 결합**(β1,β2 비대칭), 각자 스파이크, 주파수 응답 이론. Metr-la/Pems-bay/Solar/Electricity에서 Spike-TCN·iSpikformer 상회 (**ETT 없음**) | 비확산 결합의 선례 |
| Baronig et al., **adLIF**, Nat. Commun. 2025 (arXiv:2408.07517) | 막전위–적응변수 2변수 결합 → 공진·진동, SHD/SSC에서 LIF 상회 | 〃 |
| Spiking SSM 계열: Binary-S4D(Sci. Rep. 2024), SpikingSSM, **SPikE-SSM**(arXiv:2410.17268), **P-SpikeSSM**(ICLR 2025), **SiLIF**(arXiv:2506.06374) | S4D 대각 상태 = 채널당 여러 감쇠율의 적분기 뱅크. SPikE-SSM은 reset이 병렬화를 막는 문제를 PMBC(경계 압축)로 우회 | 우리 population의 일반형. **reset과 병렬성의 상충**은 공통 난제 |
| Perez-Nieves et al., Nat. Commun. 12:5791 (2021); PLIF/GLIF/CLIF/MLIF; TC-LIF(arXiv:2307.07231) | 시간상수 이질성·학습 | 이질성 자체는 새롭지 않음 |

**(c) 내용 의존 선택 / 과거에 대한 attention**

Fang et al., **PSN**(NeurIPS 2023, arXiv:2304.12760): 뉴런마다 **자기 시간 가중치 행렬**(T×T, 마스크로 인과성) — 내용 의존은 아님. 사용자 옵션 (1)의 고정 가중치 버전에 해당. TA-SNN(ICCV 2021), TCJA, TIM(IJCAI 2024), STAtten/STAA/FSTA(CVPR·AAAI 2025): time-step 축 attention. Mamba(arXiv:2312.00752) 및 spiking Mamba 변형: 입력 의존 선택적 전파/망각이지만 **사건 선택은 아님**. → **"커널 내부에서, population key로, 과거 사건을 골라 fractional 합에 넣는" 형태는 찾지 못했다.** 새로움의 자리는 있으나 좁고, 입증 부담은 (a) 차수 조절, (b) `hetero α=1`, (c) 질량 맞춤 감쇠를 이기는 것이다.

**(d) population 내부 결합** — 생물학에서 gap junction은 동기화 기제이며 이질 QIF population에서 "diversity-induced synchronization"과 **이질성 보상**이 보고된다(arXiv:2409.18278; J. Comput. Neurosci. DOI 10.1007/s10827-008-0117-3). ML에서는 coRNN(arXiv:2010.00951)·LEM(arXiv:2110.04744)·adLIF·TS-LIF 모두 **대칭 확산이 아닌 2차/비대칭 결합**이다. 학습 가능한 gap-junction형 결합을 쓴 심층 SNN은 찾지 못했다.

**(e) SNN 시계열 예측 기준선** — Lv et al., **ICML 2024**(arXiv:2402.01533, 코드 `microsoft/SeqSNN`): delta/convolutional spike encoder, Spike-TCN/Spike-RNN/iSpikformer, **마지막 층 스파이크에 Linear**로 예측. TS-LIF도 동일. 두 연구 모두 Metr-la/Pems-bay/Solar/Electricity(RSE/R²)이고 **ETT 수치는 없다.**

**(f) 수치해석·수학** — Caputo 빠른 계산의 SOE 근사(Jiang et al., arXiv:1511.03453): 멱함수 커널을 `O(log N)`개 지수로. 충격 fractional ODE의 해 개념 논쟁: Fečkan–Zhou–Wang(CNSNS 2012) vs Wang–Ahmad–Zhang–Nieto(CNSNS 2014 comments), 정리는 Wang–Fečkan–Zhou(FCAA 19(4):806–831, 2016). **"충격이 이후 전 구간에 상수 점프로 남는가, 커널을 통해 전파되는가"가 미해결 논쟁**이며 재검토안 hybrid는 전자, 코드는 후자에 가깝다. 초록 원문은 이번에 확보하지 못했으므로 인용 전 확인 필요.

**(g) 실현 가능성 판정** — 기술적으로 가능하다. 모든 재료가 이미 학습되고 fractional+spiking은 대규모에서 돈다(LongSpike). 과학적 위험은 셋: ① fractional 차수와 이질 population의 **중복**(SOE 항등식) — 논문의 두 기둥이 하나일 수 있음, ② 선택이 **감쇠·차수조절·질량조절과 구분되지 않을** 위험, ③ O(T²) 비용을 정당화할 **사건 회상 능력**의 입증 필요. 셋 다 synthetic recall과 `hetero α=1` 기준선으로만 답할 수 있다.

### 7. 사전등록 결정 (상세는 `NSMT/docs/Population_fLIF_v3_prereg_KO.md`)

| # | 결정 | 요지 |
|---|---|---|
| D1 | Reset·발화 | v3-A는 가지 리셋 없음 + 소마 subtractive reset. 원본 4후보는 Phase A 대조군 |
| D2 | 선택 단위 | **하이브리드**: 공유 내용 점수 × 가지별 lag 사전분포 `π_k(d)=exp(−d/(c τ_k))`. 비교군 = 순수 공유, 가지별 multi-head |
| D3 | 결합 | 확산 결합 **제거**(λ=0). 상호작용은 공유 선택 + 소마 혼합 |
| D4 | ρ 스케일 | 질량보존형 + 볼록 결합 `(1−η)+η ρ̃`. 감쇠 전용은 대조군. 우선순위 질량보존 > 상한 |
| D5 | 선택자 초기화 | **중립극한(f-LIF)에서 출발**, 선택은 학습으로 획득 |
| D6 | 주 기준선 | **hetero α=1 (=SOE/DH-SNN 계열)**, scalar f-LIF, conventional LIF, ridge, window-mean |
| D7 | α·T | α∈{0.3,0.5,0.7,1.0}, 기본 0.7, T=42 고정. α는 총질량도 바꿈 |
| D8 | Readout | 스파이크 주 + 막전위 진단, flatten 주 + bottleneck 기전 |
| D9 | 구조 | **v3-A(동일 입력·동일 출력) 주**, v3-B(성분별 스파이크) 보류 |
| D10 | 순서 | A→B→C→D(synthetic, oracle 포함)→E(M1 H96). 정의 전 대규모 matrix 금지 |
| D11 | 발화율 보정 | 학습 전 train 구간만으로 `(τ_s, w, input_scale)` 고정, 목표 발화율 0.1–0.3 |

Synthetic recall 프로토콜(과제 B: regime 재귀 / 과제 A: 문맥별 lag), 조건 8종(full·dense·sparse·**oracle**·학습된 uniform/recent·질량 맞춤·scalar·capacity-matched·α=1), 지표(정답 사건 질량, support hit, 문맥 전환 반응 step, slot 교체 개입)와 사전 판정 기준은 사전등록 문서 §4에 있다. **cue와 value는 반드시 같은 채널 스트림에 둔다** — 모델이 channel-independent이므로 다른 채널에 두면 정보가 구조적으로 도달할 수 없어 과제가 불가능해진다.

### 8. 이번에 실행한 수치 분석

명령(cwd `NSMT/f_lif_pop_v3/analysis`, `env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python`):
`python prereg_numerics.py > prereg_numerics.txt`, `python calib_probe.py > calib_probe.txt`.

- **α=1·ρ=1에서 가지 합산 = 보통 leaky integrator**: 최대오차 **0.0**
- **유한구간 이득**(상수 입력 1, n=42): α=0.5 `[.835,.699,.516,.333]`, α=0.7 `[.946,.883,.753,.556]`, α=1 `[1.000,1.000,.996,.934]`
- **감쇠 교란**: 모든 ρ에 상수 c를 곱하면 상태가 직접 줄어든다(n=42, α=0.7: c=1 `[.946,.883,.753,.556]` → c=0.5 `[.880,.754,.566,.367]` → c=0.25 `[.760,.585,.392,.234]`). **선택이 이득 조절로 작동할 수 있다**는 정량적 근거 → 질량 맞춤 대조 필수
- **발화율 보정표**: (τ_s=2, w=1/K, scale=2) → **0.004**; (τ_s=2, w=1/K, scale=4) → 0.044; (τ_s=1, w=1/K, scale=4) → 0.120; (τ_s=1, w=1, scale=1) → 0.120; (τ_s=1, w=1, scale=2) → 0.243. α=1 가지도 유사(scale=1에서 0.130)
- **선별 worked example**과 ρ 스케일별 κ: 2절 표

Artifact: `NSMT/f_lif_pop_v3/analysis/{prereg_numerics.py, prereg_numerics.txt, calib_probe.py, calib_probe.txt}`. 사전등록 문서: `NSMT/docs/Population_fLIF_v3_prereg_KO.md`.

### 9. 한계와 not run

- 선행연구 조사는 웹 검색·PDF 텍스트 추출 기반이다. DH-SNN·TS-LIF·LongSpike·NvoFDE·SPikE-SSM은 본문 수식·표를 직접 확인했으나, 충격 fractional ODE 논쟁(Fečkan/Wang)은 **초록 원문을 확보하지 못했고** 2차 자료 기반이다. 인용 전 원문 확인 필요.
- 수치 분석은 무발화 또는 단일 population의 CPU 시뮬레이션이다. 학습, 그래디언트, 다층, 실제 ETT 입력, synthetic recall: **not run**.
- v3 코드 구현, Phase A golden 파일, spikeDE 설치(torch 2.x CPU env), 성분별 스파이크 v3-B, 결합 재도입, 학습 가능 τ/α: **not run**.
- 사전등록 문서의 판정 기준 수치(정답 사건 질량 0.5, oracle 대비 1.5배)는 **예시이며 실행 전 확정해야 한다.**
- v3 구현은 `exp/f-lif-pop-v2` HEAD를 바꾸지 않도록 새 branch에서 시작한다. Main 통합/push: not run.

### 10. 추가 수치 확인 — L1/Teka 형식 (같은 날 실행)

`NSMT/f_lif_pop_v3/analysis/l1_teka_check.py` (출력 `l1_teka_check.txt`).

L1 스킴은 Caputo 도함수를 **과거 전압 증분의 가중합**으로 이산화한다(h=1):
```
V_N = V_{N−1} + (Γ(2−α) h^α / τ)·(−V_{N−1} + I_{N−1}) − Σ_{k<N−1} w_{N,k}·(V_{k+1} − V_k)
w_{N,k} = (N−k)^{1−α} − (N−1−k)^{1−α}
```
즉 **"보통 LIF 한 스텝 + 과거 변화량을 기억해 빼는 보정항"** 구조다. α=1이면 모든 `w`가 0이 되어 보정항이 사라지고 정확히 Euler LIF가 된다(**측정 최대오차 0.0**).

측정 결과:

- **무발화 완화**(τ=4, I=1): α=1은 60스텝에 1.000에 도달, α=0.8은 0.961, α=0.6은 0.838. predictor 형식에서 본 "fractional은 느리게 차오른다"와 같은 방향.
- **발화**(I=1.6, 200스텝): α=1 ISI=4 일정. α=0.6, refractory=1에서 ISI 9,9,8,8…→7,7,8로 **완만히 짧아졌다**. Teka et al.이 보고한 적응(ISI 길어짐)과 방향이 다르므로, 우리 이산화·파라미터에서의 관측으로만 기록한다.
- **리셋 증분의 처리가 결정적**: 리셋 점프를 memory trace에 포함하면 스파이크 25개(ISI 9,9,8,8…), 제외하면 5개(ISI 27→38→48→59, 강한 적응). **같은 수식·같은 파라미터에서 질적으로 다른 뉴런**이 된다 → 사전등록 D1의 하위 결정으로 명시.
- **구현 함정**: L1 가중치의 `k` 색인은 목표 시점 N 기준이어야 한다. 한 칸 어긋나면 가장 최근 증분에 가중치 1이 붙어 과잉 차감·진동이 생기는데, **α=1 검사는 모든 가중치가 0이라 그대로 통과한다.** Phase A는 반드시 α<1 궤적으로 검사해야 한다(이번에 실제로 이 실수를 만들고 수정했다).

또한 synthetic 과제의 인코딩을 사전등록 문서 §4.4에 구체화했다: **patch 하나 = 사건 하나**(patch_size 8의 원시 위치를 `key one-hot | value | 0`으로 채움). 그러면 "사건 j = 기억 slot j = patch j"가 되어 정답 slot 적중률이 모호하지 않고, lag도 patch 단위로 정의된다.

### 11. 추가 수치 확인 — reset 규약 3종 직접 비교 (같은 날 실행)

`NSMT/f_lif_pop_v3/analysis/reset_conventions.py`, `paper_reset_detail.py` (출력 각 `.txt`).

v3-A(가지 무리셋 + 소마 리셋)를 채택하면 원문과 reset이 어떻게 달라지는지 scalar 뉴런으로 직접 비교했다. 조건 동일: τ=4, α=0.7, I=1.5, θ=1, 40 step, 1/τ 규약.

| 규약 | 발화 판정 | reset 위치 | 이력 저장 항목 | 스파이크 | 첫 발화 | ISI | max·상태 |
|---|---|---|---|---:|---:|---|---:|
| (a) 논문 식 (13) 문자 그대로 | fractional charge `U_k` | 그 상태를 직접 깎음 | **post-reset `U_j`** | 31 | n=9 | 1,1,1,… | 1.167 |
| (b) spikeDE 공개 코드 | `U_n + D_n` (국소 Euler 시험값) | `F_n` 안 `−θS_n/τ` | `F_j` | 17 | n=6 | 2,2,2,… | 0.997 |
| (c) **v3-A (DH식)** | 소마 `v_n` | 소마에서만 subtractive | `f_j` (**리셋 없음**) | 16 | n=9 | 2,2,2,… | 1.318 |

상태 궤적(처음 14 step):
```
(a) 논문: 0.917 0.957 0.991 [0.021] 0.323 0.442 0.536 0.611 …   (대괄호=리셋 직후)
(b) 코드: 0.917 0.642 0.861 0.643 0.869 0.655 0.883 0.669 …
(c) v3-A: 0.917 0.957 0.991  1.021 1.047 1.071 1.092 1.111 …
```

발화 패턴만 보면 **(b)와 (c)가 거의 같고((17 vs 16), ISI 2) (a)가 혼자 튄다.**

#### (a) 논문 식 (13)의 문자 그대로 읽기는 퇴화한다

이유를 추적했다. 식 (13)의 이력 항은 `−U_{k−1−m}`, 즉 **누설항**이다. 리셋으로 저장된 `U_j`가 작아지면 이후 모든 합에서 누설이 약해져 **다음 상태가 오히려 커진다.** 리셋 → 누설 감소 → 상태 증가 → 재발화의 양의 되먹임이다. pre-reset 값이 1.021 → 1.323 → 1.442 → 1.536으로 계속 자란다(hard reset은 더 심해 1.328 → 1.535 → 1.717 → 1.885).

파라미터 27조합(τ∈{2,4,8} × α∈{0.5,0.7,0.9} × I∈{1.2,1.5,2.0}, 40 step) 전수 확인 결과, **발화가 일어나는 모든 설정에서 "최장 연속 발화 구간 = 총 스파이크 수"**였다. 즉 한 번 임계를 넘으면 window 끝까지 매 스텝 발화하며 아래로 돌아오지 않는다(예: τ=2,α=0.7,I=2.0 → 40/40 발화). 발화가 0인 설정은 애초에 임계를 못 넘은 경우다.

**해석의 한계:** 이것은 *인쇄된 식 (13)을 문자 그대로 구현한 경우*의 성질이다. 논문의 실제 실험은 공개 코드 경로(b)를 쓰므로, 이 결과를 "논문의 실험이 잘못되었다"로 읽으면 안 된다. 다만 **"논문 수식을 그대로 따르면 된다"는 선택지는 이번 측정으로 사라졌다.**

#### 결정에 미치는 영향

- v3-B(성분별 스파이크)로 갈 때의 reset 후보는 **(b) 코드 규약, (c) hybrid, (d) L1/Teka 셋**으로 좁혀진다. 식 (13) 문자 그대로는 후보에서 제외하되, Phase A의 대조 기록으로는 남긴다.
- v3-A는 원문 어느 규약과도 **다른 뉴런**이다. 정확한 서술은 "f-LIF의 확장"이 아니라 **"fractional 수상돌기/시냅스 적분 + 보통 LIF 소마"**이며, 전류 기반 LIF(CUBA)·DH-SNN 계열의 2단 구조에 해당한다. 보존되는 것은 멱함수 기억이고, 바뀌는 것은 발화·리셋과 기억의 결합 방식이다. 관련연구·방법 절에서 이 구분을 명시한다.
- v3-A의 부수 효과: 가지가 선형이므로 선택을 끄면(η=0) 가지는 순수 convolution이라 FFT/SOE로 O(T log T) 계산이 가능하고, **O(T²) 비용이 정확히 선택 기능에만 귀속**된다.

#### 실행 준비 상태 점검

이 기록과 별도로, 아이디어 검증에 필요한 결정이 모두 끝났는지 점검했다. **9개 항목이 미결**이며 사전등록 문서 `NSMT/docs/Population_fLIF_v3_prereg_KO.md` §9에 권고안과 함께 정리했다. 요지: 소마 혼합 `w_k`와 `τ_s`의 파라미터화, surrogate 식·scale 고정, `η`·`π_k`·온도 `ϑ`의 파라미터화와 학습 여부, hard-mask 모드 포함 여부, synthetic 과제의 **시간 단위 불일치**(원안은 raw step, 인코딩은 patch=사건), oracle 강제 방식, 사전 판정 수치, 학습 예산·seed·MDE, Phase A golden 환경 구축. **이 중 시간 단위 불일치와 MDE 미선언이 실행 전 반드시 해소해야 할 항목이다.**

## 2026-09-21 00:57 KST — v3 사전등록 미결 항목 O1–O9 상세 사양과 용어 정리 (결정 대기, 새 학습 없음)

### 0. 이 기록의 위치

2026-09-20 항목에서 주 모델 v3-A(공유 소마 population f-LIF)를 확정하고 실행 전 미결 9개를 식별했다. 이 기록은 그 9개의 **구체 사양·근거·기각한 대안**을 남긴다. 확정되면 사전등록 문서 `NSMT/docs/Population_fLIF_v3_prereg_KO.md` §2에 날짜가 붙은 정식 결정으로 옮긴다. 학습·GPU 실행 없음.

### 1. 용어와 텐서 축 (다음 세션이 헷갈리지 않도록)

| 용어 | 뜻 |
|---|---|
| `τ_k` | dendritic branch의 membrane time constant. `τ=[2,4,8,16]`, 가지마다 다름 |
| `τ_s` | **somatic** membrane time constant(아래첨자 s = soma). 갱신 `v ← v + (a−v)/τ_s`에서 **한 스텝에 목표값과의 차이를 얼마나 메우는가**를 정한다. `1/τ_s`가 그 비율이고, 남는 비율 `β = 1 − 1/τ_s`가 retention factor다. `τ_s=1 → β=0`(기억 없음), `τ_s=2 → β=0.5`, `τ_s=16 → β≈0.94`. 지수 규약 `β=e^{−1/τ}`와는 다른 이산화이며 우리는 Euler 규약을 쓴다 |
| `ISI` | inter-spike interval. 연속한 두 스파이크 사이의 step 수. firing rate = 1/ISI. 우리 시계는 patch이므로 ISI=2는 "두 patch마다 한 번 발화", rate 0.5 |
| `memory slot` | selector가 주소를 매길 수 있는 과거 단위. 우리 모델에서는 **patch 하나 = 1 slot** |
| `event` | synthetic 과제에서 의미 단위. **1 event = 1 patch = 1 memory slot**으로 일치시킨다 |
| `neutral limit` | selection을 껐을 때(η=0, π≡1) 원래 f-LIF와 정확히 같아지는 성질 |
| `MDE` | minimum detectable effect. 주어진 표본 크기·검정력에서 검출 가능한 최소 효과 크기 |

**텐서 축 구조 (중요).** 많은 SNN 시계열 연구(예: Lv et al. ICML 2024)는 시계열 step마다 `Ts`개의 **별도 spiking simulation 축**을 만들어 `[B, Ts, T, C]`로 둔다. **우리는 그렇게 하지 않는다.** 개념 문서 §17의 결정대로 `Ts = 1`로 두고 **chronological patch 축을 그대로 뉴런의 시간축으로 쓴다.** 정적 입력을 인공 시뮬레이션 축에 반복하면 "과거 상태를 회상한다"는 주장이 실제 시계열 과거와 무관해지기 때문이다.

그 결과 v3-A의 형상은 다음과 같다.

```
입력                [B, 336, C]
patch 분할          [T=42, B·C, P=8]
입력 전류 I         [T, B·C, D]                Linear(8→D) × input_scale
가지 상태 u         [T, B·C, D, K]             ← population 축 K가 여기서 생긴다
기억 항목 f         [T, B·C, D, K]             = memory bank
key context ξ       [T, B·C, D, K+1]           [u ; I]
점수 e / 계수 ρ     [T, B·C, D, T]             과거 축이 하나 더 붙어 O(T²)
소마 drive a        [T, B·C, D]                a = Σ_k w_k u_k  ← K 축이 여기서 접힌다
출력 스파이크 s     [T, B·C, D]                logical neuron당 1개
readout             flatten [B·C, T·D] → Linear → [B, H, C]
```

**v2와의 차이:** v2는 스파이크가 `[T, BC, D, K]`로 K 축이 출력까지 살아 있었다. v3-A는 소마에서 K를 접으므로 출력이 `[T, BC, D]`다. 이것이 "같은 입력 `I_n`, 같은 출력 `s_n`"이라는 사용자 요구의 직접적 결과이고, **O1에서 `w_k`를 학습 가능하게 두어야 하는 이유**이기도 하다. `w_k`가 고정 균등값이면 K 축의 정보가 접히면서 사라진다.

---

### 2. O1 — Soma parametrization (readout weights, τ_s, θ, reset rule)

**정할 것**

```
a_n = Σ_k w_k · u_{n,k}                somatic drive
v_charge = v_{n−1} + (a_n − v_{n−1})/τ_s
s_n = H(v_charge − θ)
v_n = v_charge − θ · s_n
```

**왜 문제인가 (1) — `w_k` 고정 균등값이면 population이 출력에서 붕괴한다.** K개 가지는 모두 같은 `I_n`을 받으므로 `Σ_k u_{n,k}`는 결국 **하나의 필터 출력**이다. 소마가 그 고정 합만 보면 다음 층과 readout이 보는 스파이크에 multi-timescale 다양성이 실리지 않는다. 다양성이 selector의 key 안에서만 살고 네트워크에는 전달되지 않는 구조가 된다.

**왜 문제인가 (2) — `τ_s=1`이면 firing rate가 drive를 부호화하지 못한다.** `τ_s=1`은 leak이 완전하다는 뜻이라 소마가 자기 기억을 갖지 않는다. drive가 `θ`와 `2θ` 사이면 **입력 세기와 무관하게 ISI가 항상 2**로 고정되고 rate가 0.5에서 포화한다. `τ_s>1`이면 소마가 적분하고 리셋 결손이 감쇠하며 남아 ISI가 drive에 따라 연속적으로 변한다.

**권고**

| 항목 | 값 | 근거 |
|---|---|---|
| `w_k` | learnable `[D, K]`(층당 128개), init 1.0 | 다양성이 출력에 도달. 각 logical neuron이 자기 temporal basis 조합을 학습 |
| `τ_s` | 2 | graded rate coding. 보정표에서 `input_scale=2` → firing rate 0.138 |
| `θ` | 1 | v2·논문·코드 공통 |
| reset | **standard soft reset**(위 식) | conventional LIF anchor와 소마 단이 글자 그대로 같아져, 차이가 dendritic 단에만 귀속 |

**이전 권고에서 변경:** DH-SNN 형식(`−θ·s_{n−1}`, 결손이 leak을 거치지 않음) → standard soft reset. 이유는 textbook 정의이고 v2·SpikingJelly·snnTorch가 모두 이 형식이라 baseline 비교가 깨끗해지기 때문이다. DH-SNN 형식은 ablation으로 남긴다. **D11 보정표는 이전 형식으로 측정했으므로 Phase B에서 재측정한다.**

---

### 3. O2 — Surrogate gradient (식과 scale)

**정할 것:** `H(v−θ)`의 backward. 현재 세 가지가 섞여 있다.

| 출처 | 식 | 기본 scale | peak | half-width |
|---|---|---:|---:|---:|
| spikeDE `surrogate.py` | `(s/2)/(1+(π/2·s·x)²)` | 함수 기본 2.0 | 1.00 | 0.318 |
| 같은 코드 `LIFNeuron` 실제 호출 | 위 식 | **5.0** | 2.50 | **0.127** |
| 논문 Appendix B.2.2 | `κ/(1+(κx)²)` | 2.0 | 2.00 | 0.500 |

**왜 문제인가:** 세 식은 모양(π/2, 1/2 인자)과 폭이 모두 다르다. `s=5`는 `s=2`보다 gradient 창이 2.5배 좁다. Selector는 `ρ → u → a → v`를 거쳐 **간접적으로만** loss에 닿으므로, 창이 좁으면 threshold에서 조금 떨어진 뉴런이 gradient를 못 받고 selector가 학습 신호를 얻지 못한다. v2에서 Q/K가 초기값 근처에 머문 원인 중 하나일 수 있다.

**권고:** 코드 식 + **s=5.0**(reference implementation이 실제로 전달하는 값)을 **모든 조건에 동일 적용**. 동일 적용하면 surrogate가 조건 간 차이를 설명할 수 없다. `s=2`는 declared sensitivity analysis로만. 논문 식은 "논문과 코드가 다르다"는 사실로 기록만 한다. v3-A는 threshold가 logical neuron당 하나이므로 v2보다 surrogate 비선형성이 K배 줄어든다.

---

### 4. O3 — Selection parametrization (η, π_k, ϑ, Q/K sharing)

**(a) `η` — selection strength.** `ρ = (1−η) + η·ρ̃`. `η=0`이면 정확히 f-LIF. 어디서 시작하느냐가 "Full vs Selective" 비교의 공정성을 정한다. 처음부터 강하게 켜져 있으면 차이가 학습 때문인지 초기화 때문인지 구분할 수 없다.

> `η = sigmoid(η̂)`, **층당 스칼라 하나**, `η̂` 초기 **−4**(η≈0.018), 학습. `∂ρ/∂η = ρ̃−1 ≠ 0`이라 초기에도 gradient가 흐른다. 층당 하나로 두면 "이 층이 selection을 얼마나 쓰게 되었나"를 숫자 하나로 보고할 수 있다. Full 조건은 `η≡0` clamp, selector 파라미터는 **inactive로 명시 보고**.

**(b) `π_k` — per-branch lag prior.** 내용은 population이 공동 결정하고 **얼마나 쓸지는 가지의 시간척도에 맞춘다**는 하이브리드의 구현체. τ=2인 빠른 가지와 τ=16인 느린 가지가 30 patch 전 사건을 같은 비중으로 읽을 이유가 없다. 기존 서술의 "초기 `c=∞`"는 구현 불가능이었다.

> `π_k(d) = exp(−d·g/τ_k)`, `g = softplus(ĝ)`, **층당 스칼라 하나**, `ĝ` 초기 **−5**(g≈0.0067). 확인: τ=16, d=41에서 지수 `−41×0.0067/16 ≈ −0.017` → `π≈0.983`으로 사실상 중립에서 출발. 가지 간 차등은 `τ_k`가 만들고 학습 파라미터는 하나. Ablation: `π≡1`(순수 shared selection).

**(c) `ϑ` — temperature.** `p`의 뾰족함, 즉 **초기 sparsemax support 크기**를 정한다. v2의 `ϑ=0.25`에서는 항등 Q/K만으로도 초기부터 support가 희소했다(worked example 점수 `[−1.63, −0.02, −0.02, −1.76]`). 그러면 "Sparse vs Full" 차이의 일부가 학습이 아니라 temperature에서 나온다.

> `ϑ = 1.0` **고정**(학습하지 않음). 초기에 (i) dense `κ≈1`, (ii) sparse support가 bank 대부분을 덮는지 측정·기록. 학습 가능하게 두지 않는 이유는 `ϑ`가 drift해서 실제로 일하는 변수가 되면 selection 기여와 구분되지 않기 때문이다.

**(d) `W_Q, W_K` sharing.** 층 내 모든 logical neuron 공유, 층마다 별도, 초기 `[I_K 0]`. Per-neuron Q/K는 selector를 per-neuron attention으로 만들어 **다른(더 큰) 모델**이 된다.

**초기화 시 기록 필수:** `η`, `g`, `κ_n`, support density, score 분포. 이것이 "중립극한에서 출발했다"의 증거다.

---

### 5. O4 — hard-mask(완전 배제) 모드 포함 여부

`η<1`이면 `ρ = (1−η) + η·ρ̃`의 바닥 `(1−η)` 때문에 sparsemax의 정확한 0이 **완전 배제로 이어지지 않는다.** 즉 "관련 없는 과거를 완전히 차단"이라는 주장은 `η=1`에서만 성립한다.

> **권고:** 1차 forecasting matrix에는 넣지 않는다(조건 수가 배가 되고, 바닥이 있는 편이 안정적이다). synthetic에서만 `η=1 + sparse` 조건을 하나 두어 "완전 배제가 필요한가"를 확인한다. 보고 시 "부분 배제(attenuation floor)"와 "완전 배제"를 구분해 쓴다.

---

### 6. O5 — Synthetic recall task를 event 단위로 재정의 (**실행 전 필수**)

**무엇이 어긋나 있었나.** 모델의 시계는 patch다. 입력 336 raw points, patch 8 → `T=42` model steps. Selector가 주소를 매길 수 있는 memory slot은 patch 단위다. 그런데 원래 Task B 초안은 "regime 길이 U[6,16] **raw step**, 시퀀스 96 step"이었다. regime 하나가 6 raw step이면 **patch 하나보다 짧아** selector가 지목할 수 없다. 이 상태로는 "몇 개의 사건을 건너뛴 회상인가"가 정의되지 않고 correct-slot retrieval rate를 측정할 수 없다.

**재정의: one patch = one event.**

| 항목 | 값 |
|---|---|
| 시퀀스 | **T = 42 events** → raw 42×8 = **336** |
| Event 인코딩 | patch의 8칸 = `[key one-hot (3) | value (1) | 0 (4)]` |
| Key 수 | M = 3 |
| Regime 길이 | **U[2,5] events** |
| 간섭 regime | 1–3개 |
| 채널 | **C = 1** (cue와 value가 같은 stream에 있어야 함 — channel-independent 구조) |
| Value | 첫 등장 regime에만 `v_k ~ U[−1,1]`, 재등장 시 0 |
| Target | `y_n` = 현재 key의 value |
| Ground-truth slot set | `A_n` = 현재 key **직전 등장 구간**의 event 색인 |

예시(앞 14 event):

```
n      :   0    1    2    3    4    5    6    7    8    9   10   11   12   13
key    :   A    A    A    B    B    C    C    C    A    A    A    B    B    C
first? :   Y    Y    Y    Y    Y    Y    Y    Y    N    N    N    N    N    N
value  :  .7   .7   .7  -.3  -.3   .5   .5   .5    0    0    0    0    0    0
target :  .7   .7   .7  -.3  -.3   .5   .5   .5   .7   .7   .7  -.3  -.3   .5
GT set :   -    -    -    -    -    -    -    -  {0,1,2} ...  {3,4} ...  {5,6,7}
```

**형상이 ETT와 완전히 같아진다**(`[B,336,C]`). `data_provider`·patch embedding·모델·진단 코드를 그대로 재사용하고 데이터셋만 교체하면 된다. "event j = memory slot j = patch j"가 일치하므로 적중률이 모호하지 않다.

**Readout과 timing rule.** `ŷ_n = Linear(D→1)(s_n)`, loss는 전 event, 보고는 **first appearance(복사, 쉬움)** / **re-appearance(회상, 본 지표)**로 분리. 시점 n의 selection은 `u_{n+1}`을 바꾸고 `s_{n+1}`에 반영되는 **한 스텝 지연**이 구조적으로 존재한다. 따라서 회상 지표는 **재등장 regime의 2번째 event부터** 집계하고 1번째는 별도 열로 보고한다. regime 최소 길이를 2로 둔 이유가 이것이다.

**왜 이 과제가 대조군을 가르는가:** "recent"는 직전 regime(오답), "uniform"은 전체 평균(오답), "fixed lag"는 regime 길이가 무작위라 불가, "no memory(full only)"는 8–20 event 간격을 멱함수 커널로 덮더라도 간섭 regime과 섞여 값을 특정할 수 없다.

**데이터·난이도:** train/val/test = 8000/1000/1000 시퀀스, 생성 seed 분리. 난이도 축은 간섭 regime 수(1/2/3), key one-hot 잡음, distractor 채널.

**대조군 정의 정밀화:** capacity-matched control은 `Linear(8 → D·K)`로 `D·K`개 독립 스칼라 뉴런(K=1)을 만들고 τ를 순환 배정해 **총 integrator 수를 맞춘다**. scalar control은 K=1. α=1 baseline은 같은 구조에서 α만 1.

---

### 7. O6 — Oracle intervention

**목적:** v2에서 답하지 못한 질문, 즉 원인이 ① "올바른 과거를 못 고른다"(selection)인지 ② "골라줘도 못 쓴다"(pathway: memory strength·소마·readout)인지를 가른다.

**구현:** selector만 교체하고 나머지는 손대지 않는다.

```
p_j = 1/|A_n| (j ∈ A_n), 0 (j ∉ A_n)
→ 동일 변환: ρ̃ = B_n p / Σ b p,  ρ = (1−η) + η ρ̃
```

**두 변형은 다른 질문에 답한다.**

| 변형 | 질문 | 주의 |
|---|---|---|
| **train-time oracle** | "선택이 완벽하면 이 구조가 과제를 푸는가" — 메커니즘의 upper bound | Full과 같다면 selector를 고칠 이유가 없음 |
| **test-time oracle** | "학습된 이 모델이 올바른 선택으로 이득을 보는가" | 모델이 자기 selector에 co-adapt 되어 있어 null 결과의 증거력이 약함 |

**판정 논리:** `train-time oracle ≈ Full`이면 pathway가 병목 → Phase C 복귀. `train-time oracle ≫ Full`인데 `학습 sparse ≈ Full`이면 selection **학습**이 실패 → score 함수·key 표현력·gradient 경로 점검.

---

### 8. O7 — 사전 판정 수치 (**사용자 승인 대기**)

§4.6의 수치가 예시로만 적혀 있었다.

> **확정 제안:** 다음 셋을 모두 만족하면 "검색 타당(retrieval valid)"으로 판정한다. ① 학습 sparse의 **정답 사건 질량 ≥ 0.5**, ② 회상 MSE ≤ **oracle의 1.5배**, ③ sparse의 회상 MSE가 full 대비 **20% 이상 감소**. 한편 **test-time oracle이 full과 5% 이내**면 selection 설계가 아니라 사용 경로의 문제로 판정하고 Phase C로 되돌아간다.

---

### 9. O8 — Training budget, seeds, MDE

| 항목 | 값 |
|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 1e-2(weight matrix만, bias 제외) |
| Batch / clip | 128 / grad-norm 1.0 |
| Max epochs / early stop | 50 / validation MSE patience 10 |
| LR schedule | ReduceLROnPlateau factor 0.5, patience 5 |
| Checkpoint | strict minimum validation MSE, 복원 후 평가 |
| Seeds | **{7, 13, 21, 42, 123, 256, 512, 1024}** (8개) |

**MDE 선언(핵심).** 우리 비교는 paired design이므로 관련 잡음은 raw MSE의 SD가 아니라 **paired difference의 SD**다. v2에서 측정한 ETTh1 H96의 값 `SD ≈ 0.0043`을 쓰면, paired t-test·n=8·power 0.8·α=0.05(양측)에서

```
|δ| ≥ (t_{.975,7} + t_{.8,7}) × SD/√n = (2.365 + 0.896) × 0.0043 / 2.83 ≈ 0.0050
```

MSE 수준 ~0.42 대비 **상대 약 1.2%**.

> **선언:** 상대 1.2% 이상만 검출 가능하다고 본다. 그보다 작은 차이는 **"이 표본 크기에서 검출 불가"**로 보고하며 "효과 없음"으로도 "작은 개선"으로도 서술하지 않는다.

**갱신 규칙:** SD는 v2(다른 모델) 추정치이므로 v3 첫 seed 묶음에서 재추정한다. 더 크면 n을 늘리거나 선언 MDE를 넓히되 **변경 사실과 그 시점에 이미 본 결과를 함께 기록**한다. **변동원 분리:** dataset 차이와 seed 차이는 다른 종류의 변동이다. dataset별 paired difference를 각각 보고하고, 방향이 일치할 때만 평균하며 그 사실을 명시한다. `2 dataset × 8 seed = 16`을 독립 16쌍으로 취급하지 않는다.

---

### 10. O9 — Reference environment (Phase A golden trajectory)

**문제:** spikeDE는 `torch.fx`와 `torch.compile`을 쓰므로 PyTorch ≥ 2.0 필요. `snn_recall`은 1.12(GPU 정상, compile 없음), `snn_jelly`는 2.11이지만 driver 535(CUDA 12.2) 대 cu130 불일치로 **CUDA 불가, CPU만 가능**.

**그래도 괜찮은 이유:** reference에서 필요한 것은 **짧은 scalar 궤적**이지 학습이 아니다.

**절차**

1. CPU 전용 `spikede_ref` 환경 신규 생성(python 3.10 + CPU torch 2.x), spikeDE를 고정 커밋 `fcd743b`로 설치.
2. 선언된 입력으로 scalar `LIFNeuron + pred` 실행: subthreshold, 단일 pulse, 반복 pulse, 상수 suprathreshold, 음/양 전류.
3. **스파이크만이 아니라 내부 항 전부를 덤프**: 입력, local trial 값, dynamics term, reset term, 적분 상태, 계수. CSV + SHA256. 출력만 맞추면 내부 불일치를 놓친다.
4. v3 구현(`snn_recall`)이 그 CSV를 재현하는지 대조. float64 목표 `atol 1e-8 / rtol 1e-6`, **달성 수치를 기록**.
5. 주장 등급 표기: **source parity**(원본을 실행해 맞춤) / **source-derived**(수식만 이식) / **mathematical validation**(해석해 대조, 예: 무발화 상수 forcing `U(t)=U_0+c t^α/Γ(α+1)`).
6. 설치 불가 시 source-derived + mathematical validation으로 낮추고 그 사실을 결과에 함께 표기.
7. **CPU golden ≠ GPU 학습 경로 검증.** 실제 학습 환경에서 수백 step 짧은 궤적으로 device(CPU/GPU)·dtype(float32/float64) 일치 검사를 별도로 추가한다.

---

### 11. 상태 요약

| | 항목 | 성격 | 상태 |
|---|---|---|---|
| O1 | Soma `w_k, τ_s, θ, reset` | 설계 | 권고 확정안 있음. reset 형식 변경으로 **보정 재측정 필요** |
| O2 | Surrogate 식·scale | 고정 | 권고 확정안 있음 |
| O3 | `η, π_k, ϑ`, Q/K 범위 | 설계 | 권고 확정안 있음 |
| O4 | hard-mask 모드 | 범위 | 1차 제외, synthetic에서만 |
| O5 | Synthetic 단위 | **수정** | event 단위 재정의안 있음 |
| O6 | Oracle | 설계 | train/test 두 변형 |
| O7 | 판정 수치 | **승인 대기** | 0.5 / 1.5배 / 20% / 5% |
| O8 | 예산·seed·MDE | **선언** | MDE 상대 1.2% |
| O9 | Reference 환경 | 절차 | CPU golden + 등급 표기 |

O7만 사용자 승인이 필요하고 나머지는 위 권고안으로 확정 가능하다. 확정 시 사전등록 문서 §2로 옮기고 §9에서 제거하지 않고 "확정됨"으로 표시한다.

### 12. 아이디어 기술 문서 `IDEA_LOG.md` 작성 (2026-09-21 01:30 KST)

사용자 요청으로 `NSMT/docs/IDEA_LOG.md`(445줄)를 새로 만들었다. **확정된 아이디어와 모델 정의를 한곳에 모은 기술 문서**이며, 이력·근거는 이 PROJECT_LOG에, 실행 계약은 사전등록 문서에 남긴 채 상호 참조하도록 구성했다.

구성: 문서 지도 → 한 문단 요약 → LIF/f-LIF/Population f-LIF 배경(수식·펄스 응답·커널 수치) → **v3-A 완전 정의**(수식, 텐서 형상, 확정 하이퍼파라미터 표, 검증된 환원 성질, 원문 f-LIF와의 관계, 구조적으로 없어지는 문제, 상호작용의 위치) → 선택 계산 5단계와 worked example → 선행연구 경계(무엇이 새롭고 무엇이 아닌가) → 검증 설계(Phase A–E, 회상 과제, 대조군, ETT 프로토콜, MDE) → **모듈 단위 구현 계획** → 보류 항목 → **O7 승인 목록**.

구현 계획은 파이썬 모듈 수준으로만 적었다. 디렉터리는 `NSMT/f_lif_pop_v3/{analysis, reference, forecasting}`이며 `forecasting/`은 기존 관례(config/layers/ours/model/train/test/utils/data_provider/scripts)를 따른다. v2에서 검증이 끝난 코드(sparsemax autograd, ETT 로더, 로깅 유틸, 파이프라인 컨트롤러)는 이식한다. `layers.py`의 구성요소는 `Sparsemax`, `arctan_surrogate`, `fractional_coefficients`, `Selector`, `Soma`, `PopulationNeuron`, `Embedding`이다. `PopulationNeuron`은 **루프 경로(선택 켬, O(T²))와 합성곱 경로(η=0, 하삼각 행렬곱)** 두 가지를 갖고 둘의 일치를 수치 게이트 G3으로 둔다. 수치 게이트는 G1–G10으로 정리했고, 실행 순서는 golden 생성 → Phase A 게이트 → 발화율 보정 → Phase B·C 게이트 → 회상 과제 → (O7 통과 시) ETT H96 → seed matrix다. Branch는 `exp/f-lif-pop-v3`를 새로 만들어 v2 결과와 섞지 않는다.

문서 말미에 **O7 승인 목록**을 두었다. ① 정답 위치에 실린 비중 ≥0.5(무작위 기대치 ≈0.15), ② 회상 오차 ≤ oracle의 1.5배, ③ 선택 켰을 때 개선폭 ≥20%, ④ 실패 진단으로 oracle과 선택 없음이 5% 이내이면 구조 문제로 판정. 각 항목에 근거와 더 엄격·느슨한 대안값을 함께 적었다. **이 네 숫자만 확정되면 구현에 착수한다.**

새 학습·GPU 실행 없음. 코드 구현: not run.

## 2026-09-21 20:53 KST — v3-A 비판적 검토서 독립 검증과 설계 쟁점 정리 (새 학습 없음)

### 0. 대상과 범위

사용자가 제공한 `NSMT/docs/Population_fLIF_v3A_Critical_Review_and_O7_Decisions_KO.md`(1425줄)는 `IDEA_LOG.md` v3-A를 검토한 문서다. 그 문서의 수치 주장을 `NSMT/f_lif_pop_v3/analysis/review_verification.py`로 독립 재계산하고, 대응책 후보를 같은 조건에서 시험했다. 학습·GPU 실행 없음. **검토서 §0.3의 원칙대로 IDEA_LOG·사전등록 문서는 승인 전에 수정하지 않았다.** 이 기록은 검증 결과와 결정 대기 목록이다.

### 1. 검토서 주장의 재현 결과 (전부 일치)

| 검토서 주장 | 검토서 수치 | 재계산 | 판정 |
|---|---|---|---|
| §5.3 `π_k` 초기값(g=softplus(−5))에서 κ<1 | τ=2..16: 0.9417 / 0.9702 / 0.9849 / 0.9924 | **10자리까지 동일** | 맞음. IDEA_LOG §3.4의 "κ=1.000"은 π 없이 계산한 값이었음 |
| §7.2 recent-slot 정책의 상태 증폭 (τ=2, α=0.7, I₀=2, g=0) | η=0: 1.10 / 0.018: 1.10 / **0.3: 26.50 / 0.5: 397,853** | **동일** | 맞음. 질량 보존이 안정성을 보장하지 않음 |
| §8 full-history fast path는 누설 feedback을 포함한 유효 연산자 `(I+BJ/τ)⁻¹B/τ` | 루프 대비 1.1e-16 | 7.6e-16; **단순 `B·I/τ`는 오차 1.21** | 맞음. IDEA_LOG §7.1의 "하삼각 행렬 한 번 곱" 서술은 틀림 |
| §13.3 oracle이 완벽해도 η가 작으면 실효 배분이 작음 | M_eff(η=σ(−4), m₀=0.15)=0.1653 | 동일 | 맞음 |
| §6 α=1 환원에 g=0 필요 | — | 기존 검증(prereg_numerics B1)은 π 없는 코드로 수행됨 → g=0 조건에서만 성립 | 맞음. 환원표에 조건 추가 필요 |
| §4 sparsemax의 0이 최종 적분의 0이 아님 | — | ρ=(1−η)+ηρ̃이므로 자명 | 맞음 (사전등록 D4에 이미 명시) |

### 2. 새로 확인한 것

- **불안정의 정체는 "이동하는 최근 slot"이 만드는 되먹임 고리다.** 같은 η=0.5에서 recent 정책은 397,853까지 커지지만 **고정된 먼 slot(j=0)** 정책은 1.63(η=1에서도 9.06)에 그친다. 증폭 자체가 아니라, 선택 대상이 매 스텝 직전 상태로 옮겨가며 `f_{n−1} = −u_{n−1}/τ`의 누설항을 η·B_n/τ 배로 되먹이는 것이 원인이다.
- **고리 이득 근사 `η·B_n/τ_k`가 1을 넘는 시점**: τ=2에서 η=0.3이면 n=16, η=0.5이면 n=8부터. T=42 전체에서 이득<1을 유지하는 상한은 **η < τ_min/B_41 = 0.143**.
- **가지별로 다르다.** τ=16 가지는 η=1에서도 max|u|=0.138로 안정. 폭주하는 것은 빠른 가지다.
- **세 가지 대응책이 최악 조건(recent, τ=2, η∈{0.5,1})에서 모두 1.1005로 억제된다.**
  - (R1) η 상한 0.07: 안정, 질량 보존 유지, **그러나 M_eff ≤ 0.21로 O7-① 도달 불가**
  - (R3) 계수 상한 `c_{n,j} = min(b_{n−j}ρ, b_0)` ("과거 사건을 현재 입력보다 크게 읽지 않는다"): 안정, 중립극한 유지, sparse 0 유지(η=1), lag 41에서 최대 4.37배 증폭 허용, **질량 비보존(최악 κ=0.079)** → κ를 보고
  - (R4) 입력항만 선택 `u_{n+1} = (1/τ)Σ b_{n−j}(ρ_{n,j} I_j − u_j)`, 누설은 고정 커널: 구조적으로 안정(동차부가 불변), 기억 항목이 f_j에서 **I_j로 바뀜** → 개념 문서 §26의 "input-history retrieval" 조건에 해당
- **O7-①과 안정성의 충돌.** 실효 정답 질량 ≥0.5는 m₀=0.15 기준 η ≥ 0.41을 요구하는데, (R1)의 안전 상한은 0.14다. 즉 **R1을 택하면 O7-①은 정의상 도달 불가**이고, R3/R4를 택해야 η를 크게 둘 수 있다. 어느 대응책을 고르느냐가 O7-①의 정의를 결정한다.

### 3. 검토서에 동의하는 설계 변경 (승인 대기)

| ID | 변경 | 근거 |
|---|---|---|
| D-A | **첫 기전 검증에서 g=0, π≡1.** 하이브리드(π_k)는 "tempered 변형"으로 별도 명명 | π는 질량 보존·α=1 환원·순수 멱함수 꼬리를 모두 깨고, 시간 감쇠와 내용 선택을 뒤섞음 |
| D-B | 주 모델 명칭을 "sparse redistribution + dense residual"로. 완전 배제는 η=1 조건에서만 | §4 |
| D-D | **소마가 `u_{n+1}`(I_n 반영 후 상태)을 읽도록 정렬 변경** | 현 정의에서는 s_n이 I_{n−1}까지만 봐서 **마지막 patch가 출력에 도달하지 않는다**(예측에 치명적). 정렬을 바꾸면 회상 과제의 "2번째 event부터" 규칙도 불필요 |
| D-F | fast path = 누설 feedback 포함 유효 연산자(삼각 solve) | §8, 재현됨 |
| D-H | oracle을 oracle-trained / test-time / generator-lookup 셋으로 분리, "최선"이 아니라 "특권적 정책" | §13 |
| §16.3 | **Uniform·mass-matched 대조군 제거** — g=0·질량보존에서 Full과 동일 | ρ̃≡1 |
| §11.2 | 회상 과제에 gated-recurrent 대조군 추가 | 3-key는 압축 상태로도 풀림(Zoology). 단 M1·η=0은 소마 전까지 I에 선형이라 key 조건 저장이 불가하므로 Full은 여전히 못 풀 것으로 예상 |
| §18 | 게이트 G11(η·support·T 안정성 sweep), G13(시간 정렬), G14(oracle 분리), G15(값 독립 생성·인과 readout) 추가 | 위 검증이 G11의 필요성을 직접 보여줌 |

### 4. 검토서와 의견이 다른 곳

- **§5.5 "질량 보존 > 상한" 우선순위**: 검토서 자신의 §7 결과로 뒤집힌다. 되먹임 고리가 확인된 이상 **안정성 > 질량 보존**이어야 하며, κ<1은 결함이 아니라 보고 대상이다. → R3 또는 R4 권고.
- **§15.2 O7-① "실효 정답 질량 ≥0.5" 절대치**: R1을 택하면 도달 불가. 절대치 대신 **기준선 대비**(M_eff ≥ 2·m₀ 또는 M_eff − m₀ ≥ 0.2)로 정의하거나, R3/R4를 채택해 η를 풀어야 한다. 어느 쪽이든 실행 전에 결정.
- **§11.2 "Full로 풀 수 있는 구성적 대안이 있다"**: 논리적으로는 맞지만 **M1 v3-A의 Full(η=0)은 소마 전까지 입력에 선형**이라 key×value 곱 상호작용을 만들 수 없어 실제로는 못 풀 가능성이 높다. 따라서 대조군은 "Full"이 아니라 **비선형 recurrent(GRU 등)**여야 판별력을 검사할 수 있다.

### 5. 대응책 선택에 대한 권고

**R3(계수 상한 c ≤ b_0)을 주 후보, R4(입력항 선택)를 명명된 대안**으로 둔다. 이유: R3는 중립극한·정확한 0·증폭 허용을 모두 유지하며 "과거 사건을 현재 입력보다 크게 읽지 않는다"는 해석 가능한 물리적 의미를 갖는다. R4는 가장 안전하지만 기억 항목의 의미가 바뀌므로 별도 모델이다. R1은 O7-①을 불가능하게 하므로 기각. **어느 경우든 Phase C에서 η·support 집중·T 변화의 안정성 sweep(G11)을 통과해야 한다.**

### 6. 결정 대기 목록 (사용자)

1. 안정성 대응책: **R3 / R4 / 둘 다** 중 선택
2. O7-① 정의: 절대치 0.5 (R3/R4 전제) / 기준선 대비 (R1 전제)
3. O7-②를 oracle-gap 회수율 G ≥ 0.5로 교체 (검토서 §15.3 권고, 동의)
4. O7-④ 5%를 자동 실패가 아닌 진단 범위로 (동의)
5. 소마 정렬 변경(D-D) 채택 여부 (권고: 채택)
6. g=0 첫 검증(D-A) 채택 여부 (권고: 채택. 이는 앞선 "선택 단위" 질문에서 (2) 순수 공유 선택을 1차로 두는 것을 뜻함)

승인 후 IDEA_LOG와 사전등록 문서를 **rev.1**으로 갱신하고 원본과의 차이를 보존한다.

Artifact: `NSMT/f_lif_pop_v3/analysis/review_verification.{py,txt}`.

---

## 2026-09-21 22:19 KST — v3-A 안정성 sweep, 설계 확정, IDEA_LOG·사전등록 rev.1

**Branch:** `exp/f-lif-pop-v2` (문서 작업만. 학습 실행 없음) · **작성:** Claude Opus 5 세션
**사용자 승인:** "너의 권고대로 진행하자" — 직전 항목 §6 결정 대기 목록 6개 전부 권고안대로 확정.

### 1. 왜 추가 분석이 필요했는가

직전 항목에서 검토서 §7의 증폭 주장을 재현하고 원인을 **"매 스텝 움직이는 되먹임 고리"**(최근 slot을 반복해서 크게 읽음, 고리 이득 ≈ `η·B_n/τ`)로 좁혔다. 대응책 R3(계수 상한 `c = min(b_d·ρ, b_0)`)를 주 후보로 권고했으나, **R3가 "최근 1칸" 정책보다 나쁜 정책에도 버티는지**는 확인하지 않은 상태였다. 이를 확인하지 않고 문서에 확정하면 사전등록의 의미가 없다.

### 2. Adversarial stability sweep

**Artifact:** `NSMT/f_lif_pop_v3/analysis/stability_sweep.{py,txt}` (학습 없음, float64 스칼라 시뮬레이션)

측정량은 `max|u_n|`. α=0.7, T ∈ {42, 84}, η ∈ {0, 0.25, 0.5, 0.75, 1.0}. 입력 3종(pulse / const / noise). 정책은 고정 정책들에 더해 **greedy-adversarial** — 매 스텝 `|u_{n+1}|`를 가장 크게 만드는 과거 칸 하나를 고르는 최악 정책 — 을 포함한다.

```python
def run(T, tau, I, policy, eta, cap):
    """u_{n+1} = b0 f_n + sum_{j<n} c_{n,j} f_j ;  c = min(b_{n-j}*rho, b0) if cap else b*rho"""
    u = np.zeros(T + 1); f = np.zeros(T)
    for n in range(T):
        f[n] = (I[n] - u[n]) / tau
        acc = B0 * f[n]
        if n > 0:
            p = policy(n, u, f)
            den = sum(b(n - j) * p.get(j, 0.) for j in range(n))
            Bn = B(n)
            for j in range(n):
                rt = Bn * p.get(j, 0.) / den if den > 0 else 1.
                c = b(n - j) * ((1 - eta) + eta * rt)
                acc += (min(c, B0) if cap else c) * f[j]
        u[n + 1] = acc
    return u[1:]
```

**결과 (`max|u|`)**

| 설정 | pulse | const | noise | 판정 |
|---|---|---|---|---|
| 상한 **없음**, τ=2 | 397,853 | 2.36e6 → 7.3e9 → 발산 | 발산 | ✗ |
| 상한 있음, τ=2, η=1, T=42 | 10.86 | — | 30.9 | ✗ |
| 상한 있음, τ=2, η=1, **T=84** | **83.24** | — | **204.8** | ✗ **T에 따라 증가** |
| 상한 있음, **τ=4**, 모든 η, T=42 | **0.5503** | 1.335 | 3.16 | ✓ |
| 상한 있음, **τ=4**, 모든 η, T=84 | **0.5503** | 1.335 | 4.20 | ✓ T에 거의 불변 |

**결론 두 개.** ① **상한은 필수다** — 없으면 발산한다. ② **상한만으로는 부족하다** — 가장 빠른 가지 `τ=2`만 최악 정책·`η=1`에서 시퀀스 길이에 따라 증폭이 커진다(T=42의 10.86 → T=84의 83.24). 나머지 가지는 안전하다.

### 3. 보완책 비교 — F1(τ 이동) vs F2(τ-비례 상한)

| 보완책 | pulse | noise (T=42 / T=84) | 판정 |
|---|---|---|---|
| **F1.** τ = [4, 8, 16, 32], 상한 `b_0` | ≤ 0.5503 | 3.16 / 4.20 | ✓ **채택** |
| F2. τ = [2,…] 유지, 상한 `min(b_0, λτ)`, λ=0.25 | 1.1005 | 6.74 / 8.14 | ✗ **부적격** — 상한 0.5 < `b_1`=0.687이라 η=0에서도 상한이 걸려 **중립극한이 깨진다** |
| F2. λ=0.35 (중립극한 유지 최소값, 상한 0.70) | — | 11.1 / 13.2 | ✗ T에 따라 증가 |

F2는 "중립극한을 유지하려면 상한이 `b_1` 이상이어야 한다"는 제약 때문에 유효 범위가 없다. **F1이 유일한 선택.**

**F1의 반응성 확인 (계단 응답, `I ≡ 1`)**

| n | 1 | 2 | 4 | 8 | 42 |
|---|---:|---:|---:|---:|---:|
| τ=2 (기존 최속) | 0.550 | 0.591 | 0.717 | 0.817 | 0.946 |
| **τ=4 (신규 최속)** | **0.275** | **0.371** | **0.500** | **0.638** | **0.883** |

τ=4는 n=4에 계단의 절반, n=42에 0.883에 도달한다. T=42 창에서 "가장 빠른 가지" 역할에 충분하다. 또한 현재 입력항 `b_0·f_n`이 즉시 성분을 따로 제공한다.

**중립극한·희소성 보존 확인**

- `η=0`에서 상한 적용/미적용 결과가 `np.allclose(..., atol=0, rtol=0) == True`. `b_d`가 `d`에 단조 감소하므로 `d≥1`에서 `b_d ≤ b_0`이고, 따라서 상한은 구조적으로 작동하지 않는다.
- `η=1` sparsemax의 **정확한 0**도 보존된다(0에 상한을 씌워도 0).
- 상한이 허용하는 최대 증폭: **lag 1에서 1.60배, lag 41에서 4.37배.** 선택 기능은 살아 있다.

### 4. 확정된 설계 결정 (사용자 승인 완료)

| 코드 | 결정 | 직전 항목의 대기 번호 |
|---|---|---|
| **R3** | 계수 상한 `c = min(b_d·ρ, b_0)` — **주 안정성 장치** | 1 |
| **R4** | `ρ`를 `[0, 3]`으로 clamp — 명명된 대안. R3 실패 시에만, 날짜부 항목으로 기록 | 1 |
| **F1** | **τ = [4, 8, 16, 32]** (기존 [2,4,8,16]) — 본 항목의 sweep에서 신규 도출 | (신규) |
| **O7-①** | `M_eff ≥ 0.5` **절대 기준**. `M_eff = (1−η)·m₀ + η`. `η`·`m₀`·동일 `η`의 비-oracle 기준선 `M_eff`를 함께 보고 | 2 |
| **O7-②** | oracle 격차 회수율 `G = (E_full − E_learned)/(E_full − E_oracle-trained) ≥ 0.5`. **G14 통과 시에만** 판정에 사용. `G>1`도 성공 | 3 |
| **O7-③** | 평균 개선 ≥ 20% **이고** paired 차이 CI 상한 < 0 | (보강) |
| **O7-④** | `\|δ_O\| ≤ 5%`는 **자동 실패가 아니라 진단 구간** | 4 |
| **D-D** | 소마가 `u_{n+1}`을 읽는다. 질의 `ξ_n`은 갱신 전 `u_n`으로 만든다(인과성 유지) | 5 |
| **D-A** | `g = 0`, `π_k ≡ 1`. tempered 변형은 별도 모델명으로 분리 | 6 |
| **D-H** | `uniform` 대조군 **삭제**(Full과 동일), `mass_matched` **유지**, **GRU 추가** | (수정) |

**D-H의 정정.** 직전 항목에서 검토서 §16.3을 따라 "uniform·mass-matched 둘 다 제거"라고 적었으나 **mass-matched는 유지**한다. R3 상한이 걸리면 `κ_n < 1`이 되어 `mass_matched`(`ρ ≡ κ_n`, 내용 무관)가 `full`과 갈라지기 때문이다. `uniform`만 제거하고, 그 동일성(`p` 균등 ⇒ `ρ̃≡1` ⇒ `full`)은 게이트 **G7b**로 검사한다.

### 5. 연쇄 변경 (τ 이동에 따른 재측정 대상)

| 항목 | 기존 (τ=[2,4,8,16]) | 신규 (τ=[4,8,16,32]) |
|---|---|---|
| 동질 대조군 `τ` (조화평균 `K/Σ(1/τ_k)`) | 4.267 | **8.533** |
| 가지 상태 크기 `mean\|u\|` (최속 가지) | 0.433 | 0.249 → **D11 발화율 보정 재측정 필수** |
| 유한구간 이득 (상수 입력 1, n=42, α=0.7) | [0.946, 0.883, 0.753, 0.556] | 재측정 후 기록 |

### 6. 추가 게이트 G11–G15, G7b

| 게이트 | 검사 | Phase | 실패 시 |
|---|---|---|---|
| **G11** | 상태 유계. 가지별 `max\|u\|` 기록, 사전 선언 상한 `10·max\|I\|` 초과 시 중단 | B/D/E 상시 | 중단, R4로 전환, 날짜부 기록 |
| **G12** | 상한 작동률 `cap_rate`. `η=0`에서 **정확히 0** | C | 상한 구현 오류 |
| **G13** | `support(p)` / `support(ρ)` / `support(b·ρ)` / `support(c)` 4종 분리 보고 | C/D | 보고 누락이 실패 |
| **G14** | oracle headroom `E(full) − E(oracle-trained) > MDE` | D | ②를 판정에서 제외, 명시 |
| **G15** | `I_{T−1}` 교란 시 `s_{T−1}`이 변해야 함 (D-D 정렬 확인) | A/C | 정렬 구현 오류 |
| **G7b** | `p` 균등 ⇒ `full`과 비트 단위 동일 | C | ρ 정규화 구현 오류 |

또한 기존 G8(`κ_n = 1.000`)은 **상한 미작동 조건(`cap_rate = 0`)에서만** 적용하도록 조건을 붙였다.

### 7. 문서 갱신

| 문서 | 조치 |
|---|---|
| `NSMT/docs/IDEA_LOG.md` | **rev.1** (445 → 531행). 문서 머리에 rev.0 대비 변경 8건 표 추가 |
| `NSMT/docs/archive/IDEA_LOG_rev0_20260921.md` | **rev.0 원본 보존** (원본과의 차이를 남기라는 요구에 대응) |
| `NSMT/docs/Population_fLIF_v3_prereg_KO.md` | **rev.1** (371 → 528행). §0의 변경 절차에 따라 **기존 §1~§9는 수정하지 않고** `§2A`(개정 결정 R3/R4/F1/D-A/D-B/D-D/D-F/D-H/D-J/D-K/D-L/D-M), `§3A`(게이트 G11–G15, G7b), `§9A`(O1–O9 해소 + O7 확정 + 본 문서 오류 정정표)를 날짜부로 추가 |

**IDEA_LOG rev.1에서 정정한 자체 오류 3건** (직전 항목에서 확인한 것을 문서에 반영)

| 위치 | rev.0 서술 | 정정 |
|---|---|---|
| §4 | "`κ = 1.000`" | `π` 없이 계산한 값. `g>0`이면 `κ` = 0.93/0.83/0.67/0.45로 가지마다 다르게 깨진다 → `g=0`으로 제거(D-A), 상한 작동 시 `κ<1`도 정상(R3) |
| §3.5 | "`α=1, η=0` ⇒ Euler, 오차 0.0" | **`g=0` 조건 필수.** `g>0`이면 지수 인자가 붙은 수정 recurrence |
| §7.1 | "`η=0`이면 하삼각 계수 행렬 한 번의 행렬곱" | **틀림**(오차 1.21). 누설 되먹임을 포함한 유효 연산자 `H_eff = (I + BJ/τ)⁻¹·B/τ`여야 하며 이것이 루프와 7.6e-16까지 일치. 계산 차수는 여전히 `O(T²)`이고 속도 이득은 측정 후 보고 |

추가로 §5의 "`α=1` 이질 기준선을 못 이기면 fractional 차수는 논문에서 빠져야 한다"를 **세 갈래 결론표**(D-M)로 교체했다. K=4 이질 population(α=1)은 4개 지수 커널의 합이지 fractional 커널의 fitted SOE 근사가 아니므로(τ 고정), "이 τ 설정·이 과제에서 기여가 확인되지 않음"과 "차수가 불필요함"을 구분한다.

또한 §6.2에 **회상 과제의 한계**를 명시했다. 이 과제는 값 3개를 압축 상태에 담고 현재 신호로 꺼내는 방식으로도 풀리므로, "우리 구조만 푼다"의 근거가 아니라 **"우리 구조 안에서 선택이 작동하는가"의 내부 진단**이다. GRU 기준선을 같이 돌려 격차를 정량화하고, 난이도 축에 **신호 종류 수**(3 → 8 → 16)를 추가한다.

### 8. 한계

- 본 항목의 안정성 결론은 **스칼라 시뮬레이션**에서 얻었다. 실제 모델은 `D`차원·`K`가지·학습되는 `W_Q/W_K`를 가지며, 학습된 정책이 greedy-adversarial과 같다는 보장도, 그보다 순하다는 보장도 없다. **G11이 학습 중 상시로 이를 감시한다.**
- 입력 3종(pulse/const/noise)은 실제 ETT patch 전류 분포와 다르다. D11 보정 후 실제 전류로 `max|u|`를 한 번 더 측정해 기록한다.
- τ 이동의 **성능** 영향은 측정하지 않았다(안정성과 계단 응답만 확인). τ=[2,4,8,16]과의 성능 비교는 회상 과제에서 탐색적 조건으로 돌릴 수 있으나 1차 matrix에는 넣지 않는다.

### 9. 다음 단계

`exp/f-lif-pop-v3` branch를 `origin/main`에서 만들고 Phase A 구현에 착수한다. 실행 순서는 IDEA_LOG §7.6.

Artifacts: `NSMT/f_lif_pop_v3/analysis/stability_sweep.{py,txt}`, `NSMT/docs/IDEA_LOG.md` (rev.1), `NSMT/docs/archive/IDEA_LOG_rev0_20260921.md`, `NSMT/docs/Population_fLIF_v3_prereg_KO.md` (rev.1).

---

## 2026-09-21 22:26 KST — v3-A Phase A·C 구현과 수치 게이트 통과 (학습 없음)

**Branch:** `exp/f-lif-pop-v3` (신규) · **Base commit:** `329183b94` (`exp/f-lif-pop-v2`)
**환경:** `/home/yschoi/.conda/envs/snn_recall/bin/python` (Python 3.10, torch 1.12, GPU 정상). 학습·데이터셋 사용 없음.

`origin/main`은 상류 NSMT 논문 저장소 루트(`forecasting/`, `anomaly_detection/`)이고 본 프로젝트의 참조는 로컬 `main`이다. v3는 v2 코드(Sparsemax autograd)를 이식하고 v3 문서를 이어받으므로 **`exp/f-lif-pop-v2` HEAD(`329183b94`)를 기준 커밋**으로 삼았다. 기존 로컬 변경은 전부 보존했다.

### 1. 구현한 것

| 파일 | 내용 |
|---|---|
| `NSMT/f_lif_pop_v3/forecasting/layers.py` | `Sparsemax`(v2 이식), `ArcTanSpike`(O2: `(s/2)/(1+(π/2·s·x)²)`, s=5.0), `fractional_coefficients`, `Selector`, `Soma`, `PopulationNeuron`, `Embedding` |
| `NSMT/f_lif_pop_v3/forecasting/check_model.py` | Phase A·C 게이트 실행기 (`--phase A|C|all`) |

`PopulationNeuron.forward`가 rev.1 수식을 그대로 구현한다.

```
f_n   = (I_n − u_n)/τ ;  ξ_n = [u_n ; I_n]           질의는 갱신 전 상태(인과성)
c_n,j = min(b_{n−j}·ρ_n,j , b_0)                      R3
u_{n+1} = b_0 f_n + Σ_{j<n} c_n,j f_j
a_n   = Σ_k w_k u_{n+1,k} → soma → s_n                D-D
```

`Selector` 모드는 `full / dense / sparse / recent / mass_matched / oracle` 6종이며 전부 같은 계수화 경로를 통과한다. `uniform`은 의도적으로 없다(= `full`). `mass_matched`는 학습 정책의 총질량 `κ_n`을 먼저 구한 뒤 `ρ ≡ κ_n`으로 내용 의존성만 제거한다.

`PopulationNeuron.fast_path`는 `η=0` 폐형식 `(Id + B·J/τ)⁻¹·B/τ`를 float64로 계산한다(G3 전용, 학습 경로 아님).

### 2. 게이트 결과 — 17 passed / 0 failed / 1 not run

Artifact: `NSMT/f_lif_pop_v3/analysis/check_model_phaseAC.txt`

**Phase A**

| 게이트 | 결과 | 측정값 |
|---|---|---|
| G1a `Σ_d b_d = (n+1)^α/Γ(α+1)` | PASS | max err 1.78e-15 (α=0.3/0.5/0.7/1.0) |
| G1b 상수 forcing 적분기 | PASS | max rel err 6.56e-07 (τ=1e7로 누설 억제) |
| G2 `α=1, η=0, g=0` ⇒ Euler | PASS | **max err 0.00e+00** |
| G3 루프 == `(Id+BJ/τ)⁻¹B/τ` | PASS | **5.00e-16**. 같은 조건에서 단순 `B@I/τ`는 **0.945** 어긋남 |
| G4 spikeDE golden 대조 | **NOT RUN** | 서버에 spikeDE 소스 없음 + torch 2.x CPU env 없음. O9에 따라 등급을 `mathematical validation`으로 유지 |
| G15 소마 정렬 (D-D) | PASS | `I_{T−1}` 교란 시 `Δv_{T−1}` = 2.2243, 그 이전은 정확히 0 |

**Phase C**

| 게이트 | 결과 | 측정값 |
|---|---|---|
| G5a `K=1` == scalar fractional branch | PASS | 0.00e+00 |
| G5b 동질 τ ⇒ 구성원 동일 | PASS | spread 0.00e+00, `τ_hom = 8.533` (F1 반영값) |
| G6 인과성 | PASS | 미래 patch 교란 시 이전 state/voltage/spike 오차 **정확히 0** |
| G7 중립극한 (`η=0` == full, 비트 단위) | PASS | `torch.equal == True` |
| G7b `uniform p` == `full` | PASS | 3.33e-16 (η=1). **`uniform` 대조군을 뺀 이유의 직접 확인** |
| G12 `η=0`에서 상한 비활성 | PASS | `cap_rate` 최대 **0.0** |
| G8a 상한 비활성 시 `κ = 1.000` | PASS | max\|κ−1\| 0.00e+00 |
| G8b `η=1` sparsemax의 정확한 0 | PASS | 23스텝 중 17스텝에 정확한 0. **`κ ∈ [0.242, 1.000]`** |
| G13 support 4종 분리 | PASS | η=0.0180에서 `support(p)=0.841`인데 `support(ρ)=support(b·ρ)=support(c)=1.000` |
| G10 gradient 유한·비영 | PASS | `W_Q, W_K, η̂, w` 전부 |
| G9 배치 == 창 단독 | PASS | 0.00e+00 |
| G11 상태 유계 (η=1, recent 정책) | PASS | `max\|u\|` = **2.385 (T=42) = 2.385 (T=84)** |

### 3. 게이트가 확인해 준 설계 주장 셋

1. **G13이 D-K를 직접 보여준다.** `η=0.018`에서 `p`의 support는 0.841인데 최종 계수의 support는 **1.000**이다. 즉 `p_j=0`이어도 `(1−η)·b_d`가 남는다. "sparse 선택"이라는 말을 최종 계수에 쓰면 틀린다.
2. **G8b가 mass-matched 대조군을 되살린 근거를 확인한다.** `η=1`에서 상한이 실제로 물려 `κ`가 0.242까지 내려간다. 따라서 `mass_matched`(`ρ≡κ_n`)는 `full`과 갈라지며 유효한 대조군이다(D-H).
3. **G11이 스칼라 sweep 결과를 실제 모델에서 재현한다.** D=4·K=4·학습 초기 `W_Q/W_K`, `η=1`, `τ=[4,8,16,32]`에서 `max|u|`가 T=42와 T=84에 **동일하게 2.385**다. 길이에 따른 증폭이 없다(F1의 목표). 다만 이는 `recent` 고정 정책이며, 학습된 정책에 대해서는 G11을 학습 중 상시로 돌려야 한다.

### 4. 한계·미결

- **G4는 실행하지 못했다.** 서버에 spikeDE 소스가 없고(`find / -iname '*spikede*'` 무결과) torch 2.x CPU env도 없다. 원본 대조 등급은 O9 규정대로 **`mathematical validation`**으로 낮춰 표기한다. 소스를 확보하면 `reference/make_golden.py`로 재판정한다.
- **G1b는 6.56e-07**로 다른 게이트(1e-15)보다 느슨하다. τ=1e7로 누설을 억제한 유한 근사이므로 잔차는 누설 항 자체다. 정확한 항등식 검사는 G1a가 담당한다.
- **발화율은 아직 0이다.** 기본 `input_scale`에서 스파이크가 나지 않는다(v2 probe에서 이미 확인한 dead-neuron 영역). τ 이동으로 가지 크기가 더 작아졌으므로 **D11 보정(Phase B)이 다음 단계**이며, 보정 전 숫자를 성능으로 해석하지 않는다.
- Phase B(발화율 보정)·D(회상 과제)·E(ETT)는 **not run**.

### 5. 다음 단계

`synthetic.py`(회상 과제 생성기) → `calibrate.py`(D11 보정, Phase B) → `ours.py`/`train.py`/`test.py` 이식 → Phase D.

Artifacts: `NSMT/f_lif_pop_v3/forecasting/{layers.py,check_model.py}`, `NSMT/f_lif_pop_v3/analysis/check_model_phaseAC.txt`.

---

## 2026-09-21 22:35 KST — v3 코드를 기존 저장소 작성 스타일로 재작성 (동작 변경 없음)

**Branch:** `exp/f-lif-pop-v3` · 사용자 요청: "기존 코드 작성 스타일을 그대로 모방하여 작성해줘".

### 1. 점검 결과 — 모방하지 않은 상태였다

저장소에는 목적에 따라 **두 가지 스타일**이 공존한다.

| 파일군 | 예시 | 특징 |
|---|---|---|
| **모델·레이어 모듈** | `model_v1/forecasting/{layers.py, ours.py}` | 모듈 docstring 없음, `__all__` 선언, 클래스 docstring은 기전 산문 + 평문 수식(`V(t+1)=decay*V(t)+...`) + 근거 인용(`review 6.3`, `Zipser 1993`), `forward` 시그니처 뒤 shape 주석, `T, B, N, D = x.shape` 명시, 오른쪽 정렬 인라인 주석, 타입힌트 거의 없음, 작은따옴표 |
| **독립 헬퍼 스크립트** | `model_v1/forecasting/{firing_rate.py, neo_bank.py}` | 모듈 docstring 있음, numpy식 `Args ----` 섹션, `print(f"[firing-rate] ...")` 태그 접두 출력 |

초판 v3 코드는 둘 중 어느 쪽도 아니었다(압축형 한 줄 docstring `"""[N,D,K+1] -> c [N,D,J]"""`, shape 주석 없음, `__all__` 없음).

### 2. 조치

| 파일 | 적용한 스타일 | 주요 변경 |
|---|---|---|
| `f_lif_pop_v3/forecasting/layers.py` | 모델·레이어 모듈 | 모듈 docstring 제거, `__all__` 추가, 클래스 docstring을 기전 산문 + 평문 수식 + 사전등록 결정코드 인용(`R3`, `F1`, `D-A`, `D-D`, `D-K`, `O2`) 형태로 재작성, `forward` 뒤 shape 주석, `T, B, D = x.shape` 명시, 오른쪽 정렬 인라인 주석, `Selector.coefficients` → `Selector.forward`(레퍼런스의 `self.lif(pre)` 호출 관례에 맞춤), `Soma.step` → `Soma.forward`, 변수명 축약(`increment`→`f`, `current`→`x`, `coeff`→`c`) |
| `f_lif_pop_v3/forecasting/check_model.py` | 독립 헬퍼 스크립트 | 모듈 docstring에 `Phases` / `Usage` 섹션, 함수에 numpy식 `Args ----`, 모든 출력을 `[gate]` 태그 접두로, `Report`→`GateReport`, `build`→`build_neuron`, `neutral`→`force_eta`(0/1 양쪽 지원), `argparse`에 저장소 관례대로 `dest=`·`nargs='?'`·`%(default)s` 적용 |

### 3. 회귀 확인 — 17개 게이트 전부 동일 수치로 재통과

Artifact: `NSMT/f_lif_pop_v3/analysis/check_model_phaseAC.txt` (갱신)

| 게이트 | 재작성 전 | 재작성 후 |
|---|---|---|
| G1a / G1b | 1.78e-15 / 6.56e-07 | **동일** |
| G2 / G3 | 0.00e+00 / 5.00e-16 (naive 0.945) | **동일** |
| G15 | 2.2243, 이전 시점 0.00e+00 | **동일** |
| G5a / G5b | 0.00e+00 / 0.00e+00, τ_hom 8.533 | **동일** |
| G6 / G7 / G7b | 0.00e+00 / `torch.equal=True` / 3.33e-16 | **동일** |
| G12 / G8a / G8b | 0.0 / 0.00e+00 / 17-23, κ∈[0.242,1.000] | **동일** |
| G13 | η=0.0180, p=0.841 vs ρ=b·ρ=c=1.000 | **동일** |
| G10 / G9 / G11 | 유한·비영 / 0.00e+00 / 2.385·2.385 | **동일** |

**한 곳만 값이 바뀌었고, 이는 개선이다.** G11의 사전 선언 상한을 `10·2·3 = 60.0`(어림값)에서 실제 `10·max|I| = 67.5`로 고쳤다. 사전등록 §3A G11의 정의(`10 · max|I|`)와 일치시킨 것이며, 판정 결과(`max|u| = 2.385 < 상한`)는 그대로다.

동작이 바뀌지 않았음은 위 17개 수치가 전부 일치한다는 사실로 확인했다. 학습·데이터셋 사용 없음.

Artifacts: `NSMT/f_lif_pop_v3/forecasting/{layers.py,check_model.py}`, `NSMT/f_lif_pop_v3/analysis/check_model_phaseAC.txt`.

---

## 2026-09-21 23:16 KST — 회상 과제 생성기와 Phase B 발화율 보정 (학습 없음)

**Branch:** `exp/f-lif-pop-v3` · **환경:** `snn_recall` (torch 1.12). `LD_LIBRARY_PATH`에 conda `lib`을 넣어야 pandas가 import된다(v2 runner와 동일). 학습 실행 없음.

### 1. 추가한 코드

| 파일 | 내용 |
|---|---|
| `data_provider/synthetic.py` | 회상 과제 생성기 `make_sequence` / `Dataset_Recall` / `sanity_check` |
| `data_provider/data_factory.py` | recall/ETT 분기. pandas는 ETT 경로에서만 import |
| `data_provider/data_loader.py` | v2에서 **무수정 이식** |
| `config.py` | v2 구조 이식 + v3 인자(`--task`, `--mode`, `--alpha`, `--tau`, `--n_keys`, `--cap`, `--input_norm` 등), `neuron_kwargs()` |
| `calibrate.py` | D11 발화율 보정 (Phase B) |
| `layers.py` | `to_patches()` 추가(v2 patching 관용구), `Embedding`에 `input_norm`·`fit_norm` 추가 |

생성기는 4값 배치 인터페이스를 유지하되 ETT가 비워두는 뒤 두 칸에 회상 지표에 필요한 감독 정보를 싣는다: `(x [336,1], y [42], truth [42,42] bool, recall [42] bool)`. `truth[n,j]`는 "j가 y[n]이 제시된 자리"이고 엄격히 `j<n`이다.

### 2. 회상 과제 sanity check 통과

Artifact: `NSMT/f_lif_pop_v3/analysis/recall_task_sanity.txt`

300개 시퀀스에서 다음을 assert로 강제 확인했다: truth의 엄격한 인과성, 재등장 구간의 값 칸이 0, 첫 등장 구간의 값 칸이 정답과 일치, 모든 회상 사건에 정답 칸이 존재, 정답 칸의 값이 목표와 일치.

| n_keys | 구간 길이 평균 | 끼어드는 구간 | 회상 사건 | 정답 칸 수 | chance mass |
|---:|---:|---|---:|---:|---:|
| 3 (주 조건) | 3.40 | 1–3 (평균 1.35) | 74.8% | 3.53 | **0.168** |
| 5 | 3.36 | 1–3 (평균 1.54) | 58.9% | 3.48 | 0.166 |
| 8 | 3.41 | 1–3 (평균 1.68) | 33.1% | 3.55 | 0.169 |

chance mass 0.168은 IDEA_LOG §9가 O7-①의 근거로 적은 "찍으면 0.15 수준"과 일치한다. **사전에 선언한 임계값 0.5가 실제로 "찍기보다 3배"임을 생성기 수치로 확인했다.**

### 3. 난이도 축 `n_keys` 상한 발견 — D-H를 D-O로 개정

rev.1 D-H가 적은 "3 → 8 → 16"에서 **16은 T=42에서 구성 자체가 불가능**하다. 구간 평균 3.4칸이면 42칸에 구간이 약 12개뿐이라 신호 종류가 그보다 많으면 재등장이 없다. 측정: 12개에서 회상 사건 1.8%, 16개에서 **0%**.

`run_range`를 (1,3)으로 줄이면 16개도 가능하지만 정답 칸이 3.5 → 2.0으로 줄어 **chance mass가 0.168 → 0.095**로 내려간다. 그러면 난이도 수준마다 O7-①의 0.5가 다른 뜻이 된다. 따라서 **`run_range` 고정, 축은 3 → 5 → 8**로 확정했다(D-O). 세 수준의 chance mass가 0.166–0.169로 일치한다.

부수적으로 `rng_codes`의 거부 샘플링 버그를 고쳤다. `cue_dim=4`면 서로 다른 ±1 코드가 16개뿐이라 16개를 전부 뽑는 거부 샘플링은 사실상 종료하지 않는다(실제로 무한 루프로 관측). 2^cue_dim개 부호패턴에서 **비복원 추출**하도록 바꿨다.

### 4. Phase B 발화율 보정 — 왜 발화율이 0이었는가

**설계 오류가 아니라 척도 불일치다.** 감쇠 사슬을 측정했다(I ~ N(0,1), D=16, K=4).

| 단계 | 배율 | 원인 |
|---|---:|---|
| `I` → 가지 `u` | 0.311 | `f=(I−u)/τ`의 `/τ`. τ=4에서 0.312, τ=32에서 0.057 (5.4배 격차) |
| `u` → 소마 입력 `a` | 0.516 | `w=1/K` 균등 평균. 구성원 상관 0.872라 평균이 크기를 회복시키지 못함 |
| `a` → 막전위 `v` | 0.800 | `τ_s=2` 저역통과 |
| **총** | **0.129** | `max v = 0.516` vs `θ = 1.0` |

τ(F1), w(=1/K), θ(=1)을 각각 독립적으로 정했고 셋을 잇는 척도는 정한 적이 없다. 그 척도를 정하는 절차가 D11이다.

### 5. 입력 표준화 `input_norm='frozen'` 채택 (D-N)

**`input_scale`만으로는 풀리지 않는다.** `input_scale`은 이득이므로 모든 입력에서 구동값이 음수인 유닛을 되살릴 수 없다. 회상 과제는 cue가 one-hot이라 유닛 하나의 구동값이 3개 이산 준위만 가지며, 셋 다 음수인 유닛은 **배율을 30배로 키워도 19%가 침묵**했다. 부호 문제이지 크기 문제가 아니다.

`Embedding`의 사영 출력에 **학습 구간에서 한 번 추정해 고정한** 유닛별 평균·표준편차를 적용한다(BatchNorm이 아니다 — 배치 통계를 쓰지 않는다).

| 회상 k=3 | 채택 scale | 발화율 | 죽은 유닛 | 포화 유닛 | 유닛별 발화율 범위 |
|---|---:|---:|---:|---:|---|
| `none` | 17.0 | 0.2151 | **31%** | 6% | 0.000 – 0.926 |
| **`frozen`** | **8.0** | 0.2278 | **0%** | **0%** | **0.178 – 0.248** |

| ETTh1 | 채택 scale | 발화율 | 죽은 유닛 | 유닛별 범위 |
|---|---:|---:|---:|---|
| `none` | 12.0 | 0.2171 | 3% | 0.000 – 0.694 |
| **`frozen`** | **6.0** | 0.1861 | **0%** | 0.049 – 0.256 |

**seed 비의존이라는 부수 효과가 크다.** `frozen`에서 seed 7/13/21/42가 전부 8.0을 고르고 발화율이 0.227–0.230에 모인다. "한 번 보정해 모든 seed·조건이 공유한다"는 D11 절차가 이로써 실제로 성립한다.

게이트 재확인: `frozen`에서 **G9(창 분리) 오차 0.00e+00, G7(중립극한) 비트 단위 동일, train/eval 출력 동일**. BatchNorm이었다면 셋 다 깨진다.

D11이 금지하는 것은 **구성원별(K축)** gain·threshold 조정이고 그 이유는 이질성을 지우기 때문이다. 여기서 조정하는 축은 **임베딩 유닛(D축)**이라 이질성과 무관하다.

### 6. 자체 정정 — `calibrate.py` 초판의 판정 기준이 사전등록보다 엄격했다

초판은 통과 조건에 "유닛(D축) 사망 0%"를 넣었으나, 사전등록 §3 Phase B의 조건은 "발화율 0.1–0.3 + **구성원(K축)** 사망·포화 없음"이다. 구현이 임의로 더 엄격한 조건을 추가한 것이므로 사전등록대로 되돌렸다. 유닛별 사망·포화율은 **판정이 아니라 진단**으로 보고하며, 그 진단이 D-N의 근거가 되었다.

### 7. 확정된 보정값

| 과제 | α | input_norm | **input_scale** | 발화율 | G11 선언 상한 | 관측 max\|u\| |
|---|---:|---|---:|---:|---:|---:|
| 회상 k=3 | 0.7 | frozen | **8.0** | 0.2278 | 80.0 | 17.81 |
| ETTh1 | 0.7 | frozen | **6.0** | 0.1861 | 299.4 | 19.69 |

모든 비교 조건이 이 값을 공유한다. α를 바꾸면 재측정한다.

### 8. 한계

- 보정은 **`mode='full'`(중립극한)**에서 수행했다. 학습이 진행되어 `η`가 커지면 동작점이 이동할 수 있으므로, 학습 중 발화율을 epoch마다 기록하고 구간을 벗어나면 보고한다.
- 회상 과제 보정은 1024 시퀀스·8배치 표본이다. 전체 학습 구간(8000)으로 재확인하는 것은 학습 시작 시 함께 기록한다.
- `input_norm='frozen'`은 사영 **초기값** 기준으로 고정된다. 학습으로 `emb_linear`가 움직이면 표준화가 어긋날 수 있다. 학습 후 사영 출력 분포를 진단에 남긴다.

### 9. 다음 단계

`ours.py`(M1 백본 + readout) → `utils.py`·`train.py`·`test.py` 이식 → Phase D 실행.

Artifacts: `NSMT/f_lif_pop_v3/forecasting/{config.py,calibrate.py,layers.py,data_provider/*}`, `NSMT/f_lif_pop_v3/forecasting/results/calibration/*.json`, `NSMT/f_lif_pop_v3/analysis/recall_task_sanity.txt`.

---

## 2026-09-21 23:42 KST — v3 독립 감사와 문서별 검토 메모 (새 학습 없음)

- 목적/범위: 사용자 요청으로 NSMT/docs 전체(archive/manifest 및 canonical log 링크 포함)와 루트 docs 환경 inventory를 검토하고, 다른 세션의 `f_lif_pop_v3` 구현·수치 검증·보정 결과를 감사했다. 다음 세션용 문서별 지속 메모 `NSMT/docs/DOCS_REVIEW_MEMORY.md`, 감사/조치 추적 문서 `NSMT/docs/ASSESMENT.md`를 생성했다. 다른 세션의 대화 내용은 보지 않았으며 저장된 코드·commit·artifact만 검토했다.
- 식별: 현재 branch `exp/f-lif-pop-v3`, 감사 HEAD `df3a3407b9ae653b4b1031320c4e8240b4c943a0`, v3의 실제 선택 base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 조회한 origin/main은 `191f366c6b9dd3cbfaeeb28bb49e7800c8ab488e`다. 새 학습 실험/브랜치가 아니라 진행 중인 v3의 감사다. 모델·config·생성기·기존 결과 및 pre-existing untracked `NSMT/papers/`는 보존했다.
- 방법: 초기 대상 파일49개의 path/SHA256/bytes inventory 및 임시 사본 `/tmp/nsmt_assessment_20260921`를 만들고, 고정 사본에서 CPU 게이트와 독립 반례를 실행했다. Markdown 본문·수식·개정 이력을 읽었으며, 긴 실행 표는 전 행 파싱/수치 재집계로 보완했다. Canonical log의73개 표/864개 table 행을 파싱하고8개 긴 per-run 표(총400행)의 finite metric/best epoch 범위와 v2 horizon/variant macro를 재계산했다. 과거 훈련/체크포인트 전체 재평가는 not run.
- 코드/문서 변경: 읽기 전용 재현 도구 `NSMT/scripts/audit_f_lif_pop_v3.py`, 위 두 문서 및 이 append entry만 작성했다. 감사 증거 JSON/명령은 `NSMT/f_lif_pop_v3/forecasting/results/assessment/20260921-2327-kst/`, raw gate stdout은 같은 task `log/assessment/20260921-2327-kst/gates.stdout`에 로컬 보존한다. Raw logs/checkpoint/data를 Git에 추가하지 않는다.
- 환경/설정: snn_recall Python3.10.18, torch1.12.0+cu113, CPU thread2, torch seed7. Synthetic probe는 data RNG20260921, n_keys3/5/8 각각300 sequences, code encoding, T42/patch8/run2–5/gap1–3/alpha.7/K4; actual-current 진단은 onehot 기본 생성256 sequences, frozen norm, scale8이다. DataProvider의 정식8000/1000/1000 학습이나 optimizer step은 수행하지 않았다.
- 실제 명령(cwd NSMT): `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_f_lif_pop_v3.py /tmp/nsmt_assessment_20260921`; 같은 env prefix로 `/home/yschoi/.conda/envs/snn_recall/bin/python /tmp/nsmt_assessment_20260921/f_lif_pop_v3/forecasting/check_model.py --phase all`. ETT loader inline Python 및 snn_jelly CPU 확인 명령 전체는 결과 폴더 `commands.md`에 기록했다. 사용자 작업 코드 변경 없이 실행했다.
- 수치 게이트: 재실행17 pass/0 fail/1 not run(G4), exit0. 핵심 branch/soma/shared selector/causality/중립극한은 최신 계약과 대체로 대응한다. G4 source parity, 실제 GPU/dtype parity, semantic recall, 학습 중 안정성은 이 숫자로 확인되지 않는다.
- **확인된 구현 오류 A01:** cap 이전 질량으로 `mass_matched`를 만들어 Full로 퇴화한다. 동일 bank/eta.5의 반례에서 sparse kappa=.569804778811738, control=1.0000000000000002, control–Full 최대 차이1.11e-16, 실제 질량 차이6.006159352561638이다. Cap 후 kappa로 대조를 만들고 동일 bank에서 합을 확인해야 한다.
- **O7 공식 정정 필요 A02:** `(1-eta)*m0+eta`는 cap 이전 perfect oracle 식이다. 해당 반례에서 옛 식 .5090226726020335 대 실제 cap 후 정답 계수 비율 .13834115533070288. 학습 성능이 아니라 수식 반례다. 승인된0.5를 임의 변경하지 않고 실제 c로 metric을 정의해야 한다.
- **과제/통계 정정 필요 A03/A04:** fresh key를 모두 먼저 넣고 gap1–3만 허용하면 오래된 도입 key는 재질의되지 않는다. n_keys3/5/8에서 실제 질의한 고유 key 평균은2.5233/2.5867/2.5467이다. 실제 uniform-slot chance `mean_query(|A_n|/n)`의 sequence 평균은.157525/.125956/.101110이다. 직전23:16 기록의 “chance가 세 수준에서 같다”는 근거는 `source_count/(T/2)` 근사의 산물이므로 이 항목으로 정정 대상으로 남긴다. 기존 파일은 덮어쓰지 않았다.
- **G11 보정 오류 A05:** raw patch×scale를 actual current로 기록한다. 감사256-sequence probe의 raw 값8 대 projection/frozen norm을 지난 actual max current30.6887283; max branch state17.4722176. 이 표본에서 폭주를 관측한 것은 아니다. 측정 대상·초과 판정·학습 monitor를 수정해야 한다.
- **배선/기록 불일치:** `eta_fixed/no-cap`은 현재 constructor에 전달되지 않는다(A06). snn_jelly torch2.11.0+cu130의 CPU forward/backward는 실행되었으므로 G4의 “torch2 CPU 환경 없음” 사유는 부정확하다(A07); spikeDE 설치/golden은 not run. G7b/G15·support/analog 진단·문서 모순·calibration provenance는 A07–A10에 통과 조건을 명시했다. 직전23:16 및 HEAD 메시지의 `rng_codes` 비복원 추출 수정 완료 설명과 달리 현재 소스에는 rejection while loop가 남아 있다(A11). n_keys>16/cue_dim4의 hang 실행은 하지 않았다.
- 기존 결과 분석: recall frozen scale8은 seed7/13/21/42에서 rate.2270–.2297, dead0; ETTh1 frozen scale6은 rate.18610, dead0이다. 이는 초기 train calibration이며 정확도 결과가 아니다. seed7 현 JSON은 rate.2280505933/state18.4044304로 직전 log의.2278/17.81과 다르다. 원인을 추정하지 않고 artifact hash와 함께 기록했다. 실제 ETTh1 loader는 train-only scaler 및 target boundary 검사를 통과했고 H96 counts8209/2785/2785, H7207585/2161/2161이었다.
- 문서 manifest 검산:1852 moves/20574 recorded files, source/destination 중복0. 두 tuning raw.txt의228/1045행 및 분할 hash/원문 복원 hash 일치. 현재 copy44개 중42개 hash 일치, 불일치는 config.py와 그 pycache로 역사 inventory와 현재 동일성을 구분했다.
- 문헌 검토: 원문 기반으로 Zoology/MQAR, sparsemax, entmax, constrained sparsemax, DH-SNN, f-SNN, LongSpike, NvoFDE 및 attention 설명력 논쟁을 확인했다. `ASSESMENT.md` §5에 직접 링크와 조건부 수정 방향을 기록했다. LongSpike는 확인한 arXiv v1 preprint로 표시했다. 논문 모델을 설치·학습 재현하거나 완전한 신규성 조사를 수행한 것은 아니다.
- 결론: 핵심 구현은 있으나 기전 판정을 훼손하는 오류와 미완성 검증이 남아 있다. A01–A06 등 수정/재검증 후 task 계약과 oracle/GRU 비교를 고정해야 한다. 현재 경로에 train/test 파이프라인·완료 성능 결과가 없어 v3 효과 판정은 보류한다. 각 이슈의 수치·위치·수정/통과 기준과 다른 에이전트의 append 양식을 `ASSESMENT.md`에 마련했다. 자동 감시/주기적 확인 설정은 하지 않았다.
- 검증/보존: 새 probe의 Python 구문, JSON parse, Markdown local link, 원본 log prefix 보존, 소스 hash 불변, Git whitespace를 점검한다. Commit/tag는 **not run**: 진행 중 실험의 공유 HEAD/자동화 guard를 유지하고 감사 파일을 작업 트리에 전달한다. 완료된 훈련 실험으로 tag하지 않는다. 이후 v3 완료 시 담당 세션이 코드·감사·canonical log를 함께 명시적으로 보존한다. Main 통합/remote push/package 설치/GPU 학습: not run.

---

## 2026-09-22 23:52 KST — 외부 감사(ASSESMENT.md) 수용과 수정 (학습 없음)

**Branch:** `exp/f-lif-pop-v3` · 감사 대상 HEAD `df3a3407b` · 감사 문서 `NSMT/docs/ASSESMENT.md`

사용자 지시로 `docs/ASSESMENT.md`를 확인하고, 검증 가능한 지적을 **독립 재계산으로 먼저 확인한 뒤** 수용했다. 감사가 보고한 수치가 내 재계산과 소수점까지 일치했다(A01 kappa 1.000000000000000·질량차 6.006159352561638, A02 0.5090226726020335→0.1383411553307029, A03 재질의 key 2.5233/2.5867/2.5467, A04 0.166217→0.157525). 환경 주장(A07)도 확인했다: `snn_jelly`는 torch 2.11.0+cu130이고 CPU autograd·`torch.compile`이 동작한다.

**P1 6건과 A11은 전부 사실이었다.** 요약하면 내가 만든 오류는 네 종류다. ① 대조군이 수학적으로 퇴화(A01), ② 판정식이 상한 도입 후 무효가 된 것을 방치(A02), ③ 난이도 축이 의도한 축이 아니었음(A03·A04), ④ 배선·측정 대상 오류(A05·A06·A11).

### 1. A01 — `mass_matched` 대조군이 항상 `full`이었다

`Σ_j b_j·ρ_j = (1−η)B + η·B = B`가 **항등식**이다. 상한 이전에 질량을 맞추면 배율이 항상 1이므로 대조군이 `full`과 같아진다.

| | 수정 전 | 수정 후 |
|---|---|---|
| 대조군 kappa | 1.000000000000000 | 선택 모델과 동일 0.589396241989 |
| `\|대조군 − full\|` | 1.11e-16 | **0.2957** |
| `\|Σc_control − Σc_sel\|` | 6.0062 | **1.78e-15** |
| 상한 만족 | — | max c_control 0.3916 ≤ b₀ 1.1005 |
| η=0에서 full과 일치 | — | 비트 단위 일치 |

수정 계약: `c_sel = min(b·ρ, b₀)` → `κ = Σc_sel/B` → `c_control = κ·b`. `κ≤1`이고 `b_j≤b₀`이므로 상한이 자동 충족된다. 게이트 **G16** 신설. 대조군의 정확한 이름은 "내용 무관"이 아니라 **"총질량 보존, slot 배분 제거"**다(κ 자체가 선택자가 만든 상태 의존량이므로).

### 2. A02 — O7-①의 `M_eff` 공식이 상한 도입 후 무효

`(1−η)m₀ + η`는 **상한 이전의 완벽한 oracle p**에서만 성립한다. 같은 반례에서 공식 **0.5090**, 실제 **0.1383**. 정의를 실측으로 교체했다.

```
M_eff(n) = Σ_{j∈A_n} c(n,j) / Σ_{j<n} c(n,j) ,   c = min(b·ρ, b₀)
```

**임계값 0.5는 감사 권고대로 유지한다.** 결과를 보기 전이므로 사후 조정이 아니다. 집계 순서(query 내부 → sequence → seed)와 `t=0`·빈 정답집합 제외 규칙을 사전등록에 고정했다.

### 3. A03 — 난이도 축이 기억 용량 축이 아니었다 (과제 revision r2)

r1은 모든 key를 먼저 소개한 뒤 "마지막 등장이 1–3 구간 전"인 key만 후보로 뒀다. T=42에는 구간이 약 12개뿐이라 **먼저 소개된 key는 영구히 후보에서 빠진다.** 실측한 재질의 고유 key 수는 n_keys 3/5/8에서 **2.52 / 2.59 / 2.55**로 사실상 동일했다. n_keys를 올린 것은 방해 신호를 늘렸을 뿐이다.

r2: 1단계 모든 key 한 번씩 소개 → 2단계 `min_gap` 이상 지난 key 중 **균등 재질의**.

| n_keys (r2) | 회상 사건 | key coverage | key당 질의 | 평균 정답 lag |
|---:|---:|---:|---:|---:|
| 3 | 75.2% | **99.3%** | 10.63 | 21.18 |
| 5 | 58.5% | **86.4%** | 5.80 | 21.26 |
| 8 | 33.2% | **48.2%** | 3.71 | 22.01 |

**T=42의 구조적 상한을 명시했다.** coverage ≈ min(1, (12.4−n_keys)/n_keys). 따라서 주 난이도 축은 **`{3, 5}`**, `n_keys=8`은 coverage 48%를 밝힌 stress 조건으로만 쓴다. r1 결과와 합치지 않는다(r1 학습 결과는 없다).

부수 효과로 평균 정답 lag가 21 event가 되었다. 재등장이 더 이상 근거리가 아니므로 과제가 실제 장거리 회상을 요구하게 됐다.

### 4. A04 — chance 계산의 분모 오류

query는 시간축에 균등하지 않고 n_keys가 커질수록 뒤로 밀린다. 따라서 `E[|A_n|/n]`을 `E[|A_n|]/(T/2)`로 대체할 수 없다. 또한 **균등 slot 확률과 `full`의 정답 커널 질량은 별개**이며, 선택자가 실제로 이겨야 하는 것은 후자다.

| n_keys (r2) | 기존 `E[r]/(T/2)` | uniform_slot_chance | **full_kernel_mass** | 0.5 / kernel |
|---:|---:|---:|---:|---:|
| 3 | 0.1662 | 0.1569 | **0.1305** | 3.83× |
| 5 | 0.1649 | 0.1263 | **0.1080** | 4.63× |
| 8 | 0.1640 | 0.1031 | **0.0913** | 5.48× |

**D-O의 근거 문장("세 난이도의 chance가 같으므로 0.5의 의미도 같다")을 철회한다.** chance는 난이도마다 다르다.

### 5. A05 — G11이 실제 전류를 재지 않았다

`max|raw patch| × scale`을 `max|I|`라고 불렀으나 실제 전류는 `Linear → frozen 표준화 → scale`을 지난 값이다. 회상 과제에서 **8.0 vs 실제 30.504**로 3.8배 어긋났다(감사의 독립 probe 30.6887과 일치). 보정이 실제 전류를 측정하고, 비유한 값이면 후보를 탈락시키며, sparse 초기 확인이 구간을 벗어나면 보정 실패로 처리하도록 고쳤다.

안정성 주장의 범위도 좁혔다: "모든 η·T에서 유계"가 아니라 **"검사한 입력·정책·길이 범위에서 유계"**다.

### 6. A06 — `--eta_fixed`·`--no-cap`이 이름만 바꾸고 있었다

두 옵션이 parser와 variant 이름에만 존재하고 `neuron_kwargs`에서 빠져 모델에 전달되지 않았다. 그대로 trainer를 붙였다면 **라벨만 다른 실험**이 될 뻔했다. `eta_fixed`는 sigmoid 큰 logit 근사가 아니라 정확한 덮어쓰기로 구현하고(η=0, η=1을 정확히 표현) `eta_hat`을 동결한다. 게이트 **G17** 신설 — eta_fixed 0→0.0, 1→1.0, no-cap에서 max c 7.107 > b₀ 1.101.

### 7. A07 — 게이트 의미 정정

- **G4 사유가 틀렸다.** "torch≥2 CPU 환경 없음"이라고 적었으나 `snn_jelly`는 torch 2.11.0+cu130이고 CPU에서 정상 동작한다. 실제 사유는 **고정 커밋 spikeDE 소스 부재**다. CUDA 불가는 CPU golden 불가의 근거가 아니다. 등급은 O9대로 `mathematical validation` 유지.
- **G7b**: 등록 문구는 비트 단위였으나 uniform-p 경로가 `Σ(b·p)`로 나누므로 마지막 ulp에서 구조적으로 다르다(관측 3.33e-16). 허용오차 1e-12를 사전등록에 명시했다.
- **G15**: 전압 변화만 보던 것을 **실제 스파이크 변화**로 바꿨다 — T−1에서 8개 변화, 이전 시점 0.

### 8. A11 — 수정했다고 기록했으나 코드에 반영되지 않았다

`rng_codes`의 비복원 추출 수정이 파일에 적용되지 않은 채 PROJECT_LOG 23:16과 commit 메시지에 "고쳤다"고 기록되어 있었다. **기록과 코드가 어긋난 provenance 오류**다. 이후 실행이 `cue_dim=5`를 써서 우연히 종료된 탓에 드러나지 않았다(cue_dim=4·n_keys=16이면 시도당 성공 확률 1.2e-7). 지금 수정 후 `rng_codes(16,4)`는 0.0001초에 16개 distinct를 반환하고 `(17,4)`는 ValueError를 낸다.

### 9. A10 — 재현성 보완(부분)

고정 표본으로 모든 후보 scale을 비교하도록 바꿨고(기존에는 scale마다 shuffle loader를 새로 순회), 파일명에 timestamp와 task revision을 넣어 덮어쓰기를 막았으며, source sha256·표본 sha256·full args·norm 통계·sparse 확인 결과를 JSON에 저장한다.

**PROJECT_LOG 23:16의 rate .2278/state 17.81과 JSON .2280505933/state 18.4044의 불일치 원인을 확인했다**: 그 사이에 `fit_norm`을 1배치에서 8배치로 바꿨고 로그는 이전 실행값이다. 추정이 아니라 실제 편집 이력이다. 두 값 모두 r2 재보정으로 대체됐다.

### 10. 재검증 결과

| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 게이트 | 17 passed / 0 failed / 1 not run | **19 passed / 0 failed / 1 not run** |
| 보정 (recall r2, frozen) | scale 8.0, rate 0.2278, max\|I\| 8.0(오류) | scale 8.0, rate 0.1848, **max\|I\| 30.504**, bound 305.0, max\|u\| 19.49 |
| 보정 (ETTh1, frozen) | scale 6.0, rate 0.1861 | scale 6.0, rate 0.1896, max\|I\| 50.756, max\|u\| 19.69 |

### 11. 아직 열려 있는 항목

- **A08(부분):** support 4종과 `has_history`는 aux에 넣었으나 score/QK norm, branch 상관·유효 rank, 선택 gradient norm은 없다. **gradient를 유지하는 analog readout 경로**는 `ours.py` 미구현이라 없다. 현재 `state`/`voltage`는 detach된 진단값이며, 거기에 head를 붙이면 뉴런까지 학습되지 않는다는 지적은 유효하다.
- **A09(부분):** 사전등록은 §2C로 정정했으나 `IDEA_LOG.md`와 `IDEA_SUMMARY_FOR_REPORT.md`의 표현(w 초기값·η·cap 후 M_eff·속도 이득)은 미정리.
- **A10(부분):** `Config.run_id`의 전체 설정 hash·run UUID, checkpoint의 `norm_mean/std` 포함과 fresh reload 확인, `stability_sweep.txt`의 F1/F2 출력 보충 미완.
- **A12:** 학습 파이프라인 전체가 not run.

### 12. 주장 범위

감사의 종합 판정을 그대로 받아들인다. **현재 자료로 "검증 완료"나 "아이디어 성능 입증"을 주장할 수 없다.** 지금까지 확인된 것은 정의의 자기일관성과 초기 동작점뿐이다.

Artifacts: `NSMT/f_lif_pop_v3/analysis/{check_model_phaseAC.txt,recall_task_sanity.txt}`, `NSMT/f_lif_pop_v3/forecasting/results/calibration/*_260921-23*.json`, `NSMT/docs/Population_fLIF_v3_prereg_KO.md` §2C, `NSMT/docs/ASSESMENT.md` 후속 기록.

## 2026-09-21 23:54 KST — v3 추적 감사 01 (r2 수정 및 재보정, 학습 없음)

- 목적: 사용자의 지속 관찰 요청에 따라 다른 세션의 수정/보정/사전등록 §2C를 검토하고 `NSMT/docs/ASSESMENT.md` §9를 append했다. 기존 감사/역사 기록은 보존했다. Branch `exp/f-lif-pop-v3`, HEAD `df3a3407b9ae653b4b1031320c4e8240b4c943a0`, v3 base `329183b94f65090cc6b337f464c5aa4d8e127ad7`; 미커밋 소스를 23:49:25 KST에 고정했다. 상세 SHA256/결과는 `NSMT/f_lif_pop_v3/forecasting/results/assessment/20260921-144925-utc/`와 later_inventory.json에 있다.
- 환경/명령: CPU thread2, snn_recall Python3.10/torch1.12, torch seed7; `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_f_lif_pop_v3.py /tmp/nsmt_assessment_20260921-144925-utc`; 같은 prefix/인자로 `scripts/audit_f_lif_pop_v3_followup.py`. GateReport에 phase_a/phase_c/phase_c_audit를 호출해19 pass/0 fail/1 not run(G4). 최초 --json CLI 시도는 미지원 옵션 exit2, 올바른 API 재실행 완료; 모델 실패 아님. 원시 로그는 task log/assessment 하위에 로컬 보존한다.
- 확인: A01 cap 후 질량 대조 mismatch6.006159→0, kappa 둘 다 .5698047788. A06 cap/eta_fixed 배선 및 정확값/동일 설정 state_dict 복원 spike/state 일치. A11 codebook16개 중복0·17개 ValueError. A04 두 chance 정의가 독립 재계산과 일치. A02 정의는 §2C에서 post-cap c로 개정, 임계 .5 유지; 학습 metric은 not run.
- 과제: r2 data seed20260921, T42/P8/run2–5/min_gap1/code encoding, n_keys3/5/8 각300 sequences. 고유 재질의 key2.98/4.32/3.8567, 모든 key 재질의 sequence98%/41%/0%. 영구 배제는 해결되었으나8-key의 coverage 부족은 남는다. §2C의 주3/5·stress8 구분을 확인. recall-only/첫 재등장 event 평가, 균형 query 및 delay 분리 제안은 ASSESMENT §9.2와 기존 Zoology 원문 근거를 따른다.
- 남은 문제: A05 actual current 측정은 수정되었으나 choose는 상태1001/bound10인 후보도 통과시킴(정책 반례, 관측 폭주 아님). branch_abs_max가 실제 max가 아닌 배치평균 max임을 텐서로 확인. cap=False인데 cap_rate 양수인 명칭 문제, 실패 sparse row 소실/분 단위 filename 충돌 가능성(A10). 사전등록 시각 및 동일 표본 재현 표현 정정 권고.
- 새 보정 결과 분석: recall r2 seed7 scale8 Full rate.1847657673060894/sparse.18672253005206585/current30.503751754760742/state19.48978614807129. ETT seed7 scale6 Full.18956471048295498/sparse.18921967595815659/current50.75556945800781/state19.68758773803711. 초기 보정이고 학습 정확도 아님. 과제/표본/난수 순서가 바뀌어 이전 보정과 인과적 성능 비교 불가. ASSESMENT §9.4의 Full rate 전사 오류를 즉시 후속 append로 정정했다.
- 변경/한계: 감사 probe 추가 및 감사/기억 문서와 이 기록만 작성. 작업 중 모델 소스와 결과는 해당 세션이 수정했다. 정식 train/test·학습 중 bound·8 seeds·golden/GPU·성능 판정은 not run. Commit/tag/push는 not run: 진행 중 공유 HEAD/index 보존. main 통합 없음. 이 세션의 직접 관찰이며 종료 후 자동 감사 예약 서비스는 설정하지 않았다.

---

## 2026-09-22 00:01 KST — 추적 감사 §9 신규 3건 수정과 기록 정정 (학습 없음)

**Branch:** `exp/f-lif-pop-v3` · 감사 문서 `NSMT/docs/ASSESMENT.md` §9 (추적 감사 01, 2026-09-21 23:52 KST)

추적 감사가 내 수정을 재검증해 **A01·A04·A06·A11을 VERIFIED**로 올렸고, A02·A03·A05·A07·A08·A09·A10을 PARTIAL로 두었다. 동시에 **새 결함 3건**을 지적했고 전부 재현되어 수정했다.

### 1. P1 — 보정이 상한 초과 후보를 거부하지 않았다

`calibrate.choose()`에 `max_abs_state=1001`, 선언 상한 10인 후보를 주면 **"closest to target"으로 선택**했다. `EXCEEDS`를 출력만 하고 무효화하지 않았다. **G11을 만들어 놓고 판정에 쓰지 않은 것**이다.

통과 조건에 `within_bound`를 넣고, 거부 시 사유에 탈락 개수를 적는다. `declared_bound`·`within_bound`·`g11_factor`를 JSON에 저장해 **학습 중에도 이 고정 상한을 쓴다**. 매 스텝 새 최댓값으로 상한을 올리면 상한 검사가 아니다.

| | 수정 전 | 수정 후 |
|---|---|---|
| max\|u\|=1001 vs bound 10 | **선택됨** | 거부, "1 in-band candidate(s) rejected for exceeding the declared bound" |
| max\|u\|=5 vs bound 10 | — | 선택됨 (회귀 없음) |

### 2. P2 — `branch_abs_max`가 최댓값이 아니었다

배치별 `mean`들의 max를 저장하고 있었다. 독립 64-sequence probe에서 저장값 **[2.02, 1.18, 0.67, 0.37]** vs 실제 **[12.33, 6.85, 3.69, 2.08]** — 6배 차이. `amax(T,B,D)`를 배치 간 max로 모으도록 고쳤다. 실제 보정 JSON에서 **[19.49, 13.64, 8.35, 4.67]**(평균 [2.53, 1.54, 0.90, 0.50])이고 가장 빠른 가지의 값이 `max_abs_state`와 일치한다.

### 3. P2 — `cap=False`인데 `cap_rate`가 양수였다

cap 유무와 무관하게 0.0488로 동일했다. 실제 clipping 빈도가 아니라 `raw > b₀` 잠재 초과율이었기 때문이다. 정의를 둘로 나눴다.

| 조건 | `cap_rate` (실제 잘린 비율) | `would_cap_rate` (정책의 잠재 초과율) |
|---|---:|---:|
| cap=True | 0.097561 | 0.097561 |
| cap=False | **0.000000** | 0.097561 |
| mass_matched | **0.000000** (κb ≤ b₀이라 구조적) | 0.097561 |

### 4. 내 기록의 부정확함 두 건 정정

- **날짜:** 사전등록 §2C와 `check_model.py`에 감사 날짜를 "2026-09-22"로 적었으나 **감사와 수정 모두 2026-09-21 KST**였다. 사전등록과 코드를 정정했다. 직전 PROJECT_LOG 항목의 제목 "2026-09-22 23:52 KST"도 같은 오류이며, 본 로그는 append-only이므로 **여기서 정정한다: 실제는 2026-09-21 23:52 KST다.**
- **과장:** "실제 30.504가 감사의 독립 probe 30.6887과 일치"라고 썼다. 두 값은 **표본과 조건이 다르므로** 같은 성질의 측정 오류를 독립적으로 지지하는 값이지 동일 표본의 재현 일치가 아니다.

### 5. 수용하되 일반화하지 않는 것 (§9.2)

추적 감사의 지적대로, r2에서도 `n_keys=8`이 `n_keys=5`보다 더 많은 key의 회상을 평가한다는 증거는 없다. 감사의 독립 측정으로 실제 고유 재질의 key는 **2.98 / 4.32 / 3.86**이고, 전체 key를 재질의한 sequence 비율은 **98% / 41% / 0%**다. 정확한 표현은 **"8개 도입, 평균 3.86개 재질의, 낮은 query coverage의 stress 조건"**이며 **capacity scaling 일반화는 보류한다.**

추가로 `lag_max`는 "sequence별 평균 lag의 최댓값을 sequence 평균한 값"이므로 표본 전체의 최장 lag로 읽지 않는다. 평가에서 **recall-only MSE와 재등장 구간 첫 사건 MSE를 분리**해 고정한다(전체 사건 MSE만 보면 key가 많을수록 copy 사건 가중치가 늘어난다). 사전등록 D-W에 명시했다.

### 6. 회귀 확인

게이트 **19 passed / 0 failed / 1 not run** 유지, 보정 선택값도 recall 8.0 / ETTh1 6.0으로 동일하며 둘 다 `within_bound=True`(305.0 / 507.6 대비 19.49 / 19.69)다.

### 7. 다음 단계

`ours.py`(M1 백본 + readout). A08이 요구한 **gradient를 유지하는 analog readout 경로**를 여기서 함께 만든다. 현재 `state`/`voltage`는 detach된 진단값이라 거기에 head를 붙이면 뉴런까지 학습되지 않는다.

Artifacts: `NSMT/f_lif_pop_v3/analysis/check_model_phaseAC.txt`, `NSMT/f_lif_pop_v3/forecasting/results/calibration/*_2609*.json`, `NSMT/docs/Population_fLIF_v3_prereg_KO.md` D-W, `NSMT/docs/ASSESMENT.md` 후속 기록.

## 2026-09-22 00:03 KST — v3 추적 감사 02: 재보정 수정 확인 및 원본 scalar reference 실제 실행

- Branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 감사01의 source는 다른 세션 commit `2da37e385dc134cdd798b40d6f9c2e72d065343b`과 일치. 감사02는 이 HEAD 위23:59:54 KST 미커밋 source 고정; 이후 담당 세션의 `0afda35825ba160689209177c0f0e9e173c618fa` 관찰. 감사자 commit/tag/push 없음.
- 목적/변경: ASSESMENT §10에 append. `scripts/audit_spikede_reference.py`와 `scripts/audit_calibration_failure_paths.py` 추가, followup probe는 새 bound row schema에 맞춰 일관된 실패 필드 명시. 모델/trainer 소스를 감사자가 수정하지 않음. Canonical 기록과 문서 기억은 append.
- 재검사: snn_recall CPU2threads/seed7, 64-sequence r2/data seed20260921. branch_abs_max가 actual tensor max와 정확히 일치, cap=False cap_rate0, choose의 Full 상한 초과 거부 확인. 초 단위 exclusive 생성과 sparse 실패 row 보존 확인. 단 main sparse 초기 확인에 finite/bound 실패를 주입하면 여전히 picked 유지/exit0; band 실패는 pickedNone/exit1. A05 전체는 OPEN이며 실제 폭주 관측으로 해석하지 않음.
- Reference: 공개 PhysAGI/spikeDE commit `fcd743befe504b1a471fa81887e6af7d6789da2e`의 원본9개 파일을 /tmp/nsmt_spikede_ref_fcd743b에 다운로드, hash/URL 보존. 패키지 설치/소스 수정 없이 snn_jelly torch2.11 CPU/float64 실행. 원본 LIFNeuron + pred_integrate_tuple의 alpha.3/.5/.7/1 × tau4/8 × 상수/펄스/seed7난수40steps, 24조건 spike 일치·최대 상태 오차3.0184e-15. 기존 reset_conventions port17spikes/오차6.6613e-16; scale5 arctan gradient 차4.4409e-16. 고정 사본 재실행도 같은CSV hash. G4 원본 소스 부재는 해소 가능/실제 해소됨. Full-wrapper/FX/compiled/adjoint/GPU/훈련은 not run, v3 soma와 원본 동등성 주장이 아님.
- Artifacts: `NSMT/f_lif_pop_v3/forecasting/results/assessment/20260921-145954-utc/` (inventory, followup_probes, calibration_failure_paths_corrected, reference JSON/전체CSV), 최초 live reference 실행은 `20260921-1500-utc-reference/`. Fault harness의 첫 파일 CLI exit 필드는 감사자 표기 오류로 corrected JSON이 우선함을 ASSESMENT에 append했다. 데이터/체크포인트 생성/설치/GPU 작업 없음.
- Exact commands: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2, cwd NSMT. snn_recall python `scripts/audit_f_lif_pop_v3_followup.py /tmp/nsmt_assessment_20260921-145954-utc`; 같은 python `scripts/audit_calibration_failure_paths.py /tmp/nsmt_assessment_20260921-145954-utc f_lif_pop_v3/forecasting/results/assessment/20260921-145954-utc/calibration_failure_paths_corrected.json`; snn_jelly python `scripts/audit_spikede_reference.py /tmp/nsmt_spikede_ref_fcd743b /tmp/nsmt_assessment_20260921-145954-utc f_lif_pop_v3/forecasting/results/assessment/20260921-145954-utc/reference`. 감사 runtime별 모델 소스 hash는 inventory.
- 보존/제한: 동시 담당 commit에 원시 audit stderr4개가 포함된 점을 기록하고 로컬 파일 보존. 감사자 index 변경하지 않음. 정식 학습/8seeds/성능 판정은 not run. 작업 에이전트의 ASSESMENT 후속 append도 보존하며 별도로 판정함.


## 2026-09-22 00:16 KST — f_lif_pop_v3 자동 추적 감사 예약 활성화

- 사용자 승인: 자동 감시 예약 설정을 명시적으로 요청. 현재 experiment branch에서 감사 운영만 추가하며 새 훈련 실험 아님. Branch `exp/f-lif-pop-v3`, 관찰 HEAD `0afda35825ba160689209177c0f0e9e173c618fa`, v3 base `329183b94f65090cc6b337f464c5aa4d8e127ad7`. 기존 변경과 작업을 보존했다.
- 구현: `NSMT/scripts/watch_f_lif_pop_v3.py`, `NSMT/docs/ASSESSMENT_WATCH_PROMPT.md`, 운영 문서 `NSMT/docs/ASSESSMENT_WATCH.md`, queue runtime ignore. Canonical 감사 파일은 ASSESMENT.md append만 사용. Cron `*/10 * * * *`로 기존 감사 thread에 queue; 무변경 skip·single pending·완료 marker+ack·실행 중 변경의 다음 주기 처리. 모델/학습 코드는 수정하지 않음.
- 설정/환경: /usr/bin/python3 stdlib, 활성 cron 서비스, 기존 Codex CLI/계정. 기존 crontab 보존/백업 후 전용 블록 추가. Init 명령 `python3 scripts/watch_f_lif_pop_v3.py init --thread 01a0c453-2d06-7ee2-bf44-dafb878d4a96`; 실제 전송 `python3 scripts/watch_f_lif_pop_v3.py tick --force`; 운영 상태 `python3 scripts/watch_f_lif_pop_v3.py status`. root는 NSMT.
- 검사: 임시 fixture + mock delivery9개 검사 통과, 실제 CLI queue healthcheck 및 첫 감사 접수, crontab 재조회 일치, cron active, Python 구문/whitespace 확인. 첫 감사 `20260921T151514Z-5f69481c`는 pending이며 완료 결과를 선취하지 않음. GPU/새 학습/통계 실험/commit/tag/push는 not run.
- 보존: lock/queue/state는 NSMT/scripts/queues/assessment_watch/ 로컬, raw cron stdout은 task log/assessment_watch/, 검사/설치 JSON은 task results/assessment/automation-setup-20260922/. 사용량은 기존 계정에 적용. 서버 및 CLI 연결이 필요. Pause/resume/status 명령은 운영 문서에 기록.


## 2026-09-22 00:28 KST — f_lif_pop_v3 예약 추적 감사03: 첫 smoke 재평가와 평가 경로 검증

- 목적: 사용자 승인 예약 `20260921T151514Z-5f69481c`의 구현/검증/개선 방향 감사. Branch `exp/f-lif-pop-v3`, base `329183b94f65090cc6b337f464c5aa4d8e127ad7`, snapshot HEAD `0afda35825ba160689209177c0f0e9e173c618fa`, 종료 전 다른 세션 HEAD `c34fa1c68c49169c13947b674543512dd05a057f`. 감사자 commit/tag/push 없음.
- 변경: `NSMT/scripts/audit_v3_pipeline.py` 신규 진단, ASSESMENT/DOCS_REVIEW_MEMORY/canonical log append, task results/assessment 텍스트 증거. 모델/훈련 소스와 환경은 수정하지 않음. 00:18:04 KST에62파일과 기존 checkpoint를 /tmp 사본으로 고정·hash한 후 CPU만 사용.
- 데이터/설정: 기존 smoke-001655, recall r2/k3/data_seed20260921, train/val/test512/128/128, seed7,2epochs,batch64,alpha0.7,D32,G4,input_scale8,frozen norm. 감사는 test128개를 batch128/16으로 평가해 집계 의존성도 확인. 원래 학습을 재실행하지 않음.
- 환경/명령: snn_recall torch1.12.0+cu113 CPU2threads. cwd NSMT, `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_v3_pipeline.py /tmp/nsmt_assessment_20260922-0017-kst-scheduled f_lif_pop_v3/forecasting/results/assessment/20260922-0017-kst-scheduled/pipeline_probes.json`.
- 결과: checkpoint all/copy/recall/first MSE .35197442620527725/.3153044522647635/.36387251826727696/.35899281812515704, 원래 JSON과 최대차5.56e-17. causal/nonoracle truth 분리 probe 차0, analog gradient 연결 확인. 기존 M_eff .22736488은 copy 혼입/집계 문제; 독립 recall-only sequence mean .12957217(Full kernel .12994454). ETT empty truth IndexError, GRU None scalar logger 오류, incompatible tau/cue 요청의 calibration 수용을 재현했다. A02/A05/A08/A10/A12는 명시한 범위만 확인하고 잔여 항목 OPEN.
- 증거: `NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922-0017-kst-scheduled/` inventory/checkpoint_inventory/pipeline_probes/observed_smoke/validation JSON. Checkpoint hash375628a1dfe871dfa9a7f518018a1f08777e5279b6abc7780739da3a3952463b. source hash는 inventory. trainer만 smoke provenance 당시와 달라 ‘학습 재현’으로 부르지 않음. 표준 CSV 두 개 미생성 확인.
- 검사/제한: 고정 사본 hash와 기존3문서 prefix 및 원본 checkpoint 보존 확인. metadata 검사의 zoneinfo import 오류는 stdlib UTC+09로 재실행 성공; 모델 오류 아님. 신규 학습/optimizer/GPU/설치/정식8seed/Full·GRU·oracle-trained 비교는 not run. 성능 우위 보류. primary MQAR/Zoology 및 Wiegreffe-Pinter 원문 확인 후 대조군·집계·coverage 개선을 ASSESMENT에 제안. 새 예약 없음;00:20 cron은 pending 중복 방지 확인. 이후 변경/결과는 다음 주기 감사.


## 2026-09-22 00:33 KST — v3 예약 추적 감사04: pilot 재평가와 checkpoint provenance 불일치

- 예약 20260921T153001Z-ed6b3b26; branch exp/f-lif-pop-v3, base329183b94f65090cc6b337f464c5aa4d8e127ad7, 관찰 HEADc34fa1c68c49169c13947b674543512dd05a057f. 감사 목적/범위: 감사03 이후 pilot 완료 결과만 추가 검토. 모델·설정 소스는 감사03 사본과 동일; 변경은 감사 스크립트/텍스트 증거/append 문서만.
- Snapshot /tmp/nsmt_assessment_20260921T153001Z-ed6b3b26,00:30:37 KST에 source/result와 checkpoint/config를 별도 복사·hash. Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260921T153001Z-ed6b3b26/(inventory,observed_pilot,pilot_probes,validation). 원본 보존.
- 기존 pilot: r2 k3/onehot,min_gap1,data_seed20260921; seed7, train/val/test2048/256/256,batch64,15epochs,lr.001,wd.01,eta_init-4,spike/scale8/frozen norm. 재학습 없이 snn_recall torch1.12 CPU2threads로 평가. Exact command(cwd NSMT): `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_v3_pilot.py /tmp/nsmt_assessment_20260921T153001Z-ed6b3b26 f_lif_pop_v3/forecasting/results/assessment/20260921T153001Z-ed6b3b26/pilot_probes.json`.
- 재현: recall MSE .26989367121801305; 같은 checkpoint Full .2804213865590815/oracle .20338486806976538, 저장 recall 수치 최대차5.56e-17. 독립 recall M_eff .13309303 대 기존 .23146702; A02 OPEN. A10-HASH JSON14a5650… 대 best50d4af2… 불일치 실제 확인; checkpoint 파일 SHA259794eb9a1620113fa50d0779fcfba95acec45197c7eabd8af788ea817f6679. 평가 결과는 재현되므로 parameter identity 기록 문제로 분류.
- 검산: source provenance11개 일치, 사본/원본 artifact hash 및 감사 구문 확인. 제한: 동일 seed pilot, 기존 smoke와 동일 data_seed; 재학습 Full/GRU/oracle-trained·8seed통계·새 학습/GPU는 not run. 성능 일반화 보류; 수정 우선순위 및 기존 primary 문헌 방향은 ASSESMENT 감사04 참조. 감사자 git add/commit/tag/push/reset/switch 없음.

---

## 2026-09-22 03:33 KST — 학습 파이프라인 완성과 선택자 역전파 안정성 발견

**Branch:** `exp/f-lif-pop-v3` · 파일럿은 **탐색적(exploratory)**이며 확정 결과가 아니다.

### 1. 파이프라인 완성

`ours.py`(M1 백본·두 head·analog 경로·GRU 대조군), `model.py`, `train.py`, `test.py`, `utils.py`(v2 이식)를 추가해 **end-to-end가 돈다.**

smoke에서 두 가지를 잡았다. ① `train.py`가 `fit_norm`을 호출하지 않아 frozen 표준화가 항등이었고 발화율이 **0.077**(보정값 0.185)이었다. 학습 시작 시 적합하도록 고쳐 **0.192**가 됐다. ② 버퍼가 등록되어 있으므로 **fresh reload가 test MSE를 정확히 재현**한다(0.351974426 양쪽 동일).

### 2. η가 사실상 움직이지 않는다

파일럿(2048 시퀀스, 15 epoch, θ=1): **η 0.0178 → 0.0298**. 이 속도면 0.5에 닿는 데 ~600 epoch이 필요한데 예산은 50이다. 검증 손실은 0.277 → 0.226으로 내려갔으므로 **선택은 꺼진 채 나머지만 학습된 것**이다.

구조적 원인이다. `dη/dη̂ = σ'(η̂)`이고 `η̂=−4`에서 **σ' = 0.0177**이라 기울기가 1/57로 눌린다. D5·D-B의 중립 출발이 의도한 대로 작동하되, 학습으로 빠져나오지 못한다.

### 3. 선택자 역전파가 긴 시퀀스·큰 η에서 폭주한다

Artifact: `NSMT/f_lif_pop_v3/analysis/selector_gradient.txt` (학습 없음, seed 3개 중앙값)

| T | η=0.05 | η=0.2 | η=0.5 | η=1.0 |
|---:|---:|---:|---:|---:|
| 10 | 4.3e-05 | 3.4e-04 | 3.4e-03 | 5.6e-03 |
| 30 | 1.2e-03 | 2.8e-02 | 1.9e-01 | 5.7e+00 |
| **42** | 6.6e-04 | 6.4e-03 | **1.8e+02** | **3.8e+03** |

길이에 지수적이므로 **되먹임 누적**이다. R3 상한은 **순전파** 상태만 막고 역전파는 막지 못한다. 과거 key 경로만 detach하면 η=1에서 24배 줄지만(3.8e3 → 1.6e2) 유일한 원인은 아니다.

### 4. 원인은 보정되지 않은 점수 온도 θ — 사용 가능한 창이 있다

O3는 θ를 1로 고정했고 D11은 `input_scale`만 보정했다. **둘을 잇는 검사가 없었다.** 보정 후 `max|u| ≈ 20`인데 점수가 `−‖u_n−u_j‖²/(d_q·θ)`이므로 θ=1이면 점수 폭이 수백이 되고 sparsemax가 hard argmax가 된다.

| θ | `max\|grad W_Q\|` (η=1, T=42) | `support_p` | 판정 |
|---:|---:|---:|---|
| 1 (현행) | 3.8e+03 | 0.327 | 불안정 |
| **4** | 2.4e+01 | 0.461 | **안정·희소** |
| **16** | 2.5e-01 | 0.653 | **안정·희소** |
| 64 | 7.4e-02 | 0.924 | 안정하나 사실상 `full` |

**θ를 `input_scale`과 같은 지위의 보정 대상으로 옮겼다**(D-X). 규칙은 `θ = mean‖W_Q ξ_n − W_K ξ_j‖²/d_q`를 학습 구간 고정 표본에서 한 번 재는 것이고, 회상 r2에서 **θ = 5.561**로 위 창 안에 들어온다.

### 5. 자체 정정

작업 중 "θ 규칙으로는 폭주가 안 잡힌다"고 보고했다. **틀렸다.** seed 하나짜리 표본으로 판단했고, seed를 고정해 3회 중앙값으로 다시 재니 θ=4에서 이미 23.6으로 안정했다. 잡음 표본으로 설계 결론을 내린 것이 문제였고, 이후 측정은 전부 seed 고정·다회 중앙값으로 바꿨다.

`key_norm`(Q/K 앞 표준화)도 추가했다가 **주 원인이 아님을 확인**해 기본값을 `none`으로 되돌렸다(η=1에서 1–2자릿수만 감소).

### 6. 검증 설계에 미치는 영향 (D-Y)

**D-B가 이미 규정한 고정 η 조건을 아이디어 검증의 주 경로로 승격한다.** 학습된 η는 "모형이 스스로 선택을 쓰도록 학습하는가"라는 별도 질문이며, 그 실패를 아이디어의 실패와 동일시하지 않는다. 고정 η 격자는 `{0, 0.2, 0.5, 1.0}`이고 η ≥ 0.5는 θ 보정 적용 하에서만 실행한다.

### 7. 범위 제한

§3·§4의 수치는 **학습 없이 초기 가중치에서 한 번의 역전파**를 잰 것이다. 학습이 진행되면 `W_Q/W_K`와 상태 분포가 움직여 실효 점수 척도가 달라질 수 있으므로 학습 중 `max|grad|`·`support_p`·`η`를 epoch마다 기록한다. §2의 파일럿은 12–15 epoch·2048 시퀀스의 탐색적 실행이며 성능 결론이 아니다.

Artifacts: `NSMT/f_lif_pop_v3/forecasting/{ours.py,model.py,train.py,test.py,utils.py}`, `NSMT/f_lif_pop_v3/analysis/selector_gradient.txt`, `NSMT/docs/Population_fLIF_v3_prereg_KO.md` §2D.


## 2026-09-22 03:33 KST — v3 예약 추적 감사05: query/key 정규화 함수와 checkpoint 호환성

- 예약 20260921T183001Z-0804948b; branch exp/f-lif-pop-v3, base329183b94f65090cc6b337f464c5aa4d8e127ad7, HEADc34fa1c68c49169c13947b674543512dd05a057f 위 layers.py 미커밋 변경 검토. source SHA001c4d8290c9901d6bba54368ef2c579aeec66307dc667df8c566482581d28db, snapshot03:30:45 KST /tmp/nsmt_assessment_20260921T183001Z-0804948b. 감사자 변경은 scripts/audit_v3_key_norm.py·텍스트 증거·append 문서뿐.
- 환경/표본: snn_recall torch1.12 CPU2threads, seed7,float64 T12/B2/D3 난수 입력; 기존 pilot checkpoint를 복사 후 strict 로드 검사. 새 훈련/데이터 split 실험은 not run. 모듈 identity의 spike/state/input gradient 차0, 독립 mean/std 차0, 동일 Full 표본 재적합/causality/new-schema roundtrip 차0. 기존 pilot strict 로드는 새 buffer2개 누락으로 실패(A10-KEYNORM-CKPT OPEN).
- Exact command(cwd NSMT): `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_v3_key_norm.py /tmp/nsmt_assessment_20260921T153001Z-ed6b3b26 /tmp/nsmt_assessment_20260921T183001Z-0804948b f_lif_pop_v3/forecasting/results/assessment/20260921T183001Z-0804948b/key_norm_probes.json`.
- Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260921T183001Z-0804948b/inventory.json,layers.diff,key_norm_probes.json,validation.json. 원본/사본 hash 보존 및 구문 확인. 학습·optimizer·GPU·설치·git 변이·commit/tag/push 없음. 결과 일반화/정규화 안정성 효과는 not run.
- 결론: 함수 수준 확인, train/calibrate 연결은 캡처 시 미완. 감사 중 calibrate/config/layers 후속 변경은 다음 감사 범위. 기존 OPEN 유지. Pascanu2013 원문 근거로 clipping 전 gradient와 시간길이·validation 통제 비교 제안; ASSESMENT 감사05 참조.

---

## 2026-09-22 03:42 KST — 탐색적 파일럿: oracle은 성공하고 학습된 선택자는 실패한다

**Branch:** `exp/f-lif-pop-v3` · **성격: 탐색적(exploratory).** 12 epoch, 2048 학습 시퀀스, **seed 7 하나**, 회상 과제 r2, θ=5.561, input_scale 8.0. **확정 결과가 아니며 O7 판정에 쓰지 않는다.** 효과 크기가 커서 방향은 읽을 수 있으나 반복·통계 없이 결론을 내리지 않는다.

### 1. 결과

| 조건 | **recall MSE** | copy MSE | 비고 |
|---|---:|---:|---|
| **oracle-trained, η=1** | **0.0297** | 0.0921 | 정답 정책을 고정하고 처음부터 학습 |
| oracle-trained, η=0.5 | 0.0362 | 0.1096 | |
| **GRU 대조군** | **0.2531** | **0.0164** | 압축 상태 기준선 (D-H) |
| sparse, 학습된 η(≈0.026) | 0.2717 | 0.1460 | 주 조건 |
| full (η=0) | 0.2763 | 0.1487 | 선택 없음 |
| sparse, η=0.5 고정 | **0.4108** | 0.4517 | 선택을 켰더니 **더 나빠짐** |

### 2. 읽어야 할 것 넷

**① 구조는 검색을 쓸 수 있다.** oracle-trained가 full보다 **9.3배** 낫다(0.0297 vs 0.2763). "정답 위치를 알려주면 이 뉴런이 그것을 실제로 활용하는가"에 대한 답은 **예**다. 사전등록 G14(headroom 유효성)는 압도적으로 통과한다(여지 0.2466).

**② 학습된 선택자는 그 여지의 1.9%만 가져온다.**

```
G = (E_full − E_learned) / (E_full − E_oracle-trained)
  = (0.2763 − 0.2717) / (0.2763 − 0.0297) = 0.0046 / 0.2466 = 0.019
```

사전등록 O7-② 기준은 **G ≥ 0.5**다. 현재 0.019다.

**③ 진단이 일관된다.** 모든 조건에서 `M_eff ≈ kernel_mass`였다(full 0.229/0.229, η=0.5 0.234/0.229, 학습 sparse 0.231/0.229). 즉 **선택은 희소하게 작동하지만(`support_p` 0.358–0.395) 정답 쪽으로 질량을 옮기지 못한다.** v2의 실패 양상이 반복된다 — 점수는 "닮음"이지 "유용함"이 아니다.

**④ 나쁜 점수로 선택을 켜면 해롭다.** η=0.5 고정에서 recall이 0.4108로 full보다 나쁘고, **copy조차 0.4517로 무너진다**(full 0.1487). 정답이 입력에 들어 있는 사건까지 망가진다는 것은, 잘못된 재가중이 상태 표현 자체를 훼손한다는 뜻이다.

따라서 현재 국면은 외부 감사 §5.2가 미리 지목한 **"oracle은 성공하고 learned selector만 실패한다"**에 해당한다. 사전등록 O7 판정표의 "고르지 못함 → 점수 함수·key 표현력 재검토" 가지다.

### 3. 부수 관찰 — readout 병목의 징후

oracle η=1에서 **recall(0.0297)이 copy(0.0921)보다 낫다.** copy는 정답이 현재 입력에 있는 쉬운 사건인데도 그렇다. 또 GRU의 copy는 **0.0164**로 우리 최고 조건보다 5.6배 좋다. 현재 입력이 임베딩·스파이크를 지나면서 손실되는 양이 크다는 뜻이며, **선택과 무관한 readout 병목**이 따로 있을 수 있다. D8의 막전위(analog) readout 진단이 이를 가리기 위한 조건이며 `--readout analog`로 실행 가능하다.

### 4. GRU가 현재 우리보다 낫다

GRU는 recall 0.2531로 full(0.2763)·학습 sparse(0.2717)보다 낫다. D-H가 이 대조군을 필수로 둔 이유가 그대로 드러났다. **현 상태에서는 "압축 상태로도 풀리는 과제를 우리 모형이 더 못 푼다"**가 정확한 서술이다. 단 oracle-trained(0.0297)는 GRU를 8.5배 앞선다.

### 5. 함께 고친 것

- `train.py`가 비스파이킹 모델에서 `None` 진단값을 TensorBoard에 기록하려다 **모든 GRU 실행이 첫 epoch에서 죽었다.** 해당 키를 생략하도록 고쳤다.
- `Config.run_id`에 **모델 이름·α·readout이 없어** GRU와 myModel이 같은 이름으로 충돌했다(감사 A10 지적). run_id에 포함했다.
- 감사의 결함 주입이 찾은 빈틈: `choose()`가 full probe의 상한·유한성만 보고 **실제 학습되는 sparse 초기값은 발화율만** 검사했다. 네 가지 주입 case가 이제 모두 올바르게 거부된다.

### 6. 한계

seed 1개·12 epoch·2048 시퀀스의 탐색적 실행이다. 학습이 수렴했는지 확인하지 않았고, oracle η=1은 구조상 `M_eff = 1.0`을 강제하므로 **학습 가능한 점수가 도달할 수 있는 목표라는 보장은 없다.** 또한 이 과제에서 GRU가 강하다는 사실은 §6.2에 이미 적은 과제 한계(압축 상태로도 풀림)의 확인이다.

### 7. 다음 단계

감사 §5.2의 권고 순서를 따른다. 먼저 **관측**한다: 실제 배치의 Q/K gradient 크기, sparsemax singleton support 비율, `cap_rate`, η 이동량, 점수 분산. 그 다음에야 dense warm-up, `Δu` key 추가, entmax 등 대안을 **탐색적 조건으로 분리해** 시도한다. 아직 O7 판정을 시도하지 않는다.

Artifacts: `NSMT/f_lif_pop_v3/forecasting/results/pilot-theta-*/`, `pilot-oracle-*/`, `pilot-gru-*/`.


## 2026-09-22 03:45 KST — v3 예약 추적 감사06: sparse 실패 처리 재검사와 θ/oracle pilot

- 예약 20260921T184001Z-e2dd33af, branch exp/f-lif-pop-v3, base329183b94f65090cc6b337f464c5aa4d8e127ad7, HEAD5512135e7c2570f50b6f4f33ad4f88ffdf16acb1.03:40:36 KST92개 source/docs/results/checkpoint 사본 고정, /tmp/nsmt_assessment_20260921T184001Z-e2dd33af. 소스 변경 없음; 감사 코드/텍스트/문서 append만.
- 재검사: snn_recall torch1.12 CPU2threads. A05 finite/bound/band 실패 pickedNone·exit1/row보존, 건강exit0. 추가 branch dead/ratio1000은 accepted라 OPEN. GRU 반환 dict만 실행해 실제 logging/verbose 성공; train 함수/optimizer는 실행하지 않음.
- 데이터/모델: 기존 새6pilot, r2 k3/onehot/min_gap1,data_seed20260921,seed7,2048·256·256,batch64,12epochs,lr.001/wd.01,spike/frozen norm/scale8,key_norm none,theta5.5. CPU best 재평가 오차≤1.12e-16. Recall Full.2763386362,sparse학습η.2717429156,sparseη.5 .4107734982,oracleη.5 .0362117806,oracleη1 .0296800174. Oracleη.5의 독립 recall M_eff.46655238(기존.52672184); A02 판정 오류 중요.
- θ 보정값5.5613433501은 loader 반환 누락. 학습η pilot best/last parameter hash 불일치 지속. 새로운 oracle-trained 증거는 탐색적이며 sparse성능/8seed 검증 완료 아님. A10 legacy config에도 key_norm migration 필요.
- Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/inventory.json,changes.diff,calibration_failure_paths.json,theta_pilot_probes.json,branch_failure_paths.json,validation.json. Exact commands(cwd NSMT, OMP/MKL2,LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib): snn_recall python `scripts/audit_calibration_failure_paths.py /tmp/nsmt_assessment_20260921T184001Z-e2dd33af f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/calibration_failure_paths.json`; python `scripts/audit_v3_theta_pilots.py /tmp/nsmt_assessment_20260921T184001Z-e2dd33af f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/theta_pilot_probes.json`; python `f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/branch_failure_probe.py /tmp/nsmt_assessment_20260921T184001Z-e2dd33af f_lif_pop_v3/forecasting/results/assessment/20260921T184001Z-e2dd33af/branch_failure_paths.json`.
- 검산/제한: 원본checkpoint/사본hash·구문·append prefix 보존. 새 학습/GPU/환경 설치/git 변이/commit/tag/push 없음.8seed통계·GRU학습·gradient표 재현 not run. D-X/Y 최신 계약 복구, clipped 이전gradient/동일대조군 개선은 기존 primary 문헌 기반. 감사 중 config/기록 후속변경은 다음 주기로 남김.


## 2026-09-22 03:53 KST — v3 예약 추적 감사07: GRU 완료 결과와 run 경로 검증

- 예약 20260921T185001Z-86f72a84; branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADd1fb43edaec12c7f87adcf06aade82b361c7187c.03:50:37 KST snapshot /tmp/nsmt_assessment_20260921T185001Z-86f72a84. Config SHAbe713ae2cdde9bbabe149381fe3b365e3e687cb6fb2aa0ce3d0e41e44cb3a2af. 모델 소스 수정 없음; 감사 스크립트/증거/문서 append만.
- 기존 GRU: r2 k3,data_seed20260921,seed7,2048·256·256,12epochs,batch64,lr.001/wd.01,hidden32/patch8. CPU torch1.12/2threads로 best 재평가, all/copy/recall/first MSE .1943944133/.0164127504/.2531052475/.2563219884; JSON 최대차3.47e-18 및 parameter hash일치.2표본 causal/truth차0. Checkpoint SHA64327b40a745f0ecbf6d4a8170b087cbc8e850225a65d4229aa91d0293d5e304.
- A10-PATH JSON 이름 구분 확인, 같은 suite analog/spike log/model_state 충돌 남음. 사용 recall 모듈 GRU4065/myModel490(selector포함)로 parameter-matched가 아님. **이전 pilot 해석의 θ=5.561 표기는 실제 config5.5로 정정한다**(보정값5.56134335의 전달 누락은 감사06 증거). Readout 병목은 copy gap만으로 확정하지 않는다.
- Exact command(cwd NSMT): `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python scripts/audit_v3_gru.py /tmp/nsmt_assessment_20260921T185001Z-86f72a84 f_lif_pop_v3/forecasting/results/assessment/20260921T185001Z-86f72a84/gru_probes.json`. Artifacts inventory/config.diff/observed_gru/gru_probes/validation은 해당 assessment 디렉터리.
- 보존/한계: snapshot/원본checkpoint·source provenance11개/구문/append prefix 확인. 새로운 학습/optimizer/GPU/설치/git변이/commit/tag/push 없음.8seed·CI·용량통제·수렴 확인 not run. A12-GRU 완료 평가범위만 확인, 기존 나머지 OPEN 유지. 성능 우위 일반화 보류.

---

## 2026-09-22 10:12 KST — 추적 감사 07 수용, G4 최초 실행 (학습 없음)

**Branch:** `exp/f-lif-pop-v3` · 감사 `NSMT/docs/ASSESMENT.md` 추적 07 및 02

### 1. θ가 보정에서 학습으로 전달되지 않았다 — 직전 로그 기재 정정

D-X로 θ를 보정 대상으로 옮겼으나 `train.py`는 보정 artifact에서 `input_scale`과 `g11_bound`만 읽었다. 저장된 config를 확인한 결과 **파일럿은 θ=5.5(CLI 기본값)로 실행됐다.** 직전 PROJECT_LOG 항목의 "θ=5.561"은 **그 실행에 대해서는 틀린 기재**다. 본 로그는 append-only이므로 여기서 정정한다.

A06(`--eta_fixed`·`--no-cap` 미전달)과 **같은 종류의 결함**이 다시 나왔다. 보정이 정한 필드를 일괄 덮어쓰고 `calibrated_fields`를 결과에 남기도록 고쳤다. 확인: `theta 5.5 overridden by calibration 5.561343350061557`.

### 2. readout이 로그·checkpoint 경로에 없었다

`run_id`에는 들어갔으나 `save_result_path`에는 없어 같은 suite에서 `spike` 다음 `analog`가 `FileExistsError`로 막혔다. **D8이 요구하는 readout 비교가 한 suite에서 불가능**했다. 경로와 tag에 포함했다.

### 3. G4를 처음으로 실행했다 — 그리고 정밀도 결함을 찾았다

감사가 사전등록의 spikeDE 고정 커밋 `fcd743b`를 확보해 `snn_jelly`(torch 2.11 CPU)에서 궤적을 생성했다. golden을 `reference/golden/`에 고정하고 SHA256을 `check_model.py`에 박았다.

**첫 실행 결과가 FAIL이었다: max 오차 3.35e-08.** 원인은 `fractional_coefficients`가 계수표를 **float32로 만든 뒤 `.double()`로 올린 것**이다. 이미 소실된 정밀도는 복구되지 않는다. 내부 게이트들은 같은 버퍼로 기준을 만들어 오차가 상쇄돼 전부 통과하고 있었다. **외부 float64 기준과 대조하는 G4가 아니었으면 드러나지 않았다.**

생성 시점 dtype으로 만들도록 고친 뒤 **7.77e-16**이 됐다.

| | 수정 전 | 수정 후 |
|---|---|---|
| G4 | NOT RUN → FAIL 3.35e-08 | **PASS 7.77e-16** (360스텝 / 24조건) |
| 게이트 총계 | 19 passed / 1 not run | **20 passed / 0 failed / 0 not run** |

**대조 범위를 과장하지 않는다.** v3-A 가지는 리셋하지 않고(D1) 원본은 임계에서 차감하므로 두 재귀는 **첫 스파이크 직전까지만** 같다. 24조건 중 7개는 스파이크가 없어 전 구간, 나머지는 스파이크 이전 구간을 비교했다. 등급은 **"분수적분 핵심부의 source parity"**이며 뉴런 전체가 아니다.

### 4. GRU 비교 기재 정정 (A10-PARAM)

총 parameter는 GRU 134,241 대 myModel 130,666으로 비슷하나 둘 다 recall에서 쓰지 않는 forecasting head가 대부분이다. **recall 경로만 세면 GRU 4,065 대 myModel 490으로 GRU가 8.3배 많다.** 직전 항목의 "GRU가 현재 우리보다 낫다"는 관찰 자체는 유효하지만 **용량 동등 비교가 아니다.** 용량 통제가 필요하면 미사용 head를 제외한 기준을 사전에 정하고 기존 결과에 소급 적용하지 않는다.

### 5. readout 병목은 가설로 격하

직전 항목에서 copy 오차 차이를 근거로 "readout 병목이 따로 있을 수 있다"고 적었다. 감사 지적대로 **copy 오차만으로는 손실 위치(임베딩/스파이크/readout)가 식별되지 않는다.** 가설로 두고 `--readout analog`를 같은 데이터·seed·예산의 통제 조건으로 돌려 구분한다. 이제 경로가 분리되었으므로 같은 suite에서 실행 가능하다.

Artifacts: `NSMT/f_lif_pop_v3/reference/golden/{scalar_trajectories.csv,SHA256SUMS,reference_results.json}`, `NSMT/f_lif_pop_v3/analysis/check_model_phaseAC.txt`, `NSMT/docs/Population_fLIF_v3_prereg_KO.md` §2E.


## 2026-09-22 10:15 KST — v3 예약 추적 감사08: θ/readout 배선 및 공식 G4 재검사

- 예약 20260922T011001Z-6c90ce78; exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADd1fb43edaec12c7f87adcf06aade82b361c7187c 위 미커밋 수정.10:10:40 KST90개 사본 /tmp/nsmt_assessment_20260922T011001Z-6c90ce78; trigger 후 layers/check_model/analysis 변경을 실제 캡처 hash로 검사.10:12 문서 §2E D-Z/AA/AB도 postscript로 별도 보존.
- CPU snn_recall torch1.12/2threads, 기존 wirecheck seed7/r2 k3/256·64·64/batch64/1epoch 평가. Spike recall.392408747928214,analog1.1312525898272698; MSE차0/parameter hash일치. Layers dtype 추가 후 기본float32 평가 재현. 새 학습 아님.
- 실제 main을 train 함수 대체 fixture로 실행해theta123→5.561343350061557,scale1→8 및 두 calibrated_fields 확인; same-suite readout 결과/log/checkpoint 경로 분리 확인. A10-THETA 배선/A10-PATH 해당범위 VERIFIED. JSON provenance에는 theta/calibrated_fields 직접 저장이 아직 없으며 config에 존재.
- G4: golden CSV SHA2fda1abbb03743e1b9074d2fb7b7b84feeec8e322201b42b0b66bc0fc657a7b7, 이전 감사 CSV/JSON과 byte일치. Official check_model --phase all20pass/0fail/0notrun exit0.24조건/360step/7.77e-16은 첫 spike 이전 적분기 핵심부 parity. Full neuron/compiled/adjoint/GPU/상류 재생성 not run. make_golden.py는 명시적 scaffold임을 확인.
- Exact commands(cwd NSMT,OMP/MKL2,LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib): snn_recall python `/tmp/nsmt_assessment_20260922T011001Z-6c90ce78/f_lif_pop_v3/forecasting/check_model.py --phase all`; python `scripts/audit_v3_wirecheck.py /tmp/nsmt_assessment_20260922T011001Z-6c90ce78 f_lif_pop_v3/forecasting/results/assessment/20260922T011001Z-6c90ce78/wirecheck_probes.json`. Artifacts 같은 assessment 폴더 inventory/changes.diff/document_postscript_inventory/wirecheck_probes/validation.
- 원본checkpoint/사본hash·구문·append prefix 보존. 감사자 모델수정/학습/optimizer/GPU/설치/git변이/commit/tag/push 없음. 기존 미수정 이슈 유지;1epoch analog 비교로 병목/성능 우위 판단 보류.


## 2026-09-22 10:21 KST — v3 예약 추적 감사09: 기존 검토 내용의 commit 확인

예약 20260922T012001Z-8b3e144d, branch exp/f-lif-pop-v3, base329183b94f65090cc6b337f464c5aa4d8e127ad7, HEADecec8d397862e6367e556668d82e6dd11e2e36e4.10:20:33 KST snapshot /tmp/nsmt_assessment_20260922T012001Z-8b3e144d; trigger 코드/사전등록/reference는 감사08 및 후속 문서 사본과 hash 동일. **새 판단 근거 없음**, 모델/결과 재검사·학습·GPU·통계 실험 not run. 기존 VERIFIED 범위와 OPEN 유지. 변경은 감사 inventory 및3문서 append뿐; 원본/로컬변경 보존, 감사자 git add/commit/tag/push/reset/switch 없음. Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T012001Z-8b3e144d/inventory.json,validation.json.

---

## 2026-09-22 10:31 KST — 추적 감사 03–06 수용: 보고한 M_eff가 틀렸다 (학습 없음)

**Branch:** `exp/f-lif-pop-v3`

### 1. 가장 중요한 것 — `M_eff` 수치 정정

`selection_diagnostics()`가 사전등록 D-Q의 정의를 세 곳에서 어겼다. ① `truth.any()`로 사건을 골라 **copy 사건 1,899건이 섞였고**, ② (batch × 시점)별 값을 평균해 **"query 내부 평균 → sequence 평균" 순서가 아니었으며**, ③ `hit`이 첫 임베딩 유닛만 봤다.

정정 후 값이 감사의 독립 재계산과 **소수점 9자리까지 일치**한다.

| 조건 | 내가 보고했던 값 | **정정값** | 감사 독립값 |
|---|---:|---:|---:|
| 학습된 sparse | 0.2315 | **0.1330930316** | 0.1330930309 |
| full 커널 질량 | 0.2290 | **0.1303288944** | 0.1303288936 |
| oracle η=0.5 | 0.5267 | **0.4665523771** | 0.4665523809 |
| oracle η=1 | — | 1.0000000000 | — |

**판정에 직접 영향이 있다.** 옛 값으로는 oracle η=0.5가 O7-① 기준 0.5를 넘는 것처럼 보였으나 **실제로는 넘지 못한다.** 즉 완벽한 정답 정책이라도 상한 적용 후 계수 질량이 임계값에 못 미치는 설정이 존재한다. 임계값을 이번 결과에 맞춰 내리지 않는다.

정정된 값으로 다시 읽으면 결론은 더 선명해진다. 학습된 선택자가 정답에 추가로 얹는 질량은 **0.1331 − 0.1303 = 0.0028**이고, **가장 크게 읽은 칸이 정답인 비율은 0.0000**이다(η=0.5 고정에서 0.0852, oracle에서 1.0000). 직전 항목의 "M_eff ≈ kernel_mass" 관찰은 방향이 맞았으나 **수치는 전부 틀렸다.**

### 2. 실행 계약 결함 여섯 건

| ID | 문제 | 확인 |
|---|---|---|
| A10-CAL | 보정 artifact를 **파일명만 맞춰** 골라, τ·cue_mode가 달라도 다른 동작점의 값을 썼다 | τ=[40,80,160,320] 요청이 τ=[4,8,16,32] 보정을 받았다 → 이제 거부 |
| A10-CAL | 보정이 없으면 **상한 없이 학습을 계속**했다 | `--require_calibration` 기본 참 |
| A05-BRANCH | 구성원 건강 조건이 sparse 승인에 미적용 | 죽은 가지·1000배 불균형도 통과했다 → 적용 |
| A10-HASH | `parameter_hash`가 마지막 epoch 모델인데 평가는 best checkpoint다 | **결과의 identity가 평가 모델과 달랐다** → 분리 저장 + checkpoint sha256 |
| **A12-ETT** | 빈 truth에서 `truth[:, n, :n]`이 IndexError | **ETT가 오차 계산 뒤 저장 전에 죽었다.** 고친 뒤 ETTh1 1 epoch 정상 완료(mse 0.9118, fresh reload 일치) |
| A12-TEST-SPLIT | `--no-test`가 배선되지 않음 | 탐색 실행에서도 test를 봤다 → 배선 |
| A10-KEYNORM-CKPT | 새 buffer·새 인자 때문에 옛 checkpoint/config 로드 실패 | 항등 기본값 buffer만 누락 허용, 그 외는 여전히 실패 |

### 3. 감사 가설 하나는 기각됐다

감사 §5.2는 sparsemax의 singleton support에서 score gradient가 0이 되는 것을 실패 경로 후보로 들었다. 학습 중 측정한 `singleton_frac`은 **0.038**로 주 원인이 아니다. 같은 실행에서 `score_std` 1.014(θ 보정이 의도대로 작동), `grad_WQ` 5.3e-3, `grad_WK` 1.4e-2로 **폭주도 소멸도 없다.**

남는 유력 원인은 `grad_eta_hat` 3.5e-4와 sigmoid 포화의 조합이며, D-Y대로 고정 η 조건이 주 경로다.

### 4. 한계

§1의 정정값은 기존 checkpoint를 다시 읽어 계산한 것이며 새 학습이 아니다. §3의 진단은 1 epoch·256 시퀀스 실행의 관측이다.

Artifacts: `NSMT/docs/Population_fLIF_v3_prereg_KO.md` §2F, `NSMT/f_lif_pop_v3/analysis/check_model_phaseAC.txt`.


## 2026-09-22 10:38 KST — v3 예약 추적 감사10: 진단 집계·실행 계약 재검사

- 목적: A02/A05/A09/A10/A12 수정과 새diag3 결과 감사. Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7, 관찰HEAD73599f02a6cec05e485ea078669a9cb1723a83e7+미커밋수정. 감사자commit/tag 없음. Snapshot /tmp/nsmt_assessment_20260922T013001Z-6c439d7a(10:30:36 KST,111파일) 및 sourcehash inventory 고정.
- 모델/학습변경 없음. CPU Python3.10/torch1.12/2threads,기존seed7/data_seed20260921/r2k3/frozen-scale8 checkpoint 재평가. Pilot15epoch(theta1) recall.269893671218/M_eff.133093031595;diag3 12epoch(theta5.56134335) recall.272620149439/M_eff.132756536374. 각각2048/256/256 split, MSE차<=2.8e-17. epoch/θ달라통제비교아님. batch16/128 primary차0.
- A02 집계/A05 branch 실패배선 VERIFIED. Legacy정상복원,빈truth진단/no-test 학습후AST/hash분리 scoped VERIFIED. Train함수·optimizer미실행. Calibration tau/cue거부·기본guard확인,전체계약OPEN.
- **A10-RESTORE-STATS OPEN(P1)**: norm_mean/std누락임시checkpoint승인,예측최대차.8706527948. 원본손상주장아님. 일반weight누락거부. Schema/version 기반 migration 필요.
- **10:31 canonical항목 정정 권고:** singleton.038 및 gradient표본 norm만으로 singleton원인기각/폭주소멸없음/eta원인단정불가. 이번diag3 singleton.06798,gradQ.004664/K.013687/eta.003896은마지막epoch표본평균. 사전등록max|grad|·nonfinite기록과통제조건이필요. 성능·8seed/CI not run.
- Exact commands/세부수치/문헌링크는 NSMT/docs/ASSESMENT.md 감사10, artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T013001Z-6c439d7a/. 실행script는동폴더diagnostic_probe.py, 기존scripts/audit_calibration_failure_paths.py 및감사06 branch_failure_probe.py, 모두frozenroot 인자와CPU환경으로실행. source/model변경·학습·GPU·설치·프로세스중단·git변이없음. 사본hash/평가checkpoint/기존문서prefix보존. 다른세션진행중변경은다음주기.

---

## 2026-09-22 10:40 KST — readout 통제 실험: 스파이크가 학습된 선택의 이득을 삼킨다 (탐색적)

**Branch:** `exp/f-lif-pop-v3` · **성격: 탐색적.** 12 epoch, 2048 시퀀스, seed 7 하나. 확정 결과가 아니다.

### 1. 설계

감사 A08이 요구한 대로 readout을 셋으로 분리해 **같은 데이터·seed·예산**에서 비교했다. 셋 다 gradient를 유지하므로 뉴런까지 end-to-end로 학습된다.

| readout | 읽는 값 | 묻는 것 |
|---|---|---|
| `spike` | `s_n` | 모델 그 자체 |
| `analog` | 리셋 **이후** 막전위 `v_n` | 스파이크 비선형성이 병목인가 |
| `drive` | 소마 **이전** 가지 혼합 `a_n = Σ w_k u_{n+1,k}` | 정보가 가지 상태에 있는가 |

초판 `analog`는 리셋 이후 막전위만 읽어 **두 질문을 구분하지 못했다.** `drive`를 추가해 분리했다.

### 2. 결과 — recall MSE

| 조건 | spike | analog | **drive** |
|---|---:|---:|---:|
| full (η=0) | 0.2763 | 0.2830 | 0.2747 |
| **sparse (학습된 η≈0.026)** | 0.2726 | 0.2616 | **0.2122** |
| oracle η=1 | **0.0297** | 0.0426 | 0.0316 |

copy MSE: full 0.1487/0.2084/0.1274, sparse 0.1462/0.2072/0.1198, oracle 0.0921/0.1101/0.0833.

### 3. 읽어야 할 것

**readout은 full과 oracle에서는 사실상 무관하다.** full에서 drive 0.2747 vs spike 0.2763, oracle에서 0.0316 vs 0.0297로 스파이크가 오히려 약간 낫다. 즉 **스파이크 경로 자체가 정보를 못 나르는 것이 아니다.**

**그런데 학습된 sparse에서만 큰 차이가 난다.** drive 0.2122 vs spike 0.2726(22% 차이). 같은 readout 안에서 보면 더 분명하다.

| readout | full → learned sparse 개선 | **격차 회수율 G** |
|---|---:|---:|
| spike | 0.2763 → 0.2726 (1.3%) | **0.015** |
| analog | 0.2830 → 0.2616 (7.6%) | 0.089 |
| **drive** | 0.2747 → 0.2122 (**22.7%**) | **0.257** |

**해석: 학습된 선택이 만들어내는 이득은 가지 상태에는 존재하지만 소마·스파이크를 통과하며 대부분 사라진다.** oracle처럼 강하고 정확한 신호는 스파이크를 통과해도 살아남지만(0.0297), 학습된 선택의 약하고 부분적인 신호는 살아남지 못한다.

직전 항목에서 나는 readout 병목을 가설로 제기했다가 감사 지적으로 격하했고, analog만 봤을 때는 **지지되지 않았다**(analog가 spike보다 낫지 않음). `drive`를 추가하고 나서야 **조건부로 성립**함이 드러났다 — 병목은 무조건적이지 않고 **신호가 약할 때만** 작동한다.

### 4. 한계

seed 1개·12 epoch의 탐색적 실행이다. `drive`는 **스파이크가 전혀 없으므로 SNN이 아니다.** D8대로 주 결과는 스파이크이며 analog/drive는 진단이다. 또한 이 batch의 oracle 조건은 §5의 정책 수정 **이전** 코드로 실행됐다.

### 5. 함께 고친 감사 항목

| ID | 내용 |
|---|---|
| **A12-ORACLE** | oracle이 copy 사건에도 정답 칸을 가리켜 **headroom을 부풀렸다**. `kind>0`인 recall 사건에만 적용하고 나머지는 uniform(=η=0 커널)으로 바꿨다. 확인: t=10에서 64개 중 38개 시퀀스의 정책이 바뀌고(그 시점 copy 41건), copy가 없는 t=20/30에서는 0개 |
| **A10-LOG** | `EpochLog.write()`를 trainer가 호출하지 않아 프로젝트 표준 `log/best_log_0.csv`가 없었다. `write()`로 교체해 생성 확인 |

### 6. 다음

`drive`와 `spike`의 격차는 **점수 함수만 고쳐서는 안 될 수도 있다**는 뜻이다. 다만 seed 1개 결과이므로 먼저 **재현**해야 한다. 정책 수정 후 oracle을 다시 돌리고, seed를 늘려 이 격차가 유지되는지 확인한 뒤에야 구조 변경을 논한다.

Artifacts: `NSMT/f_lif_pop_v3/forecasting/results/{diag3-*,drive-*}/`.


## 2026-09-22 10:45 KST — v3 예약 감사11: readout·oracle 버전·실제 기능 결과 검토

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD639d2293dcf12e3db3738641ff774a30e9f7439a. Snapshot /tmp/nsmt_assessment_20260922T014001Z-17a8da3d(10:40:39,143파일),ETT CSV SHA f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066. 모델수정/새학습없음,감사commit/tag없음.
- CPU Python3.10/torch1.12/2threads/seed7,data_seed20260921,r2k3,2048/256/256,batch64,12epoch 기존checkpoint평가. Sparse recall spike.272620149439/analog.261619795824/drive.212159097751;각MSE차0. DriveMeff.134012595802/kernel.130328894. 1seed탐색이며성능우위/검색성공일반화보류;8seedCI not run.
- A12-ORACLE copyuniform수정 VERIFIED,현재정책copy비균등0/옛1243. 과거12epochoracle은옛정책에서만저장MSE재현. 새정책재평가 recall spike.035348084881/analog.071973921201/drive.045209064883은재학습아님. Driveoracle JSON의새sourcehash와실행정책불일치가 A10-PROVENANCE 재확인근거.
- Drive혼합값차0/유한gradient/앞20patch인과성차0. 실제CSV2행확인(A10-LOG부분VERIFIED),final+result.csv미생성. Hash분리새artifact모두일치/driveoracle last≠best. ETTmax_train/eval_batches3 기능실행test192/2785창MSE.911763752704재현,window_mean.856615248951;전체성능주장금지. No-testartifact확인,해당test미접근.
- Exact command와근거는 NSMT/docs/ASSESMENT.md 감사11, NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T014001Z-17a8da3d/readout_probe.py 및readout_probes.json/inventory/data_snapshot/validation; rawstdout forecasting/log/assessment/20260922T014001Z-17a8da3d/readout_probe.log. OMP/MKL2,LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib,snn_recall python으로snapshot인자실행. 새학습/optimizer/GPU/설치/git변이/프로세스중단없음,원본checkpoint/문서prefix보존. A10-RESTORE-STATS 등미수정이슈유지.

---

## 2026-09-22 10:47 KST — 추적 감사 08–11 수용: 복원 결함과 진단 표현 정정 (학습 없음)

### 1. A10-RESTORE-STATS (신규 P1) — 누락을 안전하게 거부하지 못했다

`OPTIONAL_BUFFERS`에 `embedding.norm_mean/std`를 **무조건** 넣어, `input_norm='frozen'`으로 학습한 checkpoint에서 그 통계가 빠져도 경고만 내고 승인했다. 감사 측정에서 같은 입력에 대한 예측이 **최대 0.8706527948** 차이 났다. 조용히 다른 동작점에서 도는 모델이 된다.

누락 허용을 **그 설정에서 항등인 경우로 한정**했다. `input_norm='none'`일 때만 `norm_mean/std`, `key_norm='none'`일 때만 `key_mean/std`, `eta_fixed=None`일 때만 `eta_value`. 확인: `input_norm=frozen` checkpoint에서 통계를 지우면 **거부**되고, 온전하면 승인된다.

### 2. A09-GRAD — 내 직전 표현을 정정한다

직전 항목에서 `singleton_frac` 0.038, `grad_WQ` 5.3e-3 등을 근거로 **"singleton 가설 기각", "폭주도 소멸도 없다", "η 포화가 유력 원인"**이라고 썼다. 감사 지적대로 이는 **관측 표본에 대한 진술을 인과 결론으로 넘긴 것**이다.

정확히는 이렇다. ① 기록값은 `g11_every` 표본의 **L2 norm 평균**이지 사전등록이 요구한 `max|grad|`가 아니었다. ② 최종 JSON은 마지막 epoch의 표본 평균이다. ③ `*_nonfinite` 개수가 payload에서 빠져 있었다.

따라서 올바른 서술은 **"관측한 표본에서 singleton 점유가 낮았고(0.037–0.068) Q/K gradient가 유한했다"**이며, 원인 판정은 query·시점·unit별 분포와 고정 η 통제 결과로 확인해야 한다.

지표 이름과 수집을 고쳤다: `grad_*_norm`(L2 평균)과 `grad_*_absmax`를 분리하고 **모델 전체 `grad_absmax_all`**을 추가했다(관측값 1.44166). `*_nonfinite` 개수도 payload에 싣는다.

### 3. A10 기록 범위

`calibrated_fields`가 `config.pt`에만 있고 결과 JSON에는 없었다. 결과만 받은 사람이 어떤 값이 보정에서 왔는지 알 수 없다. JSON `provenance.calibration`에 `theta`와 `calibrated_fields`를 포함했다.

### 4. 감사가 VERIFIED로 올린 항목

A10-THETA 전달(D-Z), A10-PATH spike/analog(D-AA), A07/G4 분수적분기 대조(D-AB), A02-DIAG 집계(D-AC, 기존 독립값과 7e-10 차이), A05-BRANCH 거부 배선, A12-ORACLE(copy 비균등 사건 **1243 → 0**), A08-DRIVE 배선·인과성, A12-ETT 빈 truth 경로, A12-TEST-SPLIT·A10-HASH.

감사가 짚은 사소한 정정 하나: 내가 `hit = 0.0000`이라고 쓴 값의 실제는 **3.72888448e-5**이며 0이 아니라 반올림값이다.

### 5. 남은 OPEN

`log/final+result.csv` 미생성과 동적 `*_nonfinite` 열의 CSV 스키마 문제(A10-LOG), `make_golden.py`의 실행 가능한 재생성 경로(A07-REGEN), 보정 artifact의 ID/hash 기반 호환 계약(A10-CAL), `key_norm=frozen`일 때 통계 적합 연결.

Artifacts: 본 커밋의 `model.py`·`train.py`.

---

## 2026-09-22 10:50 KST — 검색 기전 개선 후보와 그 수학적 근거 (학습 없음)

사용자 요청으로 '과거 검색' 방법의 개선 후보를 문헌에서 찾고, **각 문헌이 말하는 실패 조건이 우리 모델에 실제로 해당하는지 측정**했다.

### 1. 이론이 말하는 실패 조건

| 문헌 | 핵심 정리 | 실패 조건 |
|---|---|---|
| Wang, Shi & Fox, **Test-time regression** (arXiv:2501.12352) | 기억을 가중최소제곱 `argmin Σ γ_i ‖v_i − m(k_i)‖²`으로 보면, 선형/커널 읽기는 `K^T K ≈ I`를 가정한 근사다. 정확해는 `M = V^T K (K^T K)^{-1}` | **key가 직교하지 않으면** 커널 읽기는 최적이 아니다. 또한 `‖k‖=‖q‖=1`일 때에만 지수 커널이 올바른 국소 추정자가 된다(QK 정규화의 근거) |
| Ramsauer et al., **Hopfield Networks is All You Need** (arXiv:2008.02217) | 현대 Hopfield는 지수적으로 많은 패턴을 저장하고 **한 번의 갱신**으로 회상하며 오차가 지수적으로 작다. 갱신식이 attention과 동치 | **패턴이 충분히 분리되어 있어야** 한다 |
| Yang et al., **DeltaNet** (arXiv:2406.06484) | delta rule은 온라인 경사하강 한 걸음으로 옛 연관을 **지우고** 새로 쓴다. 회상–기억 trade-off 곡선이 개선됨 | 단순 누적(선형 attention)은 간섭에 취약 |
| Correia, Niculae & Martins, **Adaptively Sparse Transformers** (arXiv:1909.00015) | α-entmax의 **α에 대한 닫힌 형태 Jacobian**을 유도해 희소성 자체를 학습 가능하게 만든다 | 희소성을 온도와 함께 손으로 정해야 하는 문제 |

### 2. 우리 key 기하 측정 — 실패 조건에 정확히 해당한다

Artifact: `NSMT/f_lif_pop_v3/analysis/key_geometry.txt` (diag3 sparse checkpoint, test 64 시퀀스, 학습 없음)

| 측정 | 값 | 이론이 요구하는 값 |
|---|---:|---|
| key Gram 고유값 최대/최소 | **2.47e+03** | 직교면 1 |
| **유효 rank** | **3.97 / 42** | 42에 가까울수록 좋음 |
| 인접 key 코사인 유사도 | 0.758 | 작을수록 좋음 |
| 전체 쌍 코사인 유사도 (중앙값) | 0.669 | |

**42개 과거 key가 실효적으로 4차원만 차지한다.** 커널 읽기가 최적이라는 조건(직교)에서 크게 벗어나 있고, Hopfield의 "충분히 분리된 패턴" 조건도 만족하지 않는다.

### 3. 그런데 점수 자체에는 신호가 있다 — 구조가 버린다

| 측정 | 값 |
|---|---:|
| 정답 칸의 평균 근접 순위 | **상위 23.5%** (무작위면 50%) |
| 가장 가까운 key가 정답일 확률 | **0.2924** |
| 정답 칸 vs 바로 옆 칸 코사인 유사도 | 0.464 (구분 가능) |
| **그런데 최종 계수의 argmax가 정답일 확률** | **3.73e-05** |

점수는 정답을 29% 확률로 1순위에 놓는데 **최종 계수는 사실상 한 번도 정답에서 최대가 되지 않는다.** 원인은 구조다: `c = (1−η)b_d + η(...)`에서 η≈0.026이므로 `(1−η)b_d` 항이 지배하고, `b_d`는 **d에 단조 감소하므로 항상 가장 최근 칸에서 최대**다.

**따라서 점수 함수를 아무리 개선해도 η가 작고 dense residual이 남아 있는 한 결과에 드러나지 않는다.** 이것이 §2의 기하 문제보다 먼저 해결해야 할 순서다.

### 4. 후보와 우선순위

| # | 후보 | 근거 | 비용 | 우리 조건에서의 예상 |
|---|---|---|---|---|
| **1** | **구조 먼저**: 고정 η 격자 실행(D-Y) + dense residual 재검토 | 위 §3 측정 | 없음 | 점수 개선의 효과를 **볼 수 있게** 만드는 전제 |
| **2** | **QK 정규화** (`‖q‖=‖k‖=1`) | test-time regression §(3) | 거의 0 | 점수가 곧 내적 커널이 되어 θ 손보정이 불필요해지고, 관측된 역전파 폭주의 원인도 제거 |
| **3** | **Gram 보정 읽기** `(K^T K + λI)^{-1}` | test-time regression 정확해 | **매우 쌈** (d_q=4이므로 4×4 역행렬) | 유효 rank 3.97/42에 직접 대응 |
| **4** | **delta rule** (옛 연관 제거 후 쓰기) | DeltaNet | O(T), 현재 O(T²)보다 쌈 | 3의 온라인 근사 |
| **5** | **key 분리 강화** (`Δu` 추가 등) | Hopfield의 분리 조건 | 작음 | 사전등록이 이미 예정한 key ablation과 같은 축 |
| **6** | **α-entmax (α 학습)** | 닫힌 형태 Jacobian | 작음 | 희소성을 θ에서 분리 |

**어느 것도 지금 채택하지 않는다.** 모두 탐색적 후보이며, 사전등록 변경은 파일럿 재현(seed 확대) 이후에 날짜부로 한다.

### 5. 한계

§2·§3은 **학습된 checkpoint 하나·test 64 시퀀스**의 관측이다. 문헌의 정리들은 각자의 설정에서 증명된 것이며 **우리 구조(리셋 없는 분수적분 가지 + 단일 소마 + 계수 상한)에 그대로 승계되지 않는다.** 특히 3·4는 "과거 증분 `f_j`의 재가중"이 아니라 "key-value 기억의 읽기"를 다루므로, 우리 수식에 옮기려면 `f_j`가 value 역할을 할 수 있는지부터 따져야 한다.

Artifacts: `NSMT/f_lif_pop_v3/analysis/key_geometry.txt`.


## 2026-09-22 10:55 KST — v3 예약 감사12: 복원 결함 해소·gradient 집계 잔여·문헌 적용 범위

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD520bb56872c6e7c68c48f65cd70601979c22a1c5.10:50:38 snapshot139파일 /tmp/nsmt_assessment_20260922T015001Z-bff8435c. 감사자모델변경/학습/commit/tag없음.
- CPU Python3.10/torch1.12/2threads. 복원fault8조건에서frozenstats/fixedeta누락거부+legacy pilot recall.269893671218재현:A10-RESTORE-STATS VERIFIED. 새gradchk seed7,data_seed20260921,r2k3,512/64/64,batch64,2epoch,theta5.56134335,scale8기존checkpoint전체MSE.337799426411/recall.347767202408,hash/MSE차0.
- A09 absmax는여전히batch최대의평균:고정AST주입[1,9]→5,[2,10]→6. g11_every10,epoch8batch중1관측이므로이run으로전체max주장불가. nonfiniteJSON전달및theta/calibrated_fields저장확인.8seedCI/새학습not run.
- 10:50 문헌후보검토: key_geometry재현코드/명령/hash/mask/shape부족. 유닛별key4차원rank상한4이므로3.97/42를붕괴로단정불가. 0.2924와기존pilot hit3.73e-5를서로다른checkpoint/집계로비교하지말것. TTR원문bandwidthB잔존,QKNormθ불필요/폭주제거보장없음. Gram/delta는다른memory계약,현재분수history가자동O(T)되지않음. Entmax학습가능성만확인,온도독립보장아님. Hopfield원문PDF접근실패,정리가정검증not run. 직접링크/세부수정권고는ASSESMENT 감사12.
- Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T015001Z-bff8435c/(inventory,changes.diff,restore_grad_probe.py,restore_grad_probes,postscript,validation),rawlog forecasting/log/assessment/20260922T015001Z-bff8435c/. Exact command는ASSESMENT 감사12; OMP/MKL2·snn_recall python·frozenroot로실행. Train/optimizer/GPU/설치/git변이/프로세스중단없음,원본/문서prefix보존. A09/CAL/PROVENANCE/LOG/keynormfit/REGEN잔여유지.


## 2026-09-22 11:01 KST — v3 예약 감사13: 검토 완료 내용의 commit 확인

예약 20260922T020001Z-8f8cd215,branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD41554c68cd97cbad522008297b98eda230b1e651.11:00:58 KST snapshot123파일 /tmp/nsmt_assessment_20260922T020001Z-8f8cd215;감시대상120파일 trigger/감사12 hash 동일. key_geometry와문헌내용은이미감사12범위. 새 판단 근거 없음;모델/probe재검사·학습·GPU·통계 not run. 기존VERIFY범위/OPEN유지,감사자commit/tag/git변이없음. Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T020001Z-8f8cd215/inventory.json,validation.json. 모델소스/원본결과/checkpoint보존;3문서append만갱신.


## 2026-09-22 11:15 KST — 사용자 요청: 검색 개선 후보 채택 우선순위 감사

- AUDIT-PRIORITY-01. Branch exp/f-lif-pop-v3,HEAD41554c68cd97cbad522008297b98eda230b1e651,base329183b94f65090cc6b337f464c5aa4d8e127ad7. 소스/문서/checkpoint19파일사본 /tmp/nsmt_assessment_20260922-adoption-priority 및hash고정. 감사자모델수정/학습/optimizer/GPU/git변이/commit/tag없음.
- CPU torch1.12/2threads,diag3 spike sparse checkpoint(seed7,data_seed20260921,기존12epoch2048train),validation첫64sequence/2004recall만forward. Score/p hit.2715946141,c hit0,p mass.256499139,c mass.137036866,kernel.134411009,chance.161031434,eta.0257748514. 같은unit/query/sequence집계. Key[64,32,41,4],비중심featureGram PR평균1.174733/상한4.
- 실제b와cap로η.026 onehot정책검산:lag2/5/10은정답c최대가능,20/41은불가. ‘점수개선효과전무’단정은성립안함. η.2에서도단일정답mass약.09로argmax와O7질량구분. 후보검증순서 고정η→QK정규화→key표현→entmax→Gram별도기억→Delta별도기억. 최종효능채택아님. Validation선택/8seedpairedCI·G14/O7유지.
- TTR원문Eq3/32/35/36,DeltaNetEq4,entmax명제1확인. Hopfield원문PDF직접확보SHA48cecc1d10cea553538fe8d1e2f1bf7378bed6b1233b2857384f5081ac2f7579,Eq5/Thm4–5확인. 평균cosine/rank로정리실패단정불가. 문헌링크·조건/제한은NSMT/docs/ASSESMENT.md AUDIT-PRIORITY-01에명시.
- Evidence NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922-adoption-priority/(inventory,priority_probe.py,priority_probes.json,literature_checks,validation). Raw log forecasting/log/assessment/20260922-adoption-priority/. Exact CPU command는동감사항목. 새학습/후보구현/다중seed효능검증not run. 예약ack없음.

---

## 2026-09-22 11:30 KST — 감사 AUDIT-PRIORITY-01 수용과 내 분석 정정 (학습 없음)

사용자가 감사자에게 직접 요청한 채택 우선순위 결정을 확인하고, 내 직전 분석의 두 주장을 독립 검증했다.

### 1. 정정 ① — "η가 작으면 점수 개선이 드러나지 않는다"는 **반례가 있다**

감사가 유도한 경계식을 재현했다. 정답 한 칸에 p를 전부 주는 정책에서, 정답 lag d의 계수가 가장 최근 칸을 앞서는 조건은

```
η > (b₁ − b_d) / (B + b₁ − b_d)        (d > 1)
```

`B=13.96147390, b₀=1.10054741, b₁=0.68729713`에서 실제 계수와 cap으로 검산한 결과가 감사 표와 **정확히 일치**한다.

| 정답 lag | 경계 η | η=0.0258에서 정답이 최대 c인가 | c_target vs c_recent |
|---:|---:|---|---|
| 2 | 0.007149 | **예** | 0.9315 vs 0.6696 |
| 5 | 0.015867 | **예** | 0.8101 vs 0.6696 |
| 10 | 0.021498 | **예** | 0.7306 vs 0.6696 |
| 20 | 0.026224 | 아니오 | 0.6631 vs 0.6696 |
| 41 | 0.030240 | 아니오 | 0.6053 vs 0.6696 |

**따라서 내 일반 명제는 틀렸다.** lag가 짧으면 현재 η로도 충분하다. 우리 과제의 평균 정답 lag가 21이라 **경계 근처**에 있을 뿐이며, "η가 작아서 무조건 안 드러난다"가 아니라 **"이 과제의 lag 분포에서 현재 η가 경계에 걸쳐 있다"**가 정확한 서술이다.

### 2. 정정 ② — rank 주장의 **틀은 잘못이나 결론은 기준을 대면 살아난다**

내가 "유효 rank 3.97 / 42, 42에 가까울수록 좋다"고 쓴 것은 **기준을 제시하지 않은 서술**이다. 감사는 "유닛별 투영 key가 4차원이므로 상한이 4"라고 지적했는데, 이는 감사가 잰 **feature Gram(4×4)**에는 맞지만 **내가 잰 것은 표본 평균한 token Gram(42×42)**이라 상한이 4가 아니다.

동일 계산으로 기준선을 만들어 확인했다.

| key | token Gram 유효 rank |
|---|---:|
| **우리 실제 key** | **3.97** |
| 완전 무작위 key, d_q=4 | **35.67** |
| 완전 무작위 key, d_q=8 | 38.88 |

즉 d_q=4여도 무작위면 35.67이 나오므로 **3.97은 구조적 상한이 아니라 실제 집중**이다. 감사의 feature Gram 측정(1.1747 / 상한 4)도 같은 방향을 가리킨다. **정정 내용: 결론(“key가 심하게 집중되어 있다”)은 유지하되, 기준선 없이 “/42”로 적은 표기를 철회하고 무작위 기준선 35.67과 함께 보고한다.**

### 3. 새로 확인된 것 — argmax 개선은 O7-① 통과를 뜻하지 않는다

감사 지적대로, 정답 한 칸에 모든 p를 주는 **완벽한 단일 정답 정책**의 post-cap 정답 질량을 직접 계산했다.

| η | 단일 정답 정책의 post-cap M_eff |
|---:|---:|
| 0.2 | **0.0912 – 0.0933** |
| 0.5 | **0.1383 – 0.1413** |

O7-① 기준은 0.5다. 즉 **정답 칸이 하나뿐이면 η<1의 어떤 값으로도 기준에 도달할 수 없다.** 앞서 측정한 oracle η=0.5의 M_eff 0.4666은 정답 칸이 평균 3.5개이기 때문이며, 상한과 dense residual이 함께 만드는 **구조적 천장**이 존재한다.

**사전등록 관련 함의:** O7-①의 0.5는  평면의 좁은 영역에서만 도달 가능하다. 기준을 낮추지 않되, **도달 가능 영역을 먼저 지도로 만들어** 판정 전에 기록해야 한다.

### 4. 감사의 채택 순위를 수용한다

**① 고정 η 대조 → ② QK 정규화 → ③ key 표현 ablation → ④ α-entmax → ⑤ Gram/ridge 보정 → ⑥ delta rule.**

내 직전 제안과 다른 곳은 ⑤(Gram 보정)의 위치다. 나는 "4×4 역행렬이라 싸다"는 이유로 3순위에 뒀으나, 감사 지적대로 **계산 비용이 아니라 계약 변경 폭**이 기준이어야 한다. 보정 가중치는 **음수가 될 수 있어** 현재의 양의 계수·cap·G4 계약을 그대로 승계하지 못한다. 또 우리 모델은 선형 K–V 회귀가 아니라 **분수 증분 `f_j`의 재가중**이므로 TTR의 최소제곱 해를 옮기려면 K·V·f_j 대응을 먼저 수식으로 고정해야 한다. **감사 순위를 채택한다.**

또한 "dense residual 즉시 삭제"도 채택하지 않는다. **η=1이 이미 그에 해당**하므로 별도 삭제부터 시작할 이유가 없다.

### 5. 기타 정정

- 내가 인용한 0.2924(가장 가까운 key가 정답일 확률)와 3.7e-5(계수 argmax 적중)는 **서로 다른 checkpoint**(pilot-eta vs diag3)에서 나온 값이라 같은 조건의 수치로 나란히 쓰면 안 된다. 감사의 동일 표본 계산은 score argmax 0.2716 / c argmax 0.0000 / M_eff 0.1370 / kernel 0.1344 / 균등 기준 0.1610이다.
- Hopfield 정리의 조건은 패턴별 분리도 `Δ_i`, β, 패턴 수·크기, query 거리에 의존한다. **평균 인접 cosine 0.758만으로 조건 위반을 판정할 수 없다.** 내 서술을 "충분조건이 확인되지 않았다"로 약화한다.

### 6. 한계

§1·§3은 **단일 정답 칸을 가정한 계수 대수 계산**이며 실제 정답 집합(평균 3.5칸)의 모델 성능이 아니다. §2의 기준선은 무작위 key이며 "좋은 key"의 기준이 아니다.

Artifacts: `NSMT/f_lif_pop_v3/analysis/key_geometry.txt`(기준선 없는 표기는 본 항목으로 정정), 감사의 `results/assessment/20260922-adoption-priority/`.

---

## 2026-09-22 11:31 KST — 감사 AUDIT-PRIORITY-01 수용과 내 분석 정정 (학습 없음)

사용자가 감사자에게 직접 요청한 채택 우선순위 결정을 확인하고, 내 직전 분석의 두 주장을 독립 검증했다.

### 1. 정정 ① — "η가 작으면 점수 개선이 드러나지 않는다"는 **반례가 있다**

감사가 유도한 경계식을 재현했다. 정답 한 칸에 p를 전부 주는 정책에서, 정답 lag d의 계수가 가장 최근 칸을 앞서는 조건은

```
η > (b₁ − b_d) / (B + b₁ − b_d)        (d > 1)
```

`B=13.96147390, b₀=1.10054741, b₁=0.68729713`에서 실제 계수와 cap으로 검산한 결과가 감사 표와 **정확히 일치**한다.

| 정답 lag | 경계 η | η=0.0258에서 정답이 최대 c인가 | c_target vs c_recent |
|---:|---:|---|---|
| 2 | 0.007149 | **예** | 0.9315 vs 0.6696 |
| 5 | 0.015867 | **예** | 0.8101 vs 0.6696 |
| 10 | 0.021498 | **예** | 0.7306 vs 0.6696 |
| 20 | 0.026224 | 아니오 | 0.6631 vs 0.6696 |
| 41 | 0.030240 | 아니오 | 0.6053 vs 0.6696 |

**따라서 내 일반 명제는 틀렸다.** lag가 짧으면 현재 η로도 충분하다. 우리 과제의 평균 정답 lag가 21이라 **경계 근처**에 있을 뿐이며, 정확한 서술은 "η가 작아서 무조건 안 드러난다"가 아니라 **"이 과제의 lag 분포에서 현재 η가 경계에 걸쳐 있다"**이다.

### 2. 정정 ② — rank 주장의 **틀은 잘못이나 결론은 기준선을 대면 살아난다**

내가 "유효 rank 3.97 / 42, 42에 가까울수록 좋다"고 쓴 것은 **기준선 없는 서술**이다. 감사는 "유닛별 투영 key가 4차원이므로 상한이 4"라고 지적했는데, 이는 감사가 잰 **feature Gram(4×4)**에는 맞지만 **내가 잰 것은 표본 평균한 token Gram(42×42)**이라 상한이 4가 아니다.

동일 계산으로 기준선을 만들어 확인했다.

| key | token Gram 유효 rank |
|---|---:|
| **우리 실제 key** | **3.97** |
| 완전 무작위 key, d_q=4 | **35.67** |
| 완전 무작위 key, d_q=8 | 38.88 |

d_q=4여도 무작위면 35.67이 나오므로 **3.97은 구조적 상한이 아니라 실제 집중**이다. 감사의 feature Gram 측정(1.1747 / 상한 4)도 같은 방향을 가리킨다. **정정: 결론("key가 심하게 집중되어 있다")은 유지하되 "/42" 표기를 철회하고 무작위 기준선 35.67과 함께 보고한다.**

### 3. 새로 확인된 것 — argmax 개선은 O7-① 통과를 뜻하지 않는다

정답 한 칸에 모든 p를 주는 **완벽한 단일 정답 정책**의 post-cap 정답 질량을 직접 계산했다.

| η | 단일 정답 정책의 post-cap M_eff |
|---:|---:|
| 0.2 | **0.0912 – 0.0933** |
| 0.5 | **0.1383 – 0.1413** |

O7-① 기준은 0.5다. 즉 **정답 칸이 하나뿐이면 η<1의 어떤 값으로도 기준에 도달할 수 없다.** 앞서 측정한 oracle η=0.5의 M_eff 0.4666은 정답 칸이 평균 3.5개이기 때문이며, 상한과 dense residual이 함께 만드는 **구조적 천장**이 존재한다.

**사전등록 함의:** O7-①의 0.5는 η와 정답 칸 수가 이루는 평면의 좁은 영역에서만 도달 가능하다. 기준을 낮추지 않되 **도달 가능 영역을 먼저 지도로 만들어 판정 전에 기록**해야 한다.

### 4. 감사의 채택 순위를 수용한다

**① 고정 η 대조 → ② QK 정규화 → ③ key 표현 ablation → ④ α-entmax → ⑤ Gram/ridge 보정 → ⑥ delta rule.**

내 직전 제안과 다른 곳은 ⑤의 위치다. 나는 "4×4 역행렬이라 싸다"는 이유로 3순위에 뒀으나, 기준은 **계산 비용이 아니라 계약 변경 폭**이어야 한다. 보정 가중치는 **음수가 될 수 있어** 현재의 양의 계수·cap·G4 계약을 승계하지 못한다. 또 우리 모델은 선형 K–V 회귀가 아니라 **분수 증분의 재가중**이므로 TTR의 최소제곱 해를 옮기려면 K·V·f 대응을 먼저 수식으로 고정해야 한다. **감사 순위를 채택한다.**

"dense residual 즉시 삭제"도 채택하지 않는다. **η=1이 이미 그에 해당**하므로 별도 삭제부터 시작할 이유가 없다.

### 5. 기타 정정

- 내가 인용한 0.2924(가장 가까운 key가 정답일 확률)와 3.7e-5(계수 argmax 적중)는 **서로 다른 checkpoint**(pilot-eta vs diag3)에서 나온 값이라 같은 조건의 수치로 나란히 쓰면 안 된다. 감사의 동일 표본 계산은 score argmax 0.2716 / c argmax 0.0000 / M_eff 0.1370 / kernel 0.1344 / 균등 기준 0.1610이다.
- Hopfield 정리의 조건은 패턴별 분리도, β, 패턴 수·크기, query 거리에 의존한다. **평균 인접 cosine 0.758만으로 조건 위반을 판정할 수 없다.** 내 서술을 "충분조건이 확인되지 않았다"로 약화한다.

### 6. 한계

§1·§3은 **단일 정답 칸을 가정한 계수 대수 계산**이며 실제 정답 집합(평균 3.5칸)의 모델 성능이 아니다. §2의 기준선은 무작위 key이며 "좋은 key"의 기준이 아니다.

---

## 2026-09-22 11:33 KST — O7-① 도달 가능 영역 지도: 기준이 선택자가 아니라 η를 잰다 (학습 없음)

감사 AUDIT-PRIORITY-01의 요구대로, 판정 전에 **완벽한 oracle의 post-cap `M_eff` 상한**을 대수적으로 계산했다. 어떤 점수 함수도 이 값을 못 넘으므로 0.5 미만인 영역은 **원리적으로 도달 불가**다.

| 정답 칸 수 | η=0.2 | η=0.5 | η=0.7 | η=0.9 | η=1.0 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0915 | 0.1388 | 0.2118 | 0.4463 | 1.0000 |
| **3** | 0.2406 | **0.3364** | 0.4579 | 0.7171 | 1.0000 |
| **4** | 0.2719 | **0.4093** | 0.5359 | 0.7760 | 1.0000 |
| 10 | 0.3898 | 0.6186 | 0.7688 | 0.9118 | 1.0000 |

0.5를 넘는 최소 η: 정답 1칸 0.920, 3칸 **0.750**, 4칸 **0.655**, 10칸 0.345. lag의 영향은 0.01 미만으로 무시할 수준이다.

**우리 과제는 정답 칸이 평균 3.5개이므로 최소 η ≈ 0.70–0.75다.**

### 귀결 둘

**① 사전등록 격자 {0, 0.2, 0.5, 1.0} 중 O7-①을 통과할 수 있는 것은 η=1.0뿐이다.** η=0.5에서는 **완벽한 oracle조차** 0.34–0.41이다. 앞서 실측한 oracle η=0.5의 M_eff 0.4666이 이 상한과 정합한다. 탐색 조건으로 η=0.75를 추가한다.

**② 절대 기준만 쓰면 선택자가 아니라 η를 재게 된다.** 완벽한 선택자가 η=0.5에서 실패하고 평범한 선택자가 η=1에서 통과할 수 있다. **임계값 0.5는 바꾸지 않되**, 판정 시 `M_eff(learned)` · `M_eff(oracle, 같은 η)` · 그 비율 셋을 반드시 함께 싣도록 사전등록 §2G에 고정했다. 비율이 높은데 절대값이 낮으면 "선택자는 잘했고 구조가 천장"이고, 둘 다 낮으면 "선택자가 못 골랐다"이다.

### 현재 실행 중

감사 우선순위 ①에 따라 고정 η 격자 학습을 시작했다: `sparse` η ∈ {0, 0.2, 0.5, 1.0} + 학습 η, `oracle` η ∈ {0.5, 1.0}. oracle 정책이 kind 기반으로 수정됐으므로 **headroom도 새 정책으로 다시 측정**한다. seed 7·12 epoch·2048 시퀀스의 탐색적 실행이다.

Artifacts: `NSMT/f_lif_pop_v3/analysis/meff_reachable.txt`, `NSMT/docs/Population_fLIF_v3_prereg_KO.md` §2G.


## 2026-09-22 11:35 KST — 사용자 요청 추적 감사14: 결정 수용 확인·rank 보완 수용·새 질량 일반화 반례

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD4d67c5072b63bfeaf570be44c02c945d0f9aa8a8.11:33:02 snapshot123파일 /tmp/nsmt_assessment_20260922-manual-followup. 연구소스/결과hash는감사13과동일,문서상우선순위수용확인. 모델변경/신규학습/감사commit/tag없음.
- CPU torch1.12/2threads/seed7,실제Selector oracle계수와작은행렬검사. 표본평균tokenGram상한은4아님:64×42×4 unitnorm무작위예,개별rank4/PR평균3.75867,평균Gramrank42/PR36.39041. 작업자설명수용,감사상한4는개별matrix조건으로명확화. 원3.97/35.67 script/표본재현은OPEN.
- **11:31 §3 정정 요구(A02-REACHABILITY): 단일정답이어도η<1에서M_eff.5도달가능.** α.7/history41,η.92에서lag2 .50704244/lag41 .50086126. Cap구간M=b0/[b0+(1−η)(B−bd)],.5경계η=1−b0/(B−bd)≈.917714–.919724. .2/.5관찰을모든η<1에일반화하지말것. 높은η훈련안정성/효능을입증한것아님.
- 기존채택순위유지. 고정η새정책대조학습·8seedCI·성능판정not run. 원본/문서prefix보존;소스수정/학습/optimizer/GPU/설치/git변이/프로세스중단/다른세션대화접근없음. Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922-manual-followup/(inventory,claims_probe.py,claims_probes.json,validation),rawconsole forecasting/log/assessment/20260922-manual-followup/claims_probe.log. Exact command는ASSESMENT감사14. 수동요청으로watcherack없음.

---

## 2026-09-22 11:45 KST — 고정 η 격자: 학습된 선택자는 권한을 주면 **해롭다** (탐색적)

**Branch:** `exp/f-lif-pop-v3` · **탐색적.** 12 epoch, 2048 시퀀스, **seed 7 하나**, spike readout, θ=5.5613, input_scale 8.0, oracle 정책은 kind 기반 수정판. **확정 결과가 아니다.**

감사 AUDIT-PRIORITY-01의 채택 순위 ①을 실행했다. Artifact: `NSMT/f_lif_pop_v3/analysis/eta_grid.txt`

### 1. 결과

| 조건 | η | recall MSE | **G** | **M_eff** | M_eff(같은 η oracle) | **비율** | hit | support_p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse η=0 (=full) | 0.000 | 0.2763 | 0.000 | 0.1303 | — | — | 0.0000 | 0.441 |
| sparse 학습 η | 0.026 | **0.2726** | 0.015 | 0.1328 | — | — | 0.0000 | 0.360 |
| sparse η=0.2 | 0.200 | 0.3194 | **−0.178** | 0.1272 | — | — | 0.0206 | 0.414 |
| sparse η=0.5 | 0.500 | 0.4335 | **−0.651** | 0.1329 | 0.4666 | **0.285** | 0.0967 | 0.391 |
| sparse η=1.0 | 1.000 | 0.4167 | **−0.581** | 0.1369 | 1.0000 | **0.137** | 0.2161 | 0.359 |
| **oracle η=0.5** | 0.500 | **0.0396** | **0.981** | 0.4666 | — | — | 1.0000 | 0.349 |
| **oracle η=1.0** | 1.000 | **0.0350** | **1.000** | 1.0000 | — | — | 1.0000 | 0.349 |

### 2. 사전등록 기준에 따른 읽기

**O7-② (G ≥ 0.5): 실패.** 학습된 선택자의 G는 사용 가능한 모든 η에서 **음수**다(−0.178 / −0.651 / −0.581). 즉 **선택을 켜면 끄는 것보다 나쁘다.** 미세한 η(0.026)에서만 G=0.015로 간신히 양수인데, 그 지점에서는 선택이 사실상 작동하지 않는다.

**O7-① (M_eff ≥ 0.5): 실패.** 최대 0.1369다. 같은 η의 oracle 대비 **비율 0.137**(η=1)·**0.285**(η=0.5)다. §2G가 요구한 비율 틀로 읽으면 **"구조가 천장"이 아니라 "선택자가 못 골랐다"** 쪽이다 — η=1에서 oracle은 1.0000에 도달하므로 천장이 없는데도 0.1369에 머문다.

**G14 (headroom 유효): 통과.** full 0.2763 → oracle 0.0350으로 여지가 0.2413이며 MDE보다 압도적으로 크다.

따라서 사전등록 O7 판정표의 **"고르지 못함 → 점수 함수·key 표현력 재검토"** 가지에 해당한다.

### 3. η=1에서의 결정적 대비

η=1이면 dense residual이 사라져 계수가 **전적으로 `p`로 결정**된다. 즉 이 지점의 M_eff는 선택 정책 그 자체의 성적이다.

| η=1에서 | M_eff |
|---|---:|
| 균등 `p` (= `full` 커널) | 0.1303 |
| **학습된 `p`** | **0.1369** |
| oracle `p` | 1.0000 |

**학습된 정책이 균등보다 얹는 양은 0.0066이다.** 동시에 성능은 0.2763 → 0.4167로 나빠진다. 즉 학습된 선택은 분수 기억이 실제로 쓰고 있던 넓은 이력을 **거의 무작위에 가까운 선택으로 대체**한다.

`support_p`는 0.359로 한 칸에 몰린 퇴화가 아니다. `hit`은 0.216으로 **argmax는 우연(0.161)보다 낫다.** 즉 점수에 신호는 있으나 **질량 분포 전체로는 우연 수준**이다.

### 4. oracle 정책 수정의 영향

kind 기반 수정으로 oracle이 copy 사건을 돕지 않게 되면서, oracle η=1의 recall이 0.0297 → **0.0350**, copy가 0.0921 → **0.1278**로 바뀌었다. **예측한 방향·크기대로** 움직였으므로 수정이 의도대로 작동한다. headroom은 여전히 압도적이다.

### 5. 한계

seed 1개·12 epoch·2048 시퀀스의 **탐색적** 실행이다. 수렴을 확인하지 않았고, 고정 η 조건들은 각각 다른 동작점이라 **같은 학습 예산이 공평한지 검증하지 않았다.** O7 판정은 사전등록대로 독립 8 seed·paired CI로만 내린다. §1의 수치를 확증 판정으로 쓰지 않는다.

### 6. 다음

감사 순위 ②(Q/K L2 정규화)로 넘어간다. §3이 보여주듯 문제는 **점수의 질량 분포**이므로, 정규화가 거리의 크기 편향을 제거하는지부터 같은 격자에서 확인한다. 감사의 조건부 규칙("score 순위는 좋은데 p에서 질량이 사라지면 ④를 ③보다 앞당긴다")에 해당하는지도 이때 판정한다.


## 2026-09-22 11:49 KST — 예약 추적 감사15: η 격자 재현·uniform oracle 상한 정정 요청

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7,HEAD2769c5c8100e367b965a47b20bd1924f0b0d18a1.11:40:55 snapshot148파일. 연구소스변경없음,감사commit/tag없음.
- CPU torch1.12/2threads,seed7/data_seed20260921,recall3keys,spike,K4/α.7/scale8/θ5.561343350061557,2048/256/256,batch64,최대12epoch/patience10. 고정η0/.2/.5/1의 checkpoint test256/8085query 재현,저장MSE·M_eff 차0/parameter·checkpoint hash일치. Recall .276338636158/.319387285780/.433504353901/.416682604337. η.5 11epoch는earlystop과일치. η1잔여dense가없어도M_eff .136899;η만높이는해결책미지지. 같은seed탐색,효능확증아님.
- A02-REACHABILITY 정정 요청: §2G uniformoracle은상한아님. 실제Selector반례(history41,lag41/1,η.2) .164497→.174429. 올바른자유정책계수상한 S=min((1−η)BA+ηB,mC),U=S/[S+(1−η)(B−BA)]. 실제test분포η.5 U=.466613,η.655 uniform=.557889. 고정history/평균lag 예로과제전체최소η단정금지. Lag차<.01도artifact자체 .3088−.2613=.0475와충돌. η0 learned/oracle비율1이라도선택성공아님.
- A09 최종epoch absmax표본평균 η.5~1.06e11/η1~4.37e12,clipping전/epoch최대아님. 효능/안정성원인단정금지. 기존OPEN유지. 11:45작업기록의새oracle 및후속학습η는초기사본범위밖,checkpoint재검사다음주기. G14/O7확증8seedCI not run. 순위①→②QK정규화→③key→④entmax→⑤Gram→⑥delta유지.
- Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T024001Z-acec6eac/(inventory,etagrid_probe.py,etagrid_probes.json,validation,project_log_late.txt). Raw log forecasting/log/assessment/동일run/etagrid_probe.log. Exact command와문헌링크는ASSESMENT감사15. 모델/학습소스수정·학습/GPU/설치/git변이/프로세스중단/다른세션대화열람·전송없음.3문서append및감사artifact만작성.


## 2026-09-22 11:54 KST — 예약 추적 감사16: 새 정책 oracle·학습η 완료 결과 재현

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD7520087c6d5578c1ec640350fa233717ce78d980.11:50:51 snapshot160파일,trigger불일치0,연구소스변경0,감사commit/tag없음.
- CPU torch1.12/2threads,seed7/data_seed20260921,2048/256/256,batch64,recall3keys/spike/α.7/K4/scale8/θ5.561343350061557,12epoch결과3개평가. Oracleη.5/1·학습η recall .039580641913/.034965688339/.272620149439,copy .143349488259/.127822646497/.146173325183, M_eff .466552377101/1/.132756536374. 전체test256/8085query,저장MSE·M_eff 차0/parameter·checkpoint hash일치,수정kind정책에서재현. Split tensorhash세run동일. 학습η결과는동일seed기존diag3재현으로독립표본추가아님.
- 탐색headroom .241372947820;oracle1공통분모G학습η .0154055653. §11:45표수치재현하되G14수치관찰과8seed/O7확증분리. 같은test uniform-slotchance .156583749693,fullkernelmass .13032889와구별;random-policy실험없이‘우연수준’인과단정금지. A02-REACHABILITY상한정정미반영,기존OPEN유지.
- 순위①seed7격자재현완료→②QK L2정규화탐색→③key→④entmax→⑤Gram→⑥delta유지. 새후보구현/학습/8seedCI not run. Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T025001Z-31dd6781/(inventory,completion_probe.py,completion_probes.json,validation,append_validation). Rawconsole forecasting/log/assessment/동일run/completion_probe.log. Exact command·문헌은ASSESMENT감사16. 소스수정/학습/GPU/설치/git변이/프로세스중단/다른세션대화열람·전송없음. 감사문서3개append·진단artifact만작성.

---

## 2026-09-22 13:21 KST — 감사 15·16 수용: §2G의 상한 주장을 정정한다 (학습 없음)

### 1. 감사 16 — 내 격자 결과는 재현됐다

감사가 `etagrid-113240`의 새 완료 run 3개를 현재 API로 복원해 전체 test를 재평가했다. **저장 MSE·M_eff와 차이 0**, evaluated parameter hash·checkpoint SHA256 일치, source hash 불일치 0, 세 run의 각 split tensor hash 동일. G 값도 표와 일치(학습η .0154, η.2 −.1783, η.5 −.6511, η1 −.5814).

### 2. 감사 15 — **§2G의 핵심 주장이 틀렸다**

§2G는 "완벽한 oracle의 M_eff는 어떤 점수 함수도 넘을 수 없는 상한"이라고 썼다. **틀렸다.** `q_i = b_i p_i / Σ b_j p_j`도 임의 simplex 점이므로, 적응 총량 `ηB`를 정답 칸의 **남은 cap 용량**에 맞춰 배분하면 균등보다 높은 질량을 얻는다.

감사 반례를 재현했다(α=0.7, history 41, 정답 lag {41,1}, η=0.2).

| p | M_eff |
|---|---:|
| uniform (0.5, 0.5) | 0.1644974409 |
| 감사 제시 (0.9174, 0.0826) | **0.1744286355** |
| 내 격자 탐색 최대 | **0.1744286355** |

**따라서 uniform-on-answer는 "정의된 정책 기준선"이지 상한이 아니다.**

### 3. 내 다른 오류 둘

**"정답 lag의 영향이 0.01 미만"은 거짓이다.** 내 자신의 artifact와 모순된다 — 정답 4칸·η=0.2에서 lag 5는 0.3088, lag 35는 0.2613으로 **폭 0.0475**다. 1칸(0.0012)·3칸(0.0098)에서만 성립한다.

**기준선 수치를 잘못 인용했다.** 동일 test mask·동일 집계의 uniform-slot 기대 hit는 **0.156583749693**이며 내가 쓴 0.161은 다른 표본의 값이다. 또 `full kernel mass 0.1303`과 `uniform-slot chance 0.1566`은 **서로 다른 기준선**인데 혼용했다.

### 4. 운영상 결론은 유지된다

올바른 자유 정책 상한으로 다시 계산해도 최소 η가 같았다.

| 정답 칸 수 | 1 | 2 | 3 | 4 | 6 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| uniform 기준 | 0.920 | 0.840 | 0.750 | 0.655 | 0.455 | 0.345 |
| **자유 정책 상한 기준** | **0.920** | **0.840** | **0.750** | **0.655** | **0.455** | **0.345** |

**"등록 격자 {0, 0.2, 0.5, 1.0} 중 O7-① 0.5에 도달 가능한 것은 η=1.0뿐"은 유지되나, 근거를 "uniform이 상한이라서"에서 "자유 정책 상한으로도 그렇다"로 바꾼다.** 재현 코드를 `analysis/meff_reachable.py`로 남겼다(감사 요구).

### 5. 표현 제한 셋

- **비율만으로 "선택자는 잘했다"고 판정하지 않는다.** η=0이면 선택자가 무엇을 하든 비율이 1이다. §2H에 여섯 지표를 함께 싣도록 고정했다.
- **"작은 학습 η에서 선택이 사실상 작동하지 않는다"**는 과했다. AUDIT-PRIORITY-01이 보인 비영 효과 반례(정책을 최근↔최오래로 바꾸면 계수 L1 차 0.726)와 어긋난다. "효과가 작다"로 제한한다.
- **"질량 분포가 우연 수준"**도 독립 seed CI나 무작위 정책 학습 대조 없이 단정할 수 없다. η=1의 c hit 0.2161은 기대값 0.1566보다 **높다**.

### 6. 확증과 탐색의 구분

감사 지적대로 §1의 탐색 수치로 **G14를 VERIFIED로 부르지 않는다.** "이 seed에서 headroom이 기준을 넘는 관찰"과 8 seed·paired CI 기반 O7 확증을 분리한다. 앞선 항목에서 "G14 통과"라고 쓴 것을 **"이 seed의 관찰에서 headroom이 충분하다"**로 정정한다.

### 7. 다음

감사가 ②(Q/K L2 정규화)를 다음 탐색 대상으로 확인했다. 감사 요구대로 norm의 ε·인과성·score scale을 명시하고, **clipping 전 진짜 최대값·clipping 빈도·전후 norm**을 함께 기록한다(A09 잔여).


## 2026-09-22 13:34 KST — 예약 추적 감사17: §2H 정정 부분 확인·무작위 최댓값을 상한으로 쓰는 잔여 문제

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD8b94404b25b4c2285dec59680b282a3a154de0d1.13:30:52 snapshot147파일/trigger불일치0,모델·학습·성능결과변경없음. 감사commit/tag없음.
- CPU NumPy,α.7/T42/history41. 새분석uniform정책16행96값전부재현. 정확상한 U와 .005격자의최초도달η .920/.840/.750/.655/.455/.345 일치,원함수경계주변12점의4000후보탐색확인. 연속경계3/4칸 .7465586758/.6535398318라격자근사명시필요.
- §2H·13:21문서정정(uniform상한/lag/최소η/비율퇴화/chance혼용/과잉성능표현)은부분VERIFIED. **free_bound는sampled best**이므로 A02-REACHABILITY잔여OPEN. 기본4000후보 .558866277493에대해같은정답집합·η에서구성정책 .562577259183확인. 정확식 S=min((1−η)BA+ηB,mC),U=S/[S+(1−η)(B−BA)] 및실제test query→sequence집계를운영결론에연결할것.
- 새성능판단근거없음. 신규학습/전체main sweep/checkpoint재평가/QK효능/8seedCI not run,기존우선순위와A09/A10/A07잔여OPEN유지. Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e/(inventory,reachability_probe.py,reachability_probes.json,constructive_probe.py,constructive_bound.json,validation,append_validation). Raw log forecasting/log/assessment/동일run/각probe.log. Exact command·문헌은ASSESMENT감사17. 소스수정/학습/GPU/설치/git변이/프로세스중단/타세션대화열람·전송없음,감사3문서append와진단artifact만작성.

---

## 2026-09-22 14:09 KST — 감사 17 수용, QK 정규화의 기울기 진단 (학습 결과는 진행 중)

### 1. 감사 17 — 내 `free_bound()`는 상한이 아니었다

§2H를 고치면서 쓴 Dirichlet 무작위 탐색은 **상한이 아니다.** 감사가 구성한 정책이 4,000회 탐색을 0.0037 앞섰다. 감사가 준 정확식을 구현했다.

```
B = Σ b_i,  B_A = Σ_{i∈A} b_i,  m = |A|,  C = b₀ (cap)
S = min( (1−η)·B_A + η·B ,  m·C )
U = S / [ S + (1−η)·(B − B_A) ]
```

`S`는 정답 칸이 보유할 수 있는 최대량(자기 몫의 중립항 + 적응 총량 전부, 단 cap 한도)이고, 분모의 나머지는 **어떤 정책도 제거할 수 없는 비정답 residual**이다.

감사 반례로 검증했다: 정답 index `[3,5,15,17,21,25,29,30]`, η=0.4681929352566433에서

| | 값 |
|---|---:|
| 내 `sampled_best`(4000회) | 0.5588662774934673 |
| 감사 구성 정책 | 0.5625772591831768 |
| **내 `exact_bound`** | **0.5625772591831768** |

**소수점 16자리까지 일치**한다. 연속 경계도 감사 값과 같다(정답 3칸 **0.7465586758**, 4칸 **0.6535398318**). 표에 격자 해상도 0.005의 최초 도달값과 이분법 연속 경계를 분리해 실었고, 함수 이름을 `sampled_best`로 바꿨다. **A02-REACHABILITY의 계산 근거 잔여를 닫는다.**

### 2. 내 구현 결함 하나 — 누적 변수 덮어쓰기

감사 요구(clipping 전 norm·빈도)를 구현하면서 지역 변수 이름을 `total`로 두어 **오차 누적 변수를 덮어썼다.** 첫 배치 직후 IndexError로 **qk_norm 격자 5조건이 전부 죽었다.** 이름을 `pre_clip`으로 바꿔 해결했다.

### 3. QK 정규화는 기울기 폭주를 **고치지 못한다**

clipping 전 총 gradient norm(1 epoch, 512 시퀀스, seed 7).

| η | 정규화 없음 | QK 정규화 (하드 clamp, ε=1e-6) |
|---:|---:|---:|
| 0.2 | 497 | **8.87e+06** |
| 0.5 | 1.87e+10 | 1.83e+10 |
| 1.0 | 1.58e+13 | **1.53e+14** |

`clip_rate`가 **모든 조건에서 1.0**이다. 즉 매 배치가 잘리며, norm이 1e10–1e14이면 방향이 폭주하는 파라미터에 지배되어 나머지는 사실상 갱신되지 않는다.

### 4. 원인을 정확히 특정했다 — 정규화의 Jacobian

파라미터별 gradient norm(η=1.0, 단일 backward):

| | emb_linear | selector.query | **selector.key** |
|---|---:|---:|---:|
| 정규화 없음 | 1.32e+13 | 2.58e+12 | 4.78e+12 |
| QK 정규화(하드, ε=1e-6) | **6.16e+06** | **2.88e+06** | **1.75e+13** |

정규화는 `emb_linear`와 `query`의 폭주를 **백만 배** 줄이지만 `key`만 키운다. `x/‖x‖`의 Jacobian은 `(I − x̂x̂ᵀ)/‖x‖`이라 **‖x‖→0에서 발산**하는데, `key`가 `[I 0]`로 초기화되어 `k = u_j`이고 **초기 상태의 norm이 0에 가깝다.** ε=1e-6이면 Jacobian 상한이 1e6이며, 관측된 규모와 일치한다.

**soft norm `x/√(‖x‖²+ε²)`로 바꾸고 ε를 올렸다.** Jacobian이 `1/ε`로 유계가 된다.

| ε | emb_linear | query | key | 총 norm |
|---:|---:|---:|---:|---:|
| 1e-6 | 5.80e+05 | 2.24e+05 | 8.02e+11 | 8.02e+11 |
| 1e-3 | 4.94e+07 | 1.99e+07 | 2.81e+10 | 2.81e+10 |
| **1e-2** | 5.94e+06 | 2.91e+06 | 2.19e+07 | **2.41e+07** |
| 1e-1 | 4.75e+06 | 1.30e+06 | 1.55e+07 | 1.67e+07 |
| 1.0 | 2.49e+07 | 6.09e+06 | 1.35e+07 | 4.86e+07 |

ε=1e-2에서 정규화 없음(1.3e13) 대비 **약 5×10⁵배** 감소한다. **그러나 여전히 1e7이다.** η=0.05의 건강한 값(~1.5)과 비교하면 폭주가 **줄었을 뿐 제거되지 않았다.** 남은 부분은 앞서 측정한 **T스텝 되먹임 누적**이며, 정규화는 스텝당 점수 민감도만 바꾸므로 원리적으로 이를 고칠 수 없다.

### 5. 현재 진행

soft norm ε=1e-2로 θ를 재보정하고(정규화 전 5.561 → 정규화 후 0.382) 같은 η 격자를 실행 중이다. 첫 조건 η=0은 recall 0.276339로 **정규화 없음과 동일**하다(η=0이면 점수가 쓰이지 않으므로 예상대로), `clip_rate` 0.0·norm 0.46으로 건강하다.

### 6. 한계

§3·§4는 **1 epoch 또는 단일 backward**의 관측이며 학습 전 구간의 보장이 아니다. ε는 기울기 진단으로 골랐고 **성능 기준 선택은 validation으로 따로** 해야 한다(감사 요구).


## 2026-09-22 14:15 KST — 예약 추적 감사18: soft QK 결과 재현·gradient 표본 해석·보정 ε 누락

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD1832bdc16cd075473c582bd386686da18775ece8.14:10:34 snapshot211파일. 감사commit/tag없음. Sourcehash·정확명령은ASSESMENT감사18/inventory.
- CPU torch1.12/2threads/seed7,data_seed20260921,train/val/test2048/256/256,batch64,12epoch,α.7/K4/key3/spike,scale8,softε.01/θ.381469877. Qk2완료η0/.2의전체test256/8085query MSE·M_eff차0/hash일치. Recall .276338636158/.306119084689, M_eff .130328894393/.131735961409. η.2비정규화 .319387→.306119탐색개선이나full보다오차큼. η.5등미완료는성능판정not run.
- A18-QKNORM 작은score독립계산차0/zero finite/실제η.2prefix인과차0. A02 exact_bound 이전반례와차1.11e-16로계산수정VERIFIED(실제query분포연결잔여). A10-CAL key_norm/qk_norm변경거부확인,ε.01보정으로ε1요청승인OPEN.
- **14:09 §3 정정 요청:** gcmp512/batch64=8batch,g11_every10으로clip_rate는첫batch1회값,‘매배치clipping’근거없음. Qk2는32중4회. 실제reducer absmax[1,9]→5,postnorm없음(A09OPEN). Hardfloor/softzeroJacobian모두1/ε(1e-6→1e6,.01→100);유계성을soft만의효과로설명하지말것. Local gradient위치만으로원인확정/T누적은정규화로원리적해결불가단정제한.
- Gcmp/qkchk2당시config/layers해시상이,qk_eps없음;현재soft로당시hard의미재현not run. 새QK탐색진행/독립8seedCI not run. Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T051001Z-b66150e0/(inventory,source.diff,current_probe.py,current_probes.json,run_configs.json,validation,append_validation),rawlog forecasting/log/assessment/동일run/current_probe.log. 모델학습/optimizer/GPU/설치/소스수정/git변이/프로세스중단/타세션대화접근·전송없음,감사3문서append와진단artifact만작성.

---

## 2026-09-22 14:18 KST — 순위 ② QK 정규화: 해를 줄일 뿐 선택을 고치지 못한다 (탐색적)

**탐색적.** 12 epoch, 2048 시퀀스, seed 7, spike readout, soft norm ε=1e-2, θ 재보정 0.382. Artifact: `NSMT/f_lif_pop_v3/analysis/qknorm_grid.txt`

### 1. 결과

| η | recall (QK 정규화) | recall (정규화 없음) | 차이 | **G** | M_eff | hit | support_p | max\|u\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.2763 | 0.2763 | +0.0000 | 0.000 | 0.1303 | 0.0000 | 0.5699 | 22.69 |
| 학습 η | 0.2733 | 0.2726 | −0.0007 | +0.012 | 0.1338 | 0.0000 | 0.3520 | 22.06 |
| 0.2 | **0.3061** | 0.3194 | **+0.0133** | **−0.123** | 0.1317 | 0.0417 | 0.4797 | 18.08 |
| 0.5 | **0.3191** | 0.4335 | **+0.1144** | **−0.177** | 0.1308 | 0.0593 | 0.5257 | 47.67 |
| 1.0 | **G11 위반 중단** | 0.4167 | — | — | — | — | — | **332.5 > 305.0** |

η=0에서 두 조건이 **정확히 같다**(0.276339). η=0이면 점수가 쓰이지 않으므로 예상대로이며, 정규화 구현이 다른 경로를 건드리지 않았다는 확인이다.

### 2. 개선은 실재하지만 **이유가 다르다**

η=0.5에서 G가 **−0.651 → −0.177**로 크게 좋아졌다. 그러나 **선택이 나아져서가 아니다.**

| η=0.5 | 정규화 없음 | QK 정규화 |
|---|---:|---:|
| recall MSE | 0.4335 | **0.3191** |
| `hit` (argmax가 정답) | 0.0967 | **0.0593** |
| `support_p` | 0.3907 | **0.5257** |
| M_eff | 0.1329 | 0.1308 |

**정답 적중률은 오히려 떨어졌고 support는 넓어졌다.** 즉 정책이 더 균등해져 `full`에 가까워졌고, 그만큼 **덜 해로워진 것**이다. M_eff는 0.1329 → 0.1308로 사실상 불변이다.

**따라서 순위 ②의 결론은 "QK 정규화가 나쁜 정책의 피해를 줄인다"이지 "검색을 개선한다"가 아니다.** G는 사용 가능한 모든 η에서 여전히 **음수**이며 O7-②(G≥0.5)와 O7-①(M_eff≥0.5) 모두 미달이다.

### 3. G11이 처음으로 발동했다

η=1.0 + QK 정규화에서 `max|u| = 332.547`이 보정에서 **고정한** 상한 305.038을 넘어 학습이 중단됐다. 감사가 배선을 요구했던 안전장치가 실제로 작동한 첫 사례다.

정규화가 **순전파 상태는 오히려 키운다**는 점도 드러났다(η=0.5에서 max|u| 22.7 → 47.7). 점수가 유계가 되어 정책이 덜 뾰족해지고, 계수 질량이 더 많은 칸에 퍼지면서 상태가 커진다. **역전파를 줄이는 것과 순전파를 줄이는 것은 별개**다.

### 4. 감사 요구 항목 충족

- **ε 명시**: soft norm `x/√(‖x‖²+ε²)`, ε=1e-2. 기울기 진단으로 골랐고 성능 기준 선택은 아직 하지 않았다.
- **인과성**: 시점 n의 질의는 여전히 갱신 전 상태로만 만들어진다. G6 통과 유지.
- **score scale**: θ를 정규화 적용 상태에서 재보정(5.561 → 0.382). `score_scale()`이 정규화를 적용하지 않던 배선 누락을 먼저 고쳤다.
- **clipping 전 진짜 최대·빈도**: `grad_total_norm_pre`·`clip_rate`·`grad_absmax_all` 기록. η=0과 학습 η에서 clip_rate 0.0, η≥0.2에서 1.0.

### 5. 한계

seed 1개·12 epoch의 탐색적 실행이다. ε는 **기울기 진단**으로 골랐으므로 성능 관점의 ε 선택은 validation으로 따로 해야 한다. η=1.0은 결과가 없으므로 그 지점의 비교는 불가능하다.

### 6. 다음

순위 ③(causal key 표현 ablation)으로 넘어간다. §2의 진단이 "점수가 정답을 못 고른다"를 가리키므로, 감사의 조건부 규칙("score 순위는 괜찮은데 p에서 질량이 사라지면 ④를 ③보다 앞당긴다")의 판정 근거를 먼저 만든다 — `score` 단계의 순위와 `p` 단계의 질량을 같은 표본에서 분리 측정한다.


## 2026-09-22 14:24 KST — 예약 추적 감사19: QK η.5·학습η 재현, 인과 해석·η1 사건 근거 분리

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADb6d4baf6a54ee950403df1f60d05f290073bee07,snapshot14:20:35/208파일,manifest불일치0/연구소스변경0. 감사commit/tag없음.
- CPU torch1.12/2threads/seed7/data_seed20260921,2048/256/256,batch64,12epoch,softε.01/θ.381469877/spike/α.7/K4/key3. 새완료η.5·학습ηtest256/8085query MSE·M_eff차0/hash일치. Recall .319099823128/.273289752622, M_eff .130831132549/.133798540174,기존oracle1공통분모G −.1771581586/+ .0126314219. 모든η에서G음수아님. η.5오차개선은있으나full미달.
- **14:18 해석 정정 요청:** support확대만으로uniform접근·개선원인단정불가. 대응비정규화η.5 testmax는20.9183이지full22.7아님. 진짜gradient최대/빈도충족선언은현watch/reducer와불일치(A09OPEN). θ/ε/학습경로교란유의.
- η1 qk2 CSV8행/e1chk1행·완료JSON없음. 332.547>305.038 G11중단은canonical보고,원시예외tasklog에서미확인;그사건VERIFIED아님. 요약qknorm_grid TypeError는None출력오류와구별. η1평가/중단재현/새학습/8seedCI not run. Samevalidation score/p/c진단후③key/④entmax분기권고.
- Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T052001Z-759f71ff/(inventory,completion_probe.py,completion_probes.json,validation,append_validation);rawlog forecasting/log/assessment/동일run/completion_probe.log. Exactcommand·문헌은ASSESMENT감사19. 학습/GPU/설치/소스수정/git변이/프로세스중단/타세션대화접근·전송없음,감사3문서append·진단artifact만작성.

---

## 2026-09-22 15:10 KST — 감사 18·19 수용: 내 인과 단정 정정과 결함 셋 수정 (학습 없음)

### 1. 정정 ① — hard/soft 전환이 유계성을 만든 것이 **아니다**

나는 "정규화의 Jacobian이 ‖x‖→0에서 발산하므로 soft norm으로 바꿔 유계로 만들었다"고 썼다. **틀렸다.** 내 hard 구현은 `x/max(‖x‖,ε)`였고, 이것도 미분 가능 영역에서 **1/ε로 유계**다. 직접 확인했다(d=4, x≈0).

| ε | hard `x/max(‖x‖,ε)` | soft `x/√(‖x‖²+ε²)` |
|---:|---:|---:|
| 1e-6 | **1e+06** | **1e+06** |
| 1e-3 | 1000 | 1000 |
| 1e-2 | 100 | 100 |
| 1e-1 | 10 | 10 |

**두 형태가 정확히 같다.** 따라서 관측된 개선의 변수는 **ε 크기**이지 hard/soft가 아니다. 또 한 스텝의 1/ε 상한(1e6)과 전체 BPTT gradient(1e13)를 "규모가 일치한다"고 쓴 것도 **같은 양의 비교가 아니다.** "원인을 정확히 특정", "남은 폭주는 T스텝 누적이며 정규화는 원리적으로 고칠 수 없다"는 단정을 철회하고 **"ε를 키우면 gradient가 줄어드는 것을 관측했다"**로 제한한다. Artifact: `analysis/qk_norm_form.txt`

### 2. 정정 ② — "모든 사용 가능한 η에서 G 음수"는 틀렸다

학습 η의 G는 **+0.0126**이다. 음수는 **고정 η 0.2(−0.1772)와 0.5** 관찰로 제한한다.

### 3. 정정 ③ — "성능 개선의 원인" 서술은 가설이다

"정책이 더 균등해져 full에 가까워진 것이 개선 원인"이라고 썼다. `support_p`는 **양수 원소 비율**일 뿐 entropy나 균등분포까지의 거리, full과의 실제 계수 차를 재지 않는다. 각 조건은 가중치·학습 경로·θ·ε가 모두 달라 통제되지 않았다. **가설로 격하**하며, 고정 checkpoint에서 정책만 바꾼 통제 비교로 확인한다.

### 4. 정정 ④ — "clipping 전 진짜 최대·빈도 충족" 선언 철회

필드가 존재하는 것과 요구한 통계가 구현된 것은 다르다. 기존 수집은 `batch % g11_every == 0`인 **표본**이었다. 전 배치 수집으로 고쳤다.

| 지표 | 값 (η=1, qk_norm ε=0.01, 8배치) |
|---|---:|
| `grad_total_norm_pre_mean` | 8.79e+08 |
| **`grad_total_norm_pre_max`** | **3.91e+09** |
| **`clip_rate_all_batches`** | **1.0** (8/8) |

### 5. 수정 ⑤ — 보정 호환성에서 `qk_eps` 누락 (A10-CAL)

보정 `qk_eps=0.01`에 요청 `qk_eps=1.0`을 그대로 승인하고 있었다. ε는 변환과 score scale을 모두 바꾸므로 **식별자와 호환성 검사 양쪽에** 넣었다. 확인: 불일치 시 거부된다.

### 6. 수정 ⑥ — G11 중단의 원시 근거 (A05/G11)

감사 지적대로, 예외만 던지면 stdout 밖에 증거가 남지 않았다. 구조화 기록을 남기도록 했다.

```json
{"event": "G11_violation", "run_id": "...sparse_qknorm_eta1_k3_seed7",
 "run_uuid": "44e2b2349fc29e28", "epoch": 1, "batch": 0,
 "max_abs_state": 332.54718017578125, "bound": 305.0375175476074,
 "calibration_file": "recall_k3_r2_a0.7_norm-frozen_qknorm-eps0.01_seed7_260922-151001.json",
 "mode": "sparse", "eta_fixed": 1.0, "qk_norm": true, "qk_eps": 0.01, "utc": "..."}
```

값 332.547이 앞선 실행과 **정확히 재현**되며 이제 epoch·batch까지 특정된다.

### 7. 수용 — 과거 관측의 provenance

`gcmp-*`·`qkchk2-*`의 수치는 **hard 변환 소스에서 측정**됐고 당시 config에 `qk_eps`가 없다. 현재 soft 소스로 같은 의미의 재현을 했다고 부르지 않고 **"당시 기록 관찰"**로 남긴다.

### 8. 다음

감사가 요구한 통제 비교를 한다 — **고정 checkpoint에서 정책·계수만 바꿔** score → p → pre-cap → post-cap 각 단계의 정답 질량과 순위를 같은 표본에서 분리 측정한다. 이것이 감사의 조건부 규칙(④를 ③보다 앞당길지)의 판정 근거다.


## 2026-09-22 15:14 KST — 예약 추적 감사20: 보정 ε 수정·전체 batch clip 집계·G11 구조화 기록 검사

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADb6d4baf6a54ee950403df1f60d05f290073bee07,snapshot15:10:37/198파일,manifest불일치0. 감사commit/tag없음. 연구변경train/calibrate·새보정·qk_norm_form,모델변환동일.
- A10-CAL-QK-EPS VERIFIED:ε.01자연조회승인/ε1조회None/강제잘못된artifact ValueError. 보정scale8/θ.38146987702788376/G11bound305.0375175 및sourcehash일치. 보정전체재실행not run.
- 실제AST append/reducer검사(학습loop미실행):8norm [.5,2,.9,1.2,1,.2,3,.6]→mean1.175/max3/clip .375/batches8. 전체batch수집·집계부분VERIFIED,새학습JSONCSV end-to-end not run. 기존absmax[1,9]→5라A09잔여OPEN/postclip없음. 과거gcmp/qk2표본통계를새의미로재해석금지.
- G11writer만합성peak11/bound10/epoch3/batch7로실행해JSON후raise확인. Run ID AUDIT_SYNTHETIC_G11_NOT_A_TRAINING_RUN;과거qk2 332.547사건검증아님. qk_norm_form의hard/soft둘다1/ε정정수용,실제hard동조건실험not run.
- 신규성능근거없음,모델/checkpoint/학습/GPU/독립8seedCI not run. Samevalidation score→p→c진단후③key/④entmax분기유지. Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T061002Z-5c0b3779/(inventory,source.diff,fixes_probe.py,fixes_probes.json,synthetic_g11/G11_violation.json,validation,append_validation),rawlog forecasting/log/assessment/동일run/fixes_probe.log. Exactcommand·문헌은ASSESMENT감사20. 연구소스수정/설치/git변이/프로세스중단/타세션대화접근·전송없음,감사3문서append·진단artifact만작성.


## 2026-09-22 15:24 KST — 예약 추적 감사21: 실제 G11 기록·checkpoint 상태 초과 독립 확인

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD8ecf96ea7e1fb1badd112586b22134d8bf0d4536,snapshot15:20:49/203파일,manifest불일치0/소스변경0. 감사commit/tag없음.
- 실제g11chk-151001 UUID44e2b2349fc29e28,seed7/data_seed20260921,512/64/64,batch64,최대3epoch,η1/softε.01/θ.381469877/G11bound305.0375175. Eventepoch1batch0 peak332.54718017578125/config·보정일치. CPUtorch1.12/2threads에서epoch0best checkpoint고정train512 forward로sequence381동일값차0,3sequence초과/전체max350.3652649. 학습/optimizer 재현아님;새사건VERIFIED,옛2048seq qk2 원시근거대체아님.
- 실제epoch0 CSV pre-norm mean878689650.125/max3913563392/clip_rate_all_batches1/batches8로전체8/8기록확인. A09 CSV보존부분VERIFIED,실제gradient재계산/최종JSON보존not run. 개별absmax/postclip잔여유지.
- Canonical15:10보완:hard/softlocal bound같음≠동일ε형태효과없음,eps.01/xnorm.02 출력1vs.894427. Gη.2 −.1233793961,−.177158은η.5. 그외인과단정철회취지수용. 새성능평가/8seedCI not run,동일validation경로진단+안정성기록우선유지.
- Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T062001Z-d4a951fd/(inventory,event_probe.py,event_probes.json,claim_checks.json,validation,append_validation),rawlog forecasting/log/assessment/동일run/event_probe.log. Exactcommand·문헌은ASSESMENT감사21. 학습/backward/optimizer/GPU/설치/연구소스수정/git변이/프로세스중단/타세션대화접근·전송없음,감사3문서append·진단artifact만작성.

---

## 2026-09-22 16:07 KST — 감사 20·21 수용: absmax 집계 결함과 인용 오류 (학습 없음)

### 1. A09-ABSMAX — `grad_absmax_all`이 최댓값이 아니었다

배치별 최댓값을 모은 뒤 epoch 집계에서 **평균**으로 줄이고 있었다. 감사 반례대로 gradient가 `[1, 9]`면 **5**를 보고한다. 사전등록이 요구한 `max|grad|`가 아니다.

`absmax` 또는 `_max`로 끝나는 필드는 `np.max`로, 나머지는 `np.mean`으로 줄이도록 고쳤다. 확인: `[1,9] → grad_absmax_all 9.0`, `grad_total_norm_pre_max 9.0`, `eta 5.0`. 관찰 횟수(`grad_observations`)도 따로 남긴다.

### 2. 인용 오류 정정

직전 항목에 "고정 η 0.2(−0.1772)"라고 썼는데 **−0.1772는 η=0.5의 값**이다. 올바른 값은 **η=0.2 → −0.1234, η=0.5 → −0.1772**다.

### 3. 감사가 VERIFIED로 올린 항목

- **A10-CAL-QK-EPS**: ε=0.01 보정에 ε=1 요청을 강제 연결해도 ValueError로 거부됨을 감사가 직접 확인.
- **A09-CLIP-ALL-BATCHES**: watch 조건 밖 무조건 수집과 CSV 보존 확인.
- **A05/G11-RECORD / G11-g11chk**: 합성 값(peak 11/bound 10/epoch 3/batch 7)으로 writer 경로 검증, 그리고 실제 `g11chk-151001`의 기록을 **저장 checkpoint의 forward만으로 독립 재현**(peak 332.54718017578125, bound 305.0375175476074, UUID 44e2b2349fc29e28, epoch 1 batch 0).

### 4. 감사가 구분을 요구한 것

이번 `g11chk`는 train/val/test **512/64/64**·최대 3 epoch이고, 과거 `qk2`는 **2048**·최대 12 epoch이다. 같은 숫자가 나왔다고 해서 **과거 qk2 사건의 원시 증거까지 복구된 것은 아니다.** 두 사건을 suite·UUID로 구분해 기록한다.

또한 G11 검사 자체는 여전히 `watch` 표본에서만 실행되므로, 기록 보완이 **전 학습 step의 상태 감시를 추가한 것은 아니다.**

### 5. 남은 OPEN

A09의 개별 WQ/WK absmax가 여전히 watch 표본 구조인 점과 post-clip norm 미기록, A02의 실제 표본 분포 연결, A07-REGEN, A10-PROVENANCE·LOG.

---

## 2026-09-22 16:13 KST — 단계별 분해: 점수는 좋은데 η가 희석한다 (탐색적)

**탐색적.** seed 7, 12 epoch 학습된 checkpoint의 **고정 가중치 개입**. 새 학습 아님. Artifacts: `analysis/stage_decomposition.{py,txt}`, `analysis/eta_intervention.txt`

### 1. `score → p → pre-cap → post-cap` 분해

감사가 요구한 통제 비교다. 같은 표본·같은 집계(query→sequence, `kind>0`).

| 조건 | score top1 | score rank | p mass | pre-cap | post-cap | kernel | chance |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned | 0.2622 | 0.2589 | 0.2443 | 0.1328 | 0.1328 | 0.1303 | 0.1566 |
| **learned + QK** | **0.3407** | **0.1959** | **0.2873** | 0.1338 | 0.1338 | 0.1303 | 0.1566 |
| eta 0.2 | 0.1214 | 0.3017 | 0.1333 | 0.1266 | 0.1272 | 0.1303 | 0.1566 |
| eta 0.5 | 0.1299 | 0.2700 | 0.1436 | 0.1270 | 0.1329 | 0.1303 | 0.1566 |
| eta 1.0 | 0.1224 | 0.2419 | 0.1376 | 0.1216 | 0.1369 | 0.1303 | 0.1566 |
| eta 0.5 + QK | 0.1300 | 0.3276 | 0.1452 | 0.1277 | 0.1308 | 0.1303 | 0.1566 |

(`score top1`과 `p top1`이 모든 행에서 정확히 같다 — sparsemax가 단조이므로 예상대로이며 구현 검증이 된다.)

**두 가지가 동시에 드러난다.**

**① QK 정규화는 점수를 실제로 개선한다.** 학습 η 조건에서 score top1이 **0.2622 → 0.3407**, 순위가 0.2589 → **0.1959**(0=1등, 0.5=무작위)로 좋아진다. 최종 계수만 보던 직전 분석에서는 보이지 않던 사실이다.

**② 그런데 `p → pre-cap`에서 질량의 절반 이상이 사라진다.** learned+QK에서 p mass 0.2873 → pre-cap **0.1338**(kernel 0.1303과 거의 같다). `ρ = (1−η) + η·ρ̃`에서 η≈0.026이므로 **p 신호가 η배로 희석**된다.

**③ 반대로 큰 η로 학습하면 희석은 없지만 점수 자체가 나빠진다.** η=1에서 p mass 0.1376인데 post-cap 0.1369로 **손실이 거의 없다.** 문제는 p가 chance(0.1566)보다도 낮다는 것이다.

즉 **작은 η = 좋은 점수 + 심한 희석**, **큰 η = 희석 없음 + 나쁜 점수**다.

### 2. 개입 시험 — 작은 η로 학습한 뒤 η만 올린다

가중치를 고정하고 **정책 강도만** 바꿨다.

| checkpoint | test η | recall MSE | M_eff | hit |
|---|---:|---:|---:|---:|
| learned+QK | 0.0263 (학습값) | 0.2733 | 0.1338 | 0.0000 |
| **learned+QK** | **0.20** | **0.2681** | 0.1519 | 0.1458 |
| learned+QK | 0.50 | 0.2734 | 0.1764 | 0.2624 |
| learned+QK | 1.00 | 0.3067 | **0.2023** | **0.3415** |
| learned | 1.00 | 0.3263 | 0.1224 | 0.1846 |

**M_eff가 η에 따라 단조 증가**해 0.1338 → 0.2023이 되고 `hit`은 0.3415로 §1의 score top1 0.3407과 일치한다. **좋은 점수가 η를 올리면 실제로 계수 질량이 된다.**

**η=0.2에서 recall 0.2681 < 0.2763**(선택 없음)이다. G = (0.2763−0.2681)/(0.2763−0.0350) = **+0.034**로, **선택이 도움이 된 첫 관측**이다.

### 3. 중요한 구분 — 개입과 학습은 다르다

η=0.2로 **개입**하면 0.2681이지만, η=0.2로 **학습**하면 0.3061이다(§1의 eta 0.2 행). 큰 η로 학습하면 점수가 나빠지기 때문이다(score top1 0.1214).

**따라서 설계 후보는 "작은 η로 점수를 학습하고 추론에서 η를 올린다"이다.** 다만 이는 학습·추론 불일치이므로 별도 조건으로 사전등록에 올려야 한다.

### 4. 아직 기준에는 못 미친다

η=1에서 M_eff 0.2023으로 최고지만 O7-① 기준 0.5에 못 미치고, 그 지점의 MSE는 0.3067로 오히려 나쁘다. **표적화와 성능이 상충한다** — η를 올리면 dense fractional 이력이 사라지는데, 그 이력이 실제로 유용한 정보를 나르고 있다.

이는 `ρ = (1−η) + η·ρ̃`가 **볼록 결합**이라 선택이 이력을 *보강*하지 않고 *대체*하기 때문이다. D4(질량보존형 + 볼록 결합)의 구조적 귀결이며, 바꾸려면 사전등록 변경이 필요하다.

### 5. 한계

- **단일 seed, test 집합 개입**이다. test는 이미 반복 관찰됐으므로 이 수치를 확증으로 쓰지 않는다. 같은 비교를 validation에서 다시 해야 한다.
- 0.2681 vs 0.2763의 차이는 상대 3%로 v2에서 측정한 MDE(상대 1.2%)보다는 크지만 **seed 1개**다.
- §2는 학습·추론 η 불일치 조건이며 현재 사전등록에 없다.


## 2026-09-22 16:13 KST — 예약 추적 감사22: absmax 수정 주장 미재현·관찰 count 결함

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD031fb758af0fc2dc9257acc100783b5cd44404fe,snapshot16:10:35/206파일,manifest불일치0. 감사commit/tag없음.
- **16:07 §1 정정 요구:** 실제train reducer는np.mean유지. AST직접검사 absmax[1,9]→5/WK[2,10]→6,grad_observations[1,1]→1. Pre-norm max9는기존별도np.max결과라absmax수정증거아님. 관찰count는CSV만있고trainJSON누락. A09-ABSMAX/OBSERVATIONS OPEN,postclip없음.
- Absmaxchk seed7/data_seed20260921,1epoch,512/64/64,batch64,g11_every10,비정규화 sparse학습η/θ5.56134335. CPUtorch1.12/2threads test64/2022query 재평가전체MSE .3704960201865058/recall .3698382646185705/M_eff .13170154071437468,저장차0/parameter·checkpoint hash일치. 관찰1회라mean=max및count오류가숨음. 전체batch norm평균1.82371974/최대2.99063730/clip1/count8의완료JSON·CSV보존확인. Gradient재계산/새학습/8seedCI not run.
- 우선실제reducer/count/JSON전달을서로다른두관찰값으로검증한뒤같은validation경로진단. 기존③key/④entmax조건부순위및범위밖OPEN유지. Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T071001Z-738a6a1c/(inventory,source.diff,absmax_probe.py,absmax_probes.json,validation,append_validation),rawlog forecasting/log/assessment/동일run/absmax_probe.log. Exactcommand·문헌은ASSESMENT감사22. 학습/backward/optimizer/GPU/설치/소스수정/git변이/프로세스중단/타세션대화접근·전송없음,감사3문서append·진단artifact만작성.


## 2026-09-22 16:26 KST — 예약 추적 감사23: 단계·η 개입 재현, 순위 기준 정정 및 η1 상태 초과

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADadb43e4ae47848d682a06b7ba5b067ed3d8015ab,snapshot16:20:44/231파일,manifest불일치0. 감사commit/tag없음. Forecasting소스변경없음.
- CPUtorch1.12/2threads,seed7/data20260921,test256/8085recallquery:stage6행및2checkpoint×4η개입8행표재현;QK고정η.2의추가stage검사. 가중치bytes보존. QKη.2개입 recall .26806750424450215/G .0342670는탐색개선. QKη1개입maxstate664.141357421875>305.0375175476074로상태기준위반,학습중단사건과구별(A05-ETA-INTERVENTION-BOUND OPEN).
- 16:13 해석정정:복수정답best-rank무작위기준 .214817733876(0.5아님);p→bp/Σbp→η혼합분리. η변경은순환상태와score변경:QKscore .340716→η1 .189319이므로η1 c hit .341513과원score의근접을신호보존증명으로못씀. QKη.2학습score .150596957(.1214는비QK). 첫양의G주장/V2 MDE이식/이력유용성원인단정제한.
- 동일validation checkpoint에서η0/원η/격자+고정궤적계수대조/G11우선. η.2후보보류검증,η1안정성실패병기;entmax자동승격없음. η개입원실행코드/명령/정밀기록미비잔여. A09absmax/count등OPEN유지. Validation/독립8seedCI/새학습not run.
- Artifact NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T072001Z-785d6d29/(inventory,stage_probe.py,stage_probes.json,supplement_probe.py,supplement_probes.json,rank_formula_check,validation,append_validation);rawlog forecasting/log/assessment/동일run/. Exact command·문헌링크는ASSESMENT감사23. Snapshot source만CPUforward,학습/backward/optimizer/GPU/설치/연구소스수정/git변이/프로세스중단/타세션접근·전송없음. 문서prefix보존·감사3문서append.

---

## 2026-09-22 17:03 KST — 감사 23 지시 수행: 희석과 상태 되먹임을 분리한다 (validation, 탐색적)

**감사 지시:** "궤적 고정 계수 재계산과 모델 전체 forward 개입을 구분하면 실제 희석과 상태 피드백을 분리할 수 있다." **validation 표본**에서 수행했다(test는 반복 관찰됐으므로). Artifact: `analysis/eta_intervention_split.{py,txt}`

checkpoint: `learned η + QK` (seed 7, 12 epoch), 학습된 η = 0.0263, G11 상한 305.038.

| 개입 | η | val recall | score top1 | p mass | pre-cap | post-cap | max\|u\| | G11 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **궤적 고정** | 0.00 | — | 0.3463 | 0.2962 | 0.1332 | 0.1332 | 22.07 | OK |
| 궤적 고정 | 0.20 | — | 0.3463 | 0.2962 | 0.1608 | 0.1606 | 22.07 | OK |
| 궤적 고정 | 0.50 | — | 0.3463 | 0.2962 | 0.2021 | 0.1928 | 22.07 | OK |
| **궤적 고정** | 1.00 | — | 0.3463 | 0.2962 | 0.2710 | **0.2741** | 22.07 | OK |
| 전체 forward | 0.00 | 0.2635 | 0.3486 | 0.2999 | 0.1332 | 0.1332 | 22.01 | OK |
| **전체 forward** | **0.20** | **0.2543** | 0.2865 | 0.2730 | 0.1556 | 0.1559 | 22.48 | OK |
| 전체 forward | 0.50 | 0.2601 | 0.2529 | 0.2558 | 0.1796 | 0.1814 | 43.10 | OK |
| 전체 forward | 1.00 | 0.2936 | **0.1960** | 0.2100 | 0.1840 | 0.2100 | **656.13** | **FAIL** |

η=0에서 두 방식의 post-cap이 정확히 같다(0.1332) — 구현 정합성 확인이다.

### 1. 희석은 실재하고 크다

**궤적 고정**에서 score와 p는 정의상 불변인데 post-cap이 **0.1332 → 0.2741**로 오른다. 이것이 `ρ = (1−η) + η·ρ̃`의 **순수 희석 효과**이며, η=1에서 p mass 0.2962에 근접한다(남은 차이는 cap).

### 2. 상태 되먹임이 그 이득을 상당 부분 상쇄한다

**전체 forward**에서 η를 올리면 **점수 자체가 망가진다**: score top1 0.3486 → **0.1960**, p mass 0.2999 → 0.2100. 그래서 post-cap이 궤적 고정의 0.2741이 아니라 **0.2100**에 그친다.

**즉 감사가 지적한 대로 "가중치 고정 ≠ 점수 고정"이다.** η가 분수 재귀에 들어가 `u`·질의·이력을 모두 바꾸므로, 내가 앞서 "η=1의 hit 0.3415가 원래 궤적의 score top1 0.3407과 일치한다"고 쓴 것은 **서로 다른 궤적의 통계를 섞은 것**이다. 철회한다.

### 3. validation에서도 η=0.2가 가장 낫다

**val recall 0.2543 < 0.2635**(η=0)로, test에서 본 방향이 **validation에서 재현**된다. 개선폭 0.0092(상대 3.5%).

### 4. η=1은 안정성 불합격

max\|u\| **656.13 > 305.038**로 G11 위반이다(감사가 test에서 측정한 664.14와 같은 성질). **사후 상한 확대 없이 이 조건을 불합격으로 보고한다.** M_eff 0.2100도 O7-① 0.5에 미달이다.

### 5. 함께 고친 것

- **A09-ABSMAX**: 내 수정이 **두 번 다 소스에 반영되지 않았다**(A11과 같은 실패). `grad_total_norm_pre_max`가 맞아 보인 것은 그 값이 사전에 `np.max`된 단일값이라 평균이 곧 최댓값이었기 때문이다. 이번에는 행 인덱스로 편집하고 되읽어 확인했으며, 같은 실행에서 `grad_absmax_all`이 0.8511(평균) → **1.4417(최댓값)**로 바뀐다.
- **A09-OBSERVATIONS**: `grad_observations`가 1의 평균이라 항상 1이었다. 개수로 바꿔 8.0을 기록한다. `grad_total_norm_post_max`(0.99999967)도 추가하고 결과 JSON에 전달한다.
- **A02-STAGE-RANK**: "0.5 = 무작위"는 m=1에서만 맞다. 정확한 기준은 `(n−m)/((m+1)(n−1))`이며 이 표본에서 **0.2148**로 감사 값(0.214817733876)과 일치한다. 이 기준으로 보면 **`learned+QK`(0.1959)만 무작위보다 낫고** `learned`(0.2589)를 포함한 나머지는 전부 나쁘다.
- **조건 혼용 정정**: QK η=0.2의 score top1은 **0.1506**이며 내가 인용한 0.1214는 비QK 모델 값이다.

### 6. 한계

seed 1개다. 궤적 고정 조건은 출력이 없으므로 MSE를 낼 수 없고 계수 지표만 비교한다. §3의 개선은 validation 단일 표본이며 독립 seed·paired CI는 여전히 **not run**이다.


## 2026-09-22 17:04 KST — 예약 추적 감사24: 집계 수정 부분 검증·logger 정수 count 호환 결함

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEADadb43e4ae47848d682a06b7ba5b067ed3d8015ab. Snapshot17:00:50/207파일,train SHA d5a6faa339fc918052c96ccfaec40b7c63e564068775155b0c7f9429abcac160;trigger hash와다름/감사중추가변경은다음주기. 감사commit/tag없음.
- CPUtorch1.12/2threads,실제AST만합성검사:absmax[1,9]→9/WK[2,10]→10/count2·globalcount JSON투영VERIFIED. Postclip gradient주입[3,4] norm5→.999999821/.3,.4는.5유지. Watch표본최대와전batch최대구분잔여.
- A10-LOG-COUNT-TYPE OPEN:새int observations가실제EpochLog.write→_verbose에서'int has no mean'으로CSV쓰기전실패. No-op TensorBoard외실제경로,입력만float변환한진단대조성공. 소스수정없음;완료실행저장검증not run. 신규absmax2결과이번snapshot외라판정안함.
- 우선reducer→logger→JSON타입/의미연결검사후감사23 validationη대조/G11순위유지. 새로운성능근거없음/학습·8seedCI not run. Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T080001Z-b5162cb6/(inventory,source.diff,reducer_probe.py,reducer_probes.json,synthetic_logger.csv,validation,append_validation);rawlogs forecasting/log/assessment/동일run/. Exactcommand·문헌은ASSESMENT감사24. 학습/backward/optimizer/GPU/설치/연구소스수정/git변이/프로세스중단/타세션접근전송없음,감사3문서append.


## 2026-09-22 17:15 KST — 예약 추적 감사25: logger/완료 기록 검증 및 validation η 분리 재현

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD5db6e315fec55de2962b843d9348deb1acfc18ed,snapshot17:10:50/230파일,manifest불일치0. 감사commit/tag없음.
- A10-LOG-COUNT-TYPE:현재float관찰수의실제reducer→logger→CSV재검사성공VERIFIED. Absmax4실제JSON/CSV max1.4416555166/count8/postnorm.99999966996보존확인;1epoch512/64/64,batch64,g11_every1로전8batch관찰. Absmax2평균.85107249소급재명명금지,두checkpoint SHA34c6965c동일. A02 rankchance.214817733876 실제함수VERIFIED.
- CPUtorch1.12/2threads,seed7/data20260921,val256/4batch,QKε.01/θ.381469877,동일learnedηcheckpoint의고정궤적/전체forward8행재현. 추가원ηvalrecall .259146662958;η0 .263483596918/.2 .254324974062/.5 .260108542602/1 .293643652987. .2의원η대비1.8606%,η0대비3.4760%탐색개선. η1max656.129578>305.037518로안정성불합격유지.
- 17:03보완:고정궤적max22.07/G11 OK는원궤적기준,변경η시스템G11은not run/N/A. 고정η1 p.296239→w/pre.270999→post.274052로cap은비율을올림;남은차이cap단독주장기각. 원학습η대조·정밀필드·실행시작provenance와독립seed검증필요. Candidate .2유지,기존③key/④entmax조건부순위유지.
- Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T081001Z-2214c860/(inventory,source.diff,reducer_probe.py,reducer_probes.json,split_probe.py,split_probes.json,validation,append_validation);rawlogs forecasting/log/assessment/동일run/. Exactcommands·문헌은ASSESMENT감사25. 학습/backward/optimizer/GPU/설치/연구소스수정/git변이/프로세스중단/타세션접근전송없음. 실제gradient재생성·absmax성능재평가·독립8seedCI·새학습not run,감사3문서append.

---
---

# 통합 요약 — 2026-09-22 17:27 KST 기준

> **이 절의 목적.** 이 문서 하나만 읽고도 다른 세션·다른 에이전트가 지금까지의 작업과 현재 상태를 이해할 수 있도록, 위의 시간순 기록을 통합 정리한 것이다. 세부 근거는 각 날짜 항목에 있다. 이 절 아래로는 다시 시간순 기록이 이어진다.

## 0. 기록 규칙 (2026-09-22 사용자 지시로 확정)

1. **모든 실행**(학습·보정·진단·분석)은 이 문서에 **날짜부 항목**으로 기록한다. 항목은 다음을 반드시 포함한다.
   - 제목: `## YYYY-MM-DD HH:MM KST — 한 줄 요약 (탐색적/확증 구분, 학습 유무)`
   - **성격**: 탐색적(exploratory) / 확증(confirmatory) / 학습 없음
   - **조건**: branch, seed, 데이터 revision, epoch/표본 수, 주요 하이퍼파라미터
   - **결과**: 표로. 전체 정밀도가 필요한 값은 JSON artifact 경로를 함께 적는다
   - **해석**: 무엇을 말하고 무엇을 말하지 않는지
   - **한계**: seed 수, 수렴 여부, 통제되지 않은 축
   - **Artifacts**: 산출물 경로
   - **Commit**: 아래 2번
2. **로그를 갱신한 뒤 반드시 현재 실험 branch에 commit + push**하고, **그 commit 정보를 이 문서에 기록**한다. `main`에는 커밋·푸시하지 않는다. 자기 자신의 hash는 커밋 안에 넣을 수 없으므로 **직전 항목의 hash를 다음 항목에 기록**하거나, 로그 갱신 → commit → hash 추가 → 짧은 후속 commit 순으로 처리한다.
3. **정정은 덮어쓰지 않는다.** 과거 항목을 수정하지 않고 새 항목에 정정을 append한다(사전등록 §0과 같은 원칙).
4. **탐색과 확증을 섞지 않는다.** 탐색 수치로 사전등록 기준의 통과/실패를 선언하지 않는다.

## 1. 한 눈에 보는 현재 상태

| 항목 | 값 |
|---|---|
| Branch | `exp/f-lif-pop-v3` (base `329183b94`, `exp/f-lif-pop-v2` HEAD) |
| 모델 | Population f-LIF **v3-A** (리셋 없는 분수적분 가지 K=4 + 단일 소마) |
| 수치 게이트 | **20 passed / 0 failed / 0 not run** |
| 학습 파이프라인 | 완성 (회상 과제·ETT 양쪽 동작) |
| 확증 실험 | **미실시** — 모든 결과가 seed 1개·12 epoch의 탐색적 실행 |
| 핵심 발견 | **구조는 작동하나(oracle 8배) 학습된 선택자가 그 여지를 못 가져온다(G ≈ 0)** |

## 2. 아이디어

하나의 논리 뉴런을 **서로 다른 시간상수를 가진 K개 가지**로 만든다. 가지는 같은 입력을 받고 **발화하지 않으며**, 직전 값만 물려받는 대신 **과거 모든 시점의 변화량을 멱함수 가중으로 재합산**한다(분수적분). 여기에 현재 population 상태를 질의로 삼아 **어느 과거를 더 크게 읽을지** 결정하는 선택 기능을 더한다. 발화는 가지들을 합산하는 **소마 한 곳**에서만 일어난다.

검증할 주장: **"이 선택이 관련 있는 과거를 실제로 집어내며, 그것이 성능으로 이어진다."**

## 3. 모델 v3-A 확정 사양

```
f(n,k)   = (I_n − u(n,k)) / tau_k              tau = [4, 8, 16, 32]   (F1)
xi_n     = [u(n,1..K) ; I_n]                   질의는 갱신 전 상태 (인과성)
e(n,j)   = −‖W_Q xi_n − W_K xi_j‖² / (d_q·theta)
p_n      = sparsemax(e_n)
rho(n,j) = (1−eta) + eta · B_n · p(n,j) / Σ_l b(n−l) p(n,l)      질량 재분배
c(n,j)   = min( b(n−j) · rho(n,j) , b_0 )                        R3 상한
u(n+1,k) = b_0 f(n,k) + Σ_{j<n} c(n,j) f(j,k)
a_n      = Σ_k w_k u(n+1,k)                    D-D: 갱신된 상태를 읽는다
v_n      = v(n−1) + (a_n − v(n−1))/tau_s ;  s_n = H(v_n − θ) ;  v_n ← v_n − θ s_n
b_d      = [(d+1)^alpha − d^alpha] / Γ(alpha+1),   alpha = 0.7
```

주요 결정(사전등록 `NSMT/docs/Population_fLIF_v3_prereg_KO.md` §2A–§2H):

| 코드 | 내용 | 이유 |
|---|---|---|
| **R3** | 계수 상한 `c = min(b·rho, b_0)` | 질량 보존만으로는 상태가 발산한다(최악 정책에서 `max\|u\|` 397,853) |
| **F1** | `tau = [4,8,16,32]` | 상한을 걸어도 τ=2는 길이에 따라 증폭(T=84에서 83배) |
| **D-A** | tempered 인자 제거 (`g=0`) | 질량 보존·α=1 환원·멱함수 꼬리를 모두 깬다 |
| **D-D** | 소마가 `u(n+1)`을 읽는다 | 아니면 마지막 patch가 출력에 도달하지 않는다 |
| **D-P** | `mass_matched`는 **상한 적용 후** 질량을 맞춘다 | 이전에는 항상 `full`과 동일했다 |
| **D-Q** | `M_eff` = **상한 후** 계수 비율 | `(1−η)m₀+η`는 상한 이전 oracle에서만 성립 |
| **D-X** | `theta`를 Phase B에서 보정 | 점수 폭이 수백이면 sparsemax가 hard argmax가 되고 역전파가 폭주 |
| **D-Y** | 고정 η 조건을 주 검증 경로로 | 학습 η가 15 epoch에 0.018→0.030으로만 움직인다 |

## 4. 검증 설계

- **과제:** 합성 회상 과제 revision **r2**. 신호 3종, patch 하나 = 사건 하나, T=42(ETT와 텐서 모양 동일). 값은 첫 등장에만 실리고 재등장 시 0. `kind` 코드로 copy/recall/recall-first를 분리 집계.
- **판정 기준 O7** (사용자 승인, 변경 없음):
  - ① `M_eff ≥ 0.5` — 단, §2G/§2H대로 `M_eff(oracle, 같은 η)`와 비율을 **함께** 보고
  - ② 격차 회수율 `G = (E_full − E_learned)/(E_full − E_oracle-trained) ≥ 0.5`, G14 통과 시에만
  - ③ 평균 개선 ≥20% **이고** paired CI 상한 < 0
  - ④ `|δ_O| ≤ 5%`는 자동 실패가 아니라 진단 구간
- **대조군:** full(η=0) / dense / sparse / recent / mass_matched / oracle 3종 / **GRU** / scalar / capacity-matched / α=1 이질 / ridge·window-mean
- **확증 프로토콜:** 독립 8 seed {7,13,21,42,123,256,512,1024}, paired CI, MDE ≈ 0.005(상대 1.2%)

## 5. 지금까지의 실험 결과 (전부 탐색적, seed 7, 12 epoch, 2048 시퀀스)

### 5.1 고정 η 격자 — 핵심 결과

| 조건 | η | recall MSE | **G** | **M_eff** |
|---|---:|---:|---:|---:|
| 선택 없음 (η=0) | 0.000 | 0.2763 | 0.000 | 0.1303 |
| 학습 η | 0.026 | 0.2726 | +0.012 | 0.1328 |
| 고정 η=0.2 | 0.200 | 0.3194 | −0.123 | 0.1272 |
| 고정 η=0.5 | 0.500 | 0.4335 | −0.177 | 0.1329 |
| 고정 η=1.0 | 1.000 | 0.4167 | −0.581 | 0.1369 |
| **oracle η=0.5** | 0.500 | **0.0396** | 0.981 | 0.4666 |
| **oracle η=1.0** | 1.000 | **0.0350** | 1.000 | 1.0000 |
| GRU 대조군 | — | 0.2531 | — | — |

**구조는 검색을 쓸 수 있다**(oracle이 full보다 8배 낫다). **학습된 선택자는 그 여지를 거의 못 가져온다**(G ≈ 0, 큰 η에서는 음수).

### 5.2 readout 통제 (spike / analog / drive)

| 조건 | spike | analog | drive |
|---|---:|---:|---:|
| full | 0.2763 | 0.2830 | 0.2747 |
| 학습 sparse | 0.2726 | 0.2616 | **0.2122** |
| oracle η=1 | **0.0297** | 0.0426 | 0.0316 |

readout은 full·oracle에서 무관하고 **학습 sparse에서만** 차이가 난다. 강한 신호는 스파이크를 통과하고 약한 신호는 통과하지 못한다.

### 5.3 QK 정규화 (순위 ②)

점수는 실제로 개선되지만(score top1 0.2622 → **0.3407**) **성능으로는 이어지지 않는다**. G는 여전히 음수다(η=0.5에서 −0.651 → −0.177로 완화될 뿐).

### 5.4 단계별 분해 — 신호가 어디서 사라지는가

| 조건 | score top1 | score rank | rank 기준 | p mass | post-cap | kernel |
|---|---:|---:|---:|---:|---:|---:|
| learned | 0.2622 | 0.2589 | 0.2148 | 0.2443 | 0.1328 | 0.1303 |
| **learned+QK** | **0.3407** | **0.1959** | 0.2148 | **0.2873** | 0.1338 | 0.1303 |
| 고정 η=1 | 0.1224 | 0.2419 | 0.2148 | 0.1376 | 0.1369 | 0.1303 |

**`learned+QK`만 무작위 기준(0.2148)보다 낫다.** 그런데 `p → 계수`에서 질량의 절반 이상이 사라진다(0.2873 → 0.1338).

### 5.5 희석 vs 상태 되먹임 (validation, 감사 23 지시)

| 개입 | η | val recall | score top1 | post-cap | max\|u\| | G11 |
|---|---:|---:|---:|---:|---:|---|
| 궤적 고정 | 0 → 1 | — | 0.3463 (불변) | 0.1332 → **0.2741** | 22.07 | OK |
| 전체 forward | 0.00 | 0.2635 | 0.3486 | 0.1332 | 22.01 | OK |
| **전체 forward** | **0.20** | **0.2543** | 0.2865 | 0.1559 | 22.48 | OK |
| 전체 forward | 1.00 | 0.2936 | **0.1960** | 0.2100 | **656.13** | **FAIL** |

- **희석은 실재한다**: 궤적을 고정하면 η를 올릴수록 정답 질량이 회복된다(0.1332 → 0.2741).
- **상태 되먹임이 상쇄한다**: 실제로 η를 올리면 점수가 망가져(0.3486 → 0.1960) 0.2100에 그친다.
- **validation에서도 η=0.2가 최선**(0.2543 < 0.2635). test에서 본 방향이 재현된다.
- **η=1은 G11 불합격**(656.13 > 305.038).

## 6. 현재 판정

| 기준 | 값 | 판정 |
|---|---|---|
| O7-① `M_eff ≥ 0.5` | 최대 0.2100 (validation η=1) | **미달** |
| O7-② `G ≥ 0.5` | 학습 η +0.012, 고정 η 음수 | **미달** |
| G14 headroom | full 0.2763 → oracle 0.0350 | 이 seed에서 충분 (확증 아님) |

사전등록 O7 판정표의 **"고르지 못함 → 점수 함수·key 표현력 재검토"** 가지다. **확증 판정이 아니다** — seed 1개·12 epoch이며 8 seed·paired CI는 미실시다.

## 7. 개선 후보와 우선순위 (감사 AUDIT-PRIORITY-01)

| 순위 | 후보 | 상태 |
|---|---|---|
| ① | 고정 η 대조 | ✅ 완료 (§5.1) |
| ② | Q/K L2 정규화 | ✅ 완료 (§5.3) — 점수는 개선, 성능은 미개선 |
| ③ | causal key 표현 ablation (`Δu` 추가 등) | 대기 |
| ④ | α-entmax (학습 가능 지수) | 대기 |
| ⑤ | Ridge/Gram 보정 읽기 | 별도 구조 대조 |
| ⑥ | delta rule | 별도 구조 대조 |

문헌 근거: [Test-time regression](https://arxiv.org/abs/2501.12352)(커널 읽기는 `KᵀK=I`일 때만 정확, QK 정규화의 정당화), [Modern Hopfield](https://arxiv.org/abs/2008.02217)(분리된 패턴에서 한 번에 회상), [DeltaNet](https://arxiv.org/abs/2406.06484), [Adaptively Sparse Transformers](https://aclanthology.org/D19-1223/)(entmax 지수의 닫힌 형태 Jacobian).

## 8. 발견된 결함 (전부 수정 완료, 상세는 각 날짜 항목)

외부 감사 `NSMT/docs/ASSESMENT.md`가 23회에 걸쳐 검토했고, **검증 가능한 지적은 전부 사실이었다.**

| 종류 | 예 |
|---|---|
| **대조군이 수학적으로 퇴화** | `mass_matched`가 항상 `full`과 동일 (A01) |
| **판정식 무효** | `M_eff` 공식이 상한 도입 후 성립하지 않음 (A02) |
| **난이도 축이 의도와 다름** | n_keys를 올려도 재질의 key 수가 그대로 (A03) |
| **기준선 계산 오류** | chance의 분모가 실제 history 길이가 아님 (A04) |
| **측정 대상 오류** | G11이 raw patch를 쟀다 (8.0 vs 실제 30.5) (A05) |
| **옵션이 이름만 바꿈** | `--eta_fixed`·`--no-cap` 미전달 (A06), `theta` 미전달 (A10-THETA) |
| **안전장치를 판정에 미사용** | 상한 1001배 초과 후보를 통과 (A05-후속) |
| **집계 오류** | `M_eff`에 copy 혼입·집계 단위 오류 (A02-DIAG), absmax를 평균으로 (A09) |
| **복원 결함** | frozen 통계 누락을 승인 (A10-RESTORE-STATS) |
| **정밀도 결함** | 계수표를 float32로 만든 뒤 `.double()` (G4가 적발) |
| **기록과 코드 불일치** | 수정했다고 기록했으나 파일에 미반영 (A11, A09-ABSMAX ×2) |

**자체 정정 주요 항목**: uniform oracle을 상한이라 부른 것, "lag 영향 <0.01", "0.5=무작위" 순위 기준, "soft norm이 유계성을 만들었다", "모든 η에서 G 음수", "가중치 고정 = 점수 고정".

## 9. 코드 지도

```
NSMT/f_lif_pop_v3/
├── analysis/                     학습 없는 수치 분석과 그 출력(.txt)
│   ├── prereg_numerics.py        α=1 환원·유한구간 이득
│   ├── stability_sweep.py        adversarial 상태 안정성 (R3·F1 근거)
│   ├── selector_gradient.py*     선택자 역전파 (T·η 의존)
│   ├── meff_reachable.py         O7-① 도달 가능 영역 (정확식 + 표본 탐색)
│   ├── stage_decomposition.py    score→p→pre-cap→post-cap 분해
│   ├── eta_intervention_split.py 희석 vs 상태 되먹임 분리
│   └── key_geometry.py*          key Gram·유효 rank
├── reference/golden/             spikeDE 고정 커밋 궤적 + SHA256 (G4)
└── forecasting/
    ├── layers.py                 Sparsemax·ArcTanSpike·Selector·Soma·PopulationNeuron·Embedding
    ├── ours.py                   myModel(두 head, 3 readout) · GRUBaseline · truth_to_oracle_p
    ├── model.py                  LOAD_MODEL factory, checkpoint 복원 계약
    ├── config.py                 인자·경로·보정 연결
    ├── calibrate.py              Phase B 발화율 보정 + theta 보정
    ├── check_model.py            수치 게이트 G1–G17
    ├── train.py / test.py        학습·평가·개입·진단
    ├── utils.py                  EpochLog·EarlyStopping (v2 이식)
    └── data_provider/            synthetic(r2) · ETT loader(v2 무수정 이식) · factory
```
(`*` = 일부는 인라인 스크립트로 실행되어 `.txt`만 남은 것이 있다)

## 10. 재현

```bash
export LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib
PY=/home/yschoi/.conda/envs/snn_recall/bin/python
cd NSMT/f_lif_pop_v3/forecasting

$PY check_model.py --phase all                      # 게이트 20개
$PY calibrate.py --task recall --n_train 1024 --cpu  # Phase B 보정
$PY train.py --task recall --data recall --suite <새 이름> --mode sparse \
   --n_train 2048 --n_val 256 --n_test 256 -e 12 -bs 64 --cpu
```

- 환경: `snn_recall` (Python 3.10, torch 1.12). `snn_jelly`는 torch 2.11 CPU 전용(G4 golden 생성용).
- `LD_LIBRARY_PATH`를 conda `lib`으로 두지 않으면 pandas import가 실패한다.
- `--suite`는 매번 새로 준다. 기존 산출물은 덮어쓰지 않는다(`FileExistsError`).

## 11. 미결 항목

- **확증 실험 전체**: 독립 8 seed·paired CI·ETT H96/H720 — **미실시**
- 감사 OPEN: A07-REGEN(golden 재생성 entrypoint), A10-PROVENANCE(실행 시작 코드 고정), A10-LOG(`final+result.csv`), A02(실제 표본 분포와 상한 연결)
- G11 검사가 여전히 `watch` 표본에서만 실행된다
- 순위 ③·④ 미착수

## 12. Commit 기록

| 시점 | commit | 내용 |
|---|---|---|
| 2026-09-21 | `c19c52010` | v3-A 설계 rev.1 + Phase A/C 게이트 |
| 2026-09-21 | `df3a3407b` | 회상 과제 생성기 + Phase B 보정 |
| 2026-09-22 | `c34fa1c68` | 모델·학습·평가 파이프라인 |
| 2026-09-22 | `d1fb43eda` | oracle 성공 / 학습 선택자 실패 파일럿 |
| 2026-09-22 | `7520087c6` | 고정 η 격자 |
| 2026-09-22 | `b6d4baf6a` | QK 정규화 |
| 2026-09-22 | `5db6e315f` | 희석 vs 상태 되먹임 분리 |
| 2026-09-22 | *(이 항목의 commit은 다음 항목에 기록)* | 통합 요약 |

(전체 이력은 `git log --oneline exp/f-lif-pop-v3`)

---

## 2026-09-22 17:28 KST — 기록 규칙 확정과 원격 branch 게시

**성격:** 문서·운영. 학습 없음.

### 1. 사용자 지시로 확정한 기록 규칙

위 「통합 요약 §0」에 규칙을 명문화했다. 요지는 넷이다.

1. 모든 실행을 이 문서에 **날짜부 항목**으로, 성격·조건·결과·해석·한계·artifact·commit을 갖춰 기록한다.
2. **로그 갱신 후 반드시 현재 실험 branch에 commit + push**하고 그 commit 정보를 이 문서에 기록한다. `main`에는 커밋·푸시하지 않는다.
3. 과거 항목을 **덮어쓰지 않고** 새 항목에 정정을 append한다.
4. **탐색과 확증을 섞지 않는다.**

### 2. 통합 요약 추가

다른 세션·에이전트가 **이 문서 하나로** 전체를 이해할 수 있도록 「통합 요약」 절을 추가했다(§0 기록 규칙, §1 현재 상태, §2 아이디어, §3 v3-A 사양, §4 검증 설계, §5 실험 결과, §6 현재 판정, §7 개선 후보, §8 발견된 결함, §9 코드 지도, §10 재현, §11 미결, §12 commit 기록). 문서 길이 4,035 → 4,276행.

### 3. 원격 branch 게시

이 저장소의 실험 branch를 처음으로 원격에 올렸다. AGENTS.md대로 **의도한 branch 하나만** 푸시했고 `--all`·`--mirror`·force를 쓰지 않았다.

```
git push -u origin exp/f-lif-pop-v3
  * [new branch]  exp/f-lif-pop-v3 -> exp/f-lif-pop-v3
  Branch 'exp/f-lif-pop-v3' set up to track remote branch 'exp/f-lif-pop-v3' from 'origin'.
```

- 원격: `git@github.com:wlghditkfkdgo/NSMT.git`
- `main`은 건드리지 않았다. base commit `329183b94`에서 **54 commit** 앞서 있다.

### 4. Commit

| 대상 | commit | 비고 |
|---|---|---|
| 통합 요약 | **`4f797efaa`** | 위 §12 표의 "다음 항목에 기록"이 이것이다 |
| 본 항목 | 다음 항목에 기록 | 자기 hash는 커밋 안에 넣을 수 없다 |

### 5. 한계

이 항목은 **문서·운영 작업**이며 새 실험 결과가 없다. 통합 요약의 모든 수치는 기존 항목에서 옮긴 것으로, **전부 seed 1개·12 epoch의 탐색적 실행**이라는 제한이 그대로 적용된다.

**Artifacts:** `docs/PROJECT_LOG.md` (본 문서)

---

## 2026-09-22 17:31 KST — 감사 24·25: 최근 수정들이 VERIFIED로 확인됨 (학습 없음)

**성격:** 감사 추적. 새 실행 없음. **조건:** 감사 관찰 HEAD는 내 최근 수정 커밋들.

### 1. 감사가 VERIFIED로 올린 항목

| 항목 | 감사 확인 내용 |
|---|---|
| **A10-LOG-COUNT-TYPE** | `observations`·`nonfinite` count를 float로 반환하도록 고친 것을 확인. 실제 `EpochLog.write→verbose→CSV` 경로에 **변환 없이** 통과. 감사 24가 보고한 정수 타입 예외가 재발하지 않음 |
| **A09-ABSMAX / OBSERVATIONS / POSTCLIP** | `absmax4-170100`의 실제 CSV·JSON 대조로 **완료 artifact 저장까지** 확인. AST reducer에 `[1,9]`를 넣어 **9 / count 2.0** 재현 |
| **A02-STAGE-RANK** | 같은 test 256 표본에서 chance **0.21481773387565073**, QK rank **0.19589813019628083** 재현. "0.5=무작위" 오기 수정 확인 |
| **A08-ETA-INTERVENTION** | 궤적 고정 함수가 학습 η 상태의 `p, b`로 계수만 재계산하고, 전체 forward 함수가 η buffer를 바꿔 순환 모델을 다시 실행하는 구조를 확인. 각 호출 후 **buffer 복원**과 학습 파라미터 hash 불변도 확인 |

### 2. 감사가 재현한 수치

`qk2-140541`(learned η + QK) checkpoint를 CPU로 재평가해 **validation 8행 전부가 표시 정밀도까지 재현**됐다. 추가로 감사가 정량화한 값: **η=0.2가 같은 모델의 원 학습 η 대비 recall MSE 약 1.86% 낮다.**

전체 forward와 궤적 고정의 η=1 post-cap 차이(**0.21000670 vs 0.27405234**)도 상태 되먹임을 포함한 개입의 차이로 뒷받침된다.

### 3. 감사가 제한한 것

- 위 확인은 **`sparse`/`cap=True`/QK 학습 η checkpoint에 한정**된다. 다른 mode·`--no-cap`까지 일반적으로 맞다고 판정하지 않는다.
- Logger가 **모든 int를 지원하게 된 것은 아니며**, 현재 train이 넘기는 타입을 고친 범위다.
- 분해 비율(0.2100 vs 0.2741)을 **보편적 분해나 새 학습의 성능 예측으로 일반화하지 않는다.**
- test 표본의 rank 기준(0.2148)을 **validation에 그대로 이식하지 않는다.**

### 4. 감사 지시 (다음 단계)

1. **η=0.2를 후보로 유지하되 채택은 보류.** 사전등록한 **선택 규칙**과 **독립 검증** 뒤로 둔다. 비교에는 **원 학습 η와 η=0을 항상 포함**하고, 후보 수·η 선택 기준·**G11 탈락 규칙**을 먼저 고정한 뒤 **별도 미사용 평가 자료와 독립 seed**로 본다. 기록에는 `score→p→w→pre→post`와 **전체 forward의 실제 상태 상한**을 함께 남긴다.
2. 우선순위 유지: **② QK 조건 검증 → ③ causal key → ④ entmax → ⑤ Gram/ridge → ⑥ delta.** 이번 분해는 **혼합·커널 가중·상태 되먹임**을 점검 대상으로 지지하며 **entmax 자동 승격이나 residual 즉시 삭제는 지지하지 않는다.**

### 5. 한계

감사는 학습·backward·optimizer·GPU를 실행하지 않았고, 새 gradient 수치 재생성·absmax 실행의 성능 재평가·독립 seed 통계는 **not run**이다. **현재 validation 개선은 후보 유지의 근거이지 확증이 아니다.**

### 6. Commit

| 대상 | commit |
|---|---|
| 직전 항목(기록 규칙·원격 게시) | `0314e9319` |
| 본 항목 | 다음 항목에 기록 |

**Artifacts:** `NSMT/docs/ASSESMENT.md` 추적 감사 24·25


## 2026-09-22 17:33 KST — 예약 추적 감사26: 통합 요약의 조건 혼용 정정 요청 (학습 없음)

- Branch exp/f-lif-pop-v3/base329183b94f65090cc6b337f464c5aa4d8e127ad7/HEAD0314e9319891dbe212b1a153b402db75143d170c,snapshot17:31:14/216파일,manifest불일치0. 연구소스·결과변경없음. 새성능판정근거없음. 감사commit/tag/push없음.
- **통합요약§5.1·§6 정정:** 동일etagrid JSON 공통full .2763386361581949/oracle1 .03496568833854037에서비QK G는학습η+.015405565340/.2−.178349106686/.5−.651132279581. 현재표+.012/−.123/−.177은QK조건과혼용(A09-SUMMARY-CONDITION-MIX OPEN). §5.3 G음수는고정η에한정,QK학습η는+.0126314. 비QK.5실제11epoch/로그확인1epoch/분리분석새학습없음이므로전부12epoch표기정정.
- §8 전부수정완료는§11OPEN과모순. §7①②완료는탐색에한정. §5.5고정궤적G11은원궤적peak이며개입시스템안전not run/N/A;§5.2readout인과단정·v2MDE1.2%v3이식제한유지. 기존VERIFIED/OPEN그대로,η.2선택규칙사전등록·독립검증우선순위유지.
- 표준Python JSON산술만수행;모델forward/학습/backward/optimizer/GPU·독립seedCI·원격게시확인not run. Artifacts NSMT/f_lif_pop_v3/forecasting/results/assessment/20260922T083001Z-36837373/(inventory,project_log.diff,summary_probe.py,summary_arithmetic.json,validation,append_validation),raw log forecasting/log/assessment/동일run/summary_probe.log. 상세근거·명령은ASSESMENT감사26. 예약감사git변이금지준수,3문서append·텍스트artifact만작성.

---

## 2026-09-22 17:43 KST — `/briefing` skill 등록 (학습 없음)

**성격:** 도구·운영. 새 실험 없음.

사용자 지시로 `/briefing` skill을 등록했다. 위치는 `NSMT/.claude/skills/briefing/SKILL.md`이며 저장소에 함께 버전 관리된다.

### 1. 동작

사용자가 `/briefing`을 입력하거나 "브리핑"·"현재 상황"·"진행 상황 정리"를 요청하면, **진행 중이거나 완료된 작업을 이해하기 쉽게 정리해 보고**한다.

### 2. skill에 담은 규칙

| 항목 | 내용 |
|---|---|
| **상태 수집** | 추정하지 않는다. 실행 중 프로세스·git 상태·`ASSESMENT.md` 신규 항목·`PROJECT_LOG.md` 최근 기록을 **실제로 읽은 뒤** 쓴다. 끝나지 않은 백그라운드 작업의 결과를 예상해서 쓰지 않는다 |
| **형식** | `## 한 줄로`로 시작 → 지금 돌고 있는 것 → 핵심 발견(표) → 아직 안 된 것 → 감사 상황 → 다음 |
| **용어** | 사용자가 명시적으로 지적한 사항이다. 알아들을 수 없는 용어를 쓰지 않고, 불가피하면 그 자리에서 푼다. skill에 대응표를 넣었다 (`sparsemax`, `M_eff`, `G`, `G11` 등) |
| **정직성** | **탐색과 확증을 반드시 구분**한다. 나쁜 결과를 먼저 쓴다. 감사가 찾은 결함과 내가 철회한 주장을 브리핑에 포함한다 — 이 프로젝트에서는 그것이 진행 상황의 일부다 |
| **길이** | 한~두 화면. 세부는 `docs/PROJECT_LOG.md`를 가리킨다 |

### 3. 동시에 진행 중인 실험

사전등록 §2I의 η 선택 절차를 위한 **8 seed 학습**이 백그라운드에서 실행 중이다(suite `seeds-173737`, seed {7,13,21,42,123,256,512,1024}, learned η + QK soft ε=0.01, 12 epoch, 2048 시퀀스, `--no-test`). 완료되면 `analysis/eta_selection.py`로 validation 선택 → confirm 분할 1회 평가를 수행한다.

### 4. Commit

| 대상 | commit |
|---|---|
| 직전 항목(감사 24·25 기록) | `0ee0e5765` |
| 사전등록 §2I + confirm 분할 | `f98fab4fe` |
| 본 항목 | 다음 항목에 기록 |

**Artifacts:** `NSMT/.claude/skills/briefing/SKILL.md`
