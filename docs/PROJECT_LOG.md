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
