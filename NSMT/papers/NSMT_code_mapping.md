# NSMT 방법 ↔ 코드 구현·옵션 대응

작성 2026-10-14, 사용자 확인 2026-10-14 (1·2 확정). `papers/NSMT_method_summary.md` 의 각 요소가 **이 저장소의 어느 코드·어느 옵션**인지 찾은 결과와 근거다. 이 대응을 옮겨 둔 실행 가능한 사본이 `NSMT/nsmt_base/` 다.

## 0. 결론

| 항목 | 결과 |
|---|---|
| 논문 코드 (forecasting) | 저장소 루트의 `forecasting/` (`train.py`, `ours.py`, `layers.py`, `config.py`, `model.py`, `test.py`, `analysis.py`). 업스트림 `origin/main` (`191f366c6`, 논문 각주의 공개 저장소) 의 `forecasting/` 와 같은 코드이며, 루트 사본은 `ours.py`에 rebuttal용 `shuffle_replay` 플래그(기본 False → forward 동일)와 `config.py`에 iTransformer 인자만 더 있다. 공개 저장소에는 `test.py`가 import 하는 `analysis.py`, `positional.py`, `noise_injector.py`가 빠져 있고 루트에만 있다. |
| 모델 클래스 | `forecasting/ours.py: myModel` (timm 이름 `mymodel`, `--model myModel`) |
| 논문 Table 13 의 32칸 전부 | 루트 `forecasting/{final3,final2,result}/…` 의 저장된 실행과 **소수 셋째 자리까지 모두 일치** (`final+result.csv` 첫 행). 데이터셋마다 **네 horizon을 모두 재현하는 설정이 하나씩** 존재(electricity만 둘). → §3 |
| 이상 탐지 (Table 9) | 루트 `anomaly_detection/result/` 의 122개 실행 중 논문 F1 과 일치하는 것이 **없다** (SMD 최고 F1 89.05 vs 논문 86.94, SMAP 72.56 vs 75.75, PSM 96.64 vs 95.46 …). 논문 수치를 낸 실행은 이 기계에 없다. 루트 `anomaly_detection/` 코드도 공개본과 다르다(`model.py`, `train.py`, `test.py`, `layers.py`, `config.py`). → §5 |
| 분류 (Table 10) | 5개 데이터셋 모두 정확도(소수 1자리)가 일치하는 실행이 있으나 **여러 설정이 같은 정확도**를 낸다(test 표본이 작아 동률). FaceDetection·JapaneseVowels 은 공개 스크립트 설정과 같은 실행이 있고, EthanolConcentration·Handwriting·Heartbeat 는 공개 스크립트와 정확히 같은 설정의 실행이 없다. → §5 |
| `NSMT/model_v1/forecasting` | 같은 계보의 **확장판**. 기본값이 논문과 다르다(§4). 논문 재현에는 쓰지 않는다. |

## 1. 확인 방법
1. 업스트림 논문 저장소(`origin/main`)의 파일을 로컬 사본들과 해시로 비교했다.
2. 루트 `forecasting/` 아래 `myModel` 실행 1,096개의 `log/final+result.csv`(첫 행 = 학습 직후 test) 를 읽어, 논문 Table 13 의 MSE/MAE 와 소수 셋째 자리에서 비교했다. 스크립트 `NSMT/scripts/nsmt_paper_match_runs.py`, `nsmt_paper_consistent_configs.py`; 출력 `papers/code_mapping_evidence/table13_matches.txt`, `table13_consistent_configs.txt`.
3. 일치한 실행의 `logargs.txt`(전체 인자 덤프)에서 옵션을 읽었다.
4. 읽은 옵션을 `origin/main:forecasting/scripts/*_96.sh` 와 대조했다. 스크립트 일부는 마지막 sweep 상태로 남아 있어(예: `ETTh1_96.sh`는 `ab1 ab3` 모델 루프, `ETTh2/ETTm1/ETTm2_96.sh`는 keep_ratio sweep) **스크립트 자체가 최종 설정은 아니다**. §3 표의 값은 저장된 실행에서 읽은 것이다.
5. 이상 탐지·분류는 `NSMT/scripts/nsmt_paper_match_ad_cls.py` 로 같은 방식(F1 소수 2자리 / 정확도 소수 1자리)으로 대조했다. 출력 `papers/code_mapping_evidence/ad_cls_matches.txt`.

## 2. 방법 요소 ↔ 코드 위치 (forecasting, `origin/main:forecasting/ours.py` 기준)

| 논문 요소 | 코드 | 비고 |
|---|---|---|
| LIF 뉴런 (§2.2) | `spikingjelly.clock_driven.neuron.MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy', surrogate_function=surrogate.Sigmoid())` (`layers.py: SpikLinearLayer`, `ours.py: Embedding`) | V_th = 1, V_reset = 0 (hard reset), decay_input=True → 논문 식과 같음. surrogate 는 spikingjelly `Sigmoid(alpha=4)`: 역전파 `4σ(4x)(1−σ(4x))`. 논문 (14)의 σ(x)=1/(1+e^{−x})와 **기울기 척도(alpha=4)가 다름**. `--tau 2.0` |
| 패치화 (§3.2) | `myModel.__init__`: `stride = patch_size // 2`, `num_patches = (L−P)/stride + 1 + 1`, `ReplicationPad1d((0, stride))`; `forward`: `x.unfold(-1, P, stride)` | **50% 겹침 패치 + 끝에 복제 패딩 한 패치** — 논문 본문에는 없는 세부. `--patch_size`, `--seq_len` |
| T = N | `self.T = self.num_patches`; `x.unsqueeze(0).repeat(self.T, …)` (direct coding, T번 반복) | 해마 경로는 같은 입력을 T번 반복, 신피질 경로는 아래 평균 풀링으로 T를 접고 N을 시간 축으로 씀 |
| 윈도우 정규화 | `forward`: `means`, `stdev` 로 윈도우·채널별 표준화, head 출력에서 복원 | 논문 본문에 없음 (RevIN류, affine 없음). 옵션 없음(항상 적용) |
| Spiking Embedding (1)-(2),(5) | `ours.py: Embedding` = `nn.Linear(P, D, bias) → BatchNorm1d(D) → MultiStepLIFNode` | `-emb`(D), `--no-bias` (논문 실행 전부 bias 없음) |
| 신피질 입력 정렬 | `forward`: `patched_x.transpose(0, 2).mean(2, keepdim=True)` → `[N, BC, 1, P]` → `encoding_neo` | 평균 풀링(T) 후 N이 dim 0(시간 축) |
| 신피질 Spiking MLP × M | `TemporalBlock`: `SpikLinearLayer(D, D)` × `time_num_layers` | `--time_layers` = M |
| 저주파 목표 (3), Alg. 1 | `ours.py: dct_matrix / dct_time / idct_time / lowpass_memory(x, keep_ratio, time_dim=2, center=True)`; `k = max(1, ceil(T·keep_ratio))`, `mask[:k]=1` | `--keep_ratio` = k. 입력은 정규화·패딩된 신호(길이 L+stride) 이고 평균 제거 후 복원(center=True). 목표는 패치화·T평균 후 `self.org_x` `[N, BC, 1, P]` |
| 약한 디코더, L_aux (4) | `ours.py: Decoder` = `nn.Linear(D, P, bias)` 하나; `train.py: rec = ((org_x − rec_x)**2).mean()` | 추론 시 `init_testing()`에서 삭제 |
| 융합 (6) | `Block.forward`: `mx = x * mx.transpose(0, 2)` | `x` = X^h `[T, BC, N, D]`, `mx` = X_neo `[N, BC, 1, D]` → `[1, BC, N, D]` |
| Memory Replay (7)-(8) | `layers.py: MutualCrossAttention(q=x, kv=mx)`: Q←X^h, K,V←X^f 각각 Linear–BN–LIF, `A = (q @ kᵀ) * D^{-0.5}`, `attn_lif(A @ v)`, `proj_linear = SpikLinearLayer` | softmax 없음. `-nh` = h. 스케일 s = D^{−1/2} |
| **신피질으로 가는 기울기 제한** | `Block.forward`: `mx = 0.05*mx + 0.95*mx.detach()` | **논문에 없음.** memory replay 를 통해 신피질로 흐르는 과제 기울기를 5%로 줄인다 (이상 탐지 코드는 0.5). 옵션 없음(상수) |
| IAND 잔차 (9)-(10) | `x = x * (1. − self.mca(x, mx))`; `x = x * (1. − self.mlp(x))` | `--connect_f IAND` 인자는 있지만 **코드에서 읽지 않는다** (항상 IAND) |
| FFN (Spiking MLP 2개) | `layers.py: MLP` = `SpikLinearLayer(D, D·r) → SpikLinearLayer(D·r, D)` | `--mlp_ratios` = r |
| 블록 수 | `data_block = [Block] × depths` | `--layers` (논문 실행 전부 1) |
| Head | `nn.Linear(D·N, pred_len, bias)` 를 스파이킹 단계 T마다 적용 → `[T, B, pred_len, C]` | 학습: T단계 각각에 L1; 평가(`analysis.extract_pred_true_for_metrics(reduce_T="mean")`): **T평균 예측**으로 MSE/MAE |
| L_total (11) | `train.py: loss = ce*(1−ratio) + rec*ratio`, `ce = nn.L1Loss()` | `ratio = args.alpha * (pred_len/720)` — **논문에 없음**: 실제 α는 `--alpha × pred_len/720` (예: pred 96이면 α의 2/15) |
| 최적화 | `AdamW(lr, weight_decay=6e-2)`; `ReduceLROnPlateau(mode='min', patience=2, factor=0.1, cooldown=1)` on val weighted loss; early stopping on val `ce`(T단계 L1), patience=`--patience` | `-lr`, `--weight_decay`(기본 0.06), `--scheduler reduce`, `-e`, `--patience`, `-bs` |
| 평가 모델 선택 | `EarlyStopping.save_checkpoint` 는 `checkpoint.pth` 에 best 를 쓰지만, `train()` 은 루프가 끝난 **마지막 epoch 상태**를 `best+model.pt` 로 저장하고 `test()` 가 그것을 읽는다 | 즉 test 수치는 best-val 가중치가 아니라 **early stop 시점(또는 마지막 epoch)의 가중치**. 논문에는 명시 없음 |
| 데이터 | `data_provider/data_loader.py`: ETT 12/4/4개월 분할, custom(electricity/traffic/weather/exchange) 7:1:2; 학습 구간 StandardScaler | `--data {ETTh1,ETTh2,ETTm1,ETTm2,custom}`, `--root_path`, `--data_path`, `--features M`, `--c_in` |
| `--gating` | `attn` 또는 `original` 은 **동작 동일**(`attn` 인자는 Block에서 안 씀); `ablation` 은 신피질·디코더 삭제 | 논문 실행은 모두 `--gating attn` |
| 시드 | `set_random_seed(seed)` (torch/numpy/random/cudnn deterministic) | 논문 본문 수치 = `--seed 42` (부록 E.2 와 일치) |
| 환경 | README: Python 3.10.18, PyTorch 1.12.0+cu113, SpikingJelly 0.0.0.0.14, timm 1.0.19; LIF backend `cupy` | 로컬 `snn_recall` env 가 이 torch 버전 |

## 3. 논문 Table 13 을 그대로 내는 옵션 (저장된 실행에서 읽음; 확정)

공통: `--model myModel --gating attn --no-bias --scheduler reduce -s --test --warm_up_epoch 0 --layers 1 --max_ratio 2 --tau 2.0 --seq_len 96 --seed 42 --weight_decay 0.06(기본) --features M`

| 데이터 | `--data` / `--c_in` | patch | keep_ratio k | emb D | heads | α (`--alpha`) | mlp_ratios | time_layers M | lr | bs | epoch | patience | 저장 위치 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | ETTh1 / 7 | 8 | 0.25 | 32 | 2 | 0.8 | 1.0 | 2 | 5e-4 | 64 | 50 | 3 | `forecasting/final3/ETTh1/260206/` |
| ETTh2 | ETTh2 / 7 | 8 | **0.3** | 64 | 8 | 0.8 | 0.5 | 1 | 5e-4 | 64 | **100** | 2 | `forecasting/final3/ETTh2/260206/` |
| ETTm1 | ETTm1 / 7 | 8 | 0.25 | 64 | 8 | 0.5 | 1.0 | 2 | **1e-3** | 64 | 50 | 3 | `forecasting/final3/ETTm1/260206–07/` |
| ETTm2 | ETTm2 / 7 | 8 | 0.25 | 64 | 16 | 0.5 | 1.0 | 2 | 5e-4 | 64 | 50 | 2 | `forecasting/final3/ETTm2/260207/` |
| traffic | custom / 862 | 32 | 0.25 | 512 | 8 | 0.5 | 2.0 | 2 | 5e-4 | 4 | 50 | 2 | `forecasting/result/traffic/260203–04/` |
| electricity | custom / 321 | 24 | 0.25 | 512 | 16 | 0.1 | 1.5 | **3** (2도 동일 수치) | 5e-4 | 8 | 50 | 2 | `forecasting/result/electricity/260203/` |
| exchange | custom / (c_in 1 기록; 스크립트 미지정) | 24 | 0.2 | 128 | 8 | 0.9 | 2.0 | 2 | 5e-4 | 64 | 50 | 3 | `forecasting/final3/exchange/260206/` |
| weather | custom / 21 | 8 | 0.15 | 128 | 8 | 0.1 | 6.0 | 1 | 5e-4 | 32 | 50 | 2 | `forecasting/final3/weather/260206–07/` |

- 각 데이터셋의 네 horizon(96/192/336/720)이 위 한 설정으로 모두 일치했다. 같은 설정을 다른 날짜에 다시 돌린 사본(`final2/`, `final3/patch_size/`, `rep/`)도 같은 수치를 냈다(결정적 실행).
- `origin/main:forecasting/scripts/{weather,electricity,exchange,traffic}_96.sh` 는 위 표와 정확히 같다. `ETTh1_96.sh`(모델 루프 `ab1 ab3`), `ETTh2_96.sh`(k sweep, 720만), `ETTm1_96.sh`·`ETTm2_96.sh`(k sweep) 는 sweep 상태로 남아 있다. ETT 네 개의 논문 설정은 `NSMT/nsmt_base/forecasting/scripts/run_ett.sh` 에 옮겨 두었다.
- exchange 의 `--c_in`: 저장 인자에는 1(기본값)로 남아 있고 스크립트도 지정하지 않는다. `c_in`은 모델 forward 에서 쓰이지 않아 결과에 영향이 없다.
- 보조 손실의 실제 가중치는 `α·pred_len/720` 이다 (§2 표 참조).

## 4. `NSMT/model_v1/forecasting` 와의 차이 (논문 재현에 쓰지 않는 이유)
`model_v1`(= `neorecall_v1` 복제본)은 같은 `myModel`에 수십 개 실험 옵션을 덧붙인 판이다. 기본값으로 돌려도 논문 코드와 다르다:
1. `--neo_tau` 기본 **8.0** (논문 코드는 신피질 LIF도 τ = 2.0). `model.py: load_mymodel` 이 `neo_tau=getattr(args, 'neo_tau', 8.0)` 를 넘긴다.
2. 신피질 층이 `NeoLayer`(단일 스텝 LIF 루프, `neo_exact=True` 기본)로 바뀌어 있다. 수학적으로 같다고 주석에 적혀 있으나 cupy `MultiStepLIFNode` 경로와 비트 동일성은 확인하지 않았다 (`--neo_legacy` 가 옛 경로).
3. `train.py` 검증이 **T평균 예측**으로 모델 선택(review 7.6)을 한다. 논문 코드는 T단계 L1(`ce`)로 early stopping 한다.
4. `test.py` 에 firing-rate 그림 등 추가.

## 5. 이상 탐지·분류 대조 결과
**코드.** 공개본은 `origin/main:anomaly_detection/`, `origin/main:classification/` (→ `NSMT/nsmt_base/` 에 그대로 복사). 루트 `anomaly_detection/` 의 `config/layers/model/test/train.py` 는 공개본과 다르다(이후 수정본). 루트 `classification/` 은 `dataloader.py`, `encoder.py` 만 다르다.

**구조 차이.** 이상 탐지 `Block.forward` 의 신피질 기울기 비율은 **0.5**(forecasting 0.05), head 는 `Linear(D·N, seq_len)` 재구성, `--anomaly_ratio`·`--c_out`. 분류는 `-T 24` 스파이킹 스텝과 `--patch_stride` 를 따로 받고 head 는 `Linear(D, classes)`, cosine 스케줄; 저주파 목표는 패치가 아니라 입력 전체(`x [B, M, L]`)에 DCT.

**이상 탐지 (Table 9).** 저장된 122개 실행 중 P/R/F1 이 논문과 일치하는 것이 없고 F1 만 일치하는 것도 없다. 데이터셋별 최고 F1(모두 `--gating original`): SMD 89.05, MSL 79.78, SMAP 72.56(ablation)/68.79, PSM 96.64, SWaT 94.18(ablation)/92.28 — 논문 86.94 / 76.51 / 75.75 / 95.46 / 92.56. 저장된 실행의 설정(patch 16, emb 64/128)은 공개 스크립트의 설정(SMD patch 8·emb 32 등)과도 다르다. **논문 AD 수치를 낸 실행·설정은 이 기계에서 확인할 수 없다.** 공개 스크립트 값: SMD(emb32, nh8, α0.2, k0, M2, mlp4, lr5e-4, bs128, patch8, e100), MSL(emb64, α0.5, M4, mlp4, patch12, e100), SMAP(emb128, α0.1, k0.1, M2, mlp2, patch8, e50), PSM(emb64, α0.9, k0.25, M4, mlp4, lr1e-4, bs256, patch16, e100), SWaT(emb128, α0.8, M1, mlp6, patch12, e50); 공통 seq_len 100.

**분류 (Table 10).** 정확도(소수 1자리)가 일치하는 실행: EthanolConcentration 9개, FaceDetection 2, Handwriting 2, Heartbeat 6, JapaneseVowels 15 — 설정이 제각각이라 정확도만으로는 논문 실행을 특정할 수 없다. 공개 스크립트와 설정이 같은 일치 실행은 FaceDetection(`260208`, emb512, M2, lr5e-3, α0.5, k0.15, mlp1.5, bs64, e50 → 55.28)과 JapaneseVowels(`260207`, emb768, nh16, M2, lr5e-3, α0.1, k0.15, mlp1.5, bs16, e15 → 98.92) 둘이다. EthanolConcentration·Handwriting·Heartbeat 는 공개 스크립트 설정과 정확히 같은 실행이 없다(Handwriting 은 k 0 vs 0.15, Heartbeat 는 patch 16 vs 24·epoch). 따라서 분류의 "논문 설정"은 **공개 스크립트 값을 그대로 쓰되 검증되지 않은 것**으로 둔다.

## 6. nsmt_base
`NSMT/nsmt_base/` 에 위 코드를 옮겼다 (`README.md`). `forecasting/scripts/run_ett.sh` 가 §3 의 ETT 설정으로 한 실행을 돌리고, ETT 재현 점검 결과는 `nsmt_base/forecasting/results/ett-paper-check-20261014/` 와 `docs/PROJECT_LOG.md` 에 기록한다.
