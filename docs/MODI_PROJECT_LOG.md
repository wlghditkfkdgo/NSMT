# MODI_PROJECT_LOG — NSMT 수정 연구(신피질 경로 f-LIF 교체) 기록

이 파일은 실제 Git 저장소 루트의 `docs/MODI_PROJECT_LOG.md`이다. `NSMT/docs/MODI_PROJECT_LOG.md`는 이 파일을 가리키는 링크로, 별도의 기록을 만들지 않는다.
2026-10-14 이후 **NSMT 수정 연구**(아래 §1의 아이디어)에 관한 모든 로그·실행·결과 분석은 이 파일 끝에 날짜별로 append한다. 기록 규칙은 `docs/PROJECT_LOG.md`의 운영 규칙(2026-09-10 도입, 1–8항)을 그대로 따른다: `main`에 직접 커밋하지 않고 `exp/<실험명>` 브랜치에서 작업하며, 실험마다 목적·브랜치·기준 commit·코드/설정 변경·데이터와 분할·환경·명령·seed·하이퍼파라미터·지표·산출물 위치·검증·한계·결론·commit/tag 를 적고, 하지 않은 것은 "not run"으로 적는다. 정정은 새 항목으로 append하며 과거 항목은 고치지 않는다. 항목 머리의 시각은 작성 시점의 `date` 출력이다.

이전 연구 계열(neorecall, f_lif_pop_v1–v3)과 NSMT 논문 재현(`nsmt_base`)의 기록은 `docs/PROJECT_LOG.md`에 있다. 이 파일은 그 뒤를 잇는 새 계열의 기록이다.

---

## 2026-10-14 21:45 KST — 아이디어 정리: 신피질 경로의 뉴런을 f-LIF로 교체 (실행 전, 설계·검토 기록)

**사용자 요청 (10-14):** (1) `NSMT/papers/NSMT_method_summary.md`로 연구 목적·방법·원하는 결과 방향 파악, (2) `nsmt_base`의 NSMT 구현 확인, (3) 아래 아이디어 검토, (4) 추가로 정해야 할 것 정리, (5) 이 문서 작성. 이후 모든 로그와 결과 분석을 이 문서에 기록한다.

**브랜치·기준:** `exp/nsmt-flif-neo`, 기준 commit `dd2d4c9ea` (= `exp/nsmt-base` HEAD = `origin/exp/nsmt-base`; `nsmt_base` 코드가 `main`에 없으므로 `origin/main`이 아니라 여기서 분기). 기존 로컬 변경(미추적 로그·보정 파일)은 보존했다. 이 항목은 **실행 전 설계 기록**이며 학습·평가는 not run.

### 1. 원 연구(NSMT 논문)의 목적·방법·원하는 결과 — 요약 (`papers/NSMT_method_summary.md`)

- **문제.** 시계열 모델은 고정 길이 look-back 윈도우를 섞어서 학습하므로 윈도우 밖의 장기 의존성을 못 본다. 실제 시계열의 에너지는 저주파(느린 추세)에 몰려 있으므로, 과거 전체 대신 **현재 윈도우의 저주파 성분**을 기억하면 된다. 스파이킹 뉴런은 저역 통과 필터처럼 동작하고 에너지가 싸다.
- **구조(CLS 이론).** 두 경로가 공통의 패치열 입력을 받는다. **신피질 경로**: 스파이킹 시간 축 T를 평균으로 접고 **패치 순서 축 N을 스파이킹 시간 축으로 쓴다(T = N)**. Spiking Embedding(Linear–BN–LIF) 뒤에 Spiking MLP M개. 학습 시 약한 디코더(선형 1층)가 DCT 저주파 목표(비율 k)를 재구성하도록 보조 손실 L_aux를 건다. **해마 경로**: 같은 Spiking Embedding 뒤에, 신피질 출력과 원소곱(AND)한 것을 key/value로, 해마 임베딩을 query로 하는 spiking cross-attention(Memory Replay) + Spiking MLP 2층 FFN. 잔차는 IAND `x ⊙ (1 − f(x))`. 마지막 head만 비스파이킹 선형층.
- **손실.** L = (1−α)·L_task + α·L_aux. 예측은 L1 학습, MSE/MAE 평가.
- **원하는 결과 방향.** (i) 보조 감독으로 신피질이 추세 기억을 학습하고, (ii) 그 기억을 cross-attention으로 돌려주면 장기 예측이 좋아지고, (iii) T = N이면 위치 부호화·중복 계산 없이 효율이 오르며, (iv) IAND로 에너지가 크게 준다. 예측·이상 탐지·분류에 하나의 설계로 대응.

### 2. `nsmt_base`의 NSMT 구현 확인 (`NSMT/nsmt_base/forecasting/`, = 공개 저장소 `origin/main:forecasting/`)

ETT 재현 점검은 16/16 칸이 논문 Table 13과 일치했다(PROJECT_LOG 2026-10-14 21:19). 이번 아이디어와 직접 관련된 구현 사실만 적는다(전체 대응은 `papers/NSMT_code_mapping.md`).

| 요소 | 코드 | 이번 아이디어와의 관계 |
|---|---|---|
| 신피질 경로 입력 | `ours.py: myModel.forward`: `patched_x.transpose(0, 2).mean(2, keepdim=True)` → `[N, B·C, 1, P]`. 해마 입력은 같은 패치를 T번 반복한 것(direct coding)이라 T 평균은 패치 그 자체 | T = N 정렬. N = num_patches = (96−8)/4 + 1 + 1 = **24** (seq 96, patch 8, stride 4, 끝 패딩 1) |
| 신피질 뉴런 **1** | `ours.py: Embedding`(`encoding_neo`) = `Linear(P→D)` → `BatchNorm1d(D)` → `MultiStepLIFNode(tau=2, hard reset, Sigmoid surrogate α=4, cupy)` | 교체 대상 후보 ① |
| 신피질 뉴런 **2** | `ours.py: TemporalBlock` = `SpikLinearLayer(D, D)` × M (`--time_layers`; ETT는 1 또는 2), 각각 Linear–BN–`MultiStepLIFNode` | 교체 대상 후보 ② |
| 신피질 출력 | `x_neo` `[N, B·C, 1, D]`, 각 배치마다 `functional.reset_net` → 막전위는 **윈도우 안 24 스텝**에서만 이어진다 | f-LIF의 기억 커널도 이 24 스텝 안에서만 작동한다 |
| 상호작용 | `Block.forward`: `mx = 0.05·mx + 0.95·mx.detach()`; `mx = x ⊙ mxᵀ`; `x = x ⊙ (1 − MCA(q=x, kv=mx))`; `x = x ⊙ (1 − MLP(x))` | **과제 손실에서 신피질으로 가는 기울기를 5%로 제한**한다(논문 본문에 없음). 신피질은 사실상 L_aux로 학습된다 |
| 보조 손실 | `train.py`: `loss = ce·(1−ratio) + rec·ratio`, `ratio = --alpha × pred_len/720`, `rec = MSE(Decoder(x_neo), 저주파 패치 목표)` | 사용자가 말한 "arbitrary loss"는 이 L_aux로 이해했다. H96에서는 실제 가중치가 α의 2/15(ETTh1: 0.8 → 0.107), H720에서는 α 그대로 |
| 해마만 있는 모델 | `--gating ablation`은 `_init_ablation`으로 신피질·디코더를 지우지만 `forward`가 `self.encoding_neo`를 무조건 호출하므로 **실행되지 않는다**(AttributeError). 루트 `forecasting/ours_ablation2.py`(`mymodel_ab2`)가 신피질을 빼고 `mx = 1`(cross-attention의 key/value = 해마 자신)로 둔 모델, 루트 `forecasting/Spikformer.py`는 ADD 잔차 self-attention 기준선 | Hippo-only 비교군의 정의가 필요하다(§4-7) |
| 에너지 집계 | `test.py` → `syops.get_model_complexity_info`: `nn.Linear`·`BatchNorm1d` 등 모듈 훅만 센다. 입력이 0/1이면 AC(0.9 pJ), 아니면 MAC(4.6 pJ); 뉴런 내부 연산(LIF의 누설·리셋, f-LIF의 과거 재합산)은 **세지 않는다** | f-LIF의 추가 연산을 어떻게 셀지 결정 필요(§4-15) |
| 평가 | head를 스파이킹 단계마다 적용해 `[T, B, H, C]`, 평가는 **T 평균 예측**의 MSE/MAE; 마지막 epoch(또는 조기 종료 시점) 가중치로 test | 유지 |

### 3. 사용자의 아이디어 (10-14 원문 요지)

**기존 연구의 설명.** 초기에는 신피질망의 역할을 (1) 스파이킹 뉴런의 시간 처리 능력으로 temporal 정보 구축, (2) look-back 윈도우들 사이의 공통 흐름(계절·추세) 포착, (3) 잡음이 제거된 일종의 장기 기억, 세 가지로 두었다. 계속된 증거 부족으로 이 주장들을 철회해야 하는 상황이므로, 주장을 유지하기 위해 실험을 수정한다.

**개선 방향.**
- CLS 이론에서 온 **이중 경로 구조와 상호작용은 유지**하되, 상호작용 방법과 신피질망의 구체 구조는 바꾼다.
- 해마는 transformer로 제한하지 않고 일반 SNN(CNN, Transformer, ResNet 등)으로 확장할 예정.
- 현재의 보조 손실(L_aux)은 (a) 제거, (b) 다른 손실로 변경, (c) 유지 중 선택.
- **[필수]** 신피질망의 스파이킹 뉴런을 **f-LIF**(Ge et al., ICLR 2026, spikeDE)로 교체. 선행 연구 원문의 f-LIF를 그대로 쓴다(구현: `NSMT/f_lif_pop_v3/forecasting/neurons.py: FLIFNeuron`, spikeDE `pred` solver와 144/144 일치 확인됨 — PROJECT_LOG 2026-10-14 19:31).
- **T = N은 반드시 유지**(스파이킹 축과 시계열 시간 축의 정렬이 이 연구의 가장 중요한 기여).

**검증.** forecasting ETT만, horizon 720·96, 3 seed. 통과 기준: (1) 장기 기억의 증거, (2) Hippo-only 모델 대비 "향상된 성능 + 적당한 에너지" 또는 "적당한 성능 + 크게 개선된 에너지".

### 4. 검토 — 추가로 정하거나 제시해야 할 것

아래 번호는 이후 기록에서 "결정 N"으로 참조한다. "제안"은 내 제안이며 사용자 확정 전까지 효력이 없다.

**A. 교체 범위와 뉴런 설정**

1. **교체 위치.** 신피질 경로의 뉴런은 두 곳이다: 임베딩 `encoding_neo`의 LIF ①, Spiking MLP `TemporalBlock` M층의 LIF ②. 둘 다 바꿀지 ②만 바꿀지. *제안: 둘 다*(경로 전체가 f-LIF, "신피질 = f-LIF 망"으로 설명이 단순).
2. **대조군의 뉴런.** 논문 코드의 LIF(spikingjelly `MultiStepLIFNode`)와 spikeDE의 LIF/f-LIF는 분수 차수 α 외에도 (i) 발화 판단 시점(spikeDE: 갱신 **전** 상태의 한 스텝 예측으로 발화 / spikingjelly: 갱신 후), (ii) 리셋(spikeDE: 우변 안의 −θ·s/τ, 과거까지 분수 커널로 전파 / spikingjelly: 0으로 hard reset), (iii) surrogate(arctan κ=5 / sigmoid α=4), (iv) 구현(파이썬 루프 / cupy 커널)이 다르다. 따라서 "f-LIF 덕분"을 말하려면 대조군은 **`neurons.py: LIFNeuron`(= `FLIFNeuron(alpha=1)`, 그 외 전부 동일)** 이어야 하고, 논문 코드 그대로(`nsmt_base`)는 **별도의 기준선**으로 둔다. *제안: 조건 = {nsmt_base(논문 LIF), neo-LIF(spikeDE LIF, α=1), neo-fLIF(α<1), Hippo-only} 4개.*
3. **f-LIF 하이퍼파라미터.** α(분수 차수: v3는 0.7, 논문은 과제마다 다름), τ(신피질 τ: 논문 코드 2.0; spikeDE 기본 0.5, 비전 모델 2.0), 적분 간격 h(v3 1.0, 논문 스크립트 0.1), solver(`pred`), 기억 길이(전체 N=24). "원문 그대로"가 무엇인지 — 특히 α와 h — 를 정해야 한다. *제안: τ = 2.0 유지(다른 요소를 논문 코드와 같게), h = 1, `pred`, 전체 이력, α는 {0.5, 0.7} 중 하나를 1차에 고정(둘 다 돌리면 run 수가 1.5배).*
4. **입력 전류 척도.** 논문 코드는 Linear→BN→뉴런이고 BN 출력(≈N(0,1))에 θ = 1이다. f-LIF로 바꿔도 BN을 유지하고 별도 보정(v3의 `input_scale`·`calibrate.py`)을 하지 않을지. *제안: BN 유지, 보정 없음(논문 코드 최소 변경). 단 smoke에서 발화율이 0 또는 포화면 재검토.*
5. **T = N의 함의 확인.** N = 24이고 배치마다 상태를 초기화하므로 f-LIF의 멱법칙(power-law) 기억은 **한 윈도우(96 스텝) 안**에서만 작동한다. 윈도우 밖(과거 윈도우)의 기억은 지금 구조에서는 가중치에만 있다. 이 전제를 받아들이는지, 아니면 윈도우 사이에 신피질 상태를 이어 주는 변형(큰 변경, 섞인 배치와 충돌)까지 고려하는지. *제안: 1차는 윈도우 안으로 한정.*

**B. 상호작용과 손실**

6. **보조 손실 (a)/(b)/(c).** 결정에 앞서 알아야 할 결합: `Block.forward`가 과제 기울기를 신피질에 **5%만** 전달하므로, L_aux를 제거(a)하면 신피질은 과제 손실의 5%로만 학습되어 사실상 학습되지 않는다. (a)를 고르면 0.05 상수(1.0으로 열지, 유지할지)도 함께 정해야 한다. 또한 실제 가중치는 `--alpha × pred_len/720`이라 H96과 H720에서 L_aux의 비중이 7.5배 다르다. (b)의 후보: 예측 구간(미래)의 저주파 재구성, 다음 윈도우의 저주파 예측, 윈도우 간 대비 손실. *제안: 1차 검증은 (c) 유지 + 0.05 유지 — 뉴런 교체 효과만 분리. (a)/(b)는 2차.*
7. **Hippo-only 모델의 정의.** `--gating ablation`은 실행되지 않는다(§2). 후보: (i) 루트 `ours_ablation2.py` 방식 — 신피질·디코더 제거, cross-attention의 key/value를 해마 자신으로(`mx = 1`), IAND·FFN·head 동일; (ii) `Spikformer.py`(ADD 잔차, 다른 attention). *제안: (i)를 `nsmt_base` 기준으로 다시 구현(루트 파일은 공개본과 계보가 다름).* 파라미터·에너지는 신피질 경로(Linear·BN M+1개 + 디코더)만큼 줄어든다.
8. **해마의 일반화(CNN/ResNet 등)는 1차 범위 밖**으로 두는지 확인. *제안: 1차는 transformer 해마 고정.*
9. **상호작용 방법 변경**은 "변경 예정"이라고만 되어 있다. 1차 검증에서는 논문의 Memory Replay(AND 융합 + cross-attention + IAND)를 그대로 두는지. *제안: 그대로 둔다.*

**C. 검증 설계**

10. **평가 분할.** 논문 프로토콜의 표준 ETT test(12/4/4개월)를 쓰는지. f_lif_pop_v3 계열은 ETT test를 "소진"으로 보고 미래 구간(행 14400–17419)을 사전등록해 썼다. `nsmt_base`와 논문 수치에 맞대려면 표준 test가 필요하다. *제안: 표준 test, 다만 설정(α 등)은 test를 보기 전에 고정하고 이 문서에 먼저 적는다(사전 고정). 튜닝 없음.*
11. **seed 3개.** 논문은 42. *제안: 42, 7, 2026.*
12. **데이터셋 범위.** ETT 4개(ETTh1/h2/m1/m2) × H{96, 720} × 3 seed × 조건 4 = 96 run. 재현 점검의 소요 시간(ETTh 1–4분, ETTm 6–15분)으로 보면 GPU 4장에서 반나절 이내. *제안: 4개 모두.* ETTh만이면 48 run.
13. **하이퍼파라미터.** 논문 Table 13 설정(`run_ett.sh`)을 모든 조건에 고정. *제안: 고정(튜닝 없음). f-LIF 조건에만 유리한 조정은 하지 않는다.*
14. **통과 기준 (2)의 수치화.** "향상된 성능", "적당한 에너지", "적당한 성능", "매우 개선된 에너지"에 숫자가 필요하다. 예: (가) 3-seed 평균 test MSE가 Hippo-only보다 낮고 8칸(4 데이터셋 × 2 horizon) 중 몇 칸 이상에서 같은 방향이며, 에너지(uJ, syops)가 Hippo-only의 X배 이하; (나) MSE 차이가 ±Y% 안이고 에너지가 Z% 이상 절감. X, Y, Z와 "몇 칸"을 정해야 한다. 참고: v3의 사전등록 기준은 "97.5% paired 상한 < 0 이고 평균 ≤ −0.005". *제안: 3 seed는 검정에 부족하므로 1차는 "8칸 중 ≥6칸 같은 방향 + 평균 차이"로 방향만 판정하고, 통과 시 seed를 늘려 확증.*
15. **에너지 집계 규칙.** syops는 모듈 훅만 세므로 f-LIF의 과거 재합산(스텝 t마다 뉴런당 t개의 곱셈-덧셈, N=24에서 뉴런당 최대 276 MAC)은 집계에 들어가지 않고, 기존 LIF 내부 연산도 마찬가지로 빠져 있다. (a) 기존 관례대로 제외, (b) 뉴런 내부 연산을 별도 열로 보고. *제안: (b) — 교체 조건이 유리해지는 방향의 누락은 보고해야 한다.* 뉴로모픽 구현에서 분수 커널이 어떤 비용인지(상태 저장 N개, 곱셈 N개)는 한계로 적는다.
16. **장기 기억 증거의 정의 (통과 기준 1).** 후보: (i) **윈도우 안 의존 길이** — 패치 j(또는 입력 구간)를 지우거나 섞었을 때 신피질 출력·예측의 변화량을 j까지의 거리별로 측정; f-LIF는 멱법칙으로 느리게, LIF는 지수로 빠르게 감쇠해야 한다(직접 측정 가능, 뉴런 교체 가설과 1:1 대응). (ii) **저주파 재구성** — 디코더 재구성 오차를 주파수 대역별로; (iii) **look-back 외삽** — T = N이라 head가 N에 묶여 있어 L을 바꿀 수 없다(불가). (iv) **윈도우 밖 기억** — 지금 구조에 없다(결정 5). *제안: (i)를 주 증거, (ii)를 보조로. "장기"의 기준 거리(예: 12패치 = 48스텝 이상)도 정한다.*
17. **속도.** `neurons.py`는 파이썬 루프(24스텝 × 층 2–3개)라 cupy LIF보다 느리다. 신피질 경로에만 쓰이므로 영향은 작을 것으로 보지만 smoke에서 측정한다.

**D. 운영**

18. **코드 위치.** `nsmt_base`는 논문 재현 상태로 두고, 복사본 `NSMT/nsmt_flif_v1/forecasting/`에 `--neo_neuron {sj_lif, lif, flif}`·`--hippo_only` 옵션을 넣는 방식. *제안: 복사본.* 기본값은 논문 코드와 비트 동일해야 하고, 그것을 재현 점검 한 칸(ETTh1 H96 seed 42 = 0.378303)으로 확인한다.
19. **실험명·브랜치.** `exp/nsmt-flif-neo`(이번에 생성). 다른 이름을 원하면 지금 바꾼다.

### 5. 1차 검증 설계 초안 (결정 1–19가 제안대로 확정될 때; 실행 전)

- **조건.** `base`(nsmt_base 그대로) / `neo-lif`(신피질 ①②를 `LIFNeuron`, α=1) / `neo-flif`(①②를 `FLIFNeuron`, α 고정) / `hippo-only`(결정 7의 (i)). 나머지 구조·손실·설정은 `run_ett.sh`의 논문 설정.
- **격자.** ETTh1·ETTh2·ETTm1·ETTm2 × H{96, 720} × seed{42, 7, 2026} = 조건당 24 run, 총 96 run. `scripts/gpu_pool2.sh`, GPU당 1개.
- **사전 점검(학습 전).** (p1) 복사본 기본값 = `nsmt_base` 비트 동일(ETTh1 H96 seed 42 test MSE 0.378303); (p2) `neo-lif`·`neo-flif`가 보는 입력 전류와 발화율이 smoke 1 epoch에서 0도 포화도 아님; (p3) `hippo-only` 파라미터 수 = base − 신피질 경로; (p4) 에너지 집계에 뉴런 내부 연산 열 추가 후 base의 기존 열이 바뀌지 않음.
- **지표.** test MSE/MAE(T 평균 예측), 파라미터 수, syops AC/MAC/에너지(uJ) + 뉴런 내부 연산 열, 발화율, 학습 시간. 장기 기억 증거: 결정 16의 (i) 거리별 민감도 곡선(조건별, 데이터셋별), (ii) 대역별 재구성 오차.
- **산출물.** `NSMT/nsmt_flif_v1/forecasting/{log,results}/<suite>/`, 실행별 `log/{best_log_0.csv, final+result.csv, logargs.txt}`는 Git, checkpoint·TensorBoard event·stdout은 로컬. 요약 표와 이 문서의 결과 항목.

### 6. 한계·not run

- 이 항목은 설계·검토 기록이다. 코드 작성, smoke, 학습, 평가 모두 not run.
- f-LIF 논문 PDF는 열지 않았다(사용자 지시: 필요할 때만). 뉴런의 수식·기본값은 `neurons.py`의 머리 주석과 PROJECT_LOG 2026-10-14 19:31의 spikeDE 대조 결과에 근거한다.
- `ours_ablation2.py`가 논문의 어느 ablation 표에 해당하는지는 확인하지 않았다(Hippo-only 정의는 결정 7에서 새로 고정한다).

### 7. commit

이 항목을 담은 commit을 `exp/nsmt-flif-neo`에 push한다. 해시는 다음 항목에 적는다. 실험이 아니므로 태그는 만들지 않는다.
