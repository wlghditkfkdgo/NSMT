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

---

## 2026-10-15 00:30 KST — 결정 1–19 확정, 조사 결과(보조 손실·에너지·선행 연구·기억 증거), 1차 실험 설계 고정 (실행 전)

**사용자 결정 (10-14, 이전 항목의 번호):** 1 교체 위치는 **둘 다**와 **MLP층만** 두 케이스 비교. 2 대조군 = `neurons.py: LIFNeuron`(α=1). 3 spikeDE **논문 공개 스크립트 값**(τ 2.0, `pred`, 전체 이력)에 **h = 1.0**, α 1차 0.5; α는 이후 튜닝 대상. 4 v3식 보정 없음(아래 설명). 5 기억은 윈도우 안으로 한정. 6 아래 제안대로 **1차 (c) 유지**. 7 Hippo-only = 신피질·디코더 제거, cross-attention key/value = 해마 자신(향후 해마를 임의의 SNN 백본으로 바꿀 것이므로 신피질은 plug-and-play 모듈). 8 해마 확장은 나중. 9 상호작용은 1차 그대로, 선행 연구 조사 뒤 개선. 10 **표준 ETT test(1차 판정) + v3 미래 구간(2차 확인, 재사용 허용)**. 11 seed 42, 7, 2026. 12 **ETTh2 + ETTm2**. 13 논문 설정 고정. 14 통과 기준은 내가 정함(아래). 15 f-LIF 논문의 에너지 계산 검토 후 결정(아래). 16 기억 증거 설계·검토(아래). 17–19 승인(복사본 `NSMT/nsmt_flif_v1/`, 브랜치 `exp/nsmt-flif-neo`).

**spikeDE "기본값" 확인** (로컬 clone `fcd743b`, 읽기 전용): 라이브러리 클래스 기본값은 τ 0.5·θ 1·κ 5·`method='gl'`; 논문 공개 스크립트(`examples/ICLR_Release/Neuromorphic/train_hardvs_cnn.py`)는 τ 2.0·`pred`·`time_interval 0.1`·`memory -1`(전체), α는 스크립트마다 0.3/0.8/0.9, 라이브러리 예제(`cde_mnist.py`) 기본 α 0.5, 논문 Table 6 ablation 최적 α 0.5. → 사용자 선택: τ 2.0, pred, 전체 이력, **h 1.0**(α=1 대조군이 NSMT LIF와 같은 스텝당 감쇠 0.5를 갖도록), α 0.5.

**결정 4의 질문 "v3식 보정":** `f_lif_pop_v3/forecasting/calibrate.py`(사전등록 D11) — v3 뉴런은 BN 없이 `Linear → ×input_scale → 뉴런`이라 학습 전에 train 분할만으로 격자 {1..30}에서 `input_scale` 하나를 골라 초기 발화율이 [0.1, 0.3](0.2에 가장 가까운 값, 죽거나 포화한 구성원 없음)에 들도록 한 절차. nsmt_base는 뉴런 앞에 BatchNorm이 있어 입력 전류가 ≈N(0,1)로 맞춰지므로 생략한다. smoke에서 신피질 발화율이 0 또는 포화면 재검토.

**결정 6 근거 — 보조 손실 목표는 기억을 요구하지 않는다 (읽기 전용 측정).** ETTh1 train 분할, 윈도우 3,000개 × 7채널, `ours.py`와 같은 전처리(윈도우 표준화, 끝 패딩 4, 길이 100 DCT, k = 0.25 → 25계수, 평균 제거·복원, 패치 8/stride 4, N = 24). 패치 n의 저주파 목표를 최근 ctx 패치로 선형 최소제곱 적합한 R²(n ≥ 6 합산):

| 목표 | ctx 1 (패치 n만) | ctx 2 | ctx 4 | ctx 7 |
|---|---|---|---|---|
| 저주파(패치 n) = 현재 L_aux | **0.9635** | 0.9779 | 0.9818 | 0.9833 |
| 저주파(패치 n+1) | 0.7399 | – | 0.7613 | 0.7867 |
| 저주파(패치 n+3) | 0.0656 | – | 0.1686 | 0.2692 |

패치별 평균만으로 현재 목표 분산의 69.6%가 설명된다. 즉 현재 L_aux는 현재 패치를 저역 통과시켜 되돌리면 거의 다 줄어들어 f-LIF의 긴 기억이 보상받을 자리가 없다. 또 `Block.forward`가 과제 기울기를 신피질에 5%만 전달하므로 (a) 제거는 신피질을 학습시키지 않는다(0.05도 함께 열어야 하며 그러면 "느린 추세 기억" 역할 설명이 사라짐). **결론: 1차는 (c) 유지로 뉴런 교체 효과만 분리; 2차에 (b) 목표를 `저주파(패치 n+ℓ)`(ℓ ∈ {2, 4}, 창 안, 마지막 ℓ 스텝 제외)로 바꾸는 실험을 권장**(기억을 써야 줄어드는 손실; 사용자 원래 주장 (2)·(3)과 일치). 측정 코드는 일회성 인라인 스크립트였으므로 구현 단계에서 `nsmt_flif_v1/forecasting/analysis_aux_target.py`로 남긴다.

**결정 15 — f-LIF 논문(Ge et al. ICLR 2026) 부록 D.3의 에너지 계산 (PDF 38–41쪽 확인):** "뉴런 내부 에너지에 널리 쓰이는 방법이 없다"고 밝히고, 시냅스는 Yao et al. 관례(SOP = ρ·T·FLOPs, E_AC 0.9 pJ, E_MAC 4.6 pJ, 45 nm 32-bit), 뉴런 내부는 따로 더한다(κ = E_MAC/E_AC). LIF(하드 리셋) 뉴런 1개 T 스텝: `T·[(2+2ρ)κ + (2+ρ)]·E_AC`(식 60); f-LIF: `T·[(κ+1)·log₂T + 1 + ρ(2κ+1)]·E_AC`(식 64, FFT 합성곱 가정); 총합 = 시냅스 + Σ뉴런(식 67). 논문 Spikformer 예: 뉴런 몫 LIF 8.0%, f-LIF 14.5%, 총합은 비슷(2.93 vs 2.91 mJ). **채택 규칙:** `final+result.csv`의 기존 13열은 그대로 두고 뒤에 `neuron_lif_uJ`(모든 스파이킹 층을 식 60으로), `neuron_model_fft_uJ`(f-LIF 층은 식 64), `neuron_model_direct_uJ`(우리 구현은 FFT가 아니라 직접 합: 스텝 t에 t번 MAC, T = 24에서 276 MAC/뉴런 + 식 60 항), `total_fft_uJ`, `total_direct_uJ`, `rho_neo`, `rho_hippo`, `n_spiking_layers`, `kappa`를 추가하고 층별 상세는 `log/neuron_energy.csv`. 판정은 `total_fft_uJ`(논문 방식), direct는 병기. 해마의 spikingjelly LIF도 식 60으로 센다(지금까지 0으로 세던 것). 예상 크기(ETTh1 H96, ρ ≈ 0.15): 해마 뉴런 내부 ≈ 13 µJ(시냅스 18.76 µJ와 같은 자릿수), 신피질 경로 LIF ≈ 0.2 / f-LIF FFT ≈ 0.44 / direct ≈ 1.2 µJ → 뉴런 교체로 인한 에너지 차이는 전체의 수 % 이하이고, Hippo-only 대비 차이는 주로 AND 융합이 해마 K/V 스파이크율을 낮추는 효과에서 온다.

**결정 14 — 통과 기준 (1차 = 방향 판정; 셀 = ETTh2/ETTm2 × H96/H720 = 4칸, 칸마다 3 seed, 같은 seed끼리 짝):**
- (2) Hippo-only 대비 — 경로 A(성능): 4칸 모두 f-LIF 조건의 평균 test MSE < Hippo-only, 상대 개선 평균 ≥ 2%, 12쌍 중 ≥ 10쌍 같은 방향, `total_fft_uJ` ≤ Hippo-only의 1.3배. 경로 B(에너지): 4칸 평균 상대 MSE 차이 ≤ +1%이고 `total_fft_uJ` ≤ Hippo-only의 0.7배. **추가 필수:** f-LIF 조건이 같은 위치의 LIF 대조군보다 4칸 중 ≥ 3칸에서 낮은 MSE(아니면 이득은 신피질 경로 자체의 것). base(논문 LIF)와의 차이는 보고만.
- (1) 장기 기억 — M1 의존 길이: 신피질 입력의 d패치 전 교란에 대한 신피질 스파이크 민감도 S(d)에서 S(12)/S(1) ≥ 0.10 이고 LIF 대조군 비율의 ≥ 3배; 멱법칙/지수 적합 R² 병기. M2 사용 증거: 신피질에 마지막 k패치만 보여줄 때 ΔMSE(k=12) ≥ 1%(f-LIF)이고 LIF 대조군보다 큼. **M2가 핵심**; M1만 통과면 "기억은 있으나 쓰이지 않음"으로 기록.
- 2차 확인(미래 구간): 같은 식을 미래 구간 MSE로 재계산하되 방향 일치 여부만. 3 seed로 검정은 하지 않으며 1차 통과 시 seed 8개 확증 실험을 따로 사전 고정.

**결정 16 — 기억 증거 실험 설계와 타당성:** 학습된 모델, test 윈도우 ≥ 500개, `encoding_neo` 입력(`[N, B·C, 1, P]`)만 치환하고 해마 입력은 원본 유지. M1: 마지막 스텝 n = 23(보조 n = 16)에서 d = 1..n의 패치 n−d를 (i) 0, (ii) **윈도우 평균 패치(주)**, (iii) 다른 윈도우의 같은 패치로 바꾸고 `x_neo[n]` 스파이크 변화율과 예측 변화를 기록, 멱법칙 `d^(α−1)`과 지수 `ρ^d` 적합 비교. M2: 마지막 k ∈ {1, 2, 4, 8, 12, 24}패치만 원본, 앞은 (ii)로 치환해 test MSE(k). **검토:** M1은 뉴런 설계상 당연한 결과(멱법칙 감쇠)라 학습된 기억의 증거가 아니며 M2(예측에 쓰이는가)가 있어야 "장기 기억"이라 말할 수 있다. 한계: 해마도 attention으로 24패치 전체를 보므로 "해마가 못 보는 과거"라는 주장은 윈도우 안에서는 불가(결정 5) — 말할 수 있는 것은 "신피질이 먼 과거를 느린 추세 요약으로 유지하고 그것이 예측에 쓰인다"까지. 비교는 반드시 같은 위치의 LIF 대조군과, 발화율을 함께 보고. 스파이크 이산성 때문에 윈도우 ≥ 500·채널 7 평균과 부트스트랩 구간. 제외한 대안: look-back 외삽(T = N이라 head 폭이 N에 묶임), 윈도우 간 상태 전달(구조 변경, 보류).

**결정 9 — 선행 연구 조사 (웹, 26편; 에이전트 조사, 미확인 항목은 표시):** CLS 영감 SNN은 연속 학습(CH-HNN, Nat. Commun. 2025, DOI 10.1038/s41467-025-56405-9; SESLR arXiv:2507.02901)·신경과학 시뮬레이션(D'Alba et al., Neurocomputing 2025)이며 **시계열 예측 + spiking cross-attention 검색을 결합한 선행 연구는 찾지 못함**. 가장 가까운 구조: **DMP**(Sun et al., Nat. Mach. Intell. 2026, arXiv:2512.07602) — 층마다 저차원 느린 상태(LMU)를 빠른 LIF의 추가 입력 전류로 주입; 기억 증거는 S/PS-MNIST 784스텝·상태 길이 sweep·경로 제거·스텝별 기울기. 상호작용 방식: 전류 주입(DMP), 수지상/정단부 곱셈 변조(Ferrand et al. ICANN 2023; TC-LIF AAAI 2024 arXiv:2308.13250; TS-LIF ICLR 2025 arXiv:2503.05108; DH-SNN Nat. Commun. 2024), spiking cross-attention은 멀티모달 융합에서만(S-CMRL arXiv:2502.12488, `x + α·CA`; SMM Transformer ICML 2026), 시계열 이중 경로 융합은 SpikeSTAG(arXiv:2508.02069)의 실수 sigmoid 게이트(단일 0.823/0.854 < concat 0.858 < gate 0.868 R²). 장기 기억 표준 증거: 지연 XOR/store-recall(DH-SNN, TS-LIF 10–50스텝, LSNN NeurIPS 2018), 784스텝 S/PS-MNIST·LRA(PSN NeurIPS 2023, SpikingSSMs AAAI 2025, P-SpikeSSM ICLR 2025, LMUFormer ICLR 2024), 스텝별 기울기(TC-LIF, DMP), 경로·기억 제거와 길이 sweep(DMP, TS-LIF), 학습된 시간상수 분석(DH-SNN), 절단·결측 강건성(f-SNN, TS-LIF). **f-SNN 자체는 이론 + 강건성만 제시하고 의존 길이 곡선이 없다.** 후속 LongSpike(arXiv:2606.12895)는 멱법칙 커널을 지수 2개 합으로 근사. 시계열 SNN 에너지 관례: SeqSNN(ICML 2024 arXiv:2402.01533)·TS-LIF 등은 SOP·0.9/4.6 pJ, 뉴런 내부는 무시; 예외 f-SNN(뉴런 항), SpikeLite(arXiv:2609.35097, 막전위 갱신을 AC로 환산), SpikySpace(arXiv:2601.02411)·DMP(22 nm 실측); 비판 Yan et al. TCAD(arXiv:2409.08290), STEP(NeurIPS 2025 arXiv:2505.11151). 미확인: SpikF(ICML 2025)·SpikF-GO의 에너지 방법, CH-HNN 상호작용 세부, PSN·SE-adLIF 기억 프로토콜, SESLR 저자·출처. **2차 상호작용 후보(우선순위):** ① DMP식 전류 주입(`u ← u + W_m·x_neo`; plug-and-play에 가장 맞음) ② 정단부식 곱셈 이득 `(1 + g(x_neo))` ③ 잔차형 cross-attention(IAND 이점 상실) ④ 실수 게이트(스파이크 순수성 상실). 해마 백본 후보: Spike-TCN/Spike-RNN/iSpikformer(SeqSNN), TS-former, SpikF.

**1차 실험 설계 (고정):** 조건 6개 — `base`(nsmt_base와 비트 동일), `neo_lif_both`, `neo_lif_mlp`, `neo_flif_both`(α 0.5), `neo_flif_mlp`(α 0.5), `hippo_only` — × ETTh2/ETTm2 × H96/H720 × seed 42/7/2026 = **72 run**. 하이퍼파라미터는 `run_ett.sh`의 논문 Table 13 값. 사전 점검: p1 기본값 = nsmt_base(ETTh1 H96 seed 42 → 0.378303), p2 smoke에서 신피질 발화율 ∉ {0, 포화}, p3 hippo_only 파라미터 = base − 신피질 경로, p4 새 에너지 열 추가 후 기존 13열 불변. 구현 순서·점검(c1–c5)·평가기(`ett_future.py`, 등록부가 같은 checkpoint의 두 번째 평가 거부)·기억 증거(`neo_memory_probe.py`)는 승인된 계획(세션 plan 파일)대로이며 다음 항목에 구현 결과를 적는다.

**commit:** 이전 항목의 commit `5f79df6dd`(exp/nsmt-flif-neo). 이 항목은 구현 commit과 함께 push한다.

---

## 2026-10-15 00:52 KST — `nsmt_flif_v1` 구현, 점검 23/23, smoke 6/6, 기본값 재현(p1) 0.378303, 평가기 시험 — 72 run 큐 시작 (tmux)

**구분:** 구현·점검 기록. 본 학습은 이 항목 끝에서 **시작**하며 결과는 다음 항목에 적는다. 브랜치 `exp/nsmt-flif-neo`, 이전 commit `5f79df6dd`.

**코드 (`NSMT/nsmt_flif_v1/forecasting/`, nsmt_base 복사본; 변경 목록은 `nsmt_flif_v1/README.md`):** `neurons.py`(v3 파일 그대로, import 한 줄만 인라인), `layers.py`(`SpikLinearLayer(neuron=)`, `NeoNeuron` 어댑터, `build_neo_neuron`), `ours.py`(임베딩·MLP층 뉴런 교체, `Block.forward(x, mx=None)` → mx 없으면 key/value = 해마 자신, `hippo_only`), `config.py`(`--neo_neuron --neo_where --neo_tau --neo_alpha --fde_method --fde_step --fde_memory --hippo_only`; 기본값이 아닐 때만 디렉터리 접미사), `model.py`, `energy.py`+`test.py`(뉴런 내부 에너지 9열을 기존 13열 뒤에 추가, 층별 `log/neuron_energy.csv`; syops에 새 뉴런 클래스를 LIF 훅으로 등록), `check_flif.py`, `ett_future.py`, `neo_memory_probe.py`, `analysis_aux_target.py`, `scripts/{run_ett.sh,smoke.sh}`, `scripts/queues/flif-v1-main.txt`(72줄). 환경 `snn_recall`(torch 1.12.0+cu113), RTX A6000.

**점검 (`results/checks/check_flif.txt`, 23/23 통과, GPU 0):** c1 기본 모델 = nsmt_base — 초기 가중치·학습/평가 모드 출력·한 번의 역전파 기울기·저장된 ETTh1 H96 checkpoint의 출력이 모두 **비트 동일**(nsmt_base 코드를 별도 프로세스로 돌려 대조), 파라미터 84,416. c2 `lif` = `flif(α=1)`: 가중치·출력 비트 동일(평가 모드 BN과 학습 모드 BN 모두), 기울기 최대 차이 2.8e-9(Euler 누적과 분수 solver의 합산 순서 차이), 배치 위치 both 3개/mlp 2개. c3 `hippo_only`: `encoding_neo/time_block/weak_decoder` 없음, 파라미터 81,664 = 84,416 − 2,752, 두 모드 모두 4차원 예측만 반환. c4 recorder의 발화율 = syops가 다음 Linear 입력에서 읽는 발화율(최대 차이 3.1e-6; syops의 LIF 노드 자체 수치는 첫 스텝만 읽으므로 비교 대상이 아님). c5 nsmt_base checkpoint로 `test()` 재실행 → 앞 13열이 원본과 동일(0.378303, 0.386383, 18.76 uJ, …), 새 열: neuron_lif 13.1408 µJ, total_fft 31.8992 µJ, rho_neo 0.2317, rho_hippo 0.1507, 스파이킹 층 11개.
- 부수 관찰: 초기화 직후 평가 모드(BN 이동 통계)에서는 신피질이 전혀 발화하지 않는다(전류 ≈ 0). 학습 모드 BN에서는 발화하며, spikeDE LIF와 논문 LIF는 신피질 스파이크의 2.4%(both)/1.3%(mlp)가 다르다(발화율 0.028 vs 0.034) — 리셋·surrogate 차이. 결정 2(대조군 = spikeDE LIF)의 근거.
- **해마 뉴런 내부 에너지(식 60)가 ≈ 13 µJ로 시냅스 에너지(18.8 µJ)와 같은 자릿수**. 신피질 뉴런 교체로 바뀌는 몫은 전체의 1–3%.

**smoke (`results/smoke-flif-20261015/raw.txt`, ETTh1 H96 seed 42, 1 epoch, GPU 1; 실험 아님):**

| cond | rc | test MSE(1 epoch) | rho_neo | total_fft / total_direct (µJ) | 시간(s) |
|---|---|---|---|---|---|
| base | 0 | 0.393413 | 0.0636 | 28.45 / 28.45 | 21 |
| neo_lif_both | 0 | 0.391952 | 0.0791 | 28.48 / 28.48 | 24 |
| neo_lif_mlp | 0 | 0.392576 | 0.0717 | 28.47 / 28.47 | 24 |
| neo_flif_both (α 0.5) | 0 | 0.391284 | 0.0807 | 28.74 / 29.51 | 32 |
| neo_flif_mlp (α 0.5) | 0 | 0.393043 | 0.0739 | 28.67 / 29.19 | 28 |
| hippo_only | 0 | 0.392141 | – | 28.50 / 28.50 | 21 |

여섯 조건이 학습·저장·평가·에너지 열까지 지난다(p2: 신피질 발화율 0.06–0.08, 죽지도 포화하지도 않음). f-LIF는 base보다 1.3–1.5배 느리다(파이썬 루프). 1 epoch의 MSE는 성능 근거가 아니다.

**p1 기본값 재현 (`results/p1-base-20261015/raw.txt`, GPU 2, 전체 학습):** `COND=base ETTh1 96 42` → **mse 0.378303, mae 0.386383**, 27 epoch(조기 종료), syn 18.76 µJ, rho_neo 0.231671 — nsmt_base·논문과 동일. 통과.

**평가기 시험 (p1 checkpoint, `results/checks/future_dryrun/`):** `ett_future.py --split test` → 0.378303/0.386383(2,785창; `final+result.csv` 재현). `--split future`(ETTh1 목표 행 [14400, 17420), 평가 행 [14304, 17420), 척도 행 [0, 8640), 2,925창) → **MSE 0.525327, MAE 0.484230**(base, seed 42; 72 run 밖의 참고값 — 이 구간은 v3 2O가 2026-09-27에 연 구간을 결정 10에 따라 재사용한 것). 같은 checkpoint의 두 번째 호출은 등록부가 거부(REFUSED). 기록 JSON에 `target_rows`와 `evaluated_rows`가 다른 필드로 남는다. 시험용 등록부는 `results/checks/future_dryrun/registry/`(`--registry_dir`)이며 본 등록부 `results/split_openings/`는 아직 비어 있다.

**기억 증거 측정기 시험 (`results/checks/probe_smoke/`, smoke 1-epoch checkpoint 2개, 64창, 윈도우 평균 치환, GPU 1, 10 s):** 동작 확인만. neo_flif_both: S(1) 0.0917, S(12) 0.0030, keep-last-12의 ΔMSE −0.0116; base: S(1) 0.0413, S(12) 7e-5. 1 epoch 모델이라 수치는 의미 없고(앞 패치를 지우면 오히려 MSE가 줄어듦), 곡선·부트스트랩 구간·CSV 형식이 나오는 것만 확인했다.

**보조 손실 측정 재현 (`analysis_aux_target.py`, `results/checks/aux_target_ETT{h1,h2}.txt`):** ETTh1은 이전 항목의 인라인 수치와 동일(현재 목표 ctx 1에서 R² 0.9635; n+3 목표 0.0656 → ctx 7에서 0.2692). ETTh2도 같은 모양(0.9576; n+3: 0.0614 → 0.2932; 패치 평균만으로 72.4%).

**실수 기록:** README를 쓰는 중 작업 디렉터리가 바뀌어 저장소 루트의 논문 README.md를 덮어썼다. `git checkout -- README.md`로 즉시 복구했고(diff 없음) 손실은 없다. 같은 이유로 `analysis_aux_target.py`가 한 번 잘못된 위치에 생성되어 옮겼다.

**본 학습 시작:** 큐 `nsmt_flif_v1/forecasting/scripts/queues/flif-v1-main.txt` 72줄(6 조건 × ETTh2/ETTm2 × H96/H720 × seed 42/7/2026, `SUITE=flif-v1-main`), `scripts/gpu_pool2.sh … 0,1,2,3`(GPU당 1개)을 **tmux 세션 `flif_main`** 에서 실행해 세션이 끊겨도 이어지게 한다. 결과 `results/flif-v1-main/raw.txt`, pool 로그 `log/flif-v1-main.pool.log`. 소요 예상 2–3시간. 결과·판정(§C 기준)은 다음 항목.

**commit:** 이 항목과 구현을 담은 commit을 `exp/nsmt-flif-neo`에 push한다(해시는 다음 항목에). 큐의 `sha=`는 그 commit을 가리킨다.

---

## 2026-10-15 02:40 KST — 1차 실험 결과 (72 run, ETTh2·ETTm2 × H96/H720 × 3 seed): **기준 (2) 불통과, 기준 (1) 불통과** — 신피질 뉴런을 f-LIF로 바꿔도 성능·에너지·기억 증거에 차이가 없음 — 태그 `exp/nsmt-flif-neo-20261015`

**구분:** 사전 고정한 1차 검증(방향 판정, 3 seed)의 **완료 기록**. 브랜치 `exp/nsmt-flif-neo`, 코드 commit `cd82f0550`(큐의 `sha=cd82f0550`). 이 항목이 속한 commit은 태그 `exp/nsmt-flif-neo-20261015`로 식별한다.

**실행:** tmux 세션 `flif_main`에서 `bash scripts/gpu_pool2.sh nsmt_flif_v1/forecasting/scripts/queues/flif-v1-main.txt 0,1,2,3`(GPU당 1개, RTX A6000 4장, 환경 `snn_recall`), 2026-10-15 00:52–02:26 KST, 72/72 완료·실패 0. 하이퍼파라미터는 `run_ett.sh`의 논문 Table 13 값(ETTh2: patch 8, k 0.3, D 64, heads 8, α 0.8, mlp 0.5, M 1, lr 5e-4, bs 64, 100 epoch, patience 2; ETTm2: k 0.25, heads 16, α 0.5, mlp 1, M 2, patience 2, 50 epoch), seq 96, seed 42/7/2026. f-LIF: α 0.5, τ 2.0, pred, h 1.0, 전체 이력. 후처리: 미래 구간 평가 72건(`ett_future.py`, 등록부 `results/split_openings/{ETTh2,ETTm2}-future-H{96,720}.json`에 모델마다 1회 기록)과 기억 측정 60건(`neo_memory_probe.py`, 512창, 윈도우 평균 치환; hippo_only 제외)을 tmux 세션 `flif_post`에서 02:27–02:38. 집계 `summarize_flif.py` → `results/flif-v1-main/summary.md`.

**산출물:** `nsmt_flif_v1/forecasting/results/flif-v1-main/{raw.txt,summary.md,future/{future.csv,*.json},memory_probe/*_probe.csv}`, 실행별 `log/flif-v1-main/<DS>/261015/<run>/log/{best_log_0.csv,final+result.csv,logargs.txt,neuron_energy.csv,h*_curve.csv,model+info+per+layer.txt}`(Git), checkpoint·TensorBoard·stdout·pool 로그(로컬).

### 표준 ETT test — MSE (3 seed 평균 ± 표준편차)

| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base (논문 LIF) | 0.283453 ±0.0030 | 0.409741 ±0.0007 | 0.175030 ±0.0003 | 0.400208 ±0.0009 |
| neo_lif_both | 0.285758 ±0.0029 | 0.411570 ±0.0030 | 0.175394 ±0.0003 | 0.400197 ±0.0004 |
| neo_lif_mlp | 0.282974 ±0.0008 | 0.412465 ±0.0025 | 0.175349 ±0.0004 | 0.400342 ±0.0009 |
| neo_flif_both | 0.282259 ±0.0009 | 0.409369 ±0.0020 | 0.176410 ±0.0008 | 0.400929 ±0.0002 |
| neo_flif_mlp | 0.285547 ±0.0026 | 0.411662 ±0.0022 | 0.175365 ±0.0003 | 0.400378 ±0.0012 |
| hippo_only | 0.283697 ±0.0027 | 0.414134 ±0.0033 | 0.175181 ±0.0004 | 0.400131 ±0.0007 |

(MAE·발화율·학습 시간은 `summary.md`. base seed 42는 논문 재현값과 같다: ETTh2 0.287624/0.408831, ETTm2 0.175206/0.398896.)

### 에너지 (µJ/창; 시냅스 syops + 뉴런 내부, 논문 방식 식 60/64 = `total_fft`; 괄호는 직접 합산 방식)

| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 56.00 | 66.89 | 63.79 | 71.62 |
| neo_lif_both | 56.74 | 66.34 | 63.75 | 71.66 |
| neo_flif_both | 56.74 (57.77) | 67.17 (68.20) | 63.78 (65.34) | 72.16 (73.71) |
| neo_flif_mlp | 57.12 (57.63) | 66.06 (66.58) | 63.88 (64.91) | 71.99 (73.03) |
| hippo_only | 58.53 | 68.12 | 65.53 | 72.66 |

시냅스만: base 33.2/44.1/38.2/46.8, hippo_only 35.8/45.7/40.5/48.3. 이중 경로 조건이 hippo_only보다 2–3% 적은 것은 AND 융합이 해마 key/value의 스파이크를 줄이기 때문이고, 신피질 뉴런 교체에 따른 차이는 1% 미만이다(직접 합산 방식으로도 f-LIF 추가분은 1–2%).

### 기준 (2) 판정 — 표준 test, Hippo-only 대비 (같은 seed끼리 짝)

| 비교 | 칸별 상대 MSE 차이 (ETTh2 H96 / H720 / ETTm2 H96 / H720) | 평균 | 12쌍 중 낮음 | 에너지 비 | 경로 A | 경로 B |
|---|---|---|---|---|---|---|
| neo_flif_both vs hippo_only | −0.51% / −1.15% / +0.70% / +0.20% | −0.19% | 6 | 0.980 | 불통과 | 불통과 |
| neo_flif_mlp vs hippo_only | +0.65% / −0.60% / +0.10% / +0.06% | +0.06% | 7 | 0.978 | 불통과 | 불통과 |
| neo_flif_both vs neo_lif_both (추가 필수) | −1.22% / −0.53% / +0.58% / +0.18% → 4칸 중 2칸 | | | | 불통과(≥3 필요) | |
| neo_flif_mlp vs neo_lif_mlp | +0.91% / −0.19% / +0.01% / +0.01% → 1칸 | | | | 불통과 | |

보고만: base 대비 neo_lif_both +0.37%, neo_lif_mlp +0.18%, neo_flif_both +0.11%, neo_flif_mlp +0.36%, **hippo_only +0.31%**(12쌍 중 5쌍 낮음). **→ (2) 불통과.** 여섯 조건이 네 칸 모두에서 ±1.2% 안에 있고, seed 간 표준편차(0.0002–0.003)와 같은 크기다.

### 2차 확인 — 미래 구간 (ETTh2 행 [14400, 17420), ETTm2 행 [57600, 69680), 문맥 96, 창 2,925/2,301/11,985/11,361; 방향만)

| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 0.172566 | 0.268724 | 0.125270 | 0.225897 |
| neo_lif_both | 0.171966 | 0.268958 | 0.125203 | 0.226077 |
| neo_lif_mlp | 0.172042 | 0.266354 | 0.125036 | 0.225819 |
| neo_flif_both | 0.172792 | 0.263927 | 0.125836 | 0.226238 |
| neo_flif_mlp | 0.172233 | 0.268027 | 0.124559 | 0.226196 |
| hippo_only | 0.172501 | 0.268647 | 0.124545 | 0.225375 |

neo_flif_both vs hippo_only: +0.17% / −1.76% / +1.04% / +0.38%(평균 −0.04%, 12쌍 중 4쌍 낮음); vs neo_lif_both 4칸 중 1칸. neo_flif_mlp vs hippo_only 평균 −0.00%, 4쌍. hippo_only vs base −0.22%(8/12 낮음). **방향도 표준 test와 같다: 차이 없음.** 미래 구간의 MSE 수준이 표준 test보다 낮은 것(ETTh2 H96 0.17 vs 0.28)은 구간의 성질이며 모델 비교와 무관하다.

### 기준 (1) 판정 — 기억 측정 (`neo_memory_probe.py`, 데이터셋별 3 seed × 2 horizon = 6 run 평균, 윈도우 512개)

| data | cond | S(1) | S(12)/S(1) | ΔMSE(마지막 12패치만)/MSE |
|---|---|---|---|---|
| ETTh2 | base | 0.0733 | 0.020 | +1.98% |
| ETTh2 | neo_lif_both / mlp | 0.0832 / 0.0707 | 0.016 / 0.014 | +1.85% / +2.45% |
| ETTh2 | neo_flif_both / mlp | 0.0844 / 0.0762 | **0.048** / 0.024 | **+3.05%** / +2.35% |
| ETTm2 | base | 0.1133 | 0.110 | +6.65% |
| ETTm2 | neo_lif_both / mlp | 0.1226 / 0.1120 | 0.091 / 0.089 | +4.80% / +5.59% |
| ETTm2 | neo_flif_both / mlp | 0.1352 / 0.1284 | 0.134 / 0.095 | +3.24% / +4.87% |

S(d) = 신피질 입력의 d패치 전 패치를 윈도우 평균으로 바꿨을 때 마지막 스텝 신피질 스파이크의 변화 비율. 판정: M1(S(12)/S(1) ≥ 0.10 이고 대조군의 ≥ 3배) — ETTh2 f-LIF 0.048(대조군의 3.0배이지만 0.10 미만), ETTm2 0.134(≥ 0.10이지만 대조군 0.091의 1.5배) → **네 칸 모두 불통과**. M2(ΔMSE(12) ≥ 1% 이고 대조군보다 큼) — ETTh2 neo_flif_both만 통과(+3.05% vs +1.85%), 나머지 세 칸은 LIF 대조군이 오히려 크거나 같음 → **불통과**. **→ (1) 불통과.**

### 해석 (말할 수 있는 것 / 없는 것)
1. **신피질 경로 자체가 이 설정에서 거의 기여하지 않는다.** hippo_only가 base와 ±0.3%, 어느 뉴런으로 바꿔도 같다. 결정 6의 측정(현재 보조 손실 목표는 현재 패치만으로 96% 맞춰짐)과 일관된다: 신피질이 기억을 쓸 이유가 없는 손실로 학습된다.
2. **f-LIF의 느린 기억은 신피질 출력에는 남는다**(ETTh2에서 S(12)/S(1)이 LIF의 3배) — 그러나 그것이 예측에 쓰이는 증거(M2)는 ETTh2 both 한 칸뿐이고, ETTm2에서는 LIF 대조군이 오래된 패치에 더 의존한다. ETTm2의 base·LIF도 S(12)/S(1) ≈ 0.09–0.11로 큰데, τ = 2 LIF의 막전위 기억(스텝당 0.5 감쇠)으로는 설명되지 않는다. 스파이크 타이밍이 한 번 바뀌면 리셋을 통해 이후가 계속 달라지는 **발산 민감도**로 보이며, 따라서 S(d)는 "기억"과 "민감도"를 구분하지 못한다(§F 한계에 추가). M2(마지막 k패치만 유지했을 때의 오차 증가)가 더 믿을 만한 측정이고, 그 측정에서 모든 모델이 +2–7% 악화되는 것은 신피질이 오래된 패치의 정보를 어느 정도 쓰고 있음을 뜻하지만 뉴런 종류와는 무관하다.
3. 에너지: 뉴런 교체는 전체의 1–2%를 바꿀 뿐이다. 이중 경로의 2–3% 절감은 AND 융합의 부수 효과다.
4. 말할 수 없는 것: "f-LIF가 쓸모없다" — 1차는 논문 설정 그대로(α 0.5 고정, 기억을 요구하지 않는 보조 손실, 5% 기울기 제한)에서의 결과다. 3 seed·2 데이터셋의 방향 판정이며 검정이 아니다.

### 한계 · not run
- α는 0.5 하나(결정 3; 튜닝은 2차). h 0.1, GL solver, 짧은 기억 창은 not run. 해마 백본 확장·상호작용 변경 not run.
- 기억 측정의 치환은 윈도우 평균 하나(0 치환·다른 창 치환·해마 입력 대조는 not run). 확증(seed 8)은 1차 불통과이므로 하지 않는다.
- 미래 구간은 v3가 2026-09-27에 연 구간을 재사용했다(결정 10). 등록부에 모델별 1회로 기록되어 있다.
- 확인된 것: p1(기본값 재현 0.378303)과 base seed 42의 ETTh2/ETTm2 값이 논문 재현 점검과 같아 코드 경로는 nsmt_base와 동일하다.

### 결론과 다음
1차 가설("신피질 뉴런을 f-LIF로 바꾸면 장기 기억 증거와 Hippo-only 대비 이점이 생긴다")은 **이 설정에서 기각**. 원인 후보는 뉴런이 아니라 **신피질이 학습되는 목표**다. 다음 단계 제안(사용자 결정 필요): (A) 결정 6의 (b) — 보조 손실 목표를 `저주파(패치 n+ℓ)`(ℓ ∈ {2, 4})로 바꿔 신피질이 추세를 앞으로 외삽하도록 하고 같은 6조건을 다시 돌린다(코드 변경은 `ours.py`의 목표 정렬과 `train.py`의 마스크뿐); (B) 5% 기울기 제한을 열어(1.0) 과제 손실이 신피질에 직접 닿게 한 조건 추가; (C) α sweep {0.3, 0.7, 0.9}은 (A)/(B)에서 신피질이 기여하기 시작한 뒤에. 상호작용 변경(§E 후보 ① 전류 주입)은 그 뒤.

**commit·태그:** 이 항목·결과·스크립트(`summarize_flif.py`, `make_post_queues.py`)를 담은 commit을 `exp/nsmt-flif-neo`에 push하고 annotated tag `exp/nsmt-flif-neo-20261015`(완료된 1차 검증)를 붙인다.

---

## 2026-10-15 10:21 KST — 2차 탐색 사전 고정: (A) 앞을 보는 보조 목표 × (B) 신피질 기울기 개방, 120 run (실행 전 기록 → 이 항목의 commit 뒤 tmux에서 시작)

**사용자 결정 (10-15):** "(A)와 (B) 동시 탐색 시작". 브랜치 `exp/nsmt-flif-neo`(계속), 이전 commit `7b7180939`(태그 `exp/nsmt-flif-neo-20261015`).

**목적.** 1차에서 신피질 경로가 기여하지 않은 원인 후보 두 가지를 동시에 바꿔, (i) 신피질이 기여하기 시작하는지, (ii) 그때 f-LIF가 LIF보다 나은지 본다. **탐색**이다(조건 묶음 5개를 한 번에 보므로 어느 하나의 통과도 확증이 아니다).

**바꾼 것 (코드, `nsmt_flif_v1/forecasting/`):**
- (A) `--aux_lead ℓ`: 신피질의 스텝 n 출력이 디코더를 거쳐 **패치 n+ℓ의 저주파 목표**를 재구성한다(`ours.py: myModel._training`에서 `rec_x[:-ℓ]`와 `org_x[ℓ:]`를 짝지음; 마지막 ℓ 스텝은 윈도우 안에 목표가 없어 손실에서 빠짐). 목표 계산(DCT, k, 패치)과 디코더(선형 1층)는 그대로. ℓ = 0이 논문.
- (B) `--neo_grad g`: `Block.forward`의 `mx = g·mx + (1−g)·mx.detach()` — memory replay를 통해 신피질로 가는 과제 기울기의 비율. 논문 코드 0.05, (B)는 1.0. 순전파 값은 g와 무관하다.
- 실행기: `LEAD`, `NG` 환경변수 → `--aux_lead --neo_grad`; 조건 묶음 이름 `L<ℓ>g<g>`를 실행 폴더 이름과 RESULT 줄(`regime=`)에 넣어 같은 조건·seed의 다른 묶음 폴더를 잘못 집지 않게 함. `ett_future.cond_name`·기억 측정 파일 이름에 `@L<ℓ>g<g>` 접미사(기본값이면 없음). `summarize_flif.py`가 여러 suite를 합치고 아래 판정을 계산(`--stage2`). 실행기 `scripts/stage2_driver.sh`(학습 → 기억 측정 → 집계).

**점검 (`results/checks/check_flif.txt`, 29/29):** 기존 c1–c5 그대로 통과(기본값 = nsmt_base 비트 동일). **c6** ℓ = 2: 초기 가중치 동일, 예측·신피질 출력 비트 동일, 목표 = ℓ=0 목표의 [2:], 재구성 = ℓ=0 재구성의 [:−2](모양 [64, 7, 1, 22, 8] vs [.., 24, ..]), sj_lif·flif 모두. **c7** g = 1.0 vs 0.05(과제 손실만 역전파): 예측 비트 동일, 해마 파라미터 22개의 기울기 비트 동일, 신피질 파라미터 9개의 기울기가 정확히 20배(최대 상대 오차 3.8e-5), sj_lif·flif 모두.

**smoke (`results/smoke-s2-20261015/raw.txt`, ETTh1 H96 seed 42, 1 epoch, GPU당 2개):** 7/7 정상 종료, 실행 폴더·`logargs.txt`의 `aux_lead/neo_grad`가 조건과 일치, 신피질 발화율 0.056–0.079. ℓ=4의 1 epoch 학습 재구성 오차 0.796(현재 목표보다 어려움). 1 epoch MSE는 근거 아님.

**Hippo-only 재사용 확인 (`results/s2-hippo-reuse-check/raw.txt`):** 새 코드로 `hippo_only ETTh2 96 42` 전체 학습 → mse 0.287275, mae 0.334102 = 1차 값과 **동일**. hippo_only는 (A)(B)와 무관한 경로이므로 1차의 12개를 재사용한다. 1차의 L0g0.05 lif/flif(both) 24개도 그대로 L0g0.05 묶음으로 쓴다.

**설계 (사전 고정):**
- 조건 묶음 5개(1차 L0g0.05에 더해): **L0g1.0**(B만), **L2g0.05**, **L4g0.05**(A만), **L2g1.0**, **L4g1.0**(A+B). 각 묶음 × {`neo_lif_both`, `neo_flif_both`} × ETTh2/ETTm2 × H96/H720 × seed 42/7/2026 = **120 run**. f-LIF는 1차와 같음(α 0.5, τ 2, pred, h 1, 전체 이력). 그 외 하이퍼파라미터는 논문 Table 13 값 그대로.
- **배치는 "둘 다"만** 쓴다(판단): 1차에서 둘 다/MLP만의 차이가 없었고(결정 1의 비교 결과), "신피질 전체가 f-LIF"인 쪽이 설명이 단순하며 1차의 유일한 M2 통과도 이쪽이었다. MLP만 배치는 필요하면 추가.
- base(논문 LIF)는 새 묶음에서 돌리지 않는다(질문에 필요한 대조군은 같은 묶음의 spikeDE LIF와 hippo_only).
- **미래 구간은 2차 탐색에 쓰지 않는다**(판단): 후보 선택에 영향을 주지 않도록 확증 단계까지 남겨 둔다. 기억 측정(512창, 윈도우 평균 치환)은 120개 모두.
- 실행: tmux 세션 `flif_s2`, `nsmt_flif_v1/forecasting/scripts/stage2_driver.sh`(내부: `scripts/gpu_pool2.sh …/flif-v1-s2.txt 0,1,2,3,0,1,2,3` → GPU당 2개; 긴 칸(ETTm2) 먼저). 환경 `snn_recall`, RTX A6000 × 4. GPU 공유 시 run당 시간이 늘 수 있으며 `time_s`로 기록된다(결과는 결정적이라 공유와 무관).

**판정 (사전 고정; 묶음 R마다, 같은 seed끼리 짝, 셀 = 데이터 × H 4칸):**
- **Q-A 신피질이 기여하는가:** `neo_lif_both@R`, `neo_flif_both@R` 각각 vs hippo_only — 평균 상대 MSE ≤ −1% 이고 12쌍 중 ≥ 8쌍 낮음.
- **Q-B f-LIF가 나은가:** `neo_flif_both@R` vs `neo_lif_both@R` — 4칸 중 ≥ 3칸 낮고 평균 ≤ −0.5%.
- 1차 기준 (2)(경로 A/B + LIF 대조군 조건)와 (1)(M1, M2)을 `neo_flif_both@R`에 그대로 적용(대조군 = 같은 묶음의 LIF).
- **확증 후보:** (2)와 M2를 모두 통과한 묶음; 여럿이면 평가된 가중치(마지막 epoch)의 평균 검증 MSE가 가장 낮은 것. 후보가 있으면 seed 8개 + 미래 구간으로 확증 실험을 따로 사전 고정한다. 후보가 없으면 그 사실을 기록하고 다음 방향(α, 상호작용 변경)을 제안한다.
- 요인 해석(보고): (A) 효과 = L2/L4 vs L0 (g 0.05에서), (B) 효과 = g 1.0 vs 0.05 (ℓ 0에서), 상호작용 = A+B 묶음.

**commit:** 이 항목과 코드 변경을 담은 commit을 push한 뒤 큐를 시작한다(큐의 `sha=`는 그 commit). 결과는 다음 항목.

---

## 2026-10-15 10:50 KST — 2차 탐색(patience 2) 중단 → 조기 종료 patience 10 + 최고 검증 가중치 test로 재설계, 168 run 사전 고정 / 정정 2건

**사용자 결정 (10-15, 학습 방식 설명 뒤):** (1) 조기 종료의 patience를 늘린다. (2) 학습률 조절은 나중에 고민한다. (3) 질문: "최고 검증 가중치로 test하려면 별도의 validation set이 필요하지 않은가?"

**학습 방식 확인 (1차 72 run의 stdout·`best_log_0.csv`; 사용자에게 보고한 내용):** AdamW(lr 5e-4, weight decay 0.06), batch 64, 기울기 자르기 없음, warmup 없음. 학습률 스케줄은 `ReduceLROnPlateau`(검증 전체 손실 기준, 개선 없는 epoch 2번까지 참고 3번째에 ×0.1, 감소 뒤 1 epoch 쉼). 조기 종료는 검증 예측 오차(T단계 L1) 기준 patience 2(ETTh2/ETTm2). **72 run 모두 조기 종료(실제 epoch: ETTh2 H96 4/6/15, H720 3/5/7, ETTm2 H96 5/7/10, H720 4/4/6 — 최소/중앙값/최대), 최대 epoch(100/50)에 닿은 run 0, 학습률이 줄어든 run 0/72**(조기 종료가 항상 먼저 걸림). test는 마지막 epoch 가중치(`best+model.pt`라는 이름이지만 마지막 상태)로 하며, 그 검증 오차는 최고 검증 epoch보다 평균 0.2–0.9% 나쁘다. ETTm2 H720은 최고 검증 epoch 중앙값이 1. 즉 논문 설정은 사실상 고정 학습률로 3–15 epoch만 학습한다. 해석상 함의: 해마는 몇 epoch 만에 배우지만 천천히 배우는 신피질(보조 손실 + 5% 기울기)에는 불리할 수 있고, 2차의 (A)는 더 어려운 목표(ℓ=4 smoke 재구성 오차 0.80)인데 조기 종료는 예측 오차만 보므로 효과가 가려질 수 있다.

**질문 (3)에 대한 답:** 새 validation set은 필요 없다. 표준 ETT 분할(`data_provider/data_loader.py`)은 이미 학습 12개월 / 검증 4개월 / test 4개월이다. ETTh 목표 행 기준 학습 [0, 8640), 검증 [8640, 11520), test [11520, 14400)이고, ETTm은 각 4배다. 검증 창의 입력 문맥(앞 96행)만 학습 구간 끝에 걸치고 목표 행은 겹치지 않는다. 조기 종료가 이미 이 검증 구간으로 "언제 멈출지"를 고르고 있으므로, 같은 구간으로 "어느 epoch의 가중치를 쓸지" 고르는 것도 test를 보지 않는다(Informer·Autoformer·PatchTST 계열의 표준 절차; `EarlyStopping`이 최고 검증 가중치를 `model_state/checkpoint.pth`에 이미 저장하고 있었다). 검증 구간을 조기 종료·가중치 선택·ℓ 후보 선택에 함께 쓰면 검증 수치 자체는 낙관적으로 치우치지만 test·미래 구간은 손대지 않는다.

**2차(patience 2) 중단:** 10:22에 시작한 `flif-v1-s2`(120 run)를 10:32에 중단했다(사용자 결정 (1)로 학습 규칙이 바뀌어 결과가 바로 대체되므로). 진행 중이던 ETTm2 H96 8개는 결과 없이 멈췄다(RESULT·SKIP 0). 중간 산출물 `log/flif-v1-s2/`는 로컬에 남기고 Git에는 넣지 않는다. 사전 고정 항목(`952c5c98b`)의 설계·판정은 아래 patience 10 설계로 이어진다.
- 실수 기록: 중단 중 `pkill -f` 패턴이 제 셸의 명령 줄에도 걸려 셸이 함께 종료됐다(메모리에 적어 둔 같은 함정). 남은 프로세스를 PID 목록으로 확인했고(없음), GPU 4장이 비었다.

**정정 2건 (과거 항목은 고치지 않고 여기 적는다):**
1. queue 파일(`**/scripts/queues/*.txt`)은 저장소 `.gitignore` 규칙으로 **Git에 들어가지 않는다.** 앞선 항목들에 "큐 72줄/120줄"이라 적은 파일은 commit에 없다. 1차 queue는 항목에 적은 반복(6 조건 × ETTh2/ETTm2 × 96/720 × 42/7/2026, `SUITE=flif-v1-main`)으로, 2차는 `952c5c98b` 항목의 반복으로 재현된다. 이번부터 queue 생성 스크립트(`scripts/make_queue_s2p10.sh`)를 commit한다.
2. 1차 RESULT 줄의 `epochs_run`은 실제 epoch 수보다 **1 많다**(`grep -c '^epoch'`가 인자 출력 줄 `epoch----…`도 셌음). 위 표의 epoch 수는 stdout의 `epoch N (lr=…)` 줄에서 다시 센 값이다. `run_ett.sh`를 고쳤다(`EP` = 학습률 출력 줄 수).

**코드 변경 (`nsmt_flif_v1/forecasting/`, 기본값은 그대로):** `config.py --test_best`; `train.py` — 조기 종료가 최고 검증 가중치를 저장한 epoch를 기록하고, `--test_best`면 평소 test(마지막 가중치, `log/final+result.csv`) 뒤에 `checkpoint.pth`로 같은 test를 한 번 더 해 `log/best/`(`final+result.csv`, `neuron_energy.csv`, `best_epoch.txt` 등)에 쓴다; `model.py load_mymodel(..., ckpt=)`; `ett_future.py`·`neo_memory_probe.py --weights {last,best}`; `run_ett.sh` — `PATIENCE`(argparse가 마지막 `--patience`를 쓰므로 뒤에 덧붙임), `TEST_BEST=1`, RESULT에 `patience lr_final lr_drops mse_best mae_best syn_energy_best total_fft_uJ_best total_direct_uJ_best rho_neo_best rho_hippo_best best_epoch`; `summarize_flif.py --weights {last,best} --summary_name`(best면 검증 MSE도 최고 epoch 값), epoch·최고 epoch·학습률 감소 횟수 표; `make_post_queues.py --weights`; `scripts/{make_queue_s2p10.sh,stage2p10_driver.sh}`.

**점검:** `check_flif.py` 29/29(기본값 = nsmt_base 비트 동일 유지). smoke(`results/smoke-p10-20261015/raw.txt`): ETTh1 H96 seed 42, 8 epoch, patience 10 — base/hippo_only/neo_flif_both@L2g1.0 모두 정상, 세 run 모두 마지막 epoch가 최고 epoch(7)라 `mse_best = mse`. **두 가중치가 갈리는 경우 확인:** ETTm2 H720 neo_flif_both seed 42, 6 epoch, patience 10 → 검증 예측 오차 최저 epoch 2(0.362329), 마지막 epoch 5; 마지막 가중치 test mse 0.403025, 최고 검증 가중치 0.402127, `best_epoch=2`. 독립 평가기 `ett_future.py --split test --weights last|best`가 0.403025 / 0.402127을 재현(MAE는 0.393109 vs 0.393110, 누적 순서에 따른 마지막 자리 차이) — `results/checks/bestval_crosscheck/`. 이 smoke에서 학습률 감소는 0회(6–8 epoch로는 조건에 닿지 않음). run당 시간: ETTm2 H720 f-LIF 6 epoch에 635 s(GPU 단독).

**설계 (사전 고정, suite `flif-v1-s2p10`, 168 run):** 1차와 2차의 모든 조건을 **patience 10**으로 다시 돌린다 — base, hippo_only, 그리고 조건 묶음 6개(L0g0.05, L0g1.0, L2g0.05, L2g1.0, L4g0.05, L4g1.0) × {`neo_lif_both`, `neo_flif_both`} = 14 조건 × ETTh2/ETTm2 × H96/H720 × seed 42/7/2026. 최대 epoch는 논문 값(ETTh2 100, ETTm2 50) 그대로. 그 외 하이퍼파라미터·f-LIF 설정은 이전과 같다. 비교는 모두 이 suite 안에서만 한다(1차·중단된 2차의 patience 2 결과와 섞지 않는다).
- **판정에 쓰는 test 가중치 = 최고 검증 가중치**(판단): patience 10이면 마지막 가중치는 최고점보다 10 epoch 뒤라 과적합된 가중치가 된다. 마지막 가중치 결과는 `summary_lastweights.md`로 함께 보고한다. 기억 측정도 최고 검증 가중치로 한다(`memory_probe_bestval/`).
- **학습률:** 스케줄 코드는 바꾸지 않는다(사용자 결정 (2)). 다만 patience가 10이 되면 기존 `ReduceLROnPlateau`(3번째 무개선 epoch에 ×0.1)가 조기 종료보다 먼저 걸릴 수 있어 **실제로 학습률이 줄어들기 시작한다.** 이것은 논문 코드의 원래 동작이 처음으로 드러나는 것이며, run마다 `lr_drops`·`lr_final`로 기록해 해석에 반영한다(patience 효과와 학습률 감소 효과는 이 설계로 분리되지 않는다).
- 판정 규칙은 `952c5c98b` 항목의 Q-A·Q-B·기준 (2)·(1)·확증 후보 규칙 그대로(대조군은 같은 suite의 hippo_only와 같은 묶음 LIF). 추가 보고: base vs hippo_only(긴 학습에서 신피질 경로가 기여하는지, 논문 LIF 기준).
- 실행: tmux 세션 `flif_s2p10`, `scripts/stage2p10_driver.sh`(GPU당 2개, 학습 → 최고 검증 가중치 기억 측정 → 요약 2개). 예상 7–10시간(ETTm2가 run당 20–40분).
- 미래 구간은 이번에도 쓰지 않는다(확증 단계용).

**commit:** 이 항목과 코드를 담은 commit을 push한 뒤 시작한다. 결과는 다음 항목.

---

## 2026-10-15 21:12 KST — 2차(patience 10) 학습 168개 중 166개 완료, 2개 실패(폴더 이름 길이) → 이름 규칙 수정 후 2개 재실행

- `flif-v1-s2p10` 학습 10:50–20:56, 기억 측정 154개 20:56–21:09, 요약 `results/flif-v1-s2p10/summary.md`(최고 검증 가중치)·`summary_lastweights.md`. 결과·판정은 2개 재실행 뒤 다음 항목에 적는다.
- **실패 2개:** `neo_flif_both` ETTm2 H720 seed 2026의 `L2g0.05`, `L4g0.05`. 원인은 실행 폴더 이름이 256바이트로 파일 시스템 한도(255)를 넘어 `os.makedirs`가 `OSError: [Errno 36] File name too long`으로 학습 시작 전에 실패한 것(0 epoch). 같은 칸의 다른 폴더 이름은 최대 255바이트로 한도에 걸려 있었다(4자리 seed + `flif` + `g0.05` + ETTm2 H720의 heads 16 조합만 넘침).
- **수정 (`nsmt_flif_v1/forecasting/`):** `config.py` — 폴더 이름의 `+neo+…` 접미사에서 solver 옵션(`-h…-pred-m…`)은 기본값이 아닐 때만 붙인다(기본 경로 이름은 그대로, nsmt_base 동일성 무관). `scripts/run_ett.sh` — 학습 실패(rc≠0)를 최고 검증 test 확인보다 먼저 검사한다. 이번 두 SKIP 줄은 실제 원인(학습 실패)이 아니라 "best-checkpoint test missing"으로 잘못 표시됐다.
- **재실행:** 같은 두 run을 tmux 세션 `flif_s2p10_fix`에서 GPU 0·1로 다시 돌리고(`PATIENCE=10 TEST_BEST=1 SUITE=flif-v1-s2p10`), 끝나면 두 run의 기억 측정과 요약 재작성까지 이어서 한다. 재실행 run의 폴더 이름만 짧고 설정은 같다.

---

## 2026-10-15 21:48 KST — 2차 결과 (patience 10, 168 run, 최고 검증 가중치): **확증 후보 없음** — 모든 신피질 조건이 Hippo-only보다 나쁨, (B) 기울기 개방은 해로움, (A) 앞 2패치 목표에서만 f-LIF > LIF(3/4칸, 단서) — 태그 `exp/nsmt-flif-neo-20261015-2`

**구분:** 사전 고정한 2차 **탐색**(조건 묶음 6개)의 완료 기록. 코드 commit `2d41bfc7c`(166 run), `4d8d8223f`(재실행 2 run; 폴더 이름만 짧음). 이 항목이 속한 commit은 태그 `exp/nsmt-flif-neo-20261015-2`로 식별한다.

**실행:** tmux `flif_s2p10`(10:50–21:09: 학습 168 → 166 완료·2 실패, 기억 측정 154, 요약), tmux `flif_s2p10_fix`(21:12–21:45: 실패 2개 재실행 + 기억 측정 2 + 요약 재작성). GPU당 2개, RTX A6000 × 4, `snn_recall`. 재실행 둘(ETTm2 H720 seed 2026 f-LIF): L2g0.05 → 최고 검증 가중치 mse 0.403105(최고 epoch 9, 20 epoch; 마지막 가중치 0.405083), L4g0.05 → 0.399760(최고 epoch 1, 12 epoch; 마지막 0.408415). 둘 다 아래 표의 칸 평균에 포함됐다. 산출물: `nsmt_flif_v1/forecasting/results/flif-v1-s2p10/{raw.txt,summary.md,summary_lastweights.md,memory_probe_bestval/}`, 실행별 `log/flif-v1-s2p10/<DS>/261015/<run>/log/{best_log_0.csv,final+result.csv,neuron_energy.csv,best/…}`(Git), checkpoint·stdout·TensorBoard(로컬). raw.txt에는 실패 2개의 SKIP 줄이 그대로 남아 있다(집계는 RESULT만 읽음).

### 표준 test MSE (최고 검증 가중치, seed 3개 평균; 전체 표는 `summary.md`)

| 조건 | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| **hippo_only** | **0.285518** | **0.411191** | 0.173930 | 0.401147 |
| base(논문 LIF) | 0.289553 | 0.415306 | 0.174388 | 0.401433 |
| neo_lif_both (L0g0.05) | 0.291403 | 0.419786 | 0.174274 | **0.400285** |
| neo_flif_both (L0g0.05) | 0.287388 | 0.413676 | 0.174612 | 0.401368 |
| neo_flif_both@L2g0.05 | 0.286142 | 0.415108 | **0.173847** | 0.401173 |
| neo_lif_both@L2g0.05 | 0.293565 | 0.420931 | 0.174392 | 0.400855 |
| neo_flif_both@L0g1.0 | 0.290010 | 0.417937 | 0.174764 | 0.403515 |
| neo_flif_both@L4g0.05 | 0.291795 | 0.417598 | 0.174706 | 0.400736 |

### 사전 고정 판정 (`summary.md` "stage 2" 표 그대로)

| 묶음 | Q-A LIF vs hippo (평균, 12쌍 중 낮음) | Q-A f-LIF vs hippo | Q-B f-LIF vs LIF (칸, 평균) | 기준 (2) f-LIF | M2 f-LIF (ETTh2 / ETTm2) | 검증 MSE f-LIF / LIF |
|---|---|---|---|---|---|---|
| L0g0.05 | +1.03%, 2 | +0.43%, 3 | 2/4, −0.59% | 불통과 | 통과 / 불통과 | 0.31421 / 0.31338 |
| L0g1.0 | +1.12%, 3 | +1.07%, 1 | 3/4, −0.04% | 불통과 | 통과 / 불통과 | 0.31463 / 0.31433 |
| L2g0.05 | +1.34%, 2 | +0.28%, 3 | **3/4, −1.04% (통과)** | 불통과 | 불통과 / 통과 | 0.31573 / 0.31409 |
| L2g1.0 | +1.39%, 2 | +1.19%, 2 | 2/4, −0.20% | 불통과 | 통과 / 통과 | 0.31535 / 0.31540 |
| L4g0.05 | +1.09%, 2 | +1.02%, 2 | 2/4, −0.06% | 불통과 | 통과 / 불통과 | 0.31405 / 0.31392 |
| L4g1.0 | +0.71%, 3 | +0.82%, 3 | 2/4, +0.11% | 불통과 | 통과 / 불통과 | 0.31568 / 0.31531 |

**확증 후보: 없음**(기준 (2)를 통과한 묶음이 없다). 보고만: hippo_only vs base −0.68%(12쌍 중 10쌍 낮음); f-LIF@L2g0.05 vs base −0.40%(12쌍 중 9쌍 낮음) — 칸별 값은 `summary.md`.

### 에너지 (µJ/창, 논문 방식 total; 최고 검증 가중치)
hippo_only 58.68 / 67.24 / 65.97 / 72.81, base 56.78 / 68.57 / 64.93 / 71.40, f-LIF 56.94 / 68.75 / 64.27 / 71.59, f-LIF@L2g0.05 56.30 / 67.83 / 64.54 / 71.07 (칸 순서는 위 표와 같음). 신피질 조건이 hippo_only보다 1–4% 적고, ETTh2 H720만 1–2% 많다(1차와 같은 모양).

### 기억 측정 (최고 검증 가중치, 512창, 윈도우 평균 치환)
- M1(S(12)/S(1) ≥ 0.10 이고 LIF 대조군의 3배): **모든 묶음·데이터 불통과**. f-LIF의 비율은 같은 묶음의 LIF보다 일관되게 2–3배 크다(예: ETTh2 L0g0.05 0.052 vs 0.016, ETTm2 L2g0.05 0.230 vs 0.109) — 오래된 패치에 대한 신피질 출력의 민감도가 더 오래 남는다. 단 ETTm2는 LIF도 0.07–0.11로 높아 발산 민감도가 섞인다(1차 항목의 해석과 같음).
- M2(마지막 12패치만 볼 때 오차 증가가 1% 이상이고 LIF보다 큼): ETTh2 6묶음 중 5개, ETTm2 6묶음 중 2개 통과 — 일관된 패턴이 아니다.

### 해석
1. **신피질 경로는 이 결합 방식에서 예측을 돕지 못한다.** 12개 신피질 조건이 모두 Hippo-only보다 평균 0.3–1.4% 나쁘고, 길게 학습하자 Hippo-only가 논문 원본보다 나아졌다(1차의 짧은 학습에서는 +0.31%로 비슷했던 것이 −0.68%로 뒤집힘). 즉 1차의 "신피질이 기여하지 않음"은 짧은 학습 때문이 아니었다.
2. **(B) 기울기 개방은 해롭다**: f-LIF·LIF 모두 Hippo-only 대비 +0.7–1.4%, 최고 검증 epoch가 일찍 온다(학습률 감소도 적음).
3. **(A) 앞 2패치 목표에서만 f-LIF가 LIF보다 낫다(3/4칸, 평균 −1.04%; 재실행 전 4/4).** 단서로만 기록한다 — 묶음 6개를 시험하면 차이가 없어도 하나가 이 정도로 나올 수 있고, 검증 MSE는 반대 방향이며(0.31573 vs 0.31409), 그 f-LIF도 Hippo-only보다 +0.28% 나쁘다. 차이는 주로 LIF 쪽이 이 목표에서 나빠진 데서 온다(LIF@L2g0.05 vs hippo +1.34%).
4. **긴 학습이 항상 낫지는 않다**: 논문 원본은 ETTh2에서 1차(짧은 학습, 마지막 가중치)보다 나빠졌다(H96 0.283453 → 0.289553, H720 0.409741 → 0.415306). 검증 구간 기준 선택이 ETTh2 test로 옮겨지지 않는 경우다(ETTh2의 알려진 분포 이동과 일관). 이 비교는 학습 규칙이 다른 suite 사이라 보고만 한다.
5. 학습률: patience 10에서 `ReduceLROnPlateau`가 처음으로 작동했다(ETTm2 H96 평균 3–4.3회, ETTh2 H96 1.7회, H720 0–1회). 최고 검증 epoch는 ETTm2 H720에서 1.3–2.3로 여전히 매우 이르다. 긴 학습과 학습률 감소의 효과는 분리되지 않는다.

### 한계 · not run
- 3 seed·2 데이터셋의 방향 판정(검정 없음), 탐색 6묶음. 미래 구간은 쓰지 않았다(확증 단계용으로 남김).
- α 0.5 하나, MLP층만 배치 not run, 논문 LIF(sj_lif)의 (A)(B) not run.
- 1차·patience 2 결과와의 비교는 학습 규칙이 달라 보고용.

### 결론과 다음
2차 탐색에서도 **확증 후보 없음.** 현재의 결합 방식(AND 융합 + cross-attention, IAND)으로는 목표를 바꾸거나(A), 기울기를 열거나(B), 길게 학습해도 신피질 경로가 Hippo-only보다 나아지지 않았다. 다음 단계 제안(사용자 결정 필요): **상호작용 방식 교체** — 선행 연구 조사(10-15 00:30 항목 §결정 9) 1순위인 DMP식 전류 주입(신피질 출력을 해마 뉴런 막전위 식의 추가 전류로)부터 시험하고, "(A)에서 f-LIF > LIF" 단서는 그 뒤 필요하면 seed를 늘려 확인한다. α 탐색은 신피질이 기여하기 시작한 뒤로 미룬다.

### 같은 commit의 다른 변경
- 감사 78·79 수용(`docs/PROJECT_LOG.md` 같은 시각 항목): A35 — "α=1 f-LIF = LIF"는 **전체 이력에서만** 성립(직접 확인: memory 8에서 상태 차이 최대 3.19, 스파이크 다름). 이 실험 계열의 모든 run과 점검 c2는 전체 이력이므로 영향 없음. 이전 MODI 항목의 "LIF = f-LIF(α=1)" 표현도 "전체 이력일 때"로 읽는다. 두 `neurons.py` 설명 보충, `spikede_parity.py`의 원본 commit 기록 수정.
- `nsmt_flif_v1/forecasting/results/checks/c5_test_log/`의 그림 4개(PNG/PDF)를 Git 추적에서 뺐다(사전 고정 commit `952c5c98b`에 의도치 않게 들어갔고, 점검을 다시 돌릴 때마다 바뀜; 로컬 파일은 유지). 점검 `check_flif.py` 29/29 재확인.

---

## 2026-10-17 20:45 KST — 3단계 설계: 신피질 경로의 역할 분석(Phase A) → 직렬 구조 "정렬 스파이킹 임베딩"(Phase B) — 의도와 구현의 간극, 사용자 결정 (실행 전)

**사용자 요청 (10-16):** 상호작용 방식을 바꾸기 전에 신피질 망의 역할과 과제에 대한 영향을 먼저 분석한다. 원 의도: (1) 스파이킹 시간 축과 시계열 토큰 축을 정렬해 뉴런의 시간 처리 능력을 시계열 시간 정보 처리에 직접 쓴다 → 에너지 절약 + 시간 표현 개선; (2) CLS 관점에서 뉴런의 기억으로 look-back 창 너머의 장기 의존성(원 시계열 전체의 추세·경향·계절성의 평균)을 갖게 한다 — LIF는 1차 ODE라 단기 기억이므로 f-LIF로 보강. 사용자 아이디어: **병렬 대신 직렬** — 해마(백본)의 spiking embedding 모듈을 지금의 신피질 망(T=N 정렬, f-LIF)으로 바꿔 시간 패턴을 담은 스파이크 임베딩을 백본에 넣는다. 브랜치 `exp/nsmt-flif-neo`, 이전 commit `1b3fe7acf`(태그 `exp/nsmt-flif-neo-20261015-2`).

**의도와 구현의 간극 (코드·저장 산출물에서 확인):**
1. **정렬은 신피질에만 있다.** 해마는 같은 패치열을 T=24번 반복해 넣는다(`ours.py`: `x.unsqueeze(0).repeat(self.T, …)`) — 해마의 스파이킹 시간 축은 시계열 시간이 아니라 같은 값을 24번 보는 반복 축이다. 에너지 측정(ETTh2 H96 neo_flif_both seed 42, 최고 검증 가중치, 창 1개, syops 층별 + 뉴런 식 60/64): 시냅스 31.8 µJ = 해마 임베딩 11.85(첫 Linear가 반복 입력에 MAC 2.06M) + 해마 백본 17.49 + head 1.76 + **신피질 0.66**; 뉴런 내부는 해마 층 8개가 각 ≈3 µJ(≈23 µJ), 신피질 2층 ≈0.6 µJ. **에너지의 97% 이상이 반복 축을 쓰는 해마에서 나온다** → 정렬의 에너지 이점이 주 경로에서 실현되지 않았다.
2. **신피질은 우회 가능한 곁가지다.** 출력은 해마 key/value에 AND 마스크(`x ⊙ x_neoᵀ`)로만 들어가고 과제 기울기는 5%. 해마는 신피질 없이도 예측할 수 있다(2차: Hippo-only가 모든 신피질 조건보다 나음).
3. **창 너머 기억은 구조적으로 불가능하다.** 배치마다 상태 초기화(`functional.reset_net`; f-LIF 상태도 forward 안), 신피질 시간 축은 창 안 24스텝. f-LIF는 24스텝 안의 감쇠를 멱법칙으로 바꿀 뿐이며 그 차이는 긴 시퀀스에서 커진다.

**직렬 아이디어 검토:** 강점 — 백본을 T_b = 1(스파이킹 스텝 1개, 토큰 N개)로 돌려 반복 축을 없앨 수 있다(위 측정의 단순 환산으로 창당 ≈56 µJ → ≈3–4 µJ, 측정으로 확인 예정); 신피질이 우회 불가능해져 과제 기울기 전부를 받는다; 어떤 SNN 백본에도 붙는 plug-and-play 앞단이 된다. 선행 연구상 SeqSNN·iSpikformer는 시계열 한 스텝을 스파이킹 하위 스텝으로 쪼개는 반복 축을 쓰며, "토큰 시간 = 스파이킹 시간으로 정렬된 f-LIF 임베딩이 반복 축을 대체"는 이와 구별된다. 위험 — T_b = 1이면 토큰마다 특징당 1비트(정밀도 손실 가능); 에너지 이득 일부는 T_b 축소 자체에서 오므로 같은 T_b = 1의 평범한 임베딩(Hippo-only T_b = 1)과 비교해야 한다; 직렬이면 신피질은 "상보적 기억"이 아니라 "시간 부호화 앞단"이 되어 CLS 서술이 바뀐다.

**사용자 결정 (10-16):** (1) 직렬 백본 스텝 **T_b = 1과 T_b = 24 둘 다**; (2) 의도 2(창 너머 장기 기억)는 **직렬 검증 뒤 별도 단계**(후보: 긴 저해상도 이력을 신피질 시간 축에 정렬, DMP식 느린 상태를 백본 뉴런에 추가 전류로 주입, 느리게 학습되는 가중치 기억); (3) 직렬의 보조 손실 **없음**; (4) 진행: Phase A 실행·보고, 그동안 Phase B 구현·점검·smoke까지; **B 본 학습(60 run)은 A 결과와 smoke를 보고하고 확인받은 뒤**.

**Phase A 설계 (학습 없음):** 2차 suite `flif-v1-s2p10`의 최고 검증 가중치 72개(base, hippo_only, neo_lif_both, neo_flif_both, neo_flif_both@L2g0.05, neo_lif_both@L2g0.05 × ETTh2/ETTm2 × H96/H720 × seed 3). 표준 test로 ΔMSE, 탐침 적합은 train·점수는 val(test로 아무것도 고르지 않음), 미래 구간 미사용. 스크립트 `nsmt_flif_v1/forecasting/neo_role_analysis.py`:
- A1 대체 시험: 신피질 출력을 (i) 다른 창의 것, (ii) 전부 1(K/V = 해마 자신), (iii) 전부 0, (iv) 토큰 순서 뒤섞기, (v) 입력 시간 역순으로 만든 것으로 바꿔 ΔMSE.
- A2 마스크 통계: 토큰별 신피질 발화율, AND 뒤 살아남는 해마 스파이크 비율, 창 사이 마스크 일치율(우연 수준 대비).
- A3 선형 탐침(ridge, 특징 표준화, α 10 고정): 특징 {신피질 전체 스텝, 신피질 마지막 스텝, 해마 임베딩(반복 축 평균) 전체, 해마 마지막 토큰, 원 패치} → 목표 {예측 구간 평균·기울기(창 정규화 공간), d패치 전 패치 평균 d = 0, 1, 4, 12, 23}.
- A4 해마 반복 축의 역할: 백본은 반복 축 방향으로 인과적이고 head는 스텝마다 같으므로 "앞 k스텝 평균 예측 = 반복을 k로 줄인 모델". k = 1, 2, 4, 8, 12, 24의 MSE와 스텝 단독 MSE.
- A5 기능적 민감도: 순전파는 같게 두고 신피질 기울기 비율을 1로 바꿔 예측 오차의 신피질 입력 기울기 노름 vs 해마 입력 기울기 노름(반복 축 합산).
- A6 에너지 분해(경로별 시냅스·뉴런)와 직렬 T_b ∈ {1, 2, 4, 24} 환산(발화율 고정 가정).

**Phase B 설계 요지:** `--arch {parallel, serial}`, `--backbone_T`. serial: 해마 `encoding` 대신 신피질(`encoding_neo` + `time_block`) 출력 [N, B·C, 1, D] → [T_b, B·C, N, D](T_b > 1이면 반복) → 백본(key/value = 백본 자신), 디코더·보조 손실 없음. 조건 5개(hippo_only T_b=1, serial LIF/f-LIF × T_b ∈ {1, 24}) × ETTh2/ETTm2 × H96/H720 × seed 3 = 60 run, 학습 규칙은 2차와 같음(patience 10, 최고 검증 가중치). 판정 수치는 본 학습 전 사전 고정 항목에서 확정.

---

## 2026-10-17 21:16 KST — Phase A 결과: 신피질 경로는 "창마다 거의 같은, 위치별 희소 마스크"로만 쓰이고 창별 내용은 쓰이지 않는다; 해마의 24스텝 반복 축은 2–4스텝이면 충분 (학습 없음, 72 checkpoint)

**실행:** `nsmt_flif_v1/forecasting/neo_role_analysis.py`, queue 생성 `scripts/make_queue_neo_role.sh`, 실행기 `scripts/neo_role_driver.sh`(tmux `neo_role`, GPU당 2개, 20:54–21:14). 대상 = `flif-v1-s2p10`의 최고 검증 가중치 72개(base, hippo_only, neo_lif_both, neo_flif_both, neo_lif_both@L2g0.05, neo_flif_both@L2g0.05 × ETTh2/ETTm2 × H96/H720 × seed 42/7/2026). **72/72 모두 깨끗한 순전파가 `log/best/final+result.csv`의 test MSE를 재현**(허용 2e-6)했고, 반복 축 24스텝 평균도 원래 MSE와 같았다. 산출물 `results/neo-role-20261017/{runs/*.json, runs.csv, summary.md, a6_energy.csv}`. 시험 중 고친 것: 예측 추출 도우미가 `@torch.no_grad()`라 A5 기울기를 직접 계산; ridge를 CPU LAPACK으로 풀면 DataLoader worker를 띄운 뒤 같은 프로세스에서 세그먼테이션 오류가 나서 run의 GPU에서 풀도록 함; 탐침 규제 강도는 train의 뒤 20%로 고른다(val·test는 선택에 쓰지 않음).

**A1 신피질 출력 대체 (test MSE 변화, seed 3개 평균, 칸 순서 ETTh2 H720 / H96 / ETTm2 H720 / H96):**

| 대체 | base | neo_lif_both | neo_flif_both | neo_flif_both@L2g0.05 |
|---|---|---|---|---|
| 다른 창의 것 | +0.25 / +0.05 / −0.03 / +0.13% | +0.10 / −0.22 / +0.03 / +0.07% | +0.18 / +0.16 / +0.04 / +0.09% | −0.22 / +0.07 / +0.01 / +0.28% |
| 시간 역순 입력 | +0.50 / −0.05 / −0.11 / −0.24% | +0.20 / +0.46 / +0.15 / +0.26% | +0.72 / −0.08 / −0.16 / +0.27% | +0.59 / +0.10 / +0.10 / +0.82% |
| 토큰 순서 섞기 | +2.21 / +0.45 / +0.18 / +2.21% | +1.76 / −1.32 / +0.54 / +2.14% | +4.31 / +1.34 / +0.36 / +2.67% | +0.51 / +0.11 / +0.25 / +2.32% |
| 전부 1(마스크 없음) | +0.74 / −1.05 / +1.11 / +4.43% | +1.23 / −2.24 / +2.85 / +6.61% | +2.80 / −1.26 / +2.65 / +4.71% | +1.04 / +4.19 / +1.21 / +7.36% |
| 전부 0 | +2.09 / −0.17 / +0.49 / +16.85% | +0.77 / −0.75 / +0.66 / +12.04% | +5.74 / +0.11 / +0.40 / +11.06% | +0.15 / +0.66 / +0.69 / +16.50% |

**A2 마스크:** AND 뒤 살아남는 해마 스파이크 13–30%. 같은 토큰 위치에서 두 창의 신피질 출력 일치율 0.86–0.92(우연 수준 0.67–0.83보다 높음).

**A4 해마 반복 축 줄이기 (재학습 없이, 24스텝 대비 MSE 변화):** k = 1: +0.7 – +4.6%; k = 2: −0.4 – +1.4%; k = 4: −0.7 – +0.5% (전 조건·칸). Hippo-only도 같다(k = 1: +0.66 – +3.93%, k = 4: +0.06 – +0.42%).

**A5 기능적 민감도:** 예측 오차의 입력 기울기 노름 중 신피질 몫 0.21–0.67(base 0.21–0.36, 앞 2패치 목표 조건 0.46–0.67).

**A3 마지막 스텝 기억 (ridge R², val):** 4패치 전 패치 — 신피질 마지막 스텝: LIF 0.14 / 0.15 / 0.52 / 0.50, **f-LIF 0.03 / 0.02 / 0.27 / 0.32**(LIF보다 낮음); 해마 마지막 토큰(기준) 0.01 / 0.01 / 0.15 / 0.16. 12패치 전 — LIF 0.12–0.25, f-LIF 0.07–0.22, 해마 0.03–0.20. 23패치 전 — LIF 0.06–0.20, f-LIF 0.05–0.16, 단 앞 2패치 목표의 ETTm2에서만 f-LIF 0.21–0.23 > LIF 0.11–0.14. 예측 구간 평균·기울기 목표는 **원 패치로도 val R²가 음수**(train과 val 구간의 분포 차이)라 판단에 쓰지 않는다.

**A6 에너지 (창당 µJ, 최고 검증 가중치, 경로별):** 총 55–71 = 해마 임베딩 ≈15(반복 입력의 첫 Linear + BN + LIF) + 해마 백본 37–47 + head 1.8(H96) / 12–14(H720) + 신피질 0.9–1.8. 직렬 환산(발화율 고정 가정): T_b = 1 → 2.5–3.9, T_b = 4 → 7.4–10.5; Hippo-only T_b = 1 → 2.3–3.0.

**해석:**
1. **신피질의 창별 내용은 쓰이지 않는다.** 다른 창의 신피질 출력으로 바꿔도 ±0.3%. 쓰이는 것은 위치별 마스크 패턴(토큰 순서를 섞으면 최대 +4.3%)과 마스크 밀도(전부 0이면 ETTm2 H96에서 +11–24%)다. 마스크는 창이 바뀌어도 거의 같다. 즉 신피질은 "느린 추세 기억"이 아니라 **학습된 위치별 희소 key/value 게이트**로 동작한다. 1·2차에서 뉴런 종류·목표·기울기를 바꿔도 결과가 같았던 이유와 맞는다.
2. **f-LIF가 오래된 패치를 더 잘 담지 않는다.** 마지막 스텝 상태에서 4·12패치 전 내용을 선형으로 읽는 정도는 오히려 LIF가 높다(1·2차의 "f-LIF는 오래된 패치 민감도가 2–3배"와 함께 보면, f-LIF의 출력은 오래된 입력에 더 오래 반응하지만 그 반응이 특정 과거 패치를 선형으로 복원할 수 있는 형태는 아니다). 이 계열에서 "f-LIF = 더 긴 기억"을 실험 근거로 쓸 수 없다.
3. **해마의 24스텝 반복 축은 거의 중복이다.** 재학습 없이 4스텝으로 줄여도 +0.5% 이내, 1스텝이어도 +4.6% 이내. 에너지의 ≈95%가 이 축에서 나온다. **직렬 구조(T_b = 1)의 근거로 쓸 수 있다**(단 T_b = 1에서 학습했을 때의 성능은 Phase B가 잰다).
4. 위 2·3은 학습된 병렬 모델에 대한 관찰이며, 직렬로 학습하면 신피질이 받는 기울기와 역할이 달라진다.

**한계·not run:** 미래 구간 미사용. 탐침은 선형(ridge)이며 비선형 판독은 not run. A5는 국소 민감도라 A1(실제 대체 효과)과 다를 수 있다(신피질 몫 0.2–0.7이지만 내용 대체 효과는 ±0.3%). A6의 직렬 환산은 발화율 고정 가정 — 실제 값은 Phase B의 측정으로 대체한다.

**commit:** 이 항목·분석 스크립트·결과, 감사 80·81 수용(`docs/PROJECT_LOG.md` 같은 시각 항목)과 두 `neurons.py` 설명 재수정을 담은 commit을 push한다(분석이므로 태그 없음). Phase B 구현·smoke는 다음 항목.

---

## 2026-10-17 21:19 KST — Phase B 구현: 직렬 "정렬 스파이킹 임베딩"(`--arch serial`, `--backbone_T`), 점검 38/38, smoke 5/5 — 본 학습(60 run)은 사용자 확인 대기

**코드 (`nsmt_flif_v1/forecasting/`; 수정 스크립트를 작업용 사본에서 먼저 시험한 뒤 Phase A가 끝나고 반영):**
- `config.py`: `--arch {parallel(기본), serial}`, `--backbone_T`(0 = 패치 수 N = 논문의 반복 횟수); serial은 hippo_only·`--aux_lead`·`--neo_grad`와 함께 쓰면 거부; 폴더 이름에 `+serial`, `+Tb<k>`(기본값이면 없음).
- `ours.py myModel`: serial이면 해마 `encoding`·디코더를 만들지 않고, 신피질(`encoding_neo` + `time_block`, 뉴런은 `--neo_neuron`) 출력 [N, B·C, 1, D]를 [1, B·C, N, D]로 바꿔 `backbone_T`번 반복해 백본(key/value = 백본 자신)에 넣는다. 보조 손실 없음(결정 3). hippo_only·parallel에서도 `backbone_T`가 해마 입력의 반복 횟수를 정한다. head 출력 정리와 `train.py`의 손실 계산은 `model.T` 대신 출력의 스텝 수를 쓴다(기본값에서는 같은 값).
- `model.py`(옵션 전달), `ett_future.cond_name`(`serial_<neuron>`, `@Tb<k>` 접미사), `scripts/run_ett.sh`(`COND=serial_lif|serial_flif`, `TB=`), `scripts/make_queue_s3.sh`(본 학습 queue 생성, 60줄), `scripts/smoke_s3.sh`.

**점검 (`results/checks/check_flif.txt`, 38/38):** 기존 c1–c7 그대로(기본값 = nsmt_base 비트 동일). **c8** serial 구조(LIF·f-LIF × T_b 1·24: 해마 임베딩·디코더 없음, 출력 스텝 = T_b); **정렬 임베딩의 인과성** — 학습된 f-LIF 신피질(2차 ETTh2 H96 seed 42, 발화율 0.126)에서 패치 j 이후를 바꾸면 토큰 j 이전 출력은 변하지 않고 토큰 j의 스파이크 28–29%가 바뀐다(j = 5, 12, 20; 지연 없음). 처음 버전은 초기화 모델로 시험해 신피질이 거의 발화하지 않아(발화율 0.0000) 사실상 빈 통과였으므로, 학습된 checkpoint를 쓰고 발화율 0.05 미만이면 실패로 고쳤다; 과제 기울기가 임베딩까지 닿음. **c9** hippo_only T_b=1 출력 = T_b=24 모델의 첫 스텝(같은 초기 가중치, 백본은 반복 축 방향으로 인과적), 명시한 T_b=N = 기본값(hippo_only·parallel).

**smoke (`results/smoke-s3-20261017/raw.txt`, ETTh1 H96 seed 42, 1 epoch, patience 10·최고 검증 test 경로 포함; 실험 아님):**

| 조건 | test MSE(1 epoch) | 시냅스 µJ | 총 에너지(논문식) µJ | 신피질 / 백본 발화율 | 파라미터 |
|---|---|---|---|---|---|
| hippo_only T_b=1 | 0.435475 | 0.64 | 1.131 | – / 0.047 | 81,664 |
| serial_lif T_b=1 | 0.424773 | 0.80 | 1.417 | 0.059 / 0.043 | 83,840 |
| serial_flif T_b=1 | 0.422256 | 0.80 | 1.663 | 0.065 / 0.039 | 83,840 |
| serial_lif T_b=24 | 0.426328 | 10.68 | 21.316 | 0.060 / 0.056 | 83,840 |
| serial_flif T_b=24 | 0.425360 | 10.85 | 21.821 | 0.065 / 0.066 | 83,840 |

다섯 조건 모두 학습·저장·최고 검증 test·에너지 열까지 지난다. **측정 에너지:** T_b=1은 1.1–1.7 µJ로, 같은 칸의 학습된 논문 모델(c5, 31.90 µJ)보다 약 95% 적다(Phase A의 환산과 일치하는 크기). 직렬 T_b=24도 21.3–21.8 µJ로 논문 모델보다 적다(반복 입력에 대한 해마 임베딩 Linear가 없어서). 1 epoch의 MSE는 판단 근거가 아니다.

**본 학습 (실행 전 — 사용자 확인 후 시작):** `scripts/make_queue_s3.sh` → 조건 5개(hippo_only T_b=1, serial_lif/serial_flif × T_b ∈ {1, 24}) × ETTh2/ETTm2 × H96/H720 × seed 42/7/2026 = 60 run, patience 10·최고 검증 가중치, suite `flif-v1-s3`. 참조는 같은 학습 규칙의 2차 suite `flif-v1-s2p10`(hippo_only T_b=24, base, 병렬 LIF/f-LIF) — c1·c9로 기본 경로가 변하지 않았음을 확인했으므로 재사용한다. 판정 수치는 확인을 받을 때 사전 고정 항목으로 확정한다.

**commit:** 이 항목과 Phase B 코드·점검·smoke 결과를 담은 commit을 push한다.

---

## 2026-10-17 21:24 KST — Phase B 본 학습 사전 고정 (사용자 승인 "시작"): 직렬 정렬 임베딩 60 run, 판정 Q1–Q4

**사용자 결정 (10-17):** 직전 항목의 설계와 판정 기준으로 시작. 브랜치 `exp/nsmt-flif-neo`, 이전 commit `045a7b5aa`.

**설계 (고정):** suite `flif-v1-s3`, `scripts/make_queue_s3.sh`(60줄, 긴 칸 먼저) — hippo_only@Tb1, serial_lif@Tb1, serial_flif@Tb1, serial_lif(백본 T=24), serial_flif(T=24) × ETTh2/ETTm2 × H96/H720 × seed 42/7/2026. 신피질 = spikeDE LIF / f-LIF(α 0.5, τ 2, pred, h 1, 전체 이력), 직렬은 보조 손실·디코더 없음, 백본 key/value = 백본 자신. 학습 규칙은 2차와 같음(patience 10, 최대 epoch 논문 값, ReduceLROnPlateau 그대로, **판정은 최고 검증 가중치**, 마지막 가중치는 `summary_lastweights.md`로 보고). 참조 = 2차 suite `flif-v1-s2p10`(hippo_only T=24, base, 병렬 LIF/f-LIF; 같은 학습 규칙, c1·c9로 기본 경로 불변 확인). 미래 구간 미사용. 실행: tmux `flif_s3`, `scripts/stage3_driver.sh`(GPU당 2개; 학습 → 직렬 run의 기억 측정(최고 검증 가중치, 512창) → `summarize_flif.py --stage3`).

**판정 (고정; 셀 = 데이터 × H 4칸, seed 3개 평균, 같은 seed끼리 짝; 에너지 = 논문 방식 총합 `total_fft_uJ`):**
- **Q1 정렬 임베딩의 기여:** serial_flif@Tb1 vs hippo_only@Tb1 — 평균 상대 MSE ≤ −1% 이고 12쌍 중 ≥ 8쌍 낮음.
- **Q2 사용자 통과 기준 2의 경로 B("적당한 성능 + 크게 개선된 에너지"):** serial_flif@Tb1 vs hippo_only(T=24, 2차) — 평균 상대 MSE ≤ +1% 이고 에너지 ≤ 0.7배.
- **Q3 f-LIF vs LIF:** 같은 직렬·같은 백본 T(1과 24 각각)에서 4칸 중 ≥ 3칸 낮고 평균 ≤ −0.5%.
- **Q4 기억(M2):** 직렬 f-LIF에서 마지막 12패치만 남길 때 오차 증가 ≥ 1% 이고 같은 백본 T의 직렬 LIF보다 큼(ETTh2·ETTm2 각각).
- 보고만: 백본 T=24 vs T=1(반복 축 제거의 정확도 비용), 직렬 vs 병렬(base, neo_flif_both), serial_lif@Tb1 vs hippo_only@Tb1.
- 탐색 단계이므로 통과는 확증이 아니다. Q1 또는 Q2를 통과하면 seed 8개 + 미래 구간 확증을 따로 사전 고정한다.

**commit:** 이 항목·실행기·집계 코드(`summarize_flif.py --stage3`)를 담은 commit을 push한 뒤 시작한다. 결과는 다음 항목.

---

## 2026-10-17 23:08 KST — Phase B 결과 (직렬 정렬 임베딩, 60 run, 최고 검증 가중치): **Q1 불통과(정렬 임베딩 = 평범한 1스텝 임베딩), Q2 불통과(에너지 0.05배이지만 MSE +2.97%), Q3 통과(f-LIF < LIF, 단 마지막 가중치로는 사라짐), Q4 4칸 중 3칸** — 태그 `exp/nsmt-flif-neo-20261017`

**구분:** 사전 고정한 3단계 Phase B **탐색**의 완료 기록. 코드 commit `045a7b5aa`, 사전 고정 `8a4e90c9e`. 이 항목이 속한 commit은 태그 `exp/nsmt-flif-neo-20261017`로 식별한다.

**실행:** tmux `flif_s3`, `scripts/stage3_driver.sh`(GPU당 2개): 학습 21:24–23:01 **60/60 완료·실패 0**, 기억 측정 48개(직렬 run, 최고 검증 가중치, 512창) 23:01–23:05, 요약 `results/flif-v1-s3/{summary.md, summary_lastweights.md}`(2차 suite `flif-v1-s2p10`과 합쳐 계산). 산출물: `results/flif-v1-s3/{raw.txt, memory_probe_bestval/}`, 실행별 `log/flif-v1-s3/<DS>/261017/<run>/log/{best_log_0.csv, final+result.csv, neuron_energy.csv, best/…}`(Git), checkpoint·stdout(로컬).

### test MSE (최고 검증 가중치, seed 3개 평균; 칸 순서 ETTh2 H96 / H720 / ETTm2 H96 / H720)

| 조건 | MSE | 총 에너지(µJ/창) |
|---|---|---|
| hippo_only (T=24, 2차) | **0.285518 / 0.411191 / 0.173930 / 0.401147** | 58.68 / 67.24 / 65.97 / 72.81 |
| base 병렬 (2차) | 0.289553 / 0.415306 / 0.174388 / 0.401433 | 56.78 / 68.57 / 64.93 / 71.40 |
| neo_flif_both 병렬 (2차) | 0.287388 / 0.413676 / 0.174612 / 0.401368 | 56.94 / 68.75 / 64.27 / 71.59 |
| serial_flif (백본 T=24) | 0.293679 / 0.421751 / 0.178317 / 0.406633 | 40.42 / 47.20 / 45.61 / 51.02 |
| serial_lif (백본 T=24) | 0.302441 / 0.426814 / 0.178948 / 0.405765 | 40.22 / 47.69 / 45.86 / 51.11 |
| **serial_flif@Tb1** | 0.296254 / 0.421816 / 0.180477 / 0.408187 | **2.84 / 3.15 / 3.45 / 3.70** |
| serial_lif@Tb1 | 0.301025 / 0.427609 / 0.180608 / 0.407056 | 2.53 / 2.82 / 2.99 / 3.23 |
| hippo_only@Tb1 | 0.296977 / 0.426399 / 0.179289 / 0.405366 | 2.23 / 2.41 / 2.48 / 2.66 |

### 사전 고정 판정 (`summary.md` "stage 3")
- **Q1 정렬 임베딩의 기여 — 불통과.** serial_flif@Tb1 vs hippo_only@Tb1: −0.24 / −1.07 / +0.66 / +0.70%, 평균 +0.01%, 12쌍 중 4쌍 낮음.
- **Q2 경로 B — 불통과.** serial_flif@Tb1 vs hippo_only(T=24): +3.76 / +2.58 / +3.76 / +1.75%, 평균 **+2.97%**, 12쌍 중 0쌍, **에너지 0.050배**(≈95% 절감). 에너지 조건은 크게 넘지만 MSE 허용(+1%)을 넘는다.
- **Q3 f-LIF vs LIF — 통과(두 백본 T 모두).** T=1: −1.58 / −1.35 / −0.07 / +0.28%, 3/4칸, 평균 −0.68%; T=24: −2.90 / −1.19 / −0.35 / +0.21%, 3/4칸, 평균 −1.06%.
- **Q4 기억 M2 — 4칸 중 3칸.** T=1: ETTh2 통과(+8.0% vs +6.7%), ETTm2 불통과(+18.3% vs +18.9%); T=24: 둘 다 통과(+8.6% vs +7.5%, +22.6% vs +22.3%).
- 보고만: 백본 T=24 vs T=1 — serial_flif −0.62%(10/12, 에너지 14배), serial_lif −0.24%(7/12), hippo_only −2.86%(12/12, 27배); serial_lif@Tb1 vs hippo_only@Tb1 +0.70%(1/12); serial_flif(T=24) vs hippo_only(T=24) +2.33%(0/12, 에너지 0.70배); serial_flif@Tb1 vs 병렬 base +2.26%, vs 병렬 neo_flif_both +2.53%.

### 마지막 가중치 기준 (`summary_lastweights.md`, 보고용)
Q1 평균 +1.30%(0/12), Q2 +4.75%(0/12, 에너지 0.049배), **Q3 T=1 평균 +0.04%(1/4칸), T=24 −0.12%(2/4칸) — 둘 다 불통과**. 즉 Q3의 f-LIF 이점은 가중치 선택(최고 검증 epoch)에 의존한다.

### 학습 양상
직렬 모델은 **매우 일찍 과적합**한다: 최고 검증 epoch가 serial_flif@Tb1 12개 중 9개가 0–2, serial_lif@Tb1 대부분 0–1. 학습 손실은 계속 내려가는데 검증 오차는 1 epoch 전후에서 최저가 된다(예: ETTh2 H96 seed 42). 같은 백본 T=1의 평범한 임베딩(hippo_only@Tb1)은 최고 epoch 1–20(중앙값 약 4). 학습률 감소는 직렬 대부분 2회(조기 종료 전 학습률이 줄어들 여유가 거의 없음).

### 해석
1. **정렬 임베딩이 시간 표현을 개선한다는 근거는 없다.** 같은 백본 1스텝에서 f-LIF 정렬 임베딩은 패치마다 독립인 평범한 임베딩과 같고(Q1 +0.01%), LIF 정렬 임베딩은 오히려 나쁘다(+0.70%). 백본 24스텝에서도 직렬은 Hippo-only보다 +2.3% 나쁘다. 즉 "토큰 시간 = 스파이킹 시간" 정렬로 과거 패치를 섞어 넣는 것이, 각 패치를 그대로 넣는 것보다 백본에 유리하지 않았다(Phase A의 "신피질 내용이 쓰이지 않음"과 같은 방향).
2. **에너지는 반복 축 제거에서 크게 줄고, 대가는 약 3%다.** 백본 1스텝은 창당 2.2–3.7 µJ(2차 Hippo-only 58.7–72.8 µJ의 약 5%)이고 MSE는 +2.9–3.0%(직렬 f-LIF와 평범한 임베딩 모두). 이 절감은 정렬 임베딩이 아니라 반복 축 제거에서 나온다(hippo_only@Tb1이 가장 적은 에너지).
3. **f-LIF는 직렬에서 LIF보다 낫다(Q3)** — 2차의 "앞 2패치 목표에서 f-LIF > LIF" 단서와 같은 방향이지만, 마지막 가중치로는 사라지고 f-LIF 정렬 임베딩도 평범한 임베딩을 넘지 못한다. 직렬 LIF가 정렬로 잃는 것을 f-LIF가 되찾는 정도로 읽는 것이 보수적이다.
4. **직렬 모델의 조기 과적합**은 정렬 임베딩(과거 패치를 누적한 입력)이 학습 구간에 빨리 맞춰진다는 뜻일 수 있다. 학습률·정규화를 조정하면 달라질 수 있으나 이번 사전 고정 범위 밖이다.

### 한계·not run
3 seed·2 데이터셋의 방향 판정(검정 없음). 백본 T ∈ {2, 4} not run(Phase A에서 재학습 없이 4스텝이면 +0.5% 이내였다). 직렬 학습률·정규화 조정 not run. 미래 구간 미사용. 의도 2(창 너머 기억)는 이번 범위 밖.

### 결론과 다음
**확증 후보 없음**(Q1·Q2 불통과). 다음 선택지(사용자 결정 필요): (a) **백본 T = 2–4의 평범한 임베딩/직렬** — Phase A·B를 합치면 반복 축을 1이 아니라 2–4로 줄일 때 정확도 손실 없이 에너지를 크게 줄일 가능성이 가장 높다(정렬 임베딩과 무관한 기여); (b) 직렬 정렬 임베딩의 조기 과적합 대응(학습률 낮춤·warmup·dropout) 뒤 Q1 재시험; (c) 의도 2(창 너머 장기 기억) 모듈 설계로 이동 — 이번 결과는 "창 안 정렬"만으로는 이점이 없음을 보였다.

**commit·태그:** 이 항목·결과를 담은 commit을 push하고 annotated tag `exp/nsmt-flif-neo-20261017`(3단계 Phase B 탐색 완료)을 붙인다.

---

## 2026-10-18 17:24 KST — 4단계 설계: 신피질 경로 단독 예측기 — 원 시계열 한 스텝 = 스파이킹 한 스텝 정렬의 효과 (실행 전)

**사용자 요청 (10-18):** 우선 신피질 경로만 떼어 추가 검증·실험을 한다. (1) 지금은 패치 단위로 연산하지만 **원 시계열의 진짜 한 스텝을 스파이킹 뉴런의 한 스텝에 대응**시킨다. (2) 신피질 경로에서 T = T′ 정렬의 효과를 순수하게 보기 위해 **보조 손실을 과제 손실로 바꾸고 해마 없이 신피질 경로 단독으로 예측**해, 정렬이 예측 과제에 어떤 효과를 내는지 확인한다. 브랜치 `exp/nsmt-flif-neo`, 이전 commit `519ddff04`(3단계 태그 `exp/nsmt-flif-neo-20261017`).

**사용자 결정 (10-18):** (1) 비정렬 대조군 = **토큰 독립·반복** — 각 스텝(또는 패치)을 독립 표본으로 보고 같은 입력을 T_s = 4번 반복(해마 방식), 뉴런 상태가 시계열 시간을 따라 이어지지 않음; (2) 예측 읽기 = **펼친 head와 마지막 스텝 막전위 둘 다**(마지막 스텝 읽기는 정렬 켜짐에서만); (3) 단위 = **step(T = 96)과 patch(T = 24) 둘 다**; (4) 기준선 = **선형 기준선(창 정규화 + Linear(96 → H)) 추가**, Hippo-only·논문 원본은 2차 결과 재사용.

**모델 `NeoForecaster`(새 파일 `nsmt_flif_v1/forecasting/neo_only.py`):** 창 정규화된 입력 [B, L, C] → 채널 독립 [B·C, L] → 토큰(step: 한 스텝, P = 1, 96개; patch: P = 8·stride 4·끝 패딩, 24개) → 기존 신피질 부품(`ours.Embedding` Linear–BN–뉴런, `ours.TemporalBlock` Spiking MLP × M; 뉴런 spikeDE LIF / f-LIF). **정렬 켜짐:** 토큰 축 = 스파이킹 시간 축(상태가 시계열 시간을 따라 이어짐). **정렬 꺼짐:** 토큰을 배치로 접고 T_s = 4번 반복, 반복 축 평균. **head:** `flat` = 토큰별 스파이크 [토큰 수 × D]를 펼쳐 Linear → H(NSMT head와 같은 방식); `last` = 마지막 층 뉴런의 마지막 스텝 막전위 [D]를 Linear → H(뉴런이 과거를 기억해야만 예측 가능; 막전위는 `neo_only.py`의 얇은 하위 클래스가 `integrate`의 상태를 기울기와 함께 보관, 공유 `neurons.py`는 불변). 창 정규화 되돌림 → [B, H, C]. 손실은 L1 과제 손실만(보조 손실·디코더·해마 없음). 선형 기준선 `LinearForecaster`(같은 파일).

**실험 설계:** 단위 {step, patch} × 뉴런 {LIF, f-LIF(α 0.5, τ 2, pred, h 1, 전체 이력)} × 구성 {정렬 켜짐 + flat, 정렬 꺼짐 + flat, 정렬 켜짐 + last} = 12 조건 × ETTh2/ETTm2 × H96/H720 × seed 42/7/2026 = 144 run + 선형 12 run = **156 run**. 이름 `neo_<unit>_<on|off>_<flat|last>_<lif|flif>`, `linear`. 하이퍼파라미터는 논문 Table 13의 D·M·lr·bs, 학습 규칙은 2차와 같음(patience 10, 최고 검증 가중치로 판정, 마지막 가중치는 보고). suite `flif-v1-s4`. 기억 측정은 정렬 켜짐 조건만(토큰 단위). 미래 구간 미사용.

**판정 (본 학습 전 사전 고정 항목에서 확정; 같은 seed끼리 짝, 셀 = 데이터 × H 4칸):** R1 정렬 효과 — 같은 단위·뉴런·flat에서 켜짐 vs 꺼짐, 4칸 중 ≥ 3칸 낮고 평균 ≤ −1%; R2 단위 효과 — 정렬 켜짐에서 step vs patch, 같은 기준; R3 뉴런 효과 — 정렬 켜짐에서 f-LIF vs LIF, ≥ 3칸·평균 ≤ −0.5%; R4(보고) — 신피질 단독 조건 vs hippo_only(T = 24)·병렬 base·선형 기준선, last vs flat, 에너지.

**점검 계획:** 기존 c1–c9 유지 + c10(출력 모양·모듈 구성, 정렬 켜짐의 인과성과 정렬 꺼짐의 토큰 독립성을 smoke로 학습된 모델에서 — 발화율이 낮으면 실패, LIF = f-LIF(α=1) 비트 동일). smoke 13개(12 조건 + 선형)로 run당 시간을 재서 본 학습 예상 시간을 사전 고정 항목에 적는다.

---

## 2026-10-18 17:33 KST — 4단계 구현·점검 52/52·smoke 13/13, 본 학습 사전 고정 (156 run) — tmux에서 시작

**코드 (`nsmt_flif_v1/forecasting/`):** `neo_only.py`(`NeoForecaster`, `LinearForecaster`; 마지막 막전위는 `LIFKeep`/`FLIFKeep` 하위 클래스가 `integrate`의 상태를 기울기와 함께 보관, `neurons.py` 불변), `model.py`(`LOAD_MODEL['neoOnly']`, `['linear']`, 체크포인트 이름 선택 지원), `config.py`(`--neo_unit {step,patch}`, `--align {on,off}`, `--align_Ts`(기본 4), `--neo_readout {flat,last}`; 부적절한 조합 거부; 실행 폴더 이름에 짧은 접미사 `+no+<s|p><1|0><f|l>[T_s]+<neuron>-a<α>` — 최악 231바이트), `scripts/run_ett.sh`(`COND=neo_<unit>_<on|off>_<flat|last>_<lif|flif>`, `COND=linear`), `ett_future.py`·`neo_memory_probe.py`(실행 설정의 모델로 로드; 이전에는 `myModel` 고정), `make_post_queues.py --only`(정렬 켜짐 run만 기억 측정), `summarize_flif.py --stage4`(R1–R4), `scripts/{make_queue_s4.sh, stage4_driver.sh}`.

**점검 (`results/checks/check_flif.txt`, 52/52):** c1–c9 그대로. **c10** — 12가지 구성의 출력 [B, H, C]와 모듈 구성(신피질·head, patch는 끝 패딩 포함); LIF = f-LIF(α=1): flat은 예측 비트 동일, last는 스파이크 비트 동일·예측 차이 1.2e-7(막전위의 float32 합산 순서 차이; 처음에는 비트 동일을 요구해 실패했고, 측정으로 원인을 확인한 뒤 기준을 고침); smoke로 1 epoch 학습한 모델에서 **정렬 켜짐 인과성**(step: 스텝 j 이후를 바꾸면 j 이전 출력 불변, 토큰 j 스파이크 16% 변화, 발화율 0.049; patch: 22%, 0.097)과 **정렬 꺼짐 토큰 독립성**(토큰 j를 바꾸면 그 토큰만 바뀌고 나머지 95/23개 토큰은 불변). 계획의 발화율 하한 0.05는 0.02로 낮췄다 — step 모델의 1 epoch 발화율이 0.049였고, "토큰 j에서 스파이크가 실제로 바뀜"이 빈 통과를 막는 조건이기 때문이다(판단).

**smoke (`results/smoke-s4-20261018/raw.txt`, ETTh1 H96 seed 42, 1 epoch, GPU당 2개; 실험 아님):**

| 조건 | test MSE | 총 에너지 µJ | 신피질 발화율 | 파라미터 | 시간 s |
|---|---|---|---|---|---|
| neo_step_on_flat_flif / lif | 0.443225 / 0.421125 | 3.28 / 1.62 | 0.052 / 0.062 | 297,184 | 99 / 59 |
| neo_step_on_last_flif / lif | 0.674608 / 0.656838 | 3.30 / 1.64 | 0.065 / 0.079 | 5,344 | 92 / 58 |
| neo_step_off_flat_flif / lif | 0.433719 / 0.448674 | 15.70 / 15.59 | 0.052 / 0.068 | 297,184 | 27 / 21 |
| neo_patch_on_flat_flif / lif | 0.423056 / 0.422615 | 0.84 / 0.59 | 0.068 / 0.062 | 76,224 | 34 / 29 |
| neo_patch_on_last_flif / lif | 0.651695 / 0.654477 | 0.89 / 0.65 | 0.054 / 0.053 | 5,568 | 34 / 31 |
| neo_patch_off_flat_flif / lif | 0.405105 / 0.405095 | 4.67 / 4.63 | 0.078 / 0.091 | 76,224 | 22 / 20 |
| linear | 0.429054 | 0.30 | – | 9,312 | 11 |

13개 모두 학습·최고 검증 test·에너지 열까지 지난다. 처음 smoke에서 선형 기준선이 `train.py`가 읽는 `pred_len` 속성이 없어 실패했고(SKIP 줄이 `raw.txt`에 남아 있음), 속성을 추가해 다시 돌렸다. 1 epoch MSE는 판단 근거가 아니다.

**본 학습 (사전 고정):** suite `flif-v1-s4`, `scripts/make_queue_s4.sh`(156줄, 느린 조건 먼저) — 12 조건 + linear × ETTh2/ETTm2 × H96/H720 × seed 42/7/2026. 학습 규칙은 2차와 같음(patience 10, 최대 epoch 논문 값, 판정은 최고 검증 가중치, 마지막 가중치는 `summary_lastweights.md`). 참조는 2차 `flif-v1-s2p10`(hippo_only T = 24, 병렬 base). 실행: tmux `flif_s4`, `scripts/stage4_driver.sh 0,1,2,3,0,1,2,3,0,1,2,3`(**GPU당 3개** — 가장 느린 step·정렬 켜짐·f-LIF가 파이썬 루프 위주라 동시 실행으로 처리량을 늘림; 결과는 결정적이라 동시 실행과 무관), 학습 → 정렬 켜짐 run의 기억 측정(최고 검증 가중치, 512창) → 요약. **예상 10–14시간**(smoke 기준 ETTm2에서 step·f-LIF는 epoch당 약 5분).

**판정 (고정; 같은 seed끼리 짝, 셀 = 데이터 × H 4칸):**
- **R1 정렬 효과:** 같은 단위·뉴런·flat에서 정렬 켜짐 vs 꺼짐 — 4칸 중 ≥ 3칸 낮고 평균 ≤ −1% (단위·뉴런 4가지 각각).
- **R2 단위 효과:** 정렬 켜짐에서 step(T = 96) vs patch(T = 24) — 같은 기준(뉴런·head 4가지 각각).
- **R3 뉴런 효과:** 정렬 켜짐에서 f-LIF vs LIF — 4칸 중 ≥ 3칸 낮고 평균 ≤ −0.5% (단위·head 4가지 각각).
- **R4 (보고만):** 각 신피질 단독 조건 vs hippo_only(T = 24)·병렬 base·linear; last vs flat; 에너지; 기억 측정(마지막 12토큰만 남길 때 오차 증가).
- 탐색 단계이므로 통과는 확증이 아니다.

**commit:** 이 항목·코드·점검·smoke 결과를 담은 commit을 push한 뒤 시작한다. 결과는 다음 항목.

---

## 2026-10-18 23:16 KST — 4단계 결과 (신피질 단독 예측, 156 run, 최고 검증 가중치): **R1 정렬 효과 4/4 불통과(정렬이 오히려 평균 +0.5–2.0% 나쁨), R2 단위 효과 4/4 불통과, R3 f-LIF > LIF 2/4 통과(마지막 가중치로는 0/4)**; 마지막 막전위 읽기의 기억 측정에서 처음으로 f-LIF의 긴 기억이 뚜렷이 보임 — 태그 `exp/nsmt-flif-neo-20261018`

**구분:** 사전 고정한 4단계 **탐색**의 완료 기록. 코드·사전 고정 commit `51d019695`. 이 항목이 속한 commit은 태그 `exp/nsmt-flif-neo-20261018`로 식별한다.

**실행:** tmux `flif_s4`, `scripts/stage4_driver.sh 0,1,2,3,0,1,2,3,0,1,2,3`(GPU당 3개): 학습 17:33–22:50 **156/156 완료·실패 0**(가장 느린 step·정렬 켜짐·f-LIF는 ETTm2 H96에서 run당 135–150분), 기억 측정 96개(정렬 켜짐 run, 최고 검증 가중치, 512창) 22:50–23:09, 요약 `results/flif-v1-s4/{summary.md, summary_lastweights.md}`(2차 suite와 합쳐 계산). 산출물: `results/flif-v1-s4/{raw.txt, memory_probe_bestval/}`, 실행별 `log/flif-v1-s4/<DS>/261018/<run>/log/…`(Git), checkpoint·stdout(로컬).

### test MSE (최고 검증 가중치, seed 3개 평균; ETTh2 H96 / H720 / ETTm2 H96 / H720)

| 조건 | f-LIF | LIF |
|---|---|---|
| step·정렬 켜짐·flat | 0.301492 / 0.442... ※ / 0.179136 / 0.402691 | 0.300828 / 0.440226 / 0.178907 / 0.402521 |
| step·정렬 꺼짐·flat | 0.301396 / 0.433858 / 0.179241 / 0.402243 | 0.301024 / 0.430462 / 0.179874 / 0.400390 |
| patch·정렬 켜짐·flat | 0.299605 / 0.422832 / 0.179561 / 0.406597 | 0.305840 / 0.430607 / 0.180011 / 0.406065 |
| patch·정렬 꺼짐·flat | 0.297340 / 0.417887 / 0.178640 / 0.407099 | **0.294910 / 0.417563 / 0.177503** / 0.407751 |
| step·정렬 켜짐·last | 0.320300 / 0.443... ※ / 0.199075 / 0.416518 | 0.319469 / 0.443844 / 0.203816 / 0.421695 |
| patch·정렬 켜짐·last | 0.318274 / 0.443448 / 0.197733 / 0.416711 | 0.314976 / 0.437... ※ / 0.192977 / 0.415652 |
| 선형 기준선 | **0.291515** / 0.422... ※ / 0.182854 / 0.407828 | |
| (참조) hippo_only T = 24, 2차 | 0.285518 / 0.411191 / 0.173930 / 0.401147 | |

※ 22:45 브리핑 시점에 seed 2개 이하였던 칸; 최종 3-seed 값은 `summary.md`의 표가 기준이다(판정은 최종 값으로 계산됨).

### 사전 고정 판정 (`summary.md` "stage 4")
- **R1 정렬 효과 — 4/4 불통과.** 정렬 켜짐 vs 꺼짐(flat): step·LIF +0.55%(2/4칸 낮음), step·f-LIF +0.49%(1/4), patch·LIF +1.96%(1/4), patch·f-LIF +0.58%(1/4). **정렬을 켠 쪽이 평균적으로 나쁘고**, 특히 ETTh2 H720(step +1.9–2.3%, patch +1.2–3.1%)과 patch·ETTh2 H96(+3.7% LIF)에서 그렇다. 에너지는 정렬 켜짐이 0.06–0.11배(꺼짐은 토큰마다 4번 반복).
- **R2 단위 효과 — 4/4 불통과.** step(T = 96) vs patch(T = 24): LIF·flat −0.22%(3/4칸, 평균 기준 미달), f-LIF·flat +0.99%, LIF·last +2.48%, f-LIF·last +0.32%. step은 에너지 2.1–3.5배.
- **R3 f-LIF vs LIF — 2/4 통과(최고 검증 가중치).** step·last −0.84%(3/4칸; ETTm2 H96 −2.33%, H720 −1.23%), patch·flat −0.99%(3/4칸; ETTh2 −2.0/−1.8%) **통과**; step·flat +0.20%(0/4), patch·last +1.27%(0/4) 불통과. **마지막 가중치로는 4가지 모두 불통과**(step·last −0.57%이지만 2/4칸, patch·flat +0.04%).
- **R4 (보고):** 신피질 단독 최선(patch·정렬 꺼짐·LIF)이 hippo_only보다 +2.13%(0/12), 논문 base보다 +1.44%, 선형보다 −0.64%(8/12). 정렬 켜짐 flat은 hippo_only보다 +3.1–4.1%, last는 +7.8–10.5%. last vs flat +3.6–6.4%. 선형 기준선 자체는 hippo_only보다 +2.81%.

### 기억 측정 — 마지막 12토큰만 남길 때 오차 증가 (최고 검증 가중치)

| 조건 | ETTh2 | ETTm2 |
|---|---|---|
| step·정렬 켜짐·last f-LIF / LIF | **+0.3% / +0.0%** | **+11.1% / +0.0%** |
| patch·정렬 켜짐·last f-LIF / LIF | +0.3% / −0.0% | +3.6% / +0.0% |
| step·정렬 켜짐·flat f-LIF / LIF | +15.5% / +17.8% | +38.5% / +42.8% |
| patch·정렬 켜짐·flat f-LIF / LIF | +7.6% / +6.6% | +19.8% / +18.0% |

**마지막 막전위 읽기에서 LIF는 12토큰보다 오래된 입력을 지워도 예측이 전혀 바뀌지 않는다(0.0%)** — τ = 2 LIF의 막전위는 12스텝이면 0.5¹² ≈ 2×10⁻⁴로 줄어 사실상 잊는다. **f-LIF는 ETTm2에서 +11.1%(step)·+3.6%(patch) 바뀐다** — 마지막 막전위가 12토큰보다 오래된 입력을 실제로 담고 예측에 쓴다. 1–4단계를 통틀어 **f-LIF의 긴 기억이 뉴런 수준에서 예측에 쓰인다는 첫 뚜렷한 증거**다(step·last의 R3 통과와 같은 방향). 단 ETTh2에서는 둘 다 거의 0이고, flat head는 토큰을 직접 읽으므로 두 뉴런 모두 크게 바뀐다(이 경우 측정은 기억이 아니라 "입력을 지운 효과").

### 해석
1. **신피질 단독 예측에서 "스파이킹 시간 = 시계열 시간" 정렬은 예측을 개선하지 않는다.** 토큰을 독립으로 처리하고(정렬 꺼짐) 위치별 head가 시간을 다루게 하는 쪽이 같거나 낫다. 원 시계열 한 스텝 단위(T = 96)로 더 촘촘히 정렬해도 마찬가지다(R2). 1–3단계 결과(해마와 결합 시 신피질 내용 미사용, 직렬 정렬 임베딩 = 평범한 임베딩)와 일관된다.
2. **f-LIF의 장점은 "뉴런이 과거를 운반해야만 하는" 설정에서만 나타난다.** 마지막 막전위만 읽을 때 f-LIF는 LIF가 완전히 잊는 12토큰 이전 정보를 쓰고(ETTm2), 예측도 LIF보다 낫다(step·last −0.84%). 그러나 그 설정 자체가 flat head보다 3–14% 나빠, 실용적 이득으로는 이어지지 않는다.
3. 정렬은 에너지 면에서는 반복(꺼짐)보다 10배 이상 싸다(0.06–0.11배) — 성능이 같다면 에너지 이점은 있다. 정렬 켜짐 flat은 창당 0.9–6.9 µJ로 hippo_only(58–73 µJ)의 2–10%이지만 MSE는 +3–4%.
4. R3의 두 통과는 마지막 가중치로는 사라진다(3단계 Q3와 같은 양상) — 가중치 선택에 민감한 약한 차이다.

### 한계·not run
3 seed·2 데이터셋의 방향 판정(검정 없음). 정렬 꺼짐의 반복 횟수는 T_s = 4 하나. last 읽기는 마지막 층 막전위 하나(여러 층·여러 스텝 읽기 not run). α 0.5 하나. 미래 구간 미사용. 창 너머 기억(의도 2)은 이번 범위 밖 — 이번 결과는 창 안 96스텝에서도 f-LIF가 LIF보다 오래 기억함을 보였으므로, 창 너머 기억 모듈(긴 저해상도 이력을 신피질 시간 축에 정렬)의 근거가 된다.

### 결론과 다음
**정렬 자체의 예측 이득은 없음(R1·R2 불통과).** f-LIF의 긴 기억은 마지막 막전위 읽기에서 처음으로 뚜렷하게 측정됐다. 다음 선택지(사용자 결정 필요): (a) **의도 2 — 창 너머 장기 기억**: 신피질 시간 축에 긴 저해상도 이력(앞선 여러 창)을 정렬해 넣고 마지막 막전위(또는 소수 스텝)로 읽어, f-LIF의 긴 기억이 창 너머 정보를 운반하는지 시험(이번 측정이 직접 근거); (b) f-LIF α 탐색(0.3–0.9)을 마지막 막전위 읽기에서 — 기억 길이와 성능의 관계; (c) 정렬 켜짐의 에너지 이점(창당 수 µJ)을 살리면서 성능 손실(+3–4%)을 줄이는 head·학습 조정.

**같은 commit:** 감사 85(10-17 23:11, 기존 판정 확인 기록)·86(10-18 17:41, 이 계열의 HEAD 변경 확인) — 둘 다 새 판단 근거 없음, 조치 없음(감사 문서 반영).

---

## 2026-10-18 23:17 KST — 정정: 4단계 결과 항목의 표에 남은 임시 값 4칸 (태그 `exp/nsmt-flif-neo-20261018`가 가리키는 commit `b01f9bdf6`)

직전 항목의 "test MSE" 표에 22:45 브리핑 시점의 임시 값(seed 2개 이하, "※" 표시) 4칸이 남은 채로 commit·태그·push됐다. 임시 값을 최종 값으로 바꾸는 스크립트가 표가 아니라 마지막 소제목 이후 부분만 처리했기 때문이다(내 실수; 스크립트는 그 부분에 남은 표시가 없다는 것만 확인했다). **판정(R1–R4)과 `results/flif-v1-s4/summary.md`는 처음부터 최종 3-seed 값으로 계산됐으므로 결론은 바뀌지 않는다.** 공개한 태그는 옮기지 않고 이 정정으로 바로잡는다. 최종 값(ETTh2 H720 칸, seed 3개 평균 ± 표준편차):

| 조건 | 직전 항목의 표시 | 최종 값 |
|---|---|---|
| step·정렬 켜짐·flat·f-LIF | 0.442... ※ | **0.442058** ±0.0057 |
| step·정렬 켜짐·last·f-LIF | 0.443... ※ | **0.443530** ±0.0001 |
| patch·정렬 켜짐·last·LIF | 0.437... ※ | **0.437656** ±0.0005 |
| 선형 기준선 | 0.422... ※ | **0.420805** ±0.0015 |

나머지 칸은 모두 최종 값이었다. 재발 방지: 기록 항목을 commit하기 전에 파일 전체에서 임시 표시(※, "...", "?")를 검사한다.

---

## 2026-10-19 02:05 KST — 5단계 설계·구현·사전 고정: MIT-BIH 분류에서 신피질 경로 LIF → f-LIF (원저자 ECG 실험 코드 기준), 점검 10/10, smoke 7/7 — 21 run tmux에서 시작

**요청 (2026-10-19, 사용자):** "분류 과제에 대해서는 정렬+f-LIF 네트워크가 어떤 정보를 학습하는 지 궁금해. MIT-BIH 데이터셋에 대하여 `nsmt_base/classification/model.py`의 구조와 학습 프레임워크에서 Neocortex path의 LIF 뉴런만 f-LIF 뉴런으로 교체하고 분류 성능 변화를 관찰해줘." 작업 중 추가 요청: "`nsmt_base/classification/ECG` 경로를 참고하여 너가 수정한 분류 코드를 검토해." (사용자가 01:46에 원저자의 MIT-BIH 실험 코드 `nsmt_base/classification/ECG/`를 추가함.)

**브랜치·기준:** `exp/nsmt-flif-neo`(계속), 기준 commit `f28772380`.

**코드 검토 결과 → 기준 변경.** 처음 수정본은 `nsmt_base/classification/`(UEA용) 모델·학습 코드를 복사해 만들었다. ECG 코드와 비교하니 MIT-BIH 실험과 다른 점이 많아 **폐기하고 ECG 코드 기준으로 다시 만들었다**(폐기본은 학습에 쓰지 않았고 Git에도 넣지 않았다).

| 항목 | UEA 분류 코드 (처음 수정본) | ECG 코드 (채택) |
|---|---|---|
| 패치 임베딩 | 원 신호에 Conv1d(kernel 16, stride 8), 토큰 22개 | 끝을 복제 패딩한 패치(32, stride 16)에 Linear, 토큰 11개 |
| 신피질 임베딩 뉴런의 시간 축 | 같은 입력을 T번 반복한 축 — **정렬 아님** | 토큰 축 — **정렬** (예측 모델과 같음) |
| 신피질 층 | `SpikLinearLayer`(다중 스텝 LIF) | `NeoLayer`(Linear → 전체 시퀀스 BN → 한 스텝씩 도는 LIF; 수동형) |
| 보조 손실의 목표 | 저역 신호의 BN 임베딩 | 원 패치의 저역 성분 (예측 모델과 같음) |
| 입력 모양 | [B, 187, 1] | [B, 1, 187] (ECG 로더) |
| 과제 손실 | CE + label smoothing 0.1 | 클래스 가중 CE(CAL, 논문 App. B.4) |
| 학습 | 내가 임의로 정한 검증 분할·조기 종료·lr 1e-3 | 30 epoch cosine, 마지막 가중치로 test |
| 신피질 제거 대조군 | 없음 | `--no_neo` (교차 주의도 함께 제거) |

**데이터:** Kachuee 5-class 심박(N/S/V/F/Q, 187 샘플, 125 Hz). 논문 App. B.1의 70:30 무작위 분할 파일을 쓴다.
- `5class_mitbih_train_7_187.csv`: 76,612개, 클래스별 [63412, 1946, 5065, 562, 5627], md5 `c2009b3556e23974e5294d6e36513289`.
- `5class_mitbih_test_3_187.csv`: 32,833개, [27176, 833, 2171, 241, 2412], md5 `ccc7248980e020ea127ca8b3d923c309`.
- 위치: `nsmt_base/classification/ECG/data/`(로컬, 3.8 GB, 이번에 `.gitignore`로 제외). 같은 md5의 사본이 `/home/yschoi/CLS_spiking_transformer/data/`에도 있다.
- 전처리 없음(원값 [0, 1]), header 없음, 결측 없음. train과 test에 똑같은 심박은 0개(확인). 무작위 심박 분할이라 같은 환자의 심박이 양쪽에 섞인다(논문과 같음; 환자 간 일반화는 재지 않는다).

**학습 (ECG 프레임워크 그대로; `ECG/scripts/_run_mitbih_3seed.sh`의 설정):**
- 30 epoch, cosine 일정(timm, 6 epoch warmup), bs 512, lr 0.01, AdamW wd 1e-4.
- emb 384, head 8, mlp_ratio 1, 층 1, 신피질 층 2, patch 32(stride 16, 토큰 11개 = T), keep_ratio 0.15, 보조 손실 비중 0.1, 신피질로 가는 과제 기울기 몫 0.5.
- 클래스 가중 CE: 가중치 [1.0, 3.4839, 2.5273, 4.7259, 2.4221] (= max(1, log(n_max / n_c))).
- 매 epoch의 "val" 열은 test 파일로 계산된다(원 코드). cosine 일정이라 조기 종료·가중치 선택이 없어서 test로 고르는 것은 없다. **마지막 epoch 가중치로 test** 하므로 별도 검증 분할은 필요 없다.

**코드 (`nsmt_flif_v1/classification/`; `ECG/`의 model·train·test·config·load_model·dataloader·utils·layers·positional을 복사해 수정):**
- `layers.py`: `SpikLinearLayer(neuron=)`, `NeoNeuron`, `build_neo_neuron` (예측 코드와 같은 어댑터).
- `model.py`: `Embedding(neuron=)`로 encoding_neo의 LIF 교체(토큰 축), `NeoLayer(neuron=)`(수동형만, 같은 BN 입력을 spikeDE 뉴런이 토큰 축으로 적분), `TemporalBlock(neuron_fn=)`, `Block(self_kv=)`(hippo_only), myModel 옵션 `neo_neuron/neo_where/neo_alpha/fde_*/hippo_only`(맞지 않는 조합은 거부).
- `load_model.py`: ECG 코드의 `from ab4 import myModelAB4` 제거(ab4 절제 모델은 복사하지 않음), 옵션 전달.
- `config.py`: `--neo_neuron --neo_where --neo_alpha --fde_method --fde_step --fde_memory --hippo_only`. 실행 폴더의 seed+조건 단계 이름에 접미사(기본 이름 불변).
- `train.py`: 디버그 스위치(anomaly detection, `CUDA_LAUNCH_BLOCKING`)를 기본으로 끔(`NSMT_DEBUG=1`이면 켬). 결과가 비트 동일함을 c2로 확인했고, 속도 때문이다.
- `test.py`: 클래스별 F1, confusion matrix, 뉴런 고유 에너지 열, syops에 spikeDE 뉴런 hook. 원래 열 뒤에 덧붙인다.
- `energy.py`: 예측 코드 복사 + 한 스텝씩 도는 LIF(ECG `NeoLayer`) 기록. 이것이 없으면 base의 신피질 뉴런 에너지가 빠진다.
- `neurons.py`: 예측 코드와 바이트 동일(c8).
- 새 파일: `check_cls.py`, `cls_analysis.py`, `summarize_cls.py`, `scripts/{run_mitbih.sh, make_queue_cls.sh, run_analysis.sh, cls_driver.sh}`.
- 참조 코드 `nsmt_base/classification/ECG/`(사용자가 추가한 원저자 코드 27개 파일)를 이 commit에 함께 기록한다. 데이터 폴더는 제외.

**점검 10/10 (`nsmt_flif_v1/classification/results/checks/check_cls.txt`):**
- c1 기본값 = ECG 코드: 초기 파라미터 62개, AdamW 3스텝 뒤 파라미터, 학습·test 출력이 모두 비트 동일.
- c2 디버그 스위치를 꺼도 base·f-LIF 모두 파라미터와 test 출력이 비트 동일.
- c3 spikeDE LIF = f-LIF(α = 1): 신피질 스파이크 비트 동일(발화율 0.30 / 0.28), 출력 차이 0.
- c4 해마 뉴런은 모든 조건에서 spikingjelly LIF. 신피질 뉴런은 조건대로 바뀐다.
- c5 신피질의 토큰 축 인과성(세 뉴런 모두): 토큰 j를 바꾸면 j 이전 출력은 그대로이고 j 이후는 바뀐다. 초기 가중치에서는 신피질이 발화하지 않아 검사가 성립하지 않으므로, 고정 난수로 입력 가중치를 키워 발화율 0.13–0.24에서 검사했다.
- c6 no_neo·hippo_only: 신피질 모듈에 기울기 없음. hippo_only만 교차 주의를 쓴다.
- c7 ECG 로더: 배치 [512, 1, 187], 클래스 수와 CAL 가중치가 파일과 일치.
- c8 `neurons.py` 바이트 동일. c9 에너지 기록기: 모든 뉴런이 T = 11로 기록됨(신피질 3층, 해마 8층). c10 7개 조건의 실행 폴더가 서로 다르고, base 이름은 ECG 규칙과 같음.
- 처음 실행에서 c3·c5는 신피질이 발화하지 않아 검사가 성립하지 않았고, c6은 비교 조건(출력이 달라야 함)이 초기 가중치에서 성립하지 않아 실패했다. 발화하도록 가중치를 다시 뽑고, c6은 "교차 주의 파라미터가 기울기를 받는가"로 바꿨다.

**smoke (`results/smoke-cls-20261019/raw.txt`, seed 42, 1 epoch, GPU당 2개; 실험 아님, 1 epoch 값은 판단 근거가 아님):**

| 조건 | 정확도 | macro-F1 | 시냅스 µJ | 총 µJ (FFT / 직접 합) | 신피질 발화율 | 시간 s |
|---|---|---|---|---|---|---|
| base (ECG 모델) | 0.879176 | 0.448076 | 16.08 | 20.53 / 20.53 | 0.0862 | 61 |
| neo_lif_both | 0.932842 | 0.600014 | 16.14 | 20.61 / 20.61 | 0.0839 | 63 |
| neo_flif_both | 0.923918 | 0.614389 | 15.99 | 20.56 / 20.80 | 0.0756 | 70 |
| neo_lif_mlp | 0.911857 | 0.543118 | 16.16 | 20.65 / 20.65 | 0.0731 | 41 |
| neo_flif_mlp | 0.940974 | 0.662233 | 15.95 | 20.50 / 20.65 | 0.0879 | 54 |
| no_neo | 0.918801 | 0.566698 | 10.87 | 12.52 / 12.52 | – | 33 |
| hippo_only | 0.871075 | 0.430451 | 18.18 | 22.52 / 22.52 | – | 53 |

7개 모두 학습·test·에너지 열까지 지난다. 분석 스크립트도 smoke 체크포인트 3개(base, neo_flif_both, no_neo)로 시험했다(`results/smoke-cls-20261019/analysis/`). 분석용 forward가 모델의 출력과 같은지 매번 확인하고, 다르면 멈춘다.

**본 학습 (사전 고정):** suite `flif-v1-cls`, 21 run = 7 조건 × seed 42 / 7 / 2026 (`scripts/make_queue_cls.sh`).

| 조건 | 신피질 뉴런 |
|---|---|
| base | ECG 모델 그대로: 임베딩은 spikingjelly 다중 스텝 LIF, `NeoLayer` 2개는 한 스텝 LIF |
| neo_lif_both / neo_flif_both | spikeDE LIF / f-LIF(α 0.5, τ 2, pred, h 1, 전체 이력)를 임베딩과 `NeoLayer` 2개 모두에 |
| neo_lif_mlp / neo_flif_mlp | `NeoLayer` 2개만 교체 (임베딩은 spikingjelly LIF) — 결정 1의 두 위치 비교 |
| no_neo | ECG 프레임워크의 신피질 제거 (교차 주의 없음) |
| hippo_only | 결정 7: 신피질 제거, 교차 주의의 key/value = 해마 |

실행: tmux `flif_cls`, `scripts/cls_driver.sh 0,1,2,3,0,1,2,3`(GPU당 2개). 순서는 학습 → 21개 run 분석(`cls_analysis.py`) → 요약(`summarize_cls.py`). 예상 1–1.5시간.

**판정 (고정; test macro-F1, 마지막 가중치, 같은 seed끼리 짝):** A가 B보다 "좋음"은 평균 차이 ≥ +0.5%p이고 3 seed 중 2개 이상에서 A > B인 경우다. "나쁨"은 평균 ≤ −0.5%p이고 2개 이상에서 A < B인 경우다. 그 밖은 "뚜렷한 차이 없음", seed가 빠지면 "미완"이다.
- **Q1 (주 질문):** f-LIF vs spikeDE LIF — both, mlp 각각.
- **Q2:** f-LIF vs ECG 모델(base) — both, mlp 각각.
- **Q3 신피질 기여:** base vs no_neo, base vs hippo_only.
- 정확도는 같은 규칙으로 보고만 한다. 에너지(시냅스 + 뉴런 식 60/64, 직접 합)도 보고만 한다.
- **분석 (보고만, 미리 정의):**
  - A1 replay: 해마가 읽는 신피질 출력을 다른 심박 / 같은 클래스 심박 / 다른 클래스 심박 / 전부 0 / 전부 1로 바꿨을 때 macro-F1 변화.
  - A2 선형 probe: 원 심박, 신피질 임베딩 스파이크, 신피질 출력 스파이크, 토큰별 발화율, 해마 head 입력에서 클래스를 얼마나 읽을 수 있나.
  - A3 토큰 축 기억: 신피질 입력 토큰 하나를 다른 심박의 것으로 바꿨을 때, d 토큰 뒤 신피질 스파이크가 바뀌는 비율과 예측이 바뀌는 비율.
  - A4 클래스별·토큰별 신피질 발화율.
- **참고값:** 원저자 스크립트 주석의 수동형 신피질 macro-F1 0.9143(3 seed; 다른 서버의 결과라 로그는 확인하지 못함). 논문 Table 5는 정확도 98.45, F1 91.91, 19.05 µJ.
- 탐색 단계(3 seed)이므로 통과는 확증이 아니다.

**환경:** `snn_recall`(Python 3.10, torch 1.12.0+cu113, spikingjelly cupy), RTX A6000 4장.

**commit:** 이 항목·코드·점검·smoke 결과·ECG 참조 코드를 담은 commit을 push한 뒤 시작한다. 결과는 다음 항목.

---

## 2026-10-19 03:00 KST — 5단계 결과 (MIT-BIH, 21 run, 마지막 가중치): **Q1 f-LIF vs spikeDE LIF 뚜렷한 차이 없음(both −0.02%p, MLP만 −0.03%p), Q2 f-LIF vs ECG 모델 뚜렷한 차이 없음, Q3 신피질 기여: no_neo보다 +1.18%p 좋음(3/3), 그러나 hippo_only와는 차이 없음(−0.25%p)**; 학습된 모델은 심박마다 다른 신피질 출력을 실제로 쓴다 — 태그 `exp/nsmt-flif-neo-20261019`

**구분:** 사전 고정한 5단계 **탐색**(3 seed)의 완료 기록. 코드·사전 고정 commit `da14fc7a8`. 이 항목이 속한 commit은 태그 `exp/nsmt-flif-neo-20261019`로 식별한다.

**실행:** tmux `flif_cls`, `scripts/cls_driver.sh 0,1,2,3,0,1,2,3`(GPU당 2개). 학습 02:06–02:56 **21/21 완료·실패 0**, 모두 30 epoch(마지막 학습률 0). run당 시간은 no_neo 387–443 s, hippo_only 664–986 s, base 924–990 s, spikeDE LIF 1091–1227 s, f-LIF 1208–1498 s. 분석 21개 02:56–02:58(실패 0), 요약 `nsmt_flif_v1/classification/results/flif-v1-cls/summary.md`. 산출물은 `results/flif-v1-cls/{raw.txt, runs.csv, conditions.csv, analysis/}`와 실행별 `log/flif-v1-cls/MITBIH/261019/<run>/<seed+조건>/log/…`(Git). checkpoint·TensorBoard·stdout은 로컬이다.

### 성능과 에너지 (test 32,833개, 3 seed 평균 ± 표준편차, %)

| 조건 | 정확도 | macro-F1 | F1 S | F1 V | F1 F | 총 에너지 µJ (FFT / 직접 합) | 신피질 발화율 |
|---|---|---|---|---|---|---|---|
| ECG 모델 (base) | 98.41 ± 0.03 | 91.58 ± 0.06 | 84.31 | 95.65 | 79.89 | 23.86 / 23.86 | 0.093 |
| spikeDE LIF, both | 98.39 ± 0.03 | 91.42 ± 0.29 | 83.77 | 95.73 | 79.72 | 23.08 / 23.08 | 0.114 |
| f-LIF, both | 98.35 ± 0.07 | 91.39 ± 0.42 | 83.11 | 95.72 | 80.18 | 23.43 / 23.67 | 0.106 |
| spikeDE LIF, MLP만 | 98.38 ± 0.02 | 91.45 ± 0.23 | 83.69 | 95.72 | 79.90 | 23.73 / 23.73 | 0.114 |
| f-LIF, MLP만 | 98.39 ± 0.04 | 91.42 ± 0.20 | 84.08 | 95.60 | 79.43 | 23.50 / 23.66 | 0.107 |
| no_neo | 98.13 ± 0.07 | 90.40 ± 0.34 | 79.95 | 95.40 | 78.88 | 13.03 / 13.03 | – |
| hippo_only | 98.43 ± 0.04 | **91.83 ± 0.23** | 84.01 | 95.90 | 81.23 | 26.02 / 26.02 | – |

- **재현 확인:** base는 논문 Table 5(정확도 98.45, F1 91.91, 19.05 µJ)와 원저자 스크립트 주석(macro-F1 0.9143)에 가깝다. base의 시냅스 에너지는 19.06 ± 0.18 µJ로 논문 값과 같다(위 표의 "총 에너지"는 여기에 뉴런 고유 에너지를 더한 값).
- 클래스별 F1 Q는 모든 조건에서 98.7–98.9%다(`summary.md`).

### 사전 고정 판정 (macro-F1, 같은 seed끼리 짝, %p)

| 질문 | A − B (seed 42 / 7 / 2026) | 평균 | 판정 |
|---|---|---|---|
| Q1 f-LIF − spikeDE LIF, both | +0.19 / −0.04 / −0.22 | −0.02 | 뚜렷한 차이 없음 |
| Q1 f-LIF − spikeDE LIF, MLP만 | −0.08 / +0.25 / −0.26 | −0.03 | 뚜렷한 차이 없음 |
| Q2 f-LIF − ECG 모델, both | +0.22 / −0.51 / −0.29 | −0.19 | 뚜렷한 차이 없음 |
| Q2 f-LIF − ECG 모델, MLP만 | −0.05 / −0.10 / −0.33 | −0.16 | 뚜렷한 차이 없음 |
| Q3 ECG 모델 − no_neo | +0.86 / +1.29 / +1.40 | +1.18 | **좋음** |
| Q3 ECG 모델 − hippo_only | −0.42 / −0.25 / −0.07 | −0.25 | 뚜렷한 차이 없음 |

- 정확도로는 6개 비교 모두 "뚜렷한 차이 없음"이다(base − no_neo +0.28%p로 기준 0.5%p 미달).
- **해석:** no_neo는 신피질과 함께 교차 주의 층을 통째로 없앤다. 신피질 없이 교차 주의를 해마 자신에게 거는 hippo_only는 base만큼 좋다(수치상 +0.25%p). 따라서 no_neo 대비 이득은 **교차 주의 층에서 오고, 신피질이 주는 key/value는 해마가 스스로 대신할 수 있다**.
- **(참고, 사전 고정 아님) 원래 통과 기준 (2) — hippo-only 대비:** f-LIF both는 macro-F1 −0.44%p(−0.19 / −0.75 / −0.36), 총 에너지 0.90배다. f-LIF MLP만은 −0.41%p, 0.90배다. "성능이 더 좋음"도 "성능은 비슷하고 에너지가 훨씬 좋음"도 아니어서 기준 (2)를 만족하지 않는다.

### 정렬 + f-LIF 신피질이 학습한 것 (분석, 사전 정의, 보고만; `summary.md` A1–A4)

**A1 — 해마는 심박마다 다른 신피질 출력을 실제로 쓴다.** 학습된 모델에서 해마가 읽는 신피질 출력을 바꾸면 macro-F1이 크게 떨어진다 (%p, 3 seed 평균).

| 조건 | 다른 심박의 것 | 같은 클래스 심박 | 다른 클래스 심박 | 전부 0 | 전부 1 |
|---|---|---|---|---|---|
| ECG 모델 | −32.95 | −13.50 | −38.16 | −66.62 | −64.05 |
| spikeDE LIF, both | −18.55 | −10.51 | −20.33 | −52.17 | −60.99 |
| f-LIF, both | −20.05 | −11.80 | −21.39 | −42.75 | −51.68 |
| spikeDE LIF, MLP만 | −23.69 | −12.13 | −29.42 | −73.21 | −43.39 |
| f-LIF, MLP만 | −23.15 | −12.55 | −26.86 | −54.60 | −45.39 |

- 예측 과제(Phase A)에서는 다른 창의 신피질 출력으로 바꿔도 ±0.3%였다. 분류에서는 반대로 심박마다 다른 내용이 쓰인다.
- 같은 클래스 심박의 것으로 바꾸면 덜 떨어진다(−10.5 ~ −13.5). 다른 클래스 심박의 것이면 더 떨어진다(−20 ~ −38). 신피질 출력에는 클래스 정보와 심박 고유의 정보가 함께 담겨 있다.
- 다만 hippo_only가 같은 성능을 내므로, 이 의존은 "신피질이 없으면 안 된다"가 아니라 "신피질과 함께 학습한 모델이 신피질에 의존하게 되었다"는 뜻이다.

**A2 — 신피질 스파이크에서 클래스를 선형으로 읽는 정도** (클래스 균형 로지스틱 회귀, 학습은 클래스당 최대 1,000개, test macro-F1 %).

| 조건 | 원 심박 | 신피질 임베딩 스파이크 | 신피질 출력 스파이크 | 토큰별 발화율(11개 수) | 해마 head 입력 |
|---|---|---|---|---|---|
| ECG 모델 | 47.04 | 61.08 | 65.53 | 35.48 | 87.89 |
| spikeDE LIF, both | 47.04 | 60.08 | 60.16 | 27.84 | 87.31 |
| f-LIF, both | 47.04 | 60.80 | 62.33 | 28.42 | 87.11 |
| f-LIF, MLP만 | 47.04 | 61.41 | 64.26 | 32.15 | 88.04 |
| no_neo / hippo_only | 47.04 | – | – | – | 83.49 / 88.25 |

- 신피질 출력은 원 심박보다 클래스가 선형으로 잘 갈리는 스파이크 부호다(+13 ~ +18%p). 그러나 해마 특징(87–88%)에는 크게 못 미친다.
- 토큰별 발화율만으로는 28–35%다. 클래스 정보는 "얼마나 많이" 발화하는지보다 "어느 뉴런이" 발화하는지에 있다.
- f-LIF와 spikeDE LIF의 차이는 +2.2%p(both)로 작다. ECG 모델의 spikingjelly LIF가 가장 높다(65.5%).
- 이 probe는 클래스 균형으로 학습해 불균형 test에서 N 클래스 오분류가 늘기 때문에 절대값이 낮다. 조건 사이 비교에만 쓴다.

**A3 — 토큰 축 기억: 한 토큰의 정보가 몇 토큰 뒤까지 남나.** 신피질 입력 토큰 하나를 다른 심박의 것으로 바꿨을 때, d 토큰 뒤 신피질 출력 스파이크가 바뀌는 비율(%, 4,096개 심박, 3 seed 평균)이다.

| 조건 | d=0 | d=1 | d=2 | d=5 | d=8 | d=10 | 토큰 0을 바꿀 때 바뀐 예측 |
|---|---|---|---|---|---|---|---|
| ECG 모델 (spikingjelly LIF) | 15.29 | 12.06 | 9.11 | 4.51 | 2.46 | 1.79 | 5.62 |
| spikeDE LIF, both | 9.28 | 9.87 | 6.52 | 1.96 | 0.92 | 0.69 | 2.34 |
| f-LIF, both | 10.88 | 12.96 | 4.13 | 2.05 | 1.26 | 1.11 | 2.47 |
| spikeDE LIF, MLP만 | 10.86 | 9.76 | 6.17 | 1.79 | 0.83 | 0.59 | 3.84 |
| f-LIF, MLP만 | 12.24 | 11.68 | 4.76 | 1.93 | 0.81 | 0.54 | 3.04 |

- 바꾼 토큰보다 앞 토큰의 스파이크는 한 번도 바뀌지 않았다(인과성; 신피질이 있는 15개 run 모두 0).
- both 배치에서 f-LIF는 d ≥ 5에서 spikeDE LIF보다 정보를 더 오래 남긴다(d = 10에서 1.11% vs 0.69%, 1.6배). 발화율(0.106 vs 0.114)로 나눠도 순서는 같다. 그러나 이 차이는 분류 성능으로 이어지지 않았다(Q1).
- MLP만 바꾼 배치에서는 f-LIF의 긴 꼬리가 없다(d = 10에서 0.54% vs 0.59%).
- ECG 모델의 spikingjelly LIF가 가장 멀리 남긴다(d = 10에서 1.79%). 발화율로 나누면 0.19로, f-LIF(0.11)·spikeDE LIF(0.06)보다 크다. 두 LIF는 리셋 방식이 다르다. spikingjelly는 0으로 되돌리고, spikeDE는 θ·s/τ만큼 뺀다.

**A4 — 클래스별·토큰별 신피질 출력 발화율** (%, f-LIF both, 3 seed 평균; 토큰 = 32샘플 패치, stride 16).

| 클래스 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N | 14.5 | 23.1 | 22.7 | 24.0 | 21.9 | 18.9 | 13.6 | 9.9 | 6.9 | 5.1 | 4.1 |
| S | 15.4 | 24.8 | 26.0 | 23.3 | 23.9 | 21.0 | 16.0 | 12.7 | 9.4 | 7.7 | 7.1 |
| V | 15.3 | 26.2 | 25.2 | 29.5 | 26.9 | 25.5 | 20.4 | 15.3 | 11.9 | 9.5 | 6.8 |
| F | 13.4 | 21.6 | 22.0 | 23.4 | 12.5 | 7.3 | 4.1 | 3.8 | 3.6 | 3.6 | 3.5 |
| Q | 15.3 | 26.4 | 29.9 | 30.0 | 30.0 | 32.2 | 23.4 | 13.2 | 4.5 | 4.1 | 3.4 |

- 신피질은 심박 신호가 있는 앞·중간 토큰에서 많이 발화하고, 0으로 채운 꼬리 쪽에서 줄어든다. 발화가 줄어드는 위치가 클래스마다 다르다. F는 t4 뒤에 거의 멈추고, Q는 t6까지 높고, V·S는 뒤쪽까지 발화가 남는다.
- 세 뉴런(spikingjelly LIF, spikeDE LIF, f-LIF)의 모양은 비슷하다. 첫 토큰 t0의 발화만 spikeDE 뉴런이 낮다(12–15% vs 22%). 원인은 확인하지 않았다.

### 결론
- **MIT-BIH 분류에서 신피질 경로의 LIF를 f-LIF로 바꿔도 분류 성능은 변하지 않는다.** spikeDE LIF 대조군 대비 −0.02 / −0.03%p, ECG 모델 대비 −0.19 / −0.16%p, 에너지도 거의 같다.
- 정렬된 신피질은 심박을 원 신호보다 클래스가 잘 갈리는 스파이크 부호로 바꾼다. 해마는 그 부호를 심박마다 읽어 쓴다(예측 과제와 다름). f-LIF는 토큰 축에서 정보를 조금 더 멀리 나르지만(both 배치), 분류에는 쓰이지 않았다.
- 신피질과 교차 주의를 함께 빼면 macro-F1이 1.18%p 떨어진다. 그러나 교차 주의를 해마 자신에게 걸면 신피질 없이도 같은 성능이다. 이 데이터에서는 신피질이 대체 불가능한 정보를 준다는 증거가 없다.
- 1–4단계 결론(f-LIF는 성능 이득이 없고 기억은 측정 가능)과 같은 방향이다.

### 한계
- 3 seed 탐색이다. seed 간 표준편차가 0.06–0.42%p라 0.5%p 기준보다 작은 차이는 가릴 수 없다.
- 무작위 심박 분할이라 같은 환자의 심박이 train·test에 섞인다. 환자 간 일반화는 재지 않았다.
- 학습 설정은 ECG 프레임워크 값을 그대로 썼다. f-LIF용 조정(α, τ, 학습률)은 하지 않았다.
- 분석 A1–A4는 마지막 가중치 한 시점의 측정이다. A2 probe는 선형이고 클래스 균형 학습이다.
- 원저자 스크립트 주석의 0.9143은 로그를 확인하지 못한 참고값이다.

### 점검
- 사전 점검 10/10(`results/checks/check_cls.txt`, commit `da14fc7a8`). 분석은 매 run에서 분석용 forward와 모델 forward가 같은지 확인했다(21/21 비트 동일, `meta.json`의 `parity`). 인과성 측정값은 신피질이 있는 15개 run 모두 0이다.
- 감사 88(`docs/PROJECT_LOG.md`, 02:11)은 f_lif_pop_v3 추적 감사로 새 판단 근거가 없다. 이 단계와 관련된 지적도 없다.

**commit/tag:** 이 항목·결과·분석·요약을 담은 commit에 태그 `exp/nsmt-flif-neo-20261019`를 붙이고 브랜치와 태그만 push한다.

---

## 2026-10-19 11:12 KST — 6단계 설계·구현·사전 고정: MIT-BIH에서 population f-LIF와 양방향 신피질 (사용자 제안 2가지), 점검 12/12, smoke 4/4 — 12 run tmux에서 시작

**요청 (2026-10-19, 사용자):** "왜 정렬이 효과적이지 못 할까?" 분석 뒤 — "우리가 설계했던 population f-LIF가 도움을 줄 수 있을까? 그리고 양 방향 설계를 더 하는 거야. 지금은 시계열 토큰 원래 순서대로 입력했지만 반대 순서로도 입력하고 두 정보를 합치는 거야." → "두 제안에 대한 실험을 MIT BIH 데이터셋에 대해 시도해주고 결과를 분석해줘."

**브랜치·기준:** `exp/nsmt-flif-neo`(계속), 기준 commit `b842e414a`(5단계 결과).

**배경 (5단계 결과 뒤 사용자 질문에 답한 분석):** 정렬이 효과를 내지 못한 원인을 네 가지로 정리했다. ① 뉴런 기억이 2–3토큰(A3: d=5에서 4.5%, d=10에서 1.8%), ② head·교차 주의가 이미 모든 토큰을 양방향으로 봄(4단계 R1: 정렬 끔이 더 좋음; hippo_only = base), ③ 인과 제약(앞 토큰이 뒤 문맥을 못 씀), ④ 손실이 기억을 요구하지 않음(보조 목표가 같은 토큰). Population f-LIF는 ①을, 양방향은 ③을 건드린다. ②④는 두 제안 모두 그대로 둔다. **예상(사전 기록):** 두 제안 모두 macro-F1 ±0.3%p 안, 즉 판정 기준 미달. 둘 다 기준 미달이면 ①③은 원인이 아니고 ②④가 남는다.

**설계:**
- **Population f-LIF** (`neurons.py`의 `PopulationFLIF`, f_lif_pop_v3의 v3 배치): 한 유닛에 시간상수 4/8/16/32의 가지 4개(리셋 없음, α 0.5·pred·전체 이력), 학습 가중치(초기 1/4)로 섞어 LIF soma(τ 2, θ 1) 하나가 발화. 신피질 임베딩과 `NeoLayer` 2개 모두에 둔다(both). 파라미터 +4,608(soma 혼합 가중치). 주의: f_lif_pop_v3에서 쓴 입력 크기·임계값 보정은 하지 않는다. 앞단 BN의 affine이 크기를 학습할 수 있다고 보고 ECG 설정 그대로 둔다.
- **양방향 신피질** (`--neo_bidir`): 역순 토큰 열을 **별도 파라미터**의 임베딩 + `NeoLayer` 2개(`encoding_neo_rev`, `time_block_rev`)에 통과시키고, 다시 뒤집어 정순 출력과 **OR(원소별 최대)** 로 합친다. OR을 고른 이유: 출력이 0/1로 남아 해마의 key/value 선형층이 AC 연산으로 유지되고(합이면 2가 생겨 MAC로 집계됨), 발화율 ~0.1에서 두 방향이 동시에 발화하는 경우는 ~1%라 잃는 정보가 작다. 신피질 파라미터 2배(1,226,112 → 1,535,616). 보조 손실은 합친 출력으로 계산한다(토큰별 목표는 그대로).
- 둘을 합친 조건과, 양방향의 뉴런 효과를 분리할 spikeDE LIF 양방향 대조군을 둔다.

**조건 (새 suite `flif-v1-cls2`, 4 조건 × seed 42/7/2026 = 12 run; 참조는 5단계 `flif-v1-cls`의 같은 코드 경로·같은 학습 규칙 run):**

| 조건 | 신피질 |
|---|---|
| neo_pop_both | population f-LIF, 정순 |
| bidir_lif_both | spikeDE LIF, 양방향 |
| bidir_flif_both | f-LIF, 양방향 |
| bidir_pop_both | population f-LIF, 양방향 |
| (참조) neo_lif_both, neo_flif_both, base, hippo_only, no_neo | 5단계 run 재사용 (c1: 기본 경로 비트 동일 확인) |

학습은 5단계와 같다(30 epoch cosine, bs 512, lr 0.01, CAL 가중 CE, 마지막 가중치로 test).

**코드 (`nsmt_flif_v1/classification/`):** `layers.build_neo_neuron`에 `pop_flif`(`pop_taus`); `model.py`에 `neo_bidir`, 역방향 모듈, 공통 `neo_forward(neo_in)`(분석 스크립트도 이것을 씀); `config.py`에 `--pop_taus --neo_bidir`와 폴더 접미사 `_neopop_flif-…`, `_bidir`; `load_model.py`; `energy.py`에 population 에너지(우리 규약: 가지 K개의 이력 항(식 64 FFT 또는 직접 합) + 리셋 없는 스텝 항 + soma 혼합 K MAC/스텝 + soma LIF 식 60; 논문에 없는 식이므로 보고용)와 역방향 모듈 접두사; `scripts/run_mitbih.sh`(4 조건), `make_queue_cls.sh`(`CONDS=`), `cls_driver.sh`(`SUITE= CONDS= REF_SUITE=`), `summarize_cls.py --ref_suite`(Q4–Q7); `check_cls.py` c11–c12.

**점검 12/12 (`results/checks/check_cls_stage6.txt`):** c1–c10 그대로 통과(기본 경로·f-LIF 경로 비트 동일 유지). **c11** population: 가지 4개·시간상수 4/8/16/32·soma τ 2, 출력 0/1·[11, 32, 1, 384], 토큰 축 인과성 3/3, 에너지 기록기가 pop_flif로 집계. 초기 가중치를 c5 방식으로 키워도(3배·7배) soma가 발화하지 않아(0.000) smoke 체크포인트(1 epoch 학습)와 실제 심박으로 검사했다 — 발화율 임베딩 0.0045, 출력 0.019. **c12** 양방향: 역방향 모듈이 별도 파라미터, 합친 출력 = max(정순, 뒤집은 역순) 비트 동일, 두 방향이 동시에 발화하는 원소 존재, 토큰 j를 바꾸면 정순 가지는 j 이후만·역순 가지는 j 이전만 바뀜(3/3), 폴더 이름 구분. 처음 c12는 "별도 파라미터" 검사가 BN의 초기값(1, 0)까지 비교해 실패했고, Linear 가중치와 뉴런 객체로 바꿨다.

**smoke (`results/smoke-cls2-20261019/raw.txt`, seed 42, 1 epoch; 실험 아님):**

| 조건 | 정확도 | macro-F1 | 총 µJ (FFT / 직접 합) | 신피질 발화율 | 파라미터 | 시간 s |
|---|---|---|---|---|---|---|
| neo_pop_both | 0.942406 | 0.722443 | 21.70 / 22.09 | 0.0098 | 1,230,720 | 50 |
| bidir_lif_both | 0.926507 | 0.601142 | 21.47 / 21.47 | 0.0541 | 1,535,616 | 76 |
| bidir_flif_both | 0.885176 | 0.464674 | 21.56 / 22.03 | 0.0593 | 1,535,616 | 54 |
| bidir_pop_both | 0.912283 | 0.548903 | 24.31 / 25.08 | 0.0081 | 1,544,832 | 88 |

**관찰(사전 기록):** population 조건의 신피질 발화율이 1 epoch 뒤 약 1%로, f-LIF(5단계 1 epoch 7.6%)의 1/8이다. 느린 가지(τ 4–32)와 1/4 혼합이 입력을 크게 줄이기 때문이다. 30 epoch 뒤에도 낮게 남으면 population 조건의 결과는 "보정 없는 설정"의 결과로 읽어야 한다. 분석 스크립트는 bidir_pop_both smoke 체크포인트로 시험했다(forward 일치 비트 동일; 양방향이므로 A3에서 바꾼 토큰 앞쪽 변화 3.3%가 나타남 — 설계대로).

**판정 (고정; 5단계와 같은 규칙 — test macro-F1, 같은 seed끼리 짝, 평균 ≥ +0.5%p이고 3 seed 중 2개 이상):**
- **Q4 population:** neo_pop_both vs neo_flif_both.
- **Q5 양방향:** bidir_flif_both vs neo_flif_both; bidir_lif_both vs neo_lif_both(뉴런과 무관한 양방향 효과).
- **Q6 둘 다:** bidir_pop_both vs neo_flif_both, vs bidir_flif_both.
- **Q7 (원래 기준 (2), 보고):** 세 조건 각각 vs hippo_only, 에너지 비율과 함께.
- 분석 A1–A4는 5단계와 같이 보고만 한다. 양방향에서 A3의 "바꾼 토큰 앞쪽 변화"는 역방향 가지의 정상 동작이다.
- 탐색 단계(3 seed)이므로 통과는 확증이 아니다.

**실행:** tmux `flif_cls2`, `SUITE=flif-v1-cls2 CONDS="bidir_pop_both bidir_flif_both neo_pop_both bidir_lif_both" REF_SUITE=flif-v1-cls scripts/cls_driver.sh 0,1,2,3,0,1,2,3`. 예상 1시간(smoke에서 양방향 population은 epoch당 약 90 s).

**commit:** 이 항목·코드·점검·smoke를 담은 commit을 push한 뒤 시작한다. 결과는 다음 항목.

---

## 2026-10-19 12:00 KST — 6단계 결과 (MIT-BIH, population f-LIF·양방향 신피질, 12 run + 5단계 참조, 마지막 가중치): **Q4 population 차이 없음(−0.07%p), Q5 양방향 차이 없음(f-LIF −0.16%p, LIF +0.24%p), Q6 둘 다 차이 없음(+0.31%p / +0.48%p), Q7 hippo_only 대비 양방향 population만 동급(−0.12%p, 에너지 1.04배)** — 사전 예상대로 기준 미달; population 뉴런은 토큰 축 기억을 평탄하게 늘리지만 클래스 정보는 덜 담는다 — 태그 `exp/nsmt-flif-neo-20261019-2`

**구분:** 사전 고정한 6단계 **탐색**(3 seed)의 완료 기록. 코드·사전 고정 commit `8714c5f80`. 이 항목이 속한 commit은 태그 `exp/nsmt-flif-neo-20261019-2`로 식별한다.

**실행:** tmux `flif_cls2`, `cls_driver.sh 0,1,2,3,0,1,2,3`(GPU당 2개). 학습 11:12–11:55 **12/12 완료·실패 0**, 모두 30 epoch. run당 시간: neo_pop_both 1045–1393 s, bidir_lif_both 700–1008 s, bidir_flif_both 1848–1986 s, bidir_pop_both 1874–2086 s. 분석 12개 11:55–11:57(실패 0, 분석용 forward와 모델 forward 12/12 비트 동일). 요약 `nsmt_flif_v1/classification/results/flif-v1-cls2/summary.md`(5단계 `flif-v1-cls`의 참조 run을 합쳐 계산). 산출물은 `results/flif-v1-cls2/{raw.txt, runs.csv, conditions.csv, analysis/}`와 실행별 `log/flif-v1-cls2/MITBIH/261019/<run>/<seed+조건>/log/…`(Git). checkpoint·TensorBoard·stdout은 로컬.

### 성능과 에너지 (test 32,833개, 3 seed 평균 ± 표준편차, %; 위 5줄은 5단계 참조)

| 조건 | 정확도 | macro-F1 | F1 S | F1 V | F1 F | 총 에너지 µJ (FFT / 직접 합) | 신피질 발화율 | 파라미터 |
|---|---|---|---|---|---|---|---|---|
| ECG 모델 (base) | 98.41 ± 0.03 | 91.58 ± 0.06 | 84.31 | 95.65 | 79.89 | 23.86 / 23.86 | 0.093 | 1,226,112 |
| spikeDE LIF, both | 98.39 ± 0.03 | 91.42 ± 0.29 | 83.77 | 95.73 | 79.72 | 23.08 / 23.08 | 0.114 | 1,226,112 |
| f-LIF, both | 98.35 ± 0.07 | 91.39 ± 0.42 | 83.11 | 95.72 | 80.18 | 23.43 / 23.67 | 0.106 | 1,226,112 |
| hippo_only | 98.43 ± 0.04 | **91.83 ± 0.23** | 84.01 | 95.90 | 81.23 | 26.02 / 26.02 | – | 1,226,112 |
| no_neo | 98.13 ± 0.07 | 90.40 ± 0.34 | 79.95 | 95.40 | 78.88 | 13.03 / 13.03 | – | 1,226,112 |
| **population f-LIF, both** | 98.39 ± 0.04 | 91.32 ± 0.04 | 83.46 | 95.88 | 79.17 | 24.83 / 25.21 | 0.079 | 1,230,720 |
| **양방향 spikeDE LIF** | 98.44 ± 0.03 | 91.65 ± 0.20 | 84.28 | 95.95 | 80.06 | 24.41 / 24.41 | 0.085 | 1,535,616 |
| **양방향 f-LIF** | 98.36 ± 0.06 | 91.23 ± 0.31 | 83.49 | 95.73 | 78.96 | 24.41 / 24.88 | 0.071 | 1,535,616 |
| **양방향 population f-LIF** | **98.45 ± 0.04** | 91.70 ± 0.35 | **84.80** | 95.93 | 79.81 | 27.09 / 27.86 | 0.037 | 1,544,832 |

### 사전 고정 판정 (macro-F1, 같은 seed끼리 짝, %p; 기준 평균 ≥ +0.5이고 3 seed 중 2개 이상)

| 질문 | A − B (seed 42 / 7 / 2026) | 평균 | 판정 |
|---|---|---|---|
| Q4 population − f-LIF | −0.50 / +0.24 / +0.05 | −0.07 | 뚜렷한 차이 없음 |
| Q5 양방향 f-LIF − f-LIF | −0.40 / +0.27 / −0.37 | −0.16 | 뚜렷한 차이 없음 |
| Q5 양방향 LIF − spikeDE LIF | −0.25 / +0.72 / +0.23 | +0.24 | 뚜렷한 차이 없음 |
| Q6 양방향 population − f-LIF | +0.14 / +0.74 / +0.06 | +0.31 | 뚜렷한 차이 없음 |
| Q6 양방향 population − 양방향 f-LIF | +0.53 / +0.47 / +0.43 | +0.48 | 뚜렷한 차이 없음 (3/3 같은 방향이지만 평균 0.48 < 0.5) |
| Q7 양방향 population − hippo_only | −0.06 / −0.01 / −0.30 | −0.12 | 뚜렷한 차이 없음 (에너지 1.04배) |
| Q7 양방향 f-LIF − hippo_only | −0.59 / −0.48 / −0.73 | −0.60 | **나쁨** (에너지 0.94배) |
| Q7 population − hippo_only | −0.69 / −0.51 / −0.31 | −0.50 | **나쁨** (에너지 0.95배) |

- 정확도로는 8개 비교 모두 "뚜렷한 차이 없음"(최대 +0.10%p).
- **사전 예상("두 제안 모두 ±0.3%p 안, 기준 미달")과 일치한다.** 가장 좋은 조합(양방향 population)도 f-LIF 대비 +0.31%p, hippo_only와 동급이고 에너지는 4% 더 든다. 원래 통과 기준 (2)(hippo_only보다 성능이 좋거나, 비슷하면서 에너지가 훨씬 적음)는 어느 조건도 만족하지 않는다.
- 양방향 population이 양방향 f-LIF보다 3 seed 모두 +0.43–0.53%p 높은 것은 유일하게 방향이 일관된 결과다. 그러나 평균이 기준에 못 미치고, 정순 population − 정순 f-LIF는 −0.07%p라 "population이 양방향에서만 돕는다"는 사후 해석이 된다. 확증 없이는 주장하지 않는다.

### 분석 (보고만; `summary.md` A1–A4)

**A3 — population 뉴런은 토큰 축 기억을 실제로 평탄하게 늘린다.** 토큰 하나를 바꿨을 때 d토큰 뒤 신피질 출력 스파이크가 바뀌는 비율(%):

| 조건 | d=0 | d=1 | d=2 | d=5 | d=8 | d=10 |
|---|---|---|---|---|---|---|
| f-LIF, both (5단계) | 10.88 | 12.96 | 4.13 | 2.05 | 1.26 | 1.11 |
| spikeDE LIF, both (5단계) | 9.28 | 9.87 | 6.52 | 1.96 | 0.92 | 0.69 |
| **population f-LIF, both** | 5.46 | 6.65 | 7.78 | 8.04 | 7.06 | 7.07 |
| 양방향 f-LIF | 11.99 | 7.44 | 2.54 | 1.25 | 0.75 | 0.63 |
| 양방향 population | 4.94 | 2.12 | 2.59 | 2.88 | 2.64 | 2.78 |

- f-LIF는 d=10에서 1.1%인데 population은 **7.1%로 감쇠가 거의 없다**(가지 τ 16·32는 11토큰 안에서 잊지 않는다). 6단계의 원인 ①(기억 2–3토큰)은 population으로 확실히 제거되었다. **그런데도 성능은 그대로다** — 기억 길이가 병목이 아니었다는 뜻이다(원인 ②④가 남는다).
- 양방향에서는 바꾼 토큰 **앞쪽**도 바뀐다(최대 6.5–13.6%; 역방향 가지의 정상 동작). 정순 population은 앞쪽 변화 0(인과성 유지).

**A2 — population의 스파이크는 클래스 정보를 덜 담는다** (클래스 균형 선형 probe, test macro-F1 %):

| 조건 | 신피질 임베딩 스파이크 | 신피질 출력 스파이크 | 해마 head 입력 |
|---|---|---|---|
| f-LIF, both (5단계) | 60.80 | 62.33 | 87.11 |
| population f-LIF, both | 51.73 | 53.57 | 86.97 |
| 양방향 f-LIF | 59.32 | 63.61 | 87.93 |
| 양방향 population | 46.14 | 47.08 | 86.74 |

- population 출력에서 읽히는 클래스는 f-LIF보다 9–15%p 낮다(원 심박 47%에 가깝다). 느린 가지가 토큰을 평균내어 토큰별 구별이 흐려지고, 발화율도 낮다(0.037–0.079). 해마 특징(87%)은 모든 조건에서 같다 — 최종 표현은 신피질 종류와 무관하다.

**A1 — 해마의 의존도.** 신피질 출력을 다른 심박의 것으로 바꿨을 때 macro-F1 변화: f-LIF −20.1, population −13.2, 양방향 LIF −22.4, 양방향 f-LIF −22.0, 양방향 population −15.1%p. population 조건은 해마가 덜 의존한다 — 정보가 적으니 덜 쓴다는 해석과 맞는다.

**A4 — 클래스별 발화 윤곽.** population은 심박 신호가 끝난 뒤쪽 토큰(t8–t10)에서도 15–22% 발화가 남아(f-LIF 3–7%) 긴 기억이 윤곽에 그대로 보인다. F 클래스에서 일찍 줄어드는 모양은 모든 뉴런에서 같다.

### 결론
- **두 제안 모두 MIT-BIH 분류 성능을 바꾸지 못했다.** 모든 사전 고정 비교가 "뚜렷한 차이 없음"이고, hippo_only를 넘는 조건은 없다. 사전 예상과 같다.
- population f-LIF는 설계대로 토큰 축 기억을 10토큰까지 평탄하게 유지한다(d=10에서 7.1% vs f-LIF 1.1%). 그 기억이 성능으로 이어지지 않았으므로, **기억 길이(원인 ①)는 정렬이 효과 없는 이유가 아니다.** 양방향도 인과 제약(원인 ③)을 없앴지만 성능이 그대로여서 **③도 이유가 아니다.** 남는 설명은 ②(head·교차 주의가 이미 모든 토큰을 봄; hippo_only = base)와 ④(손실이 토큰 사이 기억을 요구하지 않음)다.
- 다음에 정렬의 가치를 시험하려면 ② 또는 ④를 깨야 한다: head·attention이 볼 수 없는 긴 과거를 신피질 축에만 넣기(의도 2), 또는 보조 손실의 목표를 미래 토큰으로 옮기기. 이는 사용자의 결정 사항이다.

### 한계
- 3 seed 탐색. 표준편차 0.04–0.42%p.
- population 뉴런에 입력 크기·임계값 보정을 하지 않았다. 발화율이 f-LIF의 0.35–0.75배이고 양방향에서는 0.037로 낮다. 보정하면 A2의 클래스 정보 부족이 줄어들 수 있으나, A3의 기억 측정이 이미 "기억은 길지만 쓰이지 않음"을 보이므로 결론 방향은 바뀌지 않을 것으로 본다(미확인).
- 양방향 결합은 OR 하나만 시험했다. 이어 붙임이나 두 방향을 각각 key/value로 주는 방식은 시험하지 않았다.
- population 에너지 식은 우리 규약(논문에 없음)이라 절대값 비교는 참고용이다.

### 점검
- 사전 점검 12/12(`results/checks/check_cls_stage6.txt`, commit `8714c5f80`). 분석 forward 일치 12/12 비트 동일. 정순 population의 인과성 3 run 모두 0.
- 감사 90(`docs/PROJECT_LOG.md`, 11:21)은 f_lif_pop_v3 추적 감사로 새 판단 근거가 없고 이 단계와 무관하다.

**commit/tag:** 이 항목·결과·분석·요약을 담은 commit에 태그 `exp/nsmt-flif-neo-20261019-2`를 붙이고 브랜치와 태그만 push한다.

---

## 2026-10-19 12:45 KST — 7단계 설계·구현·사전 고정: 교차 주의 모듈을 신피질망(양방향 population f-LIF)으로 교체, 패치 기반 T = N 정렬, 보조 손실 없음 — 점검 13/13, smoke 6/6, 18 run tmux에서 시작

**요청 (2026-10-19, 사용자):** "self attention 모듈을 현재 neo network로 교체하는 실험에 대해서는 어떻게 생각해? 양방향 population f-LIF 뉴런이 사용되고 당연히 보조 손실은 제거되겠지" → (내 설계 제안 뒤) "해당 설정에서는 패치 기반 연산이고 T=N 정렬이 적용되어야해."

**브랜치·기준:** `exp/nsmt-flif-neo`(계속), 기준 commit `3399fb585`(6단계 결과).

**이 실험이 묻는 것:** 6단계까지 남은 원인은 ②(교차 주의·head가 이미 모든 토큰을 봄)와 ④(손실이 토큰 사이 기억을 요구하지 않음)였다. 교차 주의를 없애고 보조 손실을 빼면, 정렬된 뉴런 상태가 **토큰 사이 정보를 나르는 유일한 통로**가 되고 과제 손실이 그 뉴런에 100% 닿는다. 5단계 수치로 쓰면: 교차 주의는 no_neo(90.40) 대비 **+1.4%p(hippo_only 91.83)와 약 13 µJ**를 차지한다. 신피질망이 그 +1.4%p를 얼마나 되찾고 에너지는 얼마나 쓰는지가 질문이다. 이는 원래 통과 기준 (2)의 형태다. 주의할 점: 이 모델은 더 이상 "f-LIF 신피질을 단 NSMT"가 아니라 **주의 없는 정렬 스파이킹 재귀망**이다(두 경로 서사 없음).

**모델 (`nsmt_flif_v1/classification/neo_mixer.py`, `--model neoMixer`):**
- 패치: 187샘플 → 길이 32·보폭 16·끝 복제 패딩 → 토큰 N = 11 (ECG 모델과 같음). **T = N = 11**: 모든 스파이킹 뉴런이 토큰 축을 시간 축으로 쓴다. 해마의 반복 축(같은 입력 11번)은 없다.
- 임베딩: Linear(32→384) → BN → spikingjelly LIF(토큰 축). 조건 간 고정.
- 블록 × L: `x = x·(1 − mixer(x))`, `x = x·(1 − MLP(x))` (논문의 IAND 결합). **mixer = 신피질망** = `NeoLayer`(Linear → BN → 뉴런) 2층, 양방향이면 역순 토큰 열에 별도 2층을 두고 뒤집어 OR로 합침(6단계와 같음). MLP는 SpikLinearLayer 2개(spikingjelly LIF, 토큰 축), 조건 간 고정.
- head: 토큰 평균 → Linear(384→5). 손실은 CAL 가중 CE만 (`train.py`가 `time_block` 속성이 없는 모델에 CE만 적용; 디코더·보조 손실 없음).
- 설계상 주의: T = N에서는 임베딩·MLP의 LIF도 토큰 축 상태를 가지므로, mixer가 없는 조건(`none`)도 "토큰 독립"이 아니라 "**정렬만 있고 섞기 모듈이 없음**"이다(τ = 2 LIF의 2–3토큰 기억). 이것이 하한 대조군이다. 토큰 독립 하한은 5단계 no_neo(90.40)를 참조한다.
- 파라미터: none 311,424 / 양방향 mixer 904,320(pop 910,464) / 정순 pop 610,944 / 2블록 양방향 pop 1,805,952. 비교: hippo_only 1,226,112(교차 주의 D×D 선형 4개 포함).

**조건 (suite `flif-v1-cls3`, 6 조건 × seed 42/7/2026 = 18 run; 학습 규칙은 5단계와 같음 — 30 epoch cosine, bs 512, lr 0.01, CAL CE, 마지막 가중치로 test):**

| 조건 | mixer | 방향 | 블록 |
|---|---|---|---|
| mix_none_fwd | 없음 (하한: 정렬만) | – | 1 |
| mix_lif_bidir | spikeDE LIF | 양방향 | 1 |
| mix_flif_bidir | f-LIF | 양방향 | 1 |
| **mix_pop_bidir** (제안) | population f-LIF (가지 4/8/16/32 + LIF soma) | 양방향 | 1 |
| mix_pop_fwd | population f-LIF | 정순 | 1 |
| mix_pop_bidir_L2 | population f-LIF | 양방향 | 2 |
| (참조) hippo_only, no_neo, base, f-LIF both | 5단계 run 재사용 | | |

**코드:** `neo_mixer.py`(NeoMixer, `forward_parts`로 분석 훅), `load_model.py`(`LOAD_MODEL['neoMixer']`), `config.py`(`--mixer {none,lif,flif,pop_flif} --mixer_bidir`; `--layers` = 블록 수, `--time_layers` = mixer 층 수; 폴더 접미사 `_mix-<kind>[-bidir]-L<블록>-a<α>`), `energy.py`(mixer 스택을 신피질 경로로 집계), `scripts/run_mitbih.sh`(`COND=mix_<none|lif|flif|pop>_<fwd|bidir>[_L2]`), `cls_analysis.py`(NeoMixer 지원: "신피질 출력" = 첫 블록 mixer 출력, replay 교체도 그 자리), `summarize_cls.py`(Q8–Q10), `check_cls.py` c13.

**점검 13/13 (`results/checks/check_cls_stage7.txt`):** c1–c12 그대로 통과(기존 경로 비트 동일 유지). **c13**: 주의 모듈 없음, N = T = 11이고 스파이킹 층 7개(임베딩 + mixer 2×2 + MLP 2) 모두 T = 11로 집계, 학습 forward는 로짓만(CE만 적용되는 경로), test forward는 (로짓, 토큰 스파이크); `none`에서 토큰 j를 바꾸면 j 이후만 바뀜(앞은 불변; 3/3); 양방향 f-LIF mixer에 임베딩 토큰 j를 바꿔 넣으면 정순 스택은 j 이후만, 역순 스택은 j 이전만 바뀌고 합은 max와 비트 동일(3/3); 폴더 이름 6개 구분. 처음 c13은 `none`을 "토큰 독립"으로 기대해 실패했고(T = N에서는 임베딩 LIF도 상태를 가짐), 기대를 "정렬만, 인과"로 고쳤다. 역순 스택 검사도 토큰 대신 임베딩을 바꾸는 방식으로 고쳤다(임베딩 LIF의 인과 상태가 j 이후 임베딩을 바꿔 역순 검사를 오염시켰음).

**smoke (`results/smoke-cls3-20261019/raw.txt`, seed 42, 1 epoch; 실험 아님):**

| 조건 | 정확도 | macro-F1 | 시냅스 µJ | 총 µJ (FFT / 직접 합) | mixer 발화율 | 시간 s |
|---|---|---|---|---|---|---|
| mix_none_fwd | 0.936345 | 0.690148 | 0.99 | 1.14 / 1.14 | – | 18 |
| mix_lif_bidir | 0.946030 | 0.724317 | 1.64 | 1.99 / 1.99 | 0.057 | 37 |
| mix_flif_bidir | 0.946670 | 0.711909 | 1.60 | 2.10 / 2.41 | 0.050 | 34 |
| mix_pop_bidir | 0.913563 | 0.665180 | 1.48 | 4.21 / 4.73 | 0.008 | 42 |
| mix_pop_fwd | 0.924832 | 0.569331 | 1.25 | 2.69 / 2.95 | 0.027 | 36 |
| mix_pop_bidir_L2 | 0.922152 | 0.562941 | 2.24 | 7.67 / 8.69 | 0.012 | 55 |

**smoke에서 이미 보이는 것 (사전 기록):** 반복 축을 없앤 결과 **에너지가 한 자릿수 µJ**다(hippo_only 26.0, ECG 모델 23.9 대비 1/3–1/20). 시냅스 연산이 11배 줄기 때문이다. 따라서 기준 (2)의 "에너지가 훨씬 적음"은 거의 확실히 성립하고, 질문은 성능 쪽으로 좁혀진다. population mixer의 발화율은 6단계처럼 낮다(0.8–2.7%; 보정 없음). 분석 스크립트는 mix_pop_bidir·mix_none_fwd smoke 체크포인트로 시험했다(forward 일치 비트 동일).

**판정 (고정; test macro-F1, 같은 seed끼리 짝, 평균 ≥ +0.5%p이고 3 seed 중 2개 이상 = "좋음"; 반대면 "나쁨"):**
- **Q8 mixer 효과:** 각 mixer 조건 vs mix_none_fwd. mixer가 no_neo 수준의 하한보다 나아야 "정렬된 재귀 상태가 토큰을 섞는 장치로 쓸모 있다".
- **Q9 주의 대체 (주 질문):** mix_pop_bidir, mix_flif_bidir, mix_lif_bidir, mix_pop_bidir_L2 각각 vs hippo_only. 해석 틀: hippo_only보다 "나쁨"이 아니면(−0.5%p 이내) 에너지와 함께 기준 (2) 충족으로 본다; "나쁨"이지만 mix_none_fwd보다 "좋음"이면 "일부 대체"; mix_none_fwd와도 차이 없으면 재귀 섞기는 효과 없음.
- **Q10 (보고):** population vs f-LIF(mixer), 양방향 vs 정순, 2블록 vs 1블록, mix_none_fwd vs no_neo(정렬만의 효과).
- 에너지: 시냅스 + 뉴런 고유(population은 우리 규약), hippo_only 대비 비율.
- 분석 A1–A4는 같은 스크립트로 보고만 한다(A1은 첫 블록 mixer 출력 교체).
- **예상(사전 기록):** 양방향 mixer들은 mix_none_fwd보다 좋고(Q8 통과), hippo_only에는 못 미치되(−0.3 ~ −1.0%p) 에너지 0.1배 수준. population이 f-LIF보다 나을 근거는 없다(6단계 A2).
- 탐색 단계(3 seed)이므로 통과는 확증이 아니다.

**실행:** tmux `flif_cls3`, `SUITE=flif-v1-cls3 CONDS="mix_pop_bidir_L2 mix_pop_bidir mix_flif_bidir mix_lif_bidir mix_pop_fwd mix_none_fwd" REF_SUITE=flif-v1-cls scripts/cls_driver.sh 0,1,2,3,0,1,2,3`. 예상 40분(epoch당 18–55 s).

**감사:** 91(`docs/PROJECT_LOG.md` 12:11)은 f_lif_pop_v3 추적 감사로 새 판단 근거가 없다.

**commit:** 이 항목·코드·점검·smoke를 담은 commit을 push한 뒤 시작한다. 결과는 다음 항목.

---

## 2026-10-19 13:15 KST — 7단계 결과 (교차 주의 → 신피질망, 18 run + 5단계 참조, 마지막 가중치): **Q8 양방향 f-LIF·LIF mixer는 "정렬만" 하한보다 좋음(+0.53 / +0.54%p), population mixer는 차이 없음; Q9 네 조건 모두 hippo_only보다 나쁨(−0.68 ~ −1.21%p)이지만 에너지는 0.08–0.30배** — 정렬된 재귀 상태는 교차 주의가 주는 이득의 절반을 1/12의 에너지로 되찾는다; population은 여기서도 돕지 않는다 — 태그 `exp/nsmt-flif-neo-20261019-3`

**구분:** 사전 고정한 7단계 **탐색**(3 seed)의 완료 기록. 코드·사전 고정 commit `e05fc1b48`. 이 항목이 속한 commit은 태그 `exp/nsmt-flif-neo-20261019-3`로 식별한다.

**실행:** tmux `flif_cls3`, `cls_driver.sh 0,1,2,3,0,1,2,3`. 학습 12:45–13:11 **18/18 완료·실패 0**, 모두 30 epoch. run당 시간: none 163–172 s, 양방향 LIF 415–504, 양방향 f-LIF 547–688, 양방향 pop 691–818, 정순 pop 382–524, 2블록 1223–1381(반복 축이 없어 주의 모델의 1/2–1/6). 분석 18개 13:11–13:13(실패 0, forward 일치 18/18 비트 동일). 요약 `nsmt_flif_v1/classification/results/flif-v1-cls3/summary.md`(5단계 참조 run 포함). 산출물 `results/flif-v1-cls3/{raw.txt, runs.csv, conditions.csv, analysis/}`, 실행별 `log/flif-v1-cls3/MITBIH/261019/<run>/<seed+조건>/log/…`(Git). checkpoint·stdout은 로컬.

### 성능과 에너지 (test 32,833개, 3 seed 평균 ± 표준편차, %; 위 3줄은 5단계 참조)

| 조건 | 정확도 | macro-F1 | F1 S | F1 F | 시냅스 µJ | 총 µJ (FFT / 직접 합) | 총 에너지 / hippo_only | mixer 발화율 | 파라미터 |
|---|---|---|---|---|---|---|---|---|---|
| hippo_only (교차 주의, 상한) | 98.43 ± 0.04 | **91.83 ± 0.23** | 84.01 | 81.23 | 21.41 | 26.02 / 26.02 | 1.00 | – | 1,226,112 |
| ECG 모델 | 98.41 ± 0.03 | 91.58 ± 0.06 | 84.31 | 79.89 | 19.06 | 23.86 / 23.86 | 0.92 | – | 1,226,112 |
| no_neo (토큰 독립 하한) | 98.13 ± 0.07 | 90.40 ± 0.34 | 79.95 | 78.88 | 11.32 | 13.03 / 13.03 | 0.50 | – | 1,226,112 |
| **mix_none_fwd** (정렬만, 섞기 없음) | 98.17 ± 0.04 | 90.60 ± 0.06 | 81.36 | 78.65 | 1.02 | **1.17 / 1.17** | **0.045** | – | 311,424 |
| **mix_lif_bidir** | 98.32 ± 0.04 | 91.14 ± 0.08 | 83.64 | 78.75 | 1.79 | 2.15 / 2.15 | 0.082 | 0.075 | 904,320 |
| **mix_flif_bidir** | 98.30 ± 0.04 | 91.13 ± 0.21 | 82.81 | 79.36 | 1.75 | 2.25 / 2.56 | 0.086 | 0.057 | 904,320 |
| **mix_pop_bidir** (제안) | 98.21 ± 0.04 | 90.62 ± 0.08 | 81.70 | 78.17 | 1.57 | 4.32 / 4.83 | 0.166 | 0.033 | 910,464 |
| **mix_pop_fwd** | 98.16 ± 0.05 | 90.54 ± 0.27 | 81.78 | 78.01 | 1.27 | 2.72 / 2.98 | 0.105 | 0.067 | 610,944 |
| **mix_pop_bidir_L2** (2블록) | 98.31 ± 0.01 | 91.07 ± 0.26 | 83.21 | 78.57 | 2.47 | 7.90 / 8.93 | 0.304 | 0.031 | 1,805,952 |

### 사전 고정 판정 (macro-F1, 같은 seed끼리 짝, %p; 기준 평균 ≥ 0.5이고 3 seed 중 2개 이상)

| 질문 | A − B (seed 42 / 7 / 2026) | 평균 | 판정 |
|---|---|---|---|
| Q8 양방향 f-LIF mixer − 정렬만 | +0.79 / +0.26 / +0.54 | +0.53 | **좋음** |
| Q8 양방향 LIF mixer − 정렬만 | +0.54 / +0.59 / +0.50 | +0.54 | **좋음** |
| Q8 양방향 population mixer − 정렬만 | +0.17 / −0.05 / −0.08 | +0.02 | 차이 없음 |
| Q8 정순 population mixer − 정렬만 | +0.03 / +0.16 / −0.36 | −0.06 | 차이 없음 |
| Q9 양방향 population − hippo_only | −1.36 / −1.21 / −1.05 | −1.21 | **나쁨** (에너지 0.17배) |
| Q9 양방향 f-LIF − hippo_only | −0.74 / −0.91 / −0.44 | −0.70 | **나쁨** (0.09배) |
| Q9 양방향 LIF − hippo_only | −0.99 / −0.58 / −0.48 | −0.68 | **나쁨** (0.08배) |
| Q9 2블록 양방향 population − hippo_only | −1.04 / −0.98 / −0.26 | −0.76 | **나쁨** (0.30배) |
| (참고) 정렬만 − no_neo | −0.25 / +0.36 / +0.50 | +0.21 | 차이 없음 (에너지 0.09배) |

- 정확도로는 모든 비교가 "뚜렷한 차이 없음"(hippo_only 대비 −0.10 ~ −0.22%p).
- **사전 예상과 일치:** 양방향 mixer는 하한보다 좋고 hippo_only에는 못 미친다(예상 −0.3 ~ −1.0, 실제 −0.68 ~ −0.76). population이 f-LIF보다 나을 근거는 없다고 적었고, 실제로 population은 하한과 구별되지 않는다.
- **no_neo 기준으로 다시 쓰면:** 교차 주의가 주는 +1.43%p 중 양방향 f-LIF·LIF mixer는 **+0.73 / +0.75%p(약 절반)** 를 되찾고, 총 에너지는 hippo_only의 **1/12**(2.2 µJ)다. 사전 해석 틀로는 "일부 대체"다. 기준 (2)("성능 비슷, 에너지 훨씬 적음")는 성능 쪽(−0.5%p 이내)에서 0.2%p 차이로 미달이다.
- 반복 축을 없앤 것만으로 에너지는 1/20(정렬만 1.17 µJ)이 되고 성능은 no_neo와 같다. 이 모델의 에너지 대부분은 "같은 입력을 11번 반복"하는 해마 구조에서 나온 것이었다.

### 분석 (보고만; `summary.md` A1–A4)

**A1 — mixer 출력은 분류에 결정적으로 쓰인다.** 첫 블록 mixer 출력을 다른 심박의 것으로 바꾸면 macro-F1 −42 ~ −53%p(5단계 신피질 −20). 같은 클래스 심박이면 −9 ~ −14, 다른 클래스면 −55 ~ −65. 주의가 없으니 mixer가 유일한 토큰 섞기 통로이고, 모델이 그것에 전적으로 의존한다. 2블록 모델만 첫 블록 교체에 −5%p로 둔감하다(두 번째 블록이 다시 섞음).

**A2 — mixer 스파이크는 지금까지 중 클래스 정보를 가장 많이 담는다** (선형 probe, test macro-F1 %): 양방향 LIF 74.2, f-LIF 73.5, population 70.2(5단계 신피질 60–66). 과제 손실이 100% 닿고 보조 손실이 없어서다. 그러나 최종 head 입력은 86.5–87.0으로 hippo_only(88.3)보다 낮다. 이 1.3%p 차이가 Q9의 성능 차이와 맞물린다.

**A3 — 토큰 축 기억.** 양방향 LIF mixer는 d=10에서 1.11%, f-LIF 0.59%, population 4.6–5.4%(평탄). 6단계와 같은 그림이다: **population의 긴 기억은 여기서도 성능으로 이어지지 않는다**. 오히려 population의 발화율(0.031–0.033)이 가장 낮고 F1도 가장 낮다. 토큰 0을 바꿨을 때 예측이 바뀌는 비율은 25–27%(5단계 2–6%)로, mixer 모델은 첫 토큰에 훨씬 민감하다.

### 결론
- **정렬된 스파이킹 재귀 상태는 교차 주의를 "절반" 대체한다.** 양방향 f-LIF 또는 LIF mixer는 토큰 섞기가 없는 하한보다 분명히 좋고(+0.53/+0.54%p, 3/3), 교차 주의가 주는 이득의 약 절반을 되찾으며, 총 에너지는 1/12이다. 6단계까지 남았던 원인 ②(주의가 이미 다 본다)가 실제 병목이었음을 뜻한다. 주의를 빼자 처음으로 정렬된 뉴런 상태가 쓰였다.
- **그러나 교차 주의를 완전히 대체하지는 못한다**(−0.7%p, 3/3). 원래 통과 기준 (2)에는 0.2%p 모자란다.
- **f-LIF는 LIF와 같다**(91.13 vs 91.14). 분수 기억은 이 과제에서 끝까지 이득을 주지 않았다. **population f-LIF는 하한과 구별되지 않는다**(보정 없는 설정; 발화율 3%). 사용자가 제안한 "양방향 population f-LIF" 조합은 양방향 LIF보다 −0.5%p 나쁘다.
- 2블록으로 늘려도 1블록 LIF·f-LIF 수준(91.07)이고 에너지는 3.7배다.

### 한계
- 3 seed 탐색. population에 입력 크기 보정 없음(발화율 3%; 6단계와 같은 단서). 양방향 결합은 OR만. 하이퍼파라미터(lr, epoch, D)는 ECG 설정 그대로이며 주의 없는 모델에 맞춰 조정하지 않았다. 환자 간 일반화는 재지 않았다. population 에너지는 우리 규약.

### 점검
- 사전 점검 13/13(`results/checks/check_cls_stage7.txt`, commit `e05fc1b48`). 분석 forward 일치 18/18 비트 동일. 정순 population mixer의 인과성 3 run 모두 0.
- 감사 92(`docs/PROJECT_LOG.md` 12:51)는 f_lif_pop_v3 추적 감사로 새 판단 근거가 없다.

**다음 (사용자 결정):** 성능 쪽 0.2–0.7%p를 메우는 시도(mixer의 학습률·깊이·D 조정, OR 대신 두 방향을 이어 붙이기, 양방향 LIF로 ETT 예측 확장) 또는 여기서 닫기.

**commit/tag:** 이 항목·결과·분석·요약을 담은 commit에 태그 `exp/nsmt-flif-neo-20261019-3`를 붙이고 브랜치와 태그만 push한다.

---

## 2026-10-19 17:09 KST — 질문 기록: ETT 분석에 H720이 포함되었는가, "look-back 96 → horizon 720"은 기억을 쓰는가 (학습 없음, 4단계 기억 측정 재집계)

**질문 (사용자, 10-19):** "지금까지 나의 아이디어 검증 과정에서 일반적인 time-series forecasting 벤치마크 데이터셋에 대하여 (ETT 예측) prediction length=720으로 설정하고 수행한 결과를 바탕으로 분석을 한 것인지, 그리고 짧은 look-back window(예를 들어, 96)에 대하여 긴 prediction length(예를 들어, 720)을 예측하는 과제는 'memory'를 사용하지 않는지."

**답 1 — H720은 처음부터 포함되어 있었다.** 1–4단계의 셀은 ETTh2/ETTm2 × **H96/H720** × seed 42/7/2026이다(결정 11·12; 1차 설계 "72 run = 6 조건 × ETTh2/ETTm2 × H96/H720 × 3 seed", 2차 `flif-v1-s2p10`, 3단계 60 run, 4단계 156 run 모두 같은 4칸). 판정(기준 (1)(2), Q1–Q4, R1–R4)은 4칸 모두에서 했고, 어느 단계에서도 H720 칸만 따로 통과한 질문은 없었다. 5–7단계는 분류라 horizon이 없다.

**답 2 — look-back 96 → horizon 720은 "창 밖 기억"을 요구하지 않는다.** 입력은 horizon과 무관하게 96스텝뿐이고, 그 96스텝 전체가 head(flat)나 해마 attention에 그대로 보인다. 긴 horizon은 출력 쪽 외삽을 어렵게 할 뿐, 모델이 더 먼 과거를 기억해야 하는 요구를 만들지 않는다. 창 **안**에서 뉴런 기억이 쓰였는지는 4단계 기억 측정을 horizon별로 나눠 볼 수 있다. `nsmt_flif_v1/forecasting/probe_by_horizon.py`(새 스크립트, `results/flif-v1-s4/memory_probe_bestval/*.csv` 재집계, `results/flif-v1-s4/probe_by_horizon_keep12.csv`):

| 마지막 12토큰만 남길 때 MSE 증가 (%, 3 seed 평균, 최고 검증 가중치) | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| step·정렬·last·f-LIF | +0.52 | +0.04 | +10.51 | **+11.66** |
| step·정렬·last·LIF | +0.01 | 0.00 | +0.05 | +0.02 |
| patch·정렬·last·f-LIF | +0.44 | +0.08 | +2.91 | +4.25 |
| patch·정렬·last·LIF | −0.01 | 0.00 | +0.04 | +0.01 |
| step·정렬·flat·f-LIF / LIF | +25.33 / +28.60 | +5.60 / +7.07 | +49.31 / +57.81 | +27.78 / +27.84 |
| patch·정렬·flat·f-LIF / LIF | +10.00 / +8.72 | +5.25 / +4.40 | +26.44 / +23.19 | +13.07 / +12.88 |

- 마지막 막전위 읽기에서 **f-LIF는 H720에서도 12토큰보다 오래된 입력을 쓴다**(ETTm2 +11.7%; LIF 0.0%). 즉 기억은 horizon과 무관하게 측정된다. 그러나 4단계 R3에서 H720 칸을 포함해도 f-LIF가 LIF보다 나은 조건은 step·last(최고 검증 가중치)뿐이었다. 결론 "창 안 기억은 만들어지지만 96스텝 창에서는 이득이 없다"는 H96·H720 모두에서 같다.
- flat head 조건에서 H720의 증가분이 H96보다 작은 것(예: step·f-LIF 25.3 → 5.6)은 기억이 아니라 "먼 과거 입력을 지운 효과"가 긴 horizon 예측에서 덜 중요하다는 뜻이다. 긴 horizon일수록 예측이 창 평균·주기에 가까워지기 때문이다(선형 기준선이 H720에서 상대적으로 강한 것과 같은 현상).
- 기억이 **필요해지는** 예측 설정은 (i) 창을 길게 주되 head·attention이 그것을 직접 못 보게 하는 설계(사용자의 의도 2: 긴 저해상 이력을 신피질 축에만), 또는 (ii) 창 밖에 주기가 있는 데이터(Electricity·Traffic의 주 단위 주기 168 h vs 창 96 h; `nsmt_base/forecasting/scripts/upstream_electricity_96.sh`가 그 설정)다. ETT는 하루 주기(24 h / 96 × 15분)가 창 안에 들어 기억 요구가 약하다.

**commit:** 이 항목과 `probe_by_horizon.py`, 재집계 CSV를 commit·push한다(8단계 조사 commit과 함께).

---

## 2026-10-19 20:19 KST — 8단계 조사 결과: ECG 리듬 분류(CinC 2017·AFDB)와 SHD/SSC의 2023–2026 top-tier 선행 연구 프로토콜 조사 (학습 없음) — 공통 프로토콜과 채택안, 승인 대기

**요청 (사용자, 10-19):** 두 과제의 최근 약 3년 선행 연구를 학술 DB에서 전수 조사해 공통 실험 프로토콜을 확인하고 채택하라. 조사는 내가 직접(사용자의 첫 답 "CSV 제공"을 번복), **신뢰도 높은 국제 학회를 놓치지 말 것, top-tier 저널·학회만, MDPI 제외.** 경계 저널(B군: Scientific Reports, Computers in Biology and Medicine, Biomedical Signal Processing and Control, Physiological Measurement, Expert Systems with Applications, IEEE Access)은 **포함하되 분리 집계**(사용자 결정). ECG 범위는 CinC 2017 중심 + AFDB. 순서는 조사 → 프로토콜 확정 보고 → 승인 후 실험.

**방법 (`docs/surveys/`, 모두 Git; API 원본·PDF·전문은 `docs/surveys/raw/` 로컬):**
- 수집 `collect.py`: (1) 데이터셋 원 논문의 **인용 추적** — Semantic Scholar(`paper/{id}/citations`)와 OpenAlex(`filter=cites:`) 두 색인; 씨앗은 Clifford 2017(CinC 2017 대회 논문, 인용 887)과 Cramer 2020/2022(Heidelberg 데이터셋, 인용 363); (2) **키워드 검색** — OpenAlex 전문 검색, arXiv API, Crossref, PubMed. DBLP와 OpenReview는 이 서버에서 봇 차단(HTTP 202/403)으로 쓰지 못했고, ICLR·NeurIPS는 인용 추적과 arXiv로 잡았다. Semantic Scholar 검색은 키 없이 429라 쓰지 못했다(인용 endpoint는 됨). OpenAlex는 cursor 페이지가 키를 요구해 page 번호 방식으로 바꿨다.
- 장소 분류: A군 화이트리스트(계획 1.3) + B군 + 제외군(MDPI·Frontiers·Hindawi·CinC 프로시딩·워크숍) + 프리프린트 + 출처 불명. 빈 장소·LNCS 묶음은 Crossref DOI로 풀어 재분류. **내가 임의로 더한 장소(사용자 확인 필요):** A군에 Heart Rhythm·Europace·JAHA·TMLR·TCAD·DATE, B군에 ICASSP·Interspeech·TCAS-II. ASP-DAC가 DAC 규칙에 잘못 걸린 것을 발견해 제외했다(1편).
- 선별 `screen.py`: A/B군·주제 일치 후보의 공개 전문(arXiv → S2/OpenAlex 공개 PDF → PMC → Europe PMC XML)을 받아 데이터셋 언급을 세고(≥ 2 = 사용 후보), 문장 발췌와 규칙 힌트를 만들었다.
- 추출: 병렬 작업자(SHD 3, ECG 2)가 발췌·전문에서 항목(계획 1.4)을 채우고 근거 인용문을 적었다. `verify_quotes.py`로 인용문의 원문 존재를 기계 대조했고(SHD 290개 중 227개, ECG 202개 중 169개; 나머지는 PDF 열 배치로 끊긴 문자열이며 표본으로 실존 확인), 표본 2편은 내가 직접 원문과 대조했다. 집계 `report.py`(A군만 / A+B군 두 열), 수동 제외 `ecg_overrides.json`(4편, 이유 기록).

**건수 (PRISMA 식):**

| | SHD/SSC | ECG 리듬 |
|---|---|---|
| 수집(중복 제거, 2023–2026) | 2,416 | 5,578 |
| 주제 일치 A군 / B군 | 165 / 38 | 280 / 207 |
| 공개 전문 확보 A / B | 86 / 16 | 181 / 108 |
| 전문에서 데이터셋 실제 사용 A / B | 51 / 8 | 9 / 17 (+ 1회 언급 15편 추가 검토) |
| **추출 후 포함** A / B | **46 / 3** | **7 / 15** |
| 제외군(MDPI 등) 주제 일치 | 82 | 305 |
| 초록에 데이터셋이 보이나 전문 없음 | 11 | 16 |

**결과 1 — SHD/SSC (`docs/surveys/shd_ssc_protocols_2023_2026.md`, 포함 49편):** 시간 구간 폭은 명시한 논문 중 10 ms(6)·4 ms(5)·1 ms(5)가 축이고 절반(23/46)은 적지 않음; 스텝 수 100(10)·250(8); 채널 700 그대로(21)·140(6); **모델 선택 방식은 65%가 미기재**이고 적은 논문은 "test를 검증으로 씀"(6)·"학습 일부 분리"(5)·"화자 단위 교차검증"(2)으로 갈림; 증강 미기재 31, 사용 12(채널 이동·같은 클래스 섞기·사건 드롭); 모델은 재귀 SNN 19·순방향 14·지연 7; 뉴런은 LIF 18·**다중 구획/가지 12**·ALIF 8·이질 τ 5; seed 5개(10)·3개(6)·10개(4)·미기재 21; 보고는 평균±표준편차 24·단일 17. 출처: Nature Communications 9, Neuromorphic Computing and Engineering 7, ICLR 5, Neural Networks 4, NeurIPS 4, AAAI 3, TMLR 3.
**채택안(승인 전):** 700채널, **10 ms → 100스텝 기본 + 4 ms → 250스텝**, SHD는 **학습 10%를 떼어 검증(test는 마지막 1회)**, 증강 없음 기본 + 증강 조건 별도 행, **seed 5**, 지표 시험 정확도, 7단계 주의 없는 정렬 모델. 상세는 보고서 5절.

**결과 2 — ECG 리듬 (`docs/surveys/ecg_rhythm_protocols_2023_2026.md`, 포함 22편):** **가장 중요한 발견은 top-tier에 이 데이터셋을 쓰는 논문이 거의 없다는 것**이다. A군 전문 181편 중 CinC 2017·AFDB로 자기 모델을 평가한 논문은 7편이고, 나머지는 PTB-XL(19)·MIT-BIH 부정맥(8)·병원 자체 코호트다. CinC 2017 연구의 대부분은 B군(Scientific Reports 12편)과 제외군(305편)에 있다. A+B 22편 기준: 분할은 고정 홀드아웃(9)과 k-fold(8, k=5가 4편)가 반반, 공식 숨은 test 1; 분할 단위는 명시된 쪽에서 환자/기록 단위(11)가 다수(무작위 3은 모두 B군); 표본화율 300(7)·250(4); 입력 길이 10 s(4)·30 s/기록 전체(7+); 불균형은 보통 CE(13), 가중치·focal 0; 주 지표 F1 클래스 평균(12)·정확도(5)·AUROC(3), 잡음 포함 여부 명시 5편뿐; **seed 미기재 19/22**, 단일 값 16.
**채택안(승인 전):** CinC 2017 4클래스 v3, **기록 단위 층화 5-fold**(탐색은 fold 1 × seed 3, 확증은 5-fold), 125 Hz 재표본·0.5–40 Hz·z-score, **30 s(토큰 233) 기본 + 10 s(토큰 77) 길이 ablation**, CE+CAL 기본 + 보통 CE 조건, 주 지표 **F1(N·AF·Other 평균, 대회 점수)** 과 보조 지표, 기록 단위 평가; AFDB는 기록 단위 분할·10 s 조각으로 흐름 처리 확장 과제. 상세는 보고서 5절.

**판단에 영향을 주는 점:**
- ECG 쪽은 "관행 채택"의 근거가 약하다(A군 7편). 사용자가 top-tier만을 원한 취지로 보면, **CinC 2017은 top-tier가 쓰는 벤치마크가 아니다.** 대안은 (i) 그래도 CinC 2017을 쓰되 B군 관행을 따르고 그 사실을 명시, (ii) top-tier가 실제로 쓰는 PTB-XL(10 s, 12리드, 리듬 라벨 포함)로 바꾸기 — 단 리듬 라벨의 기억 요구는 CinC 2017보다 약하다(10 s 안 판정), (iii) AFDB 중심의 흐름 처리 과제. 결정은 사용자 몫이다.
- SHD/SSC 쪽은 표본이 충분하고 관행이 분명하다. 단 "모델 선택에 test를 쓰는" 논문이 있어, 우리가 검증 분할을 쓰면 상위 보고치와 0.5–1%p 정도 불리한 비교가 된다는 점을 표에 표시해야 한다.

**한계:** 유료 전문 미접근(SHD A군 79편, ECG A군 99편 초록만), DBLP·OpenReview 미사용, 자동 추출(인용문 대조 78–84%, 표본 수작업 확인), 장소 분류 일부 내 판단. 모두 보고서 5.4절에 적었다.

**commit:** 이 항목과 `docs/surveys/` 산출물(후보 목록·선별 표·추출 표·보고서·스크립트)을 commit·push한다. 실험은 승인 뒤 별도 사전 고정 항목으로 시작한다.

---

## 2026-10-19 21:50 KST — 9단계 설계·구현·사전 고정: 기억이 필요한 두 과제(CinC 2017 ECG 리듬, SHD)에서 NSMT · Hippo-only(Spikformer) · Neo-mixer 비교 — 점검 18/18, smoke 18/18, tmux에서 시작

**요청 (사용자, 10-19):** "1. 우선 CinC 혹은 MIT-BIH를 사용해서 보고하는 것으로 하자. PTB-XL은 10s 데이터이므로 memory 기능을 요구하지 않아. 2. 두 채택안의 승인. 하지만, 모델은 여러 구성으로 비교했으면 좋겠어. (1) 기본 NSMT (LIF/f-LIF/Population f-LIF) with T=N (2) Hippo-only (Spikformer와 유사한 Self-attention module 포함 트랜스포머 구조) (3) self-attention 대신 Neo network를 사용하는 NSMT 모델 (f-LIF/Population f-LIF) with T=N + bidirectional. 3. 확인" (3 = 8단계 보고서의 장소 분류 확인.)
**질문 답(AskUserQuestion):** Hippo-only = Spikformer식 SSA, T = 4; 추가 대조군 = 양방향 LIF mixer, 섞기 없음 하한; ECG 토큰 = 패치 64·보폭 32(N ≈ 117); population f-LIF = 보정 추가. 사용자 질문 "토큰 섞기 모듈이 뭐야?"에는 보고에서 답한다(토큰 사이에서 정보를 주고받게 하는 부분 — 트랜스포머의 자기주의, 7단계의 양방향 신피질망; 하한은 그것을 빼고 토큰을 따로 처리해 평균만 내는 모델).
**계획 승인:** 첫 승인 화면에는 계획 파일 쓰기 실패로 예전(8단계) 계획이 표시되었다. 이를 알리고 9단계 계획을 다시 써서 승인받았다(`/home/yschoi/.claude/plans/1-snuggly-galaxy.md`).

**브랜치·기준:** `exp/nsmt-flif-neo`, 기준 commit `6cb8cdc27`.

**데이터 (로컬, Git 제외):**
- CinC 2017: `training2017.zip`(99,226,822 B, md5 `7a220c80e10881bad86045e8ce97b49d`, sha256 `25fdbab3…cd1255`) + `REFERENCE-v3.csv`(md5 `238a81b4dc5572428a247bd485c00c12`). physionet.org 직접 받기가 초당 약 5 KB로 느려 PhysioNet 공식 AWS 공개 미러(`physionet-open.s3.amazonaws.com/challenge-2017/1.0.0/`)에서 받았다(크기가 원본과 같음; 미러에는 SHA256SUMS가 없음). 8,528기록, v3 라벨 N 5,076 / AF 758 / Other 2,415 / 잡음 279. 전처리 캐시 `cinc2017_v3_125hz.npz`(md5 `e6d4c804…`): 125 Hz(polyphase 5/12), 0.5–40 Hz 4차 Butterworth 양방향, 기록 z-score; 길이 9.0–61.0 s(중앙값 30 s).
- SHD: `shd_train.h5.gz`(md5 `d47c9825…`), `shd_test.h5.gz`(md5 `3062a80e…`), Zenke lab. snn_recall에 h5py가 없어 시스템 `/usr/bin/python3`(h5py 2.10)로 `shd_convert.py`를 한 번 실행 → `shd_bin10ms_1s.npz`(md5 `c5c09f6b…`): 10 ms·처음 1 s·100 × 700 스파이크 수(구간 최대 6), 학습 8,156 / 시험 2,264, 20클래스.
- 분할: CinC 기록 단위 층화 5-fold(random_state 0) → fold 0 = 학습 6,139 / 검증 683 / 시험 1,706기록(시험 라벨 1,016 / 151 / 483 / 56); SHD 학습의 10%(816)를 seed마다 층화로 검증에 뗌.

**코드 (`nsmt_flif_v1/classification/`; MIT-BIH 경로 불변 — c1·c2 비트 동일):**
- `model.py`: 다채널 토큰(`in_channels`, 토큰 = P 스텝 × C 채널, 채널 우선 펼침; C = 1·hippo_ssa 아님이면 기존 경로), `hippo_ssa`·`hippo_T`(해마 반복 수를 N과 분리, Block이 `layers.SSA_rel_scl`을 쓰고 교차 주의·신피질·디코더를 만들지 않음).
- `neo_mixer.py`: `tokens()` 다채널, `in_channels`·`pop_soma_init`. `layers.py`: `build_neo_neuron(..., pop_soma_init)` — 생성 직후 soma 혼합 가중치를 지정 값으로(9단계 1.0, 기본은 1/K로 6–7단계 재현 유지). `neurons.py` 불변(c8).
- 새 파일: `rhythm_data.py`(전처리·fold·학습 무작위 30 s 자르기·평가 30 s/15 s 창·SHD 로더·사전 검사용 k토큰 자르기), `rhythm_eval.py`(기록 단위 softmax 평균, CinC 점수 = F1_N·F1_AF·F1_Other 평균, SHD 정확도, 에너지·syops), `train_rhythm.py`, `shd_convert.py`, `summarize_rhythm.py`(관문·학습률 결정과 판정), `rhythm_analysis.py`(토큰 축 기억 측정, 보고만), `scripts/{run_rhythm.sh, make_queue_rhythm.sh, rhythm_driver.sh}`, `check_cls.py` c14–c18.
- **계획과 다른 점:** 학습 진입점을 `train.py` 분기가 아니라 새 파일 `train_rhythm.py`로 만들었다(MIT-BIH 경로를 확실히 그대로 두기 위해). 같은 손실 형태(신피질이 있으면 CE·0.9 + 보조 MSE·0.1, 없으면 CE)와 AdamW·timm cosine을 쓰고, 다른 점은 (i) 검증·시험 분리와 최고 검증 epoch의 test 1회(마지막 epoch는 `log/last/`에 보고용), (ii) RNG를 한 번만 고정하고 epoch마다 셔플·자르기가 달라짐(`train.py`는 매 epoch 다시 고정해 같은 순서를 반복). ECG 설정 파일의 `--weight_decay`가 정수형이라 `1e-4`를 받지 못해 첫 smoke가 모두 실패했고(`log/smoke-rhythm-20261019/failed_argparse/`), 실수형으로 고쳤다(기본값 그대로).

**점검 18/18 (`results/checks/check_cls_stage9.txt`):** c1–c13 그대로 통과. c14 다채널 토큰(C = 1에서 원래 경로와 입력 동일·출력 차이 0, C = 7에서 모든 토큰이 수동 조각과 일치, NSMT·mixer). c15 Hippo-only SSA(신피질·교차 주의·디코더 없음, SSA 입력 [4, B, 117, 384], 상대 위치 표 [233, 8]). c16 population 보정(soma 가중치 1.0 / 기본 0.25; **학습 모드 정규화에서 실제 심박 첫 발화율: 보정 없음 0.0% / 보정 3.7–3.8% / LIF 4.5–8.0%** — 처음에는 평가 모드로 재서 모든 뉴런이 0이라 검사가 성립하지 않았고, 학습과 같은 배치 통계 정규화로 고쳤다; 6–7단계 population의 낮은 발화율이 이 초기 침묵 때문이었음을 보여 준다). c17 CinC(8,528기록, v3 개수 일치, fold 서로소·전체 덮음, 창 평균 집계 = 수동 계산). c18 SHD(표본 수·20클래스, 구간 합 = 변환기가 남긴 스파이크 수, 검증 서로소·모든 클래스 포함).

**smoke (`results/smoke-rhythm-20261019/raw.txt`, 1 epoch, seed 42, lr 1e-3; 실험 아님 — 1 epoch에서는 어느 모델도 AF를 맞히지 못함):**

| 조건 | CinC 점수 | CinC 총 µJ (FFT) | CinC 시간 s | SHD 정확도 | SHD 총 µJ (FFT) | SHD 시간 s | 파라미터 (CinC / SHD) |
|---|---|---|---|---|---|---|---|
| nsmt_lif | 0.325715 | 2,900.9 | 259 | 0.447438 | 26,032.2 | 228 | 1,262,592 / 2,807,808 |
| nsmt_flif | 0.328383 | 2,907.5 | 305 | 0.454947 | 26,034.2 | 281 | 〃 |
| nsmt_pop | 0.312098 | 2,857.9 | 318 | 0.447438 | 26,095.8 | 288 | 1,267,200 / 2,812,416 |
| nsmt_lif_ce | 0.350622 | 2,858.1 | 258 | – | – | – | 1,262,592 |
| spk_ssa | 0.378778 | 107.6 | 34 | 0.459806 | 1,048.3 | 40 | 918,088 / 1,436,984 |
| mix_flif_bidir | 0.289900 | 29.5 | 151 | 0.441696 | 263.0 | 150 | 916,224 / 1,435,392 |
| mix_pop_bidir | 0.325180 | 63.1 | 171 | 0.529152 | 292.3 | 168 | 922,368 / 1,441,536 |
| mix_lif_bidir | 0.283584 | 24.4 | 87 | 0.453622 | 259.2 | 81 | 916,224 / 1,435,392 |
| mix_lif_bidir_ce | 0.315720 | 24.7 | 88 | – | – | – | 916,224 |
| mix_none | 0.281345 | 17.6 | 22 | 0.453622 | 251.6 | 27 | 323,328 / 842,496 |

- **에너지 규모(사전 기록):** T = N인 NSMT 해마가 에너지를 지배한다(CinC 2.9 mJ, SHD 26 mJ — SHD는 토큰 차원 1,400의 실수 입력이 반복 축 100번 곱해짐). Spikformer(T = 4)는 그 1/25, Neo-mixer는 1/100 수준이다.
- **비용:** NSMT는 배치 32에서 GPU 메모리 34–35 GB(CinC)·26–27 GB(SHD), epoch당 약 140 s / 130 s. 그래서 NSMT는 GPU당 1개, 가벼운 조건은 두 번째 pool로 같은 GPU에 함께 돌린다.

**실행 (사전 고정):** tmux `flif_rhythm`, `scripts/rhythm_driver.sh 0,1,2,3`.
- A. 병렬로 (i) **학습률 선택**: `nsmt_lif`, seed 42, lr ∈ {1e-3, 3e-3}, 두 데이터(suite `flif-v1-rhythm-lr`), (ii) **기억 필요 사전 검사(정지 관문)**: `mix_lif_bidir`·`mix_none`을 입력 8토큰·32토큰·전체로 학습(lr 1e-3, seed 42, suite `flif-v1-rhythm-crop`; CinC는 학습 무작위 자르기·평가 기록당 가운데 창 1개, SHD는 스파이크 무게중심을 가운데로 한 k구간).
- B. `summarize_rhythm.py --decide`: 학습률 = 최고 검증 점수가 높은 쪽(동률·누락이면 1e-3). **관문 = `mix_lif_bidir`의 최고 검증 점수가 전체 입력에서 8토큰보다 1%p 이상 높아야 통과; 실패하거나 run이 없으면 그 데이터의 본 학습을 하지 않고 보고한다.**
- C. 관문 통과 데이터만 본 학습: CinC 10조건 × seed 42/7/2026(fold 0, suite `flif-v1-rhythm-cinc`) = 30 run, SHD 8조건 × seed 42/7/2026/123/1024(suite `flif-v1-rhythm-shd`) = 40 run.
- D. 요약 `results/<suite>/summary.md`. 분석 `rhythm_analysis.py`(토큰 축 기억)는 본 학습 뒤 별도로 돌린다.
- **예상 시간(smoke로 갱신):** A 약 2시간, C 약 12시간(NSMT 27 run × 1.6–1.8 h를 GPU 4장에서 하나씩, 가벼운 43 run을 함께). 계획의 추정(2.5 h + 4 h)보다 2–3배 길다 — T = N = 100–117 해마의 비용.

**판정 (계획 7절 그대로; 같은 seed끼리 짝):** "좋음" = 평균 ≥ +0.5%p이고 A > B인 seed가 CinC 3개 중 2개 이상, SHD 5개 중 4개 이상.
- R1 기억 요구: 각 모델 vs `mix_none`.
- R2 NSMT 안의 뉴런: `nsmt_flif`·`nsmt_pop` vs `nsmt_lif`.
- R3 Neo-mixer 안의 뉴런: `mix_flif_bidir`·`mix_pop_bidir` vs `mix_lif_bidir`.
- R4 원래 기준 (2): Neo-mixer·NSMT vs `spk_ssa` — "좋음" 또는 "−0.5%p 이내이고 총 에너지 ≤ 0.5배"면 충족.
- R5(보고만): CAL vs 보통 CE, 토큰 축 기억, 발화율, 에너지.
- **예상:** 두 데이터 모두 하한이 무너진다. Neo-mixer는 Spikformer(T = 4)보다 에너지가 1/4–1/2이고, 성능은 불확실하다. NSMT는 에너지가 Spikformer의 25배라 R4 에너지 조건을 만족하지 못한다.

**8단계 장소 분류 확인(사용자 3번):** 사용자가 "확인"했다 — A군에 Heart Rhythm·Europace·JAHA·TMLR·TCAD·DATE, B군에 ICASSP·Interspeech·TCAS-II를 넣은 분류를 그대로 쓴다.

**감사:** 96(`docs/PROJECT_LOG.md` 20:31)은 f_lif_pop_v3 추적 감사로 새 판단 근거가 없다.

**commit:** 이 항목·코드·점검·smoke를 commit·push한 뒤 시작한다. 결과는 다음 항목.

---

## 2026-10-19 23:50 KST — 9단계 중간 기록: 학습률 선택과 기억 필요 관문 (A단계 16 run) — 두 데이터 모두 관문 통과(+22.5%p / +20.3%p), 학습률 1e-3; 본 학습 70 run 23:47 시작

**실행:** tmux `flif_rhythm`, 21:49–23:47, 16/16 완료·실패 0. 결정 `results/flif-v1-rhythm-crop/decision.json`(`summarize_rhythm.py --decide`), 원자료 `results/flif-v1-rhythm-{lr,crop}/raw.txt`. **결정은 검증 점수만으로 했다.** 이 run들도 파이프라인대로 끝에 test를 1회 계산했으므로 아래에 함께 적되, 결정에는 쓰지 않았다.

| 데이터 | run | 검증 최고(epoch) | test(최고 검증 가중치) | 시냅스 µJ | 시간 s |
|---|---|---|---|---|---|
| CinC | nsmt_lif lr 1e-3 | **0.665489** (23) | 0.619527 | 2,959.5 | 4,948 |
| CinC | nsmt_lif lr 3e-3 | 0.662513 (24) | 0.633112 | 3,173.7 | 4,524 |
| SHD | nsmt_lif lr 1e-3 | 0.982843 (34) | 0.818021 | 26,102.8 | 7,112 |
| SHD | nsmt_lif lr 3e-3 | 0.982843 (23) | 0.840106 | 26,155.9 | 6,876 |

- **학습률:** CinC는 1e-3이 검증에서 높아(0.6655 vs 0.6625) 1e-3. SHD는 검증 점수가 같아(0.982843 = 802/816) **사전 규칙대로 동률 → 1e-3.** test로는 두 데이터 모두 3e-3이 높았지만(+1.4%p, +2.2%p) 규칙상 쓰지 않는다.

| 데이터 | 모델 | 입력 8토큰 | 32토큰 | 전체(117 / 100) |
|---|---|---|---|---|
| CinC (검증) | mix_lif_bidir | 0.434027 | 0.546082 | **0.659214** |
| CinC (검증) | mix_none(하한) | 0.423530 | 0.533167 | 0.600610 |
| CinC (test) | mix_lif_bidir / mix_none | 0.449833 / 0.419357 | 0.553558 / 0.539232 | 0.602091 / 0.588476 |
| SHD (검증) | mix_lif_bidir | 0.770833 | 0.950980 | **0.974265** |
| SHD (검증) | mix_none(하한) | 0.729167 | 0.924020 | 0.954657 |
| SHD (test) | mix_lif_bidir / mix_none | 0.520318 / 0.515901 | 0.720406 / 0.690371 | 0.810071 / 0.763693 |

- **관문:** `mix_lif_bidir`의 검증 점수가 전체 입력에서 8토큰보다 CinC +22.5%p, SHD +20.3%p 높다(기준 1%p) → **두 데이터 모두 통과.** 입력이 길수록 점수가 꾸준히 오르므로, 두 과제는 긴 문맥을 실제로 요구한다(5–7단계 MIT-BIH 심박과 다름).
- 같은 길이에서 양방향 LIF mixer가 하한보다 높다(전체 입력 검증 CinC +5.9%p, SHD +2.0%p; seed 1개, 본 판정 아님).
- **주의할 점(보고):** (i) SHD는 검증(학습 화자 중 10%)과 test(학습에 없는 화자 2명 포함)의 차이가 크다(0.97 vs 0.81). 8단계 조사에서 본 "test로 모델을 고르는" 관행과 우리 규칙의 차이가 여기서 드러난다. (ii) CinC test 점수 0.60–0.63은 8단계 조사의 B군 보고치(F1 0.79–0.84, 대부분 CNN·RNN, 분할이 다름)보다 크게 낮다. 우리 모델이 작은 SNN이고 학습 설정을 조정하지 않은 영향으로 보이며(미확인), 비교의 기준은 같은 분할의 대조군들이다.

**본 학습:** 23:47 시작, 무거운 27 run(NSMT, GPU당 1개) + 가벼운 43 run(두 번째 pool). A단계의 run당 시간(CinC NSMT 약 1.3 h, SHD NSMT 약 2 h)으로 보면 끝나는 시각은 10-20 오전 11시–오후 1시 무렵이다.

**commit:** 이 항목과 A단계 결과를 commit·push한다.

---

## 2026-10-20 16:18 KST — 9단계 결과 (CinC 2017 30 run, SHD 40 run, 최고 검증 가중치): **두 과제 모두 Spikformer식 해마 단독(T = 4)이 가장 좋고, 기준 (2)는 12개 비교 모두 미충족 — 가장 가까운 SHD 양방향 population mixer는 −0.54%p·에너지 0.27배로 −0.5%p 경계에서 미달; population f-LIF는 SHD에서 NSMT·mixer 모두 LIF보다 좋음(+5.95 / +5.82%p, 5/5), CinC에서는 차이 없음** — 태그 `exp/nsmt-flif-neo-20261020`

**구분:** 사전 고정한 9단계 **탐색**의 완료 기록(CinC fold 0 × 3 seed, SHD 5 seed). 코드·사전 고정 commit `fe9686c8c`, 중간 기록 `239746ca1`·`9094bfaf9`. 이 항목이 속한 commit은 태그 `exp/nsmt-flif-neo-20261020`로 식별한다.

**사용자 질문 (10-20, 학습 중):** "'Mixer'의 뜻이 뭐야?" → 토큰 섞기 모듈(조각 사이에서 정보를 주고받게 하는 부분; MLP-Mixer 논문의 token-mixing에서 온 말), Neo-mixer = 신피질망을 토큰 섞기 모듈로 쓴 7단계 모델이라고 답했다. 이 이름을 여러 단계에서 설명 없이 썼다는 점을 인정하고, 앞으로 모델 이름에 한 줄 설명을 붙이기로 했다. "Neo-mixer를 갖는 모델의 입력에서부터 출력까지의 흐름" → 코드(`neo_mixer.py`, `neurons.py`, `rhythm_data.py`)를 다시 읽고 조각 자르기 → 임베딩(LIF) → 양방향 신피질망(OR) → `x·(1−m)` → MLP → 조각 평균 → 분류를 단계별로 설명했다. 그 설명의 요점 중 결과 해석에 쓰는 것: 섞기 없음 하한에도 임베딩·MLP의 LIF 막전위(τ 2)와 마지막 평균이라는 두 길이 남아 있으므로, 하한과의 차이는 "섞기 모듈 막전위로 더 전달된 정보"만 잰다.

**실행:** tmux `flif_rhythm`, `scripts/rhythm_driver.sh 0,1,2,3`(큐 `scripts/queues/rhythm-main-{heavy,light}.txt`). 본 학습 10-19 23:47 – 10-20 15:52, **70/70 완료·실패(SKIP) 0**. 명령 형식 `DATA=<CINC2017|SHD> COND=<조건> SUITE=flif-v1-rhythm-<cinc|shd> LR=0.001 bash nsmt_flif_v1/classification/scripts/run_rhythm.sh <seed> <gpu>`. 모든 run의 코드는 같다(`fe9686c8c`와 `9094bfaf9` 사이 `.py`·`.sh` 변화 없음; RESULT 줄의 sha는 8 run `fe9686c8c`, 62 run `9094bfaf9`).
- 환경: conda `snn_recall`(Python 3.10.18, torch 1.12.0+cu113, timm 1.0.19, numpy 1.26.4, scipy 1.15.3, scikit-learn 1.7.1), RTX A6000 × 4.
- 하이퍼파라미터(사전 고정 그대로): D 384, head 8, mlp 1, 블록 1, 신피질 층 2, keep_ratio 0.15, 보조 비중 0.1(NSMT만), τ 2, AdamW lr 1e-3·wd 1e-4, timm cosine, 배치 32; CinC 패치 64(117토큰)·30 epoch·CAL 가중 CE, SHD 패치 2(100토큰)·50 epoch·CE; f-LIF α 0.5·pred·전체 이력; population 가지 τ 4/8/16/32·soma τ 2·soma 가중치 시작값 1.0; Spikformer식 T_h 4.
- seed: CinC 42/7/2026(fold 0), SHD 42/7/2026/123/1024.
- 산출물(Git): `nsmt_flif_v1/classification/results/flif-v1-rhythm-{cinc,shd}/{raw.txt, summary.md, reach_summary.md, reach/}`, 실행별 `log/flif-v1-rhythm-{cinc,shd}/<DATA>/<날짜>/<run>/<seed+조건>/{logargs.txt, log/{final+result.csv, best_log_0.csv, best_epoch.txt, confusion_matrix.csv, neuron_energy.csv, model+info+per+layer.txt, last/…}}`, 결정 사본 `log/rhythm-decision.txt`. 로컬만: checkpoint(`model_state/*.pt, *.pth`), `test_probs.npz`, TensorBoard 이벤트, stdout, 드라이버·pool·분석 기록(`log/flif-v1-rhythm*.log`).

### CinC 2017 (fold 0 시험 1,706기록, 3 seed 평균 ± 표준편차, %)

CinC 점수 = F1(정상)·F1(AF)·F1(Other)의 평균. 시험은 기록마다 30 s 창을 15 s 간격으로 넣어 확률을 평균한 기록 단위 예측.

| 조건 | CinC 점수 | F1 정상 / AF / Other / 잡음 | AF 민감도 / 특이도 | 정확도 | 총 µJ (FFT / 직접 합) | 섞기·신피질 발화율 | 파라미터 | 시간 s |
|---|---|---|---|---|---|---|---|---|
| **Spikformer식 해마 단독 (T=4)** | **65.60 ± 1.48** | 82.31 / 59.78 / 54.70 / 50.97 | 55.63 / 97.06 | 72.51 ± 0.72 | 127.36 / 127.36 | – | 918,088 | 391–413 |
| NSMT, LIF | 61.76 ± 1.40 | 81.29 / 54.51 / 49.49 / 50.73 | 47.46 / 97.41 | 70.01 ± 0.24 | 3,419.30 / 3,419.30 | 0.0792 | 1,262,592 | 4,966–5,414 |
| NSMT, f-LIF | 61.50 ± 1.20 | 82.04 / 53.74 / 48.73 / 49.83 | 48.57 / 96.89 | 70.44 ± 0.28 | 3,480.00 / 3,519.27 | 0.0881 | 1,262,592 | 7,277–7,289 |
| NSMT, population f-LIF (보정) | 60.94 ± 2.24 | 79.37 / 55.78 / 47.67 / 37.40 | 49.89 / 97.19 | 68.29 ± 1.27 | 3,247.40 / 3,398.52 | 0.0398 | 1,267,200 | 6,814–7,305 |
| Neo-mixer 양방향, population f-LIF | 60.73 ± 1.67 | 79.61 / 55.38 / 47.20 / 40.78 | 50.55 / 96.91 | 68.29 ± 1.47 | 65.20 / 266.71 | 0.0873 | 922,368 | 7,278–7,289 |
| Neo-mixer 양방향, LIF | 60.54 ± 0.56 | 80.97 / 52.64 / 48.02 / 42.65 | 50.55 / 95.97 | 69.42 ± 1.26 | 27.99 / 27.99 | 0.0669 | 916,224 | 3,201–4,436 |
| Neo-mixer 양방향, f-LIF | 59.75 ± 0.39 | 81.52 / 51.49 / 46.25 / 42.49 | 47.90 / 96.29 | 69.64 ± 0.26 | 32.84 / 85.19 | 0.0576 | 916,224 | 7,089–7,227 |
| NSMT, LIF, 보통 CE | 59.22 ± 0.29 | 80.89 / 46.79 / 49.99 / 21.13 | 35.98 / 98.29 | 69.05 ± 1.07 | 3,403.07 / 3,403.07 | 0.0823 | 1,262,592 | 5,042–5,505 |
| Neo-mixer 양방향 LIF, 보통 CE | 58.68 ± 0.76 | 81.03 / 45.22 / 49.78 / 13.31 | 35.98 / 97.77 | 69.25 ± 1.09 | 28.27 / 28.27 | 0.0695 | 916,224 | 4,179–4,516 |
| 섞기 없음 하한 | 58.20 ± 1.94 | 80.56 / 50.73 / 43.30 / 42.53 | 47.24 / 96.21 | 68.60 ± 0.85 | 18.66 / 18.66 | – | 323,328 | 206–218 |

**사전 고정 판정 (CinC 점수, 같은 seed끼리 짝, seed 42 / 7 / 2026, %p; "좋음" = 평균 ≥ +0.5이고 3개 중 2개 이상):**

| 질문 | A − B | 평균 | 판정 | 에너지 A/B | 기준 (2) |
|---|---|---|---|---|---|
| R1 NSMT LIF − 하한 | +3.11 / +4.26 / +3.33 | +3.57 | 좋음 | 183.21 | |
| R1 NSMT f-LIF − 하한 | +1.28 / +6.03 / +2.61 | +3.31 | 좋음 | 186.46 | |
| R1 NSMT population − 하한 | +4.16 / +5.25 / −1.18 | +2.74 | 좋음 | 173.99 | |
| R1 Spikformer식 − 하한 | +8.41 / +8.40 / +5.39 | +7.40 | 좋음 | 6.82 | |
| R1 mixer f-LIF − 하한 | +1.31 / +3.37 / −0.02 | +1.56 | 좋음 | 1.76 | |
| R1 mixer population − 하한 | +0.09 / +5.00 / +2.52 | +2.53 | 좋음 | 3.49 | |
| R1 mixer LIF − 하한 | +1.36 / +4.21 / +1.47 | +2.35 | 좋음 | 1.50 | |
| R2 NSMT f-LIF − NSMT LIF | −1.82 / +1.78 / −0.73 | −0.26 | 차이 없음 | 1.02 | |
| R2 NSMT population − NSMT LIF | +1.05 / +0.99 / −4.51 | −0.83 | 차이 없음 | 0.95 | |
| R3 mixer f-LIF − mixer LIF | −0.05 / −0.84 / −1.48 | −0.79 | **나쁨** | 1.17 | |
| R3 mixer population − mixer LIF | −1.27 / +0.79 / +1.05 | +0.19 | 차이 없음 | 2.33 | |
| R4 mixer f-LIF − Spikformer식 | −7.10 / −5.03 / −5.41 | −5.84 | **나쁨** | 0.26 | 미충족 |
| R4 mixer population − Spikformer식 | −8.32 / −3.40 / −2.88 | −4.87 | **나쁨** | 0.51 | 미충족 |
| R4 mixer LIF − Spikformer식 | −7.05 / −4.19 / −3.93 | −5.05 | **나쁨** | 0.22 | 미충족 |
| R4 NSMT LIF − Spikformer식 | −5.30 / −4.14 / −2.06 | −3.83 | **나쁨** | 26.85 | 미충족 |
| R4 NSMT f-LIF − Spikformer식 | −7.13 / −2.37 / −2.79 | −4.09 | **나쁨** | 27.32 | 미충족 |
| R4 NSMT population − Spikformer식 | −4.25 / −3.15 / −6.57 | −4.66 | **나쁨** | 25.50 | 미충족 |
| R5 (보고) NSMT LIF: CAL − 보통 CE | +3.07 / +0.86 / +3.70 | +2.54 | 좋음 | 1.00 | |
| R5 (보고) mixer LIF: CAL − 보통 CE | +0.95 / +1.27 / +3.38 | +1.87 | 좋음 | 0.99 | |

- R2–R4에서 통과 후보가 없으므로 **사전 규칙대로 fold 1–4 확증은 하지 않는다.**
- CAL 가중의 이득은 주로 잡음(F1 13–21 → 43–51)과 AF(F1 45–47 → 53–55)에서 온다.
- 가장 좋은 점수 65.60도 8단계 조사의 보고치(0.79–0.84, 분할이 다름)보다 크게 낮다. 가장 약한 곳은 AF(F1 51–60, 민감도 47–56%)다.

### SHD (시험 2,264개, 5 seed 평균 ± 표준편차, %)

| 조건 | 시험 정확도 | macro-F1 | 검증 정확도(최고 epoch) | 총 µJ (FFT / 직접 합) | 섞기·신피질 발화율 | 파라미터 | 시간 s |
|---|---|---|---|---|---|---|---|
| **Spikformer식 해마 단독 (T=4)** | **86.75 ± 1.32** | 86.41 ± 1.35 | 97.82 | 1,071.24 / 1,071.24 | – | 1,436,984 | 775–1,079 |
| NSMT, population f-LIF (보정) | 86.32 ± 0.92 | 85.46 ± 1.04 | 98.87 | 26,508.12 / 26,616.32 | 0.1197 | 2,812,416 | 10,723–11,161 |
| Neo-mixer 양방향, population f-LIF | 86.21 ± 0.68 | 85.75 ± 0.66 | 98.63 | 294.58 / 438.85 | 0.1243 | 1,441,536 | 11,055–12,080 |
| NSMT, f-LIF | 81.34 ± 0.61 | 81.16 ± 0.62 | 98.21 | 26,484.08 / 26,512.40 | 0.0761 | 2,807,808 | 10,952–11,148 |
| Neo-mixer 양방향, LIF | 80.39 ± 0.85 | 80.29 ± 0.86 | 97.50 | 263.59 / 263.59 | 0.0740 | 1,435,392 | 5,012–7,056 |
| NSMT, LIF | 80.37 ± 1.05 | 80.13 ± 1.10 | 98.24 | 26,484.59 / 26,484.59 | 0.0825 | 2,807,808 | 7,105–8,444 |
| Neo-mixer 양방향, f-LIF | 78.71 ± 1.47 | 78.55 ± 1.45 | 97.35 | 267.65 / 305.41 | 0.0695 | 1,435,392 | 10,486–11,039 |
| 섞기 없음 하한 | 76.79 ± 1.07 | 76.72 ± 1.14 | 95.78 | 253.01 / 253.01 | – | 842,496 | 470–738 |

**사전 고정 판정 (시험 정확도, seed 42 / 7 / 2026 / 123 / 1024, %p; "좋음" = 평균 ≥ +0.5이고 5개 중 4개 이상):**

| 질문 | A − B | 평균 | 판정 | 에너지 A/B | 기준 (2) |
|---|---|---|---|---|---|
| R1 NSMT LIF − 하한 | +5.43 / +2.87 / +2.21 / +3.40 / +3.98 | +3.58 | 좋음 | 104.68 | |
| R1 NSMT f-LIF − 하한 | +5.70 / +4.37 / +2.74 / +3.53 / +6.40 | +4.55 | 좋음 | 104.68 | |
| R1 NSMT population − 하한 | +8.92 / +8.44 / +9.67 / +8.57 / +12.01 | +9.52 | 좋음 | 104.77 | |
| R1 Spikformer식 − 하한 | +11.44 / +9.45 / +7.60 / +10.73 / +10.56 | +9.96 | 좋음 | 4.23 | |
| R1 mixer f-LIF − 하한 | +0.04 / +2.65 / +1.19 / +2.52 / +3.18 | +1.92 | 좋음 | 1.06 | |
| R1 mixer population − 하한 | +10.73 / +8.48 / +8.66 / +7.82 / +11.40 | +9.42 | 좋음 | 1.16 | |
| R1 mixer LIF − 하한 | +4.64 / +3.71 / +2.16 / +1.55 / +5.92 | +3.60 | 좋음 | 1.04 | |
| R2 NSMT f-LIF − NSMT LIF | +0.27 / +1.50 / +0.53 / +0.13 / +2.43 | +0.97 | **좋음** | 1.00 | |
| R2 NSMT population − NSMT LIF | +3.49 / +5.57 / +7.46 / +5.17 / +8.04 | +5.95 | **좋음** | 1.00 | |
| R3 mixer f-LIF − mixer LIF | −4.59 / −1.06 / −0.97 / +0.97 / −2.74 | −1.68 | **나쁨** | 1.02 | |
| R3 mixer population − mixer LIF | +6.10 / +4.77 / +6.49 / +6.27 / +5.48 | +5.82 | **좋음** | 1.12 | |
| R4 mixer f-LIF − Spikformer식 | −11.40 / −6.80 / −6.40 / −8.22 / −7.38 | −8.04 | **나쁨** | 0.25 | 미충족 |
| R4 mixer population − Spikformer식 | −0.71 / −0.97 / +1.06 / −2.92 / +0.84 | **−0.54** | 차이 없음 | 0.27 | **미충족(경계)** |
| R4 mixer LIF − Spikformer식 | −6.80 / −5.74 / −5.43 / −9.19 / −4.64 | −6.36 | **나쁨** | 0.25 | 미충족 |
| R4 NSMT LIF − Spikformer식 | −6.01 / −6.58 / −5.39 / −7.33 / −6.58 | −6.38 | **나쁨** | 24.72 | 미충족 |
| R4 NSMT f-LIF − Spikformer식 | −5.74 / −5.08 / −4.86 / −7.20 / −4.15 | −5.41 | **나쁨** | 24.72 | 미충족 |
| R4 NSMT population − Spikformer식 | −2.52 / −1.02 / +2.08 / −2.16 / +1.46 | −0.43 | 차이 없음 | 24.75 | 미충족 |

- **경계 사례:** SHD 양방향 population mixer − Spikformer식의 평균은 전체 정밀도로 **−0.538840%p**다. 기준 (2)의 성능 조건(−0.5%p 이내)을 0.04%p 차이로 넘지 못한다. 에너지 조건(0.27배 ≤ 0.5배)은 만족한다. 사전 규칙대로 "미충족"으로 둔다. seed별 차이의 표준편차(1.61%p)가 이 0.04%p보다 훨씬 커서, 이 판정은 seed 구성에 민감하다.
- NSMT population − Spikformer식도 −0.43%p(차이 없음)이지만 에너지가 24.75배라 기준 (2)와 무관하다.
- 검증(학습 화자의 10%)과 시험(학습에 없는 화자 포함)의 차이가 모든 조건에서 10–20%p다.

### 토큰 축 기억 (보고만; `rhythm_analysis.py`, `results/flif-v1-rhythm-{cinc,shd}/reach_summary.md`)

입력 토큰 하나를 다른 표본의 같은 위치 토큰으로 바꿨을 때, d토큰 뒤(음수는 앞) 토큰 섞기 망의 출력 스파이크가 바뀌는 비율(%)이다. NSMT는 신피질망, Neo-mixer는 첫 블록의 두 방향을 합친 출력을 잰다. 최고 검증 가중치, 시험 앞쪽 512단위, seed 평균(표준편차는 `reach_summary.md`). 실행 `bash nsmt_flif_v1/classification/scripts/run_rhythm_analysis.sh 0,1,2,3 2`(tmux `flif_reach`, 15:57–16:17, **48/48 완료·실패 0**). 토큰 간격은 CinC 0.256 s(d=10 → 2.6 s, d=50 → 12.8 s), SHD 10 ms(d=10 → 0.1 s, d=50 → 0.5 s).

| 조건 | CinC d=0 | d=5 | d=10 | d=50 | d=−10 | SHD d=0 | d=5 | d=10 | d=50 | d=−10 |
|---|---|---|---|---|---|---|---|---|---|---|
| NSMT, LIF | 15.40 | 4.49 | 1.65 | 0.01 | 0 | 6.57 | 2.10 | 0.96 | 0.01 | 0 |
| NSMT, f-LIF | 14.53 | 4.24 | 2.84 | 0.81 | 0 | 7.58 | 2.32 | 1.68 | 0.31 | 0 |
| NSMT, population | 5.32 | 6.32 | 6.05 | **4.92** | 0 | 3.27 | 5.43 | 5.99 | **7.15** | 0 |
| mixer, LIF | 19.15 | 4.33 | 1.16 | 0.01 | 0.25 | 10.13 | 1.41 | 0.40 | 0.00 | 0.11 |
| mixer, f-LIF | 16.74 | 4.28 | 1.34 | 0.16 | 0.55 | 11.28 | 1.61 | 0.56 | 0.02 | 0.35 |
| mixer, population | 15.99 | 8.03 | 5.20 | **3.42** | 5.17 | 8.59 | 5.83 | 5.40 | **5.13** | 2.85 |

- LIF의 영향은 10토큰 안에서 대부분 사라지고 50토큰 뒤에는 0이다. f-LIF는 꼬리가 조금 더 길다(NSMT d=50에서 0.81 / 0.31%).
- **population은 두 과제 모두에서 평탄하게 오래 간다**(d=50에서 3.4–7.2%). SHD의 NSMT population은 거리가 멀수록 오히려 커진다(d=0 3.27 → d=50 7.15%). 리셋하지 않는 느린 가지에 바뀐 입력이 쌓이기 때문으로 본다(추정).
- **같은 긴 기억이 CinC에서도 측정되지만 성능 이득은 SHD에서만 나타났다.** 긴 기억이 있다는 것만으로 이득이 생기지는 않는다(6–7단계 MIT-BIH와 같음). 이 측정만으로 SHD 이득의 원인이 긴 기억이라고 단정할 수는 없다. population 구성은 발화율도 더 높다(mixer 0.25–0.30 vs 0.12–0.15). 바뀌는 비율은 기본 발화율에 따라 달라지므로 조건 간 절대값 비교에는 이 차이가 섞여 있다.
- NSMT 신피질망은 바꾼 토큰 앞쪽의 변화가 24 run 모두 정확히 0이다(정방향 인과성 확인). 양방향 mixer의 앞쪽 변화는 역방향 가지의 정상 동작이다.

### 결론
- **기준 (2)는 두 과제 12개 비교에서 모두 충족되지 않았다.** 긴 문맥이 필요한 두 과제에서 Spikformer식 스파이킹 자기주의(T = 4)가 정렬된 뉴런 상태로 토큰을 섞는 모든 구성(NSMT, Neo-mixer)보다 좋거나 같다. CinC에서는 차이가 3.8–5.8%p로 크고, SHD에서는 population 구성만 0.4–0.5%p 안으로 따라붙는다.
- **population f-LIF가 처음으로 성능 이득을 냈다 — SHD에서만.** SHD에서 NSMT·mixer 모두 LIF보다 +5.8–6.0%p(5/5 seed)이고 f-LIF보다도 5/5 seed에서 높다(사전 등록 비교는 아님). CinC에서는 NSMT −0.83, mixer +0.19로 차이가 없다. 5–7단계(MIT-BIH)와 CinC에서는 이득이 없었다. 토큰 축 기억 측정에서 population의 긴 기억(d=50에서 3.4–7.2%)은 두 과제 모두에 있으므로, 차이는 과제가 그 기억을 쓸 수 있는가에 있다(원인은 미확인).
- **f-LIF는 LIF와 비슷하거나 나쁘다.** NSMT 안에서는 SHD +0.97(좋음), CinC −0.26(차이 없음); mixer 안에서는 두 과제 모두 나쁨(−0.79, −1.68).
- **두 과제 모두 토큰 섞기가 필요하다(R1 14/14 좋음).** 다만 LIF·f-LIF 섞기는 하한보다 +1.6–4.6%p로 작고, Spikformer식과 population 구성은 +7.4–10.0%p(SHD)다.
- **NSMT 구조는 같은 뉴런의 Neo-mixer와 성능이 비슷한데 에너지는 50–122배다**(CinC LIF 61.76 vs 60.54, SHD 80.37 vs 80.39; 사전 등록 비교 아님). 에너지는 T = N 해마의 반복 계산이 대부분이다.
- **사전 예상과 비교:** "두 데이터 모두 하한이 무너진다"는 맞았다(크기는 구성마다 다름). "Neo-mixer 에너지는 Spikformer의 1/4–1/2"는 맞았다(0.22–0.51배). "NSMT 에너지는 Spikformer의 약 25배"도 맞았다(24.7–27.3배). "f-LIF·population 이득은 불확실"은 SHD population에서 크게 나타났다.

### 한계
- 탐색 결과다. CinC는 fold 0 하나와 seed 3개, SHD는 seed 5개다.
- 학습률은 NSMT LIF 하나로 고른 값(1e-3)을 모든 조건에 썼고, Spikformer식을 포함해 어느 모델도 따로 조정하지 않았다.
- Spikformer식은 T = 4, 나머지는 T = N(100–117)이라 스파이크 스텝 수가 다르다. 기준 (2)는 이 차이를 포함한 총 에너지로 판정했다.
- population 뉴런의 에너지는 우리 규약(FFT / 직접 합 두 방식)이다.
- CinC 점수가 문헌보다 크게 낮다(AF 인식이 약함). 절대 수준이 아니라 같은 분할의 대조군끼리만 비교한다.
- SHD는 처음 1 s만 쓰고 10 ms 구간으로 나눴다(8단계 채택안).

### 점검
- 사전 점검 18/18(`results/checks/check_cls_stage9.txt`, commit `fe9686c8c`). 70 run 모두 학습·검증·시험·에너지 열까지 끝났다(SKIP 0). 요약 스크립트는 seed 누락·비유한 값을 "미완"으로 처리한다(이번에는 해당 없음).
- 토큰 축 기억 분석 48/48(실패 0). NSMT 신피질망의 앞쪽 변화가 모두 0인 것으로 정방향 인과성을 확인했다.

### 기록 규칙 위반 공개
- 이미 commit한 중간 기록(`239746ca1`)의 제목 시각을 같은 줄에서 "23:55"에서 "23:50"으로 직접 고쳤다(commit `9094bfaf9`, 실제 작성 시각 기준). 기존 항목은 고치지 않고 정정은 새 항목으로 쓴다는 규칙을 어겼다. 원래 값은 `239746ca1`에 남아 있다. 이후 정정은 이처럼 새 항목에 적는다.

### 감사
- 감사 97(`docs/PROJECT_LOG.md` 10-19 21:53)과 98(10-20 00:01)은 f_lif_pop_v3 추적 감사로 새 판단 근거가 없다. 감사 98은 9단계 코드를 감사 범위 밖으로 기록했고, 학습·데이터·시험 검증은 하지 않았다고 적었다.

**다음 (사용자 결정):** 선택지 — (a) SHD population mixer의 경계 결과를 새 사전 고정(추가 seed 또는 다른 검증 분할)으로 확인, (b) MIT-BIH AFDB(긴 연속 기록)로 확장, (c) SSC로 확장, (d) 여기서 9단계를 닫고 정리.

**commit/tag:** 이 항목·결과·분석·요약을 담은 commit에 태그 `exp/nsmt-flif-neo-20261020`를 붙이고 브랜치와 태그만 push한다.
