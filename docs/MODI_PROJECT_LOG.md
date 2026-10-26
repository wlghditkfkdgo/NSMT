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

---

## 2026-10-20 16:57 KST — 9단계 추가 실험 설계·구현·사전 고정: NSMT 보조 손실 옵션(끄기 / DCT 없이 원 신호 재구성 / 비중 α 0.5·0.9) × 세 뉴런 × 두 과제 — 점검 19/19, smoke 8/8, tmux에서 시작

**질문과 답 (사용자, 10-20):** "현재 비교군에 있는 'NSMT' 모델 설정은 'auxiliary loss(reconstruction loss)'가 포함된 상태야?" → 예. 코드(`train_rhythm.py`)와 학습 기록(`best_log_0.csv`의 train_mse)으로 확인했다.
- NSMT 네 조건(LIF, f-LIF, population, 보통 CE)의 손실은 0.9 × CE + 0.1 × MSE다(CinC NSMT LIF seed 42 마지막 epoch: 0.9 × 0.3324 + 0.1 × 0.2482 = 0.3240 = 기록된 손실).
- 재구성 목표는 입력의 DCT 저역 15%(평균 복원, CinC 약 0–9.4 Hz, SHD 채널별 약 0–8 Hz)를 같은 방식으로 조각낸 것이다. 디코더는 신피질 출력만 읽는 선형층 하나다.
- Spikformer식·Neo-mixer·하한은 CE만 쓴다. 그래서 9단계 R4의 "NSMT 대 Spikformer식"은 구조와 손실이 함께 다른 비교였다. 이 점을 답에서 밝혔다.

**요청 (사용자, 10-20):** "1. 보조 손실 완전 OFF 2. full time-series (w/o DCT) 를 reconstruction 3. alpha를 {0.5, 0.9} 로 변경 위 세 개의 option을 추가적으로 두 가지 task에 대해서 전부 실험해줘."
**질문 답 (AskUserQuestion):** 적용 뉴런 = LIF·f-LIF·population 모두(96 run, 약 57시간). LIF → population → f-LIF 순서로 돌리고, 뉴런 묶음이 끝날 때마다 중간 보고한다.

**브랜치·기준:** `exp/nsmt-flif-neo`, 기준 commit `ad1da15a4`(9단계 결과, 태그 `exp/nsmt-flif-neo-20261020`).

**설정 (9단계 NSMT에서 바뀌는 것만):**

| 이름 | α (보조 손실 비중) | 재구성 목표 | 손실 |
|---|---|---|---|
| 기준선 (9단계 본 run 재사용) | 0.1 | DCT 저역 15% (`--keep_ratio 0.15`) | 0.9 CE + 0.1 MSE |
| `auxoff` | 0 | 계산하지만 학습에 쓰지 않음 | CE만 |
| `full` | 0.1 | 원 신호 조각 그대로 (`--keep_ratio 0`) | 0.9 CE + 0.1 MSE |
| `a05` | 0.5 | DCT 저역 15% | 0.5 CE + 0.5 MSE |
| `a09` | 0.9 | DCT 저역 15% | 0.1 CE + 0.9 MSE |

- 기준선은 9단계 본 run(같은 분할·seed·lr·epoch)을 다시 쓴다. 9단계 학습 코드와 이번 코드의 차이는 α = 0일 때만 갈리는 분기 하나다. α > 0이면 손실 식은 그대로다(아래 c19).
- "full" 목표는 CinC에서 z-score된 125 Hz 신호 64샘플, SHD에서 2구간 × 700채널 스파이크 수(0–6)다.
- **설계상 사실(예상의 근거):** 보조 손실의 기울기는 신피질 경로(신피질 임베딩과 2층)로만 간다. 분류 손실의 기울기는 교차 주의를 거쳐 신피질에도 가지만 절반으로 줄어 있다(`model.py:263`). AdamW는 기울기 크기를 파라미터마다 정규화한다. 그래서 CE만 받는 해마·head에서는 (1 − α) 배율이 거의 영향을 주지 않는다. α가 실제로 바꾸는 것은 주로 신피질 파라미터 안에서 두 손실의 균형이다.

**코드 (`nsmt_flif_v1/classification/`):**
- `train_rhythm.py`: `--alpha 0`이면 손실 = CE(디코더에 기울기 없음; MSE는 기록만).
- `scripts/run_rhythm.sh`: `nsmt_*` 조건에 접미사 `_auxoff`(α 0) · `_full`(keep 0) · `_a05` · `_a09`를 붙인다. 접미사가 없으면 α 0.1·keep 0.15로 이전과 같다.
- `scripts/make_queue_rhythm.sh`: kind `aux`(96줄 생성기), `auxsmoke`.
- 새 `scripts/rhythm_aux_driver.sh`: 한 pool, GPU당 1개.
- `summarize_rhythm.py --aux`: 9단계 suite에서 기준선과 spk_ssa를 읽고 X1–X4를 판정한다.
- `check_cls.py` c19.

**점검 19/19 (`results/checks/check_cls_aux.txt`):** c1–c18 그대로 통과. c19 내용:
- keep 0이면 재구성 목표 = 원 신호 조각이다(C = 1 경로 최대 차이 7.2e-07, C = 7 경로 비트 동일). keep 0.15이면 다르다(최대 차이 4.3, 빈 통과 아님).
- `train_epoch`를 α 0으로 돌리면 손실 = CE(같은 값)이고 디코더 기울기는 없으며 신피질 기울기는 있다.
- α 0.5·0.9이면 손실 = (1 − α) CE + α MSE이고 디코더 기울기가 생긴다.

**smoke (`results/smoke-rhythm-aux-20261020/raw.txt`, NSMT LIF, 1 epoch, seed 42; 16:48–16:57, 8/8, SKIP 0; 실험 아님):** 아직 commit하지 않은 작업 트리(기준 `ad1da15a4` + 위 코드)에서 돌렸다. 각 run의 `logargs.txt`에서 α·keep_ratio가 의도한 값임을 확인했다. 기록된 마지막 epoch 학습 손실은 사전 식(α = 0이면 CE, 아니면 (1 − α) CE + α MSE)과 최대 6e-07 차이로 일치한다.

| 데이터 | 옵션 | α / keep | 1 epoch 점수 | 학습 손실 / CE / MSE | 총 µJ (FFT) | 시간 s |
|---|---|---|---|---|---|---|
| SHD | auxoff | 0 / 0.15 | 0.440813 | 2.47139 / 2.47139 / 0.11216 | 26,027.9 | 236 |
| SHD | full | 0.1 / 0 | 0.450972 | 2.22893 / 2.46773 / 0.07975 | 26,031.8 | 238 |
| SHD | a05 | 0.5 / 0.15 | 0.458039 | 1.24356 / 2.46702 / 0.02009 | 26,029.1 | 230 |
| SHD | a09 | 0.9 / 0.15 | 0.458922 | 0.26407 / 2.46902 / 0.01908 | 26,024.4 | 233 |
| CinC | auxoff | 0 / 0.15 | 0.323558 | 1.12644 / 1.12644 / 0.73054 | 2,894.9 | 256 |
| CinC | full | 0.1 / 0 | 0.327960 | 1.05886 / 1.12624 / 0.45248 | 2,912.1 | 252 |
| CinC | a05 | 0.5 / 0.15 | 0.324345 | 0.71469 / 1.12707 / 0.30230 | 2,874.9 | 258 |
| CinC | a09 | 0.9 / 0.15 | 0.325207 | 0.38455 / 1.12714 / 0.30204 | 2,876.7 | 251 |

- 끄기에서는 디코더가 학습되지 않아 MSE가 높게 남는다(설계대로). 9단계 smoke의 같은 seed 기준선(α 0.1)은 CinC 0.325715, SHD 0.447438이다. 1 epoch 값이라 비교에 쓰지 않는다.

**실행 (사전 고정):** tmux `flif_aux`, `bash nsmt_flif_v1/classification/scripts/rhythm_aux_driver.sh 0,1,2,3`.
- 96 run = 3 뉴런(LIF, population, f-LIF) × 4 옵션 × (CinC 3 seed 42/7/2026, fold 0 + SHD 5 seed 42/7/2026/123/1024).
- lr 1e-3, 나머지는 9단계 그대로. suite는 `flif-v1-rhythm-aux-cinc`, `flif-v1-rhythm-aux-shd`.
- 큐 순서는 LIF 묶음 → population 묶음 → f-LIF 묶음이고, 묶음 안에서는 SHD 먼저, CinC 다음이다.
- **예상 시간:** 9단계 run당 시간(CinC LIF 약 1.4 h, f-LIF·population 약 2.0 h; SHD LIF 약 2.1 h, f-LIF·population 약 3.0 h)으로 약 229 GPU시간, GPU 4장에서 약 57시간(10-23 새벽 무렵).

**판정 (9단계와 같은 규칙; 같은 seed끼리 짝):** "좋음" = 평균 ≥ +0.5%p이고 A > B인 seed가 CinC 3개 중 2개 이상, SHD 5개 중 4개 이상. 반대면 "나쁨", 그 밖은 "차이 없음", seed 누락·비유한 값은 "미완".
- X1 보조 손실 끄기: `auxoff` − 기준선(뉴런별·과제별 6개).
- X2 재구성 목표: `full` − 기준선(6개).
- X3 비중: `a05` − 기준선, `a09` − 기준선(12개).
- X4 Spikformer식과: 각 옵션 − `spk_ssa`(24개)에 기준 (2)를 적용한다. NSMT 에너지가 Spikformer식의 약 25배이므로 기준 (2)는 "좋음"일 때만 충족될 수 있다.
- 보고만: 마지막 epoch 학습 CE·MSE, 신피질 발화율, 에너지, 마지막 epoch 점수, 토큰 축 기억(끝난 뒤 별도로 돌리면 보고).
- **해석 틀(사전):** X1에서 끄기가 기준선과 차이 없으면, 보조 손실은 NSMT의 성능을 설명하지 않는다. 끄기가 좋으면 보조 손실이 분류를 방해한 것이고, 나쁘면 도운 것이다. X4에서 어느 옵션도 Spikformer식을 넘지 못하면, 9단계 R4의 NSMT 패배는 보조 손실 때문이 아니다.
- **예상(사전 기록):** 끄기와 full은 기준선과 차이가 없을 것이다(±0.5%p 안; 비중 0.1이 작고, 해마·head는 Adam 때문에 거의 영향을 받지 않음). α 0.9는 신피질이 분류보다 재구성에 맞춰져 약간 나쁠 수 있다. population 조건의 SHD 이득이 신피질 상태에서 나왔다면 특히 그렇다. CinC에서 Spikformer식을 넘는 옵션은 없을 것이다(차이 3.8–4.7%p).

**운영:** 데이터·checkpoint·TensorBoard·stdout은 로컬에 둔다. 뉴런 묶음이 끝날 때마다 중간 기록을 덧붙이고 commit·push한다. 감사는 98 이후 새 항목이 없다.

**commit:** 이 항목·코드·점검·smoke를 commit·push한 뒤 시작한다.

---

## 2026-10-21 06:19 KST — 9단계 추가 실험 중간 기록 1/3: NSMT LIF 묶음(32 run) — **보조 손실을 끄거나 목표·비중을 바꿔 좋아지는 설정은 없음(8개 비교 중 "좋음" 0)**; Spikformer식과의 차이는 그대로(−4.12 ~ −6.97%p); 원 신호 재구성은 두 과제 모두 경계에서 "나쁨"

**구분:** 10-20 16:57 항목(commit `80a79940d`)에서 사전 고정한 96 run 중 LIF 묶음의 중간 기록이다(탐색). population 묶음은 06:10에 시작했고 f-LIF 묶음이 뒤따른다. 이 항목이 속한 commit은 메시지 "stage 9 follow-up interim 1/3"으로 식별한다.

**실행:** tmux `flif_aux`, 10-20 16:57 – 10-21 06:10, **LIF 32/32 완료·SKIP 0**. run당 평균 시간(GPU당 1개): SHD 6,402–7,759 s, CinC 4,151–5,449 s. 산출물(Git): `results/flif-v1-rhythm-aux-{cinc,shd}/{raw.txt, summary.md}`(summary는 중간본 — population·f-LIF 행은 미완), 실행별 `log/flif-v1-rhythm-aux-{cinc,shd}/<DATA>/<날짜>/<run>/…`(LIF run만). checkpoint·stdout은 로컬.

### 결과 (최고 검증 가중치, 시험; CinC 3 seed / SHD 5 seed 평균 ± 표준편차, %)

| 과제 | NSMT LIF 설정 | 점수 | 마지막 epoch 점수 | 마지막 epoch 학습 CE / MSE | 신피질 발화율 | 총 µJ (FFT) |
|---|---|---|---|---|---|---|
| CinC | 끄기 (α 0) | 61.47 ± 1.37 | 60.96 ± 1.30 | 0.3865 / 0.7376 | 0.0575 | 3,482.5 |
| CinC | **기준선 (α 0.1, 저역 목표; 9단계)** | 61.76 ± 1.40 | 61.61 ± 0.72 | 0.3323 / 0.2464 | 0.0792 | 3,419.3 |
| CinC | α 0.5 | 61.03 ± 0.44 | 60.72 ± 0.85 | 0.3196 / 0.1271 | 0.1053 | 3,414.7 |
| CinC | α 0.9 | 61.02 ± 1.23 | 60.58 ± 0.99 | 0.3701 / 0.0813 | 0.1284 | 3,379.9 |
| CinC | 원 신호 재구성 (α 0.1) | 61.20 ± 1.13 | 61.41 ± 0.73 | 0.3267 / 0.3500 | 0.0810 | 3,463.7 |
| CinC | Spikformer식 (참고) | 65.60 ± 1.48 | 65.36 ± 1.17 | – | – | 127.4 |
| SHD | 끄기 (α 0) | 80.50 ± 1.25 | 80.74 ± 0.87 | 0.0030 / 0.1154 | 0.0795 | 26,471.3 |
| SHD | **기준선 (α 0.1, 저역 목표; 9단계)** | 80.37 ± 1.05 | 80.76 ± 0.57 | 0.0029 / 0.0146 | 0.0825 | 26,484.6 |
| SHD | α 0.5 | 80.08 ± 1.40 | 80.53 ± 0.41 | 0.0030 / 0.0125 | 0.0874 | 26,483.2 |
| SHD | α 0.9 | 80.24 ± 1.05 | 80.13 ± 0.36 | 0.0034 / 0.0112 | 0.0960 | 26,470.9 |
| SHD | 원 신호 재구성 (α 0.1) | 79.78 ± 0.98 | 80.89 ± 0.35 | 0.0029 / 0.0691 | 0.0810 | 26,484.1 |
| SHD | Spikformer식 (참고) | 86.75 ± 1.32 | 86.77 ± 0.97 | – | – | 1,071.2 |

(끄기의 MSE는 디코더가 학습되지 않은 값이다. 원 신호 재구성의 MSE는 목표가 달라 다른 행과 직접 비교할 수 없다.)

### 사전 고정 판정 (같은 seed끼리 짝, %p; CinC seed 42 / 7 / 2026, SHD 42 / 7 / 2026 / 123 / 1024)

| 질문 | 과제 | A − B | 평균 | 판정 |
|---|---|---|---|---|
| X1 끄기 − 기준선 | CinC | +1.04 / +0.07 / −1.98 | −0.29 | 차이 없음 |
| X1 끄기 − 기준선 | SHD | +0.09 / −1.19 / −0.22 / −0.35 / +2.34 | +0.13 | 차이 없음 |
| X2 원 신호 − 기준선 | CinC | −1.48 / +2.23 / −2.44 | −0.56 | **나쁨** |
| X2 원 신호 − 기준선 | SHD | −1.10 / −0.22 / −1.77 / −0.66 / +0.80 | −0.59 | **나쁨** |
| X3 α 0.5 − 기준선 | CinC | −1.10 / +0.44 / −1.53 | −0.73 | **나쁨** |
| X3 α 0.5 − 기준선 | SHD | −3.75 / −0.13 / +1.59 / +0.31 / +0.53 | −0.29 | 차이 없음 |
| X3 α 0.9 − 기준선 | CinC | +0.01 / +1.18 / −3.43 | −0.75 | 차이 없음 (나쁜 seed 1개) |
| X3 α 0.9 − 기준선 | SHD | −1.90 / −0.97 / +0.00 / +0.88 / +1.33 | −0.13 | 차이 없음 |
| X4 각 설정 − Spikformer식 | CinC | 4개 설정 모두 | −4.12 ~ −4.58 | **나쁨** 4/4, 기준 (2) 미충족 |
| X4 각 설정 − Spikformer식 | SHD | 4개 설정 모두 | −6.25 ~ −6.97 | **나쁨** 4/4, 기준 (2) 미충족 |

- **좋아지는 설정이 없다.** 기준선 대비 8개 비교 중 "좋음" 0, "나쁨" 3, "차이 없음" 5. "나쁨" 3개는 모두 경계(−0.56 ~ −0.73%p)이고 seed별 흩어짐(1–2%p)보다 작다.
- **LIF NSMT가 Spikformer식에 지는 것은 보조 손실 때문이 아니다.** 보조 손실을 완전히 꺼도 차이는 CinC −4.12, SHD −6.25%p로 9단계(−3.83, −6.38)와 같은 수준이다.
- **보조 손실은 신피질 발화를 늘리지만 분류에는 반영되지 않는다.** 신피질 발화율이 α와 함께 단조 증가한다(CinC 0.0575 → 0.0792 → 0.1053 → 0.1284, SHD 0.0795 → 0.0825 → 0.0874 → 0.0960). 점수는 거의 그대로다.
- **사전 예상과 비교:** "끄기는 차이 없음" — 맞음(두 과제). "원 신호 재구성은 차이 없음" — 틀림(두 과제 모두 경계에서 나쁨). "α 0.9는 약간 나쁠 수 있음" — 두 과제 모두 차이 없음(CinC 평균 −0.75이지만 나쁜 seed가 1개). "CinC에서 Spikformer식을 넘는 설정은 없음" — 맞음.

### 한계
- LIF 묶음만의 중간 결과다. population(SHD에서 9단계 최고 NSMT)·f-LIF 묶음이 남았다.
- 탐색(CinC fold 0 × 3 seed, SHD 5 seed)이다. "나쁨" 판정 3개는 기준선 −0.5%p에 가깝다.

### 점검·감사
- 32 run 모두 학습·검증·시험·에너지까지 끝났다(SKIP 0). 사전 점검 19/19(`results/checks/check_cls_aux.txt`).
- 감사 99(10-20 16:21)·100(17:01)은 f_lif_pop_v3 추적 감사로 새 판단 근거가 없다. 두 감사는 9단계 commit들을 감사 범위 밖으로 기록했다.

**다음:** population 묶음(32 run, 06:10 시작) → f-LIF 묶음. 묶음마다 중간 기록. 전체가 끝나면 결과 항목·태그.

**commit:** 이 항목·LIF 결과·중간 요약을 commit·push한다.

---

## 2026-10-21 20:12 KST — 9단계 원인 분리 실험 설계·구현·사전 고정: 1-B 위치 정보 · 1-A 도달 거리 · 1-C 해마 T = 4 · 3 population soma 가중치 (76 run) — 점검 24/24, smoke 16/20 통과(4건 공유 GPU 메모리 부족; 본 학습 전 정식 배치 smoke 재실행), 드라이버 대기 시작

**배경 (사용자 질문 "문제점과 그 원인을 분석해줘", 10-21):** 9단계 결과(태그 `exp/nsmt-flif-neo-20261020`)와 보조 손실 LIF 묶음(commit `7e1b1e98a`)을 바탕으로 문제 6개와 원인을 정리해 보고했다. 요지: (1) 조각 사이 정보 통로의 성격 차이 — 자기주의는 모든 조각을 한 번에 보고, 신피질 막전위는 2–3토큰에서 잊는다(토큰 축 기억 측정); (2) **제가 만든 비대칭** — `SSA_rel_scl`에만 상대 위치 편향이 있고 NSMT 교차 주의·Neo-mixer에는 위치 정보가 없다; (3) NSMT는 학습 데이터에 더 맞고 새 데이터에 덜 맞는다(SHD 검증 98.24 > Spikformer식 97.82인데 시험 80.37 < 86.75; 검증−시험 17.9 vs 11.1); (4) 보조 손실은 원인이 아니다(LIF 묶음); (5) T = N 해마 반복이 에너지 대부분; (6) f-LIF ≈ LIF, population은 SHD에서만 이득. 사용자가 1-B·1-A·1-C·3을 진행하기로 했고, 별도 아이디어(순방향·역방향 스트림 사이의 교차 주의)는 검토 보고 뒤 "1-A/1-C 결과를 본 뒤 결정"으로 두었다.

**아이디어 검토 요지(기록):** 채널별 T×T' 주의 지도는 스파이크 입력에서 점수가 F[t,d]·R[t',d](두 비트의 AND)뿐이라 출력[t,d] = F[t,d]·Σ_t' R[t',d]V[t',d]로 **모든 토큰이 같은 합을 받는다**(토큰별 상호작용 소멸). 채널을 헤드로 묶어 내적을 취하면 "질의 = 순방향 스트림, 키/값 = 역방향 스트림인 Spikformer식 주의"가 되고, 이것이 작동하는 형태다. 위험: 주의가 들어오면 뉴런 기억의 몫이 다시 작아질 수 있다(5단계 "자기 키/값 = 신피질 키/값"). 문헌: TIM(IJCAI 2024)·STAtten(CVPR 2025)은 반복 축 위의 시간 주의, QKFormer(NeurIPS 2024)는 토큰/채널 1차원 주의, SHD 상위 결과(SE-adLIF 95.1%, DCLS 지연 학습, STSC-SNN 92.4%)는 재귀·지연 기반, f-SNN(ICLR 2026)이 우리 f-LIF. "재귀 뉴런 상태를 질의·키로 쓰는 신호 시간 축 스파이킹 주의"는 찾지 못했다.

**브랜치·기준:** `exp/nsmt-flif-neo`, 기준 commit `7e1b1e98a`. 모든 run은 9단계 프로토콜 그대로(fold 0 × seed 42/7/2026, SHD seed 42/7/2026/123/1024, lr 1e-3, 30/50 epoch, 배치 32). 기준선은 9단계 본 run 재사용. suite `flif-v1-rhythm-fix-{cinc,shd}`.

**계획 중 측정(population 초기 발화율, 학습 모드 BN, 실제 16표본, 초기 가중치; soma 가중치별):**

| 데이터 / 모델 | LIF | psi 1.0 | 1.25 | 1.5 | 2.0 | 4.0 | 8.0 |
|---|---|---|---|---|---|---|---|
| CinC NSMT 신피질 | 0.0448 | 0.0381 | 0.0638 | 0.1057 | 0.2004 | 0.3628 | 0.4305 |
| CinC mixer | 0.0798 | 0.0373 | 0.0748 | 0.1287 | 0.2589 | 0.5483 | 0.6636 |
| SHD NSMT 신피질 | 0.0579 | 0.0690 | – | – | 0.2497 | 0.3795 | 0.4422 |
| SHD mixer | 0.0977 | 0.0817 | – | – | 0.3077 | 0.5728 | 0.6796 |

**초기 발화는 9단계 보정(1.0)에서 이미 LIF와 맞았다.** CinC에서 학습 뒤 population 신피질 발화율이 LIF의 절반(0.040 vs 0.079)인 것은 학습 중에 생긴 차이다. 가설 3("초기 보정 부족")은 초기값으로는 지지되지 않는다. 사용자 결정대로 "초기 구동을 더 세게 주면 CinC 결과가 바뀌는가"를 psi 1.5(LIF의 1.3–1.6배)·2.0(4–5배)으로 시험한다.

### 조건 (새 조건 10개 + soma 가중치 2값; 76 run)

| 가설 | 조건 | 설정 | run | pool |
|---|---|---|---|---|
| 1-B | `spk_ssa_nope` | Spikformer식(T_h 4), 상대 위치 편향 없음 (`--ssa_pe 0`) | 3 + 5 | 가벼움 |
| 1-B | `nsmt_lif_pe` | NSMT LIF(T = N) + 해마 임베딩에 학습형 절대 위치 [1, N, D] (`--pos_emb`; 선형층 출력에 더한 뒤 BN·LIF) | 3 + 5 | 무거움 |
| 1-B | `mix_lif_bidir_pe` | 양방향 spikeDE-LIF Neo-mixer + 같은 위치 임베딩 | 3 + 5 | 가벼움 |
| 1-A | `mix_lif_rec_bidir` | 양방향 Neo-mixer, 층 = spikingjelly 단일 스텝 LIF 루프 + 게이트 재귀 `tanh(g)·W_rec·s_{t−1}`(g 초기 0, `NeoLayer(recurrent=True)`) | 3 + 5 | 가벼움 |
| 1-A 대조 | `mix_sjlif_bidir` | 위와 같은 spikingjelly LIF 루프, 재귀 없음 (**계획에 없던 추가**: `lif_rec`는 뉴런 구현이 9단계 `mix_lif_bidir`의 spikeDE LIF와 달라 재귀 효과를 따로 보려면 같은 뉴런의 대조군이 필요) | 3 + 5 | 가벼움 |
| 1-A | `mix_pop_bidir_pt16` | 양방향 population mixer, 가지 τ 16/32/64/128, soma 가중치 **2.0**(c24 규칙) | 3 + 5 | 가벼움 |
| 1-C | `nsmt_lif_t4` | 해마 T_h = 4(정확한 반복) + 토큰 교차 주의(질의 = 해마 토큰, 키/값 = 신피질 토큰을 T_h로 복제, 상대 위치 편향 포함, 기울기 절반 공유 유지), 신피질 LIF T = N, 보조 손실 α 0.1·DCT 유지 (`--hippo_T 4 --hippo_xattn`) | 3 + 5 | 가벼움 |
| 1-C | `nsmt_pop_t4` | 위와 같고 신피질 population(psi 1.0) | 3 + 5 | 가벼움 |
| 3 | `nsmt_pop_psi1.5`, `nsmt_pop_psi2` | NSMT population, soma 가중치 1.5 / 2.0 (CinC만) | 3 + 3 | 무거움 |
| 3 | `mix_pop_bidir_psi1.5`, `mix_pop_bidir_psi2` | 양방향 population mixer, 1.5 / 2.0 (CinC만) | 3 + 3 | 가벼움 |

- **c24 규칙과 결과:** τ 16–128의 soma 가중치는 {1, 2, 4} 중 초기 발화율이 "τ 4–32·psi 1.0 mixer"(CinC 0.0373, SHD 0.0817)에 가장 가까운 값. 측정: CinC 1 → 0.0000, 2 → 0.0116, 4 → 0.1337; SHD 1 → 0.0038, 2 → 0.0405, 4 → 0.2020 → **두 데이터 모두 2.0**. 선택값의 초기 발화는 기준의 1/3–1/2이다(규칙상 그대로 쓰되 기록).
- **계획과 다른 점(실행 순서):** 계획은 가벼움 pool 62 run을 지금 돌고 있는 보조 손실 pool 옆에 바로 시작하는 것이었다. 그러나 보조 손실 population·f-LIF NSMT run은 GPU당 약 40 GB(LIF 묶음은 30 GB)를 써서 두 번째 작업이 들어갈 자리가 없다 — 첫 smoke(배치 32)에서 SHD `mix_lif_bidir_pe`가 메모리 부족으로 실패했다(`results/smoke-rhythm-fix-20261021/raw.txt`의 SKIP 1건, 통과 1건; 그 뒤 중단). 그래서 **두 pool 모두 보조 손실 드라이버의 "ALL DONE" 뒤에 시작**한다: 무거움 pool(GPU당 1개) + 가벼움 pool A(GPU당 1개) 동시, 무거움이 끝나면 같은 큐를 읽는 가벼움 pool B를 추가(GPU당 2개).

### 코드 (`nsmt_flif_v1/classification/`; 기존 조건 출력 비트 동일 — c1·c2·c20·c21)
- `layers.py`: `SSA_rel_scl.forward(x, mask=None, kv=None)` — kv가 있으면 키·값을 kv에서.
- `model.py`: `Embedding(pos_emb=False)`(파라미터 `pos` [1, N, D], trunc_normal 0.02, BN 앞에 더함); `Block(xattn, ssa_pe)` — xattn 경로 `kv = mx.transpose(0,2).expand(T_h)`, `x·(1 − ssa(x, kv))`, MLP; `myModel(hippo_xattn, pos_emb, ssa_pe)` — `hippo_T ≠ N`을 `hippo_xattn`에도 허용, 입력은 `_tokens_mc` 경로(정확한 T_h 반복), 신피질·디코더·보조 목표는 NSMT와 같음.
- `neo_mixer.py`: `MIXERS += lif_rec, sj_lif`(`MixerStack(recurrent=True/False)` → `NeoLayer` spikingjelly 루프; `_init_weights`가 `W_rec`의 작은 초기값을 보존); `pos_emb`.
- `config.py`: `--ssa_pe {0,1}`, `--pos_emb`, `--hippo_xattn`, `--mixer lif_rec|sj_lif`; 디렉터리 접미사 `_xaT{T}`, `_pe`, `_nope`, population mixer의 `_pt…`.
- `load_model.py`: 새 인자 전달. `scripts/run_rhythm.sh`: 새 조건, `_psi<v>` 접미사, `PT16_PSI_<DATA>`(2.0), `BS` 환경변수(smoke용). `scripts/make_queue_rhythm.sh`: kind `fixlight`/`fixheavy`/`fixsmoke`. 새 `scripts/rhythm_fix_driver.sh`. `summarize_rhythm.py --fix`. `scripts/run_rhythm_analysis.sh`: `SUITES`/`CONDS` 환경변수. `check_cls.py` c20–c24.

### 점검 24/24 (`results/checks/check_cls_fix.txt`)
- c1–c19 그대로 통과(기본 경로 비트 동일).
- c20 `--ssa_pe 0`: 상대 위치 표 없음; 초기 출력이 `pe=1`(표 0 초기화)과 비트 동일; `pe=1`에서 표가 기울기를 받음(합 1.05e-01).
- c21 `--pos_emb`: 추가 파라미터는 `encoding.pos` [1, N, 384] 하나(NSMT·mixer); 임베딩 출력 모양 불변; 가중치를 옮기고 pos를 0으로 두면 출력 비트 동일; 토큰 j에만 오프셋을 주면 두 배치 모두 토큰 j의 스파이크만 바뀜; 무작위 pos면 로짓이 달라짐(아래 smoke가 잡은 오류 뒤 보강).
- c22 `--hippo_xattn`: T_h 4, 주의 q [4, B, 117, 384]·kv [4, B, 117, 384], 신피질 인과(토큰 j 교체 → j 이전 변화 0), 신피질 토큰 교체로 블록 출력 변화, 디코더 존재, 학습 손실 = 0.9 CE + 0.1 MSE. (처음 두 번은 검사 방법 문제로 실패: 평가 모드 BN은 초기 뉴런을 모두 침묵시키고, 학습 모드 BN은 배치 통계를 다시 합산해 j 이전 토큰의 문턱 근처 스파이크를 뒤집었다. BN 통계를 한 번 고정(momentum 1)한 뒤 평가 모드로 검사하도록 고쳤다.)
- c23 `lif_rec`/`sj_lif`: 재귀 층 4개(양방향 2 × 2), g 초기 0, W_rec을 0으로 둔 모델과 출력 비트 동일, g = 1이면 달라짐, g가 기울기를 받음; `sj_lif`는 W_rec 없음; 에너지 기록기가 `blocks.0.mixer` 아래 LIFNode 4개를 신피질 경로로 잡음.
- c24: 위 표.

### smoke (배치 8, 1 epoch, seed 42; 실험 아님 — 파이프라인·설정 확인용)
GPU마다 보조 손실 NSMT(약 40 GB)가 돌고 있어 정식 배치(32)로는 자리가 없었다. 배치 8로 20개를 돌렸고(`results/smoke-rhythm-fix-20261021-bs8/raw.txt`), **16개 통과, 4개는 CinC에서 메모리 부족으로 실패**(`mix_lif_bidir_pe`, `nsmt_lif_pe`, `nsmt_pop_psi1.5`, `nsmt_pop_psi2`; 모두 공유 GPU의 빈 메모리 부족, 코드 오류 아님). 실패한 네 조건 중 `nsmt_lif_pe`·`mix_lif_bidir_pe`는 SHD에서 통과했고, `nsmt_pop_psi*`는 9단계 `nsmt_pop`과 soma 값만 다르다(c16·c24). 각 run의 `logargs.txt`에서 hippo_T·hippo_xattn·pos_emb·ssa_pe·mixer·pop_taus·pop_soma_init 값이 의도와 같음을 확인했다. **본 학습 직전에 드라이버가 정식 배치로 smoke 20개를 다시 돌리고, 실패가 있으면 본 학습을 시작하지 않는다.**

**smoke가 잡은 구현 오류(수정 완료):** 첫 구현의 위치 임베딩은 표 [1, N, D]를 선형층 출력에 그대로 더했다. 해마 배치([T·B, N, D])에서는 맞지만, Neo-mixer처럼 토큰이 0번 축에 있는 배치([N·B, 1, D])에서는 브로드캐스트로 텐서가 N배로 커지고 위치도 틀리게 붙는다. SHD `mix_lif_bidir_pe` smoke의 에너지가 대조군의 6배(1,538.9 vs 260.0 µJ, 뉴런 수 100배)로 나와 발견했다. c21이 이것을 놓친 이유는 출력 모양을 검사하지 않았기 때문이다(0 위치에서 로짓이 같다는 검사만 통과). `Embedding.forward`가 두 배치를 구분해 토큰 축에 더하도록 고치고, c21에 "임베딩 출력 모양 불변"과 "토큰 j에만 큰 오프셋을 주면 토큰 j의 스파이크만 바뀜(두 배치 모두)"을 추가했다. 고친 뒤 SHD·CinC `mix_lif_bidir_pe`를 다시 smoke했다: 두 run 모두 통과(`results/smoke-rhythm-fix-20261021-bs8b/raw.txt`), 1 epoch 점수 CinC 0.3019 / SHD 0.6210, 총 에너지 **24.2 / 260.8 µJ**(대조군 24.1 / 260.0과 같은 규모), 파라미터 961,152 / 1,473,792(= 대조군 + N × 384).

| 데이터 | 조건 | 1 epoch 점수 | 총 µJ (FFT) | 시간 s |
|---|---|---|---|---|
| SHD | spk_ssa_nope | 0.5861 | 1,049.7 | 97 |
| SHD | mix_sjlif_bidir | 0.5137 | 260.0 | 829 |
| SHD | mix_lif_rec_bidir | 0.3750 | 270.3 | 1,404 |
| SHD | mix_pop_bidir_pt16 (psi 2) | 0.7372 | 292.8 | 1,677 |
| SHD | nsmt_lif_t4 | 0.6343 | 1,297.6 | 627 |
| SHD | nsmt_pop_t4 | 0.7200 | 1,328.3 | 1,225 |
| SHD | nsmt_lif_pe | 0.5702 | 26,103.8 | 840 |
| CinC | spk_ssa_nope | 0.3548 | 105.5 | 78 |
| CinC | mix_sjlif_bidir | 0.2970 | 24.1 | 979 |
| CinC | mix_lif_rec_bidir | 0.3085 | 26.9 | 1,642 |
| CinC | mix_pop_bidir_pt16 (psi 2) | 0.3354 | 62.4 | 1,715 |
| CinC | nsmt_lif_t4 | 0.3483 | 134.3 | 632 |
| CinC | nsmt_pop_t4 | 0.3435 | 144.4 | 1,423 |
| CinC | mix_pop_bidir_psi1.5 | 0.3816 | 64.5 | 1,679 |
| CinC | mix_pop_bidir_psi2 | 0.3777 | 65.6 | 1,690 |

- 에너지 규모(사전 기록): 해마 T = 4 + 신피질 T = N인 `nsmt_*_t4`는 SHD 약 1,300 µJ(Spikformer식 1,050 + 신피질 약 260), CinC 약 135–145 µJ — 9단계 NSMT(26,500 / 3,400)의 1/20–1/25. 시간은 배치 8·공유 GPU 값이라 본 학습 추정에 쓰지 않는다.

### 판정 (사전 고정; 9단계 규칙 — 평균 ≥ +0.5%p이고 CinC 3개 중 2개·SHD 5개 중 4개; 누락·비유한 값은 미완)
- **Y (1-B)**: Y1 `spk_ssa_nope` − `spk_ssa`; Y2 `nsmt_lif_pe` − `nsmt_lif`; Y3 `mix_lif_bidir_pe` − `mix_lif_bidir`; Y4 `nsmt_lif_pe`·`mix_lif_bidir_pe` − `spk_ssa_nope`(위치 정보를 양쪽 다 뺀/다 준 비교) 및 − `spk_ssa`(기준 (2)).
- **Z (1-A)**: Z1 `mix_lif_rec_bidir` − `mix_sjlif_bidir`(같은 뉴런); 보고 `mix_sjlif_bidir` − `mix_lif_bidir`(뉴런 구현 차이); Z2 `mix_pop_bidir_pt16` − `mix_pop_bidir`; Z3 둘 − `spk_ssa`(기준 (2)); 보고: 토큰 축 기억, 학습 후 g.
- **W (1-C)**: W1 `nsmt_lif_t4` − `nsmt_lif`, `nsmt_pop_t4` − `nsmt_pop`; W2 둘 − `spk_ssa`(기준 (2)); 보고: 검증−시험 차이, 에너지.
- **V (3, CinC만)**: V1 psi 1.5 / 2 − psi 1.0(NSMT·mixer); 보고: 학습 후 신피질 발화율.
- **예상(사전 기록):** Y1 상대 위치 편향 제거는 Spikformer식을 0.5–2%p 낮추지만 NSMT·mixer보다는 여전히 높다. Y2·Y3 위치 임베딩은 CinC에서 약간 돕고 SHD에서는 차이 없음. Z1 재귀는 SHD에서 좋음(문헌상 SHD는 재귀가 중요), CinC 불확실; `sj_lif` − `lif`는 차이 없음; Z2 느린 가지는 CinC 차이 없음, SHD 불확실(초기 발화가 기준의 1/3이라 침묵 위험). W1 T = 4 해마는 T = N과 차이 없거나 좋음, 에너지 1/20; W2는 5단계("자기 키/값 = 신피질 키/값")가 맞다면 기준 (2) 충족 가능(가장 불확실). V1 차이 없음.


**실행 중 스크립트 수정 사고(공개):** 이 구현을 위해 `scripts/run_rhythm.sh`를 고치는 동안 보조 손실 pool이 같은 파일로 돌고 있었다. bash는 스크립트를 바이트 위치로 읽으므로, 학습이 끝난 뒤 다음 줄을 읽은 인스턴스 4개(CinC `nsmt_pop_auxoff` seed 42/7/2026, `nsmt_pop_full` seed 42)가 바뀐 파일의 엉뚱한 위치에서 `line 53: syntax error`로 끝나 RESULT 줄을 쓰지 못했다(pool 로그에 4건). 학습·시험 산출물은 온전하다. `collect_missing_results.py`로 같은 형식의 RESULT 줄 4개를 `results/flif-v1-rhythm-aux-cinc/raw.txt`에 덧붙였고 `collected=1`로 표시했다(`time_s`는 stdout의 "done in" 값). 18:29 이후 시작한 run은 정상이다. 재발 방지: 돌고 있는 셸 스크립트는 고치지 않고 복사본을 쓴다(메모리에 기록).

**실행:** tmux `flif_fix`, `bash nsmt_flif_v1/classification/scripts/rhythm_fix_driver.sh 0,1,2,3`(보조 손실 완료를 기다렸다가 시작). 예상: 보조 손실 완료 10-23 새벽 → 무거움 14 run 약 7시간, 가벼움 62 run 약 1일 → **10-24 새벽–낮** 종료.

**감사:** 100(10-20 17:01) 이후 새 항목 없음.

**commit:** 이 항목·코드·점검·smoke를 commit·push한 뒤 시작한다.

---

## 2026-10-21 23:26 KST — 9단계 추가 실험 중간 기록 2/3: NSMT population f-LIF 묶음(32 run) — **SHD에서는 보조 손실 비중이 클수록 좋음(α 0.9 +2.37%p, Spikformer식보다 +1.93%p로 기준 (2) 충족), CinC에서는 판정이 모두 "차이 없음" 또는 경계 "나쁨"이고 Spikformer식과의 차이는 −3.9 ~ −5.3%p 그대로** — SHD의 이득은 CinC로 옮겨가지 않았다

**구분:** 10-20 16:57 사전 고정(commit `80a79940d`)의 population 묶음 중간 기록(탐색). 이 항목이 속한 commit은 메시지 "stage 9 follow-up interim 2/3"으로 식별한다.

**실행:** tmux `flif_aux`, 10-21 06:10 – 23:2x, **population 32/32 완료·SKIP 0**. CinC `nsmt_pop_auxoff` seed 42/7/2026과 `nsmt_pop_full` seed 42의 RESULT 줄은 실행 중 스크립트 수정 사고로 산출물에서 다시 만든 것이다(`collected=1`, 10-21 20:12 항목에 공개). f-LIF 묶음(32 run)이 이어서 돈다.

### 결과 (최고 검증 가중치, 시험; CinC 3 seed / SHD 5 seed 평균 ± 표준편차, %)

| 과제 | population NSMT 설정 | 점수 | 검증(최고) | 검증 − 시험 | 신피질 발화율 | 마지막 epoch 학습 CE / MSE |
|---|---|---|---|---|---|---|
| CinC | 끄기 (α 0) | 60.77 ± 0.94 | 65.74 | 4.97 | 0.0667 | 0.4524 / 0.7449 |
| CinC | **기준선 (α 0.1; 9단계)** | 60.94 ± 2.24 | 65.88 | 4.94 | 0.0398 | 0.3730 / 0.5601 |
| CinC | α 0.5 | 61.40 ± 0.60 | 67.27 | 5.87 | 0.0595 | 0.3391 / 0.3634 |
| CinC | α 0.9 | 61.67 ± 0.62 | 66.12 | 4.45 | 0.1148 | 0.4617 / 0.2655 |
| CinC | 원 신호 재구성 | 60.31 ± 0.17 | 64.59 | 4.28 | 0.0366 | 0.3641 / 0.7044 |
| CinC | Spikformer식 (참고) | 65.60 ± 1.48 | 68.79 | 3.19 | – | – |
| SHD | 끄기 (α 0) | 85.08 ± 1.03 | 98.82 | 13.74 | 0.1297 | 0.0020 / 0.1259 |
| SHD | **기준선 (α 0.1; 9단계)** | 86.32 ± 0.92 | 98.87 | 12.56 | 0.1197 | 0.0021 / 0.0229 |
| SHD | α 0.5 | 87.46 ± 0.70 | 99.12 | 11.66 | 0.1217 | 0.0023 / 0.0169 |
| SHD | **α 0.9** | **88.68 ± 0.38** | 99.07 | **10.38** | 0.0996 | 0.0024 / 0.0133 |
| SHD | 원 신호 재구성 | 86.77 ± 0.48 | 99.09 | 12.33 | 0.1156 | 0.0020 / 0.0802 |
| SHD | Spikformer식 (참고) | 86.75 ± 1.32 | 97.82 | 11.07 | – | – |

### 사전 고정 판정 (같은 seed끼리 짝, %p; CinC seed 42 / 7 / 2026, SHD 42 / 7 / 2026 / 123 / 1024)

| 질문 | 과제 | A − B | 평균 | 판정 |
|---|---|---|---|---|
| X1 끄기 − 기준선 | CinC | −2.33 / −1.39 / +3.21 | −0.17 | 차이 없음 |
| X1 끄기 − 기준선 | SHD | +1.15 / −0.80 / −1.90 / −2.56 / −2.08 | −1.24 | **나쁨** |
| X2 원 신호 − 기준선 | CinC | −2.85 / −0.96 / +1.94 | −0.62 | **나쁨** (경계) |
| X2 원 신호 − 기준선 | SHD | +1.50 / +1.33 / −1.15 / +1.28 / −0.71 | +0.45 | 차이 없음 |
| X3 α 0.5 − 기준선 | CinC | −2.29 / +0.60 / +3.06 | +0.46 | 차이 없음 |
| X3 α 0.5 − 기준선 | SHD | +1.02 / +1.81 / +0.71 / +1.81 / +0.35 | +1.14 | **좋음** |
| X3 α 0.9 − 기준선 | CinC | −0.68 / −0.18 / +3.05 | +0.73 | 차이 없음 (좋은 seed 1개) |
| X3 α 0.9 − 기준선 | SHD | +3.80 / +3.40 / +1.28 / +1.99 / +1.37 | +2.37 | **좋음** |
| X4 각 설정 − Spikformer식 | CinC | 4개 설정 | −3.93 ~ −5.28 | 나쁨 4/4, 기준 (2) 미충족 |
| X4 α 0.9 − Spikformer식 | SHD | +1.28 / +2.39 / +3.36 / −0.18 / +2.83 | +1.93 | **좋음, 기준 (2) 충족** |
| X4 α 0.5 / 원 신호 / 끄기 − Spikformer식 | SHD | | +0.71 / +0.02 / −1.67 | 차이 없음 / 차이 없음 / 나쁨 |

- **SHD:** population에서는 보조 손실이 실제로 쓰인다. 끄면 나빠지고(−1.24), 비중을 올릴수록 좋아진다(+1.14, +2.37). 이득은 모두 새 화자 일반화에서 온다(검증은 98.8–99.1%로 같고, 검증 − 시험 차이가 13.7 → 10.4%p로 준다). LIF 묶음(어떤 설정도 차이 없음)과 정반대다.
- **CinC:** 평균은 α와 함께 오르지만(60.77 → 60.94 → 61.40 → 61.67) seed 사이 방향이 갈려 판정은 모두 "차이 없음"이다. 원 신호 재구성은 경계에서 나쁨(−0.62). Spikformer식과의 차이는 −3.9 ~ −5.3%p로 그대로다.
- **발화율의 방향이 과제마다 반대다.** CinC에서는 α가 커질수록 신피질 발화율이 오르고(0.040 → 0.060 → 0.115; 10-21 BN 점검에서 찾은 "CinC population 첫 신피질 뉴런 침묵"을 보조 손실이 일부 되살린다), SHD에서는 내려간다(0.12 → 0.10). CinC에서는 발화가 되살아나도 점수로 이어지지 않았다.
- **사전 예상과 비교:** "끄기와 원 신호는 차이 없음" — CinC 끄기만 맞음(SHD 끄기는 나쁨, CinC 원 신호는 경계 나쁨). "α 0.9는 약간 나쁠 수 있음" — 두 과제 모두 틀림(SHD 좋음, CinC 차이 없음). "CinC에서 Spikformer식을 넘는 설정은 없음" — 맞음.
- **범위(사용자 지적 10-21 반영):** SHD 결과는 원래 시간 칸 처리 안에서의 비교이므로 T = N 효과로 해석하지 않는다.

### 한계
- 탐색 결과다(CinC fold 0 × 3 seed). CinC α 0.9의 평균 +0.73은 seed 1개(+3.05)가 만든 값이다.
- 에너지는 Spikformer식의 약 25배 그대로다. SHD에서 기준 (2)는 "더 좋음" 조건으로 충족했다.

### 점검·감사
- 32 run 모두 학습·검증·시험·에너지까지 끝났다(SKIP 0). 감사 102(10-21 20:21)는 새 판단 근거가 없다.

**다음:** f-LIF 묶음(32 run, 진행 중) → 중간 기록 3/3·결과 항목. 아이디어 묶음(사용자 지시 10-21; 사전 고정은 별도 항목)은 smoke가 돌고 있다.

**commit:** 이 항목과 population 결과를 commit·push한다.

---

## 2026-10-21 01:00 KST — SHD 제외 결정과 아이디어 묶음 설계·구현·사전 고정: BiX(순방향·역방향 스트림 사이 토큰 축 교차 주의)와 사용자 원안 BiX-ch × 뉴런 8종, soma 없는 population 두 형태, Spikformer 주의 BN 대조 (**CinC만 54 run**) + 원인 분리 population 보충 — 점검 31/31, smoke 20/22(실패 2 = GPU 메모리 부족); soma 없는 두 형태(24 run)는 과다 발화가 드러나 보정을 다시 정한 뒤 별도 항목으로 시작, 나머지 30 run 시작

**SHD 제외 (사용자, 10-21 23:3x):** "혹시 현재 SHD 실험이 돌고 있는 것이 있어? SHD는 더 이상 우리의 고려되는 task가 아니게 돼. T=N 주장을 유지해야 하기 때문에."
- 그 시점 돌던 run 8개가 모두 SHD였다: 보조 손실 f-LIF 묶음 `nsmt_flif_auxoff` SHD seed 42/7/2026/123(시작 7–23분 뒤) 4개, 아이디어 묶음 smoke(SHD popk, 1 epoch) 4개.
- **조치:** (i) 대기 중인 SHD run 120개를 큐에서 뺐다(각 큐의 잠금을 잡고 편집; 보조 손실 16, 아이디어 smoke 14, 아이디어 본 학습 90; 뺀 줄은 `scripts/queues/removed-shd-20261021/`에 보관). (ii) 보조 손실 SHD run 4개는 PID로 정확히 중단했다(`results/flif-v1-rhythm-aux-shd/raw.txt`에 SKIP rc=143 네 줄). pool은 곧바로 CinC run으로 넘어갔다. (iii) SHD smoke 4개는 끝나게 두었다 — 중단하면 드라이버가 "smoke 실패"로 멈추고, popk 코드 경로 확인용으로는 쓸모가 있다(결과는 판정·주장에 쓰지 않음). (iv) 원인 분리 드라이버를 CinC만 돌도록 고쳤다(실행 전이라 안전; 원인 분리 100 → **45 run**: 가벼움 33, 무거움 12).
- **기록상 의미:** 지금까지의 SHD 결과(9단계, 보조 손실 LIF·population 묶음 — 특히 "population + α 0.9가 Spikformer식보다 +1.93%p")는 기록으로 남기되 연구 주장에서 뺀다. 보조 손실 실험의 f-LIF 묶음은 CinC 12 run만 남는다(사전 고정의 SHD 20 run은 사용자 결정으로 취소). T = N 주장은 CinC(연속 ECG)에서만 다룬다.
- 같은 GPU에서 다른 작업 계열의 ETT 학습(`etthard-2u-20261021`, GPU당 약 1.5 GB)이 돌고 있다. 건드리지 않았다.

**사용자 지시 (10-21, 5개 항목)와 답:**
1. "SHD는 이미 spiking time step과 data의 time step을 일치할 수 있는 데이터 포맷 … T=N 효과를 주장할 수 있는 task가 아닌 것 같아." → 동의. 8단계 조사에서도 SHD 논문 대부분이 10 ms 칸 하나를 한 스텝으로 넣는다(T = 칸 수). 이어서 사용자가 SHD를 과제에서 뺐다(위). T = N 주장은 CinC 같은 연속 신호에서만 가능한데, 지금 그 과제에서는 T = N NSMT가 T = 4 Spikformer식에 지고 있어 받쳐 줄 결과가 없다.
2. "아까 내가 제안했던 아이디어(bidirectional + cross-attention)는 왜 실험 안 해?" → 직전 질문에서 제가 "1-A/1-C 결과를 본 뒤 결정"을 추천했고 그 선택지가 골라졌다(비용과 "주의가 들어오면 뉴런 기억의 몫이 다시 작아질" 위험 때문). 이번에 돌린다 — 머리 묶음 형태와 **사용자 원안(채널별 T×T′ 지도)** 둘 다.
3. "Population LIF 실험 옵션도 비교에 항상 넣어줘. (구조 비교 실험 시)" + 계획 의견 "Population f-LIF와 population LIF 둘 다 비교에 넣으라는 것" → 메모리에 상시 규칙으로 기록. AskUserQuestion 범위: **"아이디어 묶음만 넓힘"** — 원인 분리 묶음에는 population f-LIF 보충 3조건만 넣고 population LIF 보충은 넣지 않는다.
4. "Population LIF/f-LIF neuron에서 soma가 꼭 필요할까? … Population 축도 두면 되지 않아? 그리고 나서 head에서 정보를 합치는 것" → 문헌 정리: 지금 구조는 가지 → soma(DH-SNN, Nat. Commun. 2024)와 같고, soma 없는 형태는 층 안 τ 이질성(Perez-Nieves 등 Nat. Commun. 2021 — SHD에서 이득 보고)이 표준이며, "줄기를 head까지 분리"는 그것과 출력 집단 투표를 합친 형태로 흔하지 않다. 사용자 질문 "보통 Population coding은 어떻게 수행돼?"에 ① 값 부호화(가우시안 수용장·집단 벡터), ② 출력 집단 투표, ③ 층 안 시간 척도 이질성, ③′ 가지 → soma로 답했다. AskUserQuestion: **두 형태 모두**(층 안 τ 그룹 / 줄기 분리).
5. "Batch Normalization을 점검해줘 … BN 이후에 Spiking Neuron의 발화량이 급격하게 줄어드는 현상이 있는지도" → 아래 점검 결과. 순서: AskUserQuestion **"아이디어 묶음 먼저"**(원인 분리 76 run보다 앞).

### BN 점검 (seed 42, 13개 모델, 시험 앞쪽 48단위; 학습 뒤 = 최고 검증 가중치·평가 모드 BN, 초기 = 같은 설정 새 가중치·학습 모드 BN; 스크립트는 scratchpad의 읽기 전용 측정, 수치는 이 항목에 기록)
- **구조:** 모든 스파이킹 뉴런 앞에 Linear + BN이 있다. 예외는 **주의 출력 뉴런(attn_lif)** 하나 — NSMT 교차 주의(MCA)와 Spikformer식 SSA 모두, 주의 점수 × 값의 결과가 BN 없이 뉴런에 들어간다(저자 코드·Spikformer 설계). BN·뉴런 없는 선형층은 분류층과 재구성 디코더(출력용)뿐이다.
- **발화가 크게 줄어드는 곳:**

| 위치 | 초기 발화율 | 학습 뒤 | 비교 |
|---|---|---|---|
| Spikformer식 SSA 주의 출력 뉴런 | SHD 0.0041, CinC **0.0007** | 0.2159 / 0.1199 | 다른 층 초기 0.04–0.06 |
| NSMT 교차 주의 출력 뉴런 | 0.0148–0.0181 | 0.049–0.149 | |
| **CinC population NSMT 첫 신피질 뉴런** | 0.0056 | **0.0073 / 0.0078 / 0.0090 (seed 3개 전부)** | LIF 0.060–0.061, SHD population 0.088–0.109 |

- 원인: (a) 주의 출력 — BN이 없고 점수 배율 384^−0.5 = 0.051·문턱 1.0이라 학습 초기 거의 침묵. (b) CinC population 첫 층 — BN 출력은 정상(평균 −0.03, 표준편차 0.97, 1을 넘는 비율 0.11)인데, BN이 평균을 0으로 맞춘 입력을 느린 리셋 없는 가지(τ 32 정상 상태 표준편차 ≈ 0.13)가 평균 내 지워 soma 합이 문턱을 거의 못 넘는다(SHD는 말소리 구간에 입력이 몰려 괜찮음). 그 밖의 층은 학습 뒤 0.05–0.26으로 붕괴 없음.
- 대응: 두 BiX 형태 모두 주의 출력 앞에 BN을 둔다("연산 + BN + 뉴런 한 세트"), 공정 비교를 위해 Spikformer식에도 같은 BN을 넣은 `spk_ssa_abn`을 추가한다. (b)는 대기 중인 원인 분리 V1(soma 가중치 1.5·2.0)이 직접 건드린다.

**브랜치·기준:** `exp/nsmt-flif-neo`, 기준 commit `9efdbc87d`. 프로토콜은 9단계 CinC 그대로(fold 0 × seed 42/7/2026, lr 1e-3, 30 epoch, 배치 32, CAL 가중 CE). 기준선은 9단계 run 재사용(`mix_lif_bidir`, `mix_flif_bidir`, `mix_pop_bidir`, `mix_none`, `spk_ssa`). suite `flif-v1-rhythm-bix-cinc`.

### 모델
- **공통 골격 = Neo-mixer:** 조각 임베딩 → 블록 1개(`x ← x·(1 − 섞기)`, `x ← x·(1 − MLP(x))`) → 조각 평균 → head. 모든 뉴런은 토큰 축을 시간으로 쓴다(T = N).
- **BiX** (`--mixer_comb xattn`): 순방향 신피질 2층 F, 역방향 2층 R(기존 `MixerStack`) → 토큰 축 교차 주의(`bix.TokenXAttn`): q = 뉴런(BN(Linear(F))), k·v = 뉴런(BN(Linear(R))), 점수_h[t, t′] = q_h[t]·k_h[t′] × 384^−0.5 + 상대 위치 편향(헤드 8, 0 초기화) → @v → **BN** → 주의 뉴런 → proj(Linear + BN + 뉴런). Neo-mixer의 OR 결합만 교차 주의로 바뀐다.
- **BiX-ch** (`--mixer_comb xattn_ch`, 사용자 원안): 점수_d[t, t′] = q[t,d]·k[t′,d](채널마다 T×T′ 지도, 헤드 합·위치 편향 없음) → 출력[t,d] = Σ_t′ 점수_d[t,t′] v[t′,d] → BN → 주의 뉴런 → proj. 0/1 스파이크에서는 출력 = q[t,d]·Σ_t′ k[t′,d] v[t′,d]·배율로 줄어든다(c26이 실제 텐서에서 차이 0으로 확인 — 토큰별 상호작용이 없음). 실험으로도 확인한다.
- **뉴런 축 (8종):**
  - LIF(대조, spikeDE), f-LIF(α 0.5): 신피질(mixer) 층만, 나머지 층은 spikingjelly LIF τ 2(9단계와 같음).
  - **population LIF** = `PopulationFLIF`에서 α = 1(pred·전체 이력 → 가지가 정확한 Euler LIF, c31: 수동 반복과 차이 0). **soma 가중치 0.5** — {0.25, 0.5, 1.0} 중 초기 mixer 발화율이 LIF mixer에 가장 가까운 값(측정: CinC 0.0019 / **0.0476** / 0.3032 vs LIF 0.0798; SHD 0.0183 / **0.1004** / 0.3424 vs LIF 0.0977). population f-LIF는 9단계대로 1.0(초기 0.0373 / 0.0817).
  - **popg_lif / popg_flif** (soma 없음, 층 안 τ 그룹): 모든 뉴런 층(임베딩·신피질·주의·MLP)의 유닛을 96 × 4 그룹(τ 4/8/16/32)으로 나눔, 각 유닛이 스스로 발화·리셋. 신피질 층 = spikeDE LIF(τ_k) 또는 f-LIF(τ_k, α 0.5), 나머지 층 = spikingjelly LIF(τ_k). 다음 선형층이 모든 그룹을 함께 읽음. 비용은 기존과 같음.
  - **popk_lif / popk_flif** (soma 없음, 줄기 분리 — 사용자 원안; `stream_mixer.StreamMixer`): 임베딩 선형층 출력을 4줄기로 복제, 줄기마다 τ_k인 같은 신경망(선형층 가중치 공유, BN만 줄기별), head에서 줄기별 조각 평균을 이어 붙여 Linear(4·384 → 클래스). 계산·에너지 약 4배. K = 1·τ 2에서 Neo-mixer(f-LIF)와 로짓이 비트 동일(c28).
  - **τ 보정(사전 규칙):** τ_k 뉴런 앞 BN의 가중치 초기값 = √((2τ_k − 1)/3) × 2(τ > 2). √ 부분은 계획대로(백색 입력에서 τ 2 LIF와 막전위 분산을 맞춤)였으나 **계획의 식만으로는 τ 32 그룹이 초기 0.2–0.7%로 c27·c28(≥ 1%)에 실패**했다(리셋 뒤 느린 회복을 반영하지 못함). 그래서 τ > 2에 곱하는 공통 배율을 {1, 1.5, 2, 3} 중 "두 데이터·네 모델의 모든 층·모든 그룹이 ≥ 1%"를 만족하는 최솟값으로 정했다. 측정(그룹 τ 4/8/16/32의 층별 최소 발화율):

| 배율 | CinC (popg_lif / popk_flif) | SHD (popg_lif / popk_flif) |
|---|---|---|
| ×1 | 0.018/0.010/0.006/0.003 · 0.017/0.010/0.006/0.003 | 0.021/0.015/0.009/0.006 · 0.024/0.014/0.008/0.003 |
| ×1.5 | 0.047/0.026/0.015/**0.008** · 0.045/0.026/0.015/**0.008** | 0.048/0.033/0.023/0.014 · 0.053/0.032/0.019/0.011 |
| **×2** | 0.073/0.041/0.024/0.014 · 0.072/0.042/0.024/0.013 | 0.080/0.056/0.037/0.023 · 0.087/0.054/0.032/0.018 |
| ×3 | 0.114/0.069/0.041/0.024 · 0.113/0.069/0.041/0.024 | 0.133/0.094/0.066/0.041 · 0.140/0.095/0.059/0.036 |

- **원인 분리 보충(population f-LIF):** `nsmt_pop_pe`(무거움), `mix_pop_bidir_pe`, `mix_pop_rec_bidir`(population soma에 게이트 재귀 전류 tanh(g)·W_rec·s_{t−1}, g 초기 0 — `layers.RecurrentPopulationFLIF`; `neurons.py` 불변, c8).

### 조건 (아이디어 묶음, CinC 3 seed — SHD 제외로 144 → 54 run)
| 묶음 | 조건 | run |
|---|---|---|
| BiX | `bix_{lif, flif, pop_lif, pop_flif, popg_lif, popg_flif, popk_lif, popk_flif}` | 8 × 3 = 24 |
| BiX-ch | `bixch_{lif, flif, pop_lif, pop_flif}` (원안 형태 확인용; soma 없는 두 형태는 BiX·mixer에서 비교하므로 제외) | 4 × 3 = 12 |
| soma 비교 (Neo-mixer, OR) | `mix_pop_lif_bidir`(새 기준선), `mix_popg_{lif,flif}_bidir`, `mix_popk_{lif,flif}_bidir` | 5 × 3 = 15 |
| 주의 BN 대조 | `spk_ssa_abn` | 3 |

원인 분리 묶음: CinC만 45 run(보충 `nsmt_pop_pe`·`mix_pop_bidir_pe`·`mix_pop_rec_bidir` 9 run 포함; 아이디어 묶음 뒤 자동 시작).

### 코드 (`nsmt_flif_v1/classification/`; 기존 조건 출력 비트 동일 — c1·c2·c13–c24)
- 새 `bix.py`(`TokenXAttn`, `StreamSpikeLinear`), 새 `stream_mixer.py`(`StreamMixer`, `StreamNeurons`).
- `neo_mixer.py`: `Block(comb='or'|'xattn'|'xattn_ch')`, 뉴런 `pop_lif`·`pop_rec`·`popg_lif`·`popg_flif`(그룹 τ 전 층 + BN 보정).
- `layers.py`: `GroupNeuron`, `tau_gain`(TAU_GAIN_SLOW = 2), `group_gamma`, `sj_lif`, `RecurrentPopulationFLIF`, `build_neo_neuron('pop_lif'|'pop_rec')`, `SSA_rel_scl(attn_bn)`.
- `model.py`: `attn_bn` 전달. `config.py`: `--mixer_comb`, `--attn_bn`, mixer 선택지, 디렉터리 접미사(`_xa`, `_xach`, `_abn`, `_smix-…`). `load_model.py`: `streamMixer`. `energy.py`: population LIF는 이력 항 없는 에너지(α = 1은 Euler). `train_rhythm.py`: 끝에 최대 GPU 메모리 출력.
- `scripts/run_rhythm2.sh`: **돌고 있는 `run_rhythm.sh`의 복사본**(10-21 사고 뒤 규칙) + 새 조건; 원인 분리 큐도 이것을 쓴다(상위 집합). `scripts/make_queue_rhythm.sh`: kind `bixlight`/`bixheavy`/`bixsmoke`, fix kind에 보충 3조건. 새 `scripts/rhythm_bix_driver.sh`. `summarize_rhythm.py --bix`, `--fix`에 Y2p·Y3p·Z1p. `rhythm_analysis.py`: 줄기 축을 펼쳐 측정.

### 점검 31/31 (`results/checks/check_cls_bix.txt`)
- c1–c24 그대로 통과.
- c25 BiX: 모든 뉴런 T = N, 주의 지도 [B, 1, 8, N, N], q는 F·k/v는 R에서만, 토큰 j 교체가 j 앞 토큰 출력도 바꿈.
- c26 BiX-ch: 주의 BN 직전 출력 = q·Σk v·배율과 **차이 0**(머리 묶음 형태는 1.28로 다름).
- c27 popg: 그룹 τ·BN 보정 확인, 모든 층·그룹 초기 발화율 ≥ 1%(최소 CinC 0.0137, SHD 0.0228).
- c28 popk: K = 1·τ 2·가중치 복사 → Neo-mixer(f-LIF)와 **로짓 비트 동일**(state dict 51/51 정렬); τ가 같으면 4줄기 출력 동일; head 입력 1,536; 모든 층·줄기 ≥ 1%(최소 CinC 0.0134).
- c29 pop_rec: g = 0이면 population f-LIF mixer와 비트 동일, g가 기울기를 받음, 재귀 경로를 키우면(g = 1, W_rec × 100) 출력이 달라짐(처음엔 초기 W_rec ± 0.005가 스파이크를 하나도 바꾸지 못해 검사가 너무 약했다 — 고침).
- c30 `--attn_bn`: 추가 파라미터는 attn_bn뿐, 끄면 기본 모델과 비트 동일, 초기 주의 발화율 0.0000 → 0.0463.
- c31 population LIF: 가지 상태 = 수동 Euler 반복(차이 0), soma 출력 0/1, 초기 mixer 발화율(psi 0.5) CinC 0.0476 / SHD 0.1004.

### smoke (정식 배치 32, 1 epoch, seed 42; 10-21 23:25 – 10-22 00:57; 실험 아님)
- `results/smoke-rhythm-bix-20261021/raw.txt`: **RESULT 20, SKIP 2.** SHD 4개는 SHD 제외 전에 시작한 것이라 끝까지 두었다(판정에 쓰지 않음). CinC 18개 중 16개 통과.
- **실패 2건 — CinC `bix_popk_flif`, `mix_popk_flif_bidir`: GPU 메모리 부족**(이 job이 5.34 GB를 잡았을 때 남은 공간 11 MB). 보조 손실 CinC f-LIF NSMT(GPU당 약 40–45 GB)와 다른 작업 계열의 ETT 학습(GPU당 약 1.5 GB) 옆에서 돌렸기 때문이다. 코드 오류는 아니다(같은 모델이 SHD와 CinC LIF판에서 완주). 남는 메모리를 GPU당 약 7 GB로 본 제 추정이 GPU 0·1에서 틀렸다(45.5 / 47.3 GB 사용 중). 드라이버는 사전 규칙대로 smoke 뒤 멈췄다(00:57).
- 최대 GPU 메모리(train_rhythm 출력): BiX·BiX-ch 1.9–3.6 GB, SHD 줄기 분리 4.5–5.1 GB.
- 에너지 규모 확인: CinC 줄기 분리(LIF) BiX 168.9 µJ vs τ 그룹(LIF) BiX 52.1 µJ → 약 3.2배(4줄기 설계와 맞는 규모). Spikformer식 + 주의 BN 108.0 µJ(9단계 Spikformer식 127.4).

| CinC smoke | 1 epoch 점수 | 총 µJ (FFT) | 시간 s |
|---|---|---|---|
| bix_lif / bix_flif | 0.3107 / 0.3474 | 32.8 / 37.4 | 297 / 475 |
| bix_pop_lif / bix_pop_flif | 0.2740 / 0.3003 | 44.5 / 72.3 | 582 / 631 |
| bixch_lif / bixch_flif | 0.2772 / 0.2914 | 31.6 / 36.7 | 345 / 602 |
| bixch_pop_lif / bixch_pop_flif | 0.2800 / 0.2919 | 43.6 / 71.2 | 616 / 518 |
| mix_pop_lif_bidir | 0.3525 | 35.2 | 493 |
| spk_ssa_abn | 0.3397 | 108.0 | 50 |
| bix_popg_lif / bix_popg_flif | **0.2488 / 0.2488** | 52.1 / 49.0 | 1,136 / 1,956 |
| mix_popg_lif / mix_popg_flif | **0.2488 / 0.2488** | 30.2 / 32.3 | 667 / 1,572 |
| bix_popk_lif / mix_popk_lif | **0.2503 / 0.2487** | 168.9 / 80.4 | 1,160 / 1,125 |

- **soma 없는 두 형태의 과다 발화 (smoke가 잡은 보정 문제):** popg·popk 6개는 1 epoch 뒤 시험에서 거의 모두 "정상"으로만 예측했다(0.2488 = 정상만 맞힐 때의 CinC 점수). 학습 손실(1.13–1.18)은 BiX LIF(1.15)와 같은 수준이라 학습 실패라고 보지는 않는다. 그러나 층별 발화율이 기준 모델보다 훨씬 높다. 초기(학습 전, 실제 CinC 16표본, 학습 모드 BN)에 다시 재었다:

| 층 | LIF mixer (기준) | popg_lif τ 4/8/16/32 | popk_lif τ 4/8/16/32 |
|---|---|---|---|
| 임베딩 | 0.029 | 0.072 / 0.041 / 0.025 / 0.013 | 0.071 / 0.042 / 0.024 / 0.013 |
| 신피질 1층 | 0.037 | 0.140 / 0.144 / 0.146 / 0.146 | 0.146 / 0.142 / 0.137 / 0.130 |
| 신피질 2층 | 0.041 | **0.265 / 0.288 / 0.310 / 0.330** | **0.223 / 0.275 / 0.322 / 0.347** |
| 주의 q (BiX) | 0.038 | 0.164 / 0.124 / 0.087 / 0.061 | – |
| MLP 1층 | 0.035 | 0.075 / 0.047 / 0.029 / 0.018 | 0.078 / 0.047 / 0.027 / 0.015 |

  원인: τ 보정 배율 ×2는 "모든 층·그룹 ≥ 1%"라는 하한만 맞췄고 상한이 없었다. 하한을 묶은 것은 spikingjelly LIF 층(임베딩·MLP)의 느린 그룹이었는데, 같은 배율을 받은 spikeDE 신피질 뉴런은 리셋이 θ/τ만 빼는 방식이라(τ 32면 문턱의 1/32) 느린 그룹일수록 문턱 근처에 머물며 계속 발화한다. 그래서 신피질이 기준의 3–9배로 발화한다. 이대로 돌리면 "soma 제거" 효과와 "과다 발화" 효과가 섞인다.
  **결정:** popg·popk 8조건(24 run)은 보류하고, 뉴런 종류별로 초기 발화율을 기준(τ 2 같은 층)에 맞추는 보정 규칙을 다시 정해 점검·smoke한 뒤 **별도 사전 고정 항목**으로 시작한다(판정 정의 P1·B1·B3 해당 행은 그대로 둔다). 나머지 10조건(30 run)은 보정과 무관하므로 지금 시작한다.

### 실행
- smoke 드라이버 `rhythm_bix_driver.sh`(10-21 23:25 – 10-22 00:57, smoke 실패 규칙으로 정지) 다음은 새 드라이버 `scripts/rhythm_bix_driver2.sh`(tmux `flif_bix`)가 맡는다:
  - 큐 A = 보정이 정해진 10조건 30 run(`bix`·`bixch` × LIF·f-LIF·population LIF·population f-LIF, `mix_pop_lif_bidir`, `spk_ssa_abn`). 이 항목 commit 뒤 `scripts/queues/bix.go`를 만들면 지금 메모리 여유가 있는 GPU 2·3에서 1개씩 시작, 보조 손실 "ALL DONE" 뒤 GPU 0–3에 2개씩으로 늘림.
  - 큐 B = popg·popk 24 run. 보정 재설정 항목 commit 뒤 `scripts/queues/bix_b.go`를 만들면 정식 배치 smoke(8조건, 실패 시 정지) → 본 학습.
  - 두 큐가 끝나면 `summarize_rhythm.py --bix` → `rhythm_fix_driver.sh`(원인 분리 CinC 45 run).
- **계획과 다른 점:** (a) 계획은 아이디어 묶음을 보조 손실 종료 뒤에 시작하는 것이었으나, mixer 수준 run의 메모리가 작아(smoke 최대 1.9–3.6 GB) 메모리 여유가 있는 GPU에서 **사전 고정 직후 바로 시작**한다(사용자 결정 "아이디어 묶음 먼저"를 더 일찍 실현). (b) popg·popk `_lif`의 비신피질 층은 계획의 spikeDE LIF 대신 spikingjelly LIF(τ_k) — 다른 모든 모델과 같은 뉴런이고, K = 1 비트 동일 검사(c28)가 가능하다. (c) population LIF soma 0.5는 위 측정 규칙으로 정했다. τ 보정 배율 ×2는 smoke에서 과다 발화가 드러나 popg·popk와 함께 보류하고 다시 정한다. (d) 대기 중이던 원인 분리 드라이버(`tmux flif_fix`, 시작한 job 없음, 20:12부터 대기)는 종료하고 이 드라이버가 이어받는다.
- **예상 시간(SHD 제외 뒤):** 보조 손실은 CinC f-LIF 12 run만 남아 10-22 낮 무렵 끝난다(추정). 아이디어 묶음 54 run은 10-23 무렵, 원인 분리 45 run은 10-24 무렵 끝난다(추정).

### 판정 (사전 고정; 9단계 규칙 — 평균 ≥ +0.5%p이고 seed 3개 중 2개 이상; 누락·비유한 값은 미완)
- **B1 결합 방식:** `bix_n` − Neo-mixer(OR) 같은 뉴런 (8개 전부; 9단계 `mix_lif/flif/pop_bidir` 재사용, 나머지는 이 묶음).
- **B2 Spikformer식과:** `bix_n` − `spk_ssa`, − `spk_ssa_abn` (기준 (2)).
- **B3 BiX 안의 뉴런:** 각 뉴런 − `bix_lif`; 분수 차수 효과 = f-LIF − LIF 짝 4쌍(`flif/lif`, `pop_flif/pop_lif`, `popg_flif/popg_lif`, `popk_flif/popk_lif`).
- **B4 원안 형태:** `bixch_n` − `bix_n`, `bixch_n` − Neo-mixer(OR) (n = lif, flif, pop_lif, pop_flif).
- **P1 soma:** 차수(LIF·f-LIF)마다 그룹 − soma, 줄기 − soma, 줄기 − 그룹(분리 효과); Neo-mixer와 BiX 각각.
- **F1:** Neo-mixer population f-LIF − population LIF.
- **A1:** `spk_ssa_abn` − `spk_ssa`.
- 원인 분리 보충(원인 분리 요약에 포함): Y2p `nsmt_pop_pe` − `nsmt_pop`, Y3p `mix_pop_bidir_pe` − `mix_pop_bidir`, Z1p `mix_pop_rec_bidir` − `mix_pop_bidir`.
- 보고: 토큰 축 기억, 그룹·줄기별 발화율, 에너지, 주의 초기·학습 뒤 발화율, CinC population 첫 층 발화율, 학습 뒤 g.
- **예상(사전 기록):** B1 교차 주의는 OR보다 좋음(전역 접근) — +1%p 이상. B2 CinC는 여전히 Spikformer식보다 낮을 가능성이 큼. B3 분수 차수 효과는 대부분 차이 없음. B4 원안 형태는 머리 묶음보다 나쁘고 Neo-mixer(OR)와 비슷. P1 그룹 ≈ soma, 줄기는 차이 없거나 약간 좋음(에너지 4배). F1 차이 없음. A1 CinC에서 약간 좋음.

**감사:** 102(10-21 20:21)까지 새 판단 근거 없음.

**commit:** 이 항목·코드·점검·smoke를 commit·push한 뒤 `bix.go`를 만들어 본 학습을 시작한다.

---

## 2026-10-22 01:24 KST — 아이디어 묶음 보충 사전 고정: soma 없는 population(popg·popk) 24 run의 보정 규칙을 "층별 발화율 맞춤"으로 다시 정함 — 모든 층·그룹의 초기 발화율이 기준 모델 같은 층의 0.98–1.02배, 점검 31/31

**구분:** 01:00 항목(commit `12caf85e1`)에서 보류한 popg·popk 8조건(24 run, CinC)의 사전 고정 보충. 조건·판정 정의(B1·B3·P1 해당 행)는 01:00 항목 그대로이고, 바뀌는 것은 BN 가중치 초기값(τ 보정)뿐이다.

**왜 다시 정했나:** 10-21의 규칙(√((2τ − 1)/3) × 2)은 "모든 층·그룹 ≥ 1%"라는 하한만 맞춰, spikeDE 신피질 뉴런을 기준의 3–9배로 과다 발화시켰다(01:00 항목 표). 첫 대안으로 뉴런 종류별 표(spikingjelly LIF / spikeDE LIF / spikeDE f-LIF마다, τ 2 뉴런과 같은 초기 발화율을 내는 이득을 대표 입력에서 이분 탐색)를 만들었으나, 신피질 **2층**의 느린 그룹이 여전히 기준의 3.5배(popg_lif)·4.6배(popk_lif)·1.8–2.4배(f-LIF판)였다 — 2층 입력은 1층 스파이크라 분포가 달라 한 표로 맞지 않았다.

**규칙 (사전 고정):** `calibrate_groups.py` — 모델마다(popg_lif / popg_flif / popk_lif / popk_flif × 결합 OR / 교차 주의 = 8개), 그룹 뉴런 층을 순전파 순서대로 하나씩: 그 층이 실제로 받는 입력(위층 보정 뒤)을 잡아, τ 그룹마다 BN 가중치를 이분 탐색(25단계)해 그룹의 초기 발화율이 **기준 모델(같은 결합의 spikeDE LIF / f-LIF Neo-mixer) 같은 층의 발화율**과 같아지게 한다. 데이터: CinC fold 0, seed 42의 학습 창 0–31, 모델 seed 0, 학습 모드 BN. 결과는 고정 표 `calib/tau_gains_<model>_<mixer>_<comb>.json`으로 저장하고, 모델 생성자가 τ 4/8/16/32일 때 불러 쓴다(`layers.apply_tau_calibration`; 표가 없거나 그룹 층이 빠지면 오류로 멈춤). 모든 run이 같은 초기값에서 시작한다. 실행 기록 `log/calibrate_groups-20261022.log`.

**보정 결과 (BN 가중치, τ 4/8/16/32; 일부 층):**

| 모델 / 결합 | 임베딩 | 신피질 1층 | 신피질 2층 | 주의 q | MLP 1층 |
|---|---|---|---|---|---|
| popg_lif / OR | 1.88 / 3.73 / 7.54 / 17.38 | 1.59 / 2.39 / 3.44 / 4.82 | 1.08 / 1.37 / 1.72 / 2.41 | – | 1.95 / 3.78 / 7.78 / 17.36 |
| popg_lif / 교차 주의 | 1.86 / 3.75 / 7.80 / 17.03 | 1.62 / 2.39 / 3.35 / 4.82 | 1.12 / 1.38 / 1.74 / 2.28 | 1.37 / 2.40 / 4.36 / 8.32 | 1.94 / 3.71 / 7.71 / 17.30 |
| popg_flif / OR | 1.88 / 3.73 / 7.54 / 17.38 | 1.73 / 3.11 / 5.57 / 10.23 | 1.49 / 2.48 / 4.12 / 7.70 | – | 1.94 / 3.75 / 7.75 / 17.18 |
| popk_lif / OR | 1.91 / 3.70 / 7.69 / 17.18 | 1.64 / 2.45 / 3.44 / 4.64 | 1.37 / 1.55 / 1.64 / 1.73 | – | 1.95 / 3.81 / 7.83 / 16.78 |
| popk_flif / 교차 주의 | 1.91 / 3.72 / 7.75 / 17.36 | 1.76 / 3.11 / 5.55 / 9.89 | 1.63 / 2.56 / 3.99 / 6.24 | 1.80 / 3.06 / 5.02 / 8.56 | 1.94 / 3.78 / 7.69 / 16.45 |

(전체 8개 표와 모든 층은 `calib/*.json`.) 리셋이 θ/τ만 빼는 spikeDE LIF는 같은 τ에서도 spikingjelly LIF보다 훨씬 작은 이득(τ 32: 4.6–4.8 vs 17)만 필요하다는 것이 이전 과다 발화의 원인을 다시 확인해 준다.

**점검 31/31 (`results/checks/check_cls_bix2.txt`):** c1–c24·c25·c26·c29–c31 그대로. c27·c28을 "모든 층·그룹의 초기 발화율이 기준 모델 같은 층의 0.5–2배이고 ≥ 1%"로 강화(CinC만):

| 모델 | OR: 기준 대비 비율 범위 | 교차 주의: 비율 범위 | 최소 발화율 |
|---|---|---|---|
| popg_lif | 1.00 – 1.00 | 0.98 – 1.00 | 0.0273 / 0.0279 |
| popg_flif | 1.00 – 1.00 | 0.99 – 1.02 | 0.0273 / 0.0279 |
| popk_lif | 1.00 – 1.00 | 0.99 – 1.01 | 0.0274 / 0.0279 |
| popk_flif | 1.00 – 1.00 | 0.99 – 1.00 | 0.0274 / 0.0279 |

c28의 K = 1·τ 2 비트 동일 검사는 그대로 통과(보정은 τ 4/8/16/32에서만 적용).

**실행:** `tmux flif_bix`의 `rhythm_bix_driver2.sh`가 `scripts/queues/bix_b.go`(이 항목 commit 뒤 생성)를 보고, **보조 손실 pool이 끝나 메모리가 비면**(popk f-LIF는 공유 GPU에서 메모리 부족이 났다) 8조건 정식 배치 smoke(실패 시 정지) → 본 학습 24 run을 GPU당 2개로 돌린다. 큐 A(30 run)는 그 사이 계속 돈다.
- **운영 기록:** 큐 A는 01:02부터 드라이버의 GPU 2·3 pool에 더해, 메모리 여유가 생긴 GPU 0·1에 같은 큐를 읽는 pool을 하나 더 띄웠다(`tmux flif_bix_a01`, 큐 잠금 공유). 01:00 항목의 "GPU 2·3에서 1개씩"과 다르다. 결과에는 영향이 없다(run마다 같은 명령).

**판정·예상:** 01:00 항목 그대로. 다만 P1(soma 대 soma 없음)의 비교는 이제 **초기 발화율이 맞춰진 상태**의 비교다(soma population은 9단계·이번 묶음의 보정값 그대로).

**commit:** 이 항목·보정 스크립트·표·점검을 commit·push한 뒤 `bix_b.go`를 만든다.

---

## 2026-10-22 06:38 KST — 9단계 추가 실험(NSMT 보조 손실 옵션) 결과: **CinC에서는 세 뉴런 모두 어떤 보조 손실 설정도 기준선보다 좋지 않음(12개 비교 중 "좋음" 0, "나쁨" 5) — NSMT가 Spikformer식에 지는 것(−3.9 ~ −5.3%p)은 보조 손실 때문이 아니다**; SHD 결과(population + α 0.9가 Spikformer식보다 +1.93%p)는 SHD 제외 결정으로 연구 주장에서 뺀다 — 태그 `exp/nsmt-flif-neo-20261022`

**구분:** 10-20 16:57 사전 고정(commit `80a79940d`)의 완료 기록(탐색). 중간 기록 1/3(LIF, `7e1b1e98a`), 2/3(population, `9efdbc87d`) 다음의 f-LIF 묶음과 종합이다. 이 항목이 속한 commit에 태그 `exp/nsmt-flif-neo-20261022`를 붙인다.

**실행:** tmux `flif_aux`, 10-20 16:57 – 10-22 06:30, 드라이버 "ALL DONE". CinC 36 run(뉴런 3 × 옵션 4 × seed 3) 완료·SKIP 0(RESULT 4줄은 실행 중 스크립트 수정 사고로 산출물에서 다시 만든 것, `collected=1`). SHD는 LIF·population 40 run 완료, f-LIF 20 run은 10-21 23:3x 사용자 결정(SHD 제외)으로 취소 — 실행 중이던 4개는 중단(SKIP rc=143), 대기 16개는 큐에서 뺐다. (정정: 01:2x 항목의 보정 실행 기록 `log/calibrate_groups-20261022.log`는 Git 제외 규칙에 걸려 로컬에만 있다; 이득 값은 `calib/*.json`과 그 항목의 표에 있다.)

### CinC 결과 (fold 0 시험, 최고 검증 가중치, seed 42/7/2026 평균 ± 표준편차, CinC 점수 %)

| NSMT 뉴런 | 끄기 (α 0) | **기준선 (α 0.1, 저역)** | α 0.5 | α 0.9 | 원 신호 (α 0.1) | 신피질 발화율 (끄기 → 0.1 → 0.5 → 0.9) |
|---|---|---|---|---|---|---|
| LIF | 61.47 ± 1.37 | 61.76 ± 1.40 | 61.03 ± 0.44 | 61.02 ± 1.23 | 61.20 ± 1.13 | 0.058 → 0.079 → 0.105 → 0.128 |
| f-LIF | 61.36 ± 1.06 | 61.50 ± 1.20 | 60.86 ± 0.21 | 60.66 ± 0.53 | 61.38 ± 1.15 | 0.054 → 0.088 → 0.111 → 0.129 |
| population f-LIF | 60.77 ± 0.94 | 60.94 ± 2.24 | 61.40 ± 0.60 | 61.67 ± 0.62 | 60.31 ± 0.17 | 0.067 → 0.040 → 0.060 → 0.115 |
| Spikformer식 (참고) | | 65.60 ± 1.48 | | | | |

**사전 고정 판정 (CinC, 기준선 대비 %p; seed 42 / 7 / 2026):**

| 질문 | LIF | f-LIF | population f-LIF |
|---|---|---|---|
| X1 끄기 | −0.29, 차이 없음 | +2.25 / −0.60 / −2.07 = −0.14, 차이 없음 | −0.17, 차이 없음 |
| X2 원 신호 | −0.56, **나쁨** | +1.38 / −1.87 / +0.14 = −0.12, 차이 없음 | −0.62, **나쁨** |
| X3 α 0.5 | −0.73, **나쁨** | +0.80 / −1.42 / −1.30 = −0.64, **나쁨** | +0.46, 차이 없음 |
| X3 α 0.9 | −0.75, 차이 없음 | +1.15 / −1.67 / −2.00 = −0.84, **나쁨** | +0.73, 차이 없음 |
| X4 각 옵션 − Spikformer식 | −4.12 ~ −4.58, 나쁨 4/4 | −4.21 ~ −4.93, 나쁨 4/4 | −3.93 ~ −5.28, 나쁨 4/4 |

- **보조 손실은 CinC에서 NSMT의 성능을 설명하지 않는다.** 끄기(X1)는 세 뉴런 모두 차이 없음이고, Spikformer식과의 차이는 어떤 설정에서도 −3.9 ~ −5.3%p로 9단계(−3.83 ~ −4.66)와 같은 수준이다(기준 (2) 12/12 미충족).
- 비중을 올리면 LIF·f-LIF는 약간 나빠지는 쪽(4개 중 3개 "나쁨"), population은 평균이 오르지만 seed가 갈려 판정은 차이 없음이다. 모든 "나쁨"은 −0.56 ~ −0.84%p로 seed 흩어짐(1–2%p) 안쪽이다.
- 신피질 발화율은 세 뉴런 모두 α와 함께 오른다(보조 손실이 신피질을 더 발화시킴). 성능에는 반영되지 않는다.
- **SHD(기록만, 주장에서 제외):** LIF는 어떤 설정도 차이 없음, population은 α 0.5·0.9에서 좋음(+1.14 / +2.37%p)이고 α 0.9가 Spikformer식보다 +1.93%p였다(중간 기록 1/3·2/3). SHD는 원래 시간 칸 데이터라 T = N 주장에 쓸 수 없어 10-21에 과제에서 뺐다. 이 SHD 이득이 CinC로 옮겨가지 않았다는 점(CinC population α 0.9: 차이 없음, Spikformer식보다 −3.93%p)만 기록해 둔다.
- **사전 예상과 비교:** "끄기와 원 신호는 차이 없음" — 끄기는 3/3 맞음, 원 신호는 LIF·population에서 경계 "나쁨"이라 틀림. "α 0.9는 약간 나쁠 수 있음" — f-LIF만 맞음(LIF·population 차이 없음). "CinC에서 Spikformer식을 넘는 설정은 없음" — 맞음.

### 한계
- 탐색 결과다(CinC fold 0 × seed 3). "나쁨" 판정은 모두 기준선 −0.5%p 경계 근처다.
- 보조 손실 비중은 0–0.9만 보았고, 재구성 목표는 저역 15%와 원 신호 두 가지뿐이다.

### 점검·감사
- CinC 36 run 모두 학습·검증·시험·에너지까지 끝났다(SKIP 0). 감사 103–105는 새 판단 근거가 없다.

**다음:** 아이디어 묶음(큐 A 30 run 진행 중, 큐 B 24 run의 smoke가 06:34 시작)과 원인 분리 묶음(CinC 45 run, 아이디어 묶음 뒤 자동)이 이어진다.

**commit/tag:** 이 항목·f-LIF 결과·요약을 commit하고 태그 `exp/nsmt-flif-neo-20261022`를 붙여 브랜치와 태그만 push한다.

---

## 2026-10-22 11:01 KST — 선형 합 Neo-mixer 사전 고정(CinC 12 run)과 원인 분리 묶음 일정 변경(가벼운 33 run 지금 시작), 사용자 질문 답(층 수·잔차 연결) — 점검 32/32, smoke 4/4

**사용자 질문·제안 (10-22):**
1. "Neo mixer에서 역방향 순방향 각 나온 값을 linear 통과 하여 가중치합을 하는 것은 어때? 그리고 역방향 순방향 각각 몇개의 레이어로 쌓여있어?" → **각 방향 2층**(NeoLayer = Linear 384→384 + BN + 뉴런; 두 방향 가중치 따로; 블록 1개; 9단계 `mix_lif_bidir` run의 `logargs.txt`·모듈 목록으로 확인). 제안 분석: OR는 학습하는 것이 없고 어느 방향이 발화했는지 구분하지 못한다. 선형 합은 두 방향을 채널마다 학습해 섞고(문턱으로 AND 같은 조합도 가능), **같은 조각 위치끼리만** 합치므로 조각 사이 정보는 여전히 정렬된 막전위로만 오간다(T = N 주장과 가장 잘 맞는 구조). 그리고 BiX의 OR 대비 이득(LIF +0.96, f-LIF +1.36)이 (a) 학습되는 합치기 층 때문인지 (b) 주의의 전역 접근 때문인지를 가르는 대조군이 된다. AskUserQuestion: **지금 큐 B 옆에서 바로**, **선형층 형태**(m = 뉴런(BN(W[F; R])); 스칼라 형태는 넣지 않음).
2. "각 Spikformer 식 모델도 residual connection이 XOR인 상태야?" → 아니다. **모든 모델(NSMT, Spikformer식, Neo-mixer, BiX, 선형 합)이 IAND** `x ← x·(1 − f(x))`(저자 ECG 코드의 방식; x가 1이고 f가 0일 때만 1)를 쓴다. XOR(x + f − 2xf)도, 원 Spikformer의 덧셈 잔차(x + f, 값이 2까지)도 아니다. 그래서 하위 모듈(주의·섞기·MLP)은 스파이크를 끄기만 하고 새로 만들 수 없다. 내부 비교는 같은 조건이라 공정하지만, **우리 "Spikformer식"은 원 Spikformer와 잔차 방식이 다르다** — 문헌 비교에 쓸 때 밝혀야 한다. 잔차 방식을 원인 후보로 시험할지는 AskUserQuestion에서 **"나중에 결정"**.
3. "Spikformer식 모델과의 성능차이에 대한 원인 분석 실험은 언제 시작해?" → 원래 일정은 아이디어 묶음 큐 B 뒤(오늘 저녁 시작, 10-23 저녁 종료 추정). AskUserQuestion: **가벼운 33 run을 지금 바로**.

### A. 선형 합 Neo-mixer (사전 고정)
- **조건:** `mixlin_{lif, flif, pop_lif, pop_flif}` × CinC fold 0 × seed 42/7/2026 = 12 run(뉴런 축 규칙: LIF·f-LIF·population LIF·population f-LIF; population soma 가중치는 아이디어 묶음과 같은 0.5 / 1.0). suite `flif-v1-rhythm-lin-cinc`, 러너 `scripts/run_rhythm3.sh`(큐 B가 `run_rhythm2.sh`를 실행 중이라 복사본; 10-21 사고 뒤 규칙).
- **구현:** `neo_mixer.Block(comb='lin')`: `bix.StreamSpikeLinear(768 → 384, K = 1)` = Linear(이어 붙인 [F; R]) → BN → spikingjelly LIF(τ 2, 토큰 축 T = N). 나머지(임베딩·두 방향 2층·IAND·MLP·head)는 Neo-mixer와 같다. `config.py`: `--mixer_comb lin`, 디렉터리 접미사 `_lin`. `summarize_rhythm.py --lin`.
- **점검 32/32** (`results/checks/check_cls_lin.txt`): c1–c31 그대로, c32 = Linear (384, 768)·주의 모듈 없음, 합친 출력 0/1, `mix()` = `fuse([F; R])`, 모든 뉴런 T = N, 순방향만 바꿔도·역방향만 바꿔도 합친 출력이 바뀜(초기 발화율 0.034).
- **smoke** (`results/smoke-rhythm-lin-20261022/raw.txt`, 1 epoch, seed 42; 4/4 통과): mixlin_lif 0.2788 / 26.6 µJ / 224 s, mixlin_flif 0.2903 / 31.6 µJ / 438 s, mixlin_pop_lif 0.3161 / 38.6 µJ / 519 s, mixlin_pop_flif 0.3043 / 66.0 µJ / 478 s. 파라미터 1,211,904(population 1,218,048) = Neo-mixer 916,224 + 294,912 + 768로 설계와 일치. 최대 GPU 메모리 0.26–1.74 GB.
- **판정(같은 규칙: 평균 ≥ +0.5%p이고 seed 3개 중 2개 이상):** L1 `mixlin_n` − Neo-mixer OR(같은 뉴런; LIF·f-LIF·population f-LIF는 9단계, population LIF는 아이디어 묶음 `mix_pop_lif_bidir`), L2 `mixlin_n` − `bix_n`, L3 `mixlin_n` − Spikformer식(기준 (2)).
- **해석 틀(사전):** L1 좋음 + L2 차이 없음 → BiX의 이득은 주의가 아니라 학습되는 합치기에서 온다. L1 차이 없음 + (BiX > OR) → 주의(전역 접근)가 핵심이다.
- **예상(사전 기록):** L1은 LIF·f-LIF에서 좋음(+0.5–1%p), population은 차이 없음. L2는 차이 없거나 BiX가 약간 좋음. L3는 미충족.

### B. 원인 분리 묶음 일정 변경 (10-21 20:12 사전 고정의 조건·판정은 그대로, CinC만)
- **가벼운 33 run**(위치: `spk_ssa_nope`·`mix_lif_bidir_pe`·`mix_pop_bidir_pe`; 도달 거리: `mix_lif_rec_bidir`·`mix_sjlif_bidir`·`mix_pop_rec_bidir`·`mix_pop_bidir_pt16`; 해마 T = 4: `nsmt_lif_t4`·`nsmt_pop_t4`; mixer soma 가중치: `mix_pop_bidir_psi1.5`·`_psi2`)은 **지금** 새 드라이버 `scripts/rhythm_fix_light_driver.sh`(tmux `flif_fix_light`, 11:00 시작): 11조건 정식 배치 smoke(실패 시 정지) → `scripts/queues/fix_light.go`(이 항목 commit 뒤 생성) → GPU 4장에 1개씩 → `--fix` 요약(무거운 부분이 끝나기 전엔 중간본).
- **무거운 12 run**(`nsmt_lif_pe`·`nsmt_pop_pe`·`nsmt_pop_psi1.5`·`_psi2`, GPU당 35–45 GB)은 큐 B와 같은 GPU에 들어가지 않아 그대로 큐 B 뒤: `rhythm_fix_driver.sh`를 **무거운 부분만** 돌도록 고쳤다(실행 전이라 안전; `rhythm_bix_driver2.sh`가 큐 B 뒤에 호출; 무거운 4조건 정식 배치 smoke 뒤 시작).
- 이 때문에 지금 GPU당 최대 4개 run(큐 B 2, 선형 합 1, 원인 분리 1)이 함께 돈다. 각 run은 느려지지만 결과에는 영향이 없다(run마다 같은 명령·seed).

**commit:** 이 항목·코드·점검·smoke를 commit·push한 뒤 선형 합 12 run을 시작하고 `fix_light.go`를 만든다.

---

## 2026-10-23 17:26 KST — 진단(학습 없음, 보고만): Neo-mixer 합치기 지점에서 막전위가 스파이크보다 리듬 정보를 더 갖는가 (CinC fold 0, 8 조건 × seed 3 = 24 모델, validation) — **일관된 차이 없음**; f-LIF만 |H_f − H_r|(막전위판 XOR)를 더하면 3/3 seed에서 좋음(+3.2점)

**직전 commit:** `cd4024d50` (선형 합 Neo-mixer 사전 고정).

**목적:** 사용자 질문(10-23) "Neo-mixer에서 순방향·역방향 정보를 spike가 아닌 각 시점의 membrane potential로 {Attention, Linear combination, XOR} 등을 연산하는 것은 어때?" (`NSMT/docs/About_idea.md` 4.4). 학습 전에 먼저, 지금 학습된 모델들의 합치기 지점에 막전위 쪽에만 있는 정보가 있는지 잰다. 작업 범위는 사용자가 고른 "(나)"이고, (가)는 `exp/f-lif-pop-v3`의 사전등록 2Y(`docs/PROJECT_LOG.md`, worktree)다.

**대상:** OR로 합치는 양방향 Neo-mixer 8 조건(뉴런 축 규칙: LIF·f-LIF·population LIF·population f-LIF + soma 없는 popg·popk 두 형태) — `mix_lif_bidir`, `mix_flif_bidir`, `mix_pop_lif_bidir`, `mix_pop_bidir`(population f-LIF), `mix_popg_lif_bidir`, `mix_popg_flif_bidir`, `mix_popk_lif_bidir`, `mix_popk_flif_bidir`. seed 42·7·2026, 9단계와 아이디어 묶음 run. 가중치는 `model_state/checkpoint.pth`(train_rhythm.py가 val 점수가 오를 때마다 저장하고 'best' 시험에 다시 읽는 파일; `best+model.pt`는 마지막 epoch). 모델 자신의 val 점수가 config의 best_val_score와 24개 모두 정확히 같다.

**방법 (`nsmt_flif_v1/classification/fusion_probe.py`, sha256 b002091c…):**
- 각 방향 마지막 mixer 층의 출력 스파이크 F, R과 문턱 전 충전값 H(LIF·f-LIF: v + (I − v)/τ, soma가 있는 population: soma 충전값)를 잡는다. R과 H_r은 토큰 축으로 다시 뒤집는다.
- 특징(토큰 평균, 스트림 축은 펼침): or = max(F, R)(모델이 쓰는 것), spk = [F; R], mem = [H_f; H_r], memxor = [H_f; H_r; |H_f − H_r|], spkmem = [F; R; H_f; H_r], input = 블록 입력, head = 모델 head 입력.
- probe: 표준화 + 다항 logistic 회귀(class_weight balanced), C ∈ {0.001, 0.01, 0.1, 1}은 train 기록의 10% 내부 hold-out에서 CinC 점수로 고르고 train 전체로 다시 맞춘 뒤, validation 창 확률을 기록별로 평균해 `rhythm_eval.metrics`로 잰다(점수 = N·AF·Other F1 평균). probe train은 train 기록마다 가운데 창 하나(6139), validation은 683 기록 / 896 창. test는 만들지 않았다.
- 점검(모든 batch, 실패 시 중단): H에서 만든 스파이크 = 출력 스파이크(정확히), max(F, R) = 모델의 m0(정확히), 계측한 forward의 로짓 = 보통 forward(차이 0.0). 24 모델 모두 통과, 192개 회귀 모두 수렴.
- 명령: `nsmt_flif_v1/classification`에서 `OMP_NUM_THREADS=16 … python fusion_probe.py --models <cond:seed …> --skip_done`(GPU 0–3에 나눠서), 끝에 `--summarize`. 실행 로그와 특징 캐시(24 npz, 3.4 GB)는 로컬 `log/fusion-probe-20261023/`.

**결과 (val 점수 ×100, seed 3개 평균; `results/fusion-probe-20261023/summary.txt`·`summary.csv`):**

| 조건 | 모델 자신 | or | spk | mem | memxor | spkmem | input | head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LIF | 64.02 | 59.91 | 61.41 | 58.10 | 60.39 | 60.22 | 52.15 | 61.66 |
| f-LIF | 63.66 | 58.04 | 57.12 | 58.08 | 60.33 | 58.32 | 52.80 | 61.21 |
| population f-LIF | 64.90 | 61.14 | 61.44 | 60.62 | 61.70 | 61.11 | 55.08 | 61.86 |
| population LIF | 65.45 | 62.45 | 61.88 | 62.08 | 61.72 | 61.26 | 53.22 | 62.87 |
| popg LIF | 61.30 | 61.37 | 60.19 | 59.98 | 57.59 | 61.22 | 52.97 | 62.04 |
| popg f-LIF | 60.22 | 58.07 | 56.92 | 58.28 | 58.21 | 57.26 | 52.73 | 60.23 |
| popk LIF | 59.10 | 57.49 | 57.33 | 56.93 | 58.12 | 57.91 | 52.76 | 58.37 |
| popk f-LIF | 62.12 | 55.97 | 56.98 | 56.45 | 57.84 | 58.21 | 53.93 | 59.06 |

| 조건 | mem − spk | memxor − spk | spk − or |
|---|---|---|---|
| LIF | **−3.32** (−3.71 / −3.34 / −2.90) | −1.03 (+0.92 / −0.15 / −3.85) | **+1.51** (+0.92 / +1.04 / +2.56) |
| f-LIF | +0.96 (−1.44 / +0.40 / +3.91) | **+3.20** (+1.60 / +3.90 / +4.10) | −0.92 (−1.52 / +1.16 / −2.41) |
| population f-LIF | −0.82 (+0.11 / +0.07 / −2.64) | +0.26 (+1.79 / +0.58 / −1.60) | +0.30 |
| population LIF | +0.21 (−0.27 / +0.22 / +0.67) | −0.16 | −0.57 |
| popg LIF | −0.21 | **−2.60** (−1.94 / −1.34 / −4.52) | −1.19 |
| popg f-LIF | **+1.36** (+2.50 / +0.61 / +0.97) | +1.29 (+5.08 / +1.25 / −2.46) | −1.15 |
| popk LIF | −0.40 | +0.79 (−0.99 / +1.61 / +1.75) | −0.15 |
| popk f-LIF | −0.52 | +0.87 (−0.85 / +1.54 / +1.91) | +1.00 |

(괄호 = seed 42 / 7 / 2026. 굵게 = 세 seed 부호가 같음.)

- 발화율(F / R): LIF 0.079, f-LIF 0.066, population 0.14–0.19, popg·popk 0.10–0.13. 문턱 바로 아래(−0.5 < H − θ < 0)에 머무는 몫은 12–22%다.

**읽는 법 (탐색, 학습 없는 선형 probe):**
- **막전위(mem)가 스파이크(spk)보다 리듬 정보를 더 갖는다는 일관된 근거는 없다.** 8 조건 중 세 seed 부호가 같은 것은 LIF(막전위가 −3.3점 나쁨)와 popg f-LIF(+1.4점)뿐이다. 나머지는 ±1점 안에서 seed마다 갈린다.
- **막전위판 XOR(|H_f − H_r|)는 f-LIF에서만 3/3 seed로 좋다(+3.2점)**. popg LIF에서는 3/3 나쁘다(−2.6점). 뉴런 종류에 따라 반대라서 일반 규칙은 아니다.
- **OR로 합치며 잃는 것(spk − or)도 일관되지 않다.** LIF만 3/3에서 [F; R]이 OR보다 낫다(+1.5점). 어제 사전 고정한 선형 합(`mixlin_*`)이 겨냥한 것과 같은 지점이다.
- ETT(v3, 같은 날)에서는 스파이크가 막전위보다 7–35% 더 나빴지만, CinC Neo-mixer 합치기 지점에서는 그런 차이가 보이지 않는다. v3 모델은 출력층이 스파이크를 직접 읽고 회귀하는 구조, 여기는 토큰 평균 뒤 분류라서 같은 질문이 다른 답을 낸다.

**한계:**
- 토큰 평균 선형 probe다. 시점별 정보(IAND가 토큰마다 쓰는 것)는 보지 않는다. 모델 자신보다 head probe가 2–3점 낮다.
- probe train 기록이 모델의 학습 기록이라 내부 hold-out 점수가 부풀려져 있다(val보다 중앙값 0.19 높음). C 선택이 그만큼 흔들린다.
- 잡음 수준: 추출 batch 크기만 32 → 128로 바꿔도 probe 점수가 최대 ±0.25점 움직였다(C가 거의 동점에서 바뀐 경우 포함). seed 간 차이는 흔히 몇 점이다.
- `results/fusion-probe-20261023/bs32_partial_run/`은 이전 스크립트 버전(batch 32)으로 만든 4 모델 결과다. 최종 값과 0.003 안에서 같으며, 요약에 쓰지 않았다(보존).

**판단:** 학습 없이 본 합치기 지점에서는 "막전위로 합치면 낫다"를 지지하는 근거가 약하다. 학습해서 확인할 가치가 있는 것은 (i) 선형 합 입력을 막전위로 바꾼 것(`lin-mem`)과 (ii) f-LIF의 |H_f − H_r| 항 정도다. 진행 여부는 사용자 결정.

**commit:** 이 항목·스크립트·결과를 commit 하고 `exp/nsmt-flif-neo`만 push (특징 캐시와 실행 로그는 로컬).

---

## 2026-10-23 21:30 KST — 기록 보충: 아이디어 묶음(54 run)·원인 분리 묶음(45 run)·선형 합 Neo-mixer(12 run) 결과 — 끝났지만 이 기록에 결과 항목이 없던 세 묶음

**경위:** 세 묶음은 10-22–10-23에 모두 끝났고(`results/flif-v1-rhythm-{bix,fix,lin}-cinc/raw.txt`: 54 / 45 / 12 RESULT, SKIP 0), `summarize_rhythm.py`가 각 `summary.md`를 만들어 두었다. 그런데 결과 항목과 commit이 없었다. 이번에 막전위 합치기·해마 T = 4 정렬 실험(다음 항목)의 비교 기준이 되므로 여기에 정리하고 결과 파일을 commit 한다. 선형 합 summary는 10-22 15:32(12 중 일부만 끝났을 때) 판이라, 12 run 전부로 다시 만들었다(`summarize_rhythm.py --lin --suite flif-v1-rhythm-lin-cinc`). 나머지 두 summary는 모든 run이 끝난 뒤 만든 판 그대로다. 판정 규칙은 모두 "평균 ≥ +0.5%p이고 seed 3개 중 2개 이상"(CinC test, 최고 검증 가중치).

**선형 합 (L, 10-22 사전 고정):**
- L1 OR → 선형 합: LIF **+0.83 좋음**, f-LIF **+0.82 좋음**, population LIF **−0.71 나쁨**, population f-LIF **+1.40 좋음**.
- L2 선형 합 vs BiX: LIF −0.13 차이 없음, f-LIF −0.53 나쁨, population LIF +0.43 차이 없음, population f-LIF **+2.51 좋음**.
- L3 vs Spikformer식(기준 (2)): 네 뉴런 모두 나쁨(−3.24 ~ −5.02), 미충족.
- 사전 예상(L1 LIF·f-LIF 좋음, population 차이 없음; L2 차이 없거나 BiX 약간 좋음; L3 미충족)과 비교: L1 LIF·f-LIF 맞음, population 두 개 틀림(f-LIF 좋음, LIF 나쁨). L3 맞음.

**아이디어 묶음 (B·P·F·A, 10-21 사전 고정; `results/flif-v1-rhythm-bix-cinc/summary.md`):**
- B1 OR → 교차 주의(BiX): LIF +0.96, f-LIF +1.36 좋음; population LIF −1.14, population f-LIF −1.11 나쁨; soma 없는 popg·popk 네 개 모두 크게 나쁨(−3.6 ~ −6.1).
- B2 vs Spikformer식(기준 (2)): 8개 모두 나쁨(−3.7 ~ −12.5), 미충족. 주의 BN을 넣은 Spikformer식과 비교해도 같다.
- B4 채널별(BiX-ch) vs 머리 묶음: f-LIF 차이 없음, 나머지 나쁨.
- **P1 soma → soma 없는 population:** Neo-mixer에서 그룹(popg) −3.79 / −1.22, 스트림(popk) −3.81 / −2.43(LIF / f-LIF) 모두 나쁨. BiX 안에서는 −4.9 ~ −8.8로 더 나쁨.
- F1 Neo-mixer population에서 f-LIF(α 0.5) − LIF(α 1): −2.33 나쁨. A1 SSA 주의 BN: 차이 없음.

**원인 분리 묶음 (Y·Z·W·V, 10-21 사전 고정; `results/flif-v1-rhythm-fix-cinc/summary.md`):**
- 위치(1-B): Y1 SSA의 상대 위치 bias를 빼면 −1.11 나쁨. Y2·Y3 NSMT·Neo-mixer에 절대 위치를 넣으면 LIF −1.34 / −1.97 나쁨, population은 +0.74 차이 없음 / −1.95 나쁨. 위치를 맞춰도(Y4) Spikformer식보다 −4.1 ~ −5.9.
- 도달 거리(1-A): Z1 Neo-mixer에 recurrent LIF를 넣으면 크게 나쁨(−13.3 LIF, −10.5 population). Z2 느린 가지(τ 16–128) −0.63 나쁨. spikingjelly LIF − spikeDE LIF −1.26.
- **해마 T = 4 + 토큰 교차 주의(1-C):** W1 NSMT LIF **+1.66 좋음**(0.6342 vs 0.6176), NSMT population f-LIF **−2.28 나쁨**(0.5866 vs 0.6094). W2 vs Spikformer식: −2.18 / −6.93, 미충족. LIF t4는 점수가 오르지만 macro-F1은 내려간다(Noisy F1 약 0.50 → 0.28–0.38).
- soma 가중치(V1): 1.5·2.0 모두 차이 없음(NSMT·Neo-mixer).

**종합:** 지금까지 어떤 신피질 변형도 Spikformer식(65.6)을 넘지 못했다. 가장 가까운 것은 NSMT LIF T_h 4(63.4, −2.2)와 population LIF Neo-mixer(63.1)다. soma 없는 population은 모든 형태에서 soma가 있는 population보다 나빴다.

**commit:** 이 항목과 세 묶음의 `raw.txt`·`summary.md`(선형 합은 다시 만든 판)를 다음 항목과 함께 commit 한다.

---

## 2026-10-23 21:35 KST — 사전 고정: (A) Neo-mixer 막전위 합치기 ②·③ → ④ (CinC 36 run)와 (B) NSMT 해마 T = 4에 신피질 4개 스트림을 정렬 (CinC 12 run) — 점검 37/37, smoke 7/7

**직전 commit:** `8acab9fa0` (합치기 지점 probe).

**사용자 지시 (10-23):**
1. "Neo-mixer에서 막전위로 합치기 항목의 ②, ③ → ④ 을 수행해보자." (앞선 답의 단계표: ① probe(완료), ② 선형 합 입력을 막전위로, ③ ②에 |H_f − H_r|, ④ 막전위 주의.)
2. (다른 작업 계열 `exp/f-lif-pop-v3` 사전등록 2Z — 그쪽 기록.)
3. "[NSMT 해마만 T=4] 모델에서 해마는 non-population neuron, Neocortex path는 population neuron으로 동작한 다음, population k=4로 설정하여 해마의 spiking 시간축(T=4)으로 일치시키는 것은 어때? 그리고 나서 cross-attention으로 두 정보를 합치는 거야. neo에서는 population을 구성하는 각 뉴런이 time-series를 받는데 그 4개의 뉴런들 각각이 4개의 상태를 갖고 있는 것이고, 해마는 일반적인 Spikformer와 유사하게 동작…하나의 시계열에 대해서 4번 반복해서 입력을 형성하는 것."

### A. 막전위 합치기 (②·③·④)

- **공통:** 각 방향 마지막 mixer 층의 문턱 전 충전값 H(LIF·f-LIF: u + (I − u)/τ, soma 있는 population: soma 충전값)에서 θ를 뺀 값을 그래프에 붙인 채로 꺼낸다. `layers.spikes_and_charge`는 뉴런의 forward와 똑같은 rhs / integrate / emit 호출을 따라가며 H를 같이 기록한다. 스파이크는 forward와 비트 단위로 같다(점검 c33). 마지막 층의 출력 스파이크는 쓰지 않고 H만 쓴다.
- **② `mixlinmem_<n>`:** m = LIF(BN(W [H_f − θ; H_r − θ])), Linear 768 → 384. 선형 합(`mixlin_<n>`)과 모양·순서가 같아 같은 seed에서 초기값이 같다(c34).
- **③ `mixlinmx_<n>`:** m = LIF(BN(W [H_f − θ; H_r − θ; |H_f − H_r|])), Linear 1152 → 384.
- **④ `bixmem_<n>`:** BiX(머리 묶음, q = 순방향, k / v = 역방향)에서 q = BN(W_q (H_f − θ)), k = BN(W_k (H_r − θ))를 뉴런 없이 실수로 두고, v = LIF(BN(W_v R))은 스파이크 그대로다. 주의 BN·주의 뉴런·proj는 같다. 같은 seed에서 BiX와 초기값이 같다(c35).
- 나머지(임베딩, 두 방향 2층, IAND, MLP, head, 학습 설정)는 Neo-mixer와 같다. 합친 출력은 0/1이다.
- **뉴런:** `<n>` ∈ {lif, flif, pop_lif(soma 0.5), pop_flif(soma 1.0)}, seed 42 / 7 / 2026 → ② 12 + ③ 12 + ④ 12 = 36 run. suite `flif-v1-rhythm-mem-cinc`, 러너 `scripts/run_rhythm4.sh`(run_rhythm3.sh 복사본 + 새 조건). 학습 순서는 사용자 순서대로 ②·③ 먼저, ④ 뒤.
- **soma 없는 population(popg / popk):** 이번 묶음에 넣지 않았다. 선형 합 자체가 soma 없는 형태로 정의되어 있지 않고(10-22 구현이 거부), 층별 BN 보정 표를 새 합치기마다 다시 만들어야 하며, run당 3–8시간이다. 아이디어 묶음에서 soma 없는 형태가 모든 비교에서 soma 있는 형태보다 나빴다(P1). ②·③·④ 결과를 본 뒤 필요하면 보충한다(뉴런 축 규칙의 예외로 기록).
- **에너지:** syops는 입력이 0/1이 아니면 MAC으로 센다. 그래서 실수 입력의 합치기 Linear는 MAC이다. smoke에서 1 window 시냅스 에너지가 크게 늘었다: `mixlinmx_lif` 263 µJ, `mixlinmem_pop_flif` 222 µJ, `bixmem_pop_lif` 198 µJ(선형 합 smoke 26.6–66.0 µJ). 이유는 768 × 384 × 117 토큰 = 34.5 M MAC ≈ 159 µJ다. 주의 곱(q k^T, s v)은 이전처럼 세지 않는다.
- **판정(같은 규칙):** M1 `mixlinmem` − `mixlin`(막전위 vs 스파이크, 같은 합치기; ②의 주 비교), M2 `mixlinmx` − `mixlinmem`(③: |H_f − H_r|의 몫), M3 `mixlinmx` − `mixlin`, M4 `bixmem` − `bix`(④), M5 각각 − OR, M6 각각 − Spikformer식(기준 (2)). `summarize_rhythm.py --mem --suite flif-v1-rhythm-mem-cinc`.
- **예상(사전 기록; 근거 = 같은 날 학습 없는 probe):** M1은 대부분 차이 없음, LIF는 나쁠 수 있음(probe −3.3점). M2는 f-LIF에서 좋음(probe +3.2), 나머지 차이 없음. M4 차이 없음. M6 모두 미충족.

### B. NSMT 해마 T = 4 + 신피질 4개 스트림 정렬

- **구조:** 해마는 지금의 `nsmt_*_t4`와 같다. 각 패치를 4번 반복해 입력하고, spikingjelly LIF, 교차 주의(SSA_rel_scl, q = 해마)를 쓴다. 신피질은 soma 없는 population 4개 스트림(τ 4 / 8 / 16 / 32, 구성원이 직접 발화·리셋; stream_mixer의 `_Embed` + `_Stack` + `StreamNeurons`)이 토큰 축(T = N = 117)으로 돈다. 출력은 [N, B, 4, D]다.
  - **정렬:** 해마 스텝 t의 key / value = 신피질 스트림 t(τ 순서). 지금의 t4 모델은 한 줄짜리 신피질 출력을 4 스텝에 복사한다.
  - 보조 재구성 손실(α 0.1, DCT 저역 목표)은 스트림마다 같은 목표를 재구성한다.
- **보정:** 층별 발화율 맞춤(10-22 규칙). 신피질 세 층(임베딩, 2층) 각각에서 스트림마다 BN 이득을 이분 탐색으로 정해, 같은 층의 기준 모델(같은 계열 단일 spikeDE 뉴런의 NSMT T_h 4 모델) 발화율과 같게 했다.
  - 스크립트 `calibrate_nsmt_streams.py` → `calib/tau_gains_myModel_popk_{lif,flif}_xaT4.json`. CinC train 창 0..31, seed 0, 학습 모드 BN.
  - popk_lif 이득: 임베딩 1.68 / 2.59 / 3.88 / 5.83, 1층 1.38 / 1.54 / 1.64 / 1.76, 2층 1.23 / 1.29 / 1.36 / 1.59. 목표 0.0309 / 0.0405 / 0.0445.
  - popk_flif 이득: 임베딩 1.80 / 3.26 / 6.02 / 11.34, 1층 1.66 / 2.64 / 4.19 / 6.93, 2층 1.58 / 2.33 / 3.36 / 5.22. 목표 0.0360 / 0.0406 / 0.0434.
  - 모든 층·스트림이 목표의 1.000배로 발화한다(c36).
- **조건 (뉴런 축 규칙):**
  - 새 모델: `nsmt_popk_lif_t4`(LIF 스트림), `nsmt_popk_flif_t4`(f-LIF α 0.5 스트림).
  - 새 기준선: `nsmt_poplif_t4`(population LIF, soma 0.5, 복사), `nsmt_flif_t4`(f-LIF 하나, 복사).
  - 기존 기준선: `nsmt_lif_t4`, `nsmt_pop_t4`(원인 분리 묶음), `spk_ssa`, `nsmt_*`(9단계).
  - 4 조건 × seed 3 = 12 run. suite `flif-v1-rhythm-t4k-cinc`, 러너 `run_rhythm4.sh`.
- **판정:** K1 스트림 정렬 − soma population 복사(popk_lif − poplif_t4, popk_flif − pop_t4), K2 스트림 정렬 − 단일 뉴런 복사(popk_lif − lif_t4, popk_flif − flif_t4), K3 − Spikformer식(기준 (2)). 보고만: flif_t4 − nsmt_flif, poplif_t4 − lif_t4. `summarize_rhythm.py --t4k --suite flif-v1-rhythm-t4k-cinc`.
- **예상(사전 기록):** soma 없는 population이 이전 모든 비교에서 soma 있는 형태보다 2–4%p 나빴으므로 K1은 나쁨이거나 차이 없음, K2는 나쁨, K3 미충족. 스트림이 τ 순서로 해마 스텝에 들어가므로 해마 LIF가 스텝 사이에 상태를 이어 가는 것(τ 4 → 32)이 이득을 줄 가능성은 있다. 이것이 사용자 원안의 새로운 점이다.

### 점검·smoke

- **점검 37/37** (`results/checks/check_cls_mem.txt`): c1–c32 그대로 통과(기본 경로 비트 동일).
  - c33: 네 뉴런 모두 충전값 경로의 스파이크 = forward(비트 단위), 스파이크 = (H − θ ≥ 0), H에 gradient, 에너지 기록기가 호출을 본다.
  - c34: ②·③ Linear 모양, 출력 0/1, mix() = fuse(충전값), 실제 CinC 창에서 두 방향 마지막 층 모두 gradient, 순방향만·역방향만 바꿔도 출력이 바뀜, ②의 초기값 = 선형 합.
  - c35: ④ q는 실수·v는 0/1·출력 0/1, 주의 지도 [8, 1, 8, 117, 117], 초기값 = BiX, 두 방향 gradient. 임의 입력에서는 population LIF의 역방향이 침묵해 순방향 gradient가 0이었으나, 실제 CinC 창에서는 0이 아니다.
  - c36: 스트림 신피질 [117, B, 4, 384], 스트림 τ 4/8/16/32, kv[t] = 스트림 t(스트림 0 ≠ 3), 보정 표 적용과 발화율 1.000배, 스트림별 토큰 인과성, 보조 손실 목표가 스트림마다 같음, hippo_T 3·hippo_xattn 없음 거부.
  - c37: population LIF·f-LIF 기준선이 돌고 에너지 기록기가 pop_lif / flif로 본다.
- **smoke** (`results/smoke-rhythm-mem-20261023/raw.txt`, 1 epoch, seed 42, 7/7 끝까지):

| 조건 | params | 시냅스 에너지 µJ | 시간 s |
|---|---:|---:|---:|
| mixlinmem_pop_flif | 1,218,048 | 222.5 | 300 |
| mixlinmx_lif | 1,359,360 | 263.3 | 131 |
| bixmem_pop_lif | 1,517,896 | 198.1 | 313 |
| nsmt_popk_lif_t4 | 1,271,368 | 135.8 | 326 |
| nsmt_popk_flif_t4 | 1,271,368 | 150.9 | 634 |
| nsmt_poplif_t4 | 1,269,064 | 126.1 | 204 |
| nsmt_flif_t4 | 1,264,456 | 137.8 | 281 |

**코드:** `layers.py`(`spikes_and_charge`, `NeoNeuron.forward_charge`, `GroupNeuron.forward_charge`), `model.py`(`NeoLayer.forward_charge`; myModel의 `pop_lif`·`popk_lif`·`popk_flif`, 스트림 신피질, xattn의 K = T_h 확인, 스트림별 보조 목표), `neo_mixer.py`(`MixerStack.forward_charge`, `linmem` / `linmx` / `xattnmem`), `bix.py`(`TokenXAttn(qk_from='charge')`), `config.py`(선택지·디렉터리 접미사 `_linmem` / `_linmx` / `_xamem`), `load_model.py`(tau_calib), `calibrate_nsmt_streams.py`, `check_cls.py`(c33–c37), `summarize_rhythm.py`(`--mem`, `--t4k`), `scripts/run_rhythm4.sh`.

**학습:** 이 commit 뒤 두 pool을 띄운다. A 36 run은 GPU당 2개(②·③ 먼저, ④ 뒤), B 12 run은 GPU당 1개. 대기열은 `scripts/queues/rhythm-mem-20261023.txt`, `rhythm-t4k-20261023.txt`.

**commit:** 이 항목·코드·점검·보정 표·smoke와 앞 항목의 세 결과 묶음을 commit 하고 `exp/nsmt-flif-neo`만 push.

---

## 2026-10-23 21:22 KST — 정정: 앞 두 항목의 시각

앞 두 항목의 머리말 시각("21:30", "21:35")은 확인하지 않고 적은 값이다. 실제로는 **21:17–21:21**에 썼다(commit `2205768fa` 21:21:43). 두 pool은 21:22에 시작했다(GPU 0–3에 A 8개, B 4개 worker).

---

## 2026-10-24 03:42 KST — 막전위 합치기 결과 (CinC 36 run, test, 최고 검증 가중치): **막전위만으로는 나아지지 않고(M1 차이 없음 / 나쁨), |H_f − H_r|를 더하면 LIF·f-LIF가 크게 좋아짐(M2 +4.58 / +3.05)** — LIF는 65.61로 Spikformer식(65.60)과 같은 점수(차이 없음) 첫 도달, 단 에너지 2.1배로 기준 (2) 미충족; 막전위 주의(④)는 네 뉴런 모두 나쁨

**직전 commit:** `046103a27` (시각 정정). 사전 고정은 `2205768fa`.

**학습:** 36 run이 2026-10-23 21:22 – 10-24 03:3x에 모두 끝났다(SKIP 0; run당 1,863–5,727 s, GPU당 3개가 함께 돎). 명령은 `results/flif-v1-rhythm-mem-cinc/run_commands.txt`, GPU·시각은 `pool_runs.txt`(pool 로그 복사).

**요약:** `summarize_rhythm.py --mem --suite flif-v1-rhythm-mem-cinc` → `results/flif-v1-rhythm-mem-cinc/summary.md`.

| 뉴런 | OR | 스파이크 선형 합 | ② 막전위 선형 합 | ③ 막전위 + \|H_f − H_r\| | BiX (스파이크 q·k) | ④ 막전위 q·k |
|---|---:|---:|---:|---:|---:|---:|
| LIF | 60.54 | 61.37 | 61.03 | **65.61 ± 1.11** | 61.50 | 60.08 |
| f-LIF | 59.75 | 60.58 | 59.13 | **62.18 ± 0.42** | 61.11 | 59.87 |
| population LIF | 63.06 | 62.35 | 61.92 | 60.71 | 61.93 | 56.01 |
| population f-LIF | 60.73 | 62.13 | 60.85 | 61.14 | 59.62 | 55.03 |

(CinC 점수 = N·AF·Other F1 평균, %, seed 42·7·2026 평균. Spikformer식 65.60 ± 1.48.)

| 판정 | LIF | f-LIF | population LIF | population f-LIF |
|---|---|---|---|---|
| M1 ② − 스파이크 선형 합 | −0.34 차이 없음 | **−1.45 나쁨** | −0.44 차이 없음 | **−1.28 나쁨** |
| M2 ③ − ② | **+4.58 좋음** (+4.96 / +6.64 / +2.14) | **+3.05 좋음** (+3.05 / +1.16 / +4.94) | −1.21 나쁨 | +0.29 차이 없음 |
| M3 ③ − 스파이크 선형 합 | **+4.24 좋음** (3/3) | **+1.61 좋음** (3/3) | −1.65 나쁨 | −0.99 나쁨 |
| M4 ④ − BiX | −1.43 나쁨 | −1.24 나쁨 | −5.92 나쁨 | −4.59 나쁨 |
| M5 ③ − OR | **+5.07 좋음** | **+2.43 좋음** | −2.36 나쁨 | +0.41 차이 없음 |
| M6 ③ − Spikformer식 | **+0.02 차이 없음** (−1.84 / +2.40 / −0.50), 에너지 2.10배 → 미충족 | −3.41 나쁨 | −4.89 나쁨 | −4.46 나쁨 |

- ②·④의 M5·M6은 `summary.md`. ②는 OR과 차이 없음 또는 나쁨이고, ④는 OR보다 나쁘거나 차이 없음. 모두 Spikformer식보다 나쁘다.
- **에너지(시냅스 + 뉴런, FFT):** 실수 입력의 합치기·q·k Linear가 MAC으로 잡혀 크게 늘었다. ② 187–224 µJ, ③ 267–304 µJ, ④ 191–228 µJ. 스파이크 선형 합은 31–69, Spikformer식은 127.4 µJ다. ③ LIF는 Spikformer식의 2.10배다.
- **③ LIF의 클래스별 F1(seed 42 / 7 / 2026):** AF 0.610 / 0.664 / 0.618(Spikformer식 0.632 / 0.572 / 0.589)로 높다. Other 0.525 / 0.513 / 0.499(0.558 / 0.546 / 0.537)와 Noisy 0.450 / 0.431 / 0.362(0.456 / 0.531 / 0.542)는 낮다. 그래서 macro-F1은 59.57로 Spikformer식 61.94보다 낮다.
- ③ LIF의 val − test 차이는 2.92 ± 1.00으로, 다른 Neo-mixer(3.2–5.6)보다 작다.

**사전 예상과 비교:**
- M1 "대부분 차이 없음, LIF는 나쁠 수 있음": f-LIF·population f-LIF가 나쁨이라 절반만 맞았다. LIF는 차이 없음.
- M2 "f-LIF에서 좋음, 나머지 차이 없음": f-LIF는 맞았지만 **LIF +4.58은 예상하지 못했다**(probe에서 LIF memxor − spk는 −1.03이었다). population LIF는 나쁨.
- M4 "차이 없음": 틀렸다. 네 뉴런 모두 나쁘다.
- M6 "모두 미충족": 판정은 맞다. 다만 ③ LIF는 점수가 같아 에너지 때문에만 미충족이다.

**읽는 법 (탐색, seed 3개):**
- 이득은 "막전위를 쓴 것"에서 오지 않는다. ②(막전위만)는 스파이크와 같거나 나쁘다. 두 방향 막전위의 **차이 |H_f − H_r|**를 따로 준 것이 단일 뉴런(LIF·f-LIF)에서 크게 효과가 있다. 두 방향이 같은 토큰을 다르게 볼 때 그 크기가 정보가 되는 것으로 읽힌다(가설).
- population(soma 있음)에서는 같은 항이 돕지 않거나 해친다. soma 충전값은 범위가 넓어(표준편차 1.2–1.5, ±30, probe 기록) 차이 항이 BN 뒤에도 다르게 작동할 수 있다(가설, 검정 안 함).
- 막전위 q·k 주의(④)는 일관되게 나쁘다. 실수 q·k와 스파이크 v의 곱은 학습이 불안정하다(seed 간 표준편차 1.7–3.7).
- **아직 가려지지 않은 것:** ③의 이득이 "실수 값"에서 오는지, "차이 정보"에서 오는지. 스파이크만으로 같은 정보를 주는 대조가 필요하다. 예: [F; R; F XOR R] → AC 비용. 이 대조가 이득을 재현하면 에너지 문제 없이 같은 효과를 얻는다.

**한계:** CinC fold 0, seed 3개. 판정 규칙의 +0.5%p 문턱에 비해 seed 간 표준편차가 1–2%p다. 에너지 추정에는 주의 곱이 빠져 있다(이전과 같음).

**commit:** 이 항목·결과(`raw.txt`, `summary.md`, `pool_runs.txt`)를 commit 하고 `exp/nsmt-flif-neo`만 push. 해마 T = 4 스트림 정렬 묶음(12 run 중 9개 끝)은 끝나면 따로 기록한다.

---

## 2026-10-24 03:54 KST — 해마 T = 4 + 신피질 4개 스트림 정렬 결과 (CinC 12 run, test, 최고 검증 가중치): **LIF 스트림은 soma population 복사보다 좋음(K1 +1.02), 단일 LIF 복사와는 점수가 같고(K2 +0.10) macro-F1이 높음(58.99 vs 55.80)**; f-LIF 스트림은 단일 f-LIF 복사보다 나쁨(−3.52); Spikformer식에는 둘 다 못 미침

**직전 commit:** `3be758a38` (막전위 합치기 결과).

**정정 (앞 항목):** "36 run이 … 10-24 03:3x에 모두 끝났다"의 끝 시각은 **03:39:46**이다(`raw.txt` 수정 시각).

**학습:** 12 run이 2026-10-23 21:22 – 10-24 03:52:11에 모두 끝났다(SKIP 0, GPU당 1개 + 막전위 묶음 2개가 함께 돎). run당 시간은 f-LIF t4 2,301–2,905 s, population LIF t4 3,636–5,275 s, LIF 스트림 6,635–7,737 s, f-LIF 스트림 14,129–17,426 s다. 명령은 `results/flif-v1-rhythm-t4k-cinc/run_commands.txt`, GPU·시각은 `pool_runs.txt`.

**요약:** `summarize_rhythm.py --t4k --suite flif-v1-rhythm-t4k-cinc` → `results/flif-v1-rhythm-t4k-cinc/summary.md`.

| 모델 | CinC 점수 | macro-F1 | Noisy F1 (seed 42 / 7 / 2026) | 에너지 µJ |
|---|---:|---:|---|---:|
| Spikformer식 (해마만, T = 4) | 65.60 ± 1.48 | 61.94 | 0.456 / 0.531 / 0.542 | 127.4 |
| T_h 4, LIF 하나 복사 (원인 분리) | 63.42 ± 0.85 | 55.80 | 0.385 / 0.277 / 0.327 | 148.1 |
| T_h 4, f-LIF 하나 복사 (새) | 61.84 ± 2.42 | 55.24 | 0.375 / 0.373 / 0.315 | 148.2 |
| T_h 4, population LIF soma 복사 (새) | 62.50 ± 0.84 | 57.82 | 0.417 / 0.441 / 0.457 | 142.7 |
| T_h 4, population f-LIF soma 복사 (원인 분리) | 58.66 ± 2.25 | 52.12 | | 160.0 |
| **T_h 4, LIF 4 스트림 = 4 스텝 (새)** | **63.52 ± 0.38** | **58.99** | 0.429 / 0.477 / 0.456 | 149.1 |
| T_h 4, f-LIF 4 스트림 = 4 스텝 (새) | 58.33 ± 0.96 | 53.00 | 0.306 / 0.365 / 0.439 | 162.3 |

| 판정 | 결과 |
|---|---|
| K1 LIF 스트림 − population LIF soma 복사 | **+1.02 좋음** (+1.39 / +1.38 / +0.31) |
| K1 f-LIF 스트림 − population f-LIF soma 복사 | −0.33 차이 없음 (+2.26 / −0.24 / −3.03) |
| K2 LIF 스트림 − LIF 하나 복사 | +0.10 차이 없음 (+0.16 / −0.61 / +0.75) |
| K2 f-LIF 스트림 − f-LIF 하나 복사 | **−3.52 나쁨** (−2.04 / −2.26 / −6.24) |
| K3 vs Spikformer식 (기준 (2)) | LIF 스트림 −2.08 나쁨, f-LIF 스트림 −7.27 나쁨; 미충족 |
| K4 (보고) f-LIF t4 − f-LIF T = N | +0.34 차이 없음 (에너지 0.04배) |
| K4 (보고) population LIF soma 복사 − LIF 하나 복사 | −0.92 나쁨 |

**사전 예상과 비교:**
- K1 "나쁨이거나 차이 없음": LIF에서 틀렸다(좋음). f-LIF는 맞았다.
- K2 "나쁨": f-LIF는 맞았고, LIF는 차이 없음이다.
- K3 "미충족": 맞았다.
- 이전 묶음들에서 soma 없는 population은 soma 있는 형태에 늘 졌다. 그런데 이 정렬에서는 처음으로 LIF 스트림이 soma population 복사를 이겼다.

**읽는 법 (탐색, seed 3개):**
- LIF 스트림 정렬은 점수로는 단일 LIF 복사와 같다. 하지만 macro-F1이 3.2점 높고(Noisy F1 0.43–0.48 vs 0.28–0.38), seed 간 표준편차가 가장 작다(0.38). 원인 분리 묶음에서 "T_h 4는 점수를 올리지만 Noisy를 잃는다"고 적었는데, 그 손실이 스트림 정렬에서 거의 사라졌다.
- f-LIF 스트림은 나쁘고 느리다(run당 4–5시간). soma 없는 f-LIF 스트림의 느린 구성원(τ 16·32)은 같은 날 ETT probe에서 스파이크로 가장 많은 정보를 잃었다(구성원 probe). 같은 현상일 수 있다(가설).
- 어느 형태도 Spikformer식(65.60)에는 못 미친다(−2.1 ~ −7.3).

**한계:** CinC fold 0, seed 3개. 스트림 순서(τ 오름차순)를 바꾼 대조가 없다.

**태그:** 이 기록을 담은 commit 에 `exp/nsmt-flif-neo-20261024`(annotated, 완료: 막전위 합치기 36 + 스트림 정렬 12).

**commit:** 이 항목·결과를 commit 하고 `exp/nsmt-flif-neo`와 태그만 push.

---

## 2026-10-24 23:26 KST — 사전 고정: 네 가지 후속 (CinC 84 run) — (1) 스파이크 XOR 대조, (2) ③ LIF의 fold 1–4 확인, (3) 스트림 정렬의 순서 대조와 양방향 합치기, (4) soma 없는 popg의 선형 합 4종 — 점검 41/41, smoke 9/9, 보정 표 10개

**직전 commit:** `7afd3d2fc` (스트림 정렬 결과, 태그 `exp/nsmt-flif-neo-20261024`).

**사용자 지시 (10-24):** "4개 전부 진행". 앞 보고의 다음 결정 1–4다.

### (1) 스파이크 XOR 대조 `mixlinxor_<n>` (12 run)
- m = LIF(BN(W [F; R; |F − R|])). 0/1 스파이크에서 |F − R|은 F XOR R과 같다(c38). Linear 1152 → 384로 ③(`mixlinmx`)과 모양·순서가 같아, 같은 seed에서 초기값이 같다(c38). 막전위는 쓰지 않는다.
- 입력이 0/1이라 합치기 Linear는 AC로 잡힌다. smoke 1 window: `mixlinxor_lif` 27.7 µJ(③ LIF 263 µJ).
- 뉴런 4종(LIF, f-LIF, population LIF soma 0.5, population f-LIF soma 1.0) × seed 3. suite `flif-v1-rhythm-xor-cinc`.
- 판정: X1 xor − 스파이크 선형 합(차이 항이 스파이크로도 돕는가), X2 xor − ③(실수 차이 vs 스파이크 차이), X3 − OR, X4 − Spikformer식(기준 (2)). `summarize_rhythm.py --xor --suite flif-v1-rhythm-xor-cinc`.
- **예상(사전 기록):** X1은 LIF에서 좋음(+1 ~ +3), X2는 나쁨(−1 ~ −2, 크기 정보가 빠짐), X4는 LIF도 나쁨.

### (2) ③ LIF fold 1–4 (36 run)
- `mixlinmx_lif`, `mixlin_lif`, `spk_ssa` × fold 1–4 × seed 3. fold 0은 기존 suite에서 쓴다. CinC 분할 규칙(`rhythm_data.cinc_split`: 기록 단위 층화 5-fold, 검증 10%)은 그대로다. suite `flif-v1-rhythm-folds-cinc`.
- 판정(`summarize_rhythm.py --folds --suite flif-v1-rhythm-folds-cinc`): F1 ③ − Spikformer식, F2 ③ − 스파이크 선형 합, F3 스파이크 선형 합 − Spikformer식.
  - fold마다 기존 규칙(평균 ≥ +0.5%p, seed 2/3)으로 판정한다.
  - fold 0–4의 15쌍(fold × seed)을 합친 평균, 95% t 구간(자유도 14), A > B 쌍 수도 낸다.
- **예상:** F1 15쌍 평균 −1 ~ 0으로 차이 없음, F2 +3 ~ +4 좋음.

### (3) 스트림 정렬 대조 (12 run)
- **순서 대조** `nsmt_popk_{lif,flif}_t4_srev`: 해마 스텝 t에 스트림 3 − t(τ 32 → 4)를 넣는다. 신피질과 초기값은 원래 스트림 모델과 같다(c40).
- **양방향 합치기** `nsmt_popk_{lif,flif}_t4_bimx`(사용자의 "③과 결합"을 이렇게 정의했다): 신피질 스트림을 순방향과 역방향(자기 파라미터)으로 두 번 돌린다. 스트림마다 m_k = LIF(BN(W [H_f,k; H_r,k; |H_f,k − H_r,k|]))로 합쳐(Linear 공유, BN은 스트림별) 해마 스텝 k에 넣는다.
  - 역방향 스택도 같은 자리의 기준 층 발화율에 맞춰 보정했다(`calibrate_nsmt_streams.py --bimx` → `calib/tau_gains_myModel_popk_{lif,flif}_xaT4_bimx.json`, 6 층 모두 1.000배, c41).
  - 순방향은 인과, 역방향은 반인과다(c41).
  - **비용:** params 2.05 M(스트림 모델 1.27 M). 실수 입력 합치기가 4 스트림이라 smoke 1 window 에너지가 1,157 µJ다(f-LIF; 스트림 모델 151 µJ, Spikformer식 127 µJ). 성능이 좋아도 에너지 기준은 맞추지 못한다.
- seed 3, suite `flif-v1-rhythm-t4k2-cinc`. 판정: S1 순서 뒤집기 − 원래, S2 양방향 합치기 − 원래, S3 − 단일 뉴런 복사, S4 − Spikformer식. `summarize_rhythm.py --t4k2`.
- **예상:** S1 차이 없음, S2 LIF 좋음(+1 ~ +2)·f-LIF 차이 없음, S4 미충족.

### (4) soma 없는 popg 선형 합 (24 run)
- `mixlin` / `mixlinmem` / `mixlinmx` / `mixlinxor` × `popg_lif` / `popg_flif` × seed 3. 10-23에 뉴런 축 규칙의 예외로 뺀 부분을 보충한다.
  - 합치기 뉴런은 τ 그룹 spikingjelly LIF(`GroupNeuron('sj')`, fc1·fc2와 같은 방식)다.
  - 층별 BN 이득은 같은 합치기의 단일 spikeDE 뉴런 모델에 맞춘 표 8개(`calibrate_groups.py --combs lin linmem linmx linxor` → `calib/tau_gains_neoMixer_popg_{lif,flif}_{lin,linmem,linmx,linxor}.json`)다. 모든 층이 0.99–1.01배로 시작한다(c39).
  - suite `flif-v1-rhythm-popg-cinc`.
- 판정: G0 OR → 선형 합, G1 막전위 − 스파이크, G2 + 차이 항, G3 ③ − 스파이크, G4 + XOR, G5 soma 없음 − soma 있음(같은 합치기), G6 − Spikformer식. `summarize_rhythm.py --popg`.
- **예상:** G5 모두 나쁨(이전 P1과 같음), G2·G4 차이 없음, G6 미충족.

### 코드 (기본 경로 비트 동일, c1–c32 통과)
- `neo_mixer.py`: `linxor`, popg 선형 합 4종(합치기 τ 그룹 LIF).
- `layers.py` / `stream_mixer.py`: 충전값 경로가 모듈 호출을 거치도록 바꿨다. 계산은 같고 hook이 정상적으로 불린다. 보정 스크립트가 층을 찾으려면 필요하다. `StreamNeurons`·`_Stack` 충전값 경로를 추가했다.
- `model.py`: `stream_rev`, `neo_stream_fuse mx`(양방향 스트림 + 스트림별 합치기).
- `config.py` / `load_model.py`: 선택지와 디렉터리 접미사(`_linxor`, `_srev`, `_sfmx`).
- `calibrate_groups.py`: `--combs`, `fuse` 층, 기존 표를 덮어쓰지 않음. `calibrate_nsmt_streams.py --bimx`.
- `check_cls.py`: c38–c41. `summarize_rhythm.py`: `--xor`, `--popg`, `--t4k2`, `--folds`. 러너 `scripts/run_rhythm5.sh`(run_rhythm4.sh 복사본 + 새 조건, FOLD 사용).

### 점검·smoke
- **점검 41/41** (`results/checks/check_cls_b3.txt`). c33–c37은 충전값 경로를 바꾼 뒤에도 그대로 통과했다.
- **smoke** (`results/smoke-rhythm-b3-20261024/raw.txt`, 1 epoch, seed 42, 9/9 끝까지):

| 조건 | params | 에너지 µJ | 시간 s |
|---|---:|---:|---:|
| mixlinxor_lif | 1,359,360 | 27.7 | 124 |
| mixlinxor_pop_flif | 1,365,504 | 67.3 | 175 |
| mixlin_popg_lif | 1,211,904 | 30.6 | 373 |
| mixlinmx_popg_flif | 1,359,360 | 269.4 | 525 |
| mixlinxor_popg_lif | 1,359,360 | 33.8 | 247 |
| nsmt_popk_lif_t4_srev | 1,271,368 | 136.7 | 209 |
| nsmt_popk_flif_t4_bimx | 2,045,512 | 1,156.8 | 867 |
| spk_ssa (fold 1) | 918,088 | 108.3 | 33 |
| mixlinmx_lif (fold 1) | 1,359,360 | 263.3 | 98 |

### 학습
- 대기열 셋, 모두 `run_rhythm5.sh`:
  - (1)+(2) 48 run: `scripts/queues/rhythm-b3-xorfolds-20261024.txt`, GPU당 2개.
  - (3) 12 run: `rhythm-b3-streams-20261024.txt`, GPU당 1개, 느린 f-LIF 양방향을 먼저.
  - (4) 24 run: `rhythm-b3-popg-20261024.txt`, GPU당 1개. (1)+(2)가 끝나면 같은 대기열에 worker를 더 붙인다.
- 명령은 각 suite의 `run_commands.txt`.

**commit:** 이 항목·코드·점검·보정 표·smoke를 commit 하고 `exp/nsmt-flif-neo`만 push.

---

## 2026-10-25 04:22 KST — 후속 (1)·(2) 결과: **③ LIF는 fold 0–4 전체에서 Spikformer식과 차이 없음(15쌍 평균 −0.21%p, 95% 구간 −1.11 ~ +0.70)**, 스파이크 선형 합보다는 15/15쌍 좋음(+3.33); 스파이크 XOR는 ③의 이득을 LIF에서 1/4만 재현 — 이득은 대부분 막전위 차이의 크기에서 옴; 러너의 fold 디렉터리 탐색 버그(결과 줄 11개 누락)와 복구

**직전 commit:** `3deef0677` (네 후속 사전 고정).

**학습:** (1)+(2) 대기열 48 run이 2026-10-24 23:27 – 10-25 04:18에 끝났다(xor 마지막 03:05:39, folds 마지막 04:18:30). 명령은 각 suite의 `run_commands.txt`, GPU·시각은 `pool_runs.txt`.

### 러너 버그: fold 실행에서 결과 줄 11개 누락 (내 실수, 복구함)
- **증상:** folds suite `raw.txt`에 SKIP(rc=0, no result) 11줄이 찍혔다. 종료 코드 0인데 결과가 없다는 뜻이다.
- **원인:** `run_rhythm5.sh`는 결과 디렉터리를 `ls -td …/*+<cond>+lr<lr>+seed<s>+*/seed<s>_* | head -1`, 즉 같은 조건·seed에서 가장 최근에 바뀐 디렉터리로 찾는다. 이 패턴에 fold가 없다. 같은 조건·seed의 다른 fold가 동시에 돌면, 아직 결과가 없는 다른 fold의 디렉터리를 집어 SKIP을 썼다. fold 0만 쓰던 이전 실험에서는 드러나지 않았다.
- **확인:**
  - 11개 run의 자기 디렉터리(안쪽 이름 `…_f<fold>`)에는 모두 `final+result.csv`가 있다. 학습은 정상 종료했다.
  - RESULT로 찍힌 25줄은 모두 `dir=`의 fold 접미사가 줄의 fold와 같다. 다른 fold의 값을 잘못 적은 줄은 없다.
- **복구:** `recover_fold_results.py --suite flif-v1-rhythm-folds-cinc`.
  - SKIP 줄마다 자기 fold 디렉터리를 정확히 하나 찾는다. 저장된 config의 fold·seed·data를 대조하고, 러너와 같은 형식의 RESULT 줄을 `raw_recovered.txt`에 쓴다(새 파일, `recovered=1`, time_s는 복구할 수 없어 na).
  - 원래 `raw.txt`는 바꾸지 않았다. `summarize_rhythm.py --folds`는 두 파일을 함께 읽고 (cond, fold, seed) 중복을 거부한다.
- **영향 범위:** 지금 도는 (3)·(4) pool은 fold 0만 쓰고, 같은 조건·seed가 동시에 돌지 않는다. `run_rhythm5.sh`는 실행 중이라 고치지 않았다(실행 중 스크립트 편집 금지). 다음에 fold를 섞어 돌릴 때는 fold를 패턴에 넣은 새 러너를 쓴다.

### (2) ③ LIF, CinC fold 0–4 (`results/flif-v1-rhythm-folds-cinc/summary.md`)

| 모델 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 15쌍 평균 | macro-F1 | µJ |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| Spikformer식 | 65.60 | 66.30 | 65.89 | 64.44 | 66.46 | 65.74 ± 1.10 | 61.39 | 126.8 |
| 스파이크 선형 합, LIF | 61.37 | 63.61 | 62.26 | 61.50 | 62.28 | 62.20 ± 1.21 | 57.41 | 31.6 |
| **③ 막전위 + \|H_f − H_r\|, LIF** | 65.61 | 67.32 | 64.54 | 64.61 | 65.59 | 65.53 ± 1.57 | 60.03 | 267.4 |

| 비교 | fold별 (seed 3 평균, 판정) | 15쌍 평균 [95% 구간] | A > B |
|---|---|---|---|
| F1 ③ − Spikformer식 | f0 +0.02 차이 없음; f1 +1.02 좋음; f2 −1.36 나쁨; f3 +0.17 차이 없음; f4 −0.87 나쁨 | **−0.21 [−1.11, +0.70]** | 6/15 |
| F2 ③ − 스파이크 선형 합 | 다섯 fold 모두 좋음(+2.28 ~ +4.24) | **+3.33 [+2.63, +4.03]** | 15/15 |
| F3 스파이크 선형 합 − Spikformer식 | 다섯 fold 모두 나쁨 | −3.54 [−4.29, −2.78] | 0/15 |

- 사전 예상("F1 −1 ~ 0 차이 없음, F2 +3 ~ +4 좋음")은 둘 다 맞았다.
- ③ LIF는 다섯 fold를 합쳐도 Spikformer식과 점수 차이가 없다. 다만 macro-F1은 1.4점 낮고(Noisy·Other가 낮음), 에너지는 2.1배라 기준 (2)는 미충족이다.

### (1) 스파이크 XOR 대조 (`results/flif-v1-rhythm-xor-cinc/summary.md`)

| 뉴런 | 스파이크 선형 합 | **스파이크 + F XOR R** | ③ 막전위 + 차이 | X1 (XOR − 선형 합) | X2 (XOR − ③) | µJ (XOR / ③) |
|---|---:|---:|---:|---|---|---|
| LIF | 61.37 | **62.43** | 65.61 | **+1.06 좋음** | **−3.18 나쁨** | 33 / 267 |
| f-LIF | 60.58 | 60.20 | 62.18 | −0.38 차이 없음 | **−1.98 나쁨** | 38 / 272 |
| population LIF | 62.35 | 61.44 | 60.71 | −0.91 나쁨 | +0.73 좋음 | 42 / 275 |
| population f-LIF | 62.13 | 61.15 | 61.14 | −0.98 나쁨 | +0.01 차이 없음 | 71 / 304 |

- X3(XOR − OR)은 LIF +1.89 좋음, f-LIF·population f-LIF 차이 없음, population LIF −1.63 나쁨이다.
- X4(vs Spikformer식): 넷 모두 나쁨(−3.2 ~ −5.4), 미충족이다.
- **사전 예상과 비교:** X1 LIF 좋음(+1 ~ +3)은 맞았다. X2 나쁨(−1 ~ −2)은 LIF에서 예상보다 더 나쁘다(−3.18). X4 맞았다.
- **읽는 법:** 두 방향이 "다르다/같다"는 0/1 정보만으로는 ③의 이득(LIF +4.24) 중 약 1/4(+1.06)만 나온다. 나머지는 막전위 차이의 **크기**(얼마나 다르게 충전됐는가)에서 온다. 그래서 ③의 정확도를 AC 비용으로 얻는 길은 이 대조로는 열리지 않았다.

**commit:** 이 항목·두 suite 결과(`raw.txt`, `raw_recovered.txt`, `summary.md`, `pool_runs.txt`)·복구 스크립트·요약 스크립트 수정을 commit 하고 `exp/nsmt-flif-neo`만 push. (3)·(4)는 계속 돈다(04:21에 popg 대기열에 worker 4개를 더 붙임).

---

## 2026-10-25 11:55 KST — 후속 (3) 결과: 스트림 정렬 대조 (CinC 12 run) — **τ 순서를 뒤집으면 LIF가 나빠짐(S1 −0.97)**, 양방향 합치기는 LIF에서 나빠짐(S2 −0.72, 에너지 7.6배); f-LIF는 둘 다 차이 없음; 모두 Spikformer식에 못 미침

**직전 commit:** `1e8451931` (후속 1·2 결과).

**학습:** 12 run이 2026-10-24 23:27 – 10-25 11:53:24에 끝났다(SKIP 0). run당 시간은 LIF 순서 대조 4,301–5,720 s, LIF 양방향 9,914–13,402 s, f-LIF 순서 대조 13,199–13,709 s, f-LIF 양방향 26,771–27,344 s다. 명령은 `results/flif-v1-rhythm-t4k2-cinc/run_commands.txt`, GPU·시각은 `pool_runs.txt`. fold 0만 써서 러너 fold 버그와는 무관하다(같은 조건·seed가 동시에 돌지 않았다).

**요약:** `summarize_rhythm.py --t4k2 --suite flif-v1-rhythm-t4k2-cinc` → `results/flif-v1-rhythm-t4k2-cinc/summary.md`.

| 모델 (해마 T_h 4) | CinC 점수 | macro-F1 | 에너지 µJ |
|---|---:|---:|---:|
| Spikformer식 | 65.60 ± 1.48 | 61.94 | 127.4 |
| LIF 하나 복사 | 63.42 ± 0.85 | 55.80 | 148.1 |
| LIF 스트림, τ 4 → 32 (10-24) | **63.52 ± 0.38** | **58.99** | 149.1 |
| LIF 스트림, τ 32 → 4 (순서 대조) | 62.55 ± 1.60 | 56.48 | 150.0 |
| LIF 스트림, 양방향 합치기 | 62.80 ± 0.70 | 57.78 | 1,136.7 |
| f-LIF 하나 복사 | 61.84 ± 2.42 | 55.24 | 148.2 |
| f-LIF 스트림, τ 4 → 32 (10-24) | 58.33 ± 0.96 | 53.00 | 162.3 |
| f-LIF 스트림, τ 32 → 4 | 58.85 ± 2.28 | 53.89 | 160.8 |
| f-LIF 스트림, 양방향 합치기 | 58.22 ± 1.52 | 52.26 | 1,162.3 |

| 판정 | LIF | f-LIF |
|---|---|---|
| S1 순서 뒤집기 − 원래 | **−0.97 나쁨** (+0.97 / −2.93 / −0.95) | +0.52 차이 없음 (−1.39 / −0.27 / +3.23) |
| S2 양방향 합치기 − 원래 | **−0.72 나쁨** (−0.02 / −0.46 / −1.69), 에너지 7.62배 | −0.11 차이 없음, 7.16배 |
| S3 − 하나 복사 | 순서 −0.87 나쁨, 양방향 −0.62 나쁨 | 순서 −2.99 나쁨, 양방향 −3.63 나쁨 |
| S4 − Spikformer식 (기준 (2)) | −3.05 / −2.80 나쁨, 미충족 | −6.75 / −7.38 나쁨, 미충족 |

**사전 예상과 비교:**
- S1 "차이 없음": LIF에서 틀렸다(나쁨). f-LIF는 맞았다.
- S2 "LIF 좋음(+1 ~ +2), f-LIF 차이 없음": LIF에서 틀렸다(나쁨). f-LIF는 맞았다.
- S4 "미충족": 맞았다.

**읽는 법 (탐색, seed 3개):**
- LIF에서는 τ가 짧은 스트림부터 해마 스텝에 넣는 원래 순서(빠른 기억 → 느린 기억)가 반대 순서보다 낫다. macro-F1도 58.99 vs 56.48이다. 해마 LIF는 스텝 사이에 상태를 이어 가므로 순서가 결과를 바꾼다. 다만 seed 하나(+0.97)는 반대 방향이라 차이는 작다.
- Neo-mixer에서 크게 도왔던 막전위 차이 합치기(③)가 이 구조에서는 돕지 않는다. NSMT에서는 해마의 교차 주의가 이미 117 토큰 전체를 본다. 그래서 두 방향을 합친 신피질이 더할 정보가 적은 것으로 읽힌다(가설). Neo-mixer에는 이런 전역 접근이 없다.
- f-LIF 스트림은 어떤 배치에서도 단일 f-LIF 복사보다 3%p가량 나쁘다.

**commit:** 이 항목·결과를 commit 하고 `exp/nsmt-flif-neo`만 push. (4) popg는 24 중 20 끝, 계속 돈다.

---

## 2026-10-25 13:25 KST — 후속 (4) 결과: soma 없는 τ 그룹(popg) Neo-mixer의 선형 합 4종 (CinC 24 run) — **모든 선형 합이 OR보다 나쁨(G0 −2.08 / −2.11)**, soma 있는 population보다 2–5%p 나쁨(G5 8개 모두); 그룹 안에서는 차이 항이 LIF에서 조금 도움(G2 +1.01, G3 +1.43, G4 +0.86); 네 후속 완료

**직전 commit:** `a89298d63` (후속 3 결과).

**학습:** 24 run이 2026-10-24 23:27 – 10-25 13:23:40에 끝났다(SKIP 0, run당 시간은 아래 범위). 처음엔 GPU당 1개였고, 10-25 04:21에 같은 대기열에 worker 4개를 더 붙였다(`pool_runs.txt`, `pool_runs_2.txt`). fold 0만 써서 러너 fold 버그와는 무관하다.

**요약:** `summarize_rhythm.py --popg --suite flif-v1-rhythm-popg-cinc` → `results/flif-v1-rhythm-popg-cinc/summary.md`.

| 합치기 | popg LIF | popg f-LIF | soma 있는 population LIF | soma 있는 population f-LIF |
|---|---:|---:|---:|---:|
| OR | 59.27 | 59.51 | 63.06 | 60.73 |
| 스파이크 선형 합 | 57.19 | 57.41 | 62.35 | 62.13 |
| ② 막전위 | 57.61 | 56.67 | 61.92 | 60.85 |
| ③ 막전위 + 차이 | **58.62** | 57.31 | 60.71 | 61.14 |
| 스파이크 + XOR | 58.06 | **57.73** | 61.44 | 61.15 |

(CinC 점수 %, seed 3 평균. Spikformer식 65.60.)

| 판정 | popg LIF | popg f-LIF |
|---|---|---|
| G0 스파이크 선형 합 − OR | **−2.08 나쁨** | **−2.11 나쁨** |
| G1 ② − 스파이크 선형 합 | +0.41 차이 없음 | −0.74 나쁨 |
| G2 ③ − ② | **+1.01 좋음** | +0.64 차이 없음 |
| G3 ③ − 스파이크 선형 합 | **+1.43 좋음** | −0.10 차이 없음 |
| G4 XOR − 스파이크 선형 합 | **+0.86 좋음** (3/3) | +0.33 차이 없음 |
| G5 soma 없음 − soma 있음 (같은 합치기) | −5.16 / −4.31 / −2.08 / −3.38 나쁨 | −4.72 / −4.18 / −3.83 / −3.41 나쁨 |
| G6 − Spikformer식 | −7.0 ~ −8.4 나쁨, 미충족 | −7.9 ~ −8.9 나쁨, 미충족 |

- G5의 에너지 비율 칸이 nan인 것은 요약 스크립트가 표에 없는 soma 조건의 에너지를 모으지 않았기 때문이다. 판정과는 무관하다. 에너지는 각 suite의 `summary.md`에 있다.
- **사전 예상과 비교:** G5 "모두 나쁨"과 G6 "미충족"은 맞았다. "G2·G4 차이 없음"은 popg LIF에서 틀렸다(둘 다 좋음, 다만 작음).

**읽는 법 (탐색):** soma 없는 τ 그룹은 합치기 방식과 무관하게 soma 있는 population보다 나쁘다. 이전 P1(OR·BiX)과 같은 결론이 선형 합 4종에서도 나온다. 차이 항(막전위든 XOR든)은 이 형태에서도 LIF에서만 작게 돕는다.

### 네 후속 종합
1. **스파이크 XOR:** ③의 이득을 LIF에서 약 1/4만 재현했다. 이득의 대부분은 막전위 차이의 크기에서 온다.
2. **fold 0–4:** ③ LIF는 Spikformer식과 점수 차이가 없다(15쌍 −0.21, [−1.11, +0.70]). 스파이크 선형 합보다는 15/15 좋다. macro-F1은 1.4점 낮고 에너지는 2.1배다.
3. **스트림 정렬:** τ 4 → 32 순서가 반대 순서보다 낫다(LIF). 양방향 합치기는 돕지 않는다.
4. **popg:** 모든 선형 합이 soma 형태보다 나쁘다.

**태그:** 이 기록을 담은 commit 에 `exp/nsmt-flif-neo-20261025`(annotated, 완료: 후속 84 run).

**commit:** 이 항목·결과를 commit 하고 `exp/nsmt-flif-neo`와 태그만 push.

---

## 2026-10-25 13:25 KST — 정정: 앞 항목의 run 시간

앞 항목 "학습" 줄의 "run당 시간은 아래 범위"는 범위를 적지 않은 채 남았다. 실제 범위는 **6,396 – 18,201 s**(24 run의 `time_s`)다.

---

## 2026-10-26 17:44 KST — 분석(학습·새 평가 없음, 보고만): ③ LIF의 macro-F1 열세는 어디서 오는가 — **Noisy(−4.6)는 대부분 결정 문턱 문제(순위 AUC는 같음, 짧은 기록에 몰림), Other(−2.3)는 순위 자체가 약간 나쁨(AUC −0.012, Other → N 혼동 증가, 긴 기록에서 이득이 없음)**; AF는 오히려 나음(+2.2)

**직전 commit:** `097a0357a` (popg 기록 정정). 사용자 질문(10-26): "2번을 분석하면서, 1번에 대해서 설명해줘." 2번 = ③ LIF macro-F1 열세의 원인 분석.

**방법:** `analyze_class_gap.py` → `results/class-gap-20261026/`(`class_gap.json`, `class_gap.txt`).
- 대상: Spikformer식(`spk_ssa`), ③ LIF(`mixlinmx_lif`), 스파이크 선형 합 LIF(`mixlin_lif`) × CinC fold 0–4 × seed 3 = 각 15 run.
- 각 run이 이미 저장한 test 출력만 읽는다: 최고 검증 가중치의 `confusion_matrix.csv`와 기록별 확률 `test_probs.npz`. 새 학습·새 평가는 없다.
- 점검: (cond, fold, seed)마다 run이 정확히 하나. 저장된 혼동 행렬 = 저장된 확률의 argmax. 저장된 정답 = 데이터 캐시의 정답. 두 모델의 test 기록이 fold마다 같다. 모두 통과.
- 클래스별 recall·precision·F1, 예측 비율, 한 클래스 대 나머지 AUC(순위만 보는 값, 결정 문턱과 무관)를 낸다. 클래스 문턱을 test에서 최적으로 옮겼을 때의 F1(상한, 사용하지 않음), 기록 길이별 recall, 두 모델이 갈리는 기록 수도 낸다.

| 클래스 (15 run 합산) | Spikformer식 F1 | ③ LIF F1 | 차이 | AUC (Spikformer / ③) | 문턱 최적 F1 (Spikformer / ③) | recall / precision (Spikformer → ③) |
|---|---:|---:|---:|---|---|---|
| N | 82.9 | 82.6 | −0.3 | 0.854 / 0.847 | 83.7 / 83.5 | 87.0 / 79.2 → 87.6 / 78.2 |
| AF | 58.4 | **60.6** | **+2.2** | 0.908 / **0.925** | 60.5 / 63.5 | 57.7 / 59.1 → 54.4 / **68.4** |
| Other | 55.8 | 53.5 | **−2.3** | 0.783 / **0.771** | 59.5 / 57.8 | 50.0 / 63.2 → 48.3 / 60.0 |
| Noisy | 48.4 | 43.8 | **−4.6** | 0.946 / 0.948 | 51.1 / 49.6 | 49.9 / 46.9 → 44.9 / 42.7 |

- **Noisy (−4.6):** 순위 AUC는 같다(0.948 vs 0.946). 문턱만 최적으로 옮기면 차이가 −1.5로 준다. 즉 대부분 "어디서 Noisy라고 판정하느냐"의 문제다.
  - 짧은 기록(< 30 s, 데이터 로더가 반복해 30 s를 채우는 기록)에서 recall 42.4 vs 49.0(−6.6)으로 가장 크게 진다. 30–45 s 기록에서는 −2.7이다.
  - Noisy를 AF로 잘못 보는 비율이 6.1% vs 3.8%로 늘었다.
- **Other (−2.3):** 순위 AUC도 낮다(0.771 vs 0.783). 문턱 최적 F1도 −1.7이라, 표현 자체가 약간 약하다.
  - Other를 N으로 잘못 보는 비율이 43.6% vs 39.0%다.
  - Spikformer식은 긴 기록(> 45 s, 평가에서 창 여러 개를 평균)에서 Other recall이 46.0 → 53.3으로 오른다. ③은 48.4 → 48.8로 거의 그대로다.
- **AF (+2.2):** ③이 낫다. AUC가 높고(0.925 vs 0.908), precision이 크게 높다(68.4 vs 59.1). AF를 적게 예측하는 쪽이다(예측 비율 7.1% vs 실제 8.9%). AF를 Other로 잘못 보는 비율은 29.9% vs 23.1%로 높다.
- 두 모델이 갈리는 기록(15 run 합산): Other는 Spikformer만 맞힘 1,028 vs ③만 맞힘 899, Noisy는 142 vs 100, AF는 269 vs 194, N은 985 vs 1,083이다.
- 클래스별 차이의 평균(−0.3 + 2.2 − 2.3 − 4.6) / 4 = −1.25는 macro-F1 차이(−1.36)와 맞는다.

**읽는 법:** macro-F1 열세의 2/3가 Noisy에서 온다. 그 대부분은 순위가 아니라 판정 문턱 쪽이다. validation에서 클래스별 판정 편향을 정하면 줄일 수 있는 부분일 수 있다(가설; 문턱 최적값은 test에서 잰 상한이라 그대로 쓸 수 없다). Other는 순위 자체가 약하다. 긴 기록에서 창을 더 봐도 나아지지 않는 점은, ③의 표현이 창마다 비슷한 답을 내 평균으로 얻는 것이 적다는 뜻으로 읽힌다(가설).

**한계:** test 출력을 다시 읽은 진단이다. 문턱 최적 F1은 test에서 고른 상한이다. 길이 구간별 Noisy는 기록 수가 작다(긴 기록 10개).

**commit:** 이 항목·분석 스크립트·결과를 commit 하고 `exp/nsmt-flif-neo`만 push.
