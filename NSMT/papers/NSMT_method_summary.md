# NSMT 정리 — 연구 동기·가설과 방법 (Method)

원문: *Fast Spikes, Slow Trends: Neuro-Inspired Spiking Memory Transformer for Time-Series Analysis* (`papers/NSMT.pdf`, 익명 투고본, Submission Id 3724, 18쪽).
이 문서는 논문의 **동기·가설·연구 목적**과 **방법(Method)** 만 정리한다. 실험 설정·결과·분석(§4–§5, 부록 B–E)은 제외했다. 절 번호와 식 번호는 원문 것이다.

---

## 1. 연구 배경과 문제의식 (§1)

### 1.1 시계열 모델의 "슬라이딩 윈도우" 한계
- 시계열은 연속 신호로 기록되지만, 학습은 계산 비용과 메모리 때문에 **고정 길이 look-back window** 단위로 이루어진다.
- 흔한 구현은 윈도우를 잘라 mini-batch로 **섞어서** 학습하므로, 연속 구간 사이의 시간적 연속성이 끊긴다.
- 그 결과 모델은 **현재 윈도우 안의 정보만** 보게 되고, 윈도우 밖의 장기 의존성(global temporal dependency)을 명시적으로 다루지 못한다.
- 장기 예측(long-horizon forecasting)에서 특히 취약하다. 윈도우를 늘리면 해결되지만 Transformer는 길이에 대해 제곱 비용이 든다.

### 1.2 어떤 정보를 저장해야 하는가 — 저주파 성분
- 실제 시계열의 스펙트럼 에너지는 **저주파 성분이 지배**한다 (Fig. 1, ETTh2 예시).
- 저주파 성분은 느리게 변하는 **추세(slow trend)** 를 담으며, 장기 예측에 필요한 안정적 문맥(global context)이다.
- 따라서 과거 전체를 그대로 기억하는 대신, **현재 look-back window의 지배적 저주파 성분**을 선택적으로 저장하는 메모리가 효율적이다.

### 1.3 왜 SNN인가
- SNN은 이벤트 기반(spike) 연산으로 에너지 효율이 높고 뉴로모픽 하드웨어와 호환된다.
- 뉴런 동역학 자체가 시간 정보를 다루며, 스파이킹 뉴런은 사실상 **저역 통과 필터(low-pass filter)** 처럼 동작한다 → "느린 추세는 보존하고 고주파 잡음은 억제"하려는 목적과 맞는다.
- 기존 SNN 시계열 모델(iSpikformer, SpikF 등)의 한계:
  - 입력의 실제 시간 진행과 **분리된 별도의 spiking time axis**를 만들어 중복 계산이 생긴다.
  - ANN에서 가져온 무거운 연산(예: SpikF의 FFT, 마지막 층 GeLU)을 그대로 두어 스파이킹의 에너지 이점이 줄어든다.
  - 과제별로 설계되어 **범용(task-general)** 스파이킹 시계열 모델이 없다.

### 1.4 생물학적 영감 — Complementary Learning Systems (CLS) 이론 (§3.1)
- 뇌는 두 시스템으로 학습한다: **해마(Hippocampus)** 는 최근의 구체적 사건을 빠르게 부호화하고, **신피질(Neocortex)** 은 긴 시간에 걸쳐 느리게 구조화된 지식을 쌓는다.
- 두 시스템은 상호작용한다: 새 경험이 들어오면 해마는 신피질의 지식을 참조해 해석하고(간섭 감소), 해마가 잡은 정보는 시간이 지나며 신피질로 통합된다(memory replay/consolidation).
- 이를 시계열에 옮기면: **느린 추세(global trend)는 신피질 경로가 응축해 저장하고, 현재 윈도우의 빠른 국소 패턴은 해마 경로가 처리하며, 둘을 상호작용으로 결합**한다.

## 2. 연구 목적과 가설

**목적.** 에너지 효율적인 스파이킹 Transformer 안에 "저주파 추세 메모리"를 두어, 슬라이딩 윈도우 학습 아래에서도 장기 의존성을 모델링하고, 예측·이상 탐지·분류에 하나의 설계로 대응하는 **범용 SNN 시계열 모델**을 만든다.

**가설 (논문이 설계로 전제하는 명제).**
1. 현재 look-back window의 **저주파 성분을 재구성하도록 보조 감독**하면, 신피질 경로가 안정적인 추세 메모리를 학습한다 (저주파 보조 손실).
2. 그 추세 메모리를 **cross-attention(memory replay)** 으로 해마 경로에 되돌려 주면, 윈도우 단위 학습으로 끊긴 전역 문맥이 복원되어 장기 예측이 좋아진다.
3. 별도의 spiking time axis를 만들지 않고 **토큰(패치) 순서 축을 그대로 스파이킹 시간 축으로 쓰면(T = N)**, 뉴런 동역학이 위치·순서 정보를 스스로 부호화하므로 positional encoding과 중복 계산 없이 효율이 오른다.
4. 잔차 연결을 덧셈(ADD) 대신 **IAND** 논리 연산으로 두면 스파이크 포화를 막고 이론적 에너지를 크게 줄이면서 정확도를 유지한다.

**기여 요약 (§1 끝).**
- 패치 기반 스파이킹 Transformer + IAND 잔차 + 토큰 정렬 스파이킹 동역학 + 저주파 재구성 보조 목적.
- CLS에서 영감을 받은 **이중 경로(dual-path)**: 신피질 메모리(저주파 저장) ↔ 해마(현재 윈도우 처리), 가벼운 attention으로 검색.
- 예측·이상 탐지·분류에 공통으로 쓰이는 task-general 설계.

---

## 3. 방법 (§3 Methodology)

### 3.1 기본 뉴런: LIF (§2.2)
스파이킹 뉴런은 Leaky Integrate-and-Fire(LIF)를 쓴다.

```
H[t] = V[t-1] + (1/τ) · ( X[t] − (V[t-1] − V_reset) )
S[t] = Θ(H[t] − V_th)
V[t] = H[t]·(1 − S[t]) + V_reset·S[t]
```
- τ: 막 시간상수, X[t]: 입력 전류, H[t]: 발화 전 막전위, V_th: 임계값, Θ: Heaviside(0/1 스파이크), 발화하면 V_reset으로 hard reset.
- 역전파는 surrogate gradient를 쓴다 (부록 B.2: sigmoid σ(x) = 1/(1+e^{−x})). 임계값 V_th = 1.

### 3.2 표기 (§3.2)
- 입력 look-back window X_in ∈ R^{L×C} (길이 L, 변수 C).
- 입력을 N개의 **패치**로 자르고 direct spiking encoding을 적용해 스파이크 형태의 패치열 X ∈ R^{T×N×P} 를 얻는다 (P: 패치 차원, T: 스파이킹 시간 축, N: 시퀀스 위치 축).
- 채널 독립(channel-independent) 패치 토큰화(PatchTST 방식)를 따른다. 변수 C는 배치 축으로 펼쳐 각 변수를 독립 시계열로 다룬다.
- **T와 N을 명시적으로 구분**한다: T는 뉴런 동역학이 진행되는 축, N은 시계열의 자연스러운 순서 축. 변환된 X는 두 경로의 공통 입력이다.

### 3.3 신피질 경로 (Neocortex Network): 느린 추세 저장 (§3.3)
목적: 현재 윈도우에서 **느리게 변하는 전역 추세**를 가중치 안에 평균적인 추세 표현으로 저장하고, 해마 경로에 안정적 전역 참조를 제공한다.

**(a) Spiking temporal alignment (T = N).**
- 원래의 스파이킹 시간 축 T를 따라 **평균 풀링**해 중복된 반복을 없앤 뒤, **시퀀스 위치 축 N을 스파이킹 시간 축으로** 삼아 뉴런이 패치의 진행을 직접 처리하게 한다 (각주 1: T = N으로 설정).
- 정렬된 입력은 Spiking MLP로 투영된다:
  ```
  X^n = SpikingMLP(X) = SN( BN( Linear(X) ) )        (1)-(2)
  ```
  SN: 스파이킹 뉴런, BN: batch normalization. 이 모듈을 **Spiking Embedding Layer** 라 부른다.
- 시퀀스 순서 N이 곧 SNN의 시간 축이므로 **위치 정보가 뉴런 발화 동역학에 자연히 부호화**되어 별도의 positional encoding이 필요 없다 (Fig. 3: 해마는 표준 패치열 임베딩 X^h, 신피질은 T=N으로 정렬된 X^n을 받는다).
- 신피질망은 hidden size D의 **Spiking MLP 층 M개**를 쌓은 것이다. 층을 쌓을수록 정렬된 토큰 축을 따라 통합 깊이가 늘어 간결한 장기 요약이 만들어진다.

**(b) 저주파 보조 목적 (Low-frequency auxiliary objective).**
- look-back window X_in에 1-D **DCT**를 적용하고, 가장 낮은 주파수 계수 비율 **k** 만 남긴 뒤(나머지는 0) 역변환(IDCT)해 저주파 목표 X̂_low를 만든다:
  ```
  X̂_low = IDCT( Mask_k( DCT(X_in) ) )                (3)
  ```
  (부록 A, Algorithm 1: X_c ← DCT(X_in); K ← max(1, ⌈T·k⌉); q < K인 계수만 유지; X̂_low ← IDCT. 구현에서는 선택적으로 변환 전에 평균을 빼고 역변환 후 더해 DC 성분이 목표를 지배하지 않게 한다.)
- 신피질 출력 X_neo를 **약한 디코더(단일 fully-connected 층)** 에 통과시켜 이 목표를 재구성하도록 감독한다:
  ```
  L_aux = || Decoder(X_neo) − X̂_low ||²             (4)
  ```
- 디코더는 학습에만 쓰고 추론에서는 제거한다 (Fig. 2의 *Recon.*).

### 3.4 해마 경로 (Hippocampus Network): 효율적 국소 처리 (§3.4)
목적: 현재 윈도우의 구체적 패턴을 추출하고, 신피질에 저장된 전역 추세를 **능동적으로 검색**해 활용한다.

**(a) 임베딩.**
```
X^h = SpikingMLP(X)                                  (5)
```
식 (2)와 같은 Linear–BN–SN 구조. positional encoding은 쓰지 않는다.

**(b) Memory Replay (신피질 → 해마 상호작용).**
- CLS의 "신피질이 재활성화된 지식으로 해마 처리를 조절한다"를 cross-attention으로 설계한다.
- 먼저 X_neo를 X^h와 같은 축 순서로 되돌린다(신피질은 N을 시간 축으로 썼으므로 전치). 그리고 해마 임베딩과 신피질 출력을 **원소별 곱**으로 융합한다:
  ```
  X^f = X_neo^T ⊙ X^h                                (6)
  ```
  이진 스파이크 입력에서 원소별 곱은 **논리 AND** 게이트로 동작해 에너지가 싸다.
- 융합 신호 X^f를 **key/value**, 원래 해마 임베딩 X^h를 **query** 로 하는 spiking cross-attention(Spikformer의 SSA 방식, 각 스파이킹 시간 단계마다 수행):
  ```
  Q = SN(BN(X^h W_Q)),  K = SN(BN(X^f W_K)),  V = SN(BN(X^f W_V)),   W ∈ R^{D×d_k}, d_k = D/h
  A = (Q · K^T) · s                                  (7)   s: 스케일 상수
  MemoryReplay(X^h, X_neo) = Proj( SN(A · V) )       (8)   Proj: Spiking MLP 층
  ```
  softmax가 없으며(스파이크 Q, K의 곱), 역전파를 통해 현재 윈도우에서 과제에 유용한 신호가 신피질 경로로 흘러 메모리가 갱신된다.
- 이어서 잔차 연결로 검색된 문맥을 입력 흐름에 합치고, **Spiking MLP 두 개를 쌓은 feed-forward 블록**을 거친 뒤 두 번째 잔차 연결을 적용해 해마 경로의 최종 출력 X_hippo를 얻는다.

**(c) 에너지 효율 잔차 연결: IAND.**
- 통상적인 SNN 잔차(ADD)는 값을 누적해 다중 비트 산술이 필요하다. 대신 Fang et al.의 **IAND** 연산을 쓴다. 블록 입력 S[t], 블록 출력 A[t]에 대해
  ```
  IAND(A[t], S[t]) = (¬A[t]) ∧ S[t] = (1 − A[t]) · S[t]      (9)-(10)
  ```
- 블록 출력이 비활성(0)인 위치에서만 입력 스파이크가 통과하므로 **스파이크 포화를 막고** 깊은 망에서도 정보 흐름이 유지되며, 비트 논리만으로 계산된다. (Table 1: ADD 대비 이론 에너지가 약 10배 이상 감소, MSE는 오히려 소폭 개선 — 설계 근거로만 인용.)

### 3.5 학습 목적 (§3.5)
최종 표현 X_hippo를 과제별 head에 넣어 과제 손실 L_task를 계산하고, 보조 추세 손실과 가중합한다:
```
L_total = (1 − α) · L_task + α · L_aux,     0 < α < 1        (11)
```
α는 과제 성능과 메모리의 추세 보존 사이의 균형을 정한다.

**과제별 L_task (부록 B.4).**
- 장기 예측: MAE(L1) 를 학습 손실로 사용 (SpikF를 따름; 평가는 MSE/MAE).
- 이상 탐지: 입력과 재구성 사이의 MSE (재구성 기반 비지도 탐지, Anomaly Transformer 프로토콜).
- UEA 분류: label smoothing 0.1의 cross-entropy; MIT-BIH 부정맥 분류: context-aware loss와 결합.

### 3.6 전체 구조 요약 (Fig. 2)
```
입력 윈도우 X_in ─┬─ 패치 → Spiking Embedding (T, N) ──→ X^h ─┐
                  │                                             ├─ [Block] X^f = X_neoᵀ ⊙ X^h
                  ├─ 평균풀링(T) + T=N 정렬 → Spiking Embedding    │           Q←X^h, K,V←X^f
                  │        → Spiking MLP × M ──→ X_neo ─────────┘           x ← x ⊙ (1 − MemoryReplay)
                  │                │                                          x ← x ⊙ (1 − FFN(x))   (IAND)
                  │                └─ (학습 시) Decoder → L_aux vs X̂_low           ↓
                  └─ DCT → 저주파 k 유지 → IDCT → X̂_low                      Head → L_task
```
- 두 경로 모두 전부 스파이킹 층(Linear–BN–LIF)으로 구성되고, 마지막 head만 비스파이킹 선형층이다.
- 설계상 추가 시간 축을 만들지 않으므로(T = N) 중복 반복 계산이 사라지고, IAND 잔차와 패치 토큰화로 연산량이 작다.

---

## 4. 논문이 밝힌 한계 (부록 F, 방법에 관한 것만)
- T = N 가정은 입력 표현이나 시퀀스 구조가 다를 때 유연성을 제한한다.
- 신피질 깊이 M, 보조 손실 가중치 α, 저주파 유지 비율 k 등은 원리적 기준이 아니라 **경험적 튜닝**으로 정한다 (부록 E 민감도 분석에서 α ∈ [0.4, 0.8]에서 안정, k = 0.15 부근이 최적이라 보고 — 수치는 실험이므로 본문에서는 생략).

## 5. 용어 정리
| 용어 | 뜻 |
|---|---|
| look-back window | 모델이 한 번에 보는 과거 구간 (길이 L) |
| 패치(patch) | 윈도우를 잘라 만든 토큰, N개 |
| T / N | 스파이킹 뉴런 동역학이 진행되는 시간 축 / 시계열 순서(토큰) 축 |
| Spiking temporal alignment | 신피질 경로에서 N을 T로 쓰는 것 (T = N) |
| Spiking MLP / SN | Linear → BN → LIF 뉴런 |
| DCT 저주파 목표 (X̂_low) | DCT 계수의 하위 k 비율만 남겨 복원한 느린 추세 신호 |
| Memory Replay | 해마 query, (신피질⊙해마) key/value 의 spiking cross-attention |
| IAND 잔차 | x ⊙ (1 − 블록출력): 블록이 쉬는 자리만 입력을 통과 |
| α | L_task와 L_aux의 혼합 비율 (식 11) |
| k (keep ratio) | DCT에서 남기는 저주파 계수 비율 |
