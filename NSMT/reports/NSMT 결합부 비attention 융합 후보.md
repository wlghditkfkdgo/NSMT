# 두 경로는 spike보다 current에서, 무시하기 어렵게 합쳐라

2026-11-19 · 질문: "현재 cross-attention이 아닌 hippo와 neo의 두 tensor를 합치는 데에 '효과적일 거라고 생각하는' 방법을 'SNN 선행 연구 기반'으로 리스트업 해줘." · 근거: research note 2편(`research_notes/NSMT 결합부 비attention 융합 후보/`의 `fusion_operators.md`, `neuron_level_and_memory.md`), 이전 보고서(`reports/NSMT 결합부 spiking attention 후보.md`), 코드(`nsmt_flif_v1/classification/model.py`)

## 한 줄로

SNN 선행 연구에서 cross-attention 없이 두 tensor를 합친 방법은 24개 항목(논문 33편)으로 정리되고, 가장 유망한 방향은 Neocortex 신호를 Hippocampus neuron의 문턱 앞 current(neuron이 매 step 받는 입력 값)에 더하거나 곱해 넣는 것이다. 가능하면 spike 대신 membrane potential을 넣는다. 다만 어떤 방법도 ETT·MIT-BIH에서 시험된 적이 없고, 한쪽 경로가 무시되는 문제를 연산자만 바꿔 푼 사례도 없으므로, concat + Neocortex 단독 학습 같은 학습 쪽 점검을 함께 해야 한다.

## 우리 문제

- **두 tensor.** Hippocampus 출력은 spike [4, B, N, D]다. patch token마다 같은 입력을 4 step 반복한다. Neocortex 출력은 spike [N, B, 1, D]다. N개 token을 순서대로 읽는다(T = N). 둘은 같은 N개 token을 가리키므로, Neocortex를 4 step에 복사하면 token끼리 바로 맞물린다.
- **지금 결합.** softmax 없는 spiking cross-attention(query = Neocortex, key·value = Hippocampus) 뒤에 IAND gate x·(1 − y)가 붙는다. IAND gate는 y가 1인 곳의 Hippocampus spike를 지우는 장치다. Neocortex가 출력에 닿는 길은 Z → q → q·k → z → y → gate의 긴 사슬 하나다. 그리고 query 형태에서는 이 cross-attention이 Hippocampus의 유일한 token 섞기 통로이기도 하다(코드 확인).
- **Neocortex가 쓰이지 않는다.** 시험 때 Neocortex 출력을 0으로 바꿔도 ETT MSE가 1 % 미만으로 바뀐다. q·k 곱의 약 97 %가 0이다.
- **Neocortex가 거의 조용하다.** 11-08 측정(Neocortex membrane shortcut을 넣기 전)에서 마지막 층 발화율은 2–4 %였다. 첫 token은 한 번도 발화하지 않았다. (행, token) 위치의 70–83 %에서 Neocortex 출력 전체가 0이었다. 지금 base에서는 다시 재지 않았다.
- **attention 안의 식을 바꾸는 방향은 닫혔다.** 11-18에 score를 XNOR + Log-PE로 바꾼 시험도 개선이 없었다.
- **논문 목표.** 분류·forecasting·anomaly detection을 한 SNN model로 한다. contribution은 T = N 시간 처리, DNN보다 훨씬 낮은 전력, (수정한) Spikformer보다 평균적으로 나은 성능을 비교적 낮은 전력으로 내는 것이다. 그러니 새 결합은 Neocortex가 실제로 쓰이게 만들면서 실수 곱셈을 적게 써야 한다.

## 후보 목록

숫자는 모두 note에 적힌 논문 측정값 그대로다. "우리 구조에 넣는 방법"과 "우려"의 적합성 판단은 추론이며, 아직 시험하지 않았다. 대부분의 숫자는 **한 번 run**(같은 설정을 한 번만 학습한 숫자라 우연에 따른 흔들림을 알 수 없음)이다. 예외는 Ceolini(3-fold CV), XOResNet(평균 ± sd), DH-SNN(XOR 과제 10회 반복), Ferrand(5회), Pes(5 seed), Iyer(10회), TS-LIF(seed 3개), Luca(초기화 10개)다. 33편 가운데 ETT나 MIT-BIH를 쓴 논문은 없다.

### 표를 읽는 법

| 용어 | 쉬운 뜻 |
|---|---|
| current | neuron이 매 step 받는 입력 값. 쌓여서 문턱을 넘으면 spike가 나간다 |
| membrane potential | neuron 안에 쌓인 값. spike가 0이어도 0이 아닐 수 있다 |
| 문턱 앞에서 합치기 | 두 경로의 current를 neuron에 넣기 전에 더하는 것. 출력 spike끼리 합치는 것과 대비된다 |
| compartment | neuron 안의 따로 계산되는 부분. soma는 발화하는 몸통, dendrite는 입력을 모으는 가지다. apical dendrite는 위쪽 가지로, 맥락 입력을 받는다 |
| branch | dendrite의 가지 하나. 가지마다 기억 길이(시간상수 τ)를 따로 가질 수 있다 |
| leak, 시간상수 τ | membrane potential이 매 step 줄어드는 정도. τ가 클수록 오래 기억한다 |
| gate | 0–1 사이 값을 곱해 통과량을 조절하는 장치. 보통 sigmoid로 만든다 |
| convex gate | 두 값을 G와 1 − G 비율로 섞는 gate |
| concat + projection | 두 tensor를 channel 방향으로 이어 붙인 뒤 Linear로 원래 크기로 줄이는 것 |
| FiLM | 한쪽에서 만든 배율 γ와 이동 β로 다른 쪽을 γ·x + β로 바꾸는 방법 |
| OR / XOR | 둘 중 하나라도 1이면 1 / 둘이 다를 때만 1 |
| IMP | neuron마다 학습하는 초기 membrane potential |
| AC / MAC | 덧셈만 하는 연산(0.9 pJ) / 곱셈과 덧셈(4.6 pJ). 0/1 spike 입력은 AC로 센다 |
| spike 덧셈만으로 되는가 | 결과를 0/1 spike와 덧셈·논리 연산만으로 계산할 수 있는가. weight나 문턱에 접을 수 있는 상수 배율은 덧셈으로 본다 |
| arXiv only | 심사를 거친 학회·저널 판을 찾지 못한 논문 |
| 지표 | R²(1에 가까울수록 좋음)·RSE(낮을수록 좋음)는 forecasting 지표다. SD는 이미지 fusion 품질(높을수록 좋음), PESQ는 음성 품질, px는 pose 오차, krad는 방사선량이다. McNemar p는 두 분류기 차이를 검정한 값으로, 0.05보다 크면 차이가 없다고 본다 |

### 전체 표

| 방법 | 논문 | 두 tensor를 합치는 방식 | spike 덧셈만으로 되는가 | 보고된 근거 | 우리 구조에 넣는 방법 | 우려 |
|---|---|---|---|---|---|---|
| 1. SEW 원소별 합치기 (ADD·AND·IAND) | Fang 외, 2021, NeurIPS [1] | 출력 spike끼리 A + S, A·S, (1 − A)·S | AND·IAND는 논리 연산만. ADD는 결과가 0/1/2라 다음 층이 MAC가 된다 | DVS Gesture: ADD 97.92, IAND 95.49, AND 70.49 %. 다른 논문의 재실행(21층)에서는 순위가 바뀌었다: DVS Gesture ADD 96.18 / AND 97.22 / IAND 96.88 %. SynA 계산으로 ADD-SEW는 4,462 M MAC·21.3 mJ, AND·IAND-SEW는 840 M MAC·4.8–4.9 mJ | 값싼 대조. Z_i를 4 step에 복사해 x ∨ Z_i(OR) 또는 x + Z_i(ADD)로 합친다. 아무것도 Neocortex를 끌 수 없을 때 Neocortex spike가 쓸모 있는지 본다 | 두 modality가 아니라 residual 연구, 영상, 한 번 run. Z가 70–83 % 위치에서 0이라 AND는 거의 다 지우고 OR는 거의 더하지 않는다. 지금 IAND의 문제를 설명하는 데 가장 쓸모 있다 |
| 2. OR·XOR·OR-ADD 합치기 | SynA-ResNet: Shan 외, 2023, **arXiv only** [2]; XOResNet: Wu 외, 2026, Neurocomputing [3] | OR은 x + y − xy, XOR은 x + y − 2xy(두 쪽이 다른 곳만 1). OA는 모양이 같은 shortcut은 spike OR로, 아닌 shortcut은 neuron 앞에서 current를 더한다 | OR·XOR 예. OA의 current 더하기도 neuron 앞 덧셈이다 | XOResNet FMNIST(평균 ± sd), 11층: spike OR 90.59 ± 0.08 → current 더하기 91.28 ± 0.03 → XOR 추가 92.98 ± 0.04 %. 50층: 93.88 → 94.11 → 94.53 %. SynA: 단순 OR 95.83 %(DVS Gesture)·69.92 %(CIFAR10-DVS), SynA를 붙이면 97.92 / 78.51 % | XOR(Z_i, H_t,i)는 "Neocortex 요약과 지금 Hippocampus token이 다른 channel"을 표시한다. weight가 없다. 핵심 교훈은 "spike보다 current에서 합쳐라"다 | 한 network 안의 residual, 영상만. Z가 대부분 0이면 XOR는 H와 같아진다. SynA의 학습 filter가 shortcut을 스스로 끄는 현상("natural pruning")은 우리 실패와 같은 모양이다. SynA는 심사 전이다 |
| 3. 문턱 앞 gate current 주입 (CIS) | Zhao 외, 2026, **arXiv only** [4] | I = I_main + α ⊙ G ⊙ I_aux, G = sigmoid(BN(Conv(F_aux))). 합친 current가 neuron 하나를 움직인다. 혼자서는 spike를 못 내는 보조 응답도 membrane potential을 움직인다 | 아니오(sigmoid gate와 곱). G ≡ 1이면 current 더하기만 남아 spike 덧셈만으로 된다 | MSRS 이미지 fusion, SD: 더하기 40.89, concat + 1×1 conv 41.73, CIS 43.70. 한 방향만 주입하면 41.05 / 40.70. 학습된 α 평균 0.923·0.532. 발화율 ≈ 14.5 % | x_i ← LIF(BN(W_h x_i) + α ⊙ G_i ⊙ BN(W_n Z_i)). 또는 기존 Hippocampus LIF에 current로 더한다. Z_i 대신 Neocortex membrane potential도 넣을 수 있다. D = 64이면 ≈ 8.3 k parameter(지금 결합 attention은 ≈ 16.4 k + bias 표). Z는 4 step 동안 같아 token마다 한 번만 계산한다 | 예측이 아닌 이미지 품질 지표, 한 번 run, 이득이 작다(지표별 0.04–2.8). gate는 실수 곱이다. Neocortex가 발화한 곳에서만 정보가 간다(membrane potential판으로 우회). α가 0으로 줄 수 있다 |
| 4. Dendrite branch마다 다른 입력 (DH-SNN) | Zheng(Hanle) 외, 2024, Nature Communications [5] | 입력 stream마다 branch를 따로 두고, branch마다 시간상수를 따로 학습한다. soma는 branch current의 합으로 발화한다. soma만 reset되어 branch는 기억을 유지한다 | 예(AC, branch마다 감쇠 1회) | 두 timescale spiking XOR: dendrite 없는 보통 spiking feedforward network는 "much lower than 75 %", branch 1개는 두 timescale을 다루지 못함, branch 2개가 훨씬 나음(값은 그림에만 있음, 10회 반복). 하드웨어(TianjicX)에서 연결 제한 비용 ≤ 0.3 % | (가) pop neuron의 4 member(τ 4/8/16/32)에 `forward_branches`로 넣는다. Neocortex token current는 느린 member, Hippocampus count c_i는 빠른 member로. (나) Hippocampus neuron에 Neocortex용 느린 branch를 둔다. Neocortex branch는 W_n ≈ 4.1 k | 더하기라 한 branch의 weight를 줄이면 무시할 수 있다. XOR의 두 신호는 timescale이 달랐지만 우리 두 신호는 출처가 다르다. Hippocampus 정보가 Neocortex에 들어가면 미래 token이 섞인다 |
| 5. 두 compartment neuron + soma 직접 주입 (TS-LIF) | Feng 외, 2025, ICLR [6] | dendrite와 soma가 β로 연결된다. 입력 current가 soma에도 바로 들어간다. 출력은 κ·s_d + (1 − κ)·s_s. 논문에서는 두 compartment가 같은 입력을 받는다 | 대부분 예. κ 섞기는 실수지만 다음 층 weight에 접을 수 있다 | GRU backbone, TC-LIF 대비: Metr-la H6 R² 0.848 vs 0.828, H96 0.329 vs 0.259, Electricity H96 0.976 vs 0.941(Table 8의 6/6 칸에서 이김, seed 3개). Electricity H96 RSE: TS-former 0.261 vs iSpikformer 0.348. 추가 parameter +0.002–0.006 M | dendrite ← Neocortex token current(느림), soma ← Hippocampus count c_i(직접 주입) | 두 stream으로 쓴 적이 없다. β₂·κ → 0이면 dendrite를 무시한다. pop neuron을 대체해야 한다. ETT가 없다. 직접 주입만 뺀 ablation이 없다 |
| 6. Apical dendrite 곱셈 조절 (cLIF) | Ferrand 외, 2023, ICANN(LNCS 14254) [7] | soma current × ReLU(apical membrane potential). apical 쪽은 맥락 입력을 따로 쌓는다 | 입력은 spike(AC)지만 neuron-step마다 실수 곱 1회 | 5회 평균 ± SD. 맥락 과제 DVS 89.4 ± 1.0(곱) vs 80.4 ± 2.2(더하기) vs 75.9 ± 1.4 %(ALIF, 문턱이 적응하는 LIF), SHD 87.8 ± 0.77 / 72.2 ± 2.0 / 66.1 ± 1.7 %. 순서 기억(우연 50 %): DVS 83.9 ± 2.9(곱) vs 51.1 %(더하기) | B 방향: Hippocampus neuron current × (1 + tanh(W_a u_i)), u_i는 Neocortex membrane potential이다. Neocortex가 조용하면 그대로 통과한다. A 방향: Neocortex soma를 출력 경로로 두고 apical에 Hippocampus count를 넣는다. 우회할 수 없는 유일한 배치다 | 논문 과제는 맥락 없이는 답할 수 없다(곱에 가장 유리한 조건). ETT는 Hippocampus만으로 base와 같다. 200 neuron 1층, 문턱 0.05. ReLU 형태 + 조용한 Neocortex면 Hippocampus도 꺼진다. LIF 갱신 비용 약 +42 % |
| 7. 맥락 민감 two-point neuron (CS-TPN) | Raza & Adeel, 2024, **arXiv only** [8] | I_s + I_c·(0.1 + \|I_s\|) | 아니오(실수 곱) | spiking XOR를 같은 875 neuron으로 더 빨리 학습. 200 population에서 cost 1.0(epoch 120부터) vs 1.5(epoch 250). 비spiking 음성 향상 PESQ 1.58 vs 1.40(0 dB) | I_s = Hippocampus current, I_c = W_a u_i. 맥락이 0이면 그대로다 | 한 연구 그룹, 대부분 비spiking 결과, 시계열 없음. "up to 330x fewer neurons"는 표가 뒷받침하지 않는다 |
| 8. membrane potential에 배율 곱 (MA-SNN) | Yao 외, 2023, IEEE TPAMI(숫자는 arXiv 판) [9] | 작은 FC·conv + sigmoid가 만든 배율을 membrane potential에 곱한다. query–key 방식이 아니다. 원래는 같은 stream 안에서 쓴다 | 아니오 | DVS128 Gesture 90.63 → 96.53 %, spike −84.9 %. ImageNet Res-SNN-34 64.13 → 69.15 % | g_i = σ(W n_i 또는 W u_i)를 token i의 Hippocampus membrane potential에 4 step 모두 곱한다 | 다른 stream의 배율로 쓴 예가 없다. 실수 곱. TPAMI 판 숫자와 다를 수 있다 |
| 9. 입력이 leak를 바꾸는 gate (DGN) | Bai 외, 2026, ICLR [10] | 입력 spike가 current를 주는 동시에 leak 통로를 열어, V에 곱해지는 감쇠율을 바꾼다(LSTM의 forget gate에 비유) | 아니오(입력에 따른 곱) | 한 stream, 다른 논문 숫자와 비교. SHD feedforward 85.18 vs TC-LIF 83.08 %, recurrent 87.78 vs 88.91 %(더 낮음), TIDIGITS 99.10 % | Neocortex spike가 Hippocampus neuron의 leak 항을, H가 current 항을 맡는다 | 두 stream 결합에 쓴 적이 없다. Hippocampus는 4 step뿐이라 leak가 일할 시간이 짧다. membrane 계열 중 가장 낮은 순위 |
| 10. 입력이 시간상수를 정함 (LTC-SN) | Yin, Corradi, Bohté, 2023, Nature Machine Intelligence [11] | 1/τ = σ(W[x_t, u_{t−1}]) | 아니오(곱 + sigmoid) | online 학습(FPTT): PS-MNIST 94.77 %, DVS Gesture 97.22 %. BPTT로는 Add 과제에서 발산. 입력 의존 τ와 고정 τ의 ablation이 없다 | Neocortex neuron에서 1/τ(i) = σ(W[c_i, u_{i−1}]). Hippocampus가 token마다 Neocortex 기억을 얼마나 남길지 정한다 | 우리는 BPTT(시간을 펼쳐 역전파)로 학습한다. 입력 의존 τ는 고정 kernel이나 병렬 scan 가속을 쓸 수 없게 한다. forecasting 근거가 없다 |
| 11. 맥락이 문턱·시간상수를 정함 | Hammouamri 외, 2022, TMLR(초록·code만) [12]; Luca 외, 2026, Frontiers in Neuroscience [13] | θ = θ₀ + p·context(또는 τ) | 예(덧셈) | Luca: recurrent network는 ≈ 600 krad까지 sub-pixel 오차를 유지, 맥락 없이는 ≈ 100 krad. feedforward는 "neither threshold nor time-constant adaptation significantly improved accuracy". Hammouamri 숫자는 읽지 못함 | θ_i = θ₀ − W_θ n_i(Hippocampus 쪽). 사실상 bias current를 더하는 것과 같다 | fusion 정확도 연구가 아니다(continual learning, 하드웨어 강건성). 맥락이 scalar다. 더하기 계열이라 6번보다 약하다 |
| 12. branch 강도 gate (DendSN DBG) | Huang 외, 2024(v2 2025), **arXiv only** [14] | soma 입력 = Σ_b κ_b·Y_b. 위에서 오는 신호가 κ_b를 가린다 | AC synapse + neuron-step마다 곱 K번 | pMNIST 50 task: LIF 23.73 %, LIF + EWC(망각을 줄이는 학습 규칙) 35.01 %, DendSNN + dEWC + DBG "above 42 %". 속도 ≈ 0.92× LIF | κ_b(c_i): Hippocampus count가 Neocortex의 어느 τ member가 soma를 움직일지 고른다(neuron-step당 곱 4번) | 논문의 mask는 task별 고정 무작위 mask다. 입력에 따라 바뀌는 판은 시험되지 않았다 |
| 13. NMDA dendrite gain 조절 | Wybo 외, 2023, PNAS [15] | 맥락이 입력-출력 곡선의 기울기(gain)를 바꾼다: y = σ(g_t ⊙ (Wx − x_shift) + b) | rate 추상화는 실수 곱(spiking 시연은 따로 있음) | 공유 weight + task별 gain으로 48 task "solved all". 47-task EMNIST에서 task별 readout과 같은 성능. shunting inhibition은 bias보다 gain 조절에 더 잘 맞음 | 직접 넣기보다 "bias(더하기)보다 gain(곱하기)"을 고르는 이론 근거(6·8번) | 맥락이 이산 task ID이고 gain도 task별로 학습한다. 시계열이 없다 |
| 14. Active dendrite 맥락 gate | Iyer 외, 2022, Frontiers in Neurorobotics(비spiking) [16]; Pes 외, 2024, IEEE AICAS(첫 spike 시각 부호 SNN) [17] | Iyer: (wᵀx + b)·σ(max_j u_jᵀc). Pes: 맥락이 spike 시각을 늦추고, 너무 늦으면 사실상 꺼진다 | Iyer 아니오, Pes 예 | Iyer MT10 87.5 vs 76.6 %(10회). Pes Split MNIST 88.3 % vs ≈ 69.4 %(5 seed) | task ID 대신 Neocortex 상태로 d_i = max_j u_jᵀu_i를 만들고 Hippocampus feature를 σ(d_i)로 gate한다 | 두 논문 모두 맥락은 주어진 task ID다. Iyer는 비spiking이다 |
| 15. burst로 두 신호 나르기 | Payeur 외, 2021, Nature Neuroscience [18]; Naud & Sprekeler, 2018, PNAS(Crossref 기록만) [19]; Capone 외, 2023, PNAS(초록만) [20] | event 비율은 아래에서 오는 신호, burst 확률은 apical로 오는 위쪽 신호를 나른다 | 예(inference) | 학습 규칙 결과: CIFAR-10 test error 20.1 %(400 epoch, backprop과 비슷), MNIST ≈ 1.1 %, ImageNet top-5 error 56.1 % | 4 step에 event(≥ 1 spike, Hippocampus)와 burst(≥ 2 spike, Neocortex)의 두 단계 code | 학습 신호용이지 inference 결합용이 아니다. 4 step은 이미 Hippocampus 크기(0–4 spike)를 담고 있다. 낮은 순위 |
| 16. 발화율 cross-gate + concat (CMA) | Zhou 외, 2024, Neural Networks [21] | 한 stream의 발화율로 만든 점수(σ)를 다른 stream spike에 곱하고 둘을 이어 붙인다. q·k 곱이 없다 | 아니오(실수 점수 × spike) | DVS-SLR(0.2 s, T = 20): early concat 76.54, middle concat 77.43, late OR 72.92, late 확률 평균 75.83, CMA 83.67 %(event만 70.49, frame만 64.71). 9개 설정에서 CMA − 나은 concat = +4.4 ~ +8.1점(note 계산). LIF·IF·PLIF·LIAF neuron 모두에서 middle concat보다 나음 | g_i = σ(MLP(Z_i 또는 u_i))를 H_·,i에 곱한 뒤 concat + projection. IAND와 달리 키울 수도 있다 | 비디오류 분류, 한 번 run, 실수 곱. Neocortex가 조용하면 g = σ(bias)라는 상수가 된다 |
| 17. 순차 stream과 SSA(Spikformer식 spiking self-attention) stream 사이 convex gate (SpikeSTAG) | Hu 외, 2025, **arXiv only** [22] | G = σ(W[H_LSTM; H_SSA]), G ⊙ H_LSTM + (1 − G) ⊙ H_SSA | 아니오(sigmoid, 곱, LSTM은 실수) | Table 3, concat → gate. R²: Metr-la 0.858 → 0.868, PEMS-BAY 0.854 → 0.874, Solar 0.964 → 0.973, Electricity 0.966 → 0.987. RSE: 0.412 → 0.375, 0.413 → 0.384, 0.251 → 0.246, 0.214 → 0.207. PEMS-BAY에서는 concat이 SSA 혼자(0.860)보다 낮았다. energy 4.39 mJ vs iSpikformer 3.19 mJ | G_i = σ(W[H̄_i; Z_i]), out_i = G_i ⊙ H_i + (1 − G_i) ⊙ P Z_i, G ≈ 0.5에서 시작. 평균 (1 − G)가 Neocortex 사용 비율이다. ≈ 8.2 k parameter | 한 번 run(seed 미기재), 본문·caption·표의 설명이 엇갈린다. 순차 stream이 spiking이 아닌 LSTM이다. G → 1(Neocortex 무시)이 가능하다. 가장 비슷한 설정이다 |
| 18. SNN 안 concat·sum·FiLM·gate 비교 + 균형 학습 (TAAF) | Shen(Jiangrong) 외, 2025, ACM MM [23] | 두 spiking ResNet-18 branch를 concat 또는 가중합한다. step별 중요도로 loss를 가중하고, modality별로 gradient를 조절한다 | concat·sum 예, FiLM·gate 아니오 | AVE / CREMA-D 정확도(loss 조정 SNN): concat 69.65 / 75.27, sum 69.41 / 74.60, FiLM 66.41 / 65.72, gate 63.68 / 68.01 %. TAAF 전체(concat) 70.65 / 77.55 %. 균형 방법으로 +1.0 ~ +3.2점. "without the TAAF module … the modality dominance remains unchanged" | "concat·sum + 학습 쪽 균형"이 기본이어야 한다는 근거 | 분류, FiLM·gate 정의 없음, 한 번 run. 시간 정렬을 뺀 sum(79.17)이 전체 model보다 높은데 설명이 없다 |
| 19. 따로 학습한 뒤 이어 붙이기 | Ceolini 외, 2020, Frontiers in Neuroscience [24]; Taunyazov 외(VT-SNN), 2020, RSS [25] | Ceolini: 각 stream을 따로 학습하고, 끝 층을 버린 뒤 spike 층을 이어 붙여 새 출력 층만 학습한다. VT-SNN: 이어 붙여 함께 학습한다 | 예 | Ceolini, Loihi(3-fold CV): EMG 55.7 ± 2.7, DVS 92.1 ± 1.2, 합침 96.0 ± 0.4 %. energy 173 / 815 / 1,105 µJ. VT-SNN: tactile 0.71, vision 0.73, 합침 0.81 | [H_t,i; Z_i] → Linear(2D → D) → BN → LIF(D = 64이면 8.2 k, spike 입력이라 AC). Neocortex를 자기 head로 먼저 학습한 뒤 결합을 학습한다 | 분류, late fusion(출력 직전 합치기)뿐, Ceolini는 branch를 고정했다. 잡음 많은 약한 stream은 오히려 해칠 수 있다(El-Assal: 71.23 → 63.45 % [26]) |
| 20. 상호정보량 가중합 (MK-SGN의 SMF) | Zheng(Naichuan) 외, 2026, Neurocomputing [27] | modality 사이 상호정보량으로 정한 scalar 가중치로 더한다 | 가중치는 실수 scalar | NTU XSub: SMF 76.1, spiking attention 76.0, concat 75.4, sum 75.2 % | 낮음. scalar 하나로는 조용한 stream을 살리지 못한다 | 같은 종류의 skeleton modality 4개, 한 번 run. 본문이 attention과 concat 값을 바꿔 적었다 |
| 21. 합치는 깊이 (early·middle·late) | Bjorndahl 외, 2024, **arXiv only**(ICASSP 2025 제출) [28]; CMA Table 7 [21] | 입력·중간·출력 직전에서 concat | 예 | N-MNIST + SHD: 시각만 92.25, 청각만 95.29, early 97.35, middle 97.45, late 98.43 %. McNemar p = 0.473, 1.00, 0.720(유의차 없음). CMA: middle이 early보다 9개 중 6개 설정에서 높았고 차이는 최대 2점 | 지금도 block마다 중간 깊이에서 합친다. 위치를 옮기는 것은 우선순위가 낮다 | 관계없는 두 데이터셋을 인위로 짝지었다 |
| 22. 초기 membrane potential 쓰기 / 학습 IMP | Aydin 외, 2024, CVPR Workshops [29]; Shen(Hangchi) 외, 2024, NeurIPS [30] | Aydin: ANN이 SNN 각 층의 t = 0 membrane potential을 쓴다. IMP: neuron마다 학습하는 초기값 | Aydin 초기화는 실수, IMP는 inference 비용 0 | Aydin DHP19: 0 초기화 9.12 px vs 고른 초기화 6.11 px(마지막 구간). hybrid 5.19 / 4.97 px, 순수 SNN 21.37 / 19.17, ANN 5.03 / 4.67. 전력 0.424 W vs 3.718 W. IMP(한 번 run): ImageNet-100 SEW-R18 T = 4에서 SDT 79.10 → 79.90, LTS 80.20 → 80.80, TET 78.50 → 78.70. CIFAR10-DVS TET 84.90 → 85.90 | Neocortex neuron마다 IMP(결합은 그대로)를 둔다. 첫 token이 발화하지 못하는 문제를 겨냥한 공짜 진단이다. 결합판은 u_0 = W_0·mean_i c_i(Hippocampus 요약이 Neocortex의 시작 상태를 정함) | u_0는 창 전체 정보를 모든 Neocortex step에 넣는다(순서 문제). u_0가 크면 늘 발화한다. Aydin은 영상 workshop 논문이다 |
| 23. Hippocampus–neocortex(CLS) 결합 SNN | Shi 외(CH-HNN), 2025, Nature Communications [31]; D'Alba 외, 2025, Neurocomputing(초록만) [32] | CH-HNN: ANN이 SNN hidden neuron에 0/1 mask를 씌운다. D'Alba: 재생(replay) 단계에서 apical 증폭 | mask는 예 | CH-HNN: episode 추론만 70.70 vs 전체 70.72 %(Split Tiny-ImageNet) | mask는 원래 NSMT AND mask와 같은 계열이다. 우리 10-17 분석에서 다른 창의 Neocortex 출력으로 바꿔도 ETT MSE는 ±0.3 % 안이었다. 즉 이미 시험되어 효과가 거의 없었다. replay는 표본 사이 학습용이라 창 단위 결합에 맞지 않는다 | 둘 다 continual learning이 목적이다 |
| 24. fusion module gradient 키우기 (IEMF, 학습 방법) | He 외, 2026, Science Advances [33] | 연산자는 그대로 둔다. 단일 modality 근거가 약할 때 fusion module의 gradient만 ξ_t ∈ (0, 2γ)배로 키운다 | inference는 바뀌지 않는다 | SNN, Kinetics-Sounds, LFM 결합 54.63 → 63.53 %. concat·MSLR·OGM-GE·LFM과 데이터 3개에서 모두 이득, 학습 계산도 줄었다 | Neocortex 전용·Hippocampus 전용 근거 점수가 필요하다(보조 head 등) | regression에서 시험된 적이 없다 |

## 공통으로 보이는 것

**1. 지금 IAND gate에서는 "Neocortex 무시"가 가장 쉬운 답이다.** SEW는 IAND (1 − A)·S를 골랐다. A ≡ 0이면 그대로 통과(identity)하고, 마지막 BN을 0으로 두면 쉽게 그렇게 된다. AND는 늘 A ≡ 1이어야 해서 출력이 꺼지는 문제가 생겼다(DVS Gesture 70.49 vs IAND 95.49 %) [1]. 추론: 우리 x·(1 − y)는 같은 IAND다. y가 발화하지 않으면 Hippocampus가 그대로 지나가고 Neocortex는 아무 일도 하지 않는다. surrogate gradient(spike 대신 쓰는 부드러운 대용 미분)는 neuron 입력이 문턱에서 멀면 0에 가까워진다. 그래서 학습이 조용한 y를 깨우지 못한다. ∂out/∂y = −x라서 Hippocampus spike x가 0인 곳에서는 gradient도 0이다. 항등(γ = 1, β = 0)에서 시작하는 FiLM식 조절도 같은 "무시" 기본값을 가진다. 반대로 current 더하기, concat + projection, 0.5에서 시작하는 convex gate에서는 Neocortex 항이 늘 neuron 입력에 들어간다. 이들은 weight가 0으로 가야만 무시할 수 있다.

**2. 출력 spike보다 current(문턱 앞)에서 합친 쪽이 이겼다.** XOResNet은 spike OR보다 neuron 앞 current 더하기가 모든 깊이에서 나았다고 보고했다(11층 90.59 vs 91.28 %, 50층 93.88 vs 94.11 %) [3]. CIS는 두 current를 문턱 앞에서 합쳐 더하기·concat보다 대부분의 지표에서 나았다 [4]. TS-LIF는 soma에 직접 current 통로를 둔 neuron으로 TC-LIF를 6/6 칸에서 이겼다 [6]. 다만 TS-LIF와 TC-LIF는 직접 주입 말고도 다르다(아래 "한계"). 같은 흐름에서, cLIF는 맥락 쪽 **membrane potential**로 조절한다 [7]. 추론: 우리 Neocortex spike는 70–83 % 위치에서 0이지만 membrane potential은 0이 아니다. membrane potential을 넣으면 어떤 attention 점수로도 쓸 수 없던 위치를 쓸 수 있다.

**3. 한쪽이 맥락일 때는 곱하기가 더하기보다 나았다. 그러나 반대 근거도 있다.** SNN에서 통제된 비교는 Ferrand 하나다. 곱하기가 +9.0 / +15.6점 높았고, 순서 기억 과제에서 더하기는 우연 수준이었다(5회) [7]. Wybo도 dendrite 조절이 bias보다 gain에 더 잘 맞는다고 보고했다 [15]. 반대로 TAAF에서는 FiLM과 gate가 concat보다 3–10점 낮았다 [23]. 두 결과는 설정이 다르다. Ferrand의 과제는 맥락 없이 답할 수 없지만, TAAF의 FiLM·gate는 구현조차 정의되지 않았다. 추론: 우리 ETT에서는 Hippocampus만으로도 base와 같으므로, 곱하기에 유리한 조건이 아니다. 둘 다 시험해야 한다.

**4. SNN 안에서 연산자끼리의 차이는 작고 들쭉날쭉하다.** 분명한 것은 spike OR가 가장 나쁘고, FiLM과 단순 gate는 해칠 수 있다는 것뿐이다.

| 논문 (과제) | concat | sum / 더하기 | 다른 연산자 |
|---|---|---|---|
| TAAF (AVE / CREMA-D) [23] | 69.65 / 75.27 | 69.41 / 74.60 | FiLM 66.41 / 65.72, gate 63.68 / 68.01 |
| CMA (DVS-SLR, 0.2 s, T = 20) [21] | early 76.54, middle 77.43 | — | late OR 72.92, late 확률 평균 75.83, CMA 83.67 |
| MK-SGN (NTU XSub) [27] | 75.4 | 75.2 | spiking attention 76.0, 상호정보량 가중 76.1 |
| CIS (MSRS, SD) [4] | 41.73 | 40.89 | gate current 주입 43.70 |
| SpikeSTAG (Metr-la R²) [22] | 0.858 | — | convex gate 0.868 |
| Bjorndahl (N-MNIST + SHD) [28] | early 97.35, middle 97.45, late 98.43 | — | — |
| SEW (DVS Gesture, residual) [1] | — | ADD 97.92 | IAND 95.49, AND 70.49 |

**5. 한쪽 stream이 지배하면 연산자만 바꿔서는 풀리지 않았다.** TAAF는 균형 module이 없으면 "the modality dominance remains unchanged during the whole training process"라고 적었다. 균형 방법은 정확도를 1.0–3.2점 움직였다 [23]. IEMF는 연산자를 그대로 두고 fusion module의 gradient만 키워, 시도한 모든 연산자에서 SNN 정확도를 올렸다 [33]. Ceolini는 각 stream을 먼저 따로 학습시킨 뒤 결합만 학습했다. 혼자서는 55.7 %인 약한 EMG도 +3.9점을 더했다 [24]. 추론: "시험 때 Neocortex를 0으로 바꿔도 1 % 미만"은 전형적인 modality 불균형 증상이다. 연산자 교체에는 학습 쪽 점검을 적어도 하나 붙여야 한다.

**6. 합치는 깊이는 별로 중요하지 않았다.** Bjorndahl에서 early·middle·late 사이 차이는 유의하지 않았다(p ≥ 0.47) [28]. CMA에서도 middle이 early보다 높은 정도는 최대 2점이었다 [21]. 추론: 결합 위치를 옮기는 것보다 연산자의 "무시" 기본값과 stream 균형이 더 중요하다.

**7. 어떤 연산자도 조용한 Neocortex는 고치지 못한다.** 두 note가 같은 결론을 냈다. Neocortex 출력이 통째로 0인 70–83 % 위치에서는, spike를 받는 어떤 연산자도 나를 정보가 없다(W_n·Z는 그곳에서 BN의 상수 offset만 남는다). 길은 둘이다. Neocortex 앞부분을 고치거나(IMP 등 [30]), spike 대신 membrane potential을 넣는 것이다(실수 MAC를 받아들임). 덧붙여 Luca는 맥락 조절이 recurrent network에서만 도움이 되고 feedforward에서는 도움이 되지 않았다고 보고했다 [13]. 추론: T = N 주장을 위해서는 조절이 Neocortex의 시간 처리 안에서 일어나야 하고, Neocortex가 출력에 중요해지려면 Hippocampus neuron이 Neocortex 신호를 직접 받아야 한다.

## 추천 3가지와 시험 방법

### 순위

두 note의 상위 3개는 다음과 같다. note 1은 (1) CIS식 current 주입(DH-LIF branch를 생물판으로), (2) concat + 학습 쪽 균형, (3) SpikeSTAG convex gate 또는 CMA gain이다. note 2는 (1) cLIF apical 곱하기(CS-TPN을 안전판으로), (2) IMP 주입 + LTC-SN, (3) pop neuron branch routing(DH-SNN, DendSN, TS-LIF), 가작으로 SpikeSTAG gate다. 이를 하나로 합치면 아래와 같다.

| 순위 | 방법 | 두 note에서 | 이 순위인 이유 |
|---|---|---|---|
| 1 | **Neocortex를 Hippocampus neuron의 문턱 앞 current로 넣기** (더하기 [4], gate [4], 곱하기 [7]; spike 또는 membrane potential 입력) | note 1의 1위, note 2의 1위 | 두 note가 각자 다른 논문에서 출발해 같은 계열을 1위로 골랐다. 희소한 spike 두 개를 곱하지 않는다. token끼리 맞물린다. IAND의 공짜 "무시" 해가 없다. membrane potential판은 조용한 Neocortex를 우회한다. 근거는 CIS ablation(더하기·concat보다 나음), XOResNet(current > spike), Ferrand(5회, 곱 > 더하기)다. 1a는 spike 덧셈만으로 되어 저전력 목표와도 맞는다 |
| 2 | **concat + spiking projection, 그리고 학습 쪽 균형** [23][24][25][33] | note 1의 2위 | SNN 비교에서 가장 안정적인 단순 연산자다(TAAF: concat ≥ sum > FiLM·gate, CMA: concat > late OR, MK-SGN: concat ≥ sum). spike 덧셈만으로 된다. 지배 문제는 학습에서 풀어야 한다는 근거가 세 편 있다. 가장 깨끗한 대조이기도 하다. concat + 균형 학습 뒤에도 Neocortex를 0으로 바꾼 영향이 1 % 미만이면, ETT에서 Neocortex가 쓸 정보를 거의 나르지 않는다는 뜻이다 |
| 3 | **Neocortex를 살리고 그 안에서 합치기: 학습 IMP + pop neuron branch routing** [30][5] (선택: [29][14][6]) | note 2의 2위(IMP 부분)와 3위, note 1의 요구 (d) | T = N contribution과 바로 이어진다. 합치는 일이 Neocortex 시간 처리 안에서 일어나므로 memory 끔·순서 섞기 대조가 결정적인 시험이 된다. spike 덧셈만으로 된다. `forward_branches` hook이 이미 있다. IMP는 측정된 문제(첫 token 침묵)를 직접 겨냥하고 inference 비용이 0이다. 근거는 가장 약하다(합성 XOR, 영상, 한 번 run) |
| 4 (진단 arm) | SpikeSTAG convex gate [22], CMA 발화율 gain [21] | note 1의 3위, note 2의 가작 | 가장 비슷한 설정(forecasting, 순차 stream + SSA)이고 평균 (1 − G)로 Neocortex 사용 비율을 바로 읽을 수 있다. 그러나 arXiv only·한 번 run·설명 불일치이고, 순차 stream이 LSTM이며, 실수 곱을 쓴다. TAAF에서는 단순 gate가 concat보다 나빴다. 1순위의 선택 arm으로 둔다 |
| 보류 | LTC-SN [11], DendSN gate [14], MA-SNN [9], DGN [10], CS-TPN [8], TS-LIF 두 compartment [6] | note 2의 2위 일부 등 | 두 stream 결합 근거가 없다. LTC-SN은 BPTT 발산 보고가 있고 τ ablation이 없다 |
| 제외 | burst [18], CLS mask·replay [31][32], task ID dendrite [16][17], 문턱만 조절 [12][13], 논리 연산만 [1][2], 상호정보량 가중합 [27], 깊이 바꾸기 [28] | 두 note 모두 낮음 | 학습 신호용이거나, 우리 model에서 이미 효과가 없었거나, 더하기와 같거나, 조용한 Neocortex를 못 살린다 |

### 모든 시험에 공통

- **고정.** base는 지금 기준(Neocortex membrane shortcut을 쓰는 query population model)이다. 학습 recipe, T_h = 4, D = 64, 층 수, patch, head는 바꾸지 않는다. 모든 arm은 같은 code에서 돌린다. pool이 도는 동안에는 scratchpad 사본에서 구현하고, 합친 뒤 기본 경로가 비트 단위로 같은지 확인한다.
- **neuron과 과제.** pop LIF와 pop f-LIF를 늘 함께 돌린다. ETT는 4칸(ETTh2·ETTm2 × H96·H720, 입력 96) × seed 3개, MIT-BIH는 seed 3개다. base 쪽 arm 하나는 2 neuron × (12 + 3) = 30 run, Spikformer식 비교 model arm 하나는 15 run이다.
- **판정.** ETT는 neuron마다 12쌍에서 test MSE 상대 변화의 평균 R ≤ −1 %이고 8/12쌍 이상에서 낮아야 "좋음"이다. 비교 V1–V4와 "약함" 읽는 법은 이전 보고서를 그대로 따른다. MIT-BIH는 이전 MIT-BIH 시험의 지표와 판정 규칙을 쓴다. NaN, 결과 누락, 발산은 실패로 센다(fail-closed).
- **Spikformer식 비교 model 처리.** 두 note의 의견이 갈린다. note 1은 Spikformer식의 IAND gate도 같은 연산자로 바꾸라고 하고, note 2는 Spikformer식을 그대로 두라고 한다. 이 보고서의 1·2순위 배치는 Spikformer식 block에 Neocortex 항만 더하는 형태라 둘이 일치한다. Neocortex 항을 0으로 두면 그것이 곧 Spikformer식 대조다. 3순위는 Spikformer식에 바꿀 Neocortex가 없으므로 그대로 기준으로 둔다.
- **run마다 기록할 것.** Neocortex 출력과 결합 neuron의 발화율, 첫 token과 token 위치별 Neocortex 발화율, 시험 때 Neocortex 항을 0으로 바꿨을 때의 ΔMSE(지금은 1 % 미만), 그때 뒤집히는 결합 neuron spike의 비율, 학습된 α 또는 평균 gate 값, Neocortex 쪽과 Hippocampus 쪽으로 가는 gradient 크기의 비.
- **neo_grad.** 지금 code는 결합을 통해 Neocortex로 가는 task gradient를 절반으로 줄인다(`neo_grad = 0.5`). 이 값을 그대로 두고 적는다. 1순위에서 가장 좋은 arm만 1.0으로도 돌린다. 그러지 않으면 "차이 없음"의 일부가 절반 gradient 때문일 수 있다.
  - **정정 (11-19 저녁):** 위 문장은 틀렸다. `neo_grad = 0.5`는 code의 기본값일 뿐이고, 기준 model의 실행 스크립트(`run_ett_dual.sh`, `run_mitbih4.sh`의 `nsmt4q_*` 조건)는 `--neo_grad 1.0`을 준다. 저장된 run 설정에서도 1.0이다(ETTh2 H96 seed 42, 기준 pop LIF / pop f-LIF). 즉 지금까지의 기준 model run은 gradient를 줄이지 않았다. 따라서 "절반 gradient 때문일 수 있다"는 우려와 "1.0으로도 돌린다"는 계획은 해당 없다.
- **energy.** 기존 syops 숫자와, 새 연산을 명시적으로 센 숫자를 함께 적는다. syops는 neuron 안에서 곱하는 gain을 보지 못한다. sigmoid와 gate 곱은 MAC로 센다. Neocortex 쪽 계산(W_n·Z 등)은 Z가 4 step 동안 같으므로 token마다 한 번만 센다. 이전 보고서대로, 지금 syops는 cross-attention의 q·k 곱과 bias 덧셈도 세지 않는다. 그래서 cross-attention을 없애도 energy가 줄어든 것으로 나오지 않는다. 이 부분도 따로 센다.
- **memory·순서 대조.** 판정을 통과한 arm에만 Neocortex memory 끔과 token 순서 섞기 대조를 돌린다(arm당 60 run). 이 대조는 Hippocampus 쪽이 아니라 Neocortex 자기 token 입력과 상태에 건다.

### 1순위: Neocortex를 Hippocampus neuron의 문턱 앞 current로 넣기

| 항목 | 내용 |
|---|---|
| 무엇을 바꾸나 | **배치 (가).** Hippocampus block을 Spikformer식 비교 model의 block(self-attention + IAND, MLP + IAND)으로 되돌린다. 즉 결합 cross-attention의 query를 다시 Hippocampus에서 만든다. 그리고 attention 결과 z_i가 들어가는 attention 출력 neuron(지금 결합 attention의 결과가 들어가는 바로 그 neuron)에 Neocortex 항을 current로 더한다. Neocortex 항이 0이면 정확히 Spikformer식 비교 model이다 |
| 왜 배치 (가)인가 (추론, 코드 확인) | query 형태의 Hippocampus block에서 token 사이 섞기는 결합 cross-attention 하나뿐이다(`model.py`의 `Block.forward`, xattn 경로). attention을 지우기만 하면 Hippocampus가 token을 섞지 못한다. 그러면 Spikformer식과의 비교에 두 가지 변화가 섞인다. (가)에서는 base와 Spikformer식이 Neocortex 항 하나만 다르다. 또 attention 출력 neuron은 Hippocampus self-attention을 나르려면 발화해야 한다. 그래서 "y가 조용하면 Neocortex가 무시된다"는 공짜 해가 사라진다. 무시하려면 α가 0으로 가야 하고, α는 기록할 수 있다 |
| arm | **1a 더하기·spike**: + α ⊙ BN(W_n Z_i). CIS에서 gate를 뺀 형태다 [4]. **1b 더하기·membrane potential**: Z_i 대신 Neocortex 마지막 층의 H − θ(`forward_charge`)를 넣는다. **1c 곱하기·membrane potential**: Hippocampus current × (1 + tanh(W_a u_i)). cLIF code의 tanh 형태이고, Neocortex가 조용하면 그대로 통과한다 [7]. α는 channel마다 하나이고, 0에서 떨어진 값으로 시작한다 |
| 선택 arm | 1d CIS gate: α ⊙ σ(BN(W_g Z_i)) ⊙ BN(W_n Z_i) [4]. 1e ReLU 곱(cLIF 논문 형태). 1c와의 차이가 곧 Neocortex 사용량의 척도다. 1f SpikeSTAG식 convex gate(G를 0.5에서 시작) [22]. 1g **배치 (나)**: cross-attention과 IAND를 통째로 지우고 x_i ← LIF(BN(W_h x_i) + α ⊙ BN(W_n Z_i))로 바꾼다. 그러면 Neocortex(T = N)가 유일한 token 섞기 통로가 된다. "T = N 처리가 attention을 대신할 수 있는가"를 직접 묻는 대담한 arm이다 |
| 대조 | Spikformer식 비교 model = Neocortex 항 0. 기본 경로가 비트 단위로 같으면 기존 run을 다시 쓴다. concat 대조는 2순위의 2a다. 시험 때 Neocortex 항을 0으로 바꾼 ΔMSE를 모든 arm에서 잰다 |
| energy | 1a: AC만 쓴다. W_n(≈ 4.1 k)은 token마다 한 번 계산하고, α는 BN에 접는다. 1b: u_i가 실수라 W_n·u_i가 MAC다(token마다 한 번). 1c: 1b에 더해 neuron-step마다 곱 1회가 든다. LIF 갱신 비용의 약 +42 %이고, block당 4 × N × 64번이다(N = 24에서 6,144, N = 180에서 46,080). 1d: sigmoid와 곱이 MAC다. CIS 결합점 하나는 ≈ 8.3 k parameter다 |
| 양성 기준 | V3(새 base − Spikformer식)가 "좋음"이다. 두 neuron 모두이거나, 한 neuron이라도 한 칸을 빼도 유지되어야 한다. 시험 때 Neocortex를 0으로 바꾼 ΔMSE가 지금(1 % 미만)보다 분명히 크다. α나 gate가 0으로 무너지지 않았다. 그다음 memory 끔·순서 섞기가 새 base보다 "나쁨"이면, Neocortex의 memory와 순서가 쓰인다고 말할 수 있다 |
| run 수 | 1a–1c 90, Spikformer식 대조 0–15, 통과 arm의 memory·순서 대조 60. 선택 arm은 각 30 |

### 2순위: concat + spiking projection, 그리고 학습 쪽 균형

| 항목 | 내용 |
|---|---|
| 무엇을 바꾸나 | Hippocampus block은 1순위 배치 (가)와 같이 Spikformer식 block으로 되돌린다. head 바로 앞 한 곳에 결합 neuron f_i = LIF(BN(W[x_i; Z_i]))를 둔다(W는 2D → D). 깊이는 결과에 크게 영향이 없었으므로(공통 6번) 가장 단순한 한 곳에서 시작한다. block마다 두는 판은 선택이다 |
| arm | **2a** concat, 함께 학습. **2b** 2a + 단계 학습: Neocortex를 자기 head(ETT는 forecasting, MIT-BIH는 분류)로 먼저 학습시킨 뒤 결합을 학습한다 [24]. **2c** 2a + Neocortex 전용 보조 head(학습 때만), 또는 결합 지점에서 Hippocampus 입력을 작은 확률로 0으로 바꾸기. 후자는 논문 근거가 없는 추론이다 |
| 기존 계획과의 관계 | 11-18 밤에 사전 등록한 aux-loss 시험(NSMT reconstruction loss)도 학습 쪽 신호다. 이 보고서를 쓴 뒤 결과가 나왔다(11-19): 켜도 성능은 두 기준 model, 두 neuron, 두 과제 모두 "차이 없음"이었다. Neocortex 발화율은 약 4배가 됐고, 시험 때 Neocortex를 0으로 바꾼 손해는 pop LIF에서 −0.07 → +1.02 %로 커졌다(`docs/MODI_PROJECT_LOG.md` 11-19 항목). 원 시계열 복원 aux는 이어서 돈다. 2c는 이 결과를 보고 다시 정한다 |
| 대조 | Spikformer식 + 같은 결합 neuron(Neocortex 자리 0) 15 run. 2a와 2b·2c를 비교하면 연산자 효과와 균형 학습 효과가 나뉜다 |
| 읽는 법 | 2b나 2c 뒤에도 시험 때 Neocortex를 0으로 바꾼 영향이 1 % 미만이면, ETT에서 Neocortex가 쓸 정보를 거의 나르지 않는다는 뜻이다. 그때는 ETT에서 연산자 탐색을 멈춘다. MIT-BIH도 같은 방식으로 읽는다 |
| energy | AC만 쓴다(spike 입력, W ≈ 8.2 k). 2b·2c는 학습 비용만 늘고 inference는 2a와 같다. 보조 head는 inference 때 뺀다 |
| 양성 기준 | 1순위와 같다 |
| run 수 | 2a 30 + 대조 15, 2b 30, 2c 30 |

### 3순위: Neocortex를 살리고 그 안에서 합치기

| 항목 | 내용 |
|---|---|
| 무엇을 바꾸나 | 결합(지금 attention + IAND, 또는 1·2순위에서 이긴 연산자)은 그대로 두고 Neocortex 쪽만 바꾼다 |
| arm | **3a IMP만**: Neocortex neuron마다 학습 초기 membrane potential을 둔다 [30]. 결합은 바꾸지 않는다. **3b 3a + branch routing**: `forward_branches`로 Neocortex 자기 token current는 느린 member(τ 16/32)에, Hippocampus count c_i = Σ_t h[t, i]는 빠른 member(τ 4/8)에 넣는다 [5]. 선택 **3c**: u_0 = W_0·mean_i c_i(Hippocampus 요약이 Neocortex 시작 상태를 정함) [29], 또는 DendSN식 κ_b(c_i) gate [14] |
| c_i를 어디서 가져오나 (추론, 코드 확인) | 지금 forward는 Hippocampus embedding → Neocortex → Hippocampus block 순서로 돈다. embedding spike에서 c_i를 만들면 순서를 바꿀 필요가 없고 미래 token도 새지 않는다. 다만 embedding은 Neocortex 입력과 같은 patch에서 나오므로 새 정보가 적다. 창 전체의 맥락을 주려면 attention을 거친 Hippocampus 출력을 써야 한다. 그러면 계산 순서를 바꿔야 하고 Neocortex step i가 미래 token을 보게 된다. embedding판으로 먼저 시작한다 |
| 대조 | Spikformer식 비교 model은 그대로 기준으로 둔다. memory 끔·순서 섞기는 Neocortex 자기 token 입력과 상태에 건다 |
| energy | 3a: neuron마다 저장된 상수 하나, inference 비용 0. 3b: count 입력이라 AC만 쓴다. 3c의 κ_b gate는 neuron-step마다 곱 4번이다 |
| 양성 기준 | 3a: 첫 token이 발화하고, V1(새 base − 지금 base)이 "좋음"이거나 Neocortex를 0으로 바꾼 ΔMSE가 커진다. 3b: V3가 "좋음"이고, memory 끔·순서 섞기가 3b보다 "나쁨"이다. 이것이 T = N contribution의 가장 직접적인 근거가 된다 |
| run 수 | 3a 30, 3b 30, 3c 선택 30 |

### 진행 순서

1. 학습 없이 지금 base에서 첫 token 발화율, Neocortex 전체가 0인 위치 비율, Neocortex를 0으로 바꾼 ΔMSE를 다시 잰다. 11-08 숫자는 membrane shortcut 이전이기 때문이다.
2. 3a(IMP, 30 run)를 먼저 돌린다. 싸고, 다른 시험과 독립적이다.
3. 1a–1c(90 run)와 2a(30 + 15 run)를 함께 돌린다. Spikformer식 대조를 함께 쓴다.
4. 2a에서 Neocortex 효과가 없으면 2b·2c(60 run)로 학습 균형이 문제였는지 본다.
5. 통과한 arm에 memory·순서 대조(arm당 60 run)를 돌리고, 3b(30 run)로 넘어간다.

## 한계와 빠진 것

- **데이터.** 33편 가운데 ETT나 MIT-BIH를 쓴 논문은 없다. regression 과제를 쓴 것은 SpikeSTAG와 TS-LIF(Metr-la, PEMS-BAY, Solar, Electricity)뿐이다. 나머지는 대부분 영상·음성·event 분류다.
- **반복 횟수.** 대부분의 숫자가 한 번 run이다. 예외는 "후보 목록" 머리말에 적었다. CIS의 이득은 이미지 품질 지표에서 작았고(지표별 0.04–2.8), SpikeSTAG는 seed 수를 적지 않았다.
- **아무도 하지 않은 비교.** neuron 수준 곱하기 결합과 cross-attention 결합을 같은 model에서 비교한 SNN 논문은 없다. apical·dendrite·두 compartment 방식의 두 stream 결합을 forecasting이나 ECG에서 시험한 논문도 없다. spiking 순차 stream + spiking Transformer에서 한쪽이 다른 쪽 membrane potential을 초기화하거나 gate한 예도 없다(Aydin의 영상 ANN → SNN뿐). LTC-SN의 입력 의존 τ와 고정 τ를 비교한 ablation도 없다. 두 stream 사이 FiLM·adaLN을 쓴 SNN 논문은 TAAF의 정의 없는 baseline 말고는 찾지 못했다.
- **끝까지 읽지 못한 것.** D'Alba는 숫자를 읽지 못했다(paywall). Hammouamri는 초록과 code README만, Capone은 초록만, Naud & Sprekeler는 Crossref 기록만 봤다. Payeur와 Iyer는 일부만 읽었다. MA-SNN 숫자는 TPAMI 판이 아니라 arXiv 판에서 나왔다. DH-SNN의 XOR 정확도는 그림에만 있다. LTC-SN 표는 배치가 읽기 어려워 짝이 틀렸을 수 있다.
- **논문 안의 모순.** SpikeSTAG: 본문은 교통 데이터 이득을 R² "1.0–1.4 %"라 했지만 표로는 PEMS-BAY에서 상대 +2.3 %다. gate 없는 concat 비교(W3)를 본문과 caption이 다르게 설명하고(순차 처리 vs concat), W2의 정의도 본문과 caption이 다르다. 평균 R²가 표에서는 0.823, 본문에서는 0.818이다. TAAF: 시간 정렬을 뺀 sum(CREMA-D 79.17 %)이 전체 model보다 높은데 설명이 없다. energy는 "18.5 MJ vs 49.91 MJ"로 단위가 그대로 찍혀 있다. FiLM과 gate는 정의가 없다. MK-SGN: 본문이 attention과 concat 값을 바꿔 적었다. CS-TPN: "up to 330x fewer neurons"를 뒷받침하는 표가 없다. SEW와 SynA의 재실행에서 ADD·AND·IAND의 순위가 뒤집혔다. CMA 비교의 attention fusion baseline은 다른 논문의 module이다.
- **note 사이의 해석 차이.** note 1은 TS-LIF가 TC-LIF를 이긴 것을 "soma 직접 주입"의 근거로 썼다. note 2는 직접 주입만 뺀 ablation이 없다고 적었다. 두 neuron은 직접 주입 말고도 두 spike 섞기(κ) 등이 다르므로, 이 비교는 깨끗한 근거가 아니다. Spikformer식 처리에 대한 두 note의 차이는 위 "모든 시험에 공통"에서 정리했다. Ferrand(곱 > 더하기)와 TAAF(FiLM·gate < concat)도 서로 반대 방향이지만 설정이 다르다.
- **제외한 논문.** attention 기반이라 범위 밖: HI-AVSNN(IEEE TC 2025), S-CMRL(이전 보고서에서 다룸; 여기서는 다른 방법의 CREMA-D 숫자에만 씀), TDFormer. 한 stream이라 제외: SFOD, CE-LIF. 비spiking·venue 없음: Islah 외. STDP 학습이라 한 숫자에만 씀: El-Assal 외. 전문을 못 열었거나 기제를 확인하지 못함: CMCI(paywall), WeightAttention, SGSAFormer, SpikeVFuse, Zhang 외의 lateral connection model, Cooperator, 검색에서만 본 IR·visible membrane current 합산 preprint.
- **우리 쪽 숫자의 한계.** 70–83 %, 첫 token 침묵, 2–4 %는 Neocortex membrane shortcut 이전의 11-08 측정이고, 지금 base에서는 다시 재지 않았다. 이전 보고서에 따르면 membrane shortcut은 ETTm2에서 Neocortex 발화를 약 2배로 늘렸지만 정확도는 그대로였다. `neo_grad = 0.5`가 결과에 미친 크기는 모른다. 지금 syops는 neuron 안의 곱과 attention 핵심 연산을 세지 않는다.

## 인용 확인

2026-11-19에 arXiv API·초록 page, Crossref API(DOI 기록), 학회 proceedings page, mlanthology, web 검색으로 제목·첫 저자·연도·venue를 확인했다. 아래 12편은 보고서가 가장 많이 기대는 논문이다(추천 1–3순위의 논문 전부와, 그 밖에 가장 많이 인용한 논문).

| 논문 | 확인한 곳 | 결과 |
|---|---|---|
| Zhao 외, CIS-Fuse [4] | arXiv 초록 page | 확인. 제목·첫 저자(Rui Zhao) 일치, 2026-07-22 v1, comment·journal 기록 없음 → arXiv only |
| Ferrand 외, cLIF [7] | Crossref(DOI 10.1007/978-3-031-44207-0_32) | 확인. 제목·첫 저자(Romain Ferrand)·2023 일치, ICANN 2023 LNCS, pp. 381–392 |
| Shen(Jiangrong) 외, TAAF [23] | Crossref(DOI 10.1145/3746027.3755622) | 확인. 제목·첫 저자 일치, 33rd ACM MM 2025, pp. 11042–11051 |
| Ceolini 외 [24] | Crossref(DOI 10.3389/fnins.2020.00637) | 확인. 제목·첫 저자(Enea Ceolini) 일치, Frontiers in Neuroscience 14:637, 2020 |
| He 외, IEMF [33] | Crossref(DOI 10.1126/sciadv.ady8751) | 확인. 제목·첫 저자(Xiang He) 일치, Science Advances 12, 2026. arXiv 설명란은 아직 "under review"로 남아 있다 |
| Zheng(Hanle) 외, DH-SNN [5] | Crossref(DOI 10.1038/s41467-023-44614-z) | 확인. Nature Communications 15:277, 2024. note 2가 기억으로 적었던 마지막 저자 Lei Deng도 Crossref에서 확인했다 |
| Shen(Hangchi) 외, IMP [30] | NeurIPS 2024 proceedings page | 확인. 제목·첫 저자·NeurIPS 2024 일치 |
| Fang 외, SEW [1] | NeurIPS 2021 proceedings page, arXiv comment | 확인. 제목·첫 저자(Wei Fang)·NeurIPS 2021 일치 |
| Hu 외, SpikeSTAG [22] | arXiv 초록 page, web 검색 | 확인. 제목·첫 저자(Bang Hu) 일치, v2 2025-08-18, venue를 찾지 못함 → arXiv only |
| Wu 외, XOResNet [3] | Crossref(DOI 10.1016/j.neucom.2026.134933), arXiv API | **부분 불일치.** 제목·첫 저자(Jianfang Wu)·2026·Neurocomputing(vol. 705)은 일치한다. 그러나 출판본 저자는 4명(Jianfang Wu, Junsong Wang, Qiang Yu, Zhiwei Gao)이고, note의 "Wu & Wang"은 arXiv 판 저자다. 이 보고서에는 "Wu 외"로 고쳐 적었다 |
| Feng 외, TS-LIF [6] | ICLR 2025 proceedings page | 확인. 제목·저자 5명·ICLR 2025 일치 |
| Bai 외, DGN [10] | mlanthology | 확인. 제목·첫 저자(Qianyi Bai)·ICLR 2026 일치 |

나머지 21편도 arXiv API·Crossref·OpenAlex로 한 번에 확인했다. 제목, 첫 저자, 연도, venue가 모두 note와 맞았다. SynA-ResNet, CS-TPN, DendSN, Bjorndahl은 venue 기록이 없어 arXiv only가 맞다. Bjorndahl은 Semantic Scholar에도 arXiv 판만 있다. Capone은 PNAS 120(2023), Payeur는 Nature Neuroscience 24(2021), MA-SNN은 TPAMI 45(2023), LTC-SN은 Nature Machine Intelligence 5(2023)로 확인했다.

## 출처

확인 수준은 note의 표기다. 따로 적지 않은 것은 note 작성자가 전문을 읽은 논문이다.

1. Fang 외, "Deep Residual Learning in Spiking Neural Networks", NeurIPS 2021. https://arxiv.org/abs/2102.04159 ; https://proceedings.neurips.cc/paper/2021/hash/afe434653a898da20044041262b3ac74-Abstract.html
2. Shan 외, "SynA-ResNet: Spike-driven ResNet Achieved through OR Residual Connection", arXiv only. https://arxiv.org/abs/2311.06570
3. Wu 외, "XOResNet: Exclusive-OR Meta-Residuals Facilitate Deep Spiking Neural Networks Learning", Neurocomputing 2026. https://arxiv.org/abs/2605.30362 ; https://doi.org/10.1016/j.neucom.2026.134933
4. Zhao 외, "Current Injection Spiking Neural Network for Infrared and Visible Image Fusion", arXiv only, 2026. https://arxiv.org/abs/2607.19879
5. Zheng(Hanle) 외, "Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics", Nature Communications 2024. https://doi.org/10.1038/s41467-023-44614-z
6. Feng 외, "TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting", ICLR 2025. https://arxiv.org/abs/2503.05108 ; https://proceedings.iclr.cc/paper_files/paper/2025/hash/45bd303ea32a8193d3c19de5d5eaf138-Abstract-Conference.html
7. Ferrand 외, "Context-Dependent Computations in Spiking Neural Networks with Apical Modulation", ICANN 2023. https://doi.org/10.1007/978-3-031-44207-0_32 ; https://zenodo.org/records/11489556 ; code https://github.com/IGITUGraz/CSNN
8. Raza & Adeel, "An Overlooked Role of Context-Sensitive Dendrites", arXiv only, 2024. https://arxiv.org/abs/2408.11019
9. Yao 외, "Attention Spiking Neural Networks", IEEE TPAMI 2023(숫자는 arXiv 판). https://doi.org/10.1109/TPAMI.2023.3241201 ; https://arxiv.org/abs/2209.13929
10. Bai 외, "A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks", ICLR 2026. https://arxiv.org/abs/2509.03281 ; https://mlanthology.org/iclr/2026/bai2026iclr-braininspired/ ; https://openreview.net/forum?id=5h741EyfQM
11. Yin, Corradi, Bohté, "Accurate online training of dynamical spiking neural networks through Forward Propagation Through Time", Nature Machine Intelligence 2023. https://doi.org/10.1038/s42256-023-00650-4 ; https://arxiv.org/abs/2112.11231
12. Hammouamri 외, "Mitigating Catastrophic Forgetting in Spiking Neural Networks through Threshold Modulation", TMLR 2022(초록·code README만). https://openreview.net/forum?id=15SoThZmtU ; https://github.com/Thvnvtos/Nm-SNN
13. Luca 외, "Learned adaptive properties for mitigation of weight perturbations in embedded spiking networks", Frontiers in Neuroscience 2026. https://doi.org/10.3389/fnins.2026.1766765
14. Huang 외, "Scalable Dendritic Modeling Advances Expressive and Robust Deep Spiking Neural Networks"(v1 제목 "Flexible and Scalable Deep Dendritic Spiking Neural Networks with Multiple Nonlinear Branching"), arXiv only. https://arxiv.org/abs/2412.06355
15. Wybo 외, "NMDA-driven dendritic modulation enables multitask representation learning in hierarchical sensory processing pathways", PNAS 2023. https://doi.org/10.1073/pnas.2300558120
16. Iyer 외, "Avoiding Catastrophe: Active Dendrites Enable Multi-Task Learning in Dynamic Environments", Frontiers in Neurorobotics 2022(일부 읽음). https://doi.org/10.3389/fnbot.2022.846219
17. Pes 외, "Active Dendrites Enable Efficient Continual Learning in Time-To-First-Spike Neural Networks", IEEE AICAS 2024. https://doi.org/10.1109/AICAS59952.2024.10595872 ; https://arxiv.org/abs/2404.19419
18. Payeur 외, "Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits", Nature Neuroscience 2021(bioRxiv 판 일부 읽음). https://doi.org/10.1038/s41593-021-00857-x ; https://www.biorxiv.org/content/10.1101/2020.03.30.015511v2.full
19. Naud & Sprekeler, "Sparse bursts optimize information transmission in a multiplexed neural code", PNAS 2018(Crossref 기록만). https://doi.org/10.1073/pnas.1720995115
20. Capone 외, "Beyond spiking networks: the computational advantages of dendritic amplification and input segregation", PNAS 2023(초록만). https://doi.org/10.1073/pnas.2220743120
21. Zhou 외, "Enhancing SNN-based spatio-temporal learning: A benchmark dataset and Cross-Modality Attention model", Neural Networks 2024. https://doi.org/10.1016/j.neunet.2024.106677 ; https://arxiv.org/abs/2410.15689
22. Hu 외, "SpikeSTAG: Spatial-Temporal Forecasting via GNN-SNN Collaboration", arXiv only, 2025. https://arxiv.org/abs/2508.02069
23. Shen(Jiangrong) 외, "Spiking Neural Networks with Temporal Attention-Guided Adaptive Fusion for imbalanced Multi-modal Learning", ACM MM 2025. https://doi.org/10.1145/3746027.3755622 ; https://arxiv.org/abs/2505.14535
24. Ceolini 외, "Hand-Gesture Recognition Based on EMG and Event-Based Camera Sensor Fusion: A Benchmark in Neuromorphic Computing", Frontiers in Neuroscience 2020. https://doi.org/10.3389/fnins.2020.00637
25. Taunyazov 외, "Event-Driven Visual-Tactile Sensing and Learning for Robots", RSS 2020. https://doi.org/10.15607/rss.2020.xvi.020 ; https://arxiv.org/abs/2009.07083
26. El-Assal 외, "Spiking two-stream methods with unsupervised STDP-based learning for action recognition", Signal Processing: Image Communication 2025(한 숫자에만 사용). https://doi.org/10.1016/j.image.2025.117263 ; https://arxiv.org/abs/2306.13783
27. Zheng(Naichuan) 외, "MK-SGN: A Spiking Graph Convolutional Network with Multimodal Fusion and Knowledge Distillation for Skeleton-based Action Recognition", Neurocomputing 2026. https://doi.org/10.1016/j.neucom.2025.131796 ; https://arxiv.org/abs/2404.10210
28. Bjorndahl 외, "Digit Recognition using Multimodal Spiking Neural Networks", arXiv only, 2024. https://arxiv.org/abs/2409.00552
29. Aydin 외, "A Hybrid ANN-SNN Architecture for Low-Power and Low-Latency Visual Perception", CVPR Workshops 2024. https://doi.org/10.1109/CVPRW63382.2024.00579 ; https://arxiv.org/abs/2303.14176
30. Shen(Hangchi) 외, "Rethinking the Membrane Dynamics and Optimization Objectives of Spiking Neural Networks", NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/a867a94a427bacbe3f4de16c7ac10ba8-Abstract-Conference.html ; code https://github.com/StephenTaylor1998/IMP-SNN
31. Shi 외, "Hybrid neural networks for continual learning inspired by corticohippocampal circuits", Nature Communications 2025. https://doi.org/10.1038/s41467-025-56405-9
32. D'Alba 외, "Semantization of memories in a hippocampal–cortical spiking neural network", Neurocomputing 2025(초록만). https://doi.org/10.1016/j.neucom.2025.130323
33. He 외, "Incorporating brain-inspired mechanisms for multimodal learning in artificial intelligence", Science Advances 2026. https://doi.org/10.1126/sciadv.ady8751 ; https://arxiv.org/abs/2505.10176 ; code https://github.com/Brain-Cog-Lab/IEMF

내부 자료: `research_notes/NSMT 결합부 비attention 융합 후보/fusion_operators.md`, `research_notes/NSMT 결합부 비attention 융합 후보/neuron_level_and_memory.md`, `reports/NSMT 결합부 spiking attention 후보.md`, `nsmt_flif_v1/classification/model.py`.
