# Bidirectional SNN과 두 stream의 불일치 gating: SpikMixer muxlin novelty 점검

근거 표시 규칙: `[read in full text]` = 해당 section을 PDF, HTML, 코드 원문에서 직접 확인. `[abstract only]` = 초록만 확인. `[secondary summary]` = 검색 요약이나 다른 논문의 서술로만 확인. 따옴표 안 영어는 원문 그대로다. `(fetch)`가 붙은 항목은 요약형 fetch 도구로 HTML을 읽은 것이고, 수치는 가능한 한 로컬 PDF로 다시 대조했다. 조사일은 2026-11-07~08이다.

비교 기준 (우리 모델): [spk_mixer.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/spk_mixer.py) `[read in full text]`. `mixer`와 `mixer_rev`는 별도 weights, reversal은 token 축(`xm.flip(0)`), `H_r`와 `R`은 다시 뒤집어 정렬한다. `s_d = LIF(BN(|H_f - H_r|))`, `L = LIF(BN(W[F OR R; s_d]))`, 출력은 `m = s_d ? L : (F AND R)`이고 학습 때 산술은 `s_d*L + (1-s_d)*min(F,R)`이다. 측정 효과 (gate vs sdlin, 사용자 제공): CinC +1.56 %p, ETT MSE +0.57 % (나빠지는 방향), MIT-BIH +0.00 %p.

## 1. 어떤 SNN 논문이 입력을 정방향과 역방향으로 돌리고, 어떻게 합치는가

### Takeaway
입력을 뒤집어 두 번째 stream을 돌리는 SNN은 이미 있다 (Yin 2021 bi-SRNN, PRF Appendix H, SmolMamba, SPM 등). 확인된 fusion은 mean, concatenation, addition이고, 학습 때만 쓰는 KL loss와 firing rate 곱이 하나 있다. 본문을 읽은 논문 중 두 stream의 차이로 gating하는 것은 없었다. 다만 본문을 못 읽은 논문이 5편 남아 있고, 그중 SpikeVFuse가 가장 확인이 시급하다.

### Cited Findings

**입력 reversal이 실제로 있는 spiking 모델**

| 논문 | Reversal | Weights | 축 | Fusion | Fusion 결과가 binary인가 | Task | 등급 |
|---|---|---|---|---|---|---|---|
| [Yin, Corradi, Bohte 2021, bi-SRNN](https://arxiv.org/abs/2103.12593) | 있음. "reading the sequence from the forward and backward direction" | "two SRNN layers" (분리로 읽힘) | time step | mean. "The mean of these layer's output is then fed into the last layer, an integrator" | 아님 (평균 후 non-spiking integrator) | TIMIT | read in full text |
| [PRF Appendix H](https://arxiv.org/abs/2410.03530) | 있음. "also reverse it and pass it through an independent second PRF Neuron" | 분리 ("independent") | time step | concat 후 linear. Eq. 102 `S ∈ B^{T×N×2D}`, 이어서 `W ∈ R^{2D×D}` | concat은 binary, linear 출력은 real | LRA 중 Image, Pathfinder만 | read in full text |
| [SpArch 코드, snns.py](https://github.com/idiap/sparch/blob/main/sparch/models/snns.py) | 있음. `x_flip = x.flip(1)`을 batch 차원에 이어 붙임 | 공유 (같은 layer 통과) | time step | `torch.cat([s_f, s_b], dim=2)` | binary, width 2배 | 옵션일 뿐 결과 없음 (`bidirectional=False`가 [README](https://github.com/idiap/sparch) 예시 값) | read in full text (code) |
| [Vision SmolMamba 2026](https://arxiv.org/abs/2604.25570) | 있음. Eq. 18 `Reverser(P_l)` | "two structured SSM modules" (분리 여부는 명시 없음) | token. "along the token dimension" | 덧셈. Eq. 20 `F_l = (Z_f + Reverser(Z_b)) ⊙ Mask_l`, 이후 `SN(BN(Conv2d(F_l)))` | 덧셈 단계는 0/1/2 (추론), 이후 spike | CIFAR, ImageNet, DVS | read in full text (fetch) |
| [Efficient Spiking Point Mamba 2025](https://arxiv.org/abs/2504.14371) | 있음. Time Flip과 Token Flip을 비교. "we reverse only the time dimension" | 확인 못 함 | time step 선택 | 확인 못 함 | 확인 못 함 | point cloud | read in full text (fusion 수식은 미확인) |
| [Stan & Rhodes 2023](https://arxiv.org/abs/2401.00955) | "both Binary S4D and GSU networks are bidirectional" | 확인 못 함 | time step | 읽은 범위에 기술 없음 | 확인 못 함 | LRA. sMNIST에서는 "Bidirectionality is also disabled" | read in full text (해당 section) |
| [TRR 2024](https://arxiv.org/abs/2408.09108) | 있음. `x̂_t = x_{T+1-t}`. 학습 전용 | 같은 `SNN(·)` 호출 (공유로 읽힘, 명시 없음) | time step | KL consistency loss + penultimate firing rate의 Hadamard 곱 | 곱은 real | CIFAR, DVS, ModelNet | read in full text |

**Reversal 입력이 없거나 축이 시간이 아닌 bidirectional**

- [TEFormer](https://arxiv.org/abs/2601.18274) (arXiv 2601.18274 맞음, 제목 "Structured Bidirectional Temporal Enhancement Modeling in Spiking Transformers", arXiv 기재로 ICML 2026): 두 개의 parallel stream이 아니다. Attention의 TEA가 forward, MLP의 T-MLP가 backward를 맡아 직렬로 이어진다. Eq. 4는 `h_r = LIF(σ(W_fx X_r + W_fh h_{r+1}) ⊙ h_{r+1} + (1-σ(·)) ⊙ W_in X_r)`, r = T-2..0이고 Eq. 5는 `Y_t = LIF(BN(W_o h_t))`이다. 처음 두 block에만 적용한다. 입력을 뒤집어 넣지 않고 recurrence 순서만 역방향이다. `[read in full text]`
- [GSU-DBNet 2026](https://arxiv.org/abs/2606.23761): "a bidirectional BiGSU in the frequency path ... and a unidirectional GSU in the time path for causal temporal modeling". 시간 축은 causal로 두고 bidirectional은 비시간 축에만 쓴 설계다. `[read in full text]`
- [Zou et al., Spiking Spatiotemporal Transformer](https://arxiv.org/abs/2303.09681): "bi-directional spatiotemporal fusion"은 attention의 space-time 정보 흐름을 뜻하고, 내가 읽은 범위에 reversed 입력 stream은 없다. Attention score에 normalized Hamming similarity를 쓴다. `[read in full text, 해당 section]`
- [Parallel Delayed Memory Unit 2025](https://arxiv.org/abs/2512.01626): Bi-PDMU의 Eq. 16은 `h[k] = m[k] + Σ d→ m[i] + Σ d← m[i]`이다. 결과 표에서 Bi-PDMU는 모두 "SNN: No"이므로 spiking 사례가 아니다. `[read in full text]`

**본문을 읽지 못한 bidirectional 후보 (fusion 불명)**

- [Xiao et al. 2022, TASLP](https://doi.org/10.1109/TASLP.2022.3221011): "The core of our model is bi-directional spiking neural network (SNN)". sentiment classification과 machine translation을 다루고 에너지는 0.82 %로 보고한다. `[abstract only]`. [Spike-driven LLM](https://arxiv.org/abs/2604.16475)이 이를 "Bi-SNN (Xiao et al., 2022)"로 인용한다 `[secondary summary]`.
- [Bi-SRNN, EEG 2025](https://doi.org/10.1002/cpe.70343): ALIF 기반 "Bidirectional Spiking Recurrent Neural Network (Bi-SRNN)". Bonn dataset에서 binary 100 %, ternary 99.00 %로 보고한다. `[abstract only]`
- [Closed-Loop Arrhythmia ... Bidirectional SNN, ISCAS 2026](https://doi.org/10.1109/ISCAS66217.2026.11562549): MIT-BIH 5-class 96.44 %, 약 50.3k parameters, 43 µW. 초록에 "bidirectional"이 무엇을 뜻하는지 설명이 없다. 우리 task와 겹치므로 확인이 필요하다. `[abstract only]`
- [SpikeApneaNet 2026](https://doi.org/10.1038/s41598-026-65099-y): ECG, "spike-bidirectional gated recurrent unit (BiGRU)". 본문 fetch는 redirect로 실패했다. `[abstract only]`
- [SpikeVFuse, IEEE TMM 2026](https://doi.org/10.1109/tmm.2026.3729390): "a time-reversal-assisted bidirectional spike-modulation design to model richer cross-step dependencies". "modulation"의 구동 신호가 초록에 없다. 전문 검색에서도 본문을 찾지 못했다. 이 조사에서 가장 먼저 확인해야 할 논문이다. `[abstract only]`

**이름이 같거나 비슷하지만 다른 것 (false friends)**

- [BiSNN, arXiv 2012.08300](https://arxiv.org/abs/2012.08300)은 "Binary Weights" 논문이다. bidirectional이 아니다. `[제목만 확인]`
- [BI-SNN, Sci. Rep. 2021](https://doi.org/10.1038/s41598-021-81805-4)은 "Brain-Inspired" SNN이다. EEG에서 근육 활동을 예측한다. `[abstract only]`
- [SpikeLM](https://arxiv.org/abs/2406.03287)과 [Spike-driven LLM](https://arxiv.org/abs/2604.16475)의 "bidirectional"은 spike 부호 {-1, 0, 1}이나 부호 인코딩을 뜻한다. `[read in full text]`
- [Bidirectional dynamic threshold SNN](https://doi.org/10.3389/fnins.2025.1661916)은 객체 검출용 threshold 뉴런이다. `[abstract only, 앞부분]`. [Bipolar Silicon Synapse](https://doi.org/10.1109/vlsid68508.2026.00017)와 [Spiking Bidirectional Associative Memory](https://doi.org/10.20381/ruor-26444)도 시간 reversal과 무관하다. `[제목만 확인]`
- [BSD, Bidirectional Spike-based Distillation](https://arxiv.org/abs/2509.20284)은 "a feedforward and a backward spiking network"를 함께 학습하는 learning rule이다. layer 방향 얘기이지 시간 reversal이 아니다. `[read in full text, 해당 section]`
- [T-RevSNN](https://arxiv.org/abs/2405.16466)은 "reversible"이다 `[제목만 확인]`.
- [CMQKA](https://arxiv.org/abs/2602.00701), [CIS-Fuse](https://arxiv.org/abs/2607.19879), [SNNSIR](https://arxiv.org/abs/2508.12271), [GSCANet](https://doi.org/10.21203/rs.3.rs-10733322/v1), [DS-RSFNet](https://doi.org/10.1109/iccir70228.2026.11633338)의 bidirectional은 두 modality나 두 view 사이의 양방향 교환이다. `[앞 3편 read in full text, 해당 section; 뒤 2편 abstract only]`
- [SpikeAD](https://arxiv.org/abs/2611.00112)의 bidirectional은 feature pyramid다. `[read in full text]`
- [SpikeSSL](https://arxiv.org/abs/2610.11456)은 칼슘 이미징에서 spike를 추론하는 모델로, 내가 읽은 layer 식은 real 값이다 (Eq. 2 `h = W_out[h_fwd ∥ h_bwd] + b_out`). `[read in full text]`
- [Hybrid Bi-LSTM-SNN, ECG 2026](https://doi.org/10.54554/jtec.2026.18.03.002)은 앞단 Bi-LSTM이 non-spiking이다. `[abstract only]`
- [Combining SNNs with Filtering](https://arxiv.org/abs/2312.15889)의 bidirectional은 출력에 거는 Bessel filter다. `[read in full text]`
- [Bittar & Garner 2024](https://arxiv.org/abs/2212.01187)의 "four bidirectional recurrent LSTM layers"는 교체 대상인 LSTM baseline이다. 스파이킹 layer가 bidirectional인지는 본문에 언급이 없다. `[read in full text]`

**Spiking SSM과 Mamba 중 bidirectional을 안 쓰는 것**: 참고문헌을 빼고 본문에 "bidirectional"이 0회인 논문은 [SpikingSSMs](https://arxiv.org/abs/2408.14909), [SpikMamba](https://arxiv.org/abs/2410.16746), [SpikeMba](https://arxiv.org/abs/2404.01174), [SpikingMamba](https://arxiv.org/abs/2510.04595), [Mamba-Spike](https://arxiv.org/abs/2408.11823), [P-SpikeSSM](https://arxiv.org/abs/2406.02923), [Event-SSM](https://arxiv.org/abs/2404.18508), [Spiking S4 speech enhancement](https://arxiv.org/abs/2309.03641), [Second-Order SpikingSSM](https://arxiv.org/abs/2510.14386), [Spiking Mamba speech](https://arxiv.org/abs/2606.01135)이다. `[PDF 본문 regex]`

### Inferences
- "SNN에 bidirectional을 처음 넣었다"는 주장은 불가능하다. 내가 원문으로 확인한 가장 이른 사례는 Yin 2021이지만, 이것이 최초라는 뜻은 아니다.
- Separate weights + input reversal 구조는 PRF Appendix H가 사실상 같다. SpikMixer의 차별점은 구조가 아니라 fusion에만 있다.
- "spiking Bi-Mamba"라는 이름의 논문은 못 찾았다. 실제 사례는 SmolMamba와 SPM처럼 Vim 방식을 옮긴 것이다.

### Gaps
- 사용자가 제시한 "BiSNN"이 bidirectional을 뜻하는 논문은 Xiao 2022 외에 찾지 못했다. sEMG용 reversed-stream SNN도 못 찾았다.
- "S4-style bidirectional kernels"의 출처: [S4 arXiv 본문](https://arxiv.org/abs/2111.00396)에는 "bidirectional"이 0회였고, PRF는 "concatenates two S4 convolution kernels back-to-back" 문구를 Goel et al. (2022a), Gu et al. (2022b)에 돌린다. 두 논문과 S4 저장소 코드는 읽지 않았다.
- Yin 2021은 TIMIT의 bidirectional 구조를 "similar to" ref. 12 (Bellec 2020)라고 쓰지만, [Bellec 2020 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/)에서 "bidirectional"을 찾지 못했다 (fetch, 전체의 약 93 %까지 읽음, supplement 미포함). [Bellec 2018](https://arxiv.org/abs/1803.09574) 본문에도 0회다.

## 2. 두 stream의 비교(차이, XOR, 불일치, 오차)를 학습된 gate나 feature로 쓰는 spiking 논문이 있는가

### Takeaway
"두 spike tensor의 XOR을 다음 학습 경로의 입력으로 쓴다"는 선행이 있다 (XOResNet 2026, DST 2023). 다만 두 논문의 상황은 같은 방향의 residual branch이고, gate가 아니며, membrane이 아니라 spike 수준이다. Membrane 차이를 forward 연산의 입력으로 쓰는 사례, 특히 forward와 reversed stream 사이의 사례는 찾지 못했다.

### Cited Findings

**가장 가까운 선행: spike 수준 비교**
- [XOResNet (2026-05)](https://arxiv.org/abs/2605.30362): Eq. 10 `MR = f1 ⊕ f2 = f1 + f2 - 2(f1 × f2)`. XOR은 "providing meta-residuals for the backbone branch to specialize in learning residual information that is different from that of the shortcut branch". 효과는 OA ResNet (XOR 없음) 대비 CIFAR-10 110층 88.60 → 90.54, CIFAR-100 50층 61.85 → 66.30, miniImageNet 44층 57.59 → 68.00. 두 branch는 같은 입력의 convolution이고 reversal은 없다. `[read in full text]`
- [DST / Auxiliary Accumulation Pathway (ICML 2023)](https://arxiv.org/abs/2301.11929): 원소별 함수 `g`로 AND, IAND, OR, XOR을 비교한다 (`s ⊕ x = s(1-x) + x(1-s)`). DVS Gesture FSNN-7에서 AND 55.56, OR 96.18, XOR 96.53, IAND 97.57. firing rate는 "OR > XOR > IAND > AND". `[read in full text]`
- [α-XNOR, CVPR 2025](https://doi.org/10.1109/cvpr52734.2025.00512): Query와 Key의 spike 유사도를 XNOR 변형으로 계산한다 `[abstract only]`. [Zou et al.](https://arxiv.org/abs/2303.09681)의 Hamming similarity도 같은 계열이다 `[read in full text]`. 둘 다 attention 내부 점수이지 stream 간 gate가 아니다.

**학습 신호로서의 비교 (forward 연산에 안 들어감)**
- [TRR](https://arxiv.org/abs/2408.09108): 정방향과 역방향 출력의 KL. "We use KL divergence to push the category probabilities of the reversed output to be consistent" (fetch). 여기에 firing rate의 Hadamard 곱을 더하지만 "During inference, our SNN behaves like vanilla SNNs". 일치를 강제하는 쪽이므로 SpikMixer와 방향이 반대다. `[read in full text]`
- [BSD](https://arxiv.org/abs/2509.20284): forward와 backward 네트워크의 voltage를 ReCo loss로 정렬한다 ("spurious correlations between non-corresponding voltage pairs"). 두 네트워크의 membrane을 비교하는 드문 예지만 loss이고, 시간 reversal이 아니다. `[read in full text, 해당 section]`
- [Membrane Potential Smoothing, ICLR 2025](https://arxiv.org/abs/2502.14218): "if the difference in membrane potentials across timesteps is too large, it will lead to unstable outputs". 같은 뉴런의 시간 간 차이이고, 줄여야 할 대상으로 다룬다. `[read in full text]`
- [Urbanczik & Senn 2014](https://doi.org/10.1016/j.neuron.2013.11.030): "the rule seeks to minimize discrepancies between somatic firings and a local dendritic potential". plasticity 신호다. `[abstract only]`
- [Dendritic predictive coding](https://arxiv.org/abs/2205.05303)은 제목만 확인했다. 내가 찾은 다른 spiking 오차 연구 ([error-modulated plasticity](https://arxiv.org/abs/2510.14382))도 제목상 plasticity 쪽이다. `[제목만 확인]`

**논리 연산 fusion과 residual**
- [SEW ResNet, Fang 2021](https://arxiv.org/abs/2102.04159): ADD, AND, IAND `(1-A)·S`. "using AND and IAND as g will output spikes (i.e. binary tensors)". IAND residual의 정의가 여기 있다. `[read in full text]`
- OR: [SynA-ResNet](https://arxiv.org/abs/2311.06570)이 "OR Residual Connection"을 제안한다 `[abstract only]`. [XOResNet](https://arxiv.org/abs/2605.30362)은 OR이 non-identity mapping에서 정보 손실을 낸다고 비판하고 OR-ADD를 쓴다 `[read in full text]`. [Multi-LRA](https://doi.org/10.1016/j.ins.2024.120136)는 logical residual을 제목에 걸지만 초록이 없었다 `[제목만 확인]`.
- Membrane 수준: [MS-ResNet](https://arxiv.org/abs/2112.08954). [Spike-driven Transformer](https://arxiv.org/abs/2307.01694)는 "creating a shortcut between membrane potential of spiking neurons in various layers"로 설명하고, "Only binary spikes can support the spike-driven function"이라는 입장을 취한다. Membrane을 모듈 사이로 나르는 선행은 덧셈뿐이었다. `[read in full text]`

**Spiking gate (전부 구동 신호가 차이가 아님)**

| 논문 | Gate 구동 신호 | Gate 형태 | 등급 |
|---|---|---|---|
| [SpikeAD 2026](https://arxiv.org/abs/2611.00112) | 한 stream의 공간 합 vs 학습 threshold. Eq. 14 `m = 1[Σ S_MS ≥ σ(θ)·K·H'W']` | binary. "pass-or-suppress". 선택할 다른 branch 없음. "multiplexer control signals"는 shift-add 하드웨어 표현 | read in full text |
| [TEFormer](https://arxiv.org/abs/2601.18274) | 현재 입력 + 다음 hidden state | sigmoid, real | read in full text |
| [Stan & Rhodes, GSU](https://arxiv.org/abs/2401.00955) | GLU처럼 두 stream의 Hadamard 곱 | real 혼합 | read in full text |
| [GSCANet 2026](https://doi.org/10.21203/rs.3.rs-10733322/v1) | "Q-only, V-only, and JointQV". G4가 no-gate 대비 +2.15, +1.05 %p | head별 MLP gate | abstract only |
| [CIS-Fuse 2026](https://arxiv.org/abs/2607.19879) | 보조 modality. "Conv-BN-Sigmoid branch", membrane에 전류로 주입 | real | read in full text, 해당 section |
| [SpikeSTAG 2025](https://arxiv.org/abs/2508.02069) | LSTM과 SSA 출력에 대한 "learnable, parameterized gating function" (시계열 예측) | real | read in full text |
| [GLIF](https://arxiv.org/abs/2210.13768), [Dynamic Gated Neuron](https://arxiv.org/abs/2509.03281) | 뉴런 내부 dynamics | 뉴런 모델 | read in full text, 앞부분 |
| [SGSAFormer](https://doi.org/10.3390/electronics14010043) | Spike Gated Linear Unit | 미확인 | abstract only |
| [Spiking ConvNeXt 2026](https://www.sciencedirect.com/science/article/pii/S2666827026001738) | 학습된 convex combination. TTFS에서는 "element-wise minimum" | 미확인 | secondary summary |

- [DH-SNN](https://doi.org/10.1038/s41467-023-44614-z)의 "temporal spiking XOR problem"은 fusion이 아니라 benchmark task다. `[abstract only]`

### Inferences
- `{0, 1}` 위에서는 `|a-b| = a XOR b`, `min(a,b) = a AND b`, `max(a,b) = a OR b`다. 그래서 spike 수준으로 내려가면 s_d의 입력은 XOResNet의 XOR과 대수적으로 같다. 심사자가 "XOR residual의 변형"이라고 볼 가능성이 높다.
- 방어 가능한 차이는 네 가지다. (a) 입력이 spike가 아니라 membrane이어서, 양쪽 다 발화하지 않아도 잠재적 불일치를 볼 수 있다. (b) 그 값이 BN과 LIF를 지나므로 token 축 상태를 가진다. (c) 두 값이 같은 입력의 forward와 reversed라서 차이가 곧 시간 비대칭에 해당한다. (d) 결과가 feature가 아니라 두 경로 선택(MUX)에 쓰인다.
- (a)가 실제로 효과를 내는지는 데이터가 없다. Spike XOR로 구동한 같은 구조를 대조군으로 돌려야 이 주장이 선다 (not run).

### Gaps
- [Multi-LRA](https://doi.org/10.1016/j.ins.2024.120136) 본문을 못 읽었다. "Multi logical" 조합에 XOR이나 stream 비교가 있는지 불명이다.
- Spiking stereo와 Siamese 계열은 두 입력의 차이를 쓸 만한 자리다. [SiamSNN](https://arxiv.org/abs/2003.07584), [Siamese SNN with EMD](https://arxiv.org/abs/2203.13207), [StereoSpike](https://arxiv.org/abs/2109.13751), [SpikeStereoNet](https://arxiv.org/abs/2505.19487)는 제목이나 검색 요약만 봤다. 이 계열의 feature 수준 차이는 미확인이다. Spiking change detection은 arXiv 검색 0건이고, OpenAlex 검색은 한도 초과로 실패했다.

## 3. 심사자가 인용할 만한 non-spiking 유사 사례

### Takeaway
`|u-v|` feature와 fusion gate는 각각 오래된 표준이다. 그러나 **BiRNN의 forward와 backward 사이에서** `|f-b|`를 gate나 feature로 쓴 논문은 찾지 못했고, 가까운 것은 loss 쪽이다 (BRITS, UnFlow). 이 영역은 문헌이 방대하므로 부재 근거는 약하다.

### Cited Findings
- **Merge 표준**: [Keras `Bidirectional`](https://keras.io/api/layers/recurrent_layers/bidirectional/)의 merge_mode는 `{"sum", "mul", "concat", "ave", None}`, 기본 `"concat"`. 차이 모드가 없다. `[read in full text]`
- **두 표현의 절댓값 차이**: [InferSent](https://arxiv.org/abs/1705.02364): `(u, v, |u-v|, u*v)`, "(iii) absolute element-wise difference |u − v|". u와 v는 **서로 다른 두 문장**의 embedding이고 gate가 아니라 feature다. 그 뿌리인 [Mou 2016](https://arxiv.org/abs/1512.08422): "concatenation, element-wise product and difference". [ESIM](https://arxiv.org/abs/1609.06038): 정렬된 쌍의 차이와 곱을 "capture inference relationships such as contradiction" 용도로 쓰고, 제거 시 88.0 → 87.0 %. `[read in full text]`
- **Forward-backward 일치를 loss로**: [BRITS](https://arxiv.org/abs/1805.10572): "consistency loss" `Discrepancy(x̂_t, x̂'_t)` (MSE). [UnFlow](https://arxiv.org/abs/1711.07837): "forward-backward consistency ... for estimating occlusion maps, which mask the differences in the data loss". UnFlow는 forward-backward 불일치를 threshold로 잘라 binary mask를 만들므로, 개념상 가장 닮은 사례다. 다만 mask가 걸리는 곳은 feature가 아니라 loss다. `[read in full text]`
- **Fusion gate 산술**: [DiSAN](https://arxiv.org/abs/1709.04696): `F = sigmoid(W1 s + W2 h + b)`, `u = F⊙h + (1-F)⊙s`. forward/backward 사이가 아니라 attention 출력과 입력 사이다. 산술 형태는 `s_d*L + (1-s_d)*min(F,R)`와 같다. `[read in full text]`
- **Bidirectional Mamba**: [Vim](https://arxiv.org/abs/2401.09417): "The y_forward and y_backward are then gated by the z and added together". z는 입력에서 나오고 두 방향의 차이가 아니다. [Bi-Mamba+](https://arxiv.org/abs/2404.15772)는 방향별 block을 두고 합친다. `[read in full text]`
- **원형**: [Graves & Schmidhuber 2005](https://doi.org/10.1016/j.neunet.2005.06.042): "bidirectional networks outperform unidirectional ones". `[abstract only]`

### Inferences
- 심사자가 가장 쉽게 꺼낼 조합은 "InferSent식 `|u-v|` + GRU식 gate"다. 그래서 "차이로 gate하는 것 자체"를 새롭다고 주장하면 안 된다. 주장 범위는 SNN, membrane, 시간 역방향 stream, binary 선택까지 좁혀야 한다.

### Gaps
- Span 표현의 "LSTM-minus" 계열은 같은 방향 안의 두 위치 차이로 기억하지만, 이번에 원문을 확인하지 않았다. [Schuster & Paliwal 1997](https://doi.org/10.1109/78.650093)도 참고문헌으로만 봤다.
- BiRNN 불일치 gate는 arXiv 초록 검색 0건, web 검색 2회로 확인한 것이 전부다. 음성, 이상 탐지, 의료 시계열 저널까지 포괄하지 못했다.

## 4. |H_f - H_r| gate는 새로운가: 검색 기록과 부재 근거의 강도

### Takeaway
**찾은 범위에서는 선행이 없다.** "SNN에서 forward와 reversed stream의 membrane 절댓값 차이를 spike gate로 만들어 fusion을 선택한다"는 논문은 초록 수준에서 없다고 꽤 강하게 말할 수 있고, 본문 수준에서는 중간 강도다. 초록만 읽은 bidirectional 후보 5편, 초록도 못 본 Multi-LRA, 유료 저널 본문이 남은 위험이다. 증명이 아니므로 "To the best of our knowledge"로 써야 한다.

### Cited Findings

**검색 기록**

- **OpenAlex `title_and_abstract.search` (약 39건)**. 핵심 hit 수:
  - `spiking neural AND "absolute difference" AND gate` 0
  - `spiking neural AND "membrane potential difference" AND fusion` 0
  - `spiking neural AND "membrane difference"` 0
  - `"spiking XOR" AND fusion` 0
  - `spiking neural AND XOR AND fusion AND stream` 0
  - `spiking neural AND disagreement AND fusion` 0
  - `spiking neural AND "mismatch" AND gating AND stream` 0
  - `spiking neural AND "logical operation" AND fusion` 0
  - `spiking neural AND "cross-stream"` 0
  - `spiking neural AND multiplexer AND gate AND learned` 1 (무관)
  - `"spike XOR"` 3 (task 또는 응답 표면)
  - `spiking neural AND "IAND"` 2, `spiking neural AND "OR residual"` 1
  - `"bidirectional spiking"` 33, `"bi-directional spiking"` 4, `"bidirectional SNN"` 2, `"spiking Bi-LSTM"` 0, `"bidirectional SRNN"` 0, `spiking AND "Bi-Mamba"` 0, `"spiking" AND "bidirectional kernel"` 0
  - 일반 `search` 23건도 수행했다.
- **OpenAlex 2차 27건은 실패** (429, `retry-after: 31824`, 일일 한도 소진). Siamese, change detection, stereo difference, discrepancy, inconsistency, conflict, forward-backward discrepancy가 여기 있었다. 일부만 arXiv로 대체했다.
- **arXiv API (title, abstract, 65건)**. 예:
  - `abs:spiking AND abs:"disagreement"` 10 (무관)
  - `abs:spiking AND abs:"discrepancy" AND abs:fusion` 1 (무관)
  - `abs:spiking AND abs:"membrane" AND abs:"difference" AND abs:"gate"` 5 (관련은 GLIF 정도)
  - `abs:spiking AND abs:XOR AND abs:residual` 1 (XOResNet)
  - `abs:spiking AND abs:"gating" AND abs:"difference" AND abs:"branches"` 0
  - `abs:bidirectional AND abs:"forward and backward" AND abs:disagreement` 0
  - `abs:"forward and backward" AND abs:difference AND abs:LSTM AND abs:fusion` 0
  - `abs:spiking AND abs:"temporal reversal"` 2 (TRR, T-RevSNN)
  - `abs:spiking AND abs:"change detection" AND abs:"remote sensing"` 0
- **Europe PMC 전문 검색 (5건, 제목만 선별)**: `"spiking neural network" AND "membrane potential" AND "absolute difference"` 24, `... "bidirectional" AND "reversed"` 43, `... "XOR" AND "fusion"` 16, `... "non-causal"` 16, `... "disagreement" AND "bidirectional"` 8. 관련 논문은 없었다. 처음 18건은 503으로 실패했다.
- **Web 검색 약 12회**: "spiking gate absolute difference membrane two streams disagreement", "spiking XOR of spike streams fusion mismatch", "BiLSTM |f - b| fusion gate disagreement", "forward-backward discrepancy gating fusion", "spiking gate driven by disagreement between two pathways", "dual stream membrane potential difference stereo" 등. 전부 일치 논문 없음. [XOResNet](https://arxiv.org/abs/2605.30362)은 여기서 처음 나왔다.
- **Semantic Scholar**: 검색과 batch 모두 429 또는 timeout. 결과 없음. DBLP, OpenReview는 차단이라 쓰지 않았다.
- **본문 regex (내가 받은 arXiv PDF 66편과 Yin 2021, spiking 관련 약 55편, 참고문헌 제외)**:
  - "absolute difference": spiking 논문 0회. 비교군 InferSent, Mou에서만 등장.
  - "membrane (potential) difference" 계열: [Membrane Smoothing](https://arxiv.org/abs/2502.14218) 10회 (시간 간 차이), PRF 1회 (증명).
  - "disagree|discrepanc": spiking 논문의 hit를 문맥으로 전부 확인했고 stream 간 사용은 없음.
  - "XOR|exclusive-or|XNOR": DST 5회, XOResNet 33회, SpikingMamba 1회 (하드웨어 부호 비트).

**근거의 강도**

| 범위 | 강도 | 이유 |
|---|---|---|
| 제목, 초록에 이 기법을 명시한 SNN 논문 | 비교적 강함 | 세 색인에서 여러 표현으로 0건 |
| 본문 어딘가에 이 기법이 있는 SNN 논문 | 중간 | 약 55편 regex + bidirectional 그룹 정독. 수식으로만 쓰인 경우 regex가 놓친다 |
| 유료 저널 본문 (IEEE, Elsevier, Wiley, ACM) | 약함 | Xiao 2022, Bi-SRNN, ISCAS 2026, SpikeVFuse, Multi-LRA 본문 미확인. SpikeApneaNet은 오픈 액세스지만 fetch 실패 |
| GitHub 코드 | 없음 | 코드 검색 안 함 (SpArch만 직접 읽음) |
| 특허 | 없음 | 검색 안 함 |

### Inferences
- 주장 가능한 문장: "To the best of our knowledge, no prior spiking network selects between fusion paths with a spiking gate driven by the membrane disagreement of its forward and reversed streams."
- 주장하면 안 되는 문장: 최초의 bidirectional SNN, 최초의 gated fusion SNN, 최초의 XOR 또는 차이 기반 SNN.
- 문헌이 빠르게 움직인다. 가까운 논문 다수가 2026년 것이다 (TEFormer 1월, SmolMamba 4월, XOResNet 5월, SpikeVFuse). 제출 직전에 다시 검색해야 한다.

### Gaps
- 투고 전 우선 확인 순서: SpikeVFuse, ISCAS 2026 (MIT-BIH), SpikeApneaNet (ECG, 브라우저로 열림 가능), Xiao 2022, Multi-LRA, Bi-SRNN.
- OpenAlex 2차 27개 질의를 한도 복구 후 다시 돌려야 한다. Semantic Scholar, Google Scholar도 미수행이다.

## 5. Neuroscience 근거, 그리고 non-causal 비판

### Takeaway
Forward and reverse replay는 "양방향이 뇌에도 있다"는 동기로만 쓸 수 있고, 불일치 비교의 근거로는 약하다. 비판은 명시적으로 있다. SiLIF가 bidirectional 모델을 buffer와 연산량 문제로 공정 비교에서 제외했고, TEFormer는 스스로 한계로 적었다. SpikMixer도 전체 sequence가 있어야 돌아가므로 같은 비판을 받는다.

### Cited Findings

**Replay**
- [Foster & Wilson 2006](https://doi.org/10.1038/nature04587): "recent episodes of spatial experience are replayed in a temporally reversed order", awake period에서 경험 직후. `[abstract only]`
- [Diba & Buzsáki 2007](https://doi.org/10.1038/nn1961): "recurred in reverse order at the end of a run, but in forward order in anticipation of the run"; "bidirectional re-enactment". `[abstract only]`
- [Ambrose et al. 2016](https://doi.org/10.1016/j.neuron.2016.07.047): "only reverse replays increased their rate at increased reward ... while forward replays were unchanged"; "a functional distinction between different directions of replay". `[abstract only]`
- [Haga & Fukai 2018](https://doi.org/10.7554/eLife.34171): 같은 CA3 recurrent synapse의 symmetric STDP로 forward와 reverse replay를 모두 설명한다. `[abstract only]`
- [Foster 2017](https://doi.org/10.1146/annurev-neuro-072116-031538): 새 결과가 "necessitating revision of the original interpretations". `[abstract only]`

**불일치와 일치 검출**
- [Keller & Mrsic-Flogel 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6400266/): "Prediction error neurons project upwards in the hierarchy and encode a difference between prediction and bottom-up input". 부호가 있는 두 유형 (S-P, P-S)을 구분하고, "We have referred to this type of negative prediction error as a mismatch response" (fetch). `[read in full text]`
- [Larkum 2013](https://doi.org/10.1016/j.tins.2012.11.006): "robustly triggered by coincident input to opposite poles of the neuron". AND에 가까운 근거다. `[abstract only]`
- [Gidon et al. 2020](https://doi.org/10.1126/science.aax6239): dCaAP의 진폭은 "maximal for threshold-level stimuli but dampened for stronger stimuli"이고, 뉴런이 "classify linearly nonseparable inputs"할 수 있다. `[abstract only]`
- [Borst, Haag, Mauss 2020](https://doi.org/10.1007/s00359-019-01375-9): T4/T5 수상돌기는 "an enhancement of signals for motion along their preferred direction as well as a suppression of signals for motion along the opposite direction". 순서 방향을 비교하는 가장 가까운 회로지만, causal하게 온라인으로 작동한다. `[abstract only]`
- [Costa, Goldberger, Peng 2005](https://doi.org/10.1103/PhysRevLett.95.198102): heartbeat의 time asymmetry index는 "decreases with aging or heart disease". `[abstract only]`

**Non-causal 비판과 대응**
- [SiLIF](https://arxiv.org/abs/2506.06374), 가장 직접적인 비판: "the use of bidirectional models prevents streamlined data processing, timestep by timestep"; "the whole sequence needs to be buffered ... and the computation on the sequence needs to be repeated for each prediction"; "This leads to 100 times more compute for GSC at 10 ms resolution, doubled further due to the two paths of bidirectional models". 그래서 SiLIF는 Ding et al. (2025, Keyword Mamba)의 정확도가 "not comparable to our SNN models"라고 하고 forward mode로 재학습한다. Ding 논문 자체는 읽지 않았다. `[read in full text]`
- [TEFormer](https://arxiv.org/abs/2601.18274): "Since T-MLP uses backward temporal fusion, TEFormer targets window-based recognition rather than strictly causal online inference". `[read in full text]`
- [PDMU](https://arxiv.org/abs/2512.01626): Bi-PDMU는 "Choose for offline/bidirectional settings". `[read in full text]`
- [Spikachu](https://arxiv.org/abs/2510.20683): "to enable online operation, models must be causal"; SNN의 "inherent causality supports integration into online systems". `[read in full text]`
- [Combining SNNs with Filtering](https://arxiv.org/abs/2312.15889): bidirectional filtering은 "generally applicable to offline filtering since the whole waveform is needed before processing begins"이고, 타협안으로 sliding block을 쓰면 "only a latency penalty of half block size"다. `[read in full text]`
- [GSU-DBNet](https://arxiv.org/abs/2606.23761): bidirectional은 frequency 축에만, 시간 축은 causal. `[read in full text]`

### Inferences
- **Replay 비유의 한계 네 가지.** (1) Replay는 입력이 없는 시점의 재현이고 SpikMixer는 입력 처리 중이다. (2) 두 방향은 시점이 다르며 (run 전과 후), 같은 순간에 비교한다는 근거를 찾지 못했다. (3) Replay는 같은 synapse에서 일어나는데 (Haga & Fukai) 우리는 separate weights다. (4) Ambrose의 기능 분리는 "둘이 다르다"는 근거일 뿐 "차이를 계산한다"는 근거가 아니다. 논문에서는 "inspired by" 수준으로만 쓰는 편이 안전하다.
- **Prediction error 비유도 맞지 않는다.** 두 stream은 모두 같은 입력의 bottom-up 처리이고 어느 쪽도 다른 쪽의 예측이 아니다. 또 Keller의 오차는 부호가 있는데 우리는 절댓값이다. "mismatch" 대신 "disagreement"를 권한다.
- 더 정직한 해석은 "시간 비대칭 검출"이다. `|H_f - H_r|`는 순서에 민감한 두 적분기의 차이이므로 국소적 time irreversibility의 학습된 대리 신호로 읽을 수 있다. Costa 2005가 생리학적 의미를 준다. 다만 이것은 가설이고, MIT-BIH +0.00 %p가 "ECG니까 효과가 있다"는 단순한 설명을 반박한다.
- **Task별 causal 부담은 다르다.**
  - CinC, MIT-BIH: 창 하나당 예측 하나이므로 SiLIF의 "100배"는 해당 없고 연산 2배 + 창 전체 지연이 비용이다.
  - ETT 예측: look-back이 전부 과거 값이라 미래 정보 누출은 없다. 다만 새 예측마다 처음부터 다시 돌아야 하므로 streaming 상태 재사용이 안 된다.
  - Event-driven 하드웨어: reversed stream은 도착 순서대로 처리할 수 없다. 코드상 `H_r`를 뒤집어 정렬하므로 입력과 membrane의 N token 분량 buffer가 필요하다.
- 완화안 후보 (not run): sliding block 방식의 Mixer, 또는 GSU-DBNet처럼 bidirectional을 비시간 축으로 옮기기. 다만 후자는 T = N 구조의 의미를 바꾼다.

### Gaps
- Replay 논문은 초록 위주로 읽었다. 같은 ripple 안에서 forward와 reverse가 공존하는지, 어떤 하류 회로가 둘을 비교하는지는 조사하지 못했다.
- Gidon 2020이 XOR을 언급하는 것은 본문 사항으로 기억하지만 이번에 확인하지 않았다. 초록 표현만 인용할 것.
- Latency와 buffer 비용을 실제로 재서 보고한 bidirectional SNN은 SiLIF의 추정 외에 찾지 못했다.

## 6. Bidirectional의 보고된 이득과 부정적 결과

### Takeaway
Spiking에서 같은 구조로 방향만 바꿔 비교한 수치는 드물고 (SPM, TEFormer), 이득은 약 +0.15~1.3 %p로 작다. 학습 전용인 TRR은 neuromorphic dataset에서 +3.6~3.8 %p로 더 크지만, 추론은 단방향이다. 부정적 결과도 분명히 있다. 양방향을 다 쓰면 오히려 이득이 사라지거나 (SPM), non-spiking에서 나빠지는 경우가 있다 (Bi-PDMU cough).

### Cited Findings

| 연구 | 비교 | 수치 | 주의 | 등급 |
|---|---|---|---|---|
| [SPM](https://arxiv.org/abs/2504.14371) Table 5 | 없음 → Time Flip | OBJ-BG 88.9 → 90.2, PB-T50-RS 83.5 → 84.2, ModelNet40 91.6 → 92.3 | Token Flip만: 89.6, 83.7, 91.7. **둘 다: 89.2, 83.4, 91.6** | read in full text |
| [TEFormer](https://arxiv.org/abs/2601.18274) Table 6 (CIFAR10) | TEA/T-MLP 방향 | →← 96.24, ←← 96.09, →→ 95.95, ←→ 96.06 | 표에 표준편차 없음. 양방향인 ←→가 단방향 ←←보다 낮음 | read in full text |
| [TEFormer](https://arxiv.org/abs/2601.18274) Table 3 | vs QKFormer | SHD 88.56 → 90.19, sMNIST 94.25 → 96.20, sCIFAR 80.20 → 80.94, HMDB51-DVS 63.53 → 63.65, UCF101-DVS 61.31 → 63.16 | 구조 차이가 섞임 | read in full text |
| [TRR](https://arxiv.org/abs/2408.09108) Table 1 (학습 전용) | baseline → TRR | CIFAR10 93.67 → 94.45, CIFAR100 73.39 → 74.85, CIFAR10-DVS 73.97 → 77.60, DVS-Gesture 87.85 → 91.67, ModelNet10 92.38 → 93.45, ModelNet40 87.35 → 88.84 | inference는 단방향. 정방향 stream 하나로 추론 | read in full text |
| [Yin 2021](https://arxiv.org/abs/2103.12593) | TIMIT | Bi-SRNN 66.1, LSNN 65.4, Bi-LSTM 68.9 | 단방향 SRNN 없음. ECG에서는 단방향 SRNN 85.9 > Bi-LSTM 80.8 | read in full text |
| [PDMU](https://arxiv.org/abs/2512.01626) (**non-spiking**) | PDMU → Bi-PDMU | SHD 84.57 → 84.94, GSC 96.93 → 96.96, PSMNIST 98.22 → 98.25, WithMe 84.17/74.50 → 84.82/77.13 | **COUGHVID 82.45 → 79.63 (Acc), 81.58 → 78.02 (AUC), 82.21 → 78.23 (F1)**. SHD와 GSC에서 단방향 EPDMU가 더 높음 (86.48, 97.01) | read in full text |
| [Bi-Mamba+](https://arxiv.org/abs/2404.15772) (**non-spiking**, 시계열 예측) | backward 제거 | MSE 3.25 %, MAE 2.44 % 악화 | 데이터셋 평균 | read in full text |
| [SpikeSSL](https://arxiv.org/abs/2610.11456) (spike 추론 모델) | bidirectional IIR 제거 | Corr 0.660 → 0.597 | 내가 읽은 layer는 real 값 | read in full text |
| [Hybrid Bi-LSTM-SNN](https://doi.org/10.54554/jtec.2026.18.03.002) | vs 단독 SNN | test 96.88 %, +5.21 %p | Bi-LSTM이 non-spiking이라 방향 효과가 아님 | abstract only |
| [GSU-DBNet](https://arxiv.org/abs/2606.23761) | frequency path (bidirectional) 제거 | "significantly degrades PESQ", time path 제거보다 큼 | 축과 방향이 함께 빠짐, 수치 미추출 | read in full text |
| [Bi-SRNN](https://doi.org/10.1002/cpe.70343) | 단독 보고 | Bonn 100 %, 99.00 % | 단방향 비교는 초록에 없음 | abstract only |

- 방향 비교 수치가 없는 논문: [PRF](https://arxiv.org/abs/2410.03530) (Table 5는 Spike-Driven TCM vs S4(Bidirectional): Image 84.77 vs 88.65, Pathfinder 91.76 vs 94.20, 평균 82.39 vs 84.03이고 단방향 비교가 아님), [Stan & Rhodes](https://arxiv.org/abs/2401.00955), [SmolMamba](https://arxiv.org/abs/2604.25570) (ablation에 방향 항목 없음, fetch). `[read in full text]`

**부정적 결과 모음**
- [SPM](https://arxiv.org/abs/2504.14371): "applying both strategies degrades performance to that of unidirectional SSM"; 저자의 추측은 "likely due to the limited information in the spike input to SSM". `[read in full text]`
- [SiLIF](https://arxiv.org/abs/2506.06374): forward mode로 재학습한 Mamba가 "consistently dominated in the accuracy-efficiency trade-off". 다만 이것은 Mamba 구조에 대한 판단이고 bidirectional 자체의 평가는 아니다. `[read in full text]`
- [Bi-PDMU](https://arxiv.org/abs/2512.01626) COUGHVID (위 표). `[read in full text]`

### Inferences
- SPM의 "방향을 더 넣으면 이득이 사라진다"는 관찰은 ETT +0.57 %, MIT-BIH +0.00 %p와 분위기가 같다. Spike 입력에 정보가 적다는 설명이 우리 결과에도 맞는지는 시험된 적 없다.
- 문헌의 이득 크기(+0.15~1.3 %p)에 비추면 CinC +1.56 %p는 작지 않다. 그러나 이 값은 **gate 유무** 비교이고 위 수치는 **방향 유무** 비교이므로 직접 비교할 수 없다. 또 세 task 중 한 곳에서만 나온 효과다.

### Gaps
- SPM 본문은 Token Flip의 차이를 "0.4%, 0.4%, and 0.3%"라고 쓰지만, 표에서 추출한 값으로 계산하면 0.6, 0.5, 0.6이다. 추출 오류인지 논문의 불일치인지 그림 대조가 필요하다.
- ECG에서 reversed-stream SNN의 방향 효과 수치는 찾지 못했다 (ISCAS 2026과 SpikeApneaNet 본문을 못 읽음).

## 7. SpikMixer와의 겹침 분석

### Takeaway
**새롭지 않은 부품**: reversed stream, separate weights, OR, AND, 학습 산술의 gate 형태. **찾지 못한 부분**: membrane 절댓값 차이를 LIF에 넣어 spike gate를 만들고, 그 gate로 "학습 경로 vs AND"를 선택하는 조합. Novelty 주장은 이 조합에 한정해야 한다.

### Cited Findings

| SpikMixer 부품 | 가장 가까운 선행 | 겹침 | 차이 |
|---|---|---|---|
| Reversed stream + separate weights | [PRF App. H](https://arxiv.org/abs/2410.03530), [Yin 2021](https://arxiv.org/abs/2103.12593) | 사실상 동일 | 없음 |
| Token 축 양방향 | [SmolMamba](https://arxiv.org/abs/2604.25570) "along the token dimension", [SPM](https://arxiv.org/abs/2504.14371) Token Flip | 부분 | 그쪽은 SSM scan, 우리는 Linear-BN-LIF. SPM은 token flip이 "not meaningful"이라 함 |
| F OR R | 방향 fusion에서는 못 찾음. residual로는 [SynA-ResNet](https://arxiv.org/abs/2311.06570), [XOResNet](https://arxiv.org/abs/2605.30362) | 연산은 표준 | 적용 위치만 다름 |
| F AND R (`min`) | [SEW](https://arxiv.org/abs/2102.04159), [DST](https://arxiv.org/abs/2301.11929) 논리 residual. [TRR](https://arxiv.org/abs/2408.09108) rate 곱 | 연산은 표준 | 방향 사이 AND는 못 찾음. DST에서 AND가 가장 나빴음 (55.56) |
| 차이 입력 | [XOResNet](https://arxiv.org/abs/2605.30362), [DST](https://arxiv.org/abs/2301.11929) (spike XOR). [InferSent](https://arxiv.org/abs/1705.02364) (`|u-v|`, non-spiking) | 개념 겹침 | 우리는 membrane, reversed 쌍 |
| s_d = LIF(BN(·)) | [SpikeAD](https://arxiv.org/abs/2611.00112) binary gate | 형태 겹침 | SpikeAD는 한 stream의 활동량과 threshold 비교. 상태 없음 |
| MUX 2경로 선택 | [DiSAN](https://arxiv.org/abs/1709.04696) fusion gate (real). [Spiking ConvNeXt](https://www.sciencedirect.com/science/article/pii/S2666827026001738) `min` (secondary summary) | 산술 겹침 | Spike로 구동되는 2경로 선택은 못 찾음 |
| 전체 조합 | 없음 | | [SpikeVFuse](https://doi.org/10.1109/tmm.2026.3729390) 미확인 |

**각 근접 논문이 왜 같지 않은가**
- [TEFormer](https://arxiv.org/abs/2601.18274): 직렬 구조, 입력 reversal 없음, gate는 sigmoid이고 차이를 보지 않는다. 이름 때문에 가장 먼저 지목될 수 있으나 구조는 멀다.
- [XOResNet](https://arxiv.org/abs/2605.30362): 비교 신호라는 점에서 가장 가깝다. 그러나 시간 방향이 없고, spike 수준이며, gate가 아니라 다음 layer의 입력이다.
- [PRF](https://arxiv.org/abs/2410.03530): 구조상 가장 가깝지만 fusion은 concat + linear이고, 우리의 `sdlin`보다도 단순하다.
- [TRR](https://arxiv.org/abs/2408.09108): reversal을 쓰지만 학습 전용이며 불일치를 줄이려 한다. 우리는 추론에서 불일치를 신호로 쓴다.
- [UnFlow](https://arxiv.org/abs/1711.07837): 불일치 threshold → binary mask라는 발상이 가장 닮았다. 그러나 non-spiking이고 mask 대상이 loss다.

### Inferences
- **예상되는 심사 지적 다섯 가지.**
  1. "XOR residual과 InferSent의 합성일 뿐이다." 대응: membrane 입력의 효과를 spike XOR 대조로 보여야 한다 (not run).
  2. "효과가 일반적이지 않다." CinC +1.56 %p, MIT-BIH 0, ETT 악화. 부정할 수 없으므로 그대로 보고하고 조건을 밝혀야 한다.
  3. "Non-causal이라 SNN의 장점을 버렸다." SiLIF 인용 예상. Task별 부담을 나눠 설명하되 지연을 수치로 내야 한다.
  4. "Spike-driven이 아니다." `|H_f - H_r|`는 real 값이고 BN을 지난다. 코드 주석은 같은 real 값을 linear에 넣는 `linmx`를 MAC로 표기하고, `sdlin`의 W만 "0/1 input (AC)"로 표기한다. 따라서 `s_d` 앞단의 BN은 AC가 아니다. [Spike-driven Transformer](https://arxiv.org/abs/2307.01694)의 "Only binary spikes can support the spike-driven function" 기준으로는 지적받을 수 있다. 에너지 계산에 이 경로를 포함해야 한다.
  5. "Multiplexer" 용어 충돌. [SpikeAD](https://arxiv.org/abs/2611.00112)가 하드웨어 의미로 쓴다. 논문에서 "spiking disagreement gate" 같은 이름이 더 안전하다.
- **코드에서 본 세부 사항 (논문 기술 때 주의).**
  - `H - theta`의 차이는 두 stream의 threshold가 같으면 `theta`가 상쇄되어 `|H_f - H_r|`가 된다. Threshold가 학습되거나 stream별로 다르면 그렇지 않다. `neo_mixer.py`는 이번 범위 밖이라 확인하지 않았다.
  - `L`의 입력에 `s_d`가 들어가지만, `L`이 선택되는 token에서는 `s_d = 1`로 고정이다. 그래도 `L`의 neuron은 모든 token에서 상태를 쌓으므로 `s_d`의 영향은 membrane을 통해 남는다. 순수한 조합 논리 MUX가 아니라 상태가 있는 선택기라고 써야 정확하다.
- 대조 실험 후보 (모두 not run): spike XOR gate, 부호 있는 차이 `H_f - H_r`, weights 공유, sliding block.

### Gaps
- Token 축 T = N spiking mixer 자체의 선행은 이 노트 범위 밖이다.
- 위 "못 찾음" 항목은 모두 4절의 검색 범위에 한정된다.
