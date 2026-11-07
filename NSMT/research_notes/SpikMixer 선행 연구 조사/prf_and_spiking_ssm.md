# PRF와 가장 가까운 spiking state-space / neuron-as-token-mixer 모델 원문 검증 조사

조사일 2026-11-08. 근거 표시: [read in full text] 원문 해당 부분을 직접 읽음, [read in code] 공개 코드를 직접 읽음(실행은 안 함), [abstract only], [secondary summary], [grep only] 원문 키워드 검색만 함. 내 계산은 Inferences에 넣고 [computed]로 표시. PRF arXiv v2 PDF 28쪽은 전부 읽었고, Pattern Recognition 게재본은 못 읽었다(ScienceDirect 403).

## PRF neuron: 식, resonate, decoupled reset, 복잡도, inference 연산

### Takeaway
PRF는 spike로 되먹이는 reset이 없는 neuron이다. 막전위는 입력의 선형 convolution(복소수 kernel 하나)이고 그 실수부에 threshold만 건다. 학습은 FFT로 하고 inference는 실수 state 2개(u, r)의 순차 갱신으로 할 수 있으며, neuron당 step마다 부동소수 곱셈 5회가 필요하다.

### Cited Findings
- **decoupled reset은 LIF용이고 PRF와 별개다.** LIF(Eq. 2-3): `u_t = β(u_{t-1} - V_th s_{t-1}) + c_t`, `s_t = H(u_t - V_th)`, `β ≜ 1 - 1/τ`. 전개(Eq. 4)한 뒤 reset 항을 threshold 쪽으로 옮겨 `s_t = H(u'_t - d_t)`(Eq. 5-6)로 쓴다. `u'_t = βu'_{t-1} + c_t`, `A_t = β(1+A_{t-1})` (`u'_{t-1} ≥ d_{t-1}`일 때) 또는 `βA_{t-1}`(Eq. 10), `d_t = V_th(A_t + 1)`, `A_0 = u'_0 = 0`(Eq. 11). 병렬식(Eq. 12): `U'_T = F^{-1}(F(C_T) ⊙ F(K_T))`, `D_T = f_D(U'_T)`, `S_T = H(U'_T - D_T)`, `K_T = (β^0..β^{T-1})` — [PRF v2 §4.1](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **복잡도 주장.** "we convert the all calculation of s_t with O(L^2) complexity into a combination of u'_t with O(L log L) complexity and subsequently d_t with O(L) complexity, achieving a training speed-up of approximately L/(log L + 1)" (§4.1). 측정값은 1,024 step에서 4.6 s → 0.7 s로 6.57×, 16,384에서 9.35×, 32,768에서 16.50×이고, forward 1.77×, backward 132.55×다(§5.1). CIFAR 1024 길이 5층 MLP 학습 비교에서 8.58×, test 0.4355(parallel) 대 0.4378(sequential)이다(App. K, Fig. 10) — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **PRF neuron.** `dũ/dt = γ̃ũ + c(t)`, `ũ ≜ u + ir`, `γ̃ = γ + iθ`(Eq. 14-16). 이산식(Eq. 19-21, 27-28): `ũ_t = exp(Δγ̃)ũ_{t-1} + Δc_t`, `s_t = H(Re{ũ_t} - V_th)`, `γ̃ = -1/τ + iθ`. 병렬식: `Ũ_T = F^{-1}(F(C_T) ⊙ F(K̃_T))`, `S_T = H(Re{Ũ_T} - V_th)`, `K̃_T = (ΔA^0..ΔA^{T-1})`, `A = exp(Δγ̃)`. "When θ = 0, the model reduces to the standard LIF neuron without reset."(§4.2, Theorem 2) — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **Resonance.** 공진 주파수는 ω = θ이고 최대 진폭은 τ다(App. D, Eq. 48). 입력이 N(0, σ²)이면 u_t는 N(0, τΔ/2 σ²)로 수렴한다(Theorem 3). 초기값 민감도가 커서 θ 범위를 task별로 고른다(App. N, Tables 17-21) — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **Hidden state와 출력.** 표기상 `u_t ∈ C^D`, 출력 `y_t ∈ {0,1}^D`(Table 6). 배치는 실수 2개로 전개한다(Eq. 50-54): `φ_re = exp(-Δ/τ)cos(Δθ)`, `φ_im = exp(-Δ/τ)sin(Δθ)`, `u_t = φ_re u_{t-1} - φ_im r_{t-1} + Δc_t`, `r_t = φ_im u_{t-1} + φ_re r_{t-1}`, `s_t = H(u_t - V_th)`. LIF 대비 "two additional multiplications and one more addition than LIF"(§4.2, App. E도 같은 내용)이고 Table 7은 neuron 비용을 `E_M·5D + E_AC·3D`로 적는다 — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **FFT는 학습용(논문 기준).** Table 2에서 PRF만 Seq. Infer ✓, Par. Train ✓. Fig. 10 caption: "During inference, all metrics are computed using sequential computation." — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **공개 코드는 forward가 항상 FFT다.** `PRFNeuron.forward`는 `parallelization_step`만 부르고, docstring은 "there is no explicit reset (soft reset is equivalent to the constant threshold, see the paper)", "`sequential_step` provides the equivalent recurrent form for reference"라고 적는다. spike는 `surrogate_function(u_seq.real - v_threshold)`이다. Δ, θ는 채널별 학습 파라미터(`log_dt`, `theta_log`, lr 0.001)이고 τ 기본값은 2다 — [spikingblock.py L67-169](https://github.com/HuuYuLong/prf/blob/main/src/models/sequence/modules/spikingblock.py) [read in code]
- **reset을 더한 변형은 따로 있다.** `PRFNeuronReset`은 복소 kernel에 `scan_dynamic_threshold`를 더한다(README: "PRF with an extra reset (Table 9)"). threshold scan은 `torch.where`로 fire mask를 만들어 u 쪽으로 gradient가 흐르지 않는 형태다 — [spikingblock.py L297-387](https://github.com/HuuYuLong/prf/blob/main/src/models/sequence/modules/spikingblock.py), [README](https://github.com/HuuYuLong/prf/blob/main/README.md) [read in code]. Table 10의 backward 항목에도 reset scan 경로가 없다 — [PRF v2 App. L](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **제3자 분류.** Man Yao 그룹 리뷰표는 PRF를 "Function 1(비선형 도입) ✗, Function 2(막전위 제어) ✓"로 분류한다 — [Parallel Training in SNNs Table 1](https://arxiv.org/abs/2602.01133) [read in full text, Table 1]. D-RF는 "The SD-TCM model assumes that a learnable decay constant can effectively approximate the reset process, with a static threshold"라고 적는다 — [D-RF](https://arxiv.org/abs/2509.17186) [read in full text, §5.2]

### Inferences
- PRF neuron은 채널마다 복소 pole 1개인 S4D 형태의 선형 filter에 Heaviside를 붙인 것과 같다(Eq. 19-21과 Table 6에서 도출). spike가 막전위에 되돌아오지 않으므로 FFT 병렬화가 정확히 성립한다. 반면 reset이 있는 LIF에서는 Eq. 12의 D scan이 필요하다.
- "reset"이라는 이름에도 불구하고 PRF의 r(t)는 spike와 무관한 두 번째 선형 state다. 논문 §3-4의 "decoupled reset"과 최종 PRF neuron을 같은 기법으로 인용하면 틀린다.
- inference에 spike-driven이 아닌 연산이 필요하다: neuron당 step마다 FP 곱셈 5회(입력 spike가 없어도 oscillation 갱신이 필요), BN(Table 8), real-valued embedding과 head. App. H의 "no computation is triggered when the input is zero"는 Eq. 52-53의 무입력 갱신과 맞지 않는다.
- [computed] L/(log₂L+1)은 1,024에서 약 93×, 32,768에서 2,048×인데 측정은 6.57×, 16.50×다. 이론식은 측정값을 설명하지 못한다.
- Contribution 문장 "accelerates the back propagation by three orders of magnitude"와 측정 132.55×(2자리)가 어긋난다. Eq. 18 `dr/dt = γu + θr`도 Eq. 17(`ṙ = θu + γr`)과 어긋나는데, Eq. 53과 코드는 Eq. 17 쪽이므로 Eq. 18은 오식으로 보인다.

### Gaps
- `inference.py`로 측정한 firing rate가 FFT 경로라는 점은 코드로 확인했지만, Fig. 9-10 외에 PRF neuron 자체의 sequential 동치 수치 비교는 쪽에 없다.
- v1(2024-10-04)은 읽지 않았다. arXiv API상 최신은 v2(2024-10-30)이고 그 이후 버전은 없다.

## 아키텍처: token mixing, channel mixing, T 대 L, bidirectional

### Takeaway
SD-TCM은 `PRF → Linear → 더하기 → Spatial Neuron(상태 없는 threshold) → Linear → 더하기`이고 층 사이 신호는 실수 membrane shortcut이다. T는 항상 sequence 길이 L이며 T_h 같은 반복 축이 없다. bidirectional은 spike concat 뒤 Linear(2D→D)이고 stream 간 차이나 gate는 없다.

### Cited Findings
- **층 순서.** `S^l_T = TN(U^l_T)`, `RPE = U^l_T + Linear(S^l_T)`, `S'^l_T = SN(RPE)`, `U^{l+1}_T = RPE + Linear(S'^l_T)`(Eq. 98-101). S, S'는 `B^{T×N×D}`이고 U, RPE는 `R^{T×N×D}`다. "For token mixing, we use a PRN Neuron followed by a Linear layer, while for channel mixing, the Neuron is replaced by a Spatial Neuron."(§4.4, PRN은 원문 표기) residual은 "membrane shortcut"이다 — [PRF v2 §4.4, App. H](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **Spatial Neuron.** τ→1⁺ 극한으로 `s_t = H(c_t - V_th)`(Eq. 96). 출력에 학습 진폭 α를 곱하고 inference에서 Linear에 합친다. Text 기준 α 효과는 Table 4다: without α 85.75, α on SN & TN 85.69(-0.06), α on TN 86.11(+0.36), α on SN 86.33(+0.58) — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **코드도 같다.** `s = neuron1(u)` → `y = pro_linear1(dropout(s))` → `x = u + y` → `s = neuron2(x - 0.5)` → `y = pro_linear2(s * exp(alpha)) + x`. Spatial Neuron은 시간 상태가 없는 `surrogate.ATan()`이고 threshold는 0.5다. 입력 형태는 `(L, B, H)`다 — [SpikingBlockGate.forward L987-1033](https://github.com/HuuYuLong/prf/blob/main/src/models/sequence/modules/spikingblock.py) [read in code]
- **정규화와 바깥 residual.** Table 8: 전 task BN, Pre-norm은 ListOps False, Text True, Retrieval True, Image False, Pathfinder True — [PRF v2 Table 8](https://arxiv.org/pdf/2410.03530v2) [read in full text]. 기본 config는 `residual: R`이고 `block.py`의 residual block이 layer를 감싼다 — [base.yaml](https://github.com/HuuYuLong/prf/blob/main/configs/model/base.yaml) [read in code]
- **T와 L.** "where T denote timestep while aligning with the sequence length."(App. H, Eq. 97 뒤). LRA는 T = ListOps 2,048, Text 4,096, Retrieval 4,000, Image 1,024, Pathfinder 1,024(Table 5 입력 길이), sMNIST / psMNIST 784(Table 2). README 명령에도 `T=2048`, `T=4096`, `T=4000`, PathX `T=16384`로 적혀 있다 — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text], [README](https://github.com/HuuYuLong/prf/blob/main/README.md) [read in code]
- **Bidirectional(App. H, Eq. 102).** `S^l_T = Concat(TN(U^l_T), Rev(TN(Rev(U^l_T))))`, `S^l_T ∈ B^{T×N×2D}`. "We simply pass the input sequence through an PRF Neuron, and also reverse it and pass it through an independent second PRF Neuron. These spiking outputs are concatenated and passed through a position wise linear layer." `W ∈ R^{2D×D}`. S4 bidirectional을 따른 것이라고 적는다 — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **코드의 bidirectional.** `s = torch.concat([s, self.reverse_neuron1(u.flip(dims=[0])).flip(dims=[0])], dim=-1)`, `pro_linear1 = nn.Linear(2*h, h)`. 두 neuron은 같은 입력 u를 읽고 θ, Δ만 따로 가진다. 앞단 Linear는 공유이고 막전위는 fusion에 쓰이지 않는다 — [spikingblock.py L925-935, L993-1002](https://github.com/HuuYuLong/prf/blob/main/src/models/sequence/modules/spikingblock.py) [read in code]
- **어느 task에서.** "For the ListOps, Text, and Retrieval tasks, we use the causal architecture, while S4 employs a bidirectional architecture for all tasks. For the Image and Pathfinder tasks, we use a bidirectional architecture."(§5.3). README는 PathX에도 `bidirectional=true`를 쓴다 — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text], [README](https://github.com/HuuYuLong/prf/blob/main/README.md) [read in code]
- **이름의 "Gate".** `SpikingBlockGate`의 Gate는 α 진폭을 뜻한다("a trainable amplitude α acts as a gate", §4.4). `SpikingBlockGateSelective`의 docstring은 "selective (trainable) threshold. Experimental; not used in the paper."이고 flip + concat 구조는 같다 — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text], [spikingblock.py L1036-1182](https://github.com/HuuYuLong/prf/blob/main/src/models/sequence/modules/spikingblock.py) [read in code]
- **Hyperparameters(Table 8, Depth / Channels / Dropout / LR / Neuron LR / B / Epochs / WD).** ListOps 8 / 128 / 0 / 0.005 / 0.001 / 50 / 40 / 0.05. Text 6 / 256 / 0 / 0.005 / 0.001 / 16 / 32 / 0.05. Retrieval 6 / 256 / 0 / 0.005 / 0.001 / 32 / 20 / 0.05. Image 6 / 512 / 0.1 / 0.005 / 0.001 / 50 / 200 / 0.05. Pathfinder 6 / 256 / 0.05 / 0.005 / 0.001 / 64 / 200 / 0.03. (Δmin, Δmax)는 전부 (0.001, 0.1) — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]

### Inferences
- 층 사이 신호 U^{l+1}이 실수이므로 PRF는 "neuron 출력은 binary"이지만 "모든 연결이 0/1"은 아니다. 다음 층 PRF의 입력은 실수 누적값이다. SpikMixer의 IAND / OR 잔차와 다른 지점은 여기다.
- [computed] 코드의 `fr_scale`은 θ ~ U(0, 2π/fr_scale)이고, README의 12 / 16 / 0.6은 Table 17, 18, 21의 굵은 값 [0, π/6], [0, π/8], [0, 2π/0.6]과 일치한다. 공개 코드는 논문 표의 최고 설정을 재현하도록 고정돼 있다.
- Eq. 97의 N("flattened spike patches")은 코드와 LRA 설정에 없다. Spike-driven Transformer 표기를 옮겨온 흔적으로 보이며, 실제 PRF는 patch 없이 L개 token(pixel, 문자)을 그대로 걷는다.
- base.yaml의 `residual: R`이 실제로 적용되면 Eq. 98-101 바깥에 실수 residual이 하나 더 있는 셈이다. 코드를 돌려 확인하지는 않았다.

### Gaps
- 논문에 embedding 층과 decoder의 정확한 형태는 안 나온다. 코드 config는 `decoder: mode: pool`이다.
- bidirectional의 두 stream이 왜 앞단 Linear를 공유하는지에 대한 설명은 쪽에 없다.

## 결과와 energy: 표, 계산법, 불일치

### Takeaway
LRA 평균 accuracy는 S4 84.03 대 SD-TCM 82.39, energy는 11.748 mJ 대 0.168 mJ(98.57% 감소)다. pJ 값은 논문에 없지만, E_M = E_MAC = 4.6, E_AC = 0.9를 넣으면 Table 3의 Ours 5칸이 소수 셋째 자리까지 재현된다. 이때 bidirectional의 두 번째 stream과 BN, embedding, head는 빠져 있다.

### Cited Findings
- **Table 3 (verbatim).** En.(mJ) S4 5.104 / 3.718 / 24.439 / 19.222 / 6.256, Avg 11.748. Ours 0.075 / 0.298 / 0.211 / 0.187 / 0.067, Avg 0.168. Acc.(%) S4 59.60 / 86.82 / 90.90 / 88.65 / 94.20, Avg 84.03. Ours 59.20 / 86.33 / 89.88 / 84.77 / 91.76, Avg 82.39. Par. S4 815k / 843k / 3.6M / 3.6M / 1.3M. Ours 272k / 830k / 1.1M / 4.1M / 1.3M (ListOps / Text / Retrieval / Image / Pathfinder 순) — [PRF v2 §5.3](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **Table 5 (verbatim, NL Act-Free / FP MAC-Free 뒤 5 task와 Avg).** Random 10.00 / 50.00 / 50.00 / 10.00 / 50.00 / 34.00. Transformer ✗✗ 36.37 / 64.27 / 57.46 / 42.44 / 71.40 / 54.39. S4 (Bidirectional) ✗✗ 59.60 / 86.82 / 90.90 / 88.65 / 94.20 / 84.03. Binary S4D ✗✗ 54.80 / 82.50 / 85.03 / 82.00 / 82.60 / 77.39. + GSU & GeLU ✗✗ 59.60 / 86.50 / 90.22 / 85.00 / 91.30 / 82.52. Stoch. SpikingS4 ✗✗ 55.70 / 77.62 / 88.48 / 80.10 / 83.41 / 77.06. SpikingSSMs ✗✗ 60.23 / 80.41 / 88.77 / 88.21 / 93.51 / 82.23. Spiking LMU ✓✗ 37.30 / 65.80 / 79.76 / 55.65 / 72.68 / 62.23. ELM Neuron ✗✗ 44.55 / 75.40 / 84.93 / 49.62 / 71.15 / 69.25. Spike-Driven TCM ✓✓ 59.20 / 86.33 / 89.88 / 84.77 / 91.76 / 82.39 — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **Table 2 (sMNIST 784 / psMNIST 784, params k, Top-1).** sMNIST: LIF 85.1 / 155.1* 72.06 / 89.28, PLIF 87.92 / 91.79, GLIF 95.27 / 96.64, TC-LIF 97.35 / 99.20, ALIF 156.3* 98.70, AHP 68.4* 96.00, BHRF 68.9 99.1 ± 0.1, PRF 68.9 **99.18**. psMNIST: LIF 155.1* 80.36, GLIF 90.47, TC-LIF 95.36, ALIF 94.30, BHRF 68.9 95.2, PRF 68.9 **96.87**. 전체 학습 1.60시간, batch 256, 200 epoch — [PRF v2 §5.2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **Fire rate.** Table 11 평균: ListOps 3.53, Text 4.32, Retrieval 1.48, Image 3.47, Pathfinder 3.29. Tables 12-16에 층별 값이 있다 — [PRF v2 App. M](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **Energy 식(Table 7).** S4-LegS token `E_MAC·(H²+DH)` + `E_MAC·(HD+D²)`, channel `E_MAC·D²` ×2 + `E_MAC·2D + E_M·D`. Ours token `E_M·5D + E_AC·3D` + `E_AC·(R·D²+D)`, channel `E_AC·(R·D²+D)`. R은 firing rate다. 논문 전문에 pJ, 4.6, Horowitz가 나오지 않는다(원문 텍스트 검색, 4.6은 "4.6 seconds"만). 메모리 접근 항도 없다 — [PRF v2 App. I](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **코드의 상수.** `E_mac = 4.6`, `E_mul = 3.7`, `E_ac = 0.9`, `prf(m,T) = 5*m*T*E_mul + 3*m*T*E_ac`, `lif(m,T) = m*T*E_mac`. neuron 비용만 계산하고 synapse 항은 없다 — [neuron_dynamic_energy.py](https://github.com/HuuYuLong/prf/blob/main/neuron_dynamic_energy.py) [read in code]
- **ListOps 본문-표 역전 확인.** §5.3 본문 "Specifically, the accuracy on ListOps is 59.60%, compared to S4's 59.20%." Table 3은 S4 59.60, Ours 59.20이고 Table 5와 Table 17의 값도 59.20이다. 본문이 뒤집힌 것이 맞다 — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **PathX.** §6: "due to the sensitivity of training to neuron initialization, the PathX problem remains unsolved." 코드 README에는 PathX 명령이 포함돼 있다 — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text], [README](https://github.com/HuuYuLong/prf/blob/main/README.md) [read in code]
- **모델 선택.** README 명령과 config가 모두 `train.monitor=test/accuracy`를 쓴다. seed는 config 기본 2222이고 v2 Table 3, 5에는 표준편차가 없다. README 주석: "the commands below are the best tuned settings ... differ slightly from the configurations reported in the paper" — [README](https://github.com/HuuYuLong/prf/blob/main/README.md), [snn-listops.yaml](https://github.com/HuuYuLong/prf/blob/main/configs/experiment/lra/snn-listops.yaml) [read in code]
- **제3자 재현.** D-RF Table 3: SD-TCM† Spiking Rate 11.2 / 7.9 / 5.7 / 15.7 / 5.8, Avg 9.3, Energy "mJ" 71.4 / 277.8 / 195.7 / 503.6 / 51.0, Avg 220.6, "† Results reproduced by ourselves, as the original code is not publicly available." — [D-RF](https://arxiv.org/abs/2509.17186) [read in full text, Table 3]

### Inferences
- [computed] `E = L × depth × [4.6·5D + 0.9·3D + 2·0.9·(R·D²+D)]`에 Table 8의 depth, D와 Table 11의 R을 넣으면 0.0747 / 0.2983 / 0.2109 / 0.1871 / 0.0671 mJ이고 발표값 0.075 / 0.298 / 0.211 / 0.187 / 0.067과 일치한다. 즉 Table 3은 샘플당 energy이며 E_M = 4.6 pJ를 쓴 것으로 보인다. 공개 코드의 `E_mul = 3.7`을 넣으면 ListOps가 0.0653 mJ로 맞지 않는다.
- [computed] 같은 식에서 neuron 항 비중이 72.1 / 54.2 / 74.9 / 43.2 / 60.2%다. "FP MAC Free ✓"인 모델의 energy 대부분이 neuron의 부동소수 곱셈이다. 공정한 비교라면 LIF 기반 SpikMixer도 neuron 갱신 비용을 같은 방식으로 세야 한다.
- [computed] bidirectional인 Image, Pathfinder도 단방향 식으로 재현된다. 코드대로 두 번째 stream과 Linear(2D→D)를 세면 대략 0.321, 0.121 mJ로 1.7-1.8배다. 발표값은 두 번째 stream을 뺀 것으로 보인다.
- [computed] Table 11 평균은 층별 rate의 단순 평균이 아니다. 단순 평균은 3.97 / 5.05 / 1.73 / 4.05 / 3.84이고, 합을 (2·depth + 2)로 나누면 3.53 / 4.32 / 1.48 / 3.47 / 3.29로 정확히 일치한다. 층별 rate로 다시 계산한 energy는 0.0769 / 0.3193 / 0.2180 / 0.2040 / 0.0711 mJ로 2.5-9% 높다. 분모가 2 큰 이유는 쪽에 없다.
- [computed] Table 5의 ELM Avg 69.25는 해당 행 5칸의 평균 65.13과 다르다. 다른 행은 모두 맞는다. D-RF Table 2가 같은 69.25를 그대로 옮겼다.
- D-RF의 "mJ"는 단위 표기가 의심스럽다. PRF가 0.075 mJ(75 μJ)라 한 ListOps를 71.4 "mJ"로 적었는데 숫자가 μJ로 보면 가깝다. 다만 fire rate(11.2 대 3.53)는 표기 문제로 설명되지 않고 실제 재현 차이일 수 있다. 원 코드 공개 이전의 재현이라는 점도 감안해야 한다.
- test accuracy로 모니터링하고 θ 범위를 Tables 17-21에서 test 값으로 고른 것으로 보이므로(표의 굵은 값이 Table 3의 값), Table 3 accuracy는 낙관적일 가능성이 있다. 다만 어떤 checkpoint로 표를 만들었는지는 쪽에 없다.
- Table 5의 비교 대상 일부는 시점이 지난 값이다. Stoch. SpikingS4는 ICLR 판(P-SpikeSSM)에서 58.20 / 81.20 / 88.53 / 82.40 / 84.80으로 올랐다.

### Gaps
- S4 행의 H와 D가 논문에 없어 S4 energy는 재현하지 못했다. Table 8의 depth, D로 역산하면 Text에서 층-step당 MAC이 32,888로 3D²(196,608)보다 작아 S4는 다른 구성을 쓴 것으로 보인다.
- Pattern Recognition 판의 수치, 표 번호, energy 식은 확인하지 못했다.

## 게재 상태, 코드, 인용, follow-up

### Takeaway
PRF는 제목 그대로 Pattern Recognition vol. 183(2027년 3월호, article 114968)에 실렸고 저자 구성이 일부 바뀌었다. 공식 코드는 2026-08-15에 공개됐고 인용은 8-9건이다. ICLR 게재 근거는 찾지 못했다.

### Cited Findings
- **Journal.** 제목 "PRF: Parallel Resonate and Fire Neuron for Long Sequence Learning in Spiking Neural Networks", Pattern Recognition, volume 183, article 114968, published 2027-03(print), DOI 10.1016/j.patcog.2026.114968, Crossref 등록 2026-09-19, PII S0031320326019321. 저자 Yulong Huang, Zunchang Liu, Xiaopeng Lin, Hongwei Ren, Hongxiang Huang, Haotian Fu, Yue Zhou, Hong Xing, Zeke Xie, Bojun Cheng — [Crossref](https://api.crossref.org/works/10.1016/j.patcog.2026.114968) [read in full text, metadata]. arXiv 저자에 있던 Changchun Feng은 빠지고 Hongxiang Huang, Zeke Xie가 추가됐다 — [arXiv v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **arXiv.** v1 2024-10-04, v2 2024-10-30(최신), cs.NE. admin note: "text overlap with arXiv:2208.04933 by other authors"(S5 논문). App. J.2에 "The following describe mainly refer from Smith et al. (2023)"라고 적혀 있다 — [arXiv API](https://export.arxiv.org/api/query?id_list=2410.03530) [read in full text, metadata]
- **코드 약속.** 1쪽 각주 "The GitHub repository will be released after paper accepted." App. M "The checkpoints and statistical code can be found in the open-source repository." 둘 다 링크는 없다 — [PRF v2](https://arxiv.org/pdf/2410.03530v2) [read in full text]
- **코드.** [HuuYuLong/prf](https://github.com/HuuYuLong/prf), Apache-2.0, README "Official code release", 저장소 생성 2024-10-03, 커밋 `416e057c1` "Release: PRF neuron + SD-TCM code for LRA and sequential image tasks"(2026-08-15), 이어서 `9c077dd91` "Remove journal/manuscript references"(2026-08-15). 같은 계정이 Yulong Huang 1저자의 SpikingMamba 코드 위치로 논문에 적혀 있다 — [SpikingMamba](https://arxiv.org/abs/2510.04595) [read in full text, 초록 부분]. S4 코드베이스 기반이고 `snn-*` config는 LRA와 MNIST만 있다. `configs/experiment/forecasting/s4-informer-etth.yaml` 등은 S4 baseline용이다 — [repo tree](https://github.com/HuuYuLong/prf) [read in code]
- **코드가 보여주는 추가 사실.** baseline neuron(BHRF, ELM, TC-LIF, TS-LIF, ParaLIF, PSN 변형), `PRFNeuronTrain`(sMNIST 명령에 사용), seqCIFAR 소형 모델(`d_model=128`, `dataset.grayscale=false`). README의 표 번호(LRA Table 4, sMNIST Table 3, energy Table 5, reset Table 9)가 v2(각각 Table 3 / 5, Table 2, Table 3, 없음)와 다르다 — [README](https://github.com/HuuYuLong/prf/blob/main/README.md) [read in code]
- **인용 수.** Semantic Scholar 8건, OpenAlex cited_by 9건(목록 조회 8건) — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2410.03530), [OpenAlex](https://openalex.org/W4403884634) [read in full text, metadata]. Semantic Scholar는 아직 venue를 arXiv로 표시한다.
- **인용 논문 목록.** D-RF(NeurIPS 2025, [2509.17186](https://arxiv.org/abs/2509.17186)), Multiplication-Free Parallelizable Spiking Neurons(NeurIPS 2025 표기, [2501.14490](https://arxiv.org/abs/2501.14490)), FiTS(NeurIPS 2026 Spotlight, [2605.13071](https://arxiv.org/abs/2605.13071)), Parallel Training in SNNs([2602.01133](https://arxiv.org/abs/2602.01133)), ASRC-SNN(ICASSP 2026, DOI 10.1109/icassp55912.2026.11464148), A scalable hybrid training approach([2506.14464](https://arxiv.org/abs/2506.14464)), Edge Intelligence with SNNs([2507.14069](https://arxiv.org/abs/2507.14069)), Spikes as perturbations(Front. Comput. Neurosci. 2026), Bio-Inspired Auditory Processing([2610.29720](https://arxiv.org/abs/2610.29720)) — [OpenAlex 인용 목록](https://api.openalex.org/works?filter=cites:W4403884634) [abstract only]
- **Follow-up 수치.** D-RF LRA: 60.02 / 86.52 / 90.02 / 85.32 / 92.36, Avg 82.88(SD-TCM 82.39보다 높음) — [D-RF Table 2](https://arxiv.org/abs/2509.17186) [read in full text, Table 2]. Bio-Inspired Auditory Processing은 PRF update를 "4 multiplications and 3 additions", 17.5 pJ로 센다(PRF 원문 Table 7은 곱셈 5회) — [2610.29720](https://arxiv.org/abs/2610.29720) [read in full text, App.]
- **ETT, ECG 사용 여부.** 내려받은 5개 인용 논문(2602.01133, 2610.29720, 2605.13071, 2501.14490, 2509.17186)에서 ETT, ECG, MIT-BIH, PTB-XL 일치 0건 [grep only]. 나머지 인용 논문은 확인 안 함.

### Inferences
- README 표 번호와 PathX, 추가 baseline 때문에 journal 판은 v2보다 내용이 늘어난 것으로 보인다. PRF를 인용할 때는 journal 판 기준 수치를 따로 확인해야 하고, v2 수치와 혼용하면 안 된다.
- PRF의 token-axis neuron mixing 선행 시점은 arXiv 기준 2024-10이다. 우리 논문 심사에서는 이 날짜로 비교될 것이다.

### Gaps
- Pattern Recognition 판의 abstract와 본문은 Crossref와 OpenAlex에 없고 ScienceDirect는 403이라 읽지 못했다. 접수일, 데이터 공개 문구, PathX 결과는 미확인이다.
- ICLR 투고 여부는 웹 검색에 근거가 없고 OpenReview는 이 서버에서 막혀 있어 확인하지 못했다. 지금 시점에는 ICLR 논문으로 쓰면 안 된다.
- Hugging Face, Google Drive 등의 checkpoint 위치는 확인하지 않았다.

## 가장 가까운 relatives

### Takeaway
S4 계열(Binary S4D, P-SpikeSSM, SpikingSSMs, SPikE-SSM)은 실수 SSM이 token mixer이고 T = L이다. spiking Mamba 계열은 LIF 자체를 state로 쓰는 SpikeMamba(ECG 포함), bidirectional SSM scan을 쓰는 SmolMamba와 Spiking Point Mamba(반복 T 축 있음)로 갈린다. 예측 task는 SpikySpace, SeqSNN, TS-LIF가 다루지만 ETT는 없다.

### Cited Findings

요약표 (T = neuron이 사실상 걷는 축과 길이, 출처는 아래 항목)

| 모델 / venue | Mixer 본체 | T 대 L | bidirectional | 출력 | 데이터 | Energy 방식 |
|---|---|---|---|---|---|---|
| PRF SD-TCM / Pattern Recog. 2027 | PRF neuron + Linear | T = L | Image, Pathfinder (concat) | binary, 층 사이는 실수 | sMNIST, psMNIST, LRA | MAC 4.6, AC 0.9 추정, neuron 포함 |
| Binary S4D / Sci. Rep. 2024 | 실수 S4D + Heaviside | T = L | LRA에서 yes, sMNIST no | binary 후 GLU 또는 GSU | sMNIST, sCIFAR, LRA, Path-X | pJ 수치 없음 |
| P-SpikeSSM / ICLR 2025 | SSM kernel + Bernoulli sampler | T = L | 논문 언급 없음, config true | stochastic binary | psMNIST, SC10, LRA | 4.6 / 0.9, 합성곱 기준, I/O 제외 |
| SpikingSSMs / AAAI 2025 | S4D + LIF(SDN) | T = L | 논문 언급 없음, config true | binary 후 FC | MNIST류, LRA, Path-X, WT103, DVS128, SC | 4.6 / 0.9, feature-mix만 |
| SPikE-SSM / IEEE TCDS 2026 | S4D + refractory LIF(PMBC) | T = L | 언급 없음 | binary | LRA, WT103 등 | 4.6 / 0.9, feature-mix만 |
| Spiking LMUFormer / ICLR 2024 | LMU cell + LIF | T = L(streaming) | 없음 | binary, shortcut은 실수 | psMNIST, Speech Commands, LRA | SOP 수만 |
| SpikySpace / preprint 2026 | binary selective scan | scan은 L, spike train T = 3 | 논문 없음 | binary(Avg-IF 변환) | Metr-la, Pems-bay, Solar, Electricity | 22 nm, 이동 및 메모리 포함 |
| SpikeMamba / MICCAI 2026 | LIF가 곧 state | T = L, 외부 T = 1 | 없음 | 논문 binary, 코드 head는 membrane | PTB, PTB-XL, MIT-BIH, ADFTD, CHB-MIT | 4.6 / 0.9, 입력과 MLP만 MAC |
| Vision SmolMamba / preprint 2026 | 실수 SSM + SN | N을 SSM이 scan, T = 4 또는 16 반복 | 2-path, 합 | SN 후 합 | ImageNet, CIFAR, DVS | 4.6 / 0.9, SSM은 MAC |
| Spiking Point Mamba / arXiv 2025 | SSM 분기 + Gate 분기 | token scan, time step 기본 4 | Token Flip / Time Flip | SN, residual 실수 | 3D point cloud | 4.6 / 0.9 |

- **Binary S4D / GSU.** "Binary activations are applied element-wise per each SSM output before position-wise feature mixing". "both Binary S4D and GSU networks are bidirectional"(LRA), sMNIST에서는 "Bidirectionality is also disabled". residual은 feature mixing 뒤 덧셈이다. pJ 환산은 없다 — [Stan & Rhodes](https://arxiv.org/abs/2401.00955) [read in full text, §4.4-4.8]. LRA 54.80 / 82.50 / 85.03 / 82.00 / 82.60, Path-X 61.2, sMNIST 99.1-99.2 — [같은 곳 Fig. 4, Table 1](https://arxiv.org/abs/2401.00955) [read in full text]
- **P-SpikeSSM(이전 이름 Stochastic / S6-based).** ICLR 2025(arXiv v5). LRA: 58.20 / 81.20 / 88.53 / 82.40 / 84.80, psMNIST 98.4, SC10 95.6. Block은 `fmix[t] = gelu(Is[t]·Wm)`, `p = σ(BN(fmix[t] + xs[t]))` 뒤 SpikeSampler다. "require additional time steps to simulate network dynamics. In contrast, our proposed model eliminates this overhead". Energy는 `IFR_in·L²N + IFR_o·LN²` ACC, 합성곱 기준, ListOps 36배, "did not consider memory I/O transaction costs" — [P-SpikeSSM](https://arxiv.org/abs/2406.02923) [read in full text, §1, §3.2, §4]. 논문에 bidirectional 언급은 0건이지만 실험 config 5개는 `bidirectional: true`다 — [config](https://github.com/NeuroCompLab-psu/PSpikeSSMs/tree/main/configs/experiment/lra) [read in code]. **주의: 이 논문의 "SpikeMixer Block"은 channel 쪽 fully-connected block이다.**
- **SpikingSSMs.** AAAI 2025 vol. 39 pp. 20380-20388(SpikySpace 참고문헌 표기). `y_t = Ch_t`를 LIF 입력으로 쓴다(Eq. 8-9). SDN은 작은 CNN으로 reset 후 막전위를 예측해 학습을 병렬화한다(L = 8K에서 101.2배). test에서 SDN을 제거하면 iterative 동작이 된다. LRA VF 59.93 / 82.35 / 88.20 / 86.81 / 93.68 / Path-X 94.80(rate 0.13 / 0.10 / 0.06 / 0.22 / 0.07 / 0.10), VT 60.23 / 80.41 / 88.77 / 88.21 / 93.51 / 94.82. Energy는 4.6 / 0.9, WikiText-103 feature-mix만 계산해 1.265 J 대 65.40 mJ, "the hardware implementation and the dynamics of spiking neurons are ignored" — [SpikingSSMs](https://arxiv.org/abs/2408.14909) [read in full text]. 코드의 bidirectional은 `k = F.pad(k0,(0,L)) + F.pad(k1.flip(-1),(L,0))`로 kernel을 양방향으로 늘린 것이고 뒤에 LIF가 한 번 붙는다. reversed stream이나 stream별 막전위는 없다. `listops.yaml`, `cifar.yaml`에 `bidirectional: true` — [ss4d.py](https://github.com/shenshuaijie/SDN/blob/main/models/spike/ss4d.py) [read in code]
- **SPikE-SSM.** IEEE TCDS vol. 18 no. 4, 2026(arXiv journal_ref). soft reset, refractory LIF와 max-min boundary compression(PMBC)으로 병렬화. LRA 60.17 / 82.43 / 88.82 / 87.23 / 92.04 / Path-X 94.37, spiking rate 12 / 3 / 7 / 10 / 9 / 7%. WikiText-103 60.68 mJ — [SPikE-SSM](https://arxiv.org/abs/2410.17268) [read in full text, Tables 2, 6]. bidirectional은 참고문헌 제목 외 언급이 없다 [grep only]
- **Spiking LMUFormer.** 학습은 병렬, inference는 순차. Eq. 7-12: `U[t] = BN(Conv1d(X_S[t]))`, LIF로 `U_S[t]`, `M[t] = Ā·M_S[t-1] + B̄·U_S[t]`, `O[t] = SN(BN(Conv1d([M_S[t], X_S[t]])))`. 실수 합이 되도록 shortcut 위치를 조정했다. Table 5가 LMU를 "Token Mixer" 열에 적는다. LRA 37.30 / 65.80 / 79.76 / 55.65 / 72.68, Avg 62.24, Speech Commands 96.12, 3.09 × 10⁷ SOPs — [LMUFormer](https://arxiv.org/abs/2402.04882) [read in full text, §4-5]
- **SpikySpace.** 저자들이 "first fully spiking state-space model"이라고 주장. `h_t = SN(A_t h_{t-1} + B_t s_t)`, `y_t = SN(C_t h_t + D s_t)`, `A_t = 2^K if Δ_s,t = 1, I otherwise`(Eq. 5-8). ANN-to-SNN 변환, Avg-IF, "we use a spike train length of 3 to represent a scalar"(App. D.4). R²/RRSE, horizon {3, 6, 12, 24}. Avg R²: SpikySpace .869, iTransformer .870, Mamba .868, SpikeSTAG .863, iSpikformer .856, SpikeTCN .851, GRU .834, SpikeRNN .819. Electricity H=3 energy: SpikySpace 0.17 mJ, SpikeSTAG 4.39, iSpikformer 3.19, iTransformer 9.47(Table 2). Eq. 14 `E = T·s·E_ACC + T·s·(E_move + E_weight)`, 22 nm. Lava 구현만 있고 Loihi 측정은 future work. ETT와 ECG는 없다. PRF를 인용하지 않는다 — [SpikySpace v2](https://arxiv.org/abs/2601.02411) [read in full text, 증명 제외]. 코드 `ACC_pJ = 0.05448`, `MAC_pJ = 4.6`, `SHIFT_pJ = 0.024`, move 0.18 pJ/bit — [energy_cal.py](https://github.com/Kaiwen-Tang/SpikySpace/blob/main/energy_cal.py) [read in code]. `module/spike_fastssm.py`에 `bidirectional: bool = True`, `y = 0.5*(y_f + y_b)`가 있으나 네트워크 파일 4개(snn, qsnn, qann, baselinemamba)에서 import 일치 0건 — [spike_fastssm.py](https://github.com/Kaiwen-Tang/SpikySpace/blob/main/module/spike_fastssm.py) [read in code]
- **SpikeMamba.** MICCAI 2026, LNCS pp. 565-575. "the first architecture to realize SSMs entirely within the spike domain"이라고 주장. `v_t = α·v_{t-1}·(1-s_{t-1}) + I_t`, `s_t = Θ(v_t - V_th)`, `v_t ← v_t - s_t·V_th`, "T = 1 (single forward pass)"(Eq. 3-5, 7). 인코더 `SiLU(BN(W_enc·u_t + b_enc))`, DWConv mixing, 층 residual과 LayerNorm, 시간 평균 풀링 + MLP. Gate(Eq. 6): `g_t = Θ(σ(W_g·[s̄_t ‖ σ²_v(t)] + b_g) - τ)`, g_t = 0이면 `v_t ← v_{t-1}, s_t ← 0` — [SpikeMamba PDF](https://papers.miccai.org/miccai-2026/paper/4698_paper.pdf) [read in full text]
- **SpikeMamba 결과(Table 2, 3).** PTB-XL: SpikeMamba-M Acc 74.13, F1 62.50, AUROC 90.30, "37.09 mJ"(Medformer 72.87 / 62.02 / 89.66 / "8225.97"). PTB: -L 91.72 / 94.00 / 94.92 / "322.91"(Medformer 83.50 / 79.18 / 92.81). MIT-BIH(segment-mixed, 6:2:2): -M Acc 98.87, Prec 93.14, Rec 97.01, F1 94.81, AUROC 99.79, "48.19"(Medformer 98.99 / 95.43 / 95.14 / 95.28 / 99.76 / "2147.43"). CHB-MIT: -M Acc 88.57, AUROC 94.48. baseline은 iTransformer, MTST, Reformer, Medformer. 10-run bootstrap — [SpikeMamba PDF](https://papers.miccai.org/miccai-2026/paper/4698_paper.pdf) [read in full text]
- **SpikeMamba 코드.** MIT-BIH `seq_len = 360`, patch 없음. `for t in range(L)`로 순차 scan, 파일에 flip이나 bidirectional 없음. `SpikeOutputHead` 기본 `output_type='membrane'`이고 학습 스크립트에 override가 없다. 실제 gate는 `(r_t ≥ κ) | (Δv_t ≥ δ) | (e_t ≥ τ_g)`, `Δv_t = ||v_{t-1} - v_{t-2}||_1`로 논문 Eq. 6(sigmoid, 분산)과 다르고 출력은 `(B,)`다. energy 로그는 pJ를 1e6으로 나누고 "mJ"로 표기한다 — [spike_interface.py](https://github.com/Brain-Inspired-Computing-Lab/SpikeMamba/blob/main/mamba_ssm/ops/spike_interface.py), [spike_mamba.py](https://github.com/Brain-Inspired-Computing-Lab/SpikeMamba/blob/main/mamba_ssm/modules/spike_mamba.py), [energy_analysis.py L767-769](https://github.com/Brain-Inspired-Computing-Lab/SpikeMamba/blob/main/energy_analysis.py) [read in code]
- **Vision SmolMamba.** `O_l = SN(BN(Conv2d(X_{l-1})))`, `(H_f, H_b) = SN(DWConv(P_l, Reverser(P_l)))`, `(Z_f, Z_b) = SN(SSM(H_f, H_b))`, `F_l = (Z_f + Reverser(Z_b)) ⊙ Mask_l`(Eq. 16-20). 입력 `B×T×C×N`, Timestep 4(ImageNet, CIFAR), 16(DVS). "for sequence-based SNNs like SpikingSSM and pSpikeSSM, temporal dynamics natively unfold along the sequence dimension without an extra simulation loop." Energy에서 SSM은 MAC, 나머지는 SOP — [SmolMamba](https://arxiv.org/abs/2604.25570) [read in full text, §IV-V, Table I, V]
- **Spiking Point Mamba.** "Token Flip reverses the token dimension, while Time Flip reverses the time dimension." "Time Flip proves more effective than Token Flip", 둘을 같이 쓰면 단방향 수준으로 떨어진다. 차이는 0.3-0.4%다. time step 1-6을 실험(Table 7) — [SPM](https://arxiv.org/abs/2504.14371) [read in full text, §5.3]
- **SeqSNN(Spike-RNN 등, ICML 2024).** "we divide a time step ΔT of the time series into Ts segments"(§3.2), Ts = 4(App.), 데이터 Metr-la, Pems-bay, Solar, Electricity — [Lv et al.](https://arxiv.org/abs/2402.01533) [read in full text, §3]. 등록된 `SpikeRNN`은 `[T, B, L, C]`에 `step_mode="m"` LIF를 적용하므로 코드상 neuron이 걷는 축은 T = 4다. `SpikeRNN2D`만 L을 돈다 — [spikernn.py](https://github.com/microsoft/SeqSNN/blob/main/SeqSNN/network/snn/spikernn.py) [read in code]
- **TS-LIF(ICLR 2025).** dual-compartment neuron, CNN, RNN, Transformer 백본, Metr-la, Pems-bay, Solar, Electricity. ETT 일치 0건 — [TS-LIF](https://arxiv.org/abs/2503.05108) [grep only]
- **그 외.** SpikingMamba(TMLR 01/2026, Mamba2 KD, multi-bit SI-LIF, bidirectional 0건) [grep only]. PDMU(LMU + delay gate, bidirectional은 Eq. 16의 앞뒤 delay 가중합, 생체 EEG, COUGHVID) — [2512.01626](https://arxiv.org/abs/2512.01626) [read in full text, §II 일부]. SiLIF(speech, "the use of bidirectional models prevents streamlined data processing") — [2506.06374](https://arxiv.org/abs/2506.06374) [read in full text, 해당 문단]. SpikMamba, Mamba-Spike, SpikeMba, speech Mamba는 제목과 키워드만 확인 [abstract only]. ELM은 PRF Table 5 수치만 [secondary summary]. SpikeSSL([2610.11456](https://arxiv.org/abs/2610.11456))은 칼슘 이미징 spike 추론용이라 범위 밖이다 [abstract only].
- **이름 충돌.** "SpikeMixer: Dynamic axial mixing for spiking neural networks"(Neurocomputing, 영상, H/W 축 혼합) — [Germinted/SpikeMixer](https://github.com/Germinted/SpikeMixer) [read in code, README]. P-SpikeSSM의 "SpikeMixer Block"도 있다. arXiv 제목에서 "SpikMixer" 일치는 0건 [grep only].

### Inferences
- ETT로 평가한 관련 모델은 읽은 범위에 없다. 예측은 전부 Metr-la, Pems-bay, Solar, Electricity(SeqSNN 설정 계승)이고 지표도 R²/RRSE다. ECG는 SpikeMamba가 유일하다.
- [computed] SpikeMamba의 Eq. 7에 Table 1 값을 넣으면(gate 비율 1 가정) PTB-XL-L 1.09 mJ, -M 0.16 mJ, -T 0.02 mJ인데 표는 260.41, 37.09, 3.97이다. 코드의 1e6 나눗셈(pJ→μJ)을 보면 표의 "mJ"는 μJ일 가능성이 크다(Medformer 8226 "mJ"는 MAC 1.79조에 해당해 비현실적). 다만 -T는 MAC 항만 9.66 μJ로 3.97을 넘어 이 해석으로도 완전히 맞지 않는다. SpikeMamba의 energy 수치는 그대로 인용하면 안 된다.
- SpikeMamba는 기본값이 membrane head라면 "downstream multiplications → conditional additions" 전제가 깨진다. 논문 실험이 이 기본값인지는 확인하지 못했다. 또한 MIT-BIH는 segment-mixed split이라 inter-patient 평가와 직접 비교할 수 없다.
- SpikySpace의 "first", SpikeMamba의 "first"는 서로와 PRF를 고려하면 성립하지 않는다. "first fully spiking SSM" 주장은 인용 근거로 쓰기 어렵다.
- SpikySpace §1의 26.6×, 78.9×는 v2 Table 2로 계산한 18.8×, 55.7×와 다르다 [computed]. 초록의 "over 96.1%"도 Table 2의 94.7-98.2%와 범위가 다르다.
- SmolMamba 문장은 분류 기준으로 쓸 만하다: 반복 T를 쓰는 모델과 sequence 축에 dynamics를 푸는 모델을 저자가 명시적으로 구분한다. SpikMixer는 mixer에서 후자, embedding과 MLP에서 전자이므로 두 범주에 걸친다.

### Gaps
- SpikeMamba는 논문 실험 시 `output_type`을 어떻게 썼는지, 표의 단위가 무엇인지 저자 확인이 필요하다.
- Binary S4D의 Sci. Rep. 판과 AAAI, TCDS, MICCAI 출판본은 비교하지 않았고 arXiv 또는 저자 공개본만 읽었다.
- 검색은 arXiv 제목 및 초록 쿼리, PRF 인용 목록, GitHub 저장소 검색, 웹 검색에 한정했다. DBLP와 OpenReview는 막혀 있다. "없다"는 "이 범위에서 찾지 못했다"는 뜻이다.
- TS-LIF, SpikMamba, Mamba-Spike, speech Mamba의 내부 구조는 깊이 읽지 않았다.

## 겹침 분석: SpikMixer의 각 요소를 누가 이미 갖고 있나

### Takeaway
"attention 없는 neuron token mixing + T = sequence 길이 + 두 stream bidirectional" 조합은 PRF가 2024-10에 이미 공개했다. 찾은 범위에서 SpikMixer만 가진 것은 (1) neuron이 token 축을 걷는 mixer 옆의 독립 T_h 반복 축, (2) |H_f - H_r| spiking gate와 MUX fusion, (3) 실수 shortcut 없는 IAND / OR 잔차, (4) ETT 평가다. 단 (2)의 입력은 실수 막전위이고 BN도 쓰므로 "완전 spike-only"로 주장하기는 어렵다.

### Cited Findings

비교 기준은 [spk_mixer.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/spk_mixer.py)의 `SpkBlock` [read in code]: `self.mixer`, `self.mixer_rev`가 각자 Linear → BN → neuron을 가지고, `d = (hf - hr).abs()`, `sd = sd_neuron(sd_bn(d))`, `a = self.fuse(torch.cat([fr, sd], dim=-1))`, `m = sd * a + (1 - sd) * torch.minimum(f, r)`, `connect` = IAND / OR / ADD.

표의 근거는 위 섹션들의 출처와 같다. 각 칸의 "논문" / "코드"는 어디에서 확인했는지를 뜻한다.

| 논문 | A 무attention, neuron이 mixer | B T = N (token 축) | C bidirectional | D 옆에 T_h 반복 축 | E gate on \|H_f - H_r\| | F spike-only fusion, 잔차 |
|---|---|---|---|---|---|---|
| **SpikMixer** | 있음 | 있음(patch N) | 있음(stream별 Linear, BN, neuron) | 있음(T_h = 4) | 있음 | 있음(IAND / OR), ADD는 비binary |
| **PRF SD-TCM** | **있음** | **있음**(raw token, patch 없음) | **있음**, 논문 + 코드: θ, Δ만 분리, Linear 공유, spike concat | 없음 | 없음 | 없음: concat 후 Linear, 실수 shortcut |
| Binary S4D | 부분: 실수 SSM + Heaviside | 있음 | kernel 기반(LRA) | 없음 | 없음 | 없음: GLU, 덧셈 residual |
| P-SpikeSSM | 부분: SSM + sampler | 있음 | 논문 없음, config kernel 기반 | 없음(명시) | 없음 | 부분: 층 사이는 spike, 내부 gelu, BN, 덧셈 |
| SpikingSSMs | 부분: S4D 뒤 LIF | 있음 | 논문 없음, config kernel 기반 | 없음 | 없음 | 없음 |
| SPikE-SSM | 부분 | 있음 | 언급 없음 | 없음 | 없음 | 없음 |
| Spiking LMU | 부분: LMU + LIF("Token Mixer") | 있음(conv patch 후 streaming) | 없음 | 없음 | 없음 | 없음: 실수 shortcut |
| SpikySpace | 아니오: SSM scan | scan만 L, SN은 T | 논문 없음, 미사용 모듈 존재 | 부분: T = 3은 값 표현용 spike train | 없음 | 없음: 출력 gating은 실수 곱 |
| **SpikeMamba** | **있음**: LIF가 곧 state | **있음**(raw L, MIT-BIH 360) | 없음 | 없음(T = 1) | 없음. 다른 gate: 샘플별 1-bit, 막전위 분산(논문), 시간 차 `‖v_{t-1} - v_{t-2}‖₁`(코드) | 없음: SiLU, BN, LN, 실수 residual |
| **SmolMamba** | 아니오: 실수 SSM | 아니오: N은 SSM이 scan | **있음**: 2-path, 합 | **있음**: T = 4 | 없음 | 없음 |
| Spiking Point Mamba | 아니오 | 아니오 | 있음: Token / Time Flip | 있음: 기본 4 | 없음 | 없음 |
| SeqSNN Spike-RNN | 아니오 | 아니오: 코드상 T = 4 | 없음 | 있음: Ts = 4(값 세분화 개념) | 없음 | 없음 |

### Inferences
- **이미 선행된 부분.** A + B + C 조합은 PRF 하나로 완전히 덮인다. PRF의 App. H 문장(독립 두 neuron, 뒤집어서 흘리고 concat 후 Linear)은 SpikMixer의 `or` / `sdlin` 계열 앞단과 구조적으로 같은 아이디어다. "token 축 bidirectional spiking mixer"를 새 기여로 쓰면 안 된다.
- **차이는 있지만 약한 부분.** neuron 종류(LIF, f-LIF, population 대 PRF), mixer 층 수(2층 Linear → BN → neuron 대 PRF 1개 + Linear), stream별 앞단 Linear 분리, patch token 사용. 이들은 설계 변형이며 새 범주는 아니다.
- **D(T_h 축).** 반복 T 축 자체는 Spikformer 이래 흔하고 SmolMamba, SPM, SeqSNN에 있다. 새로운 것은 "mixer에서는 neuron이 N을 걷고 T_h는 독립 sequence들로 취급"하는 배치뿐이다. SmolMamba는 SN이 T를, 실수 SSM이 N을 담당해 축 배정이 반대다. 다만 T_h 반복이 성능에 기여한다는 근거가 없으면 기여로 내세우기 어렵고, 동일 입력의 반복이라 추가 정보가 들어오지 않는다.
- **E(stream 차이 gate)와 MUX fusion.** 찾은 범위에 선행이 없다. 가장 가까운 SpikeMamba gate는 층 전체 1-bit이고 시간 축 내 활동량을 보며 업데이트 생략이 목적이다. SpikMixer는 neuron별 spike로 다음 값을 고르는 fusion이다. SmolMamba, SPM은 합, PRF는 concat이다. 이 부분이 가장 방어하기 쉬운 차별점이지만, 그 효과를 실험으로 입증해야 기여가 된다.
- **F(spike-only).** IAND / OR 잔차는 relatives가 쓰는 실수 membrane shortcut과 실제로 다르다. 그러나 SpikMixer도 `|H_f - H_r|`(실수)를 읽고, 모든 neuron 앞에 BN(부동소수 곱)이 있으며, neuron 갱신 곱셈도 있다. "모든 연결이 0/1"은 쓸 수 있어도 "inference 전체가 spike-driven"은 PRF에 대한 비판과 똑같이 우리에게도 적용된다. energy 비교 시 PRF처럼 neuron 비용을 세고 BN, membrane gate 연산을 명시하는 편이 안전하다.
- **Task 위치.** PRF 계열은 LRA 중심이고 ETT 평가를 찾지 못했다. ECG 축에서는 SpikeMamba와 직접 비교되므로 split(MIT-BIH segment-mixed)과 energy 단위 문제를 짚어야 한다.
- **판단 요약.** SpikMixer는 "PRF에 이미 포함"은 아니지만 "PRF 아이디어의 LIF 버전 + 새 fusion"으로 읽힐 위험이 크다. PRF neuron을 SpikMixer에 넣은 baseline, PRF SD-TCM 자체를 ETT와 ECG에 돌린 baseline이 있어야 차이를 주장할 수 있다. 공개 코드가 있으므로 가능하다. 이 baseline 실행은 사용자가 결정할 사항이며 이번 조사에서는 하지 않았다.

### Gaps
- Pattern Recognition 판에 v2에 없는 fusion이나 gate가 추가됐는지는 본문을 못 읽어 단정할 수 없다. 코드의 `SpikingBlockGate`와 "not used in the paper"인 Selective 변형에는 stream 차이가 없다는 점까지만 확인했다.
- D-RF, FiTS, MPS 계열의 내부 bidirectional 구조는 깊이 보지 않았다. D-RF 원문의 "bidirectional"은 S4 행 라벨과 참고문헌에서만 일치했다 [grep only].
- SpikMixer와 PRF의 직접 성능 비교 자료는 없다(not run).
