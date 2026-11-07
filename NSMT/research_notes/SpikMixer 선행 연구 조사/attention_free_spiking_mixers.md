# Attention-free spiking token mixer 선행 연구와 SpikMixer overlap 분석

## Q1. 논문별 세부 사항: mixing 식, axis, learned weight, binary 출력, T의 의미, token-axis neuron, non-spike 연산, task, baseline, 정확도 / 에너지, 에너지 계산법, q·k^T·v 계상 여부

### Takeaway
- SSA를 없앤 spiking 모델은 2022-2026년에 이미 여러 편이 있습니다. 방식은 고정 linear transform(2023), learned N×N matrix(2024, 2026), convolution 혼합(2024, 2025), Fourier selector(2025), state-space scan(2026)입니다. "Spikformer와 동등"이라는 주장은 새롭지 않습니다.
- 지정 논문 중 두 편은 attention-free가 아닙니다. TEFormer와 Lapis는 attention을 유지합니다.
- token 축으로 도는 neuron 또는 state는 SNN-MLP(full-precision), SpikeGPT(RWKV + SN), LRF-Dyn, Vision SmolMamba(양방향)에 이미 있습니다.

### Cited Findings

#### 표기
- 표기 규칙: [read in full text]는 해당 대목을 논문 전문(PDF)에서 직접 읽었다는 뜻입니다. 전문의 일부만 읽은 경우 "(부분: 읽은 범위)"를 붙였습니다. [abstract only]는 초록만, [secondary summary]는 검색 요약이나 제3자 정리만 본 경우입니다. "내 계산"은 논문 표의 숫자로 제가 계산한 값입니다.

#### ID와 서지 확인
- arXiv 2601.18274는 실제로 TEFormer입니다. 표지에 ICML 2026(PMLR 306)이 적혀 있습니다. 양방향은 T 축 이야기이고 attention은 유지합니다. [read in full text] — [TEFormer](https://arxiv.org/abs/2601.18274)
- arXiv 2608.11865는 실제로 Lapis입니다. 제목은 "Laplacian Spiking Attention via First-Spike Timing and Membrane Leakage"이고 attention을 유지합니다. [read in full text] — [Lapis](https://arxiv.org/abs/2608.11865)
- SMixer의 arXiv 판은 찾지 못했습니다. 전문은 ICLR 2026 proceedings에서 읽었습니다. OpenReview PDF 링크는 PDF가 아닌 HTML만 돌려주었습니다. [read in full text] — [SMixer](https://proceedings.iclr.cc/paper_files/paper/2026/file/aaeb635296295aa7c9f8b91c5e10960d-Paper-Conference.pdf)
- STMixer는 NeurIPS 2024 proceedings 전문과 공식 코드를 읽었습니다. [read in full text] — [STMixer](https://proceedings.neurips.cc/paper_files/paper/2024/file/e8c20cafe841cba3e31a17488dc9c3f1-Paper-Conference.pdf), [models/STMixer.py](https://github.com/brain-intelligence-lab/STMixer_demo)
- Attention-free Spikformer(arXiv 2308.02557v2)의 PDF에는 학회명이 없습니다. [read in full text] — [Wang et al.](https://arxiv.org/abs/2308.02557)

#### 표 A. Mixing 구조

| 논문 | Mixing 식 | Axis | Learned weight | 출력 binary | T와 그 의미 | Token-axis neuron | Attention-free |
|---|---|---|---|---|---|---|---|
| SNN-MLP 2022 [read in full text] | y=ΣW^T x; u_{t+1}=τu_t(1−o_t)+y_{t+1}; o_{t+1}=u_{t+1}>V_th; r_{t+1}=max(u_{t+1},V_th) (Eq. 11) | patch의 vertical / horizontal, group 단위 | W, τ, V_th | 아니오(fp32, 뒤에 GELU) | simulation T 없음, t = patch index | 예, full-precision | 예 |
| Attention-free Spikformer 2023 [read in full text] | X′=SN(BN(LT(X))) (Eq. 10), LT = FFT의 real part 또는 Haar WT (Eq. 11-16) | sequence(1D), feature + sequence(2D) | LT는 없음 | 예 | 4(static은 같은 영상 T회 반복), 16(DVS). SN 층만 T를 별도 차원으로 쓰고 나머지는 batch에 합침 | 아니오 | 예 |
| Spiking MLP(Li) 2023 [read in full text] (부분: method, 실험, 결론) | U_h=BN(W_h I^h); U_w=BN(W_w I^w); Y=BN(W_f Concat[f(Ũ_h),f(Ũ_w),I])+X (Eq. 1-4) | height, width 각각 | 예 | 예(residual은 BN state) | ImageNet 4 / 6, DVS 10, simulation step | 아니오 | 예 |
| SpikeGPT 2023 [read in full text] (부분: 3.3-3.5절, 표 1) | Y[t+1]=SN(σ(R[t])⊙(exp(W_f)⊙exp(K[t])⊙V[t]+A[t])/(exp(W_f)⊙exp(K[t])+B[t])), A·B는 재귀 state (Eq. 4-6) | token 순서 | M_R, M_K, M_V, 벡터 W_d, W_f | 예(A, B는 real) | T = sequence 길이(1024 / 3072), 별도 T 없음 | 예, 단방향 | 예(RWKV) |
| STMixer 2024 [read in full text] | STM(X)=SN(W_STM SN(BN(X W_V))), W_STM ∈ R^{N×N}, H group별 별도 W (Eq. 5) | token | 예 | 예(residual은 membrane) | T=1 / 4(static), 25-640(DVS 시험), simulation step | 아니오 | 예 |
| SWformer 2024 [read in full text] (부분: 초록, FATM, 표) | FATM: spiking Haar learner + ConvBN + pointwise conv. H_f=Spk(W_haar·Spk(I·W_haar^T)) | 공간 + 채널 | 예(Haar 자체는 고정) | ternary 사용(negative spike) | 4, 16, simulation | 아니오 | 예(저자 표현 "attention-free") |
| SpikF 2025 [read in full text] (부분: 본문 3장, 표, 부록 C) + 코드 | H_F^i=SF_g^{-1}(M_sel⊙F^i); F^i=SF_g(G_i); M_sel=SMP(SN(BN(LN(G_i)))); S_F^i=SN(BN(H_F^i)) (Eq. 8-12) | patch(token) 축 FFT | selector만 예 | 예(중간은 complex) | T_s = 패치의 zero-order hold 반복. 코드 `x.repeat(self.T,...)` | 아니오 | 예 |
| LRF-Dyn 2026 [read in full text] (부분: 5장, 표 1, 3) | X_n[t]=A⊙X_{n−1}[t]+ΓToken_n[t] (Eq. 12) + SN | token 위치 n, 각 simulation t마다 | A, Γ, C | SN으로 변환 | T = simulation, 그 안에서 n을 재귀 | 예, 단방향, membrane state | SSA 계산 제거(저자는 attention이라 부름) |
| SMixer 2026 [read in full text] | SMixer(X)=SN(W_M X) (Eq. 3), 저자들은 W_M을 "attention weight"라 부름 | token | 예 | 예 | ImageNet T=1 / 4, DVS 10 / 16, 시계열 4 | 아니오 | 예 |
| Vision SmolMamba 2026, preprint [read in full text] (부분: IV-C, 실험, 표, VI-A·B) | (H_f,H_b)=SN(DWConv(P,Reverser(P))); (Z_f,Z_b)=SN(SSM(H_f,H_b)); F=(Z_f+Reverser(Z_b))⊙Mask; U=SN(BN(Conv2d(F)))+X⊙Mask (Eq. 18-21) | token, (b, t, c) slice마다 | 예 | SN 뒤 예, SSM state는 real | T=4 또는 16, static token 위 추가 simulation | SSM이 token 축, 양방향 | 예 |
| Max-Former 2025 [read in full text] (부분: 서론, 4.3, 부록 B·F) | stage별 Identity / DWC / SSA. 앞 stage만 SSA 제거 | 공간 | DWC 예 | 예 | CIFAR·DVS·ImageNet, simulation | 아니오 | 부분(마지막 stage SSA 유지) |
| TEFormer 2026 [read in full text] | TEA: M_{i,j}=α(1−α)^{i−j}, Ṽ=MV_t. T-MLP: h_r=LIF(σ(W_fx X_r+W_fh h_{r+1})⊙h_{r+1}+(1−σ(·))⊙W_in X_r) (Eq. 2-5) | T 축(forward + backward) | 예 | 예 | 4 / 10, direct encoding 또는 frame | 아니오 | 아니오(QK + SSA 유지) |
| Lapis 2026 [read in full text] | D_ij=‖t_i^Q−t_j^K‖₁; A_ij=exp(−D_ij/τ) (Eq. 3-4) | token 쌍 | Q, K, V, τ | TTFS 한 번 발화 | T=15 / 20은 TTFS coding window | 아니오 | 아니오 |

#### 표 B. Task, baseline, 정확도, 에너지

| 논문 | Task | Baseline | 정확도(논문 표) | 에너지 주장 | 에너지 계산법 | q·k^T·v 계상 |
|---|---|---|---|---|---|---|
| SNN-MLP | ImageNet, COCO, ADE20K | AS-MLP, Swin 등 ANN | 81.9 / 83.3 / 83.5 (4.4G / 8.5G / 15.2G FLOPs) | 없음(FLOPs만) | 없음 | 해당 없음 |
| Attention-free Spikformer | CIFAR10-DVS, DVS128, CIFAR-10/100 | SSA 재현 79.7 / 98.2, 95.3 / 77.7 | 표 2: 1D-FFT 81.1 / 99.1, 2D-WT 81.6 / 98.5. 표 3: 94.9-95.2 / 77.0-77.3 | mJ 없음. memory 4-26% 감소, 학습 29-51%, 추론 61-70% 빠름 | 측정 시간만 | 에너지 미계산 |
| Spiking MLP | ImageNet, CIFAR, CIFAR10-DVS | ResNet SNN, Spikformer-8-512 | MLP-SPE-T 66.39(T=4), B 71.64(T=6). Spikformer 73.38보다 낮음 | 1.12 mJ vs 11.58 mJ(표 VI) | sTA × 0.9 pJ, 곱셈 × 4.6 pJ | Spikformer 값의 산출 방식은 페이지에 없음 |
| SpikeGPT | Enwik8 등 언어 | Vanilla GPT, GRU 등 | 표 본문 미확인 | 32.2× 적은 연산, 33.2× 적은 에너지 | R̂·연산수. attention 함수는 E_MAC·7Td | 자기 함수 f를 MAC으로 계상 |
| STMixer | CIFAR-10/100, ImageNet, CIFAR10-DVS | Spikformer, Spikingformer 등 | ImageNet 8-512: 73.82(T=1) vs Spikformer 73.38(T=4). CIFAR-100: 81.87(T=4) vs 77.86 | 2.20 mJ(T=1) vs 11.57 mJ(T=4), T가 다름 | Eq. 4: E_AC·ΣSOP_Conv + E_MAC·(FLOP¹_Conv + FLOP^L_Linear) | 식에 항 없음. baseline은 논문 값 인용 |
| SWformer | ImageNet, CIFAR, DVS, N-Caltech 등 | SpikFormer, SD Transformer | 6-512: 74.98 vs 72.46 | 3.58 mJ vs 9.41 mJ | supplementary(미확인) | 미확인 |
| SpikF | ECL, Weather, ETTh1/2, ETTm1/2, Traffic, Exchange, Solar | iTransformer, PatchTST 등 ANN, iSpikformer(Solar) | ETTh2 MSE .372, ETTm2 .281 (L=96, H 평균) | MAE 1.9% 감소, total energy 3.16×, 표 2 기준 operational 75.05% 감소(ECL) | SOPs=α·T·FLOPs, 복소 곱은 6 SOP | 에너지 표는 ANN 모듈만 비교, 스파이킹 attention 미계상 |
| SMixer | ImageNet, CIFAR, DVS, COCO, 시계열 4종 | SpikformerV2, QKFormer, SDT-V3 교체쌍 | 표 2 교체쌍 아래 참조 | 교체쌍 0.65-0.85×(내 계산) | Eq. 12: 4.6 pJ·MACs₁+0.9 pJ·ΣACs_l, L은 linear 층 수, BN 무시 | 식에 항 없음. "the power data of other works are obtained from related papers" |
| Vision SmolMamba | ImageNet, CIFAR, DVS | Spikformer, SDT, Vim 등 | CIFAR-10 2-256: SSA 93.7, Spiking Mamba 94.7, 전체 94.6 | 91.4 / 48.2 / 31.9 µJ(SPS 제외, 표 V) | E_MAC·MACs_SSM + E_AC·SOPs (Eq. 29-30) | SSA baseline 산출 세부는 미확인 |
| Max-Former | CIFAR, DVS, ImageNet | Spikformer, QKFormer 등 | ImageNet 82.39 vs 74.81 | 14.87 vs 21.48 mJ | 부록 Eq. 23: E_MAC·FLOP¹ + E_AC·(ΣSOP_Conv + ΣSOP_FC) | 식에 항 없음 |
| LRF-Dyn | ImageNet, 분할 | Spikformer, QKFormer | Spikformer-8-512 73.38→74.51, HST-10-512 82.04→82.48 | 메모리 O(d²)→O(kd) | 읽은 범위에 에너지 식 없음 | 미확인 |
| TEFormer | CIFAR, SVHN, DVS 계열, sCIFAR, sMNIST, SHD, HMDB51-DVS, UCF101-DVS | Spikformer, SDT, QKFormer, TIM | CIFAR-10 96.24 vs 95.91(baseline) | 5.18 mJ(MR=4), 2.39 mJ(MR=2). Spikformer 3.32, SDT 1.09 | 표 9, E_MAC + E_AC | 표에 분해 없음 |
| Lapis | CIFAR-10/100, ImageNet | 내부 kernel 비교, DT-SNN, A2S | CIFAR-10 96.56(dot product 97.09), ImageNet 83.25(6 bit) | attention path 100.01→6.92 mJ(dense 대비 14.5×) | 45 nm, N_rel=LN²D를 희소성 없이 전부 계상 | 자기 relation은 계상. 타 SNN은 "under the operation-level models used by the corresponding studies" |

#### 비교 기준 3편(attention 유지, 에너지 관행의 기준)
- **Spikformer**는 SSA 비용을 식에 넣습니다. "E_Spikformer = E_MAC × FL¹_SNN Conv + E_AC × (ΣSOP_SNN Conv + ΣSOP_SNN FC + ΣSOP_SSA)" (Eq. 22). 그림 1은 SSA 한 번 계산을 5.5×10⁶ SOP(4.95 µJ)로 적고 VSA는 77×10⁶ FLOP(354.2 µJ)로 적습니다. [read in full text] (부분: 그림 1, 부록 C.2) — [Spikformer](https://arxiv.org/abs/2209.15425)
- **Spike-driven Transformer(SDT)**: SDSA=SN(SUM_c(Q_S⊗K_S))⊗V_S. "the mask (Hadamard product) operation does not consume energy". SSA 비판은 "these methods not only yield large integers in the output". 표 1 계상은 f(Q,K,V)=E_AC·T·R̂·ND입니다. [read in full text] (부분: 1, 3장, 표 1, 4, 부록 S3) — [SDT](https://arxiv.org/abs/2307.01694)
- **QKFormer**: A_t=SN(Σ_j Q_{i,j}), X′=A_t⊗K (Eq. 6). 에너지 식은 ΣSOP_QKTA + ΣSOP_SSA를 포함합니다. ImageNet 85.65%(64.96M)로 attention 계열 최고 수치를 주장합니다. [read in full text] (부분: 초록, 3장, 에너지 부록) — [QKFormer](https://arxiv.org/abs/2403.16552)

#### 논문별 핵심 문장
- **Attention-free Spikformer**: "suggests that SSA may not be the principal factor driving the Spikformer's performance." "we argue that the utilization of SSA is unnecessary in the context of SNNs." 표 1 복잡도는 SSA O(N²d) 또는 O(Nd²), 1D-FFT O(N log N)입니다. "In static datasets, a 2D image ... needs to be repeated T times". 파라미터는 9.32M→6.96M입니다. 표에 ± 값은 없습니다. [read in full text] — [Wang et al.](https://arxiv.org/abs/2308.02557)
- **STMixer**: SSA를 비동기 칩 관점에서 비판합니다. "Calculating two spike matrix multiplications or a spike matrix passing through the max-pooling layer could result in significant differences in the output." 유도(Eq. 6)는 Q, K의 activation을 빼면 XWX^TV가 되고 "the objective of WSTM is to gradually fit the attention map during the training process."입니다. 머리: "The classification head will introduce floating-point matrix multiplication, so it needs to be deployed on the CPU." 에너지 한계도 스스로 적습니다: Eq. 4는 "neglects the overhead of memory scheduling". [read in full text] — [STMixer](https://proceedings.neurips.cc/paper_files/paper/2024/file/e8c20cafe841cba3e31a17488dc9c3f1-Paper-Conference.pdf)
- **STMixer 코드 확인**: TokenMixer는 LIF→Conv1d→BN→LIF, STM(head별 Linear(N→N)), LIF→Conv1d→BN 순서입니다. Encoder는 `x = x + self.token_mix(x)`이므로 residual은 real 값 덧셈입니다. SSA로 바꾸는 줄이 주석으로 남아 있습니다. [read in full text] — [models/STMixer.py](https://github.com/brain-intelligence-lab/STMixer_demo)
- **SMixer**: "replacing SSA with Spiking-token Mixer yields comparable accuracy to the original frameworks at similar parameters". 주 기여는 mixer가 아니라 pruning(DSTSP)입니다. 표 3의 SMixer CIFAR 값(95.49 / 96.01, 80.00 / 81.87)은 STMixer 표 1의 값과 같습니다. [read in full text] — [SMixer](https://proceedings.iclr.cc/paper_files/paper/2026/file/aaeb635296295aa7c9f8b91c5e10960d-Paper-Conference.pdf)
- **SMixer 표 2 교체쌍**(내 계산, 정확도 차 / 에너지 비): V2-8-384 T=1 +0.97 / 0.65×, V2-8-512 T=1 +0.11 / 0.75×, V2-8-384 T=4 +0.32 / 0.78×, V2-8-512 T=4 +0.07 / 0.85×, HST-10-384 T=1 +0.51 / 0.82×, HST-10-512 T=1 −0.02 / 0.85×, SDT-V3 S −0.05 / 0.84×, M +0.15 / 0.85×, L −0.55 / 0.84×. — [SMixer](https://proceedings.iclr.cc/paper_files/paper/2026/file/aaeb635296295aa7c9f8b91c5e10960d-Paper-Conference.pdf)
- **SNN-MLP**: "The t in the formula represents the index of patches instead of time steps in our design." "We replace the output 1 with u, so the full-precision information is reserved." 융합은 `x = mlp(norm(x_v + x_h))`입니다. 표 4의 활성값은 "fp32"입니다. [read in full text] — [SNN-MLP](https://arxiv.org/abs/2203.14679)
- **Spiking MLP가 SNN-MLP를 평가한 문장**: "this model does not conform to a fully spiking architecture", "due to the use of floating-point valued neurons, the model has no significant advantage in terms of energy cost compared with other ANNs". 자기 모델은 "all matrix multiplications involve a real-valued matrix and a binary matrix"입니다. Spikformer에 대해서는 "achieve significantly higher accuracy but consume over ten times the energy"라고 적어 정확도 열세를 인정합니다. [read in full text] (부분) — [Li et al.](https://arxiv.org/abs/2306.12465)
- **SpikeGPT**: "Input tokens are instead streamed in sequentially to our attention mechanism (as with typical SNNs)." "the optimal approach is not to introduce a new dimension, but to leverage the existing sequence dimension for forward propagation." "both naturally operate in a unidirectional manner." [read in full text] (부분) — [SpikeGPT](https://arxiv.org/abs/2302.13939)
- **Vision SmolMamba**: "the spiking SSM performs bidirectional selective scans along the token dimension for each (b, t, c) slice." 게이트에 대해: "we remove the gating branch to eliminate redundant dense activations. The binary nature of spike generation naturally provides an implicit gating mechanism". non-spike 연산: "a small number of MAC operations arising from multiplications involving the parameters Ā, C, and the hidden state h". [read in full text] (부분) — [Vision SmolMamba](https://arxiv.org/abs/2604.25570)
- **LRF-Dyn**: "It closely parallels the charge–fire-reset dynamics of spiking neurons". "our model eliminates the SSA computation and only requires storing the membrane potential at each position". [read in full text] (부분) — [LRF-Dyn](https://arxiv.org/abs/2603.19290)
- **SpikF**: "we introduce an attention-free framework". 비판: "their inherent permutation-invariance renders them inadequate in capturing positional information in time-series data". 비스파이크 요소: spiking max-pooling, Hadamard(M_sel⊙F), complex 값, "The resulting real-valued time-series are the input to the LIF neurons". 복소 곱 계상: "a complex multiplication is considered as 6 SOPs (4 multiplications and 2 additions)". 코드에는 T축으로 합산한 selector(`selector.sum(dim=0)`)도 있습니다. [read in full text] (부분) — [SpikF](https://proceedings.mlr.press/v267/wu25m.html), [models/SpikF.py](https://github.com/WWJ-creator/SpikF)
- **TEFormer**: "the first Spiking Transformer framework that achieves bidirectional temporal fusion". 한계: "TEFormer targets window-based recognition rather than strictly causal online inference." 직접 부호화 비판: "It repeats the same static input for T timesteps, allowing temporal accumulation but also introducing severe temporal redundancy, since different timesteps contain nearly identical signals." [read in full text] — [TEFormer](https://arxiv.org/abs/2601.18274)
- **Lapis**: "removes all multiplication between query and key channels". ANN-to-SNN conversion 방식이고 ImageNet은 pretrained ViT에서 출발합니다. "The temporal relation is deliberately counted densely over all token pairs." [read in full text] — [Lapis](https://arxiv.org/abs/2608.11865)
- **Meta-SpikeFormer** (attention이 필요하다는 주장은 하지 않음): "(2022a;b) argue that compared with a specific token mixer, a genera [sic] meta Transformer block ... is more essential than a specific token mixer". 앞 두 stage를 Conv로 두고 뒤 두 stage에만 SDSA를 둡니다. ImageNet 80.0%. [read in full text] (부분: 초록, 3장) — [Meta-SpikeFormer](https://arxiv.org/abs/2404.03663)

#### 그 밖에 찾은 논문
- ASFNOformer는 SSA를 frequency-domain token mixer로 바꾼다고 합니다. MDPI가 403을 돌려주어 본문을 읽지 못했습니다. [secondary summary] — [ASFNOformer](https://doi.org/10.3390/electronics14244860)
- Spiking Point Mamba(ICCV 2025)는 3D 점구름용이며 에너지 3.5× 감소를 주장합니다. [secondary summary] — [papernotes](https://en.papernotes.org/ICCV2025/3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis/)
- attention을 개선하는 쪽(SLI-ACF, LSFormer, SCLA-BCP, SAFformer)은 attention-free가 아닙니다. [abstract only] — [SLI-ACF](https://arxiv.org/abs/2608.19238), [LSFormer](https://arxiv.org/abs/2605.13887), [SCLA-BCP](https://arxiv.org/abs/2608.08541), [SAFformer](https://arxiv.org/abs/2605.08270)
- 2610.15968은 softmax-free spiking attention을 two-level system의 영도 극한으로 다루는 이론 논문이며, attention 기여도 측정은 아닙니다. [abstract only] — [Softmax-free attention](https://arxiv.org/abs/2610.15968)

### Inferences
- 에너지 비교의 관행은 통일되어 있지 않습니다. Spikformer와 QKFormer는 attention 항을 식에 넣고, Max-Former와 SMixer의 식에는 없으며, STMixer와 SMixer는 baseline 수치를 원논문에서 그대로 가져옵니다. 따라서 논문 간 mJ 비교는 같은 계산이 아닙니다.
- STMixer의 5× 전후 에너지 차이는 대부분 T=1 대 T=4에서 옵니다. 같은 T로 비교한 SMixer 표 2의 교체쌍(0.65-0.85×)이 mixer 자체 효과에 더 가깝습니다. 다만 그 표도 baseline 값이 인용값이라 같은 코드의 측정이 아닙니다.
- Spikformer 그림 1의 값으로 어림하면 attention 곱 자체는 총 에너지의 일부입니다. 8.91 µJ(그림의 두 값 합) × 8 head × 8 block × T 4 ≈ 2.28 mJ로 11.57 mJ의 약 20% 이하입니다. 8 head라는 값과 두 값을 합치는 것은 제 가정이며, 그림의 두 값은 계산 순서에 따른 대안일 수도 있습니다.

### Gaps
- SWformer와 Spiking MLP의 Spikformer 에너지가 attention 곱을 어떻게 세는지는 확인하지 못했습니다. SWformer는 supplementary에 있고 읽지 못했습니다.
- SMixer 표 4의 Spikformer 행이 새로 돌린 값인지 페이지에 적혀 있지 않습니다. 시드 수도 읽은 범위에 없습니다.
- LRF-Dyn의 에너지 식은 읽은 범위에 없었고, SpikeGPT의 정확도 표는 읽지 않았습니다.

## Q2. 시계열이나 biosignal에 적용되었는가, vision만인가?

### Takeaway
대부분 vision입니다. 시계열은 SMixer(부분 실험), SpikF, SpikF-GO가 있습니다. 시계열에서 attention-free 쪽이 더 좋다는 주장이 이미 나와 있습니다. Biosignal 분류에 쓴 spiking mixer는 찾지 못했습니다.

### Cited Findings
- SMixer 표 4는 Metr-la, Pems-bay, Solar, Electricity의 horizon 6 / 24 / 48 / 96을 다룹니다. "the SMixer paradigm consistently outperforms Spikformer". 설정은 "A fixed setting of 4 time steps is used for all SNNs", 2 block, D=256이고 CPG 위치 부호화 모델의 SSA를 SMixer로 바꾼 것입니다. [read in full text] — [SMixer](https://proceedings.iclr.cc/paper_files/paper/2026/file/aaeb635296295aa7c9f8b91c5e10960d-Paper-Conference.pdf)
- 내 계산: 같은 표에서 SMixer R²가 Spikformer보다 높은 칸이 15/16, 동률 1(Electricity H6), 평균 차이 +0.032입니다. — [SMixer](https://proceedings.iclr.cc/paper_files/paper/2026/file/aaeb635296295aa7c9f8b91c5e10960d-Paper-Conference.pdf)
- Lv et al.의 iSpikformer R²(같은 4개 데이터, 같은 horizon, 3 seed 평균)는 Metr-la .817 .618 .440 .279, Pems-bay .879 .744 .687 .674, Solar .961 .876 .795 .738, Electricity .977 .974 .972 .963입니다. [read in full text] (부분: 표 2) — [Lv et al.](https://arxiv.org/abs/2402.01533)
- 내 계산: SMixer 표 4 값과 대조하면 SMixer가 iSpikformer보다 낮은 칸 14, 높은 칸 1, 같은 칸 1입니다. 논문이 다른 실행의 비교이고 split 동일 여부는 확인하지 않았습니다. — [SMixer](https://proceedings.iclr.cc/paper_files/paper/2026/file/aaeb635296295aa7c9f8b91c5e10960d-Paper-Conference.pdf), [Lv et al.](https://arxiv.org/abs/2402.01533)
- SpikF(ICML 2025)는 시계열 전용 attention-free 모델입니다. 표 1(L=96, H 평균)의 MSE / MAE는 ETTh2 .372 / .394, ETTm2 .281 / .320이고, iTransformer는 .383 / .407, .288 / .332입니다. 표 5(Solar 단기)는 iSpikformer보다 오차가 2.3% 낮다고 합니다. [read in full text] (부분) — [SpikF](https://proceedings.mlr.press/v267/wu25m.html)
- SpikF-GO(ECML-PKDD 2026)는 제3자 논문으로 "Among the baselines, SpikF is the strongest SNN baseline"이라고 적습니다. Spikformer w/ CPG, iSpikformer, TS-Former도 비교군에 있습니다. 표 1의 데이터에 ECG가 있으나 forecasting 벤치마크입니다. [read in full text] (부분: 초록, 2, 5장) — [SpikF-GO](https://arxiv.org/abs/2606.13901)
- TEFormer는 speech(SHD)와 serialized image(sMNIST, sCIFAR)를 다루지만 attention을 유지합니다. [read in full text] — [TEFormer](https://arxiv.org/abs/2601.18274)
- Attention-free Spikformer, STMixer, Spiking MLP, SWformer, Max-Former, Vision SmolMamba, Lapis는 읽은 범위에서 vision(또는 DVS)만 다룹니다. 이 논문들의 텍스트에서 time series / ECG / EEG / forecast를 검색해도 실험 대상으로 나오지 않았습니다. SNN-MLP는 SNN 응용 예로 ECG를 언급할 뿐입니다. [read in full text] (부분 + 단어 검색) — [SNN-MLP](https://arxiv.org/abs/2203.14679)
- ECG 분류에서 mixer 구조는 비스파이킹 MS-MLP(MIT-BIH)가 있고, 스파이킹 쪽은 STDP 계열이 나왔습니다. spiking mixer + ECG 조합은 검색에 없었습니다. [secondary summary] — [MS-MLP](https://personalpages.surrey.ac.uk/w.wang/papers/WangGCW_EUSIPCO_2022.pdf), [neuromorphic arrhythmia](https://www.nature.com/articles/s41598-025-23248-9)

### Inferences
- SMixer의 "Spikformer보다 좋다"는 비교 대상이 token을 그대로 쓰는 Spikformer일 때의 결과로 보입니다. token을 변수 단위로 뒤집은 iSpikformer와 교차 비교하면 관계가 뒤집힙니다. 시계열에서는 mixer 종류보다 token 구성이 더 크게 작용할 수 있습니다. 논문 간 비교이므로 약한 근거입니다.
- 우리의 "ETT, MIT-BIH, AD에서 Spikformer-style과 동등"은 같은 front end로 짝지은 비교라는 점에서 SMixer 표 4(baseline 출처 미기재)와 성격이 다르고, 결론도 다릅니다. reviewer는 이 차이를 물을 것입니다.
- SpikF는 ETT에서 가장 가까운 attention-free 경쟁 모델입니다. 프로젝트 로그의 최근 커밋(07986f989)은 SpikF, iSpikformer, SpikeRNN을 "not run"으로 적고 있습니다.

### Gaps
- MIT-BIH, CinC 2017, AD에 spiking mixer를 쓴 논문은 찾지 못했습니다. 검색 횟수가 적으므로 없다는 증명은 아닙니다. DBLP와 OpenReview는 이 서버에서 막혀 있습니다.
- 시계열 비교 대부분은 표에 ± 또는 짝지은 검정이 없어서 차이의 신뢰도를 판단하기 어렵습니다.

## Q3. token mixing이나 attention의 기여를 실제로 측정했는가(ablation, 제거 후 재학습)? 결론은?

### Takeaway
"다른 mixer로 교체 후 재학습"은 많습니다. 결론은 대체로 "attention이 필수는 아니다"이고 차이는 ±1 %p 안팎입니다. "mixing을 완전히 제거"한 대조는 드뭅니다. 가장 가까운 SpikF의 w/o 행은 ETT에서 거의 차이가 없다고 보고합니다. 그런데 저자들은 이 숫자를 본문에서 해석하지 않습니다.

### Cited Findings

#### 교체 후 재학습
- STMixer 표 2(CIFAR-100, STMixer-4-384-16, T=1): 전체 79.55, IPSPS→SPS 78.97, STM→SSA 78.33, SML(STM)→SML(Conv) 79.07, SML→SDT 77.17. "SML is the most significant improvement method." "the performance of the STM module surpasses that of the SSA module (+1.22%)". 내 계산: 학습법 교체(−2.38)가 mixer 교체(−1.22)보다 큽니다. [read in full text] — [STMixer](https://proceedings.neurips.cc/paper_files/paper/2024/file/e8c20cafe841cba3e31a17488dc9c3f1-Paper-Conference.pdf)
- STMixer 표 3: head 수 H=1 79.02, 2 79.07, 4 79.39, 8 79.51, 16 79.55, 32 80.15. 표 4(CIFAR10-DVS, 구조 동일, STM↔SSA): T=25 학습 모델의 event-driven 시뮬레이터 정확도 50.70 vs 16.60, T=40은 54.10 vs 10.00. "events fail to meet each other and end in SSA". supernet 탐색(그림 4, 4000개 subnet)에서 STMixer가 좌상단에 위치한다고 주장합니다. [read in full text] — [STMixer](https://proceedings.neurips.cc/paper_files/paper/2024/file/e8c20cafe841cba3e31a17488dc9c3f1-Paper-Conference.pdf)
- SMixer 표 2(9쌍, 위 내 계산): 정확도 차이 −0.55 ~ +0.97, 에너지 0.65-0.85×. [read in full text] — [SMixer](https://proceedings.iclr.cc/paper_files/paper/2026/file/aaeb635296295aa7c9f8b91c5e10960d-Paper-Conference.pdf)
- Attention-free Spikformer 표 2-3: DVS에서 재현 SSA보다 높고(79.7 → 81.1-81.6, 98.2 → 98.4-99.1), static에서 조금 낮음(95.3 → 94.9-95.2, 77.7 → 77.0-77.3). 결론 문장은 "only nominal accuracy costs on static datasets"입니다. 파라미터 없는 mixer와 비교한 점이 핵심입니다. [read in full text] — [Wang et al.](https://arxiv.org/abs/2308.02557)
- Vision SmolMamba 표 V(CIFAR-10): SSA 93.7 / 91.4 µJ, Spiking Mamba 94.7 / 48.2 µJ, Full SST-TP 94.6 / 31.9 µJ. "all configurations achieve higher accuracy than the SSA baseline while maintaining lower energy cost". 내 계산: 0.53×, 0.35×. [read in full text] (부분) — [Vision SmolMamba](https://arxiv.org/abs/2604.25570)
- SWformer 표 4: NoHaar 78.89, base 79.31(CIFAR-100). CIFAR10-DVS NoHaar 80.9, 본문의 max-pool 제거 전 83.9. [read in full text] (부분) — [SWformer](https://arxiv.org/abs/2403.11138)
- Max-Former 표 4(CIFAR-100): Identity/DWC-3/SSA 82.65, SSA/SSA/SSA 82.23, Identity/Identity/SSA 81.28. CIFAR10-DVS: DWC-3/SSA 84.2, SSA/SSA 83.9. "this substitution leads to +0.42% performance gain". [read in full text] (부분) — [Max-Former](https://arxiv.org/abs/2505.18608)
- Max-Former 서론의 파라미터 없는 pooling 실험: "adopting Avg-Pooling (low-pass) for token mixing lowers performance to 76.73% on Cifar-100". Max-Pool은 "+2.39% improvement ..., making it surpass the well-tuned Spikformer [8] baseline by 0.97%". [read in full text] (부분) — [Max-Former](https://arxiv.org/abs/2505.18608)

#### 일부 또는 전체 제거
- SpikF 표 4(MSE / MAE, H 평균): 원본 ETTh1 .440 / .428, ETTm1 .388 / .385, Weather .245 / .265. SFS 제거(w/o) .441 / .428, .388 / .387, .246 / .266. iSSA로 교체 .452 / .440, .389 / .389, .247 / .268. 본문은 "the SFS shows a 1.6% performance enhancement compared to iSSA"만 말하고 w/o 행은 해석하지 않습니다. 제거 후에도 convolution 층은 feature extraction에 남습니다. [read in full text] (부분: 표 4, 8, 부록 C.2, 코드) — [SpikF](https://proceedings.mlr.press/v267/wu25m.html)
- MetaFormer(ANN, spiking 아님)의 identity token mixer는 ImageNet 74.3%입니다. [read in full text] (부분) — [MetaFormer](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_MetaFormer_Is_Actually_What_You_Need_for_Vision_CVPR_2022_paper.pdf)
- Max-Former 부록은 "employing average pooling operators to replace SDSA as the token mixer surprisingly results in substantial performance degradation from 61.0% to 41.2%"라고 하며 [41]을 출처로 듭니다. 이 PDF의 [41]은 Spike-driven Transformer인데, 그 arXiv 텍스트와 Meta-SpikeFormer 텍스트에서 41.2와 61.0을 찾지 못했습니다. 출처 추적 실패로 표시합니다. [secondary summary] — [Max-Former](https://arxiv.org/abs/2505.18608), [SDT](https://arxiv.org/abs/2307.01694)

#### Token 축 동역학 자체의 효과
- SNN-MLP 표 5(ImageNet): AS-MLP-T 81.3, dwconv + shift 81.66, dwconv + LIF 81.87. 표 7 group 수: 2 81.60, 4 81.87, 7 81.68, Inf(전체 token) 81.53. "the performance of global LIF significantly worse than group LIF". 원인 추정은 "the introduction of long-distance information with weak correlations"입니다. [read in full text] — [SNN-MLP](https://arxiv.org/abs/2203.14679)
- Vision SmolMamba 표 III: T를 sequence 길이(64)로 둔 SpikingSSM 84.5, P-SpikeSSM 78.3, 자기 모델(T=4) 95.7(CIFAR-10). "these 1D sequence-oriented baselines fail to achieve high accuracy on CIFAR10." 인용 baseline입니다. [read in full text] (부분) — [Vision SmolMamba](https://arxiv.org/abs/2604.25570)
- LRF-Dyn 표 3(CIFAR-100): LRF-Dyn 77.78-78.57, causal SSA 74.30-76.50, LRF-SSA 77.86-78.64. [read in full text] (부분) — [LRF-Dyn](https://arxiv.org/abs/2603.19290)
- TEFormer 표 5(CIFAR-10): baseline 95.91, +T-MLP 95.98, +TEA 95.85, 둘 다 96.24. 표 6의 방향 조합은 95.95-96.24입니다. 양방향 효과가 0.3 %p 수준이고, 에너지는 MR=4에서 5.18 mJ로 Spikformer(3.32 mJ)보다 큽니다. [read in full text] — [TEFormer](https://arxiv.org/abs/2601.18274)

#### Attention 쪽 근거(반대 방향)
- Lapis 표 3(CIFAR-10, conversion, 같은 backbone): dot product(softmax) 97.09, Laplacian 96.56, Gaussian 91.44, Hamming 87.92. "confirming that graded temporal distance is necessary for competitive accuracy". [read in full text] — [Lapis](https://arxiv.org/abs/2608.11865)
- QKFormer는 attention을 유지하고 ImageNet 85.65%입니다. SMixer 표 2에서도 가장 큰 SDT-V3-L은 교체 후 −0.55입니다. [read in full text] (부분) — [QKFormer](https://arxiv.org/abs/2403.16552), [SMixer](https://proceedings.iclr.cc/paper_files/paper/2026/file/aaeb635296295aa7c9f8b91c5e10960d-Paper-Conference.pdf)
- Spiking MLP(attention 없음, 66.39-71.64)는 Spikformer(73.38)보다 낮습니다. 비슷한 시기의 mixer가 항상 동등한 것은 아닙니다. [read in full text] (부분) — [Li et al.](https://arxiv.org/abs/2306.12465)

### Inferences
- 문헌의 결론을 정리하면 "SSA는 필수가 아니다"(Wang 2023), "동등하고 비동기 친화적이다"(Deng 2024, Lu 2026), "앞 stage에서는 SSA가 해롭거나 불필요하다"(Fang 2025, Max-Former)입니다. 다만 차이 대부분이 ±1 %p이고 시드 수가 표에 적히지 않은 경우가 많습니다.
- 교체 실험 다수는 다른 요소를 함께 바꿉니다. STMixer의 표 1은 T, IPSPS, SML이 함께 다르고, 표 2의 STM→SSA는 그 안에서 T=1 한 조건입니다. 같은 front end, 같은 seed로 짝지은 교체 실험은 드뭅니다.
- ETT에서 SpikF의 w/o가 거의 그대로라는 점은 "ETT에서는 token mixer가 거의 일하지 않는다"는 가설과 맞습니다. 우리 ETT 동등 결과도 같은 이유일 수 있으므로 identity 대조 없이는 "mixer가 attention을 대신한다"고 말하기 어렵습니다. convolution이 남아 있어 완전한 대조는 아닙니다.
- SNN-MLP의 global LIF 열세와 SmolMamba의 T=64 baseline 수치는 이미지 데이터의 결과입니다. 순서가 의미 있는 시계열에 그대로 옮겨지지는 않지만, T = N이 자동으로 이득이 아니라는 선행 근거입니다.

### Gaps
- 스파이킹 모델에서 token mixer를 전부 identity로 두고(다른 층 유지) 재학습한 뒤 attention 모델과 짝지어 비교한 논문은 찾지 못했습니다. Max-Former의 Identity/Identity/SSA는 마지막 stage SSA가 남습니다.
- SLI-ACF는 "SSA only, SLI only"를 비교하지만 숫자가 그림에만 있어 읽지 못했습니다. 저자는 조합이 어느 한쪽보다 높다고 씁니다. — [SLI-ACF](https://arxiv.org/abs/2608.19238)
- 61.0 → 41.2의 원 출처를 확인하지 못했습니다.

## Q4. SpikMixer와의 overlap: 어떤 요소가 이미 있고 무엇이 없는가, reviewer는 "attention-free spiking mixer가 Spikformer와 같은 정확도"를 어떻게 볼 것인가?

### Takeaway
개별 요소 중 attention-free, T_h 반복 축, token-axis 동역학, 양방향은 모두 선행 사례가 있습니다. 찾지 못한 것은 두 stream의 membrane 차이로 만든 spiking gate와, LIF 자체를 양방향 token-axis mixer로 쓰는 조합입니다. 동등 정확도 주장만으로는 기여가 되기 어렵고, 에너지 0.98×는 문헌의 0.53-0.85×보다 약합니다. 대신 짝지은 비교의 null 결과는 문헌의 긍정 주장과 대비되는 가치가 있을 수 있습니다.

### Cited Findings

#### SpikMixer 구조(코드 확인)
- 구조는 docstring과 코드에서 확인했습니다. T_h 반복 → embedding(T_h 축) → mixer(N 축, `x.permute(2, 1, 0, 3)`, [N, B, T_h, D]) → `mixer`와 `mixer_rev` 별도 weight → 융합 → MLP(T_h 축). muxlin 융합은 `m = sd * a + (1. - sd) * torch.minimum(f, r)`이고 `d = (hf - hr).abs()`가 BN을 거쳐 `sd_neuron`으로 갑니다. 기본값은 `mixer_layers=2`입니다. [read in full text] — [spk_mixer.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/spk_mixer.py)
- 측정 결과(이 조사의 지시문 기준): ETT, MIT-BIH, AD에서 Spikformer-style과 같은 정확도, 에너지 0.98×, CinC 2017 fold 0에서만 +2.16 %p. 이 수치를 다시 확인하지는 않았습니다.

#### 요소별 overlap

| SpikMixer 요소 | 가장 가까운 선행 | 판정 | 남는 차이 |
|---|---|---|---|
| Attention-free | Wang 2023, Deng 2024, Lu 2026, Wu 2025, Fang 2024, SmolMamba 2026, Li 2023, SNN-MLP 2022 | 완전히 있음 | 없음 |
| Token-axis 동역학 | SNN-MLP(full-precision LIF), SpikeGPT(RWKV + SN, T = 길이), LRF-Dyn(X_n=A⊙X_{n−1}+ΓToken_n), SmolMamba(SSM) | 대부분 있음 | binary spike와 reset을 가진 LIF 자체가 N 축을 돌며 mixer 역할을 하는 사례는 못 찾음. SNN-MLP는 저자 스스로 fp 출력, 나머지는 real state |
| 양방향(token 축) | SmolMamba(Reverser, 두 SSM 모듈, 합산), Yin 2021 bi-SRNN(forward / backward SRNN, TIMIT) | 있음 | LIF 기반 mixer + 게이트 융합은 못 찾음. TEFormer는 T 축이라 다른 축. SNN-MLP와 Li 2023의 두 갈래는 직교 축 |
| T_h 반복 축 | Spikformer류 전부(static 입력 T회 반복), SpikF(ZOH, 코드 `x.repeat`), TEFormer의 direct encoding 설명, SWformer | 완전히 있음 | [N, B, T, D] 배치도 SmolMamba의 "(b, t, c) slice"와 동일 형태 |
| Stream 불일치 gate | 없음. 근접 사례: SDSA·QKTA의 spike mask, SpikF의 spiking selector, TEFormer의 sigmoid gate | 못 찾음 | |H_f−H_r|을 입력으로 쓰는 gate는 검색과 정독 범위에 없음. SmolMamba는 오히려 gate가 불필요하다고 주장 |
| Spiking 융합 | Li 2023 Y=BN(W_f Concat[...])+X, SmolMamba SN(BN(Conv2d(F))), SNN-MLP 합 | learned 융합은 있음 | F AND R, s_d MUX 형태는 못 찾음 |
| Residual IAND / OR / ADD | 대부분 ADD(spike 또는 membrane) | 읽은 범위에 IAND 없음 | NSMT 계열 고유로 보이나 전수 확인은 아님 |

출처: [SNN-MLP](https://arxiv.org/abs/2203.14679), [SpikeGPT](https://arxiv.org/abs/2302.13939), [LRF-Dyn](https://arxiv.org/abs/2603.19290), [Vision SmolMamba](https://arxiv.org/abs/2604.25570), [TEFormer](https://arxiv.org/abs/2601.18274), [SDT](https://arxiv.org/abs/2307.01694), [QKFormer](https://arxiv.org/abs/2403.16552), [SpikF](https://proceedings.mlr.press/v267/wu25m.html), [Li et al.](https://arxiv.org/abs/2306.12465) (이상 [read in full text], 일부는 부분 읽음)

- Yin et al.의 bi-SRNN: "we use two SRNN layers in the network, reading the sequence from the forward and backward" (TIMIT). [read in full text] (부분) — [Yin et al.](https://arxiv.org/abs/2103.12593)
- 양방향 + membrane 차이 gate를 직접 찾는 검색에서 해당 논문이 나오지 않았습니다. [secondary summary](검색 1회) — [검색 결과 중 근접 논문](https://arxiv.org/abs/2103.12593)

### Inferences

#### Reviewer가 할 법한 지적
1. **새 주장이 아닙니다.** "SSA 없이 같은 정확도"는 Wang 2023부터 반복되었습니다. SMixer는 SpikformerV2, QKFormer, SDT-V3에 넣어 "comparable accuracy"를 이미 보였고 시계열까지 다룹니다.
2. **에너지 기준이 더 높습니다.** 문헌은 mixer 교체로 0.65-0.85×(SMixer), 0.53×(SmolMamba, pruning 전)를 보고합니다. 0.98×는 "복잡도를 들인 만큼 얻는 것이 없다"로 읽힐 수 있습니다. 코드 기본값 기준으로 forward 2층 + reverse 2층 + 융합 Linear이고, SSA는 Q, K, V + Dense(Attention-free Spikformer Eq. 2-3)이므로 층 수로는 줄지 않습니다. 실제 실행 설정과 발화율은 제가 확인하지 않았습니다.
3. **더 단순한 대조가 필요합니다.** 첫 질문은 "SMixer의 SN(W_M X) 하나보다 나은가"일 것입니다. Spikformer-style 하나만 대조하면 이 질문에 답하지 못합니다. 파라미터 없는 FFT(Wang 2023)와 SpikF의 S-FFT, identity도 같은 부류의 대조입니다.
4. **spike-driven 주장은 방어하기 어렵습니다.** STMixer의 논리를 적용하면 두 spike의 곱은 비동기 하드웨어에서 깨집니다. `minimum(f, r)`, `sd * a`, IAND `x * (1 - m)`은 모두 spike끼리의 element-wise 동시 발화 연산입니다. 동기 하드웨어 전제의 SDT 논리("mask ... does not consume energy")로는 방어되지만 event-driven이라고 말하기는 어렵습니다. `(hf - hr).abs()`는 membrane을 층 사이로 직접 보내는 real 값입니다.
5. **역방향은 window 전체가 필요합니다.** `xm.flip(0)`은 N개 token이 다 모여야 합니다. TEFormer도 이 한계를 적었고, SpikeGPT는 SNN이 "can only generate spikes based on information from previous time-steps"라고 씁니다. 분류나 고정 window 예측에서는 문제가 없지만 온라인 추론과 신경모사 동기를 함께 내세우면 충돌합니다.
6. **T = N은 신기하지 않습니다.** SpikeGPT는 "as with typical SNNs"라고 적었고, SmolMamba는 sequence 모델의 T를 길이로 정의합니다. 프로젝트 메모의 SHD 판단("native bins make T = N standard")과도 같습니다.
7. **gate의 근거가 약하면 공격받습니다.** SmolMamba의 "implicit gating mechanism" 주장이 반론이 됩니다. gate on / off, OR 융합과 합산 대비 짝 비교 없이는 기여로 인정받기 어렵습니다.
8. **CinC 단일 fold**: +2.16 %p가 fold 0 하나라면 reviewer는 다른 fold를 요구합니다. 사용자 결정(11-04)으로 CinC는 main 결과도 아닙니다.
9. **에너지 산식을 공개해야 합니다.** 문헌의 산식이 제각각이므로, Spikformer-style 쪽에 Spikformer Eq. 22처럼 SOP_SSA를 넣었는지 명시해야 합니다. 저는 프로젝트의 에너지 코드를 읽지 않았습니다(지시 범위 밖).

#### 방어할 수 있는 지점
- **짝지은 null 결과.** 문헌의 우위 주장에는 T, patch splitting, 학습법 변경이 섞이거나(STMixer 표 1) baseline 출처가 불분명한 경우(SMixer 표 4)가 있습니다. 같은 front end와 seed로 시계열, MIT-BIH, AD에서 차이가 없다는 측정은 그 자체로 정보이고, "시계열에서 attention-free가 낫다"는 주장(SMixer, SpikF)에 대한 재검증이 됩니다.
- **biosignal과 AD 영역.** 이 분야의 spiking mixer 비교는 찾지 못했습니다(Q2 Gaps).
- **새로운 구조 요소.** stream 불일치 gate와 LIF 양방향 mixer는 선행을 못 찾았습니다. 다만 성능 기여를 보이지 못하면 구조의 신규성은 약한 근거입니다.
- **권장 표현.** "attention-free로 동등"이 아니라 "기존 mixer들과 같은 조건에서 비교했을 때 무엇이 다른가"로 질문을 옮겨야 합니다. 최소 대조군은 SMixer식 W_M, identity, SpikF입니다.

### Gaps
- OpenReview와 DBLP가 막혀 있어 ICLR 2025-2026 전수 조사는 못 했습니다. "못 찾음"은 "없음"이 아닙니다. 특히 gate on stream disagreement는 검색 1회와 정독한 논문 범위의 판단입니다.
- Spiking Mamba 계열(SpikingSSM, P-SpikeSSM, Spiking Point Mamba)과 그 밖의 양방향 모델은 원문을 읽지 않았습니다.
- 우리 구현의 실제 run 설정(mixer_layers, 융합 종류), 층별 발화율, 에너지 코드의 SSA 계상 여부는 이 조사에서 확인하지 않았습니다.
- STMixer의 비동기 시뮬레이터는 저자들의 자체 C++ 구현이라 외부 재현 여부를 모릅니다.
