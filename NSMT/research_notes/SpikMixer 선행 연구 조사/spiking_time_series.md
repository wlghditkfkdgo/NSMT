# SNN forecasting 선행 연구: spiking time axis 와 sequence axis 를 어떻게 다루는가 (SpikMixer 위치 판단용)

표기 규칙. 모든 주장에 근거 수준을 붙인다.
- `[read in full text]`: 논문 본문을 직접 읽음. PDF 는 `pdftotext` 로 추출해 표와 식을 확인함.
- `[read in full text: repository code]`: 저장소 코드 또는 설정 파일을 직접 읽음. 논문 본문과 별개의 근거임.
- `[abstract only]`: 초록 또는 서지 정보만 확인함. 서지 정보만 있으면 그렇게 적음.
- `[secondary summary]`: 검색 요약이나 제3자 요약에서만 확인함.
- `[given in assignment]`: 의뢰문에 적힌 우리 측 수치임. 이 조사에서 다시 측정하지 않음.

용어. `T` 는 spiking time step 수, `N` 은 token (patch) 수, `L` 은 입력 길이, `H` 는 예측 길이다. "T = N" 은 neuron 이 token 순서를 따라 한 칸씩 진행한다는 뜻이다. "반복 T" 는 같은 값을 T 번 복사해 넣고 neuron 이 그 복사본 축을 따라 진행한다는 뜻이다.

## 질문 1. 논문별 mechanism, T 축의 의미, encoder, head, 학습 규칙, energy 산출 방식

### Takeaway
SNN forecasting 의 주류는 "patch 또는 convolution 으로 값을 만들고, 이를 T 번 반복해 LIF 에 넣고, sequence 순서는 FFT, attention, convolution, MLP 같은 비신경 연산으로 처리하는" 구조이며 T 는 대체로 시간이 아니라 반복축 또는 convolution kernel 축이다. Neuron 이 sequence 를 따라 진행하는 모델도 분명히 있지만 (Spike-GRU, SpikeLite 의 encoder, SpikySpace, NSMT 의 Neocortex, 단변량 S4D) token 축 Linear mixing 뒤에 같은 축으로 neuron 을 진행시키고 순방향과 역방향을 spiking gate 로 합치는 조합은 찾지 못했다.

### Cited Findings

#### T 축 요약표 (아래 카드의 근거를 모은 것)

| 모델 | T 값 | T 축의 실제 의미 | Neuron 이 입력 순서를 따라 진행하는가 | Sequence 순서를 다루는 주 연산 |
|---|---|---|---|---|
| iSpikformer (Lv 2024) | 4 | convolution 출력 channel 축 | 아니오 | 변수 token 사이 SSA, L 은 feature 로 투영 |
| Spike-TCN (Lv 2024) | 4 | 위와 같음, 매 시계열 step 마다 membrane 0 | 아니오 | convolution |
| Spike-GRU / Spike-RNN (Lv 2024) | 4 (L 위치마다) | 총 L x T 로 중첩, membrane 유지 | 예, 주 mechanism | neuron 과 recurrent 상태 |
| TS-LIF (ICLR 2025) | 설정상 4, 본문은 "history length 에 맞춤" | backbone 에 따름 | TS-GRU 는 예, TS-former 는 아니오 | 2-compartment neuron + backbone |
| SpikF (ICML 2025) | ETT 16, 변수 많은 데이터 4 | patch 값의 zero-order hold 반복 | 아니오 | patch 축 FFT 선택, patch 축 convolution |
| SpikF 저장소의 SpikeRNN | 16 | 위와 같음 | 아니오, 이름과 달리 N 축 루프 없음 | token 축 Linear |
| SpikF-GO (ECML-PKDD 2026) | 4 | 반복 + step 별 scale, shift | 아니오 | node 축 FFT |
| SpikySpace (2026 preprint) | 3 | scalar 하나를 표현하는 spike train 길이 | state 가 L 을 따라 진행, membrane 은 아님 | spiking selective scan |
| SpikeSTAG (2025 preprint) | Ts, 값은 그림으로 탐색 | 값의 spike encoding | 실수값 LSTM 이 진행 | LSTM, graph, SSA |
| SpikeLite (2026 preprint) | encoder 는 L, attention 은 4 | encoder 는 관측 순서, 이후 반복 | encoder 에서 예 | 병렬 LIF, Linear, masked SSA |
| NSMT (제출본) | Hippocampus 는 입력을 T 번 반복 (코드상 T = num_patches), Neocortex 는 N | 반복축과 token 순서 양쪽 | Neocortex 에서 예 | spiking MLP + cross attention |
| Closed-loop S4D (2026 preprint) | P + F | 관측 순서, 단변량 | 예 | S4D 상태 |
| SpikMixer (우리) | 임베딩 4, mixer 안 N | 임베딩은 반복, mixer 는 token 순서 | mixer 에서 예, 순방향과 역방향 | token Linear + neuron 진행 + gate |

#### Lv et al., ICML 2024 (SeqSNN: iSpikformer, Spike-TCN, Spike-RNN, Spike-GRU)
- Venue: "Proceedings of the 41st International Conference on Machine Learning, PMLR 235, 2024". [read in full text] — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533); [PMLR](https://proceedings.mlr.press/v235/lv24d.html)
- 데이터: Metr-la, Pems-bay, Solar, Electricity, H = 6, 24, 48, 96. **ETT 는 본문에 없다.** 지표는 R^2 와 RSE 다. 그러므로 ETT 의 iSpikformer 수치는 이 논문에서 나온 것이 아니다. [read in full text] — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- LIF 식: U(t) = H(t - dt) + I(t), H(t) = V_reset S(t) + (1 - S(t)) beta U(t), S(t) = 1 if U(t) >= U_thr. [read in full text] — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- Temporal alignment: dT = Ts dt 로 시계열 한 step 을 Ts 조각으로 나눈다고 정의하고, encoder 가 "Ts x T x C possible spike events" 를 계산한다고 쓴다. 논문은 단순 복제가 "can disrupt the continuous nature" 한다고 비판한다. [read in full text] — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- Encoder 식: Delta 는 S = SN(BN(Linear(x_t - x_{t-1}))), Convolutional 은 S = SN(BN(Conv(X))). 비교 기준인 Repetition 은 "repeated Ts times" 이다. [read in full text] — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- Table 2: Repetition encoder 는 수렴 실패가 많다. 예로 Spike-TCN Metr-la R^2 는 `.024*`, iSpikformer Metr-la H96 은 `.021*`, Electricity H96 은 `.710*` 이다 (`*` 는 수렴 실패 표시). [read in full text] — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- Spike-TCN 은 "membrane potentials ... are set to 0 at the beginning of every time-series time step" 이고, Spike-RNN 은 "the membrane potential in Spike-RNN will persist across time steps" 이다. iSpikformer 는 iTransformer 의 self-attention 을 SSA 로 바꾸고 S_emb = SN(Linear(S)) 로 channel-wise token 을 만든다. [read in full text] — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- 코드 확인 1: `ConvEncoder` 는 convolution 출력 channel 을 T 로 permute 한 뒤 LIF 에 넣는다 (`enc.permute(1, 0, 2, 3) # T, B, C, L`). 즉 iSpikformer 의 T 는 convolution kernel 축이고, 출력은 `x[-1, :, :, :]` 로 마지막 T step 만 쓴다. [read in full text: repository code] — [encoder.py](https://github.com/microsoft/SeqSNN/blob/main/SeqSNN/module/encoding/spikingjelly/encoder.py); [ispikformer.py](https://github.com/microsoft/SeqSNN/blob/main/SeqSNN/network/snn/ispikformer.py)
- 코드 확인 2: Spike-GRU 설정은 `ISNNGRU2d, num_steps: 4` 이고 forward 는 `for i in range(length)` 로 L 을 따라가며 각 위치에서 `num_steps` 만큼 내부 루프를 돈다. 초기화는 forward 시작에서 한 번뿐이라 membrane 이 L x T 동안 유지된다. 진짜로 입력 순서를 따라 neuron 이 진행하는 경우다. [read in full text: repository code] — [spikegru.py](https://github.com/microsoft/SeqSNN/blob/main/SeqSNN/network/snn/spikegru.py); [spikegru.yml](https://github.com/microsoft/SeqSNN/blob/main/exp/forecast/model/spikegru.yml)
- Head: Y = Linear(S_hidden). 학습: Ts = 4, U_thr = 1.0, beta = 0.99, batch 128, Adam, lr 1e-4, early stopping 30 epochs, 3 seeds. 입력 길이는 Metr-la, Pems-bay 12 와 Solar, Electricity 168 이다. Dropout 과 layer normalization 은 제거한다. [read in full text] — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- 학습 loss 는 본문에 없다. 저장소 설정 한 건 (`rnn2d_electricity.yml`) 에 `loss_fn: mse` 가 있다. [read in full text: repository code] — [rnn2d_electricity.yml](https://github.com/microsoft/SeqSNN/blob/main/exp/forecast/combined/rnn2d_electricity.yml)
- Energy: Energy(l) = E_AC x SOPs(l), Energy(b) = E_MAC x FLOPs(b), 45 nm 이론값이다. Table 3 은 Electricity L = 24 한 지점만 보고한다: iTransformer 9.47 mJ, iSpikformer 3.19 mJ (66.30% 감소), Spike-GRU 1.51 mJ (GRU 6.07 mJ 대비 75.05% 감소), Spike-TCN 0.23 mJ. [read in full text] — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- Baseline: ARIMA, GP, TCN, GRU, Autoformer, iTransformer. "Results for ARIMA, GP, GRU, and Autoformer are obtained from" Fang et al. (2023) 라고 밝힌다. [read in full text] — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)

#### TS-LIF, ICLR 2025
- Venue: "Published as a conference paper at ICLR 2025". 데이터는 Lv 와 같은 4개이고 **ETT 없음**. Context 는 {12, 168} 또는 168, H = 6, 24, 48, 96. [read in full text] — [arXiv 2503.05108](https://arxiv.org/abs/2503.05108)
- Neuron 식 (5): v_d[t] = a1 v_d[t-1] + b1 v_s[t-1] + (1 - a1) c[t] - g1 s_d[t-1], v_s[t] = a2 v_s[t-1] + b2 v_d[t] + (1 - a2) c[t] - g2 s_s[t-1]. 출력은 s_mix[t] = k s_d[t] + (1 - k) s_s[t] (식 6) 이고 계수는 모두 학습한다. 정리 1 은 안정성 조건을, 식 17 은 compartment 별 transfer function 을 준다. [read in full text] — [arXiv 2503.05108](https://arxiv.org/abs/2503.05108)
- T 축: 본문은 "As for the timesteps in the SNN structures, we align them with the history length in each setting" 이라 쓰지만, 부록은 Lv 의 Ts x T x C 정렬과 convolution encoder 를 그대로 옮겨 적었고 저장소 설정은 `ISNNGRU2d, num_steps: 4` 이다. 본문 표현과 설정이 정확히 무엇을 뜻하는지는 일치하지 않는다. [read in full text]; [read in full text: repository code] — [arXiv 2503.05108](https://arxiv.org/abs/2503.05108); [spikegru.yml](https://github.com/kkking-kk/TS-LIF/blob/main/TS-LIF/exp/forecast/model/spikegru.yml)
- 결과 비교: Table 1 에서 "remaining results are sourced from Lv et al. (2024)" 이다. Table 10 은 3 seeds 표준편차다. Loss 와 stopping 은 본문에서 찾지 못했다. [read in full text] — [arXiv 2503.05108](https://arxiv.org/abs/2503.05108)
- Energy: E_MAC = 4.6 pJ, E_AC = 0.9 pJ, SOPs(l) = T x gamma x FLOPs(l), 45 nm. Table 6 (Electricity): TS-former 3.22 mJ (65.99% 감소), TS-GRU 1.58 mJ, TS-TCN 0.25 mJ. 학습 시간은 iSpikformer 49.84 s, iTransformer 7.24 s 로 함께 보고한다. [read in full text] — [arXiv 2503.05108](https://arxiv.org/abs/2503.05108)

#### SpikF, ICML 2025 (가장 가까운 경쟁 모델)
- Venue: "Proceedings of the 42nd International Conference on Machine Learning, PMLR 267, 2025". [read in full text] — [PMLR](https://proceedings.mlr.press/v267/wu25m.html); [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- Spiking Patch Encoder (식 5, 6): p_k = x_{L/N (k-1)+1 : L/N k}, S_enc = SN(BN(LN(p_k))). 본문: "temporally upsampled by a factor of Ts using a zero-order hold (ZOH) strategy". [read in full text] — [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- Spiking Frequency Selector (식 7-12): group 화 G_i, F_i = SF_g(G_i), 선택자 M_sel^i = SN(BN(LN(G_i))), max pooling, H_i^F = SF_g^{-1}(M_sel element-wise F_i), S_i^F = SN(BN(H_i^F)). [read in full text] — [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- Decoder (식 13, 14): 학습은 T step 마다 L = (1/Ts) sum ||MLP(S^k) - Y||, 추론은 평균. "SpikF is optimized using the MAE loss function". [read in full text] — [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- 코드: 모든 LIF 는 `MultiStepLIFNode` 로 T 축을 따라 진행하고, 순서 처리는 patch 축 `torch.fft.rfft` 와 kernel `[5, 1]` convolution 이 맡는다. Kernel `[3, 1]` convolution 은 T 축에 건다. Head 는 Linear, BatchNorm1d, GELU, Linear 이고 입력에 평균과 표준편차 instance normalization 을 쓴다. 즉 **neuron 은 token 순서를 따라 진행하지 않는다**. [read in full text: repository code] — [SpikF.py](https://github.com/WWJ-creator/SpikF/blob/main/models/SpikF.py)
- ETT 설정 (Table 7 + 스크립트): Ts 16, layers 1, d_patch 32, hidden 720, batch 32, lr 5e-4, epochs ETTh 5 / ETTm 10. 기본 `patch_num = 48` 이라 L = 96 에서 patch 길이는 2 이다. Early stopping 은 ETTh2 patience 2, ETTm2 patience 1, min delta 0.0002, 시험은 best-validation checkpoint 로 한다. 시험 loader 는 batch 1, `drop_last = False`. [read in full text]; [read in full text: repository code] — [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf); [run_ETTh2.sh](https://github.com/WWJ-creator/SpikF/blob/main/scripts/run_ETTh2.sh); [run_long.py](https://github.com/WWJ-creator/SpikF/blob/main/run_long.py); [exp_ETT.py](https://github.com/WWJ-creator/SpikF/blob/main/exp/exp_ETT.py)
- Seed 주의: 논문은 "three independent experimental trials with distinct random seeds" 라고 쓴다. 공개 스크립트는 `for seed in 0 42 79` 로 돌지만 `--random_seed` 를 넘기지 않고 파일 이름에만 쓴다 (`run_long.py` 기본값 0). 공개 스크립트 그대로면 같은 seed 를 3번 돌리는 셈이다. 논문 결과를 어떻게 만들었는지는 이것만으로 알 수 없다. [read in full text: repository code] — [run_ETTh2.sh](https://github.com/WWJ-creator/SpikF/blob/main/scripts/run_ETTh2.sh); [run_ETTm2.sh](https://github.com/WWJ-creator/SpikF/blob/main/scripts/run_ETTm2.sh)
- Ablation (Table 4): SFS 를 "completely discarding" 한 `w/o` 는 ETTh1 0.441 / 0.428, ETTm1 0.388 / 0.387, Weather 0.246 / 0.266 으로 원본 0.440 / 0.428, 0.388 / 0.385, 0.245 / 0.265 와 거의 같다. iSSA 로 바꾸면 0.452 / 0.440, 0.389 / 0.389, 0.247 / 0.268 로 **SFS 를 빼는 것보다 나쁘다**. Encoder 를 바꾼 경우 Delta 0.468, Conv 0.467 (ETTh1 MSE) 로 효과가 가장 크다. ETTh2, ETTm2 는 ablation 표에 없다. [read in full text] — [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- Ts 민감도: Ts 를 1, 2, 4, 8, 16 으로 바꿔도 "average variation of only 3.7%" 이다. [read in full text] — [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- 비교 방식: "The results of all benchmarks except SpikF are extracted from (Liu et al., 2024)". 다만 부록 C.4 는 SpikF 가 MAE loss 이고 다른 모델은 MSE loss 임을 인정하고 MAE loss 로 다시 돌린 iTransformer 를 Table 9, 10 에 싣는다. Table 12 는 "SNN-based models are equipped with SPE" 라고 적는다. 즉 **ETT 의 SpikeRNN, iSpikformer 수치는 Lv 의 원형이 아니라 SpikF encoder 를 붙인 변형이다.** [read in full text] — [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- Energy: SOPs = alpha x T x FLOPs, 4.6 pJ, 0.9 pJ, 45 nm. Table 2 는 ECL H720 에서 **SFS 모듈만** 센다 (SpikF 117.66 uJ, DLinear 205.19 uJ, iTransformer 3289.29 uJ). Figure 3 은 모든 모듈의 연산 energy 로 75.05% 감소를 말한다. Table 3 은 memory 와 addressing 을 더해 E_total 1.37 x 10^4 uJ 대 4.34 x 10^4 uJ (3.16배) 를 준다. **ETT energy 는 이 논문에 없다.** [read in full text] — [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)

#### SpikF 저장소의 baseline 코드 (SpikeRNN, iSpikformer)
- `SpikeRNN.py`: SPE 뒤에 block 마다 `nn.Linear(patch_num + hidden_dim, hidden_dim)` 로 **token 축** 을 mixing 하고, 결과를 입력과 concat 한 뒤 BatchNorm2d, `MultiStepLIFNode` 를 지난다. N 축 루프가 없고 "recurrent" 는 layer 사이의 전달이다. 코드로 보면 이것은 RNN 이 아니라 **반복 T 위의 spiking token-mixing MLP** 이다. [read in full text: repository code] — [SpikeRNN.py](https://github.com/WWJ-creator/SpikF/blob/main/models/SpikeRNN.py)
- `iSpikformer.py`: SPE 뒤에 `Linear(patch_num, patch_num)` 으로 q, k, v 를 만들고 `q @ k @ v` 로 변수 축 attention 을 건다. Lv 의 convolution encoder 는 쓰지 않는다. [read in full text: repository code] — [iSpikformer.py](https://github.com/WWJ-creator/SpikF/blob/main/models/iSpikformer.py)
- 내가 읽은 ETTh2, ETTm2 스크립트는 `--model SpikF` 만 돌리고 저장소 scripts 폴더는 데이터별 파일뿐이라, 이 baseline 들의 stopping 설정은 공개 자료로 확정할 수 없다. [read in full text: repository code] — [SpikF scripts](https://github.com/WWJ-creator/SpikF/tree/main/scripts)

#### NSMT 제출본 (로컬 사본)
- 서지: 제목은 "Fast Spikes, Slow Trends: Neuro-Inspired Spiking Memory Transformer for Time-Series Analysis", "Anonymous Author(s), Submission Id: 3724", ACM 템플릿의 "Conference acronym 'XX" 그대로이다. **게재처는 이 사본으로 확인되지 않는다.** [read in full text] — `/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/papers/NSMT.pdf`
- Neocortex: 입력을 T 축으로 average pooling 해 "collapsing the redundant repetition" 한 뒤 sequence 축 N 을 시간 축으로 삼는다. 각주 1: "we set the simulation steps T equal to the sequence length N (T = N)". 식 2: X_n = SN(BN(Linear(X))). "the positional information is naturally encoded through the neuronal firing dynamics, eliminating the need for explicit positional encoding". 층은 D 차원 spiking MLP 이고 **token 축 Linear mixing 은 없다.** [read in full text] — NSMT.pdf
- 보조 목적: X_low = IDCT(Mask_k(DCT(X_in))), L_aux = ||Decoder(X_neo) - X_low||^2. Hippocampus 는 표준 patch 입력을 쓰고 spiking cross-attention (Q 는 X_h, K 와 V 는 X_f) 으로 memory 를 읽는다 (식 7, 8). [read in full text] — NSMT.pdf
- Table 7 (alignment 유무): ETTh2 MSE 0.364 와 0.365, ETTm1 0.380 과 0.380, ETTm2 0.279 와 0.279 로 **정확도 차이가 없고**, energy 는 53.32 에서 37.36, 67.65 에서 44.28, 62.49 에서 41.36 uJ 로 줄어든다. 표의 "Improvement" 칸은 42.74, 52.79, 51.10% 이지만 energy 칸에서 계산하면 29.9, 34.5, 33.8% 감소다. [read in full text] — NSMT.pdf
- 수치 근거: "All results are taken from [43] and [41]" ([43] 은 SpikF, [41] 은 TimesNet) 이고 main table 은 "obtained using random seed 42" 이다. 7 seeds {7, 13, 21, 42, 55, 123, 2026} 는 SpikF 대비 one-sided t-test 에만 쓰며 ETTm1, ETTm2 평균만 보고한다 (p = 5.88e-5, 평균 차 0.006914; p = 0.00169, 평균 차 0.001244). [read in full text] — NSMT.pdf
- Energy: AC(m) = fr x T x MAC(m), E_tot = E_MAC x (FL_rec + FL_head) + E_AC x (sum AC_SpikingMLP + sum AC_MR). 본문은 "4.6uJ and 0.9uJ" 로 적었는데 다른 논문은 모두 pJ 이다. Table 6 (H720): SpikF 1050.47 (ETTh2), 1050.10 (ETTm2), NSMT 44.90, 46.72 uJ. **SpikF 값은 SpikF 논문이 아니라 NSMT 저자가 계산한 값이다.** [read in full text] — NSMT.pdf
- 학습 규칙 (로컬 코드 지도): L1 + 보조 목적, `AdamW`, ETTh2 는 patch 8, epoch 100, patience 2, seed 42. 코드상 `self.T = self.num_patches` 라 Hippocampus 의 반복 횟수도 token 수와 같다. 우리 SpikMixer 의 T_h = 4 와 다른 점이다. `train()` 은 마지막 epoch 상태를 저장하고 `test()` 가 그것을 읽어, 시험 수치는 best-validation 가중치가 아니라 "early stop 시점(또는 마지막 epoch)" 의 가중치다. 본문에는 명시가 없다. [read in full text: 로컬 문서] — `/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/papers/NSMT_code_mapping.md`

#### SpikeLite, arXiv 2609.35097v1 (2026-09, 가장 최근의 ETT 수치)
- 저자는 Fudan 의 Bang Hu, Changze Lv 등으로 SeqSNN 계열이다. 게재처 표시는 없고 코드는 "will be released" 이다. [read in full text] — [arXiv 2609.35097](https://arxiv.org/abs/2609.35097)
- FSSE (식 5, 6): 변수마다 K 개의 병렬 LIF 가 **관측 축을 따라** 진행한다. U_{l,c}^(k) = tau_c^(k) U_{l-1,c}^(k) + X_{l,c} - theta_c^(k) S_{l-1,c}^(k). 출력은 membrane 을 spike 로 gating 한 실수값 G = U S 이다. 인접 branch 차를 취해 합이 입력 X 로 복원되는 분해 B^(j) 를 만들고 (식 7, 8), 각 성분을 Linear 로 투영해 Z (R^{C x D}) 를 만든다 (식 9). [read in full text] — [arXiv 2609.35097](https://arxiv.org/abs/2609.35097)
- SSCA: rFFT 크기로 이진 channel mask 를 만들고, Z 를 Ts 번 반복한 spike 위에서 masked SSA 를 건다 (식 16). "When explicit channel interaction is unnecessary, SpikeLite uses the lighter FSSE-only channel-independent path". **ETT 가 어느 경로를 쓰는지는 본문에 없다.** [read in full text] — [arXiv 2609.35097](https://arxiv.org/abs/2609.35097)
- 설정: D = 128, RevIN, Ts = 4, Adam, batch 32, validation loss 로 early stopping, 3 seeds. Patience 와 loss 는 본문에서 확인하지 못했다. [read in full text] — [arXiv 2609.35097](https://arxiv.org/abs/2609.35097)
- Energy: "The task-specific prediction heads are excluded for all methods". ECL H720 에서 95.30 uJ 이고, SpikF 117.66, DLinear 205.19, iTransformer 3289.29 는 SpikF Table 2 의 값을 그대로 옮긴 것이다. [read in full text] — [arXiv 2609.35097](https://arxiv.org/abs/2609.35097)
- Baseline: 장기 예측 쪽은 "Their reported results are taken from the SpikF benchmark" 이다. [read in full text] — [arXiv 2609.35097](https://arxiv.org/abs/2609.35097)

#### SpikySpace, arXiv 2601.02411v2 (spiking Mamba)
- 페이지에 "Preprint" 로 표시되고 코드 링크는 익명이다. 데이터는 Metr-la, Pems-bay, Solar, Electricity, H = 3, 6, 12, 24 로 **ETT 없음**. 지표는 R^2 와 RRSE. [read in full text] — [arXiv 2601.02411](https://arxiv.org/abs/2601.02411)
- 구조: h_t = A h_{t-1} + B x_t 를 spike 와 power-of-two 양자화로 곱셈 없이 계산한다. Encoder 는 s_in = SN(x_in), s = SN(Conv1D(s_in)). Linear head. ANN-to-SNN conversion 과 Avg-IF neuron 을 쓴다. [read in full text] — [arXiv 2601.02411](https://arxiv.org/abs/2601.02411)
- T: "timestep T = 3, which means we use a spike train length of 3 to represent a scalar". 학습: batch 64, lr 5e-4, Adam, MSE, 최대 1000 epochs, patience 20. Seed 수는 본문 검색으로 찾지 못했다. [read in full text] — [arXiv 2601.02411](https://arxiv.org/abs/2601.02411)
- Energy: E_total = T s E_ACC + T s (E_move + E_weight), "commercial 22nm process" 기준. Table 2 는 Electricity H3 라고 적고 SpikySpace 0.17 mJ, SpikeSTAG 4.39, iSpikformer 3.19, iTransformer 9.47 mJ 를 싣는다. [read in full text] — [arXiv 2601.02411](https://arxiv.org/abs/2601.02411)

#### SpikF-GO, ECML-PKDD 2026
- "This paper has been accepted for presentation at ECML-PKDD 2026". [read in full text] — [arXiv 2606.13901](https://arxiv.org/abs/2606.13901); [LNCS DOI](https://doi.org/10.1007/978-3-032-37664-0_3)
- 구조: 모든 관측값을 hypervariate graph node 로 보고 (M = N x L), V 를 Ts step 복제해 U_t = V x gamma_t + beta_t 로 변조한 뒤 LIF 에 넣는다 (식 14, 15). Node 축 S-FFT, Hard Concrete frequency gate, Complex LIF gate 가 이어진다. RMSNorm 사용. [read in full text] — [arXiv 2606.13901](https://arxiv.org/abs/2606.13901)
- 데이터: ECG, COVID-19, Solar, Electricity, METR-LA, Traffic, PEMS-BAY, Wiki. 입력과 예측 모두 12, 5 runs. **ETT 없음.** MSE 학습, RevIN, Ts = 4 (SpikF 만 16). 원본 FourierGNN 코드의 normalization leakage 를 고쳤다고 적는다. Baseline 을 모두 같은 절차로 다시 돌렸다. [read in full text] — [arXiv 2606.13901](https://arxiv.org/abs/2606.13901)
- Energy: Solar H12 에서 E_total 4.70 x 10^4 uJ, FourierGNN 8.89 x 10^4, SpikF 2.08 x 10^4. "our energy analysis is theoretical and does not yet include deployment on real neuromorphic hardware". [read in full text] — [arXiv 2606.13901](https://arxiv.org/abs/2606.13901)

#### SpikingTSF (제3자 통합 benchmark)
- MIT 저장소, 작성자는 SpikF-GO 저자 Bakhshaliyev. 설정: L = 96, T = 4 (설정 파일), MAE loss, patience 15, instance normalization, Optuna 100 trials, seeds 42, 1234, 3407. "Results from this library should not be used in publications without verifying them independently". [read in full text: repository code] — [SpikingTSF](https://github.com/spikora/SpikingTSF); [PROTOCOL.md](https://github.com/spikora/SpikingTSF/blob/main/PROTOCOL.md); [SpikeRNN ETTh2 설정](https://github.com/spikora/SpikingTSF/blob/main/configs/SpikeRNN/ETTh2.yaml)
- ETTh1 SpikF 재현: H96 0.3779 (+-0.0016), H192 0.4336, H336 0.4786, H720 0.4938 (+-0.0077), 논문은 0.379, 0.432, 0.473, 0.474 이다. H720 은 4.2% 차이가 난다. ETTh2, ETTm1, ETTm2 는 "Pending hyperparameter tuning" 이다. [read in full text: repository code] — [RESULTS.md](https://github.com/spikora/SpikingTSF/blob/main/RESULTS.md)

#### 그 외 forecasting 관련 논문
- Closed-loop SSM: **단변량** ETTh1, ETTh2, ETTm1 (ETTm2 생략). S4D 에 Signed-Spk 활성화. ETTh2 P = 720, F = 96 에서 Signed-Spk 0.30+-.02 (ZM), 0.30+-.02 (Model-CL), 0.30+-.04 (Layer-CL), 5 initializations. 단변량이라 우리와 비교할 수 없다. [read in full text] — [arXiv 2610.28495](https://arxiv.org/abs/2610.28495)
- SpikeSTAG: METR-LA, Pems-bay, Solar, Electricity, 3 seeds. "LSTM-processed sequences" 와 SSA 를 spike gate 로 합치고 시간 보조 특징을 넣는다. Energy 4.39 mJ, 53.64% 감소. **ETT 없음.** 게재처 표시 없음. [read in full text] — [arXiv 2508.02069](https://arxiv.org/abs/2508.02069)
- CPG-PE: NeurIPS 2024, Lv 의 4개 데이터, 3 seeds, batch 64, 30 epochs tolerance. ETT 없음. [read in full text] — [arXiv 2405.14362](https://arxiv.org/abs/2405.14362)
- RPE (Gray-PE, Log-PE): NeurIPS 2025, 같은 4개 데이터, 3 seeds, L = 12 또는 168. ETT 없음. [read in full text] — [arXiv 2501.16745](https://arxiv.org/abs/2501.16745)
- SEDformer: 불규칙 telemetry (Wiki2000, WikiArticle), Event-Aligned LIF, 5 seeds, paired t-test, 45 nm energy. ETT 없음. [read in full text] — [arXiv 2602.02230](https://arxiv.org/abs/2602.02230)
- MTSA-SNN: ICPR 2024 (LNCS). wavelet + spiking encoder. Table II 는 하위 집합을 밝히지 않고 "ETT" 로 적으며 baseline 값은 ETTh1 의 NLinear 0.374, DLinear 0.375 와 같고 자기 값은 H96 0.235 이다. [read in full text] — [arXiv 2402.05423](https://arxiv.org/abs/2402.05423)
- SpikingSSMs 와 Mamba-Spike: 본문 검색에서 forecasting 이나 ETT 실험을 찾지 못했다. SpikingSSMs 는 sMNIST 같은 장기 순서 분류, Mamba-Spike 는 DVS Gesture, TIDIGITS, Sequential MNIST, CIFAR10-DVS 이다. [read in full text] — [arXiv 2408.14909](https://arxiv.org/abs/2408.14909); [arXiv 2408.11823](https://arxiv.org/abs/2408.11823)
- 초록 수준만 확인한 forecaster: STG-GAPE (PeerJ CS 2026), SDSTCN (IJCIS 2025), Information-Preserving Spiking (Electronics 2026, energy load 2개), STDA-Net (IEEE IEMCON 2025, Electricity, Solar), REAT-CSNN (Biomimetics 2025), binary 방향 예측 재정의 (IEEE ICNC 2026), Hybrid encoding poster (IEEE MASS 2024, Metr-la), Risk-Aware iSpikformer (EPSIC 2026). 초록에 ETT 언급은 없다. [abstract only] — [PeerJ DOI](https://doi.org/10.7717/peerj-cs.3874); [IJCIS DOI](https://doi.org/10.1007/s44196-025-01063-4); [Electronics DOI](https://doi.org/10.3390/electronics15081597); [IEMCON DOI](https://doi.org/10.1109/iemcon67450.2025.11381145); [Biomimetics DOI](https://doi.org/10.3390/biomimetics10040240); [ICNC DOI](https://doi.org/10.1109/icnc68183.2026.11416922); [MASS DOI](https://doi.org/10.1109/mass62177.2024.00076); [EPSIC DOI](https://doi.org/10.1109/epsic70071.2026.11590401)
- 서지 정보만 확인: BCDQ-TCLIF (Information Sciences 2026), TSSA-Net (LNCS 2026), Physio-S3M (Wind Engineering 2026). 초록을 얻지 못했고 웹 검색에서도 본문이 나오지 않았다. [abstract only: 제목과 서지 정보만] — [BCDQ-TCLIF DOI](https://doi.org/10.1016/j.ins.2026.124324); [TSSA-Net DOI](https://doi.org/10.1007/978-981-95-7084-3_42); [Physio-S3M DOI](https://doi.org/10.1177/0309524x261506702)

### Inferences
- T 가 시간이라는 설명과 코드가 다른 경우가 많다. Lv 의 iSpikformer 에서 T 는 convolution kernel 축이고, SpikF 에서 T 는 복사본 축이며 Ts = 1 까지 줄여도 평균 변동이 3.7% 이다. 정확도 측면에서 반복 T 는 중요한 mechanism 이 아니라는 뜻으로 읽힌다.
- Lv 의 "repetition 은 수렴하지 않는다" 와 SpikF 의 "반복이 convolution, delta 보다 낫다" 는 모순이 아니다. Lv 의 `RepeatEncoder` 는 원값을 그대로 LIF 에 넣고, SpikF 는 Linear 와 BN 을 먼저 지난다. 우리 임베딩 (Linear + BN + LIF, 반복축) 은 SpikF 쪽 형태이므로 Lv 의 비판은 그대로 적용되지 않는다.
- SpikF 의 ablation 은 ETT 계열에서 "body 를 빼도 거의 같다" 는 뜻이다. 성능의 대부분은 instance normalization, patch encoder, head 에서 온다고 봐야 한다. 우리 쪽 결과 (SpikMixer 와 Spikformer-style 의 작은 차이) 와 같은 방향이다.
- SpikySpace Table 2 의 iTransformer 9.47 mJ, iSpikformer 3.19 mJ 는 Lv Table 3 (L = 24, 45 nm) 과 값이 같고 SpikeSTAG 4.39 mJ 는 SpikeSTAG 논문값과 같다. 표는 H3 와 22 nm 를 말하므로, 다시 계산한 값이 아니라 옮겨 온 값일 가능성이 높고 그렇다면 "98.20%" 는 같은 기준의 비교가 아니다. 원자료를 보지 못했으므로 확정은 아니다.
- SpikeLite 가 "energy 최저" 라고 할 때 비교값 SpikF 117.66 uJ 는 SFS 모듈만 센 값이고 SpikeLite 는 encoder backbone 을 head 없이 센 값이다. 범위가 다른 수를 나란히 놓은 것으로 보인다.
- SpikingTSF 의 SpikeRNN, SpikGRU, TSGRU 가 ETTh2 에서 거의 같은 값에 표준편차 0.0000-0.0001 을 보이는 것은 세 모델의 body 가 예측에 거의 기여하지 않는 상태일 가능성을 시사한다. 비정상 실행일 수 있으므로 baseline 으로 쓰면 안 된다.

### Gaps
- TS-LIF 의 TS-former 가 실제로 어떤 T 를 쓰는지, 본문의 "history length 에 맞춤" 이 무엇을 뜻하는지 확정하지 못했다. 저장소에 TS-LIF 전용 실행 설정을 찾지 못했다.
- SpikeSTAG 의 Ts 기본값은 본문에서 확정하지 못했고 Figure 4 에서 탐색 범위만 보인다.
- SpikeLite 의 ETT 경로 (FSSE-only 여부), 학습 loss, patience 는 코드 미공개라 확정할 수 없다.
- STG-GAPE, SDSTCN, BCDQ-TCLIF, TSSA-Net 본문을 받지 못했다 (403 또는 로그인 리다이렉트). 이들이 ETT 를 쓰는지는 알 수 없다.
- Semantic Scholar search 는 이 서버에서 429 로 막혔고 OpenAlex 와 웹 검색으로 대신했다.

## 질문 2. 공개 수치 비교표 (ETTh2, ETTm2, H96, H720) 와 우리 수치와의 비교 가능성

### Takeaway
L = 96 에서 SpikMixer 는 4칸 평균 0.3245 로 SpikF (0.3215), SpikeLite (0.3225), NSMT 공개값 (0.3178) 보다 약간 나쁘고 iTransformer (0.3275), PatchTST (0.3275), SpikF 의 SpikeRNN (0.3270), iSpikformer (0.3325) 보다는 약간 낫다. 가장 불편한 비교는 **RLinear 0.3243 과 사실상 같고 TimeMixer 0.3168 보다 2.4% 나쁘다** 는 점이다. 격차 대부분이 같은 baseline 의 논문 간 재현 차이 (최대 4.6%) 보다 작다.

### Cited Findings

표의 값은 각 논문 표에서 그대로 옮겼고 평균 열만 내가 계산했다. 값은 MSE 다.

| 모델 | 구분 | L | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4칸 평균 (계산) | 출처 표 |
|---|---|---|---|---|---|---|---|---|
| **SpikMixer (우리)** | SNN | 96 | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | [given in assignment] |
| **Spikformer-style (우리)** | SNN | 96 | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | [given in assignment] |
| NSMT | SNN | 96 | 0.288 | 0.409 | 0.175 | 0.399 | 0.3178 | NSMT Table 13, 14 [read in full text] |
| SpikF | SNN | 96 | 0.290 | 0.416 | 0.175 | 0.405 | 0.3215 | [SpikF](https://proceedings.mlr.press/v267/wu25m.html) Table 11 [read in full text] |
| SpikeLite | SNN | 96 | .294 | .426 | .175 | .395 | 0.3225 | [SpikeLite](https://arxiv.org/abs/2609.35097) Table 5 [read in full text] |
| SpikeRNN (SPE 부착) | SNN | 96 | 0.295 | 0.428 | 0.176 | 0.409 | 0.3270 | [SpikF](https://proceedings.mlr.press/v267/wu25m.html) Table 12, NSMT Table 14 [read in full text] |
| iSpikformer (SPE 부착) | SNN | 96 | 0.308 | 0.438 | 0.178 | 0.406 | 0.3325 | [SpikF](https://proceedings.mlr.press/v267/wu25m.html) Table 12, NSMT Table 14 [read in full text] |
| SpikeRNN (제3자) | SNN | 96 | 0.3526 | 0.4512 | Pending | Pending | 없음 | [SpikingTSF RESULTS](https://github.com/spikora/SpikingTSF/blob/main/RESULTS.md) [read in full text: repository code] |
| SpikGRU (제3자) | SNN | 96 | 0.3529 | 0.4515 | Pending | Pending | 없음 | 위와 같음 |
| TSGRU (제3자) | SNN | 96 | 0.3527 | 0.4512 | Pending | Pending | 없음 | 위와 같음 |
| Spikingformer (제3자) | SNN | 96 | 0.3486 | 0.4731 | Pending | Pending | 없음 | 위와 같음 |
| iTransformer | ANN | 96 | 0.297 | 0.427 | 0.180 | 0.412 | 0.3290 | [iTransformer](https://arxiv.org/abs/2310.06625) Table 10 [read in full text] |
| iTransformer, MAE loss | ANN | 96 | 0.294 | 0.423 | 0.182 | 0.411 | 0.3275 | [SpikF](https://proceedings.mlr.press/v267/wu25m.html) Table 9, 10 [read in full text] |
| RLinear | ANN | 96 | 0.288 | 0.420 | 0.182 | 0.407 | 0.3243 | [iTransformer](https://arxiv.org/abs/2310.06625) Table 10 [read in full text] |
| PatchTST (재실행) | ANN | 96 | 0.302 | 0.431 | 0.175 | 0.402 | 0.3275 | [iTransformer](https://arxiv.org/abs/2310.06625) Table 10 [read in full text] |
| PatchTST (다른 재실행) | ANN | 96 | 0.308 | 0.436 | 0.183 | 0.412 | 0.3348 | [TimeMixer](https://arxiv.org/abs/2405.14616) Table 13 [read in full text] |
| TimeMixer | ANN | 96 | 0.289 | 0.412 | 0.175 | 0.391 | 0.3168 | [TimeMixer](https://arxiv.org/abs/2405.14616) Table 13 [read in full text] |
| TimesNet | ANN | 96 | 0.340 | 0.462 | 0.187 | 0.408 | 0.3493 | [iTransformer](https://arxiv.org/abs/2310.06625) Table 10 [read in full text] |
| FITS | ANN | 96 | 0.295 | 0.431 | 0.183 | 0.407 | 0.3290 | [SpikF](https://proceedings.mlr.press/v267/wu25m.html) Table 12 (FilterNet 인용) [read in full text] |
| DLinear (재실행) | ANN | 96 | 0.333 | 0.831 | 0.193 | 0.554 | 0.4778 | [iTransformer](https://arxiv.org/abs/2310.06625) Table 10 [read in full text] |
| DLinear (긴 입력) | ANN | 336 기본, 최적 선택 | 0.289 | 0.605 | 0.167 | 0.397 | 0.3645 | [DLinear](https://arxiv.org/abs/2205.13504) Table 2; [PatchTST](https://arxiv.org/abs/2211.14730) Table 3 [read in full text] |
| PatchTST/42 | ANN | 336 | 0.274 | 0.379 | 0.165 | 0.367 | 0.2963 | [PatchTST](https://arxiv.org/abs/2211.14730) Table 3 [read in full text] |
| PatchTST/64 | ANN | 512 | 0.274 | 0.379 | 0.166 | 0.362 | 0.2953 | [PatchTST](https://arxiv.org/abs/2211.14730) Table 3 [read in full text] |
| TSMixer | ANN | 512 | 0.274 | 0.445 | 0.163 | 0.420 | 0.3255 | [TSMixer](https://arxiv.org/abs/2303.06053) Table 3 [read in full text] |

SpikF 의 95% 구간 (Table 9, 10, 3 seeds): ETTh2 H96 "0.290+-0.001", H720 "0.416+-0.008", ETTm2 H96 "0.175+-0.001", H720 "0.405+-0.007". MAE loss iTransformer: "0.294+-0.001", "0.423+-0.005", "0.182+-0.017", "0.411+-0.009". [read in full text] — [SpikF](https://proceedings.mlr.press/v267/wu25m.html)

SpikingTSF 표준편차: SpikeRNN H96 +-0.0000, H720 +-0.0000; SpikGRU +-0.0000, +-0.0000; TSGRU +-0.0001, +-0.0001; Spikingformer +-0.0052, +-0.0145. [read in full text: repository code] — [RESULTS.md](https://github.com/spikora/SpikingTSF/blob/main/RESULTS.md)

SpikMixer 와 각 행의 차이 (계산값, + 는 우리가 더 나쁨, 칸 순서는 ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720):

| 비교 대상 | 칸별 차이 | 평균 차이 |
|---|---|---|
| Spikformer-style | +1.69%, -0.12%, +0.45%, -0.25% | +0.32% |
| NSMT 공개값 | +2.50%, +3.13%, +1.43%, +1.10% | +2.12% |
| SpikF | +1.79%, +1.39%, +1.43%, -0.40% | +0.93% |
| SpikeLite | +0.41%, -0.99%, +1.43%, +2.13% | +0.61% |
| SpikeRNN (SPE 부착) | +0.07%, -1.45%, +0.85%, -1.37% | -0.77% |
| iSpikformer (SPE 부착) | -4.16%, -3.70%, -0.28%, -0.64% | -2.41% |
| iTransformer, MAE loss | +0.41%, -0.28%, -2.47%, -1.85% | -0.92% |
| RLinear | +2.50%, +0.43%, -2.47%, -0.88% | +0.07% |
| PatchTST (iTransformer 재실행) | -2.25%, -2.13%, +1.43%, +0.35% | -0.92% |
| TimeMixer | +2.15%, +2.38%, +1.43%, +3.17% | +2.44% |

비교 가능성 (우리: L = 96, L1 loss, patience 10, best-validation 가중치, 3 seeds [given in assignment]):

| 대상 | L | 분할 | Loss | Stopping 과 가중치 | Seeds | 판단 |
|---|---|---|---|---|---|---|
| SpikF | 같음 | 같음. border 가 로컬 `data_loader.py` 와 같은 12/4/4개월 | 같음 (L1) | 다름. 최대 5 또는 10 epochs, patience 2 또는 1, min delta 2e-4, best-validation | 3 (공개 스크립트는 seed 미전달) | 가장 가깝지만 stopping 이 훨씬 짧다 |
| SpikeLite | 같음 | SpikF 절차 | 본문 미확인 | validation early stopping, patience 미기재 | 3 | 코드 공개 전까지 조건부 |
| NSMT 공개값 | 같음 | 같음 | L1 + 보조 목적 | 다름. patience 2 (ETTh2), 마지막 상태 시험 | main 표는 seed 42 한 개 | 직접 비교 불가. 같은 규칙 재실행 결과를 써야 함 |
| SpikeRNN, iSpikformer (SPE) | 같음 | 같음 | 코드 기본값 MAE | 공개 스크립트 없음 | SpikF 표 설명 없음 | 조건부 |
| iTransformer, RLinear, PatchTST (Table 10) | 같음 | 표기 크기 (8545, 2881, 2881), (34465, 11521, 11521) | L2 | "number of training epochs is fixed to 10", patience 기재 없음 | 표 10 은 seed 수 미기재, 강건성 표는 5 | L1 과 L2 차이를 감안해야 함 |
| TimeMixer | 같음 | 같은 크기 | L2 | epochs 10 (Table 7), 시험 loader 설정은 Gaps 참조 | 3 repeats | 조건부 |
| PatchTST/42, /64, TSMixer | 다름 | 표준 | L2 | 긴 입력 | 표 설명 참조 | 비교 불가, 참고용 |
| SpikingTSF | 같음 | ETT loader | MAE | patience 15, 최대 300 epochs | 3 | 절차는 가깝지만 값이 비정상적으로 보임 |

- Loss 효과의 크기: 같은 iTransformer 를 L2 에서 MAE loss 로 바꾸면 ETTh2 H96 0.297 에서 0.294, H720 0.427 에서 0.423, ETTm2 H96 0.180 에서 0.182, H720 0.412 에서 0.411 이다. 칸별 약 1% 이내다. [read in full text] — [iTransformer](https://arxiv.org/abs/2310.06625); [SpikF](https://proceedings.mlr.press/v267/wu25m.html)
- 재현 차이의 크기: 같은 L = 96 PatchTST 가 iTransformer 표와 TimeMixer 표에서 0.302 와 0.308, 0.431 과 0.436, 0.175 와 0.183, 0.402 와 0.412 로 다르다 (최대 4.6%). [read in full text] — [iTransformer](https://arxiv.org/abs/2310.06625); [TimeMixer](https://arxiv.org/abs/2405.14616)
- 시험 집합 처리: SpikF 는 시험 batch 1, iTransformer 도 시험 batch 1 이라 샘플이 빠지지 않는다. 로컬 loader 의 기본값도 `drop_last = False` 이다. TimeMixer 는 시험 `batch_size = args.batch_size`, `drop_last = args.drop_last` 라 설정에 따라 마지막 batch 가 빠질 수 있다. [read in full text: repository code] — [SpikF exp_ETT.py](https://github.com/WWJ-creator/SpikF/blob/main/exp/exp_ETT.py); [iTransformer data_factory.py](https://github.com/thuml/iTransformer/blob/main/data_provider/data_factory.py); [TimeMixer data_factory.py](https://github.com/kwuking/TimeMixer/blob/main/data_provider/data_factory.py); 로컬 `NSMT/nsmt_flif_v1/forecasting/data_provider/data_factory.py`
- NSMT 값의 흔들림: 우리 저장소 기록에 NSMT 자체가 ETTh2 H96 에서 stopping 규칙에 따라 0.2835 와 0.2896 사이로 움직인다고 적혀 있다. [read in full text: 로컬 commit 메시지 d33567956] — 로컬 git log
- 길이 효과: 같은 ANN 에서 입력을 늘리면 큰 폭으로 좋아진다. PatchTST 의 ETTh2 H720 은 L = 96 재실행 0.431 에서 L = 336 0.379 로, ETTm2 H720 은 0.402 에서 0.367 로 낮아진다. SpikF Figure 5 도 ETTh2, ETTm2 에서 look-back 증가에 따른 감소를 보고한다. [read in full text] — [PatchTST](https://arxiv.org/abs/2211.14730); [SpikF](https://proceedings.mlr.press/v267/wu25m.html)

### Inferences
- 우리 값과 SpikF, SpikeLite 의 격차 (평균 1% 이내) 는 같은 baseline 의 논문 간 차이 (최대 4.6%) 안에 있다. 공개 수치만으로 "낫다" 또는 "나쁘다" 를 말할 수 없고, 같은 규칙으로 SpikF 를 우리 쪽에서 다시 돌려야 말할 수 있다.
- RLinear 와 사실상 같다는 점이 가장 큰 약점이다. RLinear 는 linear layer 와 RevIN 뿐이다. 리뷰어는 "spiking body 가 linear baseline 대비 무엇을 더하는가" 를 물을 것이다. 우리 저장소의 11-03 진단 (원입력 linear readout 이 ETTh2 H96 에서 body 와 맞먹음) 과도 방향이 같다.
- SpikF 저장소의 SpikeRNN 은 코드상 "반복 T + token 축 Linear + LIF" 이고 SpikMixer 와 평균 0.77% 차이다. 즉 token 축 neuron 진행, 양방향, gate 를 더해 얻은 공개 수치상 이득은 많아야 1% 수준으로 보인다. 다만 stopping 조건이 다르므로 같은 규칙의 측정이 필요하다.
- 칸별로는 ETTh2 H720, ETTm2 H720 에서 SpikeRNN 보다 1.4% 낫고 H96 에서는 같거나 약간 나쁘다. 긴 horizon 쪽 이득이라는 약한 신호이지만, seed 분산 확인 전에는 결론이 아니다.
- SpikingTSF 의 SpikeRNN 0.3526 과 SpikF 표의 0.295 는 같은 이름이지만 다른 구조다 (전자는 SeqSNN 계열 convolution encoder, 후자는 SPE). 같은 모델의 재현 실패로 읽으면 안 된다. 다만 encoder 교체만으로 이 정도 차이가 난다는 뜻이면 encoder 의 영향이 body 보다 크다는 앞의 판단과 일치한다.

### Gaps
- ETTh2 와 ETTm2 에서 SpikF 의 제3자 재현값은 아직 없다 (SpikingTSF 는 ETTh1 만 완료). ETTh1 H720 은 공개값보다 4.2% 나쁘게 나왔으므로 ETTh2, ETTm2 값도 공개값 그대로 믿기 어렵다.
- TimeMixer Table 13 에 쓰인 `drop_last` 값은 스크립트에서 확인하지 못했다.
- RLinear 재실행의 loss 는 iTransformer 부록 기준 L2 로 읽었지만 baseline 별 예외를 확인하지 못했다.
- Spiking 기법 중 ETTh2, ETTm2 H96, H720 을 보고한 것은 내가 찾은 범위에서 NSMT, SpikF (및 그 baseline), SpikeLite, SpikingTSF 뿐이다. 초록만 본 논문이 ETT 를 썼을 가능성은 배제할 수 없다.

## 질문 3. 이 분야의 증거 기준과 게재처 수준

### Takeaway
기준은 낮은 편이다. 3 seeds, 다른 논문에서 옮겨 온 baseline, 45 nm 상수와 firing rate 로 계산한 이론 energy 가 표준이며 하드웨어 측정은 내가 읽은 논문에 없다. 최상위 학회 (ICML, ICLR, NeurIPS) 는 새 neuron 또는 module 에 분석을 더하고 집계상 baseline 을 이긴 논문에 열렸고, 정확도 격차 자체는 2% 안팎이어도 통과했다.

### Cited Findings

| 논문 | 게재처 | Seeds | Baseline 수치 출처 | Energy 방식과 범위 |
|---|---|---|---|---|
| Lv et al. | ICML 2024 | 3 | 일부 Fang et al. 2023 에서 옮김 | 45 nm, Electricity L = 24 한 지점 |
| TS-LIF | ICLR 2025 | 3 | Lv 에서 옮김 | 4.6, 0.9 pJ, Electricity |
| SpikF | ICML 2025 | 3, 95% 구간 | Liu 2024 에서 옮김, iTransformer 만 MAE 로 재실행 | 4.6, 0.9 pJ, ECL H720. SFS 전용 표 + 전체 연산 그림 + memory 포함 표 |
| CPG-PE | NeurIPS 2024 | 3 | Lv 계열 | 미확인 |
| RPE | NeurIPS 2025 | 3 | Lv 계열 | 미확인 |
| SpikF-GO | ECML-PKDD 2026 | 주 표 5, 부록 3 | 전부 재실행 | 45 nm, memory + 연산 + addressing, Solar H12 |
| NSMT | 제출본, 게재처 미확인 | 주 표 1 (seed 42), 검정 7 | [43] SpikF 와 [41] TimesNet 에서 옮김 | 45 nm, 실수 연산 포함, ETT H720 |
| SpikeLite | arXiv | 3 | SpikF, Lv 에서 옮김, 일부 재구현 | 45 nm, head 제외, ECL H720 |
| SpikySpace | arXiv (Preprint) | 본문에서 찾지 못함 | 미표기 | 22 nm, memory 이동 포함, Electricity H3 |
| SpikeSTAG | arXiv | 3 | Lv 계열 | Electricity |
| SEDformer | arXiv | 5, paired t-test | 자체 희소화 데이터라 재실행으로 보임 | 45 nm |
| Closed-loop S4D | arXiv | 5 initializations | S4, SpaceTime 은 옮김 | 없음 |
| 저널과 소규모 학회 | Neurocomputing, Information Sciences, PeerJ CS, IJCIS, Electronics, Biomimetics, IEEE ICNC, IEEE MASS poster, LNCS | 미확인 | 미확인 | 미확인 |

근거 표시: 위 표의 각 행은 질문 1 의 카드와 같은 출처다. [read in full text] (저널과 소규모 학회 행, CPG-PE 와 RPE 의 energy 칸은 [abstract only] 또는 미확인) — [Lv](https://arxiv.org/abs/2402.01533); [TS-LIF](https://arxiv.org/abs/2503.05108); [SpikF](https://proceedings.mlr.press/v267/wu25m.html); [CPG-PE](https://arxiv.org/abs/2405.14362); [RPE](https://arxiv.org/abs/2501.16745); [SpikF-GO](https://arxiv.org/abs/2606.13901); [SpikeLite](https://arxiv.org/abs/2609.35097); [SpikySpace](https://arxiv.org/abs/2601.02411); [SpikeSTAG](https://arxiv.org/abs/2508.02069); [SEDformer](https://arxiv.org/abs/2602.02230); [Closed-loop](https://arxiv.org/abs/2610.28495)

- Baseline 재실행은 예외적이다. SpikF-GO 는 FourierGNN 코드의 leakage 를 고치고 전부 다시 돌렸다고 적으며, "prior SNN forecasting studies lack a unified experimental comparison" 이라고 지적한다. [read in full text] — [SpikF-GO](https://arxiv.org/abs/2606.13901)
- 공통 energy 공식: SOPs = firing rate x T x FLOPs, E_AC 0.9 pJ, E_MAC 4.6 pJ. 범위는 논문마다 다르다. SpikeLite 는 "prediction heads are excluded", SpikF Table 2 는 feature extraction 만, NSMT 는 실수 재구성층과 head 를 MAC 로 포함한다. [read in full text] — [SpikF](https://proceedings.mlr.press/v267/wu25m.html); [SpikeLite](https://arxiv.org/abs/2609.35097); NSMT.pdf
- 실제 하드웨어 언급: SpikF-GO 는 "does not yet include deployment on real neuromorphic hardware" 라고 명시한다. 내가 읽은 다른 forecasting 논문에도 칩 측정값은 없다. [read in full text] — [SpikF-GO](https://arxiv.org/abs/2606.13901)
- 상위 학회 논문의 격차 크기: SpikF 는 iTransformer 대비 MAE "1.9%", energy "3.16x" 이다. 다만 이 1.9% 비교의 iTransformer 는 MSE loss 이고 SpikF 는 MAE loss 이다. MAE loss iTransformer 와 비교하면 ETTh2 H96 MAE 0.336 대 0.342, ETTm2 H96 0.251 대 0.265 이다. [read in full text] — [SpikF](https://proceedings.mlr.press/v267/wu25m.html)
- 상위 학회 논문의 분석 요소: TS-LIF 는 안정성 정리와 transfer function, RPE 는 Gray Code 와 Hamming distance 에 관한 증명, CPG-PE 는 생물학적 근거와 PE 분석, Lv 는 이 과제 최초의 체계적 틀이다. [read in full text] — [TS-LIF](https://arxiv.org/abs/2503.05108); [RPE](https://arxiv.org/abs/2501.16745); [CPG-PE](https://arxiv.org/abs/2405.14362); [Lv](https://arxiv.org/abs/2402.01533)
- 비신경망 쪽 기준 참고: TimeMixer 는 3 repeats 와 통계 검정 표를, iTransformer 는 강건성 표에 5 seeds 를 제시한다. [read in full text] — [TimeMixer](https://arxiv.org/abs/2405.14616); [iTransformer](https://arxiv.org/abs/2310.06625)

### Inferences
- 우리 절차 (3 seeds, 사전 등록 판정, 같은 stopping 으로 baseline 재실행, 실패 시 중단하는 검사) 는 이 분야 평균보다 엄격하다. 이것 자체가 차별점이 될 수 있지만 그것만으로 상위 학회를 통과하지는 못한다.
- 상위 학회 사례의 공통점은 "새 mechanism + 수학적 또는 구조적 분석 + 집계상 이김" 이다. SpikMixer 는 현재 세 번째 조건을 충족하지 못한다. Spikformer-style 대비 평균 +0.32%, energy 0.98배이므로 효과 자체가 없다고 읽히고, SpikF 대비로도 집계상 뒤진다.
- 2025-2026 년 분포는 arXiv preprint 와 2급 학회 및 저널에 몰려 있다. 같은 틀에 module 하나를 더한 논문은 상위 학회에 잘 가지 않는 것으로 보인다. 다만 표본이 적고 심사 탈락 논문은 관찰되지 않으므로 강한 결론은 아니다.
- 이 분야의 energy 비교는 범위가 맞지 않는 수를 나란히 놓는 경우가 많아, head 포함 일관된 재계산은 그 자체로 쓸모 있는 기여다.

### Gaps
- 심사 기록은 공개되어 있지 않으므로 어떤 조건이 게재를 결정했는지는 추정이다.
- RPE 와 CPG-PE 의 forecasting energy 보고 여부는 본문 전체를 대조하지 않아 미확인으로 두었다.
- SpikeSTAG 와 SpikySpace 가 이후 어디에 게재되었는지는 확인하지 못했다. SpikySpace 의 인용 표현상 투고 중일 수 있으나 근거가 없다.

## 질문 4. Spiking Mixer forecaster 가 이미 있는가, 그리고 SpikMixer 의 겹침과 빈 자리

### Takeaway
"Spiking Mixer" 이름의 forecaster 는 찾지 못했다. Spiking Mixer 는 vision, 고장 진단, 빔 탐색 쪽에 있고 forecasting 에는 없다. 그러나 **코드상의 사실로는 SpikF 저장소의 "SpikeRNN" 이 이미 반복 T 위의 spiking token-mixing MLP 이고, SpikF 와 NSMT 의 ETT 표에 올라 있다.** SpikMixer 의 고유한 부분은 "token 축 Linear 뒤 같은 축으로 neuron 진행 + 순방향과 역방향 + |H_f - H_r| spiking gate" 로 좁혀진다. 메커니즘은 비어 있지만 정확도 이득이 없어 메커니즘 신규성만으로는 약하다. 더 유망한 빈 자리는 "spiking time axis 의 의미를 통제 비교한 연구" 와 "ETT 에서 head 포함 일관 기준의 energy 비교" 이다.

### Cited Findings

#### Mixer 계열 검색 결과
- 검색: 웹 검색 3회, OpenAlex 질의 ("spiking" + "mixer" 56건, 제목 "spiking mixer" 8건, "spiking" + "MLP-Mixer" 3건). Forecasting 인 Spiking Mixer 는 없었다. "spiking" + "ETTm2" 초록 검색은 0건, "spiking" + "ETTh1" 은 4건이며 SNN 논문이 아니다. 초록에 ETT 를 적지 않는 논문도 많아 후자는 약한 근거다. [abstract only] — [OpenAlex](https://api.openalex.org/works)
- SMixer: ICLR 2026, "event-driven Spiking Mixer (SMixer)" 를 기본 구조로 쓰고 Spatial-Temporal Pruning 을 제안한다. 초록 페이지에 forecasting 언급이 없고 과제와 데이터도 적혀 있지 않다. [abstract only] — [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10011328)
- Spiking Token Mixer (STMixer): NeurIPS 2024, Spikformer 의 SSA 를 convolution, fully connected, residual 로 대체하는 event-driven 구조. [secondary summary] — [NeurIPS poster](https://nips.cc/virtual/2024/poster/93999); [DOI](https://doi.org/10.52202/079017-4092)
- Efficient Deep Spiking MLP: token block 의 lightweight axial sampling 과 곱셈 없는 추론을 다룬다. [secondary summary] — [arXiv 2306.12465](https://arxiv.org/abs/2306.12465)
- SpikeMixer: "Dynamic axial mixing for spiking neural networks", Neurocomputing 2026. 초록을 얻지 못했다. **우리 이름과 거의 같다.** [abstract only: 제목과 서지 정보만] — [DOI](https://doi.org/10.1016/j.neucom.2026.134120)
- SPMF-Mixer: Advanced Engineering Informatics 2026, attention-free, spiking wavelet, 베어링 고장 진단. 분류 과제다. [abstract only] — [DOI](https://doi.org/10.1016/j.aei.2026.105342)
- IDSM: GCCE 2023, 개별 spiking encoder + feature mixer, mmWave 빔 탐색. [abstract only] — [DOI](https://doi.org/10.1109/gcce59613.2023.10315443)
- ASFNOformer ("Frequency Domain Token Mixer in Spiking Transformer", Electronics 2025), Self-Supervised Mixer-to-Spike Learning (Harvard Dataverse 2026, 분류): 제목상 forecasting 이 아니다. [abstract only: 제목과 서지 정보만] — [ASFNOformer DOI](https://doi.org/10.3390/electronics14244860); [Dataverse DOI](https://doi.org/10.7910/dvn/uki6mp)
- 양방향 검색: 순방향과 역방향 stream 을 gate 로 합치는 SNN forecaster 를 찾지 못했다. SeqSNN 의 `is_bidir: True` 는 ANN RNN 설정이고, arXiv 2509.20284 의 "bidirectional" 은 시퀀스 방향이 아니라 feedforward 와 backward 네트워크를 뜻한다. [read in full text: repository code]; [secondary summary] — [rnn2d.yml](https://github.com/microsoft/SeqSNN/blob/main/exp/forecast/model/rnn2d.yml); [arXiv 2509.20284](https://arxiv.org/abs/2509.20284)

#### SpikMixer 구성요소별 겹침

| SpikMixer 구성요소 | 가장 가까운 선행 | 겹침 정도 | 근거 |
|---|---|---|---|
| Patch token, 반복 T, Linear + BN + LIF | SpikF SPE, SN(BN(LN(p_k))) + ZOH. NSMT 식 2 | 완전히 겹침 | [read in full text] [SpikF](https://proceedings.mlr.press/v267/wu25m.html); NSMT.pdf |
| Attention 없는 구조 | SpikF ("attention-free framework"), SpikySpace, SpikeLite 의 FSSE-only 경로 | 겹침 | [read in full text] [SpikF](https://proceedings.mlr.press/v267/wu25m.html); [SpikySpace](https://arxiv.org/abs/2601.02411); [SpikeLite](https://arxiv.org/abs/2609.35097) |
| Token 축 Linear mixing, BN, spike | SpikF 저장소 `SpikeRNN.py` (이름은 RNN), `iSpikformer.py` 의 `Linear(patch_num, patch_num)` | 대체로 겹침, 단 neuron 은 T 축 | [read in full text: repository code] [SpikeRNN.py](https://github.com/WWJ-creator/SpikF/blob/main/models/SpikeRNN.py) |
| Neuron 이 token 축을 따라 진행 (T = N) | NSMT Neocortex (T = N, 위치 정보 주장까지 같음) | 개념은 겹침. 단 NSMT 는 D 축 MLP 이고 token Linear 가 없음 | [read in full text] NSMT.pdf |
| Neuron 이 원 관측축을 따라 진행 | SpikeLite FSSE, Spike-GRU (L x T), 단변량 S4D, SEDformer 의 event-aligned | 다른 축이지만 같은 발상 | [read in full text] [SpikeLite](https://arxiv.org/abs/2609.35097); [read in full text: repository code] [spikegru.py](https://github.com/microsoft/SeqSNN/blob/main/SeqSNN/network/snn/spikegru.py) |
| Token Linear 뒤 같은 축으로 neuron 진행 | 찾지 못함 | 비어 있음 | 위 검색 범위 |
| 순방향과 역방향 stream, 분리된 가중치 | 찾지 못함 | 비어 있음 | 위 검색 범위 |
| \|H_f - H_r\| spiking gate | SpikeSTAG 의 spike-gated fusion (LSTM 과 SSA), SpikF-GO 의 Complex LIF gate, NSMT 의 IAND 융합 | gate 일반은 겹치고 차이 기반 gate 는 찾지 못함 | [read in full text] [SpikeSTAG](https://arxiv.org/abs/2508.02069); [SpikF-GO](https://arxiv.org/abs/2606.13901); NSMT.pdf |
| Spiking MLP + head | 전부 | 완전히 겹침 | [read in full text] 질문 1 카드 |
| ETT 에서 energy 보고 | NSMT Table 6 뿐 (SpikF 값도 NSMT 가 계산) | 드물다 | [read in full text] NSMT.pdf |

- 우리 수치: SpikMixer energy 는 window 당 약 9-11 uJ 이고 Spikformer-style 의 0.98배다. NSMT 공개값은 H720 에서 44.90, 46.72 uJ, 같은 표의 SpikF 는 1050 uJ 근처다. [given in assignment]; [read in full text] — NSMT.pdf
- 시간 축 ablation 선행: NSMT Table 7 이 alignment 유무를 비교하지만 정확도 차이 없음, energy 차이만 보고한다. SpikF 는 Ts 를 1-16 으로 바꾸어 변동 3.7% 를 보고한다. Lv 는 Ts 4-16 에서 "R^2 values remain relatively stable" 이라고 한다. SpikF-GO 는 Ts = 8 또는 12 까지 개선된다고 한다. 어느 것도 "neuron 진행 방향과 순서를 바꾸는" 통제 실험이 아니다. [read in full text] — NSMT.pdf; [SpikF](https://proceedings.mlr.press/v267/wu25m.html); [Lv](https://arxiv.org/abs/2402.01533); [SpikF-GO](https://arxiv.org/abs/2606.13901)
- 입력 순서 섞기 실험은 ANN 쪽에 있다 (DLinear Table 5, "Shuf." 와 "Half-EX."). 내가 읽은 SNN forecasting 논문에서는 같은 실험을 찾지 못했다. [read in full text] — [DLinear](https://arxiv.org/abs/2205.13504)
- 통합 benchmark 는 SpikingTSF 가 시작했다. 비교 축은 모델과 neuron 이고 시간 축의 의미가 아니다. 결과표는 미완성이며 사용 전 독립 검증을 요구한다. [read in full text: repository code] — [SpikingTSF](https://github.com/spikora/SpikingTSF)

### Inferences
- 판단 1 (신규성). "최초의 spiking Mixer forecaster" 는 문구로는 쓸 수 있지만 위험하다. SpikF 저장소의 SpikeRNN 이 이미 같은 형태에 가까우므로, 리뷰어나 저자가 지적할 수 있다. 쓰더라도 "neuron 이 token 축을 따라 진행하는 양방향 mixer" 로 좁혀야 한다. 이름 충돌 (SpikeMixer, SMixer) 도 있어 이름을 재검토할 이유가 있다.
- 판단 2 (근거 강도). 현재 공개 비교에서 SpikMixer 의 고유 요소가 만든 이득은 측정되지 않았다. Spikformer-style 대비 +0.32%, energy 0.98배, RLinear 와 동률, SpikeRNN (SPE) 대비 -0.77% 이다. 이 상태로는 구조 제안 논문의 핵심 표가 성립하지 않는다.
- 판단 3 (빈 자리 A, 가장 유망). **Spiking time axis 의 의미를 통제 비교한 연구.** 같은 front end, 같은 stopping, 3 seeds 이상에서 T 를 반복, token 순방향, token 역방향, 섞은 token 순서, 양방향으로만 바꾸고, RLinear, SPE + token Linear (SpikeRNN 형태), SpikF, 같은 token 수의 신경 없는 MLP 를 baseline 으로 둔다. 질문은 "neuron 이 순서를 실제로 쓰는가, 쓴다면 어떤 예측 길이에서인가" 이다. 내가 찾은 범위에 이 설계는 없고, 우리 저장소에 이미 CinC 쪽 순서 실험과 head 진단이 있어 비용이 낮다. 다만 결과가 "차이 없음" 일 가능성이 높으며, 그러면 부정적 결과 논문이 되어 상위 학회는 어려울 수 있다.
- 판단 4 (빈 자리 B). **ETT 에서 head 포함 일관 기준의 energy 비교.** 지금은 SFS 전용, head 제외, 22 nm, 옮겨 온 수치가 섞여 있고 ETT 값은 NSMT 표뿐이다. SpikF 를 우리 계산기로 다시 세었을 때 T = 16 과 T = 4 의 차이, 실수 head 와 BN 의 몫이 얼마인지 보이면 실용적 기여가 된다. 정확도가 같은 상태에서 energy 차이가 크다면 이것이 가장 방어하기 쉬운 주장이다. 단, 9-11 uJ 와 44.90 uJ 를 바로 비교하면 안 된다. 계산 범위와 horizon 을 맞춰야 한다.
- 판단 5 (빈 자리 C). 긴 입력. T = N 은 N 이 커지면 step 수와 비용이 함께 늘어나고, ANN 은 L = 336 이상에서 크게 좋아진다. SpikF 는 look-back 을 720 까지 늘렸지만 T = N 모델의 비용과 정확도를 같이 본 공개 결과는 찾지 못했다. 시도 가치는 있으나 ETT 범위 결정 (ETTh2, ETTm2, H96, H720) 과 입력 길이 96 고정 여부를 먼저 정해야 한다.
- 판단 6 (회피할 자리). 새 neuron (TS-LIF, BCDQ-TCLIF), 새 PE (CPG-PE, RPE), Fourier 선택 (SpikF, SpikF-GO), graph (SpikeSTAG, SpikF-GO), state space (SpikySpace, 단변량 S4D) 는 이미 차지되었고 빠르게 늘고 있다. Fudan 과 MSRA 계열이 2024-2026 에 연속 발표 중이다 (Lv, CPG-PE, RPE, SpikeSTAG, SpikeLite).
- 판단 7 (재현 위험). 우리가 baseline 으로 삼을 SpikF ETT 값은 짧은 stopping, 데이터별 조정, 공개 스크립트의 seed 미전달 문제가 있다. 비교에는 공개값이 아니라 같은 규칙의 재실행값을 쓰는 것이 안전하다.

### Gaps
- SpikeMixer (Neurocomputing 2026) 본문을 읽지 못했다. OpenAlex 메타데이터상 SSRN 사본이 있으나 자동 내려받기가 실패했다. 제목과 축 관련 표현으로 보아 forecasting 이 아니라 일반 SNN 구조로 추정하지만 확인되지 않았다. 사람이 확인해야 한다.
- 검색으로 찾지 못했다는 것은 없다는 증거가 아니다. DBLP 와 OpenReview 는 이 서버에서 막혀 있고 Semantic Scholar search 는 429 였다. ICLR 2026, NeurIPS 2026 수락 논문의 완전한 대조는 못 했다.
- SMixer 의 본문은 읽지 않았으므로 시계열 실험 부록 유무를 확인하지 못했다. 초록 페이지에는 언급이 없다.
- Zenodo 자가 등록 문서 다수 (예: "Spiking Neural Networks for Multivariate Time Series Forecasting") 는 심사 근거가 없어 제외했다. 초록상 구조적 정보도 거의 없다.
- 우리 모델의 SOP 계산이 T = N 구간과 반복 T 구간을 위 논문들과 같은 방식으로 세는지는 이 조사 범위 밖이다. 비교 전에 별도로 맞춰야 한다.
