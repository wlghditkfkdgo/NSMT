# PRF 이후 SpikMixer에 남은 것은 gate뿐이다

**PRF는 attention 없는 neuron token mixing, T = sequence length, forward + reversed 두 stream을 2024-10에 이미 공개했다.** SpikMixer에서 새 범주로 남는 것은 하나다. forward와 reversed stream의 membrane 차이 `|H_f − H_r|`로 binary gate `s_d`를 만들고, 그 gate가 learned path와 `F AND R` 중 하나를 고르는 조합이다. 이 조합은 이번 검색 범위에서 찾지 못했다. 다만 가장 급한 후보 SpikeVFuse의 본문을 읽지 못했다. 효과도 일반적이지 않다. gate 유무 비교는 CinC fold 0 +1.56 %p, ETT R +0.57 %(나쁜 방향), MIT-BIH +0.00 %p다. Spikformer-style 대비로도 ETT, MIT-BIH, AD는 no clear difference이고 energy는 0.98–1.03배다. 뚜렷한 우위는 CinC fold 0의 +2.16 %p뿐이고, 사용자는 11-04에 CinC를 main result에서 뺐다. T_h repeat 축, token-axis dynamics, bidirectional, IAND·OR residual, ETT 평가는 고유점이 아니다. reviewer의 첫 질문은 "PRF와 더 단순한 mixer를 같은 규칙으로 돌렸는가"일 것이고, 지금 답은 not run이다. 어느 길로 갈지는 사용자 결정으로 남는다.

**근거 표시.** `[full text]`는 원문 해당 부분을 직접 읽었다는 뜻이고, 일부만 읽었으면 `[full text, 일부]`로 적는다. `[code]`는 공개 code를 읽었지만 실행하지 않았다는 뜻이다. `[abstract only]`, `[secondary summary]`, `[grep only]`는 각각 초록만, 제3자 요약만, 본문 키워드 검색만 봤다는 뜻이다. `[calculation]`은 논문 표의 값이 아니라 노트 또는 이 보고서가 계산한 값이고, `[inference]`는 추론이다. `[local]`은 우리 저장소의 code, CSV, 결과, log를 이 보고서 작성 중 읽기 전용으로 다시 확인했다는 뜻이다. training은 돌리지 않았다. "찾지 못했다"는 이번 검색 범위의 결과이고 없다는 뜻이 아니다. R은 칸별 seed 평균 MSE 상대 차이의 평균이고 음수가 좋은 쪽이다. LRA는 긴 sequence 분류 5개 task(ListOps, Text, Retrieval, Image, Pathfinder)다. SSM은 state-space model, NAO는 N·AF·Other F1의 평균, SPE는 SpikF의 Spiking Patch Encoder, RevIN은 입력별 normalization이다.

**이 보고서의 SpikMixer.** code의 `SpkMixer`, fusion `muxlin`이다. patch token을 T_h = 4번 그대로 복사하고, embedding과 MLP의 neuron은 이 repeat 축을 4 step 걷는다. mixer는 `[N, B, T_h, D]`로 축을 바꾼 뒤 layer마다 `Linear(D→D) → BN → LIF`를 token 축으로 N step 돈다. reversed stream은 `xm.flip(0)`을 자기 weights로 돌리고 다시 뒤집는다. `s_d = LIF(BN(|H_f − H_r|))`, `L = LIF(BN(W[F OR R; s_d]))`이고 출력은 `s_d·L + (1−s_d)·min(F, R)`이다. `s_d`가 1이면 L, 0이면 `F AND R`을 고른다. 이 선택을 MUX라 부른다. `sdlin`은 MUX 없이 L만 쓰는 control이다. layer 사이 residual은 IAND `x(1−m)`, OR `max(x, m)`, ADD `x + m` 중 하나다 ([spk_mixer.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/spk_mixer.py)) [local]. **Linear는 D 축이므로 token 사이 정보는 neuron state로만 움직인다** ([model.py `NeoLayer`](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/model.py), [neo_mixer.py docstring](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/neo_mixer.py)) [local].

## PRF는 SpikMixer mixer의 앞 세 요소를 이미 공개했다

PRF neuron은 spike를 되먹이는 reset이 없다. membrane은 입력의 complex convolution이고 실수부에만 threshold를 건다: `ũ_t = exp(Δγ̃)ũ_{t−1} + Δc_t`, `s_t = H(Re{ũ_t} − V_th)`. 논문은 "When θ = 0, the model reduces to the standard LIF neuron without reset"라고 적는다 ([PRF v2](https://arxiv.org/pdf/2410.03530v2)) [full text]. 채널마다 complex pole 하나인 linear filter에 Heaviside를 붙인 것과 같다 [inference]. layer는 `PRF neuron → Linear → 더하기 → Spatial Neuron → Linear → 더하기`이고 Spatial Neuron은 state가 없는 threshold다. **T는 "aligning with the sequence length"이고, patch도 repeat 축도 없다.** bidirectional은 App. H Eq. 102다: "reverse it and pass it through an independent second PRF Neuron" 뒤에 spike를 concat하고 `W ∈ R^{2D×D}`를 건다. Image와 Pathfinder에만 쓰고 ListOps, Text, Retrieval은 causal이다 ([PRF v2](https://arxiv.org/pdf/2410.03530v2)) [full text]. 공개 code도 같다 ([spikingblock.py](https://github.com/HuuYuLong/prf/blob/main/src/models/sequence/modules/spikingblock.py)) [code]. 게재본은 Pattern Recognition vol. 183, article 114968이다 ([Crossref](https://api.crossref.org/works/10.1016/j.patcog.2026.114968)). 게재본 본문은 읽지 못했다.

| Task | L | 구조 | S4 accuracy | SD-TCM accuracy | S4 mJ | SD-TCM mJ | 평균 firing rate % |
|---|---:|---|---:|---:|---:|---:|---:|
| ListOps | 2,048 | causal | 59.60 | 59.20 | 5.104 | 0.075 | 3.53 |
| Text | 4,096 | causal | 86.82 | 86.33 | 3.718 | 0.298 | 4.32 |
| Retrieval | 4,000 | causal | 90.90 | 89.88 | 24.439 | 0.211 | 1.48 |
| Image | 1,024 | bidirectional | 88.65 | 84.77 | 19.222 | 0.187 | 3.47 |
| Pathfinder | 1,024 | bidirectional | 94.20 | 91.76 | 6.256 | 0.067 | 3.29 |
| 평균 | | | 84.03 | 82.39 | 11.748 | 0.168 | |

출처는 [PRF v2 Table 3, 5, 11](https://arxiv.org/pdf/2410.03530v2) [full text]다. 이 energy는 그대로 기준으로 삼기 어렵다. 논문에 pJ 값이 없지만, E_M = E_MAC = 4.6과 E_AC = 0.9를 넣으면 Ours 5칸이 재현된다 [calculation]. 이때 두 번째 stream, BN, embedding, head가 빠진다. code대로 두 번째 stream을 세면 Image와 Pathfinder는 약 0.321, 0.121 mJ로 1.7–1.8배다 [calculation]. neuron의 floating-point 곱셈이 energy의 43–75 %를 차지한다 [calculation]. 모델 선택은 README 명령 기준 `train.monitor=test/accuracy`다 ([README](https://github.com/HuuYuLong/prf/blob/main/README.md)) [code]. 제3자 재현은 SD-TCM firing rate 평균 9.3 %를 적는다. 다만 원 code 공개 전의 재현이다 ([D-RF Table 3](https://arxiv.org/abs/2509.17186)) [full text].

| 요소 | PRF SD-TCM | SpikMixer | 판정 |
|---|---|---|---|
| Token mixing | neuron state, Linear는 D 축 | 같음, `Linear(D→D) → BN → LIF` | **같은 방식** |
| Neuron | complex pole, reset 없음, step당 FP 곱셈 5회 | LIF, reset 있음, τ = 2 | 변형 |
| T | T = L, raw token 784–4,096 | mixer만 T = N, patch token: MIT-BIH 11, ETT 24, CinC 117 | 같은 발상, 길이는 크게 다름 |
| 실제로 쓰는 길이 | 긴 범위 task로 평가 | test 시 context를 제한해 보면 방향당 4–16 token이면 충분 [local] | **우리는 긴 기억을 주장할 수 없음** |
| Repeat 축 | 없음 | embedding과 MLP에 T_h = 4 | PRF와는 차이, 다른 논문이 덮음 |
| Bidirectional | 같은 입력 u를 θ, Δ만 다른 두 neuron이 읽음 | stream마다 Linear, BN, neuron | 변형 |
| Fusion | spike concat → Linear(2D→D) | `s_d` MUX | **실질적 차이** |
| Layer 사이 신호 | 실수 membrane shortcut | IAND·OR은 binary, ADD는 정수 | 차이, 단 task별 선택에 따라 다름 |
| 데이터 | sMNIST, psMNIST, LRA | ETT, MIT-BIH, CinC, AD | 차이, 성과는 아님 |

context 측정은 test 시 개입이다. log는 이것을 "training된 모델이 기대는 정도"로 해석하고 필요성의 근거로 쓰지 않는다 ([PROJECT_LOG 11-05](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/docs/PROJECT_LOG.md)) [local].

## 나머지 갈래도 남은 조각을 대부분 덮는다

**Attention 없는 mixing은 오래된 주장이다.** Wang 2023은 FFT나 Haar를 쓰는 `X′ = SN(BN(LT(X)))`로 "we argue that the utilization of SSA is unnecessary in the context of SNNs"라고 썼다 ([Wang et al.](https://arxiv.org/abs/2308.02557)) [full text]. STMixer는 `W_STM ∈ R^{N×N}`로 SSA를 대체한다. neuron은 T 축을 걷는다. ImageNet 8-512는 73.82(T=1) 대 Spikformer 73.38(T=4), energy는 2.20 대 11.57 mJ다. 이 energy 차이는 대부분 T 차이에서 온다 [inference]. 같은 표에서 training 방법 교체(−2.38)가 mixer 교체(−1.22)보다 크다 [calculation] ([STMixer](https://proceedings.neurips.cc/paper_files/paper/2024/file/e8c20cafe841cba3e31a17488dc9c3f1-Paper-Conference.pdf)) [full text]. SMixer는 `SN(W_M X)`를 쓰고 주 contribution은 pruning이다. 교체쌍 9개는 accuracy −0.55 ~ +0.97 %p, energy 0.65–0.85배다 [calculation] ([SMixer](https://proceedings.iclr.cc/paper_files/paper/2026/file/aaeb635296295aa7c9f8b91c5e10960d-Paper-Conference.pdf)) [full text]. TEFormer와 [Lapis](https://arxiv.org/abs/2608.11865)는 attention을 유지하므로 이 갈래가 아니다 [full text].

**T = sequence length도 새 축이 아니다.** S4 계열과 Spiking LMU는 T = L이다 ([Binary S4D](https://arxiv.org/abs/2401.00955), [P-SpikeSSM](https://arxiv.org/abs/2406.02923), [SpikingSSMs](https://arxiv.org/abs/2408.14909), [LMUFormer](https://arxiv.org/abs/2402.04882)) [full text]. SpikeGPT는 "the optimal approach is not to introduce a new dimension, but to leverage the existing sequence dimension"이라고 썼다 ([SpikeGPT](https://arxiv.org/abs/2302.13939)) [full text, 일부]. SNN-MLP는 patch index를 t로 쓰는 full-precision LIF다. 또 "global LIF significantly worse than group LIF"(81.53 대 81.87)를 보고한다 ([SNN-MLP](https://arxiv.org/abs/2203.14679)) [full text]. SmolMamba 표의 T = 64 baseline은 CIFAR-10에서 84.5, 78.3이고 자기 모델은 95.7이다. 인용값이고 이미지다 ([SmolMamba](https://arxiv.org/abs/2604.25570)) [full text, 일부]. T = N이 자동으로 이득은 아니라는 선행 근거다 [inference]. **우리 NSMT Neocortex도 이미 patch token + T = N이다** ([NSMT 제출본](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/papers/NSMT.pdf)) [full text].

**Bidirectional은 입력 reversal 방식과 축으로 갈린다.** Yin 2021은 "two SRNN layers ... reading the sequence from the forward and backward direction" 뒤 mean을 non-spiking integrator에 넣는다. TIMIT 66.1이고 Bi-LSTM은 68.9다 ([Yin et al.](https://arxiv.org/abs/2103.12593)) [full text, 일부]. SmolMamba는 token 축을 양쪽으로 scan하고 더한다. SPM은 Time Flip이 Token Flip보다 낫고, 둘을 같이 쓰면 단방향 수준으로 떨어진다고 보고한다 ([SPM](https://arxiv.org/abs/2504.14371)) [full text]. TEFormer의 bidirectional은 parallel stream이 아니다. attention이 forward, T-MLP가 backward를 맡아 직렬이고 축은 T다 ([TEFormer](https://arxiv.org/abs/2601.18274)) [full text]. 같은 구조에서 방향만 바꾼 이득은 +0.15–1.3 %p로 작다 [calculation]. 시간 축에서는 비판도 분명하다. SiLIF는 "the whole sequence needs to be buffered"라고 적고 forward mode로 다시 training한다 ([SiLIF](https://arxiv.org/abs/2506.06374)) [full text].

**시계열과 ECG는 일부만 겹친다.** ETT 쪽 SNN은 찾은 범위에서 NSMT, SpikF(와 그 baseline), SpikeLite, SpikingTSF다. SpikF에서는 neuron이 token을 걷지 않는다. token 순서는 patch 축 FFT selector와 convolution이 맡는다. T_s를 1–16으로 바꿔도 "average variation of only 3.7%"다. selector를 뺀 `w/o`는 ETTh1, ETTm1, Weather에서 원본과 거의 같다. 제거 뒤에도 convolution은 남는다 ([SpikF](https://proceedings.mlr.press/v267/wu25m.html)) [full text, 일부]. SeqSNN, TS-LIF, SpikySpace, SMixer의 forecasting은 Metr-la, Pems-bay, Solar, Electricity이고 ETT가 없다 ([Lv et al.](https://arxiv.org/abs/2402.01533), [TS-LIF](https://arxiv.org/abs/2503.05108), [SpikySpace](https://arxiv.org/abs/2601.02411)) [full text]. ECG에서 SSM 계열 relative는 SpikeMamba뿐이다. LIF 자체가 state이고 MIT-BIH, PTB, PTB-XL을 쓰지만 단방향이다 ([SpikeMamba](https://papers.miccai.org/miccai-2026/paper/4698_paper.pdf), [spike_interface.py](https://github.com/Brain-Inspired-Computing-Lab/SpikeMamba/blob/main/mamba_ssm/ops/spike_interface.py)) [full text, code]. ECG SNN 다수는 원래 CNN이나 MLP라서 "attention-free"는 ECG 쪽에서 새 주장이 아니다 ([Scrugli 2026](https://zenodo.org/records/18887975)) [full text, 일부].

### 구성요소별 겹침

덮음은 같은 mechanism, 부분은 같은 기능을 다른 mechanism으로 한다는 뜻이다. 없음은 읽은 범위에 없다는 뜻이고, 미확인은 노트에 기록이 없다는 뜻이다.

| 논문 (근거) | Token mixing 수단 | Neuron이 걷는 축, T | 옆의 repeat 축 | Reversed stream, stream별 weights | Stream 비교 신호 | Fusion, gate | Residual | ETT / ECG |
|---|---|---|---|---|---|---|---|---|
| **SpikMixer** [local] | LIF state, Linear는 D 축 | mixer N step, embedding·MLP T_h step | T_h = 4, exact copy | token 축, Linear·BN·neuron 전부 분리 | `\|H_f − H_r\|`, real, BN 뒤 LIF | `s_d·L + (1−s_d)·min(F,R)` | IAND, OR, ADD | ETT, MIT-BIH, CinC |
| [PRF](https://arxiv.org/pdf/2410.03530v2) [full text, code] | **덮음.** PRF neuron state, reset 없음 | **덮음.** T = L, raw token | 없음 | **덮음, 더 좁음.** θ, Δ만 분리, Image·Pathfinder만 | 없음 | 다름. concat + Linear(2D→D) | 없음. real membrane shortcut | 없음. 인용 5편 grep 0건, 나머지 미확인 |
| [SpikF](https://proceedings.mlr.press/v267/wu25m.html) [full text 일부, code] | 부분. patch 축 FFT selector | 없음. T 축만, ETT T_s = 16 | **덮음.** ZOH, `x.repeat` | 없음 | 없음 | selector는 주파수 선택, stream fusion 아님 | 미확인 | **ETT 덮음**, ECG 없음 |
| [SpikF `SpikeRNN.py`](https://github.com/WWJ-creator/SpikF/blob/main/models/SpikeRNN.py) [code] | 부분. `Linear(patch_num + hidden_dim, hidden_dim)`, N 축 learned | 없음. T 축만, 이름과 달리 N 축 loop 없음 | **덮음.** T = 16 | 없음 | 없음 | 입력과 concat, gate 없음 | 미확인 | **ETT 덮음**, ECG 없음 |
| [STMixer](https://proceedings.neurips.cc/paper_files/paper/2024/file/e8c20cafe841cba3e31a17488dc9c3f1-Paper-Conference.pdf) [full text, [code](https://github.com/brain-intelligence-lab/STMixer_demo)] | 부분. `W ∈ R^{N×N}` | 없음. T 축, T = 1 / 4 | **덮음.** simulation step | 없음 | 없음 | 없음 | 없음. `x + token_mix(x)` | 없음. vision |
| [SMixer](https://proceedings.iclr.cc/paper_files/paper/2026/file/aaeb635296295aa7c9f8b91c5e10960d-Paper-Conference.pdf) [full text] | 부분. `SN(W_M X)` | 없음. T 축, 시계열 T = 4 | **덮음** | 없음 | 없음 | 없음 | 미확인 | ETT 없음, 다른 forecasting 4종 |
| [SmolMamba](https://arxiv.org/abs/2604.25570) [full text 일부] | 부분. real SSM이 N을 scan, 뒤에 SN | 부분. SSM은 N, SN은 T | **덮음.** T = 4 / 16, 같은 `(b, t, c)` slice | **덮음.** token 축 Reverser, weights 분리 명시 없음 | 없음 | 다름. 합 + Mask. "implicit gating" 주장 | real 합 | 없음. vision |
| [SpikeMamba](https://papers.miccai.org/miccai-2026/paper/4698_paper.pdf) [full text, code] | **덮음(단방향).** reset 있는 LIF가 state, DWConv 포함 | **덮음.** T = L, raw, MIT-BIH 360 | 없음. 외부 T = 1 | 없음. code에 flip 없음 | 없음. 활동량 기반 | 다름. sample별 1-bit로 update 생략, 논문 Eq. 6과 code가 다름 | real, LN | ETT 없음, **ECG 덮음** |
| [Yin 2021](https://arxiv.org/abs/2103.12593) [full text 일부] | 부분. recurrent weights | **덮음.** 입력 time step | 없음 | **덮음.** 두 SRNN, 분리로 읽힘 | 없음 | mean → non-spiking integrator | 미확인 | bidirectional은 TIMIT, ECG(QTDB)는 단방향 |
| [XOResNet](https://arxiv.org/abs/2605.30362) [full text] | 없음. convolution ResNet | 미확인 | 미확인 | 없음. 같은 입력의 두 branch | **가장 가까움.** `f1 ⊕ f2`, spike 수준 | gate 아님. XOR이 다음 branch의 입력 | **덮음.** XOR, OR-ADD | 없음. vision |
| [TEFormer](https://arxiv.org/abs/2601.18274) [full text] | 없음. attention 유지 | T 축 backward recurrence | **덮음**, "severe temporal redundancy" 비판 | 다름. 입력 reversal 없음, 직렬 | 없음 | 다름. sigmoid real gate | 미확인 | 없음. SHD, sMNIST 등 |
| [LRF-Dyn](https://arxiv.org/abs/2603.19290) [full text 일부], 표에 추가 | 부분. `X_n[t] = A⊙X_{n−1}[t] + ΓToken_n[t]` + SN | 부분. 각 t에서 n 축 recurrence, 단방향 | **덮음.** simulation T | 없음 | 없음 | 없음 | 미확인 | 없음. ImageNet 등 |

표 밖에서 [SEW ResNet](https://arxiv.org/abs/2102.04159)은 IAND residual `(1−A)·S`를 정의했다. [DST](https://arxiv.org/abs/2301.11929)는 AND, IAND, OR, XOR을 비교했다(DVS Gesture 55.56, 97.57, 96.18, 96.53) [full text]. [SynA-ResNet](https://arxiv.org/abs/2311.06570)은 OR residual을 제안한다 [abstract only].

## 노트 사이 불일치는 대부분 인용 근거로 풀린다

| # | 불일치 | 근거 | 판정 |
|---:|---|---|---|
| 1 | PRF bidirectional: PRF 노트는 "θ, Δ만 분리, 앞단 Linear 공유". bidirectional 노트는 "independent", "사실상 동일, 차이 없음" | "independent"는 원문에서 "second PRF Neuron"에만 붙는다. Eq. 102 뒤 `W ∈ R^{2D×D}`는 하나다. code는 `self.reverse_neuron1(u.flip(dims=[0]))`, `pro_linear1 = nn.Linear(2*h, h)`이고 두 neuron이 같은 u를 읽는다 | **해결.** 독립인 것은 neuron parameter뿐이다. stream별 앞단 Linear는 없고 Linear는 fusion 뒤 하나다. "차이 없음"은 과장이다. SpikMixer는 Linear, BN, neuron이 모두 stream별이다 |
| 2 | Bidirectional spiking mixer가 ETT·ECG에 있는가 | SpikeMamba는 ECG이지만 flip이 없다. SmolMamba는 bidirectional이지만 vision이다. Yin은 bidirectional이 TIMIT, ECG는 "using only past information"이다. [ISCAS 2026](https://doi.org/10.1109/ISCAS66217.2026.11562549)과 [SpikeApneaNet](https://doi.org/10.1038/s41598-026-65099-y)은 [abstract only]이다. SeqSNN `is_bidir`는 ANN RNN이고, SpikySpace의 bidirectional module은 network 파일에서 import되지 않는다 [code] | **ETT는 이번 검색에서 못 찾음. ECG는 미해결.** SpikeMamba와 SmolMamba는 각각 한 조건만 갖는다. 다만 ECG 후보 두 편이 unread이고, "spiking + ETTm2" 초록 검색 0건도 약한 근거다 |
| 3 | T_h 축: PRF 노트는 "SpikMixer만", attention 노트는 "완전히 있음" | SpikF `SpikeRNN.py`(repeat + N 축 Linear + LIF), SmolMamba(T = 4 + token 축 bidirectional scan), LRF-Dyn(simulation T + n 축 recurrence) | **해결. 고유점 아님.** 남는 차이는 N을 걷는 요소가 binary LIF라는 점뿐이다 |
| 4 | IAND·OR residual: PRF 노트와 attention 노트는 고유로 봄 | SEW 2021, DST 2023, SynA-ResNet, XOResNet | **해결. 고유점 아님.** 그 노트들의 읽은 범위 한계였다 |
| 5 | ETT 평가를 PRF 노트가 고유로 봄 | NSMT, SpikF, SpikeLite, SpikingTSF | **해결.** PRF 계열과 대비할 때만 참이다 |
| 6 | 시계열 노트: "token 축 Linear 뒤 같은 축 neuron: 비어 있음" | `self.w_in = nn.Linear(dim, dim)`, `u = self.bn(self.w_in(x).reshape(-1, D))` [local] | **해결. 설명이 틀렸다.** SpikMixer에 N 축 Linear는 없다. 그 자리의 "비어 있음"은 근거가 없다 |
| 7 | 시계열 노트: "forward·reversed, 분리 weights: 찾지 못함" | 그 노트의 범위는 forecaster다. PRF App. H와 Yin 2021이 있다 | **해결.** SNN 전체에서는 덮인다. forecaster 안에서만 못 찾았다 |
| 8 | 시계열 노트: "Spiking Mixer forecaster 없음", SMixer 본문 미독 | attention 노트가 SMixer 본문을 읽었다. 표 4는 시계열 4종, H 6–96, "A fixed setting of 4 time steps" | **해결. 있다.** ETT가 아닐 뿐이다. "최초의 spiking mixer forecaster"는 쓸 수 없다 |
| 9 | attention 노트: "reset 있는 LIF가 N 축 mixer인 사례 못 찾음", Mamba 계열 미독 | SpikeMamba `v_t = α·v_{t−1}·(1−s_{t−1}) + I_t`, `v_t ← v_t − s_t·V_th` | **해결. 단방향으로 덮인다.** 남는 것은 LIF + reversed stream의 결합이다 |
| 10 | "ECG는 SpikeMamba가 유일"(PRF 노트) 대 ECG 노트의 다수 SNN | 범위가 다르다. 또 ECG 노트 표에는 SpikeMamba가 빠져 있다 | **해결.** 가까운 relative 중 유일하다. 아래 MIT-BIH 표에 넣었다 |
| 11 | 이름: ECG 노트는 "충돌 없음" | [SpikeMixer, Neurocomputing 2026](https://doi.org/10.1016/j.neucom.2026.134120), [README](https://github.com/Germinted/SpikeMixer)(H/W 축), P-SpikeSSM의 "SpikeMixer Block". arXiv 제목 "SpikMixer" 일치 0건 [grep only] | **해결.** 철자 완전 일치는 못 찾았고 한 글자 차이는 있다. 본문은 unread다 |
| 12 | bidirectional 노트: "spike XOR control: not run" | `sd`(`neuron(BN(W[F; R; s_d]))`) − spike XOR(`W[F; R; \|F − R\|]`), CinC fold 0: LIF +2.55 %p [+1.14, +4.06], f-LIF +2.58, population LIF −0.19 ([PROJECT_LOG 10-28 05:45](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/docs/PROJECT_LOG.md)) [local] | **부분 해결.** T_h 없는 Neo-mixer의 fusion 입력으로는 측정됐다. SpkMixer의 MUX gate와 ETT, MIT-BIH에서는 not run이다 |
| 13 | ECG 노트: MIT-BIH "validation split 없음" | tuning은 train file의 1/5로 고르고, 최종 수치는 전체 train file 30 epoch, last-epoch test다 ([PROJECT_LOG 11-01 18:48](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/docs/PROJECT_LOG.md)) [local] | **해결.** 최종 run에만 맞는 설명이다 |
| 14 | ETT 차이: 시계열 노트 +0.32 %, log R +0.45 % | 앞은 4칸 평균 MSE의 비이고 뒤는 칸별 상대 차이의 평균이다 [calculation] | **해결.** 집계가 다르다. 판정 규칙은 R이다 |
| 15 | SeqSNN Spike-RNN의 축 | 시계열 노트는 GRU와 묶어 L을 걷는다고 적는다. PRF 노트의 [spikernn.py](https://github.com/microsoft/SeqSNN/blob/main/SeqSNN/network/snn/spikernn.py)에서는 `SpikeRNN`이 T = 4, `SpikeRNN2D`만 L을 돈다. [spikegru.py](https://github.com/microsoft/SeqSNN/blob/main/SeqSNN/network/snn/spikegru.py)는 L × T다 [code] | **해결.** class마다 다르다. SpikF, SeqSNN, SpikingTSF의 "SpikeRNN"은 서로 다른 구조다 |
| 16 | membrane 불일치 gate의 선행이 정말 없는가 | SpikeVFuse는 "time-reversal-assisted bidirectional spike-modulation"이고 구동 신호가 초록에 없다 [abstract only] | **미해결** |

## 남은 차별점은 gate 조합 하나이고 효과는 CinC에만 보인다

부품 대부분은 이미 있다. reversed stream은 PRF와 Yin에 있다. OR, AND residual은 SEW와 DST에 있다. 두 spike 비교는 XOResNet과 DST에 있다. training 때의 산술 형태 `F⊙h + (1−F)⊙s`는 [DiSAN](https://arxiv.org/abs/1709.04696)에 있고 `|u−v|` feature는 [InferSent](https://arxiv.org/abs/1705.02364)에 있다. 불일치를 threshold로 잘라 binary mask를 만드는 발상은 [UnFlow](https://arxiv.org/abs/1711.07837)에 있지만, mask가 걸리는 곳은 loss다 [full text]. `{0, 1}` 위에서 `|a−b|`는 `a XOR b`이므로, spike 수준으로 내려가면 입력은 XOResNet과 대수적으로 같다 [inference]. 그래서 **주장 범위는 SNN, membrane 입력, reversed 쌍, binary 경로 선택까지 함께 좁혀야 한다.**

| 후보 차별점 | 선행 판정 | 근거 강도 | 효과 측정 |
|---|---|---|---|
| **Forward·reversed membrane 차이로 만든 binary gate가 learned path와 AND를 고르는 조합** | 이번 검색에서 못 찾음 | 제목·초록 비교적 강함, 본문 중간, 유료 journal 약함, GitHub·특허는 검색 안 함 | `muxlin` − `sdlin`: CinC +1.56 %p(+3.64, +0.67, +0.37), ETT R +0.57 %, MIT-BIH +0.00 %p [local] |
| Membrane 입력 (spike XOR 대비) | XOR은 덮임, membrane 쪽은 못 찾음 | 위와 같음 | Neo-mixer CinC fold 0에서만 LIF +2.55 %p. population LIF는 −0.19. SpkMixer에서는 not run |
| Stream별 Linear·BN·neuron 전부 분리 | 설계 변형 | PRF는 θ, Δ만, Yin은 분리로 읽힘 | weights 공유 control not run |
| Binary LIF가 N을 걷고 옆에 T_h repeat | 좁은 변형 | SmolMamba, LRF-Dyn과 layout이 같고 scan 요소만 다름 | T_h sweep은 log 키워드 검색에서 못 찾음 [local]. T_h 없는 Neo-mixer sd 64.98 / 33.9 µJ, SpkMixer sdlin 66.96 / 129.8 µJ(+1.98 %p, 3.83배) [calculation]. 다른 차이도 섞여 있어 control이 아니다 |
| Spike-only residual | 고유점 아님 | SEW, DST, SynA-ResNet | 선택 결과도 ETT IAND, CinC OR, MIT-BIH ADD, MSL OR, SMAP·PSM ADD [local] |
| 같은 모델로 ETT와 ECG 평가 | 빈 자리일 수 있음, 성과는 아님 | ECG 후보 unread | Spikformer-style과 no clear difference |
| 같은 front end, seed, 규칙의 짝 비교와 사전등록 | 절차상 차별점 | SpikF-GO도 "lack a unified experimental comparison"이라고 지적 ([SpikF-GO](https://arxiv.org/abs/2606.13901)) [full text] | 측정됨. 결과는 null에 가까움 |

쓸 수 있는 문장의 상한은 bidirectional 노트의 초안과 같다: "To the best of our knowledge, no prior spiking network selects between fusion paths with a spiking gate driven by the membrane disagreement of its forward and reversed streams." SpikeVFuse를 읽기 전에는 이 문장도 잠정이다. 최초의 bidirectional SNN, 최초의 gated fusion SNN, 최초의 XOR 또는 차이 기반 SNN, 최초의 spiking mixer forecaster는 쓸 수 없다. 전체 inference가 spike-driven이라는 표현과 긴 기억 주장도 쓸 수 없다. code 수준에서 이 gate는 순수한 조합 논리가 아니다. L의 neuron은 모든 token에서 state를 쌓으므로 상태가 있는 선택기다 [local]. "Multiplexer"라는 말은 [SpikeAD](https://arxiv.org/abs/2611.00112)가 하드웨어 의미로 쓴다 [full text]. "mismatch"보다 "disagreement"가 안전하다. 두 stream 중 어느 쪽도 다른 쪽의 prediction이 아니기 때문이다 [inference].

test 시 개입 결과는 해석에 주의가 필요하다. gate를 닫으면 CinC −29.88 %p, MIT-BIH −50.89 %p이고, 항상 열면 −6.94, −33.45 %p다. ETT 네 칸 평균은 닫으면 +1.65 %, 열면 −0.50 %다. **ETTh2 H96은 gate를 항상 열면 MSE −2.12 %(3/3)다.** gate가 손해일 수 있다는 사후 관찰이고 retraining으로 확인되지 않았다 ([PROJECT_LOG 11-05](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/docs/PROJECT_LOG.md)) [local]. 게다가 SmolMamba는 gate 제거를 주장한다: "The binary nature of spike generation naturally provides an implicit gating mechanism" [full text, 일부].

## ETT 공개 수치는 구별되지 않고 CinC·MIT-BIH는 비교할 수 없다

### ETT, seq_len 96 (test MSE, scaled data)

| 모델 | 구분 | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4칸 평균 [calculation] | 출처 |
|---|---|---:|---:|---:|---:|---:|---|
| **SpikMixer (validation-selected)** | 우리 | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | [local](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/results/compare-nsmt-20261105/summary.md) |
| Spikformer-style (tuned) | 우리, 같은 규칙 | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 위와 같음 |
| NSMT rerun-b | 우리, 같은 규칙 | 0.2896 | 0.4153 | 0.1744 | 0.4014 | 0.3202 | 위와 같음 |
| NSMT 공개값 | SNN, seed 42 | 0.288 | 0.409 | 0.175 | 0.399 | 0.3178 | [NSMT Table 13, 14](/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/papers/NSMT.pdf) |
| SpikF | SNN | 0.290 ± 0.001 | 0.416 ± 0.008 | 0.175 ± 0.001 | 0.405 ± 0.007 | 0.3215 | [Table 9–11](https://proceedings.mlr.press/v267/wu25m.html) |
| SpikeLite | SNN, preprint | 0.294 | 0.426 | 0.175 | 0.395 | 0.3225 | [Table 5](https://arxiv.org/abs/2609.35097) |
| SpikeRNN (SPE 부착) | SNN | 0.295 | 0.428 | 0.176 | 0.409 | 0.3270 | [SpikF Table 12](https://proceedings.mlr.press/v267/wu25m.html) |
| iSpikformer (SPE 부착) | SNN | 0.308 | 0.438 | 0.178 | 0.406 | 0.3325 | 위와 같음 |
| SpikeRNN (제3자) | SNN | 0.3526 | 0.4512 | Pending | Pending | 없음 | [SpikingTSF](https://github.com/spikora/SpikingTSF/blob/main/RESULTS.md) |
| TimeMixer | ANN | 0.289 | 0.412 | 0.175 | 0.391 | 0.3168 | [Table 13](https://arxiv.org/abs/2405.14616) |
| RLinear | ANN | 0.288 | 0.420 | 0.182 | 0.407 | 0.3243 | [iTransformer Table 10](https://arxiv.org/abs/2310.06625) |
| iTransformer, MAE loss | ANN | 0.294 | 0.423 | 0.182 | 0.411 | 0.3275 | [SpikF Table 9, 10](https://proceedings.mlr.press/v267/wu25m.html) |
| PatchTST | ANN | 0.302 | 0.431 | 0.175 | 0.402 | 0.3275 | [iTransformer Table 10](https://arxiv.org/abs/2310.06625) |
| PatchTST (다른 재실행) | ANN | 0.308 | 0.436 | 0.183 | 0.412 | 0.3348 | [TimeMixer Table 13](https://arxiv.org/abs/2405.14616) |
| iTransformer | ANN | 0.297 | 0.427 | 0.180 | 0.412 | 0.3290 | [Table 10](https://arxiv.org/abs/2310.06625) |
| TimesNet | ANN | 0.340 | 0.462 | 0.187 | 0.408 | 0.3493 | 위와 같음 |
| DLinear | ANN | 0.333 | 0.831 | 0.193 | 0.554 | 0.4778 | 위와 같음 |

**같은 규칙으로 측정한 비교는 위 세 줄뿐이다.** SpikMixer − Spikformer-style은 R +0.45 %(3/12), no clear difference, energy 0.98배다. SpikMixer − NSMT rerun-b는 R +1.46 %(0/12), worse, energy 0.16배다. energy 이득은 SpikMixer의 몫이 아니다. Spikformer-style도 NSMT의 약 1/6이다 ([PROJECT_LOG 11-07](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/docs/PROJECT_LOG.md)) [local].

| 대상 | Split, metric | Loss | Stopping | Seeds | 한계 |
|---|---|---|---|---|---|
| SpikF | 같음, border가 로컬 loader와 같음 | L1, 같음 | 최대 5 / 10 epoch, patience 2 / 1, min delta 2e-4 | 3, 공개 script는 seed를 넘기지 않음 [code] | 가장 가깝지만 stopping이 훨씬 짧음. 제3자 ETTh1 H720 재현은 4.2 % 나쁨 |
| SpikeLite | SpikF 절차로 적음 | 미확인 | patience 미기재 | 3 | code 미공개, ETT 경로 미확인 |
| NSMT 공개값 | 같음 | L1 + auxiliary | patience 2, 마지막 상태 시험 | 1 | 같은 규칙 재실행 값을 써야 함. ETTh2 H96 0.2835 ↔ 0.2896 |
| SpikeRNN, iSpikformer (SPE) | 같음 | code 기본 MAE | 공개 script 없음 | 표 설명 없음 | Lv 원형이 아니라 SPE 부착 변형 |
| iTransformer 표의 ANN | 같은 크기 | L2 | epoch 10 고정 | 표 10 미기재 | L1·L2 차이는 칸별 약 1 % |
| TimeMixer | 같은 크기 | L2 | epoch 10 | 3 | test `drop_last` 값 미확인 |
| SpikingTSF | 같음 | MAE | patience 15 | 3 | std 0.0000이고 사용 전 독립 확인을 요구함, baseline으로 쓰면 안 됨 [inference] |

공개값끼리의 4칸 평균 MSE 비는 SpikF +0.93 %, SpikeLite +0.61 %, SpikeRNN −0.77 %, RLinear +0.07 %, TimeMixer +2.44 %다 [calculation]. 같은 PatchTST가 표에 따라 최대 4.6 % 다르다 [calculation]. **그래서 공개 수치만으로는 SpikF보다 낫다거나 나쁘다고 말할 수 없다.** 가장 불편한 값은 RLinear다. RevIN과 linear layer뿐인 모델과 사실상 같다. 같은 규칙으로 RLinear를 돌린 기록은 찾지 못했다. 또 mixer를 test 시 끄면 ETT MSE는 +1.61 %(10/12 나빠짐)이고 ETTh2 H96은 오히려 −0.56 %다 [local]. SpikF `w/o`와 같은 방향이다. ETT가 mixer를 가르기에 약한 task라는 해석과 맞는다 [inference]. L을 늘리는 효과는 이 모든 차이보다 크다. PatchTST의 ETTh2 H720은 L = 96에서 0.431, L = 336에서 0.379다 ([PatchTST](https://arxiv.org/abs/2211.14730)) [full text].

### CinC 2017 위치

| 출처 | Test split | Metric, 값 | 근거 | 우리와 비교 가능? |
|---|---|---|---|---|
| **SpikMixer (LIF, OR)** | 공개 8,528의 5-fold 중 fold 0, 1,706 record, 3 seeds, 30 epoch | NAO 68.52 ± 0.81, macro-F1 64.35, accuracy 73.97, 129.4 µJ/window | [local](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/results/flif-v1-rhythm-spk-mux-cinc/summary.md) | 기준 |
| Spikformer-style (tuned) | 같음 | NAO 66.36 ± 1.04, 131.5 µJ | [local](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/docs/PROJECT_LOG.md) | **예.** +2.16 %p(+2.86, +2.54, +1.09), 0.98배 |
| [Teijeiro](https://www.cinc.org/archives/2017/pdf/166-054.pdf), Challenge 1위 | hidden 3,658, 재라벨 포함 | NAO 0.831 | [full text] | 아니오. test set이 다름 |
| [공식 결과](https://physionet.org/files/challenge-2017/1.0.0/results_all_F1_scores_for_each_classification_type.csv) 68개 entry | hidden | NAO 중앙값 0.7742 | [calculation] | 아니오 |
| [Zihlmann](https://arxiv.org/abs/1710.06122) CNN, CRNN | stratified 5-fold | 증강 없음 75.8, 74.6, 증강 79.0, 79.2 | [full text, 일부] | 아니오. fold와 training 예산이 다름 |
| [Andreotti](https://www.cinc.org/archives/2017/pdf/360-239.pdf) ResNet | 5-fold | NAO 63.1, 75.2 | [calculation] | 아니오 |
| [Datta](https://www.cinc.org/archives/2017/pdf/173-154.pdf), [Zabihi](https://www.cinc.org/archives/2017/pdf/069-336.pdf), [Hong](https://www.cinc.org/archives/2017/pdf/178-245.pdf) | 각자 CV | NAO 82.64, 81.85, 86.55 | [full text, 일부] | 아니오 |
| [Feng 2022](https://doi.org/10.1016/j.bspc.2022.103749), SNN | random, 기록 전체 | Acc 84.4, F1 70.7, 1000 time steps | [secondary summary](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362428/) | 아니오. F1 정의를 모름 |
| [Rana & Kim 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175061/), SNN | 70:30 | Acc 85.8, F1 70.4 | [full text, 일부] | 아니오. 평균 방식 불명 |
| [Banerjee 특허](https://patents.google.com/patent/US20240176987A1/en), SNN | 6,822 / 1,705 | Acc 77–78.5, F1 없음 | [full text, 일부] | 아니오 |

**이 숫자로 "CinC SNN 최고"는 주장할 수 없다.** 공개 데이터 CV 값과의 거리는 −6에서 −18이다 [calculation]. 비교 가능한 값은 아니고 대략의 거리일 뿐이다. 격차의 대부분은 AF와 Other에 있고 Noisy는 비슷하다 [calculation]. 30 epoch, crop만 쓰는 증강, fold 1개, 앙상블 없음이라서 일부는 training 예산 차이일 수 있다 [inference]. +2.16 %p도 강하지 않다. Spikformer-style의 lr 2e-3이 grid 끝이다 [local]. 비슷한 초기 모델 ③(T_h 없는 Neo-mixer, 실수 membrane 차이)은 fold 0에서 +0.02였고 5 fold 합산 −0.21 %p [−1.11, +0.70]이었다 ([PROJECT_LOG 10-26 20:18](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/docs/PROJECT_LOG.md)) [local]. SpikMixer의 fold 1–4는 not run이다. window당 µJ를 보고한 CinC SNN은 찾지 못했다. 다만 accuracy가 낮은 상태에서 이 빈칸은 약하다 [inference].

### MIT-BIH 위치

| 출처 | Split | 값 | 근거 | 우리와 비교 가능? |
|---|---|---|---|---|
| **SpikMixer (ADD, lr 0.02, 2 layers)** | beat-level 70/30, 환자 ID 없음, last-epoch test | macro-F1 91.99 ± 0.10, Q 제외 90.25 [calculation], accuracy 98.56, N 99.24, S 85.94, V 96.31, F 79.49, Q 98.95, 9.44 µJ | [local](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/results/compare-nsmt-20261105/summary.md) | 기준 |
| Spikformer-style | 같음 | tuned 91.78, default 92.22 | [local] | **예.** +0.20, −0.24 %p, no clear difference, 0.98–0.99배 |
| NSMT rerun | 같음 | 91.58 ± 0.05, 23.86 µJ | [local] | **예.** +0.40 %p, no clear difference, 0.40배 |
| [SpikeMamba-M](https://papers.miccai.org/miccai-2026/paper/4698_paper.pdf) | segment-mixed, 6:2:2, 360 sample | Acc 98.87, F1 94.81, AUROC 99.79. energy 단위는 의심 [inference] | [full text] | 아니오 |
| [ECGformer](https://arxiv.org/abs/2401.05434) | 80/20, non-spiking | 98 %, macro-F1 0.86 | [full text, 일부] | 아니오. split 비율이 다름 |
| [Scrugli 2026](https://zenodo.org/records/18887975) | intra 70/10/20, inter Chazal | intra 98.4, F1 88.6. inter 92.44, macro-F1 0.425, RR branch 0.561. FPGA 36.86 µJ 측정 | [full text, 일부] | 아니오 |
| [Galvis-Chacón 2025](https://www.recercat.cat/handle/20.500.14342/5924) | intra, inter | 98.23, 86.38 | [secondary summary] | 아니오 |
| [Banjo 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306364/) | patient-specific, DS2 | 99.02, 89.31 ± 1.51 | [full text, 일부] | 아니오 |
| [BioSpike-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC13509974/) | patient-disjoint, 2 s window | macro-F1 95.8 ± 0.4, window 단위 | [full text, 일부] | 아니오. 기준값으로 쓰지 않는 편이 안전 [inference] |
| [Wang 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8181174/) | inter, non-spiking | F1_S 84.4, F1_V 93.4 | [full text, 일부] | 아니오 |
| [ISCAS 2026](https://doi.org/10.1109/ISCAS66217.2026.11562549) | 초록에 없음 | 5-class 96.44 %, 43 µW | [abstract only] | 아니오 |

DS1/DS2는 MIT-BIH record를 22개씩 나눈 inter-patient split이다. **intra-patient accuracy는 spiking과 non-spiking 모두 98–99에 몰려 변별력이 거의 없다.** 같은 모델도 inter-patient로 가면 크게 떨어진다(Scrugli 98.4 → 92.44, Galvis-Chacón 98.23 → 86.38). 우리 91.99는 inter-patient로 옮겨지지 않는다고 봐야 한다 [inference]. 9.44 µJ도 ASIC이나 FPGA 측정값과 계산 방식이 달라 어느 쪽과도 비교 가능하지 않다. AD 3종은 R +1.05 %(6/9), no clear difference이고 세 모델 모두 training 없는 점수 수준이다 [local].

## 우리 쪽에서 고칠 것과 아직 모르는 것

| 항목 | 사실 | 상태 |
|---|---|---|
| CinC 3,750 sample | "3,750 samples at the challenge rate"는 틀렸다. Challenge 원본은 300 Hz다. 125 Hz로 resample(polyphase 5/12)한 30 s가 3,750이다. token은 117이다 | **확인됨** ([rhythm_data.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/rhythm_data.py), [config.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/config.py)) |
| MIT-BIH Q class 수 | 우리 Q는 8,039 beat다(train 5,627, test 2,412). AAMI 절차대로 paced 기록을 빼면 15 beat다 ([Cao et al.](https://arxiv.org/abs/2206.14200)) [full text, 일부] | 개수 **확인됨** (CSV). paced beat가 남았다는 해석과 Kaggle 판 출처는 **미확인** [inference] |
| Q의 영향 | Q F1 98.95가 macro의 1/5이다. Q 제외 macro-F1은 90.25다. 반올림된 class F1로 계산한 값이고 다시 평가하지 않았다. 항상 N이면 accuracy 82.8 %다 | [calculation], 재평가는 **not run** |
| Beat-level split | 188열이고 환자 ID가 없다. class마다 7:3이다. 같은 환자가 train과 test에 들어간다. "highly discouraged"라고 적힌 방식이다 ([Cao et al.](https://arxiv.org/abs/2206.14200)) [full text, 일부] | 파일 구조 **확인됨**. 환자 겹침의 정도는 **미확인** |
| **`denoise_patient_mitbih_*` 파일 (노트에 없음)** | [data 폴더](/home/yschoi/CLS_spiking_transformer/data/)에 2025-05-20자 train·test 180–500 sample 파일이 있다. 180 기준 50,366 / 49,473행, label {0: 412, 1: 45,855, 2: 5, 3: 870, 4: 3,224}, {0: 384, 1: 44,036, 2: 6, 3: 1,989, 4: 3,058}이다. label 2가 5–6행이라 Q 8 / 7과 비슷하므로 DS1/DS2 계열로 보인다 [inference]. .py, .sh 참조는 찾지 못했다 | 개수 **확인됨**. 출처와 split은 **미확인**. "raw에서 다시 잘라야 한다"는 ECG 노트 판단은 이 파일을 확인한 뒤로 미뤄야 한다 |
| CinC score 정의 | `f[:3].mean()`, 즉 N, AF, Other F1 평균이다. 공식 최종 점수와 공식이 같다. 공식 CSV로 Teijeiro 0.8314가 재현된다 [calculation]. PhysioNet 페이지의 4 class 평균 문구와는 다르다 ([Clifford](https://www.cinc.org/archives/2017/pdf/065-469.pdf), [PhysioNet](https://physionet.org/content/challenge-2017/1.0.0/)) | **확인됨** ([rhythm_eval.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/rhythm_eval.py)) |
| CinC test set | hidden test가 아니라 공개 8,528의 `StratifiedKFold(5, shuffle, random_state=0)` fold 0이다. hidden은 재라벨로 class 분포도 다르다. 어떤 Challenge 점수와도 비교 불가다 | **확인됨** (code). 환자 ID 유무는 **미확인** |
| 61 % MAC | AC는 0/1 입력의 연산(0.9 pJ), MAC은 실수 입력의 연산(4.6 pJ)이다. seed 42: 47.507 M AC + 14.7425 M MAC = 110.57 µJ이고 MAC이 67.82 µJ(61.3 %)다 | **확인됨** ([raw.txt](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/results/flif-v1-rhythm-spk-mux-cinc/raw.txt)) |
| MAC의 구성 (노트에 없음) | 14,742,528 = embedding Linear 11,501,568(64 × 384 × 117 × 4) + BN 9개 3,234,816 + head 6,144다. 기록값과 정확히 일치한다. **embedding Linear만 52.91 µJ다.** T_h 복사본은 같은 값이므로 그 3/4는 같은 수의 재계산이다 [inference]. Spikformer-style도 같은 구조다 | [calculation]. layer별 기록과 맞춰 보지는 않았으므로 항목별 값은 **미확인**, 합계만 일치 확인 |
| STMixer식 비판 | 비판은 "two spike matrix multiplications ... could result in significant differences in the output"다 ([STMixer](https://proceedings.neurips.cc/paper_files/paper/2024/file/e8c20cafe841cba3e31a17488dc9c3f1-Paper-Conference.pdf)) [full text]. `min(f, r)`, `sd * a`, IAND `x * (1 − m)`은 spike끼리의 동시 발생 연산이다. `(hf − hr).abs()`는 spike 곱은 아니지만 real membrane을 읽어 layer 사이로 보내고, 두 stream을 같은 token에 맞춰야 하므로 같은 비판의 대상이 된다 | code **확인됨** ([spk_mixer.py L114, L133, L143](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/spk_mixer.py)). 적용 판단은 [inference]. 비동기 환경의 영향은 **not run** |
| Energy에 안 잡히는 연산 (노트에서 미확인) | hook은 Linear, BN, neuron에만 있다. `abs`, `min`, `max`, spike 곱, flip 버퍼는 세지 않는다. `s_d` 앞 BN만 MAC 1.65 µJ로 잡힌다 [calculation]. Spikformer-style의 `q·kᵀ·v`도 안 세므로 0.98배의 양쪽이 모두 불완전하다 | **확인됨** ([energy.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/energy.py), [PROJECT_LOG 11-07](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/docs/PROJECT_LOG.md)). 크기는 **not run** |
| "Spike-only" 표현 | MIT-BIH, SMAP, PSM의 선택은 ADD다. MIT-BIH에서 ADD와 OR의 validation 차이는 0.016 %p로 사실상 동률이다 [calculation] | **확인됨** ([selected_res.json](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/results/flif-v1-mitbih-spkmux-tune/selected_res.json)) |
| Task마다 다른 모델 | ETT: IAND, 방향당 1 layer. MIT-BIH: ADD, 2 layers. CinC: OR, 2 layers. 논문에 한 모델처럼 쓰면 지적받는다 [inference] | **확인됨** ([ETT selected_layers.json](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/results/flif-v1-ett-spkmux-tune/selected_layers.json)) |
| `H − θ` | LIF threshold는 1로 고정이라 두 stream에서 상쇄된다. `\|H_f − H_r\|` 표기는 LIF에서 맞다 | LIF **확인됨** ([layers.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/layers.py)). population은 **미확인** |
| NSMT 제출본 표기 | 본문이 "4.6uJ and 0.9uJ"로 적는다. Table 7 "Improvement"는 42.74, 52.79, 51.10 %인데 energy 칸으로 계산하면 29.9, 34.5, 33.8 %다 [calculation] | 논문 **확인됨** (시계열 노트 [full text]) |

## reviewer 반론 대부분은 아직 답이 측정되지 않았다

| 반론 | 근거가 될 문헌 | 답이 될 측정 | 상태 |
|---|---|---|---|
| "PRF의 LIF 버전 + 새 fusion 아닌가" | PRF App. H | PRF SD-TCM, 그리고 PRF neuron을 넣은 SpikMixer를 같은 규칙으로 | **not run.** 공개 code는 LRA, MNIST config만 있음 |
| "같은 accuracy는 새 주장이 아니다. 더 단순한 mixer와 비교했나" | Wang, STMixer, SMixer | `SN(W_M X)`, identity, SpikF, SpikeRNN | **not run** |
| "ETT에서 mixer가 일을 하는가" | SpikF `w/o`, RLinear | mixer 없는 retraining, 같은 규칙의 RLinear | test 시 제거만 측정(+1.61 %). retraining **not run** |
| "gate 효과가 일반적이지 않다" | SmolMamba "implicit gating", SPM | muxlin − sdlin 3 task | **측정됨, 불리함.** CinC만 better. 다른 fold **not run** |
| "XOR residual과 `\|u−v\|`의 합성일 뿐이다" | XOResNet, DST, InferSent | SpkMixer에서 spike XOR gate와 membrane gate 비교 | Neo-mixer CinC fold 0만 측정. SpkMixer와 ETT, MIT-BIH **not run** |
| "spike-driven이 아니다" | STMixer, [SDT](https://arxiv.org/abs/2307.01694) "Only binary spikes can support the spike-driven function" | 미계상 연산 포함 energy, 비동기 simulator | code 확인됨. 영향 **not run.** SDT의 "mask ... does not consume energy"는 동기 하드웨어 전제의 방어다 |
| "Non-causal이라 SNN 장점을 버렸다" | SiLIF, TEFormer | latency, 버퍼 측정, sliding block | **not run.** task별 부담 구분은 [inference]: CinC, MIT-BIH는 window당 예측 1개, ETT는 미래 누출 없음 |
| "T = N은 새롭지 않고 짧은 window의 spiking convolution과 뭐가 다른가" | SpikeGPT, SmolMamba, SNN-MLP | spiking convolution baseline, shuffled·memoryless retraining | context 4–16 token은 측정됨(test 시). baseline **not run** |
| "T_h 4번 복사가 무엇을 더하나" | TEFormer "severe temporal redundancy", SpikF 3.7 % | SpkMixer T_h sweep | log 키워드 검색에서 못 찾음, **not run**. embedding MAC은 T_h에 비례 |
| "energy 이점이 없다" | SMixer 0.65–0.85배, SmolMamba 0.53배 | `q·kᵀ·v`를 넣은 재계산, 같은 T | 0.98–1.03배 측정됨. 재계산 **not run**. 문헌 값은 baseline이 인용값이라 같은 측정이 아님 [inference] |
| "MIT-BIH split이 약하고 Q가 점수를 올린다" | Cao, Wang 2021, Scrugli | DS1/DS2, Q 제외, S·V F1 | **not run.** `denoise_patient_*` 확인이 먼저 |
| "CinC fold 하나이고 non-spiking보다 낮다" | Clifford, Zihlmann | fold 1–4, 같은 pipeline의 non-spiking baseline, 더 긴 training | **not run.** Neo-mixer ③은 5 fold에서 차이 없음 |
| "neuron 선택은 왜 LIF인가" | PRF, f-LIF | ETT·MIT-BIH의 population LIF, population f-LIF 행 | SpkMixer는 LIF만 training됨, **not run**. CinC Neo-mixer의 `sd` fusion에서는 population이 더 낮음(61.25 대 LIF 64.98) |
| "Replay에서 왔다는 말은 무슨 뜻인가" | [Foster & Wilson](https://doi.org/10.1038/nature04587), [Keller](https://pmc.ncbi.nlm.nih.gov/articles/PMC6400266/) | 측정 대상 아님. "inspired by" 수준으로 제한 | 비유의 한계 네 가지는 [inference]: 입력 없는 시점, 다른 시점, 같은 synapse, 비교 근거 없음 |
| "이름이 SpikeMixer와 거의 같다" | Neurocomputing 2026 | 이름 재검토 | 사용자 결정 |

## 다음 경로는 사용자 결정이고 아무것도 시작하지 않았다

**사용자는 11-04에 이렇게 정했다: "최종 논문에는 `CinC 과제를 main result`를 넣지 않을 예정이야. 추가적인 `memory 기능` 혹은 `T=N` 근거를 뒷받침할 작은 실험으로만 넣을 예정이니, 다른 테스크 먼저 진행해줘."** ([PROJECT_LOG 11-05](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/docs/PROJECT_LOG.md)) CinC 중심 논문은 이 결정의 명시적 번복이 필요하다. 이 보고서는 그 결정을 내리지 않는다. 11-07 log에도 이 번복 여부가 사용자 결정 대기로 남아 있다. 아래 표는 선택지일 뿐 합의가 아니다. 구조 비교에는 population 행을 함께 둔다는 사용자 규칙(10-21)이 있어 run 수가 늘어난다.

| 선택지 | 내용 | 전제 | 열리는 주장 | 위험 |
|---|---|---|---|---|
| A | SpikMixer를 논문 중심에서 내린다 | 없음 | log 기준 main task 후보는 query pop LIF다 | 구조 신규성을 포기 |
| B | CinC 중심의 좁은 논문: fold 1–4, Spikformer-style lr grid 확장, 두 번째 긴 task, non-spiking baseline | **사용자의 명시적 번복** | 긴 biosignal에서 같은 energy에 더 높은 점수 | fold 평균에서 사라질 수 있음. Sleep-EDF는 약 93 token, SNN 74.2 [calculation, [NeuroSleep](https://arxiv.org/abs/2602.15888)] |
| C | Spiking time axis의 통제 비교: repeat, forward, reversed, shuffled, bidirectional + RLinear, SPE + Linear, SpikF, PRF | ETT scope 유지 | neuron이 순서를 실제로 쓰는가. 이 설계는 찾지 못했다 | null일 가능성이 높음 [inference] |
| D | ETT energy 재계산: head, BN, 미계상 연산, `q·kᵀ·v`, SpikF T 16 대 4 | training 없음 | 일관 기준의 energy 비교 | CinC에서 embedding + BN이 약 53 %로 공통이라 차이가 작을 수 있음 |
| E | Gate control 묶음: SpkMixer spike XOR gate, signed 차이, weights 공유, sliding block | SpkMixer 유지 | membrane 입력 주장의 근거 | 효과가 없으면 차별점이 사라짐 |
| F | MIT-BIH inter-patient 평가 | `denoise_patient_*` 출처 확인 | leakage 반론 대응 | 문헌 기준 큰 폭 하락, 단일 beat 입력에 RR 없음 |
| G | training 없는 정리: 아래 재확인, 문구 정정, 이름 재검토 | 없음 | 위 경로 모두의 전제 | 없음 |

## 제출 전에 다시 봐야 할 출처

| 순서 | 출처 | 왜 급한가 | 막힌 이유 또는 상태 |
|---:|---|---|---|
| 1 | [SpikeVFuse, IEEE TMM 2026](https://doi.org/10.1109/tmm.2026.3729390) | "time-reversal-assisted bidirectional spike-modulation". modulation이 stream 차이로 구동되면 마지막 차별점이 사라진다 | [abstract only], 본문 못 찾음 |
| 2 | [PRF 게재본, Pattern Recognition 2027](https://doi.org/10.1016/j.patcog.2026.114968) | v2에 없는 fusion이나 gate 유무 미확인. README 표 번호와 reset 변형이 v2와 다르다. v2 수치와 섞으면 안 된다 | ScienceDirect 403 |
| 3 | [SpikeMixer, Neurocomputing 2026](https://doi.org/10.1016/j.neucom.2026.134120) | 이름 충돌. README는 영상의 H/W 축 혼합이지만 본문 미확인 | SSRN 사본 내려받기 실패 |
| 4 | [ISCAS 2026 Bidirectional SNN](https://doi.org/10.1109/ISCAS66217.2026.11562549) | MIT-BIH가 겹치고 "bidirectional"의 뜻이 초록에 없다 | [abstract only] |
| 5 | [SpikeApneaNet](https://doi.org/10.1038/s41598-026-65099-y) | ECG + spike-BiGRU | [abstract only], fetch 실패 |
| 6 | [Xiao 2022](https://doi.org/10.1109/TASLP.2022.3221011), [Bi-SRNN EEG](https://doi.org/10.1002/cpe.70343), [Multi-LRA](https://doi.org/10.1016/j.ins.2024.120136) | bidirectional fusion 방식, logical residual 조합 | abstract only 또는 제목만 |
| 7 | [Feng 2022](https://doi.org/10.1016/j.bspc.2022.103749), [SpikeLite](https://arxiv.org/abs/2609.35097) code, SWformer supplementary | F1 정의, ETT 경로, energy 계산 | 전문 또는 code 미공개 |
| 8 | 검색 자체 | OpenAlex 2차 27개 질의 실패(429). Semantic Scholar 429. DBLP, OpenReview 차단. GitHub 코드와 특허는 검색 안 함. stereo, Siamese 계열은 제목만 봄 | 제출 직전 다시 돌릴 필요. 가까운 논문 다수가 2026년 것이다 |

## Conclusion

이번 조사로 질문이 바뀌었다. "SpikMixer가 새로운가"가 아니라 "유일하게 새로운 부품이 일을 하는가"다. 지금 답은 불편하다. 새로운 부품의 효과는 main task에서 보이지 않고, 효과가 보이는 곳은 main task가 아니다. 반대로 우리가 강한 곳은 구조가 아니라 절차다. 같은 front end와 seed의 짝 비교, 사전등록, energy 재계산은 이 분야에서 드물다. 다만 그 결과는 null에 가까우므로, 이 강점을 살리면 논문의 성격 자체가 달라진다.

또 하나 정리된 것은 "T = sequence length"의 뜻이다. PRF의 T = L은 784–4,096 raw token을 긴 범위 task로 평가한다. SpikMixer의 T = N은 11–117 patch token이고, τ = 2 LIF가 실제로 쓰는 길이는 방향당 4–16 token이다. 표기는 같아도 주장은 다르다. 이 차이를 숨기면 reviewer가 찾아내고, 먼저 밝히면 "긴 기억"이 아니라 "짧은 bidirectional 국소 feature"라는 다른 설명이 된다. 그 설명을 측정된 주장으로 만들 spiking convolution 비교는 아직 없다. 가장 싼 다음 확인은 training이 필요 없는 SpikeVFuse 한 편이고, 그 결과에 따라 위 선택지의 값이 달라진다.
