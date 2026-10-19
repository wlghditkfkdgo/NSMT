# SHD / SSC — 2023–2026 선행 연구 프로토콜 조사

생성: `docs/surveys/report.py` (수집 `collect.py`, 선별 `screen.py`, 추출 `raw/extract/*.jsonl`). 수치는 추출 표의 값을 그대로 집계한 것이며, "not stated"는 본문에 명시되지 않은 항목이다.

## 1. 모집단과 선별 (PRISMA 식 건수)

- 수집(2023–2026, 경로별 원건수): S2CIT 302, OACIT 241, OASRCH 555, ARXIV 132, XREF 2000, PUBMED 14 → 중복 제거 후 2416편
- 장소(venue) 분류 · 전체: A군 592, B군 68, 제외군(MDPI·Frontiers·Hindawi·CinC 프로시딩·워크숍) 100, 프리프린트 전용 128, 출처 불명/기타 1528
- 제목·초록 주제 일치 · A군 165, B군 38
- 공개 전문 확보 · A군 86/165, B군 16/38 (나머지는 초록만)
- 전문에서 대상 데이터셋을 실제로 사용(언급 ≥ 2) · A군 51, B군 8
- 추출 후 포함(자기 모델의 결과를 보고) · **A군 46, B군 3**

## 2. 항목별 빈도

**시간 구간 폭** (A군 n=46 / A+B군 n=49)

| 값 | A군 | A+B군 |
|---|---|---|
| 1 ms | 5 (11%) | 7 (14%) |
| 2 ms | 3 (7%) | 3 (6%) |
| 4 ms | 5 (11%) | 5 (10%) |
| 5 ms | 1 (2%) | 1 (2%) |
| 10 ms | 6 (13%) | 6 (12%) |
| other | 1 (2%) | 1 (2%) |
| event-driven (no bin) | 1 (2%) | 1 (2%) |
| none stated | 1 (2%) | 1 (2%) |
| not stated | 23 (50%) | 24 (49%) |

**스텝 수** (A군 n=46 / A+B군 n=49)

| 값 | A군 | A+B군 |
|---|---|---|
| ≤50 | 1 (2%) | 2 (4%) |
| 100 | 10 (22%) | 10 (20%) |
| 140 | 1 (2%) | 1 (2%) |
| 150 | 0 (0%) | 1 (2%) |
| 200 | 0 (0%) | 1 (2%) |
| 250 | 8 (17%) | 8 (16%) |
| >250 | 6 (13%) | 6 (12%) |
| other | 1 (2%) | 1 (2%) |
| not stated | 19 (41%) | 19 (39%) |

**입력 채널 수** (A군 n=46 / A+B군 n=49)

| 값 | A군 | A+B군 |
|---|---|---|
| 700 | 21 (46%) | 23 (47%) |
| 140 | 6 (13%) | 6 (12%) |
| 70 | 1 (2%) | 1 (2%) |
| other | 2 (4%) | 2 (4%) |
| not stated | 16 (35%) | 17 (35%) |

**모델 선택(검증) 방식** (A군 n=46 / A+B군 n=49)

| 값 | A군 | A+B군 |
|---|---|---|
| test set used for selection | 6 (13%) | 6 (12%) |
| cross-validation on train | 2 (4%) | 2 (4%) |
| held-out part of train / official val | 5 (11%) | 6 (12%) |
| no validation set (procedure not stated) | 3 (7%) | 3 (6%) |
| not stated | 30 (65%) | 32 (65%) |

**증강** (A군 n=46 / A+B군 n=49)

| 값 | A군 | A+B군 |
|---|---|---|
| none stated | 3 (7%) | 3 (6%) |
| used | 12 (26%) | 12 (24%) |
| not stated | 31 (67%) | 34 (69%) |

**모델 계열** (A군 n=46 / A+B군 n=49)

| 값 | A군 | A+B군 |
|---|---|---|
| spiking transformer | 1 (2%) | 1 (2%) |
| state-space | 3 (7%) | 3 (6%) |
| delays | 7 (15%) | 8 (16%) |
| resonate-and-fire | 2 (4%) | 2 (4%) |
| recurrent SNN | 19 (41%) | 20 (41%) |
| feedforward SNN | 14 (30%) | 15 (31%) |

**뉴런 모형** (A군 n=46 / A+B군 n=49)

| 값 | A군 | A+B군 |
|---|---|---|
| multi-compartment / dendritic / branches | 12 (26%) | 12 (24%) |
| adaptive (ALIF) | 8 (17%) | 8 (16%) |
| heterogeneous time constants | 5 (11%) | 5 (10%) |
| plain LIF | 18 (39%) | 20 (41%) |
| other | 3 (7%) | 4 (8%) |

**seed/run 수** (A군 n=46 / A+B군 n=49)

| 값 | A군 | A+B군 |
|---|---|---|
| 1 / single | 1 (2%) | 1 (2%) |
| 3 | 6 (13%) | 6 (12%) |
| 5 | 10 (22%) | 10 (20%) |
| 10 | 4 (9%) | 4 (8%) |
| other (20) | 1 (2%) | 1 (2%) |
| other (6) | 1 (2%) | 1 (2%) |
| other (8) | 2 (4%) | 3 (6%) |
| not stated | 21 (46%) | 23 (47%) |

**보고 형식** (A군 n=46 / A+B군 n=49)

| 값 | A군 | A+B군 |
|---|---|---|
| mean only | 3 (7%) | 4 (8%) |
| mean±std (table or figure) | 24 (52%) | 24 (49%) |
| single value | 17 (37%) | 19 (39%) |
| not stated | 2 (4%) | 2 (4%) |

## 3. 포함 논문

| # | 연도 | 장소 | 군 | 제목 | DOI/arXiv | 추출 신뢰도 |
|---|---|---|---|---|---|---|
| 1 | 2026 | Neuromorphic Computing and Engineering | A | A scalable hybrid training approach for recurrent spiking neural networks | 10.1088/2634-4386/ae46d4 | high |
| 2 | 2026 | Nature Communications | A | Compute-in-memory implementation of state space models for event sequence processing | 10.1038/s41467-025-68227-w | medium |
| 3 | 2026 | Nature Communications | A | Model-agnostic linear-memory online learning in spiking neural networks | 10.1038/s41467-026-68453-w | medium |
| 4 | 2026 | IEEE Transactions on Neural Networks and Lear | A | PMSN: A Parallel Multi-Compartment Spiking Neuron for Multiscale Temporal Processing | 10.1109/tnnls.2026.3708510 | medium |
| 5 | 2026 | IEEE Transactions on Neural Networks and Lear | A | Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects | 10.1109/tnnls.2026.3736737 | medium |
| 6 | 2025 | Nature Communications | A | A multisynaptic spiking neuron for simultaneously encoding spatiotemporal dynamics | 10.1038/s41467-025-62251-6 | medium |
| 7 | 2025 | Nature Machine Intelligence | A | Algorithm–hardware co-design of neuromorphic networks with dual memory pathways | 10.1038/s42256-026-01255-3 | medium |
| 8 | 2025 | Neuromorphic Computing and Engineering | A | Beyond rate coding: surrogate gradients enable spike timing learning in spiking neural net | 10.1088/2634-4386/ae46d5 | medium |
| 9 | 2025 | Neural Information Processing Systems | A | Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling | 10.48550/arXiv.2509.17186 | medium |
| 10 | 2025 | Nature Communications | A | Efficient and robust temporal processing with neural oscillations modulated spiking neural | 10.1038/s41467-025-63771-x | medium |
| 11 | 2025 | Nature Communications | A | Efficient event-based delay learning in spiking neural networks | 10.1038/s41467-025-65394-8 | high |
| 12 | 2025 | Neural Information Processing Systems | A | HetSyn: Versatile Timescale Integration in Spiking Neural Networks via Heterogeneous Synap | 10.48550/arXiv.2508.11644 | high |
| 13 | 2025 | Neuromorphic Computing and Engineering | A | Loss shaping enhances exact gradient learning with Eventprop in spiking neural networks | 10.1088/2634-4386/ada852 | high |
| 14 | 2025 | Design, Automation and Test in Europe | A | Minimum Time Maximum Fault Coverage Testing of Spiking Neural Networks | 10.23919/DATE64628.2025.10993048 | medium |
| 15 | 2025 | Neural Information Processing Systems | A | Multiplication-Free Parallelizable Spiking Neurons with Efficient Spatio-Temporal Dynamics | 10.52202/085713-2213 | high |
| 16 | 2025 | Neural Networks | A | NSPDI-SNN: An efficient lightweight SNN based on nonlinear synaptic pruning and dendritic  | 10.48550/arXiv.2508.21566 | low |
| 17 | 2025 | Design Automation Conference | A | Replay4NCL: An Efficient Memory Replay-based Methodology for Neuromorphic Continual Learni | 10.1109/DAC63849.2025.11132839 | medium |
| 18 | 2025 | International Conference on Learning Represen | A | Rethinking Spiking Neural Networks from an Ensemble Learning Perspective | 10.48550/arXiv.2502.14218 | medium |
| 19 | 2025 | Lecture Notes in Computer Science, volume 150 | A | Scaling Up Resonate-and-Fire Networks for Fast Deep Learning | 10.48550/arXiv.2504.00719 | high |
| 20 | 2025 | AAAI Conference on Artificial Intelligence | A | SpikCommander: A High-performance Spiking Transformer with Multi-view Learning for Efficie | 10.48550/arXiv.2511.07883 | high |
| 21 | 2025 | Proceedings of the National Academy of Scienc | A | Spiking world model with multicompartment neurons for model-based reinforcement learning | 10.1073/pnas.2513319122 | low |
| 22 | 2025 | International Conference on Learning Represen | A | Temporal Flexibility in Spiking Neural Networks: Towards Generalization Across Time Steps  | 10.48550/arXiv.2503.17394 | medium |
| 23 | 2025 | Neural Networks | A | Towards Efficient and Accurate Spiking Neural Networks via Adaptive Bit Allocation | 10.1016/j.neunet.2025.108350 | medium |
| 24 | 2025 | Neuromorphic Computing and Engineering | A | Traces propagation: memory-efficient and scalable forward-only learning in spiking neural  | 10.1088/2634-4386/ae2ef9 | high |
| 25 | 2024 | Nature Communications | A | Advancing spatio-temporal processing through adaptation in spiking neural networks | 10.1038/s41467-025-60878-z | high |
| 26 | 2024 | International Conference on Machine Learning | A | Balanced Resonate-and-Fire Neurons | 10.48550/arXiv.2402.14603 | high |
| 27 | 2024 | Nature Communications | A | DenRAM: neuromorphic dendritic architecture with RRAM for efficient temporal processing wi | 10.1038/s41467-024-47764-w | high |
| 28 | 2024 | Neural Computation | A | Elucidating the theoretical underpinnings of surrogate gradient learning in spiking neural | 10.48550/arXiv.2404.14964 | high |
| 29 | 2024 | Trans. Mach. Learn. Res. | A | Exploring the Limitations of Layer Synchronization in Spiking Neural Networks | 10.48550/arxiv.2408.05098 | high |
| 30 | 2024 | Nature Communications | A | Mosaic: in-memory computing and routing for small-world spike-based neuromorphic systems | 10.1038/s41467-023-44365-x | medium |
| 31 | 2024 | Neural Networks | A | Multi-compartment neuron and population encoding powered spiking neural network for deep d | 10.1016/j.neunet.2024.106898 | medium |
| 32 | 2024 | Science Advances | A | Spatiotemporal audio feature extraction with dynamic memristor-based time-surface neurons | 10.1126/sciadv.adl2767 | medium (manual, from snippets; 2-class subset so excluded from the 20-class protocol tables by note) |
| 33 | 2024 | Trans. Mach. Learn. Res. | A | TSkips: Efficiency Through Explicit Temporal Delay Connections in Spiking Neural Networks | 10.48550/arXiv.2411.16711 | high |
| 34 | 2024 | Nature Communications | A | Temporal dendritic heterogeneity incorporated with spiking neural networks for learning mu | 10.1038/s41467-023-44614-z | medium |
| 35 | 2024 | Neuromorphic Computing and Engineering | A | Temporal hierarchy in spiking neural networks | 10.1088/2634-4386/ae7ab5 | medium |
| 36 | 2024 | Neuromorphic Computing and Engineering | A | Understanding the functional roles of modelling components in spiking neural networks | 10.1088/2634-4386/ad6cef | high |
| 37 | 2024 | Neural Networks | A | Zero-Shot Temporal Resolution Domain Adaptation for Spiking Neural Networks | 10.1016/j.neunet.2025.108483 | high |
| 38 | 2023 | Neural Information Processing Systems | A | Addressing the speed-accuracy simulation trade-off for adaptive spiking neurons | 10.48550/arXiv.2311.11390 | medium |
| 39 | 2023 | Neuromorphic Computing and Engineering | A | ETLP: event-based three-factor local plasticity for online learning with neuromorphic hard | 10.1088/2634-4386/ad6733 | high |
| 40 | 2023 | Neural Computation | A | Exploring Trade-Offs in Spiking Neural Networks | 10.1162/neco_a_01609 | medium |
| 41 | 2023 | AAAI Conference on Artificial Intelligence | A | Harnessing Manycore Processors with Distributed Memory for Accelerated Training of Sparse  | 10.48550/arXiv.2311.04386 | medium |
| 42 | 2023 | International Conference on Learning Represen | A | Heterogeneous Neuronal and Synaptic Dynamics for Spike-Efficient Unsupervised Learning: Th | 10.48550/arXiv.2302.11618 | medium |
| 43 | 2023 | International Conference on Learning Represen | A | Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spaci | 10.48550/arXiv.2306.17670 | high |
| 44 | 2023 | Trans. Mach. Learn. Res. | A | S-TLLR: STDP-inspired Temporal Local Learning Rule for Spiking Neural Networks | 10.48550/arXiv.2306.15220 | high |
| 45 | 2023 | AAAI Conference on Artificial Intelligence | A | TC-LIF: A Two-Compartment Spiking Neuron Model for Long-term Sequential Modelling | 10.48550/arXiv.2308.13250 | high |
| 46 | 2023 | International Conference on Learning Represen | A | The Expressive Leaky Memory Neuron: an Efficient and Expressive Phenomenological Neuron Mo | 10.48550/arxiv.2306.16922 | high |
| 47 | 2025 | Scientific Reports | B | Backpropagation-free spiking neural networks with the forward–forward algorithm | 10.1038/s41598-026-41671-4 | high |
| 48 | 2025 | Scientific Reports | B | Biologically-informed excitatory and inhibitory ratio for robust spiking neural network tr | 10.1038/s41598-025-03408-7 | high |
| 49 | 2023 | IEEE International Conference on Acoustics, S | B | Adaptive Axonal Delays in Feedforward Spiking Neural Networks for Accurate Spoken Word Rec | 10.1109/ICASSP49357.2023.10094768 | high |

## 4. 추출했으나 제외 (대상 데이터셋에서 자기 모델 결과를 보고하지 않음): 10편

- 2026 | Neuromorphic Computing and Engineering | A review of event-based vision sensor fusion: architectures, evaluation and robustness cha
- 2025 | IEEE International Conference on Acousti | Adaptive Central Frequencies Locally Competitive Algorithm for Speech
- 2024 | IEEE Access | Continual Learning With Neuromorphic Computing: Foundations, Methods, and Emerging Applica
- 2025 | Scientific Reports | Incremental self-organization of spatio-temporal spike pattern detection
- 2025 | Neuromorphic Computing and Engineering | NeuroMorse: a temporally structured dataset for neuromorphic computing
- 2024 | Interspeech (USB) | Neuromorphic Keyword Spotting with Pulse Density Modulation MEMS Microphones
- 2023 | Scientific Reports | Oxygenated hemoglobin as prognostic marker among patients with systemic sclerosis screened
- 2023 | Science Advances | SpikingJelly: An open-source machine learning infrastructure platform for spike-based inte
- 2025 | Patterns | Three-factor learning in spiking neural networks: An overview of methods and trends from a
- 2024 | Advances in Neural Computation, Machine  | Сравнение методов кодирования аудиосигналов для многослойных спайковых нейронных сетей в з

## 부록 A. 초록에서 데이터셋을 언급하지만 공개 전문이 없어 추출하지 못한 논문: 11편

| 연도 | 장소 | 군 | 제목 | DOI |
|---|---|---|---|---|
| 2026 | Neuromorphic Computing and Engineering | A | FeNN-DMA: A RISC-V system-on-chip for spiking neural network acceleration | 10.1088/2634-4386/ae688e |
| 2026 | Neural Networks | A | Multi-scale chunked residual encoding and temporal stochastic interpolation padding in SNN | 10.1016/j.neunet.2026.108873 |
| 2026 | 2026 6th International Conference on Adv | A | Three-Compartment Spiking Neural Networks for Lightweight Multi-Timescale Feature Fusion | 10.1109/AANN71145.2026.11709320 |
| 2025 | IEEE Transactions on Biomedical Circuits | A | A 0.66-mm2 0.49 pJ/SOP SNN Processor With Temporal-Spatial Post-Neuron-Processing and Mode | 10.1109/TBCAS.2025.3582246 |
| 2025 | Neuromorphic Computing and Engineering | A | EFLOP: a sparsity-aware metric for evaluating computational cost in spiking and non-spikin | 10.1088/2634-4386/addee8 |
| 2025 | Biomedical Circuits and Systems Conferen | A | Efficient FPGA Implementation of Spiking Neural Networks with Integrated Axonal Delays | 10.1109/BioCAS67066.2025.00133 |
| 2025 | Neural Networks | A | Efficient speech command recognition leveraging spiking neural networks and progressive ti | 10.1016/j.neunet.2025.108253 |
| 2025 | Neural Networks | A | Towards parameter-free attentional spiking neural networks | 10.1016/j.neunet.2025.107154 |
| 2024 | Neural Computation | A | Trainable Reference Spikes Improve Temporal Information Processing of SNNs With Supervised | 10.1162/neco_a_01702 |
| 2026 | IEEE Access | B | Decoder-Aligned Cumulative-Prefix Training for Low-Latency Spiking Neural Network Classifi | 10.1109/access.2026.3711605 |
| 2023 | Scientific Reports | B | Oxygenated hemoglobin as prognostic marker among patients with systemic sclerosis screened | 10.1038/s41598-023-28608-x |

## 부록 B. 제외군(MDPI·Frontiers·Hindawi·CinC·워크숍) 주제 일치 후보: 82편 (장소별 상위)

- Frontiers in Neuroscience: 27
- Neuro Inspired Computational Elements Workshop: 12
- Electronics: 7
- Frontiers Comput. Neurosci.: 4
- Applied Sciences: 4
- Sensors: 4
- Frontiers in Computational Neuroscience: 4
- Mathematics: 4
- International Workshop on Applied Reconfigurable Computing: 2
- International Workshop on Machine Learning for Signal Processing: 2
- Frontiers in Neuroinformatics: 2
- 2023 IEEE 13th Annual Computing and Communication Workshop and Conference (CCWC): 1
- Frontiers in Cellular Neuroscience: 1
- Entropy: 1
- Proceedings of the 24th International Workshop on Mobile Computing Systems and Applications: 1

## 부록 C. A/B군이 아닌 그 밖의 장소(출처 불명 포함) 주제 일치 후보: 502편 (장소별 상위 25)

- Neurocomputing: 18
- International Conference on Systems: 10
- IEEE Transactions on Cognitive and Developmental Systems: 9
- (없음): 6
- International Symposium on Circuits and Systems: 6
- International Conference on Artificial Intelligence Circuits and Systems: 5
- IEEE International Joint Conference on Neural Network: 5
- IEEE Journal of Selected Topics in Quantum Electronics: 5
- Lecture notes in computer science: 4
- IEEE Transactions on Cybernetics: 4
- eLife: 4
- IEEE Transactions on Emerging Topics in Computing: 4
- Archiv für das Studium der neueren Sprachen und Literaturen: 4
- Applied Physics Letters: 4
- Information Sciences: 4
- Neural Processing Letters: 4
- Informationen Deutsch als Fremdsprache: 4
- Physical Review E: 4
- Journal of Lightwave Technology: 4
- IEEE International Symposium on Circuits and Systems proceedings: 3
- iScience: 3
- SoftwareX: 3
- 2023 11th International IEEE/EMBS Conference on Neural Engineering (NER): 3
- International Journal of Neural Systems: 3
- Proceedings of the Great Lakes Symposium on VLSI 2023: 3

## 5. 공통 프로토콜 판독과 우리 채택안 (SHD/SSC)

### 5.1 판독 — 무엇이 "관행"이고 무엇이 관행이 **아닌가**
빈도표(2절)의 A군 기준이다. 비율의 분모는 포함 논문 수이고, "not stated"가 많은 항목은 그 자체가 결론이다.

| 항목 | 관행으로 볼 수 있는 것 | 근거 | 주의 |
|---|---|---|---|
| 시간 구간 폭 | 명시한 논문 중 **10 ms(스텝 100개)** 와 **4 ms(스텝 250개)** 가 두 축, 1 ms(스텝 1000개)는 지연 학습·사건 구동 계열 | 10 ms 6편, 4 ms 5편, 1 ms 5편, 2 ms 3편 | 절반(23/46)이 폭을 적지 않고 스텝 수만 적거나 둘 다 적지 않음 |
| 스텝 수 | **100** 또는 **250** | 100: 10편, 250: 8편 | SHD 길이가 가변이라 1 s로 자르거나 끝을 0으로 채움 |
| 입력 채널 | **700 그대로** 가 다수, 140(5배 묶음)이 그다음 | 700: 21편, 140: 6편 | 하드웨어 논문은 70·256·350으로 줄임 |
| 모델 선택 | **관행이 없다.** 65%가 선택 방식을 적지 않음. 적은 논문은 "test를 검증으로 씀"(6편)과 "학습 데이터 일부 분리"(5편)·"화자 단위 교차검증"(2편)으로 갈림 | 표 그대로 | SHD에 공식 검증 세트가 없어 생긴 문제. test로 고르는 관행은 수치를 부풀린다 |
| 증강 | 적지 않은 논문이 2/3. 적은 논문은 채널 이동(shift)·같은 클래스 섞기(blend)·사건 드롭이 주류 | used 12편, none 3편 | 최신 상위 결과(지연 학습·DCLS·SE-adLIF)는 증강을 씀 |
| 모델 계열 | **재귀 SNN**이 가장 흔하고, 순방향 SNN·지연 학습이 뒤따름 | 재귀 19, 순방향 14, 지연 7 | 스파이킹 트랜스포머는 1편뿐 |
| 뉴런 | 보통 LIF 다음으로 **다중 구획·가지(dendritic) 뉴런** 과 적응(ALIF)·이질 시간상수가 많음 | 다중 구획 12, ALIF 8, 이질 τ 5 | 우리 population 설계(DH-SNN 계열)가 바로 이 부류 |
| seed 수 | **5개**가 최빈, 3개가 그다음; 절반 가까이는 적지 않음 | 5: 10편, 3: 6편, 10: 4편, 미기재 21편 | |
| 보고 형식 | **평균 ± 표준편차** 가 절반, 단일 값이 1/3 | 24 / 16 | 단일 값 논문의 상당수는 표에는 단일, 그림에는 분산 |
| 지표 | 시험 정확도 (전원) | | F1은 쓰지 않음 (클래스 균형) |

### 5.2 우리 채택안 (사전 고정 후보; 사용자 승인 뒤 확정)
1. **데이터:** SHD(20클래스, 학습 8,156 / 시험 2,264)와 SSC(35클래스, 공식 학습/검증/시험 분할). SSC는 규모가 커서 SHD 결과가 나온 뒤 조건을 줄여 돌린다.
2. **입력:** 700채널 그대로, **10 ms 구간 → 스텝 100개**(1 s로 자르고 끝은 0 채움)를 기본으로 하고, **4 ms → 250 스텝**을 두 번째 설정으로 둔다. 구간 안 스파이크 수(정수)를 입력으로 쓴다. 이 두 설정이 가장 많이 쓰였고, 우리 모델(T = N 정렬)에서는 스텝 수가 곧 토큰 수라서 기억 길이 비교(100 vs 250)가 자연스럽다.
3. **모델 선택:** SHD는 **학습 데이터의 10%를 떼어 검증**하고(seed마다 다시 분할), test는 마지막에 한 번만 본다. SSC는 공식 검증 세트. "test를 검증으로 쓰는" 관행은 따르지 않으며, 비교 표에는 그 논문들을 표시한다.
4. **증강:** 기본은 **없음**(뉴런 효과를 분리하기 위해). 채널 이동 ±(shift)와 같은 클래스 섞기(blend)를 켠 조건을 **별도 행**으로 둔다 — 상위 결과와의 비교를 위해.
5. **seed:** **5개**, 평균 ± 표준편차. 판정은 짝지은 seed 차이로 한다(5–7단계 규칙 유지).
6. **지표:** 시험 정확도. 에너지는 우리 규약(시냅스 + 뉴런 고유)으로 함께 보고하되 외부 수치와 직접 비교하지 않는다.
7. **구조(우리 질문에 맞춤):** 7단계의 주의 없는 정렬 모델(`neo_mixer`)을 주 모델로: 임베딩 Linear(700 → D) → BN → 뉴런(스텝 = 토큰) → 블록 × L {양방향 mixer, MLP} → 스텝 평균 → 20클래스. 조건은 {섞기 없음(하한), 양방향 LIF, 양방향 f-LIF, 양방향 population f-LIF} + attention 상한. 외부 참고선은 같은 입력 설정(10 ms·100스텝)의 재귀 SNN 보고치(표 3의 논문들)로 한다.
8. **사전 검사:** 본 학습 전에 작은 모델로 "마지막 k스텝만 남기기"를 재서 과제가 기억을 요구함을 확인한다(예상: 무너짐).

### 5.3 한계
- A군 165편 중 전문을 확보한 것은 86편(52%)이고, 유료 저널(TNNLS 19, Neural Networks 13, TCAS 10 등)의 상당수는 초록만으로 "SHD 사용"을 확정할 수 없어 포함하지 못했다(부록 A에 초록에 데이터셋이 보이는 11편을 적음).
- 추출은 자동 보조(병렬 작업자 3개)로 했고, 근거 인용문 290개 중 227개(78%)를 원문과 기계 대조로 확인했다(나머지는 PDF 열 배치로 인용문이 끊긴 경우이며, 표본 검사로 실존을 확인). 각 논문 행의 `extraction_confidence`를 함께 적었다.
- DBLP·OpenReview를 쓸 수 없어 ICLR/NeurIPS 커버리지는 인용 추적과 arXiv에 의존했다(두 경로에서 ICLR 5편·NeurIPS 4편이 포함됨).

