# ECG 리듬 분류 (CinC 2017 · MIT-BIH AFDB) — 2023–2026 선행 연구 프로토콜 조사

생성: `docs/surveys/report.py` (수집 `collect.py`, 선별 `screen.py`, 추출 `raw/extract/*.jsonl`). 수치는 추출 표의 값을 그대로 집계한 것이며, "not stated"는 본문에 명시되지 않은 항목이다.

## 1. 모집단과 선별 (PRISMA 식 건수)

- 수집(2023–2026, 경로별 원건수): S2CIT 396, OACIT 354, OASRCH 2784, ARXIV 89, XREF 4000, PUBMED 96 → 중복 제거 후 5578편
- 장소(venue) 분류 · 전체: A군 787, B군 262, 제외군(MDPI·Frontiers·Hindawi·CinC 프로시딩·워크숍) 332, 프리프린트 전용 297, 출처 불명/기타 3900
- 제목·초록 주제 일치 · A군 280, B군 207
- 공개 전문 확보 · A군 181/280, B군 108/207 (나머지는 초록만)
- 전문에서 대상 데이터셋을 실제로 사용(언급 ≥ 2) · A군 9, B군 17
- 추출 후 포함(자기 모델의 결과를 보고) · **A군 7, B군 15**

## 2. 항목별 빈도

**데이터셋** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| CinC 2017 + AFDB | 2 (29%) | 6 (27%) |
| CinC 2017 | 4 (57%) | 10 (45%) |
| AFDB | 1 (14%) | 6 (27%) |

**분할 방식** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| official hidden test | 1 (14%) | 1 (5%) |
| k-fold CV | 2 (29%) | 8 (36%) |
| fixed hold-out | 3 (43%) | 9 (41%) |
| other | 1 (14%) | 3 (14%) |
| not stated | 0 (0%) | 1 (5%) |

**k-fold의 k** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| 5 | 1 (14%) | 4 (18%) |
| 10 | 0 (0%) | 2 (9%) |
| other k | 1 (14%) | 1 (5%) |
| not stated | 5 (71%) | 15 (68%) |

**분할 단위** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| patient/record-wise | 3 (43%) | 11 (50%) |
| random (sample-wise) | 0 (0%) | 3 (14%) |
| not stated | 4 (57%) | 8 (36%) |

**표본화율** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| 300 Hz | 2 (29%) | 7 (32%) |
| 250 Hz | 1 (14%) | 4 (18%) |
| 125/128 Hz | 0 (0%) | 2 (9%) |
| 100 Hz | 1 (14%) | 1 (5%) |
| 500 Hz | 0 (0%) | 1 (5%) |
| other | 1 (14%) | 1 (5%) |
| not stated | 2 (29%) | 6 (27%) |

**입력 길이** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| 30 s | 1 (14%) | 7 (32%) |
| 10 s | 2 (29%) | 4 (18%) |
| 60 s | 1 (14%) | 1 (5%) |
| 5 s | 0 (0%) | 1 (5%) |
| other | 1 (14%) | 6 (27%) |
| not stated | 2 (29%) | 3 (14%) |

**불균형 처리** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| resampling / balanced subset | 1 (14%) | 3 (14%) |
| no imbalance handling stated (plain loss) | 3 (43%) | 13 (59%) |
| not stated | 3 (43%) | 6 (27%) |

**주 지표** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| F1 (mean of classes) | 5 (71%) | 12 (55%) |
| accuracy | 1 (14%) | 5 (23%) |
| sensitivity/specificity | 1 (14%) | 2 (9%) |
| AUROC/AUC | 0 (0%) | 3 (14%) |

**F1 평균에 잡음 클래스 포함 여부** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| included | 0 (0%) | 2 (9%) |
| excluded | 0 (0%) | 3 (14%) |
| not stated | 7 (100%) | 17 (77%) |

**증강** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| none stated | 3 (43%) | 5 (23%) |
| used | 0 (0%) | 3 (14%) |
| not stated | 4 (57%) | 14 (64%) |

**seed/run 수** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| 5 | 1 (14%) | 3 (14%) |
| not stated | 6 (86%) | 19 (86%) |

**보고 형식** (A군 n=7 / A+B군 n=22)

| 값 | A군 | A+B군 |
|---|---|---|
| mean only | 0 (0%) | 1 (5%) |
| mean±std (table or figure) | 2 (29%) | 4 (18%) |
| other | 0 (0%) | 1 (5%) |
| single value | 5 (71%) | 16 (73%) |

## 3. 포함 논문

| # | 연도 | 장소 | 군 | 제목 | DOI/arXiv | 추출 신뢰도 |
|---|---|---|---|---|---|---|
| 1 | 2025 | Heart Rhythm | A | Enhanced detection of atrial fibrillation in single-lead electrocardiograms using a Cloud- | 10.1016/j.hrthm.2024.12.048 | high |
| 2 | 2025 | AAAI Conference on Artificial Intelligence | A | Parallel-Learning of Invariant and Tempo-variant Attributes of Single-Lead Cardiac Signals | 10.48550/arXiv.2502.21162 | medium |
| 3 | 2024 | European Heart Journal - Digital Health | A | Decoding 2.3 million ECGs: interpretable deep learning for advancing cardiovascular diagno | 10.1093/ehjdh/ztae014 | medium |
| 4 | 2024 | Patterns | A | Decorrelative network architecture for robust electrocardiogram classification | 10.1016/j.patter.2024.101116 | high |
| 5 | 2024 | International Conference on Learning Represen | A | Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electroc | 10.48550/arXiv.2402.09450 | high |
| 6 | 2024 | IEEE Transactions on Biomedical Circuits and  | A | Towards Hardware Supported Domain Generalization in DNN-Based Edge Computing Devices for H | 10.1109/TBCAS.2024.3418085 | medium |
| 7 | 2023 | European Heart Journal - Digital Health | A | Automatic screening of patients with atrial fibrillation from 24-h Holter recording using  | 10.1093/ehjdh/ztad018 | high |
| 8 | 2026 | Scientific Reports | B | An efficient deep CNN based BiLSTM framework with RanA optimization for accurate cardiac a | 10.1038/s41598-026-38227-x | medium |
| 9 | 2026 | Scientific Reports | B | Beyond full fine-tuning: towards enhanced generalizability in downstream ECG foundation mo | 10.1038/s41598-026-71351-2 | high |
| 10 | 2026 | Physiological Measurement | B | Multi-window temporal analysis for enhanced arrhythmia classification: leveraging long-ran | 10.1088/1361-6579/ae3937 | high |
| 11 | 2026 | Scientific Reports | B | Robust and generalizable atrial fibrillation detection from ECG using time-frequency fusio | 10.1038/s41598-026-64715-1 | high |
| 12 | 2026 | Scientific Reports | B | sRQA: an integrative pipeline for symbolic recurrence quantification analysis | 10.1038/s41598-026-74593-2 | high |
| 13 | 2025 | Scientific Reports | B | A knowledge embedded multimodal pseudo-siamese model for atrial fibrillation detection | 10.1038/s41598-025-87115-3 | high |
| 14 | 2025 | Scientific Reports | B | An integrated algorithm for single lead electrocardiogram signal analysis using deep learn | 10.1038/s41598-025-18910-1 | medium |
| 15 | 2025 | Scientific Reports | B | Robust screening of atrial fibrillation with distribution classification | 10.1038/s41598-025-10090-2 | high |
| 16 | 2025 | Biomedical Signal Processing and Control | B | Self-supervised inter-intra period-aware ECG representation learning for detecting atrial  | 10.1016/j.bspc.2024.106939 | medium |
| 17 | 2025 | Scientific Reports | B | Semantic ECG hash similarity graph | 10.1038/s41598-025-07838-1 | medium |
| 18 | 2025 | Scientific Reports | B | Transformer-based heart language model with electrocardiogram annotations | 10.1038/s41598-024-84270-x | high |
| 19 | 2024 | Scientific Reports | B | Evaluation of the performance of a machine learning based atrial fibrillation screening al | 10.1038/s41598-024-74157-2 | high |
| 20 | 2023 | Expert Systems with Applications | B | Automated atrial fibrillation classification based on denoising stacked autoencoder and op | 10.1016/j.eswa.2023.120975 | high |
| 21 | 2023 | Scientific Reports | B | Deep residual-dense network based on bidirectional recurrent neural network for atrial fib | 10.1038/s41598-023-40343-x | medium |
| 22 | 2023 | Scientific Reports | B | Robust electrocardiogram delineation model for automatic morphological abnormality interpr | 10.1038/s41598-023-40965-1 | high |

## 4. 추출했으나 제외 (대상 데이터셋에서 자기 모델 결과를 보고하지 않음): 19편

- 2025 | IEEE Transactions on Biomedical Engineer | A Dataset and Benchmarks for Atrial Fibrillation Detection from Electrocardiograms of Inte
- 2023 | IEEE Transactions on Biomedical Engineer | A Review on Atrial Fibrillation Detection From Ambulatory ECG — 수동 제외: review; its own 5-fold AFDB check reports numbers only in image tables (not extractable) -> no protocol facts
- 2023 | Scientific Reports | A framework for comparative study of databases and computational methods for arrhythmia de
- 2026 | Scientific Reports | A joint CNN-Bi-LSTM-transformer architecture with SHAP explanations for multi-label arrhyt
- 2025 | Scientific Reports | A neuromorphic approach to early arrhythmia detection
- 2024 | Physiol. Meas. 39 (2018) 115006 (15pp) | A novel wavelet-based filtering strategy to remove powerline interference from electrocard — 수동 제외: published 2018 (Physiol. Meas. 39, 115006); metadata year 2024 is wrong, and the paper is a powerline-interference filter evaluated on CinC 2017 excerpts, not a classifier
- 2026 | Scientific Reports | Benchmarking machine learning and deep learning models with a novel graph-attention SAGECo
- 2023 | IEEE Transactions on Biomedical Engineer | Characterization of Atrial Fibrillation Episode Patterns: A Comparative Study — 수동 제외: episode-pattern agreement analysis on AFDB; no trained classifier, no split
- 2025 | IEEE International Conference on Acousti | CoFE: A Framework Generating Counterfactual ECG for Explainable Cardiac AI-Diagnostics — 수동 제외: counterfactual-generation framework; an AF classifier is trained on CinC 2017 only as a demonstration target and no CinC 2017 performance is reported
- 2024 | npj Digital Medicine | Development and validation of machine learning algorithms based on electrocardiograms for 
- 2023 | IEEE Access | ECG Biometric Recognition: Review, System Proposal, and Benchmark Evaluation
- 2026 | Physiological Measurement | ECG-image-database: large-scale paired ECG images and time-series with real-world artifact
- 2024 | European Heart Journal - Digital Health | Evaluating the impacts of digital ECG denoising on the interpretive capabilities of health
- 2026 | Physiological Measurement | From PhysioNet to foundation models—a history and potential futures
- 2023 | IEEE journal of biomedical and health in | In-Distribution and Out-of-Distribution Self-Supervised ECG Representation Learning for Ar
- 2026 | Scientific Reports | Long-term continuous monitoring of wearable ECG signals for atrial fibrillation detection
- 2023 | Comput. Biol. Medicine | MVKT-ECG: Efficient Single-lead ECG Classification on Multi-Label Arrhythmia by Multi-View
- 2026 | Scientific Reports | Multi-label arrhythmia classification from Holter scatter plots using deep learning and cl
- 2026 | European Heart Journal - Digital Health | Novel machine learning fusion architectures integrating electrocardiogram representations:

## 부록 A. 초록에서 데이터셋을 언급하지만 공개 전문이 없어 추출하지 못한 논문: 16편

| 연도 | 장소 | 군 | 제목 | DOI |
|---|---|---|---|---|
| 2026 | European Heart Journal - Digital Health | A | Cross-dataset ECG classification using deep metric learning with an extended ECG context w | 10.1093/ehjdh/ztaf143.033 |
| 2025 | Circulation | A | Abstract 4370378: Camera is All You Need: Low-cost Atrial Fibrillation Detection using Fac | 10.1161/circ.152.suppl_3.4370378 |
| 2024 | IEEE Transactions on Biomedical Engineer | A | A Novel Deep Ensemble Method for Selective Classification of Electrocardiograms | 10.1109/TBME.2024.3476088 |
| 2024 | European Heart Journal | A | A multilabel deep learning model for the detection of conduction and rhythm disorders from | 10.1093/eurheartj/ehae666.3437 |
| 2023 | Artificial Intelligence in Medicine | A | Few-shot transfer learning for personalized atrial fibrillation detection using patient-ba | 10.1016/j.artmed.2023.102644 |
| 2026 | IEEE Access | B | FedShield: Configurable Mean–Median Robust Aggregation for Poisoning-Resilient Federated H | 10.1109/ACCESS.2026.3728130 |
| 2026 | IEEE Access | B | When and How to Focus: A Task-Dependent Analysis of Efficient Attentive Architectures for  | 10.1109/ACCESS.2026.3658301 |
| 2025 | IEEE Access | B | A Deep Learning-Based Ultra-Lightweight Architecture for Atrial Fibrillation Detection Usi | 10.1109/access.2025.3567093 |
| 2025 | IEEE Access | B | Combination Hand-Crafted Features and Semi-Supervised Features Selection From Deep Feature | 10.1109/ACCESS.2024.3524764 |
| 2024 | IEEE Access | B | Atrial Fibrillation Detection From Electrocardiogram Signal on Low Power Microcontroller | 10.1109/ACCESS.2024.3422329 |
| 2024 | Expert Systems with Applications | B | Directional statistics-inspired end-to-end atrial fibrillation detection model based on EC | 10.1016/j.eswa.2023.123112 |
| 2024 | Comput. Biol. Medicine | B | Novel interpretable Feature set extraction and classification for accurate atrial fibrilla | 10.1016/j.compbiomed.2024.108872 |
| 2024 | Comput. Biol. Medicine | B | Towards federated transfer learning in electrocardiogram signal analysis | 10.1016/j.compbiomed.2024.107984 |
| 2024 | IEEE Access | B | TransMixer-AF: Advanced Real-Time Detection of Atrial Fibrillation Utilizing Single-Lead E | 10.1109/ACCESS.2024.3467181 |
| 2023 | IEEE Access | B | Atrial Fibrillation Detection Algorithm Based on Graph Convolution Network | 10.1109/access.2023.3291352 |
| 2023 | Physiological Measurement | B | Atrial fibrillation detection with signal decomposition and dilated residual neural networ | 10.1088/1361-6579/acfa61 |

## 부록 B. 제외군(MDPI·Frontiers·Hindawi·CinC·워크숍) 주제 일치 후보: 305편 (장소별 상위)

- Sensors: 63
- Frontiers in Cardiovascular Medicine: 34
- Diagnostics: 25
- Bioengineering: 24
- Applied Sciences: 23
- Computing in cardiology: 21
- Frontiers in Physiology: 19
- Electronics: 9
- Algorithms: 7
- Frontiers in Medicine: 6
- International Conference on Computing in Cardiology: 6
- Frontiers in Artificial Intelligence: 5
- Mathematics: 4
- Frontiers in Digital Health: 4
- Frontiers in Bioengineering and Biotechnology: 3

## 부록 C. A/B군이 아닌 그 밖의 장소(출처 불명 포함) 주제 일치 후보: 1547편 (장소별 상위 25)

- The Journal of Thoracic and Cardiovascular Surgery: 30
- Journal of Clinical Medicine: 26
- Heart Rhythm O2: 20
- PLoS ONE: 18
- Heliyon: 18
- Italian National Conference on Sensors: 16
- Institute of Electrical and Electronics Engineers (IEEE): 14
- PubMed: 13
- Cureus: 12
- Communications Medicine: 12
- Reviews in Cardiovascular Medicine: 12
- Scientific Data: 10
- Lecture notes in computer science: 10
- International Journal of Innovative Technologies in Social Science: 10
- Journal of Cardiovascular Development and Disease: 10
- BMC Medical Informatics and Decision Making: 10
- International Journal of Advanced Computer Science and Applications: 9
- Revista Argentina de Cardiología: 9
- IEEE Sensors Journal: 8
- PLOS Digital Health: 8
- BMC Cardiovascular Disorders: 8
- Digital Health: 8
- Cardiovascular Digital Health Journal: 8
- Open Heart: 7
- Biomedicines: 7

## 5. 공통 프로토콜 판독과 우리 채택안 (ECG 리듬 분류: CinC 2017 · AFDB)

### 5.1 먼저 알아야 할 사실 — top-tier에는 이 데이터셋을 쓰는 논문이 거의 없다
- A군(top-tier) 주제 일치 280편 중 공개 전문을 확보한 181편을 읽은 결과, **CinC 2017 또는 AFDB로 자기 모델을 학습·평가한 논문은 7편**뿐이다(나머지 172편은 PTB-XL 19편, MIT-BIH 부정맥 DB 8편, 그 밖에는 병원 자체 코호트·Holter·웨어러블 데이터). 초록만 있는 99편 중 초록에 두 데이터셋이 보이는 것은 5편이다(부록 A).
- CinC 2017 연구의 대부분은 B군(Scientific Reports 14편 등)과 제외군(MDPI·Frontiers 305편), 출처 불명군(1,547편)에 있다. 즉 **"top-tier의 관행"이라고 부를 만한 표본은 7편이고, A+B를 합쳐도 22편**이다. 아래 빈도는 그 한계 안에서 읽어야 한다.
- 2023–2026의 top-tier ECG 논문은 대체로 대규모 12리드 병원 데이터(수십만 건)로 학습하고 외부 데이터로 검증하는 임상 연구다. CinC 2017은 **외부 검증용 소규모 세트**로 쓰이는 경우가 많다(Heart Rhythm 2025: 8,528개 전부를 외부 test로; EHJ-DH 2024: 학습 후 holdout).

### 5.2 판독 (A군 7편 / A+B군 22편)

| 항목 | 관행으로 볼 수 있는 것 | 근거 | 주의 |
|---|---|---|---|
| 분할 | **고정 홀드아웃**(A 3 / A+B 9)과 **k-fold 교차검증**(2 / 8)이 반반; 공식 숨은 test는 1편 | 표 그대로 | k는 5가 가장 흔함(4편), 10은 2편. 대회 공식 test는 더 이상 쓸 수 없음 |
| 분할 단위 | **환자(기록) 단위**가 명시된 쪽의 다수(3 / 11); 무작위 샘플 단위 3편은 모두 B군 | | A군 4편은 단위를 적지 않음. AFDB는 기록 단위가 필수(같은 기록의 10 s 조각이 train·test에 섞이면 누출) |
| 표본화율 | 300 Hz 그대로(2 / 7) 또는 250 Hz로 통일(1 / 4); 100·125/128 Hz도 있음 | | 재표본은 AFDB(250 Hz)와 맞추려는 선택 |
| 입력 길이 | **10 s**(2 / 4)와 **30 s 또는 기록 전체**(1+1 / 7+)가 두 축 | | 10 s는 30 s 기록을 3조각으로 잘라 표본을 늘리는 쪽, 30 s는 기록 단위 라벨에 맞추는 쪽 |
| 불균형 | 가중치·focal을 쓴 논문 0; 균형 부분집합 3; 나머지는 **보통 CE** | | 논문 B.4의 "CAL"은 이 표본에서 아무도 쓰지 않음 |
| 주 지표 | **F1(클래스 평균)** 이 다수(5 / 12), 정확도(1 / 5), AUROC(0 / 3) | | 잡음 클래스 포함 여부를 적은 논문은 5편뿐(포함 2·제외 3) |
| 추론 결합 | 10 s 조각을 쓴 논문은 조각별 예측을 기록 단위로 평균·다수결하거나(일부) 조각 단위로 평가(일부) | 추출 표 `inference_aggregation` | 기록 단위 지표로 통일해야 비교 가능 |
| seed | **거의 적지 않음**(미기재 6 / 19); 5 seed 1 / 3편 | | 단일 값 보고가 5 / 16 |
| 증강 | 미기재가 대다수; 명시적으로 없음 3 / 5 | | |

### 5.3 우리 채택안 (사전 고정 후보; 사용자 승인 뒤 확정)
1. **주 과제:** CinC 2017 학습 세트 8,528기록, 4클래스(N / AF / Other / Noisy), v3 라벨. 숨은 test는 쓸 수 없으므로 **기록 단위 층화 5-fold 교차검증**(fold = test, 나머지의 10%를 검증)을 쓴다. 홀드아웃 1회보다 분산 추정이 되고, k=5가 표본에서 가장 흔하다. 본 실험은 fold 1개 × seed 3으로 조건을 가리고, 후보가 나오면 5-fold 전체로 확증한다.
2. **보조 과제(흐름 처리 확장):** AFDB 23기록, AF vs 비AF 2클래스, **기록 단위 leave-one-record-out 또는 5-fold**, 10 s 조각(겹침 없음). 뉴런 상태를 리셋하지 않는 흐름 설정은 이 과제에서만 시험한다.
3. **입력:** 300 Hz → **125 Hz 재표본**(MIT-BIH 5–7단계와 맞춤; 표본에 250·128·100 Hz 모두 있어 관행 위반은 아님), 0.5–40 Hz 대역 통과, 기록 단위 z-score. **길이 30 s**(기록 단위 라벨에 맞춤; 짧은 기록은 반복 채움, 긴 기록은 30 s 창을 겹쳐 밀어 확률 평균) → 패치 32·보폭 16으로 **토큰 N = T ≈ 233**. 10 s 조각(N ≈ 77) 설정을 **길이 ablation**으로 둔다 — 기억 길이 효과를 분리하는 데 핵심.
4. **손실·불균형:** CE + 클래스 가중(CAL, 5단계와 같음)을 기본으로 하되, 표본의 관행(보통 CE)과 다르므로 **보통 CE 조건을 한 줄 더** 둔다. 잡음 클래스는 학습에 포함.
5. **지표:** **F1_N·F1_AF·F1_Other의 평균(대회 점수, 잡음 제외)** 을 주 지표로 보고하고, 잡음 포함 4클래스 macro-F1·정확도·AF 민감도/특이도·혼동 행렬을 함께 적는다. 기록 단위로 평가한다.
6. **seed:** 3개(탐색) → 확증 시 5개, 평균 ± 표준편차, 짝지은 seed 차이로 판정(5–7단계 규칙).
7. **구조:** 7단계의 주의 없는 정렬 모델(`neo_mixer`)을 주 모델로, 조건 {섞기 없음(하한), 양방향 LIF, 양방향 f-LIF, 양방향 population f-LIF} + attention 상한(토큰 233개에서 attention 비용이 커서 에너지 대비가 분명해짐). 외부 참고선은 같은 데이터·k-fold를 쓴 B군 논문의 F1(표 3)로 하되 분할이 달라 직접 비교하지 않는다.
8. **사전 검사:** 본 학습 전에 "마지막 k토큰만 남기기"로 과제가 기억을 요구함을 확인한다(AF 판별은 간격의 열이므로 무너져야 정상).

### 5.4 한계
- A군 표본이 7편이라 "관행"은 A+B 22편으로 읽었고, 두 열을 나란히 적었다. B군의 Scientific Reports(12편)가 표본을 지배한다.
- A군 280편 중 99편은 유료 저널(Heart Rhythm 27, Europace 18, Circulation 11 등)이라 초록만 보았다. Europe PMC 공개 전문 경로로 EHJ-Digital Health 등 97편을 추가 확보했지만, 그 안에서도 CinC 2017·AFDB 사용은 드물었다.
- 추출은 병렬 작업자 2개가 했고 근거 인용문 202개 중 169개(84%)를 원문과 기계 대조로 확인했다(나머지는 열 배치·XML 변환으로 문자열이 끊긴 경우). 각 논문 행에 `extraction_confidence`를 적었다.
- 2018년 논문 1편이 메타데이터 연도 오류로 섞여 있어 수동 제외했고, 분류기가 아닌 3편(리뷰·에피소드 분석·반사실 생성 틀)도 수동 제외했다(4절·`ecg_overrides.json`).

