# ECG와 긴 biosignal 분류의 SNN 선행 연구: MIT-BIH, CinC 2017, 다른 긴 신호, non-spiking 기준, SpikMixer와의 overlap

## Q1. MIT-BIH SNN 논문: 모델, encoding, split, T, 정확도·F1, 에너지와 계산법, hardware. beat-level split인 우리에게 믿을 만한 비교는 무엇이고, 이 split은 약하다고 보는가

### Takeaway
- 네, 약하다고 봅니다. 같은 환자의 beat가 train과 test에 함께 들어가는 split은 이 분야 문헌에서 "highly discouraged", "not highly reliable"로 적혀 있습니다. 그래도 random beat split 결과가 TBioCAS, DATE, IEEE Sensors J에 실립니다. 2024-2026년 논문은 inter-patient(DS1/DS2)를 두 번째 평가로 함께 싣는 쪽으로 움직이고 있습니다.
- 같은 모델의 intra-patient와 inter-patient 차이가 큽니다. Scrugli 2026은 accuracy 98.4 → 92.44, F1 88.6 → macro-F1 0.425입니다. Galvis-Chacón 2025는 multi-class accuracy 98.23 → 86.38입니다. 우리 macro-F1 91.99는 inter-patient로 옮겨지지 않는다고 봐야 합니다.
- 우리 데이터에는 reviewer가 짚을 문제가 하나 더 있습니다. 5 class 중 Q가 8,039 beat입니다. AAMI 절차대로 paced 기록 4개를 빼면 Q는 15 beat로 줄기 때문에, 우리 데이터에는 paced beat가 남아 있다고 추정합니다. Q F1 98.95가 macro-F1의 1/5을 차지합니다. 또 CSV에 환자 ID 열이 없어서, DS1/DS2를 돌리려면 raw MIT-BIH에서 beat를 다시 잘라야 합니다.

### Cited Findings

#### 표기
- [read in full text]는 해당 대목을 논문 전문(PDF, PMC XML, 저자 원고)에서 직접 읽었다는 뜻입니다. 일부만 읽었으면 "(부분: 읽은 범위)"를 붙였습니다. "(도구 요약)"은 전문을 fetch 도구의 요약으로 읽었다는 뜻입니다. [abstract only]는 초록만, [secondary summary]는 review나 제3자 정리, 검색 요약만 본 경우입니다. "내 계산"은 표의 숫자로 제가 계산한 값입니다. ScienceDirect와 MDPI는 이 서버에서 403이라, 그쪽 논문은 다른 경로로 확인했습니다.

#### SNN 논문 표: inter-patient 또는 환자 분리를 보고한 것
| 논문, venue | 모델 | encoding | split | T | accuracy·F1 (원문 숫자) | 에너지와 계산법 | hardware | 근거 |
|---|---|---|---|---|---|---|---|---|
| [Yan, Zhou, Wong 2021, BSPC 63:102170](https://doi.org/10.1016/j.bspc.2020.102170) | 2단계 CNN을 SNN으로 변환 | rate | inter-patient, 4 class (N, SVEB, VEB, F) | 읽은 범위에 없음 | "90% accuracy for 4 classes with an inter-patient paradigm" ([BEL 2024 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362428/)). N F1 94.4, V F1 66.8 ([Rana & Kim Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175061/)). "90% accuracy with 61.5% sensitivity and 76% specificity" ([Scrugli 2026](https://zenodo.org/records/18887975)). [Survey](https://arxiv.org/abs/2506.01737)는 94.5%로 적어 서로 다릅니다 | "0.077W" vs CNN "10.4W" (BEL review). 계산법은 못 읽음 | Scrugli 표는 ASIC로 분류 | [secondary summary] |
| [Rana & Kim 2024, Sensors 24:3426](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175061/) | 12층 ANN을 6층 Att-SNN으로 변환, attention module | SpikingJelly LIF | "70:30" random과 Inter-Patient(22/22 record) 둘 다 | "best accuracy with five timesteps"(시작값 100) | 5 class Table 1: Acc 93.8, F1 85.4, Pre 88.0. Table 3(Inter-Patient): Acc 93.8, N F1 95.5, V F1 73.5. 두 표의 Acc가 같아 어느 split 값인지 분명하지 않습니다 | "278 mW", 측정법은 "tested our model in FPGA using Vivado 2019.2"뿐 | FPGA | [read in full text] (부분: 평가 절과 표 1-3) |
| [Galvis-Chacón et al. 2025, Neurocomputing](https://doi.org/10.1016/j.neucom.2025.132259) | axonal delay SNN (D-SNN) | 못 읽음 | intra와 inter 둘 다, AAMI | 못 읽음 | binary inter 83.15, intra 98.27. multi-class inter 86.38, intra 98.23 | "estimated 1.91 µJ per inference"(intra multi-class, L1과 pruning 후) | 없음(추정) | [secondary summary] [기관 저장소 초록](https://www.recercat.cat/handle/20.500.14342/5924) 발췌, 전문은 2028-02-26까지 embargo |
| [Scrugli et al. 2026, IEEE Sensors J](https://doi.org/10.1109/jsen.2026.3663715) | all-spiking, peak detection과 classification 통합 | raw와 1차·2차 미분의 delta modulation | 주 결과 intra-patient 70/10/20. 추가로 Chazal inter-patient | 못 읽음 | intra: 98.4, Se 88.11, Spe 97.66, F1 88.6, class F1 N 99.19, S 80.42, V 95.76. inter: accuracy 92.44, macro-F1 0.425(S sensitivity 0.22%). RR branch 추가 후 92.78, macro-F1 0.561(S sensitivity 64.06%) | "we measured an average power consumption of 9.1 mW", 36.86 µJ | Lattice iCE40-UltraPlus FPGA, 측정 | [read in full text] (부분: [저자 원고](https://zenodo.org/records/18887975)의 split, 결과, 확장 평가) |
| [Banjo & Ghoraani 2026, Sensors](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306364/) | 2단계 quantized CSNN (QCSNN), RR feature | direct input | 주 결과 patient-specific 80/20. 탐색용으로 DS1/DS2 | 못 읽음 | intra 99.02. DS2 Stage-2 accuracy 89.31 ± 1.51 | PYNQ-Z2, 2.02 W, 11.54 ms per beat. 본문이 "GPU 2.8× lower energy"와 "FPGA 4.01× lower energy"를 함께 적어 서로 맞지 않습니다 | FPGA | [read in full text] (부분: 절차, 표 설명, 한계) |
| [BioSpike-Net (Ding & Tian 2026), Biomimetics](https://pmc.ncbi.nlm.nih.gov/articles/PMC13509974/) | LI-SConv 3층, novelty-gated attention, GRU predictor | sensory-adaptive encoder | patient-disjoint 5-fold, 2 s window 50% overlap, DS1/DS2 아님, seed 42 | T = 1 (direct LIF) | 97.6 ± 0.3, macro-F1 95.8 ± 0.4 | 45 nm 추정 6.4 µJ, E_MAC 4.6 pJ, E_AC 0.9 pJ. "memory access and routing overheads are excluded"라고 한계 명시 | 없음 | [read in full text] (부분: 절차, 결과, 한계) |

#### SNN 논문 표: patient-specific 또는 random beat split
| 논문, venue | 모델 | encoding | split | T | accuracy·F1 (원문 숫자) | 에너지와 계산법 | hardware | 근거 |
|---|---|---|---|---|---|---|---|---|
| [Amirshahi & Hashemi 2019, IEEE TBioCAS](https://arxiv.org/abs/1905.02954) | STDP와 R-STDP | Poisson | patient-specific: DS1 대표 beat와 그 환자의 첫 5분으로 학습, DS2의 24명 시험 | beat당 200 ms 제시 | 97.9. F1 88.0, Se 80.2 ([BEL review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362428/)) | "1.78 µJ per beat", 시뮬레이션과 사건당 값(spike 50 pJ, synaptic event 147 pJ) | 칩 없음(추정) | [read in full text] (도구 요약) |
| [Scrugli et al. 2024, DATE](https://doi.org/10.23919/date58400.2024.10546736) | SLAYER SNN | delta modulator | "intra-patient testing approach, thus randomly splitting" 70/10/20 | 못 읽음 | 98.4 | 11.8 mW 측정, 4.32 ms, 50.98 µJ | Lattice iCE40-UltraPlus | [read in full text] (부분: [저자 원고](https://zenodo.org/records/13871255)) |
| [Xing et al. 2022, Electronics](https://doi.org/10.3390/electronics11121889) | CNN 변환, channel attention | amplitude 또는 binary | "randomly selected data" (BEL review) | 못 읽음 | 98.2 또는 98.26, F1 89.0 또는 89.44. Rana Table 3은 inter-patient 92.07, N F1 95.7, V F1 72.5로 인용 | 346.33 µJ, 1.37 ms ([BEL review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362428/)). [Choi review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362400/)는 324.51, 1.32 ms | FPGA | [secondary summary] |
| [Yan et al. 2024, SparrowSNN, arXiv](https://arxiv.org/abs/2406.06543) | hybrid ANN-SNN, SSF 활성화 | 180 sample beat | beat 60/20/20, train에 SMOTE. 기록 4개 제외 | T = 3-31 | T = 31에서 98.61. 본문 한 곳은 "validation accuracy"라고 적습니다 | 11.76 nJ, 표준 셀 합성 기반 | ASIC 설계 | [read in full text] (부분: 데이터 절, 표 II-III) |
| [Chu et al. 2022, TBioCAS](https://doi.org/10.1109/tbcas.2022.3189364) | spike-driven 처리기, STBP | level crossing | 초록에 없음 | 초록에 없음 | "98.22%", 5 category | "0.75 μJ/classification" | FPGA 검증, 40 nm ASIC 평가 | [abstract only] |
| [Buettner & George 2021, ISVLSI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362428/) | CNN 변환, 6-bit | rate | review의 "Generalized" | 못 읽음 | 97.8, F1 87.9 | 7.22 ms, 0.42 mJ, 58 mW | Intel Loihi | [secondary summary] |
| [Bauer, Muir, Indiveri 2019, TBioCAS](https://arxiv.org/abs/1911.05521) | reservoir, 이상 탐지(binary trigger) | 비동기 event | 찾지 못함 | 연속 | overall Se 92.11, Spe 97.6, PPV 65.42. [Survey](https://arxiv.org/abs/2506.01737)의 "91.3%"는 표에서 확인 못 함 | DYNAP-SE 516.1 µW, "experimental results measured from the chip" | 180 nm DYNAP-SE | [read in full text] (부분: 표 3과 전력 표) |
| [Corradi et al. 2019, IJCNN](https://doi.org/10.1109/ijcnn.2019.8852279) | recurrent SNN과 LIF readout | event 압축 | 초록에 없음 | 연속 | 17 패턴, "overall classification accuracy of 95%" | 숫자 없음 | 혼합 신호 처리기 | [abstract only] |
| [MSPAN (Jiang et al. 2021), Frontiers](https://doi.org/10.3389/fnins.2021.761127) | DiSNN | Poisson | 초록에 없음 | 못 읽음 | 4 class 93.6, 시스템 92.25 | "0.178 μJ per heartbeat", "By evaluation" | memristor CIM(시뮬레이션) | [abstract only] |
| [Gerber et al. 2022, arXiv](https://arxiv.org/abs/2209.01266) | delay chain | event | 찾지 못함 | 연속 | 5 label 82.0, raw 분류 85.6 | 찾지 못함 | DYNAP-SE | [read in full text] (부분) |
| [Banerjee, Dey, Pal 2022](https://mlanthology.org/neuripsw/2022/banerjee2022neuripsw-snn) / [특허](https://patents.google.com/patent/US20240176987A1/en) | 1층 STDP feed-forward | peak-based | 13K / 3.1K, 환자 분리 언급 못 찾음 | 못 읽음 | 4 class 94.3 | 33 SOP/ts | 없음(추정) | [read in full text] (부분: 특허 표 2, 3) |
| [Range Encoding, AICAS 2025](https://doi.org/10.1109/aicas64808.2025.11173111) | SNN | range | 초록에 없음 | T = 2 | 98.15 | "0.07µJ", 55 nm 합성 | 합성 | [abstract only] |
| [Diware et al. 2025, DATE](https://doi.org/10.23919/date64628.2025.10993026) | SoC | adaptive multi-threshold | 초록에 없음 | 못 읽음 | "high accuracy" | "up to 5.1x" | TSMC 40 nm | [abstract only] |
| [SPTU-Lite, ICASSP 2025](https://doi.org/10.1109/icassp49660.2025.10889748) | U자 CNN-Transformer, spiking FC | 못 읽음 | 초록에 없음 | 못 읽음 | 99.62 | 없음, parameter -95%만 | 없음 | [abstract only] |
| [Spiking-ResNet, GCCE 2025](https://doi.org/10.1109/gcce65946.2025.11275376) | CSNN | sparse | 초록에 없음 | 못 읽음 | F1 94.25 | 없음 | 없음 | [abstract only] |
| [Kovacs & Samiee 2022, CinC](https://doi.org/10.22489/cinc.2022.049) | VP + SNN, 242 parameter | spike trail | "balanced ... approximately 60/40" | 못 읽음 | N vs VEB 97.16 | 없음 | 없음 | [abstract only] |
| Mao 2022 TBioCAS, Fan, Liu ISSCC 2022 | SNN 처리기 | dual-binary, LC-ADC | 못 읽음 | 못 읽음 | 98.6, 96.8, 90.5 | 0.3 µJ, 0.16, 0.53 pJ/spike | ASIC | [secondary summary] [SparrowSNN 표 III](https://arxiv.org/abs/2406.06543), [Scrugli 2026 표 VII](https://zenodo.org/records/18887975) |

- [Scrugli 2024 표](https://zenodo.org/records/13871255)의 각주는 Amirshahi를 inter-patient로 분류합니다. 그러나 Amirshahi 본문과 [BEL review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362428/)는 patient-specific으로 적습니다. 환자 본인의 첫 5분을 학습에 쓰므로 inter-patient가 아닙니다.
- [Survey](https://arxiv.org/abs/2506.01737)는 Corradi를 "17-class MIT-BIH Atrial Fibrillation dataset"이라고 적습니다. [Corradi 초록](https://doi.org/10.1109/ijcnn.2019.8852279)에는 MIT/BIH "Arrhythmia Database"로 되어 있습니다.

#### 이 split이 약하다고 보는 근거
- "overall, intra-patient approach should be highly discouraged"이고, inter-patient는 "avoids the information leakage issue"라고 적습니다. Kachuee 모델을 다시 구현하면 intra 93.1에서 inter 81.2로 떨어지고, inter에서 S class Pre/Rec가 0.0/0.0, F가 1.0/1.3입니다. Kachuee가 test set에서도 불균형을 보정했다고 지적합니다. [read in full text] (부분) — [Cao et al., arXiv 2206.14200](https://arxiv.org/abs/2206.14200)
- "This evaluation paradigm is not highly reliable in the real world", "the heartbeats from the same patients may be used for both the training and the testing". AAMI 절차로 "four paced recordings are not used". [read in full text] (도구 요약과 XML 부분) — [Wang et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8181174/)
- "intra-patient protocols typically yield higher accuracy than the inter-patient AAMI protocol due to shared patient morphology". 이 논문은 intra를 "not a methodological shortcut"이라고 방어합니다. 착용 기기를 사용자별로 보정하는 시나리오를 전제로, 환자마다 class별로 80/20을 나눕니다. [read in full text] (부분) — [Banjo & Ghoraani 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306364/)
- "robust generalization across unseen patients remains an unsolved challenge for ultra-low-power SNN-based ECG classifiers". [read in full text] (부분) — [Scrugli 2026 원고](https://zenodo.org/records/18887975)
- 환자 분리 기준을 평가 항목으로 삼은 review가 있고, "few properly consider patient-independent partitioning and hardware limitations"라고 적습니다. [abstract only] — [arXiv 2503.07276](https://arxiv.org/abs/2503.07276)
- SNN 논문도 leakage를 직접 언급합니다. "We did not use random sample level splitting because this may lead to data leakage". [read in full text] (부분) — [Kolhar 2025, Sci Rep](https://pmc.ncbi.nlm.nih.gov/articles/PMC12612072/)

#### 우리 데이터의 실제 상태(로컬에서 읽기만 함)
- `5class_mitbih_train_7_187.csv`는 76,612행, `5class_mitbih_test_3_187.csv`는 32,833행, 합계 109,445행입니다. 열은 188개(187 sample + label)이고 환자나 기록 ID 열은 없습니다. class 합계는 N 90,588, S 2,779, V 7,236, F 803, Q 8,039입니다. 내 계산으로 train/test 비율은 class마다 7:3입니다. — 로컬 파일 `/home/yschoi/CLS_spiking_transformer/data/`
- 공개 Kaggle 판은 "109446, with 5 categories and a sampling frequency of 125Hz", train 87,554, test 21,892입니다. test support는 N 18,118, S 556, V 1,448, F 162, Q 1,608입니다. [read in full text] (부분) — [ECGformer](https://arxiv.org/abs/2401.05434)
- Q class의 크기 차이: AAMI에서 paced 기록을 빼면 DS1 8, DS2 7, 합 15 beat입니다. 80/20 표에서는 Q가 test 1,608, 전체 8,043입니다. [read in full text] (부분) — [Cao et al.](https://arxiv.org/abs/2206.14200)
- [Kachuee 2018](https://arxiv.org/abs/1805.00794)은 lead II를 125 Hz로 줄이고, 10 s 창을 나눈 뒤 R-R 길이의 1.2배를 잘라 zero padding합니다. "we have not used any form of filtering"이라고 적습니다. test는 4,079 beat(class당 약 819)이고, 환자 분리 여부는 쓰여 있지 않습니다. [read in full text] (도구 요약)
- 우리 결과: macro-F1 91.99 ± 0.10, accuracy 98.56, class F1 N 99.24, S 85.94, V 96.31, F 79.49, Q 98.95, 9.44 µJ입니다. 같은 표의 Spikformer식은 tuned 91.78, default 92.22입니다. — 로컬 `NSMT/cinc_mixer_v1/results/compare-nsmt-20261105/summary.md`
- 프로젝트 기록은 MIT-BIH 절차를 "검증 분할 없음, 마지막 epoch 가중치"로 적습니다. — 로컬 `docs/PROJECT_LOG.md`(2026-11-01 항목)

### Inferences
- 우리 Q 8,039는 paced beat가 남아 있다는 뜻으로 보입니다. Kaggle 판 총량과 class 비율이 일치하므로 우리 CSV는 그 corpus를 7:3으로 다시 나눈 것으로 보이지만, 파일 출처 기록을 직접 확인하지는 못했습니다.
- Q를 뺀 4 class 평균은 90.25입니다(내 계산: (99.24 + 85.94 + 96.31 + 79.49) / 4). paced beat가 없는 DS1/DS2와 숫자를 나란히 놓으면 안 됩니다. accuracy도 항상 N이라고 답하면 82.8%(내 계산: 90,588 / 109,445)이므로 정보가 적습니다.
- 믿을 만한 비교는 세 단계입니다.
  1. 같은 조건 안의 짝 비교: 같은 split, seed, epoch에서 Spikformer식과 NSMT를 비교합니다. 이미 있는 비교이고 차이가 없는 쪽(91.99 vs 91.78, 92.22)입니다.
  2. 같은 corpus 계열의 외부 값: [ECGformer](https://arxiv.org/abs/2401.05434)(80/20, 98%, macro-F1 0.86, F 0.66, S 0.73)입니다. split 비율이 달라 엄밀히 비교 가능하지는 않습니다.
  3. random beat split의 SNN과는 class별 F1만 나란히 놓습니다: [Scrugli 2026](https://zenodo.org/records/18887975) N 99.19, S 80.42, V 95.76과 우리 99.24, 85.94, 96.31입니다. paced 포함 여부와 split이 달라 비교 가능하다고 쓰면 안 됩니다.
- MIT-BIH를 논문에 넣으려면 DS1/DS2 재평가가 사실상 필요해 보입니다. 2024-2026년의 Rana, Galvis-Chacón, Scrugli, Banjo가 모두 두 평가를 함께 싣습니다. 이 경우 beat를 raw에서 다시 자르고, 보고는 S와 V F1 중심으로 하며, Q는 macro에서 뺍니다.
- 예상은 비관적입니다. 형태만 보는 모델은 S class에서 무너지기 쉽습니다(Scrugli S sensitivity 0.22%, Cao 재구현 S 0.0/0.0). [Banjo](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306364/)와 Scrugli는 RR 정보로 상당 부분 회복합니다. 우리 11 token 단일 beat 입력에는 RR feature가 따로 없습니다.
- BioSpike-Net의 patient-disjoint macro-F1 95.8은 다른 환자 분리 결과와 차이가 너무 큽니다. 2 s window 단위 평가이고 저자도 "window-level"을 한계로 적습니다. 기준값으로 쓰지 않는 편이 안전합니다.
- 에너지는 우리 강점이 아닙니다. 9.44 µJ(45 nm 연산 수 추정)는 ASIC 0.3-0.75 µJ보다 크고 FPGA 측정 36.86 µJ보다 작지만, 계산 방식이 달라 어느 쪽과도 비교 가능하지 않습니다.

### Gaps
- Yan 2021, Xing 2022, Chu 2022, Buettner 2021, Mao 2022의 전문을 못 읽었습니다(ScienceDirect, MDPI 403, IEEE 유료). split과 T는 review 표기나 초록으로만 압니다. BEL review의 "Generalized"는 정의가 없고, 본문 설명으로 보면 "randomly selected data"입니다.
- 우리 CSV의 생성 과정(Kaggle 판에서 왔는지, paced 기록을 뺐는지)은 파일만으로 확정하지 못했습니다.
- OpenAlex는 검색 3건과 DOI 조회 22건 후 429를 돌려줬고, Semantic Scholar는 처음부터 429였습니다. 이후는 arXiv API, Europe PMC, 웹 검색으로 찾았으므로 SNN 표가 전수라고 할 수 없습니다.

## Q2. CinC 2017이나 긴 AF 기록의 SNN 논문, 다른 긴 biosignal의 SNN 숫자, 믿을 만한 두 번째 긴 sequence 과제

### Takeaway
- CinC 2017 SNN 결과는 있습니다. 찾은 것은 세 그룹입니다: Feng 2022(acc 84.4, F1 70.7), Rana & Kim 2024(acc 85.8, F1 70.4), Banerjee 2022/2024(4 class acc 77-78.5)입니다. 우리는 accuracy 73.97, macro-F1 64.35, NAO 평균 68.52입니다. split이 달라 비교 가능하다고 할 수는 없지만, 이 숫자만 보면 "CinC 2017 SNN 최고"는 주장할 수 없고 오히려 낮습니다.
- CinC 2017에 window당 µJ를 보고한 SNN은 찾지 못했습니다. 우리가 내세울 수 있는 빈칸이 있다면 정확도가 아니라 이쪽입니다.
- 두 번째 과제로는 Sleep-EDF가 가장 낫습니다. 30 s × 100 Hz = 3,000 sample, 단일 채널, subject 분리 CV가 표준입니다. SNN 값(74.2%, macro-F1 70.5)이 non-spiking(약 77-82%)보다 낮아 개선 여지도 보입니다. PPG-DaLiA에서는 SNN 숫자를 찾지 못했습니다.

### Cited Findings

#### CinC 2017의 SNN
| 논문 | 모델, encoding, T | split | 점수(원문) | 에너지 | 우리와 비교 가능? | 근거 |
|---|---|---|---|---|---|---|
| [Feng, Geng, Chu, Fu, Hong 2022, BSPC 77:103749](https://doi.org/10.1016/j.bspc.2022.103749) | 14층 CNN을 변환, Poisson, 1000 time steps | "randomly selected data", 기록 전체(9-60 s), 4 class | [BEL review 표](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362428/): "Acc F1 ROC-AUC PR-AUC Ppr Sen 84.4, 70.7, 94.5 75.6, 76.0, 67.2". [Choi review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362400/): "Proposed: 84.4%, ANN (ReLU): 84.1%", ROC-AUC 0.935 vs 0.890(두 review의 AUC가 다름). [Survey](https://arxiv.org/abs/2506.01737): 84.41%, ANN보다 0.35% 위, 1000 time steps | Choi review에서 Not reported | 아니오. F1 정의, fold, label 판을 못 읽음 | [secondary summary], 공개 초록 일부는 [BL catalog](https://eld.bl.uk/catalog/100159146571_0x000001) |
| [Rana & Kim 2024, Sensors](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175061/) | 12층 ANN을 6층 Att-SNN으로 변환, 200 epoch | "70:30" random. 기록 길이 처리 방식은 안 적음 | Proposed: Acc 85.8, F1 70.4, Pre 75.7. 12-layer ANN: 84.1, 70.1, 73.4. Normal F1 92.0, Noise F1 50.0 | CinC 전용 숫자 없음(278 mW는 MIT-BIH 표) | 아니오. split 비율과 평균 방식이 불명 | [read in full text] (부분) |
| [Banerjee, Dey, Pal 2022](https://mlanthology.org/neuripsw/2022/banerjee2022neuripsw-snn), [2024 IJCNN](https://doi.org/10.1109/ijcnn60899.2024.10650023), [특허 US20240176987](https://patents.google.com/patent/US20240176987A1/en) | reservoir와 feed-forward, Gaussian, delta, peak-based | train 6,822, test 1,705, length "2k to 9k", 4 class | 특허 표 2 accuracy: Gaussian+reservoir 69, delta+reservoir 72, peak+reservoir 78.5, peak+feed-forward 77. "normal and abnormal AF classes only"일 때 "92% accuracy" | 28-36 SOP/ts(추정) | 아니오. accuracy만 있고 F1 없음 | [read in full text] (부분: 특허 표 2, 3), 논문은 [abstract only] |

- [Survey](https://arxiv.org/abs/2506.01737)는 "92% accuracy on the PhysioNet 2017 dataset using a reservoir-based SNN"이라고 적습니다. 그러나 [특허 본문](https://patents.google.com/patent/US20240176987A1/en)은 92%를 feed-forward SNN의 normal 대 AF 2 class 결과로 적습니다. 4 class 값이 아닙니다.
- 특허 표는 SOA 83을 Challenge F1에서 가져오고 자기 값은 accuracy로 적어, 지표가 섞여 있습니다.

#### 우리 값(로컬 raw.txt, 읽기만 함)
- 조건 `spk_lif_muxlin_or`, fold 0, seed 7, 42, 2026: score 0.675, 0.695, 0.686. accuracy 72.63, 74.79, 74.50(내 계산 평균 73.97). AF sensitivity 0.603, 0.649, 0.589, AF specificity 0.972, 0.972, 0.977. n = 1,706 기록. F1 N 83.51, AF 65.16, Other 56.88, Noisy 51.85, macro 64.35. 검증 score는 72.46입니다. — 로컬 `NSMT/cinc_mixer_v1/results/flif-v1-rhythm-spk-mux-cinc/{raw.txt,summary.md}`
- 에너지 129.4 µJ는 window당입니다. seed 42의 synapse 110.57 µJ는 내 계산으로 0.9 pJ × 47.507 M AC + 4.6 pJ × 14.7425 M MAC와 정확히 맞고, 그 61%(67.82 µJ)가 MAC입니다. 나머지는 neuron 항 약 19 µJ입니다. 평가는 record당 여러 window를 평균하므로 record당 에너지는 이보다 큽니다. — 같은 raw.txt, [rhythm_eval.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/rhythm_eval.py)

#### 다른 긴 AF 기록과 ECG
- MIT-BIH AF Database: 70/10/20으로 91.5 accuracy, 84.45 sensitivity, 89.92 precision, 94.42 specificity, F1 86.87입니다. "no prior SNN-based studies have reported results on this benchmark"라고 적습니다. [read in full text] (부분) — [Scrugli 2026 원고](https://zenodo.org/records/18887975)
- Chapman-Shaoxing와 Ningbo(12-lead, 4 class): patient-wise 80/10/10, rate coding T = 50, 표의 accuracy 94.4, "Event-driven: ~90% fewer 2nd-layer ops", 640,644 parameter입니다. 측정 에너지는 없습니다. [read in full text] (부분) — [Kolhar 2025, Sci Rep](https://pmc.ncbi.nlm.nih.gov/articles/PMC12612072/)
- PPG AF 탐지: 2 s window, patient-disjoint 5-fold, ROC-AUC 0.982, average precision 0.976입니다. [read in full text] (부분) — [BioSpike-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC13509974/)
- QTDB ECG waveform: 2 채널, seq length 1301, 6 waveform을 매 step 분류하는 online 과제입니다. adaptive SRNN 85.9 vs Bi-LSTM 80.8입니다. 전체 기록을 하나로 분류하는 과제는 아닙니다. [read in full text] (부분) — [Yin et al. 2021](https://arxiv.org/abs/2103.12593)

#### 다른 긴 biosignal
| 과제 | 입력 길이 | split | SNN 숫자 | non-spiking 숫자 | 근거 |
|---|---|---|---|---|---|
| Sleep-EDF Expanded SC (78명) | 단일 EEG, 30 s, 100 Hz = 3,000 sample | subject-independent 5-fold | [NeuroSleep](https://arxiv.org/abs/2602.15888): Acc 74.2, macro-F1 70.5, κ 65.6, 0.932 M parameter, FLOPs 0.770 G, effective OPs 0.357 G. 시간축으로 푼 Spiking CNN 기준선 44.5. Physiological Measurement 심사 중, local attention 사용 | [DREAM 표](https://arxiv.org/abs/2312.03196): DREAM 81.94 ± 2.18 / 74.79, AttnSleep 79.09 ± 2.67 / 73.59, DeepSleepNet 76.58 ± 4.70 / 72.65. fold 방식이 같은지 확인 필요 | [read in full text] (부분) |
| DEAP EEG | 32 채널, 2 s, 256 sample | 읽은 범위에 없음 | [SparrowSNN](https://arxiv.org/abs/2406.06543): arousal 85.31, 17.87 nJ. [SpikeWavformer](https://pmc.ncbi.nlm.nih.gov/articles/PMC12572938/)는 DEAP와 KUL(숫자는 안 옮김) | 안 찾음 | [read in full text] (부분) |
| CHB-MIT 발작 | 다채널, 길이는 표에 없음 | cross-patient(제목) | AUC 96.64, RACC 90.80, 0.14 mJ vs 7.2 mJ, [Zhang 2023](https://doi.org/10.3389/fnins.2023.1303564) | 표의 ANN 90.95 / 81.53 | [secondary summary] [Choi review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362400/) |
| NinaPro DB5 sEMG | 표에 없음. 다른 sEMG SNN은 200 ms window | 표에 없음 | Vitale 2022 74% vs TCN 69.2%, Loihi 5.7 ms, 41 mW. DB2 55.92 | TCN 69.2 | [secondary summary] [Choi review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362400/), 200 ms는 [Sava et al.](https://arxiv.org/abs/2309.16425) |
| PPG-DaLiA 심박 회귀 | 8 s segment, 2 s 간격 | LOSO, 15명 | 찾지 못함. [npj Unconv. Comput. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11964916/)는 DYNAP-SE의 HR 추정이지만 데이터가 BIDMC, WristPPG | [Reiss 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6679242/) CNN ensemble, S5 세션 18.51 bpm. [PULSE](https://arxiv.org/abs/2210.11415) 4.44 / 4.03(검색 요약) | [read in full text] (부분), PULSE는 [secondary summary] |
| BIDMC PPG 호흡률 | 16 s, 125 Hz, 53 기록 | 읽은 범위에 없음 | [Yang et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11207339/) MAE 1.37 ± 0.04, E_AC·E_MAC 공식 추정 | CNN-RNN, VGG-8 | [read in full text] (부분) |
| PTB-XL | 12-lead, 10 s | 공식 fold 8/1/1 | 전용 SNN 벤치마크 못 찾음. [BioSpike-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC13509974/)는 3 class 매핑으로 2 s window | [XResNet macro AUC 약 0.929](https://arxiv.org/abs/2305.17043) | [secondary summary] |
| rPPG(카메라) | 영상 | 여러 데이터 | [Spiking-PhysFormer, Neural Networks 2025](https://arxiv.org/abs/2402.04798), 45 nm 추정. attention 사용 | PhysFormer | [read in full text] (부분) |

- 웨어러블 벤치마크: [NeuroBench](https://arxiv.org/abs/2304.04640)는 일반 프레임워크이고, 찾아본 요약에서는 ECG나 biosignal 과제를 확인하지 못했습니다. [secondary summary]
- [Survey](https://arxiv.org/abs/2506.01737) 전문에는 PPG, photoplethysmography, sleep이라는 단어가 나오지 않습니다(내 문자열 검색, 0건). 이 분야가 SNN에서 비어 있다는 간접 신호일 수 있지만, survey의 수집 범위 때문일 수도 있습니다. [read in full text]

### Inferences
- [Rana & Kim](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175061/)의 70.4가 4 class 산술 평균이라면, N 92.0과 Noisy 50.0에서 NAO 평균은 약 77.2입니다(내 계산, 가정 포함). 우리 68.52보다 약 8.7 위입니다. Feng은 Ppr 76.0, Sen 67.2에서 F1 약 71.3이 나와 70.7과 비슷하므로, class 평균값으로 보입니다.
- 세 그룹 모두 기록 단위 random 분할로 보이므로 우리와 같은 계열입니다. 다만 fold, label 판, 평균 방식을 확인하지 못했으므로 "비교 가능"이라고 쓰면 안 됩니다. 논문에는 "reported values under different random record splits"처럼 적는 것이 안전합니다.
- 차별점은 정확도가 아니라 세 가지 조합입니다: Spikformer식과 같은 조건에서 비교한 결과, attention-free 구조, window당 에너지 수치. Feng의 1000 time steps와 대비는 할 수 있지만, 정확도가 낮은 상태에서는 약합니다.
- 우리 MAC 비중 61%는 circuits 쪽 reviewer가 물을 만한 항목입니다. spike 구동이 아닌 연산이 대부분의 synapse 에너지를 쓴다는 뜻이기 때문입니다.
- 두 번째 과제 추천 순서:
  1. Sleep-EDF: 길이가 같은 자릿수입니다. 같은 patch 규칙이면 약 94 token입니다(내 계산: (3000 - 64) / 32 + 2). 단일 채널이고 subject 분리가 기본이라 MIT-BIH 같은 leakage 논쟁이 없습니다. SNN 기준선이 [NeuroSleep](https://arxiv.org/abs/2602.15888) 74.2로 약하고, 그 모델은 attention을 씁니다.
  2. MIT-BIH AFDB: SNN 값이 Scrugli 하나뿐이지만 분할이 환자 분리가 아니고 표준 절차가 약합니다.
  3. QTDB: T = 길이의 SNN 기준이 있으나 매 step 분류라 token-mean head가 맞지 않습니다.
  4. PPG-DaLiA: SNN 비교값이 없고, 프로젝트 기록상 10-29부터 보류 중입니다.
  다채널인 DEAP, CHB-MIT, PTB-XL은 1 채널 구조를 바꿔야 하고, sEMG는 window가 짧아 "긴 sequence" 주장에 맞지 않습니다.
- 사용자는 11-04에 CinC를 "기억과 T = N의 작은 근거"로만 쓰고 주 결과로 삼지 않기로 정했습니다(프로젝트 기억). 이 논문 구상은 그 결정을 바꾸는 것이므로 명시적인 판단이 필요합니다.

### Gaps
- Feng 2022 전문을 못 읽었습니다. F1 평균 방식, fold, label 판, class별 F1은 알 수 없습니다.
- Rana & Kim은 CinC 기록의 분할 단위와 길이 처리를 쓰지 않았습니다.
- Banerjee 표는 특허 본문에서 읽었습니다. NeurIPS workshop PDF(OpenReview)는 이 서버에서 막혀 같은 표인지 대조하지 못했으나, MIT-BIH 94.3과 33 SOP/ts가 [BEL review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362428/)와 일치합니다.
- CinC에 환자 ID가 있는지는 읽은 자료에서 확인하지 못했습니다. [Clifford 2017](https://www.cinc.org/archives/2017/pdf/065-469.pdf)과 공개 파일 구성에는 보이지 않습니다.
- [Verma 2025, MIMIC-IV](https://doi.org/10.1007/s00521-025-11585-9), [Hybrid Bi-LSTM-SNN(JTEC 2026)](https://doi.org/10.54554/jtec.2026.18.03.002), [BrainScaleS-2 AF 시연](https://arxiv.org/abs/2103.15960)은 제목만 보았습니다.
- PPG, sleep, EEG, sEMG는 전수 조사가 아닙니다.

## Q3. CinC 2017 non-spiking 최고 수준과 우리 점수 정의: 68.52가 얼마나 먼가

### Takeaway
- 우리 점수의 공식은 공식 Challenge 점수와 같습니다: (F1 N + F1 AF + F1 Other) / 3, Noisy 제외. 다만 test set이 다릅니다. Challenge 값은 hidden 3,658 기록이고, 우리는 공개 8,528 기록의 stratified 5-fold 중 fold 0(1,706 기록)입니다. 따라서 어떤 Challenge 점수와도 비교 가능하지 않습니다.
- 공개 데이터 CV 값끼리 보면 우승팀 82-87, 단일 end-to-end 신경망은 대체로 63-79입니다(Hong의 개선 ResNet 83.1은 예외). 68.52는 증강 없는 ResNet 하나(63.1)보다는 높고, 증강 없는 CNN과 CRNN(75.8, 74.6)보다 6-7 낮으며, 우승팀보다 13-18 낮습니다.
- 격차의 대부분은 AF와 Other입니다. Noisy는 상위권과 비슷합니다.

### Cited Findings

#### 우리 평가 방법(코드 읽기 전용)
- [rhythm_eval.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/rhythm_eval.py): `m['score'] = float(f[:3].mean())`, 즉 N, AF, Other F1 평균입니다. 4 class macro-F1, accuracy, AF sensitivity와 specificity도 함께 냅니다. softmax를 record별로 평균한 뒤 argmax하므로, 평가 단위는 window가 아니라 record입니다.
- [rhythm_data.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/rhythm_data.py): 8,528 AliveCor 기록, REFERENCE-v3, label N/A/O/~ → 0/1/2/3입니다. 300 Hz를 **125 Hz**로 resample(polyphase 5/12)하고 0.5-40 Hz 4차 Butterworth zero-phase, record별 z-score를 적용합니다. record 단위 `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`에서 test = fold, 나머지의 10%를 stratified validation으로 씁니다. 학습은 epoch마다 record당 random 30 s crop(30 s보다 짧으면 반복해 채움)이고, 평가는 30 s window를 15 s 간격으로 두며 마지막 window는 record 끝에 맞춥니다.
- [config.py](/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1/config.py)의 기본값은 `win_s` 30, `hop_s` 15, `fs_target` 125입니다.
- **조사 요청 문구의 정정:** "3,750 samples at the challenge rate"는 맞지 않습니다. 3,750 sample은 125 Hz에서 30 s입니다. Challenge 원본은 300 Hz입니다. 내 계산으로 patch 64, stride 32에서 (3750 - 64) / 32 = 116.2이므로 116 + end padding 1 = 117 token이 맞습니다.
- 로컬 REFERENCE-v3 count는 N 5,076, A 758, O 2,415, ~ 279로, Clifford Table 2의 training V3와 일치합니다.
- 학습 쪽 증강은 random crop뿐입니다(코드 기준).

#### 공식 점수 정의의 불일치와 해소
- PhysioNet 페이지는 "F1 = (F1n + F1a + F1o + F1p)/4"와 "an average of the four F1 values"라고 적습니다. [read in full text] (부분) — [PhysioNet Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/)
- Clifford 요약 논문의 식 (1)은 "F1 = (F1n + F1a + F1o) / 3"입니다. [read in full text] — [Clifford et al. 2017](https://www.cinc.org/archives/2017/pdf/065-469.pdf)
- 공식 결과 파일로 확인했습니다. Teijeiro는 test 0.903, 0.8547, 0.7366, 0.5622입니다. 3 class 평균은 0.8314로 Clifford의 0.831과 맞고, 4 class 평균 0.7641은 맞지 않습니다(내 계산). 최종 점수는 3 class 평균입니다. — [results_all_F1_scores_for_each_classification_type.csv](https://physionet.org/files/challenge-2017/1.0.0/results_all_F1_scores_for_each_classification_type.csv)
- [Teijeiro](https://www.cinc.org/archives/2017/pdf/166-054.pdf)도 "the mean F1 measure of the N, A, and O classes"라고 적습니다. [read in full text] (부분)

#### Hidden test가 공개 데이터와 다른 점
- 총 12,186 기록: public 8,528, hidden 3,658입니다. 300 Hz, 0.5-40 Hz, 9-61 s입니다. [read in full text] — [Clifford et al. 2017](https://www.cinc.org/archives/2017/pdf/065-469.pdf)
- 라벨 재검토 후 분포가 달라졌습니다. V3 test는 N 66.6%, AF 7.8%, Other 18.7%, Noisy 6.9%입니다. V3 training은 N 59.5%, AF 8.9%, Other 28.3%, Noisy 3.3%입니다. "Only test data were updated with the new labels", "a very few training recordings were also updated"라고 적습니다. — 같은 곳
- 재라벨 1,129 기록의 κ = 0.245로, 전문가 일치가 낮습니다. "the training data could be improved"라고 적습니다. — 같은 곳
- 공식 진행 기간에 쓴 라벨은 V2이고, V3는 최종 평가용입니다. — 같은 곳

#### Challenge와 대표 non-spiking 값
| 출처 | 모델 | 점수 정의 | split, 라벨 | 값(원문) | 우리와 비교 가능? | 근거 |
|---|---|---|---|---|---|---|
| [Teijeiro et al.](https://www.cinc.org/archives/2017/pdf/166-054.pdf) | abductive 해석 feature, XGBoost, RNN, LDA stacker | NAO 평균 | 8-fold CV, hidden | CV: XGBoost 0.83, RNN 0.83, LDA-stacker 0.85. hidden [0.831](https://www.cinc.org/archives/2017/pdf/065-469.pdf) | 아니오 | [read in full text] (부분: 표 1) |
| [Datta et al.](https://www.cinc.org/archives/2017/pdf/173-154.pdf) | cascaded binary classifier | NAO 평균 | 5-fold CV, hidden | CV: 0.9095, 0.7978, 0.7719, FT 0.8264, std 0.0026. hidden: 0.92, 0.82, 0.75, 0.83. 초록의 "0.92, 0.86 and 0.74"는 표와 다름 | 아니오 | [read in full text] (부분) |
| [Zabihi et al.](https://www.cinc.org/archives/2017/pdf/069-336.pdf) | random forest | NAO 평균 | 80%에서 10-fold, hidden | CV: F1n 90.49, F1a 79.43, F1o 75.64, F1p 61.11, F1 81.85 ± 2.57. hidden 82.6 | 아니오 | [read in full text] (부분) |
| [Hong et al. ENCASE](https://www.cinc.org/archives/2017/pdf/178-245.pdf) | expert + center wave + deep feature, XGBoost 앙상블 | F1 NAO | 반복 CV | E+C+D F1 NAO 0.8655. XGBoost E 0.8077. [Rajpurkar 식] ResNet 0.7792, ResNet Improved 0.8309. Online 0.83-0.84 | 아니오 | [read in full text] (부분: 표 4) |
| [Zihlmann et al.](https://arxiv.org/abs/1710.06122) | CNN, CRNN(spectrogram, LSTM) | overall = NAO 평균(내 계산으로 표와 일치) | stratified 5-fold, hidden | 증강: CNN 79.0, CRNN 79.2(N 88.8, A 76.4, O 72.6, ~ 64.5, acc 82.3). 증강 없음: CNN 75.8, CRNN 74.6(acc 79.2). hidden "82.1%" | 아니오 | [read in full text] (부분: 표 1, 2) |
| [Andreotti et al.](https://www.cinc.org/archives/2017/pdf/360-239.pdf) | feature 기반, ResNet(Rajpurkar 모델) | 표의 validation Total은 4 class 평균, test Total은 NAO 평균(내 계산으로 표와 일치) | 5-fold, hidden 부분집합 | ResNet 34L 30 s AUG: 90.2, 65.7, 69.8, 64.0, Total 72.4(NAO 75.2, 내 계산). ResNet 16CF 60 s TRAIN: 82.6, 46.6, 60.0, 60.2, Total 62.4(NAO 63.1), test 79. 같은 모델 AUG: Total 72.1(NAO 74.3), test 83. 최종 제출 79. feature 기반 72.0(NAO 79.2), test 79 | 아니오 | [read in full text] (부분: 표 1) |
| [Hannun et al. 2019, Nat Med](https://pmc.ncbi.nlm.nih.gov/articles/PMC6784839/) | 34층 ResNet, 200 Hz raw | class 평균 F1 | public 8,528(10%는 early stopping), hidden 3,658 | "class average F1 of 0.83". class별 값은 Supplementary Table 7이라 본문에 없음 | 아니오 | [read in full text] (도구 요약, 본문) |
| [Rajpurkar et al. 2017](https://arxiv.org/abs/1707.01836) | 34층 CNN | 자체 corpus | 약 30,000 환자, test 336 | CinC 2017 값이 아닙니다 | 아니오 | [read in full text] (부분) |
| [Clifford 2017 Table 4](https://www.cinc.org/archives/2017/pdf/065-469.pdf) | 상위 8팀과 투표 | NAO 평균 | hidden, V3 | 0.831, 0.829, 0.826, 0.825, 0.822, 0.821, 0.821, 0.818. 전체 75 투표 0.855, LASSO 0.858, LASSO+ 0.868 | 아니오 | [read in full text] |

- "Training" 열 0.89-0.97은 학습 데이터에 대한 재평가 값으로, Clifford는 "significant overtraining was seen even in the winning entries"라고 적습니다. — [Clifford et al. 2017](https://www.cinc.org/archives/2017/pdf/065-469.pdf)

#### 공식 결과 파일의 분포(내 계산)
- 파일의 68개 entry 기준 NAO 평균의 최고는 0.8314, 중앙값은 0.7742입니다. 53개가 0.6852 이상입니다. class별 hidden 중앙값은 N 0.893, AF 0.753, Other 0.669, Noisy 0.506이고, 상위 10개 평균은 0.908, 0.824, 0.735, 0.556입니다. — [공식 결과 CSV](https://physionet.org/files/challenge-2017/1.0.0/results_all_F1_scores_for_each_classification_type.csv)

### Inferences
- 공개 데이터 CV 값과의 차이(내 계산, 비교 가능 아님, 대략의 거리만):

| 기준(NAO, 공개 데이터 CV) | 값 | 68.52와의 차 |
|---|---:|---:|
| Hong ENCASE | 86.55 | -18.0 |
| Teijeiro stacker | 85 | -16.5 |
| Datta | 82.64 | -14.1 |
| Zabihi | 81.85 | -13.3 |
| Zihlmann CRNN(증강) | 79.2 | -10.7 |
| Andreotti feature 기반 | 79.2 | -10.7 |
| Zihlmann CNN(증강 없음) | 75.8 | -7.3 |
| Andreotti ResNet 34L(AUG) | 75.2 | -6.7 |
| Zihlmann CRNN(증강 없음) | 74.6 | -6.1 |
| Andreotti ResNet 16CF(TRAIN) | 63.1 | +5.4 |

- CV 값과 hidden 값의 관계는 한 방향이 아닙니다. 신경망 entry는 hidden이 더 높습니다(Zihlmann 79.2 → 82.1, Andreotti 74.3 → 83, 63.1 → 79). 상위 feature 기반 entry는 비슷하거나 낮습니다(Datta 82.64 → 82.9, Zabihi 81.85 → 82.6, Teijeiro 85 → 83.1, Hong 86.55 → 82.5). 원인으로는 재라벨로 달라진 class 분포, Zihlmann의 5개 앙상블, Andreotti의 부분집합 평가가 후보이지만 분리해 확인하지는 못했습니다. 어느 경우든 "68.52 vs 0.83"의 차 14.5를 그대로 거리로 쓰면 안 됩니다.
- class별 격차(hidden 값이라 참고용): AF 65.16 vs 상위 10개 82.4, 중앙값 75.3. Other 56.88 vs 73.5, 66.9. N 83.51 vs 90.8, 89.3. Noisy 51.85 vs 55.6, 50.6. Noisy만 제 자리입니다.
- 항상 N이라고만 답하면 NAO 24.9, macro 18.7입니다(내 계산, V3 공개 분포). 우리 값이 이보다는 훨씬 높습니다.
- 우리 학습 조건은 외부 기준에 비해 짧습니다. 30 epoch(Zihlmann은 500 + 100 + 이후 단계, Rana는 200), 증강은 crop뿐(Zihlmann은 증강으로 CRNN이 74.6 → 79.2), fold 1개, 앙상블 없음입니다. 격차의 일부는 구조가 아니라 학습 예산에서 왔을 수 있으므로, 외부 기준과 비교하려면 5-fold와 긴 학습이 먼저 필요합니다. Spikformer식과의 +2.16은 같은 30 epoch 조건 안의 값이라 이 문제와 별개입니다.
- 논문 내부에 같은 파이프라인의 non-spiking 1-D ResNet 또는 CNN 기준선을 두는 것이 가장 정직한 방법으로 보입니다. Rana의 ANN이 acc 84.1, F1 70.1이므로 그 기준을 넘는지 함께 보여줘야 합니다.
- 우리 Other F1이 약한 것은 Clifford가 말한 라벨 불확실성과 겹치는 지점입니다. 다만 Datta도 "a good amount of other rhythms are wrongly classified as normal"이라고 적었고 0.77을 얻었으므로, 라벨만으로는 설명되지 않습니다.

### Gaps
- 다른 팀의 CV가 V2, V3 중 어느 라벨로 계산됐는지는 대부분 안 적혀 있습니다. Andreotti의 noisy 284개는 V2 count와 같습니다.
- Hannun의 class별 F1(Supplementary Table 7)을 직접 확인하지 못했습니다.
- 2018년 이후 CinC 2017을 5-fold로 다시 평가한 non-spiking 연구는 체계적으로 찾지 못했습니다(OpenAlex 429). 위 목록은 Challenge 시기 논문 중심입니다.
- Andreotti의 83은 hidden 전체가 아니라 부분집합(TEST-DB)일 수 있습니다.

## Q4. MIT-BIH의 non-spiking과 spiking 최고 수준: inter-patient와 intra-patient

### Takeaway
- inter-patient(DS1/DS2)는 accuracy보다 S와 V의 F1로 보고합니다. non-spiking은 F1_S 84.4, F1_V 93.4(Wang 2021)와 F1_S 82.34, F1_V 95.2(Self-ONN)가 찾은 범위의 상단입니다. spiking은 accuracy 86-94, macro-F1 0.43-0.56 정도로 크게 낮습니다.
- intra-patient(beat-level)는 spiking과 non-spiking 모두 98-99 accuracy에 몰려 있어 변별력이 거의 없습니다. 우리 98.56도 이 구간입니다.

### Cited Findings

#### Inter-patient, non-spiking
| 논문 | 방법 | F1_S | F1_V | 기타 | 근거 |
|---|---|---:|---:|---|---|
| [Wang et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8181174/) | adversarial CNN + RR | 84.4 | 93.4 | pat_F1 88.9, accuracy 94.7, S 78.8 sensitivity, V 92.5 sensitivity | [read in full text] (도구 요약) |
| 같은 표: Niu 2020 | multiperspective CNN + symbol + RR | 76.5 | 89.7 | pat_F1 83.1 | 같은 표 [secondary summary] |
| 같은 표: Mondéjar-Guerra 2019 | SVM 앙상블 | 60.7 | 94.3 | pat_F1 77.5 | 같은 표 [secondary summary] |
| 같은 표: Garcia 2017 | VCG, SVM | 57.1 | 70.7 | pat_F1 63.9 | 같은 표 [secondary summary] |
| [Self-ONN (arXiv 2204.03768)](https://arxiv.org/abs/2204.03768) | operational NN + feature injection | 82.34 | 95.2 | N F1 99.15 | [read in full text] (부분: 초록) |
| [Mousavi & Afghah 2019](https://arxiv.org/abs/1812.07421) | seq2seq | - | - | S PPV 92.57, Se 88.94. V PPV 99.50, Se 99.94 | [abstract only] |
| [Cao et al.](https://arxiv.org/abs/2206.14200) | ResNet-18 + STFT | - | - | accuracy 90.8, S Pre/Rec 13.0/9.0. Kachuee 재구현 81.2 | [read in full text] (부분) |
| [Sellami & Hwang, Mondéjar-Guerra](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306364/) | | - | - | accuracy 88.34, 94.5 | [secondary summary] |

- Mousavi의 inter 값은 다른 논문보다 크게 높아, 그대로 기준으로 삼기 전에 절차 확인이 필요합니다. (제 판단이며, 전문은 읽지 않았습니다.)

#### Inter-patient, spiking
- Yan 2021 90%, Rana & Kim 2024 93.8(N F1 95.5, V F1 73.5), Galvis-Chacón 2025 86.38(multi) / 83.15(binary), Scrugli 2026 92.44(macro-F1 0.425) → 92.78(0.561, RR branch), F 제외 시 93.53(F1 0.756), Banjo 2026 89.31 ± 1.51입니다. 근거와 등급은 Q1 표에 있습니다.
- Scrugli는 "[14] reports 93.7% accuracy with 55.5% specificity", "Yan et al. [17] achieve 90% accuracy with 61.5% sensitivity and 76% specificity"라고 정리합니다. [read in full text] (부분) — [Scrugli 2026 원고](https://zenodo.org/records/18887975)

#### 같은 논문의 intra와 inter
| 논문 | intra | inter | 근거 |
|---|---|---|---|
| [Galvis-Chacón 2025](https://www.recercat.cat/handle/20.500.14342/5924) | 98.23 (multi), 98.27 (binary) | 86.38, 83.15 | [secondary summary] |
| [Scrugli 2026](https://zenodo.org/records/18887975) | 98.4, F1 88.6 | 92.44, macro-F1 0.425 | [read in full text] (부분) |
| [Banjo 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306364/) | 99.02 | 89.31 ± 1.51 | [read in full text] (부분) |
| [Cao 재구현 Kachuee](https://arxiv.org/abs/2206.14200) | 93.1 | 81.2 | [read in full text] (부분) |
| [Mousavi & Afghah](https://arxiv.org/abs/1812.07421) | S Se 100, PPV 96.46 | S Se 88.94, PPV 92.57 | [abstract only] |

#### Intra-patient, beat-level, non-spiking
- [Kachuee 2018](https://arxiv.org/abs/1805.00794): 93.4%, class당 약 819 beat의 균형 test입니다. [Cao et al.](https://arxiv.org/abs/2206.14200)은 불균형 test로 재구현해 93.1, S Pre/Rec 38.1/82.6, F 26.6/93.8이라고 적습니다.
- [ECGformer](https://arxiv.org/abs/2401.05434): 87,554 / 21,892 판에서 98%, macro precision 0.95, recall 0.80, F1 0.86. N 0.99, S 0.73, V 0.93, F 0.66, Q 0.98입니다. [read in full text] (부분)
- [SPTU-Lite](https://doi.org/10.1109/icassp49660.2025.10889748) 99.62는 spiking FC를 쓰는 hybrid이며, split은 초록에 없습니다. [abstract only]
- [Mousavi & Afghah](https://arxiv.org/abs/1812.07421): intra-patient F PPV 98.68, Se 97.40입니다. [abstract only]

### Inferences
- 우리 macro-F1 91.99는 같은 corpus의 ECGformer 0.86보다 높습니다. 특히 S 85.94와 F 79.49가 높은데, split이 70/30 대 80/20으로 다르고 ECGformer의 절차를 재현하지 않았으므로 우위라고 쓰면 안 됩니다.
- spiking inter-patient 결과의 저점이 낮아서, 우리 모델이 DS1/DS2에서 Scrugli의 0.425를 넘는 것만으로도 의미 있을 수 있습니다. 다만 아직 실행하지 않았고 결과는 알 수 없습니다.
- 공통 절차가 없으면 숫자만 나열하는 비교가 됩니다. 입력(RR 포함 여부)과 class 정의를 표에 함께 적어야 합니다.

### Gaps
- Niu, Mondéjar-Guerra, Garcia는 Wang 표를 통해서만 봤습니다. 원문 절차를 대조하지 않았습니다.
- DS1/DS2 기준의 SNN 표준 비교표는 찾지 못했습니다. 각 논문의 class 정의(4 또는 5)도 서로 다릅니다.
- 이 라운드에서 Luz 2016, de Chazal 2004 원문은 읽지 않았고, 인용을 통해 확인했습니다.

## Q5. 어떤 증거 수준과 venue가 SNN ECG 논문을 받는가. hardware 측정이 필요한가, 연산 수 추정도 받는가

### Takeaway
- 읽은 논문 범위에서 hardware 측정은 일반적으로 필수가 아닙니다. 응용 쪽 venue(BSPC, Neurocomputing, Sensors, Biomimetics, Frontiers, Neural Networks)는 연산 수 추정을 받습니다. circuits 쪽(TBioCAS, DATE, IEEE Sensors J)은 대부분 칩, 합성, FPGA 측정을 갖추지만, TBioCAS에도 시뮬레이션 추정 예가 있습니다.
- 최근 논문은 에너지보다 평가 절차의 엄격함으로 차별화합니다. 환자 분리, fold 평균 ± SD, 통계 검정, class별 F1, 추정의 한계 명시입니다. JBHI에 실린 SNN ECG 논문은 찾지 못했습니다.

### Cited Findings
| venue | 논문 | 에너지 증거 | split 수준 | 근거 |
|---|---|---|---|---|
| IEEE TBioCAS | [Chu 2022](https://doi.org/10.1109/tbcas.2022.3189364) | FPGA 검증, 40 nm ASIC 평가, 0.75 μJ | 초록에 없음 | [abstract only] |
| IEEE TBioCAS | [Bauer 2019](https://arxiv.org/abs/1911.05521) | 칩에서 측정 | 찾지 못함 | [read in full text] (부분) |
| IEEE TBioCAS | [Amirshahi 2019](https://arxiv.org/abs/1905.02954) | 시뮬레이션, 사건당 값 추정, 칩 없음 | patient-specific | [read in full text] (도구 요약) |
| BSPC | [Yan 2021](https://doi.org/10.1016/j.bspc.2020.102170) | 0.077 W, 계산법 못 읽음 | inter-patient | [secondary summary] |
| BSPC | [Feng 2022](https://doi.org/10.1016/j.bspc.2022.103749) | 보고 없음 | random | [secondary summary] |
| Neurocomputing | [Galvis-Chacón 2025](https://doi.org/10.1016/j.neucom.2025.132259) | "estimated 1.91 µJ" | 둘 다 | [secondary summary] |
| Neural Networks | [Spiking-PhysFormer 2025](https://arxiv.org/abs/2402.04798) | Horowitz 45 nm 기반 추정, transformer block 12.2배 | 데이터별 | [read in full text] (부분) |
| Neural Networks(투고본) | [Das et al.](https://arxiv.org/abs/1708.05356) | ECG 심박 추정, MAPE 보고, 에너지는 못 읽음 | 못 읽음 | [read in full text] (부분) |
| Sensors | [Rana & Kim 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175061/) | Vivado 기반 278 mW | 둘 다 | [read in full text] (부분) |
| Sensors | [Banjo 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306364/) | PYNQ-Z2 측정, Wilcoxon 검정 | intra + 탐색 inter | [read in full text] (부분) |
| Sensors | [Yang 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11207339/) | FLOPs × E_MAC, E_AC 공식 | 절차는 못 읽음 | [read in full text] (부분) |
| Biomimetics | [BioSpike-Net 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13509974/) | 45 nm 추정, 한계 명시, paired bootstrap, Holm-Bonferroni | patient-disjoint | [read in full text] (부분) |
| Electronics | [Xing 2022](https://doi.org/10.3390/electronics11121889) | FPGA | random | [secondary summary] |
| Sci Rep | [Kolhar 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12612072/) | 에너지 값 없이 "~90% fewer 2nd-layer ops", "Est." | patient-wise, 5-fold | [read in full text] (부분) |
| Frontiers | [MSPAN 2021](https://doi.org/10.3389/fnins.2021.761127) | memristor 시뮬레이션 | 초록에 없음 | [abstract only] |
| Frontiers | [SpikeWavformer 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12572938/) | 에너지 분석 절 있음(값은 안 옮김) | 못 읽음 | [read in full text] (부분) |
| IEEE Sensors J | [Scrugli 2026](https://doi.org/10.1109/jsen.2026.3663715) | FPGA 측정 9.1 mW | intra + Chazal inter | [read in full text] (부분) |
| IEEE Sensors J | [Vitale 2022](https://doi.org/10.1109/JSEN.2022.3194678) | Loihi | 못 읽음 | [secondary summary] |
| DATE | [Scrugli 2024](https://doi.org/10.23919/date58400.2024.10546736), [Diware 2025](https://doi.org/10.23919/date64628.2025.10993026) | FPGA 측정, 40 nm SoC | intra, 못 읽음 | 부분, [abstract only] |
| ICASSP | [SPTU-Lite 2025](https://doi.org/10.1109/icassp49660.2025.10889748) | 없음(parameter 수만) | 초록에 없음 | [abstract only] |
| IJCNN | [Corradi 2019](https://doi.org/10.1109/ijcnn.2019.8852279), [Banerjee 2024](https://doi.org/10.1109/ijcnn60899.2024.10650023) | 칩, SOP 추정 | 초록에 없음 | [abstract only] |
| Physiol. Meas.(심사 중) | [NeuroSleep](https://arxiv.org/abs/2602.15888) | "sparsity-adjusted effective operations", 측정 없음 | subject-independent | [read in full text] (부분) |

- 연산 수 추정의 표준 값 예시: E_MAC 4.6 pJ, E_AC 0.9 pJ(45 nm)이고, BioSpike-Net은 "definitive numbers require deployment on neuromorphic hardware"라고 스스로 적습니다. [read in full text] (부분) — [BioSpike-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC13509974/)
- 일부 논문은 추정이 하한이라고 설명합니다: "neuromorphic routing overhead would increase SNN inference cost in practice, making our estimates a lower bound". — 같은 곳
- 위 표에 쓴 venue 목록과 저자는 각 링크와 [Choi review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362400/), [Survey 참고문헌](https://arxiv.org/abs/2506.01737), Europe PMC 검색 결과로 확인했습니다.

### Inferences
- 우리 45 nm 연산 수 추정(E_MAC 4.6, E_AC 0.9)은 Neurocomputing, BSPC, Sensors, Neural Networks, Frontiers 쪽에서는 받아들여진 형식과 같습니다. neuron 항을 더하는 점은 오히려 보수적입니다. 단, memory access, routing을 뺀다는 한계를 BioSpike-Net처럼 적어야 합니다.
- TBioCAS나 DATE를 노리면 FPGA나 합성 결과 없이는 약해 보입니다. Amirshahi의 추정은 2019년 사례이고, 최근 TBioCAS 예(Chu)는 hardware가 있습니다. 표본이 작아 일반화는 조심스럽습니다.
- 정확도가 낮은 상태에서 에너지만 앞세우기도 어렵습니다. 129 µJ per window는 MIT-BIH SNN의 nJ-µJ 값과 입력 길이가 달라 직접 비교가 안 되고, CinC SNN에는 비교값이 없습니다. "Spikformer식과 같은 에너지에 더 높은 점수"라는 짝 비교가 가장 안전합니다.
- 받을 만한 증거 묶음의 최소 형태: CinC 5-fold 평균 ± SD와 class별 F1, 같은 파이프라인의 non-spiking 기준선, Spikformer식과의 짝 검정, MIT-BIH DS1/DS2와 beat-level 병기, 환자 또는 subject 분리된 두 번째 과제, 추정의 한계 명시입니다.

### Gaps
- JBHI에 실린 SNN ECG 논문은 웹 검색에서 찾지 못했습니다. 없다는 뜻은 아니며, IEEE Xplore를 직접 볼 수 없었습니다.
- venue별 심사 기준 문서는 찾지 않았습니다. 위 내용은 실린 논문에서 거꾸로 추정한 것입니다.
- BSPC, Neurocomputing 논문의 에너지 계산법은 전문을 못 읽어 확인하지 못했습니다.

## Q6. Attention-free spiking mixer, bidirectional SNN, T = N token-axis neuron을 ECG에 적용한 것이 있는가: overlap과 빈칸

### Takeaway
- 전체 조합(patch token + token 축 forward·reversed LIF + spiking gate fusion + spiking MLP + token-mean head)을 ECG에 쓴 논문은 찾지 못했습니다. 다만 요소 하나하나로 나누면 새로운 부분은 좁습니다.
- "attention-free" 자체는 ECG SNN에서 새롭지 않습니다. 대부분의 ECG SNN은 원래 CNN이나 MLP이고 attention을 쓰는 쪽이 소수입니다. 이 표현은 spiking transformer 계열과 대비할 때만 의미가 있습니다.
- "T = N"도 sample 또는 frame 단위 입력을 시간축으로 흘리는 SNN(Yin 2021의 QTDB 1301 step, level-crossing 처리기)과 개념이 겹칩니다. 다른 점은 patch token을 쓰고 T_h를 반복한다는 것입니다.
- 빈칸은 있지만 성능 빈칸은 아닙니다. CinC 2017에서 attention 기반 spiking 구조와 같은 조건으로 비교해 이긴 attention-free 구조, 그리고 window당 에너지 수치가 빈칸입니다.

### Cited Findings
| SpikMixer 요소 | ECG 또는 biosignal에서 찾은 가장 가까운 것 | overlap 판단 | 근거 |
|---|---|---|---|
| 1 채널 patch token | [ECGformer](https://arxiv.org/abs/2401.05434)(non-spiking, 신호를 나눠 token화). [NeuroSleep](https://arxiv.org/abs/2602.15888)의 tokenization(spiking, EEG). [SPTU-Lite](https://doi.org/10.1109/icassp49660.2025.10889748)(U자 hybrid, spiking FC) | 부분 겹침. ECG용 전부 spiking인 patch token 모델은 못 찾음 | [read in full text] (부분), SPTU는 [abstract only] |
| attention-free | [Yan 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362428/)(CNN 변환), [Chu 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362400/)(recurrent MLP), [Amirshahi 2019](https://arxiv.org/abs/1905.02954), [Scrugli 2026](https://zenodo.org/records/18887975) 등 다수. 반대로 attention 사용: [Rana 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175061/), [Xing 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362400/), [BioSpike-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC13509974/), [NeuroSleep](https://arxiv.org/abs/2602.15888) | 많이 겹침. 새 주장이 안 됨 | 혼합, 각 링크 등급 참조 |
| forward와 reversed LIF | [Yin 2021](https://arxiv.org/abs/2103.12593)의 bi-SRNN: "two SRNN layers ... reading the sequence from the forward and backward direction", "The mean of these layer's output". 단 TIMIT 전용이고, ECG는 "online and streaming task using only past information". non-spiking: [ECG-Mamba-V2](https://arxiv.org/abs/2510.13046)(양방향 state space, ECG), [BiT-MamSleep](https://arxiv.org/abs/2411.01589)(양방향 Mamba, 수면). Bi-LSTM은 흔한 기준선. [Hybrid Bi-LSTM-SNN](https://doi.org/10.54554/jtec.2026.18.03.002)은 제목만 봄 | 구조는 존재. ECG의 순수 spiking 양방향은 못 찾음. 빈칸이 좁음 | Yin [read in full text] (부분), 나머지 [secondary summary] |
| T = N token 축 neuron | 입력 시점을 그대로 SNN 시간축으로 쓰는 모델: [Yin 2021](https://arxiv.org/abs/2103.12593) QTDB seq 1301, [Chu 2022](https://doi.org/10.1109/tbcas.2022.3189364) level crossing, [Scrugli 2026](https://zenodo.org/records/18887975) delta. 반대로 변환 SNN은 입력 전체를 T번 반복: [Feng](https://arxiv.org/abs/2506.01737) 1000 step, [Rana](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175061/) 5 step | 개념 겹침. patch token 단위이고 T_h = 4 반복과 결합한 형태는 못 찾음 | 혼합 |
| spiking gate fusion | [BioSpike-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC13509974/)의 novelty gating은 attention 가중치라 다름. 같은 것은 못 찾음 | 겹침 없음(찾은 범위) | [read in full text] (부분) |
| spiking MLP-Mixer | [Li et al., arXiv 2306.12465](https://arxiv.org/abs/2306.12465)는 이미지 데이터 평가. [MD-Mixer, CVPR 2026](https://cvpr.thecvf.com/virtual/2026/poster/38330)는 vision | ECG 적용은 못 찾음. 일반 overlap은 같은 폴더 `attention_free_spiking_mixers.md` 참조 | [secondary summary] |
| 이름 "SpikMixer" | 검색에서 같은 이름의 논문이나 프로젝트를 찾지 못함 | 충돌 없음(찾은 범위) | [secondary summary] 웹 검색 |

### Inferences
- 논문의 중심 주장은 "attention-free"가 아니라 "spiking transformer의 token 혼합을 token 축 LIF 양방향 stepping으로 바꿔도, 같은 에너지에 긴 biosignal에서 더 높다"가 되어야 합니다. 이 비교 축에서만 새로운 정보가 됩니다.
- reviewer가 "patch를 시간축으로 흘리는 spiking RNN 아닌가"라고 물을 수 있습니다. 프로젝트 기억에도 SHD처럼 native bin이 있으면 T = N이 표준이라 제외했다고 되어 있습니다. 답은 recurrent weight가 없다는 점, T_h 반복, gate fusion이 무엇을 더하는지를 ablation으로 보이는 것입니다. 이미 있는 OR, sdlin, muxlin 비교가 이 근거가 됩니다.
- 현재 근거의 약점은 Spikformer식 대비 이득이 CinC에 치우친 점입니다. MIT-BIH는 차이가 없고, 긴 과제가 하나뿐입니다. 두 번째 긴 과제(Sleep-EDF)에서 같은 방향이 나오지 않으면 "긴 sequence" 주장은 성립하기 어렵습니다.
- 빈칸의 크기를 과장하면 안 됩니다. 찾지 못했다는 것은 이번 검색 범위에서의 결과입니다.

### Gaps
- 검색 범위: OpenAlex는 검색 3건과 DOI 조회 22건 후 429, Semantic Scholar는 429로 실패, arXiv API, Europe PMC, 웹 검색은 정상 동작했습니다. DBLP와 OpenReview는 막혀 있습니다. 2026년 후반 논문은 색인이 늦을 수 있습니다.
- ECG-Mamba-V2, BiT-MamSleep, MD-Mixer, Li et al.은 검색 요약만 봤습니다.
- SpikMixer의 구조 설명은 조사 요청의 문장과 로컬 코드 파일 목록에서 가져왔고, 이번 조사에서 `spk_mixer.py` 전문을 다시 읽지는 않았습니다.
