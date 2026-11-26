# Inter-patient MIT-BIH는 처음부터 이웃 beat를 썼다

전제부터 바로잡는다. 환자 단위 분할(inter-patient: test 환자가 학습에 없는 분할)에 이웃 beat 정보를 더하는 것은 새 방법이 아니다. 이 분할을 정의한 논문이 이미 그렇게 했다. de Chazal 외 2004의 12개 configuration은 모두 RR 간격(연속한 두 beat 사이 시간) feature 네 개를 넣었다 ([de Chazal 외 2004, 본문](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)). 이후 inter-patient 방법도 대부분 RR을 쓴다. 여러 beat의 파형을 직접 넣는 갈래도 2018–2019년부터 있다 ([Guo 외 2019](https://arxiv.org/abs/1810.04121); [Mousavi & Afghah 2019](https://arxiv.org/abs/1812.07421)). 비어 있는 곳은 셋이다. 과거 beat만 쓰는(causal) 다중 beat 입력은 드물다. beat나 RR의 순서를 섞어 본 논문은 찾지 못했다. DS1/DS2(de Chazal의 학습용·test용 record 묶음)에서 앞 beat를 network 입력으로 받는 SNN도 찾지 못했다. Ablation은 이웃 beat 정보의 값을 S class(심실 위, 곧 심방이나 방실 접합부에서 시작한 조기 beat)에서 일관되게 보인다. 같은 ResNet에 RR을 더하면 S F1이 **31.4 → 46.7**로 오른다 ([Cheng 외 2020, arXiv만](https://arxiv.org/abs/2007.04871)). 단일 beat만 보는 Kachuee CNN을 DS1로 다시 학습하면 DS2의 S beat를 **하나도** 찾지 못한다 ([Cao 외, arXiv만](https://arxiv.org/abs/2206.14200)). 하지만 가장 큰 이득은 Hindawi 논문에서 나왔고, 쓰인 RR feature 대부분은 다음 beat나 test record 전체의 평균을 쓴다. 넘어야 할 DNN 수준은 **S F1 약 76–82, V F1 약 90–95**다. 이 수치도 거의 다 다음 beat를 본다. SNN 쪽 inter-patient 최선은 **S Se 67 / +P 46**이다(sensitivity / positive predictivity, 다른 논문의 표). Reviewer가 점검할 것은 DS2로 model을 골랐는지, test 환자 data를 썼는지, sequence가 한 record의 연속 beat인지, 미래 beat를 썼는지다. DS2 S beat의 **75 %가 record 232 하나**에 있다는 점도 결과 해석을 흔든다. 과거만 보고 순서에 민감한 변형에는 자리가 있다. 다만 가장 큰 위험이 있다. "앞 RR 간격과 최근 평균 RR"이라는 순서 없는 요약이 이득 대부분을 가져갈 수 있다. 그래서 성패는 model 비교가 아니라 두 대조에 달려 있다. 과거 beat를 섞어 다시 학습하는 대조와, causal RR 숫자만 붙인 baseline이다.

## 읽는 법과 용어

link가 붙은 문장은 조사 노트가 원문에서 확인한 것이다. 수치마다 출처 종류를 붙인다. **본문**은 전문을 읽은 것, **초록만**은 초록만 본 것이다. **다른 논문의 표**는 다른 논문의 비교표에서 옮긴 것이다. **계산**은 노트가 논문의 수치로 F1 등을 계산한 값이다. 논문이 인쇄한 값이 아니다. 발표처 표시는 **arXiv만**(심사 통과판을 찾지 못함), **MDPI**, **Hindawi**, **Frontiers**, **학회만**(학회 논문만 있음), **preprint**(medRxiv), **심사 전**(F1000Research), **GitHub 비공식**(issue 토론)이다. "추론"은 노트나 이 보고서의 판단이다. "찾지 못했다"는 검색에 나오지 않았다는 뜻이지, 없다는 증명이 아니다. 연구실 수치는 실험 기록([MODI 기록](../../docs/MODI_PROJECT_LOG.md))에서 옮겼다.

| 용어 | 뜻 |
|---|---|
| R-peak, RR 간격 | R-peak는 QRS(심실이 수축할 때의 큰 파형)의 꼭짓점이다. RR 간격은 연속한 두 R-peak 사이 시간이다. 심박 리듬을 나타낸다. |
| pre-RR / post-RR | 현재 beat와 바로 **앞** beat 사이의 RR / 현재 beat와 바로 **다음** beat 사이의 RR. post-RR은 다음 beat가 와야 계산된다. |
| local / average RR | 현재 beat 주변 RR 여러 개의 평균(de Chazal은 둘러싼 10개) / record 전체 RR의 평균. 둘 다 현재 beat 뒤의 beat도 쓴다. |
| 과거 창 평균 | 현재 beat 앞의 RR만으로 낸 평균(예: 최근 80개). causal하다. |
| DS1 / DS2 | de Chazal 2004가 정한 MIT-BIH record 두 묶음. 각 22개 record다. DS1로 학습과 model 선택을 하고, DS2는 마지막 test에만 쓴다. |
| inter-patient / intra-patient | test 환자가 학습에 전혀 없는 분할 / 같은 환자의 beat가 학습과 test 양쪽에 있는 분할. 무작위 beat 분할은 intra-patient다. |
| patient-specific, assisted | test 환자 record의 일부(예: 앞 5분)를 학습이나 보정에 쓰는 설정. 전문가가 일부 beat를 고쳐 주면 assisted라고 부른다. |
| AAMI class | 미국 의료기기 표준(AAMI EC57)의 다섯 묶음. N = 정상과 각차단 등, S = SVEB(심방·방실 접합부에서 시작한 조기 beat), V = VEB(심실에서 시작한 beat), F = 심실 beat와 정상 beat의 융합, Q = paced beat와 분류 불가. 자세한 대응은 아래 표에 있다. |
| Se, +P, F1, FPR | sensitivity(Se) = 실제 그 class 가운데 맞힌 비율. positive predictivity(+P) = 그 class로 예측한 것 가운데 맞은 비율. 논문에 따라 PPV, Ppr, precision으로 적는다. F1 = 둘의 조화평균. FPR = 다른 class를 그 class로 잘못 부른 비율. |
| macro-F1 | class별 F1의 단순 평균. N이 약 90 %인 data라서 accuracy보다 정직하다. |
| causal / 한 beat 지연 / 중심 창 | 현재와 과거 beat만 쓴다 / 다음 beat 하나를 기다린다(post-RR) / 앞뒤 beat를 다 쓴다. |
| seq2seq, BiLSTM | beat 열을 받아 beat마다 label을 내는 model / 앞뒤 두 방향으로 읽는 LSTM. BiLSTM은 미래 beat를 본다. |
| SMOTE | 소수 class의 합성 표본을 만들어 class 비율을 맞추는 기법. |
| 섞어서 재학습 | 표본마다 무작위로 순서를 섞은 data로 처음부터 학습하고, 섞은 data로 test 한다. "이 과제가 순서에 보상하는가"를 본다. |

## 이웃 beat 정보는 de Chazal 2004의 기본 설정이었다

### 출발점: 지금 설정에는 이웃 beat가 없다

연구실의 MIT-BIH는 Kachuee 외 2018의 beat dataset이다. 125 Hz에서 187점이다. 각 표본은 R-peak에서 시작해 RR 중앙값의 1.2배를 자르고 0으로 채운다. R-peak만 찾고 P파와 T파는 찾지 않는다. 그래서 그 beat 자신의 P파와 pre-RR은 표본에 없다. 대신 다음 beat의 P파가 들어 있고, 다음 R-peak가 일찍 오면 그것까지 들어간다. 이 분할은 무작위 beat 분할이다. 여기서 순서를 전혀 못 보는 Spikformer(DE)가 macro-F1 **91.27 ± 0.16**이었다. 상대 위치 bias를 준 판은 91.94 ± 0.49였다. 순서의 값은 −0.67 %p에 그쳤다 ([MODI 기록 11-19, 11-26](../../docs/MODI_PROJECT_LOG.md)). 그래서 이웃 beat가 들어가는 환자 단위 설정을 검토했다.

### 분할을 만든 논문부터 RR을 썼다

나쁜 소식부터 적는다. 사용자는 "평범한 inter-patient 분할만 있다"고 보았다. 문헌은 반대다. de Chazal 외 2004는 MIT-BIH 48개 record 가운데 paced record 4개(102, 104, 107, 217)를 빼고 22개씩 DS1과 DS2로 나눴다. RR feature는 넷이다. pre-RR, post-RR, record 전체 평균 RR, 둘러싼 10개 RR의 평균이다. **8개 feature set과 12개 configuration이 모두 이 네 개를 포함한다.** 그래서 이 논문은 RR이 얼마나 돕는지 재지 않았다. configuration 사이에서 바뀐 것은 lead, 파형 표본 방식, 진폭 scaling, 파형 분할 사용 여부뿐이다 ([de Chazal 외 2004, 본문](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)). 이 논문은 Crossref 기준 **1,362회** 인용됐다 ([Crossref](https://api.crossref.org/works/10.1109/tbme.2004.827359)). 인용 수는 database마다 다르다.

이후의 고전적 inter-patient 방법도 거의 다 RR을 넣었다. Garcia 외 2017의 정리표에 Soria & Martinez 2009, Mar 외 2011, Ye 외 2012, Lin & Yang 2014, Huang 외 2014가 모두 RR이나 시간 간격 feature와 함께 올라 있다 ([Garcia 외 2017, 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/)). Llamedo & Martínez 2011은 "RR series에서 뽑은 feature"를 쓴다 ([초록만](https://pubmed.ncbi.nlm.nih.gov/20729162/)). deep learning 시대에도 단일 beat 파형 옆에 RR 숫자를 붙이는 설계가 주류다. 예는 PeerJ CS 2020 ([본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC7924512/)), Wang 외 2021 ([MDPI, 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC7831114/)), Zhang J. 외 2021 ([Hindawi, 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC8181174/)), Self-ONN 2023 ([arXiv판 본문](https://arxiv.org/abs/2204.03768)), iScience 2024 ([본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10933543/)), CDLR-net 2025 ([초록만](https://pubmed.ncbi.nlm.nih.gov/40914799/))다.

PubMed 표본으로 세어도 같다. inter-patient ECG 검색에 190건이 나왔다. 그 가운데 63건이 제목이나 초록에서 MIT-BIH, DS1/DS2, de Chazal을 언급했다. 그 63건 가운데 약 20건의 초록이 다른 beat의 정보(RR, beat 사이 간격, 이웃 조각, rhythm feature, 앞 beat, seq2seq)를 명시했다 ([PubMed 검색](https://pubmed.ncbi.nlm.nih.gov/?term=%28inter-patient%5Btiab%5D+OR+interpatient%5Btiab%5D%29+AND+%28ECG%5Btiab%5D+OR+heartbeat%5Btiab%5D%29)). 이것은 하한이다. 본문에서만 RR을 쓰는 논문이 있고, ESWA·BSPC·IEEE Access 같은 주요 학술지는 PubMed에 없다. 노트가 성격을 확인한 약 35개 학습 방법으로 세면 약 28–30개가 다른 beat의 정보를 쓰고, 약 5개만 단일 beat다(추론, 노트의 분류).

| 갈래 | 무엇을 넣나 | 대표 논문 (출처 종류) | 미래 beat를 보나 |
|---|---|---|---|
| 단일 beat + RR 숫자 | pre-RR, post-RR, local·record 평균 RR을 숫자로 붙인다 | de Chazal 2004 (본문), Garcia 2017 (본문), Mondéjar-Guerra 2019 (공개 code), Wang 2021 (MDPI, 본문), Zhang J. 2021 (Hindawi, 본문), Self-ONN 2023 (arXiv판 본문), Farag 2023 (MDPI, 본문), iScience 2024 (본문) | 거의 모두 본다. post-RR, record 평균, ±10 s 평균 |
| 여러 beat의 파형 | 10초 조각, beat sequence, 앞 beat | Guo 2019 (arXiv판 본문), Mousavi & Afghah 2019 (학회만, arXiv 본문), Sellami & Hwang 2019 (특허로 입력 확인), MF-CBRNN 2020 (초록만), Jiang 2020 (학회만, 초록), capsule + Seq2Seq 2022 (초록만), JTEHM 2022 (본문), Applied Sciences 2024 (MDPI, 검색 요약만) | 대부분 본다. Sellami만 과거 쪽이다 |
| 단일 beat만 | beat 하나의 파형 | Takalo-Mattila 2018 (학회만, 다른 논문의 표), Xu 외 2019 JBHI (초록만), Cheng 2020의 RR 없는 판 (arXiv만), Yan 외 2021 SNN | 보지 않는다 |
| 과거만 쓰는 feature | 앞 beat에서만 계산한 값 | Sellami & Hwang (특허), Computers in Biology and Medicine(CBM) 2026 hybrid feature (초록만) | 보지 않는다 |

여러 beat를 넣는 갈래는 출처를 꼼꼼히 봐야 한다. Guo 2019는 beat가 아니라 10초 조각을 분류한다. 조각의 label은 그 안에서 가장 많은 annotation이고, N이 아닌 annotation이 있으면 N은 뺀다 ([Guo 외, arXiv판 본문](https://arxiv.org/abs/1810.04121)). Mousavi & Afghah는 T파 사이로 자른 beat를 280점으로 맞춰 sequence로 넣고 BiLSTM encoder를 쓴다. 저자는 "현재 state가 이전과 미래 입력을 동시에 본다"고 적었다 ([arXiv 본문](https://arxiv.org/abs/1812.07421)). Sellami & Hwang 2019는 "여러 beat"를 쓴다고 했다 ([서울시립대 기록](https://pure.uos.ac.kr/en/publications/a-robust-deep-convolutional-neural-network-with-batch-weighted-lo/)). 관련 특허는 입력을 현재 beat와 **바로 앞 beat** 하나로 적는다. 각 170점(R 앞 70점, 뒤 100점)이다 ([US11298067B2](https://patents.google.com/patent/US11298067)). 다른 논문은 같은 입력을 "1–3 beat"로 적었다 ([Lee 외 2025, 검색 요약만](https://www.sciencedirect.com/science/article/pii/S0169260725003153)). 노트끼리도 표현이 갈린다. Wu 외 2022는 Sellami를 단일 beat로 분류했다 ([Hindawi, 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC9481402/)). 여러 beat를 넣었지만 inter-patient가 아닌 예도 있다. Frontiers가 호스팅한 CNN + BiLSTM 다중 beat 논문은 Kachuee식 전처리에 87,554 / 12,892의 무작위 분할을 썼다 ([Frontiers, PMC9760867](https://pmc.ncbi.nlm.nih.gov/articles/PMC9760867/)). 비교표에 넣으면 안 된다.

드문 것은 따로 있다(추론). 첫째, 과거 beat만 쓰는 raw 다중 beat 입력이다. 선례는 Sellami의 앞 beat 하나와, 앞 beat에서만 계산한 feature를 "환자 정보와 미래 정보의 누출을 막으려고" 쓴 2026년 논문 정도다 ([CBM 2026, 초록만](https://pubmed.ncbi.nlm.nih.gov/42727300/)). 둘째, 수십 beat에 걸친 긴 과거다. 셋째, 과거 beat의 **순서**가 중요한지 재는 시험이다. 이것은 하나도 찾지 못했다.

## Ablation 여섯 편이 모두 이득을 보였고, 순서를 잰 논문은 없다

### 먼저 의심할 점

이득의 방향은 일관된다. 크기는 믿기 어렵다. 이유는 셋이다. 가장 큰 이득(S Se +72.5 %p, +63.9 %p)은 둘 다 Hindawi의 Journal of Healthcare Engineering에서 나왔다. 그 RR feature는 일부가 test record 전체나 다음 beat를 쓴다. 그리고 이득의 크기는 단일 beat baseline이 얼마나 약한지에 달려 있다. baseline이 S를 거의 못 찾을 때(Se 6.3 %, 16.3 %) RR이 구해 준다. baseline이 강하면 이득은 몇 점이다(노트의 추론). 무엇보다 **beat 순서나 RR 순서를 섞는 대조는 어느 논문에도 없다.** "더 많은 beat 정보"와 "순서 있는 과거"를 가른 적이 없다.

### 같은 model에 이웃 beat 정보를 더한 결과

| 논문 (출처 종류) | 분할 | 무엇을 더했나 | S: 없음 → 있음 | V: 없음 → 있음 | 미래 정보 |
|---|---|---|---|---|---|
| Cheng 외 2020 (arXiv만, 본문) | DS1 → DS2 | 같은 ResNet에 pre-RR + post-RR | Se 25.8 → 44.5, +P 41.8 → 52.2, **F1 31.4 → 46.7** | F1 85.8 → 89.2 | post-RR, record 평균으로 정규화 |
| Zhang J. 외 2021 (Hindawi, 본문) | DS1 → DS2 | pre-RR 비율 두 개(record 전체 평균 대비, 최근 10개 대비) | **Se 6.3 → 78.8**, +P 42.2 → 90.8 | Se 88.7 → 92.5, +P 91.3 → 94.3 | record 전체 평균 |
| Wu 외 2022 (Hindawi, 본문) | DS1 → DS2 | 창을 256점에서 508점(R 앞 250, 뒤 257)으로 | **Se 16.34 → 80.23**, +P 14.60 → 49.40 | Se 85.15 → 90.99, +P 77.32 → 83.09 | 앞뒤 약 ±0.7 s (노트의 계산, 360 Hz 가정) |
| iScience 2024 EXP2 (본문) | DS1 + **DS2 각 record의 앞 5분** → DS2 | pre-RR, post-RR, 비율 | Se 77.4 → 81.5, +P 87.2 → 94.9, F1 82.0 → 87.7 | F1 97.0 → 97.6 | post-RR. 분할은 patient-adapted |
| iScience 2024 EXP3 (본문) | 비표준 20 / 24 record | 같음 | F1 71.4 → 75.0 | F1 94.3 → 95.0 | post-RR |
| Alforja & Aban 2026 (preprint, GitHub README) | DS1 → DS2, DS1 안 record 단위 5-fold로 선택 | RR feature 6개 | macro F1 **+0.025 – +0.093** (model 네 종류, bootstrap 구간이 0을 포함하지 않음) | — | README 기준 미상 |
| Banjo & Ghoraani 2026 (MDPI, 본문), **SNN** | DS1 → DS2 (탐색용) | RR 4개(앞, 다음, 비, 차)를 head에 | Stage-1 macro F1 +0.115, Stage-2 +0.066, Stage-2 accuracy +6.2 %p | — | post-RR |

출처: [Cheng](https://arxiv.org/abs/2007.04871), [Zhang J.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8181174/), [Wu](https://pmc.ncbi.nlm.nih.gov/articles/PMC9481402/), [iScience 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10933543/), [Alforja & Aban](https://github.com/janusaban/interpatient-ecg-classification), [Banjo & Ghoraani](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306364/).

V class의 이득은 작다. 단일 beat만으로도 V는 잘 잡힌다. Cheng의 RR 없는 ResNet이 이미 V Se 95.6이었다 ([arXiv만](https://arxiv.org/abs/2007.04871)). 반대로 S는 단일 beat로 거의 못 잡는다. 문헌의 설명도 같다. S beat는 "형태 feature만으로는 대개 구별하기 어렵다" ([Teijeiro 외, arXiv 동반 논문](https://arxiv.org/abs/1609.05632)). 그 특징은 조기 출현과 RR 변화 같은 rhythm 관계에 있다 ([Zhao 2026, 석사 논문](https://yorkspace.library.yorku.ca/items/606f08b1-f591-40cb-9568-116baaca373d)).

### 단일 beat는 환자가 바뀌면 S를 잃는다

연구실 형식과 가장 가까운 증거가 가장 나쁘다. Cao 외는 Kachuee의 187점 CNN을 두 분할로 다시 구현했다. 무작위 분할에서는 accuracy 93.1, S precision / recall 38.1 / 82.6이었다. DS1로 학습하고 DS2로 test 하자 accuracy 81.2, **S 0.0 / 0.0**이었다 ([Cao 외, arXiv만, 본문](https://arxiv.org/abs/2206.14200)). 파형을 쓰지 않고 RR만 쓰는 반대쪽 극단도 있다. 파형 없이 RR만 넣은 XGBoost가 S Se / +P / F1 60 / 36 / 45, V 58 / 75 / 65를 냈다 ([Farag 2023, MDPI, 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183)). 파형과 RR이 서로를 보완한다는 뜻이다(추론).

de Chazal 2004 안에도 관련 증거가 있다. RR을 뺀 configuration은 없지만, 같은 classifier를 DS1 안에서 두 방식으로 쟀다. beat 단위로 fold를 나누면 accuracy 86.1, S Se 77.6이었다. record 단위로 나누면 79.0, 59.9였다 ([de Chazal 외 2004, 본문](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)). 환자가 바뀔 때 가장 크게 무너지는 class가 S다.

### 몇 beat가 필요한가: 소수의 근거

beat 수를 바꾼 결과는 둘뿐이다. 둘 다 DS1/DS2가 아니거나 수치가 없다. Lee 외 2025는 입력 beat 수를 바꿨고 "inter-patient 조건에서 4 beat에서 정확도가 가장 높고 그 뒤로는 개선이 미미했다"고 썼다(accuracy 94.82 %). 분할은 HiCardi 착용형 data와 MIT-BIH의 환자 단위 5-fold다 ([CMPB 2025, 초록만](https://pubmed.ncbi.nlm.nih.gov/40554829/)). Merdjanovska & Rashkovska 2023은 여러 분절 방식 가운데 "beats2.4s"가 macro F1이 가장 높다고 했다. 이것은 현재 beat와 앞뒤 이웃 beat 하나씩을 담은 중심 창이다 ([Sci Rep 2023, 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10356811)). 둘 다 몇 beat 안에서 포화한다는 쪽을 가리킨다. 긴 축적의 근거는 아니다(추론).

### 순서: 시험은 없고 정성적 단서만 있다

190건 PubMed 표본의 어느 초록도 beat를 섞거나 순서를 바꾼 시험을 언급하지 않았다. 노트가 전문을 읽은 9편(Mousavi, Guo, Self-ONN, Zhang J. 2021, Wu 2022, iScience 2024, JTEHM 2022, PeerJ 2020, Wang 2021)에도 없다 ([PubMed 검색](https://pubmed.ncbi.nlm.nih.gov/?term=%28inter-patient%5Btiab%5D+OR+interpatient%5Btiab%5D%29+AND+%28ECG%5Btiab%5D+OR+heartbeat%5Btiab%5D%29)). causal과 양방향 context를 같은 model로 비교한 논문도 없다.

정성적 단서는 하나 있고 중요하다. Self-ONN 논문은 이렇게 썼다. record 202, 222, 232, 234에서는 S beat 여러 개가 이어지는데 "첫 S beat만 timing 이상을 보인다". record 234에는 50개 연속 supraventricular beat로 된 junctional tachycardia 구간이 있다. 그리고 "beat 단위 분류를 노리는 모든 algorithm은 결국 이런 record에서 고생한다" ([Self-ONN, arXiv판 본문](https://arxiv.org/abs/2204.03768)). 이 record들은 DS2 S 점수를 좌우하는 record와 겹친다. DS2 S beat 1,837개 가운데 232에 1,382개, 222에 209개가 있다 ([de Chazal 외 2004, Table VII, 본문](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)). 그러니 DS2의 S 점수는 "앞 RR 하나로는 안 되는" record에서 대부분 결정된다(추론). run이 언제 시작했는지를 기억하는 순서 있는 과거가 도울 수 있는 자리다. 측정한 사람은 없다.

이전 보고서의 한 수치를 바로잡는다. 이전 보고서는 이웃 beat context의 이득을 "검색 요약에서만 본 SVEB F1 51.08 → 61.25"로 적었다. 원문을 보면 이것은 Guo 2019 초록의 문장이고, 표의 값은 61.94다. 그리고 이전 방법들과의 비교이지 같은 model의 ablation이 아니다. 10초 조각 단위의 점수라 beat 단위 결과와도 견줄 수 없다 ([Guo 외, arXiv판 본문](https://arxiv.org/abs/1810.04121)). 실제 ablation은 위 표의 여섯 편이다.

## 넘어야 할 선: DNN은 S F1 76–82, SNN은 S +P 46에 머문다

### 규칙: DS1로만 고르고 DS2는 마지막에 한 번 연다

de Chazal 2004의 규칙은 다음과 같다. 48개 record는 각 약 30분, 두 lead, 360 Hz다. DS1은 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230이다. DS2는 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234다. DS1으로 classifier를 고른다. DS2는 "classifier 개발의 어느 단계에서도 쓰지 않았다". beat 위치는 database의 수정된 annotation을 쓴다. 저자는 자동 검출을 쓰면 RR feature에 오류가 더해진다고 적었다 ([de Chazal 외 2004, 본문](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)).

| AAMI class | 포함하는 MIT-BIH beat 종류 (de Chazal Table I) |
|---|---|
| N | 정상, 좌각차단, 우각차단, 심방 탈출, 접합부 탈출 |
| S (SVEB) | 심방 조기, 변행 심방 조기, 접합부 조기, 상심실 조기 |
| V (VEB) | 심실 조기 수축, 심실 탈출 |
| F | 심실과 정상의 융합 |
| Q | paced, paced와 정상의 융합, 분류 불가 |

지표 규칙도 정해져 있다. VEB 통계는 F나 Q를 V로 부른 것을 상도 벌도 주지 않는다. SVEB 통계는 Q를 S로 부른 것을 무시한다. 모든 beat를 같은 무게로 세는 "gross" 통계다. de Chazal의 최종 DS2 결과(configuration IX)는 SVEB Se 75.9, +P 38.5, FPR 4.7, VEB Se 77.7, +P 81.9, FPR 1.2다. 가장 큰 오류는 N을 S로 부른 1,904개와 N을 F로 부른 3,509개였다 ([본문](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)). 고전 방법의 약점은 S Se가 아니라 S +P다. 그래서 Se와 +P(또는 F1)를 함께 보고해야 한다(노트의 추론).

PhysioNet은 이 분할의 약점 하나를 공식적으로 적고 있다. 48개 record는 47명에게서 나왔다. record 201과 202는 같은 사람이고 "같은 쪽에 두어야 한다" ([PhysioNet mitdb](https://physionet.org/content/mitdb/1.0.0/)). de Chazal 분할은 201을 DS1에, 202를 DS2에 둔다. 체계적 review도 이 점을 지적했다 ([Silva 외 2025](https://arxiv.org/abs/2503.07276); 이 review는 protocol 노트에서는 arXiv만, SNN 노트에서는 [Res. Biomed. Eng. 2026 게재판](https://link.springer.com/article/10.1007/s42600-026-00485-z)으로 확인됐다).

### beat 수부터 논문마다 다르다

| 출처 (출처 종류) | DS1 N / S / V / F / Q (합) | DS2 N / S / V / F / Q (합) |
|---|---|---|
| de Chazal 2004 Table II (본문; class 합은 노트 계산) | 45,868 / 942 / 3,787 / 415 / 8 (51,020) | 44,258 / 1,837 / 3,221 / 388 / 7 (49,711) |
| 흔히 인용되는 값, "de Chazal을 따름" ([Silva 외 2025 Table 3](https://arxiv.org/abs/2503.07276), arXiv판; DS2 N 44,259는 [Cao 외 Table 2](https://arxiv.org/abs/2206.14200), arXiv만) | 45,866 / 944 / 3,788 / 415 / 8 (51,021) | 44,259 / 1,837 / 3,221 / 388 / 7 (49,712) |
| [Mousavi & Afghah 2019](https://arxiv.org/abs/1812.07421) (학회만, arXiv 본문) | N 45,796 / S 941 / V 3,780 | N 44,196 / S 1,836 / V 3,216 |
| [Sellami & Hwang 특허](https://patents.google.com/patent/US11298067) | 합 50,993 | 합 49,682 |
| [Mommen 외 2026](https://arxiv.org/abs/2609.32854) (arXiv만) | S 3,778 (e, j beat를 S로 넣음) | S 3,208 |
| 연구실 저장소 `denoise_patient_mitbih_*` (출처 기록 없음, [MODI 기록 11-26](../../docs/MODI_PROJECT_LOG.md)) | 합 50,084 | 합 49,477 |

de Chazal 원문의 text도 Table II와 같다("942 SVEBs … 3787 VEBs"). 널리 인용되는 값은 원문보다 DS1에서 N +2, S +2, V +1, DS2에서 N +1이 많다. 그 값의 출처는 찾지 못했다. 첫 beat와 끝 beat를 다르게 처리한 재추출로 보이지만 확인하지 못했다(노트의 추론). 연구실 저장소의 환자 단위로 보이는 파일은 어느 값과도 맞지 않는다. 이미 "쓰지 않는다"로 기록돼 있고, 이 표가 그 결정을 뒷받침한다(추론).

### DNN baseline: 높은 수치는 거의 다 다음 beat를 본다

| 방법 (발표처, 연도) | 입력 | S Se | S +P | S F1 | V Se | V +P | V F1 | 미래 정보 | 출처 종류 |
|---|---|---|---|---|---|---|---|---|---|
| [de Chazal 외](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf) (TBME 2004) | beat + RR, 2 lead | 75.9 | 38.5 | 51.1 (계산) | 77.7 | 81.9 | 79.7 (계산) | post-RR, record 평균, 중심 10개 평균 | 본문 |
| [Llamedo & Martínez](https://pubmed.ncbi.nlm.nih.gov/20729162/) (TBME 2011) | beat + RR series | 77 | 39 | 51.8 (계산) | 81 | 87 | 83.9 (계산) | post-RR 선택 여부 미확인 | 초록만 |
| [Zhang 외](https://pubmed.ncbi.nlm.nih.gov/24529208/) (CBM 2014) | beat + beat 사이 간격 | 79.06 | 35.98 | 49.46 | 85.48 | 92.75 | 88.96 | 미확인 | 초록만; F1은 [Wang 2021 표](https://pmc.ncbi.nlm.nih.gov/articles/PMC7831114/) |
| [Garcia 외](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/) (Sci Rep 2017) | beat + pre/post/local RR | 62.0 | 53.0 | 57.2 | 87.3 | 59.4 | 70.7 | post-RR | 본문 |
| [Teijeiro 외](https://arxiv.org/abs/1609.05632) (JBHI 2018) | 규칙 기반, record 전체 해석. DS1 학습 없음 | 87.17 | 83.98 | 85.5 (계산) | 94.63 | 96.79 | 95.7 (계산) | record 전체 | arXiv 동반 논문의 표 |
| Mondéjar-Guerra 외 (BSPC 2019) | beat + RR + 파형 feature, SVM ensemble | 78.1 | 49.7 | 60.7 (계산) | 94.7 | 93.9 | 94.3 (계산) | post-RR, record 평균으로 나눈 사본 | 다른 논문의 표 두 개 ([Farag](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183), MDPI; [Mommen](https://arxiv.org/abs/2609.32854), arXiv만) |
| Sellami & Hwang (ESWA 2019) | 현재 + 앞 beat (특허 기준) | 82.04 | 30.44 | — | 92.05 | 72.13 | — | 과거만 (특허 기준) | 다른 논문의 표 ([Self-ONN arXiv판](https://arxiv.org/abs/2204.03768)) |
| [Niu 외](https://pubmed.ncbi.nlm.nih.gov/31545750/) (JBHI 2020) | beat + rhythm 기호화 | 76.5 | 76.6 | **76.6** | 85.7 | 94.1 | 89.7 | 미확인 | F1은 초록; Se / +P는 다른 논문의 표 ([Cheng](https://arxiv.org/abs/2007.04871)) |
| [Wang 외](https://pmc.ncbi.nlm.nih.gov/articles/PMC7831114/) (Entropy 2021, MDPI) | CWT + CNN + RR | 74.56 | 89.54 | **81.37** | 95.65 | 93.25 | 94.43 | post-RR, 평균 RR을 뺌(창 미기재) | 본문 |
| [Zahid 외](https://pubmed.ncbi.nlm.nih.gov/35786545/) (TBME 2023, Self-ONN + RR) | beat + RR | 82.50 | 82.19 | **82.34** | 96.10 | 94.41 | 95.2 | post-RR, ±10 s 평균 | 초록 (arXiv판 값은 다름) |
| [Farag](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183) (Sensors 2023, MDPI), model 6 | beat + RR | 81.60 | 82.68 | **82.14** | 95.00 | 95.63 | 95.31 | post-RR만. 평균은 과거 80 / 400개 | 본문 |
| [Cheng 외](https://arxiv.org/abs/2007.04871) (2020, arXiv만), RR 없음 | beat (1.96 s, 2 lead) | 25.8 | 41.8 | 31.4 | 95.6 | 78.9 | 85.8 | — | 본문 |
| Kachuee CNN, DS1 → DS2 재학습 ([Cao 외](https://arxiv.org/abs/2206.14200), arXiv만) | beat (187점) | 0.0 | 0.0 | 0 | 92.4 | 30.9 | 46.3 (계산) | — | 본문 |
| [Mousavi & Afghah](https://arxiv.org/abs/1812.07421) (ICASSP 2019, 학회만) | beat sequence, BiLSTM seq2seq | 88.94 | 92.57 | 90.7 (계산) | 99.94 | 99.50 | 99.7 (계산) | 양방향 | arXiv 본문. **이상치, 다음 절 참조** |
| [Guo 외](https://arxiv.org/abs/1810.04121) (Biocybern. Biomed. Eng. 2019) | 10초 조각, 조각 label | 62.70 | 61.21 | 61.94 | 91.25 | 88.30 | 89.75 | 앞뒤 beat | arXiv판 본문. **beat 단위가 아님** |

현실적인 선은 이렇다(노트의 추론). 엄격한 protocol에서 "최고 수준에 가깝다"고 하려면 S F1이 약 76(Niu 2020)에서 82(Zahid 2023, Farag 2023)는 돼야 한다. V F1은 약 90–95다. 고전 방법(S F1 약 50–61)은 쉽게 넘는다. 문제는 이 선을 만든 model이 모두 **다음 beat를 본다**는 점이다. Farag 2023이 그나마 실시간에 가장 가깝다. 평균은 과거 창으로만 내고, post-RR 하나만 미래다. 즉 한 beat 지연이다 ([Farag 2023, MDPI, 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183)). 과거만 쓰는 model의 발표된 DS2 선은 사실상 없다. 앞 beat에서만 feature를 뽑은 2026년 논문은 linear SVC로 weighted F1 0.780, 네 class macro F1 0.493을 냈다. 같은 예산의 CNN은 "S class가 무너졌다"고 한다 ([CBM 2026, 초록만](https://pubmed.ncbi.nlm.nih.gov/42727300/)).

reviewer가 들고 올 수 있는 확인 안 된 높은 수치도 있다. capsule + Seq2Seq 2022는 S Se 99.56 / +P 92.23을 적었다 ([CMPB 2022, 초록만](https://pubmed.ncbi.nlm.nih.gov/34879327/)). seq2seq라서 다음 절의 sequence 구성 문제가 그대로 걸린다. Mousavi code를 물려받았는지는 확인하지 못했다. MKP-ECG 2026은 accuracy 99.14 %, macro-F1 92.77 %를 적었다 ([Interdiscip. Sci. 2026, 초록만](https://pubmed.ncbi.nlm.nih.gov/43077017/)). 이런 수치는 본문을 확인하기 전에는 목표로 쓰지 않는 것이 맞다(추론).

### SNN baseline: 다섯 편, 모두 단일 beat

SNN 쪽은 얇고 약하다. 이것은 기회이자 위험이다.

| 논문 | 발표처 | 실제 분할 | 입력 | S Se / +P | V Se / +P / F1 | Accuracy | 출처 종류 |
|---|---|---|---|---|---|---|---|
| Yan, Zhou, Wong 2021 | BSPC | DS1/DS2, 4 class | raw 단일 beat, CNN → SNN, 2단계 | 67 / 46 (F1 54.5) | 77 / 59 / 66.8 | 90 | 다른 논문의 표 ([Silva 2025 Table 9](https://arxiv.org/abs/2503.07276); [Rana & Kim Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175061/), MDPI; F1 54.5는 [Mommen](https://arxiv.org/abs/2609.32854), arXiv만) |
| Xing 외 2022 | Electronics (MDPI) | 주 결과는 무작위. DS1/DS2 비교를 따로 | R 중심 300점 beat | 보고 없음 | 69.04 / 76.46 / 72.56 | 92.07 | 다른 논문의 표 두 개(서로 일치) |
| [Rana & Kim 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175061/) | Sensors (MDPI) | DS1/DS2와 70:30 | R 앞뒤 250점 | 보고 없음 | 54 / 75.0 / 73.5 | 93.8 | 본문 |
| [Galvis-Chacón 외 2026](https://www.recercat.cat/handle/20.500.14342/5924) | Neurocomputing | inter와 intra | LIF + axonal delay | 미확인 | 미확인 | 86.38 (4 class) | 검색 요약만. 본문은 2028년까지 공개 유예 |
| [Banjo & Ghoraani 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306364/) | Sensors (MDPI) | 주 결과는 환자별 무작위. DS1/DS2는 탐색용 | 180점 beat + RR 4개 | "더 낮다"(수치 없음) | F1 0.804 | Stage-2 89.31 ± 1.51 | 본문 |
| [Mishra 외 2026](https://f1000research.com/articles/15-1668) | F1000Research (심사 전) | DS1/DS2 | Pan-Tompkins feature 11개(RR 포함) | — | — | 약 0.89 (macro-recall 약 0.31) | 본문 |
| (경계) [Mao 외 2022](https://doi.org/10.1109/TBCAS.2022.3185720) | TBioCAS | 칩 위에서 환자별 학습 | 단일 beat | — | — | 93.67 → 97.36 (8 record) | 초록 + 다른 논문의 표 |

SNN 하드웨어 논문들이 "최고 수준"으로 인용하는 97–99 %는 무작위 분할이나 환자별 학습의 수치다. 예는 SparrowSNN 98.61 ([arXiv만](https://arxiv.org/abs/2406.06543)), Scrugli 외 98.4 ([DATE 2024, 학회만](https://past.date-conference.com/proceedings-archive/2024/DATA/950_pdf_upload.pdf)), Xing 98.26 (MDPI)이다. 같은 그룹이 inter-patient로 재면 86–93 %다(노트의 추론). 분할 표기도 틀린 곳이 있다. Scrugli 외는 Amirshahi & Hashemi와 Yan의 정확도를 모두 "inter-patient 결과"로 적었다 ([DATE 2024, 본문](https://past.date-conference.com/proceedings-archive/2024/DATA/950_pdf_upload.pdf)). 그러나 Amirshahi는 각 test 환자 record의 앞 5분을 학습에 넣었다 ([arXiv 본문](https://arxiv.org/abs/1905.02954)). 한 체계적 review는 AAMI class, inter-patient, MIT-BIH, 내장형 구현의 네 조건을 다 만족한 연구가 **5편(4.1 %)**뿐이고 그중 3편이 SNN(Yan, Xing, Mao)이라고 셌다 ([Silva 외 2025](https://arxiv.org/abs/2503.07276)).

SNN에 이웃 beat 정보가 들어간 경우는 둘뿐이다. Banjo & Ghoraani는 RR 숫자 넷을 classifier head에 붙였다. 저자들은 "SNN 기반 ECG classifier는 RR feature를 대체로 빠뜨려 왔다"고 썼다 ([Banjo & Ghoraani 2026, MDPI, 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306364/)). Mishra는 RR을 feature 11개 가운데 하나로 넣었다. **network가 앞 beat의 파형을 직접 보는 SNN은 inter-patient에서 하나도 없다.** 연속 신호를 그대로 흘려 넣는 SNN은 있다. 하지만 모두 무작위, 한 record, 환자별 분할이다 ([Bauer 외 2019](https://arxiv.org/abs/1911.05521); [Liu 2019, 학사 논문](https://arxiv.org/abs/1905.12116); [Payvand 외 2023, 학회만](https://arxiv.org/abs/2305.06941)). Bauer는 학습 stream을 무작위로 뽑은 beat 30,000개를 이어 붙여 만들었다. 그 stream의 리듬은 인공이다(노트의 추론). Spikformer식 ECG 논문과, memory와 attention 두 경로를 가진 SNN ECG 논문은 찾지 못했다.

SNN 행만으로 비교하면 비교가 약해 보인다. 실제 경쟁 상대는 비spiking inter-patient 문헌이다(노트의 추론). SNN 행만 두면 reviewer가 DNN inter-patient 행을 요구할 가능성이 높다(추론).

## Reviewer가 보는 함정: 섞인 sequence, DS2 tuning, record 232

### 가장 무거운 함정: Mousavi & Afghah의 sequence

이 분야에서 이웃 beat를 넣은 가장 높은 수치는 Mousavi & Afghah 2019다. S F1 90.7, V F1 99.7(계산)이다. intra-patient 99.92 %와 inter-patient 99.53 %가 거의 같다 ([arXiv 본문](https://arxiv.org/abs/1812.07421)). 다른 같은 model 비교에서는 inter-patient accuracy가 대개 13–25점 낮다(아래 "같은 방법, 두 분할" 표). 공개 code에 문제가 보고됐다. GitHub issue에 따르면 code는 class별(N, S, V)로 beat를 골라 섞고, 행으로 이어 붙이고, 10개씩 sequence로 잘랐다. 그래서 "sequence 하나의 label이 거의 같고", beat들이 "같은 record에서 온 것이 전혀 아니다". 저자는 "지적이 맞다 … sequence는 같은 ECG record의 beat로 구성돼야 한다. code를 GitHub에 갱신하지 못했다"고 답했다 ([issue #2, GitHub 비공식](https://github.com/MousaviSajad/ECG-Heartbeat-Classification-seq2seq-model/issues/2)). 논문 본문은 반대로 "연속 ECG 신호를 beat sequence로 나눈다"고 적는다. 이 결과를 연속 beat로 다시 평가한 심사 통과 논문은 찾지 못했다. 그래서 near-perfect 수치는 "beat context가 효과 있다"는 증거가 아니다(노트의 추론). 이 model을 확장한 Jiang 2020 ([학회만, 초록](https://vuir.vu.edu.au/46468/))과 capsule + Seq2Seq 2022가 같은 구성을 물려받았는지는 확인하지 못했다.

### 점검표

| 함정 | 누가 지적했나 | 사례 | 지키는 법 |
|---|---|---|---|
| 무작위 beat 분할 | de Chazal 2004; [Luz & Menotti 2011](https://pubmed.ncbi.nlm.nih.gov/22255458/) (학회만, 초록만); Garcia 2017; Cao 외 (arXiv만) | Romdhane 외는 intra 분할인데 inter라고 적었다 ([Cao](https://arxiv.org/abs/2206.14200)) | DS1/DS2 |
| DS2로 model 선택이나 early stopping | de Chazal 2004 | "inter-patient에서는 DS1이 학습, DS2가 test이고 검증 set이 없다" ([issue #14](https://github.com/MousaviSajad/ECG-Heartbeat-Classification-seq2seq-model/issues/14), GitHub 비공식) | DS1 안에서 record 단위 fold로 고른다. [Alforja & Aban](https://github.com/janusaban/interpatient-ecg-classification) (preprint)이 그렇게 했다 |
| test 환자의 data 사용 | de Chazal 2004 (Hu 외 비판); [Teijeiro 표](https://arxiv.org/abs/1609.05632)가 "assisted"를 따로 둠; [Guo](https://arxiv.org/abs/1810.04121)가 비교에서 뺌 | iScience 2024 EXP2는 DS2 record의 앞 5분을 학습에 넣었다. 전문가가 test record의 일부 beat를 고쳐 주는 assisted 방법([de Chazal & Reilly 2006, 초록](https://pubmed.ncbi.nlm.nih.gov/17153211/))은 DS2에서 S Se 92–94까지 간다([Teijeiro 표](https://arxiv.org/abs/1609.05632)) | 다른 표로 분리 |
| test set의 resampling, augmentation | [Cao 외](https://arxiv.org/abs/2206.14200) (Kachuee에 대해); [Mousavi](https://arxiv.org/abs/1812.07421) (Acharya 2017에 대해) | Kachuee test set의 F class 수가 class 전체와 같다 | 학습 set에만 |
| 조각 label을 beat 결과처럼 보고 | 노트의 판단(추론) | Guo 2019의 DS2 SVEB test 표본은 4,225개다. DS2의 S beat는 약 1,836개다 | beat 단위 지표 |
| 미래 정보: post-RR, record 평균, 중심 창, BiLSTM | de Chazal 정의; [Saenz-Cogollo 2020](https://doi.org/10.3390/a13040075) (MDPI, 검색 요약만): record 단위 정규화는 "보행 중이나 실시간 사용에 맞지 않을 수 있다" | record 평균: [Zhang J.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8181174/), [Wang](https://pmc.ncbi.nlm.nih.gov/articles/PMC7831114/), [Mondéjar code](https://github.com/mondejar/ecg-classification/blob/master/python/load_MITBIH.py), [Cheng](https://arxiv.org/abs/2007.04871). ±10 s: [Self-ONN](https://arxiv.org/abs/2204.03768) | 지연과 사용 범위를 밝히고, causal 판을 따로 |
| test record의 label 없는 통계 | 노트의 관찰(추론). 이것을 함정으로 명시한 논문은 찾지 못했다 | Guo 2019는 학습 set과 test set을 "따로" 표준화했다 | 학습 set 통계만. [Mishra](https://f1000research.com/articles/15-1668)가 그렇게 했다 |
| 같은 사람의 201/202가 양쪽에 | [PhysioNet](https://physionet.org/content/mitdb/1.0.0/); [Silva 2025](https://arxiv.org/abs/2503.07276) | 표준 분할 자체 | 둘을 같은 쪽에 둔 판을 덧붙인다(선택) |
| S의 record 쏠림 | de Chazal Table VII; [Alforja & Aban](https://github.com/janusaban/interpatient-ecg-classification) (preprint): "record 232가 DS2 S beat의 75 %"; [EMBC 2019](https://pubmed.ncbi.nlm.nih.gov/31947298/) (학회만, 초록만): 현재 protocol이 S 개선을 과적합 없이 어렵게 만든다 | 작은 방법 변화로 S 점수가 크게 흔들린다(추론) | record별 S 결과, DS2 → DS1 역방향도 보고 |
| accuracy만 보고 | [Garcia 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/) | random forest가 accuracy 94.45 %인데 S beat 1,837개 가운데 26개만 찾았다 ([Alforja & Aban](https://github.com/janusaban/interpatient-ecg-classification), preprint). Mishra는 accuracy 약 0.89, macro-recall 약 0.31 | S, V의 Se / +P / F1과 macro-F1 |
| class 정의와 beat 수 | de Chazal Table V; [Mommen 2026](https://arxiv.org/abs/2609.32854) (arXiv만) | e, j beat를 S에 넣거나, Q(DS2 7개)를 뺀다 | AAMI 대응과 정확한 beat 수를 적는다 |
| 정답 R-peak 사용 | de Chazal 2004 | 거의 모든 논문 | 밝힌다 |

### 같은 방법, 두 분할

| 방법 | intra-patient | inter-patient | 출처 종류 |
|---|---|---|---|
| de Chazal configuration I, DS1 안 fold | beat 단위: accuracy 86.1, S Se 77.6 | record 단위: 79.0, 59.9 | [본문](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf) |
| Ye 외 2012 | 99.3 | 86.4 | [초록만](https://pubmed.ncbi.nlm.nih.gov/22907960/) |
| Kachuee CNN (재구현) | 93.1, S recall 82.6 | 81.2, S recall 0.0 | [Cao 외](https://arxiv.org/abs/2206.14200), arXiv만 |
| Romdhane 외 (재구현) | 82.7 | 62.1 | 같음 |
| Ye 2010 / Yu & Chou 2008 / Song 2005 | 96.50 / 95.4 / 98.7 | 75.2 / 75.2 / 76.3 (S Se 3.2 / 1.8 / 27.0) | 다른 논문의 표 ([Mousavi](https://arxiv.org/abs/1812.07421), 학회만). inter 값은 Luz & Menotti 2011의 재구현으로 보이나 확인하지 못했다 |
| Galvis-Chacón 2026 (SNN) | 98.23 | 86.38 | [검색 요약만](https://www.recercat.cat/handle/20.500.14342/5924) |
| Mousavi & Afghah 2019 | 99.92 | 99.53 | [arXiv 본문](https://arxiv.org/abs/1812.07421). 이상치 |

### 판마다 다른 수치

| 논문 | 한 곳 | 다른 곳 |
|---|---|---|
| Zahid 외 2023 (Self-ONN + RR) | [arXiv v2 표](https://arxiv.org/abs/2204.03768): S Sen 83.37 / Ppr 83.51, accuracy 98.19 | [TBME 초록](https://pubmed.ncbi.nlm.nih.gov/35786545/): S recall 82.50 / precision 82.19 |
| Guo 외 2019 | [초록](https://arxiv.org/abs/1810.04121): S F1 61.25 | 표: 61.94 |
| Sellami & Hwang | [특허 표](https://patents.google.com/patent/US11298067): PPV 48.25 %, sensitivity 90.90 % | 특허 본문: PPV 55.59 %, sensitivity 69.04 % |
| Sellami & Hwang accuracy | [Self-ONN 표](https://arxiv.org/abs/2204.03768): 88.34 (Takalo-Mattila가 89.91) | [Wu 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9481402/) (Hindawi): Sellami 89.91 |
| Wang 외 2021 (MDPI) | [자기 표](https://pmc.ncbi.nlm.nih.gov/articles/PMC7831114/): S 74.56 / 89.54; 초록 accuracy 98.74 | [Self-ONN 표](https://arxiv.org/abs/2204.03768): S 70.75 / 77.0. 자기 confusion matrix로는 accuracy 약 97.48 |
| de Chazal 2004 accuracy | confusion matrix: 85.9 (계산) | [Garcia 표](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/) 83; [Mondéjar README](https://github.com/mondejar/ecg-classification) 0.862; [PeerJ 표](https://pmc.ncbi.nlm.nih.gov/articles/PMC7924512/) 86.24 |
| Niu 외 2020 | [초록](https://pubmed.ncbi.nlm.nih.gov/31545750/): accuracy 96.4 | [Wu 2022 표](https://pmc.ncbi.nlm.nih.gov/articles/PMC9481402/): 95.87 |
| Farag 2023 (MDPI) | [초록](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183)의 S Se 85.3은 model 1 | 최종 model 6은 81.60 |
| PeerJ CS 2020 | [표](https://pmc.ncbi.nlm.nih.gov/articles/PMC7924512/): accuracy 92.53, macro-F1 66.09 | 초록: 96.27 |
| Yan 외 2021 (SNN) | [Silva 표](https://arxiv.org/abs/2503.07276): S Se 67 %, 77 mW | Silva 본문: S 91 %, V 85 % (2단계만의 값으로 보임, 확인 안 됨); [Rana 표](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175061/): 181 mW |
| Mondéjar-Guerra 2019 | [JTEHM 표](https://pmc.ncbi.nlm.nih.gov/articles/PMC9455809/): SVEB Sen 0.95 (열 오류로 보임) | 다른 두 표: 78.1 |
| Mommen 2026 Table 5 | [본문](https://arxiv.org/abs/2609.32854): deep CNN이 V F1 0.943으로 가장 높다 | 표는 0.943을 SVM ensemble에 둔다. 열이 밀린 것으로 보인다 |

## 과거만 보는 순서 변형: 자리는 있지만 RR 숫자 대조가 성패를 가른다

### 먼저 위험 여섯 가지

첫째, 조각은 이미 다 있다. 앞 beat 입력(Sellami 특허), 과거만 쓴 feature(CBM 2026), 과거 창 평균(Farag 2023, Mondéjar code), 다중 beat sequence(Mousavi, Guo)다. "inter-patient + 앞 beat" 자체는 새 주장이 될 수 없다.

둘째, 받아들여진 causal protocol이 없다. 높은 baseline은 모두 다음 beat를 본다. 과거만 보는 model은 다음 beat를 보는 수치와 나란히 놓인다. 같은 model로 "post-RR을 뺐을 때"나 "record 평균을 과거 평균으로 바꿨을 때"를 잰 논문도 찾지 못했다. 그래서 causal의 비용을 모른다. 유일한 과거만의 결과는 낮다(CBM 2026, 네 class macro F1 0.493, 초록만).

셋째, 순서가 필요 없을 수 있다. S 판정의 주된 단서는 "이 beat가 이 환자의 최근 리듬보다 일찍 왔다"이다. 이것은 앞 RR 하나와 과거 평균의 비율로 계산된다. 순서는 필요 없다(추론). RR만 쓴 XGBoost도 S F1 45였다 ([Farag 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183)). 연구실 기록도 같은 쪽이다. 지금 MIT-BIH에서 순서를 못 보는 model은 0.67 %p 안에 왔다. CinC에서 Neocortex가 순서로 얻은 몫은 +1.65 / +1.25 %p였다 ([MODI 기록 11-03, 11-19](../../docs/MODI_PROJECT_LOG.md)).

넷째, DS2의 S 점수는 사실상 한 환자다. record 232 하나가 DS2 S beat의 75.2 %(계산)다. 어떤 이득이 나와도 "232에서의 이득"일 수 있다.

다섯째, data를 새로 만들어야 한다. Kachuee 형식은 그 beat의 P파와 pre-RR이 없다. 같은 형식의 CNN은 DS1 → DS2에서 S를 하나도 못 찾았다 ([Cao 외, arXiv만](https://arxiv.org/abs/2206.14200)). 저장소의 환자 단위 파일은 출처 기록이 없다.

여섯째, 비교 상대가 바뀐다. 지금의 MIT-BIH SNN baseline은 대부분 무작위 분할이다. 새 설정에서는 SNN 행이 다섯 편뿐이고, reviewer는 DNN inter-patient 행을 볼 것이다(추론).

### 비어 있는 자리

그래도 빈자리는 분명하다. 세 갈래의 조사(방법, protocol, SNN)가 같은 곳을 가리킨다. inter-patient에서 network가 앞 beat를 직접 보는 SNN은 없다. beat 순서나 RR 순서를 섞는 대조는 어느 분할에서도 없다. causal과 비causal을 같은 model로 비교한 결과도 없다. 그리고 순서가 도울 만한 자리가 정성적으로 알려져 있다. S beat가 이어지는 run에서는 첫 beat만 일찍 온다(Self-ONN). 그 record들(232, 222)이 DS2 S 점수를 좌우한다. 이것은 "run이 언제 시작했는가"라는 순서 있는 과거의 문제다(추론). causal은 SNN의 주장과도 맞는다. Yan 외는 "대부분의 착용형 기기는 복잡한 전처리를 못 한다"는 이유로 raw beat를 썼다 ([BSPC 2021, 초록](https://doi.org/10.1016/j.bspc.2020.102170)). 다음 beat를 기다리지 않는 저전력 streaming 판정은 그 동기의 자연스러운 연장이다(추론).

### 설계와 대조 (전부 추론)

입력은 raw MIT-BIH record에서 새로 자른다. 현재 beat의 창은 다음 R-peak 전에 끝낸다. 그 앞에 과거 K개 beat를 둔다. token 하나를 beat 하나로 두면 Neocortex가 beat를 한 step씩 읽는다(T = N). 파형을 고정 길이로 맞추면 RR이 지워진다. 그래서 각 beat token에 그 beat의 pre-RR을 함께 싣는다. 입력을 과거로 끊으면 attention 경로도 미래를 못 본다. 그러면 두 경로의 차이는 미래 접근이 아니라 순서와 축적 방식에서만 나온다.

| 대조 | 무엇을 가르나 | 선례와 메모 |
|---|---|---|
| K = 0, 1, 4, 16, 64 | 축적 곡선. 이득이 몇 beat에서 포화하나 | Lee 2025는 4 beat에서 최고였다(DS1/DS2 아님, 초록만) |
| 과거 beat 순서를 표본마다 섞어 재학습 (현재 beat는 끝에 고정) | 순서. beat 파형과 RR의 묶음은 그대로 두고 배열만 지운다 | DS1/DS2 선례 없음 |
| 위치 정보 없는 attention + 평균 pooling, 같은 token | 순서 없이 어디까지 가나 | 출력이 섞어도 정확히 같은지 먼저 확인 |
| 단일 beat + causal RR 숫자(pre-RR, 과거 창 평균) | 순서 없는 요약으로 충분한가. **가장 중요한 대조** | Farag 2023 형식에서 post-RR을 뺀 것 |
| memory-less Neocortex | 축적이 neuron state에 있나 | 순서는 못 가른다 |
| 한 beat 지연판(post-RR 추가)과 중심 창판 | 발표 수치와 잇고, causal의 비용을 잰다 | 같은 model의 비교 선례 없음 |
| DS2 → DS1 역방향, record별 S (232, 222) | record 232 한 명의 영향 | Self-ONN은 DS2 → DS1에서 S Sen 79.43 / Ppr 80.97을 보고했다 ([arXiv판](https://arxiv.org/abs/2204.03768)) |

학습과 선택은 DS1 안에서 record 단위 fold로 한다. DS2는 마지막에 한 번 연다. 지표는 AAMI 규칙의 S, V Se / +P / F1, macro-F1, record별 S다. beat 수는 de Chazal Table II와 맞춰 적는다. 정규화 통계는 학습 set에서만 낸다.

| 결과 | 쓸 수 있는 문장 |
|---|---|
| causal RR 숫자 baseline이 다중 beat model과 같다 | 이웃 beat의 값은 앞 RR과 과거 평균 리듬으로 요약된다. 순서와 긴 축적은 주장할 수 없다 |
| K가 클수록 좋고, 섞어 재학습해도 같다 | 축적만 보였다. "과거 beat를 더 볼수록 좋다"까지다 |
| 섞어 재학습하면 떨어지고, 위치 정보 없는 attention이 따라오지 못한다 | Neocortex가 beat 순서를 쓴다. 이득이 232와 222의 S run에 몰려 있는지 함께 적는다 |
| 위치 정보를 준 attention이 따라온다 | 순차 memory가 꼭 필요한 것은 아니다. 주장은 energy 쪽으로 옮겨 간다 |
| causal 판이 한 beat 지연판보다 크게 낮다 | 실시간 판정의 비용을 같은 model로 처음 잰 결과로 보고한다 |

## 조사로 확인하지 못한 것

| 확인하지 못한 것 | 왜 중요한가 |
|---|---|
| Sellami & Hwang 2019의 단일 beat 대 다중 beat 수치 (특허는 "개선할 수 있다"고만 적음) | 가장 가까운 causal 다중 beat 선례의 효과 크기 |
| Niu 2020 ablation 수치, Lin & Yang 2014의 정규화 RR 비교, Mondéjar-Guerra 2019의 feature 종류별 결과 | RR만으로 어디까지 가는지의 다른 근거. 전문이 열리지 않았다 |
| Jiang 2020과 capsule + Seq2Seq 2022가 Mousavi의 class별 sequence 구성을 물려받았는지 | 높은 seq2seq 수치를 baseline으로 쓸 수 있는지 |
| Mousavi model을 연속 beat로 다시 평가한 논문 | GitHub 토론 말고는 근거가 없다 |
| 201/202 누출, record 평균 정규화, test set 표준화가 DS2 지표에 주는 영향 | 정량화한 재분석을 찾지 못했다 |
| 널리 인용되는 45,866 / 944 / 3,788 beat 수의 출처 | beat 수 보고의 기준 |
| Galvis-Chacón 2026의 S, V 수치와 RR 사용 여부, Banjo & Ghoraani의 SVEB 수치(부록), Yan 2021 원문 표 | SNN baseline의 S 선 |
| Chu 2021·2022, Corradi 2019의 분할 | SNN 수치의 분류 |
| 2024–2026년 deep inter-patient 논문 (예: Sci Rep 2026 계층형, S Se 93.23 %라는 검색 요약; +P 미확인) | 최신 DNN 선이 더 높을 수 있다 |
| Luz 외 2016 survey의 함정 목록, Luz & Menotti 2011의 수치 | 함정 목록의 원 출처 |
| MIT-BIH rhythm annotation(이단맥 등)으로 순서 정의 class를 만들 수 있는지 | 순서가 정의상 필요한 label의 후보. 이전 노트에서도 확인하지 못했다 |
| DS2만의 N 비율 | accuracy의 기준선 |

## 결론

질문이 바뀌었다. "inter-patient에 이웃 beat를 넣은 연구가 있는가"의 답은 "분할의 출발부터 있었다"다. 남은 질문은 "과거 beat의 배열이 앞 RR 하나와 과거 평균 RR 이상의 정보를 주는가"다. 이것은 아무도 재지 않았다. 그래서 그 측정 자체가 논문의 한 부분이 될 수 있다. 다만 결과가 "RR 숫자로 충분하다"로 나와도 받아들일 준비가 필요하다. 이 문헌에서 S를 살린 것은 거의 언제나 RR 숫자였기 때문이다.

순서가 이득을 낼 자리와 점수가 결정되는 자리가 같다는 점이 이번 조사의 가장 쓸모 있는 발견이다. DS2 S 점수의 대부분은 record 232와 222에서 나온다. 그 record들은 S beat가 이어지고 첫 beat만 일찍 오는 곳이다. Neocortex의 순서 사용이 어디선가 보인다면 거기서 보일 것이다. 동시에 그 이득은 사실상 한 환자의 것이다. 그러니 이 설정을 고른다면 세 가지를 처음부터 함께 낸다. 과거 beat를 섞어 다시 학습한 대조, causal RR 숫자 baseline, record별 결과와 DS2 → DS1 역방향이다. 이 셋 없이 낸 SNN의 DS2 수치는 순서에 대해 아무것도 말해 주지 못한다.
