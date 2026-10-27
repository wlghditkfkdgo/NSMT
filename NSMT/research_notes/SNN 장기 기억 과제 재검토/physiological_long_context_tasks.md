# 긴 연속 생체·센서 신호 과제 중 "맥락(기억)이 길수록 성능이 오른다"는 근거가 있는 과제: 스파이킹 시계열 모델 벤치마크 후보 (2023–2026 프로토콜 중심)

작성: 2026-10-27. 범위: ECG(심방세동 구간 검출, 수면무호흡), EEG(수면 단계, 발작 검출), PPG 심박 추적, 손목 가속도 활동 인식. 이벤트 기반 데이터(SHD, SSC, DVS)는 범위 밖이라 제외했다.

표기 규칙
- 숫자는 모두 이 세션에서 직접 연 자료(PhysioNet 페이지, arXiv PDF 전문, PMC 전문, OpenAlex API로 읽은 초록, PMLR 페이지)에서 가져왔고 줄마다 URL을 달았다. "초록만 확인"은 OpenAlex API로 초록만 읽고 본문은 못 연 경우다.
- "배경(2023 이전)"은 정의용 옛 자료다. "IEEE 저널"은 ML 최상위 학회 자료가 부족해 쓴 의공학 저널 자료라는 뜻이다.
- 용어 풀이
  - 에폭(epoch, 수면): 수면 채점 단위인 30초 구간. 학습 반복 횟수(epoch)와 다르다.
  - 시퀀스-투-시퀀스(seq2seq) 채점: 연속된 여러 에폭을 한꺼번에 넣고 각 에폭의 단계를 모두 출력하는 방식.
  - LOSO: 한 피험자(또는 세션)를 빼고 학습한 뒤 그 사람으로 시험하는 교차검증.
  - macro-F1(MF1): 클래스별 F1의 단순 평균. Cohen's kappa(κ): 우연 일치를 뺀 일치도.
  - 이벤트 기반 F1(발작): 발작 "사건" 단위로 맞음/놓침/오경보를 세어 낸 F1. 표본 기반 점수는 1초 단위 라벨을 하나씩 비교한다.
  - HMM 평활(smoothing): 창(window)별 예측 뒤에 은닉 마르코프 모델로 "상태는 잘 안 바뀐다"는 사전지식을 얹어 예측열을 고르는 후처리. 정보를 새로 더하기보다 라벨이 이어지는 성질을 쓰는 것이다.
  - T = N: 우리 모델의 neocortex 경로가 원시 시계열 토큰(패치나 샘플) 하나를 스파이킹 시간 단계 하나에 맞추는 방식. 뉴런 상태가 토큰 사이로 정보를 나른다.

---

## Q1. ECG: 긴 기록에서 심방세동(AF)·부정맥 구간 검출 (AFDB, LTAFDB, Icentia11k, CPSC 2021, Holter)

### Takeaway
AF를 창 단위로 분류하는 과제는 10~30초 창으로 이미 거의 천장(AFDB F1 0.986)에 닿아 있어서, 기억이 필요하다는 주장을 시험하기 어렵다. 맥락이 의미 있게 작동하는 판은 셋이다. (a) 기록 전체를 보고 지속성 AF, 발작성 AF, 비AF를 가르면서 시작·끝 지점까지 찍는 CPSC 2021, (b) 고정된 표현을 선형 분류기로만 평가하는 자기지도 학습에서 16초 대신 5~10분 창을 쓰는 경우(Icentia11k, AUPRC 0.712 → 0.877), (c) 환자 식별. 이웃 창을 BiGRU로 묶는 방식의 이득은 "조금"에 그쳤다.

### Cited Findings
**데이터베이스 사실 (PhysioNet 페이지 직접 확인)**
- AFDB: 장기 ECG 25건(대부분 발작성 AF). 23건에 신호 2채널이 있다. 기록당 10시간, 250 Hz, 12-bit. 리듬 주석은 수작업(AFIB, AFL, J, N)이고, 박동 주석(.qrs)은 자동 검출이라 검수되지 않았다. 일부 기록에는 수정본 .qrsc가 있다. 라이선스 ODC-By 1.0, 압축 해제 605.9 MB — [PhysioNet AFDB](https://physionet.org/content/afdb/)
- LTAFDB: 발작성 또는 지속성 AF 환자의 장기 ECG 84건. 2채널, 128 Hz, 기록은 대개 24~25시간이다. atr 주석은 MEDICALgorithmics 시스템 출력을 기술자가 검수한 것이고, AF Termination Challenge(2004)가 이 기록의 1분 발췌를 썼다. ODC-By 1.0, 3.4 GB — [PhysioNet LTAFDB](https://physionet.org/content/ltafdb/)
- Icentia11k: 환자 11,000명, 라벨된 박동 20억 개. 가슴 부착 단일 리드 250 Hz, 최대 2주(대부분 1주) 착용. 박동 라벨은 N/PAC/PVC, 리듬 라벨은 NSR/AF/AFL이다. 기술자 20명이 기록 전체를 보며 주석했다. CC BY-NC-SA 4.0(비상업), 1.1 TB — [PhysioNet Icentia11k](https://physionet.org/content/icentia11k-continuous-ecg/)
- CPSC 2021(발작성 AF 이벤트 검출): 12-lead Holter 또는 3-lead 웨어러블에서 뽑은 lead I/II, 200 Hz, 길이는 기록마다 다르다. AF 에피소드는 최소 5박동이다. 1단계 학습셋은 730건(AF 환자 12명 중 발작성 5명, 비AF 42명), 2단계 학습셋은 706건(AF 37명 중 발작성 18명, 비AF 14명)이다. 시험셋은 공개하지 않으며, 적어도 한 부분은 다른 기기로 수집됐다. CC BY 4.0, 1.3 GB — [PhysioNet CPSC 2021](https://physionet.org/content/cpsc2021/)
- CPSC 2021 채점은 두 단계다. 먼저 기록을 N / 지속성 AF / 발작성 AF로 분류해 점수 행렬로 Ur를 매기고, 다음으로 예측한 시작·끝 지점이 주석의 ±1박동 안이면 +1, ±2박동 안이면 +0.5를 준다(Ue). 최종 점수는 U = (1/N) Σ (Ur_i + M_a,i / max(M_r,i, M_a,i) × Ue_i)이고, 에피소드를 과다 예측하면 깎인다 — [PhysioNet CPSC 2021](https://physionet.org/content/cpsc2021/)
- SHDB-AF(일본 Holter, 2024 공개): 발작성 AF 환자 122명, 기록 128건, 그중 98건에 박동 단위 리듬 주석이 있다. ODC-By, 8.3 GB. 같은 페이지가 공개 AF 데이터로 MITDB, AFDB, LTAFDB, IRIDIA-AF, Icentia11k, CPSC2021을 든다 — [PhysioNet SHDB-AF](https://physionet.org/content/shdb-af/)

**프로토콜과 최근 성능**
- RawECGNet(2023, arXiv, IEEE 저널 투고본): 기록을 겹치지 않는 30초 창으로 자르고, 창 안에서 가장 많이 나온 리듬을 라벨로 삼으며 200 Hz로 재표본한다. 원천 데이터 UVAF는 환자 2,242명, 51,386시간이고 환자 단위로 학습 2,137명 / 시험 105명으로 나눴다. 외부 시험셋은 RBDB(116명)와 SHDB(100명)다. 외부 F1은 RBDB 0.91~0.94, SHDB 0.93으로, RR 간격 기반 ArNet2(RBDB 0.89~0.91, SHDB 0.91)보다 높다. ArNet2의 입력은 60박동 창이다 — [Ben-Moshe et al., RawECGNet](https://arxiv.org/abs/2401.05411)
- RawECGNet의 2단계는 앞 p개, 뒤 s개 창의 특징을 BiGRU로 묶는다. 절제 실험에서 BiGRU를 더하면 성능이 "slightly" 올랐고, 큰 이득은 다중 리드 학습과 DSU 층에서 나왔다 — [RawECGNet](https://arxiv.org/abs/2401.05411)
- Kraft & Rumm(Sensors 2025, 초록만 확인): 10초 창 하나를 보는 1D ConvNeXtV2(파라미터 770k)를 Icentia11k, CPSC-2018/2021, LTAF, PTB-XL, PCC-2017로 학습하고 MIT-AFDB를 외부 시험셋으로 썼다. AFDB F1은 0.986이다 — [doi:10.3390/s25134109](https://doi.org/10.3390/s25134109)
- Icentia11k 맥락 길이 통제 실험(arXiv 2026-08, EMBC 2026 채택이라고 표기됨). 입력 길이를 16초, 1분, 5분, 10분으로 바꾸고 Transformer 백본과 학습 절차는 고정했다. 하류 과제는 AF/AFL 대 정상 분류(양성 창 약 7%)이고, 학습에는 환자당 라벨 창 1개(1,100명 → 1,100창)를 쓴다 — [Sameh et al.](https://arxiv.org/abs/2608.12695)
  - 자기지도 학습 + CNN 패치 임베딩, 인코더 고정(선형 분류기만 학습): 16초 AUC 0.977 / AUPRC 0.712, 10분 AUC 0.980 / AUPRC 0.877. 환자 검색 Recall@1은 0.737 → 0.907 — [Sameh et al., Table I](https://arxiv.org/pdf/2608.12695)
  - 같은 조건에서 인코더까지 미세조정하면 16초 AUC 0.995 / AUPRC 0.946, 10분 AUC 0.989 / AUPRC 0.960이다. 16초 모델의 예측을 37개 창(592초)에 걸쳐 평균하면 AUC 0.991 / AUPRC 0.903이다. 저자들은 가장 큰 이득이 5~10분 창에서 나온다고 썼다 — [Sameh et al.](https://arxiv.org/pdf/2608.12695)
- Hu et al.(J Pers Med 2023, 초록만 확인, 저널 등급 낮음): CPSC2021로 학습한 ResNet + Transformer의 AF 리듬 성능은 정확도 98.67%, 민감도 87.69%, 특이도 98.56%다. 시작점 검출 민감도는 95.90%, 끝점은 87.70%이고 FPR은 0.46%다 — [doi:10.3390/jpm13050820](https://doi.org/10.3390/jpm13050820)
- 2023 TBME 리뷰는 보행 ECG 기반 AF 검출의 주요 난점으로 신호 품질 평가, QRS 검출, 다른 이상 리듬이 섞일 때의 AF 식별을 꼽는다(초록만 확인) — [Ma et al., IEEE TBME 2023](https://doi.org/10.1109/tbme.2023.3321792)

**스파이킹 / 뉴로모픽 사용 여부**
- Banerjee et al.(IJCNN 2024, 초록만 확인)은 피크 기반 "Spiking Hamilton" 인코더와 저수지형 SNN, 순방향 SNN으로 ECG의 AF를 분류했다. 데이터셋은 "five well known ECG dataset"이라고만 적혀 있어 이름을 확인하지 못했다 — [doi:10.1109/ijcnn60899.2024.10650023](https://doi.org/10.1109/ijcnn60899.2024.10650023)
- Bauer, Muir, Indiveri(IEEE TBioCAS 2019, 배경, 초록만 확인)는 다채널 ECG를 비동기 이벤트로 바꾸고, 저수지 방식 스파이킹 순환망을 DYNAP-SE에서 돌려 정상 박동과 병적 리듬을 실시간으로 구분했다 — [doi:10.1109/tbcas.2019.2953001](https://doi.org/10.1109/tbcas.2019.2953001)
- PPG 기반 AF 검출 SNN 칩(IEEE Access 2026, 초록만 확인): MIMIC III와 MIMIC PERform AF에서 정확도 98.42%, 40 nm ASIC, 0.018 mW — [doi:10.1109/access.2026.3658395](https://doi.org/10.1109/access.2026.3658395)
- 대조: 대표적인 SNN ECG 연구는 박동 단위 MIT-BIH 분류다. 예를 들어 Chu et al.(TBioCAS 2022)은 5클래스 98.22%, 분류당 0.75 μJ를 보고했다(초록만 확인) — [doi:10.1109/tbcas.2022.3189364](https://doi.org/10.1109/tbcas.2022.3189364)

### Inferences
- AFDB 창 단위 AF 분류는 10초 창으로 F1 0.986이니, 기억 경로가 낄 여지가 거의 없다. AFDB/LTAFDB를 쓰려면 과제를 "시작·끝 지점 검출"이나 "AF 부담(시간 비율) 추정"으로 바꿔야 맥락이 의미를 갖는다.
- CPSC 2021은 기록 단위의 세 클래스 판정(지속성 vs 발작성 vs 비AF)과 박동 단위의 경계 검출을 함께 요구한다. 이 판정은 기록 전체를 봐야 한다는 뜻이라, 맥락이 정보를 더하는 구조가 과제 정의 자체에 들어 있다. 다만 공식 시험셋이 비공개라 학습셋 1,436건을 환자 단위로 직접 나눠야 한다. 파일명 `data_<환자>_<기록>` 형식에서 환자 번호를 읽을 수 있을 것으로 보이나, 검증은 하지 않았다.
- Icentia11k 결과는 긴 맥락의 이득이 "고정 표현 + 선형 분류" 조건에서 크고(AUPRC +0.165), 끝단까지 미세조정하면 작아진다(AUPRC +0.014, AUC는 오히려 −0.006)는 점을 보여준다. 기억을 시험하려면 평가 조건을 미리 고정해야 한다.
- SNN 논문 가운데 AFDB/LTAFDB 구간 검출, CPSC 2021, Icentia11k를 환자 단위 분할로 쓴 사례는 OpenAlex 제목 검색에서 찾지 못했다. 이 영역은 비어 있는 셈이다(근거는 검색 결과 없음일 뿐, 존재하지 않는다는 증명은 아니다).

### Gaps
- CPSC 2021 공식 순위와 우승 점수(U)는 찾지 못했다(챌린지 요약 논문을 열지 못함).
- AF 검출에서 창 길이(예: 10/30/60/100박동)를 바꾼 2023–2026 절제 연구를 직접 열어 확인하지 못했다. ArNet2의 60박동 선택 근거도 미확인이다.
- Kraft & Rumm의 AFDB 평가가 기록 전체 외부 시험인지는 초록만으로는 알 수 없다(MDPI 페이지 403).
- Holter 기반 다중 부정맥(예: 심실빈맥 예측) 공개 벤치마크는 조사하지 못했다.

---

## Q2. EEG: 수면 단계 분류(다중 에폭 맥락)와 발작 검출·예측

### Takeaway
수면 단계 분류는 맥락 효과가 가장 잘 측정된 과제다. 에폭 1개에서 약 15개(7.5분)로 늘리면 macro-F1이 0.752에서 0.796으로 뚜렷하게 오른다. 반면 15~20에폭을 넘어 200에폭(100분)까지 늘려도 이득은 없거나(S4Sleep: 차이 유의하지 않음) 작다(L-SeqSleepNet: 정확도 +0.3~1.2%p). 2026년 연구는 이 이득의 대부분이 "국소 평활"에서 온다고 주장한다. 발작 검출은 긴 클립이 일관되게 낫지 않았고(TUSZ에서 60초 클립이 12초보다 대개 나쁨), 보류 데이터에서 최상위 이벤트 F1도 32%에 그쳐 기억 효과를 따로 떼어 보기 어렵다.

### Cited Findings
**수면 단계 — 데이터**
- Sleep-EDF Expanded: 밤샘 PSG 197건. EEG(Fpz-Cz, Pz-Oz)와 EOG는 100 Hz다. SC(Sleep Cassette) 153건은 1987–1991년 25~101세 건강인을 집에서 연속 두 주야 동안 각각 약 20시간씩 기록했다. ST(Sleep Telemetry) 44건은 temazepam 연구 22명의 병원 기록이다. 채점은 R&K 기준(W, R, 1, 2, 3, 4, M, ?)이다. ODC-By, 8.1 GB — [PhysioNet Sleep-EDFx](https://physionet.org/content/sleep-edfx/)
- 흔히 쓰는 하위셋 SleepEDF-20은 20명 39건이고 LOSO로 평가한다(L-SeqSleepNet Table II). SHHS는 5,791명으로 큰 데이터이며, L-SeqSleepNet은 그중 5,463명을 70/30으로 나눠 썼다 — [Phan et al., L-SeqSleepNet](https://arxiv.org/abs/2301.03441)

**수면 단계 — 맥락 길이 근거**
- L-SeqSleepNet(IEEE JBHI 2023, IEEE 저널) 서론: 여러 연구가 20~30에폭(10~15분)을 효과적이라고 봤고, 그보다 긴 시퀀스는 계산만 크게 늘고 이득은 거의 없다고 보고돼 왔다. 수면 주기 하나는 약 180~220에폭이다 — [L-SeqSleepNet](https://arxiv.org/pdf/2301.03441)
- 같은 논문 Table I(SHHS 정확도): SeqSleepNet은 20 / 100 / 200에폭에서 87.2 / 87.4 / 87.5, SleepTransformer는 21 / 100 / 200에폭에서 87.7 / 86.6 / 86.3이다. Transformer는 길어질수록 오히려 나빠졌다 — [L-SeqSleepNet](https://arxiv.org/pdf/2301.03441)
- L-SeqSleepNet은 긴 시퀀스를 하위 시퀀스로 접어 처리한다. Table IV(SHHS)에서 L = 100(10×10)은 정확도 88.1 / κ 0.833 / MF1 80.7, L = 200(20×10)은 88.4 / 0.837 / 81.6, L = 400(20×20)은 88.4 / 0.838 / 81.4다. 1,000스텝당 학습 시간은 354 / 449 / 605초이고, SeqSleepNet을 L = 200으로 돌리면 1,465.6초가 걸린다. 저자들은 "한 수면 주기의 정보면 충분할 것"이라고 결론 내렸다 — [L-SeqSleepNet](https://arxiv.org/pdf/2301.03441)
- L-SeqSleepNet Table III의 최종 성능: SHHS는 정확도 88.4 / κ 0.838 / MF1 81.4로 SeqSleepNet 87.2 / 0.820 / 80.2, XSleepNet2 87.5 / 0.826 / 81.0보다 높다. SleepEDF-20(LOSO)에서는 사전학습 없이 86.3 / 0.813 / 79.3, SHHS 사전학습 후 88.6 / 0.845 / 82.9이고, SeqSleepNet은 85.6 / 0.803 / 78.6이다 — [L-SeqSleepNet](https://arxiv.org/pdf/2301.03441)
- S4Sleep 설계 공간 연구(arXiv, v3 2025-01): 같은 계열 모델에서 단일 에폭 입력(CNN+S4(TS), 단일 채널)은 SEDF(full) MF1 0.752다(Table D.10). 15에폭 입력 S4Sleep(ts)는 0.796 ± 0.004다(Table 5). 저자들은 모든 비교에서 단일 에폭과 다중 에폭의 차이가 유의했다고 썼다 — [Wang & Strodthoff, S4Sleep](https://arxiv.org/abs/2310.06715)
- 장거리 상관 재검토(Biosignal 2024 워크숍, 3쪽): S4Sleep(TS)를 SEDF 197건, 8:1:1 분할로 학습하면서 입력을 점진적으로 늘렸다. MF1은 10에폭 0.790, 20에폭 0.794, 40에폭 0.795, 50에폭 0.796, 100에폭 0.794, 200에폭 0.793이고, 부트스트랩 95% 구간 기준으로 유의한 차이가 없다. 200에폭으로 처음부터 학습하면 0.723으로 크게 떨어진다. 저자들은 "very long-range interactions"의 진단적 가치에 의문을 제기했다 — [Wang & Strodthoff 2024](https://arxiv.org/pdf/2402.17779)
- "Random Transformers as Adaptive Sequence Smoothers"(arXiv 2026-05, 초록 페이지만 확인): 학습하지 않은 무작위 Transformer만으로도 수면 단계 성능이 오르고 경험적 평활보다 낫다고 보고한다. 저자들은 Transformer 기반 채점의 이득 대부분이 학습된 장거리 의존성이 아니라 구조적 귀납 편향(국소 평활)에서 온다고 주장한다. 수치와 데이터셋은 초록에 없다 — [Liu et al.](https://arxiv.org/abs/2605.09905)
- 다른 표기의 비교값(S4Sleep Table 5): SleepTransformer는 SEDF-SC 153건, 10-fold에서 MF1 0.743 / 정확도 0.814다. S4Sleep(ts)는 SEDF-20 LOSO에서 MF1 0.804 / 정확도 0.876, 단일 채널이다 — [S4Sleep](https://arxiv.org/pdf/2310.06715)

**수면 단계 — 스파이킹 사용**
- PicoSleepNet(IEEE JBHI 2025, 초록만 확인): level-crossing으로 만든 스파이크를 희소 순환 SNN에 넣는다. 정확도는 Sleep-EDF-20 / Sleep-EDF-78 / ISRUC에서 83.5 / 77.9 / 79.4%, MF1은 75.2 / 68.1 / 77.2%다. 파라미터는 14.0~25.8 K다 — [doi:10.1109/jbhi.2025.3602502](https://doi.org/10.1109/jbhi.2025.3602502)
- 제목만 확인한 SNN 수면 논문: "Sleep Stage Classification with SNN and Transformers using Multi-Channel EEG"(CCC 2024) — [doi](https://doi.org/10.23919/ccc63176.2024.10662685), "Hybrid spiking neural network for sleep EEG signals"(Sci China Inf Sci 2022) — [doi](https://doi.org/10.1007/s11432-021-3380-1)

**발작 — 데이터와 프로토콜**
- CHB-MIT: 소아 22명에서 나온 23케이스. 256 Hz, 대부분 EEG 23채널, 10-20 배치다. 파일은 대개 1시간짜리(일부 2시간, 4시간)이고 파일 사이에 끊김이 있다. .edf 664개 중 발작 포함 129개, 발작 198회. ODC-By, 42.6 GB — [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/)
- SzCORE(Epilepsia 2024, arXiv 2402.13005)가 정리한 공개 데이터는 넷이다. CHB-MIT 23명 / 982시간 / 발작 198회 / 256 Hz, TUH 675명 / 1,476시간 / 4,029회 / 평균 파일 10분, Siena 14명 / 128시간 / 47회, SeizeIT1 42명 / 4,211시간 / 182회 — [Dan et al., SzCORE](https://arxiv.org/pdf/2402.13005)
- SzCORE 채점 규칙: 표본 기반 점수는 1 Hz 라벨을 쓰고 겹침이 50%를 넘으면 발작으로 본다. 이벤트 기반 점수는 조금이라도 겹치면 검출로 인정한다. 발작 시작 전 30초, 끝난 뒤 60초까지 허용하고, 90초보다 가까운 사건은 합치며, 5분이 넘는 사건은 나눈다. 개인별 모델은 피험자당 발작 3회 이상, 1시간 30분 이상의 데이터가 필요하다. 이 경우 시간순 교차검증을 쓰는데, 처음 5시간 이상과 발작 1회 이상으로 학습한 뒤 다음 1시간을 평가하는 일을 반복한다 — [SzCORE](https://arxiv.org/pdf/2402.13005)
- TUH EEG Corpus는 서명한 신청서를 이메일로 보내야 접근할 수 있다 — [NEDC TUH EEG 페이지](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)

**발작 — 맥락 길이 근거**
- Tang et al.(ICLR 2022, 배경 직전 연도): TUSZ v1.5.2를 쓴다. 학습셋은 파일 4,599개 / 환자 592명 / 45,174.72분, 시험셋은 900개 / 45명 / 9,031.58분이고, 시험 환자는 학습에 없다. 12초 클립과 60초 클립의 검출 AUROC(5회 평균)는 다음과 같다 — [Tang et al.](https://arxiv.org/pdf/2104.08336)
  - Dense-CNN 0.812 → 0.796, LSTM 0.786 → 0.715, CNN-LSTM 0.749 → 0.682
  - Dist-DCRNN(사전학습 없음) 0.824 → 0.793, Corr-DCRNN(사전학습) 0.861 → 0.850, Dist-DCRNN(사전학습) 0.866 → 0.875
  - 즉 60초가 나은 경우는 사전학습한 Dist-DCRNN 하나뿐이었다.
- SzCORE 챌린지(ICML 2026 / PMLR 306): 비공개 보류 데이터는 65명, 4,360시간(피험자당 18~98시간)이고 28개 알고리즘을 평가했다. 1위 SzTransformer는 이벤트 F1 32%, 민감도 37%, 정밀도 29%, 하루 오경보 1.34회다. 상위 5개의 민감도는 30~58%, 정밀도는 19~29%, 하루 오경보는 1.34~14회다 — [Dan et al., PMLR](https://proceedings.mlr.press/v306/dan26a.html); [arXiv 2505.18191](https://arxiv.org/pdf/2505.18191)
- 같은 챌린지에서 제출 모델의 입력 창은 1초부터 600초까지 다양했다. 600초 입력의 DeepSOZ-HEM은 F1 30%(민감도 58, 정밀도 21, 하루 오경보 14), 12초 입력의 S4Seizure v1은 F1 27%였다. 잘 놓친 발작은 짧았다(중앙값 48초, 잘 잡힌 발작은 118초). 개발자 자체 보고 성능은 보류 데이터 성능보다 높았다 — [arXiv 2505.18191](https://arxiv.org/pdf/2505.18191)
- 검색 요약은 같은 논문 v1(2025-05)이 1위 F1을 43%로 적었다고 전하지만 v1을 직접 열지는 않았다. 현행판과 PMLR판의 32%를 기준으로 삼는다 — [arXiv 2505.18191](https://arxiv.org/abs/2505.18191)

**발작 — 스파이킹 사용 (모두 초록만 확인)**
- Yang, Eshraghian et al.(Neuromorph. Comput. Eng. 2023): 스파이킹 ConvLSTM. leave-one-out AUC는 Freiburg 92.7%, CHB-MIT 89.0%, EPILEPSIAE 81.1% — [doi:10.1088/2634-4386/acbab8](https://doi.org/10.1088/2634-4386/acbab8)
- Zhang et al.(Front. Neurosci. 2024): 순환 스파이킹 합성곱 구조 EESNN으로 환자 간 발작 검출을 했다. ANN과 비슷하거나 낫다고 주장한다 — [doi:10.3389/fnins.2023.1303564](https://doi.org/10.3389/fnins.2023.1303564)
- Chen, Liu et al.(ISCAS 2024) Spiking Conformer: CHB-MIT 원시 EEG에서 검출 민감도 94.9% / 특이도 99.3%, 예측 96.8% / 89.5%. 비스파이킹 모델보다 연산이 10배 넘게 적다 — [doi:10.1109/iscas58744.2024.10558341](https://doi.org/10.1109/iscas58744.2024.10558341)

### Inferences
- 수면 단계의 맥락 이득은 거의 "1 → 10~20에폭" 구간에 몰려 있다(MF1 +0.044). 길이는 5~10분이고, 수면 단계가 몇 분씩 이어지는 성질과 맞물린다. 그래서 T = N 상태가 이 이득을 재현해도 그것이 "기억"인지 "평활"인지 가리려면 대조군이 꼭 필요하다. 하나는 에폭별 단독 모델에 HMM이나 중앙값 평활을 얹은 것, 다른 하나는 같은 맥락 길이의 self-attention이다.
- 장기(수면 주기 규모) 맥락에서 L-SeqSleepNet이 본 +1.2%p는 S4Sleep 재검토에서 재현되지 않았다. 따라서 "100분 기억이 필요하다"는 주장은 근거가 약하다. 이것도 미리 등록할 만한 가설이다.
- 발작 검출은 (a) 긴 클립이 일관되게 낫지 않고, (b) 이벤트 단위 지표의 절대 수준이 낮고 편차가 크며, (c) 데이터가 크다(CHB-MIT 42.6 GB, TUH 1,476시간). 기억 효과를 깨끗하게 분리하는 첫 과제로는 위험하다. 기존 SNN 발작 논문은 구간 단위 민감도·특이도·AUC로 보고했고 SzCORE 이벤트 지표는 쓰지 않았다(초록 기준). SzCORE 방식으로 평가하면 차별점이 될 수 있다.

### Gaps
- MASS(SS3)의 접근 조건과 크기는 이번에 확인하지 않았다. SHHS도 NSRR 데이터 요청이 필요한 것으로 알고 있으나, 페이지를 받지 못해 미확인이다.
- XSleepNet과 SeqSleepNet 원 논문(2019–2021, 배경)의 맥락 절제 수치는 직접 열지 않았다.
- 발작 "예측"(발작 전 구간)에서 맥락 길이를 바꾼 2023–2026 절제 연구는 찾지 못했다. SNN 쪽은 Spiking Conformer가 CHB-MIT 예측을 보고했다는 사실만 확인했다.
- PicoSleepNet이 여러 에폭을 넣는지(시퀀스 맥락) 아니면 단일 에폭인지는 초록으로 판단할 수 없다.

---

## Q3. 기타: PPG 심박 추적, 긴 창 활동 인식, 호흡·수면무호흡(ECG)

### Takeaway
PPG 심박 추적(PPG-DaLiA)은 시간 맥락 효과가 직접 측정돼 있다. 8초 창 하나만 쓰면 MAE가 3.59에서 4.86 BPM으로 나빠지고, 시간 방향 메시지 전달(belief propagation)을 빼면 4.02 BPM이 된다. 손목 가속도 활동 인식(CAPTURE-24)은 HMM 평활이 κ를 0.704에서 0.814로 크게 올리지만, 이것은 라벨 지속성을 쓴 효과다. Apnea-ECG는 "라벨 1분 + 앞뒤 2분" 입력이 표준 관행인데, 맥락 길이 절제 수치는 직접 확인하지 못했다. 스파이킹 쪽에는 PPG 호흡수 추정에서 창이 길수록 오차가 줄어든 SNN 결과(16/32/64초 → MAE 1.37/1.23/1.15)가 있다.

### Cited Findings
**PPG-DaLiA / WESAD / WildPPG**
- PPG-DaLiA(배경, Sensors 2019): 15명, 총 약 36시간, 1인당 약 150분. 일상 활동 8종과 전이 구간으로 이뤄진다. 손목 Empatica E4의 PPG는 64 Hz, 가속도는 32 Hz다. 정답은 가슴 ECG에서 구한 8초 창(2초 이동)의 평균 순간 심박이고, 지표는 MAE(BPM)다. 저자들은 LOSO 평가를 권장한다 — [Reiss et al., Deep PPG (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6679242/)
- BeliefPPG(UAI 2023): 8초 창 7개를 2초씩 겹쳐 쓰고 시간 방향 belief propagation을 얹었다. LoSo MAE는 PPG-DaLiA 3.57 ± 1.4, WESAD 4.28 ± 2.0이고, 오프라인 Viterbi 복호를 쓰면 각각 3.18, 4.02다(DaLiA에서 11% 감소). 같은 표에서 Deep PPG는 DaLiA 7.65 ± 4.2다 — [Bieri et al., BeliefPPG](https://arxiv.org/pdf/2306.07730)
- BeliefPPG 절제(5-fold, PPG-DaLiA MAE): 최종 모델(64 bins) 3.59 → 메시지 전달 제거 4.02 → 데이터 증강까지 제거 4.02 → 8초 창 하나로 축소 4.86. 8초 창 하나만 보는 U-Net 단독은 7.93, CorNET Light는 5.31이다 — [BeliefPPG Table 3](https://arxiv.org/pdf/2306.07730)
- WildPPG(NeurIPS 2024): 16명이 하루 동안 13.5시간 이동하며 기록했다. 몸 4곳의 웨어러블을 합쳐 216시간이고, 기존 최신 HR 방법들이 이 데이터에서 어려움을 겪는다고 보고한다(초록 페이지만 확인) — [Meier et al., WildPPG](https://arxiv.org/abs/2412.17540)
- 스파이킹: Yang et al.(Sensors 2024, 초록만 확인)은 PPG로 호흡수를 추정하는 SNN(피드백 IF 뉴런)을 BIDMC 데이터로 평가했다. 창 16 / 32 / 64초에서 MAE가 1.37 / 1.23 / 1.15였다 — [doi:10.3390/s24123980](https://doi.org/10.3390/s24123980)

**CAPTURE-24 (손목 가속도, 자유 생활)**
- 151명, 가속도 3,883시간(그중 주석 2,562시간). Axivity AX3, 100 Hz, ±8 g. 정답은 착용형 카메라와 수면 일지로 만들었다. 라이선스 CC BY 4.0 — [Chan et al., CAPTURE-24 (Sci Data 2024)](https://arxiv.org/pdf/2402.19229)
- 벤치마크는 겹치지 않는 10초 창(총 922,199개)을 쓴다. 참가자 P001~P100이 개발용(그중 학습 80명, 검증 20명), P101~P151이 시험용이다. RNN 입력은 최대 8창(80초)이다 — [CAPTURE-24](https://arxiv.org/pdf/2402.19229)
- 신체활동 수준 분류(Table 5a)의 F1 / κ: RF 0.711 / 0.665 → RF+HMM 0.789 / 0.772, CNN 0.716 / 0.704 → CNN+HMM 0.800 / 0.814, RNN(8창) 0.776 / 0.790 → RNN+HMM 0.787 / 0.800 — [CAPTURE-24](https://arxiv.org/pdf/2402.19229)
- 일상 활동 분류(Table 5b): CNN 0.485 / 0.626 → CNN+HMM 0.576 / 0.735, RNN 0.534 / 0.697 → RNN+HMM 0.537 / 0.714. 저자들은 HMM이 RNN 위에서도 성능을 올린 것을 "longer sequence modelling would be fruitful"의 근거로 들었다 — [CAPTURE-24](https://arxiv.org/pdf/2402.19229)

**Apnea-ECG (단일 리드 ECG로 분 단위 무호흡 검출)**
- 70건(학습 35, 시험 35). 기록당 7~10시간, 100 Hz. 1분마다 무호흡 유무 주석이 있다. 8건에는 호흡 노력, 기류, SpO2 신호가 함께 있다. ODC-By. PhysioNet 페이지는 .apn 주석이 학습셋 35건에만 있다고 적고 있다 — [PhysioNet Apnea-ECG](https://physionet.org/content/apnea-ecg/)
- Wang et al.(PeerJ 2019, 배경): 라벨된 1분과 앞뒤 2분을 합친 5분(1분 구간 5개)을 입력으로 쓴다. 공개셋 35건으로 학습하고 보류셋 35건으로 시험해 구간 단위 정확도 87.6%, 민감도 83.1%, 특이도 90.3%, AUC 0.950을 얻었다. 기록 단위 정확도는 97.1%다. 근거로 "이웃 구간이 유용하다"는 2000년대 연구를 인용한다 — [Wang et al. (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6756143/)
- Nguyen et al.(APSIPA 2022, arXiv): 같은 5분 표본화를 쓴다. 학습 구간은 16,709개(무호흡 38.74%), 시험 구간은 16,945개(38.30%)이고, 정확도 91.13%, 민감도 92.58%, 특이도 88.75%다 — [arXiv 2208.03408](https://arxiv.org/pdf/2208.03408)
- DAN-MTIF(IEEE OJEMB 2024, 초록만 확인): 길이가 다른 구간 세 개에서 RR 간격과 R 진폭 특징을 뽑아 attention으로 합친다. 구간 단위 정확도는 0.9106이고, 저자들은 다중 시간척도가 단일 시간척도보다 낫다고 주장한다 — [doi:10.1109/ojemb.2024.3405666](https://doi.org/10.1109/ojemb.2024.3405666)
- 스파이킹: Tyagi & Agrawal의 "Automatic detection of sleep apnea from a single-lead ECG signal based on SNN"(Comput Biol Med 2024, 초록 없음) — [doi](https://doi.org/10.1016/j.compbiomed.2024.108877). RSN-Count(IEEE TBME 2024, 초록만 확인)는 재귀 SNN으로 밤새 기록한 오디오와 SpO2에서 무호흡 사건 수를 직접 센다(N = 33) — [doi:10.1109/tbme.2024.3498097](https://doi.org/10.1109/tbme.2024.3498097)

### Inferences
- PPG-DaLiA에서 8초 창을 7개 창 묶음으로 바꾸면 신호 길이는 8 + 6×2 = 20초가 된다(산술 추정). 여기에 시퀀스 전체에 걸친 메시지 전달을 더하면 MAE가 약 26% 줄어든다(4.86 → 3.59). 움직임 잡음 아래서 "직전 심박"이 정보를 주기 때문이다. 단순 평활보다 상태 추적에 가까운 과제다. 데이터가 작아(36시간) 4 GPU에서 많은 반복 실행이 쉽다. 같은 폴더의 `long_sequence_benchmarks.md`에 비스파이킹 장기 시퀀스 모델이 쓰는 Log-NCDE식 PPG-DaLiA 회귀 프로토콜이 따로 정리돼 있으니, 두 프로토콜을 혼동하지 말아야 한다.
- CAPTURE-24의 큰 이득은 사후 HMM에서 나온다. 따라서 이 데이터는 "상태가 평활 이상의 정보를 주는가"를 시험하기에 좋은 음성 대조(이길 기준선) 역할을 한다. 다만 데이터가 크다(3,883시간 × 100 Hz × 3축).
- Apnea-ECG는 ECG 단일 리드라 CinC 2017 파이프라인을 거의 그대로 쓸 수 있다. 공식 학습/시험 분할(기록 단위)이 고정돼 있고 작다(70건 × 약 8시간 × 100 Hz). 반면 시험셋이 35건뿐이라 기록 단위 지표의 분해능이 낮다. Wang et al.도 이 점을 지적한다.

### Gaps
- Apnea-ECG에서 1분, 3분, 5분 입력을 같은 모델로 비교한 수치를 직접 확인하지 못했다. SE-MSCNN(KBS 2022), RAFNet(Neural Networks 2023)은 본문 접근 실패다. RAFNet의 91.40%는 대학 보도자료를 거친 검색 요약이라 미검증이다.
- PPG-DaLiA의 2024–2026 최신 LoSo 최고치와 그 출처는 확인하지 못했다(석사논문 수치만 검색에 보임, 미검증).
- 보행(gait)과 호흡 신호 단독 과제에서 맥락 절제 근거는 조사하지 못했다.
- PPG-DaLiA, CAPTURE-24, WildPPG를 쓴 SNN 논문은 찾지 못했다(검색 결과 없음).

---

## Q4. 데이터셋별 비교표 (기록 길이, 표본화율, 크기, 접근, 표준 분할, 지표, 대표 수치, SNN 사용, 맥락 근거)

### Takeaway
"공개, 작음, 표준 분할 있음, 맥락 근거 직접 측정됨"을 모두 갖춘 후보는 Sleep-EDF(20/78), PPG-DaLiA, Apnea-ECG다. 과제 정의 자체가 장기 맥락을 요구하는 후보는 CPSC 2021이다. AFDB/LTAFDB 창 단위 분류, CHB-MIT/TUSZ, Icentia11k 전체는 천장, 잡음, 규모 때문에 1차 후보로 약하다.

### Cited Findings
| 과제 / 데이터 | 기록 길이·표본화율 | 크기·접근 | 표준 분할·지표 | 대표 수치 (출처) | SNN 사용 | 맥락 길이 근거 |
|---|---|---|---|---|---|---|
| AFDB (AF 구간) | 10 h, 250 Hz, 2채널 ([PN](https://physionet.org/content/afdb/)) | 23건 신호, 605.9 MB, ODC-By ([PN](https://physionet.org/content/afdb/)) | 흔히 외부 시험셋. 창 단위 F1 | 10 s 창 F1 0.986 ([초록](https://doi.org/10.3390/s25134109)) | 확인 못 함 | 없음, 천장 |
| LTAFDB | 24–25 h, 128 Hz, 2채널 ([PN](https://physionet.org/content/ltafdb/)) | 84건, 3.4 GB, ODC-By | 학습원으로 흔히 씀 ([초록](https://doi.org/10.3390/s25134109)) | — | 확인 못 함 | 미확인 |
| Icentia11k | 최대 2주, 250 Hz, 단일 리드 ([PN](https://physionet.org/content/icentia11k-continuous-ecg/)) | 11k명, 1.1 TB, CC BY-NC-SA | 환자 단위. AF AUROC/AUPRC | 고정 인코더 AUPRC 16 s 0.712 → 10 min 0.877 ([arXiv](https://arxiv.org/pdf/2608.12695)) | 확인 못 함 | 있음(5–10분에서 최대) |
| CPSC 2021 | 기록 길이 가변, 200 Hz, 2리드 ([PN](https://physionet.org/content/cpsc2021/)) | 1,436건 공개, 시험셋 비공개, 1.3 GB, CC BY 4.0 | U 점수(기록 분류 + 시작·끝 ±1/±2박동) | 2023 J Pers Med 시작점 민감도 95.90% ([초록](https://doi.org/10.3390/jpm13050820)) | 확인 못 함 | 과제 정의상 기록 전체 필요 |
| SHDB-AF | Holter(길이 미확인) ([PN](https://physionet.org/content/shdb-af/)) | 122명/128건, 8.3 GB, ODC-By | 외부 시험셋(30 s 창 F1) ([arXiv](https://arxiv.org/abs/2401.05411)) | RawECGNet F1 0.93 | 확인 못 함 | BiGRU 이웃 창 "slightly" |
| Apnea-ECG | 7–10 h, 100 Hz ([PN](https://physionet.org/content/apnea-ecg/)) | 70건, ODC-By | 공개 35 학습 / 보류 35 시험. 분 단위 정확도·민감도·특이도, 기록 단위 | 정확도 91.13% ([arXiv](https://arxiv.org/pdf/2208.03408)); 87.6% ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6756143/)) | 있음 (CBM 2024 단일 리드 ECG SNN, 사용 데이터셋 미확인) | 관행(±2분). 수치 절제 미확인 |
| Sleep-EDF-20/78 | 밤샘(SC는 약 20 h 주야 기록), EEG 100 Hz ([PN](https://physionet.org/content/sleep-edfx/)) | 197건, 8.1 GB, ODC-By | SEDF-20 LOSO, SEDF-78 k-fold. 정확도/MF1/κ | SEDF-20 단일 채널 MF1 0.793–0.806 (L-SeqSleepNet 0.793, S4Sleep(ts) 0.804, XSleepNet2 0.806; [S4Sleep Table 5](https://arxiv.org/pdf/2310.06715)) | 있음 (PicoSleepNet MF1 75.2/68.1) | 1 → 15 에폭 MF1 0.752 → 0.796, 이후 평탄 |
| SHHS | 밤샘, C4-A1 ([L-Seq](https://arxiv.org/pdf/2301.03441)) | 5,791명, NSRR 요청(미확인) | 70/30 등. 정확도/MF1/κ | 정확도 88.4 / κ 0.838 ([L-Seq](https://arxiv.org/pdf/2301.03441)) | 확인 못 함 | 20 → 200 에폭 +0.3%p(SeqSleepNet) |
| CHB-MIT | 파일 1–4 h, 총 982 h, 256 Hz, 23채널 ([PN](https://physionet.org/content/chbmit/); [SzCORE](https://arxiv.org/pdf/2402.13005)) | 42.6 GB, ODC-By | 개인별 시간순 CV 또는 LOSO. SzCORE 이벤트 F1 | SNN LOO AUC 89.0% ([초록](https://doi.org/10.1088/2634-4386/acbab8)) | 있음 (여러 편) | 미확인 |
| TUSZ | 파일 평균 10분, 1,476 h, 250–1000 Hz ([SzCORE](https://arxiv.org/pdf/2402.13005)) | 신청서 필요 ([NEDC](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)) | 공식 학습/시험(환자 분리). AUROC 또는 이벤트 F1 | 12 s 0.866 vs 60 s 0.875 (사전학습 Dist-DCRNN) ([ICLR'22](https://arxiv.org/pdf/2104.08336)) | 확인 못 함 | 긴 클립이 대개 나쁨 |
| PPG-DaLiA | 약 150 min/명, PPG 64 Hz ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6679242/)) | 15명, 36 h, 공개 | LoSo, 8 s 창 / 2 s 이동, MAE | 3.57 BPM (온라인) ([UAI'23](https://arxiv.org/pdf/2306.07730)) | 확인 못 함 | 8 s 단독 4.86 vs 20 s + 메시지 전달 3.59 |
| CAPTURE-24 | 약 24 h/명, 100 Hz 3축 ([Sci Data](https://arxiv.org/pdf/2402.19229)) | 151명, 3,883 h, CC BY 4.0 | P001–100 개발 / P101–151 시험, 10 s 창. F1/κ | CNN+HMM κ 0.814 | 확인 못 함 | HMM이 κ +0.11 (평활 효과) |

### Inferences
- 표의 "SNN 사용" 칸은 OpenAlex 제목·초록 검색 결과다. 여기서 "확인 못 함"은 "없다"의 증명이 아니다.
- 지표가 이질적이다. 창 단위(F1, AUROC), 에폭 단위(MF1, κ), 사건 단위(SzCORE F1), 기록 단위(CPSC U, 무호흡 기록 분류)가 섞여 있다. 기억 효과를 주장하려면 맥락이 실제로 바꿀 수 있는 단위의 지표를 미리 하나 골라야 한다. 예를 들어 CPSC는 U 점수, 수면은 MF1이다.

### Gaps
- SHDB-AF와 SHHS의 기록 길이, NSRR 접근 조건은 페이지를 열지 못해 미확인이다.
- CAPTURE-24와 Sleep-EDF의 실제 디스크 용량(GB)은 일부만 확인했다(Sleep-EDF 8.1 GB).

---

## Q5. 우리 모델에 "T = N"이 자연스럽게 맞고, A6000 4장으로 많은 반복 실행이 가능한 과제는?

### Takeaway
1순위는 Sleep-EDF-20/78 시퀀스 채점(+ HMM 평활 대조군)과 Apnea-ECG(±2분 맥락)다. 둘 다 공개, 소형, 표준 분할이 있고 SNN 비교 대상도 있다. 2순위는 PPG-DaLiA 심박 추적이다. 맥락 효과가 상태 추적형이고 데이터가 아주 작다. "과제 자체가 장기 맥락을 요구한다"는 강한 주장을 원하면 CPSC 2021(환자 단위 자체 분할)이 가장 직접적이다. 발작(CHB-MIT/TUSZ)과 Icentia11k 전체는 규모와 잡음 때문에 2차 후보로 미룬다.

### Cited Findings
- 수면 맥락 이득은 1 → 15에폭에서 생기고(MF1 0.752 → 0.796), 10 → 200에폭에서는 평탄하다(0.790~0.796, 유의차 없음) — [S4Sleep](https://arxiv.org/pdf/2310.06715); [Wang & Strodthoff 2024](https://arxiv.org/pdf/2402.17779)
- 무작위(미학습) Transformer만으로도 수면 채점이 좋아진다는 주장은 맥락 이득의 상당 부분이 평활이라는 해석을 지지한다 — [Liu et al. 2026](https://arxiv.org/abs/2605.09905)
- CAPTURE-24에서 사후 HMM만으로 κ가 크게 오른다(CNN 0.704 → 0.814) — [CAPTURE-24](https://arxiv.org/pdf/2402.19229)
- PPG-DaLiA에서는 시간 방향 메시지 전달을 빼면 MAE가 3.59 → 4.02, 8초 단일 창으로 줄이면 4.86이 된다 — [BeliefPPG](https://arxiv.org/pdf/2306.07730)
- Apnea-ECG 표준 입력은 라벨 1분 ± 2분(5분)이다 — [Wang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6756143/); [Nguyen et al.](https://arxiv.org/pdf/2208.03408)
- CPSC 2021은 기록 단위 3클래스 판정과 ±1~2박동 정밀도의 경계 검출을 함께 채점한다 — [PhysioNet CPSC 2021](https://physionet.org/content/cpsc2021/)
- 데이터 규모: Apnea-ECG 70건 × 7–10 h × 100 Hz, AFDB 605.9 MB, LTAFDB 3.4 GB, CPSC2021 1.3 GB, Sleep-EDFx 8.1 GB, CHB-MIT 42.6 GB, Icentia11k 1.1 TB — 각 PhysioNet 페이지([apnea-ecg](https://physionet.org/content/apnea-ecg/), [afdb](https://physionet.org/content/afdb/), [ltafdb](https://physionet.org/content/ltafdb/), [cpsc2021](https://physionet.org/content/cpsc2021/), [sleep-edfx](https://physionet.org/content/sleep-edfx/), [chbmit](https://physionet.org/content/chbmit/), [icentia11k](https://physionet.org/content/icentia11k-continuous-ecg/))

### Inferences
- **T = N이 자연스러운 이유.** 아래 과제는 모두 고유한 시간 구간이 없는 연속 신호이고, 토큰마다 라벨이 붙는 시퀀스-투-시퀀스 구조다. 그래서 "토큰 1개 = 스파이킹 시간 단계 1개"가 그대로 맞는다. 수면은 토큰 = 30초 에폭(또는 그 안의 패치)이고, 라벨은 에폭마다 있다. Apnea-ECG는 토큰 = 수 초 패치이고 라벨은 분마다, AF 구간은 토큰 = 박동 또는 1초 패치, 발작은 SzCORE의 1 Hz 라벨이 있으니 토큰 = 1초다. PPG-DaLiA는 토큰 = 2초 이동 단위, CAPTURE-24는 토큰 = 10초 창이다. CinC 2017은 기록 하나에 라벨 하나라서 상태가 할 일이 "한 번 요약"뿐이었다. 위 과제들은 매 토큰마다 상태가 쓰이니 기억의 기여를 토큰 단위로 잴 수 있다.
- **계산량 추정.** 수면을 토큰 = 에폭으로 두면 시퀀스 길이는 20~200단계라 매우 싸다. 토큰 = 1초 패치로 두면 200에폭 = 6,000단계라 스파이킹 순차 시뮬레이션이 무거워진다. 맥락 이득이 15~20에폭에서 포화하니, 토큰 = 에폭에 시퀀스 15~40에폭이면 충분할 것이다. Apnea-ECG(70건)와 PPG-DaLiA(36시간)는 여러 시드와 여러 변형을 4 GPU로 돌리기 쉽다. CHB-MIT, TUSZ, Icentia11k 전체는 반복 실행이 비싸다.
- **기억 주장을 위한 최소 대조군.** 앞의 근거를 종합하면 세 가지가 필요하다. (1) 토큰별 단독 모델 + HMM/중앙값 평활, (2) 같은 맥락 길이의 spiking self-attention, (3) 맥락 잘라내기 곡선(1, 5, 15, 40 토큰). T = N 상태가 (1)과 (2)를 함께 넘어야 "평활이나 어텐션 이상의 정보"라고 말할 수 있다. CAPTURE-24와 2026 평활 논문을 보면 (1)이 생각보다 강할 가능성이 크다.
- **추천 순서(추정).**
  1. Sleep-EDF-20(LOSO)과 SEDF-78. 맥락 근거가 가장 단단하고 SNN 비교 대상(PicoSleepNet)이 있다.
  2. Apnea-ECG. 같은 ECG 계열이고 공식 분할이 있으며 ±2분 맥락이 관행이다.
  3. PPG-DaLiA. 상태 추적형 맥락이고 매우 작다.
  4. CPSC 2021. 과제 정의상 장기 맥락이 필요하다. 다만 시험셋이 비공개라 자체 환자 분할이 필요하고 U 점수 구현도 필요하다.
  5. 이후 후보: SzCORE 방식의 CHB-MIT, Icentia11k 부분집합(5~10분 맥락).
- 이벤트 기반 데이터(SHD/SSC/DVS)는 고유한 시간 구간이 있어 T = N이 원래 표준이므로 제외했다. 2026-10-21 결정과도 일치한다.

### Gaps
- 각 과제에서 A6000 한 장당 1회 학습 시간을 실측한 자료는 없다. 위 계산량 판단은 데이터 크기와 시퀀스 길이에서 나온 추정이다.
- PicoSleepNet 외에 수면 시퀀스 맥락을 명시적으로 쓴 SNN 연구가 있는지는 확인하지 못했다(CCC 2024, Sci China 2022는 초록을 못 읽음).
- CPSC 2021 기록의 환자 식별자 규칙은 검증하지 않았다.
