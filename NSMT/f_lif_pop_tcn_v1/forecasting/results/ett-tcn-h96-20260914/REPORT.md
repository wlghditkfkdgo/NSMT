# f-LIF population: 2차 causal Spike-TCN forecasting 결과

입력336/예측96, patch8, D32/K4, train-only 표준화. 본 실험24개(3seed), 보조 last-head 학습은 not run.
모든 수치는 전체 test window/horizon/channel의 train-standardized MSE/MAE이다.

이질적 집단에 검색을 추가하면 macro MSE 0.380509 → 0.381907 (+0.367%). ETTh1 ΔMSE=+0.004596; ETTh2 ΔMSE=-0.001800.
본 실험24개 중 4개가10epoch 상한에 도달했다. 아래 결과는 짧은 예산의 예비 검증이다.

## 본 실험: 두 데이터셋 macro (seed별 평균 후 mean ± sample SD)

| Variant | MSE | MAE |
|---|---:|---:|
| heterogeneous_no_memory | 0.380509 ± 0.003947 | 0.417491 ± 0.004645 |
| heterogeneous_retrieval | 0.381907 ± 0.008823 | 0.419841 ± 0.007770 |
| homogeneous_no_memory | 0.397856 ± 0.024672 | 0.430200 ± 0.013963 |
| homogeneous_retrieval | 0.387563 ± 0.013150 | 0.425702 ± 0.009946 |

## 데이터셋별 결과 (flatten, 3seed)

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_no_memory | 0.423997 ± 0.005138 | 0.444196 ± 0.002801 |
| ETTh1 | heterogeneous_retrieval | 0.428593 ± 0.003501 | 0.448515 ± 0.003437 |
| ETTh1 | homogeneous_no_memory | 0.430014 ± 0.006015 | 0.447499 ± 0.004904 |
| ETTh1 | homogeneous_retrieval | 0.438824 ± 0.008541 | 0.455427 ± 0.008467 |
| ETTh2 | heterogeneous_no_memory | 0.337020 ± 0.013031 | 0.390786 ± 0.011963 |
| ETTh2 | heterogeneous_retrieval | 0.335220 ± 0.014525 | 0.391166 ± 0.012821 |
| ETTh2 | homogeneous_no_memory | 0.365699 ± 0.046494 | 0.412902 ± 0.028436 |
| ETTh2 | homogeneous_retrieval | 0.336301 ± 0.029292 | 0.395977 ± 0.023174 |

## 검색 효과: 같은 seed의 retrieval on − off

음수는 검색 모델의 오차가 더 낮음을 뜻한다. 아래는 두 데이터셋의 seed별 macro다.

| Population | ΔMSE mean ± SD | ΔMAE mean ± SD |
|---|---:|---:|
| homogeneous | -0.010294 ± 0.016728 | -0.004499 ± 0.009543 |
| heterogeneous | 0.001398 ± 0.004935 | 0.002350 ± 0.003135 |

Last-state head: not run (이번 구조 비교는 flatten/3seed).

## 같은 checkpoint의 memory 개입

양수 Δ는 해당 개입이 full retrieval보다 오차를 높였음을 뜻한다. 각 개입은 전체 test에 적용했다.

| Head | Data | Variant | off − full MSE | uniform − full | recent − full |
|---|---|---|---:|---:|---:|
| flatten | ETTh1 | heterogeneous_retrieval | -0.005142 | -0.001652 | 0.002332 |
| flatten | ETTh1 | homogeneous_retrieval | 0.001797 | -0.000754 | 0.001767 |
| flatten | ETTh2 | heterogeneous_retrieval | -0.000616 | -0.001995 | 0.000231 |
| flatten | ETTh2 | homogeneous_retrieval | -0.006661 | -0.002792 | 0.001247 |

## 해석 범위 및 이어서 확인할 사항

- 10epoch/early-stop3의 구조 비교 예비 실험이다. 최적 성능/수렴/통계적 유의성을 주장하지 않는다.
- 효과를 population 이질성, 검색 on/off, 두 요인의 interaction으로 나누어 읽는다.
- 모든 조건의 nominal parameter 수를 맞췄지만 retrieval off의 Q/K/gate는 미사용이다.
- Flatten head는 모든 과거 spike를 직접 읽는다. 1차와 2차는 parameter 수/깊이가 달라 matched-capacity 비교가 아니다.
- 표의 diagnostics는 마지막 block, 각 run JSON에는 embedding/두 block별 진단도 있다. 모두 첫 test8window만이며 정답 memory slot label은 없다.
- off/uniform/recent는 고정 checkpoint의 개입이다. 해당 방식으로 재학습한 대조와 다르다.
- 강도가 작은 gamma=.05, fixed tau, pre-query/post-value convention의 한정된 검증이다.
- 이전 population coding 실험은 입력96/window norm/Gaussian/deeper backbone이 달라 직접 비교할 수 없다.
- 학습 source hash/데이터 hash/명령/환경/checkpoint 경로는 각 run JSON, 검증은 check_summary.json.
- 장기학습, learned tau, pre-reset memory, fractional prior, hard selection, synthetic recall 학습, 에너지 측정: not run.
- Canonical 진행 기록: repository root docs/PROJECT_LOG.md.
