# f-LIF population: 2차 causal Spike-TCN forecasting 결과

입력336/예측720, patch8, D32/K4, train-only 표준화. 본 실험24개(3seed), 보조 last-head 학습은 not run.
모든 수치는 전체 test window/horizon/channel의 train-standardized MSE/MAE이다.

이질적 집단에 검색을 추가하면 macro MSE 0.800043 → 0.809756 (+1.214%). ETTh1 ΔMSE=+0.001337; ETTh2 ΔMSE=+0.018089.
본 실험24개 중 2개가10epoch 상한에 도달했다. 아래 결과는 짧은 예산의 예비 검증이다.

## 본 실험: 두 데이터셋 macro (seed별 평균 후 mean ± sample SD)

| Variant | MSE | MAE |
|---|---:|---:|
| heterogeneous_no_memory | 0.800043 ± 0.046684 | 0.628503 ± 0.022646 |
| heterogeneous_retrieval | 0.809756 ± 0.031170 | 0.631460 ± 0.010588 |
| homogeneous_no_memory | 0.787103 ± 0.062192 | 0.624228 ± 0.019320 |
| homogeneous_retrieval | 0.811149 ± 0.040830 | 0.627735 ± 0.012471 |

## 데이터셋별 결과 (flatten, 3seed)

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_no_memory | 0.568987 ± 0.026019 | 0.547698 ± 0.015097 |
| ETTh1 | heterogeneous_retrieval | 0.570324 ± 0.016979 | 0.547276 ± 0.009581 |
| ETTh1 | homogeneous_no_memory | 0.574332 ± 0.011656 | 0.556559 ± 0.007015 |
| ETTh1 | homogeneous_retrieval | 0.558881 ± 0.011577 | 0.549593 ± 0.004695 |
| ETTh2 | heterogeneous_no_memory | 1.031099 ± 0.067494 | 0.709309 ± 0.030266 |
| ETTh2 | heterogeneous_retrieval | 1.049188 ± 0.054940 | 0.715645 ± 0.014577 |
| ETTh2 | homogeneous_no_memory | 0.999875 ± 0.121084 | 0.691898 ± 0.040718 |
| ETTh2 | homogeneous_retrieval | 1.063416 ± 0.092372 | 0.705877 ± 0.029111 |

## 검색 효과: 같은 seed의 retrieval on − off

음수는 검색 모델의 오차가 더 낮음을 뜻한다. 아래는 두 데이터셋의 seed별 macro다.

| Population | ΔMSE mean ± SD | ΔMAE mean ± SD |
|---|---:|---:|
| homogeneous | 0.024045 ± 0.030471 | 0.003507 ± 0.007906 |
| heterogeneous | 0.009713 ± 0.071216 | 0.002957 ± 0.028049 |

Last-state head: not run (이번 구조 비교는 flatten/3seed).

## 같은 checkpoint의 memory 개입

양수 Δ는 해당 개입이 full retrieval보다 오차를 높였음을 뜻한다. 각 개입은 전체 test에 적용했다.

| Head | Data | Variant | off − full MSE | uniform − full | recent − full |
|---|---|---|---:|---:|---:|
| flatten | ETTh1 | heterogeneous_retrieval | -0.019497 | -0.002820 | 0.007008 |
| flatten | ETTh1 | homogeneous_retrieval | -0.008102 | -0.000646 | 0.004169 |
| flatten | ETTh2 | heterogeneous_retrieval | -0.023199 | -0.000465 | 0.003628 |
| flatten | ETTh2 | homogeneous_retrieval | -0.043576 | -0.008340 | -0.003596 |

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
