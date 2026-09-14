# f-LIF population: 1차 forecasting 결과

입력336/예측720, patch8, D32/K4, train-only 표준화. 본 실험24개(3seed), 보조8개(1seed).
모든 수치는 전체 test window/horizon/channel의 train-standardized MSE/MAE이다.

이질적 집단에 검색을 추가하면 macro MSE 0.679887 → 0.711945 (+4.715%). ETTh1 ΔMSE=+0.013167; ETTh2 ΔMSE=+0.050948.
본 실험24개 중 6개가10epoch 상한에 도달했다. 아래 결과는 짧은 예산의 예비 검증이다.

## 본 실험: 두 데이터셋 macro (seed별 평균 후 mean ± sample SD)

| Variant | MSE | MAE |
|---|---:|---:|
| heterogeneous_no_memory | 0.679887 ± 0.053368 | 0.580342 ± 0.022406 |
| heterogeneous_retrieval | 0.711945 ± 0.031368 | 0.592568 ± 0.014067 |
| homogeneous_no_memory | 0.717480 ± 0.033274 | 0.602915 ± 0.016036 |
| homogeneous_retrieval | 0.736275 ± 0.041440 | 0.611361 ± 0.018349 |

## 데이터셋별 결과 (flatten, 3seed)

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_no_memory | 0.560893 ± 0.017403 | 0.547541 ± 0.010001 |
| ETTh1 | heterogeneous_retrieval | 0.574060 ± 0.025006 | 0.553331 ± 0.012048 |
| ETTh1 | homogeneous_no_memory | 0.565094 ± 0.016115 | 0.554434 ± 0.009455 |
| ETTh1 | homogeneous_retrieval | 0.596723 ± 0.017414 | 0.569794 ± 0.004608 |
| ETTh2 | heterogeneous_no_memory | 0.798882 ± 0.093012 | 0.613144 ± 0.038870 |
| ETTh2 | heterogeneous_retrieval | 0.849830 ± 0.040151 | 0.631804 ± 0.016112 |
| ETTh2 | homogeneous_no_memory | 0.869866 ± 0.080837 | 0.651395 ± 0.037567 |
| ETTh2 | homogeneous_retrieval | 0.875826 ± 0.074851 | 0.652928 ± 0.036402 |

## 검색 효과: 같은 seed의 retrieval on − off

음수는 검색 모델의 오차가 더 낮음을 뜻한다. 아래는 두 데이터셋의 seed별 macro다.

| Population | ΔMSE mean ± SD | ΔMAE mean ± SD |
|---|---:|---:|
| homogeneous | 0.018795 ± 0.019601 | 0.008447 ± 0.008822 |
| heterogeneous | 0.032058 ± 0.022135 | 0.012225 ± 0.009633 |

## 마지막 상태 head (seed7만; 탐색적)

| Data | Variant | MSE | MAE |
|---|---|---:|---:|
| ETTh1 | heterogeneous_no_memory | 0.776105 | 0.651288 |
| ETTh1 | heterogeneous_retrieval | 0.773437 | 0.651113 |
| ETTh1 | homogeneous_no_memory | 0.816555 | 0.669777 |
| ETTh1 | homogeneous_retrieval | 0.815257 | 0.669294 |
| ETTh2 | heterogeneous_no_memory | 1.071878 | 0.759345 |
| ETTh2 | heterogeneous_retrieval | 1.074415 | 0.761079 |
| ETTh2 | homogeneous_no_memory | 1.124452 | 0.779799 |
| ETTh2 | homogeneous_retrieval | 1.123408 | 0.780260 |

## 같은 checkpoint의 memory 개입

양수 Δ는 해당 개입이 full retrieval보다 오차를 높였음을 뜻한다. 각 개입은 전체 test에 적용했다.

| Head | Data | Variant | off − full MSE | uniform − full | recent − full |
|---|---|---|---:|---:|---:|
| flatten | ETTh1 | heterogeneous_retrieval | -0.017590 | -0.001159 | 0.010669 |
| flatten | ETTh1 | homogeneous_retrieval | -0.009694 | -0.000743 | 0.005077 |
| flatten | ETTh2 | heterogeneous_retrieval | -0.026809 | -0.006319 | 0.007987 |
| flatten | ETTh2 | homogeneous_retrieval | -0.006627 | -0.000066 | 0.002025 |
| last | ETTh1 | heterogeneous_retrieval | 0.010337 | 0.001545 | 0.000714 |
| last | ETTh1 | homogeneous_retrieval | 0.004639 | -0.000127 | -0.001070 |
| last | ETTh2 | heterogeneous_retrieval | 0.023894 | 0.000459 | 0.001757 |
| last | ETTh2 | homogeneous_retrieval | 0.008413 | 0.000131 | 0.002456 |

## 해석 범위 및 이어서 확인할 사항

- 10epoch/early-stop3의 첫 확인 실험이다. 최적 성능/수렴/통계적 유의성을 주장하지 않는다.
- 효과를 population 이질성, 검색 on/off, 두 요인의 interaction으로 나누어 읽는다.
- 모든 조건의 nominal parameter 수를 맞췄지만 retrieval off의 Q/K/gate는 미사용이다.
- Flatten head는 모든 과거 spike를 직접 읽는다. Last head는 용량과 정보 접근도 달라진다.
- Internal diagnostics는 첫 test8window만이다. ETT에는 정답 memory slot label이 없다.
- off/uniform/recent는 고정 checkpoint의 개입이다. 해당 방식으로 재학습한 대조와 다르다.
- 강도가 작은 gamma=.05, fixed tau, pre-query/post-value convention의 한정된 검증이다.
- 이전 population coding 실험은 입력96/window norm/Gaussian/deeper backbone이 달라 직접 비교할 수 없다.
- 학습 source hash/데이터 hash/명령/환경/checkpoint 경로는 각 run JSON, 검증은 check_summary.json.
- 장기학습, learned tau, pre-reset memory, fractional prior, hard selection, synthetic recall 학습, 에너지 측정: not run.
- Canonical 진행 기록: repository root docs/PROJECT_LOG.md.
