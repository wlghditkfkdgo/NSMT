# f-LIF population: 1차 forecasting 결과

입력336/예측96, patch8, D32/K4, train-only 표준화. 본 실험24개(3seed), 보조8개(1seed).
모든 수치는 전체 test window/horizon/channel의 train-standardized MSE/MAE이다.

이질적 집단에 검색을 추가하면 macro MSE 0.385644 → 0.382667 (-0.772%). ETTh1 ΔMSE=+0.001877; ETTh2 ΔMSE=-0.007831.
본 실험24개 중 13개가10epoch 상한에 도달했다. 아래 결과는 짧은 예산의 예비 검증이다.

## 본 실험: 두 데이터셋 macro (seed별 평균 후 mean ± sample SD)

| Variant | MSE | MAE |
|---|---:|---:|
| heterogeneous_no_memory | 0.385644 ± 0.021461 | 0.423791 ± 0.016197 |
| heterogeneous_retrieval | 0.382667 ± 0.018490 | 0.420998 ± 0.014005 |
| homogeneous_no_memory | 0.407324 ± 0.017954 | 0.438958 ± 0.011965 |
| homogeneous_retrieval | 0.397969 ± 0.005556 | 0.432702 ± 0.003111 |

## 데이터셋별 결과 (flatten, 3seed)

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_no_memory | 0.424253 ± 0.007231 | 0.442967 ± 0.005307 |
| ETTh1 | heterogeneous_retrieval | 0.426130 ± 0.008795 | 0.443502 ± 0.006371 |
| ETTh1 | homogeneous_no_memory | 0.454216 ± 0.004506 | 0.461030 ± 0.003113 |
| ETTh1 | homogeneous_retrieval | 0.459988 ± 0.010425 | 0.464297 ± 0.006328 |
| ETTh2 | heterogeneous_no_memory | 0.347035 ± 0.035695 | 0.404615 ± 0.027095 |
| ETTh2 | heterogeneous_retrieval | 0.339204 ± 0.028189 | 0.398495 ± 0.021664 |
| ETTh2 | homogeneous_no_memory | 0.360431 ± 0.031991 | 0.416887 ± 0.021279 |
| ETTh2 | homogeneous_retrieval | 0.335950 ± 0.000864 | 0.401106 ± 0.000712 |

## 검색 효과: 같은 seed의 retrieval on − off

음수는 검색 모델의 오차가 더 낮음을 뜻한다. 아래는 두 데이터셋의 seed별 macro다.

| Population | ΔMSE mean ± SD | ΔMAE mean ± SD |
|---|---:|---:|
| homogeneous | -0.009355 ± 0.016715 | -0.006257 ± 0.011577 |
| heterogeneous | -0.002977 ± 0.003053 | -0.002792 ± 0.002239 |

## 마지막 상태 head (seed7만; 탐색적)

| Data | Variant | MSE | MAE |
|---|---|---:|---:|
| ETTh1 | heterogeneous_no_memory | 0.730584 | 0.601024 |
| ETTh1 | heterogeneous_retrieval | 0.745731 | 0.603236 |
| ETTh1 | homogeneous_no_memory | 0.846589 | 0.657536 |
| ETTh1 | homogeneous_retrieval | 0.807533 | 0.636966 |
| ETTh2 | heterogeneous_no_memory | 0.500898 | 0.506874 |
| ETTh2 | heterogeneous_retrieval | 0.497146 | 0.504376 |
| ETTh2 | homogeneous_no_memory | 0.565604 | 0.538102 |
| ETTh2 | homogeneous_retrieval | 0.564842 | 0.536918 |

## 같은 checkpoint의 memory 개입

양수 Δ는 해당 개입이 full retrieval보다 오차를 높였음을 뜻한다. 각 개입은 전체 test에 적용했다.

| Head | Data | Variant | off − full MSE | uniform − full | recent − full |
|---|---|---|---:|---:|---:|
| flatten | ETTh1 | heterogeneous_retrieval | 0.003029 | 0.000863 | 0.001602 |
| flatten | ETTh1 | homogeneous_retrieval | 0.005977 | 0.000335 | 0.000606 |
| flatten | ETTh2 | heterogeneous_retrieval | 0.013447 | 0.000672 | 0.000188 |
| flatten | ETTh2 | homogeneous_retrieval | 0.002746 | 0.000228 | -0.000238 |
| last | ETTh1 | heterogeneous_retrieval | 0.004609 | 0.000278 | -0.001580 |
| last | ETTh1 | homogeneous_retrieval | 0.003501 | -0.001479 | -0.001112 |
| last | ETTh2 | heterogeneous_retrieval | 0.012811 | -0.003208 | -0.005655 |
| last | ETTh2 | homogeneous_retrieval | 0.007113 | 0.000012 | -0.000433 |

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
