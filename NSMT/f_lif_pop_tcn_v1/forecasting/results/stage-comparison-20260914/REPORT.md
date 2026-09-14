# 1차 patch SNN / 2차 causal population Spike-TCN

ETTh1/ETTh2 각각의 전체 test MSE/MAE. 데이터셋 평균을 seed별 계산한 뒤3seed mean±sample SD.
각 architecture/horizon은24 runs. Prediction length가 다른 결과는 합산하지 않는다.

| Architecture | Horizon | Variant | MSE ± SD | MAE ± SD |
|---|---:|---|---:|---:|
| patch_snn | 96 | heterogeneous_no_memory | 0.385644 ± 0.021461 | 0.423791 ± 0.016197 |
| patch_snn | 96 | heterogeneous_retrieval | 0.382667 ± 0.018490 | 0.420998 ± 0.014005 |
| patch_snn | 96 | homogeneous_no_memory | 0.407324 ± 0.017954 | 0.438958 ± 0.011965 |
| patch_snn | 96 | homogeneous_retrieval | 0.397969 ± 0.005556 | 0.432702 ± 0.003111 |
| patch_snn | 720 | heterogeneous_no_memory | 0.679887 ± 0.053368 | 0.580342 ± 0.022406 |
| patch_snn | 720 | heterogeneous_retrieval | 0.711945 ± 0.031368 | 0.592568 ± 0.014067 |
| patch_snn | 720 | homogeneous_no_memory | 0.717480 ± 0.033274 | 0.602915 ± 0.016036 |
| patch_snn | 720 | homogeneous_retrieval | 0.736275 ± 0.041440 | 0.611361 ± 0.018349 |
| spike_tcn | 96 | heterogeneous_no_memory | 0.380509 ± 0.003947 | 0.417491 ± 0.004645 |
| spike_tcn | 96 | heterogeneous_retrieval | 0.381907 ± 0.008823 | 0.419841 ± 0.007770 |
| spike_tcn | 96 | homogeneous_no_memory | 0.397856 ± 0.024672 | 0.430200 ± 0.013963 |
| spike_tcn | 96 | homogeneous_retrieval | 0.387563 ± 0.013150 | 0.425702 ± 0.009946 |
| spike_tcn | 720 | heterogeneous_no_memory | 0.800043 ± 0.046684 | 0.628503 ± 0.022646 |
| spike_tcn | 720 | heterogeneous_retrieval | 0.809756 ± 0.031170 | 0.631460 ± 0.010588 |
| spike_tcn | 720 | homogeneous_no_memory | 0.787103 ± 0.062192 | 0.624228 ± 0.019320 |
| spike_tcn | 720 | homogeneous_retrieval | 0.811149 ± 0.040830 | 0.627735 ± 0.012471 |

## 이질적 population에서 검색 추가

| Architecture | Horizon | no-memory MSE | retrieval MSE | Relative change |
|---|---:|---:|---:|---:|
| patch_snn | 96 | 0.385644 | 0.382667 | -0.772% |
| patch_snn | 720 | 0.679887 | 0.711945 | +4.715% |
| spike_tcn | 96 | 0.380509 | 0.381907 | +0.367% |
| spike_tcn | 720 | 0.800043 | 0.809756 | +1.214% |

## 비교 조건과 제한

- 48쌍의 실제 run에서 데이터 hash/전처리/splits, 입력/출력/patch/폭/head/seed, optimizer budget 및 뉴런 설정 일치를 검사했다.
- 2차에는 두 convolution/population layer와 current residual이 추가된다. Parameter는24714개 더 많으며 초기 backbone weights 및 effective capacity는 architecture 간 같지 않다.
- 각 architecture 내부의 네 조건은 같은 nominal parameters/초기값을 사용한다. Retrieval off의 Q/K/gate는 미사용이다.
- 모든 최종 모델은 최소 validation MSE로 선택했다. Maximum10/early-stop3은 수렴 보장이 없으며 LR 감소 뒤 회복 시간이 짧을 수 있다.
- Test off/uniform/recent 개입은 재학습 대조가 아니다. 이를 사용한 checkpoint/hyperparameter 재선택은 하지 않았다.
- Layer diagnostics는 첫 test8windows뿐이다. 더 깊어진 backbone의 효과와 기억 선택의 효과를 정확도 하나로 동일시하지 않는다.
- 2차는 chronological persistent state/current residual을 사용하는 Spike-TCN 변형이다. 논문 원본 재현, matched-capacity sweep, longer-budget convergence, synthetic recall, 에너지 측정: not run.
- 각 horizon의 원본 REPORT/CSV/run JSON/audits와 canonical docs/PROJECT_LOG.md를 함께 읽는다.
