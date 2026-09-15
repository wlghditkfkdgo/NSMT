# PopulationLIF v2 — patchtst / H720

ETTh1/ETTh2 × seeds7,13,21 × homogeneous/heterogeneous × off/dense/sparse =36 runs.
동일 점수(-learned Q/K squared distance)·null 후보·retrieval-aware gate에서 softmax와 sparsemax를 비교한다.
모든 최종 checkpoint는 validation MSE 최소값으로 선택했다. 수치는 전체 test의 train-standardized MSE/MAE다.

| Variant | MSE ± SD | MAE ± SD |
|---|---:|---:|
| heterogeneous_dense | 0.787864 ± 0.039247 | 0.622330 ± 0.016759 |
| heterogeneous_no_memory | 0.798103 ± 0.039595 | 0.626114 ± 0.016795 |
| heterogeneous_sparse | 0.779744 ± 0.038431 | 0.619437 ± 0.016433 |
| homogeneous_dense | 0.835611 ± 0.067934 | 0.642131 ± 0.022011 |
| homogeneous_no_memory | 0.848380 ± 0.106832 | 0.644202 ± 0.037498 |
| homogeneous_sparse | 0.836971 ± 0.076719 | 0.642083 ± 0.024944 |

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_dense | 0.562113 ± 0.029066 | 0.547279 ± 0.016002 |
| ETTh1 | heterogeneous_no_memory | 0.560886 ± 0.023048 | 0.547470 ± 0.010958 |
| ETTh1 | heterogeneous_sparse | 0.561193 ± 0.026981 | 0.546152 ± 0.014506 |
| ETTh1 | homogeneous_dense | 0.576292 ± 0.031959 | 0.559713 ± 0.016645 |
| ETTh1 | homogeneous_no_memory | 0.564852 ± 0.056757 | 0.550215 ± 0.028731 |
| ETTh1 | homogeneous_sparse | 0.565982 ± 0.030568 | 0.554542 ± 0.016881 |
| ETTh2 | heterogeneous_dense | 1.013615 ± 0.100608 | 0.697381 ± 0.042467 |
| ETTh2 | heterogeneous_no_memory | 1.035320 ± 0.100694 | 0.704758 ± 0.043039 |
| ETTh2 | heterogeneous_sparse | 0.998295 ± 0.098997 | 0.692721 ± 0.042020 |
| ETTh2 | homogeneous_dense | 1.094930 ± 0.124484 | 0.724549 ± 0.042452 |
| ETTh2 | homogeneous_no_memory | 1.131907 ± 0.160934 | 0.738189 ± 0.050459 |
| ETTh2 | homogeneous_sparse | 1.107960 ± 0.146447 | 0.729623 ± 0.049604 |

## 선택 동작 (첫 test8windows, 마지막 population layer)

| Variant | Selected/available slots | Empty read fraction | Real probability mass |
|---|---:|---:|---:|
| heterogeneous_dense | 0.999344 | 0.000000 | 0.916438 |
| heterogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| heterogeneous_sparse | 0.396958 | 0.033394 | 0.931737 |
| homogeneous_dense | 0.997632 | 0.000000 | 0.893939 |
| homogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| homogeneous_sparse | 0.418903 | 0.054458 | 0.919918 |

- Sparsemax는 일부 weight를 정확히0으로 만들 수 있지만 매 query의 sparsity를 보장하지 않는다. 실제 density/empty rate를 함께 읽는다.
- 실제 read 분모는 max(real_mass,1e-12)이다. 아주 작은 양의 real mass에서는 weight 합이1보다 작다. Lag mass/mean lag/entropy 해석에 이 floor를 반영해야 하며 check_read_mass.json에서 각 checkpoint로 실제 수식을 재검증한다.
- Off/dense/sparse의 nominal parameter/초기값은 같고 off의 retrieval 파라미터는 미사용이다. Homogeneous는 redundant-state 대조다.
- off/uniform/recent는 같은 checkpoint의 평가 개입이다. uniform/recent는 null을 우회하여 과거 후보를 강제 선택하고 gate를 재계산한다.
- 서로 다른 backbone은 용량/연산이 다르다. PatchTST와 TSMixer는 causal population-spiking 변형이며 원 논문 재현이 아니다.
- 알려진 slot의 operator 검사는 통과했지만 end-to-end synthetic recall 학습/에너지 측정/정답 ETT lag 검증은 not run.
- 과거slot mask는 raw input의 후속 recurrent-state 영향을 삭제하지 않는다. Dense search와 sparse use는 별개다.
- 이전 v1 결과는 scorer/gate/budget 등이 달라 sparsemax만의 효과를 비교하는 대조가 아니다. v2 dense와 sparse가 해당 대조다.
