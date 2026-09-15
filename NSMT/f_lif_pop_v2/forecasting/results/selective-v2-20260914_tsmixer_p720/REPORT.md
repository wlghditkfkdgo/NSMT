# PopulationLIF v2 — tsmixer / H720

ETTh1/ETTh2 × seeds7,13,21 × homogeneous/heterogeneous × off/dense/sparse =36 runs.
동일 점수(-learned Q/K squared distance)·null 후보·retrieval-aware gate에서 softmax와 sparsemax를 비교한다.
모든 최종 checkpoint는 validation MSE 최소값으로 선택했다. 수치는 전체 test의 train-standardized MSE/MAE다.

| Variant | MSE ± SD | MAE ± SD |
|---|---:|---:|
| heterogeneous_dense | 0.757781 ± 0.032110 | 0.607161 ± 0.012592 |
| heterogeneous_no_memory | 0.781466 ± 0.037816 | 0.614956 ± 0.018120 |
| heterogeneous_sparse | 0.756560 ± 0.023316 | 0.609909 ± 0.014726 |
| homogeneous_dense | 0.792932 ± 0.011242 | 0.628939 ± 0.009179 |
| homogeneous_no_memory | 0.802469 ± 0.037864 | 0.627306 ± 0.012955 |
| homogeneous_sparse | 0.787126 ± 0.019379 | 0.627744 ± 0.010980 |

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_dense | 0.550526 ± 0.014068 | 0.540981 ± 0.003388 |
| ETTh1 | heterogeneous_no_memory | 0.544470 ± 0.008716 | 0.537555 ± 0.000546 |
| ETTh1 | heterogeneous_sparse | 0.550787 ± 0.014527 | 0.541237 ± 0.004020 |
| ETTh1 | homogeneous_dense | 0.573107 ± 0.020141 | 0.556904 ± 0.008573 |
| ETTh1 | homogeneous_no_memory | 0.566062 ± 0.032757 | 0.551239 ± 0.017484 |
| ETTh1 | homogeneous_sparse | 0.573389 ± 0.022117 | 0.557327 ± 0.009664 |
| ETTh2 | heterogeneous_dense | 0.965035 ± 0.077919 | 0.673341 ± 0.027903 |
| ETTh2 | heterogeneous_no_memory | 1.018461 ± 0.077909 | 0.692357 ± 0.036226 |
| ETTh2 | heterogeneous_sparse | 0.962334 ± 0.061026 | 0.678580 ± 0.033444 |
| ETTh2 | homogeneous_dense | 1.012757 ± 0.038878 | 0.700975 ± 0.023174 |
| ETTh2 | homogeneous_no_memory | 1.038876 ± 0.076890 | 0.703373 ± 0.029687 |
| ETTh2 | homogeneous_sparse | 1.000864 ± 0.058634 | 0.698160 ± 0.029066 |

## 선택 동작 (첫 test8windows, 마지막 population layer)

| Variant | Selected/available slots | Empty read fraction | Real probability mass |
|---|---:|---:|---:|
| heterogeneous_dense | 0.998060 | 0.000000 | 0.904333 |
| heterogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| heterogeneous_sparse | 0.385492 | 0.039090 | 0.927002 |
| homogeneous_dense | 0.998450 | 0.000000 | 0.916003 |
| homogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| homogeneous_sparse | 0.438660 | 0.034274 | 0.944490 |

- Sparsemax는 일부 weight를 정확히0으로 만들 수 있지만 매 query의 sparsity를 보장하지 않는다. 실제 density/empty rate를 함께 읽는다.
- 실제 read 분모는 max(real_mass,1e-12)이다. 아주 작은 양의 real mass에서는 weight 합이1보다 작다. Lag mass/mean lag/entropy 해석에 이 floor를 반영해야 하며 check_read_mass.json에서 각 checkpoint로 실제 수식을 재검증한다.
- Off/dense/sparse의 nominal parameter/초기값은 같고 off의 retrieval 파라미터는 미사용이다. Homogeneous는 redundant-state 대조다.
- off/uniform/recent는 같은 checkpoint의 평가 개입이다. uniform/recent는 null을 우회하여 과거 후보를 강제 선택하고 gate를 재계산한다.
- 서로 다른 backbone은 용량/연산이 다르다. PatchTST와 TSMixer는 causal population-spiking 변형이며 원 논문 재현이 아니다.
- 알려진 slot의 operator 검사는 통과했지만 end-to-end synthetic recall 학습/에너지 측정/정답 ETT lag 검증은 not run.
- 과거slot mask는 raw input의 후속 recurrent-state 영향을 삭제하지 않는다. Dense search와 sparse use는 별개다.
- 이전 v1 결과는 scorer/gate/budget 등이 달라 sparsemax만의 효과를 비교하는 대조가 아니다. v2 dense와 sparse가 해당 대조다.
