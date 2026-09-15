# PopulationLIF v2 — tcn / H720

ETTh1/ETTh2 × seeds7,13,21 × homogeneous/heterogeneous × off/dense/sparse =36 runs.
동일 점수(-learned Q/K squared distance)·null 후보·retrieval-aware gate에서 softmax와 sparsemax를 비교한다.
모든 최종 checkpoint는 validation MSE 최소값으로 선택했다. 수치는 전체 test의 train-standardized MSE/MAE다.

| Variant | MSE ± SD | MAE ± SD |
|---|---:|---:|
| heterogeneous_dense | 0.840394 ± 0.032524 | 0.642261 ± 0.013403 |
| heterogeneous_no_memory | 0.809613 ± 0.051103 | 0.627039 ± 0.022168 |
| heterogeneous_sparse | 0.854352 ± 0.075129 | 0.647211 ± 0.021696 |
| homogeneous_dense | 0.842025 ± 0.090161 | 0.643018 ± 0.023468 |
| homogeneous_no_memory | 0.842784 ± 0.083232 | 0.642835 ± 0.024398 |
| homogeneous_sparse | 0.825472 ± 0.021632 | 0.642080 ± 0.005551 |

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_dense | 0.564849 ± 0.008611 | 0.547765 ± 0.008940 |
| ETTh1 | heterogeneous_no_memory | 0.558062 ± 0.005598 | 0.540193 ± 0.006187 |
| ETTh1 | heterogeneous_sparse | 0.578498 ± 0.011229 | 0.555402 ± 0.003390 |
| ETTh1 | homogeneous_dense | 0.577889 ± 0.024687 | 0.558181 ± 0.013770 |
| ETTh1 | homogeneous_no_memory | 0.585228 ± 0.016353 | 0.562243 ± 0.007600 |
| ETTh1 | homogeneous_sparse | 0.599236 ± 0.029176 | 0.569354 ± 0.012981 |
| ETTh2 | heterogeneous_dense | 1.115939 ± 0.069330 | 0.736757 ± 0.026901 |
| ETTh2 | heterogeneous_no_memory | 1.061164 ± 0.102910 | 0.713885 ± 0.046229 |
| ETTh2 | heterogeneous_sparse | 1.130206 ± 0.142219 | 0.739020 ± 0.040962 |
| ETTh2 | homogeneous_dense | 1.106161 ± 0.179188 | 0.727854 ± 0.051552 |
| ETTh2 | homogeneous_no_memory | 1.100341 ± 0.153964 | 0.723428 ± 0.042740 |
| ETTh2 | homogeneous_sparse | 1.051708 ± 0.028062 | 0.714806 ± 0.011641 |

## 선택 동작 (첫 test8windows, 마지막 population layer)

| Variant | Selected/available slots | Empty read fraction | Real probability mass |
|---|---:|---:|---:|
| heterogeneous_dense | 0.994239 | 0.000000 | 0.897835 |
| heterogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| heterogeneous_sparse | 0.358352 | 0.059007 | 0.897516 |
| homogeneous_dense | 0.995947 | 0.000023 | 0.876122 |
| homogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| homogeneous_sparse | 0.340796 | 0.100215 | 0.839178 |

- Sparsemax는 일부 weight를 정확히0으로 만들 수 있지만 매 query의 sparsity를 보장하지 않는다. 실제 density/empty rate를 함께 읽는다.
- 실제 read 분모는 max(real_mass,1e-12)이다. 아주 작은 양의 real mass에서는 weight 합이1보다 작다. Lag mass/mean lag/entropy 해석에 이 floor를 반영해야 하며 check_read_mass.json에서 각 checkpoint로 실제 수식을 재검증한다.
- Off/dense/sparse의 nominal parameter/초기값은 같고 off의 retrieval 파라미터는 미사용이다. Homogeneous는 redundant-state 대조다.
- off/uniform/recent는 같은 checkpoint의 평가 개입이다. uniform/recent는 null을 우회하여 과거 후보를 강제 선택하고 gate를 재계산한다.
- 서로 다른 backbone은 용량/연산이 다르다. PatchTST와 TSMixer는 causal population-spiking 변형이며 원 논문 재현이 아니다.
- 알려진 slot의 operator 검사는 통과했지만 end-to-end synthetic recall 학습/에너지 측정/정답 ETT lag 검증은 not run.
- 과거slot mask는 raw input의 후속 recurrent-state 영향을 삭제하지 않는다. Dense search와 sparse use는 별개다.
- 이전 v1 결과는 scorer/gate/budget 등이 달라 sparsemax만의 효과를 비교하는 대조가 아니다. v2 dense와 sparse가 해당 대조다.
