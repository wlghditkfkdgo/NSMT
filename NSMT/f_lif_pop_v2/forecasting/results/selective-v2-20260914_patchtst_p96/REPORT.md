# PopulationLIF v2 — patchtst / H96

ETTh1/ETTh2 × seeds7,13,21 × homogeneous/heterogeneous × off/dense/sparse =36 runs.
동일 점수(-learned Q/K squared distance)·null 후보·retrieval-aware gate에서 softmax와 sparsemax를 비교한다.
모든 최종 checkpoint는 validation MSE 최소값으로 선택했다. 수치는 전체 test의 train-standardized MSE/MAE다.

| Variant | MSE ± SD | MAE ± SD |
|---|---:|---:|
| heterogeneous_dense | 0.375810 ± 0.004424 | 0.413817 ± 0.002602 |
| heterogeneous_no_memory | 0.376346 ± 0.005147 | 0.414718 ± 0.004728 |
| heterogeneous_sparse | 0.376121 ± 0.006987 | 0.416054 ± 0.004805 |
| homogeneous_dense | 0.388525 ± 0.002644 | 0.424769 ± 0.001372 |
| homogeneous_no_memory | 0.386814 ± 0.001612 | 0.423532 ± 0.001645 |
| homogeneous_sparse | 0.387346 ± 0.007030 | 0.424883 ± 0.003459 |

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_dense | 0.426480 ± 0.005509 | 0.442295 ± 0.002136 |
| ETTh1 | heterogeneous_no_memory | 0.423771 ± 0.005497 | 0.440850 ± 0.003904 |
| ETTh1 | heterogeneous_sparse | 0.424740 ± 0.003779 | 0.443467 ± 0.002877 |
| ETTh1 | homogeneous_dense | 0.435983 ± 0.011280 | 0.448541 ± 0.007449 |
| ETTh1 | homogeneous_no_memory | 0.437757 ± 0.010707 | 0.451273 ± 0.005195 |
| ETTh1 | homogeneous_sparse | 0.439900 ± 0.011124 | 0.453178 ± 0.005668 |
| ETTh2 | heterogeneous_dense | 0.325141 ± 0.005557 | 0.385339 ± 0.003159 |
| ETTh2 | heterogeneous_no_memory | 0.328922 ± 0.004802 | 0.388587 ± 0.006534 |
| ETTh2 | heterogeneous_sparse | 0.327502 ± 0.011837 | 0.388641 ± 0.008239 |
| ETTh2 | homogeneous_dense | 0.341067 ± 0.008556 | 0.400997 ± 0.006423 |
| ETTh2 | homogeneous_no_memory | 0.335871 ± 0.011879 | 0.395791 ± 0.007949 |
| ETTh2 | homogeneous_sparse | 0.334792 ± 0.005354 | 0.396588 ± 0.002649 |

## 선택 동작 (첫 test8windows, 마지막 population layer)

| Variant | Selected/available slots | Empty read fraction | Real probability mass |
|---|---:|---:|---:|
| heterogeneous_dense | 0.999670 | 0.000000 | 0.929329 |
| heterogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| heterogeneous_sparse | 0.374973 | 0.042667 | 0.906463 |
| homogeneous_dense | 0.999485 | 0.000000 | 0.908583 |
| homogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| homogeneous_sparse | 0.410973 | 0.030231 | 0.929636 |

- Sparsemax는 일부 weight를 정확히0으로 만들 수 있지만 매 query의 sparsity를 보장하지 않는다. 실제 density/empty rate를 함께 읽는다.
- 실제 read 분모는 max(real_mass,1e-12)이다. 아주 작은 양의 real mass에서는 weight 합이1보다 작다. Lag mass/mean lag/entropy 해석에 이 floor를 반영해야 하며 check_read_mass.json에서 각 checkpoint로 실제 수식을 재검증한다.
- Off/dense/sparse의 nominal parameter/초기값은 같고 off의 retrieval 파라미터는 미사용이다. Homogeneous는 redundant-state 대조다.
- off/uniform/recent는 같은 checkpoint의 평가 개입이다. uniform/recent는 null을 우회하여 과거 후보를 강제 선택하고 gate를 재계산한다.
- 서로 다른 backbone은 용량/연산이 다르다. PatchTST와 TSMixer는 causal population-spiking 변형이며 원 논문 재현이 아니다.
- 알려진 slot의 operator 검사는 통과했지만 end-to-end synthetic recall 학습/에너지 측정/정답 ETT lag 검증은 not run.
- 과거slot mask는 raw input의 후속 recurrent-state 영향을 삭제하지 않는다. Dense search와 sparse use는 별개다.
- 이전 v1 결과는 scorer/gate/budget 등이 달라 sparsemax만의 효과를 비교하는 대조가 아니다. v2 dense와 sparse가 해당 대조다.
