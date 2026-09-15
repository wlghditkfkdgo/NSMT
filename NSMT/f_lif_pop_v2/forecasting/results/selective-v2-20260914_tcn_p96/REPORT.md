# PopulationLIF v2 — tcn / H96

ETTh1/ETTh2 × seeds7,13,21 × homogeneous/heterogeneous × off/dense/sparse =36 runs.
동일 점수(-learned Q/K squared distance)·null 후보·retrieval-aware gate에서 softmax와 sparsemax를 비교한다.
모든 최종 checkpoint는 validation MSE 최소값으로 선택했다. 수치는 전체 test의 train-standardized MSE/MAE다.

| Variant | MSE ± SD | MAE ± SD |
|---|---:|---:|
| heterogeneous_dense | 0.384109 ± 0.006163 | 0.422137 ± 0.005505 |
| heterogeneous_no_memory | 0.373195 ± 0.005423 | 0.414063 ± 0.005773 |
| heterogeneous_sparse | 0.371949 ± 0.003504 | 0.412151 ± 0.000938 |
| homogeneous_dense | 0.391371 ± 0.015574 | 0.426185 ± 0.010558 |
| homogeneous_no_memory | 0.389710 ± 0.001491 | 0.425079 ± 0.001710 |
| homogeneous_sparse | 0.383846 ± 0.003422 | 0.422972 ± 0.004383 |

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_dense | 0.424042 ± 0.006557 | 0.446590 ± 0.004936 |
| ETTh1 | heterogeneous_no_memory | 0.421020 ± 0.005360 | 0.442969 ± 0.003117 |
| ETTh1 | heterogeneous_sparse | 0.417401 ± 0.006741 | 0.439405 ± 0.007090 |
| ETTh1 | homogeneous_dense | 0.440195 ± 0.007981 | 0.455594 ± 0.002491 |
| ETTh1 | homogeneous_no_memory | 0.439560 ± 0.010712 | 0.454737 ± 0.008492 |
| ETTh1 | homogeneous_sparse | 0.437868 ± 0.000840 | 0.454736 ± 0.004022 |
| ETTh2 | heterogeneous_dense | 0.344176 ± 0.018703 | 0.397683 ± 0.015945 |
| ETTh2 | heterogeneous_no_memory | 0.325370 ± 0.016049 | 0.385157 ± 0.014270 |
| ETTh2 | heterogeneous_sparse | 0.326496 ± 0.012553 | 0.384897 ± 0.008940 |
| ETTh2 | homogeneous_dense | 0.342547 ± 0.023918 | 0.396775 ± 0.018629 |
| ETTh2 | homogeneous_no_memory | 0.339861 ± 0.011193 | 0.395420 ± 0.007749 |
| ETTh2 | homogeneous_sparse | 0.329824 ± 0.006129 | 0.391208 ± 0.004883 |

## 선택 동작 (첫 test8windows, 마지막 population layer)

| Variant | Selected/available slots | Empty read fraction | Real probability mass |
|---|---:|---:|---:|
| heterogeneous_dense | 0.994616 | 0.000000 | 0.904684 |
| heterogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| heterogeneous_sparse | 0.339157 | 0.056818 | 0.887815 |
| homogeneous_dense | 0.997939 | 0.000000 | 0.908752 |
| homogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| homogeneous_sparse | 0.326302 | 0.107284 | 0.822386 |

- Sparsemax는 일부 weight를 정확히0으로 만들 수 있지만 매 query의 sparsity를 보장하지 않는다. 실제 density/empty rate를 함께 읽는다.
- 실제 read 분모는 max(real_mass,1e-12)이다. 아주 작은 양의 real mass에서는 weight 합이1보다 작다. Lag mass/mean lag/entropy 해석에 이 floor를 반영해야 하며 check_read_mass.json에서 각 checkpoint로 실제 수식을 재검증한다.
- Off/dense/sparse의 nominal parameter/초기값은 같고 off의 retrieval 파라미터는 미사용이다. Homogeneous는 redundant-state 대조다.
- off/uniform/recent는 같은 checkpoint의 평가 개입이다. uniform/recent는 null을 우회하여 과거 후보를 강제 선택하고 gate를 재계산한다.
- 서로 다른 backbone은 용량/연산이 다르다. PatchTST와 TSMixer는 causal population-spiking 변형이며 원 논문 재현이 아니다.
- 알려진 slot의 operator 검사는 통과했지만 end-to-end synthetic recall 학습/에너지 측정/정답 ETT lag 검증은 not run.
- 과거slot mask는 raw input의 후속 recurrent-state 영향을 삭제하지 않는다. Dense search와 sparse use는 별개다.
- 이전 v1 결과는 scorer/gate/budget 등이 달라 sparsemax만의 효과를 비교하는 대조가 아니다. v2 dense와 sparse가 해당 대조다.
