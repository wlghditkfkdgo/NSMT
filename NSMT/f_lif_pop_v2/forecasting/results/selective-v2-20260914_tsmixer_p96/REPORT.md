# PopulationLIF v2 — tsmixer / H96

ETTh1/ETTh2 × seeds7,13,21 × homogeneous/heterogeneous × off/dense/sparse =36 runs.
동일 점수(-learned Q/K squared distance)·null 후보·retrieval-aware gate에서 softmax와 sparsemax를 비교한다.
모든 최종 checkpoint는 validation MSE 최소값으로 선택했다. 수치는 전체 test의 train-standardized MSE/MAE다.

| Variant | MSE ± SD | MAE ± SD |
|---|---:|---:|
| heterogeneous_dense | 0.368598 ± 0.002025 | 0.409724 ± 0.000768 |
| heterogeneous_no_memory | 0.367226 ± 0.003632 | 0.409205 ± 0.001661 |
| heterogeneous_sparse | 0.368727 ± 0.005457 | 0.409290 ± 0.003321 |
| homogeneous_dense | 0.391448 ± 0.008942 | 0.428576 ± 0.008141 |
| homogeneous_no_memory | 0.387066 ± 0.000689 | 0.426148 ± 0.002534 |
| homogeneous_sparse | 0.391366 ± 0.006573 | 0.429257 ± 0.004431 |

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_dense | 0.415073 ± 0.005102 | 0.434215 ± 0.004583 |
| ETTh1 | heterogeneous_no_memory | 0.414036 ± 0.005065 | 0.433990 ± 0.004679 |
| ETTh1 | heterogeneous_sparse | 0.414836 ± 0.005019 | 0.433820 ± 0.004201 |
| ETTh1 | homogeneous_dense | 0.440476 ± 0.003973 | 0.453527 ± 0.004508 |
| ETTh1 | homogeneous_no_memory | 0.438481 ± 0.002417 | 0.452881 ± 0.004142 |
| ETTh1 | homogeneous_sparse | 0.439667 ± 0.005900 | 0.454029 ± 0.005869 |
| ETTh2 | heterogeneous_dense | 0.322123 ± 0.006511 | 0.385232 ± 0.004561 |
| ETTh2 | heterogeneous_no_memory | 0.320416 ± 0.005044 | 0.384420 ± 0.003313 |
| ETTh2 | heterogeneous_sparse | 0.322618 ± 0.006481 | 0.384760 ± 0.003434 |
| ETTh2 | homogeneous_dense | 0.342420 ± 0.013915 | 0.403624 ± 0.011862 |
| ETTh2 | homogeneous_no_memory | 0.335652 ± 0.001128 | 0.399414 ± 0.000926 |
| ETTh2 | homogeneous_sparse | 0.343066 ± 0.011321 | 0.404485 ± 0.007570 |

## 선택 동작 (첫 test8windows, 마지막 population layer)

| Variant | Selected/available slots | Empty read fraction | Real probability mass |
|---|---:|---:|---:|
| heterogeneous_dense | 0.999590 | 0.000000 | 0.936090 |
| heterogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| heterogeneous_sparse | 0.373709 | 0.033804 | 0.924830 |
| homogeneous_dense | 0.999077 | 0.000000 | 0.936171 |
| homogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| homogeneous_sparse | 0.390757 | 0.024619 | 0.943729 |

- Sparsemax는 일부 weight를 정확히0으로 만들 수 있지만 매 query의 sparsity를 보장하지 않는다. 실제 density/empty rate를 함께 읽는다.
- 실제 read 분모는 max(real_mass,1e-12)이다. 아주 작은 양의 real mass에서는 weight 합이1보다 작다. Lag mass/mean lag/entropy 해석에 이 floor를 반영해야 하며 check_read_mass.json에서 각 checkpoint로 실제 수식을 재검증한다.
- Off/dense/sparse의 nominal parameter/초기값은 같고 off의 retrieval 파라미터는 미사용이다. Homogeneous는 redundant-state 대조다.
- off/uniform/recent는 같은 checkpoint의 평가 개입이다. uniform/recent는 null을 우회하여 과거 후보를 강제 선택하고 gate를 재계산한다.
- 서로 다른 backbone은 용량/연산이 다르다. PatchTST와 TSMixer는 causal population-spiking 변형이며 원 논문 재현이 아니다.
- 알려진 slot의 operator 검사는 통과했지만 end-to-end synthetic recall 학습/에너지 측정/정답 ETT lag 검증은 not run.
- 과거slot mask는 raw input의 후속 recurrent-state 영향을 삭제하지 않는다. Dense search와 sparse use는 별개다.
- 이전 v1 결과는 scorer/gate/budget 등이 달라 sparsemax만의 효과를 비교하는 대조가 아니다. v2 dense와 sparse가 해당 대조다.
