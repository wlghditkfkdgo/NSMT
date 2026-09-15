# PopulationLIF v2 — patch / H720

ETTh1/ETTh2 × seeds7,13,21 × homogeneous/heterogeneous × off/dense/sparse =36 runs.
동일 점수(-learned Q/K squared distance)·null 후보·retrieval-aware gate에서 softmax와 sparsemax를 비교한다.
모든 최종 checkpoint는 validation MSE 최소값으로 선택했다. 수치는 전체 test의 train-standardized MSE/MAE다.

| Variant | MSE ± SD | MAE ± SD |
|---|---:|---:|
| heterogeneous_dense | 0.706692 ± 0.056545 | 0.591711 ± 0.023375 |
| heterogeneous_no_memory | 0.685652 ± 0.053363 | 0.582620 ± 0.021507 |
| heterogeneous_sparse | 0.721476 ± 0.038874 | 0.597785 ± 0.016600 |
| homogeneous_dense | 0.713817 ± 0.042492 | 0.601710 ± 0.019110 |
| homogeneous_no_memory | 0.681632 ± 0.055167 | 0.588881 ± 0.020140 |
| homogeneous_sparse | 0.649910 ± 0.034544 | 0.575184 ± 0.013379 |

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_dense | 0.560415 ± 0.020543 | 0.550091 ± 0.011353 |
| ETTh1 | heterogeneous_no_memory | 0.553551 ± 0.023521 | 0.545229 ± 0.014516 |
| ETTh1 | heterogeneous_sparse | 0.560954 ± 0.019902 | 0.550680 ± 0.010581 |
| ETTh1 | homogeneous_dense | 0.583428 ± 0.015605 | 0.562644 ± 0.008291 |
| ETTh1 | homogeneous_no_memory | 0.581082 ± 0.031239 | 0.564492 ± 0.013720 |
| ETTh1 | homogeneous_sparse | 0.587875 ± 0.024077 | 0.565376 ± 0.013857 |
| ETTh2 | heterogeneous_dense | 0.852969 ± 0.109647 | 0.633332 ± 0.045942 |
| ETTh2 | heterogeneous_no_memory | 0.817753 ± 0.088790 | 0.620010 ± 0.034376 |
| ETTh2 | heterogeneous_sparse | 0.881998 ± 0.072022 | 0.644889 ± 0.031179 |
| ETTh2 | homogeneous_dense | 0.844206 ± 0.100586 | 0.640777 ± 0.046028 |
| ETTh2 | homogeneous_no_memory | 0.782183 ± 0.098968 | 0.613270 ± 0.036416 |
| ETTh2 | homogeneous_sparse | 0.711944 ± 0.045112 | 0.584991 ± 0.012949 |

## 선택 동작 (첫 test8windows, 마지막 population layer)

| Variant | Selected/available slots | Empty read fraction | Real probability mass |
|---|---:|---:|---:|
| heterogeneous_dense | 0.999274 | 0.000000 | 0.881230 |
| heterogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| heterogeneous_sparse | 0.385989 | 0.091833 | 0.866983 |
| homogeneous_dense | 0.997401 | 0.000000 | 0.898756 |
| homogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| homogeneous_sparse | 0.456939 | 0.057825 | 0.913030 |

- Sparsemax는 일부 weight를 정확히0으로 만들 수 있지만 매 query의 sparsity를 보장하지 않는다. 실제 density/empty rate를 함께 읽는다.
- 실제 read 분모는 max(real_mass,1e-12)이다. 아주 작은 양의 real mass에서는 weight 합이1보다 작다. Lag mass/mean lag/entropy 해석에 이 floor를 반영해야 하며 check_read_mass.json에서 각 checkpoint로 실제 수식을 재검증한다.
- Off/dense/sparse의 nominal parameter/초기값은 같고 off의 retrieval 파라미터는 미사용이다. Homogeneous는 redundant-state 대조다.
- off/uniform/recent는 같은 checkpoint의 평가 개입이다. uniform/recent는 null을 우회하여 과거 후보를 강제 선택하고 gate를 재계산한다.
- 서로 다른 backbone은 용량/연산이 다르다. PatchTST와 TSMixer는 causal population-spiking 변형이며 원 논문 재현이 아니다.
- 알려진 slot의 operator 검사는 통과했지만 end-to-end synthetic recall 학습/에너지 측정/정답 ETT lag 검증은 not run.
- 과거slot mask는 raw input의 후속 recurrent-state 영향을 삭제하지 않는다. Dense search와 sparse use는 별개다.
- 이전 v1 결과는 scorer/gate/budget 등이 달라 sparsemax만의 효과를 비교하는 대조가 아니다. v2 dense와 sparse가 해당 대조다.
