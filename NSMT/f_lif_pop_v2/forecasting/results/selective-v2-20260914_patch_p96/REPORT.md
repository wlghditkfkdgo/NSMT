# PopulationLIF v2 — patch / H96

ETTh1/ETTh2 × seeds7,13,21 × homogeneous/heterogeneous × off/dense/sparse =36 runs.
동일 점수(-learned Q/K squared distance)·null 후보·retrieval-aware gate에서 softmax와 sparsemax를 비교한다.
모든 최종 checkpoint는 validation MSE 최소값으로 선택했다. 수치는 전체 test의 train-standardized MSE/MAE다.

| Variant | MSE ± SD | MAE ± SD |
|---|---:|---:|
| heterogeneous_dense | 0.377628 ± 0.008199 | 0.419095 ± 0.006241 |
| heterogeneous_no_memory | 0.378484 ± 0.011626 | 0.419706 ± 0.008360 |
| heterogeneous_sparse | 0.381363 ± 0.005930 | 0.421432 ± 0.004041 |
| homogeneous_dense | 0.396104 ± 0.008466 | 0.431852 ± 0.005043 |
| homogeneous_no_memory | 0.395326 ± 0.006549 | 0.431763 ± 0.003007 |
| homogeneous_sparse | 0.391533 ± 0.003132 | 0.427766 ± 0.001204 |

| Data | Variant | MSE ± SD | MAE ± SD |
|---|---|---:|---:|
| ETTh1 | heterogeneous_dense | 0.426393 ± 0.004418 | 0.445142 ± 0.004326 |
| ETTh1 | heterogeneous_no_memory | 0.423974 ± 0.005129 | 0.443024 ± 0.004525 |
| ETTh1 | heterogeneous_sparse | 0.426684 ± 0.005367 | 0.444722 ± 0.004513 |
| ETTh1 | homogeneous_dense | 0.449270 ± 0.003327 | 0.458933 ± 0.001549 |
| ETTh1 | homogeneous_no_memory | 0.447416 ± 0.001958 | 0.458227 ± 0.003037 |
| ETTh1 | homogeneous_sparse | 0.447584 ± 0.005594 | 0.456953 ± 0.003783 |
| ETTh2 | heterogeneous_dense | 0.328862 ± 0.012632 | 0.393047 ± 0.008857 |
| ETTh2 | heterogeneous_no_memory | 0.332993 ± 0.019259 | 0.396389 ± 0.013671 |
| ETTh2 | heterogeneous_sparse | 0.336041 ± 0.007756 | 0.398143 ± 0.004906 |
| ETTh2 | homogeneous_dense | 0.342938 ± 0.013875 | 0.404772 ± 0.008887 |
| ETTh2 | homogeneous_no_memory | 0.343235 ± 0.012793 | 0.405299 ± 0.007531 |
| ETTh2 | homogeneous_sparse | 0.335482 ± 0.000705 | 0.398578 ± 0.001514 |

## 선택 동작 (첫 test8windows, 마지막 population layer)

| Variant | Selected/available slots | Empty read fraction | Real probability mass |
|---|---:|---:|---:|
| heterogeneous_dense | 0.998776 | 0.000000 | 0.881238 |
| heterogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| heterogeneous_sparse | 0.326631 | 0.149941 | 0.772621 |
| homogeneous_dense | 0.998048 | 0.000000 | 0.838302 |
| homogeneous_no_memory | 0.000000 | 1.000000 | 0.000000 |
| homogeneous_sparse | 0.390572 | 0.107705 | 0.825785 |

- Sparsemax는 일부 weight를 정확히0으로 만들 수 있지만 매 query의 sparsity를 보장하지 않는다. 실제 density/empty rate를 함께 읽는다.
- 실제 read 분모는 max(real_mass,1e-12)이다. 아주 작은 양의 real mass에서는 weight 합이1보다 작다. Lag mass/mean lag/entropy 해석에 이 floor를 반영해야 하며 check_read_mass.json에서 각 checkpoint로 실제 수식을 재검증한다.
- Off/dense/sparse의 nominal parameter/초기값은 같고 off의 retrieval 파라미터는 미사용이다. Homogeneous는 redundant-state 대조다.
- off/uniform/recent는 같은 checkpoint의 평가 개입이다. uniform/recent는 null을 우회하여 과거 후보를 강제 선택하고 gate를 재계산한다.
- 서로 다른 backbone은 용량/연산이 다르다. PatchTST와 TSMixer는 causal population-spiking 변형이며 원 논문 재현이 아니다.
- 알려진 slot의 operator 검사는 통과했지만 end-to-end synthetic recall 학습/에너지 측정/정답 ETT lag 검증은 not run.
- 과거slot mask는 raw input의 후속 recurrent-state 영향을 삭제하지 않는다. Dense search와 sparse use는 별개다.
- 이전 v1 결과는 scorer/gate/budget 등이 달라 sparsemax만의 효과를 비교하는 대조가 아니다. v2 dense와 sparse가 해당 대조다.
