# f-LIF population: causal Spike-TCN prototype (2차)

1차 shallow patch SNN에 두 개의 causal temporal convolution block을 추가한다. 기존 `Config`, `LOAD_MODEL`, `myModel`, `Embedding`, `train_one_epoch`, `val_one_epoch`, `EpochLog`, `EarlyStopping` 및 neorecall artifact 구조를 유지한다. 복사 출처는 `f_lif_pop_v1/forecasting`; 그 코드의 원형과 전체 repository 구조 검토는 해당 README/source_review 및 canonical PROJECT_LOG에 있다.

## Architecture

`[B,336,C] → channel-independent patch8 → Linear(8,32)×2 → population LIF → TemporalBlock(dilation1) → TemporalBlock(dilation2) → Linear(32×4,32) → flatten → Linear(42×32,H)`.

각 block은 다음 두 줄을 구현한다. `I`는 입력 전류, `S`는 K개 뉴런의 binary spike다.

```text
I_l = I_(l-1) + 2 * CausalConv1d(flatten_DK(S_(l-1)))
S_l = PopulationLIF_l(I_l)
```

Kernel3, dilations1/2, left padding만 사용한다. Convolution 경로 자체의 receptive field는7patch(56관측치)이고, 각 LIF의 recurrent state 및 memory는 전체42patch 과거에 접근한다. 따라서 retrieval-off 대조도 recurrence를 가진다. Residual은 spike 이전 전류에 더하며, 최종 head에는 마지막 층의 spike만 전달한다. Batch/time/window normalization, dropout, 추가 simulation 반복축은 없다.

세 population layer 모두 1차와 **동일한 PopulationLIF 구현**을 사용한다. K4/tau[2,4,8,16], beta=exp(-1/tau); homogeneous는 평균 beta를 반복한다. 같은 logical neuron의 constituent는 같은 전류를 받는다. Current pre-retrieval query로 과거 post-reset membrane vector를 검색하고 `v=u_bar+.05*sigmoid(gate)*memory`로 증거를 보강한다. Q/K cosine softmax temperature.25, reset만 detach, history는 full BPTT. 모든 state는 독립 window마다 초기화한다. 각 층은 자체 memory/Q/K/gate를 갖고, retrieval on/off 및 test intervention은 세 층 모두에 적용한다.

논문 근거: [Lv et al., Efficient and Effective Time-Series Forecasting with Spiking Neural Networks, §3.3](https://arxiv.org/html/2402.01533v2). 논문 Spike-TCN은 SEW shortcut 및 시계열 시점별 membrane reset을 사용한다. 이 실험은 chronological persistent population state와 **current residual**을 쓰는 변형이며 논문 원본 재현이 아니다. 논문의 성능/효율 수치를 이 구현의 결과로 사용하지 않는다.

## Experiment and training

각 H96/H720별 ETTh1/ETTh2 × homogeneous/heterogeneous × retrieval off/on × seeds7/13/21 =24 runs, 합계48. 본 비교는 flatten head만 학습한다. Last head shape 검사는 하되 학습은 not run. 동일 architecture 내 네 조건은 nominal parameter count/초기 trainable parameters가 같고 retrieval-off Q/K/gate는 미사용이다. 1차보다 convolution과 두 population layer만큼 깊이/parameter가 늘므로 두 architecture의 matched-capacity 비교는 아니다.

Data: `NSMT/forecasting/dataset/ETT-small/{ETTh1,ETTh2}.csv`. Train[0,8640), validation targets[8640,11520), test targets[11520,14400), 앞선336행 context, train-only StandardScaler,7variables,stride1,drop_lastFalse. H96 windows8209/2785/2785; H720 windows7585/2161/2161.

AdamW lr.001/wd.01, batch128, grad clip1, 최대10epochs, ReduceLROnPlateau(factor.5,patience1), early stopping3. Strict minimum validation MSE checkpoint를 복원하고 validation을 재현한 뒤 전체 test MSE/MAE 및 horizon별 값을 측정한다. Final epoch를 자동 선택하지 않는다. 짧은 예산이며 수렴/통계적 유의성을 주장하지 않는다. 첫 LR 감소 뒤 회복 시간이 짧을 수 있다. Longer-budget 비교는 별도 후속 항목이다.

## Run (cwd NSMT)

Python `/home/yschoi/.conda/envs/snn_recall/bin/python`, torch1.12.0+cu113, SpikingJelly0.0.0.0.14. Shell은 LD_LIBRARY_PATH를 해당 환경 lib에 맞춘다. GPU0–3에 각각1process, CPU threads2, deterministic, TF32 off. 환경/명령/source/data/checkpoint hash는 각 run JSON에 저장한다.

```bash
bash f_lif_pop_tcn_v1/forecasting/scripts/run_stage2.sh
# Or one horizon; never reuse a suite name:
bash f_lif_pop_tcn_v1/forecasting/scripts/run_ett.sh --suite ett-tcn-h96-20260914 --pred_len 96
bash f_lif_pop_tcn_v1/forecasting/scripts/run_ett.sh --suite ett-tcn-h720-20260914 --pred_len 720
```

`check_model.py --device cpu|cuda:0 --output <json>` verifies mechanism/causal convolution. `summarize.py --suite <suite>` and `check_summary.py --suite <suite>` aggregate and audit all24 runs. `check_reload.py --suite <suite> --pred_len 96|720` creates a fresh config/model object and re-evaluates a selected saved checkpoint. Add the conda Python and `LD_LIBRARY_PATH` prefix when running Python files directly.

## Artifacts and interpretation

`log/<suite>/<dataset>/<date>/<config>/seed+head+variant/` contains `logargs.txt`, `log/best_log_0.csv`, `log/final+result.csv`, `log/train_0` and `log/val_0` TensorBoard events, `model_state/{config.pt,best+model.pt}`, history/provenance/horizon metrics and forecast example. `results/<suite>/` contains full-precision JSON, manifest/completion, CSV comparisons, REPORT and audits. `scripts/queues/` keeps local lock/status files. Checkpoints/raw events/stdout stay local; text results/code/config/docs are versioned. No artifact is deleted to make a commit.

Metrics use all test elements with train-standardized targets. Dataset-first macro means and sample SD are over three seeds. Memory diagnostics sample the first8 test windows only: top-level is the last layer, `diagnostics.layers` contains each layer. Off/uniform/recent are **same-checkpoint interventions**, not separately trained controls. They cannot select a replacement best model using test data. Causal patch states do not imply sub-patch causality: all8 observed values form one patch.

Canonical append-only record: repository root [`docs/PROJECT_LOG.md`](../../../docs/PROJECT_LOG.md). `NSMT/docs/PROJECT_LOG.md` links to the same file. Resume using run JSON or per_run.csv log_path. New training uses a new suite. Code/log snapshots and completed runs have distinct annotated experiment tags. Main integration/push, learned tau/fractional kernel, synthetic recall, matched-capacity ablations, uniform/recent retraining and energy measurement are not part of this run.
