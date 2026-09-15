# PopulationLIF v2 — selective membrane-memory experiment

원 concept의 **Branch B: 직접 막전위 검색**을 구현한다. 논리적 뉴런은 같은 전류를 받는 K개 LIF이며, 과거 post-reset population state를 현재 pre-retrieval state로 검색한다. 이번 수정은 read mask/크기 구분/retrieval-aware use gate에 집중한다. 정확한 fractional derivative 구현이 아니다.

기준 commit: `f8215f54106980bad7c782bf08acaef871175c34` (`exp/f-lif-pop-tcn-v1-20260914`). 별도 branch `exp/f-lif-pop-v2`. 기존 v1 코드·결과·checkpoint는 보존한다. 원 concept 및 사용자 implementation review는 수정하지 않고 내용과 SHA256를 사전 기록에 남긴다.

## 원문과 구현의 대응

| 원문 요구/선택 | v2 구현 | 검증/한계 |
|---|---|---|
| §5 population 이질성 | K4/tau[2,4,8,16], shared current; homogeneous=mean beta repeated | 구성원 동일성/다양성 검사; homogeneous는 redundant-state 대조 |
| §7/14 memory semantics | 각 과거 patch의 post-reset membrane, full BPTT/reset만 detach | mask는 read 제외이며 history 삭제나 raw input 삭제가 아님 |
| §8 current-conditioned relevance | learned Q/K의 negative mean squared distance | 크기 비례 상태 구분, task relevance 자체는 학습 결과로 판단 |
| §10 read selection | sparsemax의 positive support를 명시적 mask로 사용 | 제외 weight 정확히0; 모든 query에서 sparse임을 보장하지 않음 |
| §11 memory use | null 후보 + real probability mass × retrieval-aware sigmoid gate | null만 선택되면 M/gate/evidence 정확히0; top-k 강제 선택 없음 |
| §12 population-level read | [BC,D,J] mask/weights를 K 전체에 공유 | constituent별 별도 slot선택 없음 |
| §13 population output | [T,BC,D,K] binary spikes | head에서 DK projection까지 보존 |
| §17 actual ordered time | patch8/stride8로 시간순 진행, 모든 backbone causal | 인과성은 patch단위, patch 내부8개 관측은 함께 입력 |
| §20 sparse vs efficient | 모든 과거 score 계산 후 sparse use | 계산/에너지 이득 주장 안 함 |
| §24 미정 설계 | fixed tau/post-reset/no fractional prior/no cross-window history | 다른 선택은 아직 비교하지 않음 |
| §26 대조군 | homogeneous/heterogeneous × off/dense/sparse | dense와 sparse는 scorer/null/gate/초기parameters를 공유 |
| §27/28 검색 검증 | known-slot/context switch/distractor/null operator 검사, 실제 support/empty/lag 진단 | end-to-end synthetic forecasting 학습과 ETT 정답lag 검증은 not run |

## Neuron equations

```text
charged_k = beta_k * state_k + (1 - beta_k) * current
score_j = -mean_K((Wq * charged - Wk * history_j)^2) / temperature
probability = sparsemax([score_1, ..., score_J, learned_null_logit])
real_mass = sum(probability[:J])
mask_j = probability_j > 0
weight_j = probability_j / real_mass       (all zero when real_mass=0)
memory = sum_j(weight_j * history_j)
gate = real_mass * sigmoid(Linear([charged, memory, best_score-null_logit, real_mass]))
voltage = charged + gamma * gate * memory
spike = surrogate(voltage - threshold)
state = voltage - threshold * spike.detach()
```

Dense control uses softmax instead of sparsemax over the same real+null logits. Off bypasses reading but allocates the same parameters. The mask is the learned score distribution's support; it is not an independently learned binary parameter. The actual frozen implementation uses `weight = probability / max(real_mass, 1e-12)`: weights sum to1 when real_mass is at least1e-12, to a smaller value for tiny positive mass, and to0 for zero mass. The separate gate also scales by real_mass. Sparse normalization changes support, relative weights and null mass together, so this is a comparison of **dense vs sparse selection rules**, not just multiplication by a fixed mask.

Q/K are identity-initialized bias-free K×K transforms **without L2 normalization**. Value amplitude remains intact. Null logit starts at−1 and is learned. Gate Linear(2K+2,1) starts at0 so its sigmoid starts at.5; real_mass can still force it to0. Gamma.05,temperature.25,threshold1,input_scale2. The beta+gamma bound is conservative magnitude control, not a proof of gradient/perturbation stability. Sparsemax has inactive regions with zero score gradient; support/null collapse must be checked in measured diagnostics.

Sparsemax follows [Martins & Astudillo 2016, Alg.1 and Eq.14](https://proceedings.mlr.press/v48/martins16.pdf). A custom backward uses support times the centered incoming gradient. This avoids torch1.12 CUDA scatter backward's deterministic-mode limitation. CPU/GPU finite-difference Jacobian checks pass. Sorting/comparing all history still costs work; sparsity is not an efficiency claim.

## Four staged architectures

All use channel-independent `[B,L,C] → [T,BC,patch8] → Linear8→D32 ×2 → PopulationLIF` and the same `Linear(DK,32) → flatten → Linear(42×32,H)` readout. Common embedding/readout are initialized before additional blocks, so their initial values also match across backbones for a given seed/horizon. All non-patch backbones add two current-residual blocks and hence two population layers. No window/temporal normalization, dropout, or repeated simulation-time axis is used.

1. **patch**: the shallow patch SNN, no additional block.
2. **tcn**: spike-to-current Conv1d(DK,D,kernel3,dilation1/2), left padding, current residual, PopulationLIF. Conv-only field7patches; neuron recurrence and retrieval can access the whole preceding window. Adapted from the [Spike-TCN family](https://arxiv.org/html/2402.01533v2); persistent patch state/current residual differ from that paper's reset/SEW choices.
3. **patchtst**: causal multi-head **softmax attention** over projected population spikes (D32,4heads,learned position), current residual, PopulationLIF; then a per-patch two-layer feature MLP/current residual/PopulationLIF. This is a **hybrid spiking PatchTST adaptation**, not an all-spiking Spikformer or exact [PatchTST](https://arxiv.org/abs/2211.14730) reproduction. Causal attention prevents future observed patches entering past neuron states.
4. **tsmixer**: lower-triangular two-layer temporal MLP over projected spikes, current residual/PopulationLIF; then the same feature block. Time and hidden-feature mixing follow the [TSMixer idea](https://arxiv.org/abs/2303.06053). This causal, channel-independent population-spiking adaptation differs from the original ANN and its multivariate mixing/normalization.

Backbone parameter counts differ. Off vs dense vs sparse within each backbone has matched nominal parameters; off's retrieval parameters are unused. Causal TSMixer also has permanently masked upper-triangular nominal parameters. No matched-effective-capacity claim is made.

## Experiment matrix and training

Each architecture: ETTh1/ETTh2 × seeds7/13/21 × homogeneous/heterogeneous × off/dense/sparse × H96/H720 =72 runs. Four ordered stages =288 planned runs. Each horizon is a separate36-run suite. Every stage advances only after complete runs and audits; advancement does not depend on test improvement. No test-driven hyperparameter or architecture selection.

Train[0,8640), validation target[8640,11520), test target[11520,14400), preceding336 context. Train-only StandardScaler,7variables,stride1,drop_lastFalse. H96 windows8209/2785/2785, H7207585/2161/2161. Same local CSVs as v1. Full test MSE/MAE use all window×horizon×channel elements with float64 accumulation.

Defaults: AdamW lr.001,wd.01,batch128,gradient clip1,maximum30epochs,early-stop6,ReduceLROnPlateau factor.5/patience2. Scheduler tracks validation MSE. Strict minimum validation MSE checkpoint is restored and independently reproduced before testing. Epoch cap is a budget, not a convergence claim. Different v1 training/scoring settings mean v1 is historical context; v2 dense is the primary sparse-selection control.

## Code and run (cwd NSMT)

`Config`, `LOAD_MODEL`, `myModel`, `Embedding`, `train_one_epoch`, `val_one_epoch`, `EpochLog`, `EarlyStopping`, data_provider and log layout follow v1/model_v1/neorecall. Loader and logging utilities are copied without behavior changes. New components are in layers.py/backbones.py; ours.py selects a small backbone. `NSMT/forecasting/f-LIF_pop_v2.py` is the import entry.

```bash
# Full ordered pipeline, local completion commits/tags:
bash f_lif_pop_v2/forecasting/scripts/run_pipeline.sh
# Single new run, separate suite:
env LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib /home/yschoi/.conda/envs/snn_recall/bin/python f_lif_pop_v2/forecasting/train.py --suite my-new-suite --architecture patch --pred_len 720 --read_mode sparse
```

The shell uses `/home/yschoi/.conda/envs/snn_recall/bin/python` and that environment's lib path (torch1.12.0+cu113/SpikingJelly0.0.0.0.14). GPUs0–3, two workers per GPU by default, CPUthreads2/process, deterministic/TF32off. Preflight H720 smoke peak allocated memory was under3.7GiB/process; two workers fit the48GiB A6000. GPU placement is recorded; these shared-device wall times are not standalone efficiency measurements.

The launcher refuses existing outputs and stops later stages if a run/audit fails. `scripts/queues/<pipeline>.json` shows live stage/expected HEAD, suite JSONs show per-job PID/status/stdout. Full pipeline stdout is local under log. Training source and expected HEAD are checked before launches. Keep source/HEAD stable until a stage completes; automatic stage commits then update the expected HEAD. A branch/HEAD change stops pending work and preserves files. `--resume` can re-audit a fully trained suite after a postprocessing failure, checking identical source hashes and training budgets, then continue the missing stages. It never retrains or overwrites individual results/checkpoints. Restart/resume of a partially trained suite is **not implemented**.

## Logging and artifacts

`log/<suite>/<data>/<date>/<config>/seed+variant/`: logargs.txt; log/best_log_0.csv, final+result.csv, train_0/val_0 TensorBoard, history/provenance/horizon CSV/example JSON; model_state/config.pt and best+model.pt. Checkpoints/raw events/stdout remain local. `results/<suite>/`: full-precision run JSONs, manifest/completion, per_run/per_task/macro, paired and paired_macro_by_seed CSVs, layer diagnostics, REPORT, aggregate and audits. Macro first averages datasets within seed, then reports3seed mean/sample SD, separately for each horizon.

Diagnostics sample the first8 test windows: support density, empty-read fraction, real_mass, lag distribution, spike rates, membrane diversity and evidence scale per layer. Sum of lag_mass equals the mean of `real_mass/max(real_mass,1e-12)`; it can be below the nonempty-read fraction. The historical mean_lag field divides the lag first moment by nonempty fraction, so it is attenuated for reads affected by the floor, rather than a strictly normalized conditional mean. The same caveat applies to read entropy. `check_read_mass.json` rechecks this formula from every saved checkpoint and records the affected fraction. The entire test is also evaluated with off/uniform/recent interventions. Uniform/recent force a real-history choice (bypass null), then recompute the gate; these are policy interventions, not independently trained controls or a test-based checkpoint selection.

Each completed architecture appends a detailed dated record to the one canonical repository-root docs/PROJECT_LOG.md (`NSMT/docs/PROJECT_LOG.md` links there), commits only the named task artifacts/code/log, and creates annotated `exp/f-lif-pop-v2-<architecture>-<completion-date>`. Snapshot tags are explicitly snapshots. Raw artifacts are never deleted for Git. No main integration or remote push occurs.
