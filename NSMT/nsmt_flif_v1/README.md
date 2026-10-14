# nsmt_flif_v1 — NSMT with the neocortex-path neuron replaced (spikeDE LIF / f-LIF)

Copy of `NSMT/nsmt_base/forecasting/` (the paper's code, repo commit `dd2d4c9ea`; ETT Table 13 reproduced 16/16)
with the neocortex-path spiking neuron made replaceable and a hippocampus-only control. **Defaults are
bit-identical to nsmt_base** (`check_flif.py` c1: same init, outputs, gradients, checkpoint outputs; c5: the
nsmt_base ETTh1 H96 checkpoint gives the same `final+result.csv` first 13 columns). Design, decisions and all
logs: repo-root `docs/MODI_PROJECT_LOG.md`.

## What changed (relative to nsmt_base/forecasting)
| file | change |
|---|---|
| `neurons.py` | **new**: verbatim copy of `f_lif_pop_v3/forecasting/neurons.py` (spikeDE LIF / f-LIF classes, checked against the original spikeDE code) with the two helpers it imported inlined |
| `layers.py` | `SpikLinearLayer(neuron=...)`; `NeoNeuron` adapter ([T, B, D, N'] / [T, B, N', D] ↔ [T, B·N', D]); `build_neo_neuron` |
| `ours.py` | `Embedding(neuron=)`, `TemporalBlock(neuron_fn=)`, `Block.forward(x, mx=None)` (mx None → key/value = hippocampus), `myModel(neo_neuron, neo_where, neo_tau, neo_alpha, fde_method, fde_step, fde_memory, hippo_only)` |
| `config.py` | `--neo_neuron {sj_lif,lif,flif}` `--neo_where {both,mlp}` `--neo_tau` `--neo_alpha` `--fde_method` `--fde_step` `--fde_memory` `--hippo_only`; non-default conditions get a directory-name suffix |
| `model.py` | passes the new options (`getattr` defaults keep old `config.pt` loadable) |
| `energy.py` + `test.py` | neuron-intrinsic energy after the f-LIF paper App. D.3 (eq. 60 LIF, eq. 64 f-LIF/FFT, plus our direct-sum variant); 9 columns appended after the original 13 of `final+result.csv`; per-layer `log/neuron_energy.csv` |
| `check_flif.py` | checks c1–c5 (`results/checks/check_flif.txt`) |
| `ett_future.py` | second-stage evaluation on the ETT future interval (ETTh rows [14400, 17420), ETTm [57600, 69680)); registry refuses a second look at the same checkpoint |
| `neo_memory_probe.py` | long-term-memory evidence: lag-sensitivity and keep-last-k curves with the neocortex input perturbed only |
| `analysis_aux_target.py` | the read-only measurement behind decision 6 (the current L_aux target is 96% linear in the current patch) |
| `scripts/run_ett.sh` | `COND={base,neo_lif_both,neo_lif_mlp,neo_flif_both,neo_flif_mlp,hippo_only}`; same paper hyperparameters as nsmt_base; `scripts/smoke.sh`; `scripts/queues/flif-v1-main.txt` (72 runs) |

## Run (from `NSMT/`)
```bash
COND=neo_flif_both bash nsmt_flif_v1/forecasting/scripts/run_ett.sh ETTh2 96 42 0      # <DS> <PL> <seed> <gpu>
bash scripts/gpu_pool2.sh nsmt_flif_v1/forecasting/scripts/queues/flif-v1-main.txt 0,1,2,3
cd nsmt_flif_v1/forecasting && python check_flif.py -nd 0 && bash scripts/smoke.sh 0
python ett_future.py --run <run_dir> --split future -nd 0 ; python neo_memory_probe.py --run <run_dir> -nd 0
```
Environment `snn_recall` (Python 3.10, torch 1.12.0+cu113, spikingjelly cupy backend). Outputs: `log/<suite>/`
(stdout, per-run `log/` + `model_state/`), `results/<suite>/raw.txt`. Checkpoints, TensorBoard events and stdout
stay local; `log/{best_log_0.csv,final+result.csv,logargs.txt,neuron_energy.csv}` and text results go to Git.
