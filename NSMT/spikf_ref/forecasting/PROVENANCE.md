# Provenance of spikf_ref/forecasting (2026-11-25/26, exp/f-lif-pop-v3)

Copied from the local SpikF clone `/home/yschoi/SpikF` (git commit `041e398` "release", with local uncommitted edits; the clone itself is only read). The released training scripts are `scripts/run_ETT{h1,h2,m1,m2}.sh` of that commit.

| file | sha256 (first 16) of the copied source | release blob | local edit in the clone |
|---|---|---|---|
| `models/SpikF.py` | `addf99f5a537e099` | `9a66a58826` | yes |
| `models/iSpikformer.py` | `5a21e1976cb4a1a6` | `8bde0d1c45` | yes |
| `exp/exp_ETT.py` | `231741b98435136a` | `55cffa5b38` | yes |
| `exp/exp_basic.py` | `2117f078843addb8` | `e328530d83` | no |
| `exp/__init__.py` | `e3b0c44298fc1c14` | `e69de29bb2` | no |
| `data_provider/ETT_data_loader.py` | `85df956288e784f6` | `98009ca29c` | no |
| `utils/tools.py` | `bdbc499ef835c7fc` | `09b55332c7` | yes |
| `utils/metrics.py` | `0da0131ef7b0f475` | `5d182f5c81` | no |
| `run_long.py` | `1d77e83093e485ee` | `3a0d16e182` | yes |

Local edits of the clone (before the copy; not ours): `models/SpikF.py` one commented line `# self.activ = MultiStepLIFNode(...)`
(the released model has only `nn.GELU()`); `exp/exp_ETT.py` syops energy logging, forecast plot and the iSpikformer branch;
`run_long.py` the `-s/--save` flag; `utils/tools.py` the plot helper; `models/iSpikformer.py` rewritten (imported, not used here).
Training (data, optimizer, loss, schedule, early stopping) is the released code.

Our edits in this copy (all marked `2026-11-25` / `2026-11-26` in the code):
- `models/SpikF.py`: `activ='gelu'|'lif'`; 'lif' = `MultiStepLIFNode(tau, detach_reset=True)` (the neuron of the SFS blocks) on the head
  hidden layer, run over the T axis only (the head tensor [T*B, D, hidden] is viewed as [T, B, D, hidden] around it). 'gelu' is the
  released path (bit-identical outputs, check-spikf-act-20261125/check_unit U1).
- `run_long.py`: `--activ`, `--out` (per-run checkpoint / test folders, `result.json`), `torch.cuda.set_device(args.rank)` instead of the
  hard-coded `set_device(1)`, `--fast_eval` / `--eval_bs` / `--syops_bs` (speed only; check_fast).
- `exp/exp_ETT.py`: passes `activ`; per-epoch losses and the tested epoch into `result.json`; under `--fast_eval` validation and test in
  batches (losses weighted by batch size), the syops loader at batch 1 with the neurons reset before counting.
