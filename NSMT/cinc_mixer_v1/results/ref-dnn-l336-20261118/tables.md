# DNN results at look-back L = 336 on ETTh2 / ETTm2 (copied 2026-11-18 from the arXiv PDFs with pdftotext -layout; PDFs kept local)

Sources:
- PatchTST, Nie et al., ICLR 2023, arXiv 2211.14730, Table 3. Text: "PatchTST/42 means the number of input patches is 42, which has the
  default look-back window L = 336." Baselines: "default look-back window L = 96 for Transformer-based models, and L = 336 for DLinear",
  FEDformer / Autoformer / Informer re-run for L in {24, 48, 96, 192, 336, 720} with the best result kept.
- DLinear, Zeng et al., AAAI 2023, arXiv 2205.13504, Table 2. Text: "we report L=336 for LTSF-Linear and L=96 for Transformers by default."

| model (L) | ETTh2 H96 MSE / MAE | ETTh2 H720 MSE / MAE | ETTm2 H96 MSE / MAE | ETTm2 H720 MSE / MAE | source |
|---|---|---|---|---|---|
| PatchTST/42 (336) | 0.274 / 0.336 | 0.379 / 0.422 | 0.165 / 0.255 | 0.367 / 0.385 | PatchTST Table 3 |
| PatchTST/64 (512) | 0.274 / 0.337 | 0.379 / 0.422 | 0.166 / 0.256 | 0.362 / 0.385 | PatchTST Table 3 |
| DLinear (336) | 0.289 / 0.353 | 0.605 / 0.551 | 0.167 / 0.260 | 0.397 / 0.421 | both tables |
| NLinear (336) | 0.277 / 0.338 | 0.394 / 0.436 | 0.167 / 0.255 | 0.368 / 0.384 | DLinear Table 2 |
| Linear (336) | 0.288 / 0.352 | 0.698 / 0.595 | 0.168 / 0.262 | 0.413 / 0.435 | DLinear Table 2 |

This branch at input 336 (seed mean of 3, test MSE; results/flif-v1-ett-spkmux-pn336/summary.md):

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|--:|--:|--:|--:|
| LIF SpkMixer 336 | 0.2964 | 0.3916 | 0.1726 | 0.3778 |
| SF-de 336 | 0.3044 | 0.3952 | 0.1721 | 0.3697 |

Not checked: whether the test protocol is the same (test-set batching, normalisation such as RevIN vs this branch's window normalisation).
The values above are for orientation only until that is checked.
