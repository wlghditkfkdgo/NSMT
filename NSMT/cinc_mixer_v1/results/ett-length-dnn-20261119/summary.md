# ETT input length by the DNN criterion

Rules and sources: module docstring of compare_dnn.py.

## Input 96

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4-cell mean |
|---|--:|--:|--:|--:|--:|
| LIF SpkMixer (ours) | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 |
| SF-de (ours) | 0.2921 | 0.4229 | 0.1768 | 0.4036 | 0.3238 |
| iTransformer (iTransformer Table 10) | 0.297 | 0.427 | 0.180 | 0.412 | 0.3290 |
| iTransformer, MAE loss (SpikF Table 9, 10) | 0.294 | 0.423 | 0.182 | 0.411 | 0.3275 |
| RLinear (iTransformer Table 10) | 0.288 | 0.420 | 0.182 | 0.407 | 0.3242 |
| PatchTST (re-run) (iTransformer Table 10) | 0.302 | 0.431 | 0.175 | 0.402 | 0.3275 |
| PatchTST (other re-run) (TimeMixer Table 13) | 0.308 | 0.436 | 0.183 | 0.412 | 0.3347 |
| TimeMixer (TimeMixer Table 13) | 0.289 | 0.412 | 0.175 | 0.391 | 0.3167 |
| TimesNet (iTransformer Table 10) | 0.340 | 0.462 | 0.187 | 0.408 | 0.3493 |
| FITS (SpikF Table 12) | 0.295 | 0.431 | 0.183 | 0.407 | 0.3290 |
| DLinear (re-run) (iTransformer Table 10) | 0.333 | 0.831 | 0.193 | 0.554 | 0.4778 |

| ours | vs DNN | R % (< 0 = ours better) | cells lower | ours better on average |
|---|---|--:|--:|---|
| LIF SpkMixer | iTransformer | -1.32 | 4/4 | yes |
| LIF SpkMixer | iTransformer, MAE loss | -1.04 | 3/4 | yes |
| LIF SpkMixer | RLinear | -0.10 | 2/4 | yes |
| LIF SpkMixer | PatchTST (re-run) | -0.64 | 2/4 | yes |
| LIF SpkMixer | PatchTST (other re-run) | -3.12 | 4/4 | yes |
| LIF SpkMixer | TimeMixer | +2.29 | 0/4 | no |
| LIF SpkMixer | TimesNet | -7.01 | 4/4 | yes |
| LIF SpkMixer | FITS | -1.48 | 3/4 | yes |
| LIF SpkMixer | DLinear (re-run) | -23.95 | 4/4 | yes |
| LIF SpkMixer | **summary** | | | better on average than 8 of 9 DNN rows |
| SF-de | iTransformer | -1.61 | 4/4 | yes |
| SF-de | iTransformer, MAE loss | -1.33 | 4/4 | yes |
| SF-de | RLinear | -0.40 | 2/4 | yes |
| SF-de | PatchTST (re-run) | -0.94 | 2/4 | yes |
| SF-de | PatchTST (other re-run) | -3.40 | 4/4 | yes |
| SF-de | TimeMixer | +1.99 | 0/4 | no |
| SF-de | TimesNet | -7.27 | 4/4 | yes |
| SF-de | FITS | -1.77 | 4/4 | yes |
| SF-de | DLinear (re-run) | -24.24 | 4/4 | yes |
| SF-de | **summary** | | | better on average than 8 of 9 DNN rows |

## Input 336

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4-cell mean |
|---|--:|--:|--:|--:|--:|
| LIF SpkMixer (ours) | 0.2964 | 0.3916 | 0.1726 | 0.3778 | 0.3096 |
| SF-de (ours) | 0.3044 | 0.3952 | 0.1721 | 0.3697 | 0.3103 |
| PatchTST/42 (PatchTST Table 3) | 0.274 | 0.379 | 0.165 | 0.367 | 0.2963 |
| DLinear (DLinear Table 2 / PatchTST Table 3) | 0.289 | 0.605 | 0.167 | 0.397 | 0.3645 |
| NLinear (DLinear Table 2) | 0.277 | 0.394 | 0.167 | 0.368 | 0.3015 |
| Linear (DLinear Table 2) | 0.288 | 0.698 | 0.168 | 0.413 | 0.3917 |

| ours | vs DNN | R % (< 0 = ours better) | cells lower | ours better on average |
|---|---|--:|--:|---|
| LIF SpkMixer | PatchTST/42 | +4.75 | 0/4 | no |
| LIF SpkMixer | DLinear | -8.56 | 2/4 | yes |
| LIF SpkMixer | NLinear | +3.09 | 1/4 | no |
| LIF SpkMixer | Linear | -11.70 | 2/4 | yes |
| LIF SpkMixer | **summary** | | | better on average than 2 of 4 DNN rows |
| SF-de | PatchTST/42 | +5.09 | 0/4 | no |
| SF-de | DLinear | -8.30 | 2/4 | yes |
| SF-de | NLinear | +3.42 | 0/4 | no |
| SF-de | Linear | -11.44 | 2/4 | yes |
| SF-de | **summary** | | | better on average than 2 of 4 DNN rows |

