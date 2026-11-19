# AD energy test (B), M = 1: residual selection (validation only) at the selected lr

Score = seed-mean best validation MSE / that of OR. "res" = lowest of {OR, IAND, ADD}; "spike" = lowest of {OR, IAND} (0/1 residuals); ties keep OR.

| dataset | residual | val MSE per seed (42, 7, 2026) | mean | score | best epoch (mean) |
|---|---|---|---:|---:|---:|
| MSL | or (lr 0.002) | 0.050482, 0.050469, 0.048094 | 0.049682 | 1.00000 | 9.0 |
| MSL | iand (lr 0.002) | 0.098438, 0.085232, 0.087082 | 0.090251 | 1.81658 | 6.0 |
| MSL | add (lr 0.002) | 0.049157, 0.047035, 0.045282 | 0.047158 | 0.94920 | 8.7 |
| MSL | **selected res add, spike or** | | | | |
| SMAP | or (lr 0.002) | 0.014686, 0.014076, 0.014848 | 0.014537 | 1.00000 | 9.0 |
| SMAP | iand (lr 0.002) | 0.020116, 0.018366, 0.018799 | 0.019094 | 1.31348 | 9.0 |
| SMAP | add (lr 0.002) | 0.011880, 0.011780, 0.012105 | 0.011922 | 0.82011 | 9.0 |
| SMAP | **selected res add, spike or** | | | | |
| PSM | or (lr 0.002) | 0.015472, 0.015797, 0.015890 | 0.015720 | 1.00000 | 7.0 |
| PSM | iand (lr 0.002) | 0.014579, 0.014281, 0.014761 | 0.014540 | 0.92498 | 5.0 |
| PSM | add (lr 0.002) | 0.012281, 0.012760, 0.012310 | 0.012450 | 0.79202 | 7.7 |
| PSM | **selected res add, spike iand** | | | | |

selected_res.json: {"res": {"MSL": "add", "SMAP": "add", "PSM": "add"}, "spike": {"MSL": "or", "SMAP": "or", "PSM": "iand"}}
