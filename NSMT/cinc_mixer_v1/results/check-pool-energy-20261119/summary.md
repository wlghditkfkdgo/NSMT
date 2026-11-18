# Mean-pooling energy recount (pooled-integer rule), LIF SpkMixer

| task | arm | energy µJ, earlier rule | energy µJ, pooled-integer rule | ratio vs LIF base (earlier -> pooled) | ratio vs SF-de (earlier -> pooled) |
|---|---|--:|--:|---|---|
| ETT | LIF base | 10.28 | 10.28 | 1 | 0.9637 |
| ETT | LIF + mean pooling | 13.01 | 7.18 | 1.2678 -> 0.6958 | 1.2218 -> 0.6705 |
| ETT | SF-de | 10.67 | 10.67 | | 1 |
| MIT-BIH | LIF base | 9.44 | 9.44 | 1 | 0.9993 |
| MIT-BIH | LIF + mean pooling | 20.56 | 6.00 | 2.1787 -> 0.6360 | 2.1772 -> 0.6356 |
| MIT-BIH | SF-de | 9.45 | 9.45 | | 1 |

ETT per cell (seed mean, µJ): ETTh2 H96: base 9.06, mean 11.78 -> 5.91, SF-de 9.37; ETTh2 H720: base 10.67, mean 13.52 -> 7.64, SF-de 10.96; ETTm2 H96: base 9.89, mean 12.51 -> 6.76, SF-de 10.34; ETTm2 H720: base 11.49, mean 14.23 -> 8.40, SF-de 11.99
