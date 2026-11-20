# Aux-target memory check (pre-registered, seed 42, no relative position bias) — summary

See the docstring of aux_check.py. e = share of the aux target's variance left unexplained on the validation split (0 = perfect, 1 = no better than the position-wise average). memory = normal Neocortex, no memory = every token alone, shuffled = the Neocortex reads the tokens in a shuffled order.

## e per run

| task | neuron | target | e memory | e no memory | e shuffled |
|---|---|---|---|---|---|
| ETTh2 | pop LIF | token itself (s = 0) | 0.5640 | 0.4737 | 0.6376 |
| ETTh2 | pop LIF | past, 2 tokens back (s = 2) | 0.7456 | 0.9824 | 0.9288 |
| ETTh2 | pop LIF | past, 4 tokens back (s = 4) | 0.8579 | 0.9999 | 0.9691 |
| ETTh2 | pop LIF | next patch (s = -2) | 0.8526 | 0.9839 | 0.9262 |
| ETTh2 | pop f-LIF | token itself (s = 0) | 0.4079 | 0.1354 | 0.3201 |
| ETTh2 | pop f-LIF | past, 2 tokens back (s = 2) | 0.6946 | 0.8421 | 0.8402 |
| ETTh2 | pop f-LIF | past, 4 tokens back (s = 4) | 0.9090 | 0.9778 | 0.9442 |
| ETTh2 | pop f-LIF | next patch (s = -2) | 0.8258 | 0.8387 | 0.8431 |
| ETTm2 | pop LIF | token itself (s = 0) | 0.2422 | 0.0501 | 0.4136 |
| ETTm2 | pop LIF | past, 2 tokens back (s = 2) | 0.3896 | 0.6473 | 0.7753 |
| ETTm2 | pop LIF | past, 4 tokens back (s = 4) | 0.4213 | 0.8897 | 0.9439 |
| ETTm2 | pop LIF | next patch (s = -2) | 0.5829 | 0.6422 | 0.7778 |
| ETTm2 | pop f-LIF | token itself (s = 0) | 0.1234 | 0.0509 | 0.1233 |
| ETTm2 | pop f-LIF | past, 2 tokens back (s = 2) | 0.3933 | 0.6449 | 0.6438 |
| ETTm2 | pop f-LIF | past, 4 tokens back (s = 4) | 0.4585 | 0.8814 | 0.8709 |
| ETTm2 | pop f-LIF | next patch (s = -2) | 0.5706 | 0.6354 | 0.6483 |
| MIT-BIH | pop LIF | token itself (s = 0) | 0.3155 | 0.0770 | 0.6294 |
| MIT-BIH | pop LIF | past, 2 tokens back (s = 2) | 0.3747 | 0.6139 | 0.7397 |
| MIT-BIH | pop LIF | past, 4 tokens back (s = 4) | 0.3595 | 0.8306 | 0.7288 |
| MIT-BIH | pop LIF | next patch (s = -2) | 0.6288 | 0.5036 | 0.9067 |
| MIT-BIH | pop f-LIF | token itself (s = 0) | 0.1765 | 0.0721 | 0.2029 |
| MIT-BIH | pop f-LIF | past, 2 tokens back (s = 2) | 0.3352 | 0.6009 | 0.6894 |
| MIT-BIH | pop f-LIF | past, 4 tokens back (s = 4) | 0.3732 | 0.8208 | 0.7327 |
| MIT-BIH | pop f-LIF | next patch (s = -2) | 0.5472 | 0.4888 | 0.5985 |

## verdict (d = g(s) − g(0); needs memory: d ≥ 0.10 and e memory ≤ 0.90; ETT: mean of the two cells, d > 0 in both; deciding control = no memory if it can reconstruct its own token (e < 0.90), else shuffled)

| task | neuron | target | e memory | no memory: d (per cell) | no memory: own-token e | shuffled: d (per cell) | shuffled: own-token e | deciding | verdict |
|---|---|---|---|---|---|---|---|---|---|
| ETT | pop LIF | past, 2 tokens back (s = 2) | 0.5676 | +0.3885 (+0.3272 / +0.4498) needs memory | 0.4737 | +0.1620 (+0.1096 / +0.2143) needs memory | 0.6376 | no memory | needs memory |
| ETT | pop LIF | past, 4 tokens back (s = 4) | 0.6396 | +0.4464 (+0.2323 / +0.6605) needs memory | 0.4737 | +0.1944 (+0.0376 / +0.3511) needs memory | 0.6376 | no memory | needs memory |
| ETT | pop LIF | next patch (s = -2) | 0.7177 | +0.2365 (+0.2216 / +0.2514) needs memory | 0.4737 | +0.0118 (+0.0000 / +0.0235) does not need memory | 0.6376 | no memory | needs memory |
| ETT | pop f-LIF | past, 2 tokens back (s = 2) | 0.5439 | +0.3721 (+0.4201 / +0.3241) needs memory | 0.1354 | +0.2421 (+0.2335 / +0.2507) needs memory | 0.3201 | no memory | needs memory |
| ETT | pop f-LIF | past, 4 tokens back (s = 4) | 0.6837 | +0.4185 (+0.3415 / +0.4954) needs memory | 0.1354 | +0.2679 (+0.1231 / +0.4126) needs memory | 0.3201 | no memory | needs memory |
| ETT | pop f-LIF | next patch (s = -2) | 0.6982 | +0.2114 (+0.2854 / +0.1373) needs memory | 0.1354 | +0.0915 (+0.1051 / +0.0779) does not need memory | 0.3201 | no memory | needs memory |
| MIT-BIH | pop LIF | past, 2 tokens back (s = 2) | 0.3747 | +0.4778 (+0.4778) needs memory | 0.0770 | +0.0512 (+0.0512) does not need memory | 0.6294 | no memory | needs memory |
| MIT-BIH | pop LIF | past, 4 tokens back (s = 4) | 0.3595 | +0.7096 (+0.7096) needs memory | 0.0770 | +0.0555 (+0.0555) does not need memory | 0.6294 | no memory | needs memory |
| MIT-BIH | pop LIF | next patch (s = -2) | 0.6288 | +0.1133 (+0.1133) needs memory | 0.0770 | -0.0359 (-0.0359) does not need memory | 0.6294 | no memory | needs memory |
| MIT-BIH | pop f-LIF | past, 2 tokens back (s = 2) | 0.3352 | +0.3700 (+0.3700) needs memory | 0.0721 | +0.3278 (+0.3278) needs memory | 0.2029 | no memory | needs memory |
| MIT-BIH | pop f-LIF | past, 4 tokens back (s = 4) | 0.3732 | +0.5519 (+0.5519) needs memory | 0.0721 | +0.3332 (+0.3332) needs memory | 0.2029 | no memory | needs memory |
| MIT-BIH | pop f-LIF | next patch (s = -2) | 0.5472 | +0.0459 (+0.0459) does not need memory | 0.0721 | +0.0249 (+0.0249) does not need memory | 0.2029 | no memory | does not need memory |

| task | targets that pass (both neurons) | best |
|---|---|---|
| ETT | past, 2 tokens back (s = 2), past, 4 tokens back (s = 4), next patch (s = -2) | past, 4 tokens back (s = 4) |
| MIT-BIH | past, 2 tokens back (s = 2), past, 4 tokens back (s = 4) | past, 4 tokens back (s = 4) |

## report only: linear baselines on the data (ridge, train -> validation; no trained model)

| task | target | e_lin from token i alone | e_lin from tokens i, i-1, ... (history) |
|---|---|---|---|
| ETTh2 | token itself (s = 0) | 0.0000 | 0.0000 |
| ETTh2 | past, 2 tokens back (s = 2) | 0.8387 | 0.0000 |
| ETTh2 | past, 4 tokens back (s = 4) | 0.9539 | 0.0000 |
| ETTh2 | next patch (s = -2) | 0.8366 | 0.6547 |
| ETTm2 | token itself (s = 0) | 0.0000 | 0.0000 |
| ETTm2 | past, 2 tokens back (s = 2) | 0.6747 | 0.0000 |
| ETTm2 | past, 4 tokens back (s = 4) | 0.9295 | 0.0000 |
| ETTm2 | next patch (s = -2) | 0.6723 | 0.6070 |
| MIT-BIH | token itself (s = 0) | 0.0000 | 0.0000 |
| MIT-BIH | past, 2 tokens back (s = 2) | 0.7213 | 0.0000 |
| MIT-BIH | past, 4 tokens back (s = 4) | 0.9514 | 0.0000 |
| MIT-BIH | next patch (s = -2) | 0.6963 | 0.5915 |

## report only: task metric, Neocortex firing, energy (seed 42; not judged)

| task | neuron | variant | target | validation | test | rho_neo | energy µJ |
|---|---|---|---|---|---|---|---|
| ETTh2 | pop LIF | memory | s = 0 | 0.2175 | 0.2908 | 0.092526 | 10.90 |
| ETTm2 | pop LIF | memory | s = 0 | 0.1242 | 0.1762 | 0.18929 | 12.13 |
| ETTh2 | pop LIF | no memory | s = 0 | 0.2149 | 0.2914 | 0.018301 | 10.60 |
| ETTm2 | pop LIF | no memory | s = 0 | 0.1237 | 0.1766 | 0.074974 | 11.52 |
| ETTh2 | pop LIF | shuffled | s = 0 | 0.2139 | 0.2941 | 0.125152 | 11.03 |
| ETTm2 | pop LIF | shuffled | s = 0 | 0.1231 | 0.1758 | 0.225658 | 12.27 |
| ETTh2 | pop f-LIF | memory | s = 0 | 0.2170 | 0.2891 | 0.09628 | 10.89 |
| ETTm2 | pop f-LIF | memory | s = 0 | 0.1245 | 0.1761 | 0.236351 | 12.28 |
| ETTh2 | pop f-LIF | no memory | s = 0 | 0.2141 | 0.2949 | 0.056024 | 10.76 |
| ETTm2 | pop f-LIF | no memory | s = 0 | 0.1239 | 0.1766 | 0.085418 | 11.59 |
| ETTh2 | pop f-LIF | shuffled | s = 0 | 0.2198 | 0.2884 | 0.110445 | 10.96 |
| ETTm2 | pop f-LIF | shuffled | s = 0 | 0.1238 | 0.1775 | 0.273525 | 12.29 |
| ETTh2 | pop LIF | memory | s = 2 | 0.2165 | 0.2894 | 0.070421 | 10.74 |
| ETTm2 | pop LIF | memory | s = 2 | 0.1249 | 0.1765 | 0.196073 | 12.18 |
| ETTh2 | pop LIF | no memory | s = 2 | 0.2139 | 0.2956 | 0.001006 | 10.48 |
| ETTm2 | pop LIF | no memory | s = 2 | 0.1244 | 0.1770 | 0.043113 | 11.37 |
| ETTh2 | pop LIF | shuffled | s = 2 | 0.2162 | 0.2886 | 0.08177 | 10.82 |
| ETTm2 | pop LIF | shuffled | s = 2 | 0.1231 | 0.1759 | 0.144719 | 11.85 |
| ETTh2 | pop f-LIF | memory | s = 2 | 0.2170 | 0.2884 | 0.086116 | 10.83 |
| ETTm2 | pop f-LIF | memory | s = 2 | 0.1240 | 0.1756 | 0.20298 | 12.09 |
| ETTh2 | pop f-LIF | no memory | s = 2 | 0.2123 | 0.2964 | 0.032946 | 10.66 |
| ETTm2 | pop f-LIF | no memory | s = 2 | 0.1241 | 0.1767 | 0.064186 | 11.47 |
| ETTh2 | pop f-LIF | shuffled | s = 2 | 0.2158 | 0.2943 | 0.1044 | 10.94 |
| ETTm2 | pop f-LIF | shuffled | s = 2 | 0.1236 | 0.1758 | 0.176316 | 11.94 |
| ETTh2 | pop LIF | memory | s = 4 | 0.2164 | 0.2880 | 0.060208 | 10.70 |
| ETTm2 | pop LIF | memory | s = 4 | 0.1246 | 0.1759 | 0.178436 | 12.05 |
| ETTh2 | pop LIF | no memory | s = 4 | 0.2126 | 0.2985 | 0.000262 | 10.48 |
| ETTm2 | pop LIF | no memory | s = 4 | 0.1235 | 0.1765 | 0.021666 | 11.18 |
| ETTh2 | pop LIF | shuffled | s = 4 | 0.2156 | 0.2946 | 0.077061 | 10.84 |
| ETTm2 | pop LIF | shuffled | s = 4 | 0.1234 | 0.1754 | 0.109083 | 11.67 |
| ETTh2 | pop f-LIF | memory | s = 4 | 0.2160 | 0.2870 | 0.056544 | 10.67 |
| ETTm2 | pop f-LIF | memory | s = 4 | 0.1237 | 0.1758 | 0.244586 | 12.22 |
| ETTh2 | pop f-LIF | no memory | s = 4 | 0.2182 | 0.2889 | 0.011174 | 10.55 |
| ETTm2 | pop f-LIF | no memory | s = 4 | 0.1233 | 0.1754 | 0.045079 | 11.36 |
| ETTh2 | pop f-LIF | shuffled | s = 4 | 0.2141 | 0.2965 | 0.087549 | 10.87 |
| ETTm2 | pop f-LIF | shuffled | s = 4 | 0.1230 | 0.1759 | 0.132748 | 11.72 |
| ETTh2 | pop LIF | memory | s = -2 | 0.2186 | 0.2899 | 0.083443 | 10.83 |
| ETTm2 | pop LIF | memory | s = -2 | 0.1246 | 0.1761 | 0.155361 | 11.90 |
| ETTh2 | pop LIF | no memory | s = -2 | 0.2144 | 0.2937 | 0.000754 | 10.50 |
| ETTm2 | pop LIF | no memory | s = -2 | 0.1236 | 0.1758 | 0.043926 | 11.35 |
| ETTh2 | pop LIF | shuffled | s = -2 | 0.2134 | 0.2977 | 0.106084 | 10.95 |
| ETTm2 | pop LIF | shuffled | s = -2 | 0.1236 | 0.1755 | 0.141827 | 11.83 |
| ETTh2 | pop f-LIF | memory | s = -2 | 0.2193 | 0.2887 | 0.070924 | 10.76 |
| ETTm2 | pop f-LIF | memory | s = -2 | 0.1242 | 0.1765 | 0.185305 | 11.98 |
| ETTh2 | pop f-LIF | no memory | s = -2 | 0.2151 | 0.2896 | 0.034906 | 10.66 |
| ETTm2 | pop f-LIF | no memory | s = -2 | 0.1231 | 0.1758 | 0.06583 | 11.48 |
| ETTh2 | pop f-LIF | shuffled | s = -2 | 0.2142 | 0.2992 | 0.098429 | 10.92 |
| ETTm2 | pop f-LIF | shuffled | s = -2 | 0.1228 | 0.1765 | 0.169904 | 11.90 |
| MIT-BIH | pop LIF | memory | s = 0 | 0.9202 | 0.9080 | 0.079989 | 11.49 |
| MIT-BIH | pop LIF | no memory | s = 0 | 0.9185 | 0.9055 | 0.021257 | 10.33 |
| MIT-BIH | pop LIF | shuffled | s = 0 | 0.9233 | 0.9038 | 0.042371 | 10.82 |
| MIT-BIH | pop f-LIF | memory | s = 0 | 0.9171 | 0.9083 | 0.098055 | 12.95 |
| MIT-BIH | pop f-LIF | no memory | s = 0 | 0.9119 | 0.9011 | 0.032055 | 10.49 |
| MIT-BIH | pop f-LIF | shuffled | s = 0 | 0.9107 | 0.9026 | 0.055219 | 12.39 |
| MIT-BIH | pop LIF | memory | s = 2 | 0.9135 | 0.9077 | 0.078232 | 11.47 |
| MIT-BIH | pop LIF | no memory | s = 2 | 0.9122 | 0.9024 | 0.018324 | 10.07 |
| MIT-BIH | pop LIF | shuffled | s = 2 | 0.9097 | 0.9033 | 0.053948 | 11.03 |
| MIT-BIH | pop f-LIF | memory | s = 2 | 0.9227 | 0.9095 | 0.097155 | 12.86 |
| MIT-BIH | pop f-LIF | no memory | s = 2 | 0.9229 | 0.9099 | 0.03435 | 10.38 |
| MIT-BIH | pop f-LIF | shuffled | s = 2 | 0.9146 | 0.9044 | 0.060707 | 12.43 |
| MIT-BIH | pop LIF | memory | s = 4 | 0.9174 | 0.9096 | 0.089262 | 11.57 |
| MIT-BIH | pop LIF | no memory | s = 4 | 0.9131 | 0.9015 | 0.023781 | 10.31 |
| MIT-BIH | pop LIF | shuffled | s = 4 | 0.9157 | 0.9016 | 0.062486 | 11.17 |
| MIT-BIH | pop f-LIF | memory | s = 4 | 0.9181 | 0.9069 | 0.099195 | 13.02 |
| MIT-BIH | pop f-LIF | no memory | s = 4 | 0.9203 | 0.9046 | 0.027035 | 10.36 |
| MIT-BIH | pop f-LIF | shuffled | s = 4 | 0.9079 | 0.8999 | 0.058309 | 12.32 |
| MIT-BIH | pop LIF | memory | s = -2 | 0.9190 | 0.9100 | 0.079662 | 11.40 |
| MIT-BIH | pop LIF | no memory | s = -2 | 0.9152 | 0.9023 | 0.015615 | 10.01 |
| MIT-BIH | pop LIF | shuffled | s = -2 | 0.9088 | 0.9018 | 0.05636 | 11.27 |
| MIT-BIH | pop f-LIF | memory | s = -2 | 0.9173 | 0.9094 | 0.094602 | 12.99 |
| MIT-BIH | pop f-LIF | no memory | s = -2 | 0.9224 | 0.9042 | 0.028962 | 10.22 |
| MIT-BIH | pop f-LIF | shuffled | s = -2 | 0.9179 | 0.9057 | 0.050156 | 12.47 |
