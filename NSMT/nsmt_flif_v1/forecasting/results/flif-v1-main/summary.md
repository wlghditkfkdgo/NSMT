# flif-v1-main — summary (72 RESULT rows)

## test MSE (mean ± std over seeds)
| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 0.283453 ±0.0030 (n=3) | 0.409741 ±0.0007 (n=3) | 0.175030 ±0.0003 (n=3) | 0.400208 ±0.0009 (n=3) |
| neo_lif_both | 0.285758 ±0.0029 (n=3) | 0.411570 ±0.0030 (n=3) | 0.175394 ±0.0003 (n=3) | 0.400197 ±0.0004 (n=3) |
| neo_lif_mlp | 0.282974 ±0.0008 (n=3) | 0.412465 ±0.0025 (n=3) | 0.175349 ±0.0004 (n=3) | 0.400342 ±0.0009 (n=3) |
| neo_flif_both | 0.282259 ±0.0009 (n=3) | 0.409369 ±0.0020 (n=3) | 0.176410 ±0.0008 (n=3) | 0.400929 ±0.0002 (n=3) |
| neo_flif_mlp | 0.285547 ±0.0026 (n=3) | 0.411662 ±0.0022 (n=3) | 0.175365 ±0.0003 (n=3) | 0.400378 ±0.0012 (n=3) |
| hippo_only | 0.283697 ±0.0027 (n=3) | 0.414134 ±0.0033 (n=3) | 0.175181 ±0.0004 (n=3) | 0.400131 ±0.0007 (n=3) |

## test MAE
| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 0.331197 ±0.0021 (n=3) | 0.429935 ±0.0004 (n=3) | 0.251395 ±0.0002 (n=3) | 0.393193 ±0.0005 (n=3) |
| neo_lif_both | 0.332688 ±0.0011 (n=3) | 0.431034 ±0.0019 (n=3) | 0.251461 ±0.0003 (n=3) | 0.393058 ±0.0001 (n=3) |
| neo_lif_mlp | 0.330441 ±0.0015 (n=3) | 0.431401 ±0.0015 (n=3) | 0.251448 ±0.0002 (n=3) | 0.393076 ±0.0004 (n=3) |
| neo_flif_both | 0.330719 ±0.0006 (n=3) | 0.429167 ±0.0010 (n=3) | 0.252805 ±0.0016 (n=3) | 0.393529 ±0.0004 (n=3) |
| neo_flif_mlp | 0.332332 ±0.0020 (n=3) | 0.431101 ±0.0013 (n=3) | 0.251650 ±0.0003 (n=3) | 0.393180 ±0.0006 (n=3) |
| hippo_only | 0.331289 ±0.0020 (n=3) | 0.432524 ±0.0023 (n=3) | 0.251435 ±0.0004 (n=3) | 0.393233 ±0.0005 (n=3) |

## total energy, paper convention (µJ; syops synaptic + neuron eq. 60/64)
| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 56.003 | 66.890 | 63.792 | 71.615 |
| neo_lif_both | 56.744 | 66.342 | 63.753 | 71.658 |
| neo_lif_mlp | 56.427 | 67.070 | 63.579 | 71.915 |
| neo_flif_both | 56.737 | 67.169 | 63.784 | 72.155 |
| neo_flif_mlp | 57.116 | 66.058 | 63.876 | 71.991 |
| hippo_only | 58.527 | 68.122 | 65.534 | 72.656 |

## total energy, direct-sum convention (µJ)
| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 56.003 | 66.890 | 63.792 | 71.615 |
| neo_lif_both | 56.744 | 66.342 | 63.753 | 71.658 |
| neo_lif_mlp | 56.427 | 67.070 | 63.579 | 71.915 |
| neo_flif_both | 57.772 | 68.204 | 65.337 | 73.708 |
| neo_flif_mlp | 57.634 | 66.576 | 64.911 | 73.026 |
| hippo_only | 58.527 | 68.122 | 65.534 | 72.656 |

## synaptic energy only (µJ, syops)
| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 33.173 | 44.093 | 38.243 | 46.770 |
| neo_lif_both | 33.720 | 43.650 | 38.233 | 46.780 |
| neo_lif_mlp | 33.470 | 44.297 | 38.160 | 47.020 |
| neo_flif_both | 33.453 | 44.063 | 37.957 | 46.800 |
| neo_flif_mlp | 33.910 | 43.193 | 38.127 | 46.807 |
| hippo_only | 35.810 | 45.660 | 40.480 | 48.260 |

## neocortex firing rate
| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 0.1069 | 0.1044 | 0.1900 | 0.1572 |
| neo_lif_both | 0.1307 | 0.1094 | 0.1686 | 0.1527 |
| neo_lif_mlp | 0.1176 | 0.1086 | 0.1642 | 0.1566 |
| neo_flif_both | 0.1195 | 0.1140 | 0.1610 | 0.1444 |
| neo_flif_mlp | 0.1299 | 0.1026 | 0.1650 | 0.1508 |
| hippo_only | nan | nan | nan | nan |

## hippocampus firing rate
| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 0.0655 | 0.0640 | 0.1162 | 0.0830 |
| neo_lif_both | 0.0752 | 0.0584 | 0.1152 | 0.0847 |
| neo_lif_mlp | 0.0719 | 0.0626 | 0.1105 | 0.0855 |
| neo_flif_both | 0.0719 | 0.0630 | 0.1068 | 0.0843 |
| neo_flif_mlp | 0.0761 | 0.0590 | 0.1107 | 0.0838 |
| hippo_only | 0.0729 | 0.0599 | 0.1126 | 0.0809 |

## training time (s)
| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 84 | 99 | 599 | 338 |
| neo_lif_both | 151 | 94 | 677 | 396 |
| neo_lif_mlp | 100 | 103 | 554 | 376 |
| neo_flif_both | 128 | 117 | 633 | 456 |
| neo_flif_mlp | 139 | 93 | 599 | 405 |
| hippo_only | 92 | 77 | 557 | 364 |

## criterion (2) — standard ETT test
**neo_flif_both vs hippo_only:** cells = ETTh2 H96: -0.51% (E×0.969, n=3), ETTh2 H720: -1.15% (E×0.986, n=3), ETTm2 H96: +0.70% (E×0.973, n=3), ETTm2 H720: +0.20% (E×0.993, n=3); mean -0.19%, pairs lower 6/12, energy ×0.980 → route A fail, route B fail

**neo_flif_both vs neo_lif_both (same placement):** lower MSE in 2/4 cells (ETTh2 H96: -1.22%, ETTh2 H720: -0.53%, ETTm2 H96: +0.58%, ETTm2 H720: +0.18%) → fail (need ≥ 3/4)

→ (2) **not passed** for neo_flif_both

**neo_flif_mlp vs hippo_only:** cells = ETTh2 H96: +0.65% (E×0.976, n=3), ETTh2 H720: -0.60% (E×0.970, n=3), ETTm2 H96: +0.10% (E×0.975, n=3), ETTm2 H720: +0.06% (E×0.991, n=3); mean +0.06%, pairs lower 7/12, energy ×0.978 → route A fail, route B fail

**neo_flif_mlp vs neo_lif_mlp (same placement):** lower MSE in 1/4 cells (ETTh2 H96: +0.91%, ETTh2 H720: -0.19%, ETTm2 H96: +0.01%, ETTm2 H720: +0.01%) → fail (need ≥ 3/4)

→ (2) **not passed** for neo_flif_mlp

(report only) neo_lif_both vs base: mean +0.37%, pairs lower 5/12

(report only) neo_lif_mlp vs base: mean +0.18%, pairs lower 3/12

(report only) neo_flif_both vs base: mean +0.11%, pairs lower 3/12

(report only) neo_flif_mlp vs base: mean +0.36%, pairs lower 1/12

(report only) hippo_only vs base: mean +0.31%, pairs lower 5/12

## future interval (72 rows) — MSE
| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 0.172566 ±0.0006 (n=3) | 0.268724 ±0.0021 (n=3) | 0.125270 ±0.0004 (n=3) | 0.225897 ±0.0007 (n=3) |
| neo_lif_both | 0.171966 ±0.0011 (n=3) | 0.268958 ±0.0021 (n=3) | 0.125203 ±0.0006 (n=3) | 0.226077 ±0.0001 (n=3) |
| neo_lif_mlp | 0.172042 ±0.0005 (n=3) | 0.266354 ±0.0043 (n=3) | 0.125036 ±0.0003 (n=3) | 0.225819 ±0.0003 (n=3) |
| neo_flif_both | 0.172792 ±0.0004 (n=3) | 0.263927 ±0.0022 (n=3) | 0.125836 ±0.0011 (n=3) | 0.226238 ±0.0002 (n=3) |
| neo_flif_mlp | 0.172233 ±0.0010 (n=3) | 0.268027 ±0.0017 (n=3) | 0.124559 ±0.0002 (n=3) | 0.226196 ±0.0000 (n=3) |
| hippo_only | 0.172501 ±0.0010 (n=3) | 0.268647 ±0.0040 (n=3) | 0.124545 ±0.0001 (n=3) | 0.225375 ±0.0006 (n=3) |

## future interval — MAE
| cond | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| base | 0.285607 ±0.0008 (n=3) | 0.361594 ±0.0011 (n=3) | 0.237213 ±0.0001 (n=3) | 0.326957 ±0.0009 (n=3) |
| neo_lif_both | 0.285460 ±0.0019 (n=3) | 0.361598 ±0.0014 (n=3) | 0.237297 ±0.0004 (n=3) | 0.326905 ±0.0003 (n=3) |
| neo_lif_mlp | 0.285322 ±0.0017 (n=3) | 0.359636 ±0.0032 (n=3) | 0.237136 ±0.0001 (n=3) | 0.326872 ±0.0004 (n=3) |
| neo_flif_both | 0.285891 ±0.0014 (n=3) | 0.358077 ±0.0015 (n=3) | 0.238023 ±0.0019 (n=3) | 0.327352 ±0.0005 (n=3) |
| neo_flif_mlp | 0.284720 ±0.0006 (n=3) | 0.360842 ±0.0005 (n=3) | 0.236745 ±0.0003 (n=3) | 0.327119 ±0.0004 (n=3) |
| hippo_only | 0.285664 ±0.0019 (n=3) | 0.361278 ±0.0033 (n=3) | 0.237014 ±0.0005 (n=3) | 0.326736 ±0.0005 (n=3) |

## future interval — direction check (energy columns not available here)
**neo_flif_both vs hippo_only:** cells = ETTh2 H96: +0.17% (E×nan, n=3), ETTh2 H720: -1.76% (E×nan, n=3), ETTm2 H96: +1.04% (E×nan, n=3), ETTm2 H720: +0.38% (E×nan, n=3); mean -0.04%, pairs lower 4/12, energy ×nan → route A fail, route B fail

**neo_flif_both vs neo_lif_both (same placement):** lower MSE in 1/4 cells (ETTh2 H96: +0.48%, ETTh2 H720: -1.87%, ETTm2 H96: +0.51%, ETTm2 H720: +0.07%) → fail (need ≥ 3/4)

→ (2) **not passed** for neo_flif_both

**neo_flif_mlp vs hippo_only:** cells = ETTh2 H96: -0.16% (E×nan, n=3), ETTh2 H720: -0.23% (E×nan, n=3), ETTm2 H96: +0.01% (E×nan, n=3), ETTm2 H720: +0.36% (E×nan, n=3); mean -0.00%, pairs lower 4/12, energy ×nan → route A fail, route B fail

**neo_flif_mlp vs neo_lif_mlp (same placement):** lower MSE in 1/4 cells (ETTh2 H96: +0.11%, ETTh2 H720: +0.63%, ETTm2 H96: -0.38%, ETTm2 H720: +0.17%) → fail (need ≥ 3/4)

→ (2) **not passed** for neo_flif_mlp

(report only) neo_lif_both vs base: mean -0.06%, pairs lower 5/12

(report only) neo_lif_mlp vs base: mean -0.35%, pairs lower 7/12

(report only) neo_flif_both vs base: mean -0.26%, pairs lower 7/12

(report only) neo_flif_mlp vs base: mean -0.22%, pairs lower 7/12

(report only) hippo_only vs base: mean -0.22%, pairs lower 8/12

## criterion (1) — memory probe
| data | N | cond | S(1) | S(12)/S(1) | ΔMSE(keep 12)/MSE | n |
|---|---|---|---|---|---|---|
| ETTh2 | 24 | base | 0.0733 | 0.0196 | +1.98% | 6 |
| ETTh2 | 24 | neo_flif_both | 0.0844 | 0.0480 | +3.05% | 6 |
| ETTh2 | 24 | neo_flif_mlp | 0.0762 | 0.0242 | +2.35% | 6 |
| ETTh2 | 24 | neo_lif_both | 0.0832 | 0.0157 | +1.85% | 6 |
| ETTh2 | 24 | neo_lif_mlp | 0.0707 | 0.0137 | +2.45% | 6 |
| ETTm2 | 24 | base | 0.1133 | 0.1102 | +6.65% | 6 |
| ETTm2 | 24 | neo_flif_both | 0.1352 | 0.1339 | +3.24% | 6 |
| ETTm2 | 24 | neo_flif_mlp | 0.1284 | 0.0953 | +4.87% | 6 |
| ETTm2 | 24 | neo_lif_both | 0.1226 | 0.0914 | +4.80% | 6 |
| ETTm2 | 24 | neo_lif_mlp | 0.1120 | 0.0891 | +5.59% | 6 |

ETTh2 neo_flif_both: M1 fail (ratio 0.048 vs control 0.016); M2 PASS (ΔMSE(keep12) +3.05% vs control +1.85%)
ETTh2 neo_flif_mlp: M1 fail (ratio 0.024 vs control 0.014); M2 fail (ΔMSE(keep12) +2.35% vs control +2.45%)
ETTm2 neo_flif_both: M1 fail (ratio 0.134 vs control 0.091); M2 fail (ΔMSE(keep12) +3.24% vs control +4.80%)
ETTm2 neo_flif_mlp: M1 fail (ratio 0.095 vs control 0.089); M2 fail (ΔMSE(keep12) +4.87% vs control +5.59%)
