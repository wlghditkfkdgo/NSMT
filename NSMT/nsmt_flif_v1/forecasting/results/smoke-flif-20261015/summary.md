# smoke-flif-20261015 — summary (6 RESULT rows)

## test MSE (mean ± std over seeds)
| cond | ETTh1 H96 |
|---|---|
| base | 0.393413 ±0.0000 (n=1) |
| neo_lif_both | 0.391952 ±0.0000 (n=1) |
| neo_lif_mlp | 0.392576 ±0.0000 (n=1) |
| neo_flif_both | 0.391284 ±0.0000 (n=1) |
| neo_flif_mlp | 0.393043 ±0.0000 (n=1) |
| hippo_only | 0.392141 ±0.0000 (n=1) |

## test MAE
| cond | ETTh1 H96 |
|---|---|
| base | 0.403238 ±0.0000 (n=1) |
| neo_lif_both | 0.402818 ±0.0000 (n=1) |
| neo_lif_mlp | 0.403453 ±0.0000 (n=1) |
| neo_flif_both | 0.402579 ±0.0000 (n=1) |
| neo_flif_mlp | 0.403058 ±0.0000 (n=1) |
| hippo_only | 0.405207 ±0.0000 (n=1) |

## total energy, paper convention (µJ; syops synaptic + neuron eq. 60/64)
| cond | ETTh1 H96 |
|---|---|
| base | 28.451 |
| neo_lif_both | 28.478 |
| neo_lif_mlp | 28.471 |
| neo_flif_both | 28.738 |
| neo_flif_mlp | 28.674 |
| hippo_only | 28.505 |

## total energy, direct-sum convention (µJ)
| cond | ETTh1 H96 |
|---|---|
| base | 28.451 |
| neo_lif_both | 28.478 |
| neo_lif_mlp | 28.471 |
| neo_flif_both | 29.514 |
| neo_flif_mlp | 29.191 |
| hippo_only | 28.505 |

## synaptic energy only (µJ, syops)
| cond | ETTh1 H96 |
|---|---|
| base | 16.360 |
| neo_lif_both | 16.370 |
| neo_lif_mlp | 16.360 |
| neo_flif_both | 16.380 |
| neo_flif_mlp | 16.390 |
| hippo_only | 16.490 |

## neocortex firing rate
| cond | ETTh1 H96 |
|---|---|
| base | 0.0636 |
| neo_lif_both | 0.0791 |
| neo_lif_mlp | 0.0717 |
| neo_flif_both | 0.0807 |
| neo_flif_mlp | 0.0739 |
| hippo_only | nan |

## hippocampus firing rate
| cond | ETTh1 H96 |
|---|---|
| base | 0.0524 |
| neo_lif_both | 0.0538 |
| neo_lif_mlp | 0.0540 |
| neo_flif_both | 0.0546 |
| neo_flif_mlp | 0.0556 |
| hippo_only | 0.0629 |

## training time (s)
| cond | ETTh1 H96 |
|---|---|
| base | 21 |
| neo_lif_both | 24 |
| neo_lif_mlp | 24 |
| neo_flif_both | 32 |
| neo_flif_mlp | 28 |
| hippo_only | 21 |

## criterion (2) — standard ETT test
**neo_flif_both vs hippo_only:** cells = ETTh1 H96: -0.22% (E×1.008, n=1); mean -0.22%, pairs lower 1/1, energy ×1.008 → route A fail, route B fail

**neo_flif_both vs neo_lif_both (same placement):** lower MSE in 1/1 cells (ETTh1 H96: -0.17%) → fail (need ≥ 3/4)

→ (2) **not passed** for neo_flif_both

**neo_flif_mlp vs hippo_only:** cells = ETTh1 H96: +0.23% (E×1.006, n=1); mean +0.23%, pairs lower 0/1, energy ×1.006 → route A fail, route B fail

**neo_flif_mlp vs neo_lif_mlp (same placement):** lower MSE in 0/1 cells (ETTh1 H96: +0.12%) → fail (need ≥ 3/4)

→ (2) **not passed** for neo_flif_mlp

(report only) neo_lif_both vs base: mean -0.37%, pairs lower 1/1

(report only) neo_lif_mlp vs base: mean -0.21%, pairs lower 1/1

(report only) neo_flif_both vs base: mean -0.54%, pairs lower 1/1

(report only) neo_flif_mlp vs base: mean -0.09%, pairs lower 1/1

(report only) hippo_only vs base: mean -0.32%, pairs lower 1/1

## criterion (1) — memory probe
| data | N | cond | S(1) | S(12)/S(1) | ΔMSE(keep 12)/MSE | n |
|---|---|---|---|---|---|---|
| ETTh1 | 24 | base | 0.0413 | 0.0017 | -3.70% | 1 |
| ETTh1 | 24 | neo_flif_both | 0.0917 | 0.0327 | -5.46% | 1 |


