# Query form, population neurons: T = N memory and token order on CinC folds 0-4 — summary

See the module docstring for sources and rules. Test CinC score (%), best-validation weights.

## fold means (seeds 42 / 7 / 2026)

| model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 5-fold mean | energy µJ |
|---|---|---|---|---|---|---|---|
| Spikformer-style @30 | 65.60 | 66.30 | 65.89 | 64.44 | 66.46 | 65.74 | 126.8 |
| query pop_lif ordered | 66.60 | 67.81 | 66.84 | 68.30 | 69.09 | 67.73 | 158.0 |
| query pop_lif memoryless | 65.22 | 65.42 | 64.63 | 65.29 | 66.08 | 65.33 | 148.8 |
| query pop_lif Neocortex-shuffled | 66.18 | 66.55 | 65.37 | 65.72 | 66.53 | 66.07 | 158.7 |
| query pop_flif ordered | 66.47 | 68.91 | 65.78 | 67.95 | 68.77 | 67.58 | 179.7 |
| query pop_flif memoryless | 65.28 | 65.99 | 64.96 | 66.63 | 68.16 | 66.20 | 150.2 |
| query pop_flif Neocortex-shuffled | 65.97 | 66.83 | 65.43 | 66.76 | 66.62 | 66.32 | 178.1 |

## A (main) memory share = ordered − memoryless

| comparison | pooled A − B (%p) | 95 % t interval | A > B pairs | verdict | per fold (mean, verdict) | energy A/B |
|---|---|---|---|---|---|---|
| A pop_lif | +2.40 | [+1.48, +3.32] | 14/15 | better | f0 +1.38 better; f1 +2.39 better; f2 +2.21 better; f3 +3.01 better; f4 +3.00 better | 1.06 |
| A pop_flif | +1.37 | [+0.59, +2.16] | 12/15 | better | f0 +1.19 better; f1 +2.93 better; f2 +0.82 no clear difference; f3 +1.32 better; f4 +0.61 better | 1.20 |

## B (main) order share = ordered − Neocortex-shuffled

| comparison | pooled A − B (%p) | 95 % t interval | A > B pairs | verdict | per fold (mean, verdict) | energy A/B |
|---|---|---|---|---|---|---|
| B pop_lif | +1.65 | [+0.64, +2.67] | 11/15 | better | f0 +0.42 no clear difference; f1 +1.26 better; f2 +1.47 better; f3 +2.58 better; f4 +2.55 better | 1.00 |
| B pop_flif | +1.25 | [+0.58, +1.93] | 12/15 | better | f0 +0.50 no clear difference; f1 +2.08 better; f2 +0.35 no clear difference; f3 +1.19 better; f4 +2.15 better | 1.01 |

## C order-free share = Neocortex-shuffled − memoryless

| comparison | pooled A − B (%p) | 95 % t interval | A > B pairs | verdict | per fold (mean, verdict) | energy A/B |
|---|---|---|---|---|---|---|
| C pop_lif | +0.74 | [-0.07, +1.56] | 10/15 | better | f0 +0.96 better; f1 +1.13 better; f2 +0.74 better; f3 +0.43 no clear difference; f4 +0.45 no clear difference | 1.07 |
| C pop_flif | +0.12 | [-0.76, +1.01] | 8/15 | no clear difference | f0 +0.69 better; f1 +0.85 better; f2 +0.47 no clear difference; f3 +0.14 no clear difference; f4 -1.54 worse | 1.19 |

## report: − Spikformer-style @30

| comparison | pooled A − B (%p) | 95 % t interval | A > B pairs | verdict | per fold (mean, verdict) | energy A/B |
|---|---|---|---|---|---|---|
| Neocortex-shuffled pop_lif − Spikformer-style @30 | +0.33 | [-0.51, +1.17] | 10/15 | no clear difference | f0 +0.58 better; f1 +0.25 no clear difference; f2 -0.52 worse; f3 +1.28 better; f4 +0.07 no clear difference | 1.25 |
| memoryless pop_lif − Spikformer-style @30 | -0.41 | [-0.90, +0.08] | 4/15 | no clear difference | f0 -0.38 no clear difference; f1 -0.88 worse; f2 -1.26 worse; f3 +0.85 better; f4 -0.38 no clear difference | 1.17 |
| Neocortex-shuffled pop_flif − Spikformer-style @30 | +0.58 | [-0.29, +1.46] | 10/15 | better | f0 +0.38 no clear difference; f1 +0.53 better; f2 -0.46 no clear difference; f3 +2.32 better; f4 +0.15 no clear difference | 1.40 |
| memoryless pop_flif − Spikformer-style @30 | +0.46 | [-0.42, +1.35] | 8/15 | no clear difference | f0 -0.31 no clear difference; f1 -0.32 no clear difference; f2 -0.93 worse; f3 +2.19 better; f4 +1.70 better | 1.18 |

## report only (added after the results): per-class test F1 (%) and firing rates, 5-fold × 3-seed means

CinC score = mean F1 of N / AF / Other (Noisy excluded); macro-F1 includes Noisy. rho = mean firing rate.

| model | macro-F1 | F1 N | F1 AF | F1 Other | F1 Noisy | rho_neo | rho_hippo |
|---|---|---|---|---|---|---|---|
| Spikformer-style @30 | 61.39 | 82.91 | 58.50 | 55.81 | 48.34 | — | 0.0972 |
| query pop_lif ordered | 62.36 | 82.29 | 64.06 | 56.83 | 46.28 | 0.0629 | 0.1155 |
| query pop_lif memoryless | 62.24 | 82.51 | 58.37 | 55.10 | 52.97 | 0.0011 | 0.0971 |
| query pop_lif Neocortex-shuffled | 61.70 | 82.53 | 59.43 | 56.26 | 48.56 | 0.0767 | 0.1145 |
| query pop_flif ordered | 62.27 | 82.36 | 63.21 | 57.16 | 46.35 | 0.0644 | 0.1174 |
| query pop_flif memoryless | 61.67 | 82.74 | 60.21 | 55.66 | 48.08 | 0.0074 | 0.1016 |
| query pop_flif Neocortex-shuffled | 61.72 | 82.52 | 59.98 | 56.47 | 47.92 | 0.0607 | 0.1184 |

| difference (%p, 15-pair mean) | macro-F1 | F1 N | F1 AF | F1 Other | F1 Noisy |
|---|---|---|---|---|---|
| A pop_lif | +0.12 | -0.22 | +5.68 | +1.73 | -6.70 |
| B pop_lif | +0.67 | -0.24 | +4.63 | +0.58 | -2.29 |
| C pop_lif | -0.54 | +0.01 | +1.06 | +1.16 | -4.41 |
| A pop_flif | +0.60 | -0.38 | +3.00 | +1.50 | -1.73 |
| B pop_flif | +0.55 | -0.16 | +3.23 | +0.69 | -1.57 |
| C pop_flif | +0.05 | -0.22 | -0.23 | +0.81 | -0.16 |
