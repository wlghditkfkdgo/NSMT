# MIT-BIH residual test: IAND and membrane shortcut (MS) vs ADD (LIF SpkMixer muxlin, 2 mixer layers)

Rules: summarize_mitres.py docstring. 30 new runs (24 validation + 6 final), commit `ab3aeba47`.

## lr selection (seed-mean last-epoch validation macro-F1 %, fold 0)

| residual | 0.0025 | 0.005 | 0.01 | 0.02 | selected |
|---|--:|--:|--:|--:|---|
| IAND | 92.003 | 92.233 | 92.339 | 91.721 | 0.01 |
| MS | 92.286 | 92.577 | 92.447 | 92.225 | 0.005 |

## Test (seed means)

| arm | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy µJ (syops) |
|---|--:|--:|---|--:|
| IAND (new) | 92.21 | 98.59 | 99.27 / 86.42 / 96.31 / 80.23 / 98.83 | 9.00 |
| MS (new) | 92.19 | 98.58 | 99.26 / 85.79 / 96.22 / 80.68 / 99.01 | 7.69 |
| ADD base | 91.99 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 37.06 |
| SF-de PE on | 91.94 | 98.53 | 99.23 / 85.14 / 96.09 / 80.23 / 99.03 | 7.80 |
| SF-de PE off | 91.27 | 98.33 | 99.11 / 82.07 / 95.93 / 80.57 / 98.68 | 7.62 |
| MS lr 0.02 (11-08, report) | 91.89 | 98.55 | 99.24 / 85.57 / 96.44 / 79.24 / 98.98 | 7.92 |

| kind | A | B | d %p (42 / 7 / 2026) | mean %p | verdict | energy A/B | adoption bar (>= +0.1 %p, <= 1.5x) |
|---|---|---|---|--:|---|--:|---|
| R1 | IAND (new) | ADD base | +0.38 / +0.43 / -0.13 | +0.23 | **no clear difference** | 0.243 | passes |
| R2 | MS (new) | ADD base | +0.41 / +0.26 / -0.06 | +0.20 | **no clear difference** | 0.208 | passes |
| R3 | IAND (new) | SF-de PE on | +0.51 / +0.06 / +0.24 | +0.27 | **no clear difference** | 1.154 | passes |
| R4 | MS (new) | SF-de PE on | +0.54 / -0.12 / +0.32 | +0.25 | **no clear difference** | 0.986 | passes |
| R5 | IAND (new) | MS (new) | -0.04 / +0.18 / -0.08 | +0.02 | **no clear difference** | 1.171 | does not pass |
| R6 | IAND (new) | SF-de PE off | +0.83 / +1.45 / +0.53 | +0.94 | **A better** | 1.182 | passes |
| R7 | MS (new) | SF-de PE off | +0.87 / +1.27 / +0.61 | +0.92 | **A better** | 1.009 | passes |

Candidates (not worse than ADD, energy <= 1.5x SF-de): IAND, MS; recommendation (pre-registered rule): **IAND**
