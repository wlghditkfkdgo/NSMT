# MUX(a) select probe (validation, CinC fold 0, LIF, residual OR; mean of seeds 42 / 7 / 2026, %)

## Firing and overlap

| fusion | F | R | F&R | F^R | s_d | P(s_d given F&R) | P(s_d given F^R) | P(s_d given neither) | P(F&R given s_d) | P(F^R given s_d) |
|---|---|---|---|---|---|---|---|---|---|---|
| or | 8.55 | 8.70 | 1.58 | 14.08 | — | — | — | — | — | — |
| sdor | 6.02 | 6.04 | 0.44 | 11.17 | 10.44 | 4.32 | 45.88 | 5.99 | 0.18 | 49.09 |
| sdlin | 4.22 | 4.26 | 0.31 | 7.86 | 10.09 | 3.15 | 38.11 | 7.71 | 0.10 | 29.70 |

## What MUX(a) would change (mixer residual)

| fusion | x | m | m != x | trained residual change | MUX add / delete (data m) | MUX add / delete (data F v R) | MUX != trained (m) | MUX != trained (F v R) |
|---|---|---|---|---|---|---|---|---|
| sdor | 7.25 | 16.91 | 20.44 | 15.05 | 9.29 / 0.00 | 4.54 / 0.54 | 5.76 | 11.05 |
| sdlin | 7.66 | 11.09 | 16.41 | 9.92 | 1.71 / 0.97 | 2.63 / 0.86 | 9.18 | 12.33 |

## Gradient reach (mean gradient weight at the measured activations)

| fusion | to m: trained / MUX | to x: trained / MUX | select near threshold | select learns (near threshold and m != x) |
|---|---|---|---|---|
| sdor | 54.83 / 10.44 | 45.17 / 89.56 | 11.72 | 6.13 |
| sdlin | 51.71 / 10.09 | 48.29 / 89.91 | 11.85 | 2.64 |

## Where s_d fires

| fusion | channel rate q10 / q50 / q90 | dead channels (< 0.1 %) | > 50 % | by token decile (first ... last) | by repeat 1-4 | per window: N / AF / Other / Noisy |
|---|---|---|---|---|---|---|
| sdor | 8.28 / 10.44 / 12.50 | 0.00 | 0.00 | 11.4 10.9 10.5 10.6 10.3 10.4 10.1 10.4 10.2 9.5 | 6.85 11.51 12.18 11.25 | 9.98 / 13.09 / 10.55 / 9.69 |
| sdlin | 7.29 / 9.97 / 13.05 | 0.00 | 0.00 | 11.5 10.5 10.1 10.1 9.9 9.9 9.8 9.9 9.8 9.2 | 4.91 11.22 12.68 11.53 | 9.25 / 13.68 / 10.56 / 9.62 |

## Plug-in validation score (mixer residual replaced, no retraining; out of distribution)

| fusion | full | mux | mux_fr | mux_rate | keep_x (mixer ignored) | take_m (no residual) |
|---|---|---|---|---|---|---|
| or | 67.41 | — | — | — | 25.33 (-42.08; -41.1/-38.7/-46.4) | 61.80 (-5.60; -8.0/-5.1/-3.6) |
| sdor | 70.28 | 65.23 (-5.06; -2.5/-4.9/-7.8) | 32.97 (-37.31; -38.1/-40.0/-33.8) | 55.26 (-15.03; -16.4/-16.1/-12.6) | 25.31 (-44.97; -45.0/-45.8/-44.1) | 68.25 (-2.03; -0.8/-1.9/-3.3) |
| sdlin | 72.02 | 41.05 (-30.97; -26.2/-34.4/-32.3) | 33.90 (-38.13; -33.0/-43.3/-38.2) | 38.68 (-33.35; -29.3/-37.9/-32.8) | 32.23 (-39.80; -34.4/-44.6/-40.4) | 62.85 (-9.18; -8.6/-10.0/-9.0) |

## Per seed (F&R, s_d, select learns, %)

- or/s42: F&R 1.57; repro 0.0e+00; 34 s
- or/s7: F&R 1.74; repro 0.0e+00; 32 s
- or/s2026: F&R 1.44; repro 0.0e+00; 32 s
- sdor/s42: F&R 0.51, s_d 10.48, select learns 6.16, MUX change 9.32; repro 0.0e+00; 64 s
- sdor/s7: F&R 0.43, s_d 10.20, select learns 5.94, MUX change 9.06; repro 0.0e+00; 57 s
- sdor/s2026: F&R 0.39, s_d 10.65, select learns 6.27, MUX change 9.51; repro 0.0e+00; 53 s
- sdlin/s42: F&R 0.27, s_d 10.04, select learns 2.56, MUX change 2.57; repro 0.0e+00; 63 s
- sdlin/s7: F&R 0.29, s_d 10.03, select learns 2.63, MUX change 2.69; repro 0.0e+00; 64 s
- sdlin/s2026: F&R 0.37, s_d 10.19, select learns 2.73, MUX change 2.79; repro 0.0e+00; 63 s
