# Fusion MUX probe (validation, CinC fold 0, LIF, residual OR; mean of seeds 42 / 7 / 2026, %)

## Firing

| fusion | F | R | F&R | F xor R | s_d | P(s_d given F&R) | P(s_d given F xor R) | P(s_d given neither) | m | OR residual adds | L | P(L given s_d) | P(L given not s_d) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdor | 6.02 | 6.04 | 0.44 | 11.17 | 10.44 | 4.32 | 45.88 | 5.99 | 16.91 | 15.05 | — | — | — |
| muxor | 7.39 | 7.53 | 0.75 | 13.42 | 9.79 | 8.23 | 44.24 | 4.41 | 6.69 | 5.90 | — | — | — |
| sdlin | 4.22 | 4.26 | 0.31 | 7.86 | 10.09 | 3.15 | 38.11 | 7.71 | 11.09 | 9.92 | 11.09 | 19.60 | 10.14 |
| muxlin | 6.47 | 5.73 | 0.84 | 10.51 | 11.30 | 8.17 | 36.67 | 8.32 | 4.41 | 3.93 | 9.03 | 32.16 | 6.09 |

## Gradient weight and where s_d fires

| fusion | to F (mean weight) | to L | select learns | s_d channel q10 / q50 / q90 | channels < 0.1 % | s_d by repeat 1-4 | muxor: m from the s_d branch |
|---|---|---|---|---|---|---|---|
| sdor | 25.25 | — | 9.79 | 8.28 / 10.44 / 12.50 | 0.00 | 6.85 11.51 12.18 11.25 | — |
| muxor | 50.25 | — | 4.29 | 7.92 / 9.74 / 11.78 | 0.00 | 6.81 10.54 11.32 10.47 | 89.72 |
| sdlin | 49.98 | 100.00 | 11.85 | 7.29 / 9.97 / 13.05 | 0.00 | 4.91 11.22 12.68 11.53 | — |
| muxlin | 49.79 | 11.30 | 2.15 | 7.89 / 11.00 / 14.90 | 0.00 | 6.46 12.43 13.67 12.63 | — |

## Per seed

- sdor/s42: val 69.62 (repro 0.0e+00); s_d 10.48, m 17.04, select learns 9.92; 32 s
- sdor/s7: val 71.08 (repro 0.0e+00); s_d 10.20, m 16.57, select learns 9.45; 23 s
- sdor/s2026: val 70.15 (repro 0.0e+00); s_d 10.65, m 17.12, select learns 10.00; 24 s
- muxor/s42: val 69.89 (repro 0.0e+00); s_d 9.77, m 6.60, select learns 4.24; 23 s
- muxor/s7: val 69.31 (repro 0.0e+00); s_d 9.72, m 6.53, select learns 4.21; 24 s
- muxor/s2026: val 69.50 (repro 0.0e+00); s_d 9.86, m 6.93, select learns 4.41; 23 s
- sdlin/s42: val 69.89 (repro 0.0e+00); s_d 10.04, m 10.92, select learns 11.92; 31 s
- sdlin/s7: val 71.30 (repro 0.0e+00); s_d 10.03, m 11.01, select learns 11.68; 31 s
- sdlin/s2026: val 74.88 (repro 0.0e+00); s_d 10.19, m 11.35, select learns 11.96; 32 s
- muxlin/s42: val 72.84 (repro 0.0e+00); s_d 11.41, m 4.25, select learns 2.13; 31 s
- muxlin/s7: val 70.96 (repro 0.0e+00); s_d 11.19, m 4.64, select learns 2.15; 31 s
- muxlin/s2026: val 73.58 (repro 0.0e+00); s_d 11.30, m 4.33, select learns 2.17; 29 s
