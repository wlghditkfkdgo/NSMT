| model | shuffle | macro-F1 unshuffled | mean drop %p (seeds 42 / 7 / 2026) | seeds dropped | verdict | accuracy drop %p |
|---|---|--:|---|--:|---|--:|
| SpkMixer muxlin ADD | mixer | 91.99 | +35.06 (+35.75 / +35.07 / +34.35) | 3 / 3 | **order used** | +13.545 |
| SpkMixer muxlin ADD | full | 91.99 | +35.06 (+35.75 / +35.07 / +34.35) | 3 / 3 | **order used** | +13.545 |
| SF-de (SSA + rel. bias) | SSA | 91.94 | +26.47 (+25.97 / +25.47 / +27.95) | 3 / 3 | **order used** | +13.921 |
| SF-de (SSA + rel. bias) | full | 91.94 | +26.47 (+25.97 / +25.47 / +27.95) | 3 / 3 | **order used** | +13.921 |

Paired contrast drop_mux(mixer) - drop_SF-de(SSA): mean +8.59 %p (+9.78 / +9.60 / +6.40) -> **muxlin depends more**
