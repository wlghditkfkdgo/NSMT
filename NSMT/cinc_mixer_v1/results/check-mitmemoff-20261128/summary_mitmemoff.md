# MIT-BIH memoryless control (SpkMixer muxlin ADD, token_memory off vs LIF memory; 3 seeds)

- macro-F1: memoryless 90.13 vs LIF memory 91.99; per seed 42 / 7 / 2026: -1.72 / -1.78 / -2.06 %p; mean -1.85 %p -> **worse**
- accuracy: 98.03 vs 98.56 %
- class F1 N / S / V / F / Q: memoryless 98.94 / 78.59 / 95.14 / 79.42 / 98.58; LIF memory 99.24 / 85.94 / 96.31 / 79.49 / 98.95
- syops energy: 36.15 vs 37.06 uJ (ratio 0.975)
- check: mixer-shuffle drop of the memoryless model +0.0000 / +0.0000 / +0.0000 %p (|drop| < 0.05 required)
