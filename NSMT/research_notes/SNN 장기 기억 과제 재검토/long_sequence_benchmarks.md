# Continuous-signal long-sequence benchmarks used by non-spiking long-sequence models (2023–2026): protocols, numbers, evidence of long-range need, compute

Scope note: every number below comes from a source I opened in this session (arXiv HTML/PDF full text, official GitHub code/README, or PMC/UCI pages). Pre-2023 sources are marked **[background]**. "UEA-6" means the six-dataset UEA suite introduced by Log-NCDE (Walker et al., ICML 2024). Accuracies are in %, ± is the standard deviation over 5 runs unless stated.

## Q1. Which continuous-signal benchmarks do 2023–2026 long-sequence papers use?

### Takeaway
Since 2024 the main time-series testbed for long-sequence models has been the **UEA-6 suite** (EigenWorms, SelfRegulationSCP1/SCP2, EthanolConcentration, Heartbeat, MotorImagery) plus **PPG-DaLiA heart-rate regression** (49,920-step windows), all run under one shared protocol (Log-NCDE code, the same 162-configuration grid, 5 seeds). Most new state-space / recurrent / neural-CDE papers reuse it: LinOSS (ICLR 2025), Rough Transformers (NeurIPS 2024), SLiCE and LrcSSM (NeurIPS 2025), D-LinOSS (2025), plus 2026 preprints. The older S4-family set is still cited and was reported in ICLR/ICML 2023 papers: **raw Speech Commands (16,000 steps)**, **BIDMC vital signs (4,000 steps)**, **sMNIST/psMNIST/sCIFAR (784–1,024 steps)** and Long Range Arena (LRA).

### Cited Findings
- Log-NCDE (ICML 2024) picked its six UEA datasets with two rules: more than 200 total cases, and the six longest series. The reason given is that "datasets with many observations have previously proved challenging for deep learning approaches". Original train and test cases are merged and re-split 70:15:15 at random, following Morrill et al. (2021) — [Walker et al., Log-NCDE, arXiv v3](https://arxiv.org/html/2402.18512v3)
- Log-NCDE also introduced the PPG-DaLiA regression protocol: 15 subjects, ~150 min each, 6 channels (blood volume pulse, electrodermal activity, temperature, 3-axis acceleration), a per-subject 70:15:15 split, and sliding windows of length 49,920 with step 4,992 — [Log-NCDE v3](https://arxiv.org/html/2402.18512v3); code [Benjamin-Walker/log-neural-cdes](https://github.com/Benjamin-Walker/log-neural-cdes)
- LinOSS (ICLR 2025) adopts "a recently proposed long-range sequential benchmark introduced in Walker et al. (2024)" (UEA-6) and PPG-DaLiA ("very long-range interactions"), with the same grid search "to ensure perfect comparability". It adds a weather forecasting task (720→720) — [Rusch & Rus, LinOSS](https://arxiv.org/html/2410.03943); code [tk-rusch/linoss](https://github.com/tk-rusch/linoss)
- D-LinOSS (2025): UEA-6 + PPG-DaLiA + weather + the synthetic adding task at lengths 500/2,000/5,000 — [Boyer et al., D-LinOSS](https://arxiv.org/html/2505.12171); code github.com/jaredbmit/damped-linoss (URL as given in the paper)
- LrcSSM (NeurIPS 2025 per Semantic Scholar venue): UEA-6 (described as short-horizon <1,500 vs long-horizon >1,500) + PPG-DaLiA, "the exact same seeds for the dataset splitting" as LinOSS — [Farsang & Grosu, LrcSSM](https://arxiv.org/html/2505.21717); code [MoniFarsang/LrcSSM](https://github.com/MoniFarsang/LrcSSM)
- SLiCE / Structured Linear CDEs (NeurIPS 2025 per Semantic Scholar): UEA-6 using "the same data splits" as Log-NCDE, plus formal-language and A5 state-tracking tasks — [Walker et al., SLiCE](https://arxiv.org/html/2505.17761); code [structured-linear-cdes](https://github.com/Benjamin-Walker/structured-linear-cdes)
- Rough Transformers (NeurIPS 2024), a long-context Transformer for continuous time: UEA long datasets (70/15/15 re-split, baselines taken from Walker et al.), BIDMC Heart Rate (4,000 steps), and synthetic sinusoids up to 250K steps — [Moreno-Pino et al., RFormer](https://arxiv.org/html/2405.20799)
- DEER (parallel nonlinear RNN training; arXiv 2309.12252): EigenWorms with a 5-layer GRU at full length 17,984, and sCIFAR with a multi-head GRU — [Lim et al., DEER](https://arxiv.org/html/2309.12252)
- Two 2026 preprints still use the protocol: RiccatiSSM (UEA-6 + PPG-DaLiA) — [arXiv 2609.35441](https://arxiv.org/html/2609.35441); "Reshape and Recur" (UEA + PPG-DaLiA "with the same predefined random seeds"). The latter lists the benchmark as "commonly used" by Walker 2024, Moreno-Pino 2024, Nzoyem 2025, Rusch & Rus 2025, Boyer 2025, Pourcel & Ernoult 2026 and Karuvally 2026 — [arXiv 2605.16048](https://arxiv.org/html/2605.16048)
- Semantic Scholar lists 43 citing papers for Log-NCDE. They include NeurIPS 2025 papers (time-reversal-symmetry-breaking RHEL, adaptive unitary SSMs, BioOSS, PD-SSM) and a **spiking** second-order SSM for wearables (SHaRe-SSM) — [Semantic Scholar citations API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2402.18512/citations?fields=title,year,venue&limit=200)
- S5 (ICLR 2023): raw Speech Commands 35-way at 16 kHz (16,000 steps) with zero-shot 8 kHz testing, irregular pendulum regression (50 images), sMNIST/psMNIST/sCIFAR, and LRA — [Smith et al., S5](https://arxiv.org/html/2208.04933)
- Liquid-S4 (ICLR 2023): LRA, BIDMC vital signs (HR/RR/SpO2, length 4,000), sCIFAR, raw Speech Commands full 35 labels and SC10 — [Hasani et al., Liquid-S4](https://arxiv.org/html/2209.12951)
- LRU (ICML 2023): sCIFAR + LRA only (no physiological signals) — [Orvieto et al., LRU](https://arxiv.org/html/2303.06349)
- A **spiking** paper already uses the same long-sequence subset. SHaRe-SSM (2025) takes "the longest benchmark datasets": EigenWorms (17,984) and PPG-DaLiA (49,920), with the LinOSS hyperparameter protocol — [SHaRe-SSM](https://arxiv.org/html/2510.14386)
- EEG long-sequence work with Mamba exists (e.g., EEGM2, tested at 10–100 s, up to 12,800 samples). I saw it only in a search-result summary and did not open or verify it — [EEGM2 arXiv 2502.17873](https://arxiv.org/html/2502.17873v2)

### Inferences
- If the goal is to be comparable with current non-spiking long-sequence work on continuous signals, the de-facto standard is UEA-6 + PPG-DaLiA under the Log-NCDE code and seeds. Speech Commands raw, BIDMC and sMNIST/psMNIST/sCIFAR are the second tier (S4-family legacy). LRA is mostly text and image tokens and is only context.
- A spiking comparator already exists on EigenWorms + PPG-DaLiA (SHaRe-SSM). A spiking T = N result on these two tasks would therefore have a direct spiking baseline as well as the non-spiking ones.

### Gaps
- I found no xLSTM results on UEA-6, PPG-DaLiA, BIDMC or raw Speech Commands. In the papers I opened, xLSTM/sLSTM/mLSTM appear only in SLiCE's formal-language table.
- I did not open the NeurIPS 2025 adaptive-unitary-SSM, BioOSS or PD-SSM papers, so I cannot say whether they report UEA-6 numbers.
- The Mamba (2023) raw-audio experiments (SC09 generation, YouTubeMix) were not opened. They are not included.

## Q2. Protocols and representative numbers per benchmark

### Takeaway
UEA-6 is small: 236–561 cases per dataset, 405–17,984 steps, 5 random 70/15/15 re-splits, accuracy. Three of its six datasets sit at or near chance (the guessing level), or near the majority-class rate, for every model. PPG-DaLiA is a dense sequence-to-sequence regression scored by MSE in normalised units. The reported numbers come in **two different versions**, because Log-NCDE re-ran all baselines after an optimiser bug (v2 → v3). Check which version a paper copies before comparing against it.

### Cited Findings — dataset facts

| Dataset | What the "time" axis is | Length | Channels | Cases (orig. train/test) | Classes | Source |
|---|---|---|---|---|---|---|
| EigenWorms | video frames of C. elegans posture projected on 6 "eigenworm" shapes; truncated to the shortest series | 17,984 | 6 | 259 (table: 128/131; text says 131 train/128 test — inconsistent within the source) | 5 (wild-type + 4 mutants) | [UEA archive paper [background]](https://arxiv.org/abs/1811.00075) |
| SelfRegulationSCP1 | EEG slow cortical potentials, 256 Hz × 3.5 s | 896 | 6 | 561 (268/293) | 2 | same |
| SelfRegulationSCP2 | EEG from an ALS patient, 256 Hz × 4.5 s | 1,152 | 7 | 380 (200/180) | 2 | same; source notes "it is not clear if there is any information contained in this dataset that is useful for the classification task" |
| EthanolConcentration | **light spectrum** (226–1,101.5 nm at 0.5 nm), *not time* | 1,751 | 3 | 524 (261/263) | 4 | same |
| Heartbeat | **spectrogram** of 5 s heart-sound audio (0.061 s windows, 70% overlap); each channel is a frequency band; 113 normal / 296 abnormal | 405 | 61 | 409 (204/205) | 2 | same |
| MotorImagery | ECoG 8×8 grid, 1,000 Hz × 3 s | 3,000 | 64 | 378 (278/100) | 2 | same |
| PPG-DaLiA (Log-NCDE protocol) | wrist BVP 64 Hz, ACC 32 Hz, EDA 4 Hz, TEMP 4 Hz, all repeated up to 64 Hz in the code | 49,920 per window | 6 | 15 subjects, ~150 min each | regression: 390 heart-rate values per window | [UCI page](https://archive.ics.uci.edu/dataset/495/ppg+dalia); [process_ppg.py](https://github.com/Benjamin-Walker/log-neural-cdes/blob/main/data_dir/process_ppg.py); [Log-NCDE v3](https://arxiv.org/html/2402.18512v3) |
| BIDMC32 HR / RR / SpO2 | PPG + ECG at 125 Hz (RFormer adds time as a 3rd channel) | 4,000 | 2 | HR: 5,550 / 2,399; RR: 5,471 / 2,399 (TSER archive split) | regression | [TSER archive paper [background]](https://arxiv.org/abs/2006.12672); [RFormer](https://arxiv.org/html/2405.20799); [Liquid-S4](https://arxiv.org/html/2209.12951) |
| Speech Commands (raw, 35-way) | 1 s audio at 16 kHz | 16,000 (8,000 for the zero-shot test) | 1 | 24,482 / 5,246 / 5,247 (train/val/test, S4 preprocessing) | 35 | [S5 App. G](https://arxiv.org/html/2208.04933) |
| sMNIST / psMNIST / sCIFAR | flattened pixels (psMNIST adds a fixed permutation) | 784 / 784 / 1,024 | 1 / 1 / 3 | standard MNIST/CIFAR-10 | 10 | [S5 App. G](https://arxiv.org/html/2208.04933) |

### Cited Findings — protocol details
- **UEA-6 protocol (Log-NCDE):** hyperparameters are chosen by grid search on validation accuracy over one fixed random split. Models are then compared by "average test set accuracy over five different random splits". Optimiser Adam, batch 32. Baseline blocks are built on the S5 code: norm → recurrent layer → GLU → dropout 0.1 → skip, with a linear encoder and decoder; LRU and S5 use batch norm, S6 and Mamba use layer norm — [Log-NCDE v3](https://arxiv.org/html/2402.18512v3)
- **EigenWorms duplicates:** "As of June 1st 2024 … 23 duplicated time series" were found and removed. Older EigenWorms numbers include them — [Log-NCDE v3](https://arxiv.org/html/2402.18512v3)
- **Grid (LinOSS/D-LinOSS):** lr ∈ {1e-5, 1e-4, 1e-3} × layers ∈ {2, 4, 6} × hidden ∈ {16, 64, 128} × state ∈ {16, 64, 256} × include-time ∈ {T, F} = 162 configurations per dataset, then 5 seeds — [LinOSS App.](https://arxiv.org/html/2410.03943); [D-LinOSS](https://arxiv.org/html/2505.12171)
- **PPG-DaLiA protocol:** hyperparameters are chosen on validation MSE. Final score = "average mean squared error over five different runs on the same fixed data split". Stacked recurrent models needed batch size 4 "due to memory constraints" — [Log-NCDE v3](https://arxiv.org/html/2402.18512v3). The code does several more things. It min–max scales every channel and the HR label to [−1, 1] per subject over the whole recording. Each subject's recording is cut into contiguous time blocks (one of 6 block orders, picked by `random.randint`). Input windows are 49,920 steps with stride 4,992. Label windows are 390 values with stride 39, so the target is a 390-step heart-rate sequence, not one number — [process_ppg.py](https://github.com/Benjamin-Walker/log-neural-cdes/blob/main/data_dir/process_ppg.py)
- **PPG-DaLiA label definition:** "the ECG-signal was segmented with a shifted window approach (window length: 8 s, window shift: 2 s). We defined ground truth heart rate as the mean instantaneous heart rate within each 8-s window" — [Reiss et al., Deep PPG, Sensors 2019 [background]](https://pmc.ncbi.nlm.nih.gov/articles/PMC6679242/)
- **Erratum / two number sets:** on 28 Oct 2024 the Log-NCDE authors "discovered a minor error in the code that affected the optimiser … all experiments have been re-run". The arXiv paper was updated to v3 — [log-neural-cdes README](https://github.com/Benjamin-Walker/log-neural-cdes). The changes are visible: EigenWorms v2 → v3 is LRU 85.0 → 87.8, S5 83.9 → 81.1, NCDE 62.2 → 75.0, NRDE 77.2 → 83.9, Log-NCDE 82.8 → 85.6, and the Log-NCDE average 64.4 → 64.3 — [v2](https://arxiv.org/html/2402.18512v2) vs [v3](https://arxiv.org/html/2402.18512v3). LinOSS, D-LinOSS and SLiCE use the v3 numbers. LrcSSM's table marks its baselines as "reported from [LinOSS]", but its NRDE/NCDE/Log-NCDE/LRU/S5 values match **v2**, e.g. NRDE Worms 77.2 and Log-NCDE 82.8 — [LrcSSM](https://arxiv.org/html/2505.21717)
- **Speech Commands:** results are the mean across three random seeds. The 8 kHz column is zero-shot testing on naively decimated audio. Bidirectional models are used for the LRA and speech tasks, causal models for sMNIST/psMNIST/pendulum — [S5](https://arxiv.org/html/2208.04933)
- **BIDMC (RFormer):** test RMSE over five seeds. HR is forecast "at the sample's conclusion" from 4,000-step, 125 Hz series — [RFormer](https://arxiv.org/html/2405.20799)

### Cited Findings — UEA-6 test accuracy (v3/LinOSS baseline set unless noted)

| Model (venue) | EigenWorms | SCP1 | SCP2 | Ethanol | Heartbeat | Motor | Avg | Source |
|---|---|---|---|---|---|---|---|---|
| LRU | 87.8±2.8 | 82.6±3.4 | 51.2±3.6 | 21.5±2.1 | 78.4±6.7 | 48.4±5.0 | 61.7 | [Log-NCDE v3](https://arxiv.org/html/2402.18512v3) |
| S5 | 81.1±3.7 | 89.9±4.6 | 50.5±2.6 | 24.1±4.3 | 77.7±5.5 | 47.7±5.5 | 61.8 | same |
| S6 | 85.0±16.1 | 82.8±2.7 | 49.9±9.5 | 26.4±6.4 | 76.5±8.3 | 51.3±4.7 | 62.0 | same |
| Mamba | 70.9±15.8 | 80.7±1.4 | 48.2±3.9 | 27.9±4.5 | 76.2±3.8 | 47.7±4.5 | 58.6 | same |
| NCDE | 75.0±3.9 | 79.8±5.6 | 53.0±2.8 | 29.9±6.5 | 73.9±2.6 | 49.5±2.8 | 60.2 | same |
| NRDE | 83.9±7.3 | 80.9±2.5 | 53.7±6.9 | 25.3±1.8 | 72.9±4.8 | 47.0±5.7 | 60.6 | same |
| Log-NCDE (ICML 2024) | 85.6±5.1 | 83.1±2.8 | 53.7±4.1 | 34.4±6.4 | 75.2±4.6 | 53.7±5.3 | 64.3 | same |
| LinOSS-IMEX (ICLR 2025) | 80.0±2.7 | 87.5±4.0 | 58.9±8.1 | 29.9±1.0 | 75.5±4.3 | 57.9±5.3 | 65.0 | [LinOSS](https://arxiv.org/html/2410.03943) |
| LinOSS-IM (ICLR 2025) | 95.0±4.4 | 87.8±2.6 | 58.2±6.9 | 29.9±0.6 | 75.8±3.7 | 60.0±7.5 | 67.8 | same |
| D-LinOSS (2025) | 93.9±3.2 | 88.9±3.0 | 58.6±2.3 | 29.9±0.6 | 75.8±4.9 | 61.1±2.0 | 68.0 | [D-LinOSS](https://arxiv.org/html/2505.12171) |
| BD-SLiCE (NeurIPS 2025) | 86±4 | 85±2 | 54±8 | 29±7 | 77±6 | 53±3 | 64.0 | [SLiCE Table 5](https://arxiv.org/html/2505.17761) |
| Transformer | OOM | 84.3±6.3 | 49.1±2.5 | 40.5±6.3 | 70.5±0.1 | 50.5±3.0 | 59.0 (5 sets) | [RFormer](https://arxiv.org/html/2405.20799) via [LrcSSM](https://arxiv.org/html/2505.21717) |
| RFormer (NeurIPS 2024) | 90.3±0.1 | 81.2±2.8 | 52.3±3.7 | 34.7±4.1 | 72.5±0.1 | 55.8±6.6 | 64.5 | same |
| LrcSSM (NeurIPS 2025) | 90.6±1.4 | 85.2±2.1 | 53.9±7.2 | 36.9±5.3 | 72.7±5.7 | 58.6±3.1 | 66.3 | [LrcSSM](https://arxiv.org/html/2505.21717) (compared against **v2** baselines) |
| GRU, 5 layers, full length (DEER; 3 seeds) | 88.0±4.4 | – | – | – | – | – | – | [DEER](https://arxiv.org/html/2309.12252) |
| LEM / UnICORNN (from older papers, quoted in DEER) | 92.3±1.8 / 90.3±3.0 | – | – | – | – | – | – | [DEER Table 1](https://arxiv.org/html/2309.12252) |
| SHaRe-SSM (**spiking**, 2025), IM / IMEX | 92.8±3.3 / 90.0±5.7 | – | – | – | – | – | – | [SHaRe-SSM](https://arxiv.org/html/2510.14386) |
| RHEL-Lin / RHEL-Nonlin (NeurIPS 2025, quoted in SHaRe-SSM) | 75.0±9.9 / 50.1±6.7 | – | – | – | – | – | – | [SHaRe-SSM Table 2](https://arxiv.org/html/2510.14386) (secondary) |

- RiccatiSSM (2026) says its EigenWorms runs ranged "from 77.8% to 91.7%" across seeds — [RiccatiSSM](https://arxiv.org/html/2609.35441)
- Majority-class accuracy on the **original fixed train/test split** (not the 70/15/15 re-splits): EigenWorms 0.42, EthanolConcentration 0.251, Heartbeat 0.722, MotorImagery 0.5, SCP1 0.502, SCP2 0.5. Classical classifiers on that split for EthanolConcentration: HIVE-COTE 0.791, STC (shapelets) 0.821 — [Ruiz et al., MTSC bake-off [background]](https://arxiv.org/abs/2007.13156)

### Cited Findings — PPG-DaLiA (test MSE × 10⁻², normalised labels, 5 runs on one split)
- LRU 12.17±0.49; S5 12.63±1.25; S6 12.88±2.05; Mamba 10.65±2.20; NCDE 13.54±0.69; NRDE 9.90±0.97; Log-NCDE 9.56±0.59 — [Log-NCDE v3](https://arxiv.org/html/2402.18512v3) (v2 values were different, e.g. LRU 15.64, NRDE 18.79, Log-NCDE 9.20 — [v2](https://arxiv.org/html/2402.18512v2))
- LinOSS-IMEX 7.5±0.46; LinOSS-IM 6.4±0.23 — [LinOSS](https://arxiv.org/html/2410.03943)
- D-LinOSS 6.16±0.73 — [D-LinOSS](https://arxiv.org/html/2505.12171); LrcSSM 10.89±0.96 — [LrcSSM](https://arxiv.org/html/2505.21717); RiccatiSSM 7.15±1.01 — [RiccatiSSM](https://arxiv.org/html/2609.35441)
- Spiking: SHaRe-SSM 11.8±0.9 (IM), 9.1±0.2 (IMEX); non-spiking RHEL-Lin 9.5±1.0, RHEL-Nonlin 8.4±0.5 (quoted in SHaRe-SSM) — [SHaRe-SSM](https://arxiv.org/html/2510.14386). Caution: SHaRe-SSM's table lists LinOSS IM = 7.5 and IMEX = 6.4 (and IMEX EigenWorms 80.0±4.4), whereas the LinOSS paper has IM 6.4, IMEX 7.5 (IMEX EigenWorms 80.0±2.7) — compare [LinOSS](https://arxiv.org/html/2410.03943)

### Cited Findings — BIDMC, Speech Commands, pixel sequences, LRA
- **BIDMC RMSE (HR / RR / SpO2):** Liquid-S4 0.303 / 0.158 / 0.066; S4-LegS 0.332 / 0.247 / 0.090; S4D-Lin 0.379 / 0.226 / 0.114; UnICORNN 1.39 / 1.06 / 0.869; NRDE 2.97 / 1.49 / 1.29; XGBoost 4.72 / 1.67 / 1.52; LSTM 10.7 / 2.28 / –; Transformer 12.2 / 2.61 / 3.02 — [Liquid-S4 Table 2](https://arxiv.org/html/2209.12951)
- **BIDMC HR (RFormer, 5 seeds):** RFormer 2.66±0.21; Transformer 8.24±2.24; NRDE 2.97±0.45 (reported) / 4.04±0.11 (rerun); NCDE 9.82±0.34; GRU 13.06±0.00; ODE-RNN 13.06±0.00 — [RFormer Table 1](https://arxiv.org/html/2405.20799). The TSER archive's own BIDMC32HR RMSE values for classical/deep regressors are ~9.4–15.3 — [TSER [background]](https://arxiv.org/abs/2006.12672). These scales do not match the S4-family 0.3 values, so the target normalisation or split must differ.
- **Speech Commands raw SC35, 16 kHz / 8 kHz zero-shot:** S5 96.52 / 94.53; Liquid-S4 96.78 / 90.00; S4D-Lin 96.25 / 91.58; S4-LegS 96.08 / 91.32; ConvNet (26.2M params) 95.51 / 7.26; ResNet-18 77.86 / 8.74; InceptionNet 61.24 / 5.18 — [S5 Table 8](https://arxiv.org/html/2208.04933)
- **SC10 raw, 16 kHz / 8 kHz:** Liquid-S4 98.51 / 95.9; S4-LegS 98.32 / 96.30; WaveGAN-D 96.25 / x; CKConv 71.66 / 65.96; Performer 30.77 / 30.68; NRDE 16.49 / 15.12; Transformer "x" (infeasible on a single GPU) — [Liquid-S4 Table 6](https://arxiv.org/html/2209.12951)
- **sMNIST / psMNIST / sCIFAR:** S4 99.63 / 98.70 / 91.80; S5 99.65 / 98.67 / 90.10; CCNN 99.72 / 98.84 / 93.08; LSTM 98.9 / 95.11 / 63.01; Transformer 98.9 / 97.9 / 62.2 — [S5 Table 10](https://arxiv.org/html/2208.04933). Liquid-S4 sCIFAR 92.02 — [Liquid-S4](https://arxiv.org/html/2209.12951); LRU sCIFAR 89.0±0.1 — [LRU](https://arxiv.org/html/2303.06349); DEER multi-head GRU sCIFAR 90.25 — [DEER](https://arxiv.org/html/2309.12252)
- **LRA (context only), sCIFAR / ListOps / Text / Retrieval / Pathfinder / PathX:** LRU 89.0 / 60.2 / 89.4 / 89.9 / 95.1 / 94.2; S4 (paper) 91.1 / 59.6 / 86.8 / 90.9 / 94.2 / 96.4; S5 (paper) 90.1 / 62.2 / 89.3 / 91.4 / 95.3 / 98.6 — [LRU Table 3](https://arxiv.org/html/2303.06349)
- **Synthetic delay task:** D-LinOSS reports the adding problem at lengths 500/2,000/5,000. LinOSS-IMEX "fails to outperform the baseline of random guessing (with MSE of 0.167) for any task" — [D-LinOSS](https://arxiv.org/html/2505.12171)

### Inferences
- UEA-6 averages are dominated by datasets where every model sits near chance or the majority rate: Ethanol (chance 25%, deep models 21–41%), SCP2 (50%, models 48–59%), Motor (50%, models 47–61%) and Heartbeat (majority ≈72%, models 70–78%). Differences of 1–3 points in the 6-dataset average are within the ±5–16 point per-dataset standard deviations. EigenWorms is the dataset that separates models most (70.9 to 95.0).
- When citing baselines, quote the v3 set (as LinOSS, D-LinOSS and SLiCE do) and state the duplicate-removed EigenWorms.

### Gaps
- EigenWorms frame rate (Hz) is not verified: timeseriesclassification.com blocked automated access with a bot check, and the UEA paper text I read gives no rate.
- I did not verify the S4/NRDE BIDMC protocol (split sizes and target normalisation), so the 0.3-scale RMSEs cannot be reconciled with the TSER-scale RMSEs.
- The S5-reported Speech Commands split (24,482 training examples) looks smaller than the full Speech Commands corpus. I did not verify the subset definition beyond S5's appendix.
- Only the arXiv versions were checked. The LinOSS/LrcSSM camera-ready tables may differ.

## Q3. Evidence that these tasks need long-range memory, and criticism that they do not

### Takeaway
Published support for "long memory is needed" on these continuous benchmarks is mostly indirect: the sequences are long, and models with short or unstable memory perform worse or run out of memory. I found **no truncation study** (accuracy vs. how much of the input is kept) for UEA-6 or PPG-DaLiA. The criticism is direct and concrete. LRA tasks are mostly local (small convolutions get close to state of the art, except Pathfinder). The PPG-DaLiA targets are by construction 8-s local heart rates. EigenWorms can be compressed 600× in time with signature features (summary statistics of path segments) without loss. Several UEA-6 datasets are not temporal (spectra, spectrograms) or carry little information.

### Cited Findings — supporting a long-range requirement (indirect)
- LinOSS frames UEA-6 as "learning long-range interactions" and PPG-DaLiA as "very long-range interactions" (sequences ~50k). LinOSS-IM is best on "the two datasets with the longest sequences, namely EigenWorms and MotorImagery" — [LinOSS](https://arxiv.org/html/2410.03943)
- LrcSSM splits UEA-6 into short (<1,500: Heartbeat, SCP1, SCP2) and "tasks that require learning long-range interactions" (>1,500: Ethanol, Motor, EigenWorms). The authors "suspect that these [short] datasets lack interesting input correlations" — [LrcSSM](https://arxiv.org/html/2505.21717)
- On BIDMC HR (4,000 steps), "GRU and ODE-RNN fail to capture the information in the signal". Their RMSE is 13.06 with zero spread across seeds (it looks like a constant prediction), while NRDE reaches 2.97 and RFormer 2.66 — [RFormer](https://arxiv.org/html/2405.20799)
- On EigenWorms, a vanilla Transformer runs out of GPU memory (OOM) at 17,984 steps — [RFormer Table 2](https://arxiv.org/html/2405.20799). A full-length 5-layer GRU trained with DEER reaches 88.0±4.4, against a folded ODE-RNN at 47.9 and NCDE (step 4) at 66.7 — [DEER](https://arxiv.org/html/2309.12252)
- Raw speech: weak CNN baselines on 16,000-step audio (InceptionNet 61.24, ResNet-18 77.86) are well below SSMs (~96), and Transformers are "infeasible" on SC10 at 16 kHz on a single GPU — [S5](https://arxiv.org/html/2208.04933); [Liquid-S4](https://arxiv.org/html/2209.12951)
- sCIFAR (1,024 steps): LSTM 63.01 vs S4 91.80 — [S5 Table 10](https://arxiv.org/html/2208.04933)
- LRA Pathfinder is the one LRA task that resists small receptive fields: convolution kernels of size 5–31 stay at 51–52% (chance level); kernel 61 gives 72.28%, vs MEGA 96.01% — [Miralles-González et al. 2025](https://arxiv.org/html/2501.14850v1)

### Cited Findings — criticism / evidence that long memory is not required
- **LRA is mostly local:** "with small kernels of size 61 we get close to state-of-the-art results in all tasks. Particularly concerning is the case of text, where kernels of size 5 suffice". Kernel 61 reaches 83.62% on Image (MEGA 90.44%), 89.04% on Retrieval (91.25%) and 85.41% on Text (90.43%). The authors argue LRA "tasks are mainly positional and local" and that its validity "as a long-range dependency modeling benchmark" is in question — [Miralles-González et al. 2025](https://arxiv.org/html/2501.14850v1)
- **The LRA ranking depends on pretraining:** vanilla Transformers "match the performance of S4 on Long Range Arena when properly pretrained" (self-pretraining on the task data). With that pretraining, S4's best PathX-256 result improves "by 20 absolute points", and the structured SSM parameterisations "become mostly redundant" — [Amos et al., ICLR 2024](https://arxiv.org/html/2310.02980)
- **PPG-DaLiA targets are local:** ground truth is the mean ECG heart rate in 8-s windows shifted by 2 s — [Reiss et al. 2019 [background]](https://pmc.ncbi.nlm.nih.gov/articles/PMC6679242/). In the Log-NCDE code, each 49,920-step window carries 390 such labels (one per 128 input samples at 64 Hz) — [process_ppg.py](https://github.com/Benjamin-Walker/log-neural-cdes/blob/main/data_dir/process_ppg.py). The window length therefore measures how well a model handles long input, not how far back it must remember.
- **EigenWorms can be compressed heavily:** "the Eigenworms dataset … experiences a 20% performance drop when employing univariate signatures, but is able to achieve state-of-the-art performance with a 600× compression rate in the temporal dimension when signatures are applied across all channels". The authors' reading: "the relationships between different channels … may hold more importance than the temporal information itself, which can often be redundant". The best signature setting was about 30 windows over 18k steps — [RFormer §5](https://arxiv.org/html/2405.20799)
- **EigenWorms is small and noisy:** DEER notes "as the EigenWorms dataset is very small, slight changes could change the results quite significantly" — [DEER](https://arxiv.org/html/2309.12252). Seed spreads reach ±16.1 (S6) and ±15.8 (Mamba) — [Log-NCDE v3](https://arxiv.org/html/2402.18512v3) — and 77.8–91.7% across RiccatiSSM runs — [RiccatiSSM](https://arxiv.org/html/2609.35441)
- **Several UEA-6 datasets are not temporal, or have little signal:** EthanolConcentration's axis is wavelength, and Heartbeat is a spectrogram with frequency bands as channels — [UEA archive paper [background]](https://arxiv.org/abs/1811.00075). On the original split, for Heartbeat, MotorImagery and SCP2 "few, if any, classifiers do better than predicting the majority class", and SCP2 probably has "no useful information in the attribute space" — [MTSC bake-off [background]](https://arxiv.org/abs/2007.13156). Shapelet/ensemble classifiers that ignore long-range order reach 0.79–0.82 on EthanolConcentration, against 21–41% for deep sequence models under the re-split protocol — [bake-off](https://arxiv.org/abs/2007.13156); [Log-NCDE v3](https://arxiv.org/html/2402.18512v3)
- **Speech Commands:** a large CNN (26.2M parameters) reaches 95.51% at 16 kHz. Its collapse at 8 kHz (7.26%) shows sensitivity to sampling rate, not a memory failure — [S5 Table 8](https://arxiv.org/html/2208.04933)

### Inferences
- For the T = N question, these benchmarks fall into three groups. (a) **Long but plausibly local:** PPG-DaLiA, Speech Commands, LRA Image/Text, Heartbeat. A patch-attention model with good local features should not need a long-lived state here; PPG-DaLiA's per-2 s labels are a streaming target where an 8-s context suffices in principle. (b) **Long, where the information may sit in whole-sequence statistics or channel interactions:** EigenWorms, possibly MotorImagery. A recurrent state could help as an accumulator, but signature summaries already do well. (c) **Long-range by construction:** psMNIST (the permutation destroys locality), LRA Pathfinder/PathX, and the adding problem (the delay length is set directly).
- The cleanest test of whether the T = N state carries information beyond attention over patches is probably not a stock benchmark. It is a controlled variant: truncate or shuffle patch order on EigenWorms/MotorImagery and measure the drop, or use an adding/delayed-recall task built on continuous signals. Published papers do not report these curves, so the user would have to run them.
- On PPG-DaLiA a T = N state has a natural streaming role: one label every 128 samples across 49,920 steps. But a model needs only about the last 8 s for each label, so a gain there would show streaming efficiency rather than long memory.

### Gaps
- I found no published accuracy-vs-truncation (or vs. window length) curves for any UEA-6 dataset, PPG-DaLiA, BIDMC or raw Speech Commands.
- I found no analysis that shuffles patch order on these datasets to test whether order matters.
- RFormer's 600× compression result is on EigenWorms only, and Figure 6 values were not extracted numerically.

## Q4. Which are small enough for many runs on 4× A6000, and which are heavy?

### Takeaway
UEA-6 is cheap. Per-1,000-step times on one RTX 4090 are seconds to ~2 minutes for SSMs. Estimated full training is ~4–26 minutes per run on average for LRU/S5/S6/Mamba, ~35 minutes for Log-NCDE and ~2 hours for NCDE/NRDE. LrcSSM reports "<1 up to 2–3 hours per data split" on an A100. PPG-DaLiA is memory-heavy: batch size 4 for SSMs, and LinOSS moved it to an A100. Raw Speech Commands has ~100× more input samples per epoch than EigenWorms, so it is the heavy one. A spiking model run as T = N over 17,984–49,920 steps without a parallel scan will cost much more than these SSM figures.

### Cited Findings
- Average over UEA-6, as GPU memory / estimated total run time on an RTX 4090: LRU 4,122 MB / 466 s; S5 2,815 MB / 245 s; S6 2,608 MB / 578 s; Mamba 4,450 MB / 1,554 s; NCDE 1,760 MB / 6,650 s; NRDE 2,676 MB / 7,284 s; Log-NCDE 2,000 MB / 2,128 s. Run time = time for 1,000 steps × average total steps — [Log-NCDE v3 Table 2](https://arxiv.org/html/2402.18512v3)
- Time per 1,000 training steps on an RTX 4090 (s), **EigenWorms**: LRU 94, S5 31, Mamba 122, S6 68, LinOSS-IMEX 37, LinOSS-IM 90, Log-NCDE 1,956, NCDE 24,595, NRDE 5,386. GPU memory: LinOSS-IM 10,654 MB, Mamba 13,486 MB, LRU 10,716 MB. **MotorImagery**: LinOSS-IM 11 s, LRU 51 s, Log-NCDE 730 s, NCDE 3,778 s, NRDE 7,616 s. SCP1/SCP2/Ethanol/Heartbeat take 3–255 s for every SSM — [LinOSS Table 5](https://arxiv.org/html/2410.03943)
- SLiCE re-timed on an H100, average per 1,000 steps: BD-SLiCE 68.1 s vs Log-NCDE 1,321.7 s (~20× faster), with LRU 26.9 s, S5 21.9 s and S6 20.1 s — [SLiCE Table 3](https://arxiv.org/html/2505.17761)
- LrcSSM: "We used A100 GPUs with 80 GB of memory. Training time ranged from less than 1 up to 2–3 hours per data split" — [LrcSSM App. B](https://arxiv.org/html/2505.21717)
- PPG-DaLiA: SSMs used batch size 4 "due to memory constraints" — [Log-NCDE v3](https://arxiv.org/html/2402.18512v3). LinOSS ran it "on Nvidia Tesla A100 GPUs due to higher memory demands" (other tasks ran on V100/RTX 4090) — [LinOSS](https://arxiv.org/html/2410.03943)
- DEER on EigenWorms: one epoch takes 18 s with DEER vs about 116 s sequentially, for a 5-layer GRU with 24 hidden units (V100-class hardware is mentioned in the benchmarking section) — [DEER](https://arxiv.org/html/2309.12252)
- Full protocol size: 162 configurations per dataset, then 5 seeds — [LinOSS](https://arxiv.org/html/2410.03943); [D-LinOSS](https://arxiv.org/html/2505.12171)
- LRU training speeds on LRA were measured on A100s (steps/s, Table 4) — [LRU](https://arxiv.org/html/2303.06349)

### Inferences
- Rough data volume per training epoch (my arithmetic from the dataset facts above): EigenWorms ≈ 0.7 × ~236 cases × 17,984 ≈ 3 M time steps. PPG-DaLiA ≈ 15 subjects × ~71 windows × 49,920 ≈ 50 M (assuming ~150 min at 64 Hz and a 70% training block). Raw SC35 ≈ 24,482 × 16,000 ≈ 390 M. MotorImagery ≈ 0.7 × 378 × 3,000 ≈ 0.8 M. SCP1/SCP2/Ethanol/Heartbeat are each under 0.7 M. sMNIST/psMNIST ≈ 60,000 × 784 ≈ 47 M.
- **Light (many seeds × variants feasible in hours on 4× A6000):** SCP1, SCP2, Heartbeat, EthanolConcentration, MotorImagery, and EigenWorms (small case count, but each sequence is 17,984 steps, so memory per sample is the limit, not epoch time). **Medium:** sMNIST/psMNIST/sCIFAR, BIDMC (about 5.5k × 4,000 steps). **Heavy:** PPG-DaLiA (49,920-step windows; memory-bound, small batches) and raw Speech Commands (≈ 390 M steps per epoch).
- A T = N spiking path is a nonlinear recurrence with reset, so the parallel scans that make LRU/S5/LinOSS fast do not directly apply. Its cost will be closer to the sequential GRU (≈ 116 s per EigenWorms epoch for a tiny model) or NCDE than to the SSM numbers. The CinC 117-token setup is three orders of magnitude shorter than EigenWorms at T = N, unless EigenWorms is patched too: 17,984 / 64 ≈ 281 patch tokens, which is close to the CinC token scale.
- The project memory notes that PopulationLIF TCN jobs slow down 7–11× when two share an A6000. If T = N spiking runs behave similarly, budget one job per GPU for the 18k–50k-step tasks.

### Gaps
- I found no verified wall-clock or GPU-memory figures for raw Speech Commands training in 2023–2026 papers. The "heavy" label rests on data volume only.
- I found no published runtime for spiking T = N models on EigenWorms/PPG. SHaRe-SSM reports an energy estimate (≈ 52.1× more efficient than LinOSS for its best IMEX model; average spike rate 0.42 per step per block on EigenWorms) but not wall-clock time — [SHaRe-SSM](https://arxiv.org/html/2510.14386)
- The PPG-DaLiA training-window count (~71 per subject) is my estimate from the code's windowing, not a reported figure.
