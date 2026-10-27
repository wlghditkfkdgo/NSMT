# Long-range temporal-memory benchmarks with continuous inputs in SNN papers, 2023–2026

Research date: 2026-10-27. Method: I downloaded the arXiv PDFs (or the Europe PMC open-access full text for the Nature Communications papers), converted them to text, and read the tables and protocol sections directly. Every number below comes from a source I opened; the link next to each number is that source. When a paper quotes another paper's number (for example, a non-spiking baseline in a comparison table), I say "as reported in" and link the paper where I read it. Anything I could not check is listed under Gaps.

Terms used throughout:
- **T = N**: one input element (a pixel, an audio sample, a time-series sample, or one patch or frame) per spiking time step, so the sequence axis is the spiking time axis.
- **Repeated presentation**: the whole input (or every token) is shown for T extra spiking steps (often T = 4), separate from the sequence length L.
- **Continuous input**: real-valued samples, as opposed to native spike or event bins (SHD, SSC, DVS, N-MNIST) or discrete tokens (text bytes, ListOps symbols, words).

Papers opened (venue checked in the PDF header or the publisher record unless marked otherwise):
- SpikingSSMs, Shen et al., AAAI 2025 — [arXiv 2408.14909](https://arxiv.org/abs/2408.14909), [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/34245)
- P-SpikeSSM, Bal & Sengupta, ICLR 2025 — [arXiv 2406.02923](https://arxiv.org/abs/2406.02923)
- PSN (Parallel Spiking Neurons), Fang et al., NeurIPS 2023 — [arXiv 2304.12760](https://arxiv.org/abs/2304.12760)
- TC-LIF, Zhang et al., AAAI 2024 — [arXiv 2308.13250](https://arxiv.org/abs/2308.13250)
- Balanced Resonate-and-Fire (BRF), Higuchi et al., ICML 2024 — [arXiv 2402.14603](https://arxiv.org/abs/2402.14603)
- SE-adLIF, Baronig et al., Nature Communications 2025 (DOI 10.1038/s41467-025-60878-z, printed in the arXiv text) — [arXiv 2408.07517](https://arxiv.org/abs/2408.07517)
- DH-SNN, Zheng et al., Nature Communications 15:277 (2024) — [PMC10766638](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/) (read through the [Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10766638/fullTextXML))
- Rhythm-SNN, Yan et al., Nature Communications 16 (2025) — [PMC12484961](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484961/) (read through the [Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12484961/fullTextXML))
- LMUFormer / Spiking LMUFormer, Liu et al., ICLR 2024 — [arXiv 2402.04882](https://arxiv.org/abs/2402.04882)
- PMSN, Chen et al., accepted by IEEE TNNLS (stated in the arXiv text) — [arXiv 2408.14917](https://arxiv.org/abs/2408.14917)
- SeqSNN (Spike-TCN / Spike-GRU / iSpikformer), Lv et al., ICML 2024 — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- CPG-PE, Lv et al., NeurIPS 2024 — [arXiv 2405.14362](https://arxiv.org/abs/2405.14362)
- TS-LIF, ICLR 2025 — [arXiv 2503.05108](https://arxiv.org/abs/2503.05108)
- Binary S4D, Stan & Rhodes (venue not verified) — [arXiv 2401.00955](https://arxiv.org/abs/2401.00955)
- SPikE-SSM, Zhong et al. (venue not verified) — [arXiv 2410.17268](https://arxiv.org/abs/2410.17268)
- DCLS-Delays, Hammouamri et al. (ICLR 2024 per my background knowledge; not found in the arXiv text) — [arXiv 2306.17670](https://arxiv.org/abs/2306.17670)
- Non-spiking protocol sources: S4D [arXiv 2206.11893](https://arxiv.org/abs/2206.11893); Log-NCDE, ICML 2024 [arXiv 2402.18512](https://arxiv.org/abs/2402.18512)
- Lower-tier or recent items, noted briefly: SHaRe-SSM (NeurIPS 2025 workshop) [arXiv 2510.14386](https://arxiv.org/abs/2510.14386); SpikySpace [arXiv 2601.02411](https://arxiv.org/abs/2601.02411); SSM-based delays [arXiv 2512.01906](https://arxiv.org/abs/2512.01906); Phase SSM (ICONS 2026) [arXiv 2608.07754](https://arxiv.org/abs/2608.07754); closed-loop forecasting with SSM SNNs [arXiv 2610.28495](https://arxiv.org/abs/2610.28495)

## Which continuous-input sequence benchmarks appear in SNN papers 2023–2026 (top venues) to claim temporal memory?

### Takeaway
The continuous-input benchmarks that 2023–2026 top-venue SNN papers actually use to claim long memory are:
- sequential MNIST and permuted sequential MNIST (784 pixel steps), by far the most common;
- sequential CIFAR, either column by column (32 steps) or pixel by pixel (1,024 steps, which is also the LRA "Image" task);
- the continuous-pixel LRA tasks Pathfinder (1,024 steps) and Path-X (16,384 steps);
- raw 16 kHz Speech Commands (16,000 steps).

Less common:
- the QT-database ECG wave-labelling task (spike-encoded by level crossing, so not raw);
- DEAP EEG (one paper);
- Google Speech Commands as spectrogram frames;
- time-series forecasting (traffic, solar, electricity);
- since 2025, the UEA very-long time-series classification set (EigenWorms and others) and PPG-DaLiA regression (49,920 steps), introduced to SNNs by PMSN.

I found no 2023–2026 top-venue SNN paper using the adding or copy problems. The memory toy tasks those papers do use (delayed spiking XOR, STORE-RECALL) take spike inputs.

### Cited Findings
- **Spiking state-space models (spiking SSMs).** SpikingSSMs (AAAI 2025) evaluates on sMNIST, psMNIST, all six LRA tasks including Path-X, WikiText-103, DVS128-Gesture and Speech Commands (35 classes) — [SpikingSSMs](https://arxiv.org/abs/2408.14909)
- P-SpikeSSM (ICLR 2025) evaluates on psMNIST, SC10 raw signals of length 16,000, SC35, and five LRA tasks (no Path-X) — [P-SpikeSSM](https://arxiv.org/abs/2406.02923)
- SPikE-SSM evaluates on sMNIST, psMNIST, sCIFAR10, six LRA tasks and WikiText-103 — [SPikE-SSM](https://arxiv.org/abs/2410.17268)
- Binary S4D (Stan & Rhodes) evaluates on sMNIST, sCIFAR10 (LRA Image) and all LRA tasks including Path-X — [Stan & Rhodes](https://arxiv.org/abs/2401.00955)
- **Neuron-model papers.** TC-LIF (AAAI 2024): S-MNIST, PS-MNIST, GSC, SHD, SSC — [TC-LIF](https://arxiv.org/abs/2308.13250)
- BRF (ICML 2024): S-MNIST, PS-MNIST, QTDB ECG, SHD — [BRF](https://arxiv.org/abs/2402.14603)
- SE-adLIF (Nat. Commun. 2025): SHD, SSC, QTDB ECG; a spring-mass oscillator prediction and generation task; raw-audio compression on LibriTTS at 24 kHz — [SE-adLIF](https://arxiv.org/abs/2408.07517)
- DH-SNN (Nat. Commun. 2024): delayed spiking XOR, multi-timescale spiking XOR, SHD, SSC, S-MNIST, PS-MNIST, GSC (12 classes), TIMIT, DEAP EEG, robot place recognition — [DH-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/)
- Rhythm-SNN (Nat. Commun. 2025): S-MNIST, PS-MNIST, SHD, ECG, GSC, VoxCeleb1, PTB, DVS-Gesture, STORE-RECALL working memory, Intel N-DNS speech denoising — [Rhythm-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484961/)
- PSN (NeurIPS 2023) tests long-term dependencies with sequential CIFAR10 and CIFAR100 fed column by column — [PSN](https://arxiv.org/abs/2304.12760)
- PMSN (TNNLS) covers:
  - S-MNIST and PS-MNIST (784 steps) and SHD (250 steps);
  - sequential CIFAR-10 at T = 32 (column) and T = 1,024 (pixel);
  - six UEA multivariate classification sets (sequence lengths 405 to 17,984);
  - PPG-DaLiA regression (length 49,920);
  - [PMSN](https://arxiv.org/abs/2408.14917)
- LMUFormer (ICLR 2024): psMNIST, Speech Commands V2 (35 classes; 128 spectrogram frames), LRA (five tasks) — [LMUFormer](https://arxiv.org/abs/2402.04882)
- **Time-series forecasting.**
  - SeqSNN (ICML 2024): Metr-la, Pems-bay, Solar, Electricity at horizons 6/24/48/96 — [SeqSNN](https://arxiv.org/abs/2402.01533)
  - CPG-PE (NeurIPS 2024): the same four sets — [CPG-PE](https://arxiv.org/abs/2405.14362)
  - TS-LIF (ICLR 2025): the same four sets, plus a delayed spiking XOR test — [TS-LIF](https://arxiv.org/abs/2503.05108)
- **Delays.** DCLS-Delays evaluates SHD, SSC, and non-spiking GSC turned into Mel spectrograms "with 140 frequency bins and approximately 100 timesteps". These are pre-binned frames, not raw samples — [DCLS](https://arxiv.org/abs/2306.17670)
- **Lower-tier or recent spiking SSM work.** SHaRe-SSM ("A Second-Order SpikingSSM for Wearables", NeurIPS 2025 workshop) uses UCI-HAR, SHAR, EigenWorms (17,984) and PPG-DaLiA (49,920), following the LinOSS protocol — [SHaRe-SSM](https://arxiv.org/abs/2510.14386)
- **Excluded native time-bin data (one line, as requested).** SHD, SSC and DVS-Gesture appear in TC-LIF, BRF, SE-adLIF, DH-SNN, Rhythm-SNN, PMSN and DCLS. They are excluded here because T = N is their standard input format — [PMSN](https://arxiv.org/abs/2408.14917)

### Inferences
- The neuron-model line (TC-LIF, DH-SNN, BRF, SE-adLIF, Rhythm-SNN, PMSN) relies on S-MNIST and PS-MNIST as its main continuous memory test. The SSM line (Binary S4D, SpikingSSMs, P-SpikeSSM, SPikE-SSM) relies on LRA and raw Speech Commands.
- Only a few continuous-input tasks are long and "real-signal" in nature: raw SC10/SC35 at 16k steps, UEA EigenWorms at 17,984 steps, PPG-DaLiA at 49,920 steps, and raw 24 kHz audio compression. Of these, only PMSN (a TNNLS journal paper) and a workshop paper use the UEA and PPG tasks.
- Three of the six LRA tasks (ListOps, Text, Retrieval) and WikiText-103 and PTB have discrete-token inputs. They do not meet the "continuous input" criterion even though SNN papers use them for long-range claims. Only LRA Image, Pathfinder and Path-X have continuous (grayscale pixel) inputs.

### Gaps
- I did not open every 2023–2026 SNN paper. Candidates I did not check: GLIF, CLIF, LM-H, PSU, TSkip, S5-RF, SpikeGPT, "Spiking LMU" variants, Yin et al. 2021 (Nature Machine Intelligence; background only), and LSNN 2018 (background only).
- Adding and copy tasks may appear in papers I did not open. I found none in the papers above.
- I could not verify the venues of Binary S4D (Stan & Rhodes) or SPikE-SSM. Semantic Scholar returned empty records, and neither PDF names a venue.
- DCLS-Delays as ICLR 2024 is from background knowledge; the arXiv text I extracted did not show the header.

## Protocol details per benchmark: length, encoding, split, metric, SNN results, best non-spiking reference, and memory evidence

### Takeaway
The table below is the dense reference. The strongest continuous-input settings for separating memory mechanisms are:
- psMNIST (plain LIF collapses to chance in a feed-forward net);
- pixel-level sCIFAR at 1,024 steps (in one paper, LIF falls from 81.5% column-wise to 45.1% pixel-wise);
- raw Speech Commands at 16,000 steps (non-spiking Transformer runs out of memory or fails, Performer reaches 30.8%);
- Path-X at 16,384 steps (Transformer stays at chance, Binary S4D 61.2%, SpikingSSM 94.8%);
- PPG-DaLiA at 49,920 steps and EigenWorms at 17,984 steps.

### Cited Findings

**Dense table (continuous-input tasks; "SNN results" are test accuracy in % unless stated)**

| Benchmark | Steps N | What one spiking step receives | Split / metric | Representative SNN results (venue) | Non-spiking reference | Memory evidence in the paper | Source |
|---|---|---|---|---|---|---|---|
| S-MNIST (pixel by pixel) | 784 | One real-valued gray pixel per step, fed into the first spiking layer, which acts as the encoder | 60k train / 10k test (TC-LIF); 54k / 6k / 10k (BRF); accuracy | TC-LIF recurrent 99.20 (AAAI24); DH-SRNN 98.9 (NC24); BRF 99.0±0.1, BHRF 99.1±0.1 (ICML24); Rhythm-DH-SRNN 99.00 (NC25); SpikingSSM 99.6 (AAAI25); PMSN 99.53 (TNNLS); SPikE-SSM-Full 99.53 | S4 99.63 (as reported in SpikingSSMs); ANN-S4D 99.50 (SPikE-SSM); LSTM 98.20 (as reported in TC-LIF) | Reproduced LIF: 72.06 feed-forward / 89.28 recurrent (larger nets, TC-LIF); FFSNN 59.24 vs Rhythm-FFSNN 96.43 (Rhythm-SNN) | [TC-LIF](https://arxiv.org/abs/2308.13250), [DH-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/), [BRF](https://arxiv.org/abs/2402.14603), [Rhythm-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484961/), [SpikingSSMs](https://arxiv.org/abs/2408.14909), [PMSN](https://arxiv.org/abs/2408.14917), [SPikE-SSM](https://arxiv.org/abs/2410.17268) |
| PS-MNIST (fixed random pixel permutation) | 784 | One pixel per step, permuted order | Same as S-MNIST; LMUFormer uses 50k / 10k / 10k; accuracy | P-SpikeSSM 98.4 (ICLR25); SpikingSSM 98.4; Spiking LMUFormer 97.92 (ICLR24); PMSN 97.78; PSN 97.76 (reproduced in PMSN); Rhythm-DH-SRNN 96.73; TC-LIF 95.36; BHRF 95.2, BRF 95.0±0.2; DH-SRNN 94.52; Spike-driven Transformer 96.21 (90k params, as reported in PMSN) | S4 98.70 (SpikingSSMs), 98.7 (P-SpikeSSM); LMUFormer (non-spiking) 98.55. LSTM is 88.00 in TC-LIF's table but 95.1 in P-SpikeSSM's: the papers disagree | Reproduced LIF feed-forward 11.30 / 10.00 (chance) vs TC-LIF feed-forward 80.89 / 83.98; vanilla RF 9.9±0.8 (BRF); FFSNN 42.96 vs Rhythm-FFSNN 95.01 | [TC-LIF](https://arxiv.org/abs/2308.13250), [BRF](https://arxiv.org/abs/2402.14603), [P-SpikeSSM](https://arxiv.org/abs/2406.02923), [LMUFormer](https://arxiv.org/abs/2402.04882), [PMSN](https://arxiv.org/abs/2408.14917), [Rhythm-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484961/) |
| Sequential CIFAR-10 / 100, column by column | 32 | One image column (32 pixels × 3 channels) per step, like a patch token | Standard CIFAR split; accuracy | CIFAR-10: PSN 88.45, sliding PSN 86.70, masked PSN 85.81, GLIF 83.66, PLIF 83.49, LIF 81.50, LIF without reset 79.50 (NeurIPS23); PMSN 90.97 (TNNLS). CIFAR-100: PSN 62.21, LIF 55.45; PMSN 66.08 | none reported | PSN calls it "widely used to evaluate the learning ability ... for long-term dependencies"; accuracy-vs-order curve (Fig. 5) | [PSN](https://arxiv.org/abs/2304.12760), [PMSN](https://arxiv.org/abs/2408.14917) |
| Sequential CIFAR-10, pixel by pixel (= LRA "Image": 1,024 gray-scale pixels) | 1,024 | One pixel per step | LRA split (Tay et al.); accuracy | PMSN 82.14 vs SPSN 70.23, masked PSN 57.83, PSN 55.24, LIF 45.07, LIF without reset 43.30 (TNNLS). LRA Image: SpikingSSM-VT 88.21, SPikE-SSM 87.23, P-SpikeSSM 82.40, Binary S4D 82.00, Spiking LMUFormer 55.65 | S4D-Lin 87.93; Transformer 42.44 (as reported in SpikingSSMs); ANN-S4D 87.11 (SPikE-SSM) | Same network, T = 32 → T = 1,024: LIF 81.50 → 45.07, PSN 88.45 → 55.24, PMSN 90.97 → 82.14 (parameter counts differ between the two settings) | [PMSN](https://arxiv.org/abs/2408.14917), [SpikingSSMs](https://arxiv.org/abs/2408.14909), [SPikE-SSM](https://arxiv.org/abs/2410.17268), [P-SpikeSSM](https://arxiv.org/abs/2406.02923), [Stan & Rhodes](https://arxiv.org/abs/2401.00955) |
| LRA Pathfinder | 1,024 | One pixel per step (32×32 image) | LRA split; accuracy | SpikingSSM-VF 93.68; SPikE-SSM 92.04; P-SpikeSSM 84.80; Binary S4D 82.60; Spiking LMUFormer 72.68 | S4D-Lin 93.96; Transformer 71.40 | Binary S4D: binary spikes cost at most 6 points on Image/ListOps/Retrieval/Text but degrade Pathfinder and Path-X more | [SpikingSSMs](https://arxiv.org/abs/2408.14909), [SPikE-SSM](https://arxiv.org/abs/2410.17268), [P-SpikeSSM](https://arxiv.org/abs/2406.02923), [Stan & Rhodes](https://arxiv.org/abs/2401.00955) |
| LRA Path-X | 16,384 (128×128) | One pixel per step | LRA split; accuracy (50% = chance) | SpikingSSM-VT 94.82 / VF 94.80; SPikE-SSM 94.37; Binary S4D 61.20 | S4D-Inv 92.80 (substituted because S4D-Lin fails); per Stan & Rhodes, "the Transformer fails to converge beyond random accuracy (50%)" and S4 reaches 88.1 | Chance-level Transformer vs SSMs; Binary S4D is 30 points below its own S4D baseline | [SpikingSSMs](https://arxiv.org/abs/2408.14909), [SPikE-SSM](https://arxiv.org/abs/2410.17268), [Stan & Rhodes](https://arxiv.org/abs/2401.00955) |
| Raw Speech Commands SC10 | 16,000 | One raw 16 kHz audio sample per step ("raw unprocessed signals of length 16000") | Splits as in Kidger et al. 2020 / Gu et al.; accuracy | P-SpikeSSM 95.6 (ICLR25) | S4 98.3; CKConv 71.7; Performer 30.8; NRDE 16.5; Transformer "×" (fails or not runnable) — all as reported in P-SpikeSSM | Long-sequence baselines fail or score low at 16k steps | [P-SpikeSSM](https://arxiv.org/abs/2406.02923) |
| Raw Speech Commands SC35 | 16,000 | One raw sample per step (S4D protocol: "1-second raw audio waveform comprising 16000 samples") | S4D protocol; accuracy | SpikingSSM 96.09 (305K params, AAAI25); P-SpikeSSM 96.23 (input form not stated in the passage I read) | S4D-Lin 96.25 (306K) — the same value as in S4D's own Table 4 | none specific | [SpikingSSMs](https://arxiv.org/abs/2408.14909), [S4D](https://arxiv.org/abs/2206.11893), [P-SpikeSSM](https://arxiv.org/abs/2406.02923) |
| GSC V2, 35 classes, spectrogram frames | 128 frames | One spectrogram frame per step after a 1-D conv patch embedding (kernel 3) | Official 84,843 / 9,981 / 11,005; accuracy | Spiking LMUFormer 96.12; Spike-driven ViT* 94.85; RadLIF 94.51; **Spikformer\* 93.38** (spiking self-attention swapped in for the LMU) | LMUFormer with states 96.92; AST 98.11 | Accuracy vs number of frames: 99% of final accuracy (95.17%) after 87/128 frames, which already exceeds Spikformer* using all frames | [LMUFormer](https://arxiv.org/abs/2402.04882) |
| GSC V1, 12 classes, log-Mel frames | 101 frames | 40 log-Mel × 3 derivative orders = 120 channels per 10 ms frame | SNN-standard 12-class setup; accuracy | DH-SFNN 94.05, DH-SRNN 93.86 (NC24); Rhythm-DH-SFNN 94.47 (NC25); TC-LIF recurrent 94.84 (AAAI24; TC-LIF describes GSC as v2 with 35 commands — see Gaps) | none in these papers | none (frames are pre-binned features) | [DH-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/), [Rhythm-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484961/), [TC-LIF](https://arxiv.org/abs/2308.13250) |
| QTDB ECG wave labelling (6 classes per time step: P, PQ, QR, RS, ST, TP) | Segments of 1,300 ms (one step per ms) | **Not raw:** two leads, each level-crossing encoded with threshold L = 0.3 into up and down spike trains (4 inputs) | 557 / 61 / 141 segments; per-time-step accuracy | SE-adLIF 2 layers 88.18±0.36, 1 layer 86.88±0.40 (NC25); BHRF 87.0±0.4, BRF 85.8±0.7, ALIF 85.9 (ICML24); Rhythm-DH-SRNN 87.43 (NC25) | none (ALIF from Yin et al. 2021 is the usual reference) | LIF 1 layer 77.8±1.85 (SE-adLIF table); FFSNN 55.99 (Rhythm-SNN) | [BRF](https://arxiv.org/abs/2402.14603), [SE-adLIF](https://arxiv.org/abs/2408.07517), [Rhythm-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484961/) |
| DEAP EEG emotion (3-class or 2-class valence and arousal) | 384 (3 s × 128 Hz) | Real-valued 32-channel EEG sample per step; the first layer acts as the encoder | 25,600 segments; 90% / 10% random split; accuracy | Accuracy only in Fig. 7 and Supplementary Table S6 (not opened) | MLP, CNN, spiking CNN in Supplementary Table S6 | More dendritic branches → higher accuracy (Fig. 7) | [DH-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/) |
| UEA-MTSCA 6 sets (EigenWorms 17,984; SCP1 896; SCP2 1,152; Ethanol 1,751; Heartbeat 405; Motor 3,000) | 405–17,984 | One multichannel sample per step | Log-NCDE protocol: original train and test merged and re-split 70:15:15, mean over 5 random splits; accuracy | PMSN: EigenWorms 95.0, SCP1 88.7, SCP2 58.9, Ethanol 34.9, Heartbeat 78.7, Motor 56.1, average 68.7 (TNNLS) | LinOSS-IM average 67.8 (EigenWorms 95.0); Log-NCDE 64.3; S5 61.8; Mamba 58.6 (as reported in PMSN) | Chosen because "datasets with many observations have previously proved challenging" (Log-NCDE) | [PMSN](https://arxiv.org/abs/2408.14917), [Log-NCDE](https://arxiv.org/abs/2402.18512) |
| PPG-DaLiA heart-rate regression | 49,920 | One 6-channel wrist-sensor sample per step (128 Hz) | 70:15:15 per person; sliding window 49,920, stride 4,992; 5 runs on one fixed split; MSE ×10⁻² | PMSN 5.17±0.53 (TNNLS) | LinOSS-IM 6.40±0.23; LinOSS-IMEX 7.50; Log-NCDE 9.56; LRU 12.17; S5 12.63; Mamba 10.65 (as reported in PMSN) | Longest continuous task found in any SNN paper | [PMSN](https://arxiv.org/abs/2408.14917), [Log-NCDE](https://arxiv.org/abs/2402.18512) |
| Raw-audio compression (LibriTTS train-clean-100, 24 kHz) | Full utterance at 24 kHz | Raw PCM sample per step (the paper says "rescaled each sample x individually"; whether "sample" means a clip or a PCM value was not checked) | LibriTTS clean test set; SI-SNR and ViSQOL | SE-adLIF SI-SNR 8.39±0.05, ViSQOL 3.91; EF-adLIF 7.80 / 3.82; LIF 6.90 / 3.46 (only 2 of 4 LIF runs converged) | Neural codec RVQ 4.14 / 4.38 at 6 kbps; OPUS 1.35 / 3.69; EVS 5.51 / 4.11 | Not framed as a memory test | [SE-adLIF](https://arxiv.org/abs/2408.07517) |
| Spring-mass system prediction then free-running generation | 500 (1 ms steps) | Real-valued displacements x[k] ∈ R⁴ per step | 5 random systems; MSE (figure only) | SE-adLIF beats LIF and LSTM at matched size (Fig. 4e) | LSTM | Only displacements are observed, so the network "must ... keep track of the longer-time dynamics" to infer velocity | [SE-adLIF](https://arxiv.org/abs/2408.07517) |
| Forecasting: Metr-la, Pems-bay (lookback 12), Solar, Electricity (lookback 168) | 12 or 168 | Spike-TCN / Spike-GRU: each series step is split into Ts = 4 spiking steps (so N × 4 steps). iSpikformer / Spikformer: variate tokens with separate T = 4 spiking steps | Chronological split 0.7/0.2/0.1 (traffic), 0.6/0.2/0.2 (Solar, Electricity); RSE and R²; horizons 6/24/48/96; 3 seeds | SeqSNN (ICML24); CPG-PE (NeurIPS24); TS-LIF (ICLR25) | TCN, GRU, Autoformer, iTransformer | TS-LIF: longer training windows (24 → 96 → 168) improve R² and RSE | [SeqSNN](https://arxiv.org/abs/2402.01533), [CPG-PE](https://arxiv.org/abs/2405.14362), [TS-LIF](https://arxiv.org/abs/2503.05108) |

**Spike-input memory probes (not continuous, but the explicit memory-vs-delay tests in this literature)**
- **Delayed spiking XOR.** Two Poisson spike patterns (firing probability 0.6 or 0.2, 10 ms each, 1 ms steps) are separated by a noisy delay of variable length. The target is the XOR of the two firing-rate levels. In DH-SNN, one-branch DH-SFNNs "significantly outperform vanilla SFNNs ... showing longer-term memory"; the results are in a figure only — [DH-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/)
- TS-LIF Table 11 (two-layer MLP with 20 hidden units) gives accuracy vs delay. ReLU stays at 0.5 throughout — [TS-LIF](https://arxiv.org/abs/2503.05108)

  | Delay (steps) | LIF | TS-LIF |
  |---|---|---|
  | 10 | 0.748 | 0.994 |
  | 20 | 0.504 | 0.977 |
  | 30 | 0.504 | 0.791 |
  | 40 | 0.500 | 0.585 |
  | 50 | 0.500 | 0.585 |

- **STORE-RECALL.** Binary values are Poisson-encoded (50 Hz, 100 ms windows). The network stores the value at the STORE command and outputs it at RECALL. 3-layer SRNN with 20 neurons per layer; ALIF and DEXAT neurons, each with and without rhythmic modulation. Rhythm variants have lower recall error (figure only) — [Rhythm-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484961/)

**Discrete-token long-range tasks (for completeness; do not meet the continuous criterion)**
- LRA ListOps (up to 2K), Text (4K bytes), Retrieval (4K): SpikingSSM-VT 60.23 / 80.41 / 88.77; P-SpikeSSM 58.20 / 81.20 / 88.53. References: S4D-Lin 60.52 / 86.97 / 90.96, Transformer 36.37 / 64.27 / 57.46 — [SpikingSSMs](https://arxiv.org/abs/2408.14909), [P-SpikeSSM](https://arxiv.org/abs/2406.02923)
- WikiText-103 perplexity: SpikingSSM 33.94 (75M), SpikeGPT 39.75 (213M), S4 20.95, Transformer 20.51 — [SpikingSSMs](https://arxiv.org/abs/2408.14909)
- PTB perplexity at sequence length 200: Rhythm-PLIF 114.73 vs PLIF 128.69 — [Rhythm-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484961/)

### Inferences
- The S-MNIST and PS-MNIST numbers are saturated: spiking models reach 98.4–99.6, close to S4's 98.7–99.6. psMNIST still discriminates weak memory clearly, though: plain LIF or vanilla RF at chance in feed-forward nets, roughly 72–80% recurrent.
- PMSN's T = 32 vs T = 1,024 comparison on the same image data is the cleanest "same content, longer sequence" design in this literature. It is the closest analogue to comparing patch tokens with raw-sample tokens for CinC.
- Spiking self-attention baselines are rare on these tasks. Where they appear, they trail the recurrent or SSM models:
  - Spikformer* 93.38 vs Spiking LMUFormer 96.12 on GSC-35 (LMUFormer);
  - Spike-driven Transformer 96.21 vs PMSN 97.78 on PS-MNIST, as reported in PMSN.
  - The PS-MNIST gap is small, so PS-MNIST alone may not separate a T = N memory path from spiking self-attention.
- The QTDB "ECG" task commonly used by SNN papers is spike-encoded by level crossing and labelled per time step. It is not a raw-ECG T = N task like CinC.

### Gaps
- DH-SNN delayed-XOR accuracies and DEAP accuracies are only in figures or the supplement; I did not extract numbers.
- Rhythm-SNN STORE-RECALL errors are only in a figure.
- The SE-adLIF spring-mass MSEs are only in a figure.
- The LinOSS, Log-NCDE, S5 and Mamba numbers in the UEA and PPG rows are as reported in PMSN. I did not cross-check them against the LinOSS paper.
- I did not transcribe the SeqSNN / CPG-PE / TS-LIF forecasting numbers: the table layout came out garbled after text extraction, and I could not attach rows to model names reliably.
- TC-LIF states it uses GSC v2 with 35 commands, but compares against the 92.10 SRNN+ALIF result that DH-SNN lists under the 12-class v1 setup. TC-LIF's text does not give its preprocessing.
- The PSN accuracies on psMNIST appear only as PMSN's reproductions; the PSN paper itself reports sequential CIFAR.
- I could not find how the "Spike-driven Transformer" and "QKFormer" psMNIST/sMNIST numbers quoted in PMSN were configured (tokens, T).

## Which benchmarks feed one raw sample per spiking time step (the T = N alignment)?

### Takeaway
T = N with one raw continuous sample per step is the default in this literature:
- sMNIST and psMNIST (pixel per step, real-valued input to the first spiking layer);
- pixel-level sCIFAR and LRA Image, Pathfinder and Path-X;
- raw SC10 and SC35 (one 16 kHz sample per step);
- UEA and PPG-DaLiA (one sensor sample per step);
- DEAP EEG in DH-SNN;
- SE-adLIF's raw-audio and spring-mass tasks.

T = N at the token level, comparable to our patch tokens, appears in PSN and PMSN (one image column per step, T = 32), in LMUFormer (one spectrogram frame per step) and in the GSC log-Mel setups (one 10 ms frame per step).

Spiking transformers for time series (iSpikformer, Spikformer with CPG-PE) do not use T = N: they keep a separate spiking-step axis T (= 4) next to the sequence axis L. P-SpikeSSM explicitly contrasts its one-step-per-element design with spiking transformers that "require additional time steps".

### Cited Findings
- TC-LIF: "each image is converted into a vector of 784 time steps, with each pixel representing one input value at a certain time step" — [TC-LIF](https://arxiv.org/abs/2308.13250)
- DH-SNN: in S-MNIST, "each time a pixel comes to the model". For GSC, S-MNIST, PS-MNIST and DEAP, "the real-value inputs are directly fed into DH-SNNs"; the first layer "acts as an encoding layer that converts non-spiking inputs to spiking outputs" — [DH-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/)
- DH-SNN DEAP: 128 Hz EEG, 60 s per trial cut into 20 segments of 3 s, giving data of shape "25600 × 32 × 384 (#samples × #channels × temporal length)" — [DH-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/)
- SpikingSSMs: the SSM output y_t at sequence step t "is treated as the input current" of an LIF neuron (u′_t = τu_{t−1} + y_t; s_t = H(u′_t − v_th)). Spiking time and sequence index are therefore the same — [SpikingSSMs](https://arxiv.org/abs/2408.14909)
- P-SpikeSSM: spiking transformers "process input sequences in parallel but require additional time steps to simulate network dynamics. In contrast, our proposed model eliminates this overhead". Input sequences are formatted as R^{L×1}; SC10 uses "raw unprocessed signals of length 16000" — [P-SpikeSSM](https://arxiv.org/abs/2406.02923)
- PSN: "the image will be sent to the SNN column by column" — 32 steps, one column per step — [PSN](https://arxiv.org/abs/2304.12760)
- PMSN: sequential CIFAR "column-by-column scanning ... (T = 32)" vs "pixel-by-pixel scanning (T = 1024)" — [PMSN](https://arxiv.org/abs/2408.14917)
- LMUFormer: SC accuracy is evaluated while "gradually increasing the sequence length from 0 to its full length of 128 samples" — 128 spectrogram frames, one per recurrent step — [LMUFormer](https://arxiv.org/abs/2402.04882)
- SeqSNN "temporal alignment": "we divide a time step ∆T of the time series into Ts segments ... ∆T = Ts∆t". The time index t in the series and in the SNN "are now sharing the same meaning". Ts = 4 — [SeqSNN](https://arxiv.org/abs/2402.01533)
- CPG-PE: "the input spike matrix X ∈ {0,1}^{T×B×L×D}, where T is the time step in SNNs ... L is the sequence length". T and L are separate axes, so spiking time ≠ token axis in its Spikformer setups — [CPG-PE](https://arxiv.org/abs/2405.14362)
- SE-adLIF audio compression uses raw "Pulse-Code-Modulation (PCM) sampled at 24 kHz" (rescaled per sample; details not transcribed). In the spring-mass task the network "received the mass displacements x[k] ∈ R⁴ as input at each time step k" — [SE-adLIF](https://arxiv.org/abs/2408.07517)
- QTDB ECG is T = N in time but not raw: "the two ECG signals were normalized and encoded into two separate spike trains via level-cross encoding" (threshold L = 0.3) — [BRF](https://arxiv.org/abs/2402.14603)
- Rhythm-SNN follows the same preprocessing ("a variant of the level-crossing encoding method") — [Rhythm-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484961/)

### Inferences
- T = N cannot be claimed as novel for raw continuous sequences. It is the standard format for sMNIST, psMNIST, raw SC and UEA in both neuron-model and SSM SNN papers.
- What is less common, and arguably our distinguishing choice, is putting a T = N recurrent "memory path" next to a spiking self-attention path that uses repeated presentation. Most spiking time-series transformers (iSpikformer, CPG-PE Spikformer) put T on a separate axis.
- In spiking SSMs, the long memory lives in a real-valued linear state (HiPPO-initialised S4D or S4 kernels); the LIF neuron is only the activation applied to y_t. Their LRA and Path-X success is therefore not evidence that LIF membrane state carries long memory. The neuron-model papers (TC-LIF, DH-SNN, BRF, SE-adLIF, PMSN, Rhythm-SNN) are the ones that test membrane or dendrite state as memory, mostly on S-MNIST and PS-MNIST.

### Gaps
- Whether P-SpikeSSM's SC35 result (96.23%) uses raw audio is not stated in the passage I read.
- Whether PMSN's pixel-level sCIFAR uses grayscale or RGB pixels per step was not checked. LRA Image is grayscale per Stan & Rhodes.
- I did not check whether SpikySpace or other 2026 forecasting SNNs use T = N.

## Which benchmarks are considered "memory-demanding" by the authors, and with what evidence?

### Takeaway
Authors call psMNIST, sCIFAR and LRA (especially Path-X) memory- or long-range-demanding. The evidence they give is of four kinds:
- plain LIF or vanilla resonate-and-fire nets fall to chance on PS-MNIST;
- accuracy collapses when the same images go from 32-step to 1,024-step presentation;
- Transformers stay at chance on Path-X while SSM-based spiking models reach about 94–95%;
- gradient-through-time plots show more gradient reaching early steps.

Explicit accuracy-vs-delay curves exist only for spike-input toy tasks (delayed spiking XOR, STORE-RECALL). For raw continuous tasks, the only accuracy-vs-length evidence I found is PMSN's T = 32 vs 1,024 comparison and LMUFormer's accuracy-vs-frames curve. The LMUFormer curve shows the opposite: Speech Commands is largely decided by about two-thirds of the frames.

### Cited Findings
- **PS-MNIST.** TC-LIF says the model "needs to capture complex, non-local, and long-term dependencies between pixels". Reproduced feed-forward LIF scores 11.30 / 10.00 (chance) vs TC-LIF 80.89 / 83.98. TC-LIF also plots error gradients over time on S-MNIST and shows TC-LIF delivers "more gradients to the earlier time steps" — [TC-LIF](https://arxiv.org/abs/2308.13250)
- DH-SNN: S-MNIST and PS-MNIST "are mainly used to evaluate the long-term memory capability of spatiotemporal networks" — [DH-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/)
- P-SpikeSSM: psMNIST "demands more sophisticated models capable of effectively retaining and synthesizing information over time" — [P-SpikeSSM](https://arxiv.org/abs/2406.02923)
- LMUFormer: psMNIST models "need to remember and integrate information over time, i.e., have the ability to handle long-range dependencies" — [LMUFormer](https://arxiv.org/abs/2402.04882)
- BRF: vanilla RF reaches 9.9±0.8% on PS-MNIST vs 98.0% on S-MNIST; BRF reaches 95.0% — [BRF](https://arxiv.org/abs/2402.14603)
- Rhythm-SNN: FFSNN scores 42.96 on PS-MNIST vs 95.01 for Rhythm-FFSNN. Normalised temporal gradients show Rhythm-SNNs "allocate more gradients to earlier time steps, facilitating the learning of long-range temporal dependencies" — [Rhythm-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484961/)
- **Sequential CIFAR.** PSN: "The sequential image classification task is widely used to evaluate the learning ability of a network for long-term dependencies" (section "Learning Long-term Dependencies") — [PSN](https://arxiv.org/abs/2304.12760)
- PMSN calls pixel-level T = 1,024 a setting that "poses a greater challenge to learning long-term dependency". The numbers: LIF falls from 81.50 (T = 32) to 45.07 (T = 1,024); PSN from 88.45 to 55.24; PMSN from 90.97 to 82.14 — [PMSN](https://arxiv.org/abs/2408.14917)
- **LRA and Path-X.** SpikingSSMs describes LRA sequences of 1K–16K steps as designed "to assess model abilities in long-context understanding". On Path-X, SpikingSSM reaches 94.80 — [SpikingSSMs](https://arxiv.org/abs/2408.14909)
- Stan & Rhodes call Path-X "the most difficult long-range modelling task in the LRA". They report "the Transformer fails to converge beyond random accuracy (50%)" while their S4D-Lin baseline reaches 92.5 and Binary S4D 61.2 — [Stan & Rhodes](https://arxiv.org/abs/2401.00955)
- **Raw SC10 (16k).** In P-SpikeSSM's table, Performer reaches 30.8, NRDE 16.5 and Transformer "×", vs S4 98.3 and P-SpikeSSM 95.6. The paper frames SC10 as one of its "long-range dependency based datasets" — [P-SpikeSSM](https://arxiv.org/abs/2406.02923)
- **UEA and PPG-DaLiA.** PMSN says both "involve significantly longer sequences, with several tasks containing more than 10,000 timesteps" — [PMSN](https://arxiv.org/abs/2408.14917)
- Log-NCDE chose the six UEA sets because "datasets with many observations have previously proved challenging for deep learning approaches" — [Log-NCDE](https://arxiv.org/abs/2402.18512)
- **Spring-mass system (SE-adLIF).** Velocity is unobserved, so "it is impossible to accurately predict the displacements ... from the displacements at time k alone. The network must therefore learn to keep track of the longer-time dynamics" — [SE-adLIF](https://arxiv.org/abs/2408.07517)
- **Delay tasks (spike input).**
  - DH-SNN uses delayed spiking XOR to "test the memory capability of the models by configuring different delay values" — [DH-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/)
  - TS-LIF's table shows LIF at chance from delay 20 while TS-LIF holds 0.977 at 20 and 0.585 at 40–50 — [TS-LIF](https://arxiv.org/abs/2503.05108)
  - Rhythm-SNN frames STORE-RECALL as a test of "working memory capacity" — [Rhythm-SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12484961/)
- **Counter-evidence for Speech Commands as a memory task.** The spiking LMUFormer "achieves 99% of its original performance while getting a 32.03% (1 − 87/128) reduction in the sequence length" (95.17 vs 96.12) — [LMUFormer](https://arxiv.org/abs/2402.04882)
- **Long-memory motivation in forecasting.** TS-LIF motivates itself by the claim that LIF "rapid membrane potential decay impairs its ability to capture long-term" dependencies. Its forecasting evidence is that longer training windows (24 → 96 → 168) help — [TS-LIF](https://arxiv.org/abs/2503.05108)

### Inferences
- For our purpose (a task where a T = N memory path should add information beyond spiking self-attention), the strongest continuous candidates in this literature are:
  1. **psMNIST**: cheap, standard, and LIF fails clearly. But a Spike-driven Transformer already reaches 96.21%, so the expected gap to attention is small.
  2. **Pixel-level sCIFAR (T = 1,024) vs column-level (T = 32)**: a built-in length manipulation on identical content, mirroring a "raw sample vs patch" comparison for CinC.
  3. **Raw SC10 (16k steps)**: attention baselines fail or score low, but full attention over 16k tokens would be too expensive for a Spikformer-style path anyway.
  4. **UEA EigenWorms (17,984) and PPG-DaLiA (49,920)**: physiological or biomechanical signals closest in domain to ECG, with an established 70:15:15, 5-split protocol and recent SNN numbers (PMSN).
- On CinC with 117 patch tokens, the sequence is about as short as the column-level sCIFAR setting (T = 32), where all neuron types land between 79.5% and 91.0% (PSN and PMSN tables). In the literature, the memory advantage only becomes large at hundreds to tens of thousands of steps (psMNIST 784, sCIFAR 1,024, Path-X 16k). This is consistent with our null result at N = 117.
- Speech Commands, by LMUFormer's own curve, is not a strong long-memory task (99% of accuracy from about 68% of the frames). It is better treated as a streaming-efficiency benchmark than a memory benchmark.

### Gaps
- None of the opened papers report accuracy-vs-delay or accuracy-vs-length curves for a continuous raw-signal task, except PMSN (32 vs 1,024) and LMUFormer (accuracy vs frames). The adding and copy problems with continuous inputs did not appear in any 2023–2026 SNN paper I opened.
- I did not find any SNN paper that compares a T = N recurrent path against spiking self-attention on UEA or PPG-DaLiA. That comparison appears to be open.
