# Forecasting / regression / anomaly-detection benchmarks where temporal order and long accumulation matter (state: 2026-11-25)

Reading conventions used in these notes

- "L" = input (look-back) length in time steps, "H" = forecast horizon.
- Every number below was read by me in the cited source (table / section given). Percent changes marked "(computed)" are my own arithmetic on numbers from the cited table.
- "arXiv-only" = I found no peer-reviewed version; "venue from memory" = I did not re-verify the venue in this session.
- Sources were read as full text (arXiv HTML / proceedings PDF converted to text), not from search snippets, unless stated.

---

## 1. Evidence about the standard long-term forecasting benchmarks (ETT, Weather, Electricity, Traffic, Exchange, ILI, Solar): do they reward order and long history?

### Takeaway
Longer history clearly helps on Traffic and Electricity (PatchTST: about 15–25 % lower MSE going from L = 96 to 336) and much less on ETT (about 5–10 %, and L = 720 is often no better than 336); the gain is tied to daily / weekly periodicity and is model-dependent. Order matters only in a coarse sense: shuffling every time step hurts a lot, but models change little when only the older half is shuffled or when the input window is moved 96 steps into the past, and published work attributes the benefit to trend / periodicity that a single linear map can read.

### Cited Findings

Order (shuffling) experiments

- Zeng et al. (AAAI 2023), Table 5, L = 96, time-step-level shuffling of the raw input before embedding. ETTh1, Linear model: MSE 0.395 → 0.824 at H = 96, 0.520 → 0.846 at H = 720; "average drop" 81.06 % for full shuffle and 4.78 % for swapping the two halves ("Half-Ex."). FEDformer 73.28 % / 3.44 %, Autoformer 56.91 % / 0.46 %, Informer 1.98 % / 0.18 %. — [Zeng et al., Table 5](https://ar5iv.labs.arxiv.org/html/2205.13504)
- Same table, Exchange: Linear 27.26 % (shuffle) / 46.81 % (half exchange); FEDformer −0.09 % / 0.20 %, Autoformer 0.09 % / 1.12 %, Informer −0.12 % / −0.18 %, i.e. the Transformers do not react to shuffling at all on Exchange. The shuffling experiment was run only on Exchange and ETTh1. — [Zeng et al., Table 5 and text](https://ar5iv.labs.arxiv.org/html/2205.13504)
- Zeng et al., Table 3 ("Close" vs "Far" input, H = 720, L = 96): using the 96 steps that end 96 steps before the forecast start instead of the most recent 96 steps changes MSE only from 0.251 to 0.265 (FEDformer, Electricity), 0.631 to 0.645 (FEDformer, Traffic), 0.255 to 0.287 (Autoformer, Electricity), 0.677 to 0.675 (Autoformer, Traffic). Authors' reading: "these models only capture similar temporal information from the adjacent time series sequence"; "long-term forecasting depends on whether models can capture the trend and periodicity well only". — [Zeng et al., Table 3](https://ar5iv.labs.arxiv.org/html/2205.13504)
- Tan et al. (NeurIPS 2024), Table 6, patch-based LLM forecasters and their LLM-free ablations, ETTh1 H = 96: shuffling the whole input degrades error by 50.3–68.5 %; shuffling only the first (older) half degrades it by only 3.3–12.1 %; swapping the halves 5.5–92.2 % depending on the model; ILI H = 24: full shuffle 68.9–118.0 %, older-half shuffle 13.0–56.2 %. Conclusion of the paper: LLM-based models are no more sensitive to order than their ablations. — [Tan et al., Table 6 and Sec. 4.4](https://ar5iv.labs.arxiv.org/html/2406.16964)
- SpikF (ICML 2025), Appendix A.1 / Figure 7: with a uniformly shuffled input, "SpikF achieves a 0.8 % improvement in MAE when sequential properties are incorporated, while iTransformer shows no performance improvement" (dataset of Figure 7 not stated in the text). — [SpikF, PMLR 267, App. A.1](https://proceedings.mlr.press/v267/wu25m.html)
- A 2026 augmentation paper (arXiv-only) trains with partially re-ordered overlapping patches ("Temporal Patch Shuffle") and reports 2.08–10.51 % average MSE improvement over the strongest competing augmentation across nine long-term benchmarks and five backbones, i.e. partial re-ordering inside the window acts as a useful regularizer rather than destroying the task. — [TPS, arXiv 2604.09067](https://arxiv.org/html/2604.09067)

Look-back length

- PatchTST (ICLR 2023), Table 9, supervised PatchTST/42, MSE at L = 96 → 336 → 720:
  - Traffic H = 96: 0.477 → 0.367 → 0.365 (−23 % at 336, computed); H = 720: 0.518 → 0.434 → 0.436 (−16 %).
  - Electricity H = 96: 0.174 → 0.130 → 0.130 (−25 %); H = 720: 0.237 → 0.202 → 0.197 (−15 % / −17 %).
  - Weather H = 96: 0.178 → 0.152 → 0.147 (−15 % / −17 %); H = 720: 0.350 → 0.320 → 0.304 (−9 % / −13 %).
  - ETTh1 H = 96: 0.393 → 0.375 → 0.376 (−5 %); H = 720: 0.480 → 0.449 → 0.458 (−6 % / −5 %).
  - ETTh2 H = 96: 0.294 → 0.274 → 0.279 (−7 % / −5 %); H = 720: 0.412 → 0.379 → 0.394 (−8 % / −4 %).
  - ETTm1 H = 96: 0.321 → 0.290 → 0.295 (−10 % / −8 %); H = 720: 0.450 → 0.420 → 0.416 (−7 % / −8 %).
  - ETTm2 H = 96: 0.178 → 0.165 → 0.162 (−7 % / −9 %); H = 720: 0.400 → 0.367 → 0.350 (−8 % / −13 %).
  - ILI (L in parentheses in the table header, 24–144): non-monotone (H = 24: 3.062 at the shortest, best 1.281 at the third length, 1.470 at the longest).
  — [PatchTST, Table 9 (Appendix A.4.2)](https://ar5iv.labs.arxiv.org/html/2211.14730)
- PatchTST Table 1 (Traffic, H = 96): L = 96 gives MSE 0.518, L = 336 gives 0.397; an input covering 380 steps but sub-sampled every 4 steps to the same 96 tokens gives 0.447, "indicating that longer look-back window conveys more important information even with the same number of input tokens". — [PatchTST, Table 1 and Sec. 1](https://ar5iv.labs.arxiv.org/html/2211.14730)
- Zeng et al.: with L ∈ {24 … 720}, "existing Transformer-based models' performance deteriorates or stays stable when the look-back window size increases. In contrast, the performances of all LTSF-Linear are significantly boosted" (Figure 4: Traffic and Electricity, H = 720; Appendix Figure 6: boosted "for most datasets (e.g., ETTm1 and Traffic)"); Exchange "do[es] not show improved results with a long look-back window", attributed to "low information-to-noise ratio". Only figures, no table of numbers. — [Zeng et al., Sec. 5.2 and App. C.2](https://ar5iv.labs.arxiv.org/html/2205.13504)
- Zeng et al., weight visualization: on Traffic the linear model puts high weight on lags that are multiples of 24 and 168 steps ("daily periodicity and a weekly periodicity"). — [Zeng et al., App. (Figure 7 discussion)](https://ar5iv.labs.arxiv.org/html/2205.13504)
- iTransformer (ICLR 2024), Figure 6 (L ∈ {48, 96, 192, 336, 720}, H = 96): Transformer forecasters "do not necessarily benefit from the increased lookback length", while the inverted versions (an MLP over the time axis per variable) improve with longer L. Figure only. — [iTransformer, Sec. 4.2 "Increasing lookback length"](https://ar5iv.labs.arxiv.org/html/2310.06625)
- "How Biased is Time Series Forecasting?" (arXiv 2502.09683v3; venue not verified): with L tuned over {96, 192, 336, 512, 720} for every model, models that treat each variable separately "favor longer windows (336 or 720), with 96 chosen only 11 % of the time", models that mix variables pick 96 or 192 in 55 % of cases; "hourly datasets (ETTh1 and ETTh2) typically prefer shorter windows around 96 … while 15-minute interval datasets (ETTm1 and ETTm2) favor longer windows (greater than 336)". Fixing L = 96 "biases results against univariate models". — [Sec. 6.4 and Sec. 7](https://arxiv.org/html/2502.09683v3)
- "Scaling Law for Time Series Forecasting" (arXiv 2405.15124; NeurIPS 2024 from memory): "optimal horizon does exist, and it will increase if the amount of available training data increases"; beyond it "performance degrades due to the inherent limitations of dataset size". — [Abstract and Sec. 1](https://arxiv.org/html/2405.15124)
- SpikF (ICML 2025), "Extension of Input Sequence" (Figure 5, ETTh2 and ETTm2, L ∈ {48, 96, 192, 336, 720}): "when the length of input sequence expands from 48 to 720, an averaged prediction promotion is 9.0 % in terms of MSE and 2.4 % in terms of MAE". — [SpikF, Sec. 4.3](https://proceedings.mlr.press/v267/wu25m.html)
- Foundation-model scale: a 2026 study of a generic patch Transformer on GIFT-Eval reports MASE 0.721 / 0.717 / 0.696 for context 2048 / 4096 / 8192 (Table 2) and states that "these gains are most pronounced in time series that exhibit long-term seasonality". arXiv-only. — [arXiv 2602.06909, Sec. 5.2.3](https://arxiv.org/html/2602.06909v1)

What the benchmarks consist of (periodicity, simplicity)

- SparseTSF (ICML 2024 from memory): "the basis for accurate long-term time series forecasting lies in the inherent periodicity and trend of the data"; the model down-samples by the known period and needs "fewer than 1k parameters". — [SparseTSF, Sec. 1](https://arxiv.org/html/2405.00946)
- CycleNet (NeurIPS 2024 from memory): the benchmark datasets "all exhibit stable cyclic patterns, such as daily and weekly, which form the realistic basis for performing long-horizon forecasting" (cycle length 24 for ETTh1, 168 for Electricity); a learned per-dataset cycle plus Linear / MLP is state of the art at L = 96 except on Traffic, where it is "inferior to iTransformer" because Traffic "exhibits spatiotemporal characteristics and temporal lag characteristics". — [CycleNet, Sec. 4](https://arxiv.org/html/2409.18479)
- Chen et al. (ICML 2025), "A Closer Look at Transformers for Time Series Forecasting": dependencies within one variable "are the primary contributors to prediction performance on benchmarks, while inter-variate dependencies have a minor impact". — [PMLR 267 abstract](https://proceedings.mlr.press/v267/chen25f.html)
- TFB (PVLDB 2024): (1) VAR and linear regression beat recent methods on some datasets; (3) "Transformer-based methods outperform linear-based methods on datasets with marked seasonality, and nonlinear patterns"; the "drop last" test trick changes reported MSE with batch size (PatchTST on ETTh2, L = 512, H = 336: 0.4203 at batch 1 vs 0.3483 at batch 512; Table 2). TFB searches L over {96, 336, 512}. — [TFB, Sec. 1, Table 2, Sec. 5](https://arxiv.org/html/2403.20150)

### Inferences
- Ranking of the standard datasets by "does longer history pay" (from PatchTST Table 9): Traffic ≈ Electricity (−15 to −25 %) > Weather (−9 to −17 %) > ETTm1 / ETTm2 (−7 to −13 %) > ETTh1 / ETTh2 (−5 to −8 %, worse again at 720) > ILI / Exchange (no consistent gain). So Traffic / Electricity do reward history more than ETT. No comparable look-back table for Solar was found.
- The gain saturates near L = 336 (two weeks of hourly data); on Traffic and Electricity L = 720 adds nothing (0.367 → 0.365, 0.130 → 0.130). This matches a "cover the weekly period" explanation, not open-ended accumulation.
- The published order evidence is consistent with the lab's own observation: order is used through a position-specific linear read-out (Zeng's Linear loses 81 % on ETTh1 when shuffled), the distant half of the window contributes little order information (3–12 % on ETTh1 when only the older half is shuffled), and attention bodies are close to order-blind (Informer, iTransformer ≈ 0 %, SpikF 0.8 %).
- The lab's "longer input does not help any of our models on ETT" is not a property of ETT alone: PatchTST (−7 to −13 % on ETTh2 / ETTm2 from 96 to 336) and SpikF (9.0 % MSE from 48 to 720 on ETTh2 / ETTm2) do gain. The benefit is model- and tuning-dependent (scaling-law and look-back-bias papers), and small in absolute terms on ETT.

### Gaps
- No paper was found that shuffles patch tokens (not time steps) across several standard datasets and several look-back lengths; Zeng's and Tan's tests are time-step-level and limited to Exchange, ETTh1 and ILI.
- No order-sensitivity or look-back table was found for Solar-Energy.
- TimeMixer's and Moirai's / Chronos' own context-length ablations were not read in this session; the only foundation-model context ablation read is arXiv 2602.06909.
- Zeng's and iTransformer's look-back results exist only as figures; no numbers can be quoted.

---

## 2. Spiking forecasting papers: datasets, horizons, metrics, numbers, and order / memory ablations

### Takeaway
Spiking forecasting work splits into two protocols: the SeqSNN protocol (Metr-la, Pems-bay, Solar, Electricity; H ∈ {6, 24, 48, 96}; R² and RSE; L = 12 or 168) and the long-term protocol of SpikF (eight standard datasets, L = 96, H ∈ {96 … 720}, MSE / MAE). None of them contains a controlled test that order or accumulated neuron memory is what produces the accuracy; the closest evidence is a positional-encoding ablation (CPG-PE), a 0.8 % shuffle effect (SpikF), and a synthetic delayed-XOR test (TS-LIF).

### Cited Findings

SeqSNN (Lv et al., ICML 2024)

- Datasets Metr-la, Pems-bay, Solar, Electricity; horizons 6 / 24 / 48 / 96; metrics R² (higher is better) and RSE (lower is better); three seeds. Average over the 16 cells (Table 1): ARIMA R² .713 / RSE .583; TCN .770 / .483; GRU .733 / .573; Autoformer .753 / .569; iTransformer .776 / .480; Spike-TCN .750 / .518; Spike-GRU .759 / .517; Spike-RNN .768 / .503; iSpikformer .775 / .476. — [SeqSNN, Table 1](https://arxiv.org/html/2402.01533v2)
- Per-cell examples (Table 1): Electricity H = 96 R²: ARIMA .863, TCN .962, iTransformer .964, iSpikformer .963; Metr-la H = 96 R²: ARIMA .265, TCN .330, iTransformer .285, iSpikformer .279, Spike-RNN .283. — [SeqSNN, Table 1](https://arxiv.org/html/2402.01533v2)
- Model size and energy (Table 3, Electricity, H = 24): TCN 0.460 M parameters, GRU 1.288 M, iTransformer / iSpikformer 1.634 M; iSpikformer 3.19 mJ vs iTransformer 9.47 mJ. — [SeqSNN, Table 3](https://arxiv.org/html/2402.01533v2)
- Memory-related statements: Spike-RNN / Spike-GRU beat the ANN GRU baseline while Spike-TCN is below TCN; the authors attribute this to Spike-TCN "doesn't track the temporal states". Decay-rate study (Figure 4, H = 24, Metr-la and Solar, β ∈ {0.80 … 0.99}): the text says "when the value of β increases, there is a noticeable decrease in R²" and in the next sentence that a higher β "is beneficial for retaining long-term information" — the two sentences conflict and the figure values are not given in text. — [SeqSNN, Sec. 4.2, 4.3.2](https://arxiv.org/html/2402.01533v2)
- No shuffling, no look-back-length ablation. The "temporal analysis" uses synthetic periodic signals only. — [SeqSNN, Sec. 4.5](https://arxiv.org/html/2402.01533v2)

CPG-PE (NeurIPS 2024)

- Same four datasets / horizons / metrics, three seeds (Table 1). Averages: Spikformer without positional encoding R² .719 / RSE .559; with the original Spikformer relative encoding .733 / .541; with CPG-PE .744 / .519; Spike-RNN .768 / .503 → with CPG-PE .773 / .482; Spike-TCN .750 / .518 → .760 / .506; ANN Transformer .752 / .512; ANN RNN .776 / .477. — [CPG-PE, Table 1](https://arxiv.org/html/2405.14362)

TS-LIF (ICLR 2025)

- Same protocol; "history length" 168 for Solar / Electricity and {12, 168} for Metr-la / Pems-bay (Table 3); SNN time steps "align[ed] … with the history length". Averages (Table 1): TS-TCN R² .777 / RSE .475; TS-GRU .778 / .460; TS-former .783 / .441 (iTransformer .776 / .480). Electricity H = 96: TS-former R² .977, RSE .261. — [TS-LIF, Table 1, Table 3, App. A.3](https://arxiv.org/html/2503.05108)
- Memory evidence is synthetic: delayed spiking XOR, accuracy at delay 10 / 20 / 30 / 40 / 50 steps: LIF 0.748 / 0.504 / 0.504 / 0.500 / 0.500, TS-LIF 0.994 / 0.977 / 0.791 / 0.585 / 0.585 (Table 11). Robustness to missing input values is tested (Tables 2, 9); no shuffle or look-back ablation on the forecasting datasets. — [TS-LIF, App. A.11](https://arxiv.org/html/2503.05108)

SpikF (Wu, Huo, Chen; ICML 2025)

- Long-term protocol "following iTransformer": ECL, Weather, ETTh1 / h2 / m1 / m2, Traffic, Exchange; L = 96; H ∈ {96, 192, 336, 720}; MSE / MAE. Averages over the four horizons (Table 1), SpikF vs iTransformer: ECL 0.183 / 0.275 vs 0.178 / 0.270; Weather 0.245 / 0.265 vs 0.258 / 0.278; ETTh1 0.440 / 0.428 vs 0.454 / 0.447; ETTh2 0.372 / 0.394 vs 0.383 / 0.407; ETTm1 0.388 / 0.385 vs 0.407 / 0.410; ETTm2 0.281 / 0.320 vs 0.288 / 0.332; Traffic 0.497 / 0.296 vs 0.428 / 0.282; Exchange 0.360 / 0.402 vs 0.360 / 0.403. — [SpikF, Table 1](https://proceedings.mlr.press/v267/wu25m.html)
- SNN rows on ETT (Table 12, averages MSE / MAE): ETTh1 SpikF 0.440 / 0.428, SpikeRNN 0.452 / 0.434, iSpikformer 0.473 / 0.446; ETTh2 0.372 / 0.394, 0.380 / 0.396, 0.392 / 0.407; ETTm1 0.388 / 0.385, 0.389 / 0.389, 0.402 / 0.398; ETTm2 0.281 / 0.320, 0.283 / 0.319, 0.283 / 0.322. — [SpikF, Table 12](https://proceedings.mlr.press/v267/wu25m.html)
- Solar short-term (Table 5, RSE at H = 6 / 24 / 48 / 96): SpikF 0.189 / 0.346 / 0.444 / 0.514; iSpikformer 0.204 / 0.333 / 0.465 / 0.521; iTransformer 0.191 / 0.348 / 0.448 / 0.563. The paper states the Solar look-back is 128 "following Lv et al." whereas TS-LIF's Table 3 lists 168 for the same protocol — the two papers disagree on the Solar input length. — [SpikF, Table 5 and App. B](https://proceedings.mlr.press/v267/wu25m.html); [TS-LIF, Table 3](https://arxiv.org/html/2503.05108)
- Order / memory evidence: 0.8 % MAE from correct order (App. A.1); 9.0 % MSE / 2.4 % MAE from L = 48 → 720 on ETTh2 / ETTm2 (Figure 5); replacing the frequency module by spiking inverted self-attention costs 1.6 % (Table 4: ETTh1 0.440 → 0.452). — [SpikF, Sec. 4.3, Table 4, App. A.1](https://proceedings.mlr.press/v267/wu25m.html)

Later / other spiking forecasters

- SpikF-GO (arXiv 2606.13901; footnote: accepted at ECML-PKDD 2026): eight datasets (ECG, COVID-19, Solar, Electricity, METR-LA, Traffic, PEMS-BAY, Wiki), main setting input 12 → horizon 12, metrics R² and MAE; "Increasing the input window from L = 96 to L = 168 yields negligible improvement". — [SpikF-GO, Sec. 4 and Figure 2 text](https://arxiv.org/html/2606.13901)
- SpikySpace (arXiv 2601.02411, arXiv-only): spiking state-space model; the four SeqSNN datasets; horizons {3, 6, 12, 24}; R² and RRSE; "performs best at shorter horizons but sees diminishing returns when the horizon is 24", attributed to the recursive state. — [SpikySpace, Sec. 4](https://arxiv.org/html/2601.02411)
- SpikeSTAG (arXiv 2508.02069v2, arXiv-only as far as verified): graph + LSTM + spiking attention; ablation Table 3 (R² / RSE): LSTM path alone Metr-la 0.823 / 0.436, Electricity 0.960 / 0.246; spiking attention alone 0.854 / 0.417 and 0.965 / 0.223; gated combination 0.868 / 0.375 and 0.987 / 0.207. — [SpikeSTAG, Table 3](https://arxiv.org/html/2508.02069v2)
- SHaRe-SSM (arXiv 2510.14386v2; an ICML 2026 paper by the same authors, "A Spiking Heterogeneous Harmonic Resonate-and-Fire State Space Model for Time Series", PMLR 306:1033-1060, is listed) reports a spiking state-space model on PPG-DaLiA and EigenWorms — see Question 3. — [arXiv v2](https://arxiv.org/html/2510.14386v2); [PMLR 306](https://proceedings.mlr.press/v306/agrawal26c.html)

### Inferences
- On the SeqSNN protocol a classical ARIMA already reaches R² .96 on Electricity at H = 6 / 24 and all neural models sit within about .01 R² of each other on Electricity; differences are larger on Metr-la / Pems-bay, where the input is only 12 steps. This protocol therefore cannot show long accumulation: either the history is 12 steps, or the dataset is strongly periodic and near-saturated.
- The CPG-PE ablation is the only spiking result that isolates order information: giving an order-blind spiking attention model positional encoding moves the average R² from .719 to .744, while the recurrent Spike-RNN (which has order built in) moves only from .768 to .773. That is, on these datasets order information is worth about .025 R² to an order-blind model and a recurrent model already has it.
- All spiking forecasters fit the lab's size range (SeqSNN: 0.46–1.63 M parameters).

### Gaps
- No spiking forecasting paper found (through November 2026) reports (a) an input-shuffle test with numbers per dataset, (b) a memory-off ablation (e.g. membrane reset every step), or (c) a stateful evaluation carrying neuron state across windows.
- SpikeSTAG's venue could not be confirmed; SHaRe-SSM numbers were read from the arXiv v2 text, not from the ICML 2026 camera-ready.
- The Semantic-Scholar citation graph of SeqSNN / SpikF was not enumerated; small 2026 spiking forecasting papers may be missing.

---

## 3. Regression / state-estimation tasks where a running state must be integrated over time

### Takeaway
The only regression benchmark found with (i) an accepted streaming protocol, (ii) published spiking baselines and (iii) direct evidence that carried neuron state replaces an input buffer is NeuroBench's primate motor prediction (a stateful LIF network without buffering matches a network fed the last 50 samples: R² 0.593 vs 0.593). Mackey-Glass has memory by construction and tiny published baselines. For C-MAPSS, battery state of charge, inertial odometry and BIDMC the published evidence for long ordered accumulation is weak or negative, and no usable spiking baselines were found.

### Cited Findings

Primate motor prediction (NeuroBench; "Primate Reaching" recordings)

- Task: predict two-dimensional fingertip velocity from 96-channel (monkey Indy) or 192-channel (monkey Loco) cortical recordings; metric R² averaged over X / Y; published in Nature Communications 2025 (DOI 10.1038/s41467-025-56739-4, 209 citations in OpenAlex on 2026-11-25). — [NeuroBench, arXiv v5, "NHP Motor Prediction"](https://arxiv.org/pdf/2304.04640); [DOI](https://doi.org/10.1038/s41467-025-56739-4)
- Baselines (Table 4): ANN (input = "the 50 most recent data samples … buffered", layers N-32-48-2) R² 0.593 (Indy) / 0.558 (Loco), footprint 20 824 / 33 496 bytes; SNN ("uses the data samples directly as input to the network, without buffering", N-50-2 LIF neurons, decay β = 0.96, non-resetting leaky read-out) R² 0.593 / 0.568, footprint 19 648 / 38 848 bytes, activation sparsity 0.997 / 0.999; model execution rate 250 Hz. The authors conclude this demonstrates "the adequacy of stateful, binary-activation neuron models for sparse regression tasks". — [NeuroBench, Table 4 and text](https://arxiv.org/pdf/2304.04640)
- Variants that re-bin the last 50 samples into 7 bins and reset the network for every prediction reach "significantly greater R²" at "two orders of magnitude" more effective operations for the spiking version (values only in Figure 4). — [NeuroBench, text below Table 4](https://arxiv.org/pdf/2304.04640)
- IEEE BioCAS 2024 Grand Challenge on the same data and harness: winning accuracy entry, an autoencoder-GRU, reports R² 0.71 and states that the duration of the input "plays a crucial" role; a hybrid spiking entry reports all three of its networks beat the NeuroBench baselines. Both arXiv versions of challenge papers. — [AEGRU, arXiv 2410.22283](https://arxiv.org/pdf/2410.22283); [Hybrid SNN, arXiv 2409.04428](https://arxiv.org/html/2409.04428v2)

Chaotic-system prediction

- NeuroBench chaotic function prediction: one-dimensional Mackey-Glass, 14 series with varying delay parameter, metric sMAPE; baselines at delay 17 (Table 5): echo state network (186 reservoir neurons) sMAPE 14.79, LSTM (100 hidden units plus an explicit buffer of the previous 50 points) sMAPE 13.37; no spiking baseline in the paper; error rises as the delay parameter goes from 17 to 30 (Figure 5). — [NeuroBench, Table 5](https://arxiv.org/pdf/2304.04640)
- DelRec (arXiv 2509.24852, arXiv-only): recurrent spiking network, one hidden layer of 128 LIF neurons, input window 150 steps, predict the value H ∈ {20, 30, 40, 50} steps ahead for delay parameter ∈ {17, 30, 40, 50}; series of 6000 points split 60 / 20 / 20; metric normalized MSE; learned recurrent delays improve over no-delay and fixed-delay recurrent networks "across all temporal difficulty conditions" (numbers only in Figures 3 and 10). The paper notes this task "requires continuously tracking a non-linear trajectory rather than collapsing temporal information into a single label". — [DelRec, Sec. 2 and Methods 4.3](https://arxiv.org/html/2509.24852)
- Chaotic ODE benchmark (Lorenz, coupled Lorenz, Hopfield, double pendulum, blinking rotlet, cell cycle) proposed as forecasting data with real cross-variable dependence; on it, models that mix variables win, the reverse of the standard datasets (7 of the standard datasets: variable-independent models win 21 to 7); RevIN normalization "can be detrimental to chaotic ODE datasets". arXiv 2502.09683v3. — [Sec. 6, Table 4, Table 13](https://arxiv.org/html/2502.09683v3)

Heart rate from PPG (PPG-DaLiA) — new information only

- A spiking model now has a published number on the Log-NCDE protocol: SHaRe-SSM (spiking, second-order state-space model) test MSE ×10⁻² 9.1 ± 0.2 (one discretization) and 11.8 ± 0.9 (the other); same table: Log-NCDE 9.6 ± 0.6, NRDE 9.9 ± 1.0, NCDE 13.5 ± 0.7, Mamba 10.7 ± 2.2, LRU 12.2 ± 0.5, S5 12.6 ± 1.3, LinOSS 7.5 ± 0.5 and 6.4 ± 0.2, RHEL 9.5 ± 1.0 / 8.4 ± 0.5 (five runs). EigenWorms accuracy in the same table: spiking 92.8 ± 3.3 / 90.0 ± 5.7, LinOSS 95.0 ± 4.4 / 80.0 ± 4.4, Log-NCDE 85.6 ± 5.1. — [SHaRe-SSM, arXiv v2, Table 2](https://arxiv.org/html/2510.14386v2)
- The best non-spiking number on this protocol is therefore no longer Log-NCDE (9.6) but LinOSS (6.4), as reproduced in that table. — [SHaRe-SSM, arXiv v2, Table 2](https://arxiv.org/html/2510.14386v2)

BIDMC respiratory rate / heart rate / SpO2 (Neural RDE protocol)

- Task: predict the value "at the end of the sample" from PPG and ECG, 125 Hz, series length 4 000 (32 s), 3 channels including time, data from the TSR archive; metric L2 loss. Best results (Table 2, depth-3 Neural RDE): respiratory rate 1.49 ± 0.08, heart rate 2.97 ± 0.45, SpO2 1.29 ± 0.15; Neural CDE at step 1: 2.79 / 9.82 / 2.83; folded ODE-RNN best 1.62 / 6.75 / 1.98. — [Neural RDE, Table 2 and Sec. 4](https://ar5iv.labs.arxiv.org/html/2009.08295)

Remaining useful life (NASA C-MAPSS turbofan)

- Controlled protocol audit (engrXiv preprint, not peer reviewed): compact MLP / temporal CNN / LSTM, factorial grid over subset, target cap, window length, normalization, seed (1080 runs). Best configurations (Table 5): FD001 LSTM, window 30, RMSE 15.97 ± 0.45; FD002 LSTM, window 50, 22.73 ± 0.28; FD003 MLP, window 50, 16.76 ± 0.22; FD004 LSTM, window 50, 21.78 ± 0.11. Architecture explains 75.0 % of the RMSE variation on FD001 and 74.8 % on FD003; on FD002 / FD004 the largest shares are the target cap and the normalization policy, not the architecture or window. — [C-MAPSS audit, Table 5 and Sec. 5.3](https://engrxiv.org/preprint/download/8145/13229/11468)
- Controlled comparison (arXiv 2604.27234, arXiv-only), 30-cycle window: single-layer LSTM RMSE 14.93 (FD001) / 14.20 (FD003); 1D CNN 16.97 / 15.68; XGBoost on per-window summary statistics (mean, standard deviation, last value, …) 13.36 on FD003. — [arXiv 2604.27234, abstract and Sec. III](https://arxiv.org/html/2604.27234v1)
- Spiking: the only "spiking" C-MAPSS paper found is on spiking neural P systems evolved with NEAT (Algorithms 2022), a different formalism from LIF networks. — [Custode et al., Algorithms 15(3):98](https://doi.org/10.3390/a15030098)

Battery state of charge

- A 2026 arXiv-only model argues the opposite of long accumulation: it performs "realistic online SOC estimation using only a short historical window (20 s)", and criticizes methods that "rely on long historical input sequences". — [C2L-Net, arXiv 2605.08653, abstract](https://arxiv.org/html/2605.08653)

Inertial odometry

- EqNIO appendix (sequence-length ablation on the TLIO architecture): "increasing the context window reduces MSE but not ATE" (the trajectory error that matters), confirming the same observation in the original TLIO paper. — [EqNIO, App. 8.10](https://arxiv.org/html/2408.06321v3)

A published test for "does this task need temporal processing" (spiking)

- Neuromorphic Sequential Arena (IJCAI 2025) compares three trainings of the same LIF network: full backpropagation through time, backpropagation without the temporal gradient, and a network whose membrane state is not carried at all ("NoTD"). Table 2 drops when state is not carried: activity recognition (WISDM, length 200) 81.27 → 67.75 %, EEG motor imagery (length 500) 65.20 → 52.48 %, audio denoising (a regression task, SI-SNR) 11.47 → 9.47 dB, speech recognition error 20.70 → 27.10 %. All models are about 0.1 M parameters for the four small tasks. — [NSA, Table 1, Table 2, Supplement](https://arxiv.org/html/2505.22035v1)

### Inferences
- Primate motor prediction is the one place where "neuron state accumulates the stream" is itself the published baseline design, and where a state-carrying spiking network equals a buffered feed-forward network at similar size. But the effective memory is short (the matching buffer is 50 samples at 250 Hz, i.e. 0.2 s), so it supports "state replaces a window", not "a long past is needed".
- Mackey-Glass makes order necessary by construction and lets the required memory be tuned with the delay parameter, but a feed-forward network on a 50-point buffer is an accepted baseline input, so it also does not separate "ordered accumulation" from "a window of raw values".
- C-MAPSS: an order-free summary of a 30-cycle window (XGBoost on window statistics, 13.36 on FD003) is competitive with a sequence model (LSTM 14.20), and protocol choices move RMSE more than architecture on two of four subsets. This is evidence against C-MAPSS as a demonstration of ordered accumulation, although the small-model requirement is easily met.
- BIDMC targets are rates inside a 32-second window; the benchmark was introduced to test long-sequence training (4 000 steps), and the reported differences are between methods of shortening the sequence. Nothing in the source shows that order beyond local waveform shape or history beyond the window matters.
- PPG-DaLiA remains the regression benchmark with the longest sequences and now has a spiking baseline (9.1) that beats Log-NCDE (9.6) but not LinOSS (6.4); this raises the bar for the lab's on-hold attempt rather than changing the earlier negative evidence.

### Gaps
- No LIF-type spiking baseline was found for C-MAPSS, BIDMC, battery state of charge / health, or inertial odometry. A spiking battery state-of-health paper ("SpikeSOH", arXiv 2603.22323) and a reservoir-spiking state-of-charge paper (Trakya Univ. J. Eng. Sci. 2026) surfaced in search but were not read; no numbers can be given.
- History-length ablations for battery state of charge with a fixed protocol (e.g. LG HG2 data, LSTM vs feed-forward) were not found in a readable primary source.
- WESAD and IEEE SPC 2015 heart-rate baselines, financial and load forecasting with regime memory: not investigated in this session.
- The official BioCAS 2024 leaderboard was not found; R² 0.71 is the winning team's own report.
- Lorenz / Mackey-Glass numbers for Legendre Memory Units and for spiking reservoirs were not read (search results only).

---

## 4. Anomaly detection: which benchmarks are reliable after the critiques, and does long context / order matter?

### Takeaway
After the critiques, the credible evaluation is TSB-AD (NeurIPS 2024) with the VUS-PR measure, and on it methods that ignore temporal order are at the top (multivariate: PCA 0.31 = CNN 0.31 = LSTM 0.31 > Transformer detectors 0.12–0.19). No published evidence shows that long context or correct order is needed on SMD, MSL, SMAP, SWaT or PSM; the available evidence points the other way.

### Cited Findings
- Wu and Keogh (IEEE TKDE; extended abstract ICDE 2022): "The majority of the individual exemplars in these datasets [Yahoo, Numenta, NASA, etc.] suffer from one or more of four flaws … much of the apparent progress in recent years may be illusionary"; they introduce the UCR Time Series Anomaly Archive. — [arXiv 2009.13807 abstract](https://arxiv.org/abs/2009.13807)
- Kim et al. (AAAI 2022): with the point-adjustment protocol "even a random anomaly score can easily turn into a state-of-the-art TAD method"; "an untrained model obtains comparable detection performance to the existing methods even when PA is forbidden"; for the untrained model "F1 increases even more when the window size gets longer". — [Kim et al., abstract and Sec. 3.3](https://ar5iv.labs.arxiv.org/html/2109.05257)
- Sarfraz et al., "Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?" (arXiv 2405.02678v3; ICML 2024 from memory): "On both standard point-wise F1 and range-wise F1 metrics, the simple baselines such as PCA reconstruction error perform better on all datasets while other baselines such as 1-NN distance and L2-norm are often very close to the best performing methods"; "state-of-the-art deep anomaly detection models effectively learn linear mappings"; forecasting-mode baselines used "a time window of size 5". — [Quo Vadis, abstract and Sec. 5](https://arxiv.org/pdf/2405.02678)
- TSB-AD (Liu and Paparrizos, NeurIPS 2024 Datasets and Benchmarks): 40 algorithms; curated from 40 datasets; includes cleaned SMD, SWaT, PSM, SMAP, MSL series in the multivariate part and UCR, NAB, Yahoo etc. in the univariate part (dataset table in Appendix). VUS-PR is chosen as "the most robust" measure. — [TSB-AD paper, Sec. 4–5 and Appendix](https://proceedings.neurips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf)
- TSB-AD Table 5, mean VUS-PR. Multivariate part: CNN 0.31, OmniAnomaly 0.31, PCA 0.31, LSTMAD 0.31, USAD 0.30, AutoEncoder 0.30, KMeansAD 0.29, … TimesNet 0.19, TranAD 0.18, AnomalyTransformer 0.12. Univariate part: Sub-PCA 0.42, KShapeAD 0.40, POLY 0.39, Series2Graph 0.39, MOMENT (fine-tuned) 0.39, … CNN 0.34, LSTMAD 0.33, TimesFM 0.30, Chronos 0.27, TimesNet 0.26, AnomalyTransformer 0.12. — [TSB-AD, Table 5](https://proceedings.neurips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf)
- TSB-AD discussion: "Statistical-based methods generally demonstrate robust performance, while neural network-based methods do not exhibit the superiority often attributed to them"; "Simpler architectures such as CNNs and LSTMs generally outperform more complex designs, such as advanced transformer architectures"; foundation models "struggle with sequence anomalies … [they] estimate only one new value per step using a limited look-back window. When faced with long sequence anomalies, the constrained temporal context often leads to reduced performance". — [TSB-AD, Sec. 5.2.4](https://proceedings.neurips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf)
- Window size: Koran et al. (arXiv 2408.06620, arXiv-only) on graph / attention detectors: "Not only is there a lack of correlation between window size and performance, but using a window size of 1 actually yielded" the highest F1 for one model; also notes "SWAT suffers from issues such as labelling discrepancy". — [arXiv 2408.06620, Sec. "Window Size Effect"](https://arxiv.org/pdf/2408.06620)
- Positive context evidence, from a zero-shot foundation-model paper (arXiv 2509.21190, arXiv-only) evaluated on TSB-AD-derived data with input windows 1 000 to 13 000: "For datasets with long-term patterns like UCR, Power, SMAP, and SMD, performance generally improves with a larger window, as this allows the model to establish a more robust baseline of 'normal' behavior"; the authors add the optimal context "is task-dependent". — [TimeRCD, "Impact of Context Window Size"](https://arxiv.org/html/2509.21190)
- Spiking anomaly detection: the one dedicated spiking method found (Vacuum Spiker, arXiv 2510.06910, arXiv-only) is evaluated on Dodgers Loop Sensor, CalIt2 and similar small datasets, not on the SMD / MSL / SMAP / SWaT / PSM family; OpenAlex also lists journal papers on spiking neural P systems for multivariate anomaly detection (Knowledge-Based Systems 2026; Applied Soft Computing 2026; Expert Systems 2025), which are a different formalism and were not read. — [Vacuum Spiker, Sec. 4.2](https://arxiv.org/html/2510.06910v1)

### Inferences
- PCA on single time points has no notion of order or history, yet it ties the best neural detectors on the multivariate TSB-AD part and PCA reconstruction error is the strongest simple baseline in the 2024 position paper. On these benchmarks a model cannot demonstrate ordered accumulation by its detection score.
- The one result in favour of context (larger windows help on UCR / SMAP / SMD for a zero-shot model) concerns establishing a reference of normal behaviour over thousands of steps, not temporal order; it is arXiv-only and model-specific.
- For the lab's "one model for three tasks" goal, anomaly detection can be reported on TSB-AD-M with VUS-PR as a credibility choice, but it should not be expected to show a benefit of the sequential path.

### Gaps
- No paper was found that shuffles the input window of a detector on SMD / MSL / SMAP / SWaT / PSM and reports the change.
- No LIF-type spiking baseline on TSB-AD or on the five classic multivariate benchmarks was found.
- The UCR Anomaly Archive's own leaderboard (KDD Cup 2021) and per-dataset VUS-PR values of TSB-AD (Tables 6 and 7) were not extracted.

---

## 5. Streaming / stateful evaluation: are there accepted protocols where a model carries state across consecutive windows?

### Takeaway
In forecasting, the accepted "online" protocols (FSNet, OneNet, and the corrected setting of ICLR 2025) carry state in the weights — the model is updated by gradient steps as the stream advances — while each prediction still reads a fixed window (60 or 96 steps); they say nothing about carried neuron state. An accepted protocol where neuron state itself is carried along a continuous stream exists in neuromorphic benchmarking (NeuroBench motor prediction), and there a state-carrying spiking network equals a feed-forward network given a 50-sample buffer.

### Cited Findings
- FSNet (ICLR 2023): online forecasting with "look-back window length … 60" and horizons H ∈ {1, 24, 48}; the reported quantity is the cumulative MSE / MAE while the model is updated on the stream; datasets ETT, ECL, Traffic, Weather plus synthetic drift data. — [FSNet, Sec. 4.1](https://ar5iv.labs.arxiv.org/html/2202.11672)
- Lau, Shao, Yeung (ICLR 2025): the FSNet / OneNet setting "suffers from information leakage" because the model is updated with ground truth for time steps it is then evaluated on; they delay feedback by H steps. With L = 96, H ∈ {1, 24, 48}, six datasets (Electricity, ETTh2, ETTm1, Exchange, Traffic, Weather), online updating "generally outperforms batch learning", e.g. Traffic H = 48: DLinear 0.769 (batch) vs 0.681 (online), PatchTST 0.621 vs 0.454, iTransformer 0.517 vs 0.457 (Table 2); "in some cases, batch learning outperforms online approaches" (e.g. ECL with iTransformer: 4.936 vs 6.015 at H = 48). Table 15 studies L ∈ {24, 48, 96, 192}. — [DSOF, Sec. 4, Tables 1–2, 15](https://proceedings.iclr.cc/paper_files/paper/2025/file/46e624c244cff669223d488defd4e835-Paper-Conference.pdf)
- NeuroBench motor prediction: the spiking baseline receives each new sample "without buffering" and its membrane state persists along the recording; it reaches R² 0.593 / 0.568, the buffered feed-forward network 0.593 / 0.558 (Table 4). — [NeuroBench, Table 4](https://arxiv.org/pdf/2304.04640)
- NeuroBench Mackey-Glass: even the recurrent LSTM baseline "utilizes explicit memory by buffering 50 previous datapoints"; the echo state network runs as a state-carrying reservoir (sMAPE 14.79 vs 13.37). — [NeuroBench, "Chaotic Function Prediction"](https://arxiv.org/pdf/2304.04640)
- Neuromorphic Sequential Arena: removing the carried membrane state ("NoTD") from an LIF network costs 13.5 points on activity recognition, 12.7 on EEG motor imagery and 2.0 dB on audio denoising (Table 2) — a within-sample (not across-window) measure of the value of carried state. — [NSA, Table 2](https://arxiv.org/html/2505.22035v1)

### Inferences
- No accepted long-term-forecasting protocol was found in which hidden or neuron state is carried from one evaluation window to the next and compared with re-starting from zero. The lab would have to define such a protocol itself, and should expect reviewers to ask for the comparison "same model, state reset at every window".
- The gains reported for online forecasting (up to tens of percent on Traffic) come from adapting weights to distribution shift; they are evidence that the standard datasets are non-stationary, not that a longer remembered past helps a fixed model.

### Gaps
- OneNet (NeurIPS 2023) was not read; it is represented here only through the ICLR 2025 paper's description.
- Stateful recurrent-network evaluation outside forecasting (language-model style carried hidden state, streaming keyword spotting) was not surveyed.
- No study was found that measures "carry state vs reset" for any model on ETT, Electricity or Traffic.

---

## 6. Which 3–5 candidates best satisfy all three conditions (order + accumulation demonstrably matter; reliable baselines, ideally spiking; reachable by a 0.2–1.5 M-parameter model)?

### Takeaway
No surveyed forecasting or anomaly-detection benchmark satisfies all three conditions with published proof; the candidates below each satisfy two and have a clearly stated weakness on the third. The best-supported order is: (1) Electricity / Traffic (optionally Solar) with long look-back, (2) NeuroBench primate motor prediction, (3) Mackey-Glass / chaotic prediction as a controlled probe, (4) PPG-DaLiA only if re-opened, (5) C-MAPSS as a weak fallback. Anomaly-detection benchmarks and the online-forecasting protocols should not be used to demonstrate order or accumulation.

### Cited Findings

Candidate 1 — Electricity and Traffic long-term forecasting with L swept from 96 to 336 (and Solar on the SeqSNN protocol)

- History demonstrably matters: PatchTST MSE falls 23–25 % (H = 96) and 15–16 % (H = 720) from L = 96 to 336 on both datasets, versus 5–8 % on ETTh1 / ETTh2. — [PatchTST, Table 9](https://ar5iv.labs.arxiv.org/html/2211.14730)
- Spiking baselines exist on both protocols: SpikF at L = 96 (ECL 0.183 / 0.275, Traffic 0.497 / 0.296; iTransformer 0.178 / 0.270 and 0.428 / 0.282), and SeqSNN / CPG-PE / TS-LIF on Electricity and Solar at H ∈ {6 … 96} (best spiking average: TS-former R² .783, RSE .441). — [SpikF, Table 1](https://proceedings.mlr.press/v267/wu25m.html); [TS-LIF, Table 1](https://arxiv.org/html/2503.05108)
- Small models suffice: 0.46–1.63 M parameters in SeqSNN Table 3; a period-aware model with under 1 k parameters is competitive on the long-term protocol. — [SeqSNN, Table 3](https://arxiv.org/html/2402.01533v2); [SparseTSF](https://arxiv.org/html/2405.00946)
- Weakness: the benefit is attributed to daily / weekly periodicity that a linear map reads directly (weights at lags 24 and 168; "Close" vs "Far" inputs nearly equivalent), and attention models show ≈ 0–1 % order sensitivity. — [Zeng et al., Table 3 and App.](https://ar5iv.labs.arxiv.org/html/2205.13504); [SpikF, App. A.1](https://proceedings.mlr.press/v267/wu25m.html)

Candidate 2 — Primate motor prediction (NeuroBench; BioCAS 2024 challenge)

- Stateful streaming is the standard protocol; a spiking baseline exists and equals the buffered feed-forward network (R² 0.593 vs 0.593, 0.568 vs 0.558); models are tens of kilobytes; top reported result R² 0.71 (autoencoder-GRU). — [NeuroBench, Table 4](https://arxiv.org/pdf/2304.04640); [AEGRU](https://arxiv.org/pdf/2410.22283)
- Weakness: the input is multi-channel cortical spike counts (not a generic sensor series), and the memory that matters is about 50 samples; no order-shuffle result is published.

Candidate 3 — Mackey-Glass (NeuroBench protocol) / chaotic ODE forecasting

- Order and memory are required by construction and tunable through the delay parameter; published baselines are tiny (echo state network with 186 neurons sMAPE 14.79, LSTM with 100 units sMAPE 13.37); a recurrent spiking network with 128 neurons is evaluated on windows of 150 steps in DelRec (arXiv-only, figures only). — [NeuroBench, Table 5](https://arxiv.org/pdf/2304.04640); [DelRec](https://arxiv.org/html/2509.24852)
- Weakness: synthetic; the accepted baselines read a 50-point buffer, so a windowed feed-forward model is also sufficient; spiking baselines with tabulated numbers are missing.

Candidate 4 — PPG-DaLiA (on hold)

- New since the internal report: a spiking state-space model reports 9.1 ± 0.2 (×10⁻² MSE) on the Log-NCDE protocol, better than Log-NCDE 9.6 ± 0.6; LinOSS 6.4 ± 0.2 is the best number in the same table. — [SHaRe-SSM, arXiv v2, Table 2](https://arxiv.org/html/2510.14386v2)

Candidate 5 — C-MAPSS remaining useful life

- Small models reach the usual level (single-layer LSTM RMSE 14.93 on FD001; compact LSTM 15.97 ± 0.45 in a controlled audit), and best windows are 30–50 cycles. — [arXiv 2604.27234](https://arxiv.org/html/2604.27234v1); [C-MAPSS audit, Table 5](https://engrxiv.org/preprint/download/8145/13229/11468)
- Weakness: order-free window statistics are competitive (XGBoost 13.36 on FD003), protocol choices dominate on FD002 / FD004, and no LIF-type spiking baseline was found. — same sources.

Excluded, with the reason found

- SMD / MSL / SMAP / SWaT / PSM and TSB-AD: order-blind PCA ties the best neural detectors (VUS-PR 0.31). — [TSB-AD, Table 5](https://proceedings.neurips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf)
- Metr-la / Pems-bay on the SeqSNN protocol: input of only 12 steps in the base setting. — [TS-LIF, Table 3](https://arxiv.org/html/2503.05108)
- Exchange and ILI: no consistent gain from longer input; Transformers insensitive to shuffling on Exchange. — [Zeng et al., Table 5](https://ar5iv.labs.arxiv.org/html/2205.13504); [PatchTST, Table 9](https://ar5iv.labs.arxiv.org/html/2211.14730)
- Inertial odometry: longer context lowers the training loss but not the trajectory error. — [EqNIO, App. 8.10](https://arxiv.org/html/2408.06321v3)
- Online forecasting protocols: benefit comes from weight updates, not carried state. — [DSOF](https://proceedings.iclr.cc/paper_files/paper/2025/file/46e624c244cff669223d488defd4e835-Paper-Conference.pdf)

### Inferences
- Summary table (my assessment from the cited evidence):

| Candidate | Order / accumulation shown in print | Reliable baselines | Spiking baselines | Small model reaches published level | Main risk |
|---|---|---|---|---|---|
| Electricity / Traffic, L 96 → 336 | history: yes (15–25 %); order: only coarse | yes (PatchTST, iTransformer, TFB) | yes (SpikF; SeqSNN family) | yes | gain is periodicity a flat head can read; 321 / 862 variables cost compute |
| Primate motor prediction | carried state = 50-sample buffer | yes (Nature Commun. benchmark, challenge) | yes | yes (tens of kB) | short memory; domain far from generic sensors |
| Mackey-Glass / chaotic | by construction | partly (NeuroBench sMAPE) | weak (arXiv, figures) | yes | synthetic; a buffer also solves it |
| PPG-DaLiA | not shown (lab's own trial negative) | yes (Log-NCDE protocol) | yes, new (9.1) | unknown | target moved to 6.4 |
| C-MAPSS | weak | many, protocol-sensitive | none found | yes | order-free features competitive |

- Because every public candidate can also be solved from a window of raw values, a convincing demonstration of the sequential path will need an added control on whichever dataset is chosen — the published templates are (a) Zeng's "Close vs Far" and shuffle tests, (b) the Neuromorphic Sequential Arena's "state not carried" training, and (c) NeuroBench's "no buffer vs 50-sample buffer" pairing.
- If the paper must stay inside the forecasting family reviewers know, Electricity is the lowest-risk choice: largest documented look-back gain, both spiking protocols available, and 321 variables rather than Traffic's 862.

### Gaps
- No public forecasting / regression / anomaly-detection benchmark was found with a published result of the form "a model that integrates the stream in order beats an equally sized model that sees the same history without order". This absence is itself the main finding of the search.
- Peak memory / runtime of small spiking models on Traffic (862 variables) at L = 336 is not reported in any source read.
- Candidates outside this assignment's scope (classification, delayed-cue tasks, EigenWorms) are covered by the earlier internal report and other researchers and are not ranked here.
