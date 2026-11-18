# Spiking attention and attention-like token mixers for sequences (time-series forecasting, language, speech, long sequences)

Scope note for the report writer. All numbers below were read from the papers' own PDFs (arXiv full text or official proceedings PDF), downloaded and text-extracted on 2026-11-17, unless marked otherwise. "Venue" was checked against the camera-ready header in the PDF and/or the DOI / proceedings page (OpenAlex record or PMLR page). Items marked **arXiv-only** have no peer-reviewed version that I could find and should be treated as lower reliability. The context model (for judging fit): channel-independent ETTh2 / ETTm2 forecasting, N = 24 / 84 / 180 tokens, D = 64; coupling = binary q (Neocortex spikes) x binary k, v (Hippocampus spikes), scores = q k^T * D^-0.5 + learned relative-position bias, no softmax, LIF on scores @ v.

## Q1. Lv et al. (ICML 2024) iSpikformer / Spike-TCN / Spike-RNN, its numbers, how attention is spiked, and the forecasting follow-ups (TS-LIF, SpikF, others) with their attention forms and ETT numbers

### Takeaway
Lv et al. (ICML 2024) report **no ETT results**: its benchmarks are Metr-la, Pems-bay, Solar and Electricity, scored with R2 / RSE. Its iSpikformer is plain Spikformer SSA placed inside iTransformer, so the tokens are variates (channels), not time. TS-LIF (ICLR 2025) changes only the neuron and also has no ETT results. The only peer-reviewed spiking forecaster I found with standard ETT numbers is **SpikF (ICML 2025)**. It drops attention altogether and uses a spiking FFT-based frequency selector. Its stated reason is that spiking self-attention is order-blind: shuffle the tokens and the outputs just shuffle with them, and spiking positional encodings have not fixed this. It reports ETTh2 average MSE / MAE 0.372 / 0.394 and ETTm2 0.281 / 0.320 (look-back 96, averaged over H = 96–720). An ablation on ETT that swaps its mixer for inverted spiking self-attention (iSSA) shows only a small loss (ETTh1 MSE 0.440 → 0.452).

### Cited Findings
**Lv et al., "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks" (ICML 2024, PMLR 235; header "Proceedings of the 41st International Conference on Machine Learning"; code github.com/microsoft/SeqSNN)**
- Framework: a spike encoder turns the series X ∈ R^{T×C} into spike trains with Ts SNN time steps per data step (B × Ts × T × C). Proposed encoders: delta encoder S = SN(BN(Linear(x_t − x_{t−1}))) and convolutional encoder S = SN(BN(Conv(X))). The decoder is a single Linear on the final spikes. — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- How attention is spiked: "we follow Spikformer to implement the spiking self-attention (SSA) mechanism and use it to replace the original self-attention layer in iTransformer". A channel-wise spiking embedding S_emb = SN(Linear(S)) ∈ R^{H×C} gives **C variate tokens**, so attention mixes channels, not time steps. Settings: 2 Spikformer blocks, SSA spiking-neuron threshold 0.25, SNN time step 4 (Appendix). Spike-TCN resets membrane potentials to 0 at every series time step, which allows parallel training. Spike-RNN/GRU keeps the membrane state across steps. — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- Datasets and metrics: Metr-la, Pems-bay, Solar, Electricity, with horizons 6 / 24 / 48 / 96, scored by R2 and RSE and averaged over 3 seeds. ETT is not used anywhere in the paper. — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)
- Table 1 averages (R2↑ / RSE↓): iSpikformer .775 / .476, iTransformer .776 / .480, Spike-RNN .768 / .503, Spike-TCN .750 / .518, Autoformer .753 / .569. iSpikformer on Electricity: R2 .977 / .974 / .972 / .963 and RSE .263 / .284 / .338 / .348 for horizons 6 / 24 / 48 / 96. Table 3 (Electricity) energy: iSpikformer 1.634 M params, 3.55 G OPs, 3.19 mJ, R2 .974. — [arXiv 2402.01533](https://arxiv.org/abs/2402.01533)

**TS-LIF, "A Temporal Segment Spiking Neuron Network for Time Series Forecasting" (ICLR 2025; PDF header "Published as a conference paper at ICLR 2025"; code github.com/kkking-kk/TS-LIF)**
- Mechanism: a two-compartment neuron (dendrite + soma), extended with direct somatic current injection and dendritic spike generation. Dendritic and somatic spikes are mixed by a coefficient κ. The paper argues that the dendrite (α1 near 1) acts as a low-pass, moving-average path and the soma tracks fast changes. The **attention is unchanged**: "TS-former" is iTransformer built on TS-LIF, i.e. the iSpikformer SSA with the neuron swapped. — [arXiv 2503.05108](https://arxiv.org/abs/2503.05108)
- Table 1 uses the same four datasets as Lv et al. (no ETT). TS-former averages R2 .783 / RSE .441, against iSpikformer .775 / .476. TS-GRU averages .778 / .460. The paper states a 25.0% RSE improvement for TS-former on Electricity at the 96-step horizon (RSE .261 vs .348). Table 2 adds a missing-value robustness test on Electricity. — [arXiv 2503.05108](https://arxiv.org/abs/2503.05108)

**SpikF, "Spiking Fourier Network for Efficient Long-term Prediction" (Wu, Huo, Chen; ICML 2025, PMLR 267:67342–67368; code github.com/WWJ-creator/SpikF)**
- Venue and abstract: "an attention-free framework". The paper claims an average 1.9% MAE reduction over SOTA models on 8 long-term benchmarks and 3.16× lower total energy. — [PMLR page](https://proceedings.mlr.press/v267/wu25m.html); [ICML slides](https://icml.cc/media/icml-2025/Slides/46411.pdf); [code README](https://github.com/WWJ-creator/SpikF)
- Motivation, stated directly: "the traditional self-attention mechanism suffers from permutation-invariance … positional encoding in Spiking Transformers remains underexplored". The paper adds that Spikformer with CPG-PE "exhibit[s] only marginal performance improvements" and still trails SpikeTCN and SpikeRNN by 2.6% and 7.7% in RSE. — [SpikF PDF (PMLR)](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- Mechanism (Sec. 3.4, Spiking Frequency Selector): spiking patches are grouped at fixed intervals. One branch applies a spiking FFT (S-FFT). The other branch builds a spiking frequency selector from Linear + BN + spiking neuron, followed by spiking max-pooling. The selected spectrum goes back to the time domain through a grouped S-iFFT, and the resulting real values drive LIF neurons. The paper says S-FFT "enables the spiking selector to maintain a global receptive field across the entire time-series while leveraging positional information". Complexity is FFT-based, O(L log L) per patch group. That bound is my inference; the paper only plots S-FFT SOPs against sparsity. — [SpikF PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- Table 1 (look-back L = 96, MSE / MAE averaged over H = 96 / 192 / 336 / 720): **ETTh2** SpikF 0.372 / 0.394, iTransformer 0.383 / 0.407, RLinear 0.374 / 0.398, PatchTST 0.387 / 0.407. **ETTm2** SpikF 0.281 / 0.320, iTransformer 0.288 / 0.332, RLinear 0.286 / 0.327, PatchTST 0.281 / 0.326. ETTh1 SpikF 0.440 / 0.428, ETTm1 0.388 / 0.385. — [SpikF PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- Per-horizon SpikF values as reproduced in SpikeLite's Table 5 (**secondary source, arXiv-only**; their averages match SpikF Table 1): ETTh2 H96 .290 / .336 and H720 .416 / .436; ETTm2 H96 .175 / .251 and H720 .405 / .397. I did not check these against SpikF's own appendix table. — [SpikeLite arXiv 2609.35097](https://arxiv.org/abs/2609.35097)
- **Ablation against Spikformer-style attention (Table 4)**, replacing the SFS mixer with inverted spiking self-attention (iSSA), MSE / MAE: ETTh1 0.440 / 0.428 → 0.452 / 0.440; ETTm1 0.388 / 0.385 → 0.389 / 0.389; Weather 0.245 / 0.265 → 0.247 / 0.268. The text summarises this as "the SFS shows a 1.6% performance enhancement compared to iSSA". In the same table, the spiking patch encoder beats the delta encoder by 6.6% and the convolutional encoder by 3.2%. — [SpikF PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- **Order / shuffle control (Appendix A.1, Fig. 7)**: the paper proves that both standard and inverted (spiking) self-attention are permutation-equivariant. "we uniformly shuffled the input sequence … SpikF achieves a 0.8% improvement in MAE when sequential properties are incorporated, while iTransformer shows no performance improvement". — [SpikF PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- Appendix A.2 argues that the LIF reset limits each neuron's receptive field to the span since its last spike, "thus, relying solely on SNN internal dynamics is insufficient". — [SpikF PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- Table 5 (Solar, RSE, look-back 128, H = 6 / 24 / 48 / 96): SpikF 0.189 / 0.346 / 0.444 / 0.514, iSpikformer 0.204 / 0.333 / 0.465 / 0.521, iTransformer 0.191 / 0.348 / 0.448 / 0.563. — [SpikF PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)

**Other spiking forecasters found (lower reliability or no ETT)**
- SpikeLite (Hu, **Lv**, et al.; **arXiv-only**, 2609.35097) has two parts: a Frequency-Selective Spiking Encoder, and Sparse Spiking Channel Attention, which learns a binary mask that restricts which channels exchange information inside spike-driven self-attention. Without channel interaction it reduces to a channel-independent, encoder-only path. Under the SpikF protocol (Table 2, averaged over H and 3 seeds): ETTh2 .374 / .394, ETTm2 .278 / .319, overall average .343 / .345 (SpikF .346 / .346). — [arXiv 2609.35097](https://arxiv.org/abs/2609.35097)
- SpikF-GO (ECML PKDD 2026, LNCS 16943) combines a spiking Fourier graph operator with a FourierGNN-style hypervariate graph. Its datasets are ECG, COVID-19, Solar, ECL, METR-LA, Traffic, PEMS-BAY and Wiki, so **no ETT**. It repeats the argument that "standard self-attention is permutation-invariant". — [arXiv 2606.13901](https://arxiv.org/abs/2606.13901)
- SpikySpace (**arXiv-only**, 2601.02411) is a spiking selective-scan SSM (Mamba-like) with bit-shift approximations PTsoftplus / PTSiLU. It is evaluated on four multivariate benchmarks (no ETT; its Table 1 compares with iSpikformer). It claims "up to 3.0%" better accuracy than the leading SNN and >96.1% lower energy. — [arXiv 2601.02411](https://arxiv.org/abs/2601.02411)
- SEDformer (**arXiv-only**, "Under review", 2602.02230) targets irregular telemetry series. It uses an event-aligned LIF whose leak depends on the interval between observations, plus a "membrane-based linear attention whose scores depend on inter-event gaps". — [arXiv 2602.02230](https://arxiv.org/abs/2602.02230)
- Closed-loop forecasting with SSM-based SNNs (**arXiv-only**, 2610.28495) uses **univariate** ETT-small without ETTm2, an S4D backbone and Signed-Spiked activations (−1 / 0 / +1). Its protocol is not standard, so the numbers are not comparable to Table 1 above. — [arXiv 2610.28495](https://arxiv.org/abs/2610.28495)

### Inferences
- No peer-reviewed paper I found gives ETT numbers for a **spiking attention** forecaster under the standard protocol. The nearest is SpikF's iSSA ablation on ETTh1 / ETTm1, which uses inverted variate tokens, not time tokens. Any claim that a spiking attention variant "works on ETT" therefore has to be backed by our own runs.
- SpikF's shuffle control (ordered vs shuffled input: 0.8% MAE for SpikF, about 0% for iTransformer) matches our finding that scrambling the Neocortex order makes no clear difference on ETT. It suggests the exploitable order signal at look-back 96 is about 1% of MAE, even for a mixer designed to be order-sensitive. A better mixer may still show only small ETT gains. Shuffle and memory-removal controls should be standard for any candidate.
- In the Lv / TS-LIF / SpikeLite line, temporal information sits in the encoder or the neuron, not in attention. Attention there is cross-variate. This is the opposite of our setup, where the coupling attention mixes **time tokens** in a channel-independent model.

### Gaps
- I did not read SpikF's full per-horizon appendix table (I only have the per-horizon values reproduced by SpikeLite), and I did not check SpikF's results at look-back 336 / 720. Fig. 5 reports ETTh2 / ETTm2 loss for look-backs 48–720, but I only saw the text "the loss values generally decrease as the input sequence length increases".
- I found no NeurIPS 2025, ICLR 2026 or AAAI 2026 spiking forecasting paper with ETT results. Searches turned up only arXiv preprints (SpikySpace, SEDformer, SpikeLite). SpikeSTAG (cited by SpikySpace and SpikeLite) and "Rethinking Time-Series Forecasting as Binary Event Prediction with SNNs" (ICNC 2026, per OpenAlex) were not read.
- Lv et al.'s PMLR page was not fetched. The venue rests on the camera-ready header in the arXiv PDF.

## Q2. Spiking language / long-sequence mixers (SpikeGPT, SpikeLM, SpikeBERT, SpikingBERT, SpikeLLM, SpikingSSMs, P-SpikeSSM, LMUFormer and spiking SSMs): mechanism, causal handling, complexity, spike-drivenness, results

### Takeaway
Credible order-aware spiking mixers for long sequences are **recurrent or state-space**, not attention. Examples: SpikeGPT's spiking RWKV (TMLR 2024; linear in N, causal by recurrence, exponential decay as the position bias), SpikingSSMs (AAAI 2025), P-SpikeSSM (ICLR 2025) and LMUFormer (ICLR 2024). In each, order is built into the operator rather than learned from spike content. None of them is fully binary inside its recurrent state. SpikeLM (ICML 2024) and SpikeLLM (ICLR 2025) are about spike encoding and quantization, not about new token mixers. SpikeBERT is a distilled Spikformer: arXiv preprint, later a Neural Networks journal article.

### Cited Findings
**SpikeGPT (Zhu, Zhao, Li, Eshraghian; TMLR 06/2024; OpenReview id gcf1anBL9e; code github.com/ridgerchu/SpikeGPT)**
- Venue: PDF header "Published in Transactions on Machine Learning Research (06/2024)". — [arXiv 2302.13939](https://arxiv.org/abs/2302.13939); [OpenReview forum](https://openreview.net/forum?id=gcf1anBL9e)
- Mechanism (Spiking RWKV, Eq. 4): inputs are streamed one token at a time, with R = X[t]M_R, K = X[t]M_K, V = X[t]M_V, and "all inputs to the three MLP layers are in the form of spiking activations". The output is Y[t+1] = SN(σ(R[t]) ⊙ (exp(W_f) ⊙ exp(K[t]) ⊙ V[t] + A[t]) / (exp(W_f) ⊙ exp(K[t]) + B[t])). The states are A[t] = exp(K[t]) ⊙ V[t] + exp(W_d) ⊙ A[t−1] and B[t] = exp(K[t−1]) + exp(W_d) ⊙ B[t−1]. W_d and W_f are learnable decay vectors: W_d decays past information and W_f weights the current step. There is "no matrix-matrix multiplication to dynamically adjust the attention map", and the module "naturally operate[s] in a unidirectional manner", so no causal mask is needed. Token shift is used as well. — [arXiv 2302.13939](https://arxiv.org/abs/2302.13939)
- Complexity: O(T·d), versus O(T²·d) for the Transformer (Table 2). It is **not fully spike-driven**: the WKV state uses exp, division and real-valued accumulators, and spikes appear only at SN(·) outputs and as inputs to the linear layers. — [arXiv 2302.13939](https://arxiv.org/abs/2302.13939)
- Enwik8 (Table 2, test BPC): SpikeGPT 46M scores 1.283 at T = 1024 and 1.262 at T = 3072. Baselines: Transformer 1.137 (43.0M), Linear Transformer 1.207, Performer 1.199, Reformer 1.195, Synthesizer 1.298, SHA-LSTM 1.330. The paper notes SpikeGPT may be overfitting. — [arXiv 2302.13939](https://arxiv.org/abs/2302.13939)
- WikiText-103 perplexity as reported by SpikingSSMs (Table 4): SpikeGPT 39.75 (213M), SpikingSSM 33.94 (75M), Transformer 20.51 (231M), S4 20.95 (249M). — [arXiv 2408.14909](https://arxiv.org/abs/2408.14909)

**SpikeLM (Xing et al.; ICML 2024, PMLR 235; code github.com/Xingrun-Xing/SpikeLM)**
- Mechanism: "elastic bi-spiking" replaces {0, 1} spikes with {−α, 0, α}, i.e. a sign plus a learned amplitude, and adds spike-frequency control. Inside attention, "By encoding the key and value as spikes … all matrix multiplications are converted to additions. We set β = 0 in key and value", so keys and values carry no membrane memory. The query and the attention weights stay non-spiking. The baselines are BERT / BART-style, so order comes from those architectures' standard position embeddings (my inference; not separately described). — [arXiv 2406.03287](https://arxiv.org/abs/2406.03287)
- GLUE dev average (Table 3): SpikeLM 76.5 (T = 4) and 75.7 (T = 1); SpikingFormer transferred to language 68.9 (T = 4) and 67.1 (T = 1); LIF-BERT 54.9; SpikeBERT 59.7; BERT-base 83.2; BERT 3-layer 75.9. Ablation (Table 6): {0, 1} 54.9 → {−1, 0, 1} 55.7 → {−α, 0, α} 76.5. — [arXiv 2406.03287](https://arxiv.org/abs/2406.03287)

**SpikeBERT (Lv et al.)**: arXiv preprint (PDF header "Preprint"). OpenAlex lists a 2025 journal version in Neural Networks (doi 10.1016/j.neunet.2025.108482), not AAAI. It is Spikformer adapted to language and trained by two-stage knowledge distillation from BERT. — [arXiv 2308.15122](https://arxiv.org/abs/2308.15122); [Neural Networks DOI](https://doi.org/10.1016/j.neunet.2025.108482)

**SpikingBERT (Bal & Sengupta)**: AAAI 2024 (AAAI 38(10)), distillation with implicit differentiation. Venue verified only; not read in detail. — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/28975)

**SpikeLLM (ICLR 2025; PDF header "Published as a conference paper at ICLR 2025")**: Generalized Integrate-and-Fire neurons compress spike length from T to (T/L)·log2 L bits. An "Optimal Brain Spiking" step gives outlier channels more steps. It works inside quantization pipelines (OmniQuant / GPTQ) on LLaMA. Abstract: "SpikeLLM reduces 11.01% WikiText2 perplexity and improves 2.55% accuracy of common scene reasoning on a LLAMA-7B W4A4 model". It is a spiking quantization of linear-layer activations, **not a new token mixer**. — [arXiv 2407.04752](https://arxiv.org/abs/2407.04752)

**SpikingSSMs (Shen et al.; AAAI 2025, AAAI 39(19); code github.com/shenshuaijie/SDN)**
- Venue: [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/34245) and [arXiv 2408.14909](https://arxiv.org/abs/2408.14909)
- Mechanism: an S4D-style SSM whose output nonlinearity is replaced by a LIF neuron ("SpikingSSM replaces the non-linear function of original SSM with an LIF neuron"). A small "surrogate dynamic network" (SDN) predicts the after-reset membrane potential so training can run in parallel. The SSM convolution runs by FFT in O(L log L), and inference is recurrent. Causal. The inputs to the feature-mixing layers are binary, but the SSM state and outputs are real-valued, so it is **partially spike-driven**. — [arXiv 2408.14909](https://arxiv.org/abs/2408.14909)
- LRA (Table 3; ListOps / Text / Retrieval / Image / Pathfinder / Path-X / Avg): SpikingSSM-VT 60.23 / 80.41 / 88.77 / 88.21 / 93.51 / 94.82 / 84.33; SpikingSSM-VF average 84.30; S4D-Lin 85.52; Binary S4D 74.69; Spiking LMUFormer 60.20; Transformer 53.66. Spiking rate is about 0.10. sCIFAR10 (Table 5): LIF 85.45%, 1480 ms, versus SDN 85.57%, 230 ms. — [arXiv 2408.14909](https://arxiv.org/abs/2408.14909)

**P-SpikeSSM (Bal & Sengupta; ICLR 2025; code github.com/NeuroCompLab-psu/PSpikeSSMs)**
- Mechanism: a "SpikeSampler" draws spikes stochastically from an SSM-based neuron model, which allows parallel computation. A matching surrogate gradient handles the stochastic sampling. "SpikeMixer" mixes populations and "ClampFuse" adds a residual. The paper marks it "Yes*": GELU is used, but there are no floating-point matrix multiplications. — [arXiv 2406.02923](https://arxiv.org/abs/2406.02923)
- LRA (Table 2; ListOps / Text / Retrieval / Image / Pathfinder): 58.20 / 81.20 / 88.53 / 82.40 / 84.80, against Spiking LMUFormer 37.30 / 65.80 / 79.76 / 55.65 / 72.68 and S4 (original) 58.35 / 76.02 / 87.09 / 87.26 / 86.05. Speech Commands: SC10 95.6% (Table 3) and SC35 96.23%. — [arXiv 2406.02923](https://arxiv.org/abs/2406.02923)

**LMUFormer (Liu et al.; ICLR 2024; code github.com/zeyuliu1037/LMUFormer)**
- Mechanism: the token mixer is a Legendre Memory Unit, a linear time-invariant recurrent memory. It adds a convolutional patch embedding and a convolutional channel mixer, and the spiking version also keeps states inside those modules. Inference is sequential (streaming) and training is parallel. — [arXiv 2402.04882](https://arxiv.org/abs/2402.04882)
- Speech Commands V2, 35 classes (Table 2): Spiking LMUFormer 96.12%, LMUFormer 96.53%, LMUFormer with states 96.92%. **Same network with the LMU replaced by Spikformer SSA ("Spikformer*") 93.38%; by Spike-Driven Self-Attention ("Spike-driven ViT*") 94.85%.** LRA average (Table 4): Spiking LMUFormer 62.24, LMUFormer 61.08, Transformer 54.39, S4 80.48. — [arXiv 2402.04882](https://arxiv.org/abs/2402.04882)

### Inferences
- In SRWKV, SSMs and LMU, position enters through a **fixed or learned decay / recurrence operator** applied along the token axis. Order is structural and cannot be ignored, unlike a content-only binary q·k dot product. Our Neocortex path is already such a recurrence, but its output is used only as a query, so the coupling can still ignore its order content. A mixer in the SRWKV / SSM style would make the recurrence part of the mixing operator itself (for example, a decayed accumulation over Hippocampus k / v along tokens).
- With N ≤ 180 and D = 64, the O(N²) cost of attention is negligible. The linear-complexity advantages of SpikeGPT, SSMs and LMU are therefore irrelevant for choosing among candidates; order handling and trainability should decide.
- None of SpikeGPT, SpikingSSMs or LMUFormer keeps the recurrent state binary. A "fully spike-driven" replacement would follow P-SpikeSSM's style (no floating-point matmul, though GELU remains) or keep a binary interface around a real-valued membrane state, which is what LIF neurons already do.

### Gaps
- No spiking language or long-sequence mixer above reports ETT or any time-series forecasting result.
- I did not read the SpikeGPT WikiText-103 numbers in the SpikeGPT paper itself (39.75 is quoted from SpikingSSMs), nor the SpikingBERT details.
- Related newer items found but not read: LongSpike, fractional-order spiking SSM (arXiv 2606.12895; relevant to f-LIF); Dyn-SSM (IEEE TCDS 2026, arXiv 2410.17268); Sorbet (arXiv 2409.15298).
- Whether LMUFormer's Spikformer* swap is parameter-matched is not stated in the passages I read.

## Q3. Positional encoding for spiking transformers (relative / spike-friendly PE): what works and the evidence

### Takeaway
Two peer-reviewed spike-native position encodings exist, both from the Lv / Microsoft SeqSNN group:
- **CPG-PE** (NeurIPS 2024): absolute, binary oscillator codes concatenated to the features.
- **Gray-PE / Log-PE** (NeurIPS 2025): relative. They require replacing the dot-product score with an **XNOR (Hamming-similarity) score**.

On forecasting (Metr-la / Pems-bay / Solar / Electricity) they improve Spikformer by only about +0.01–0.03 in average R2. Most important for our coupling: **adding a relative bias straight onto dot-product binary attention (ALiBi or Log-PE on plain Spikformer) failed or collapsed** in their experiments. That is structurally what our "q k^T·D^-0.5 + learned relative bias" does.

### Cited Findings
**CPG-PE, "Advancing Spiking Neural Networks for Sequential Modeling with Central Pattern Generators" (Lv et al.; NeurIPS 2024, DOI 10.52202/079017-0845; code in microsoft/SeqSNN)**
- Mechanism: N pairs of CPG cells (2N in total), each pair a Heaviside of a cosine / sine of t at its own frequency minus a threshold: CPG-PE_{2i−1}(t) = H(cos(·) − v_thres), CPG-PE_{2i}(t) = H(sin(·) − v_thres). The frequency of pair i is set by constants η (period control) and τ (base period) and by i / N. The exact argument is garbled in my text extraction (Eqs. 11–12), so check the PDF before reimplementing. t runs over the flattened T × L (SNN step × position). The codes are concatenated to the input spikes, X1 = [X ‖ X′], and then X_out = SN(BN(Linear(X1))) maps D + 2N back to D. Parameter-free, binary, absolute. — [arXiv 2405.14362](https://arxiv.org/abs/2405.14362); [DOI](https://doi.org/10.52202/079017-0845)
- Table 1, forecasting averages (R2 / RSE): Spikformer w/o PE .719 / .559; Spikformer with its original conv relative position embedding .733 / .541; with Float-PE .734 / .531; **with CPG-PE .744 / .519**; with CPG-Full .744 / .523. Also SpikeTCN .750 → .760 and SpikeRNN .768 → .773 with CPG-PE. ANN Transformer reference .752 / .512. — [arXiv 2405.14362](https://arxiv.org/abs/2405.14362)
- Text classification (Table 2, 6 benchmarks, 5 seeds): Spikformer w/o PE 77.25, Random-PE 77.23 (control at the same parameter count, 110.4M), Float-PE 78.44, **CPG-PE 79.41**. Fine-tuned BERT scores 84.33. — [arXiv 2405.14362](https://arxiv.org/abs/2405.14362)

**Gray-PE / Log-PE, "Toward Relative Positional Encoding in Spiking Transformers" (Lv et al.; NeurIPS 2025, DOI 10.52202/085713-2601; code microsoft/SeqSNN)**
- Mechanism: (1) XNOR-based SSA, AttnMap = Σ_D ¬(Q ⊕ K), which counts matching 0s as well as matching 1s. (2) Gray-PE concatenates the binary Gray code of each position to both Q and K: AttnMap = Σ ¬([Q ‖ G(l)] ⊕ [K ‖ G(l)]). Theorem 1: two positions that differ by 2^n have Gray-code Hamming distance 1 if n = 0 and 2 if n ≥ 1. (3) Log-PE adds R_ij = ⌈log2((L−1)/(|i−j|+1))⌉ to the XNOR map, following ALiBi, so integers stay integers. The full linear form (L−1)/(|i−j|+1) ("C-RPE") is rejected because it "would catastrophically overshadow the original spiking attention activations". — [arXiv 2501.16745](https://arxiv.org/abs/2501.16745); [DOI](https://doi.org/10.52202/085713-2601)
- Table 1 (forecasting, 3 seeds; averages of R2 / RSE over 4 datasets × 4 horizons):
  - Spikformer: conv-PE (original) .733 / .541; **ALiBi .413 / .909 (collapsed; Solar R2 = .080 at every horizon)**; RoPE .720 / .559; CPG-PE .744 / .519.
  - Spikformer-XNOR: conv-PE .733 / .542; Gray-PE .748 / .513; **Log-PE .750 / .509**.
  - QKFormer: original .695 / .565; CPG-PE .732 / .531; XNOR + Gray-PE .742 / .519; XNOR + Log-PE .746 / .514.
  - SDT-V1: original .721 / .557; Log-PE .734 / .538.
  - ANN Transformer + RoPE .756 / .507.
  - Look-backs are 12 (Metr-la, Pems-bay) and 168 (Solar, Electricity). The paper finds Log-PE better than Gray-PE on the 168-step datasets.
  — [arXiv 2501.16745](https://arxiv.org/abs/2501.16745)
- **Table S1 (dot product vs XNOR with an additive bias; prediction length 24; R2 / RSE)**:
  - Spikformer (dot product) + Log-PE: Metr-la .484 / .763; Electricity .710* / 1.03*, where * means "failure to converge".
  - Spikformer-XNOR + Log-PE: Metr-la .535 / .719; Electricity .974 / .300.
  - Spikformer-XNOR + C-RPE: Metr-la .158* / .967*.
  - QKFormer + Log-PE: Electricity .710* / 1.03*.
  - The authors conclude "the dot product, which does not use Hamming distance to measure relative distance, is not suitable for RPE methods". Finding (1) of the main text: Spikformers "directly equipped with RoPE or ALiBi exhibit poor performance across all benchmarks".
  — [arXiv 2501.16745](https://arxiv.org/abs/2501.16745)
- Images: QKFormer is insensitive to PE on image classification (Table S2: original 79.67 average, Gray-PE 2D 79.56). — [arXiv 2501.16745](https://arxiv.org/abs/2501.16745)

**Corroboration from SpikF (ICML 2025)**: the authors write that Spikformer with CPG-PE gives "only marginal performance improvements" and still trails SpikeTCN / SpikeRNN in RSE. — [SpikF PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)

### Inferences
- Our coupling score, q k^T·D^-0.5 + learned relative bias on binary q and k, is the "dot product + additive RPE" combination that collapsed or failed to converge in the NeurIPS 2025 study (ALiBi on Spikformer; Log-PE on dot-product Spikformer / QKFormer for Electricity). Ours is learned rather than fixed, so it may be trained to a harmless (near-zero) or dominating scale. Either way, the bias may not be carrying usable order information. A cheap diagnostic is to log the learned bias magnitude against the spread of q·k values and check whether the bias is near-constant along the diagonals. A cheap intervention is XNOR scoring plus Log-PE or Gray-PE (binary concatenation on q and k), which remains fully spike-driven.
- Measured spike-PE gains on forecasting are small: about +0.01–0.03 average R2 over Spikformer's default, and ±0 for QKFormer on images. Expect ETT effects within seed noise unless the task really rewards order, which SpikF's shuffle test suggests ETT barely does (about 0.8% MAE).
- Gray-PE resolves relative distances only at power-of-two steps and degrades for long L (the paper says so for L ≥ 10²). With our N = 84 / 180, Log-PE is the more appropriate of the two by the paper's own finding.

### Gaps
- Neither PE paper reports ETT. Neither tests a cross-attention setup like ours, where q and k come from different paths, so Gray-PE's symmetric concatenation to both q and k would need adapting.
- Not read: "Rethinking Attention Locality in Spiking Transformers" (arXiv 2608.08541) and "Spiking Sequence Machines and Transformers" (arXiv 2605.00662), both found in OpenAlex; venue and content unverified.

## Q4. Spiking attention with explicit temporal (across-time-step) interaction for sequences

### Takeaway
Two peer-reviewed plug-ins add interaction **across SNN time steps** to Spikformer-style attention:
- **TIM** (IJCAI 2024): a recurrent, convolution-filtered query, Q_TIM[t] = α·f(Q_TIM[t−1]) + (1−α)·Q[t].
- **STAtten** (CVPR 2025): block-wise attention over pairs of time steps, at the same O(T·N·D²) cost as Spikformer.

They give +1–3 points on neuromorphic and sequential-image tasks (SHD +1.2 for TIM). Both act on the SNN time-step axis, not the token axis. In our model the Neocortex output is copied to the 4 time steps and the Hippocampus repeats the same token 4 times, so they would mix near-identical copies unless re-targeted to the token axis.

### Cited Findings
**TIM, "An Efficient Temporal Interaction Module for Spiking Transformer" (Shen et al.; IJCAI 2024, DOI 10.24963/ijcai.2024/347; code BrainCog-X/Brain-Cog examples/TIM)**
- Mechanism (Eq. 7): Attention[t] = A_TIM[t] K[t]^T V[t] = α·f(Q_TIM[t−1]) K[t]^T V[t] + (1−α)·Q[t] K[t]^T V[t], where f is a 1-D convolution. This gives the query a leaky memory of past time steps. The paper says TIM adds few parameters. Default α = 0.5 per the paper's tuning (Table 5 sweeps α ∈ {0, 0.2, 0.4, 0.5, 0.6, 0.8}; every α > 0 beats α = 0). — [arXiv 2401.11687](https://arxiv.org/abs/2401.11687); [DOI](https://doi.org/10.24963/ijcai.2024/347)
- Results: SHD, Spikformer 85.1 → TIM 86.3 (Table 2). Ablation (Table 3): CIFAR10-DVS 78.5 → 81.6; N-Caltech101 77.7 → 79.0. Also UCF101-DVS 63.8 and HMDB51-DVS 58.6. — [arXiv 2401.11687](https://arxiv.org/abs/2401.11687)

**STAtten, "Spiking Transformer with Spatial-Temporal Attention" (Lee, Li, Kim, Xiao, Panda; CVPR 2025, DOI 10.1109/cvpr52734.2025.01302)**
- Mechanism (Eq. 8): STAtten(X[b]) = LIF(Q[b] K[b]^T V[b] · α), where block b groups B consecutive time steps (T = 4, B = 2 → pairs [0, 1] and [2, 3]). It is computed in the order Q(K^T V), with no softmax. Complexity is O(T·N·D²), the same as Spikformer and QKFormer (Table 1). The paper reports that full spatial-temporal attention costs much more memory and that spike similarity weakens with temporal distance, which motivates the blocks. — [arXiv 2409.19764](https://arxiv.org/abs/2409.19764); [DOI](https://doi.org/10.1109/cvpr52734.2025.01302)
- Sequential CIFAR (Table 2, s-CIFAR10 / s-CIFAR100): Spikformer 79.29 / 57.17 → STAtten 82.45 / 58.75; Spike-driven Transformer 80.32 / 61.08 → 83.41 / 64.30; SpikingReformer 79.25 / 57.48 → 81.84 / 58.30. — [arXiv 2409.19764](https://arxiv.org/abs/2409.19764)

**Recurrent mixers also count as cross-step interaction**: SpikeGPT's SRWKV (state A[t], B[t] with decay exp(W_d)), SpikingSSMs and LMUFormer (see Q2) all mix across sequence steps through a state. In those models the SNN step equals the token step. — [arXiv 2302.13939](https://arxiv.org/abs/2302.13939); [arXiv 2408.14909](https://arxiv.org/abs/2408.14909); [arXiv 2402.04882](https://arxiv.org/abs/2402.04882)

### Inferences
- TIM's recurrent query is the published analogue closest to our design idea, "the query carries history (Neocortex) while keys / values carry the current content". Its measured benefit is small (+1.2 on SHD over Spikformer). This suggests that giving only the query a memory yields modest gains, which matches our null results.
- Applied literally, STAtten's block-wise cross-time-step attention would mix our T_h = 4 repeated presentations, which carry no new order information. To be useful it would have to be re-indexed so that its "time blocks" are token blocks. That is essentially a local, windowed token attention.

### Gaps
- Neither TIM nor STAtten reports time-series forecasting. TIM's only sequence-audio result is SHD (a dataset our project dropped).
- I found no peer-reviewed spiking speech transformer with an explicit causal / relative-position mechanism in this pass. Spiking ASR transformers were not searched in depth.

## Q5. Which mechanisms have beaten Spikformer-style attention on sequence tasks under (near-)matched settings, and by how much

### Takeaway
Every documented win over Spikformer-style SSA on sequences is a few points or less, and none is on ETT with long look-back. The clearest same-network swaps are:
- LMU replacing SSA: +2.74 points on Speech Commands V2-35.
- SpikF's frequency selector replacing iSSA: about 1–3% MSE on ETTh1 / ETTm1 / Weather.
- STAtten: +1.6 to +3.2 on sequential CIFAR.
- TIM: +1.2 on SHD.
- Spike-native PE (CPG / Gray / Log): about +0.01–0.03 R2 on forecasting.

SpikeLM's +7.6 GLUE points over a transferred SpikingFormer comes from spike encoding, not from the mixer.

### Cited Findings
- **LMUFormer vs Spikformer\*** (same network, LMU replaced by SSA): Speech Commands V2-35, 96.12% vs 93.38% (+2.74). Against Spike-driven ViT\* 94.85% the gain is +1.27. Parameter matching not stated in what I read. — [arXiv 2402.04882](https://arxiv.org/abs/2402.04882)
- **SpikF SFS vs iSSA** (same SpikF pipeline, mixer swapped), MSE: ETTh1 0.440 vs 0.452 (−2.7%); ETTm1 0.388 vs 0.389; Weather 0.245 vs 0.247. The paper's summary figure is a "1.6% performance enhancement". SpikF vs iSpikformer (Solar RSE, numbers from Lv et al.): 0.189 / 0.346 / 0.444 / 0.514 vs 0.204 / 0.333 / 0.465 / 0.521, so SpikF is worse at H = 24. — [SpikF PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wu25m/wu25m.pdf)
- **STAtten vs Spikformer**: s-CIFAR10 +3.16, s-CIFAR100 +1.58. Plugged into SDT: +3.09 and +3.22. Same O(T·N·D²) complexity. — [arXiv 2409.19764](https://arxiv.org/abs/2409.19764)
- **TIM vs Spikformer**: SHD 85.1 → 86.3. — [arXiv 2401.11687](https://arxiv.org/abs/2401.11687)
- **Spike PE on Spikformer (forecasting average R2)**: no PE .719; original conv-PE .733; CPG-PE .744 (NeurIPS 2024); XNOR + Log-PE .750 (NeurIPS 2025); Spikformer + ALiBi .413 (collapse). — [arXiv 2405.14362](https://arxiv.org/abs/2405.14362); [arXiv 2501.16745](https://arxiv.org/abs/2501.16745)
- **TS-LIF neuron in iSpikformer** (attention unchanged): average R2 .775 → .783, RSE .476 → .441. — [arXiv 2503.05108](https://arxiv.org/abs/2503.05108)
- **SpikeLM vs SpikingFormer transferred to language** (same training scheme): GLUE 76.5 vs 68.9 (T = 4), 75.7 vs 67.1 (T = 1). The difference is elastic bi-spiking; SpikingFormer's query and attention weights were made non-spiking for the comparison. — [arXiv 2406.03287](https://arxiv.org/abs/2406.03287)
- **SSMs vs other spiking mixers on LRA (not vs Spikformer)**: SpikingSSM average 84.33 vs Spiking LMUFormer 60.20 vs Binary S4D 74.69. P-SpikeSSM Pathfinder 84.80 vs Spiking LMUFormer 72.68. — [arXiv 2408.14909](https://arxiv.org/abs/2408.14909); [arXiv 2406.02923](https://arxiv.org/abs/2406.02923)

### Inferences
Candidate summary for replacing the coupling attention (a synthesis of the findings above; "spike-driven" follows each paper's own claim):

| Candidate (venue) | How order is handled | Cost in N | Fully spike-driven? | ETT evidence | vs Spikformer-style | Code |
|---|---|---|---|---|---|---|
| XNOR-SSA + Log-PE / Gray-PE (NeurIPS 2025) | relative: integer log-distance bias / Gray-code concatenation on Q, K | O(N²) | yes (binary / integer) | none (Metr-la, Pems-bay, Solar, Electricity) | +0.017 R2 average; dot product + bias collapsed | SeqSNN |
| CPG-PE (NeurIPS 2024) | absolute binary oscillator codes, concatenated | unchanged | yes | none | +0.011 R2 vs conv-PE | SeqSNN |
| SpikF S-FFT selector (ICML 2025) | Fourier transform over patches (position-sensitive by construction) | about N log N (inferred) | claimed spiking; real-valued iFFT output into LIF | **yes**: ETTh2 .372 / .394, ETTm2 .281 / .320 (L = 96) | about 1–3% MSE vs iSSA on ETTh1 / ETTm1 / Weather | yes |
| SpikeGPT SRWKV (TMLR 2024) | causal recurrence, exponential decay | O(N) | no (exp / division in state) | none | not compared | yes |
| SpikingSSMs (AAAI 2025) | causal SSM convolution / recurrence + LIF | O(N log N) train, O(1) per step | partial | none | not compared | yes |
| P-SpikeSSM (ICLR 2025) | SSM-based stochastic spiking neuron | parallel, linear-type (not stated in what I read) | "Yes*" (GELU, no FP matmul) | none | not compared | yes |
| LMUFormer spiking (ICLR 2024) | LMU recurrence (Legendre memory) | O(N) | partial (not verified) | none | +2.74 on SC-35 vs SSA swap | yes |
| TIM (IJCAI 2024) | recurrent conv-filtered query across SNN steps | O(N²)-type attention | not stated | none | +1.2 on SHD | yes |
| STAtten (CVPR 2025) | block-wise cross-time-step Q(KᵀV) | O(T·N·D²) | yes (Spikformer-like) | none | +1.6 to +3.2 on s-CIFAR | yes |

- On credibility (peer-reviewed + code + reproducible protocol): the strongest for our purpose are SpikF (the only one with ETT, plus an iSSA ablation and a shuffle control), the NeurIPS 2025 XNOR + Log-PE (directly targets our failure mode, an additive bias on a binary dot product), and the SSM family (SpikingSSMs, P-SpikeSSM) for order-as-structure.
- The pattern across papers is consistent: order-carrying recurrence or Fourier structure beats content-only spiking attention by small margins, and published ETT order sensitivity is small (0.8% MAE in SpikF's shuffle test). A replacement for our coupling is unlikely to produce a large ETT gain. It should be judged with the same shuffle and memory-removal controls we already use, ideally at input 336 / 720, where order and memory have more room to matter.

### Gaps
- No paper gives a parameter-matched Spikformer-SSA vs order-aware spiking mixer comparison on ETT at look-back 336 / 720. No paper tests a cross-path (q from one spiking path, k / v from another) coupling like ours.
- Parameter counts for most of the "swap" ablations above (LMUFormer Spikformer\*, SpikF iSSA) were not checked.
- Fully spike-driven status of LMUFormer's LMU state, TIM's convolution-filtered query and SpikF's S-iFFT output is not verified beyond the papers' own labels.
