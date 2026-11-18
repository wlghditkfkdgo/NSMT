# How reliable are the accuracy and energy claims for spiking attention? Energy-accounting critiques, hardware studies, benchmarks, reproducibility problems, and checks for swapping one spiking attention for another

Scope note (written 2026-11-17). Every number below comes from a source opened in this session unless it is marked otherwise. Evidence tags:
- **[full text]**: read from the paper body, tables or appendix (PDF converted with `pdftotext`, or the arXiv HTML).
- **[abstract]**: from the abstract or metadata only (OpenAlex `abstract_inverted_index`, arXiv abstract page, proceedings page).
- **[GitHub issue]**: from the official repository's issue tracker, read through the GitHub REST API. These are user reports and maintainer replies, not peer-reviewed, and some are unresolved.
- **[arXiv-only]**: no peer-reviewed version found, so lower reliability.
- **[secondary]**: a search-engine summary or a third-party summary page. Treat it as a lead, not a fact.

Venues were checked against OpenAlex (DOI, volume, pages) and arXiv API `comment` / `journal_ref` fields. OpenReview was not reachable, so ICLR venues rest on arXiv comments or the OpenReview link in the official repository. Context: the target comparison is attention variants inside an SNN for ETT forecasting (N = 24–180 tokens, D = 64, T = 4 in the attention path). Energy is estimated with syops-style AC/MAC counts (0.9 pJ / 4.6 pJ, 45 nm). The current attention is binary q k^T (integer counts) + real-valued relative-position bias, × binary v, then LIF.

---

## Q1. What do energy-accounting critiques and hardware studies say, especially about attention?

### Takeaway
Every independent re-evaluation found (IEEE TETCI 2023, IEEE TCAD 2026, ICASSP 2024, CVPR 2024, NeurIPS 2025 benchmark, AICAS 2024) says the same thing. Counting ACs against MACs leaves out memory access and data movement, and also leaves out the cost of storing state across time steps. Once those are counted, SNNs beat an equally capable quantized ANN only at low T and very high sparsity. For spiking self-attention in particular, two studies that do count memory find that Spikformer-style attention uses **more** memory-access energy than INT8 ANN attention (T = 10), and that a full spiking Transformer uses more total energy than a quantized Transformer. Hardware papers (ISCA 2025, IEEE TC 2026) treat spike×spike attention as a distinct, harder workload: AND + popcount producing 6–10-bit scores, and two dynamic operands that must both be stored. They do not treat it as free masking.

### Cited Findings

**General SNN-vs-ANN energy critiques (peer-reviewed)**
- Dampfhoffer, Mesquida, Valentian, Anghel, "Are SNNs Really More Energy-Efficient Than ANNs? An In-Depth Hardware-Aware Study", *IEEE TETCI* vol. 7, no. 3, pp. 731–741 (online 2022, issue 2023), DOI 10.1109/TETCI.2022.3214509 **[abstract]**. Their critique: "Many studies do not consider memory accesses, which account for an important fraction of the energy consumption, use naive ANN hardware implementations, or lack generality". The simpler IF model is more energy-efficient than LIF and temporal continuous-synapse models. SNNs with IF "can compete with efficient ANN implementations when there is a very high spike sparsity, i.e. between 0.15 and 1.38 spikes per synapse per inference, depending on the ANN implementation". They propose hybrid ANN–SNN architectures, using event-based processing only in high-sparsity layers. The paper analyses general layers, not attention. — [OpenAlex record + abstract](https://api.openalex.org/works/doi:10.1109/tetci.2022.3214509); [DOI](https://doi.org/10.1109/TETCI.2022.3214509). The HAL full text ([HAL](https://hal-cea.archives-ouvertes.fr/cea-03852141v1)) was blocked by a bot wall, so the full text was not read.
- Yan, Bai, Tang, Wong, "Reconsidering the Energy Efficiency of Spiking Neural Networks Inference from Analytical Perspectives", now in *IEEE TCAD* (2026, early access), DOI 10.1109/TCAD.2026.3718799 **[abstract + full text of arXiv v6]**. Fair baseline: a rate-coded SNN with T steps is compared with a "capacity-matched" QNN of ⌈log2(T+1)⌉ bits. Headline: "under typical neuromorphic hardware conditions, SNNs with moderate time windows (T = 5) require an average spike rate (s_r) below 5.7% to outperform equivalent QNNs". — [OpenAlex/TCAD abstract](https://doi.org/10.1109/TCAD.2026.3718799); [arXiv HTML v6](https://arxiv.org/html/2409.08290)
  - Details from arXiv v6 **[full text]**:
    - Compute constants E_ACC = E_CMP = E_SUB = 0.05448 pJ (22 nm Design Compiler synthesis).
    - Sparse data-movement cost 3.0 pJ/bit/hop (normalised from Loihi 1); dense cost 0.25 pJ/bit/hop (in-house 22 nm NoC).
    - Break-even spike rate ≈ 0.297 at T = 1 and ≈ 0.082 at T = 3 (8-bit weights, N_src = 4096).
    - Trace-driven VGG16 (Table II), SNN/QNN energy ratio: 0.987 on CIFAR-10 at T = 3; **1.406** at T = 5; **1.262** on CIFAR-100 at T = 3.
    - Transformer workloads (Sec. IV-D): GLUE at T = 3; spiking Llama-2-7B at T = 15 has ratio 3.793 (SNN costs more). The paper says the ratio roughly follows T/⌈log2(T+1)⌉ and is dominated by dense data movement.
    - The paper does not analyse attention mechanisms specifically.
  - **Version caveat:** the headline numbers changed across arXiv versions. v1 said VGG16 at T = 6 needs neuron sparsity > 93%, and that the optimized SNN used 69% of the energy of an optimized ANN on spatial-dataflow hardware, at 94.18% accuracy ([arXiv v1](https://arxiv.org/abs/2409.08290v1)). A search summary of a later version quoted 85% / 78% instead ([search-indexed copy](https://arxiv.org/html/2409.08290v1), **[secondary]**, version unclear). Cite the TCAD version.
- Bhattacharjee, Yin, Moitra, Panda, "Are SNNs Truly Energy-efficient? — A Hardware Perspective", *ICASSP 2024*, pp. 13311–13315, DOI 10.1109/ICASSP48485.2024.10448269 **[abstract]**. They say SNN energy claims "rely on certain impractical estimation metrics". Using SATA (sparsity-aware systolic array) and SpikeSim (IMC crossbars), "the actual energy-efficiency improvements of recent SNN algorithmic works differ significantly from their estimated values due to various hardware bottlenecks". The named bottlenecks are "repeated computations & data movements over timesteps, neuronal module overhead and vulnerability of SNNs towards crossbar non-idealities". The abstract gives no numbers. — [OpenAlex](https://doi.org/10.1109/ICASSP48485.2024.10448269); [arXiv](https://arxiv.org/abs/2309.03388)
- Shen, Zhao, Li, Li, Zeng, "Are Conventional SNNs Really Efficient? A Perspective from Network Quantization", *CVPR 2024*, pp. 27528–27537 **[abstract]**. An SNN with T steps is comparable to a T-bit-activation quantized ANN. They propose a "Bit Budget" in place of SynOps-based comparisons, and argue that SNN-vs-ANN comparisons ignore progress in ANN efficiency. — [CVF open access](https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Are_Conventional_SNNs_Really_Efficient_A_Perspective_from_Network_Quantization_CVPR_2024_paper.html)
- Lemaire et al., "An Analytical Estimation of Spiking Neural Networks Energy Efficiency", ICONIP 2022, LNCS 13623, pp. 574–587 — venue verified only, content **not read**. — [DOI](https://doi.org/10.1007/978-3-031-30105-6_48)

**Where 0.9 pJ / 4.6 pJ come from (primary source, decoded)**
- Horowitz, ISSCC 2014, Fig. 1.1.9 ("Rough energy costs for various operations in 45nm 0.9V") **[full text, decoded from the PDF's shifted font]**:
  - Integer add: 8-bit 0.03 pJ; 32-bit 0.1 pJ.
  - FP add: 16-bit 0.4 pJ; **32-bit 0.9 pJ**.
  - Integer mult: 8-bit 0.2 pJ; 32-bit 3.1 pJ.
  - FP mult: 16-bit 1.1 pJ; **32-bit 3.7 pJ**.
  - 64-bit cache read: 8 KB 10 pJ; 32 KB 20 pJ; 1 MB 100 pJ.
  - DRAM: 1.3–2.6 nJ.
  - So E_AC = 0.9 pJ is an **FP32 add** and E_MAC = 4.6 pJ = 3.7 + 0.9 is an **FP32 multiply-add**.
  - — [Horowitz PDF](https://www.gwern.net/doc/cs/hardware/2014-horowitz-2.pdf); [DOI](https://doi.org/10.1109/ISSCC.2014.6757323)
- Pedram et al. (Horowitz group) Table I, 45 nm: 16-bit add 0.18 pJ, multiply 0.62 pJ; 32K-word SRAM read 11 pJ; DRAM 640 pJ (16-bit columns). — [ar5iv 1602.04183](https://ar5iv.labs.arxiv.org/html/1602.04183)

**Critiques that are specifically about spiking attention / spiking Transformers**
- Song, Katti, Simeone, Rajendran, "Stochastic Spiking Attention", *IEEE AICAS 2024*, pp. 31–35, DOI 10.1109/AICAS59952.2024.10595893 **[full text]**.
  - Setup: ViT-Small (6 layers, 8 heads), INT8 parameters. "Spikformer SNN – an SNN with its dot-product attention implemented with integer multipliers".
  - Energy counts compute **and** memory read/write at 45 nm, assuming all data sit in on-chip SRAM.
  - Table II, one attention block, T = 10 (processing / memory / total, µJ):
    - ANN (INT8) attention: 7.77 / 89.96 / 97.73.
    - **Spikformer attention: 6.20 / 102.85 / 109.05**.
    - Proposed stochastic attention: 1.23 / 52.80 / 54.03.
  - Their wording: "Since memory access occurs at each time step, SNNs do not exhibit any advantage over 8-bit quantized ANNs for memory accesses. We observed that in 10 time steps, Spikformer attention consumes more energy for memory access than INT8-ANN attention."
  - Accuracy (Table I, CIFAR-10): ANN 83.66; Spikformer T = 4/8/10: 83.32 / 83.4 / 83.41.
  - — [arXiv 2402.09109](https://arxiv.org/abs/2402.09109)
- Shen et al., "STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking", *NeurIPS 2025* (arXiv comment "Accepted by NeurIPS 2025"; proceedings DOI 10.52202/085713-5334; the track, main or Datasets & Benchmarks, was not verified) **[full text, partial]**.
  - Energy model:
    - Quantized ANN compute = B·R_b·E_AC·F; spiking compute = T·R_s·E_AC·F.
    - Memory per access: 32 bits (float), B bits (quantized), **32T bits for spiking**, "because membrane potentials are kept across timesteps".
    - Constants E_MAC = 4.6 pJ, E_AC = 0.9 pJ, E_Mem = 3.12 pJ per bit (Table 9 caption as extracted; not cross-checked).
  - Table 10 (Transformer-8-512), compute / memory / total:
    - Float: 41.77 / 1.39 / 43.16 mJ.
    - Quantized: 16.34 / 0.17 / **16.51 mJ**.
    - **Spiking: 11.57 / 5.59 / 17.16 mJ**.
    - The text did not state the T and B used for this table.
  - Their wording: the omission of memory "in existing energy models can result in an overestimation of the energy efficiency of SNNs", and quantized ANNs "may be comparably or more energy-efficient".
  - — [arXiv abs](https://arxiv.org/abs/2505.11151); [HTML v2](https://arxiv.org/html/2505.11151v2)

**Hardware studies of spiking-Transformer accelerators**
- Xu, Yin, Iyer, Li, "Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning", *ISCA 2025*, pp. 944–957, DOI 10.1145/3695053.3731063 **[full text via arXiv HTML]**.
  - Workload share: attention + MLP blocks are 66.5–91.0% of total FLOPs (Sec. 2.2); "over 30% of computational overhead comes from SSA layers in typical spiking transformers" (Sec. 5.1).
  - The attention core does "AND" between Q and K, then accumulates. The attention score S "has a higher bit width, ranging from 6 to 10 model-dependent bits", so the design keeps S stationary to limit moving multi-bit scores (Sec. 5.5). Binary V "selects the right S data to be accumulated".
  - Error-constrained pruning of Q/K/V (Model 1, CIFAR-10) cuts attention-map compute by 62.7% and memory access by 38.1%.
  - Accuracy effect of that pruning: +1.24 (CIFAR-10), +0.53 (CIFAR-100), −0.13 (ImageNet-100), +0.7 (DVS-Gesture).
  - Results: average 5.91× speedup and 6.11× energy reduction against the PTB systolic SNN accelerator (both 28 nm, same PE count), and 299× speedup against a Jetson Nano.
  - Models evaluated: T = 10/8/4/20/8; N = 64/64/196/64/256; D = 384/384/128/128/384.
  - — [arXiv HTML](https://arxiv.org/html/2505.12281v1); [DOI](https://doi.org/10.1145/3695053.3731063)
- "FireFly-T: High-Throughput Sparsity Exploitation for Spiking Transformer Acceleration With Dual-Engine Overlay Architecture", *IEEE Transactions on Computers* vol. 75, no. 6, pp. 2185–2199 (2026), DOI 10.1109/TC.2026.3672901 **[abstract]**. "existing hardware accelerators lack support for spiking attention". The design adds a separate binary engine for attention and optimises "the core AND-PopCount operation in spiking attention" (LUT6 on Xilinx FPGAs). Results: 1.39× and 2.40× higher energy efficiency than FireFly v2 and SpikeTA. — [OpenAlex/DOI](https://doi.org/10.1109/TC.2026.3672901); [arXiv](https://arxiv.org/abs/2505.12771)
- "An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer", IEEE APACE 2024, pp. 250–253 (a minor venue) **[secondary]**. An FPGA design targeting the spike-driven self-attention layer, which "earlier SNN accelerators largely ignored"; evaluated on CIFAR-10 only. — [arXiv 2501.07825](https://arxiv.org/abs/2501.07825); [DOI](https://doi.org/10.1109/APACE62360.2024.10877394)
- ASTER (arXiv 2511.06770, **[arXiv-only][secondary]**): "existing neuromorphic platforms and accelerators lack direct hardware support for the sparse mask-add operations central to spiking self-attention" (search summary only; not opened). — [arXiv HTML](https://arxiv.org/html/2511.06770v1)
- Spiking Token Mixer (STMixer), Deng, Wu, Du, Gu, *NeurIPS 2024*, pp. 128825–128846, DOI 10.52202/079017-4092 **[abstract/secondary]**. Event-driven asynchronous chips "do not support some of the proposed structures" in spiking Transformers. Spikformer's SSA multiplies two spike matrices, which asynchronous hardware cannot run. STMixer uses only conv, FC and residual operations, and replaces the Q·K attention map with a learnable token-mixing weight matrix (weight × spike). It reports parity with spiking Transformers in synchronous settings at very low timesteps. The full PDF was too large to fetch, so its numbers were **not verified**. — [NeurIPS poster page](https://neurips.cc/virtual/2024/poster/93999); [DOI](https://doi.org/10.52202/079017-4092)

### Inferences
- The syops convention prices every AC as an FP32 add (0.9 pJ) and charges nothing for memory. For spiking attention it therefore errs in **two opposite directions**:
  - It **over-charges** the integer popcounts / accumulates in q k^T. Horowitz gives 0.03 pJ for an 8-bit integer add and 0.1 pJ for a 32-bit one.
  - It **under-charges** reading Q/K/V spikes and membrane states. One 64-bit read from an 8 KB cache (10 pJ) costs about 11 FP32 adds.
  - The two measured attention studies (AICAS 2024, STEP Table 10) both find that memory dominates. So the net error most likely flatters the SNN. This is an inference from the two studies, not a measurement of our model.
- With T = 4, the capacity-matched baseline of the TCAD 2026 paper is a 3-bit quantized ANN (⌈log2 5⌉ = 3). By that paper's analysis, an attention path claiming an energy advantage over a float ANN says little about an advantage over a 3-bit ANN.
- None of the hardware papers treats spike×spike attention as "no energy". All of them build dedicated logic (AND/popcount, multi-bit score accumulators, buffers for two dynamic operands). An accounting that charges the Q⊙K mask or the AND as zero therefore has no hardware study behind it.

### Gaps
- Dampfhoffer et al. and Lemaire et al. full texts were not read (HAL bot wall; no abstract in OpenAlex for Lemaire). Their exact memory-energy tables are unverified.
- No study was found that measured a spiking self-attention on real neuromorphic silicon (Loihi 2, Speck, Tianjic) against an INT ANN attention. All attention-specific numbers above are analytical (AICAS 2024, STEP) or from accelerators that are synthesized or run on FPGA.
- No energy re-evaluation of spiking attention for **time-series** forecasting (small N, small D) was found. All re-evaluations use vision or NLP shapes.
- The STEP Table 10 settings (T, B, R_s, R_b) were not in the extracted text.

---

## Q2. Do AC-only claims hold for each attention form, and which papers count energy incorrectly or inconsistently?

### Takeaway
"AC-only" holds strictly only when one operand of every product is binary **and** the other is a stored weight. Spike×spike products (q k^T) are AND + popcount and produce multi-bit integers. Integer × binary products (attention map × V) are integer select-accumulates. Real-valued additive terms (scale, bias, BN that is not folded, SEW residual sums) make the inputs non-binary, and NeuroBench counts those operations as MACs. The reference papers handle this inconsistently. Spikformer leaves the scale factor out. SDT V1 counts its mask (Hadamard) as zero energy. Models with SEW residuals feed integer "spikes" into conv/linear layers yet price them as ACs, and an SDT maintainer says this is done "so that the energy values don't look too high". QKFormer's attention input is not binary in the released code. Each paper uses its own per-operation firing-rate convention.

### Cited Findings

**Per-form accounting in the original papers**
- **Spikformer (SSA), ICLR 2023** (OpenReview id frE4fUwz_h, linked from the official repo; arXiv v2 read) **[full text]**.
  - SSA'(Q,K,V) = SN(Q K^T V · s), and they "add a scaling factor s to control the large value of the matrix multiplication result. s does not affect the property of SSA."
  - Binary Q·K "degrades the matrix dot-product calculation to logical AND operation and summation operation".
  - Energy (App. C.2, Eq. 21–22): SOPs(l) = fr × T × FLOPs(l). E = 4.6 pJ × FLOPs(first conv) + 0.9 pJ × (SOPs of conv + FC + SSA). There is no term for s, for BN, for the AND, or for memory.
  - Their VSA-vs-SSA comparison prices VSA as FLOPs × 4.6 pJ and SSA as SOPs × 0.9 pJ. Fig. 1: 5.5×10⁶ SSA ops vs 77×10⁶ VSA ops; VSA 354.2 µJ (one head, one step, 8-512 on ImageNet).
  - — [arXiv 2209.15425](https://arxiv.org/abs/2209.15425); [OpenReview](https://openreview.net/forum?id=frE4fUwz_h)
- **Spike-driven Transformer V1 (SDSA), NeurIPS 2023** **[full text]**.
  - Claim: "the Hadamard product between spikes is equivalent to the element-wise mask" and "can be considered as not consuming energy". The column sum "consumes almost no energy since the matrix to be summed … is very sparse (typically, the ratio of non-zero elements is less than 0.02)".
  - VSA needs 2N²D MACs while SDSA needs 0.02ND ACs.
  - Table 4: ANN self-attention energy E1 vs SDSA energy E2 (E2 includes Q/K/V generation):
    - 8-384: 6.7e8 vs 1.6e7 pJ (42.6×).
    - 8-512: 1.2e9 vs 2.1e7 (57.2×).
    - 8-768: 2.7e9 vs 3.1e7 (**87.2×**).
  - SDT V1 itself criticises Spikformer-type SSA: it "yield[s] large integers in the output (thus requiring an additional scale multiplication for normalization…)". It also criticises SEW residuals: their "shortcut is spike addition and thus outputs multi-bit (integer) spikes. This shortcut can satisfy event-driven, but introduces integer multiplication".
  - — [NeurIPS 2023 PDF](https://papers.neurips.cc/paper_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf)
- **SDT maintainer on SEW-based models** (BICLab/Spike-Driven-Transformer #23, reply by jkhu29, 2025-04-07, Chinese, translated) **[GitHub issue]**. SEW residuals make Spikformer-type models not strictly spike-driven, because spike + spike outputs are not just 0/1. "This means the energy of their conv and linear layers cannot be replaced by additions, i.e., it should not be computed with 0.9 pJ/FLOPs. However, so that the energy values don't look too high, models using SEW residuals generally ignore this. For a strict energy comparison, take this into account." — [issue #23](https://github.com/BICLab/Spike-Driven-Transformer/issues/23)
- **SDT V1 attention-energy convention** (#24, maintainer reply 2025-06-12) **[GitHub issue]**:
  - "In SDSA-1, the energy of attention function f(Q,K,V) is 0.9pJ · R̂·TND", where R̂ is "the firing rate of Q·K".
  - In the released code, "Q plays the role of V in the traditional sense", which is why a user measured Q firing > 10× K and V. The paper's naming does not match the code.
  - — [issue #24](https://github.com/BICLab/Spike-Driven-Transformer/issues/24)
- **SDT V2 (Meta-SpikeFormer), ICLR 2024** (arXiv comment "Accepted by ICLR2024") **[GitHub issue]**. A user notes that V2 replaced V1's Hadamard product with matrix multiplication, so "the second matrix multiplication is integer × binary, not binary".
  - Maintainer reply: "When one side of the GEMM is a 0/1 matrix, it simplifies to addressing and addition; only the addition becomes integer addition". If the other side is float, it becomes "addressing and floating-point addition; floating-point addition is not easy to implement on neuromorphic hardware" (translated).
  - A second thread (#4) asks whether RepConv's two consecutive convs (the second takes non-binary input) are still an SNN. The maintainer says they fuse by re-parameterization at inference.
  - — [V2 issue #11](https://github.com/BICLab/Spike-Driven-Transformer-V2/issues/11); [V2 issue #4](https://github.com/BICLab/Spike-Driven-Transformer-V2/issues/4); [arXiv 2404.03663](https://arxiv.org/abs/2404.03663)
- **SDT V3 (E-SpikeFormer), IEEE TPAMI** 47(4):2973–2990 (2025), DOI 10.1109/TPAMI.2025.3530246 **[full text, partial]**.
  - Attention: U' = Linear_{1/γ}(SN(Q_S K_S^T V_S · scale)); "the scale factor can be folded into the SN threshold".
  - Training uses integer activations at T = 1. Inference expands each step into D virtual spike steps, and the paper proves the integer equals the IF-SR spike count.
  - Mean spikes per neuron: 0.96 (vanilla) vs 0.39 (SFA).
  - The paper does not state that an integer k is billed as k ACs (this is in the supplement, not read). The text gives 3.7 mJ for the 10M ablation while Table V gives 3.0 mJ, an internal inconsistency.
  - Issue #7 asks whether MultiSpike's division by `Norm` still allows AC-only. It was closed with no reply.
  - Issue #1: the maintainers say "spike-driven inference needs to be implemented on neuromorphic chips… [it] does not need to be reflected in the code".
  - — [arXiv HTML 2411.16061](https://arxiv.org/html/2411.16061); [V3 issue #7](https://github.com/BICLab/Spike-Driven-Transformer-V3/issues/7); [V3 issue #1](https://github.com/BICLab/Spike-Driven-Transformer-V3/issues/1)
- **QKFormer (QKTA), NeurIPS 2024 Spotlight**, DOI 10.52202/079017-0416 **[full text]**.
  - QKTA: A_t = SN(Σ_i Q_{i,j}), X' = A_t ⊗ K ("equivalent to the mask operation"). Claimed complexity O(D) time, O(N) space.
  - Energy (Eq. 23): E = E_AC × (ΣSOP_Conv + ΣSOP_QKTA + ΣSOP_SSA) + E_MAC × FLOP_Conv1.
  - Figure 1 says "The inputs are binary spikes". Maintainer reply in issue #5: in the code, the input to QK attention is `x + x_feat` (SEW "activation-before-addition"), i.e., not spikes. A user who added an LIF there lost about 2%.
  - QKFormer's own Table 2 shows hierarchical QKFormer using **more** energy than size-matched Spikformer: HST-10-768 at T = 4 is 38.91 mJ vs Spikformer-8-768 at 21.48 mJ; HST-10-384 is 15.13 mJ vs Spikformer-8-384 at 7.73 mJ.
  - — [arXiv HTML](https://arxiv.org/html/2403.16552v2); [issue #5](https://github.com/zhouchenlin2096/QKFormer/issues/5)
- **Residual type changes the energy estimate by about 3×** (Max-Former, "Spiking Neural Networks Need High-Frequency Information", *NeurIPS 2025*, DOI 10.52202/085713-3455) **[full text]**. Table 3 on ImageNet compares QKFormer (pre-spike / SEW shortcut) with MS-QKFormer (membrane shortcut), same parameters:
  - HST-10-384 at T = 4: 15.13 mJ / 78.80% vs **5.52 mJ / 76.48%**.
  - HST-10-768 at T = 1: 8.52 mJ / 81.69% vs 6.79 mJ / 77.78%.
  - The QKFormer rows equal QKFormer's own published numbers. Whether they were re-run is not stated.
  - Energy method: SOPs = fr × T × FLOPs; first conv priced as MAC; BN excluded as "folded"; max-pool and depthwise conv treatment not stated.
  - — [arXiv HTML v6](https://arxiv.org/html/2505.18608v6); [DOI](https://doi.org/10.52202/085713-3455)
- **STAtten, CVPR 2025**, pp. 13948–13958, DOI 10.1109/CVPR52734.2025.01302 **[full text]**.
  - Attention energy is E_AC·T·N·D²·(S_Q+S_K+S_V). The paper does not say whether q k^T is counted as AC or how the scale α is treated.
  - Table 6 (ImageNet): SDT 12.42 mJ vs STAtten+SDT 12.36 mJ; SDT-V2 52.40 vs 52.38; Spikformer 21.48; ViT-B/16 254.84 mJ.
  - The paper claims "no additional energy overhead". STAtten is listed as O(TND²), the same as SDT-V2, while SDT is O(TND). Why its energy tracks SDT is not explained.
  - — [arXiv HTML](https://arxiv.org/html/2409.19764); [DOI](https://doi.org/10.1109/CVPR52734.2025.01302)
- **SpikingResformer, CVPR 2024**, pp. 5610–5619 **[GitHub issue]**. The SOP counter's `matmul` takes a flag saying which operand is spikes ('r' or 'both'). The author's explanation: when the right matrix is spikes, the left is treated "like a weight matrix… the SOP calculation then resembles a linear layer's". Follow-up questions went unanswered: how the 'both' (spike×spike) case is billed, why the stem is omitted, and whether "SOPs = FLOPs × potential value" makes sense. — [issue #7](https://github.com/xyshi2000/SpikingResformer/issues/7)
- **NeuroBench** (Yik et al., *Nature Communications* 16:1545, 2025, DOI 10.1038/s41467-025-56739-4) **[docs]**. "Synaptic operations with non-binary activation are considered multiply-accumulates (MACs), while those with binary activation are considered accumulates (ACs)". Effective counts skip zero activations and zero weights. "Pooling and normalization are not considered in any metrics currently". — [NeuroBench docs](https://neurobench.readthedocs.io/en/latest/metrics/workload_metrics/synaptic_operations.html); [DOI](https://doi.org/10.1038/s41467-025-56739-4)
- **The scale factor matters for accuracy, so it is not a removable detail.** QKFormer Table 5: Spikformer SSA on CIFAR-100 without scaling gives 76.95% vs 78.21% with scaling. — [QKFormer HTML](https://arxiv.org/html/2403.16552v2)

### Inferences
- **Three incompatible conventions exist for spike×spike attention**, and the choice can change the attention energy by an order of magnitude at low firing rates:
  - Spikformer: fr(input) × FLOPs.
  - SDT V1: the firing rate of Q⊙K (and the mask is free).
  - NeuroBench: count only where both operands are non-zero; any non-binary operand makes the operation a MAC.
  - A swap study must fix one convention and state it.
- **For our attention form**, read under the papers' own logic:
  - q k^T on binary q, k is AND + popcount, giving integers 0…d_head. Bishop needs 6–10-bit accumulators for these scores.
  - Adding a real-valued relative-position bias B makes the score matrix real-valued. That is a dense N²-per-head-per-step FP add which is **not gated by spikes**, so syops-style SOP counting would miss it unless added explicitly.
  - (S + B)·v with binary v equals S·v + B·v:
    - S·v is an integer select-accumulate (AC-like on integers, per the SDT V2 maintainer).
    - B·v is a learned weight × spike product. That is the same operation STMixer uses as its "event-driven friendly" token mixer, and the only part an asynchronous chip supports natively (per STMixer's argument).
  - Any scale multiply on the integer map is a MAC unless folded into the LIF threshold, as SDT V3 states is possible. With B added before the LIF, the scale must also be folded into B.
- Reported energy numbers across papers are not comparable. QKFormer (38.91 mJ) vs Spikformer (21.48 mJ) at similar size, SDT (12.42 mJ) vs SDT-V2 (52.40 mJ), and SEW vs MS QKFormer (15.13 vs 5.52 mJ) show that architecture and residual choices move estimates far more than the attention swap itself (STAtten: 12.42 → 12.36 mJ).

### Gaps
- No peer-reviewed paper was found that re-computes the published Spikformer / SDT / QKFormer / STAtten energies under one consistent convention. The critiques of specific papers come from GitHub maintainers (SDT on SEW) and from hardware or benchmark papers, not from a dedicated erratum.
- The DSSA (SpikingResformer) and STMixer energy formulas were not read in full.
- No source states whether the 0.9 pJ AC should be replaced by an integer-add cost for popcount-style attention. This is our inference from Horowitz's table.

---

## Q3. Surveys and benchmarks that compare attention variants under one training recipe: which differences survive?

### Takeaway
No peer-reviewed survey (TPAMI, TNNLS, Neural Networks, Nature-family) was found that compares spiking attention variants under one matched recipe. The only matched-recipe benchmark is STEP (NeurIPS 2025). Under one recipe, most reported gaps shrink to ≤ 1 point on CIFAR. Replacing Q and K with **frozen random projections** costs < 0.35 points, which says attention contributes little in those models. The conv front-end and the residual type dominate. Within-paper ablations (Spikformer, SDT, QKFormer, STAtten) are single-run, with attention-swap effects of 0.03–2 points.

### Cited Findings
- **STEP reproductions, Table 1** (reproduced vs paper; batch 128, T = 4, 400 epochs unless noted) **[full text]**:
  - Spikformer: CIFAR-10 95.12 (95.51); CIFAR-100 77.37 (78.21).
  - SDT: 95.77 (95.60); 78.29 (78.40).
  - QKFormer: 96.24 (96.18); 79.72 (81.15).
  - Spikingformer: 95.53 (95.81); 79.12 (79.21).
  - Spikformer + SEMM: 94.98 (95.78); 77.59 (79.04).
  - Spiking Wavelet: 95.31 (96.10); 76.99 (79.30).
  - SGLFormer (batch 16): 95.88 (96.76); 80.61 (82.26).
  - **SpikingResformer end-to-end: 95.69 (97.40); 79.45 (85.98)**. The gap is attributed to the original's transfer-learning setup.
  - ImageNet-1K under unified scripts (Table 2): QKFormer 73.88 (compact variant, 200 epochs), Spikformer 73.69 (300 epochs), SDT 71.96 (300 epochs).
  - — [STEP HTML](https://arxiv.org/html/2505.11151v2)
- **STEP randomized attention, Table 8** (Q and K frozen as random projections, only V trained; CIFAR-10):
  - Spikformer 95.12 → 94.96.
  - SDT 95.77 → 95.45.
  - Spikformer+SEMM 94.98 → 95.57.
  - ViT (ANN) 90.89 → 88.46.
  - "Spiking Transformers maintain performance under randomized attention (drop < 0.35%)", which the authors read as attention contributing little to spatial feature extraction. — [STEP HTML](https://arxiv.org/html/2505.11151v2)
- **STEP attention swap to SDSA-v3, Table 7** (4-conv SPS, CIFAR-10): Spikformer 95.57, SDT 96.38, SEMM 95.83. Shrinking the conv front-end collapses accuracy far more:
  - 1-conv SPS with original attention (Table 8): Spikformer 78.21, SDT 77.34, SEMM 89.24.
  - With SDSA-v3: Spikformer 89.97, SDT 87.33, SEMM 84.95.
  - — [STEP HTML](https://arxiv.org/html/2505.11151v2)
- **STEP on temporal / sequential tasks, Table 5**: Spikformer sMNIST 98.84, psMNIST 97.97, sCIFAR 84.26, while a non-spiking ViT+SPS reaches 85.62 on sCIFAR. The authors: "spike-based attention mechanisms are more suited for spatial rather than temporal modeling". Their overall conclusion: current spiking Transformers "rely heavily on convolutional frontends and lack strong temporal modeling". — [STEP abs](https://arxiv.org/abs/2505.11151)
- **Within-paper matched ablations (single runs)**:
  - Spikformer Table 1 (same framework, attention swapped; CIFAR10-DVS / CIFAR-10 / CIFAR-100): A_I 79.40 / 93.96 / 76.94; A_LeakyReLU 79.80 / 93.85 / 76.73; A_ReLU 79.40 / 94.34 / 77.00; A_softmax 80.00 / 94.97 / 77.92; **A_SSA 80.90 / 95.19 / 77.86**. Spread ≤ 1.5 points; SSA ≈ softmax on CIFAR-100. — [Spikformer arXiv](https://arxiv.org/abs/2209.15425)
  - SDT V1 Table 6 (Spiking Transformer-2-512, CIFAR-10 / 100): Spikformer baseline 93.12 / 73.17; +SDSA 93.09 (−0.03) / 72.83 (−0.34); **+MS shortcut 93.93 (+0.81) / 74.63 (+1.46)**; full model 93.82 / 74.41. The residual change moves accuracy more than the attention change. — [SDT PDF](https://papers.neurips.cc/paper_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf)
  - QKFormer Table 6 (CIFAR-100 / CIFAR10-DVS): QKTA+SSA 81.15 / 84.00; SSA only 81.23 / 84.10; **QKTA only 79.09 / 80.70**. The paper's own linear-attention form is about 2 points worse alone. — [QKFormer HTML](https://arxiv.org/html/2403.16552v2)
  - STAtten Tables 3–5: plug-in gains of +0.23 to +1.80 points. Examples: CIFAR-10 QKFormer +0.23, SDT +0.43; CIFAR-100 SDT +1.45; ImageNet SDT-8-768 +1.79 at 224 and +1.32 at 288; CIFAR10-DVS +1.0 to +1.8. Single runs. SDT-V2 baselines are the authors' re-implementations. For dynamic datasets an extra pooling branch is added, so it is not a pure swap. — [STAtten HTML](https://arxiv.org/html/2409.19764)
- **Surveys found**:
  - "Toward Large-scale Spiking Neural Networks: A Comprehensive Survey and Future Directions" (arXiv 2409.02111, **[arXiv-only]**; no journal_ref). It compiles ImageNet results as reported, not re-run. — [arXiv](https://arxiv.org/html/2409.02111v1)
  - A short IEEE IPAS 2025 review, "A Review On Fusion Of Spiking Neural Networks And Transformers" (not read). — [DOI](https://doi.org/10.1109/IPAS63548.2025.10924503)

### Inferences
- In models with a strong conv tokenizer, swapping one spiking attention for another moves accuracy by about the same amount as freezing attention at random (STEP Table 8). A swap study without a "random / frozen attention" or "no attention" control cannot show that the attention form matters.
- The residual design (SEW vs MS) and the front-end are larger levers than the attention form, for both accuracy (SDT Table 6, Max-Former Table 3, STEP Tables 7–8) and estimated energy (Max-Former Table 3). The swap must hold these fixed.
- No benchmark covers spiking attention on time-series **forecasting** under a matched recipe. STEP's sequential tasks are classification (sMNIST / sCIFAR) and show no spiking-attention advantage there.

### Gaps
- No TPAMI / TNNLS / Neural Networks / Nature-family survey with matched-recipe attention comparisons was found in 2024–2026 searches. Absence is not proof; a targeted search in ACM Computing Surveys was not done.
- The STEP checklist reportedly answers "No" on error bars, yet Table 12 reports std and confidence intervals (see Q4). The number of runs behind Table 12 is unknown.

---

## Q4. Reproducibility: recipe confounds, seed variance, errata, failed reproductions

### Takeaway
None of the reference attention papers (Spikformer, SDT V1/V3, QKFormer, STAtten, Max-Former) reports seeds, standard deviations or confidence intervals for its main tables. Recipes differ in epochs (200–1000), effective batch size (gradient accumulation), test resolution (224 / 288 / 384), T (1 vs 4), pre-training or transfer, and distillation. Public issue trackers contain several failed reproductions. They also contain one serious **unconfirmed code defect** in QKFormer's attention: membrane state is mixed across samples in a batch, and released-checkpoint accuracy reportedly drops 4.5–5.6 points once it is fixed. The QKFormer authors have not answered in that issue.

### Cited Findings
- **QKFormer attention batch-mixing defect** (zhouchenlin2096/QKFormer #14, opened 2025-06-27, still open; #17, 2026-02-04, reports the same defect independently) **[GitHub issue]**.
  - Report: "the last lif (proj_lif) in Spiking_Self_Attention get input with shape [T×B, C, N] rather than [T, B, C, N]", so the LIF integrates across samples.
  - Released-checkpoint test with the given `test.py`: HST-10-384 gives 78.732% at the default batch 32. Fixed by batch size 1: 74.202% (−4.530). Fixed in code: 74.196% (−4.536).
  - HST-10-768 at 224: 84.248% → 78.698% (−5.55).
  - #17: "the original operation leaks membrane potential; membrane potentials of different samples in the same batch get mixed" (translated).
  - A third party (bic-L, the Max-Former author) "personally agree[s] with the original post". They reproduced QKFormer with a membrane shortcut after correcting it, and say QKFormer still beats the earlier baseline.
  - The QKFormer authors had not responded in the thread at the time of reading. The defect is present in all task folders (cifar10, cifar100, cifar10-dvs, dvs128, imagenet).
  - — [issue #14](https://github.com/zhouchenlin2096/QKFormer/issues/14); [issue #17](https://github.com/zhouchenlin2096/QKFormer/issues/17)
- **QKFormer training-recipe reproductions** **[GitHub issue]**:
  - #9: ImageNet 10-384 trained from scratch on 32× RTX 4090 reached 70.4% (paper 78.80%); released weights did reproduce. The author answered that results use direct training on 8× V100 and posted the original logs.
  - #6: a user got 68% on ImageNet. The fix was `--accum_iter`: the default of 3 "can not produce the experiment results"; 2 worked on 8× A100. The author reports a 0.6-point drop when training on Ascend 910C and calls that normal. Another user reports CIFAR10-DVS at T = 10 about 78% (paper result at T = 16).
  - — [issue #9](https://github.com/zhouchenlin2096/QKFormer/issues/9); [issue #6](https://github.com/zhouchenlin2096/QKFormer/issues/6)
- **Spikformer reproductions** **[GitHub issue]**:
  - #33 (2026-04-01): CIFAR-10 4-384 with the original code and `cifar10.yml` gave 91%. Dividing LR by 8 for one A100, by the linear scaling rule, gave 86.37%. Closed without a reply.
  - #12: ImageNet 8-512 lagged the paper's curve (about 40% vs > 50% at epoch 25) with batch 24. The author: "batch size is one of the most important hyperparameters. We use 8 Nvidia V100 for 8 days." Still open.
  - #13: the 4-256 config fails on the head count; another user (not a listed author) suggests 8 heads and says the "best metric is about 94". The paper's 4-256 figure was not checked.
  - #23: a user's energy recomputation with the Spikingformer calculator gave 0.2457 mJ on CIFAR-10. The paper reports no CIFAR-10 energy to check it against.
  - — [#33](https://github.com/ZK-Zhou/spikformer/issues/33); [#12](https://github.com/ZK-Zhou/spikformer/issues/12); [#13](https://github.com/ZK-Zhou/spikformer/issues/13); [#23](https://github.com/ZK-Zhou/spikformer/issues/23)
- **SDT V1** **[GitHub issue]**:
  - #25 (open, no reply): firing-rate statistics from Table 2 could not be reproduced. The user measured 8-768, block 0, first ImageNet sample: q_lif 0.20–0.23, k_lif 0.020–0.023, v_lif 0.023–0.029, talking_heads_lif 0.35–0.42.
    - Paper Table 2 [full text]: 8-512, averaged over T = 4 and 8 blocks. Q_S 0.0091, K_S 0.0090, g(Q_S,K_S) 0.0713, V_S 0.1649, SDSA output 0.0209. The text says K and V vectors are "typically less than 0.01".
    - The user's measurement covers a different model size, a single block and a single sample, so it is not a like-for-like check.
    - The low-rate tensors measured about 0.02, roughly 2× the paper's 0.009.
  - #24: the maintainer explains that the code's Q plays the paper's V role (see Q2), so the code's q_lif (≈ 0.2) corresponds to the paper's V_S (0.1649).
  - #23 (comment): a user reports CIFAR10-DVS around 75% vs the paper's 80% even after the suggested tricks.
  - #1: a CIFAR-10 shortfall to about 85% was a user data-loading bug (resolved).
  - — [#25](https://github.com/BICLab/Spike-Driven-Transformer/issues/25); [#23](https://github.com/BICLab/Spike-Driven-Transformer/issues/23); [#1](https://github.com/BICLab/Spike-Driven-Transformer/issues/1)
- **Seeds and variance in the papers**:
  - QKFormer, STAtten and Max-Former report single numbers; none mentions seeds, variance or error bars ([QKFormer](https://arxiv.org/html/2403.16552v2); [STAtten](https://arxiv.org/html/2409.19764); [Max-Former](https://arxiv.org/html/2505.18608v6)).
  - SDT V3 reports no seeds or variance ([arXiv HTML](https://arxiv.org/html/2411.16061)).
  - STEP Table 12 gives Spikformer CIFAR-10 95.16 with std 0.09 and 95% CI [95.04, 95.27]; the number of runs is not stated ([STEP HTML](https://arxiv.org/html/2505.11151v2)).
- **Recipe confounds visible in reported numbers**:
  - QKFormer's 85.65% uses 384×384 test input on a 224-trained model (T = 4, 64.96M params, 200 epochs). At 224 the same model gives 84.22%, and 81.69% at T = 1. ([QKFormer HTML](https://arxiv.org/html/2403.16552v2))
  - SDT-8-768 is 76.32% at 224 vs 77.07% at 288 ([STAtten table](https://arxiv.org/html/2409.19764)).
  - SDT V1 notes that Att-MS-Res-SNN reaches the same 77.1% but with 78.37M params and 1000 epochs, against SDT's 66.34M and 310 epochs. SDT's own 77.1% uses 288×288 inference; it is 76.32% at 224 ([SDT PDF](https://papers.neurips.cc/paper_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf)).
  - SDT V3 uses distillation and MIM pre-training ([arXiv HTML](https://arxiv.org/html/2411.16061)).
  - SpikingResformer's CIFAR results used transfer learning ([STEP HTML](https://arxiv.org/html/2505.11151v2)).
  - STEP keeps original batch sizes and epochs for ImageNet by design.

### Inferences
- A layout bug like QKFormer's ([T·B] fed to a multi-step LIF) is easy to make in a model whose second path runs neurons along the **token axis**. If T, batch and token dimensions are merged before a multi-step neuron, its state leaks across samples or tokens, and test accuracy then depends on the evaluation batch size. Evaluating at batch size 1 vs the default and requiring identical outputs is a cheap guard.
- Published single-run attention gains (0.23–1.8 points) mostly fall within plausible seed variation on small datasets. STEP's std of 0.09 on CIFAR-10 is one data point with an unknown number of runs. Our 3-seed, ±1% MSE rule is stricter than anything in the reference papers.
- Firing rates are the input to every energy estimate, yet they cannot be checked from the released code without knowing which tensor plays which role (SDT #24 / #25). Logging per-tensor rates with explicit names is needed for any attention-swap energy claim.

### Gaps
- No formal erratum or corrigendum was found for Spikformer, SDT, QKFormer, SpikingResformer or STAtten. The QKFormer defect remains a third-party report with an independent duplicate and agreement from another author, but no author confirmation.
- The STAtten GitHub repository moved (API "Moved Permanently"), so its issues were not read. SpikingResformer issue #5 (SOP computation) and the remaining V2 threads could not be read because of GitHub API rate limits.
- Seed variance for spiking Transformers on time-series forecasting (ETT MSE) was not found in these sources.

---

## Q5. What checks should be made when one spiking attention replaces another?

### Takeaway
The sources together point to a checklist with four parts:
- **Hold fixed** everything other than the attention: T, residual type, front-end, recipe, effective batch, evaluation batch and input length.
- **Add controls**: a random or frozen attention, and a capacity-matched quantized-ANN attention.
- **Report energy per module** with a stated per-operation convention, counting non-binary operands as MAC and adding the dense terms and memory traffic.
- **Run several seeds** and check that evaluation does not depend on batch size.

### Cited Findings
- STEP's protocol: one uniform recipe for all baselines, no dataset- or model-specific tricks, report deviations from published numbers, run module-wise ablations (attention, neuron, encoding, conv front-end), and account for memory access in energy. — [STEP HTML](https://arxiv.org/html/2505.11151v2)
- Random-attention control (STEP Table 8): freezing Q/K at random costs < 0.35 points. — [STEP HTML](https://arxiv.org/html/2505.11151v2)
- Capacity-matched baseline: compare the T-step SNN with a ⌈log2(T+1)⌉-bit QNN, and model data movement. — [TCAD 2026](https://doi.org/10.1109/TCAD.2026.3718799)
- Bit-budget view: T steps ≈ T-bit activations, so compare at equal bit budgets. — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Are_Conventional_SNNs_Really_Efficient_A_Perspective_from_Network_Quantization_CVPR_2024_paper.html)
- Count non-binary operands as MACs and report both dense and effective operation counts. — [NeuroBench docs](https://neurobench.readthedocs.io/en/latest/metrics/workload_metrics/synaptic_operations.html)
- The SDT maintainer: models with SEW (integer) residuals should not price conv/linear at 0.9 pJ per FLOP "for a strict energy comparison". — [SDT #23](https://github.com/BICLab/Spike-Driven-Transformer/issues/23)
- Count memory for state across time steps (32T bits per membrane access in STEP's model), and the repeated data movement over timesteps and the neuron-module overhead named by Bhattacharjee et al. — [STEP HTML](https://arxiv.org/html/2505.11151v2); [ICASSP 2024](https://doi.org/10.1109/ICASSP48485.2024.10448269)
- Prefer IF over LIF where possible, since leak adds energy (Dampfhoffer). — [TETCI](https://doi.org/10.1109/TETCI.2022.3214509)
- Hold the residual type fixed. Moving between SEW and membrane shortcuts changes accuracy by 1–2.3 points and estimated energy by about 2.7× ([Max-Former Table 3](https://arxiv.org/html/2505.18608v6); [SDT Table 6](https://papers.neurips.cc/paper_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf)).
- Check that "binary" inputs to the attention really are binary in code, as opposed to what the figure says. QKFormer's attention input is `x + x_feat` ([QKFormer #5](https://github.com/zhouchenlin2096/QKFormer/issues/5)).
- Report multi-bit score widths and memory traffic for attention separately from the projections, as Bishop does: scores are 6–10 bits, and pruning changes attention compute and memory by 62.7% and 38.1% ([Bishop](https://arxiv.org/html/2505.12281v1)).

### Inferences
Recommended checks for our ETT attention swap. These are synthesised from the findings above, not taken from any single source.
1. **Matched structure.** Keep the same T = 4, the same neuron type and threshold, the same residual (membrane-shortcut vs spike-add), the same projections and bias usage, and the same token count N per cell. Report trainable parameters per variant. Note that a relative-position bias adds N²·heads parameters, and a STMixer-style mixer adds N² parameters.
2. **Matched recipe.** Keep epochs, LR, effective batch (including any accumulation), early-stopping rule and input length identical. QKFormer's `accum_iter` and Spikformer's batch-size sensitivity show how easily these drift.
3. **Controls.** Include (a) no token mixing or identity, (b) frozen random Q/K (STEP's test), and (c) a non-spiking or k-bit-quantized attention twin with ⌈log2(T+1)⌉ = 3 bits for T = 4.
4. **Seeds.** Use ≥ 3 seeds per cell and report the std of the relative MSE change. Do not accept a gain smaller than the seed spread. Reference papers give no variance to compare against.
5. **Batch-invariance test.** Run identical inputs at evaluation batch 1 vs the default and require identical outputs. This guards against [T·B]-style state leaks (QKFormer #14), especially in the token-axis path.
6. **Firing rates.** Log per-tensor firing rates with explicit names (q, k, v, score > 0, attention output, post-LIF) per variant. Report energy only with these measured rates (SDT #24 / #25).
7. **Energy split and convention.** Report the attention core (q k^T, map × v, bias add, scale, LIF update) separately from the Q/K/V/output projections. For each term state:
   - whether an operand is binary (AC) or non-binary (MAC, per NeuroBench);
   - for spike×spike products, which rate convention is used (Spikformer fr × FLOPs, SDT R̂(Q⊙K), or both-non-zero);
   - that dense terms are counted (the relative-bias add is N² per head per step regardless of spikes; any unfolded scale or BN).
8. **Sensitivity to energy assumptions.** Recompute the totals with integer-add costs for popcount accumulators (Horowitz 8-bit add 0.03 pJ, 32-bit add 0.1 pJ) and with a memory term (for example Horowitz's 10 pJ per 64-bit read from an 8 KB cache, or STEP's 32T-bit membrane accesses). State whether the ranking of variants is robust to these changes.
9. **Wording of claims.** Call energy figures "theoretical operation-count estimates (45 nm, no memory)" unless memory is modelled. Do not claim any advantage over a quantized ANN without the matched twin.

### Gaps
- No source gives a validated per-operation energy model for spike×spike popcount attention at small N and D (24–180 tokens, D = 64). Our sensitivity analysis would rest on generic 45 nm numbers.
- No source quantifies seed variance for spiking attention on ETT. The ±1% threshold cannot be cross-checked against the literature.
- STMixer's full numbers (accuracy vs Spikformer, energy) were not verified. Its argument that weight × spike token mixing is the asynchronous-friendly form is taken from the abstract and secondary summaries.
