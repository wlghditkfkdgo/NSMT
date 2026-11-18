# Softmax-free spike-driven self-attention family (Spikformer SSA and successors) as candidates for the NSMT coupling cross-attention

Scope note. Primary sources were read as full text: arXiv PDFs, CVF / NeurIPS-proceedings PDFs, and three official code files. Venues were checked on official or DOI pages where noted. "Self-stated" means the venue appears only in the authors' camera-ready header or arXiv comment. Notation: N = token count, D = embedding channels, d = D/H = per-head channels, T = spiking time steps. "Inference" bullets are my reasoning about the NSMT coupling. The NSMT coupling as described: query = Neocortex spikes, key/value = Hippocampus spikes, scores = q kᵀ·D^-0.5 + learned per-head relative-position bias, no softmax, out = LIF(scores @ v), then gating x ← x·(1−y). None of the papers below tested that setting.

## 1. Spikformer SSA (ICLR 2023) and Spikformer V2: exact formulation and stated reason for dropping softmax

### Takeaway
SSA is SN(Q Kᵀ V · s) with binary Q, K, V = SN(BN(XW)), and it is computed per time step. Softmax was dropped because binary Q and K already give a non-negative map, and the authors argue that float Q/K plus softmax add nothing for spike-form X and V. The scale s matters: without it, training on CIFAR loses accuracy, and dot-product variants with large values do not converge on ImageNet. Spikformer V2 keeps SSA unchanged. Its changes are the stem and self-supervised pre-training, and it remains arXiv-only.

### Cited Findings
- Equations (Sec. 3.3): Q = SN_Q(BN(XW_Q)), K = SN_K(BN(XW_K)), V = SN_V(BN(XW_V)), with Q, K, V ∈ R^{T×N×D} (Eq. 14). SSA′(Q,K,V) = SN(Q Kᵀ V ∗ s) (Eq. 15) and SSA(Q,K,V) = SN(BN(Linear(SSA′(Q,K,V)))) (Eq. 16). The block is X′ = SSA(X) + X, followed by X = MLP(X′) + X′ (Eqs. 7–8). — [Spikformer arXiv 2209.15425](https://arxiv.org/abs/2209.15425)
- Multi-head form (App. A): Q, K, V are reshaped to T×H×N×d with D = H×d. MSSA′ = [SSA′_1(q1,k1,v1); …; SSA′_H(qH,kH,vH)] and MSSA = SN(BN(Linear(MSSA′))) (Eqs. 17–19). — [Spikformer arXiv](https://arxiv.org/abs/2209.15425)
- Reasons the paper gives for not using VSA: "1) The float-point matrix multiplication of QF, KF and softmax function which contains exponent calculation and division operation, do not comply with the calculation rules of SNNs. 2) The quadratic space and time complexity of the sequence length of VSA do not meet the efficient computational requirements of SNNs." — [Spikformer arXiv §3.3](https://arxiv.org/abs/2209.15425)
- Reason for dropping softmax: Q and K from SN are "naturally non-negative (0 or 1), resulting in a non-negative attention map… Hence it does not need the softmax to ensure the non-negativeness of the attention map", which the paper calls softmax's "most important role in VSA (Qin et al., 2022)". The paper adds that X and V are spike-form "containing limited information", so "float-point-form QF, KF and softmax is redundant for modeling the spike-form X, V". — [Spikformer arXiv §1, §3.3](https://arxiv.org/abs/2209.15425)
- Arithmetic: with binary Q, K, V the matrix product "degrades… to logical AND operation and summation", for example Σ_i q_i k_i = Σ_{q_i=1} k_i. — [Spikformer arXiv §3.3](https://arxiv.org/abs/2209.15425)
- Intermediates are integer counts, not spikes: QKᵀ entries lie in [0, d], and SSA needs s "to control the large value of the matrix multiplication result". s = 0.125 on ImageNet and CIFAR. A learnable scale is used for the neuromorphic datasets. The threshold of the SN after QKᵀV·s is 0.5, and all other neurons use 1 (App. C.1, §4.2). — [Spikformer arXiv](https://arxiv.org/abs/2209.15425)
- The official code computes K first: `x = k.transpose(-2,-1) @ v; x = (q @ x) * self.scale` with `scale = 0.125`, and `attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, …)`. — [ZK-Zhou/spikformer imagenet/model.py](https://raw.githubusercontent.com/ZK-Zhou/spikformer/main/imagenet/model.py)
- Complexity: SSA is "decomposable". QKᵀ first costs O(N²d) and KᵀV first costs O(Nd²), and the second order is cheaper when N > d (§3.3). Fig. 1 reports 5.5×10⁶ operations for SSA against 77×10⁶ for VSA, with VSA at 354.2 µJ. — [Spikformer arXiv](https://arxiv.org/abs/2209.15425)
- T axis: "SSA is independently conducted on each time step". App. B adds that T is an independent dimension only for spiking-neuron layers and is merged with batch elsewhere. — [Spikformer arXiv §3.3, App. B](https://arxiv.org/abs/2209.15425)
- Table 1, attention-map variants inside the same Spikformer. Columns are Acc / OPs (M) / energy of one Q-K-V computation (µJ) on CIFAR10-DVS | CIFAR10 | CIFAR100:
  - A_I: 79.40/16.8/77 | 93.96/6.3/29 | 76.94/6.3/29
  - A_LeakyReLU: 79.80/16.8/77 | 93.85/6.3/29 | 76.73/6.3/29
  - A_ReLU: 79.40/16.8/77 | 94.34/6.3/29 | 77.00/6.3/29
  - A_softmax: 80.00/19.1/88 | 94.97/6.6/30 | 77.92/6.6/30
  - A_SSA: 80.90/0.66/0.594 | 95.19/1.1/0.990 | 77.86/1.3/1.170

  SSA is −0.06 against softmax on CIFAR100, +0.22 on CIFAR10, and +0.90 on CIFAR10-DVS, at roughly 25–150× lower attention energy. — [Spikformer arXiv Table 1](https://arxiv.org/abs/2209.15425)
- Table 5 ablation:
  - CIFAR10/100, Spikformer-4-384, T=4: SSA 95.19/77.86, VSA (spike V) 94.97/77.92, VSA with float V 95.17/78.37.
  - ImageNet, Spikformer-8-512, T=4: SSA 73.38, VSA 72.70, VSA float-V 73.96. The variants I, ReLU and LeakyReLU "do not converge" because "the value of dot-product value of Query, Key, and Value is large, which makes the surrogate gradient of the output spike neuron layer disappear".
  - Time-step sweep for SSA: T=1/2/4/6 gives 93.51/74.36, 93.59/76.28, 95.19/77.86 and 95.34/78.61 on CIFAR10/100, and 70.14, 71.09, 73.38 and 73.70 on ImageNet.

  — [Spikformer arXiv Table 5, §4.3](https://arxiv.org/abs/2209.15425)
- Independent evidence that s matters: QKFormer Table 5 reports CIFAR100 / CIFAR10-DVS of 76.95% / 79.30% for "Spikformer (SSA, w/o scaling)" against 78.21% / 80.90% for "Spikformer (SSA)". — [QKFormer arXiv 2403.16552](https://arxiv.org/abs/2403.16552)
- SpikingResformer (App. B) tried to derive a variance-based scale for SSA and concluded that the form of QKᵀV is "overly complex and lacks practicality. Thus it is difficult to design the scaling factor for SSA." — [SpikingResformer arXiv 2403.14302](https://arxiv.org/abs/2403.14302)
- Venue: ICLR 2023 poster on the official site. Code at ZK-Zhou/spikformer. — [iclr.cc poster 12139](https://iclr.cc/virtual/2023/poster/12139); [GitHub](https://github.com/ZK-Zhou/spikformer)
- Spikformer V2 reuses the same SSA, "SSA operates on each time step", and repeats the same Table 1 numbers. The change is a Spiking Convolutional Stem (SCS) plus masked-reconstruction SSL pre-training. Reported results: 80.38% for an 8-layer model at 4 time steps, and 81.10% for a 172M 16-layer model at 1 time step after SSL. — [Spikformer V2 arXiv 2401.02020](https://arxiv.org/abs/2401.02020)
- Spikformer V2 venue: OpenAlex lists only arXiv, and no journal or conference version was found. The arXiv stamp reads v2, 18 Aug 2026. — [arXiv 2401.02020](https://arxiv.org/abs/2401.02020)

### Inferences
- The NSMT coupling (q kᵀ · D^-0.5 + bias, LIF(scores @ v)) is SSA with three differences: cross-attention sources, a relative-position bias, and a scale of D^-0.5 instead of s = 0.125. With D = 64, D^-0.5 = 0.125, the same value. If the scale is per head (d = 4 or 8), the factor would differ. This should be checked in the code.
- The additive per-head relative-position bias is added to QKᵀ before multiplying by V. That forces QKᵀ to be materialised first (O(N²d) per head) and blocks the KᵀV-first reordering that the "linear" spiking-attention papers rely on. Any candidate that is "linear in N" loses that property if the bias is kept in the same place.
- With 8 or 16 heads at D = 64 (d = 8 or 4), each QKᵀ entry can only take d+1 integer values (0..d). The scores are therefore very coarse. This is the regime that the α-XNOR and Hamming-attention papers criticise (Section 6).

### Gaps
- Spikformer gives no ablation of SSA in a cross-attention setting, and none on sequences, audio or time series.
- I did not find a peer-reviewed Spikformer V2. Treat it as arXiv-only (lower reliability).

## 2. Spike-driven Transformer SDSA (NeurIPS 2023) and Spike-driven Transformer V2 / Meta-SpikeFormer SDSA-1..4 (ICLR 2024)

### Takeaway
SDSA (V1) replaces matrix products with Hadamard masks. The column sum over tokens of K⊗V passes through a spiking neuron to form a binary 1×D channel gate, which masks Q. The cost is O(ND), but it is "hard" channel attention with no token-to-token routing. In isolation it cost −0.03 / −0.34 points against Spikformer SSA on CIFAR-10/100. Meta-SpikeFormer tested four operators and kept SDSA-3 = SN_s(Q Kᵀ V), which is Spikformer's SSA with s folded into the threshold. It was chosen because it had the highest accuracy: SDSA-1 is −0.8 and SDSA-2 is −1.2 on ImageNet.

### Cited Findings
- SDT Eq. 13 (SDSA-V1): SDSA(Q,K,V) = g(Q_S,K_S) ⊗ V_S = SN(SUM_c(Q_S ⊗ K_S)) ⊗ V_S. Here ⊗ is the Hadamard product (equivalent to a mask), SUM_c is the column sum, and g and SUM_c both output a D-dimensional row vector. Float Q, K, V come from linear layers on the spike input S ∈ R^{T×N×D} and then pass through SN. — [SDT arXiv 2307.01694 §3.3](https://arxiv.org/abs/2307.01694)
- SDT Eq. 14, introduced with "we here assume T = 1": SDSA = Q_S ⊗ g(K_S, V_S) = Q_S ⊗ SN(SUM_c(K_S ⊗ V_S)). The paper calls it "a linear attention… SN(·) is the kernel function". In the H = D special case (Eq. 15), complexity is "O(0 + ND)", and "the whole SDSA only needs about 0.02ND times of addition". — [SDT arXiv §3.3](https://arxiv.org/abs/2307.01694)
- The official code implements the Eq. 14 order: `kv = k.mul(v); kv = kv.sum(dim=-2, keepdim=True); kv = self.talking_heads_lif(kv); x = q.mul(kv)`. The LIF uses v_threshold = 0.5, and the module also defines `scale = 0.125` and an `attn_lif` with threshold 0.5. — [BICLab/Spike-Driven-Transformer module/ms_conv.py](https://raw.githubusercontent.com/BICLab/Spike-Driven-Transformer/main/module/ms_conv.py)
- Nature of the attention: "Eq. 14 actually converts the soft vanilla self-attention to hard self-attention… binary self-attention scores to directly mask unimportant channels in the sparse spike Value tensor. Although this introduces a slight loss of accuracy (Table 6), SDSA(·) consumes almost no energy." No scale is used: Q_S and K_S are "very sparse (typically less than 0.01…), the value of summing Q_S ⊗ K_S column by column does not fluctuate much, thus the scale operation is not needed" (Supp. S3). — [SDT arXiv](https://arxiv.org/abs/2307.01694)
- Measured firing rates for SDT-8-512 on ImageNet, block 1 average (Table S1): Q_S 0.0195, K_S 0.0213, g(Q_S,K_S) 0.1139, SDSA output V̂_S 0.0419, V_S 0.2959. — [SDT arXiv Table S1](https://arxiv.org/abs/2307.01694)
- Energy model (Table 1). Vanilla: Q,K,V cost E_MAC·3ND², f(Q,K,V) costs E_MAC·2N²D, scale costs E_M·N², and softmax costs E_MAC·2N². SDT: Q,K,V cost E_AC·T·R·3ND², f(Q,K,V) costs E_AC·T·R̂·ND, and scale and softmax are absent. The abstract claims "up to 87.2× lower computation energy than vanilla self-attention". — [SDT arXiv](https://arxiv.org/abs/2307.01694)
- Clean ablation, Table 6, Spiking Transformer-2-512 on CIFAR-10 / CIFAR-100:
  - Baseline [Spikformer]: 93.12 / 73.17
  - + SDSA: 93.09 (−0.03) / 72.83 (−0.34)
  - + MS (membrane shortcut): 93.93 (+0.81) / 74.63 (+1.46)
  - Full SDT: 93.82 (+0.73) / 74.41 (+1.24)

  — [SDT arXiv Table 6](https://arxiv.org/abs/2307.01694)
- Whole-model ImageNet comparison (Table 3), which mixes the SDSA and membrane-shortcut changes: Spikformer 8-512 is 29.68M, 11.57 mJ, T=4, 73.38%. SDT 8-512 is 29.68M, 4.50 mJ, T=4, 74.57% (1.13 mJ and 71.68% at T=1). SDT 8-768 is 66.34M, 6.09 mJ, 77.07%. — [SDT arXiv Table 3](https://arxiv.org/abs/2307.01694)
- Venue: NeurIPS 2023 main track (official proceedings). Code at BICLab/Spike-Driven-Transformer. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract.html); [GitHub](https://github.com/BICLab/Spike-Driven-Transformer)
- SDT-V2 block (Eqs. 11–14): Q_S = SN(RepConv1(U)), K_S = SN(RepConv2(U)), V_S = SN(RepConv3(U)), U′ = U + RepConv4(SDSA(Q_S,K_S,V_S)), U″ = U′ + ChannelMLP(U′), where RepConv is a 3×3 re-parameterised convolution. The paper states that SDSA differs from VSA because "Query, Key, Value are spiking tensors… do not have softmax and scale… linear with the token number N". — [SDT-V2 arXiv 2404.03663 §3](https://arxiv.org/abs/2404.03663)
- SDT-V2 operators (App. A.2):
  - SDSA-1 (Eq. 18): Q_S ⊗ SN(SUM_c(K_S ⊗ V_S)), O(ND), "same computational complexity as hydra attention".
  - SDSA-2 (Eq. 19): SN(SUM_c(Q_S)) ⊗ V_S, which removes K entirely.
  - SDSA-3 (Eq. 20): SN_s(Q_S K_Sᵀ V_S) = SN_s((Q_S K_Sᵀ)V_S), where SN_s has threshold s·u_th. It is "inspired by the spiking self-attention SN(Q_S K_Sᵀ V_S ∗ s) in Zhou et al. (2023). Because Q_S K_Sᵀ V_S yield large integers, a scale factor s for normalization is needed to avoid gradient vanishing… we directly merge the s into the threshold".
  - SDSA-4: SDSA-3 with a learnable threshold initialised at s·u_th.
  - Statement: "We exploit SDSA-3 by default. All SDSAs only have addition, no softmax and scale."

  — [SDT-V2 arXiv](https://arxiv.org/abs/2404.03663)
- SDT-V2 official code for SDSA-3: `x = k.transpose(-2,-1) @ v; x = (q @ x) * self.scale` with `scale = 0.125`, and `attn_lif` uses v_threshold = 0.5. During training the scale is an explicit multiply. — [BICLab/Spike-Driven-Transformer-V2 classification/models.py](https://raw.githubusercontent.com/BICLab/Spike-Driven-Transformer-V2/main/classification/models.py)
- Ablation (Table 5, ImageNet-1K, Meta-SpikeFormer C=48, T=1). The baseline is 31.3M params, 7.8 mJ, 75.4%.
  - SDSA-3→SDSA-1: 31.3M, 7.2 mJ, 74.6 (−0.8)
  - SDSA-3→SDSA-2: 28.6M, 6.3 mJ, 74.2 (−1.2)
  - SDSA-3→SDSA-4: 31.3M, 7.7 mJ, 75.4 (+0.0)
  - RepConv-1/2/3→Linear: 27.2M, 6.0 mJ, 75.0 (−0.4)
  - Membrane shortcut→vanilla shortcut: does not converge
  - Fully Transformer-based SNN blocks: 26.2M, 5.4 mJ, 71.7 (−3.7)

  Stated reason for the choice: "SDSA-3 has the highest computational complexity… and its accuracy is also the best." — [SDT-V2 arXiv Table 5, §4.5](https://arxiv.org/abs/2404.03663)
- SDT-V2 also evaluates HAR-DVS event-based action recognition (Table 2), COCO detection (Table 3) and ADE20K segmentation (Table 4). — [SDT-V2 arXiv](https://arxiv.org/abs/2404.03663)
- Venue: ICLR 2024 is self-stated in the camera-ready header ("Published as a conference paper at ICLR 2024") and the arXiv comment ("Accepted by ICLR2024"). Code at BICLab/Spike-Driven-Transformer-V2. — [arXiv 2404.03663](https://arxiv.org/abs/2404.03663); [GitHub](https://github.com/BICLab/Spike-Driven-Transformer-V2)

### Inferences
- SDT Eq. 13 and Eq. 14 are presented as two "equivalent versions", but they are not algebraically identical. SN is nonlinear, and SUM_c is taken over different products (Q⊗K against K⊗V). Code and SDT-V2 use the Eq. 14 form. Whoever re-implements SDSA should follow the code form, not Eq. 13.
- SDT-V2 App. A.2 says that "the time complexity of Q_S(K_Sᵀ V_S) and (Q_S K_Sᵀ)V_S are O(N²D) and O(ND²)". That appears swapped relative to standard analysis and to Spikformer's own statement, where KᵀV first is O(Nd²). It looks like a typo.
- SDSA-1/2 contain no token-to-token interaction. The only cross-token operation is a column sum giving one D-bit gate per time step. As a coupling operator (Q from Neocortex, K/V from Hippocampus), SDSA-1 would output Neocortex spikes masked by a Hippocampus-derived channel gate. Information flows the opposite way from the current module, where Hippocampus V content is routed to Neocortex-query positions. SDSA-2 (no K) would gate V by a channel summary of Q only.
- SDSA-3 is the same operator as SSA, and therefore the same as NSMT's current coupling apart from the bias and threshold placement. Swapping it in would only move s into the threshold.

### Gaps
- No paper in this family reports SDSA-1/2/3 on sequential or forecasting tasks. The SDT-V2 ablation is on ImageNet only, at T=1.
- The official ICLR 2024 site was not checked directly; DBLP and OpenReview are blocked from this machine.

## 3. Spike-driven Transformer V3 / E-SpikeFormer (TPAMI 2025): does it change attention?

### Takeaway
No new attention operator. The E-SDSA module keeps SN(Q_S K_Sᵀ V_S · scale) from Meta-SpikeFormer, uses Linear in place of RepConv for Q, K and V, and expands V by a factor γ. The large gains come from Spike Firing Approximation (SFA) training, which uses integer activations during training and spike trains at inference, and from architecture changes. No attention-only ablation is reported.

### Cited Findings
- E-SDSA: Q_S = SN(Linear(U)), K_S = SN(Linear(U)), V_S = SN(Linear_γ(U)), U′ = Linear_{1/γ}(SN(Q_S K_Sᵀ V_S ∗ scale)). "SN(Q_S K_Sᵀ V_S ∗ scale) is the SDSA operator, which is consistent with Meta-SpikeFormer [28], and constant scale factor can be reparameterized into SN's threshold". A SpikeSepConv is inserted before E-SDSA, and the V channels are expanded by γ (iRMB technique). — [E-SpikeFormer arXiv 2411.16061 §3.3](https://arxiv.org/abs/2411.16061)
- SFA: training uses an integer activation S_l^D ∈ [0, D] at a single time step. Proposition 1 states that it equals the sum of spikes of an IF-SR neuron (V_th = 1) over D time steps, so inference is spike-driven. — [E-SpikeFormer arXiv §3.2, §4.1](https://arxiv.org/abs/2411.16061)
- Ablation (Table 5, 10M model, ImageNet; Step is time steps × D):
  - Baseline Meta-SpikeFormer 1×1: 3.3 mJ, 68.5%
  - Baseline 4×1: 11.9 mJ, 71.3%
  - + SFA training, 1×4: 7.3 mJ (−4.6), 77.8% (+6.5)
  - + E-SpikeFormer architecture, 1×4: 3.0 mJ (−4.3), 78.5% (+0.7)

  The headline ImageNet results are 78.5 / 79.8 / 84.0 / 86.2% for 10M / 19M / 83M / 173M models. — [E-SpikeFormer arXiv Table 5, abstract](https://arxiv.org/abs/2411.16061)
- Venue: IEEE TPAMI 2025, DOI 10.1109/TPAMI.2025.3530246 (OpenAlex and the arXiv DOI field). Code at BICLab/Spike-Driven-Transformer-V3. — [DOI](https://doi.org/10.1109/TPAMI.2025.3530246); [GitHub](https://github.com/BICLab/Spike-Driven-Transformer-V3)

### Inferences
- During SFA training, Q_S, K_S and V_S are integers in [0, D], not binary. Q Kᵀ V therefore involves integer multiplication during training, and the "spike" form exists only at inference, unrolled over D steps. Adopting SFA in NSMT would be a training-scheme change, not an attention replacement.

### Gaps
- No SDSA-vs-SSA or attention-operator ablation in E-SpikeFormer.

## 4. QKFormer (NeurIPS 2024): Q-K token attention / channel attention (no V), linear complexity

### Takeaway
QKTA computes a binary N×1 token mask from the per-token spike count of Q through a spiking neuron, and applies it to K. QKCA does the same with a 1×D channel mask. There is no V, no scale and no token-to-token routing. Used alone, QKTA is clearly worse than SSA (CIFAR100 79.09 vs 81.23; CIFAR10-DVS 80.70 vs 84.10). The paper therefore keeps SSA in the last stage by default, and the mixed QKTA + SSA model is about equal to all-SSA.

### Cited Findings
- Q = SN_Q(BN(XW_Q)) and K = SN_K(BN(XW_K)) (Eq. 5).
  - QKTA (Eq. 6): A_t = SN(Σ_{i} Q_{i,j}), a sum over channels for each token, giving A_t ∈ {0,1}^{N×1}. Then X′ = A_t ⊗ K ("column mask… token mask").
  - QKCA: A_c = SN(Σ_j Q_{i,j}), a sum over tokens, giving a 1×D vector, and X′ = A_c ⊗ K ("channel mask").
  - Post-projection: X″ = SN(BN(Linear(X′))) (Eq. 7).
  - "The channel dimension is D/h in the multi-head Q-K attention".
  - "Same with [11], time step T is an independent dimension for the spiking neuron layer".

  — [QKFormer arXiv 2403.16552 §3.2](https://arxiv.org/abs/2403.16552)
- Complexity (Table 1): time complexity is O(N²D) for VSA, O(N²D) for SSA, O(ND) for SDSA, O(D) for QKTA and O(N) for QKCA. Space complexity is O(N²+ND) for VSA and SSA, O(ND) for SDSA, O(N) for QKTA and O(D) for QKCA. The time complexity "can reach up to O(N∗D)" when implemented with broadcast element-wise multiplication. — [QKFormer arXiv §3.2](https://arxiv.org/abs/2403.16552)
- No scaling: with i.i.d. Bernoulli spikes, "the max theoretical variance of QKTA is 16, but the maximum theoretical variance of SSA is over 3000" (N = 196, d = 64). This is the paper's reason that QKTA "can discard scaling operations… but SSA can not". — [QKFormer arXiv §3.3, §4.3, Fig. 4](https://arxiv.org/abs/2403.16552)
- Firing rates (Table 4, QKFormer 64.9M on ImageNet), stage 1 / stage 2: Q 0.0432 / 0.0231, K 0.1784 / 0.0847, A_t 0.3477 / 0.2655, X′ 0.0832 / 0.0350, X″ 0.1478 / 0.0577. — [QKFormer arXiv Table 4](https://arxiv.org/abs/2403.16552)
- Mixed integration: "QKTA is conducted in the former stage… we use SSA in the last stage of QKFormer and QKTA in the former stages by default". — [QKFormer arXiv §3.4](https://arxiv.org/abs/2403.16552)
- Table 6, attention ablation. Columns are CIFAR100 (Acc, Param) and CIFAR10-DVS (Acc, Param):
  - QKTA + SSA (baseline): 81.15%, 6.74M | 84.00%, 1.50M
  - QKCA + SSA: 81.07%, 6.74M | 84.30%, 1.50M
  - QKTA + QKCA: 81.04%, 6.44M | 83.10%, 1.44M
  - SSA only: 81.23%, 6.79M | 84.10%, 1.52M
  - QKCA only: 81.00%, 6.44M | 80.70%, 1.44M
  - QKTA only: 79.09%, 6.44M | 80.70%, 1.44M

  The paper's summary: "using a single type of Q-K attention… leads to suboptimal performance. In particular, the performance decline in QKTA is more obvious." — [QKFormer arXiv Table 6](https://arxiv.org/abs/2403.16552)
- Table 5, CIFAR100 / CIFAR10-DVS: Spikformer (SSA, w/o scaling) 76.95% / 79.30%; Spikformer (SSA) 78.21% / 80.90%; Spikformer + SPEDS 80.26% / 82.20%. — [QKFormer arXiv Table 5](https://arxiv.org/abs/2403.16552)
- Memory: the paper reports that SSA used much more GPU memory than QKTA as the token count grew, "about 10×" at the largest size shown in Fig. 3(b). The exact token count is garbled in my text extraction. — [QKFormer arXiv §4.3](https://arxiv.org/abs/2403.16552)
- Venue: NeurIPS 2024 Spotlight (arXiv comment), NeurIPS proceedings DOI 10.52202/079017-0416 (OpenAlex). Code at zhouchenlin2096/QKFormer. — [DOI](https://doi.org/10.52202/079017-0416); [GitHub](https://github.com/zhouchenlin2096/QKFormer)

### Inferences
- As a coupling replacement, QKTA with Q from Neocortex and K from Hippocampus would give a per-token binary gate on Hippocampus K, driven by the Neocortex spike count at the same token index. This works because both paths have the same N, aligned by index. It removes all cross-token routing: token i of the output depends only on token i of each path. It is closer to a learned per-token gate than to attention. QKCA would give one channel gate per time step.
- Because QKTA used alone lost about 2 points on CIFAR100 and 3.4 points on CIFAR10-DVS relative to SSA, the paper's own evidence argues against it as a sole attention.

### Gaps
- No sequence or time-series results. All ablations are on CIFAR100 and CIFAR10-DVS.

## 5. SpikingResformer (CVPR 2024): Dual Spike Self-Attention (DSSA) and its scaling factors

### Takeaway
DSSA forms a binary N×(N/p²) attention map. It computes SN(X·Wᵀ·Xᵀ·c1) using a "dual spike transformation" in which a weight is accumulated only when two spikes coincide (AND). It then outputs SN(AttnMap·X·W·c2). The scales come from a variance analysis: c1 = 1/√(f_X d) and c2 = 1/√(f_Attn·HW/p²), using running firing rates. Without scaling, DSSA does not converge. The paper's derived scale also improved SDSA (83.06 → 83.96 on ImageNet100).

### Cited Findings
- DST definitions (Eqs. 7–8): DST(X, Y; f) = X f(Y) = X Y W and DST_T(X, Y; f) = X f(Y)ᵀ = X Wᵀ Yᵀ, where "X ∈ {0,1}^{T×p×m} and Y ∈ {0,1}^{T×m×q} represent the dual spike inputs". f is a generalised linear transformation, such as a convolution with BN folded in. — [SpikingResformer arXiv 2403.14302 §4.2](https://arxiv.org/abs/2403.14302)
- DSSA (Eqs. 10–11): AttnMap(X) = SN(DST_T(X, X; f)·c1) with f(X) = BN(Conv_p(X)), a p×p convolution with stride p. DSSA(X) = SN(DST(AttnMap(X), X; f)·c2). "In DSSA, there is no need for the softmax function since the spiking neuron layer inherently generates a binary attention map composed of spikes." — [SpikingResformer arXiv §4.2](https://arxiv.org/abs/2403.14302)
- Spike-driven property: I_{i,j}[t] = Σ_{(x_{i,k}[t] ∧ y_{j,l}[t])≠0} w_{l,k}. "The spike input here is the logical AND of dual spikes… a synaptic operation requiring dual spikes to trigger" (Eqs. 14, 16). — [SpikingResformer arXiv §4.4](https://arxiv.org/abs/2403.14302)
- Scaling: Theorem 1 gives E = 0 and Var = f_x·m for a DST output, assuming f(Y) has zero mean and unit variance. This leads to c1 = 1/√(f_X d) and c2 = 1/√(f_Attn HW/p²). Firing rates are tracked "with an exponential moving average with a momentum of 0.999" during training and fixed at inference (Supp. E). — [SpikingResformer arXiv §4.3, Supp. E](https://arxiv.org/abs/2403.14302)
- Multi-head (Eq. 17): MHDSSA(X) = BN(Conv1([DSSA_i(SN(X))]_{i=1}^h)). — [SpikingResformer arXiv §5.2](https://arxiv.org/abs/2403.14302)
- Ablation (Table 3, ImageNet100, parameters matched; SOPs G / Energy mJ / Acc %):
  - SpikingResformer-S: 2.43 / 2.18 / 88.06
  - w/o multi-stage: 1.84 / 1.66 / 85.32
  - w/o group-wise conv: 2.37 / 2.13 / 84.64
  - w/o DSSA (replaced by SSA or SDSA): "not converge"
  - w/o p×p conv: 4.34 / 3.91 / 86.14
  - w/o scaling (or with 1/√d): "not converge"

  "We believe this is because SSA and SDSA do not fit the multi-scale input… the scaling factor is the key to convergence." — [SpikingResformer arXiv Table 3, §6.3](https://arxiv.org/abs/2403.14302)
- Proposed scale for SDSA: c = 1/√(HW f_Q f_K (1−f_Q f_K)) (Supp. B). Table S1 (ImageNet100): Spike-driven Transformer-8-384 scores 83.06 without scaling and 83.96 with it. SpikingResformer-S with SDSA does not converge without scaling and reaches 87.74 with it. — [SpikingResformer arXiv Supp. B, Table S1](https://arxiv.org/abs/2403.14302)
- Venue: CVPR 2024, DOI 10.1109/CVPR52733.2024.00536 (OpenAlex). Code at xyshi2000/SpikingResformer. — [DOI](https://doi.org/10.1109/CVPR52733.2024.00536); [GitHub](https://github.com/xyshi2000/SpikingResformer)

### Inferences
- DST is defined for two different spike inputs X and Y, so a cross version maps directly onto NSMT. AttnMap = SN(X_neo Wᵀ X_hipᵀ · c1) gives a binary N_q×N_kv map, and out = SN(AttnMap · f(X_hip) · c2). Unlike SSA, the attention map is binary (a spike), and the integer counts become sums of real weights, which are AC operations. That is closer to "pure spike" than SSA's integer QKᵀ.
- Complexity remains quadratic in N, reduced by p² on the key side. With N = 24 to 180 tokens and 1-D tokens, the p×p downsampling has no direct analogue, so a 1-D stride-p version would need its own testing.
- DSSA's firing-rate-dependent scales are a principled replacement for NSMT's fixed D^-0.5 scale.

### Gaps
- No sequence or time-series results. Whether DSSA's binary N×N map is too lossy for small N was not tested in the paper.

## 6. Spikingformer, Masked Spiking Transformer, Spiking Wavelet Transformer, Spiking Token Mixer, STAtten, TIM, and 2025–2026 attention-computation changes (SSSA, A²OS²A, α-SSA, SDHA, Max-Former)

### Takeaway
Most "new" spiking attentions either keep the SSA product QKᵀV or replace the similarity measure:
- Kept the product: Spikingformer, STAtten, TIM, E-SDSA.
- Replaced the similarity: α-XNOR (a weight α for matching zeros), Hamming (±1 mapping), SSSA (spike-count based), and A²OS²A (real-valued K, ternary V).

Several report gains over SSA of 1–3 points on CIFAR100 or similar. SWformer, STMixer and Max-Former's early stages are attention-free token mixers: they have no query/key and cannot do cross-attention. Spikingformer had to keep softmax for language (GLUE).

### Cited Findings
- **Spikingformer (AAAI 2026 Oral; arXiv since 2023).** PSSA: X′ = SN(X); Q = SN_Q(ConvBN_Q(X′)), K = SN_K(ConvBN_K(X′)), V = SN_V(ConvBN_V(X′)); Attention(Q,K,V) = ConvBN(SN(QKᵀV ∗ s)) (Eqs. 12–14), with "s… the scaling factor as in (Zhou et al. 2023b)". The attention arithmetic is unchanged from SSA; the change is pre-activation residuals that avoid integer×float multiplications. — [Spikingformer arXiv 2304.11954](https://arxiv.org/abs/2304.11954); [AAAI DOI](https://doi.org/10.1609/aaai.v40i3.37207); [GitHub](https://github.com/TheBrainLab/Spikingformer)
- Spikingformer on GLUE: "we retained the softmax layer in the attention map computation, consistent with SpikeLM (Xing et al. 2024). We find this layer is crucial for model convergence in language tasks, which is different from visual tasks." — [Spikingformer arXiv appendix "Additional analysis on GLUE"](https://arxiv.org/abs/2304.11954)
- **Masked Spiking Transformer (ICCV 2023).** It uses ANN-to-SNN conversion of a Swin Transformer, with QCFS activations replaced by IF neurons. Q_spk[t] = IF(X[t]W_q), K_spk[t] = IF(X[t]W_k), and A_spk[t] = IF(Q_spk[t] K_spk[t]ᵀ/√d) (Eqs. 6–7). The novelty is Random Spike Masking (RSM), which prunes input spikes and gives "26.8% [reduction] in power consumption when the masking ratio is 75%". It needs long inference time steps: the paper discusses the low-latency setting as fewer than 20 steps, and its best results use more. — [MST arXiv 2210.01208](https://arxiv.org/abs/2210.01208); [ICCV DOI](https://doi.org/10.1109/ICCV51070.2023.00169)
- **Spiking Wavelet Transformer (ECCV 2024)** is "an attention-free architecture". Its Frequency-Aware Token Mixer (FATM) sums three branches on spikes: a spiking Haar-wavelet learner, a conv learner, and a pointwise conv. The spiking Haar transform is H_f = Spk(W_haar · Spk(I · W_haarᵀ)), with negative spikes. It reports "22.03% reduction in parameter count, and a 2.52% performance improvement on the ImageNet dataset compared to vanilla Spiking Transformers". Table 4 FATM ablation: on CIFAR100 (4-384, T=4), base 79.31 vs NoHaar 78.89; on CIFAR10-DVS (2-256, T=16), base 83.9 vs NoHaar 80.9. — [SWformer arXiv 2403.11138](https://arxiv.org/abs/2403.11138); [ECCV/LNCS DOI](https://doi.org/10.1007/978-3-031-73116-7_2); [GitHub](https://github.com/bic-L/Spiking-Wavelet-Transformer)
- **Spiking Token Mixer (NeurIPS 2024).** STM(X) = SN(W_STM V) = SN(W_STM SN(BN(XW_V))), where W_STM ∈ R^{N×N} is a trainable, input-independent token-mixing matrix, one per head group (Eq. 5). Motivation: "eliminates the requirement of performing matrix multiplication between two spike matrices", for event-driven asynchronous chips. Table 2 (CIFAR-100, STMixer-4-384-16, T=1): STMixer 79.55 against 78.33 with STM→SSA. Table 3: heads H = 1→32 give 79.02→80.15. Table 4 (CIFAR10-DVS in a simulated event-driven setting, with bias): STMixer vs Spikformer 66.73 vs 64.21 at T=25 and 75.91 vs 67.74 at T=40. — [STMixer NeurIPS PDF](https://www.proceedings.com/content/079/079017-4092open.pdf); [DOI](https://doi.org/10.52202/079017-4092); [GitHub](https://github.com/brain-intelligence-lab/STMixer_demo)
- **STAtten (CVPR 2025).** Block-wise spatial-temporal SSA: STAtten(X[b]) = LIF(Q[b] K[b]ᵀ V[b] · α), where [b] = [iB:(i+1)B, :, d] (Eq. 8). Q, K, V ∈ R^{(T/B)×N×D} are reshaped to R^{(T/B·N)×D}; KᵀV ∈ R^{D×D} is computed first, then Q(KᵀV) (Eq. 9). Complexity is O(TND²), the "same… as previous spatial-only" spiking attentions. Defaults are B=2 at T=4 (static data) and B=4 at T=16 (neuromorphic). — [STAtten arXiv 2409.19764](https://arxiv.org/abs/2409.19764); [CVPR DOI](https://doi.org/10.1109/CVPR52734.2025.01302); [GitHub](https://github.com/Intelligent-Computing-Lab-Yale/STAtten)
- STAtten results:
  - Sequential CIFAR (each image column is one time step, 32 steps), Table 2, s-CIFAR10 / s-CIFAR100: Spikformer 79.29 / 57.17 → STAtten 82.45 / 58.75; SDT 80.32 / 61.08 → 83.41 / 64.30; SpikingResformer 79.25 / 57.48 → 81.84 / 58.30.
  - CIFAR10/100 (T=4), Table 3: Spikformer-4-384 93.99 / 75.06 → 94.36 / 75.85.
  - ImageNet, Table 4: SDT-8-768 76.32 → 78.11.
  - Energy, Table 6: SDT 12.42 mJ vs STAtten+SDT 12.36 mJ.
  - Limitation stated by the authors: block processing "requires complete temporal information" within a block, which is hard on step-by-step neuromorphic chips.

  — [STAtten arXiv](https://arxiv.org/abs/2409.19764)
- **TIM (IJCAI 2024).** It is added to Spikformer's SSA query path. Attention[t] = A_TIM[t] K[t]ᵀ V[t] = α f(Q_TIM[t−1]) K[t]ᵀ V[t] + (1−α) Q[t] K[t]ᵀ V[t] (Eq. 7), where f is a 1-D convolution. The query thus mixes a history term with the current query; Eq. 6 did not extract cleanly, and Eq. 7 implies Q_TIM[t] = α f(Q_TIM[t−1]) + (1−α) Q[t]. Ablations:
  - Table 3: CIFAR10-DVS 78.5 → 81.6; N-CALTECH101 77.7 → 79.0.
  - Table 4, "local TIM" with no cross-step interaction: 78.9 and 76.6.
  - Table 5, α sweep, best α: 0.6 on CIFAR10-DVS (81.6) and 0.4 on N-CALTECH101 (79.0).
  - Table 2, SHD audio: Spikformer 85.1 → TIM 86.3.

  — [TIM arXiv 2401.11687](https://arxiv.org/abs/2401.11687); [IJCAI DOI](https://doi.org/10.24963/ijcai.2024/347); [code](https://github.com/BrainCogX/Brain-Cog/tree/main/examples/TIM)
- **SNN-ViT / Saccadic Spike Self-Attention (ICLR 2025).**
  - Similarity: a cross-entropy-derived similarity CroAtt(Q, K) = Q′K′ᵀ with Q′ = Σ_D Q and K′ = Σ_D K, i.e. per-token spike counts (Eq. 6).
  - Saccadic neuron: H = M_w·Patch with a learnable lower-triangular M_w over T, and S = Θ(H − V_th) (Eq. 8).
  - SSSA (Eq. 9): Θ(M_w Q′×K′ᵀ L − V_th)·V.
  - SSSA-V2 (Eq. 10) treats (K′ᵀL) as a learnable scale on the threshold, giving overall O(2D+N) per the paper.
  - Ablation (Table 5, CIFAR100): SSA 5.76M O(N²D) 76.95; SSSA+DP 77.12; SSSA+LIF 78.84; SSSA-V1 O(N²) 79.71; SSSA-V2 O(D) 79.60. Table 4: baseline 76.95 → +SSSA 79.60 (+2.65).
  - Venue: ICLR 2025 poster on the official site. No official code repository was found by GitHub search.

  — [SNN-ViT arXiv 2502.12677](https://arxiv.org/abs/2502.12677); [iclr.cc poster 28227](https://iclr.cc/virtual/2025/poster/28227)
- **A²OS²A (CVPR 2025).** Q = SN^b_Q(BN(XW_Q)) is binary, K = ReLU_K(BN(XW_K)) is full-precision, and V = SN^t_V(BN(XW_V)) is a ternary {−1, 0, 1} spiking neuron (Eqs. 22–27). A²OS²A(Q,K,V) = SN(Q·Kᵀ·V) (Eq. 28). The paper says no scale is needed "since K is a real-valued vector without boundaries", and the result can be negative.
  - Table 1 is against SDT, not Spikformer, on CIFAR10 / CIFAR100. For 2-256: 94.39 / 76.00 → 94.91 / 76.96. For 2-512: 95.51 / 78.83 (the text says 78.43) → 96.42 / 79.90.
  - No energy numbers were reported. No code repository was found.

  — [A²OS²A arXiv 2503.00226](https://arxiv.org/abs/2503.00226); [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Spiking_Transformer_Introducing_Accurate_Addition-Only_Spiking_Self-Attention_for_Transformer_CVPR_2025_paper.html)
- **α-XNOR / α-SSA (CVPR 2025).**
  - α-XNOR(x, y) = 1 if x = y = 1, α if x = y = 0, and 0 otherwise (Eq. 14). Sim(x, y) = Σ_i α-XNOR(x_i, y_i) (Eq. 18), defined pairwise for X, Y ∈ R^{n×d} (Eq. 19). α = 0 recovers the dot product, and α = 1 recovers XNOR.
  - α-SSA: Score = Sim_α-XNOR(Q, K) (Eq. 24) and Attn′ = (k·Score + b)V (Eq. 25). k and b are chosen from the firing rates so that E[Attn′] = 0 and V[Attn′] = 1, then Attn = SN(Attn′), with the float multiply merged into the neuron threshold.
  - Ablation (Table 3, CIFAR100, Spikformer, 9.32M): baseline 78.21; α=0 79.04; α=1 79.33; α=0.3 80.22.
  - ImageNet (Table 1): Spikformer-8-512 73.38 vs α-SSA-ViT-8-512 75.39.
  - Paper pages 5444–5454.

  — [α-XNOR CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.html)
- **SpikeVideoFormer / Spike-Driven Hamming Attention (ICML 2025).**
  - SDHA = SN(1/(2D)·(2Q_s−1)(2K_s−1)ᵀ V_s) (Eq. 9), or equivalently SN_{2D}((2Q_s−1)(2K_s−1)ᵀ V_s) (Eq. 10). It is derived from the normalised Hamming similarity f_H = ½ + (1/2D)(2q−1)ᵀ(2k−1) (Eq. 8). The map {0,1}→{−1,1} is done "through element-wise bit shifts".
  - Complexity remains O(ND²) because KᵀV can be computed first.
  - Ablation (Table 6), with PA-MPJPE↓ on MMHPSD and mIoU on Cityscapes: Hamming→dot product 39.8 → 45.7 (+5.9) and 73.1 → 65.9 (−7.8). Threshold scale s = 1/2D → 1/8 gives 46.7 (+6.9) and 68.5 (−4.6).
  - Code at JimmyZou/SpikeVideoFormer.

  — [SpikeVideoFormer arXiv 2505.10352](https://arxiv.org/abs/2505.10352); [PMLR v267](https://proceedings.mlr.press/v267/zou25b.html)
- **Max-Former (NeurIPS 2025).** It adds an "extra Max-Pool in patch embedding and Depth-Wise Convolution (DWC) in place of early-stage self-attention", and keeps SSA in the final stage. It does not change the attention computation itself. — [Max-Former arXiv 2505.18608](https://arxiv.org/abs/2505.18608); [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/956834836f36dd07df7064ff42ca69f2-Abstract-Conference.html); [GitHub](https://github.com/bic-L/MaxFormer)

### Inferences
- Three lines of work independently argue that the plain binary dot product QKᵀ with a fixed scale is a weak similarity for sparse spikes: α-XNOR (+2.01 on CIFAR100 over Spikformer), Hamming (dot product clearly worse on video tasks), and SSSA (SSA + dot product barely above baseline). In α-SSA, the affine normalisation k·Score + b alone gave +0.83 (α = 0, i.e. still the dot product). This suggests that the scale and offset before the attention-output LIF matter as much as the similarity measure. NSMT's fixed D^-0.5 scale is in the same weak position.
- With d = 4 or 8 channels per head in NSMT, the Hamming and α-XNOR similarities differ from the dot product much more than at d = 64. With few channels, "both zero" matches are a large share of the information. This is untested.
- STAtten's sequential-CIFAR setup (one time step = one sequence position) is the closest published analogue to NSMT's Neocortex path, where the time step is the token index. STAtten gained 1.6–3.2 points there over spatial-only SSA by attending across time-step blocks.

### Gaps
- Not covered in depth: STSA (Wang et al., IJCAI 2023, spatial-temporal self-attention), DISTA / DS2TA (arXiv), Quantized Spike-driven Transformer (ICLR 2025), and conversion-based softmax approximations (SpikeZIP-TF, ICML 2024; SpikedAttention, NeurIPS 2024). Their formulas were not read here.
- No official code repositories were found for SNN-ViT, A²OS²A, α-XNOR or MST. GitHub repository search by title keywords is not exhaustive.
- MST's attention-times-V step and its full inference time-step settings were not extracted.

## 7. Cross-attention compatibility, and which operations are binary or accumulate-only versus integer or real

### Takeaway
Every mechanism with separate Q and K/V projections allows Q to come from a different source than K/V, since none needs softmax: SSA, SDSA-3/4, DSSA, α-SSA, SDHA, A²OS²A and STAtten. Only SDSA-1, QKTA/QKCA and SSSA-V2 require token-aligned or same-shape inputs. Each also collapses attention to a per-token or per-channel gate. None of the softmax-free mechanisms produces a pure-spike attention map except DSSA's binary map. All others pass integer or real intermediates into an LIF. STMixer and SWformer have no query and cannot serve as cross-attention.

### Cited Findings
- Summary table. Equations, shapes and numbers come from the sources in Sections 1–6. The "cross-attention" column is my inference. Complexity entries marked * are my own derivation from the shapes, not a statement in the paper.

| Mechanism (venue) | Core op | Intermediates before LIF | Complexity | T usage | Q from another source? What flows to the output |
|---|---|---|---|---|---|
| SSA (ICLR'23) | SN(QKᵀV·s), s=0.125 | integers: QKᵀ ∈ [0,d], QKᵀV ∈ [0,N·d] | O(N²d) or O(Nd²) | per step; LIF across T | Yes (N_q ≠ N_kv fine). Output = V content routed to Q positions |
| SDSA-1 (NeurIPS'23) | Q ⊗ SN(Σ_tokens K⊗V) | integer column counts → binary 1×D gate | O(ND) | gate LIF across T | Requires K, V aligned. Output = Q spikes × channel gate |
| SDSA-2 (ICLR'24) | SN(Σ_tokens Q) ⊗ V | integer counts → binary 1×D gate | O(ND) | same | Gate from Q only. Output = V × channel gate (no K) |
| SDSA-3/4 (ICLR'24; E-SDSA TPAMI'25) | SN_{s·u_th}(QKᵀV) | integers (same as SSA) | linear in N (KᵀV first) | same | Yes. Same as SSA |
| QKTA / QKCA (NeurIPS'24) | SN(Σ_ch Q) ⊗ K / SN(Σ_tok Q) ⊗ K | integer counts → binary N×1 or 1×D gate | O(N)/O(D) per paper | per step | Requires the same N for Q and K. Output = K × per-token (or per-channel) gate |
| DSSA (CVPR'24) | SN(X Wᵀ Yᵀ c1), then SN(Map·Y W·c2) | sums of real weights (AC); binary N×N/p² map | O(N·N/p²·d)* | per step | Yes (DST defined for X ≠ Y). Output = projected Y routed by a binary map |
| α-SSA (CVPR'25) | SN((k·Sim_α + b)V) | real (α-weighted counts) | as SSA* | per step | Yes (defined for X, Y). Output = V routed |
| SDHA (ICML'25) | SN((2Q−1)(2K−1)ᵀV/2D) | signed integers | O(ND²) | per step; joint T·N tokens | Yes. Output = V routed with signed scores |
| A²OS²A (CVPR'25) | SN(Q·Kᵀ·V), K real ReLU, V ternary | real | as SSA* | per step | Yes. Output = V routed; K not a spike |
| SSSA-V2 (ICLR'25) | Θ(M_w Q′ − α V_th)·V, Q′ = per-token count | integer counts; lower-triangular mixing over T | O(2D+N) (paper) | M_w mixes all T steps | K absorbed into a learnable threshold scale. Output = V × per-token gate from Q |
| STAtten (CVPR'25) | LIF(Q[b]K[b]ᵀV[b]·α) over B-step blocks | integers | O(TND²) | attends across steps within a block | Yes. Output = V routed across tokens and steps |
| TIM (IJCAI'24) | α f(Q_TIM[t−1])KᵀV + (1−α)QKᵀV | real (α mix, conv) | as SSA* | 1-D conv over past steps on the Q path | Yes; history only on the Q side |
| STMixer (NeurIPS'24) | SN(W_STM V), W_STM ∈ R^{N×N} learned | real weights × spikes (AC) | O(N²D)* | per step | No query or key; fixed N; not cross-attention |
| SWformer (ECCV'24) | Haar-wavelet + conv token mixer | spikes incl. negative | conv-like | per step | No; attention-free |

- SpikingResformer's DST is explicitly defined on "dual spike inputs" X and Y of arbitrary shapes (Eqs. 7–8), so cross inputs are within its stated definition. — [SpikingResformer arXiv](https://arxiv.org/abs/2403.14302)
- α-XNOR similarity is defined for X, Y ∈ R^{n×d} as a pairwise matrix (Eq. 19). — [α-XNOR CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.html)
- QKTA and QKCA use the Hadamard product between an attention vector derived from Q and K. Both are N×D spike matrices, so they must share the token count. — [QKFormer arXiv Eq. 6](https://arxiv.org/abs/2403.16552)
- The SDSA official code multiplies q element-wise by a K⊗V summary over the token axis. — [ms_conv.py](https://raw.githubusercontent.com/BICLab/Spike-Driven-Transformer/main/module/ms_conv.py)
- STAtten notes that the non-softmax design is what "enables flexible reordering of Q, K, V operations". — [STAtten arXiv §4.4](https://arxiv.org/abs/2409.19764)

### Inferences
- For NSMT, the realistic drop-in candidates that keep "Hippocampus content routed by Neocortex queries" are SSA/SDSA-3 (what NSMT already has), DSSA (binary map plus derived scales), α-SSA (similarity plus affine normalisation), SDHA (signed similarity) and A²OS²A (real K, ternary V; less spike-pure).
- SDSA-1/2, QKTA/QKCA and SSSA-V2 change the semantics to a per-channel or per-token gate. They would turn the coupling into Neocortex-gated Hippocampus features without token routing. That is a different hypothesis, not a cheaper version of the same one.
- NSMT's Neocortex spikes are copied to the 4 Hippocampus steps, so Q is identical across T. Mechanisms whose novelty lies in mixing over T are partly neutralised on the query side: TIM's history on Q, SSSA's lower-triangular M_w, and STAtten's step blocks. STAtten-style blocks would still mix the 4 Hippocampus steps on the K/V side.
- Keeping NSMT's learned relative-position bias removes the linear-in-N advantage of every "linear" variant, and is not discussed by any of these papers. A spike-friendly position bias was not found in this family.

### Gaps
- No paper in this family evaluates a cross-attention (Q from a different network path). All of the above is extrapolation.
- No paper reports how these mechanisms behave at very small per-head d (4–8). Spikformer used 8–16 heads on DVS with D = 256 (d = 16–32).

## 8. Which mechanisms report results beyond ImageNet/CIFAR (sequence, audio, event, language, time series)

### Takeaway
Non-image evidence is thin:
- Audio: TIM on SHD (+1.2 over Spikformer).
- Sequential pixels: STAtten on sequential CIFAR (+1.6 to +3.2).
- Video: SpikeVideoFormer (Hamming attention clearly better than dot product).
- Event/action: SDT-V2 on HAR-DVS; TIM on UCF101-DVS and HMDB51-DVS; many papers on CIFAR10-DVS, DVS128 Gesture and N-Caltech101.
- Language: Spikingformer on GLUE, with softmax re-introduced.
- Time-series forecasting: only iSpikformer (ICML 2024), which uses plain SSA across variates (iTransformer tokens), not across time.

### Cited Findings
- TIM, SHD (spoken digits, 20 classes): Spikformer 85.1% vs TIM 86.3% (Table 2). It also reports UCF101-DVS 63.8% and HMDB51-DVS 58.6%. — [TIM arXiv](https://arxiv.org/abs/2401.11687)
- STAtten, sequential CIFAR10/100 with 32 column steps: gains of +3.16 / +1.58 over Spikformer and +3.09 / +3.22 over SDT (Table 2). — [STAtten arXiv](https://arxiv.org/abs/2409.19764)
- SpikeVideoFormer: Kinetics-400 classification, MMHPSD pose tracking, Cityscapes and VSPW segmentation. Hamming→dot-product ablation as in Section 6. — [SpikeVideoFormer arXiv](https://arxiv.org/abs/2505.10352)
- Spikingformer on GLUE: pretrained on Wikipedia-English, with softmax kept because it is "crucial for model convergence in language tasks". — [Spikingformer arXiv](https://arxiv.org/abs/2304.11954)
- SDT-V2 (Meta-SpikeFormer): HAR-DVS action recognition, COCO detection and ADE20K segmentation (Tables 2–4). E-SpikeFormer: detection, segmentation and neuromorphic action recognition. SNN-ViT: remote-sensing detection (NWPU VHR-10, SSDD; Table 3). — [SDT-V2 arXiv](https://arxiv.org/abs/2404.03663); [E-SpikeFormer arXiv](https://arxiv.org/abs/2411.16061); [SNN-ViT arXiv](https://arxiv.org/abs/2502.12677)
- iSpikformer (Lv et al., ICML 2024 per the camera-ready footer "Proceedings of the 41st ICML, Vienna, PMLR 235"):
  - Built on iTransformer, which "treats the independent time series as tokens"; the paper "follow[s] Spikformer to implement the spiking self-attention (SSA)".
  - Two Spikformer blocks, with the SSA neuron threshold set to 0.25.
  - Datasets: Metr-la, Pems-bay, Solar, Electricity.
  - Result: "iSpikformer achieves the lowest average RSE compared to all other methods and it nearly matches the R² performance of iTransformer, with only a marginal decrease of 0.001".
  - Code at microsoft/SeqSNN.

  — [Lv et al. arXiv 2402.01533](https://arxiv.org/abs/2402.01533); [GitHub](https://github.com/microsoft/SeqSNN)
- QKFormer, SpikingResformer, SWformer, STMixer, A²OS²A and α-XNOR report only image and neuromorphic-vision classification (CIFAR, ImageNet, CIFAR10-DVS, DVS128, N-Caltech101 and similar) in the parts read here. — [QKFormer](https://arxiv.org/abs/2403.16552); [SpikingResformer](https://arxiv.org/abs/2403.14302); [SWformer](https://arxiv.org/abs/2403.11138); [STMixer](https://www.proceedings.com/content/079/079017-4092open.pdf); [A²OS²A](https://arxiv.org/abs/2503.00226); [α-XNOR](https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.html)

### Inferences
- Spikingformer found that language needed softmax, and STAtten and TIM gained most on sequence-like inputs. Together these suggest that results from image-classification ablations, where SSA is roughly equal to softmax, may not transfer to sequence tasks with fine token-to-token dependence. That is the NSMT forecasting setting. This is an inference from different tasks, not a measured result.

### Gaps
- No paper in this family tests SSA, SDSA or any successor on ETT-style channel-independent patch forecasting, or with time step = token index on the query side.
- iSpikformer's SNN time step (T) value is cut by a page break in the extracted text: "The time step of SNNs is" followed by a page break, so T is unverified. Spike-TCN in the same paper uses T = 4.
