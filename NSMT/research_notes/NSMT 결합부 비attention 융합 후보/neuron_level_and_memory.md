# Neuron-level and memory-system (non-attention) ways to merge two spike streams: candidates for the NSMT Hippocampus–Neocortex coupling

Date: 2026-11-19.

**Scope.** This note covers SNN papers, plus a few non-spiking reference papers, in which two input streams meet inside one neuron (separate compartments or dendritic branches). It also covers papers in which one stream sets the other's gain, threshold, leak or initial membrane state. It adds hippocampus–neocortex (CLS) SNN couplings and an SNN time-series model that merges a sequential stream with a parallel stream.

**Not repeated here.**
- Cross-attention variants: `reports/NSMT 결합부 spiking attention 후보.md`.
- Memory neurons as single-stream upgrades of the Neocortex (TC-LIF, PMSN, DH-SNN psMNIST numbers, D-RF, BRF, TS-LIF forecasting numbers): `research_notes/NSMT query 구조 개선 제안/spiking_neuron_memory.md`. This note covers only their use as **two-input** neurons.
- CLS / fast–slow memory systems at large: `research_notes/NSMT query 구조 개선 제안/cls_objectives_readout.md`.

**Evidence labels.**
- [full text]: I read the paper (PDF, HTML, or the Europe PMC full-text XML).
- [code]: I read the official code.
- [abstract]: I saw only the abstract or the publisher record.
- [page]: I used a proceedings or publisher listing to confirm the venue.
- "Inference": my own reasoning about our model. None of it has been tested.

## 0. What a candidate must plug into (our facts)

These facts come from the attention report, the 11-08 probe and the code (`nsmt_flif_v1/classification/model.py`, `layers.py`, `energy.py`).

- **Hippocampus output.** h ∈ {0,1}^[T_h = 4, B', N, D = 64]. All 4 steps see the same token embedding, so per token the 4 steps carry a magnitude code: 0–4 spikes. A per-token summary costs additions only: c_i = Σ_t h[t, :, i, :].
- **Neocortex output.** n ∈ {0,1}^[N, B', 1, D]. Its membrane state is readable (`NeoNeuron.forward_charge` returns H − θ).
- **Neocortex neuron.** The population neuron has K = 4 members (τ 4/8/16/32) that feed one soma. `NeoNeuron.forward_branches` already accepts one input current per member, with shape [T, B, N', D, K]; it was added on 11-13 for the band tests.
- **Current fusion.** Each block takes q from the Neocortex and k, v from the Hippocampus, then applies x ← x·(1 − y) (the IAND gate). The Neocortex receives only half of the task gradient through the fusion: `neo_grad = 0.5`, implemented as `mx = g*mx + (1-g)*mx.detach()`.
- **Measured symptoms** (11-08, query pop LIF before membrane shortcut, ETT):
  - The last Neocortex layer fires 2–4 % of the time.
  - The first token never fires. It would need 8.6–10 units of current, while the BN'd currents have mean ≈ 0 and SD ≈ 1.
  - At 70–83 % of (row, token) positions the whole Neocortex vector is 0.
  - Zeroing the Neocortex at test time changes MSE by < 1 %.
- **Adapters every candidate needs (inference).**
  - Hippocampus → token axis: use c_i, or the Hippocampus membrane at step 4.
  - Neocortex → Hippocampus steps: broadcast n_i (or u_i) over the 4 steps. The q path already does this.
- **Order caveat (inference).** The Hippocampus attends over the whole window. If Hippocampus information enters the Neocortex at token i, then Neocortex step i can see future tokens. Memory-off and token-shuffle controls must then act on the Neocortex's own token input and state, not on the Hippocampus side.
- **Energy convention (`energy.py`).**
  - One LIF update per step costs [(2 + 2ρ)κ + (2 + ρ)]·E_AC, with κ = 4.6/0.9 ≈ 5.11. At low firing that is about 2 MAC + 2 AC.
  - One extra real-valued multiply per neuron-step (a gain) is +1 MAC, about +42 % of that neuron's update cost. My arithmetic: κ / (2κ + 2) = 5.11 / 12.22.
  - Per Hippocampus block, that is 4 × N × 64 extra multiplies: 6,144 at N = 24 and 46,080 at N = 180. Per Neocortex layer it is N × 64 (1,536 at N = 24).
  - Additive routing adds only ACs.

## 1. Summary table

Fit and forcing are inferences. "Forces use" asks whether the mechanism makes the output depend on the Neocortex, as opposed to merely letting it.

| # | Method (paper, venue, year; check level) | Where the two streams meet | Spike-driven? | Strongest evidence (paper-measured) | Fit to our shapes | Forces Neocortex use? |
|---|---|---|---|---|---|---|
| M1 | **cLIF apical multiplicative modulation** (Ferrand et al., ICANN 2023, LNCS 14254; full text + code) | Basal current × ReLU(apical membrane) inside one LIF | Spike inputs, but one real multiply per neuron-step | Same network, 5 seeds: multiplicative 89.4 vs additive 80.4 vs ALIF 75.9 % (DVS); sequence memory: multiplicative 83.9 % vs additive 51.1 % (chance 50) | Token-aligned; either direction | **Strong**: multiplicative. With the Neocortex as the output soma it cannot be bypassed |
| M2 | **Context-sensitive two-point neuron (CS-TPN)** (Raza & Adeel, arXiv 2408.11019, 2024; **arXiv only**; full text) | Is + Ic·(0.1 + \|Is\|) at the soma | Real multiply | Spiking XOR learns faster with fewer neurons; AV speech enhancement PESQ 1.58 vs 1.40 at 0 dB (non-spiking CNN) | Token-aligned | Medium: identity when context = 0 |
| M3 | **Burst multiplexing** (Payeur et al., Nat Neurosci 2021; Naud & Sprekeler, PNAS 2018; Capone et al., PNAS 2023; bioRxiv full text / abstract) | Event rate = feedforward, burst probability = feedback (apical) | Spikes | A burst-dependent learning rule reaches near-backprop CIFAR-10 error (20.1 %) | 4 Hippocampus steps could carry event vs burst | Weak: designed for learning signals |
| M4 | **Dendritic NMDA gain modulation** (Wybo et al., PNAS 2023; full text) | Per-neuron gain g_t on a shared feedforward drive | Rate abstraction; spiking demo | Shared weights + per-task gains solve 48 tasks; matches task-specific readouts on 47-task EMNIST | Supports gain (not bias) as the fusion form | Theory support for M1 / M9 |
| M5 | **Active dendrites, context gating** (Iyer et al., Front Neurorobot 2022, non-spiking; Pes et al., AICAS 2024, TTFS SNN; full text) | (wᵀx + b)·σ(max_j u_jᵀc); TTFS: context delays spike time | No (Iyer) / yes (Pes) | MT10 87.5 vs 76.6 %; Split MNIST TTFS 88.3 % vs ≈ 69.4 % without dendrites | Needs a continuous context in place of a task ID | Medium |
| M6 | **DH-SNN branch routing** (Zheng et al., Nat Commun 2024; full text) | Each input stream on its own dendritic branch with its own τ; soma sums | **Yes** (AC synapses) | Two signals on two branches beat one branch and vanilla (< 75 %) on multi-timescale XOR | **Direct**: our pop neuron already has 4 branches and a per-member input hook | Weak (additive) |
| M7 | **TS-LIF two-compartment neuron used with two inputs** (Feng et al., ICLR 2025; full text) | Dendrite ↔ soma coupled by learnable β; both compartments spike; κ mixes them | Mostly; κ-mix is real-valued | Forecasting gains over LIF-based SNNs with one input (prior note); +0.002–0.006 M params | Dendrite ← Neocortex, soma ← Hippocampus count; runs over N | Weak to medium |
| M8 | **DendSN dendritic branch gating (DBG)** (Huang et al., arXiv 2412.06355, 2024/2025; **arXiv only**; full text) | Branch strengths κ_b gated by a top-down signal | AC synapses; K multiplies | pMNIST 50 tasks: LIF 23.73 %, LIF + EWC 35.01 %, DendSNN + dEWC + DBG > 42 % | Hippocampus count could choose which Neocortex τ-branch drives the soma | Medium (multiplicative, tiny) |
| M9 | **Membrane-potential attention gating (MA-SNN)** (Yao et al., IEEE TPAMI 2023; arXiv-v1 full text) | Sigmoid weights multiply the membrane potential before firing | Real multiplies | DVS Gesture 90.63 → 96.53 % with spikes −84.9 %; Res-SNN-34 ImageNet 64.13 → 69.15 % | Gate from Neocortex on Hippocampus membrane, per token × channel | Medium to strong |
| M10 | **Context-set threshold / time constant** (Hammouamri et al., TMLR 2022; Luca et al., Front Neurosci 2026; abstract + code / full text) | θ (or τ) = base + p·context | Additions | Recurrent: tracking tolerance ≈ 600 vs ≈ 100 krad; feedforward: **no gain** | Hippocampus-side θ shift ≈ additive bias | Weak |
| M11 | **Liquid time-constant spiking neuron (LTC-SN)** (Yin, Corradi, Bohté, Nat Mach Intell 2023; full text) | τ_m⁻¹ = σ(W[x_t, u_{t−1}]) | Real multiply + sigmoid | PS-MNIST 94.77 %, DVS Gesture 97.22 % (online FPTT); BPTT **diverged** on Add | Hippocampus count sets the Neocortex leak per token | Medium; the interaction lives inside T = N |
| M12 | **State injection: ANN→SNN membrane init + learnable initial membrane potential (IMP)** (Aydin et al., CVPRW 2024; Shen et al., NeurIPS 2024; full text) | One network writes the other's membrane at t = 0 | Init: real; IMP: free at inference | Zero init 9.12 px vs learned init 6.11 px; IMP +0.2 to +1.0 pt (single runs) | Fixes the measured "first token never fires" | Weak by itself, but a precondition |
| M13 | **CLS couplings in SNNs** (CH-HNN, Shi et al., Nat Commun 2025, full text; D'Alba et al., Neurocomputing 2025, abstract) | ANN → binary masks on SNN hidden neurons; apical (DIT) modulation during replay | Masks: yes | Episode inference alone ≈ full model (70.70 vs 70.72 %) | Mask = our old AND mask (already ≈ null on ETT) | Weak in our setting |
| M14 | **Gated sequential + parallel fusion for SNN forecasting (SpikeSTAG DSF)** (Hu et al., arXiv 2508.02069, 2025; **arXiv only**; full text) | G = σ(W[H_seq; H_par]); G⊙H_seq + (1 − G)⊙H_par | No (LSTM and gate are real-valued) | Gate vs concat: R² +0.009 to +0.021, RSE −0.005 to −0.037 on 4 datasets | Token-aligned convex gate; mean G = Neocortex share | Weak (G → 0 is allowed), but a good diagnostic |

## 2. Methods

### A. Two-compartment / apical-gain neurons

#### M1. cLIF: contextual LIF with multiplicative apical modulation (cSNN)

1. **Citation.** Romain Ferrand, Maximilian Baronig, Thomas Limbacher, Robert Legenstein, "Context-Dependent Computations in Spiking Neural Networks with Apical Modulation", ICANN 2023, LNCS vol. 14254, pp. 381–392, Springer. DOI 10.1007/978-3-031-44207-0_32.
   - Peer-reviewed conference. ICANN is not on the preferred list.
   - I read the author preprint on Zenodo (record 11489556). Its footnote calls the ICANN chapter the version of record.
   - Code: `IGITUGraz/CSNN`, file `module/clif.py`.
   - Check level: [full text] [code].
2. **Mechanism** (Eqs. 1–2 of the preprint; matches `clif.py`).
   - Apical leaky integrator: V^a_j(t + Δt) = α V^a_j(t) + (1 − α) R_m I^a_j(t + Δt), with α = exp(−Δt/τ_a).
   - Soma: V_j(t + Δt) = β V_j(t) + (1 − β) R_m I_j(t + Δt) · ReLU(V^a_j(t + Δt)) − V_th s_j(t), with reset by subtraction.
   - Currents: I^a = W^{a,in} c + W^{a,rec} s + b^a, where c is the context spike input; I = W^{in} x + W^{rec} s + b.
   - Parameters: V_th = 0.05, τ_m = 20 ms, Δt = 1 ms.
   - The code offers other interaction functions: `relu_mult` s·ReLU(a) (the paper), `tanh_mult` s·(1 + tanh(a)), `add`, `relu_add`, `tanh_add`.
   - Additive baseline in the paper: V ← βV + (1 − β)R_m I + V^a − V_th s.
   - The paper derives f/I curves. Apical input both lowers the rheobase and raises the f/I gain. Additive apical input only lowers the rheobase.
   - **Spike-driven?** Synaptic inputs are spikes (AC). The gain needs one real multiply per neuron-step.
3. **Evidence** (paper-measured; 200 recurrent cLIF neurons, BPTT, mean ± SD over n = 5 runs).
   - Single-sample context task. The context is a class cue as 200 Hz Poisson on 1 of K context neurons; the output is "is the input of the cued class?":

     | Model | DVS Gesture | SHD |
     |---|---|---|
     | cSNN (multiplicative) | 89.4 ± 1.0 | 87.8 ± 0.77 |
     | additive cSNN | 80.4 ± 2.2 | 72.2 ± 2.0 |
     | LSNN (ALIF) | 75.9 ± 1.4 | 66.1 ± 1.7 |

     cSNNs also learned clearly faster, with learning rates tuned separately for each model.
   - Sequence classification (Table 1), accuracy / precision:

     | Model | Apical recurrence | DVS | SHD |
     |---|---|---|---|
     | cSNN | yes | 95.1 / 88.5 | 95.9 / 89.4 |
     | cSNN | no | 93.3 / 82.5 | 93.08 / 80.8 |
     | additive cSNN | yes | 86.9 / 64.3 | 84.5 / 59.0 |
     | LSNN | – | 84.8 / 58.4 | 78.4 / 47.8 |

   - Sequence memory (Table 2; chance = 50 %), DVS / SHD:

     | Model | Apical recurrence | τ_a | DVS | SHD |
     |---|---|---|---|---|
     | cSNN | yes | U(20, 200) ms | 83.9 ± 2.9 | 75.7 ± 2.5 |
     | cSNN | yes | 200 ms | 76.5 | 66.6 |
     | cSNN | yes | 20 ms | 74.7 | 66.2 |
     | cSNN | no | U(20, 200) ms | 59.6 | 53.1 |
     | additive cSNN | yes | U(20, 200) ms | 51.1 | 55.8 |
     | LSNN | – | – | 51.9 | 51.0 |

     The authors stress two findings: τ_a diversity mattered (7–9 points), and "the additive variant with apical recurrence performed close to chance level".
   - The authors' explanation: with multiplicative integration, "gradients for the apical compartment are influenced directly by the somatic activation, which is not the case in the additive case".
4. **Cost.** An apical weight matrix (context dim × neurons), an optional apical recurrent matrix, one apical state per neuron, and one multiply per neuron-step. No energy figures are reported.
5. **Fit to our shapes and forcing (inference).** Both directions are token-aligned.
   - **Direction A: the Neocortex as the pyramidal cell.**
     - Basal input = the Neocortex's own token current, as now.
     - Apical input = the Hippocampus per-token count c_i through W_a, which is an AC on counts.
     - The apical leaky integrator runs over tokens. Its τ_a could reuse the population spread 4/8/16/32; the paper's τ_a-diversity result supports this.
     - The head (or the next block) reads the Neocortex soma, so every route from the Hippocampus to the output passes through the T = N dynamics. This is the only candidate in this note that **cannot be bypassed**.
     - Risk: if the sequential bottleneck loses information, MSE rises. That result is informative in itself.
   - **Direction B: the Hippocampus as the pyramidal cell.** This replaces x·(1 − y) and/or the cross-attention.
     - Hippocampus attention/MLP output neurons keep their current I_h.
     - Their apical input is the Neocortex **membrane** u_neo,i (not its spikes), broadcast over the 4 steps.
     - Use g = 1 + tanh(W_a u) as the safe default (identity when the Neocortex is silent), or ReLU(W_a u) as in the paper (the Hippocampus stays silent without Neocortex drive).
     - The membrane is non-zero at the 70–83 % of positions where Neocortex spikes are all 0. No attention score can use those positions.
   - Both directions are multiplicative. The gradient into the Neocortex is scaled by the Hippocampus drive, instead of passing through the tiny q·k content term that the attention report measured.
6. **Concerns.**
   - The paper's tasks *require* the context: the label cannot be computed without it. That is the best case for multiplicative gating. On ETT the Hippocampus alone already matches the base.
   - Only 1 recurrent layer of 200 neurons; no deep-network, transformer or time-series test; V_th = 0.05 is far from our threshold of 1.
   - The ReLU form plus a silent Neocortex can kill the Hippocampus and collapse training.
   - The multiply breaks "AC only": about +42 % of a LIF update in each modulated layer (§0).

#### M2. Context-sensitive two-point neurons (CS-TPN), spiking version

1. **Citation.** Mohsin Raza, Ahsan Adeel, "An Overlooked Role of Context-Sensitive Dendrites", arXiv 2408.11019 (v1, 20 Aug 2024). **arXiv only**; no venue is listed.
   - The same group's "Cooperator" (arXiv 2305.10449) is the RL/transformer version; I did not open it.
   - Check level: [full text, arXiv HTML].
2. **Mechanism.**
   - Somatic drive: Mod(I_s, I_c) = I_s + I_c·(0.1 + |I_s|) (Eq. 13).
   - With a "universal" context U: + I_c·I_u·(2 + |I_s|) (Eq. 14).
   - The apical context C integrates feedback, proximal, distal and universal context. It amplifies coherent feedforward input and attenuates conflicting input.
   - Spikes: V_s crosses a dynamic threshold (−50 mV, +2 mV per spike, decaying with 27 ms). A spike becomes a singlet or a burst according to feedback: p = σ(βu + α).
   - Real multiplies.
3. **Evidence** (paper-measured).
   - Spiking XOR (burst-dependent plasticity, BDSP):
     - CS-TPN + BDSP learns faster than BDSP alone with the same 875 neurons; the version with U uses 800.
     - With a population of 200: cost 1.0 from epoch 120, versus cost 1.5 at epoch 250 for the standard network.
     - Context-insensitive TPNs fire about 3× more often.
   - MACs after training: 52,369 versus 1,161,313. CS-TPN uses 1.394× more MACs *during* training.
   - A 50-layer non-spiking AV speech-enhancement CNN: PESQ 1.33 / 1.58 / 1.77 versus 1.23 / 1.40 / 1.52 at −6 / 0 / 6 dB.
   - The text claims "up to 330x fewer neurons", which no table directly supports.
4. **Cost.** One context pathway per neuron, about 2 multiplies per neuron-step.
5. **Fit (inference).** This is an algebraic alternative to M1's gain.
   - It is the identity when I_c = 0, which is safe for a silent Neocortex.
   - Context alone gives a small drive (0.1·I_c), so the Neocortex can act even where the Hippocampus current is 0.
   - The interaction term I_c·|I_s| scales the Neocortex effect with the Hippocampus drive.
   - Plug in I_s = Hippocampus current and I_c = W_a u_neo,i.
6. **Concerns.** Single group; arXiv only. Most quantitative results come from non-spiking deep models. No time series.

#### M3. Burst multiplexing of feedforward and feedback signals (BurstProp family)

1. **Citations.**
   - Alexandre Payeur, Jordan Guerguiev, Friedemann Zenke, Blake A. Richards, Richard Naud, "Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits", *Nature Neuroscience* 24, 1010–1019 (2021). DOI 10.1038/s41593-021-00857-x. An author correction (DOI 10.1038/s41593-021-00970-x) fixed figure-6c labels. Check level: [full text, bioRxiv v2, partly read].
   - Background: Richard Naud, Henning Sprekeler, "Sparse bursts optimize information transmission in a multiplexed neural code", *PNAS* 115 (2018). DOI 10.1073/pnas.1720995115. Check level: [Crossref record only].
   - Cristiano Capone, Cosimo Lupo, Paolo Muratore, Pier Stanislao Paolucci, "Beyond spiking networks: the computational advantages of dendritic amplification and input segregation", *PNAS* 120, e2220743120 (2023). DOI 10.1073/pnas.2220743120; arXiv 2211.02553. Check level: [abstract].
2. **Mechanism.**
   - Two-compartment pyramidal neurons: bottom-up input reaches the soma; top-down input reaches the apical dendrite as I_d.
   - A somatic event becomes a burst with probability f(V_d), a sigmoid of apical depolarization.
   - Across an ensemble, the event rate e(t) carries the feedforward signal and the burst probability p(t) = b(t)/e(t) carries the feedback signal. These are the paper's words.
   - Capone et al.: three compartments. Coincidence of basal and apical input produces high-frequency bursts, used for target-based learning.
   - Spike-driven at inference.
3. **Evidence** (paper-measured). All results are about *learning* (credit assignment), not inference-time fusion.
   - MNIST ≈ 1.1 % test error.
   - CIFAR-10: 20.1 % test error after 400 epochs, similar to backprop.
   - ImageNet: 56.1 % top-5 error, better than fixed feedback weights.
   - Capone: abstract-level claims only (3-D trajectory store and recall; hierarchical imitation).
4. **Cost.** A dendritic state and burst logic per neuron. No energy figures.
5. **Fit (inference).**
   - Our 4 Hippocampus steps per token leave room for a two-level code: an event (≥ 1 spike, driven by the Hippocampus) and a burst (≥ 2 spikes, with the probability set by a Neocortex apical input).
   - A downstream Linear on spike counts would read both.
   - But the 4 steps already encode Hippocampus magnitude (0–4 spikes), so the two codes would compete for the same channel.
6. **Concerns.** No deep-learning SNN uses this for two-stream fusion at inference. The mechanism is tied to burst-dependent plasticity. **Low priority.**

#### M4. Dendritic NMDA gain modulation for multitask learning

1. **Citation.** Willem A. M. Wybo, Matthias C. Tsai, Viet Anh Khoa Tran, Bernd Illing, Jakob Jordan, Abigail Morrison, Walter Senn, "NMDA-driven dendritic modulation enables multitask representation learning in hierarchical sensory processing pathways", *PNAS* 120(32), e2300558120 (2023). DOI 10.1073/pnas.2300558120; PMC10410730. Check level: [full text, Europe PMC XML].
2. **Mechanism.**
   - Context inputs (NMDA synapses on dendritic branches) both raise the slope and lower the threshold of the feedforward input–output curve.
   - Reduced layer form: y = σ(g_t ⊙ (W x − x_shift) + b), with task-specific gains g_t and shared W, x_shift and b.
   - Somatic shunting inhibition was "better fitted by pure gain modulation than bias modulation".
   - The rate abstraction uses real multiplies. A spiking demonstration uses a reduced L5 pyramidal-cell model.
3. **Evidence** (paper-measured).
   - Networks with one shared set of feedforward weights and task-specific neuron gains "solved all 48 tasks" (2-D binary tasks).
   - Multitask EMNIST (47 one-vs-all tasks): independent task-specific gain and bias reached the same performance as task-specific readouts. Gain plus a shared x-shift was slightly lower.
   - Spiking demonstration: 100 hidden neurons with 40 dendritic compartments each. Plastic AMPA + NMDA context synapses learn multitask EMNIST under a global error signal.
4. **Cost.** One gain per neuron per context.
5. **Fit (inference).** This is theoretical and biophysical support for choosing a **gain** (M1, M9) over a bias or threshold (M10) as the form of Neocortex → Hippocampus influence.
   - The Neocortex would act as a per-window context that reweights a shared Hippocampus feature basis.
6. **Concerns.** The context is a discrete task ID with *learned per-task* gains, not a gain computed from another stream. Not tested on time series.

### B. Dendritic branch routing and gating

#### M5. Active dendrites with context segments (non-spiking) and its TTFS spiking variant

1. **Citations.**
   - Abhiram Iyer, Karan Grewal, Akash Velu, Lucas Oliveira Souza, Jeremy Forest, Subutai Ahmad, "Avoiding Catastrophe: Active Dendrites Enable Multi-Task Learning in Dynamic Environments", *Frontiers in Neurorobotics* 16, 846219 (2022). DOI 10.3389/fnbot.2022.846219; arXiv 2201.00042. **Non-spiking.** Check level: [full text, PMC, partly read].
   - Lorenzo Pes, Rick Luiken, Federico Corradi, Charlotte Frenkel, "Active Dendrites Enable Efficient Continual Learning in Time-To-First-Spike Neural Networks", IEEE AICAS 2024. DOI 10.1109/AICAS59952.2024.10595872; arXiv 2404.19419. Check level: [full text, arXiv HTML].
2. **Mechanism.**
   - Iyer: ŷ = (wᵀx + b) × σ(d), with d = the dendritic segment response u_jᵀc of largest absolute value (sign kept), followed by k-WTA sparsity. Here c is a context vector.
   - Pes (TTFS): the context shifts spike *time*, t̃ = t + S/(1 + e^u) with S = 4. A delay large enough to push a neuron past the window makes it a "dead neuron", which works as a gate. This step is spike-driven.
3. **Evidence** (paper-measured).
   - Iyer, Meta-World MT10: 87.5 % vs 76.6 % (MLP with context input, 10 runs); best 5 runs: 95.6 vs 88.2 %.
   - Iyer, permuted MNIST: 94.6 % (10 tasks) and 81.4 % (100 tasks); with SI, 97.2 / 91.6 %.
   - Pes, Split MNIST: 88.3 % with dendrites vs about 69.4 % sequential without them (derived from the reported gaps; 5 seeds).
   - Pes, FPGA: 80.0 % at 37.3 ms per image.
4. **Cost.** Iyer: one segment bank (n_segments × context dim) per neuron. Pes: 8-bit dendritic delays.
5. **Fit (inference).**
   - Replace the task ID by a continuous Neocortex state: d_i = max_j u_jᵀ u_neo,i, then gate the Hippocampus features with σ(d_i). The max over segments works like choosing among a few learned "context prototypes".
   - The TTFS variant maps onto our 4 steps: the step at which a Hippocampus neuron first fires already encodes magnitude, so a Neocortex-driven delay would trade magnitude for context.
6. **Concerns.** In both papers the context is a given task ID. Iyer is non-spiking. k-WTA is not used in our model.

#### M6. DH-SNN: temporal dendritic heterogeneity with input routing to branches

1. **Citation.** Hanle Zheng, Zhong Zheng, Rui Hu, Bo Xiao, Yujie Wu, Fangwen Yu, Xue Liu, Guoqi Li, Lei Deng, "Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics", *Nature Communications* 15, 277 (2024). DOI 10.1038/s41467-023-44614-z; PMC10766638.
   - Crossref lists the first 8 authors; the last author, Lei Deng, is from memory and was not re-checked.
   - Check level: [full text, Europe PMC XML].
2. **Mechanism.**
   - Each dendritic branch d keeps a current i_d that decays by its own learnable timing factor α_d and integrates only **its own disjoint subset** of presynaptic inputs.
   - The soma integrates the sum of the branch currents. Only the soma is reset, so the branches keep memory across spikes.
   - Spike-driven (AC). Each branch adds one decay multiply per step.
3. **Evidence** (paper-measured).
   - Multi-timescale spiking XOR: Signal 1 (low frequency) goes to Branch 1 and Signal 2 (high frequency) to Branch 2.
     - The vanilla SFNN scores "much lower than 75 %"; a one-branch DH-SFNN cannot handle both timescales.
     - Two branches do much better, especially with a "beneficial initialization": a large α for the slow signal's branch.
     - Learned α values move to match each branch's signal timescale. Exact accuracies are only in Fig. 4b.
   - On hardware (TianjicX), the connection-pattern restriction cost ≤ 0.3 % accuracy.
   - Sparse branch connectivity means "no obvious" extra storage or compute compared with vanilla SNNs.
   - psMNIST and SHD numbers are in the prior note.
4. **Cost.** One state and one decay per branch. Total synapse count unchanged.
5. **Fit (inference).** Our population neuron is already 4 branches (τ 4/8/16/32) + soma, and `forward_branches` takes one current per member.
   - Route the Neocortex token current to the slow members (τ 16/32) and the Hippocampus count c_i to the fast members (τ 4/8), or to a fifth member.
   - The soma then integrates both streams over tokens (T = N), and the Neocortex output carries both.
   - This is the cheapest neuron-level fusion and the natural **additive control** for M1.
6. **Concerns.**
   - Additive: the model can shrink one branch's weights, and Ferrand's additive cSNN was much worse whenever the context was needed.
   - Order caveat (§0): the Hippocampus branch brings in future tokens.
   - The XOR evidence uses two signals that differ in timescale. Ours differ in source, not obviously in timescale.

#### M7. TS-LIF (two-compartment, forecasting) used as a two-input neuron

1. **Citation.** Shibo Feng, Wanjin Feng, Xingyu Gao, Peilin Zhao, Zhiqi Shen, "TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting", ICLR 2025; arXiv 2503.05108; code `kkking-kk/TS-LIF`. Check level: [full text, arXiv HTML] [page, ICLR proceedings].
2. **Mechanism** (Eqs. 5–6). All coefficients are learnable; v_th = 1.
   - Dendrite: v_d[t] = α₁ v_d[t−1] + β₁ v_s[t−1] + (1 − α₁) c[t] − γ₁ s_d[t−1].
   - Soma: v_s[t] = α₂ v_s[t−1] + β₂ v_d[t] + (1 − α₂) c[t] − γ₂ s_s[t−1]. The (1 − α₂) c[t] term is the direct somatic injection.
   - Spikes: s_d = H(v_d − v_th), s_s = H(v_s − v_th); output s_mix = κ s_d + (1 − κ) s_s.
   - Stability needs the eigenvalues of the coupled system inside the unit circle.
   - **In the paper both compartments receive the same c[t].**
3. **Evidence** (paper-measured, from the prior note): Electricity H96 RSE: TS-former 0.261 vs iSpikformer 0.348 (iTransformer 0.335).
   - Overhead (Table 6, Electricity): +0.004 / +0.002 / +0.006 M params for TCN / GRU / former.
   - Energy: 0.25 vs 0.23, 1.58 vs 1.51, and 3.22 vs 3.19 mJ for the same three backbones.
   - There is no ablation of the somatic injection or the dual spikes.
4. **Cost.** Two states per neuron and a few multiplies per step. The κ-mix is real-valued unless it is folded into the next layer's weights.
5. **Fit (inference).**
   - Dendrite ← Neocortex token current (memory: α₁ large, low-pass); soma ← Hippocampus count c_i (direct injection); learned β₁, β₂.
   - It runs over N tokens and holds both streams' dynamics in one neuron type.
   - The paper's frequency-separation example (α₁ = 0.95, α₂ = 0.05) matches the split "Neocortex = slow, Hippocampus = fast".
6. **Concerns.**
   - Two-stream use is untested.
   - β₂ → 0 and κ → 0 let the model ignore the dendrite, as with any additive coupling.
   - It would replace our population neuron rather than extend it.

#### M8. DendSN: dendritic branch gating by a top-down signal

1. **Citation.** Yifan Huang, Wei Fang, Zhengyu Ma, Guoqi Li, Yonghong Tian.
   - v1 title: "Flexible and Scalable Deep Dendritic Spiking Neural Networks with Multiple Nonlinear Branching" (Dec 2024).
   - v2 title: "Scalable Dendritic Modeling Advances Expressive and Robust Deep Spiking Neural Networks" (22 Dec 2025).
   - arXiv 2412.06355. **arXiv only**: I found no venue.
   - Check level: [full text, arXiv HTML v2].
2. **Mechanism.**
   - P leaky dendritic compartments: V^d_i[t] = α V^d_i[t−1] + X_i[t].
   - Each branch b applies an affine transform and an L2 norm: Y_b = ψ(‖(V^d − ξ)/ζ_b‖), where ψ is a Mexican-hat wavelet or the identity.
   - Soma input: Z = Σ_b κ_b Y_b, with κ_b the branch strengths; the soma is a LIF.
   - DBG: the task index q fixes a random Bernoulli(ρ) mask, K* = K ⊙ DBG(q). The same mask is used in training and inference.
3. **Evidence** (paper-measured).
   - pMNIST, 50 tasks: LIF 23.73 %; LIF + full-network EWC 35.01 %; DendSNN + decoder EWC + DBG "above 42 %"; best ρ = 0.1–0.2.
   - Throughput is about 0.92× of LIF with Triton kernels; memory about +12 % with checkpointing.
4. **Cost.** About +9 parameters per neuron in their L5PC fit; K multiplies per neuron-step for κ_b.
5. **Fit (inference).** Make κ_b depend on the Hippocampus: Z_i = Σ_b κ_b(c_i) Y_b(Neocortex).
   - At each token, the Hippocampus then chooses which timescale branch of the Neocortex memory drives the soma.
   - This is a multiplicative, content-dependent timescale selection inside the T = N neuron. It costs only K = 4 multiplies per neuron-step.
6. **Concerns.** DBG's masks are fixed random per-task masks, not input-dependent, so the input-dependent version is untested. arXiv only.

### C. Neuromodulation: gain, threshold, leak

#### M9. Multi-dimensional attention on the membrane potential (MA-SNN)

1. **Citation.** Man Yao, Guangshe Zhao, Hengyu Zhang, Yifan Hu, Lei Deng, Yonghong Tian, Bo Xu, Guoqi Li, "Attention Spiking Neural Networks", *IEEE TPAMI* 45(8), 9393–9410 (2023). DOI 10.1109/TPAMI.2023.3241201; arXiv 2209.13929.
   - Check level: [full text of the arXiv "under review" version via ar5iv] [page, Crossref]. The published numbers may differ.
2. **Mechanism** (Eq. 14, TCSA).
   - Temporal attention on the input: X_TA = g_t(X) ⊙ X.
   - Channel attention on the integrated membrane: U_CA = g_c(H + X_TA) ⊙ (H + X_TA).
   - Spatial attention: U = g_s(U_CA) ⊙ U_CA; then the neuron fires from U.
   - The g's are sigmoid outputs of small FC or conv layers. The paper counts the extra cost as ΔMAC.
   - This is *not* query–key attention. It is a squeeze-and-excite style gain on the membrane.
3. **Evidence** (paper-measured).
   - DVS128 Gesture, 3-layer: 90.63 → 96.53 %, spikes −84.9 %, energy efficiency 3.4×.
   - Gait: 87.59 → 92.29 %, spikes −81.6 %, 3.2×.
   - ImageNet, T = 1: Res-SNN-18 61.70 → 63.97 %; Res-SNN-34 64.13 → 69.15 %; Res-SNN-104 71.57 → 73.82 %.
   - Location study: channel attention on the integrated membrane potential was best (93.88 %, lowest spike rate).
4. **Cost.** Small FC or conv layers; for example, CBAM-SNN adds 4,608 parameters on the 3-layer Gesture net. One multiply per gated membrane element.
5. **Fit (inference).** Cross-stream version: g_i = σ(W n_i) or σ(W u_neo,i), computed from the Neocortex, multiplies the Hippocampus membrane at token i on all 4 steps (per channel).
   - This is M1-B without a separate compartment.
   - MA's large spike reduction suggests a side benefit: the Neocortex could switch off Hippocampus spikes where they do not matter.
6. **Concerns.**
   - In MA the gate comes from the *same* stream; the cross-stream use is untested.
   - Real multiplies.
   - The numbers come from the arXiv text, not the TPAMI version.

#### M10. Context-set firing threshold or time constant

1. **Citations.**
   - Ilyass Hammouamri, Timothée Masquelier, Dennis G. Wilson, "Mitigating Catastrophic Forgetting in Spiking Neural Networks through Threshold Modulation", *TMLR* 2022 (OpenReview id 15SoThZmtU); code `Thvnvtos/Nm-SNN`. Check level: [abstract] [code README].
   - Sarah Luca, T. Patrick Xiao, Frances S. Chance, Sapan Agarwal, Corinne Teeter, G. William Chapman, "Learned adaptive properties for mitigation of weight perturbations in embedded spiking networks", *Frontiers in Neuroscience* 20 (2026). DOI 10.3389/fnins.2026.1766765. Check level: [full text, first 100k characters].
2. **Mechanism.**
   - Hammouamri: an *evolved* neuromodulatory network sets each layer's thresholds from the previous layer's spiking activity.
   - Luca: θ_j = θ₀ + p_j·c (Eq. 4) or τ_j = τ₀ + p_j·c (Eq. 5), with a scalar trial context c ∈ [0, 1] and learnable p_j.
   - Both need additions only.
3. **Evidence** (paper-measured).
   - Hammouamri: much less forgetting than a fixed-threshold SNN (abstract; numbers not read).
   - Luca, 10 initialisations, small-object tracking with radiation-perturbed weights: a context-adapted *recurrent* network kept sub-pixel error up to about 600 krad; without context, error was tolerated only to about 100 krad.
   - Luca, *feedforward* networks: "neither threshold nor time-constant adaptation significantly improved accuracy".
4. **Cost.** Negligible.
5. **Fit (inference).**
   - A Neocortex-driven threshold shift on Hippocampus neurons, θ_i = θ₀ − W_θ n_i, behaves almost like an extra bias current. That makes it the *additive* family: Ferrand's f/I analysis says additive apical input only lowers the rheobase.
   - Luca's recurrent-versus-feedforward result is the useful part. It suggests modulation should act inside the recurrent T = N Neocortex (M11, M8), not on the feedforward Hippocampus, when the goal is to make dynamics matter.
6. **Concerns.** Neither paper is about fusion accuracy (continual learning; hardware robustness). The context is a scalar or an evolved signal.

#### M11. Liquid time-constant spiking neuron (LTC-SN): input-dependent leak

1. **Citation.** Bojian Yin, Federico Corradi, Sander M. Bohté, "Accurate online training of dynamical spiking neural networks through Forward Propagation Through Time", *Nature Machine Intelligence* 5, 518–527 (2023). DOI 10.1038/s42256-023-00650-4; arXiv 2112.11231. Check level: [full text, ar5iv of arXiv v2].
2. **Mechanism** (Eq. 9).
   - Membrane time constant from input and state: τ_m⁻¹ = σ(W_τm [x_t, u_{t−1}]).
   - Adaptation time constant: ρ = σ(W_τadp [x_t, b_{t−1}]).
   - Membrane: u_t = u_{t−1} + (−u_{t−1} + x_t)/τ_m.
   - Adaptive threshold: b_t = ρ b_{t−1} + (1 − ρ) s_{t−1}, θ_t = 0.1 + 1.8 b_t.
   - Real multiplies plus a sigmoid per neuron-step.
3. **Evidence** (paper-measured; the table layout was hard to read, so treat the pairings with care).
   - Online FPTT: S-MNIST 97.37 %, PS-MNIST 94.77 %, R-MNIST 98.63 %, DVS Gesture 97.22 %.
   - Add task: LTC-SRNNs converged with FPTT, but "LTC-SRNNs trained with BPTT diverged from exploding gradients".
   - **There is no ablation of input-dependent versus fixed τ.**
4. **Cost.** One D × 2D linear map per layer per step: an AC on spikes, a MAC on the membrane. A sigmoid per neuron-step.
5. **Fit (inference).**
   - For a Neocortex neuron: τ_m⁻¹(i) = σ(W [c_i (Hippocampus), u_{i−1}]). At each token the Hippocampus decides how much Neocortex memory to keep (selective forgetting).
   - The cross-stream interaction then sits entirely inside the T = N dynamics. Any gain is by construction a gain *from Neocortex dynamics*, so memory-off and shuffle controls test it directly.
   - With the population neuron, gate each member's τ separately, or gate the member mixing (overlaps with M8).
6. **Concerns.**
   - The reported BPTT instability matters: we train with BPTT.
   - An input-dependent τ breaks time-invariance, so any fixed-kernel or parallel scan used to speed up the Neocortex neuron no longer applies. Check what our SpikeDE neurons use.
   - No forecasting evidence. Order caveat (§0).

### D. State injection

#### M12. One stream writes the other's membrane state at t = 0 (hybrid ANN→SNN init; learnable IMP)

1. **Citations.**
   - Asude Aydin, Mathias Gehrig, Daniel Gehrig, Davide Scaramuzza, "A Hybrid ANN-SNN Architecture for Low-Power and Low-Latency Visual Perception", CVPR Workshops 2024, pp. 5701–5711. DOI 10.1109/CVPRW63382.2024.00579; arXiv 2303.14176; code `uzh-rpg/hybrid_ann_snn`. **Workshop paper.** Check level: [full text, arXiv HTML v2].
   - Hangchi Shen, Qian Zheng, Huamin Wang, Gang Pan, "Rethinking the Membrane Dynamics and Optimization Objectives of Spiking Neural Networks", NeurIPS 2024; code `StephenTaylor1998/IMP-SNN`. Check level: [full text, proceedings PDF].
2. **Mechanism.**
   - Aydin: an ANN (U-Net) maps its features through Conv–BN–LeakyReLU–Conv–BN to the initial membrane potential of **every SNN layer**, V_0^k = s_i^k, and the SNN predicts a correction to the ANN output. The ANN re-initialises the SNN at 5–10 Hz; the SNN runs at 100 Hz with soft reset.
   - Shen: a learnable per-neuron initial membrane potential ŝ[0] ∈ R^N, initialised from U(−λ, λ). It costs nothing at inference.
3. **Evidence** (paper-measured).
   - Aydin, DHP19 2D pose:
     - MPJPE (cam2 / cam3): hybrid 5.19 / 4.97 px; pure SNN 21.37 / 19.17; pure ANN 5.03 / 4.67.
     - Power: 0.424 W vs 3.718 W for the ANN (−88 %).
     - State-init ablation: zero init 9.12 px at the last bin (7.08 average) vs the chosen init 6.11 (6.02); adding a final sigmoid gave 16.59.
     - Output init alone drifted back toward pure-SNN error.
   - Shen (single runs):
     - ImageNet-100, SEW-R18, T = 4: SDT 79.10 → IMP + SDT 79.90; LTS 80.20 → IMP + LTS 80.80; TET 78.50 → IMP + TET 78.70.
     - CIFAR10-DVS, VGG, T = 10: TET 84.90 → IMP + TET 85.90; best IMP + TET-S (λ = 0.2) 87.10.
   - Aydin's stated limitation: the ANN is not conditioned on the SNN state; the authors name ANN → SNN cross-attention as future work.
4. **Cost.** IMP: one stored constant per neuron. Hybrid init: an extra mapping network. For us, a Linear from Hippocampus counts would be AC.
5. **Fit (inference).** This targets a measured problem: the Neocortex's first token can never fire.
   - The cheapest diagnostic is a learnable IMP per Neocortex neuron, with no fusion at all. If the base improves, early-token silence was costing accuracy.
   - The fusion version initialises the Neocortex membrane per window from a Hippocampus summary (u_0 = W_0·mean_i c_i): "the Hippocampus sets the context, the Neocortex integrates over tokens".
   - It forces little by itself, but it is a **precondition** for any fusion. While the query is 0 at 70–83 % of positions, no coupling can carry Neocortex information (attention report, §11-08).
6. **Concerns.**
   - u_0 carries whole-window information into every Neocortex step (order caveat).
   - A large u_0 can make the Neocortex fire constantly and lose its token dynamics. Track the firing rate per token position.
   - Aydin's result is vision and a workshop paper.

### E. Hippocampus–neocortex (CLS) couplings in SNNs

#### M13. Mask modulation (CH-HNN) and apical-modulated replay (hippocampal–cortical LIF network)

1. **Citations.**
   - Qianqian Shi, Faqiang Liu, Hongyi Li, Guangyu Li, Luping Shi, Rong Zhao, "Hybrid neural networks for continual learning inspired by corticohippocampal circuits", *Nature Communications* 16, 1272 (2025). DOI 10.1038/s41467-025-56405-9; PMC11788432. Check level: [full text, Europe PMC XML].
   - F. D'Alba, N. Kushawaha, L. Fruzzetti, E. Falotico, "Semantization of memories in a hippocampal–cortical spiking neural network", *Neurocomputing* 640, 130323 (2025). DOI 10.1016/j.neucom.2025.130323. Check level: [abstract + highlights via OpenAlex]. The repository PDF was blocked.
2. **Mechanism.**
   - CH-HNN: an ANN (mPFC–CA1 analogue) produces modulation signals for each input that "serve as masks, selectively activating neurons in the hidden layers of the SNNs" (DG–CA3 analogue). Metaplasticity is added on the SNN. The masks are binary, so this step is spike-friendly.
   - D'Alba: a plastic LIF network with hippocampus, perceptual neocortex and semantic/contextual neocortex, trained in two phases:
     - an encoding phase;
     - a sleep-like phase in which hippocampal reactivation propagates to the neocortex.
   - Apical-amplification mechanisms from Dendritic Integration Theory, plus neuromodulators, modulate assemblies during learning and replay; this is compared with STDP.
3. **Evidence** (paper-measured).
   - CH-HNN (from the prior note): episode inference alone 70.70 % vs the full model 70.72 % on Split Tiny-ImageNet. Metaplasticity mainly adds stability.
   - D'Alba: split/rotated MNIST, CIFAR-10/100 and a soft-gripper task; numbers not read.
4. **Cost.** CH-HNN: a whole ANN. D'Alba: replay phases.
5. **Fit (inference).**
   - CH-HNN's coupling (one network's per-input mask over another's hidden units) is the family of the original NSMT AND mask, where Neocortex spikes multiplied the Hippocampus key/value.
   - Our 10-17 analysis found that mask acted as a near-constant per-position gate: swapping in another window's Neocortex output changed ETT MSE by only ±0.3 % (attention report). So this coupling has, in effect, already been tested in our model with a null result.
   - Replay and consolidation couplings (D'Alba; sleep replay) act *across samples* during learning. Our Hippocampus stores nothing across windows, so they do not map onto per-window inference fusion.
6. **Concerns.** Both papers target continual learning, not fusion accuracy.

### F. Sequential + parallel merge in SNN time series

#### M14. SpikeSTAG Dual-Path Spike Fusion (DSF)

1. **Citation.** Bang Hu, Changze Lv, Mingjie Li, Yunpeng Liu, Xiaoqing Zheng, Fengzhe Zhang, Wei Cao, Fan Zhang, "SpikeSTAG: Spatial-Temporal Forecasting via GNN-SNN Collaboration", arXiv 2508.02069 (v2, 18 Aug 2025). **arXiv only.** Check level: [full text, arXiv HTML v2].
2. **Mechanism.**
   - Spiking SSA branch: SSA(X) = SpikeSoftmax(QKᵀ/√d_k)V.
   - LSTM branch: produces real-valued hidden states.
   - Gate: G = σ(W[H_LSTM; H_SSA]); fusion: H_fused = G ⊙ H_LSTM + (1 − G) ⊙ H_SSA (Eq. 13).
   - Only the SSA branch is spike-driven.
3. **Evidence** (paper-measured, Table 3; W3 = concat without gate, W4 = gated DSF), R² / RSE:

   | Dataset | W3 (concat) | W4 (gated) |
   |---|---|---|
   | METR-LA | 0.858 / 0.412 | 0.868 / 0.375 |
   | PEMS-BAY | 0.854 / 0.413 | 0.874 / 0.384 |
   | Solar | 0.964 / 0.251 | 0.973 / 0.246 |
   | Electricity | 0.966 / 0.214 | 0.987 / 0.207 |

   - LSTM-only (W1) is worst on every dataset. The text defines W2 differently from the table caption.
   - Average R² is 0.823 in the table but 0.818 in the text; iSpikformer is 0.808.
   - Energy (Electricity, H3): 4.39 mJ vs 3.19 mJ for iSpikformer, which the paper does not highlight.
4. **Cost.** One 2D → D linear map + sigmoid per token, real-valued.
5. **Fit (inference).**
   - A token-aligned convex gate between the Neocortex (sequential) and the Hippocampus (parallel), per token and channel: z_i = G_i ⊙ f(u_neo,i) + (1 − G_i) ⊙ f(c_i).
   - The mean and distribution of G give a direct, readable "Neocortex share" measure. That is useful as a diagnostic even if the gate is not adopted.
   - The convex form means whatever the Hippocampus does not supply must come from the Neocortex, but G → 0 is allowed. A floor or entropy term on G would be our addition, not the paper's.
6. **Concerns.** arXiv only, with text–table inconsistencies. Seeds not stated. The sequential branch is an LSTM, not LIF. Not spike-driven.

## 3. Cross-cutting observations (inference unless marked)

1. **Multiplicative beats additive when one stream is context.**
   - The only controlled SNN comparison is Ferrand et al. (paper-measured): +9.0 / +15.6 points on single-sample tasks, and additive near chance on sequence memory.
   - Wybo et al. and Ferrand's f/I analysis agree: apical or dendritic modulation changes gain and threshold, while additive input changes only the threshold.
   - By this criterion, M6, M7 and M10 are the additive family. M1, M2, M4, M8, M9 and M11 are the multiplicative family.
2. **Modulate with the Neocortex membrane, not its spikes.** cLIF gates with the apical *membrane* (ReLU(V_a)). Our Neocortex spikes are all 0 at 70–83 % of positions; its membrane is not. This sidesteps the silent-query problem that no attention score can fix.
3. **Identity-at-zero versus forcing.**
   - ReLU(V_a), as in the paper, forces use but is fragile: a silent Neocortex kills the Hippocampus.
   - 1 + tanh(V_a) (in the cLIF code) and CS-TPN's I_s + I_c(0.1 + |I_s|) are safe defaults that allow ignoring.
   - Run both. A large gap between them is itself a measure of how much the Neocortex is used.
4. **Where to modulate.**
   - Luca et al. (paper-measured): context-set threshold or τ helped recurrent but not feedforward networks.
   - For the T = N claim, put the modulation inside the Neocortex's token dynamics (M11, M8, M7).
   - To make the Neocortex matter to the output, put a Neocortex-driven gain on the Hippocampus (M1-B, M9).
5. **Architectural forcing.** Only M1-A (the Neocortex soma as the sole path to the head) cannot be bypassed. Every other candidate relies on gradients.
6. **Order controls.** When Hippocampus information enters the Neocortex per token (M1-A, M6, M7, M8, M11, M12), Neocortex step i sees the whole window through the Hippocampus. Memory-off and shuffle controls must act on the Neocortex's own token input and state.
7. **`neo_grad = 0.5`.** The current code halves the task gradient into the Neocortex through the fusion. Any new fusion test should state this setting or also run 1.0; otherwise part of a "no difference" may come from the halved gradient. The effect size is unknown.
8. **Energy.**
   - Each multiplicative form adds about 1 MAC per modulated neuron-step, about +42 % of a LIF update in our convention.
   - Additive branch routing adds only ACs.
   - Our `syops` counting would not see a gain applied inside a neuron's forward. Count it explicitly, as the attention report asks for the attention core.

## 4. Best three for our problem

1. **Apical multiplicative gain (M1, with the M2 form as a safe variant).**
   - *Why:*
     - It is the only controlled SNN evidence that multiplicative two-stream integration beats additive integration, by 9–16 points, with additive at chance on the memory task (5 seeds).
     - Gating with the Neocortex *membrane* gets around the silent Neocortex.
     - It replaces IAND or attention with a token-aligned, per-channel gain at a cost comparable to one extra neuron update.
     - The τ_a-diversity result maps onto our population τ.
   - *Minimal test (inference):*
     - Arm 1: B-direction gain g = 1 + tanh(W_a u_neo,i) on Hippocampus attention/MLP output currents.
     - Arm 2: the additive counterpart, I_h + W_a u.
     - Arm 3: the ReLU form.
     - Run all arms for pop LIF and pop f-LIF on the usual ETT cells (ETTh2 / ETTm2 × H96 / H720 × 3 seeds), using the report's judgment rule.
     - The Spikformer-style model has no Neocortex to fuse, so it stays an unchanged reference.
     - The Neocortex claim needs base-new to be worse with the Neocortex memory switched off or its tokens shuffled.
     - Also run M1-A (Neocortex soma as the output) as the bold variant.
2. **Hippocampus-conditioned Neocortex dynamics: state injection (M12) plus an input-dependent leak (M11).**
   - *Why:*
     - It attacks the measured early-token silence directly; a learnable IMP is a free first diagnostic.
     - It puts the cross-stream interaction inside T = N, so any gain is attributable to Neocortex dynamics and the memory/order controls are decisive.
   - *Weaker evidence:* no input-dependent-τ ablation exists; BPTT diverged on the Add task; Aydin is a vision workshop paper (zero vs learned init: −33 % error at the last bin).
3. **Branch routing in the existing population neuron (M6 DH-SNN; optionally M8-style Hippocampus-gated branch strengths, or M7 TS-LIF two-compartment).**
   - *Why:*
     - Cheapest option (AC only); the hook already exists (`forward_branches`).
     - Nat Commun evidence that putting two signals on separate branches with matched τ beats one branch and vanilla LIF.
     - It is the natural additive control for #1.
     - Its gated form, κ_b(c_i), is multiplicative but tiny (4 multiplies per neuron-step).

Honourable mention: the SpikeSTAG convex gate (M14), mainly as a diagnostic (mean G = Neocortex share).

Not recommended:
- burst multiplexing (M3: a learning-signal role);
- CLS replay or consolidation (M13: works across samples; the mask form is already ≈ null in our model);
- task-ID dendritic gating as published (M5: discrete context);
- threshold-only modulation (M10: equivalent to additive).

## 5. Gaps (searched, not found)

- No paper tests apical, dendritic or two-compartment *two-stream* fusion on forecasting or ECG.
- No SNN paper compares neuron-level multiplicative fusion with cross-attention fusion in the same model.
- No spiking recurrent or SSM stream + spiking transformer hybrid in which one stream initialises or gates the other's membrane, apart from Aydin's ANN → SNN in vision.
- No ablation of input-dependent versus fixed τ (LTC-SN).
- Papers I could not fully read:
  - D'Alba: numbers not read (paywall / Cloudflare).
  - DendSN: no venue found.
  - CS-TPN and SpikeSTAG: arXiv only.
  - MA-SNN: numbers from the arXiv text.
- Seen but outside scope or unverified:
  - Islah et al. (arXiv 2312.05484): non-spiking apical/basal deep network; no venue.
  - TDFormer (arXiv 2505.15840): top-down feedback through the attention key; attention-based.
  - CE-LIF (arXiv 2308.15150): contextual-embedding threshold from the same stream.
  - A membrane-level current-summation IR/visible fusion preprint was seen in search only and is left out.

## 6. Sources

- Ferrand et al. 2023:
  - https://doi.org/10.1007/978-3-031-44207-0_32
  - https://zenodo.org/records/11489556
  - https://github.com/IGITUGraz/CSNN
  - https://tugraz.elsevierpure.com/en/publications/context-dependent-computations-in-spiking-neural-networks-with-ap/
- Raza & Adeel 2024: https://arxiv.org/abs/2408.11019 ; https://arxiv.org/html/2408.11019v1
- Payeur et al. 2021:
  - https://doi.org/10.1038/s41593-021-00857-x
  - https://www.biorxiv.org/content/10.1101/2020.03.30.015511v2.full
  - correction: https://doi.org/10.1038/s41593-021-00970-x
- Naud & Sprekeler 2018: https://doi.org/10.1073/pnas.1720995115
- Capone et al. 2023: https://doi.org/10.1073/pnas.2220743120 ; https://arxiv.org/abs/2211.02553
- Wybo et al. 2023: https://doi.org/10.1073/pnas.2300558120 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC10410730/
- Iyer et al. 2022: https://doi.org/10.3389/fnbot.2022.846219 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC9100780/
- Pes et al. 2024: https://doi.org/10.1109/AICAS59952.2024.10595872 ; https://arxiv.org/abs/2404.19419 ; https://arxiv.org/html/2404.19419v2
- Zheng et al. 2024 (DH-SNN): https://doi.org/10.1038/s41467-023-44614-z ; https://pmc.ncbi.nlm.nih.gov/articles/PMC10766638/
- Feng et al. 2025 (TS-LIF):
  - https://proceedings.iclr.cc/paper_files/paper/2025/hash/45bd303ea32a8193d3c19de5d5eaf138-Abstract-Conference.html
  - https://arxiv.org/abs/2503.05108
  - https://arxiv.org/html/2503.05108v1
- Huang et al. (DendSN): https://arxiv.org/abs/2412.06355 ; https://arxiv.org/html/2412.06355v2
- Yao et al. 2023 (MA-SNN): https://doi.org/10.1109/TPAMI.2023.3241201 ; https://arxiv.org/abs/2209.13929 ; https://ar5iv.labs.arxiv.org/html/2209.13929
- Hammouamri et al. 2022:
  - https://mlanthology.org/tmlr/2022/hammouamri2022tmlr-mitigating/
  - https://openreview.net/forum?id=15SoThZmtU
  - https://github.com/Thvnvtos/Nm-SNN
- Luca et al. 2026: https://doi.org/10.3389/fnins.2026.1766765 ; https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1766765/full
- Yin, Corradi, Bohté 2023 (LTC-SN): https://doi.org/10.1038/s42256-023-00650-4 ; https://arxiv.org/abs/2112.11231 ; https://ar5iv.labs.arxiv.org/html/2112.11231
- Aydin et al. 2024:
  - https://doi.org/10.1109/CVPRW63382.2024.00579
  - https://arxiv.org/html/2303.14176v2
  - https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/html/Aydin_A_Hybrid_ANN-SNN_Architecture_for_Low-Power_and_Low-Latency_Visual_Perception_CVPRW_2024_paper.html
- Shen et al. 2024 (IMP):
  - https://proceedings.neurips.cc/paper_files/paper/2024/hash/a867a94a427bacbe3f4de16c7ac10ba8-Abstract-Conference.html
  - https://neurips.cc/virtual/2024/poster/96543
  - https://github.com/StephenTaylor1998/IMP-SNN
- Shi et al. 2025 (CH-HNN): https://doi.org/10.1038/s41467-025-56405-9 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11788432/
- D'Alba et al. 2025: https://doi.org/10.1016/j.neucom.2025.130323
- Hu et al. 2025 (SpikeSTAG): https://arxiv.org/abs/2508.02069 ; https://arxiv.org/html/2508.02069v2
- Project context:
  - `NSMT/reports/NSMT 결합부 spiking attention 후보.md`
  - `NSMT/research_notes/NSMT query 구조 개선 제안/spiking_neuron_memory.md`
  - `NSMT/research_notes/NSMT query 구조 개선 제안/cls_objectives_readout.md`
  - `NSMT/nsmt_flif_v1/classification/{model.py,layers.py,energy.py}`
