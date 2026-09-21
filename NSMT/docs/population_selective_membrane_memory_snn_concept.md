# Conceptual Research Specification: Population-Based Selective Membrane Memory for Time-Series SNNs

> **Purpose of this document**
>
> This document records the research idea developed in discussion so that a coding/research agent can understand the intended model **before any implementation decisions are made**. It is deliberately a **conceptual and scientific specification, not an implementation guide**. It does not prescribe software structure, APIs, tensor operations, training scripts, libraries, or coding procedures.
>
> The main idea is to move from **fixed, indiscriminate temporal memory** toward **content-adaptive, selectively retrieved membrane memory**, and then to make each logical spiking unit a **population of heterogeneous LIF neurons** rather than a single scalar-state LIF neuron.
>
> No final model name has been chosen.

---

## 1. Executive summary

The starting point is the fractional-order SNN (f-SNN) framework of Ge et al. (ICLR 2026) [1]. A conventional LIF neuron is governed by first-order dynamics and can be written in discrete form as

\[
U_t = \beta U_{t-1} + X_t,
\]

followed by spike generation and reset. Its dependence on the distant past is compressed into the immediately preceding state and, under the standard linear subthreshold interpretation, decays exponentially.

The f-SNN paper replaces the ordinary derivative with a fractional derivative. In its fractional LIF (f-LIF) formulation, the present state depends on a weighted history of past dynamics. The induced memory kernel has a power-law tail rather than a purely exponential one. This gives persistent, non-Markovian temporal memory. The paper argues that such fractional dynamics capture long-range dependence that a finite collection of ordinary integer-order modes cannot reproduce exactly [1]. Earlier biological modeling work likewise described fractional LIF dynamics as incorporating a temporally weighted voltage-memory trace and related this to spike adaptation and multi-timescale neuronal behavior [2,3].

The new research idea asks a different question:

> **Instead of allowing all available past states to influence the present according to a fixed temporal kernel, can a spiking neuron learn which past states are actually relevant to the current time step and selectively retrieve only those memories?**

The intended mechanism has two conceptual components:

1. **Selective temporal memory**
   - Past internal states are treated as memory.
   - The relevance of a past state is conditioned on the current state.
   - More relevant past time steps receive larger weights.
   - Irrelevant past states may be suppressed by a soft or hard mask.
   - A fractional/power-law recency kernel may optionally act as a temporal prior, but relevance is not determined by temporal distance alone.

2. **Population-based logical neuron**
   - One **logical neuron** is not a single LIF unit.
   - Instead, one logical neuron is a **population of \(K\) constituent LIF neurons**.
   - The population is intentionally heterogeneous, especially in temporal dynamics such as membrane time constants.
   - The \(K\)-dimensional membrane-potential pattern is treated as the state representation of the logical neuron.
   - Past population-state vectors form memory slots.
   - Relevance is evaluated between the current population state and past population states.
   - The population concept is inspired by population coding: one quantity/state is represented by a pattern over multiple neurons rather than by a single scalar neuron response [4,5].
   - However, this proposal is **not merely conventional input population coding**. The population becomes the persistent internal state and memory-retrieval unit of the spiking dynamics.

The current main conceptual direction is therefore:

\[
\boxed{
\text{heterogeneous LIF population}
\rightarrow
\text{population membrane-state memory}
\rightarrow
\text{current-conditioned relevance}
\rightarrow
\text{selective retrieval of past population states}
\rightarrow
\text{current spiking dynamics}
}
\]

The central research hypothesis is that long-term temporal modeling should not be framed only as **how long a neuron remembers**, but also as **which past memory should be used now**.

---

## 2. Scientific motivation

### 2.1 Conventional LIF memory is compact but strongly constrained

For a standard discrete LIF neuron,

\[
U_t = \beta U_{t-1} + X_t,
\qquad 0 < \beta < 1,
\]

unrolling the recurrence gives, in the simple linear subthreshold case,

\[
U_t
=
X_t + \beta X_{t-1} + \beta^2 X_{t-2} + \cdots.
\]

Therefore, old information is not literally absent, but its influence is tied to a fixed exponentially decaying structure.

A single membrane potential is also a severe compression of temporal history. Different input trajectories may produce the same current scalar membrane potential. Once those trajectories collapse to the same scalar state, a memory-selection rule based only on that scalar cannot distinguish them.

This motivates both parts of the proposal:

- richer state representation through a **population**;
- adaptive access to history through **selective retrieval**.

### 2.2 Fractional-order SNNs provide long memory, but their memory is primarily time-structured

Ge et al. [1] replace first-order dynamics with fractional-order dynamics:

\[
\tau D^\alpha U(t)
=
-U(t)+RI_{\mathrm{in}}(t),
\qquad 0<\alpha\le 1.
\]

At \(\alpha=1\), the conventional LIF case is recovered. For \(0<\alpha<1\), the Caputo derivative is nonlocal in time: the present derivative depends on an integral over prior history.

After discretization, the f-LIF charging rule contains a history-weighted sum. Suppressing constants and indexing details, the fractional kernel has the characteristic form

\[
c_m^{(\alpha)}
\propto
(m+1)^\alpha-m^\alpha,
\]

whose long-lag behavior is algebraic rather than exponentially vanishing [1]. Ge et al. connect this to persistent memory and Mittag-Leffler/power-law relaxation.

An important conceptual correction must be preserved:

> The f-LIF discretization is **not simply "a weighted sum of past membrane potentials."**  
> In the formulation of Ge et al., the history enters through past evaluations of the neuronal dynamics, including the leak and input terms. A representative form is
>
> \[
> U_t
> =
> U_0
> +
> \sum_j
> c_{t-j}^{(\alpha)}
> \left(-U_{\text{past}} + X_{\text{past}}\right).
> \]
>
> Therefore, directly retrieving and adding stored membrane-potential vectors is a **new model design**, not merely a masked implementation of the original fractional differential equation.

This distinction matters for scientific claims. If direct membrane-state retrieval is used, the resulting neuron should not automatically be described as an exact fractional-order neuron.

### 2.3 The unresolved weakness: temporal persistence is not the same as contextual relevance

A fixed long-memory kernel answers the question

> "How strongly should information from lag \(d\) persist?"

but not necessarily

> "Which past state is useful for the current context?"

For many time series, the most useful previous state need not be the most recent state.

Examples include:

- recurring operating regimes;
- periodic or quasi-periodic behavior with variable phase;
- repeated motifs separated by irregular intervals;
- long-range dependencies interrupted by irrelevant observations;
- state transitions such as \(A\rightarrow B\rightarrow A\), where the earlier \(A\) state may be more useful than the immediately preceding \(B\) state.

The proposed selective memory is intended to make temporal influence **content-dependent**, not only lag-dependent.

---

## 3. The central conceptual distinction: persistent memory vs. selective memory

The f-SNN idea can be summarized as

\[
\text{past influence}
=
\text{fixed function of temporal distance and fractional order}.
\]

The proposed idea instead aims for

\[
\text{past influence}
=
\text{temporal prior}
\times
\text{current-conditioned relevance}
\times
\text{selection mask}.
\]

A generic conceptual decomposition is

\[
\widetilde c_{t,j}
=
\underbrace{c_{t-j}^{(\alpha)}}_{\text{optional temporal prior}}
\;
\underbrace{a_{t,j}}_{\text{current-conditioned relevance}}
\;
\underbrace{m_{t,j}}_{\text{selection}},
\]

where

- \(t\): current time step;
- \(j<t\): past time step;
- \(c_{t-j}^{(\alpha)}\): optional lag-dependent fractional/power-law prior;
- \(a_{t,j}\): learned or otherwise defined relevance of past state \(j\) **for current time \(t\)**;
- \(m_{t,j}\): soft or hard selection mask.

The crucial index is \(a_{t,j}\), not merely \(a_j\).

A global \(a_j\) would mean "time step \(j\) is generally important."  
A conditional \(a_{t,j}\) means "time step \(j\) is important **for the present state at \(t\)**."

The latter is the intended idea.

---

## 4. Two distinct conceptual branches

The discussion has identified two scientifically different variants. They should not be conflated.

### 4.1 Branch A: selective fractional-history dynamics

This is the conservative extension of f-LIF.

The original f-LIF history term is retained conceptually, but the contribution of past history is modulated by relevance and/or a mask:

\[
V_t
=
U_0
+
\sum_{j<t}
c_{t-j}^{(\alpha)}
a_{t,j}
m_{t,j}
F_j
+
\text{current contribution},
\]

where \(F_j\) denotes the historical neuronal-dynamics term rather than simply \(U_j\).

Scientific interpretation:

- closest to the fractional-order starting point;
- explicitly asks whether the fixed fractional memory kernel should be made selective;
- retains a clear connection to the f-SNN paper;
- nevertheless changes the dynamics enough that the original theoretical guarantees cannot automatically be inherited.

### 4.2 Branch B: direct membrane-state retrieval

This branch is more aligned with the current idea.

Past membrane states themselves are treated as explicit memory items:

\[
\mathcal{M}_t
=
\{
\mathbf u_1,
\mathbf u_2,
\dots,
\mathbf u_{t-1}
\}.
\]

The present state determines which past states are relevant, and the selected past states are aggregated into a retrieved memory:

\[
\mathbf M_t
=
\sum_{j<t}
w_{t,j}\mathbf u_j.
\]

Scientific interpretation:

- more directly implements the idea that **membrane potential is memory**;
- naturally supports explicit relevance-based retrieval;
- makes temporal selection inspectable;
- is no longer simply an f-LIF discretization;
- is closer conceptually to a memory-augmented spiking neuron or a state-retrieval neuron.

**Current emphasis:** Branch B, especially when combined with the population-based logical neuron described below.

---

## 5. Population-based logical neuron

### 5.1 Redefining the unit of computation

In a conventional description, one neuron may be one LIF element with one membrane potential.

The proposed abstraction distinguishes:

- **constituent LIF neuron**: an ordinary spiking element;
- **logical neuron**: a population of \(K\) constituent LIF neurons representing one higher-level state.

For logical neuron \(n\),

\[
\mathcal N_n
=
\{
\mathrm{LIF}_{n,1},
\mathrm{LIF}_{n,2},
\dots,
\mathrm{LIF}_{n,K}
\}.
\]

Its membrane state is

\[
\mathbf u_{t,n}
=
[
u_{t,n,1},
u_{t,n,2},
\dots,
u_{t,n,K}
]^\top
\in\mathbb R^K.
\]

Thus,

\[
\text{single-LIF state: } u_{t,n}\in\mathbb R
\]

becomes

\[
\text{population state: } \mathbf u_{t,n}\in\mathbb R^K.
\]

The population-state vector is the intended atomic memory representation.

### 5.2 Why a population should not be a set of identical replicas

If all \(K\) constituent LIF neurons have

- identical inputs;
- identical parameters;
- identical initial states;
- deterministic identical dynamics,

then

\[
u_{t,n,1}
=
u_{t,n,2}
=
\cdots
=
u_{t,n,K},
\]

so the \(K\)-dimensional vector carries no more information than a single scalar.

Therefore, the population is only meaningful if the constituent responses are intentionally diverse.

The most natural source of diversity for the current idea is **temporal heterogeneity**.

### 5.3 Heterogeneous temporal responses

An illustrative subthreshold LIF-like update for constituent \(k\) is

\[
\bar u_{t,n,k}
=
\beta_k u_{t-1,n,k}
+
(1-\beta_k) I_{t,n},
\]

with different \(\beta_k\) values across the population.

The exact equation above is a conceptual example, not a finalized neuron definition. Its purpose is to show the intended role:

- smaller \(\beta_k\): faster response, shorter persistence;
- larger \(\beta_k\): slower response, longer persistence.

The resulting population pattern contains multiple views of the recent temporal trajectory.

This motivation has empirical support from prior SNN literature. Perez-Nieves et al. [6] found that heterogeneity in membrane and synaptic time constants improved learning, stability, and robustness, especially on tasks with rich temporal structure. Other heterogeneous SNN work similarly uses different LIF time constants to create multiple timescales [7].

Importantly, these results support the **usefulness of heterogeneous timescales**, but they do **not** establish the effectiveness of the new selective memory-retrieval mechanism. That remains a new hypothesis to test.

---

## 6. Relationship to population coding

### 6.1 What is borrowed from population coding

The conceptual inspiration is:

> A quantity or state can be represented by the joint response pattern of multiple neurons rather than by the activity of a single neuron.

For example, Zhang et al.'s MDC-SAN [4] encodes each input-state dimension using a population of neurons with learnable receptive fields. A more recent temporal-coding study likewise distributes one input dimension across a population of LIF neurons with Gaussian receptive fields [5].

These works demonstrate a conventional population-coding interpretation:

\[
\text{one input dimension}
\rightarrow
\text{activity pattern over multiple neurons}.
\]

### 6.2 What is different in the proposed idea

The proposed population is not intended to be merely an input encoder.

Instead,

\[
\text{one logical neuron's internal state}
\rightarrow
\text{membrane-potential pattern over multiple LIF neurons}.
\]

That population state persists over time and becomes the object of memory retrieval.

Therefore, the proposed role is:

\[
\boxed{
\text{population as an internal temporal-state representation}
}
\]

rather than only

\[
\text{population as an input-value encoding}.
\]

This distinction is central to the intended research contribution.

### 6.3 Why the population representation may help retrieval

A scalar LIF state can alias different temporal histories.

Consider two different input histories that happen to produce the same scalar membrane value for one LIF time constant. A population with several time constants may produce distinct state vectors.

Illustrative example, assuming no spike/reset for simplicity:

\[
A=[0,1,1],
\qquad
B=[2,0,1].
\]

With one particular LIF setting, both histories can yield the same current state:

\[
u_A = u_B = 0.75.
\]

A heterogeneous three-neuron population may instead produce patterns such as

\[
\mathbf u_A
=
[0.960,\,0.750,\,0.360],
\]

\[
\mathbf u_B
=
[0.864,\,0.750,\,0.456].
\]

The exact numbers are only a toy calculation. The conceptual point is:

> Even when one scalar membrane value aliases two histories, a multi-timescale population pattern can potentially separate them.

Therefore, relevance between the current state and a past state can be evaluated in a richer state space.

This does **not** imply that a finite \(K\)-dimensional vector uniquely identifies all possible histories. It only motivates why population-state retrieval may be more expressive than scalar-state retrieval.

---

## 7. Population membrane states as memory slots

For each logical neuron \(n\), the historical memory at current time \(t\) is conceptually

\[
\mathcal M_{t,n}
=
\{
\mathbf u_{1,n},
\mathbf u_{2,n},
\dots,
\mathbf u_{t-1,n}
\}.
\]

Each memory slot corresponds to one earlier time step and contains the population state.

A central interpretive statement is:

> Selecting \(\mathbf u_{j,n}\) means selecting the internal state that existed at time \(j\), which itself may summarize information preceding \(j\). It does **not** mean selecting only the raw observation \(X_j\).

This is important for later interpretability claims.

A zero weight on memory slot \(j\) also does not guarantee that raw input \(X_j\) has no influence on the present, because its effect may have propagated into later states.

---

## 8. Current-conditioned relevance

The intended retrieval mechanism should answer:

> "Given the state I am in now, which earlier internal states are most relevant?"

The current population state before memory retrieval can be denoted

\[
\bar{\mathbf u}_{t,n}.
\]

A generic relevance score is

\[
r_{t,j,n}
=
\operatorname{Rel}
\left(
\bar{\mathbf u}_{t,n},
\mathbf u_{j,n}
\right).
\]

Here \(\operatorname{Rel}(\cdot,\cdot)\) is intentionally left abstract.

The conceptual requirement is not a particular similarity function. The requirement is that relevance be:

1. **conditioned on the current time step**;
2. **computed from sufficiently rich state information**;
3. **causal when the task requires causality**;
4. capable of assigning larger importance to a distant state than to a recent but irrelevant state.

It may also be useful to distinguish:

- similarity of state;
- usefulness for the task.

These are not necessarily identical.

A past state that is numerically similar to the current state is not automatically the state that best improves prediction or classification.

Therefore, "relevance" should be understood as an intended **task-relevant relation**, not merely raw Euclidean similarity.

---

## 9. Optional fractional temporal prior

The proposal does not require abandoning the insight of fractional dynamics.

A fractional/power-law memory kernel can be retained as a **prior over temporal distance**, while content relevance determines deviations from that prior.

One conceptual form is

\[
e_{t,j,n}
=
r_{t,j,n}
+
\lambda_p \log p_\alpha(t-j),
\]

where

\[
p_\alpha(d)
\propto
(d+1)^\alpha-d^\alpha.
\]

Interpretation:

- \(p_\alpha(d)\) says that temporal distance still matters;
- \(r_{t,j,n}\) says that content/context matters;
- a sufficiently relevant distant state can outrank a nearby irrelevant state.

This would create a useful conceptual bridge:

\[
\boxed{
\text{fractional memory as temporal prior}
+
\text{content-based relevance as adaptive correction}
}
\]

However, the fractional prior is **not yet a required component**.

A scientifically important question is whether a fractional prior contributes anything once content-dependent retrieval exists. Uniform, exponential, fractional, or no explicit temporal prior are conceptually separable alternatives.

---

## 10. Selective readout of past population states

Let \(\mathcal A_{t,n}\) denote the subset of past time steps selected for the current logical neuron.

Normalized relevance weights can be represented abstractly as

\[
w_{t,j,n}
\ge 0,
\qquad
\sum_{j\in\mathcal A_{t,n}}
w_{t,j,n}=1.
\]

The retrieved memory is

\[
\boxed{
\mathbf M_{t,n}
=
\sum_{j\in\mathcal A_{t,n}}
w_{t,j,n}
\mathbf u_{j,n}
}
\]

or, in a soft-selection form,

\[
\mathbf M_{t,n}
=
\sum_{j<t}
m_{t,j,n}a_{t,j,n}
\mathbf u_{j,n}.
\]

The intended semantic roles are:

- \(\mathcal A_{t,n}\) or \(m_{t,j,n}\): **which memories are read**;
- \(a_{t,j,n}\) or \(w_{t,j,n}\): **how strongly selected memories contribute**.

This distinction should remain explicit.

A binary mask alone only decides inclusion/exclusion. It does not by itself express graded relevance among the retained memories.

---

## 11. Memory use should be conceptually separable from memory selection

A useful distinction is:

1. **Read selection**: Which past memories should be retrieved?
2. **Read strength**: How much should the retrieved memory influence the present?

Even if a retrieval rule always identifies some "best" past state, there may be situations where no previous memory is useful.

Therefore, the conceptual model may include a state-dependent memory-use factor \(g_{t,n}\in[0,1]\):

\[
\mathbf V_{t,n}
=
\bar{\mathbf u}_{t,n}
+
\gamma\,g_{t,n}\mathbf M_{t,n},
\]

where \(\gamma\) represents the global strength of memory influence.

This equation captures an important behavioral requirement:

> The model should be able to distinguish **"which memory is best"** from **"whether memory should be used at all."**

The exact mechanism for \(g_{t,n}\) is intentionally unresolved.

---

## 12. Population-level retrieval vs. constituent-level retrieval

The current preferred conceptual interpretation is:

> The \(K\) constituent LIF neurons jointly determine which historical time steps are relevant, and the same temporal selection is applied to the whole population-state vector.

Thus, selection is conceptually associated with the logical neuron:

\[
r_{t,j,n}
=
\operatorname{Rel}
(
\bar{\mathbf u}_{t,n},
\mathbf u_{j,n}
).
\]

This is preferable as the first scientific formulation because it preserves a clean meaning:

> "Logical neuron \(n\) retrieved its state from time \(j\)."

An alternative would allow every constituent LIF neuron to select a different historical time step. That may be more expressive, but it changes the interpretation substantially and increases complexity.

It should therefore be treated as a later alternative, not assumed to be the same model.

---

## 13. What is a "logical neuron output"?

This is deliberately not finalized.

Two conceptual possibilities exist.

### 13.1 Population spike output

Each constituent LIF neuron emits its own spike:

\[
\mathbf s_{t,n}
=
[
s_{t,n,1},\dots,s_{t,n,K}
]
\in\{0,1\}^K.
\]

This preserves the population representation across layers or subsequent processing.

Advantages conceptually:

- no early collapse of the population pattern;
- constituent diversity remains observable;
- the "logical neuron" is explicitly a population-valued unit.

### 13.2 Single logical spike output

The population may eventually be collapsed to one binary logical spike.

However, this introduces a new design question:

> What operation maps \(K\) constituent membrane/spike states to one binary event, and which states are reset?

A spike count or average firing value is not itself a binary spike.

Therefore, a single-spike interpretation requires a separately justified population readout rule.

**Current discussion has not locked this decision.** The population-spike representation is the cleaner conceptual starting point.

---

## 14. Reset state and memory semantics

A key unresolved question is whether a memory slot stores:

1. **pre-reset membrane state**, or
2. **post-reset membrane state**.

These have different meanings.

### Pre-reset memory

It retains the strong membrane response that caused a spike.

Potential benefit:
- preserves evidence immediately before firing.

Potential risk:
- a state that already triggered a spike can be repeatedly reintroduced as memory.

### Post-reset memory

It stores the state that actually continues forward in the recurrence after spike/reset.

Potential benefit:
- consistent with the dynamical state that persists after firing.

Potential risk:
- hard reset may erase informative pre-spike amplitude.

The working discussion leaned toward **post-reset state as the primary memory value**, because it is the state that truly persists in the recurrence. However, this is **not a finalized scientific decision**. Pre-reset vs. post-reset memory is a critical ablation.

---

## 15. Read selection is not memory deletion

A mask value of zero at current time \(t\),

\[
m_{t,j,n}=0,
\]

should be conceptually interpreted as

> "do not read this memory now"

rather than automatically

> "delete this memory forever."

A state that is irrelevant at time \(t\) may become useful at time \(t+q\).

Therefore, two separate problems exist:

- **read policy**: what is used now;
- **write/retention/eviction policy**: what remains available for future retrieval.

The current idea primarily concerns **read selection**.

Memory retention or eviction is a separate research extension and should not be silently merged into the core proposal.

---

## 16. Why this may be useful for time-series analysis

The following are **research hypotheses**, not established facts.

### H1. Better discrimination of temporal histories

A heterogeneous population may distinguish temporal trajectories that collapse to the same scalar LIF state.

Expected consequence:
- better retrieval decisions when current relevance depends on the trajectory, not just the current value.

### H2. Adaptive long-range dependency

A distant past state may receive a larger weight than a recent state when it better matches the current regime or task context.

Expected consequence:
- long-range dependencies need not be represented by uniformly long persistence.

### H3. Recurrent regime retrieval

For sequences such as

\[
A\rightarrow B\rightarrow A,
\]

the current \(A\) regime may benefit from selectively retrieving an earlier \(A\)-like population state rather than emphasizing the immediately preceding \(B\) states.

Potential domains:
- industrial sensor data;
- physiological signals;
- traffic or load time series;
- human activity;
- repeated operational cycles.

### H4. Reduced interference from irrelevant history

A full-memory mechanism necessarily exposes the present to many old states. Selective retrieval may reduce interference from irrelevant historical states.

This does **not** mean that masking automatically guarantees noise robustness. A corrupted current query can retrieve the wrong memories.

The empirical question is whether relevance-based retrieval retains useful history while degrading less as irrelevant history increases.

### H5. Multiple temporal scales inside one logical unit

Heterogeneous LIF constituents can provide fast and slow state components.

The population can therefore be interpreted as a multi-timescale state representation:

\[
\mathbf u_{t,n}
=
[
\text{fast response},
\text{medium response},
\text{slow response},
\dots
].
\]

This is consistent with evidence that neuronal timescale heterogeneity can help temporal processing [6,7].

### H6. More interpretable temporal access patterns

The retrieval distribution \(w_{t,j,n}\) can show which historical internal states are accessed at each time.

Potential analyses include:
- preferred lag distributions;
- state-dependent shifts in preferred lag;
- whether the model retrieves repeated motifs;
- whether different logical neurons retrieve different temporal contexts.

However, retrieval weight should not automatically be equated with causal importance. Intervention/ablation is required before making a stronger interpretability claim.

---

## 17. Time-series semantics and causality

For the proposal to support the claim "retrieve relevant past time steps," the SNN temporal index must have a meaningful relationship to chronological sequence order.

If a time-series sample is

\[
\mathbf X
\in
\mathbb R^{L\times C},
\]

the intended interpretation is that the state evolves across actual ordered observations or temporally ordered patches.

This should be distinguished from a common SNN setup where an entire static input is repeated across an artificial simulation-time axis. In that case, retrieving an earlier SNN simulation state is not the same as retrieving an earlier real-world time-series observation.

For causal forecasting or online anomaly detection:

\[
j < t
\]

must hold for every retrieved memory. Causality must also hold in whatever representation generates the relevance score. A causal memory mask alone is insufficient if the current representation already contains future information.

---

## 18. Special concern for anomaly detection

Selective memory can be especially attractive for anomaly detection, but it also creates a unique risk.

If an anomalous state is written into memory and later judged relevant, repeated retrieval may propagate anomaly-contaminated state information.

This may:
- normalize abnormal behavior;
- blur anomaly boundaries;
- create false persistence after an anomaly;
- alternatively, amplify anomalies.

Therefore, anomaly detection should not simply inherit the same memory assumptions as classification or forecasting.

A separate scientific question is:

> Should the memory bank represent all past states, or primarily trusted/normal states?

This question belongs to future model design and is not resolved here.

---

## 19. Stability considerations

Directly feeding retrieved membrane states back into present membrane dynamics can create a feedback loop:

\[
\text{large past state}
\rightarrow
\text{large current state}
\rightarrow
\text{large future stored state}
\rightarrow \cdots
\]

Therefore, the proposal requires explicit stability analysis.

For a simplified nonnegative normalized-memory case, one can reason about a sufficient condition such as

\[
\beta+\gamma < 1
\]

under restrictive assumptions. This is only an illustrative stability intuition, not a theorem for the full proposed neuron.

Important caveat:

- the f-SNN robustness theorem in Ge et al. [1] is proved for particular fractional dynamics and perturbation settings;
- once history coefficients become content-dependent, or direct membrane retrieval is added, that theorem does not automatically apply.

The new model requires its own stability/robustness analysis.

---

## 20. Computational-efficiency caveat

Sparsity in the final weighted sum does **not** automatically mean efficient retrieval.

If relevance is first evaluated against every past state and only then a few memories are selected, the model still performs dense historical comparison.

Therefore, two different notions must not be conflated:

1. **sparse memory use**: only a few past states affect the final retrieved memory;
2. **efficient memory search**: only a limited amount of work is required to identify those states.

This is scientifically important because the original f-SNN can exploit properties of a stationary temporal kernel. Ge et al. discuss full-history convolution and short-memory truncation [1]. Once coefficients are explicitly content-dependent in both \(t\) and \(j\), the stationary-convolution structure is generally lost.

Mamba provides a relevant analogy: Gu and Dao [11] emphasize that making state-space dynamics input-dependent enables selective propagation/forgetting but removes the simple time-invariant convolutional formulation, creating an efficiency challenge that requires separate treatment.

Therefore, no energy- or complexity-efficiency claim should be made merely because the memory weights are sparse.

---

## 21. Relationship to existing literature and novelty boundaries

The proposal touches several established research lines. Its novelty must be stated carefully.

### 21.1 Fractional-order spiking dynamics

**Ge et al., ICLR 2026 [1]**
- fractional derivative inside SNN neuron dynamics;
- long-range/power-law memory;
- f-IF/f-LIF;
- compatibility with CNNs, Transformers, GNNs;
- theoretical distinction from finite integer-order systems.

**Teka et al., PLOS Computational Biology 2014 [2]**
- biological fractional LIF;
- voltage-memory trace;
- spike adaptation and noisy-input response.

**Deng et al., IEEE TBioCAS 2022 [3]**
- fractional spiking neuron tied to dendritic fractal modeling.

**Distinction of the proposed idea:**  
The new idea is not simply "use a fractional derivative." It asks whether memory should be **selectively retrieved based on present context**.

### 21.2 Population coding

**Zhang et al., AAAI 2022 (MDC-SAN) [4]**
- each input-state dimension is encoded using an individual neuron population with learnable receptive fields;
- population coding is combined with richer neuron dynamics.

**Tavanaei-style/related population temporal coding example [5]**
- one normalized input dimension distributed over multiple LIF neurons with overlapping Gaussian receptive fields.

**Distinction of the proposed idea:**  
The population is not only an input encoder. It becomes the **logical neuron state** and the **unit of temporal memory retrieval**.

### 21.3 Neuronal heterogeneity

**Perez-Nieves et al., Nature Communications 2021 [6]**
- heterogeneous membrane/synaptic time constants;
- improvements especially on temporally rich tasks;
- improved stability and robustness.

**Heterogeneous recurrent SNN work [7]**
- different LIF membrane constants and thresholds create multiple timescales.

**Distinction of the proposed idea:**  
Heterogeneous LIF populations are not by themselves novel enough. Their intended role here is to provide a richer **retrieval key/value state** for selective temporal memory.

### 21.4 Temporal attention in SNNs

**Yao et al., ICCV 2021, TA-SNN [8]**
- temporal-wise attention evaluates the significance of frames;
- irrelevant input frames can be discarded.

**Yao et al., TPAMI 2023, Attention SNN [9]**
- temporal, channel, and spatial attention;
- attention weights regulate membrane potentials and spiking responses.

**Distinction of the proposed idea:**  
The target of selection here is **stored past internal population membrane states**, not merely input frames or a generic temporal/channel/spatial modulation of current activations.

This distinction must be demonstrated experimentally rather than asserted.

### 21.5 Masked temporal spiking neurons

**Fang et al., NeurIPS 2023, Parallel Spiking Neuron (PSN) [10]**
- temporal input weights can be fully connected;
- a mask enforces causality;
- sliding PSN shares weights across time.

**Distinction of the proposed idea:**  
A mask over temporal weights is not itself novel. The proposed selection is intended to be **current-content-conditioned retrieval of internal membrane-state memory**, with population-state representations.

### 21.6 Selective state-space models

**Gu & Dao, Mamba [11]**
- state-space parameters depend on input;
- information can be selectively propagated or forgotten based on content.

**Distinction of the proposed idea:**  
Mamba is not an SNN neuron model and does not define a population of spiking LIF states as an explicit historical memory bank. However, its central motivation—that fixed dynamics can fail at content-based reasoning—is conceptually relevant and creates a strong comparison point.

---

## 22. What must not be claimed prematurely

The following statements are **not yet justified**:

- "This is the first selective-memory spiking neuron."
- "The model is biologically equivalent to population coding."
- "A finite LIF population is equivalent to a fractional neuron."
- "Masking automatically improves energy efficiency."
- "Relevance weights are causal explanations."
- "Selective memory guarantees robustness."
- "The original f-SNN theoretical guarantees remain valid."
- "The population representation uniquely reconstructs temporal history."
- "Top-\(K\) or sparse retrieval is necessarily cheaper than full-history fractional memory."
- "A selected memory time step corresponds only to the raw observation at that time."

These should be treated as research questions, not assumptions.

---

## 23. Finite LIF population vs. fractional memory

Ge et al. [1] present a theoretical result that a fractional IF impulse response has an algebraic tail and cannot be represented **exactly** by a finite weighted ensemble of ordinary LIF modes for general inputs; an exact representation is associated with a continuum of leak factors.

This result is directly relevant to the proposed population idea.

A finite heterogeneous population can approximate or enrich multi-timescale dynamics, but it should **not** be described as exactly reproducing the f-LIF memory.

Therefore, the intended argument is not:

\[
\text{"population replaces fractional dynamics exactly."}
\]

It is:

\[
\boxed{
\text{"a finite heterogeneous population provides a richer state representation
on which selective memory retrieval can operate."}
}
\]

That is a different claim.

---

## 24. Important unresolved design questions

The coding/research agent should understand that the following are deliberately unresolved scientific choices.

### 24.1 What exactly is stored as memory?
Candidates:
- post-reset population membrane state;
- pre-reset population membrane state;
- a transformed population state;
- the original f-LIF dynamics term rather than direct membrane state.

Current discussion: post-reset membrane state is a reasonable working convention, but not finalized.

### 24.2 What defines heterogeneity?
Candidates:
- membrane time constants;
- thresholds;
- synaptic time constants;
- receptive fields;
- combinations of these.

Current emphasis: start conceptually from heterogeneous temporal responses, especially membrane time constants, because the main purpose is richer temporal-state representation.

### 24.3 Are heterogeneous parameters fixed or learned?
Both are scientifically plausible. This is not decided.

### 24.4 What exactly defines relevance?
Possibilities include:
- state similarity;
- learned task-conditioned compatibility;
- state plus current input;
- state plus local temporal derivative/change;
- combination of content relevance and lag prior.

No final relevance function is chosen.

### 24.5 Soft selection or hard selection?
A continuous distribution offers smooth weighting.  
A hard subset provides explicit sparse retrieval.

No final choice is locked.

### 24.6 How many past states may be selected?
No fixed retrieval count has been decided.

### 24.7 Should every logical neuron retrieve independently?
Current conceptual direction: each logical neuron may have its own retrieval distribution, while the \(K\) constituents within one population share the selected time steps.

### 24.8 Should retrieval be shared across layers or channels?
Unresolved.

### 24.9 Is the fractional temporal prior necessary?
Unresolved and should be tested rather than assumed.

### 24.10 What is the population output?
- \(K\) constituent spikes;
- one aggregated logical spike;
- another readout.

Current conceptual preference: retain \(K\) spikes initially because collapsing the population too early may discard the intended representation, but this is not finalized.

### 24.11 Does memory persist across independent samples/windows?
For the basic time-series interpretation, memory should conceptually belong to the sequence being modeled. Persistent cross-sample memory would be a separate streaming/stateful setting and requires explicit justification.

---

## 25. Key failure modes to investigate

### 25.1 Population redundancy
If constituent neurons behave nearly identically, the population is merely duplicated capacity.

### 25.2 Retrieval collapse
The model may always select:
- the most recent state;
- a fixed lag;
- one memory slot;
- uniformly all history.

Any of these would weaken the intended content-adaptive interpretation.

### 25.3 Relevance-query corruption
Noise in the current state can cause retrieval of the wrong historical states.

### 25.4 Memory feedback amplification
Retrieved high membrane states can inflate later membrane states and become self-reinforcing.

### 25.5 Capacity confounding
A population model uses more physical LIF units than a scalar-neuron model. Gains may arise simply from increased capacity.

### 25.6 Attention confounding
If a generic attention mechanism over past states produces the gain, the contribution may be "attention in an SNN" rather than a new neuron-level memory principle.

### 25.7 Temporal prior dominance
If the fractional prior overwhelms relevance, the model degenerates toward fixed lag-based memory.

### 25.8 Relevance dominance
If learned relevance completely ignores the temporal prior, the fractional component may be unnecessary.

### 25.9 Causality leakage
A relevance representation that uses future information invalidates forecasting/online claims.

### 25.10 Interpretability overclaim
Large retrieval weight is not automatically causal importance.

### 25.11 Sparse-but-expensive search
Sparse final memory usage can still require dense all-history matching.

### 25.12 Anomaly contamination
In anomaly detection, abnormal states can be stored and later reused.

---

## 26. Scientific comparison structure

The proposal should eventually be evaluated by separating the effects of **population representation**, **heterogeneity**, and **selective retrieval**.

Conceptually important comparisons include:

| Scientific comparison | Question |
|---|---|
| Standard LIF | Is any new memory mechanism needed? |
| f-LIF / f-SNN | Does selective memory add value beyond fixed power-law history? |
| Homogeneous LIF population | Does merely increasing the number of constituent neurons help? |
| Heterogeneous LIF population without retrieval | Does multi-timescale population state alone help? |
| Population + fixed temporal weighting | Is content-dependent relevance necessary? |
| Population + dense relevance weighting | Is sparsity/selection necessary? |
| Population + random/recent-state selection | Is learned relevance better than trivial selection? |
| Scalar membrane memory + retrieval | Does population state improve retrieval over scalar state? |
| Input-history retrieval | Is membrane-state memory specifically useful? |
| Spike-history retrieval | Is membrane state better than spike-only history? |
| Fractional prior vs. no prior | Does a power-law temporal bias remain useful after content selection? |
| Pre-reset vs. post-reset memory | What state is the better memory representation? |

These comparisons are scientific requirements for attribution of effects, not implementation instructions.

---

## 27. Synthetic experiments that would directly test the idea

Before relying only on large real datasets, the concept is well suited to controlled tasks where the relevant lag is known.

A useful conceptual synthetic task would have:

- a current observable context variable;
- two or more candidate historical lags;
- the correct target determined by a different lag depending on current context;
- irrelevant intermediate observations;
- potentially variable lag distances.

Example:

\[
\text{if context}=A,\quad y_t=f(x_{t-8}),
\]

\[
\text{if context}=B,\quad y_t=f(x_{t-64}).
\]

This directly tests whether:

- retrieval changes with current context;
- the correct lag receives high weight;
- the population state helps distinguish histories;
- the model does more than learn a fixed recency kernel.

Another useful controlled case is repeated-regime retrieval:

\[
A\rightarrow B\rightarrow A,
\]

where the target at the second \(A\) depends on information from the earlier \(A\), not the intervening \(B\).

---

## 28. Evaluation beyond task accuracy

The central claim concerns memory behavior, so accuracy alone is insufficient.

Important scientific observables include:

### Retrieval behavior
- distribution of selected lags;
- entropy/sparsity of retrieval;
- state-dependent change in selected lag;
- frequency of selecting distant history;
- agreement with known relevant lag in synthetic data.

### Population behavior
- diversity among constituent membrane trajectories;
- diversity of time constants or response profiles;
- whether the population collapses to redundant states;
- whether different logical neurons specialize to different temporal patterns.

### Intervention tests
- remove the highest-weight memory;
- replace it with an irrelevant memory;
- remove distant memory while retaining recent memory;
- compare prediction change.

These are needed before interpreting retrieval weights as meaningful temporal explanations.

### Robustness
- input noise;
- missing segments;
- temporal jitter;
- irrelevant inserted subsequences;
- regime shifts.

### Complexity/efficiency
Any efficiency analysis should account for:
- constituent LIF count;
- synaptic computation;
- history storage;
- relevance computation;
- memory read/write;
- selection overhead;
- spike activity.

A reduced firing rate alone is not sufficient to establish lower total cost.

---

## 29. Suggested terminology

To keep discussion precise, use the following terminology unless a later paper draft chooses different names.

### Constituent LIF neuron
One ordinary LIF element inside a population.

### Logical neuron
The population of \(K\) constituent LIF neurons treated as one conceptual unit.

### Population state
The vector

\[
\mathbf u_{t,n}\in\mathbb R^K.
\]

### Memory slot
A stored historical population state associated with time \(j\).

### Memory bank / history memory
The collection of available past memory slots.

### Relevance score
Current-conditioned score \(r_{t,j,n}\) indicating how useful past slot \(j\) is at current time \(t\).

### Read mask
Soft or hard selection determining which historical states participate in retrieval.

### Temporal prior
Lag-dependent bias, potentially fractional/power-law.

### Retrieved memory
The weighted aggregate \(\mathbf M_{t,n}\) of selected past population states.

### Memory-use gate
A factor controlling whether/how strongly the retrieved memory affects current neuronal dynamics.

Avoid calling the direct membrane-retrieval model "fractional LIF" unless its dynamics are actually derived as a fractional-order system.

---

## 30. The intended conceptual contribution

The strongest framing is not simply:

> "Use more neurons."

It is not simply:

> "Apply attention to SNNs."

It is not simply:

> "Mask fractional memory."

The intended research contribution is closer to:

> **A logical spiking neuron represents its internal temporal state through a heterogeneous LIF population and uses the resulting population membrane pattern to selectively retrieve the past internal states most relevant to the present context.**

This creates a division of roles:

\[
\boxed{
\text{population}
=
\text{how temporal state is represented}
}
\]

and

\[
\boxed{
\text{selective retrieval}
=
\text{which past state is used now}
}
\]

with an optional third component:

\[
\boxed{
\text{fractional prior}
=
\text{how temporal distance biases memory before contextual selection}
}
\]

The high-level scientific shift is therefore:

\[
\boxed{
\text{"How long should a neuron remember?"}
\quad\rightarrow\quad
\text{"Which memory is relevant now, and how should it be represented?"}
}
\]

---

## 31. Relationship between the main conceptual components

A useful conceptual hierarchy is:

### Level 1 — Constituent neuronal dynamics
Ordinary LIF elements provide spike-generating stateful dynamics.

### Level 2 — Population state
Several heterogeneous LIF elements jointly represent one logical neuron's temporal state.

### Level 3 — Historical memory
Population states from prior time steps become explicit memory slots.

### Level 4 — Content-adaptive retrieval
The present population state evaluates the usefulness of past population states.

### Level 5 — Temporal prior
Optionally, a fractional/power-law prior biases retrieval according to elapsed time.

### Level 6 — Present spiking dynamics
Retrieved memory influences the current logical neuron's state and eventual spiking behavior.

The novelty should be evaluated at the interaction among these levels rather than at any single component in isolation.

---

## 32. References and verified literature context

### [1] Ge et al. — Fractional-order SNN framework

Chengjie Ge, Yufeng Peng, Zihao Li, Qiyu Kang, Xueyang Fu, Xuhao Li, Qixin Zhang, Junhao Ren, Zheng-Jun Zha.  
**"Fractional-Order Spiking Neural Network."**  
International Conference on Learning Representations (ICLR), 2026.

Official proceedings:  
https://proceedings.iclr.cc/paper_files/paper/2026/hash/80b4df828ee59926a5f2422f1c072d88-Abstract-Conference.html

Key relevance:
- fractional-order dynamics generalize ordinary LIF/IF;
- power-law/persistent memory;
- non-Markovian history dependence;
- theoretical comparison with finite integer-order systems;
- short-memory approximation and computational-complexity discussion.

### [2] Teka, Marinov, and Santamaria — Fractional LIF and voltage-memory trace

Wondimu Teka, Toma M. Marinov, Fidel Santamaria.  
**"Neuronal Spike Timing Adaptation Described with a Fractional Leaky Integrate-and-Fire Model."**  
PLOS Computational Biology, 10(3): e1003526, 2014.  
DOI: 10.1371/journal.pcbi.1003526

https://doi.org/10.1371/journal.pcbi.1003526

Key relevance:
- fractional LIF as a non-Markovian model;
- temporally weighted voltage-memory trace;
- multi-timescale adaptation;
- reliable spiking under noisy input in the studied setting.

### [3] Deng et al. — Fractional spiking neuron and dendritic fractal model

Yabin Deng, Bijing Liu, Zenan Huang, Xiaojie Liu, Shan He, Qiuhong Li, Donghui Guo.  
**"Fractional Spiking Neuron: Fractional Leaky Integrate-and-Fire Circuit Described with Dendritic Fractal Model."**  
IEEE Transactions on Biomedical Circuits and Systems, 16(6):1375–1386, 2022.  
DOI: 10.1109/TBCAS.2022.3218294

https://doi.org/10.1109/TBCAS.2022.3218294

Key relevance:
- biological/circuit motivation for fractional LIF;
- multiple-timescale dynamics.

### [4] Zhang et al. — Population coding in an SNN

Duzhen Zhang, Tielin Zhang, Shuncheng Jia, Bo Xu.  
**"Multi-Sacle Dynamic Coding Improved Spiking Actor Network for Reinforcement Learning."**  
Proceedings of the AAAI Conference on Artificial Intelligence, 36(1):59–67, 2022.  
DOI: 10.1609/aaai.v36i1.19879

https://doi.org/10.1609/aaai.v36i1.19879

Key relevance:
- each input-state dimension is encoded by an individual neuron population;
- population coding combined with richer neuronal dynamics;
- establishes a concrete SNN precedent for representing one variable through multiple neurons.

### [5] Frontiers in Computational Neuroscience — LIF population coding with receptive fields

**"Enhanced representation learning with temporal coding in sparsely spiking neural networks."**  
Frontiers in Computational Neuroscience, 2023.  
DOI: 10.3389/fncom.2023.1250908

https://doi.org/10.3389/fncom.2023.1250908

Key relevance:
- one input dimension represented collectively by a population of LIF neurons;
- overlapping Gaussian receptive fields;
- supports the general population-coding inspiration.

### [6] Perez-Nieves et al. — Neural heterogeneity and multiple timescales

Nicolas Perez-Nieves, Vincent C. H. Leung, Pier Luigi Dragotti, Dan F. M. Goodman.  
**"Neural heterogeneity promotes robust learning."**  
Nature Communications, 12:5791, 2021.  
DOI: 10.1038/s41467-021-26022-3

https://doi.org/10.1038/s41467-021-26022-3

Key relevance:
- individual membrane and synaptic time constants;
- heterogeneous temporal dynamics improve performance particularly on temporally rich tasks;
- increased learning stability and robustness in the experiments.

### [7] Heterogeneous recurrent SNN

**"Heterogeneous recurrent spiking neural network for spatio-temporal classification."**  
Frontiers in Neuroscience, 2023.  
DOI: 10.3389/fnins.2023.994517

https://doi.org/10.3389/fnins.2023.994517

Key relevance:
- heterogeneous LIF membrane time constants and thresholds;
- multiple timescales in a spiking network.

### [8] Yao et al. — Temporal attention over event-stream frames

Man Yao, Huanhuan Gao, Guangshe Zhao, Dingheng Wang, Yihan Lin, Zhaoxu Yang, Guoqi Li.  
**"Temporal-Wise Attention Spiking Neural Networks for Event Streams Classification."**  
Proceedings of ICCV, 2021.  
DOI: 10.1109/ICCV48922.2021.01006

https://doi.org/10.1109/ICCV48922.2021.01006

Key relevance:
- temporal-wise importance estimation;
- irrelevant frames can be discarded;
- demonstrates that temporal selection in SNNs is established prior art.

### [9] Yao et al. — Attention regulating SNN membrane potential

Man Yao, Guangshe Zhao, Hengyu Zhang, Yifan Hu, Lei Deng, Yonghong Tian, Bo Xu, Guoqi Li.  
**"Attention Spiking Neural Networks."**  
IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8):9393–9410, 2023.  
DOI: 10.1109/TPAMI.2023.3241201

https://doi.org/10.1109/TPAMI.2023.3241201

Key relevance:
- temporal/channel/spatial attention in SNNs;
- attention weights modulate membrane potentials and spiking responses.

### [10] Fang et al. — Masked temporal spiking dynamics

Wei Fang, Zhaofei Yu, Zhaokun Zhou, Ding Chen, Yanqi Chen, Zhengyu Ma, Timothée Masquelier, Yonghong Tian.  
**"Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies."**  
NeurIPS 2023.

https://proceedings.neurips.cc/paper_files/paper/2023/hash/a834ac3dfdb90da54292c2c932c997cc-Abstract-Conference.html

Key relevance:
- temporal input weighting in a spiking neuron family;
- masked PSN for causal temporal structure;
- demonstrates that "temporal mask inside a spiking neuron" alone is not sufficient novelty.

### [11] Gu and Dao — Selective state-space modeling

Albert Gu, Tri Dao.  
**"Mamba: Linear-Time Sequence Modeling with Selective State Spaces."**  
arXiv:2312.00752, 2023.

https://arxiv.org/abs/2312.00752

Key relevance:
- fixed state-space dynamics are made input-dependent;
- selective propagation/forgetting addresses content-based reasoning;
- useful conceptual analogy for distinguishing time-invariant memory from content-adaptive memory;
- also illustrates that content-dependent dynamics can disrupt simple convolutional computation.

---

## 33. Final statement for the research/coding agent

The project should be understood as investigating a **neuron-level selective memory principle for time-series SNNs**.

The idea has evolved through the following reasoning:

1. f-LIF shows that persistent nonlocal history can enrich SNN temporal dynamics.
2. Persistent access to all history does not solve the problem of deciding which history is relevant to the present.
3. A scalar membrane potential may be too compressed to serve as a reliable key for memory relevance.
4. Population coding motivates representing one logical quantity by multiple neuronal responses.
5. Neuronal heterogeneity motivates giving those constituent neurons distinct temporal response profiles.
6. Therefore, define one logical neuron as a heterogeneous LIF population.
7. Treat the population membrane-state vector as the logical neuron's internal temporal representation.
8. Store historical population states as memory slots.
9. Let the current population state determine which past population states are relevant.
10. Retrieve a weighted subset of those states.
11. Allow the retrieved memory to influence current spiking dynamics.
12. Optionally preserve fractional/power-law decay as a temporal prior rather than as the sole determinant of memory strength.

The intended conceptual message is:

\[
\boxed{
\text{Long-term memory should be both multi-timescale and selective.}
}
\]

or, more specifically,

\[
\boxed{
\text{A spiking neuron should not only preserve the past;
it should learn which past internal state is useful now.}
}
\]

The population is intended to make that decision from a richer internal state than a single scalar membrane potential.

This document intentionally leaves software and implementation choices unspecified.
