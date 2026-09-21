# PopulationLIF Implementation Review

## Scope

This review checks whether the provided `PopulationLIF` module matches the research idea discussed so far.

The review intentionally ignores minor implementation details, coding style, optimization, and engineering concerns. The only question is:

> **Does the current implementation faithfully realize the intended population-based selective membrane-memory concept?**

---

## Overall Verdict

**Mostly yes, but not completely.**

The current module correctly implements the main **population-based, content-adaptive dense memory retrieval** mechanism that was discussed.

However, one core part of the original idea is still missing:

> **The current implementation does not explicitly mask out irrelevant past time steps.**

In other words, the model currently performs **dense soft retrieval over all past states**, rather than **sparse/masked retrieval over only selected relevant states**.

Therefore, the most accurate description of the current implementation is:

> **A population-based content-adaptive dense membrane-memory retrieval model.**

It should not yet be described as a fully sparse or explicitly masked selective-memory model.

---

## What Is Correctly Implemented

### 1. One logical neuron is represented by a population of LIF neurons

The intended idea was:

\[
	ext{one logical neuron}
=
\{\mathrm{LIF}_1,\mathrm{LIF}_2,\ldots,\mathrm{LIF}_K\}.
\]

The implementation correctly realizes this by introducing a population dimension of size `K = num_population`.

The logical neuronal state is therefore represented as a population membrane vector rather than a scalar membrane potential.

**Status: Correct.**

---

### 2. The population members have heterogeneous temporal dynamics

The heterogeneous version assigns different time constants and therefore different leak factors:

\[
	au_k
ightarrow
eta_k.
\]

This produces different temporal responses among the constituent LIF neurons.

The homogeneous control preserves the same population size while forcing all population members to share the same average leak factor.

This is consistent with the intended comparison between:

- heterogeneous population;
- homogeneous population with the same number of constituent neurons.

**Status: Correct.**

---

### 3. Constituent LIF neurons of the same logical neuron share the same current input

The current input is broadcast across the population dimension, while each constituent neuron responds differently because of its own leak factor.

This matches the intended interpretation:

> The population provides multiple temporal views of the same logical input/state.

**Status: Correct.**

---

### 4. The current population state is used as the retrieval query

The current pre-memory population state

\[
ar{\mathbf u}_t
\]

is used to construct the query.

This is consistent with the intended current-conditioned retrieval mechanism:

\[
r_{t,j}
=
\operatorname{Rel}
\left(
ar{\mathbf u}_t,
\mathbf u_j
ight).
\]

Thus, relevance is not determined only by temporal distance.

**Status: Correct.**

---

### 5. Past post-reset population membrane states are stored as memory

The model stores the membrane state after spike generation and subtractive reset.

Therefore, a memory slot corresponds to a historical population state:

\[
\mathbf u_j \in\mathbb R^K.
\]

This matches the working convention discussed previously, where post-reset membrane states were used as memory values.

**Status: Correct.**

---

### 6. Retrieval is performed at the population level

A single relevance score is computed for each historical time step using the complete population vector.

The resulting temporal weight is shared across all \(K\) constituent LIF neurons of the same logical neuron.

Therefore, the model implements:

\[
	ext{one temporal retrieval decision per logical population}
\]

rather than allowing every constituent LIF neuron to independently retrieve a different time step.

This matches the preferred conceptual formulation.

**Status: Correct.**

---

### 7. More relevant historical population states receive larger weights

The current and historical population representations are compared using normalized query/key vectors.

The relevance scores are converted into temporal weights with a softmax:

\[
w_{t,j}
=
rac{\exp(r_{t,j})}
{\sum_{\ell<t}\exp(r_{t,\ell})}.
\]

The retrieved memory is then

\[
\mathbf M_t
=
\sum_{j<t}
w_{t,j}\mathbf u_j.
\]

Therefore, historical states judged more relevant to the current population state contribute more strongly.

This directly implements the intended idea:

> Give larger weight to past membrane states that are more relevant to the current time step.

**Status: Correct.**

---

### 8. Retrieved memory directly influences the current neuronal state

The retrieved memory is added to the current charged population state:

\[
\mathbf v_t
=
ar{\mathbf u}_t
+
	ext{memory evidence}.
\]

Therefore, retrieval affects spike generation rather than being used only for auxiliary analysis.

This matches the intended neuron-level memory mechanism.

**Status: Correct.**

---

### 9. Memory influence is controlled by a gate

The model separately controls how strongly retrieved memory affects the current state.

This preserves the conceptual distinction between:

1. **which past memory is most relevant**, and
2. **whether/how strongly memory should be used at the current time step**.

**Status: Correct.**

---

### 10. The output preserves the population spike representation

The output is

\[
[T, BC, D, K].
\]

Each constituent LIF neuron can emit its own spike.

Therefore, the population representation is not immediately collapsed into a single binary logical spike.

This matches the preferred initial interpretation discussed previously.

**Status: Correct.**

---

### 11. Memory does not leak across independent forward windows

The membrane state and history are initialized inside each forward call.

Therefore, separate independent sequence windows do not unintentionally share memory.

This is consistent with the intended basic time-series setting.

**Status: Correct.**

---

### 12. The intervention modes are conceptually appropriate

The provided evaluation interventions allow comparison with:

- memory disabled;
- uniform history weighting;
- most-recent-state retrieval.

These correspond to important scientific controls for determining whether learned relevance actually matters.

**Status: Correct.**

---

## The Main Missing Component

### Explicit sparse/masked selection is not implemented

The original research idea was stronger than simply assigning different weights to all historical states.

The intended mechanism was:

\[
\mathbf M_t
=
\sum_{j<t}
m_{t,j}w_{t,j}\mathbf u_j,
\]

where

\[
m_{t,j}
\]

acts as a mask or selection variable so that irrelevant historical states can be excluded.

The current implementation instead uses:

\[
w_{t,j}
=
\operatorname{softmax}(r_{t,j}),
\]

followed by

\[
\mathbf M_t
=
\sum_{j<t}
w_{t,j}\mathbf u_j.
\]

Because softmax generally gives every historical position a positive weight,

\[
w_{t,j} > 0,
\]

all available past states participate in the retrieved memory.

Some may receive very small weights, but they are not explicitly removed.

Therefore:

- **content-adaptive weighting:** implemented;
- **explicit selective masking:** not implemented;
- **sparse retrieval:** not implemented.

---

## Why This Distinction Matters

The current model can validly claim:

> The model assigns larger weights to past population membrane states that are more relevant to the current state.

The current model should **not yet** claim:

> The model selects only relevant historical states.

or

> Irrelevant historical states are masked out.

or

> The neuron performs sparse memory retrieval.

Those statements would require an actual mechanism that excludes historical states from retrieval.

---

## Fractional Temporal Prior

A fractional or power-law temporal prior is not currently present.

This is **not considered an implementation error**, because the previous discussion explicitly treated the fractional prior as optional.

The current model therefore represents a **content-based selective-memory direction without an explicit fractional temporal prior**.

**Status: Acceptable by design.**

---

## Final Assessment Table

| Intended component | Current implementation |
|---|---|
| One logical neuron = \(K\) LIF population | Correct |
| Heterogeneous temporal dynamics | Correct |
| Same logical input shared across population | Correct |
| Population membrane vector as neuronal state | Correct |
| Past post-reset membrane states stored as memory | Correct |
| Current population state used as retrieval query | Correct |
| Population-level temporal retrieval | Correct |
| More relevant history receives larger weight | Correct |
| Retrieved memory influences current membrane state | Correct |
| Separate memory-use gate | Correct |
| Population spike output retained | Correct |
| No cross-window memory leakage | Correct |
| Homogeneous-population control | Correct |
| Memory-off intervention | Correct |
| Uniform-history intervention | Correct |
| Recent-only intervention | Correct |
| Explicit masking of irrelevant history | **Missing** |
| Sparse selection of only selected past states | **Missing** |
| Fractional temporal prior | Not included, but optional |

---

## Final Verdict

The implementation is **conceptually correct for the dense retrieval version of the proposed model**.

The main research idea currently realized is:

\[
oxed{
	ext{heterogeneous LIF population}
ightarrow
	ext{population membrane-state memory}
ightarrow
	ext{current-conditioned relevance weighting}
ightarrow
	ext{retrieved memory}
ightarrow
	ext{current spiking dynamics}
}
\]

The only major mismatch with the fuller original proposal is:

\[
oxed{
	ext{explicit sparse/masked selection of historical states is absent}
}
\]

Therefore, the model should currently be treated as:

> **Population-based content-adaptive dense membrane-memory retrieval**

rather than:

> **Population-based sparse selective membrane-memory retrieval.**

No other major conceptual mismatch was identified in this review.
