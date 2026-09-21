# Research Assessment Memo: Feasibility and Current Interpretation of the Population-Based Selective Membrane Memory Idea

## Purpose

This memo summarizes the current research judgment after reviewing the implementation history, experimental protocol, completed v1/v2 results, and the ongoing backbone expansion.

It is intended for a coding/research agent who will continue the work.

This document is **not an implementation guide**. It focuses on:

- what has already been established;
- what has **not** yet been established;
- what the current results actually imply;
- what research questions now matter most;
- how to interpret upcoming experiments without overclaiming.

The main source for this assessment is the current project handoff document and the completed v1/v2 experiment summaries.

---

# 1. High-level conclusion

The current status should be understood as:

> **The mechanism is feasible and trainable, but the central semantic retrieval hypothesis is not yet proven.**

More concretely:

- The population-based neuron can be trained stably.
- Heterogeneous LIF populations do create a measurable effect.
- Sparse retrieval with a null option is actually active in practice.
- The retrieved memory materially affects predictions.
- However, it has **not yet been shown that the model reliably retrieves the “correct” or task-relevant past state**.
- It has also **not yet been shown that heterogeneous population states improve retrieval quality compared with scalar or homogeneous states**.

Therefore, the research idea should **not be abandoned**, but the core validation target must now become more precise.

The key question is no longer:

> “Can the model selectively read memory?”

That has already been demonstrated at the mechanism level.

The key question is now:

> **“Can the current population state identify which past state is actually useful for the current prediction?”**

And, more specifically:

> **“Does a heterogeneous population representation improve that identification ability over simpler state representations?”**

---

# 2. What is already established

## 2.1 Mechanism feasibility is confirmed

The v2 design fixes the major limitations identified in v1:

- explicit sparse support is possible;
- null/empty read is possible;
- amplitude information is preserved in the relevance score;
- retrieval-aware gating is used;
- dense and sparse reads can be compared under a common scoring structure.

The sparse version is not merely “soft attention with small values.”

Observed support statistics show that it actually excludes many historical slots and sometimes chooses no real memory at all.

For example, in the completed patch experiments:

- H96 heterogeneous sparse support density is about 0.33;
- H720 heterogeneous sparse support density is about 0.39;
- nonzero empty-read fractions are observed.

Therefore:

\[
\boxed{
\text{explicit selective read behavior exists}
}
\]

This is an important positive result.

---

## 2.2 Heterogeneous population dynamics are not redundant in all settings

In the completed v2 patch H96 runs, the no-memory heterogeneous population performs better than the no-memory homogeneous population.

This supports the hypothesis that a population with different temporal dynamics can create a richer state representation.

The most relevant conceptual interpretation is:

\[
\boxed{
\text{multiple leak/time-scale responses can provide useful temporal state diversity}
}
\]

However, this effect does not hold uniformly at H720.

Therefore, the correct interpretation is:

> **Heterogeneity appears useful in some temporal regimes, but it is not yet a universally beneficial inductive bias.**

It should not be claimed that heterogeneous population dynamics always improve forecasting.

---

## 2.3 The memory pathway is actually used by the trained model

Test-time interventions show that turning memory off can alter prediction quality.

In H96 heterogeneous sparse, disabling retrieval at test time worsens MSE.

This indicates that the trained network is not simply ignoring the memory branch.

Therefore:

\[
\boxed{
\text{memory is functionally coupled to the predictor}
}
\]

This matters because it rules out a trivial failure mode in which the retrieval mechanism exists in code but is unused by the trained model.

---

# 3. What is not yet established

## 3.1 The main synergy hypothesis is not yet supported

The original research concept expected:

\[
\text{heterogeneous population}
+
\text{selective retrieval}
\]

to outperform the no-memory version because the richer population state should act as a better retrieval key.

The completed patch results do **not** currently establish this.

### H96

The heterogeneous dense model is only slightly better than heterogeneous no-memory.

The heterogeneous sparse model is worse than heterogeneous no-memory.

### H720

Both heterogeneous dense and heterogeneous sparse are worse than heterogeneous no-memory.

Therefore:

\[
\boxed{
\text{heterogeneous population + selective retrieval synergy is not yet demonstrated}
}
\]

This is currently the central unresolved point.

---

## 3.2 Sparse selection itself is not yet proven to be superior

The model now produces sparse support, but sparse support alone is not equivalent to useful retrieval.

In H96, dense retrieval is slightly better than sparse retrieval for heterogeneous populations.

In H720, sparse retrieval is worse than both dense and no-memory in the heterogeneous case.

Thus, the following statement is not yet justified:

> “Explicit sparse selection improves forecasting.”

The current evidence only supports:

> “Explicit sparse selection is technically realized and changes model behavior.”

---

## 3.3 Retrieval weights are not yet validated as task-relevant memory

A major unresolved issue is that the model can select memories without necessarily selecting the *right* memories.

Current real-data diagnostics show:

- support density;
- empty-read fraction;
- real mass;
- lag-related summary behavior;
- intervention response.

But these do **not** tell us whether a selected historical slot was truly relevant to the target.

This is the most important conceptual gap.

The model currently answers:

> “Which past state looks relevant under the learned scoring function?”

But the research claim requires evidence for:

> “Which past state was actually useful for the forecasting task?”

These are not automatically the same.

---

# 4. Why the current forecasting setup may underestimate memory usefulness

## 4.1 The flatten forecast head can directly access all temporal outputs

In the current patch architecture, the final forecast head receives the full sequence of spike representations across all patches.

Conceptually:

\[
S_1,S_2,\ldots,S_T
\rightarrow
\text{flatten}
\rightarrow
\text{forecast}
\]

This creates an important confound.

The original motivation for selective memory was:

> Important historical information should be retrieved into the current neuronal state.

However, if the final predictor already receives all historical outputs directly, it can bypass the memory mechanism.

For example, if time step \(j\) contains useful information, the predictor can read \(S_j\) directly rather than requiring:

\[
S_j
\rightarrow
\text{memory retrieval}
\rightarrow
S_t
\rightarrow
\text{forecast}.
\]

Therefore, the current patch experiment is better interpreted as:

> **Does neuron-level memory improve a predictor that already has direct access to all temporal outputs?**

It is **not** a pure test of whether the neuron itself can preserve and retrieve useful temporal information.

This distinction is critical.

---

## 4.2 This makes a bottlenecked readout scientifically important

A readout that only uses the final or heavily compressed state would create a stronger requirement:

\[
\boxed{
\text{historical information must survive or be retrieved into later states}
}
\]

Under such a bottleneck, the contrast between:

- standard LIF;
- heterogeneous population LIF;
- population + retrieval

would more directly test memory functionality.

The earlier v1 “last-head” direction was therefore conceptually relevant, even though its existing numerical results are not suitable for final conclusions because of:

- v1 scoring;
- limited seed coverage;
- short training budget.

The important point is the **experimental logic**, not those specific results.

---

# 5. H720 should not be interpreted as a “long-memory failure”

The input history is the same for H96 and H720:

\[
\text{input length}=336
\]

and, after patching,

\[
T=42.
\]

Thus, the available memory bank is the same size in both settings.

Only the forecast horizon changes.

Therefore, worse H720 retrieval performance does **not** mean:

> “The model cannot remember 720 time steps.”

It means:

> **Given the same 336-step historical context, the current retrieved-memory evidence is not helping the longer-horizon prediction.**

This is a forecasting-horizon issue, not a direct memory-length result.

---

# 6. The homogeneous sparse H720 result is an important warning signal

In H720 patch experiments, homogeneous sparse achieves the best macro MSE among the six conditions.

This is important because it weakens a simple version of the original story:

> “Heterogeneous temporal population states are necessary to make retrieval effective.”

At least in the current data:

- homogeneous sparse can outperform homogeneous no-memory;
- heterogeneous sparse can perform worse than heterogeneous no-memory.

However, this effect is strongly dataset-dependent.

The improvement appears mainly in ETTh2, while ETTh1 does not show the same pattern.

Therefore, the current interpretation should be:

\[
\boxed{
\text{retrieval benefit is strongly dataset-dependent}
}
\]

and not:

> “Homogeneous is better than heterogeneous.”

The latter would be premature.

---

# 7. Intervention results suggest that “memory pathway usefulness” and “retrieval quality” are different questions

This distinction is one of the most important conclusions from the current experiments.

## H96 heterogeneous sparse

Turning memory off worsens performance.

This suggests:

\[
\text{memory pathway is useful}
\]

But replacing learned sparse retrieval with uniform or recent retrieval can perform similarly or slightly better.

This suggests:

\[
\text{learned memory selection may not yet be clearly superior}
\]

Therefore:

> The model may benefit from having a memory pathway, while the learned relevance mechanism itself remains imperfect.

---

## H720 heterogeneous sparse

Turning memory off improves performance.

This means:

> In that trained model, the memory pathway is actively harmful at inference.

This is different from a separately trained no-memory model comparison.

It raises a potentially important hypothesis:

\[
\boxed{
\text{retrieval may alter training dynamics even when inference-time retrieval is not beneficial}
}
\]

For example, retrieval could act as:

- a regularizer;
- a perturbation mechanism;
- a path that changes representation learning;
- a source of optimization bias.

This is not yet proven, but it is a better interpretation than simply assuming that improved sparse models must be using useful memories at inference.

---

# 8. The current likely bottleneck is not sparsity, but the definition of relevance

The v2 implementation already solves the major technical issue from v1:

- it can exclude memories;
- it can select null;
- it preserves amplitude;
- it uses a richer gate.

The remaining deeper question is:

\[
\boxed{
\text{What should “relevant” mean?}
}
\]

The current score compares transformed current and historical population states.

This is reasonable, but it may still favor:

> states that are similar to the current state

rather than:

> states that are useful for predicting the target.

These are not necessarily equivalent.

A useful past state may be:

- similar;
- complementary;
- a precursor to the current regime;
- an earlier state that preceded a future transition;
- structurally related but not close in Euclidean space.

Therefore, the current research bottleneck is likely no longer:

> “Can sparse retrieval be implemented?”

It is more likely:

> **“Can the retrieval criterion identify task-relevant historical information?”**

---

# 9. The highest-priority next scientific experiment: controlled synthetic recall

The strongest next validation should use a task where the correct historical dependency is known by construction.

This is more important than simply collecting more real-data MSE.

## Example 1: context-dependent lag

Construct a task such that:

\[
\text{context}=A
\Rightarrow
y_t=f(x_{t-8}),
\]

\[
\text{context}=B
\Rightarrow
y_t=f(x_{t-32})
\]

or another pair of clearly separated lags.

The model should then be evaluated on:

- prediction accuracy;
- correct-slot retrieval rate;
- whether selected support contains the true relevant lag;
- how retrieval changes when context changes;
- whether heterogeneous population improves correct-slot retrieval.

This directly tests the core claim.

---

## Example 2: regime recurrence

Use a controlled pattern such as:

\[
A \rightarrow B \rightarrow A.
\]

The target in the second \(A\) regime should depend on the earlier \(A\) state rather than the intervening \(B\) state.

This tests whether the model can:

- ignore recent but irrelevant history;
- retrieve a more distant matching regime;
- adapt retrieval based on current context.

---

# 10. The most important research questions now

The project should explicitly center the following questions.

## Q1. Can the model identify the correct past state?

This is the fundamental retrieval-validity question.

\[
\boxed{
\text{Does learned retrieval select the task-relevant historical slot?}
}
\]

---

## Q2. Does heterogeneous population state improve retrieval?

This tests the main population-retrieval interaction hypothesis.

\[
\boxed{
\text{Is retrieval more accurate when the key is a heterogeneous population state?}
}
\]

This should be compared against:

- scalar state;
- homogeneous population;
- heterogeneous population.

---

## Q3. Is retrieval useful only when there is an information bottleneck?

This tests whether flatten/all-time-step readout hides the value of memory.

Compare settings where:

- all temporal states are directly visible to the forecast head;
- only a compressed or final state is visible.

If selective memory only helps under a bottleneck, that is still a meaningful result.

It would simply define the task regime more precisely.

---

## Q4. Is sparse selection better than dense weighting?

This remains open.

The comparison should not only use forecasting MSE, but also:

- retrieval correctness;
- support quality;
- robustness to distractors;
- sensitivity to irrelevant-history length.

---

## Q5. Does the fractional prior add anything once content-based retrieval exists?

This remains a later-stage question.

Possible outcomes:

- no prior is best;
- exponential prior helps;
- fractional/power-law prior helps;
- prior helps only under long distractor intervals.

This should be treated as a separate research question, not assumed.

---

# 11. How to interpret the ongoing 288-run backbone matrix

The TCN, PatchTST-like, and TSMixer-like runs are still useful.

They answer:

> **Does the PopulationLIF retrieval mechanism provide practical benefit across different temporal backbones?**

This is an important robustness/generalization question.

However, these runs do **not** directly answer:

> **Does the neuron correctly retrieve task-relevant memory?**

The reason is that these backbones themselves provide temporal processing paths:

- TCN uses causal convolution;
- PatchTST-like block uses causal attention;
- TSMixer-like block uses temporal mixing.

These can partially bypass or duplicate the role of the internal memory mechanism.

Therefore, the large backbone matrix and the controlled memory-validity experiments should be treated as **complementary**, not interchangeable.

---

# 12. Possible outcome branches for the project

The project should be prepared for several scientifically valid outcomes.

## Outcome A: full hypothesis supported

If synthetic tasks show:

- correct memory retrieval;
- context-dependent lag selection;
- heterogeneous population improves retrieval;
- real forecasting gains appear in at least some settings,

then the original idea is strongly supported.

The contribution can be framed around:

> heterogeneous population state as a richer temporal retrieval key.

---

## Outcome B: retrieval is valid, but real forecasting gains are limited

If controlled tasks show correct retrieval but ETT gains are small:

> the mechanism is valid, but its practical benefit is task-dependent.

This is still a viable research result.

The paper should then target problems where:

- explicit long-range recall is required;
- irrelevant distractor history exists;
- the final predictor cannot directly access all historical states.

---

## Outcome C: population helps, retrieval does not

If heterogeneous populations repeatedly help but retrieval remains neutral or harmful:

> the strongest contribution may shift toward multi-timescale population neurons rather than selective memory.

In that case, selective retrieval should not remain the main story.

---

## Outcome D: retrieval helps, heterogeneity is unnecessary

If sparse/dense retrieval works equally well or better with homogeneous populations:

> selective membrane memory remains viable, but population coding/heterogeneity is not the core contribution.

The model story should be simplified accordingly.

---

## Outcome E: controlled retrieval also fails

If the model cannot recover known relevant slots even in synthetic controlled tasks:

> the current relevance formulation should be reconsidered before running larger experiment grids.

This would be the clearest sign that the main retrieval mechanism needs redesign.

---

# 13. Current research judgment

At this stage, the idea remains **scientifically feasible and worth continuing**.

The strongest positive evidence is:

1. selective support is technically realized;
2. null reads occur;
3. heterogeneous populations produce real representational effects;
4. the memory path affects predictions;
5. the model trains stably under the current protocol.

The strongest negative or unresolved evidence is:

1. heterogeneous + sparse retrieval synergy is not established;
2. sparse retrieval is not consistently better than dense or no-memory;
3. H720 heterogeneous retrieval is harmful in the completed patch experiments;
4. homogeneous sparse can outperform heterogeneous sparse;
5. current diagnostics do not prove that selected memories are task-relevant;
6. flatten/all-time-step heads can bypass the need for explicit neuronal memory.

Therefore, the project should **not** conclude either:

- “the idea works,” or
- “the idea fails.”

The most accurate current status is:

\[
\boxed{
\text{mechanism feasibility confirmed}
}
\]

but

\[
\boxed{
\text{semantic retrieval validity unresolved}
}
\]

---

# 14. Most important next priority

The highest-priority research task should be:

> **Directly test whether the model retrieves the correct historical state under controlled conditions.**

The key measurement should not be only forecast MSE.

It should include:

- correct-slot retrieval rate;
- support-hit rate;
- retrieval under distractors;
- retrieval under context-dependent lag changes;
- homogeneous vs heterogeneous population comparison;
- scalar vs population-state comparison.

Only after this is established should real-data forecasting gains be interpreted as evidence for selective memory quality.

---

# 15. Short version for future sessions

If a future agent needs only one paragraph:

> The current v2 system successfully implements and trains a population-based membrane-memory neuron with true sparse support and null reads. Heterogeneous populations show useful effects in some settings, and the memory pathway measurably affects predictions. However, the key research claim—that heterogeneous population states enable better retrieval of task-relevant past states—has not yet been demonstrated. Current ETT forecasting results are mixed, and the flatten head can directly access all temporal outputs, partially bypassing the need for memory. The highest-priority next validation is therefore a controlled synthetic recall task with known relevant lags/slots and distractors, where retrieval correctness can be measured directly and compared across scalar, homogeneous, and heterogeneous population states. The large TCN/PatchTST/TSMixer matrix remains useful for practical robustness, but it should not be treated as a substitute for direct semantic retrieval validation.
